---
draft: false
title: "arXiv @ 2023.08.19"
date: 2023-08-19
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.19"
    identifier: arxiv_20230819
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.DS (2)](#csds-2)
- [cs.IR (1)](#csir-1)
- [cs.SE (5)](#csse-5)
- [cs.CR (5)](#cscr-5)
- [cs.AI (5)](#csai-5)
- [cs.CV (29)](#cscv-29)
- [cs.LG (12)](#cslg-12)
- [eess.IV (3)](#eessiv-3)
- [cs.CL (22)](#cscl-22)
- [stat.ME (1)](#statme-1)
- [cs.DC (2)](#csdc-2)
- [math.OC (1)](#mathoc-1)
- [eess.AS (3)](#eessas-3)
- [cs.IT (1)](#csit-1)
- [q-bio.GN (1)](#q-biogn-1)
- [eess.SY (1)](#eesssy-1)
- [quant-ph (1)](#quant-ph-1)
- [econ.GN (1)](#econgn-1)
- [cs.RO (1)](#csro-1)

## cs.DS (2)



### (1/97) Efficient Algorithms for Attributed Graph Alignment with Vanishing Edge Correlation (Ziao Wang et al., 2023)

{{<citation>}}

Ziao Wang, Weina Wang, Lele Wang. (2023)  
**Efficient Algorithms for Attributed Graph Alignment with Vanishing Edge Correlation**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs-IT, cs.DS, math-IT  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2308.09210v1)  

---


**ABSTRACT**  
Graph alignment refers to the task of finding the vertex correspondence between two positively correlated graphs. Extensive study has been done on polynomial-time algorithms for the graph alignment problem under the Erd\H{o}s--R\'enyi graph pair model, where the two graphs are Erd\H{o}s--R\'enyi graphs with edge probability $q_\mathrm{u}$, correlated under certain vertex correspondence. To achieve exact recovery of the vertex correspondence, all existing algorithms at least require the edge correlation coefficient $\rho_\mathrm{u}$ between the two graphs to satisfy $\rho_\mathrm{u} > \sqrt{\alpha}$, where $\alpha \approx 0.338$ is Otter's tree-counting constant. Moreover, it is conjectured in [1] that no polynomial-time algorithm can achieve exact recovery under weak edge correlation $\rho_\mathrm{u}<\sqrt{\alpha}$.   In this paper, we show that with a vanishing amount of additional attribute information, exact recovery is polynomial-time feasible under vanishing edge correlation $\rho_\mathrm{u} \ge n^{-\Theta(1)}$. We identify a local tree structure, which incorporates one layer of user information and one layer of attribute information, and apply the subgraph counting technique to such structures. A polynomial-time algorithm is proposed that recovers the vertex correspondence for all but a vanishing fraction of vertices. We then further refine the algorithm output to achieve exact recovery. The motivation for considering additional attribute information comes from the widely available side information in real-world applications, such as the user's birthplace and educational background on LinkedIn and Twitter social networks.

{{</citation>}}


### (2/97) Mitigating Misinformation Spreading in Social Networks Via Edge Blocking (Ahad N. Zehmakan et al., 2023)

{{<citation>}}

Ahad N. Zehmakan, Khushvind Maurya. (2023)  
**Mitigating Misinformation Spreading in Social Networks Via Edge Blocking**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs-SI, cs.DS  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2308.08860v1)  

---


**ABSTRACT**  
The wide adoption of social media platforms has brought about numerous benefits for communication and information sharing. However, it has also led to the rapid spread of misinformation, causing significant harm to individuals, communities, and society at large. Consequently, there has been a growing interest in devising efficient and effective strategies to contain the spread of misinformation. One popular countermeasure is blocking edges in the underlying network.   We model the spread of misinformation using the classical Independent Cascade model and study the problem of minimizing the spread by blocking a given number of edges. We prove that this problem is computationally hard, but we propose an intuitive community-based algorithm, which aims to detect well-connected communities in the network and disconnect the inter-community edges. Our experiments on various real-world social networks demonstrate that the proposed algorithm significantly outperforms the prior methods, which mostly rely on centrality measures.

{{</citation>}}


## cs.IR (1)



### (3/97) A Model-Agnostic Framework for Recommendation via Interest-aware Item Embeddings (Amit Kumar Jaiswal et al., 2023)

{{<citation>}}

Amit Kumar Jaiswal, Yu Xiong. (2023)  
**A Model-Agnostic Framework for Recommendation via Interest-aware Item Embeddings**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.09202v1)  

---


**ABSTRACT**  
Item representation holds significant importance in recommendation systems, which encompasses domains such as news, retail, and videos. Retrieval and ranking models utilise item representation to capture the user-item relationship based on user behaviours. While existing representation learning methods primarily focus on optimising item-based mechanisms, such as attention and sequential modelling. However, these methods lack a modelling mechanism to directly reflect user interests within the learned item representations. Consequently, these methods may be less effective in capturing user interests indirectly. To address this challenge, we propose a novel Interest-aware Capsule network (IaCN) recommendation model, a model-agnostic framework that directly learns interest-oriented item representations. IaCN serves as an auxiliary task, enabling the joint learning of both item-based and interest-based representations. This framework adopts existing recommendation models without requiring substantial redesign. We evaluate the proposed approach on benchmark datasets, exploring various scenarios involving different deep neural networks, behaviour sequence lengths, and joint learning ratios of interest-oriented item representations. Experimental results demonstrate significant performance enhancements across diverse recommendation models, validating the effectiveness of our approach.

{{</citation>}}


## cs.SE (5)



### (4/97) A Comparative Study of Text Embedding Models for Semantic Text Similarity in Bug Reports (Avinash Patil et al., 2023)

{{<citation>}}

Avinash Patil, Kihwan Han, Sabyasachi Mukhopadhyay. (2023)  
**A Comparative Study of Text Embedding Models for Semantic Text Similarity in Bug Reports**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-LG, cs-SE, cs.SE  
Keywords: BERT, Embedding  
[Paper Link](http://arxiv.org/abs/2308.09193v1)  

---


**ABSTRACT**  
Bug reports are an essential aspect of software development, and it is crucial to identify and resolve them quickly to ensure the consistent functioning of software systems. Retrieving similar bug reports from an existing database can help reduce the time and effort required to resolve bugs. In this paper, we compared the effectiveness of semantic textual similarity methods for retrieving similar bug reports based on a similarity score. We explored several embedding models such as TF-IDF (Baseline), FastText, Gensim, BERT, and ADA. We used the Software Defects Data containing bug reports for various software projects to evaluate the performance of these models. Our experimental results showed that BERT generally outperformed the rest of the models regarding recall, followed by ADA, Gensim, FastText, and TFIDF. Our study provides insights into the effectiveness of different embedding methods for retrieving similar bug reports and highlights the impact of selecting the appropriate one for this task. Our code is available on GitHub.

{{</citation>}}


### (5/97) Enhancing API Documentation through BERTopic Modeling and Summarization (AmirHossein Naghshzan et al., 2023)

{{<citation>}}

AmirHossein Naghshzan, Sylvie Ratte. (2023)  
**Enhancing API Documentation through BERTopic Modeling and Summarization**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-LG, cs-SE, cs.SE  
Keywords: BERT, NLP, Natural Language Processing, Summarization, Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2308.09070v1)  

---


**ABSTRACT**  
As the amount of textual data in various fields, including software development, continues to grow, there is a pressing demand for efficient and effective extraction and presentation of meaningful insights. This paper presents a unique approach to address this need, focusing on the complexities of interpreting Application Programming Interface (API) documentation. While official API documentation serves as a primary source of information for developers, it can often be extensive and lacks user-friendliness. In light of this, developers frequently resort to unofficial sources like Stack Overflow and GitHub. Our novel approach employs the strengths of BERTopic for topic modeling and Natural Language Processing (NLP) to automatically generate summaries of API documentation, thereby creating a more efficient method for developers to extract the information they need. The produced summaries and topics are evaluated based on their performance, coherence, and interoperability.   The findings of this research contribute to the field of API documentation analysis by providing insights into recurring topics, identifying common issues, and generating potential solutions. By improving the accessibility and efficiency of API documentation comprehension, our work aims to enhance the software development process and empower developers with practical tools for navigating complex APIs.

{{</citation>}}


### (6/97) Towards Automatically Addressing Self-Admitted Technical Debt: How Far Are We? (Antonio Mastropaolo et al., 2023)

{{<citation>}}

Antonio Mastropaolo, Massimiliano Di Penta, Gabriele Bavota. (2023)  
**Towards Automatically Addressing Self-Admitted Technical Debt: How Far Are We?**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.08943v1)  

---


**ABSTRACT**  
Upon evolving their software, organizations and individual developers have to spend a substantial effort to pay back technical debt, i.e., the fact that software is released in a shape not as good as it should be, e.g., in terms of functionality, reliability, or maintainability. This paper empirically investigates the extent to which technical debt can be automatically paid back by neural-based generative models, and in particular models exploiting different strategies for pre-training and fine-tuning. We start by extracting a dateset of 5,039 Self-Admitted Technical Debt (SATD) removals from 595 open-source projects. SATD refers to technical debt instances documented (e.g., via code comments) by developers. We use this dataset to experiment with seven different generative deep learning (DL) model configurations. Specifically, we compare transformers pre-trained and fine-tuned with different combinations of training objectives, including the fixing of generic code changes, SATD removals, and SATD-comment prompt tuning. Also, we investigate the applicability in this context of a recently-available Large Language Model (LLM)-based chat bot. Results of our study indicate that the automated repayment of SATD is a challenging task, with the best model we experimented with able to automatically fix ~2% to 8% of test instances, depending on the number of attempts it is allowed to make. Given the limited size of the fine-tuning dataset (~5k instances), the model's pre-training plays a fundamental role in boosting performance. Also, the ability to remove SATD steadily drops if the comment documenting the SATD is not provided as input to the model. Finally, we found general-purpose LLMs to not be a competitive approach for addressing SATD.

{{</citation>}}


### (7/97) CodeCoT and Beyond: Learning to Program and Test like a Developer (Dong Huang et al., 2023)

{{<citation>}}

Dong Huang, Qingwen Bu, Heming Cui. (2023)  
**CodeCoT and Beyond: Learning to Program and Test like a Developer**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2308.08784v1)  

---


**ABSTRACT**  
In natural language processing, transformer-based large language models (LLMs) like GPT-x models developed by OpenAI have revolutionized the landscape. Despite their impressive capabilities, these models often encounter challenges when handling tasks that differ from their training data, resulting in compromised performance. To address this, few-shot learning has emerged as a valuable technique, allowing LLMs to adapt with minimal task-specific data. One innovative strategy, known as Chain-of-Thought Prompting (CoT), has been introduced to guide LLMs in revealing cognitive processes during multi-step reasoning. In this paper, we propose Code Chain-of-Thought~(CodeCoT), which consists of two components: the Vanilla CodeCoT and the Self-exam CodeCoT. The latter incorporates self-examination, empowering the model to iteratively generate code, formulate test cases, and refine its outputs. Specifically, the process entails the generation of test examples by the model corresponding to the code it is tasked to implement. If it fails on the test examples, then it regenerates the code based on the erroneous code and associated error types. Through comprehensive experiments, we observed that both techniques significantly enhance code generation accuracy across various LLM variants. Our evaluation results reveal that CodeCoT improves the code generation effectiveness, including an unprecedented pass@1 accuracy of 79.27\% using the Self-exam CodeCoT approach on the gpt-3.5-turbo-0613 model in the HumanEval dataset.

{{</citation>}}


### (8/97) On the Effectiveness of Log Representation for Log-based Anomaly Detection (Xingfang Wu et al., 2023)

{{<citation>}}

Xingfang Wu, Heng Li, Foutse Khomh. (2023)  
**On the Effectiveness of Log Representation for Log-based Anomaly Detection**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.08736v1)  

---


**ABSTRACT**  
Logs are an essential source of information for people to understand the running status of a software system. Due to the evolving modern software architecture and maintenance methods, more research efforts have been devoted to automated log analysis. In particular, machine learning (ML) has been widely used in log analysis tasks. In ML-based log analysis tasks, converting textual log data into numerical feature vectors is a critical and indispensable step. However, the impact of using different log representation techniques on the performance of the downstream models is not clear, which limits researchers and practitioners' opportunities of choosing the optimal log representation techniques in their automated log analysis workflows. Therefore, this work investigates and compares the commonly adopted log representation techniques from previous log analysis research. Particularly, we select six log representation techniques and evaluate them with seven ML models and four public log datasets (i.e., HDFS, BGL, Spirit and Thunderbird) in the context of log-based anomaly detection. We also examine the impacts of the log parsing process and the different feature aggregation approaches when they are employed with log representation techniques. From the experiments, we provide some heuristic guidelines for future researchers and developers to follow when designing an automated log analysis workflow. We believe our comprehensive comparison of log representation techniques can help researchers and practitioners better understand the characteristics of different log representation techniques and provide them with guidance for selecting the most suitable ones for their ML-based log analysis workflow.

{{</citation>}}


## cs.CR (5)



### (9/97) RatGPT: Turning online LLMs into Proxies for Malware Attacks (Mika Beckerich et al., 2023)

{{<citation>}}

Mika Beckerich, Laura Plein, Sergio Coronado. (2023)  
**RatGPT: Turning online LLMs into Proxies for Malware Attacks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI, ChatGPT, GPT, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.09183v1)  

---


**ABSTRACT**  
The evolution of Generative AI and the capabilities of the newly released Large Language Models (LLMs) open new opportunities in software engineering. However, they also lead to new challenges in cybersecurity. Recently, researchers have shown the possibilities of using LLMs such as ChatGPT to generate malicious content that can directly be exploited or guide inexperienced hackers to weaponize tools and code. Those studies covered scenarios that still require the attacker in the middle of the loop. In this study, we leverage openly available plugins and use an LLM as proxy between the attacker and the victim. We deliver a proof-of-concept where ChatGPT is used for the dissemination of malicious software while evading detection, alongside establishing the communication to a command and control (C2) server to receive commands to interact with a victim's system. Finally, we present the general approach as well as essential elements in order to stay undetected and make the attack a success. This proof-of-concept highlights significant cybersecurity issues with openly available plugins and LLMs, which require the development of security guidelines, controls, and mitigation strategies.

{{</citation>}}


### (10/97) Forensic Data Analytics for Anomaly Detection in Evolving Networks (Li Yang et al., 2023)

{{<citation>}}

Li Yang, Abdallah Moubayed, Abdallah Shami, Amine Boukhtouta, Parisa Heidari, Stere Preda, Richard Brunner, Daniel Migault, Adel Larabi. (2023)  
**Forensic Data Analytics for Anomaly Detection in Evolving Networks**  

---
Primary Category: cs.CR  
Categories: 68T01, I-2-6; C-2-0, cs-AI, cs-CR, cs-LG, cs-NI, cs.CR  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.09171v1)  

---


**ABSTRACT**  
In the prevailing convergence of traditional infrastructure-based deployment (i.e., Telco and industry operational networks) towards evolving deployments enabled by 5G and virtualization, there is a keen interest in elaborating effective security controls to protect these deployments in-depth. By considering key enabling technologies like 5G and virtualization, evolving networks are democratized, facilitating the establishment of point presences integrating different business models ranging from media, dynamic web content, gaming, and a plethora of IoT use cases. Despite the increasing services provided by evolving networks, many cybercrimes and attacks have been launched in evolving networks to perform malicious activities. Due to the limitations of traditional security artifacts (e.g., firewalls and intrusion detection systems), the research on digital forensic data analytics has attracted more attention. Digital forensic analytics enables people to derive detailed information and comprehensive conclusions from different perspectives of cybercrimes to assist in convicting criminals and preventing future crimes. This chapter presents a digital analytics framework for network anomaly detection, including multi-perspective feature engineering, unsupervised anomaly detection, and comprehensive result correction procedures. Experiments on real-world evolving network data show the effectiveness of the proposed forensic data analytics solution.

{{</citation>}}


### (11/97) Watch Out! Smartwatches as criminal tool and digital forensic investigations (Seungjae Jeon et al., 2023)

{{<citation>}}

Seungjae Jeon, Jaehyun Chung, Doowon Jeong. (2023)  
**Watch Out! Smartwatches as criminal tool and digital forensic investigations**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.09092v1)  

---


**ABSTRACT**  
In the rapidly advancing technological landscape, smartwatches have materialized as multifunctional devices integral to our daily routines. Smartwatches store a substantial amount of personal information, potentially serving as repositories of digital evidence. Thus, digital forensic researchers have devoted considerable effort to exploring smartwatch forensic techniques. However, it has been observed that prior studies have primarily treated smartwatches as mere storage mediums for digital evidence, neglecting their potential role in criminal activities. This paper presents the information leakage perpetrated through smartwatches. We represent crime scenarios in an environment where smartphones are not available, considering that the perception that smartphones can be used as tools for criminal behavior prevails in many organizations, while the potential of similar-use smartwatches is often overlooked. We detail mechanisms for information leakage via file transfer and camera control using smartwatches. Additionally, we present methods to investigate each crime incident through smartwatch forensics. Finally, we describe the limitations of post-incident responses and propose proactive measures to prepare for potential crimes involving smartwatches. Keywords: Information Leakage, Smartwatch Forensics, Android Forensics, Mobile Device Management, Security Policy

{{</citation>}}


### (12/97) Smart Bulbs can be Hacked to Hack into your Household (Davide Bonaventura et al., 2023)

{{<citation>}}

Davide Bonaventura, Sergio Esposito, Giampaolo Bella. (2023)  
**Smart Bulbs can be Hacked to Hack into your Household**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2308.09019v1)  

---


**ABSTRACT**  
The IoT is getting more and more pervasive. Even the simplest devices, such as a light bulb or an electrical plug, are made "smart" and controllable by our smartphone. This paper describes the findings obtained by applying the PETIoT kill chain to conduct a Vulnerability Assessment and Penetration Testing session on a smart bulb, the Tapo L530E by Tp-Link, currently best seller on Amazon Italy. We found that four vulnerabilities affect the bulb, two of High severity and two of Medium severity according to the CVSS v3.1 scoring system. In short, authentication is not well accounted for and confidentiality is insufficiently achieved by the implemented cryptographic measures. In consequence, an attacker who is nearby the bulb can operate at will not just the bulb but all devices of the Tapo family that the user may have on her Tapo account. Moreover, the attacker can learn the victim's Wi-Fi password, thereby escalating his malicious potential considerably. The paper terminates with an outline of possible fixes.

{{</citation>}}


### (13/97) Towards a Practical Defense against Adversarial Attacks on Deep Learning-based Malware Detectors via Randomized Smoothing (Daniel Gibert et al., 2023)

{{<citation>}}

Daniel Gibert, Giulio Zizzo, Quan Le. (2023)  
**Towards a Practical Defense against Adversarial Attacks on Deep Learning-based Malware Detectors via Randomized Smoothing**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2308.08906v1)  

---


**ABSTRACT**  
Malware detectors based on deep learning (DL) have been shown to be susceptible to malware examples that have been deliberately manipulated in order to evade detection, a.k.a. adversarial malware examples. More specifically, it has been show that deep learning detectors are vulnerable to small changes on the input file. Given this vulnerability of deep learning detectors, we propose a practical defense against adversarial malware examples inspired by randomized smoothing. In our work, instead of employing Gaussian or Laplace noise when randomizing inputs, we propose a randomized ablation-based smoothing scheme that ablates a percentage of the bytes within an executable. During training, our randomized ablation-based smoothing scheme trains a base classifier based on ablated versions of the executable files. At test time, the final classification for a given input executable is taken as the class most commonly predicted by the classifier on a set of ablated versions of the original executable. To demonstrate the suitability of our approach we have empirically evaluated the proposed ablation-based model against various state-of-the-art evasion attacks on the BODMAS dataset. Results show greater robustness and generalization capabilities to adversarial malware examples in comparison to a non-smoothed classifier.

{{</citation>}}


## cs.AI (5)



### (14/97) ChatGPT-HealthPrompt. Harnessing the Power of XAI in Prompt-Based Healthcare Decision Support using ChatGPT (Fatemeh Nazary et al., 2023)

{{<citation>}}

Fatemeh Nazary, Yashar Deldjoo, Tommaso Di Noia. (2023)  
**ChatGPT-HealthPrompt. Harnessing the Power of XAI in Prompt-Based Healthcare Decision Support using ChatGPT**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.09731v1)  

---


**ABSTRACT**  
This study presents an innovative approach to the application of large language models (LLMs) in clinical decision-making, focusing on OpenAI's ChatGPT. Our approach introduces the use of contextual prompts-strategically designed to include task description, feature description, and crucially, integration of domain knowledge-for high-quality binary classification tasks even in data-scarce scenarios. The novelty of our work lies in the utilization of domain knowledge, obtained from high-performing interpretable ML models, and its seamless incorporation into prompt design. By viewing these ML models as medical experts, we extract key insights on feature importance to aid in decision-making processes. This interplay of domain knowledge and AI holds significant promise in creating a more insightful diagnostic tool.   Additionally, our research explores the dynamics of zero-shot and few-shot prompt learning based on LLMs. By comparing the performance of OpenAI's ChatGPT with traditional supervised ML models in different data conditions, we aim to provide insights into the effectiveness of prompt engineering strategies under varied data availability. In essence, this paper bridges the gap between AI and healthcare, proposing a novel methodology for LLMs application in clinical decision support systems. It highlights the transformative potential of effective prompt design, domain knowledge integration, and flexible learning approaches in enhancing automated decision-making.

{{</citation>}}


### (15/97) Diversifying AI: Towards Creative Chess with AlphaZero (Tom Zahavy et al., 2023)

{{<citation>}}

Tom Zahavy, Vivek Veeriah, Shaobo Hou, Kevin Waugh, Matthew Lai, Edouard Leurent, Nenad Tomasev, Lisa Schut, Demis Hassabis, Satinder Singh. (2023)  
**Diversifying AI: Towards Creative Chess with AlphaZero**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09175v1)  

---


**ABSTRACT**  
In recent years, Artificial Intelligence (AI) systems have surpassed human intelligence in a variety of computational tasks. However, AI systems, like humans, make mistakes, have blind spots, hallucinate, and struggle to generalize to new situations. This work explores whether AI can benefit from creative decision-making mechanisms when pushed to the limits of its computational rationality. In particular, we investigate whether a team of diverse AI systems can outperform a single AI in challenging tasks by generating more ideas as a group and then selecting the best ones. We study this question in the game of chess, the so-called drosophila of AI. We build on AlphaZero (AZ) and extend it to represent a league of agents via a latent-conditioned architecture, which we call AZ_db. We train AZ_db to generate a wider range of ideas using behavioral diversity techniques and select the most promising ones with sub-additive planning. Our experiments suggest that AZ_db plays chess in diverse ways, solves more puzzles as a group and outperforms a more homogeneous team. Notably, AZ_db solves twice as many challenging puzzles as AZ, including the challenging Penrose positions. When playing chess from different openings, we notice that players in AZ_db specialize in different openings, and that selecting a player for each opening using sub-additive planning results in a 50 Elo improvement over AZ. Our findings suggest that diversity bonuses emerge in teams of AI agents, just as they do in teams of humans and that diversity is a valuable asset in solving computationally hard problems.

{{</citation>}}


### (16/97) MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models (Yilin Wen et al., 2023)

{{<citation>}}

Yilin Wen, Zifeng Wang, Jimeng Sun. (2023)  
**MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: GPT, GPT-3.5, GPT-4, Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2308.09729v1)  

---


**ABSTRACT**  
LLMs usually exhibit limitations in their ability to incorporate new knowledge, the generation of hallucinations, and the transparency of their decision-making process. In this paper, we explore how to prompt LLMs with knowledge graphs (KG), working as a remedy to engage LLMs with up-to-date knowledge and elicit the reasoning pathways from LLMs. Specifically, we build a prompting pipeline that endows LLMs with the capability of comprehending KG inputs and inferring with a combined implicit knowledge and the retrieved external knowledge. In addition, we investigate eliciting the mind map on which LLMs perform the reasoning and generate the answers. It is identified that the produced mind map exhibits the reasoning pathways of LLMs grounded on the ontology of knowledge, hence bringing the prospects of probing and gauging LLM inference in production. The experiments on three question & answering datasets also show that MindMap prompting leads to a striking empirical gain. For instance, prompting a GPT-3.5 with MindMap yields an overwhelming performance over GPT-4 consistently. We also demonstrate that with structured facts retrieved from KG, MindMap can outperform a series of prompting-with-document-retrieval methods, benefiting from more accurate, concise, and comprehensive knowledge from KGs.

{{</citation>}}


### (17/97) Probabilistic Results on the Architecture of Mathematical Reasoning Aligned by Cognitive Alternation (Minzheng Li et al., 2023)

{{<citation>}}

Minzheng Li, Xiangzhong Fang, Haixin Yang. (2023)  
**Probabilistic Results on the Architecture of Mathematical Reasoning Aligned by Cognitive Alternation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, math-PR  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.08714v1)  

---


**ABSTRACT**  
We envision a machine capable of solving mathematical problems. Dividing the quantitative reasoning system into two parts: thought processes and cognitive processes, we provide probabilistic descriptions of the architecture.

{{</citation>}}


### (18/97) Consciousness in Artificial Intelligence: Insights from the Science of Consciousness (Patrick Butlin et al., 2023)

{{<citation>}}

Patrick Butlin, Robert Long, Eric Elmoznino, Yoshua Bengio, Jonathan Birch, Axel Constant, George Deane, Stephen M. Fleming, Chris Frith, Xu Ji, Ryota Kanai, Colin Klein, Grace Lindsay, Matthias Michel, Liad Mudrik, Megan A. K. Peters, Eric Schwitzgebel, Jonathan Simon, Rufin VanRullen. (2023)  
**Consciousness in Artificial Intelligence: Insights from the Science of Consciousness**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-LG, cs.AI, q-bio-NC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08708v2)  

---


**ABSTRACT**  
Whether current or near-term AI systems could be conscious is a topic of scientific interest and increasing public concern. This report argues for, and exemplifies, a rigorous and empirically grounded approach to AI consciousness: assessing existing AI systems in detail, in light of our best-supported neuroscientific theories of consciousness. We survey several prominent scientific theories of consciousness, including recurrent processing theory, global workspace theory, higher-order theories, predictive processing, and attention schema theory. From these theories we derive "indicator properties" of consciousness, elucidated in computational terms that allow us to assess AI systems for these properties. We use these indicator properties to assess several recent AI systems, and we discuss how future systems might implement them. Our analysis suggests that no current AI systems are conscious, but also shows that there are no obvious barriers to building conscious AI systems.

{{</citation>}}


## cs.CV (29)



### (19/97) How Does Pruning Impact Long-Tailed Multi-Label Medical Image Classifiers? (Gregory Holste et al., 2023)

{{<citation>}}

Gregory Holste, Ziyu Jiang, Ajay Jaiswal, Maria Hanna, Shlomo Minkowitz, Alan C. Legasto, Joanna G. Escalon, Sharon Steinberger, Mark Bittman, Thomas C. Shen, Ying Ding, Ronald M. Summers, George Shih, Yifan Peng, Zhangyang Wang. (2023)  
**How Does Pruning Impact Long-Tailed Multi-Label Medical Image Classifiers?**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.09180v1)  

---


**ABSTRACT**  
Pruning has emerged as a powerful technique for compressing deep neural networks, reducing memory usage and inference time without significantly affecting overall performance. However, the nuanced ways in which pruning impacts model behavior are not well understood, particularly for long-tailed, multi-label datasets commonly found in clinical settings. This knowledge gap could have dangerous implications when deploying a pruned model for diagnosis, where unexpected model behavior could impact patient well-being. To fill this gap, we perform the first analysis of pruning's effect on neural networks trained to diagnose thorax diseases from chest X-rays (CXRs). On two large CXR datasets, we examine which diseases are most affected by pruning and characterize class "forgettability" based on disease frequency and co-occurrence behavior. Further, we identify individual CXRs where uncompressed and heavily pruned models disagree, known as pruning-identified exemplars (PIEs), and conduct a human reader study to evaluate their unifying qualities. We find that radiologists perceive PIEs as having more label noise, lower image quality, and higher diagnosis difficulty. This work represents a first step toward understanding the impact of pruning on model behavior in deep long-tailed, multi-label medical image classification. All code, model weights, and data access instructions can be found at https://github.com/VITA-Group/PruneCXR.

{{</citation>}}


### (20/97) FedPerfix: Towards Partial Model Personalization of Vision Transformers in Federated Learning (Guangyu Sun et al., 2023)

{{<citation>}}

Guangyu Sun, Matias Mendieta, Jun Luo, Shandong Wu, Chen Chen. (2023)  
**FedPerfix: Towards Partial Model Personalization of Vision Transformers in Federated Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.09160v1)  

---


**ABSTRACT**  
Personalized Federated Learning (PFL) represents a promising solution for decentralized learning in heterogeneous data environments. Partial model personalization has been proposed to improve the efficiency of PFL by selectively updating local model parameters instead of aggregating all of them. However, previous work on partial model personalization has mainly focused on Convolutional Neural Networks (CNNs), leaving a gap in understanding how it can be applied to other popular models such as Vision Transformers (ViTs). In this work, we investigate where and how to partially personalize a ViT model. Specifically, we empirically evaluate the sensitivity to data distribution of each type of layer. Based on the insights that the self-attention layer and the classification head are the most sensitive parts of a ViT, we propose a novel approach called FedPerfix, which leverages plugins to transfer information from the aggregated model to the local client as a personalization. Finally, we evaluate the proposed approach on CIFAR-100, OrganAMNIST, and Office-Home datasets and demonstrate its effectiveness in improving the model's performance compared to several advanced PFL methods.

{{</citation>}}


### (21/97) EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding (Karttikeya Mangalam et al., 2023)

{{<citation>}}

Karttikeya Mangalam, Raiymbek Akshulakov, Jitendra Malik. (2023)  
**EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.09126v1)  

---


**ABSTRACT**  
We introduce EgoSchema, a very long-form video question-answering dataset, and benchmark to evaluate long video understanding capabilities of modern vision and language systems. Derived from Ego4D, EgoSchema consists of over 5000 human curated multiple choice question answer pairs, spanning over 250 hours of real video data, covering a very broad range of natural human activity and behavior. For each question, EgoSchema requires the correct answer to be selected between five given options based on a three-minute-long video clip. While some prior works have proposed video datasets with long clip lengths, we posit that merely the length of the video clip does not truly capture the temporal difficulty of the video task that is being considered. To remedy this, we introduce temporal certificate sets, a general notion for capturing the intrinsic temporal understanding length associated with a broad range of video understanding tasks & datasets. Based on this metric, we find EgoSchema to have intrinsic temporal lengths over 5.7x longer than the second closest dataset and 10x to 100x longer than any other video understanding dataset. Further, our evaluation of several current state-of-the-art video and language models shows them to be severely lacking in long-term video understanding capabilities. Even models with several billions of parameters achieve QA accuracy less than 33% (random is 20%) on the EgoSchema multi-choice question answering task, while humans achieve about 76% accuracy. We posit that \name{}{}, with its long intrinsic temporal structures and diverse complexity, would serve as a valuable evaluation probe for developing effective long-term video understanding systems in the future. Data and Zero-shot model evaluation code are open-sourced for both public and commercial use under the Ego4D license at http://egoschema.github.io

{{</citation>}}


### (22/97) ICAR: Image-based Complementary Auto Reasoning (Xijun Wang et al., 2023)

{{<citation>}}

Xijun Wang, Anqi Liang, Junbang Liang, Ming Lin, Yu Lou, Shan Yang. (2023)  
**ICAR: Image-based Complementary Auto Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09119v1)  

---


**ABSTRACT**  
Scene-aware Complementary Item Retrieval (CIR) is a challenging task which requires to generate a set of compatible items across domains. Due to the subjectivity, it is difficult to set up a rigorous standard for both data collection and learning objectives. To address this challenging task, we propose a visual compatibility concept, composed of similarity (resembling in color, geometry, texture, and etc.) and complementarity (different items like table vs chair completing a group). Based on this notion, we propose a compatibility learning framework, a category-aware Flexible Bidirectional Transformer (FBT), for visual "scene-based set compatibility reasoning" with the cross-domain visual similarity input and auto-regressive complementary item generation. We introduce a "Flexible Bidirectional Transformer (FBT)" consisting of an encoder with flexible masking, a category prediction arm, and an auto-regressive visual embedding prediction arm. And the inputs for FBT are cross-domain visual similarity invariant embeddings, making this framework quite generalizable. Furthermore, our proposed FBT model learns the inter-object compatibility from a large set of scene images in a self-supervised way. Compared with the SOTA methods, this approach achieves up to 5.3% and 9.6% in FITB score and 22.3% and 31.8% SFID improvement on fashion and furniture, respectively.

{{</citation>}}


### (23/97) Learning Lightweight Object Detectors via Multi-Teacher Progressive Distillation (Shengcao Cao et al., 2023)

{{<citation>}}

Shengcao Cao, Mengtian Li, James Hays, Deva Ramanan, Yi-Xiong Wang, Liang-Yan Gui. (2023)  
**Learning Lightweight Object Detectors via Multi-Teacher Progressive Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09105v1)  

---


**ABSTRACT**  
Resource-constrained perception systems such as edge computing and vision-for-robotics require vision models to be both accurate and lightweight in computation and memory usage. While knowledge distillation is a proven strategy to enhance the performance of lightweight classification models, its application to structured outputs like object detection and instance segmentation remains a complicated task, due to the variability in outputs and complex internal network modules involved in the distillation process. In this paper, we propose a simple yet surprisingly effective sequential approach to knowledge distillation that progressively transfers the knowledge of a set of teacher detectors to a given lightweight student. To distill knowledge from a highly accurate but complex teacher model, we construct a sequence of teachers to help the student gradually adapt. Our progressive strategy can be easily combined with existing detection distillation mechanisms to consistently maximize student performance in various settings. To the best of our knowledge, we are the first to successfully distill knowledge from Transformer-based teacher detectors to convolution-based students, and unprecedentedly boost the performance of ResNet-50 based RetinaNet from 36.5% to 42.0% AP and Mask R-CNN from 38.2% to 42.5% AP on the MS COCO benchmark.

{{</citation>}}


### (24/97) ImGeoNet: Image-induced Geometry-aware Voxel Representation for Multi-view 3D Object Detection (Tao Tu et al., 2023)

{{<citation>}}

Tao Tu, Shun-Po Chuang, Yu-Lun Liu, Cheng Sun, Ke Zhang, Donna Roy, Cheng-Hao Kuo, Min Sun. (2023)  
**ImGeoNet: Image-induced Geometry-aware Voxel Representation for Multi-view 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.09098v1)  

---


**ABSTRACT**  
We propose ImGeoNet, a multi-view image-based 3D object detection framework that models a 3D space by an image-induced geometry-aware voxel representation. Unlike previous methods which aggregate 2D features into 3D voxels without considering geometry, ImGeoNet learns to induce geometry from multi-view images to alleviate the confusion arising from voxels of free space, and during the inference phase, only images from multiple views are required. Besides, a powerful pre-trained 2D feature extractor can be leveraged by our representation, leading to a more robust performance. To evaluate the effectiveness of ImGeoNet, we conduct quantitative and qualitative experiments on three indoor datasets, namely ARKitScenes, ScanNetV2, and ScanNet200. The results demonstrate that ImGeoNet outperforms the current state-of-the-art multi-view image-based method, ImVoxelNet, on all three datasets in terms of detection accuracy. In addition, ImGeoNet shows great data efficiency by achieving results comparable to ImVoxelNet with 100 views while utilizing only 40 views. Furthermore, our studies indicate that our proposed image-induced geometry-aware representation can enable image-based methods to attain superior detection accuracy than the seminal point cloud-based method, VoteNet, in two practical scenarios: (1) scenarios where point clouds are sparse and noisy, such as in ARKitScenes, and (2) scenarios involve diverse object classes, particularly classes of small objects, as in the case in ScanNet200.

{{</citation>}}


### (25/97) Identity-Aware Semi-Supervised Learning for Comic Character Re-Identification (Gürkan Soykan et al., 2023)

{{<citation>}}

Gürkan Soykan, Deniz Yuret, Tevfik Metin Sezgin. (2023)  
**Identity-Aware Semi-Supervised Learning for Comic Character Re-Identification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-IR, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.09096v1)  

---


**ABSTRACT**  
Character re-identification, recognizing characters consistently across different panels in comics, presents significant challenges due to limited annotated data and complex variations in character appearances. To tackle this issue, we introduce a robust semi-supervised framework that combines metric learning with a novel 'Identity-Aware' self-supervision method by contrastive learning of face and body pairs of characters. Our approach involves processing both facial and bodily features within a unified network architecture, facilitating the extraction of identity-aligned character embeddings that capture individual identities while preserving the effectiveness of face and body features. This integrated character representation enhances feature extraction and improves character re-identification compared to re-identification by face or body independently, offering a parameter-efficient solution. By extensively validating our method using in-series and inter-series evaluation metrics, we demonstrate its effectiveness in consistently re-identifying comic characters. Compared to existing methods, our approach not only addresses the challenge of character re-identification but also serves as a foundation for downstream tasks since it can produce character embeddings without restrictions of face and body availability, enriching the comprehension of comic books. In our experiments, we leverage two newly curated datasets: the 'Comic Character Instances Dataset', comprising over a million character instances and the 'Comic Sequence Identity Dataset', containing annotations of identities within more than 3000 sets of four consecutive comic panels that we collected.

{{</citation>}}


### (26/97) SimFIR: A Simple Framework for Fisheye Image Rectification with Self-supervised Representation Learning (Hao Feng et al., 2023)

{{<citation>}}

Hao Feng, Wendi Wang, Jiajun Deng, Wengang Zhou, Li Li, Houqiang Li. (2023)  
**SimFIR: A Simple Framework for Fisheye Image Rectification with Self-supervised Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09040v1)  

---


**ABSTRACT**  
In fisheye images, rich distinct distortion patterns are regularly distributed in the image plane. These distortion patterns are independent of the visual content and provide informative cues for rectification. To make the best of such rectification cues, we introduce SimFIR, a simple framework for fisheye image rectification based on self-supervised representation learning. Technically, we first split a fisheye image into multiple patches and extract their representations with a Vision Transformer (ViT). To learn fine-grained distortion representations, we then associate different image patches with their specific distortion patterns based on the fisheye model, and further subtly design an innovative unified distortion-aware pretext task for their learning. The transfer performance on the downstream rectification task is remarkably boosted, which verifies the effectiveness of the learned representations. Extensive experiments are conducted, and the quantitative and qualitative results demonstrate the superiority of our method over the state-of-the-art algorithms as well as its strong generalization ability on real-world fisheye images.

{{</citation>}}


### (27/97) MarginMatch: Improving Semi-Supervised Learning with Pseudo-Margins (Tiberiu Sosea et al., 2023)

{{<citation>}}

Tiberiu Sosea, Cornelia Caragea. (2023)  
**MarginMatch: Improving Semi-Supervised Learning with Pseudo-Margins**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.09037v1)  

---


**ABSTRACT**  
We introduce MarginMatch, a new SSL approach combining consistency regularization and pseudo-labeling, with its main novelty arising from the use of unlabeled data training dynamics to measure pseudo-label quality. Instead of using only the model's confidence on an unlabeled example at an arbitrary iteration to decide if the example should be masked or not, MarginMatch also analyzes the behavior of the model on the pseudo-labeled examples as the training progresses, to ensure low quality predictions are masked out. MarginMatch brings substantial improvements on four vision benchmarks in low data regimes and on two large-scale datasets, emphasizing the importance of enforcing high-quality pseudo-labels. Notably, we obtain an improvement in error rate over the state-of-the-art of 3.25% on CIFAR-100 with only 25 labels per class and of 3.78% on STL-10 using as few as 4 labels per class. We make our code available at https://github.com/tsosea2/MarginMatch.

{{</citation>}}


### (28/97) Uni-NLX: Unifying Textual Explanations for Vision and Vision-Language Tasks (Fawaz Sammani et al., 2023)

{{<citation>}}

Fawaz Sammani, Nikos Deligiannis. (2023)  
**Uni-NLX: Unifying Textual Explanations for Vision and Vision-Language Tasks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.09033v1)  

---


**ABSTRACT**  
Natural Language Explanations (NLE) aim at supplementing the prediction of a model with human-friendly natural text. Existing NLE approaches involve training separate models for each downstream task. In this work, we propose Uni-NLX, a unified framework that consolidates all NLE tasks into a single and compact multi-task model using a unified training objective of text generation. Additionally, we introduce two new NLE datasets: 1) ImageNetX, a dataset of 144K samples for explaining ImageNet categories, and 2) VQA-ParaX, a dataset of 123K samples for explaining the task of Visual Question Answering (VQA). Both datasets are derived leveraging large language models (LLMs). By training on the 1M combined NLE samples, our single unified framework is capable of simultaneously performing seven NLE tasks including VQA, visual recognition and visual reasoning tasks with 7X fewer parameters, demonstrating comparable performance to the independent task-specific models in previous approaches, and in certain tasks even outperforming them. Code is at https://github.com/fawazsammani/uni-nlx

{{</citation>}}


### (29/97) ARAI-MVSNet: A multi-view stereo depth estimation network with adaptive depth range and depth interval (Song Zhang et al., 2023)

{{<citation>}}

Song Zhang, Wenjia Xu, Zhiwei Wei, Lili Zhang, Yang Wang, Junyi Liu. (2023)  
**ARAI-MVSNet: A multi-view stereo depth estimation network with adaptive depth range and depth interval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09022v1)  

---


**ABSTRACT**  
Multi-View Stereo~(MVS) is a fundamental problem in geometric computer vision which aims to reconstruct a scene using multi-view images with known camera parameters. However, the mainstream approaches represent the scene with a fixed all-pixel depth range and equal depth interval partition, which will result in inadequate utilization of depth planes and imprecise depth estimation. In this paper, we present a novel multi-stage coarse-to-fine framework to achieve adaptive all-pixel depth range and depth interval. We predict a coarse depth map in the first stage, then an Adaptive Depth Range Prediction module is proposed in the second stage to zoom in the scene by leveraging the reference image and the obtained depth map in the first stage and predict a more accurate all-pixel depth range for the following stages. In the third and fourth stages, we propose an Adaptive Depth Interval Adjustment module to achieve adaptive variable interval partition for pixel-wise depth range. The depth interval distribution in this module is normalized by Z-score, which can allocate dense depth hypothesis planes around the potential ground truth depth value and vice versa to achieve more accurate depth estimation. Extensive experiments on four widely used benchmark datasets~(DTU, TnT, BlendedMVS, ETH 3D) demonstrate that our model achieves state-of-the-art performance and yields competitive generalization ability. Particularly, our method achieves the highest Acc and Overall on the DTU dataset, while attaining the highest Recall and $F_{1}$-score on the Tanks and Temples intermediate and advanced dataset. Moreover, our method also achieves the lowest $e_{1}$ and $e_{3}$ on the BlendedMVS dataset and the highest Acc and $F_{1}$-score on the ETH 3D dataset, surpassing all listed methods.Project website: https://github.com/zs670980918/ARAI-MVSNet

{{</citation>}}


### (30/97) FashionLOGO: Prompting Multimodal Large Language Models for Fashion Logo Embeddings (Yulin Su et al., 2023)

{{<citation>}}

Yulin Su, Min Yang, Minghui Qiu, Jing Wang, Tao Wang. (2023)  
**FashionLOGO: Prompting Multimodal Large Language Models for Fashion Logo Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, Language Model, OCR  
[Paper Link](http://arxiv.org/abs/2308.09012v1)  

---


**ABSTRACT**  
Logo embedding plays a crucial role in various e-commerce applications by facilitating image retrieval or recognition, such as intellectual property protection and product search. However, current methods treat logo embedding as a purely visual problem, which may limit their performance in real-world scenarios. A notable issue is that the textual knowledge embedded in logo images has not been adequately explored. Therefore, we propose a novel approach that leverages textual knowledge as an auxiliary to improve the robustness of logo embedding. The emerging Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in both visual and textual understanding and could become valuable visual assistants in understanding logo images. Inspired by this observation, our proposed method, FashionLOGO, aims to utilize MLLMs to enhance fashion logo embedding. We explore how MLLMs can improve logo embedding by prompting them to generate explicit textual knowledge through three types of prompts, including image OCR, brief captions, and detailed descriptions prompts, in a zero-shot setting. We adopt a cross-attention transformer to enable image embedding queries to learn supplementary knowledge from textual embeddings automatically. To reduce computational costs, we only use the image embedding model in the inference stage, similar to traditional inference pipelines. Our extensive experiments on three real-world datasets demonstrate that FashionLOGO learns generalized and robust logo embeddings, achieving state-of-the-art performance in all benchmark datasets. Furthermore, we conduct comprehensive ablation studies to demonstrate the performance improvements resulting from the introduction of MLLMs.

{{</citation>}}


### (31/97) Semantic Information for Object Detection (Jean-Francois Nies, 2023)

{{<citation>}}

Jean-Francois Nies. (2023)  
**Semantic Information for Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.08990v1)  

---


**ABSTRACT**  
In this paper, we demonstrate that the concept of Semantic Consistency and the ensuing method of Knowledge-Aware Re-Optimization can be adapted for the problem of object detection in intricate traffic scenes. Furthermore, we introduce a novel method for extracting a knowledge graph from a dataset of images provided with instance-level annotations, and integrate this new knowledge graph with the existing semantic consistency model. Combining both this novel hybrid knowledge graph and the preexisting methods of frequency analysis and external knowledge graph as sources for semantic information, we investigate the effectiveness of knowledge-aware re-optimization on the Faster-RCNN and DETR object detection models. We find that limited but consistent improvements in precision and or recall can be achieved using this method for all combinations of model and method studied.

{{</citation>}}


### (32/97) Automatic Signboard Recognition in Low Quality Night Images (Manas Kagde et al., 2023)

{{<citation>}}

Manas Kagde, Priyanka Choudhary, Rishi Joshi, Somnath Dey. (2023)  
**Automatic Signboard Recognition in Low Quality Night Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Yolo  
[Paper Link](http://arxiv.org/abs/2308.08941v1)  

---


**ABSTRACT**  
An essential requirement for driver assistance systems and autonomous driving technology is implementing a robust system for detecting and recognizing traffic signs. This system enables the vehicle to autonomously analyze the environment and make appropriate decisions regarding its movement, even when operating at higher frame rates. However, traffic sign images captured in inadequate lighting and adverse weather conditions are poorly visible, blurred, faded, and damaged. Consequently, the recognition of traffic signs in such circumstances becomes inherently difficult. This paper addressed the challenges of recognizing traffic signs from images captured in low light, noise, and blurriness. To achieve this goal, a two-step methodology has been employed. The first step involves enhancing traffic sign images by applying a modified MIRNet model and producing enhanced images. In the second step, the Yolov4 model recognizes the traffic signs in an unconstrained environment. The proposed method has achieved 5.40% increment in mAP@0.5 for low quality images on Yolov4. The overall mAP@0.5 of 96.75% has been achieved on the GTSRB dataset. It has also attained mAP@0.5 of 100% on the GTSDB dataset for the broad categories, comparable with the state-of-the-art work.

{{</citation>}}


### (33/97) Point-aware Interaction and CNN-induced Refinement Network for RGB-D Salient Object Detection (Runmin Cong et al., 2023)

{{<citation>}}

Runmin Cong, Hongyu Liu, Chen Zhang, Wei Zhang, Feng Zheng, Ran Song, Sam Kwong. (2023)  
**Point-aware Interaction and CNN-induced Refinement Network for RGB-D Salient Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2308.08930v1)  

---


**ABSTRACT**  
By integrating complementary information from RGB image and depth map, the ability of salient object detection (SOD) for complex and challenging scenes can be improved. In recent years, the important role of Convolutional Neural Networks (CNNs) in feature extraction and cross-modality interaction has been fully explored, but it is still insufficient in modeling global long-range dependencies of self-modality and cross-modality. To this end, we introduce CNNs-assisted Transformer architecture and propose a novel RGB-D SOD network with Point-aware Interaction and CNN-induced Refinement (PICR-Net). On the one hand, considering the prior correlation between RGB modality and depth modality, an attention-triggered cross-modality point-aware interaction (CmPI) module is designed to explore the feature interaction of different modalities with positional constraints. On the other hand, in order to alleviate the block effect and detail destruction problems brought by the Transformer naturally, we design a CNN-induced refinement (CNNR) unit for content refinement and supplementation. Extensive experiments on five RGB-D SOD datasets show that the proposed network achieves competitive results in both quantitative and qualitative comparisons.

{{</citation>}}


### (34/97) A White-Box False Positive Adversarial Attack Method on Contrastive Loss-Based Offline Handwritten Signature Verification Models (Zhongliang Guo et al., 2023)

{{<citation>}}

Zhongliang Guo, Yifei Qian, Ognjen Arandjelović, Lei Fang. (2023)  
**A White-Box False Positive Adversarial Attack Method on Contrastive Loss-Based Offline Handwritten Signature Verification Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2308.08925v1)  

---


**ABSTRACT**  
In this paper, we tackle the challenge of white-box false positive adversarial attacks on contrastive loss-based offline handwritten signature verification models. We propose a novel attack method that treats the attack as a style transfer between closely related but distinct writing styles. To guide the generation of deceptive images, we introduce two new loss functions that enhance the attack success rate by perturbing the Euclidean distance between the embedding vectors of the original and synthesized samples, while ensuring minimal perturbations by reducing the difference between the generated image and the original image. Our method demonstrates state-of-the-art performance in white-box attacks on contrastive loss-based offline handwritten signature verification models, as evidenced by our experiments. The key contributions of this paper include a novel false positive attack method, two new loss functions, effective style transfer in handwriting styles, and superior performance in white-box false positive attacks compared to other white-box attack methods.

{{</citation>}}


### (35/97) Frequency Perception Network for Camouflaged Object Detection (Runmin Cong et al., 2023)

{{<citation>}}

Runmin Cong, Mengyao Sun, Sanyi Zhang, Xiaofei Zhou, Wei Zhang, Yao Zhao. (2023)  
**Frequency Perception Network for Camouflaged Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.08924v1)  

---


**ABSTRACT**  
Camouflaged object detection (COD) aims to accurately detect objects hidden in the surrounding environment. However, the existing COD methods mainly locate camouflaged objects in the RGB domain, their performance has not been fully exploited in many challenging scenarios. Considering that the features of the camouflaged object and the background are more discriminative in the frequency domain, we propose a novel learnable and separable frequency perception mechanism driven by the semantic hierarchy in the frequency domain. Our entire network adopts a two-stage model, including a frequency-guided coarse localization stage and a detail-preserving fine localization stage. With the multi-level features extracted by the backbone, we design a flexible frequency perception module based on octave convolution for coarse positioning. Then, we design the correction fusion module to step-by-step integrate the high-level features through the prior-guided correction and cross-layer feature channel association, and finally combine them with the shallow features to achieve the detailed correction of the camouflaged objects. Compared with the currently existing models, our proposed method achieves competitive performance in three popular benchmark datasets both qualitatively and quantitatively.

{{</citation>}}


### (36/97) Identity-Seeking Self-Supervised Representation Learning for Generalizable Person Re-identification (Zhaopeng Dou et al., 2023)

{{<citation>}}

Zhaopeng Dou, Zhongdao Wang, Yali Li, Shengjin Wang. (2023)  
**Identity-Seeking Self-Supervised Representation Learning for Generalizable Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.08887v1)  

---


**ABSTRACT**  
This paper aims to learn a domain-generalizable (DG) person re-identification (ReID) representation from large-scale videos \textbf{without any annotation}. Prior DG ReID methods employ limited labeled data for training due to the high cost of annotation, which restricts further advances. To overcome the barriers of data and annotation, we propose to utilize large-scale unsupervised data for training. The key issue lies in how to mine identity information. To this end, we propose an Identity-seeking Self-supervised Representation learning (ISR) method. ISR constructs positive pairs from inter-frame images by modeling the instance association as a maximum-weight bipartite matching problem. A reliability-guided contrastive loss is further presented to suppress the adverse impact of noisy positive pairs, ensuring that reliable positive pairs dominate the learning process. The training cost of ISR scales approximately linearly with the data size, making it feasible to utilize large-scale data for training. The learned representation exhibits superior generalization ability. \textbf{Without human annotation and fine-tuning, ISR achieves 87.0\% Rank-1 on Market-1501 and 56.4\% Rank-1 on MSMT17}, outperforming the best supervised domain-generalizable method by 5.0\% and 19.5\%, respectively. In the pre-training$\rightarrow$fine-tuning scenario, ISR achieves state-of-the-art performance, with 88.4\% Rank-1 on MSMT17. The code is at \url{https://github.com/dcp15/ISR_ICCV2023_Oral}.

{{</citation>}}


### (37/97) SRMAE: Masked Image Modeling for Scale-Invariant Deep Representations (Zhiming Wang et al., 2023)

{{<citation>}}

Zhiming Wang, Lin Gu, Feng Lu. (2023)  
**SRMAE: Masked Image Modeling for Scale-Invariant Deep Representations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.08884v1)  

---


**ABSTRACT**  
Due to the prevalence of scale variance in nature images, we propose to use image scale as a self-supervised signal for Masked Image Modeling (MIM). Our method involves selecting random patches from the input image and downsampling them to a low-resolution format. Our framework utilizes the latest advances in super-resolution (SR) to design the prediction head, which reconstructs the input from low-resolution clues and other patches. After 400 epochs of pre-training, our Super Resolution Masked Autoencoders (SRMAE) get an accuracy of 82.1% on the ImageNet-1K task. Image scale signal also allows our SRMAE to capture scale invariance representation. For the very low resolution (VLR) recognition task, our model achieves the best performance, surpassing DeriveNet by 1.3%. Our method also achieves an accuracy of 74.84% on the task of recognizing low-resolution facial expressions, surpassing the current state-of-the-art FMD by 9.48%.

{{</citation>}}


### (38/97) Language-enhanced RNR-Map: Querying Renderable Neural Radiance Field maps with natural language (Francesco Taioli et al., 2023)

{{<citation>}}

Francesco Taioli, Federico Cunico, Federico Girella, Riccardo Bologna, Alessandro Farinelli, Marco Cristani. (2023)  
**Language-enhanced RNR-Map: Querying Renderable Neural Radiance Field maps with natural language**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.08854v1)  

---


**ABSTRACT**  
We present Le-RNR-Map, a Language-enhanced Renderable Neural Radiance map for Visual Navigation with natural language query prompts. The recently proposed RNR-Map employs a grid structure comprising latent codes positioned at each pixel. These latent codes, which are derived from image observation, enable: i) image rendering given a camera pose, since they are converted to Neural Radiance Field; ii) image navigation and localization with astonishing accuracy. On top of this, we enhance RNR-Map with CLIP-based embedding latent codes, allowing natural language search without additional label data. We evaluate the effectiveness of this map in single and multi-object searches. We also investigate its compatibility with a Large Language Model as an "affordance query resolver". Code and videos are available at https://intelligolabs.github.io/Le-RNR-Map/

{{</citation>}}


### (39/97) Bag of Tricks for Long-Tailed Multi-Label Classification on Chest X-Rays (Feng Hong et al., 2023)

{{<citation>}}

Feng Hong, Tianjie Dai, Jiangchao Yao, Ya Zhang, Yanfeng Wang. (2023)  
**Bag of Tricks for Long-Tailed Multi-Label Classification on Chest X-Rays**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.08853v1)  

---


**ABSTRACT**  
Clinical classification of chest radiography is particularly challenging for standard machine learning algorithms due to its inherent long-tailed and multi-label nature. However, few attempts take into account the coupled challenges posed by both the class imbalance and label co-occurrence, which hinders their value to boost the diagnosis on chest X-rays (CXRs) in the real-world scenarios. Besides, with the prevalence of pretraining techniques, how to incorporate these new paradigms into the current framework lacks of the systematical study. This technical report presents a brief description of our solution in the ICCV CVAMD 2023 CXR-LT Competition. We empirically explored the effectiveness for CXR diagnosis with the integration of several advanced designs about data augmentation, feature extractor, classifier design, loss function reweighting, exogenous data replenishment, etc. In addition, we improve the performance through simple test-time data augmentation and ensemble. Our framework finally achieves 0.349 mAP on the competition test set, ranking in the top five.

{{</citation>}}


### (40/97) MixBag: Bag-Level Data Augmentation for Learning from Label Proportions (Takanori Asanomi et al., 2023)

{{<citation>}}

Takanori Asanomi, Shinnosuke Matsuo, Daiki Suehiro, Ryoma Bise. (2023)  
**MixBag: Bag-Level Data Augmentation for Learning from Label Proportions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.08822v1)  

---


**ABSTRACT**  
Learning from label proportions (LLP) is a promising weakly supervised learning problem. In LLP, a set of instances (bag) has label proportions, but no instance-level labels are given. LLP aims to train an instance-level classifier by using the label proportions of the bag. In this paper, we propose a bag-level data augmentation method for LLP called MixBag, based on the key observation from our preliminary experiments; that the instance-level classification accuracy improves as the number of labeled bags increases even though the total number of instances is fixed. We also propose a confidence interval loss designed based on statistical theory to use the augmented bags effectively. To the best of our knowledge, this is the first attempt to propose bag-level data augmentation for LLP. The advantage of MixBag is that it can be applied to instance-level data augmentation techniques and any LLP method that uses the proportion loss. Experimental results demonstrate this advantage and the effectiveness of our method.

{{</citation>}}


### (41/97) Sensor Fusion by Spatial Encoding for Autonomous Driving (Quoc-Vinh Lai-Dang et al., 2023)

{{<citation>}}

Quoc-Vinh Lai-Dang, Jihui Lee, Bumgeun Park, Dongsoo Har. (2023)  
**Sensor Fusion by Spatial Encoding for Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10707v1)  

---


**ABSTRACT**  
Sensor fusion is critical to perception systems for task domains such as autonomous driving and robotics. Recently, the Transformer integrated with CNN has demonstrated high performance in sensor fusion for various perception tasks. In this work, we introduce a method for fusing data from camera and LiDAR. By employing Transformer modules at multiple resolutions, proposed method effectively combines local and global contextual relationships. The performance of the proposed method is validated by extensive experiments with two adversarial benchmarks with lengthy routes and high-density traffics. The proposed method outperforms previous approaches with the most challenging benchmarks, achieving significantly higher driving and infraction scores. Compared with TransFuser, it achieves 8% and 19% improvement in driving scores for the Longest6 and Town05 Long benchmarks, respectively.

{{</citation>}}


### (42/97) URL: Combating Label Noise for Lung Nodule Malignancy Grading (Xianze Ai et al., 2023)

{{<citation>}}

Xianze Ai, Zehui Liao, Yong Xia. (2023)  
**URL: Combating Label Noise for Lung Nodule Malignancy Grading**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.08772v1)  

---


**ABSTRACT**  
Due to the complexity of annotation and inter-annotator variability, most lung nodule malignancy grading datasets contain label noise, which inevitably degrades the performance and generalizability of models. Although researchers adopt the label-noise-robust methods to handle label noise for lung nodule malignancy grading, they do not consider the inherent ordinal relation among classes of this task. To model the ordinal relation among classes to facilitate tackling label noise in this task, we propose a Unimodal-Regularized Label-noise-tolerant (URL) framework. Our URL contains two stages, the Supervised Contrastive Learning (SCL) stage and the Memory pseudo-labels generation and Unimodal regularization (MU) stage. In the SCL stage, we select reliable samples and adopt supervised contrastive learning to learn better representations. In the MU stage, we split samples with multiple annotations into multiple samples with a single annotation and shuffle them into different batches. To handle label noise, pseudo-labels are generated using the similarity between each sample and the central feature of each class, and temporal ensembling is used to obtain memory pseudo-labels that supervise the model training. To model the ordinal relation, we introduce unimodal regularization to keep the ordinal relation among classes in the predictions. Moreover, each lung nodule is characterized by three orthographic views. Experiments conducted on the LIDC-IDRI dataset indicate the superiority of our URL over other competing methods. Code is available at https://github.com/axz520/UR.

{{</citation>}}


### (43/97) Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes (Zehan Wang et al., 2023)

{{<citation>}}

Zehan Wang, Haifeng Huang, Yang Zhao, Ziang Zhang, Zhou Zhao. (2023)  
**Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Dialog, Dialogue, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.08769v1)  

---


**ABSTRACT**  
3D scene understanding has gained significant attention due to its wide range of applications. However, existing methods for 3D scene understanding are limited to specific downstream tasks, which hinders their practicality in real-world applications. This paper presents Chat-3D, which combines the 3D visual perceptual ability of pre-trained 3D representations and the impressive reasoning and conversation capabilities of advanced LLMs to achieve the first universal dialogue systems for 3D scenes. Specifically, we align 3D representations into the feature space of LLMs, thus enabling LLMs to perceive the 3D world. Given the scarcity of 3D scene-text data, we propose a three-stage training strategy to efficiently utilize the available data for better alignment. To enhance the reasoning ability and develop a user-friendly interaction scheme, we further construct a high-quality object-centric 3D instruction dataset and design an associated object-centric prompt. Our experiments show that Chat-3D achieves an impressive ability to comprehend diverse instructions for 3D scenes, engage in intricate spatial reasoning, and incorporate external knowledge into its responses. Chat-3D achieves a 75.6% relative score compared with GPT-4 on the constructed instruction dataset.

{{</citation>}}


### (44/97) BOTT: Box Only Transformer Tracker for 3D Object Tracking (Lubing Zhou et al., 2023)

{{<citation>}}

Lubing Zhou, Xiaoli Meng, Yiluan Guo, Jiong Yang. (2023)  
**BOTT: Box Only Transformer Tracker for 3D Object Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.08753v1)  

---


**ABSTRACT**  
Tracking 3D objects is an important task in autonomous driving. Classical Kalman Filtering based methods are still the most popular solutions. However, these methods require handcrafted designs in motion modeling and can not benefit from the growing data amounts. In this paper, Box Only Transformer Tracker (BOTT) is proposed to learn to link 3D boxes of the same object from the different frames, by taking all the 3D boxes in a time window as input. Specifically, transformer self-attention is applied to exchange information between all the boxes to learn global-informative box embeddings. The similarity between these learned embeddings can be used to link the boxes of the same object. BOTT can be used for both online and offline tracking modes seamlessly. Its simplicity enables us to significantly reduce engineering efforts required by traditional Kalman Filtering based methods. Experiments show BOTT achieves competitive performance on two largest 3D MOT benchmarks: 69.9 and 66.7 AMOTA on nuScenes validation and test splits, respectively, 56.45 and 59.57 MOTA L2 on Waymo Open Dataset validation and test splits, respectively. This work suggests that tracking 3D objects by learning features directly from 3D boxes using transformers is a simple yet effective way.

{{</citation>}}


### (45/97) Learning Through Guidance: Knowledge Distillation for Endoscopic Image Classification (Harshala Gammulle et al., 2023)

{{<citation>}}

Harshala Gammulle, Yubo Chen, Sridha Sridharan, Travis Klein, Clinton Fookes. (2023)  
**Learning Through Guidance: Knowledge Distillation for Endoscopic Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.08731v1)  

---


**ABSTRACT**  
Endoscopy plays a major role in identifying any underlying abnormalities within the gastrointestinal (GI) tract. There are multiple GI tract diseases that are life-threatening, such as precancerous lesions and other intestinal cancers. In the usual process, a diagnosis is made by a medical expert which can be prone to human errors and the accuracy of the test is also entirely dependent on the expert's level of experience. Deep learning, specifically Convolution Neural Networks (CNNs) which are designed to perform automatic feature learning without any prior feature engineering, has recently reported great benefits for GI endoscopy image analysis. Previous research has developed models that focus only on improving performance, as such, the majority of introduced models contain complex deep network architectures with a large number of parameters that require longer training times. However, there is a lack of focus on developing lightweight models which can run in low-resource environments, which are typically encountered in medical clinics. We investigate three KD-based learning frameworks, response-based, feature-based, and relation-based mechanisms, and introduce a novel multi-head attention-based feature fusion mechanism to support relation-based learning. Compared to the existing relation-based methods that follow simplistic aggregation techniques of multi-teacher response/feature-based knowledge, we adopt the multi-head attention technique to provide flexibility towards localising and transferring important details from each teacher to better guide the student. We perform extensive evaluations on two widely used public datasets, KVASIR-V2 and Hyper-KVASIR, and our experimental results signify the merits of our proposed relation-based framework in achieving an improved lightweight model (only 51.8k trainable parameters) that can run in a resource-limited environment.

{{</citation>}}


### (46/97) Learning A Coarse-to-Fine Diffusion Transformer for Image Restoration (Liyan Wang et al., 2023)

{{<citation>}}

Liyan Wang, Qinyu Yang, Cong Wang, Wei Wang, Jinshan Pan, Zhixun Su. (2023)  
**Learning A Coarse-to-Fine Diffusion Transformer for Image Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.08730v1)  

---


**ABSTRACT**  
Recent years have witnessed the remarkable performance of diffusion models in various vision tasks. However, for image restoration that aims to recover clear images with sharper details from given degraded observations, diffusion-based methods may fail to recover promising results due to inaccurate noise estimation. Moreover, simple constraining noises cannot effectively learn complex degradation information, which subsequently hinders the model capacity. To solve the above problems, we propose a coarse-to-fine diffusion Transformer (C2F-DFT) for image restoration. Specifically, our C2F-DFT contains diffusion self-attention (DFSA) and diffusion feed-forward network (DFN) within a new coarse-to-fine training scheme. The DFSA and DFN respectively capture the long-range diffusion dependencies and learn hierarchy diffusion representation to facilitate better restoration. In the coarse training stage, our C2F-DFT estimates noises and then generates the final clean image by a sampling algorithm. To further improve the restoration quality, we propose a simple yet effective fine training scheme. It first exploits the coarse-trained diffusion model with fixed steps to generate restoration results, which then would be constrained with corresponding ground-truth ones to optimize the models to remedy the unsatisfactory results affected by inaccurate noise estimation. Extensive experiments show that C2F-DFT significantly outperforms diffusion-based restoration method IR-SDE and achieves competitive performance compared with Transformer-based state-of-the-art methods on $3$ tasks, including deraining, deblurring, and real denoising.

{{</citation>}}


### (47/97) Long-Range Grouping Transformer for Multi-View 3D Reconstruction (Liying Yang et al., 2023)

{{<citation>}}

Liying Yang, Zhenwei Zhu, Xuxin Lin, Jian Nong, Yanyan Liang. (2023)  
**Long-Range Grouping Transformer for Multi-View 3D Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.08724v1)  

---


**ABSTRACT**  
Nowadays, transformer networks have demonstrated superior performance in many computer vision tasks. In a multi-view 3D reconstruction algorithm following this paradigm, self-attention processing has to deal with intricate image tokens including massive information when facing heavy amounts of view input. The curse of information content leads to the extreme difficulty of model learning. To alleviate this problem, recent methods compress the token number representing each view or discard the attention operations between the tokens from different views. Obviously, they give a negative impact on performance. Therefore, we propose long-range grouping attention (LGA) based on the divide-and-conquer principle. Tokens from all views are grouped for separate attention operations. The tokens in each group are sampled from all views and can provide macro representation for the resided view. The richness of feature learning is guaranteed by the diversity among different groups. An effective and efficient encoder can be established which connects inter-view features using LGA and extract intra-view features using the standard self-attention layer. Moreover, a novel progressive upsampling decoder is also designed for voxel generation with relatively high resolution. Hinging on the above, we construct a powerful transformer-based network, called LRGT. Experimental results on ShapeNet verify our method achieves SOTA accuracy in multi-view reconstruction. Code will be available at https://github.com/LiyingCV/Long-Range-Grouping-Transformer.

{{</citation>}}


## cs.LG (12)



### (48/97) Online Transition-Based Feature Generation for Anomaly Detection in Concurrent Data Streams (Yinzheng Zhong et al., 2023)

{{<citation>}}

Yinzheng Zhong, Alexei Lisitsa. (2023)  
**Online Transition-Based Feature Generation for Anomaly Detection in Concurrent Data Streams**  

---
Primary Category: cs.LG  
Categories: cs-DB, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.10893v1)  

---


**ABSTRACT**  
In this paper, we introduce the transition-based feature generator (TFGen) technique, which reads general activity data with attributes and generates step-by-step generated data. The activity data may consist of network activity from packets, system calls from processes or classified activity from surveillance cameras. TFGen processes data online and will generate data with encoded historical data for each incoming activity with high computational efficiency. The input activities may concurrently originate from distinct traces or channels. The technique aims to address issues such as domain-independent applicability, the ability to discover global process structures, the encoding of time-series data, and online processing capability.

{{</citation>}}


### (49/97) Reinforcement Learning for Battery Management in Dairy Farming (Nawazish Ali et al., 2023)

{{<citation>}}

Nawazish Ali, Abdul Wahid, Rachael shaw, Karl Mason. (2023)  
**Reinforcement Learning for Battery Management in Dairy Farming**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.09023v1)  

---


**ABSTRACT**  
Dairy farming is a particularly energy-intensive part of the agriculture sector. Effective battery management is essential for renewable integration within the agriculture sector. However, controlling battery charging/discharging is a difficult task due to electricity demand variability, stochasticity of renewable generation, and energy price fluctuations. Despite the potential benefits of applying Artificial Intelligence (AI) to renewable energy in the context of dairy farming, there has been limited research in this area. This research is a priority for Ireland as it strives to meet its governmental goals in energy and sustainability. This research paper utilizes Q-learning to learn an effective policy for charging and discharging a battery within a dairy farm setting. The results demonstrate that the developed policy significantly reduces electricity costs compared to the established baseline algorithm. These findings highlight the effectiveness of reinforcement learning for battery management within the dairy farming sector.

{{</citation>}}


### (50/97) Deep-seeded Clustering for Unsupervised Valence-Arousal Emotion Recognition from Physiological Signals (Antoine Dubois et al., 2023)

{{<citation>}}

Antoine Dubois, Carlos Lima Azevedo, Sonja Haustein, Bruno Miranda. (2023)  
**Deep-seeded Clustering for Unsupervised Valence-Arousal Emotion Recognition from Physiological Signals**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.09013v1)  

---


**ABSTRACT**  
Emotions play a significant role in the cognitive processes of the human brain, such as decision making, learning and perception. The use of physiological signals has shown to lead to more objective, reliable and accurate emotion recognition combined with raising machine learning methods. Supervised learning methods have dominated the attention of the research community, but the challenge in collecting needed labels makes emotion recognition difficult in large-scale semi- or uncontrolled experiments. Unsupervised methods are increasingly being explored, however sub-optimal signal feature selection and label identification challenges unsupervised methods' accuracy and applicability. This article proposes an unsupervised deep cluster framework for emotion recognition from physiological and psychological data. Tests on the open benchmark data set WESAD show that deep k-means and deep c-means distinguish the four quadrants of Russell's circumplex model of affect with an overall accuracy of 87%. Seeding the clusters with the subject's subjective assessments helps to circumvent the need for labels.

{{</citation>}}


### (51/97) Cross-city Few-Shot Traffic Forecasting via Traffic Pattern Bank (Zhanyu Liu et al., 2023)

{{<citation>}}

Zhanyu Liu, Guanjie Zheng, Yanwei Yu. (2023)  
**Cross-city Few-Shot Traffic Forecasting via Traffic Pattern Bank**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.09727v1)  

---


**ABSTRACT**  
Traffic forecasting is a critical service in Intelligent Transportation Systems (ITS). Utilizing deep models to tackle this task relies heavily on data from traffic sensors or vehicle devices, while some cities might lack device support and thus have few available data. So, it is necessary to learn from data-rich cities and transfer the knowledge to data-scarce cities in order to improve the performance of traffic forecasting. To address this problem, we propose a cross-city few-shot traffic forecasting framework via Traffic Pattern Bank (TPB) due to that the traffic patterns are similar across cities. TPB utilizes a pre-trained traffic patch encoder to project raw traffic data from data-rich cities into high-dimensional space, from which a traffic pattern bank is generated through clustering. Then, the traffic data of the data-scarce city could query the traffic pattern bank and explicit relations between them are constructed. The metaknowledge is aggregated based on these relations and an adjacency matrix is constructed to guide a downstream spatial-temporal model in forecasting future traffic. The frequently used meta-training framework Reptile is adapted to find a better initial parameter for the learnable modules. Experiments on real-world traffic datasets show that TPB outperforms existing methods and demonstrates the effectiveness of our approach in cross-city few-shot traffic forecasting.

{{</citation>}}


### (52/97) CONVERT:Contrastive Graph Clustering with Reliable Augmentation (Xihong Yang et al., 2023)

{{<citation>}}

Xihong Yang, Cheng Tan, Yue Liu, Ke Liang, Siwei Wang, Sihang Zhou, Jun Xia, Stan Z. Li, Xinwang Liu, En Zhu. (2023)  
**CONVERT:Contrastive Graph Clustering with Reliable Augmentation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.08963v1)  

---


**ABSTRACT**  
Contrastive graph node clustering via learnable data augmentation is a hot research spot in the field of unsupervised graph learning. The existing methods learn the sampling distribution of a pre-defined augmentation to generate data-driven augmentations automatically. Although promising clustering performance has been achieved, we observe that these strategies still rely on pre-defined augmentations, the semantics of the augmented graph can easily drift. The reliability of the augmented view semantics for contrastive learning can not be guaranteed, thus limiting the model performance. To address these problems, we propose a novel CONtrastiVe Graph ClustEring network with Reliable AugmenTation (COVERT). Specifically, in our method, the data augmentations are processed by the proposed reversible perturb-recover network. It distills reliable semantic information by recovering the perturbed latent embeddings. Moreover, to further guarantee the reliability of semantics, a novel semantic loss is presented to constrain the network via quantifying the perturbation and recovery. Lastly, a label-matching mechanism is designed to guide the model by clustering information through aligning the semantic labels and the selected high-confidence clustering pseudo labels. Extensive experimental results on seven datasets demonstrate the effectiveness of the proposed method. We release the code and appendix of CONVERT at https://github.com/xihongyang1999/CONVERT on GitHub.

{{</citation>}}


### (53/97) Interpretable Graph Neural Networks for Tabular Data (Amr Alkhatib et al., 2023)

{{<citation>}}

Amr Alkhatib, Sofiane Ennadir, Henrik Boström, Michalis Vazirgiannis. (2023)  
**Interpretable Graph Neural Networks for Tabular Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.08945v1)  

---


**ABSTRACT**  
Data in tabular format is frequently occurring in real-world applications. Graph Neural Networks (GNNs) have recently been extended to effectively handle such data, allowing feature interactions to be captured through representation learning. However, these approaches essentially produce black-box models, in the form of deep neural networks, precluding users from following the logic behind the model predictions. We propose an approach, called IGNNet (Interpretable Graph Neural Network for tabular data), which constrains the learning algorithm to produce an interpretable model, where the model shows how the predictions are exactly computed from the original input features. A large-scale empirical investigation is presented, showing that IGNNet is performing on par with state-of-the-art machine-learning algorithms that target tabular data, including XGBoost, Random Forests, and TabNet. At the same time, the results show that the explanations obtained from IGNNet are aligned with the true Shapley values of the features without incurring any additional computational overhead.

{{</citation>}}


### (54/97) Causal Adversarial Perturbations for Individual Fairness and Robustness in Heterogeneous Data Spaces (Ahmad-Reza Ehyaei et al., 2023)

{{<citation>}}

Ahmad-Reza Ehyaei, Kiarash Mohammadi, Amir-Hossein Karimi, Samira Samadi, Golnoosh Farnadi. (2023)  
**Causal Adversarial Perturbations for Individual Fairness and Robustness in Heterogeneous Data Spaces**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08938v1)  

---


**ABSTRACT**  
As responsible AI gains importance in machine learning algorithms, properties such as fairness, adversarial robustness, and causality have received considerable attention in recent years. However, despite their individual significance, there remains a critical gap in simultaneously exploring and integrating these properties. In this paper, we propose a novel approach that examines the relationship between individual fairness, adversarial robustness, and structural causal models in heterogeneous data spaces, particularly when dealing with discrete sensitive attributes. We use causal structural models and sensitive attributes to create a fair metric and apply it to measure semantic similarity among individuals. By introducing a novel causal adversarial perturbation and applying adversarial training, we create a new regularizer that combines individual fairness, causality, and robustness in the classifier. Our method is evaluated on both real-world and synthetic datasets, demonstrating its effectiveness in achieving an accurate classifier that simultaneously exhibits fairness, adversarial robustness, and causal awareness.

{{</citation>}}


### (55/97) IMM: An Imitative Reinforcement Learning Approach with Predictive Representation Learning for Automatic Market Making (Hui Niu et al., 2023)

{{<citation>}}

Hui Niu, Siyuan Li, Jiahao Zheng, Zhouchi Lin, Jian Li, Jian Guo, Bo An. (2023)  
**IMM: An Imitative Reinforcement Learning Approach with Predictive Representation Learning for Automatic Market Making**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-fin-TR  
Keywords: Reinforcement Learning, Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.08918v1)  

---


**ABSTRACT**  
Market making (MM) has attracted significant attention in financial trading owing to its essential function in ensuring market liquidity. With strong capabilities in sequential decision-making, Reinforcement Learning (RL) technology has achieved remarkable success in quantitative trading. Nonetheless, most existing RL-based MM methods focus on optimizing single-price level strategies which fail at frequent order cancellations and loss of queue priority. Strategies involving multiple price levels align better with actual trading scenarios. However, given the complexity that multi-price level strategies involves a comprehensive trading action space, the challenge of effectively training profitable RL agents for MM persists. Inspired by the efficient workflow of professional human market makers, we propose Imitative Market Maker (IMM), a novel RL framework leveraging both knowledge from suboptimal signal-based experts and direct policy interactions to develop multi-price level MM strategies efficiently. The framework start with introducing effective state and action representations adept at encoding information about multi-price level orders. Furthermore, IMM integrates a representation learning unit capable of capturing both short- and long-term market trends to mitigate adverse selection risk. Subsequently, IMM formulates an expert strategy based on signals and trains the agent through the integration of RL and imitation learning techniques, leading to efficient learning. Extensive experimental results on four real-world market datasets demonstrate that IMM outperforms current RL-based market making strategies in terms of several financial criteria. The findings of the ablation study substantiate the effectiveness of the model components.

{{</citation>}}


### (56/97) Beyond Sharing: Conflict-Aware Multivariate Time Series Anomaly Detection (Haotian Si et al., 2023)

{{<citation>}}

Haotian Si, Changhua Pei, Zhihan Li, Yadong Zhao, Jingjing Li, Haiming Zhang, Zulong Diao, Jianhui Li, Gaogang Xie, Dan Pei. (2023)  
**Beyond Sharing: Conflict-Aware Multivariate Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2308.08915v1)  

---


**ABSTRACT**  
Massive key performance indicators (KPIs) are monitored as multivariate time series data (MTS) to ensure the reliability of the software applications and service system. Accurately detecting the abnormality of MTS is very critical for subsequent fault elimination. The scarcity of anomalies and manual labeling has led to the development of various self-supervised MTS anomaly detection (AD) methods, which optimize an overall objective/loss encompassing all metrics' regression objectives/losses. However, our empirical study uncovers the prevalence of conflicts among metrics' regression objectives, causing MTS models to grapple with different losses. This critical aspect significantly impacts detection performance but has been overlooked in existing approaches. To address this problem, by mimicking the design of multi-gate mixture-of-experts (MMoE), we introduce CAD, a Conflict-aware multivariate KPI Anomaly Detection algorithm. CAD offers an exclusive structure for each metric to mitigate potential conflicts while fostering inter-metric promotions. Upon thorough investigation, we find that the poor performance of vanilla MMoE mainly comes from the input-output misalignment settings of MTS formulation and convergence issues arising from expansive tasks. To address these challenges, we propose a straightforward yet effective task-oriented metric selection and p&s (personalized and shared) gating mechanism, which establishes CAD as the first practicable multi-task learning (MTL) based MTS AD model. Evaluations on multiple public datasets reveal that CAD obtains an average F1-score of 0.943 across three public datasets, notably outperforming state-of-the-art methods. Our code is accessible at https://github.com/dawnvince/MTS_CAD.

{{</citation>}}


### (57/97) Development of a Knowledge Graph Embeddings Model for Pain (Jaya Chaturvedi et al., 2023)

{{<citation>}}

Jaya Chaturvedi, Tao Wang, Sumithra Velupillai, Robert Stewart, Angus Roberts. (2023)  
**Development of a Knowledge Graph Embeddings Model for Pain**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.08904v1)  

---


**ABSTRACT**  
Pain is a complex concept that can interconnect with other concepts such as a disorder that might cause pain, a medication that might relieve pain, and so on. To fully understand the context of pain experienced by either an individual or across a population, we may need to examine all concepts related to pain and the relationships between them. This is especially useful when modeling pain that has been recorded in electronic health records. Knowledge graphs represent concepts and their relations by an interlinked network, enabling semantic and context-based reasoning in a computationally tractable form. These graphs can, however, be too large for efficient computation. Knowledge graph embeddings help to resolve this by representing the graphs in a low-dimensional vector space. These embeddings can then be used in various downstream tasks such as classification and link prediction. The various relations associated with pain which are required to construct such a knowledge graph can be obtained from external medical knowledge bases such as SNOMED CT, a hierarchical systematic nomenclature of medical terms. A knowledge graph built in this way could be further enriched with real-world examples of pain and its relations extracted from electronic health records. This paper describes the construction of such knowledge graph embedding models of pain concepts, extracted from the unstructured text of mental health electronic health records, combined with external knowledge created from relations described in SNOMED CT, and their evaluation on a subject-object link prediction task. The performance of the models was compared with other baseline models.

{{</citation>}}


### (58/97) Mitigating Semantic Confusion from Hostile Neighborhood for Graph Active Learning (Tianmeng Yang et al., 2023)

{{<citation>}}

Tianmeng Yang, Min Zhou, Yujing Wang, Zhengjie Lin, Lujia Pan, Bin Cui, Yunhai Tong. (2023)  
**Mitigating Semantic Confusion from Hostile Neighborhood for Graph Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.08823v1)  

---


**ABSTRACT**  
Graph Active Learning (GAL), which aims to find the most informative nodes in graphs for annotation to maximize the Graph Neural Networks (GNNs) performance, has attracted many research efforts but remains non-trivial challenges. One major challenge is that existing GAL strategies may introduce semantic confusion to the selected training set, particularly when graphs are noisy. Specifically, most existing methods assume all aggregating features to be helpful, ignoring the semantically negative effect between inter-class edges under the message-passing mechanism. In this work, we present Semantic-aware Active learning framework for Graphs (SAG) to mitigate the semantic confusion problem. Pairwise similarities and dissimilarities of nodes with semantic features are introduced to jointly evaluate the node influence. A new prototype-based criterion and query policy are also designed to maintain diversity and class balance of the selected nodes, respectively. Extensive experiments on the public benchmark graphs and a real-world financial dataset demonstrate that SAG significantly improves node classification performances and consistently outperforms previous methods. Moreover, comprehensive analysis and ablation study also verify the effectiveness of the proposed framework.

{{</citation>}}


### (59/97) Explainable AI for tool wear prediction in turning (Saleh Valizadeh Sotubadi et al., 2023)

{{<citation>}}

Saleh Valizadeh Sotubadi, Rui Liu, Vinh Neguyen. (2023)  
**Explainable AI for tool wear prediction in turning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08765v1)  

---


**ABSTRACT**  
This research aims develop an Explainable Artificial Intelligence (XAI) framework to facilitate human-understandable solutions for tool wear prediction during turning. A random forest algorithm was used as the supervised Machine Learning (ML) classifier for training and binary classification using acceleration, acoustics, temperature, and spindle speed during the orthogonal tube turning process as input features. The ML classifier was used to predict the condition of the tool after the cutting process, which was determined in a binary class form indicating if the cutting tool was available or failed. After the training process, the Shapley criterion was used to explain the predictions of the trained ML classifier. Specifically, the significance of each input feature in the decision-making and classification was identified to explain the reasoning of the ML classifier predictions. After implementing the Shapley criterion on all testing datasets, the tool temperature was identified as the most significant feature in determining the classification of available versus failed cutting tools. Hence, this research demonstrates capability of XAI to provide machining operators the ability to diagnose and understand complex ML classifiers in prediction of tool wear.

{{</citation>}}


## eess.IV (3)



### (60/97) Data diversity and virtual imaging in AI-based diagnosis: A case study based on COVID-19 (Fakrul Islam Tushar et al., 2023)

{{<citation>}}

Fakrul Islam Tushar, Lavsen Dahal, Saman Sotoudeh-Paima, Ehsan Abadi, W. Paul Segars, Ehsan Samei, Joseph Y. Lo. (2023)  
**Data diversity and virtual imaging in AI-based diagnosis: A case study based on COVID-19**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09730v1)  

---


**ABSTRACT**  
Many studies have investigated deep-learning-based artificial intelligence (AI) models for medical imaging diagnosis of the novel coronavirus (COVID-19), with many reports of near-perfect performance. However, variability in performance and underlying data biases raise concerns about clinical generalizability. This retrospective study involved the development and evaluation of artificial intelligence (AI) models for COVID-19 diagnosis using both diverse clinical and virtually generated medical images. In addition, we conducted a virtual imaging trial to assess how AI performance is affected by several patient- and physics-based factors, including the extent of disease, radiation dose, and imaging modality of computed tomography (CT) and chest radiography (CXR). AI performance was strongly influenced by dataset characteristics including quantity, diversity, and prevalence, leading to poor generalization with up to 20% drop in receiver operating characteristic area under the curve. Model performance on virtual CT and CXR images was comparable to overall results on clinical data. Imaging dose proved to have negligible influence on the results, but the extent of the disease had a marked affect. CT results were consistently superior to those from CXR. Overall, the study highlighted the significant impact of dataset characteristics and disease extent on COVID assessment, and the relevance and potential role of virtual imaging trial techniques on developing effective evaluation of AI algorithms and facilitating translation into diagnostic practice.

{{</citation>}}


### (61/97) JPEG Quantized Coefficient Recovery via DCT Domain Spatial-Frequential Transformer (Mingyu Ouyang et al., 2023)

{{<citation>}}

Mingyu Ouyang, Zhenzhong Chen. (2023)  
**JPEG Quantized Coefficient Recovery via DCT Domain Spatial-Frequential Transformer**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09110v1)  

---


**ABSTRACT**  
JPEG compression adopts the quantization of Discrete Cosine Transform (DCT) coefficients for effective bit-rate reduction, whilst the quantization could lead to a significant loss of important image details. Recovering compressed JPEG images in the frequency domain has attracted more and more attention recently, in addition to numerous restoration approaches developed in the pixel domain. However, the current DCT domain methods typically suffer from limited effectiveness in handling a wide range of compression quality factors, or fall short in recovering sparse quantized coefficients and the components across different colorspace. To address these challenges, we propose a DCT domain spatial-frequential Transformer, named as DCTransformer. Specifically, a dual-branch architecture is designed to capture both spatial and frequential correlations within the collocated DCT coefficients. Moreover, we incorporate the operation of quantization matrix embedding, which effectively allows our single model to handle a wide range of quality factors, and a luminance-chrominance alignment head that produces a unified feature map to align different-sized luminance and chrominance components. Our proposed DCTransformer outperforms the current state-of-the-art JPEG artifact removal techniques, as demonstrated by our extensive experiments.

{{</citation>}}


### (62/97) LesionMix: A Lesion-Level Data Augmentation Method for Medical Image Segmentation (Berke Doga Basaran et al., 2023)

{{<citation>}}

Berke Doga Basaran, Weitong Zhang, Mengyun Qiao, Bernhard Kainz, Paul M. Matthews, Wenjia Bai. (2023)  
**LesionMix: A Lesion-Level Data Augmentation Method for Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.09026v1)  

---


**ABSTRACT**  
Data augmentation has become a de facto component of deep learning-based medical image segmentation methods. Most data augmentation techniques used in medical imaging focus on spatial and intensity transformations to improve the diversity of training images. They are often designed at the image level, augmenting the full image, and do not pay attention to specific abnormalities within the image. Here, we present LesionMix, a novel and simple lesion-aware data augmentation method. It performs augmentation at the lesion level, increasing the diversity of lesion shape, location, intensity and load distribution, and allowing both lesion populating and inpainting. Experiments on different modalities and different lesion datasets, including four brain MR lesion datasets and one liver CT lesion dataset, demonstrate that LesionMix achieves promising performance in lesion image segmentation, outperforming several recent Mix-based data augmentation methods. The code will be released at https://github.com/dogabasaran/lesionmix.

{{</citation>}}


## cs.CL (22)



### (63/97) Characterizing Information Seeking Events in Health-Related Social Discourse (Omar Sharif et al., 2023)

{{<citation>}}

Omar Sharif, Madhusudan Basak, Tanzia Parvin, Ava Scharfstein, Alphonso Bradham, Jacob T. Borodovsky, Sarah E. Lord, Sarah Masud Preum. (2023)  
**Characterizing Information Seeking Events in Health-Related Social Discourse**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.09156v1)  

---


**ABSTRACT**  
Social media sites have become a popular platform for individuals to seek and share health information. Despite the progress in natural language processing for social media mining, a gap remains in analyzing health-related texts on social discourse in the context of events. Event-driven analysis can offer insights into different facets of healthcare at an individual and collective level, including treatment options, misconceptions, knowledge gaps, etc. This paper presents a paradigm to characterize health-related information-seeking in social discourse through the lens of events. Events here are board categories defined with domain experts that capture the trajectory of the treatment/medication. To illustrate the value of this approach, we analyze Reddit posts regarding medications for Opioid Use Disorder (OUD), a critical global health concern. To the best of our knowledge, this is the first attempt to define event categories for characterizing information-seeking in OUD social discourse. Guided by domain experts, we develop TREAT-ISE, a novel multilabel treatment information-seeking event dataset to analyze online discourse on an event-based framework. This dataset contains Reddit posts on information-seeking events related to recovery from OUD, where each post is annotated based on the type of events. We also establish a strong performance benchmark (77.4% F1 score) for the task by employing several machine learning and deep learning classifiers. Finally, we thoroughly investigate the performance and errors of ChatGPT on this task, providing valuable insights into the LLM's capabilities and ongoing characterization efforts.

{{</citation>}}


### (64/97) Semantic Consistency for Assuring Reliability of Large Language Models (Harsh Raj et al., 2023)

{{<citation>}}

Harsh Raj, Vipul Gupta, Domenic Rosati, Subhabrata Majumdar. (2023)  
**Semantic Consistency for Assuring Reliability of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2308.09138v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) exhibit remarkable fluency and competence across various natural language tasks. However, recent research has highlighted their sensitivity to variations in input prompts. To deploy LLMs in a safe and reliable manner, it is crucial for their outputs to be consistent when prompted with expressions that carry the same meaning or intent. While some existing work has explored how state-of-the-art LLMs address this issue, their evaluations have been confined to assessing lexical equality of single- or multi-word answers, overlooking the consistency of generative text sequences. For a more comprehensive understanding of the consistency of LLMs in open-ended text generation scenarios, we introduce a general measure of semantic consistency, and formulate multiple versions of this metric to evaluate the performance of various LLMs. Our proposal demonstrates significantly higher consistency and stronger correlation with human evaluations of output consistency than traditional metrics based on lexical consistency. Finally, we propose a novel prompting strategy, called Ask-to-Choose (A2C), to enhance semantic consistency. When evaluated for closed-book question answering based on answer variations from the TruthfulQA benchmark, A2C increases accuracy metrics for pretrained and finetuned LLMs by up to 47%, and semantic consistency metrics for instruction-tuned models by up to 7-fold.

{{</citation>}}


### (65/97) Linearity of Relation Decoding in Transformer Language Models (Evan Hernandez et al., 2023)

{{<citation>}}

Evan Hernandez, Arnab Sen Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas, Yonatan Belinkov, David Bau. (2023)  
**Linearity of Relation Decoding in Transformer Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09124v1)  

---


**ABSTRACT**  
Much of the knowledge encoded in transformer language models (LMs) may be expressed in terms of relations: relations between words and their synonyms, entities and their attributes, etc. We show that, for a subset of relations, this computation is well-approximated by a single linear transformation on the subject representation. Linear relation representations may be obtained by constructing a first-order approximation to the LM from a single prompt, and they exist for a variety of factual, commonsense, and linguistic relations. However, we also identify many cases in which LM predictions capture relational knowledge accurately, but this knowledge is not linearly encoded in their representations. Our results thus reveal a simple, interpretable, but heterogeneously deployed knowledge representation strategy in transformer LMs.

{{</citation>}}


### (66/97) MaScQA: A Question Answering Dataset for Investigating Materials Science Knowledge of Large Language Models (Mohd Zaki et al., 2023)

{{<citation>}}

Mohd Zaki, Jayadeva, Mausam, N. M. Anoop Krishnan. (2023)  
**MaScQA: A Question Answering Dataset for Investigating Materials Science Knowledge of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cond-mat-mtrl-sci, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.09115v1)  

---


**ABSTRACT**  
Information extraction and textual comprehension from materials literature are vital for developing an exhaustive knowledge base that enables accelerated materials discovery. Language models have demonstrated their capability to answer domain-specific questions and retrieve information from knowledge bases. However, there are no benchmark datasets in the materials domain that can evaluate the understanding of the key concepts by these language models. In this work, we curate a dataset of 650 challenging questions from the materials domain that require the knowledge and skills of a materials student who has cleared their undergraduate degree. We classify these questions based on their structure and the materials science domain-based subcategories. Further, we evaluate the performance of GPT-3.5 and GPT-4 models on solving these questions via zero-shot and chain of thought prompting. It is observed that GPT-4 gives the best performance (~62% accuracy) as compared to GPT-3.5. Interestingly, in contrast to the general observation, no significant improvement in accuracy is observed with the chain of thought prompting. To evaluate the limitations, we performed an error analysis, which revealed conceptual errors (~64%) as the major contributor compared to computational errors (~36%) towards the reduced performance of LLMs. We hope that the dataset and analysis performed in this work will promote further research in developing better materials science domain-specific LLMs and strategies for information extraction.

{{</citation>}}


### (67/97) mCL-NER: Cross-Lingual Named Entity Recognition via Multi-view Contrastive Learning (Ying Mo et al., 2023)

{{<citation>}}

Ying Mo, Jian Yang, Jiahao Liu, Qifan Wang, Ruoyu Chen, Jingang Wang, Zhoujun Li. (2023)  
**mCL-NER: Cross-Lingual Named Entity Recognition via Multi-view Contrastive Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2308.09073v1)  

---


**ABSTRACT**  
Cross-lingual named entity recognition (CrossNER) faces challenges stemming from uneven performance due to the scarcity of multilingual corpora, especially for non-English data. While prior efforts mainly focus on data-driven transfer methods, a significant aspect that has not been fully explored is aligning both semantic and token-level representations across diverse languages. In this paper, we propose Multi-view Contrastive Learning for Cross-lingual Named Entity Recognition (mCL-NER). Specifically, we reframe the CrossNER task into a problem of recognizing relationships between pairs of tokens. This approach taps into the inherent contextual nuances of token-to-token connections within entities, allowing us to align representations across different languages. A multi-view contrastive learning framework is introduced to encompass semantic contrasts between source, codeswitched, and target sentences, as well as contrasts among token-to-token relations. By enforcing agreement within both semantic and relational spaces, we minimize the gap between source sentences and their counterparts of both codeswitched and target sentences. This alignment extends to the relationships between diverse tokens, enhancing the projection of entities across languages. We further augment CrossNER by combining self-training with labeled source data and unlabeled target data. Our experiments on the XTREME benchmark, spanning 40 languages, demonstrate the superiority of mCL-NER over prior data-driven and model-based approaches. It achieves a substantial increase of nearly +2.0 $F_1$ scores across a broad spectrum and establishes itself as the new state-of-the-art performer.

{{</citation>}}


### (68/97) Contrasting Linguistic Patterns in Human and LLM-Generated Text (Alberto Muñoz-Ortiz et al., 2023)

{{<citation>}}

Alberto Muñoz-Ortiz, Carlos Gómez-Rodríguez, David Vilares. (2023)  
**Contrasting Linguistic Patterns in Human and LLM-Generated Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09067v1)  

---


**ABSTRACT**  
We conduct a quantitative analysis contrasting human-written English news text with comparable large language model (LLM) output from 4 LLMs from the LLaMa family. Our analysis spans several measurable linguistic dimensions, including morphological, syntactic, psychometric and sociolinguistic aspects. The results reveal various measurable differences between human and AI-generated texts. Among others, human texts exhibit more scattered sentence length distributions, a distinct use of dependency and constituent types, shorter constituents, and more aggressive emotions (fear, disgust) than LLM-generated texts. LLM outputs use more numbers, symbols and auxiliaries (suggesting objective language) than human texts, as well as more pronouns. The sexist bias prevalent in human text is also expressed by LLMs.

{{</citation>}}


### (69/97) Reinforced Self-Training (ReST) for Language Modeling (Caglar Gulcehre et al., 2023)

{{<citation>}}

Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, Nando de Freitas. (2023)  
**Reinforced Self-Training (ReST) for Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.08998v2)  

---


**ABSTRACT**  
Reinforcement learning from human feedback (RLHF) can improve the quality of large language model's (LLM) outputs by aligning them with human preferences. We propose a simple algorithm for aligning LLMs with human preferences inspired by growing batch reinforcement learning (RL), which we call Reinforced Self-Training (ReST). Given an initial LLM policy, ReST produces a dataset by generating samples from the policy, which are then used to improve the LLM policy using offline RL algorithms. ReST is more efficient than typical online RLHF methods because the training dataset is produced offline, which allows data reuse. While ReST is a general approach applicable to all generative learning settings, we focus on its application to machine translation. Our results show that ReST can substantially improve translation quality, as measured by automated metrics and human evaluation on machine translation benchmarks in a compute and sample-efficient manner.

{{</citation>}}


### (70/97) Evaluation of really good grammatical error correction (Robert Östling et al., 2023)

{{<citation>}}

Robert Östling, Katarina Gillholm, Murathan Kurfalı, Marie Mattson, Mats Wirén. (2023)  
**Evaluation of really good grammatical error correction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2308.08982v1)  

---


**ABSTRACT**  
Although rarely stated, in practice, Grammatical Error Correction (GEC) encompasses various models with distinct objectives, ranging from grammatical error detection to improving fluency. Traditional evaluation methods fail to fully capture the full range of system capabilities and objectives. Reference-based evaluations suffer from limitations in capturing the wide variety of possible correction and the biases introduced during reference creation and is prone to favor fixing local errors over overall text improvement. The emergence of large language models (LLMs) has further highlighted the shortcomings of these evaluation strategies, emphasizing the need for a paradigm shift in evaluation methodology. In the current study, we perform a comprehensive evaluation of various GEC systems using a recently published dataset of Swedish learner texts. The evaluation is performed using established evaluation metrics as well as human judges. We find that GPT-3 in a few-shot setting by far outperforms previous grammatical error correction systems for Swedish, a language comprising only 0.11% of its training data. We also found that current evaluation methods contain undesirable biases that a human evaluation is able to reveal. We suggest using human post-editing of GEC system outputs to analyze the amount of change required to reach native-level human performance on the task, and provide a dataset annotated with human post-edits and assessments of grammaticality, fluency and meaning preservation of GEC system outputs.

{{</citation>}}


### (71/97) Beam Retrieval: General End-to-End Retrieval for Multi-Hop Question Answering (Jiahao Zhang et al., 2023)

{{<citation>}}

Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Yong Liu, Shen Huang. (2023)  
**Beam Retrieval: General End-to-End Retrieval for Multi-Hop Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.08973v1)  

---


**ABSTRACT**  
Multi-hop QA involves finding multiple relevant passages and step-by-step reasoning to answer complex questions. While previous approaches have developed retrieval modules for selecting relevant passages, they face challenges in scenarios beyond two hops, owing to the limited performance of one-step methods and the failure of two-step methods when selecting irrelevant passages in earlier stages. In this work, we introduce Beam Retrieval, a general end-to-end retrieval framework for multi-hop QA. This approach maintains multiple partial hypotheses of relevant passages at each step, expanding the search space and reducing the risk of missing relevant passages. Moreover, Beam Retrieval jointly optimizes an encoder and two classification heads by minimizing the combined loss across all hops. To establish a complete QA system, we incorporate a supervised reader or a zero-shot GPT-3.5. Experimental results demonstrate that Beam Retrieval achieves a nearly 50% improvement compared with baselines on challenging MuSiQue-Ans, and it also surpasses all previous retrievers on HotpotQA and 2WikiMultiHopQA. Providing high-quality context, Beam Retrieval helps our supervised reader achieve new state-of-the-art performance and substantially improves (up to 28.8 points) the QA performance of zero-shot GPT-3.5.

{{</citation>}}


### (72/97) CMB: A Comprehensive Medical Benchmark in Chinese (Xidong Wang et al., 2023)

{{<citation>}}

Xidong Wang, Guiming Hardy Chen, Dingjie Song, Zhiyi Zhang, Zhihong Chen, Qingying Xiao, Feng Jiang, Jianquan Li, Xiang Wan, Benyou Wang, Haizhou Li. (2023)  
**CMB: A Comprehensive Medical Benchmark in Chinese**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.08833v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) provide a possibility to make a great breakthrough in medicine. The establishment of a standardized medical benchmark becomes a fundamental cornerstone to measure progression. However, medical environments in different regions have their local characteristics, e.g., the ubiquity and significance of traditional Chinese medicine within China. Therefore, merely translating English-based medical evaluation may result in \textit{contextual incongruities} to a local region. To solve the issue, we propose a localized medical benchmark called CMB, a Comprehensive Medical Benchmark in Chinese, designed and rooted entirely within the native Chinese linguistic and cultural framework. While traditional Chinese medicine is integral to this evaluation, it does not constitute its entirety. Using this benchmark, we have evaluated several prominent large-scale LLMs, including ChatGPT, GPT-4, dedicated Chinese LLMs, and LLMs specialized in the medical domain. It is worth noting that our benchmark is not devised as a leaderboard competition but as an instrument for self-assessment of model advancements. We hope this benchmark could facilitate the widespread adoption and enhancement of medical LLMs within China. Check details in \url{https://cmedbenchmark.llmzoo.com/}.

{{</citation>}}


### (73/97) Factuality Detection using Machine Translation -- a Use Case for German Clinical Text (Mohammed Bin Sumait et al., 2023)

{{<citation>}}

Mohammed Bin Sumait, Aleksandra Gabryszak, Leonhard Hennig, Roland Roller. (2023)  
**Factuality Detection using Machine Translation -- a Use Case for German Clinical Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Clinical, Machine Translation  
[Paper Link](http://arxiv.org/abs/2308.08827v1)  

---


**ABSTRACT**  
Factuality can play an important role when automatically processing clinical text, as it makes a difference if particular symptoms are explicitly not present, possibly present, not mentioned, or affirmed. In most cases, a sufficient number of examples is necessary to handle such phenomena in a supervised machine learning setting. However, as clinical text might contain sensitive information, data cannot be easily shared. In the context of factuality detection, this work presents a simple solution using machine translation to translate English data to German to train a transformer-based factuality detection model.

{{</citation>}}


### (74/97) Linguistically-Informed Neural Architectures for Lexical, Syntactic and Semantic Tasks in Sanskrit (Jivnesh Sandhan, 2023)

{{<citation>}}

Jivnesh Sandhan. (2023)  
**Linguistically-Informed Neural Architectures for Lexical, Syntactic and Semantic Tasks in Sanskrit**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.08807v1)  

---


**ABSTRACT**  
The primary focus of this thesis is to make Sanskrit manuscripts more accessible to the end-users through natural language technologies. The morphological richness, compounding, free word orderliness, and low-resource nature of Sanskrit pose significant challenges for developing deep learning solutions. We identify four fundamental tasks, which are crucial for developing a robust NLP technology for Sanskrit: word segmentation, dependency parsing, compound type identification, and poetry analysis. The first task, Sanskrit Word Segmentation (SWS), is a fundamental text processing task for any other downstream applications. However, it is challenging due to the sandhi phenomenon that modifies characters at word boundaries. Similarly, the existing dependency parsing approaches struggle with morphologically rich and low-resource languages like Sanskrit. Compound type identification is also challenging for Sanskrit due to the context-sensitive semantic relation between components. All these challenges result in sub-optimal performance in NLP applications like question answering and machine translation. Finally, Sanskrit poetry has not been extensively studied in computational linguistics.   While addressing these challenges, this thesis makes various contributions: (1) The thesis proposes linguistically-informed neural architectures for these tasks. (2) We showcase the interpretability and multilingual extension of the proposed systems. (3) Our proposed systems report state-of-the-art performance. (4) Finally, we present a neural toolkit named SanskritShala, a web-based application that provides real-time analysis of input for various NLP tasks. Overall, this thesis contributes to making Sanskrit manuscripts more accessible by developing robust NLP technology and releasing various resources, datasets, and web-based toolkit.

{{</citation>}}


### (75/97) Do you really follow me? Adversarial Instructions for Evaluating the Robustness of Large Language Models (Zekun Li et al., 2023)

{{<citation>}}

Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan. (2023)  
**Do you really follow me? Adversarial Instructions for Evaluating the Robustness of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10819v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown remarkable proficiency in following instructions, making them valuable in customer-facing applications. However, their impressive capabilities also raise concerns about the amplification of risks posed by adversarial instructions, which can be injected into the model input by third-party attackers to manipulate LLMs' original instructions and prompt unintended actions and content. Therefore, it is crucial to understand LLMs' ability to accurately discern which instructions to follow to ensure their safe deployment in real-world scenarios. In this paper, we propose a pioneering benchmark for automatically evaluating the robustness of LLMs against adversarial instructions. The objective of this benchmark is to quantify the extent to which LLMs are influenced by injected adversarial instructions and assess their ability to differentiate between these adversarial instructions and original user instructions. Through experiments conducted with state-of-the-art instruction-following LLMs, we uncover significant limitations in their robustness against adversarial instruction attacks. Furthermore, our findings indicate that prevalent instruction-tuned models are prone to being overfitted to follow any instruction phrase in the prompt without truly understanding which instructions should be followed. This highlights the need to address the challenge of training models to comprehend prompts instead of merely following instruction phrases and completing the text.

{{</citation>}}


### (76/97) Chinese Spelling Correction as Rephrasing Language Model (Linfeng Liu et al., 2023)

{{<citation>}}

Linfeng Liu, Hongqiu Wu, Hai Zhao. (2023)  
**Chinese Spelling Correction as Rephrasing Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.08796v1)  

---


**ABSTRACT**  
This paper studies Chinese Spelling Correction (CSC), which aims to detect and correct potential spelling errors in a given sentence. Current state-of-the-art methods regard CSC as a sequence tagging task and fine-tune BERT-based models on sentence pairs. However, we note a critical flaw in the process of tagging one character to another, that the correction is excessively conditioned on the error. This is opposite from human mindset, where individuals rephrase the complete sentence based on its semantics, rather than solely on the error patterns memorized before. Such a counter-intuitive learning process results in the bottleneck of generalizability and transferability of machine spelling correction. To address this, we propose $Rephrasing Language Modeling$ (ReLM), where the model is trained to rephrase the entire sentence by infilling additional slots, instead of character-to-character tagging. This novel training paradigm achieves the new state-of-the-art results across fine-tuned and zero-shot CSC benchmarks, outperforming previous counterparts by a large margin. Our method also learns transferable language representation when CSC is jointly trained with other tasks.

{{</citation>}}


### (77/97) Task Relation Distillation and Prototypical Pseudo Label for Incremental Named Entity Recognition (Duzhen Zhang et al., 2023)

{{<citation>}}

Duzhen Zhang, Hongliu Li, Wei Cong, Rongtao Xu, Jiahua Dong, Xiuyi Chen. (2023)  
**Task Relation Distillation and Prototypical Pseudo Label for Incremental Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2308.08793v1)  

---


**ABSTRACT**  
Incremental Named Entity Recognition (INER) involves the sequential learning of new entity types without accessing the training data of previously learned types. However, INER faces the challenge of catastrophic forgetting specific for incremental learning, further aggravated by background shift (i.e., old and future entity types are labeled as the non-entity type in the current task). To address these challenges, we propose a method called task Relation Distillation and Prototypical pseudo label (RDP) for INER. Specifically, to tackle catastrophic forgetting, we introduce a task relation distillation scheme that serves two purposes: 1) ensuring inter-task semantic consistency across different incremental learning tasks by minimizing inter-task relation distillation loss, and 2) enhancing the model's prediction confidence by minimizing intra-task self-entropy loss. Simultaneously, to mitigate background shift, we develop a prototypical pseudo label strategy that distinguishes old entity types from the current non-entity type using the old model. This strategy generates high-quality pseudo labels by measuring the distances between token embeddings and type-wise prototypes. We conducted extensive experiments on ten INER settings of three benchmark datasets (i.e., CoNLL2003, I2B2, and OntoNotes5). The results demonstrate that our method achieves significant improvements over the previous state-of-the-art methods, with an average increase of 6.08% in Micro F1 score and 7.71% in Macro F1 score.

{{</citation>}}


### (78/97) Exploring Demonstration Ensembling for In-context Learning (Muhammad Khalifa et al., 2023)

{{<citation>}}

Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, Lu Wang. (2023)  
**Exploring Demonstration Ensembling for In-context Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2308.08780v2)  

---


**ABSTRACT**  
In-context learning (ICL) operates by showing language models (LMs) examples of input-output pairs for a given task, i.e., demonstrations. The standard approach for ICL is to prompt the LM with concatenated demonstrations followed by the test input. This approach suffers from some issues. First, concatenation offers almost no control over the contribution of each demo to the model prediction. This can be sub-optimal when some demonstrations are irrelevant to the test example. Second, due to the input length limit of some transformer models, it might be infeasible to fit many examples into the context, especially when dealing with long-input tasks. In this work, we explore Demonstration Ensembling (DENSE) as an alternative to simple concatenation. DENSE predicts outputs using subsets (i.e., buckets) of the demonstrations and then combines the output probabilities resulting from each subset to produce the final prediction. We study different ensembling methods using GPT-j and experiment on 12 language tasks. Our experiments show weighted max ensembling to outperform vanilla concatenation by as large as 2.4 average points. Code available at https://github.com/mukhal/icl-ensembling.

{{</citation>}}


### (79/97) Differential Privacy, Linguistic Fairness, and Training Data Influence: Impossibility and Possibility Theorems for Multilingual Language Models (Phillip Rust et al., 2023)

{{<citation>}}

Phillip Rust, Anders Søgaard. (2023)  
**Differential Privacy, Linguistic Fairness, and Training Data Influence: Impossibility and Possibility Theorems for Multilingual Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: BERT, BLOOM, Language Model, Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2308.08774v1)  

---


**ABSTRACT**  
Language models such as mBERT, XLM-R, and BLOOM aim to achieve multilingual generalization or compression to facilitate transfer to a large number of (potentially unseen) languages. However, these models should ideally also be private, linguistically fair, and transparent, by relating their predictions to training data. Can these requirements be simultaneously satisfied? We show that multilingual compression and linguistic fairness are compatible with differential privacy, but that differential privacy is at odds with training data influence sparsity, an objective for transparency. We further present a series of experiments on two common NLP tasks and evaluate multilingual compression and training data influence sparsity under different privacy guarantees, exploring these trade-offs in more detail. Our results suggest that we need to develop ways to jointly optimize for these objectives in order to find practical trade-offs.

{{</citation>}}


### (80/97) Discrete Prompt Compression with Reinforcement Learning (Hoyoun Jung et al., 2023)

{{<citation>}}

Hoyoun Jung, Kyung-Joong Kim. (2023)  
**Discrete Prompt Compression with Reinforcement Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.08758v1)  

---


**ABSTRACT**  
Instruction-tuned Language Models (LMs) are widely used by users to address various problems with task-specific prompts. Constraints associated with the context window length and computational costs encourage the development of compressed prompts. Existing methods rely heavily on training embeddings, which are designed to accommodate multiple token meanings. This presents challenges in terms of interpretability, a fixed number of embedding tokens, reusability across different LMs, and inapplicability when interacting with black-box APIs. This study proposes prompt compression with reinforcement learning (PCRL), a novel discrete prompt compression method that addresses these issues. PCRL employs a computationally efficient policy network that directly edits prompts. The PCRL training approach can be flexibly applied to various types of LMs, as well as decoder-only and encoder-decoder architecture, and can be trained without gradient access to LMs or labeled data. PCRL achieves an average reduction of 24.6% in token count across various instruction prompts while preserving performance. Further, we demonstrate that the learned policy can be transferred to larger LMs, and through various analyses, we aid the understanding of token importance within prompts.

{{</citation>}}


### (81/97) An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning (Yun Luo et al., 2023)

{{<citation>}}

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, Yue Zhang. (2023)  
**An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM, Language Model  
[Paper Link](http://arxiv.org/abs/2308.08747v2)  

---


**ABSTRACT**  
Catastrophic forgetting (CF) is a phenomenon that occurs in machine learning when a model forgets previously learned information as it learns new information. As large language models (LLMs) have shown excellent performance, it is interesting to uncover whether CF exists in the continual fine-tuning of LLMs. In this study, we empirically evaluate the forgetting phenomenon in LLMs' knowledge, from the perspectives of domain knowledge, reasoning, and reading comprehension. The experiments demonstrate that catastrophic forgetting is generally observed in LLMs ranging from 1b to 7b. Furthermore, as the scale increases, the severity of forgetting also intensifies. Comparing the decoder-only model BLOOMZ with the encoder-decoder model mT0, BLOOMZ suffers less forgetting and maintains more knowledge. We also observe that LLMs can mitigate language bias (e.g. gender bias) during continual fine-tuning. Moreover, we find that ALPACA can maintain more knowledge and capacity compared with LLAMA during the continual fine-tuning, which implies that general instruction tuning can help mitigate the forgetting phenomenon of LLMs in the further fine-tuning process.

{{</citation>}}


### (82/97) PMET: Precise Model Editing in a Transformer (Xiaopeng Li et al., 2023)

{{<citation>}}

Xiaopeng Li, Shasha Li, Shezheng Song, Jing Yang, Jun Ma, Jie Yu. (2023)  
**PMET: Precise Model Editing in a Transformer**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Attention, Language Model, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.08742v1)  

---


**ABSTRACT**  
Model editing techniques modify a minor proportion of knowledge in Large Language Models (LLMs) at a relatively low cost, which have demonstrated notable success. Existing methods assume Transformer Layer (TL) hidden states are values of key-value memories of the Feed-Forward Network (FFN). They usually optimize the TL hidden states to memorize target knowledge and use it to update the weights of the FFN in LLMs. However, the information flow of TL hidden states comes from three parts: Multi-Head Self-Attention (MHSA), FFN, and residual connections. Existing methods neglect the fact that the TL hidden states contains information not specifically required for FFN. Consequently, the performance of model editing decreases. To achieve more precise model editing, we analyze hidden states of MHSA and FFN, finding that MHSA encodes certain general knowledge extraction patterns. This implies that MHSA weights do not require updating when new knowledge is introduced. Based on above findings, we introduce PMET, which simultaneously optimizes Transformer Component (TC, namely MHSA and FFN) hidden states, while only using the optimized TC hidden states of FFN to precisely update FFN weights. Our experiments demonstrate that PMET exhibits state-of-the-art performance on both the \textsc{counterfact} and zsRE datasets. Our ablation experiments substantiate the effectiveness of our enhancements, further reinforcing the finding that the MHSA encodes certain general knowledge extraction patterns and indicating its storage of a small amount of factual knowledge. Our code is available at \url{https://github.com/xpq-tech/PMET.git}.

{{</citation>}}


### (83/97) Enhancing Phrase Representation by Information Bottleneck Guided Text Diffusion Process for Keyphrase Extraction (Yuanzhen Luo et al., 2023)

{{<citation>}}

Yuanzhen Luo, Qingyu Zhou, Feng Zhou. (2023)  
**Enhancing Phrase Representation by Information Bottleneck Guided Text Diffusion Process for Keyphrase Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Keyphrase Extraction, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.08739v1)  

---


**ABSTRACT**  
Keyphrase extraction (KPE) is an important task in Natural Language Processing for many scenarios, which aims to extract keyphrases that are present in a given document. Many existing supervised methods treat KPE as sequential labeling, span-level classification, or generative tasks. However, these methods lack the ability to utilize keyphrase information, which may result in biased results. In this study, we propose Diff-KPE, which leverages the supervised Variational Information Bottleneck (VIB) to guide the text diffusion process for generating enhanced keyphrase representations. Diff-KPE first generates the desired keyphrase embeddings conditioned on the entire document and then injects the generated keyphrase embeddings into each phrase representation. A ranking network and VIB are then optimized together with rank loss and classification loss, respectively. This design of Diff-KPE allows us to rank each candidate phrase by utilizing both the information of keyphrases and the document. Experiments show that Diff-KPE outperforms existing KPE methods on a large open domain keyphrase extraction benchmark, OpenKP, and a scientific domain dataset, KP20K.

{{</citation>}}


### (84/97) Decoding Emotions: A comprehensive Multilingual Study of Speech Models for Speech Emotion Recognition (Anant Singh et al., 2023)

{{<citation>}}

Anant Singh, Akshat Gupta. (2023)  
**Decoding Emotions: A comprehensive Multilingual Study of Speech Models for Speech Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Emotion Recognition, Multilingual  
[Paper Link](http://arxiv.org/abs/2308.08713v1)  

---


**ABSTRACT**  
Recent advancements in transformer-based speech representation models have greatly transformed speech processing. However, there has been limited research conducted on evaluating these models for speech emotion recognition (SER) across multiple languages and examining their internal representations. This article addresses these gaps by presenting a comprehensive benchmark for SER with eight speech representation models and six different languages. We conducted probing experiments to gain insights into inner workings of these models for SER. We find that using features from a single optimal layer of a speech model reduces the error rate by 32\% on average across seven datasets when compared to systems where features from all layers of speech models are used. We also achieve state-of-the-art results for German and Persian languages. Our probing results indicate that the middle layers of speech models capture the most important emotional information for speech emotion recognition.

{{</citation>}}


## stat.ME (1)



### (85/97) Spectral information criterion for automatic elbow detection (L. Martino et al., 2023)

{{<citation>}}

L. Martino, R. San Millan-Castillo, E. Morgado. (2023)  
**Spectral information criterion for automatic elbow detection**  

---
Primary Category: stat.ME  
Categories: cs-AI, eess-SP, stat-ME, stat-ML, stat.ME  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09108v1)  

---


**ABSTRACT**  
We introduce a generalized information criterion that contains other well-known information criteria, such as Bayesian information Criterion (BIC) and Akaike information criterion (AIC), as special cases. Furthermore, the proposed spectral information criterion (SIC) is also more general than the other information criteria, e.g., since the knowledge of a likelihood function is not strictly required. SIC extracts geometric features of the error curve and, as a consequence, it can be considered an automatic elbow detector. SIC provides a subset of all possible models, with a cardinality that often is much smaller than the total number of possible models. The elements of this subset are elbows of the error curve. A practical rule for selecting a unique model within the sets of elbows is suggested as well. Theoretical invariance properties of SIC are analyzed. Moreover, we test SIC in ideal scenarios where provides always the optimal expected results. We also test SIC in several numerical experiments: some involving synthetic data, and two experiments involving real datasets. They are all real-world applications such as clustering, variable selection, or polynomial order selection, to name a few. The results show the benefits of the proposed scheme. Matlab code related to the experiments is also provided. Possible future research lines are finally discussed.

{{</citation>}}


## cs.DC (2)



### (86/97) Towards Lightweight Data Integration using Multi-workflow Provenance and Data Observability (Renan Souza et al., 2023)

{{<citation>}}

Renan Souza, Tyler J. Skluzacek, Sean R. Wilkinson, Maxim Ziatdinov, Rafael Ferreira da Silva. (2023)  
**Towards Lightweight Data Integration using Multi-workflow Provenance and Data Observability**  

---
Primary Category: cs.DC  
Categories: 65Y05, 68P15, I-2; H-2; C-4; J-2, cs-AI, cs-DB, cs-DC, cs-LG, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09004v1)  

---


**ABSTRACT**  
Modern large-scale scientific discovery requires multidisciplinary collaboration across diverse computing facilities, including High Performance Computing (HPC) machines and the Edge-to-Cloud continuum. Integrated data analysis plays a crucial role in scientific discovery, especially in the current AI era, by enabling Responsible AI development, FAIR, Reproducibility, and User Steering. However, the heterogeneous nature of science poses challenges such as dealing with multiple supporting tools, cross-facility environments, and efficient HPC execution. Building on data observability, adapter system design, and provenance, we propose MIDA: an approach for lightweight runtime Multi-workflow Integrated Data Analysis. MIDA defines data observability strategies and adaptability methods for various parallel systems and machine learning tools. With observability, it intercepts the dataflows in the background without requiring instrumentation while integrating domain, provenance, and telemetry data at runtime into a unified database ready for user steering queries. We conduct experiments showing end-to-end multi-workflow analysis integrating data from Dask and MLFlow in a real distributed deep learning use case for materials science that runs on multiple environments with up to 276 GPUs in parallel. We show near-zero overhead running up to 100,000 tasks on 1,680 CPU cores on the Summit supercomputer.

{{</citation>}}


### (87/97) Multi-FedLS: a Framework for Cross-Silo Federated Learning Applications on Multi-Cloud Environments (Rafaela C. Brum et al., 2023)

{{<citation>}}

Rafaela C. Brum, Maria Clicia Stelling de Castro, Luciana Arantes, Lúcia Maria de A. Drummond, Pierre Sens. (2023)  
**Multi-FedLS: a Framework for Cross-Silo Federated Learning Applications on Multi-Cloud Environments**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AWS, GCP  
[Paper Link](http://arxiv.org/abs/2308.08967v1)  

---


**ABSTRACT**  
Federated Learning (FL) is a distributed Machine Learning (ML) technique that can benefit from cloud environments while preserving data privacy. We propose Multi-FedLS, a framework that manages multi-cloud resources, reducing execution time and financial costs of Cross-Silo Federated Learning applications by using preemptible VMs, cheaper than on-demand ones but that can be revoked at any time. Our framework encloses four modules: Pre-Scheduling, Initial Mapping, Fault Tolerance, and Dynamic Scheduler. This paper extends our previous work \cite{brum2022sbac} by formally describing the Multi-FedLS resource manager framework and its modules. Experiments were conducted with three Cross-Silo FL applications on CloudLab and a proof-of-concept confirms that Multi-FedLS can be executed on a multi-cloud composed by AWS and GCP, two commercial cloud providers. Results show that the problem of executing Cross-Silo FL applications in multi-cloud environments with preemptible VMs can be efficiently resolved using a mathematical formulation, fault tolerance techniques, and a simple heuristic to choose a new VM in case of revocation.

{{</citation>}}


## math.OC (1)



### (88/97) Hitting the High-Dimensional Notes: An ODE for SGD learning dynamics on GLMs and multi-index models (Elizabeth Collins-Woodfin et al., 2023)

{{<citation>}}

Elizabeth Collins-Woodfin, Courtney Paquette, Elliot Paquette, Inbar Seroussi. (2023)  
**Hitting the High-Dimensional Notes: An ODE for SGD learning dynamics on GLMs and multi-index models**  

---
Primary Category: math.OC  
Categories: cs-LG, math-OC, math-PR, math-ST, math.OC, stat-ML, stat-TH  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2308.08977v1)  

---


**ABSTRACT**  
We analyze the dynamics of streaming stochastic gradient descent (SGD) in the high-dimensional limit when applied to generalized linear models and multi-index models (e.g. logistic regression, phase retrieval) with general data-covariance. In particular, we demonstrate a deterministic equivalent of SGD in the form of a system of ordinary differential equations that describes a wide class of statistics, such as the risk and other measures of sub-optimality. This equivalence holds with overwhelming probability when the model parameter count grows proportionally to the number of data. This framework allows us to obtain learning rate thresholds for stability of SGD as well as convergence guarantees. In addition to the deterministic equivalent, we introduce an SDE with a simplified diffusion coefficient (homogenized SGD) which allows us to analyze the dynamics of general statistics of SGD iterates. Finally, we illustrate this theory on some standard examples and show numerical simulations which give an excellent match to the theory.

{{</citation>}}


## eess.AS (3)



### (89/97) Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement (Ye-Xin Lu et al., 2023)

{{<citation>}}

Ye-Xin Lu, Yang Ai, Zhen-Hua Ling. (2023)  
**Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.08926v1)  

---


**ABSTRACT**  
Phase information has a significant impact on speech perceptual quality and intelligibility. However, existing speech enhancement methods encounter limitations in explicit phase estimation due to the non-structural nature and wrapping characteristics of the phase, leading to a bottleneck in enhanced speech quality. To overcome the above issue, in this paper, we proposed MP-SENet, a novel Speech Enhancement Network which explicitly enhances Magnitude and Phase spectra in parallel. The proposed MP-SENet adopts a codec architecture in which the encoder and decoder are bridged by time-frequency Transformers along both time and frequency dimensions. The encoder aims to encode time-frequency representations derived from the input distorted magnitude and phase spectra. The decoder comprises dual-stream magnitude and phase decoders, directly enhancing magnitude and wrapped phase spectra by incorporating a magnitude estimation architecture and a phase parallel estimation architecture, respectively. To train the MP-SENet model effectively, we define multi-level loss functions, including mean square error and perceptual metric loss of magnitude spectra, anti-wrapping loss of phase spectra, as well as mean square error and consistency loss of short-time complex spectra. Experimental results demonstrate that our proposed MP-SENet excels in high-quality speech enhancement across multiple tasks, including speech denoising, dereverberation, and bandwidth extension. Compared to existing phase-aware speech enhancement methods, it successfully avoids the bidirectional compensation effect between the magnitude and phase, leading to a better harmonic restoration. Notably, for the speech denoising task, the MP-SENet yields a state-of-the-art performance with a PESQ of 3.60 on the public VoiceBank+DEMAND dataset.

{{</citation>}}


### (90/97) Graph Neural Network Backend for Speaker Recognition (Liang He et al., 2023)

{{<citation>}}

Liang He, Ruida Li, Mengqi Niu. (2023)  
**Graph Neural Network Backend for Speaker Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.08767v1)  

---


**ABSTRACT**  
Currently, most speaker recognition backends, such as cosine, linear discriminant analysis (LDA), or probabilistic linear discriminant analysis (PLDA), make decisions by calculating similarity or distance between enrollment and test embeddings which are already extracted from neural networks. However, for each embedding, the local structure of itself and its neighbor embeddings in the low-dimensional space is different, which may be helpful for the recognition but is often ignored. In order to take advantage of it, we propose a graph neural network (GNN) backend to mine latent relationships among embeddings for classification. We assume all the embeddings as nodes on a graph, and their edges are computed based on some similarity function, such as cosine, LDA+cosine, or LDA+PLDA. We study different graph settings and explore variants of GNN to find a better message passing and aggregation way to accomplish the recognition task. Experimental results on NIST SRE14 i-vector challenging, VoxCeleb1-O, VoxCeleb1-E, and VoxCeleb1-H datasets demonstrate that our proposed GNN backends significantly outperform current mainstream methods.

{{</citation>}}


### (91/97) The DKU-MSXF Speaker Verification System for the VoxCeleb Speaker Recognition Challenge 2023 (Ze Li et al., 2023)

{{<citation>}}

Ze Li, Yuke Lin, Xiaoyi Qin, Ning Jiang, Guoqing Zhao, Ming Li. (2023)  
**The DKU-MSXF Speaker Verification System for the VoxCeleb Speaker Recognition Challenge 2023**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2308.08766v1)  

---


**ABSTRACT**  
This paper is the system description of the DKU-MSXF System for the track1, track2 and track3 of the VoxCeleb Speaker Recognition Challenge 2023 (VoxSRC-23). For Track 1, we utilize a network structure based on ResNet for training. By constructing a cross-age QMF training set, we achieve a substantial improvement in system performance. For Track 2, we inherite the pre-trained model from Track 1 and conducte mixed training by incorporating the VoxBlink-clean dataset. In comparison to Track 1, the models incorporating VoxBlink-clean data exhibit a performance improvement by more than 10% relatively. For Track3, the semi-supervised domain adaptation task, a novel pseudo-labeling method based on triple thresholds and sub-center purification is adopted to make domain adaptation. The final submission achieves mDCF of 0.1243 in task1, mDCF of 0.1165 in Track 2 and EER of 4.952% in Track 3.

{{</citation>}}


## cs.IT (1)



### (92/97) Unfolding for Joint Channel Estimation and Symbol Detection in MIMO Communication Systems (Swati Bhattacharya et al., 2023)

{{<citation>}}

Swati Bhattacharya, K. V. S. Hari, Yonina C. Eldar. (2023)  
**Unfolding for Joint Channel Estimation and Symbol Detection in MIMO Communication Systems**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.08917v2)  

---


**ABSTRACT**  
This paper proposes a Joint Channel Estimation and Symbol Detection (JED) scheme for Multiple-Input Multiple-Output (MIMO) wireless communication systems. Our proposed method for JED using Alternating Direction Method of Multipliers (JED-ADMM) and its model-based neural network version JED using Unfolded ADMM (JED-U-ADMM) markedly improve the symbol detection performance over JED using Alternating Minimization (JED-AM) for a range of MIMO antenna configurations. Both proposed algorithms exploit the non-smooth constraint, that occurs as a result of the Quadrature Amplitude Modulation (QAM) data symbols, to effectively improve the performance using the ADMM iterations. The proposed unfolded network JED-U-ADMM consists of a few trainable parameters and requires a small training set. We show the efficacy of the proposed methods for both uncorrelated and correlated MIMO channels. For certain configurations, the gain in SNR for a desired BER of $10^{-2}$ for the proposed JED-ADMM and JED-U-ADMM is upto $4$ dB and is also accompanied by a significant reduction in computational complexity of upto $75\%$, depending on the MIMO configuration, as compared to the complexity of JED-AM.

{{</citation>}}


## q-bio.GN (1)



### (93/97) MoCLIM: Towards Accurate Cancer Subtyping via Multi-Omics Contrastive Learning with Omics-Inference Modeling (Ziwei Yang et al., 2023)

{{<citation>}}

Ziwei Yang, Zheng Chen, Yasuko Matsubara, Yasushi Sakurai. (2023)  
**MoCLIM: Towards Accurate Cancer Subtyping via Multi-Omics Contrastive Learning with Omics-Inference Modeling**  

---
Primary Category: q-bio.GN  
Categories: cs-AI, cs-LG, q-bio-GN, q-bio.GN  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.09725v1)  

---


**ABSTRACT**  
Precision medicine fundamentally aims to establish causality between dysregulated biochemical mechanisms and cancer subtypes. Omics-based cancer subtyping has emerged as a revolutionary approach, as different level of omics records the biochemical products of multistep processes in cancers. This paper focuses on fully exploiting the potential of multi-omics data to improve cancer subtyping outcomes, and hence developed MoCLIM, a representation learning framework. MoCLIM independently extracts the informative features from distinct omics modalities. Using a unified representation informed by contrastive learning of different omics modalities, we can well-cluster the subtypes, given cancer, into a lower latent space. This contrast can be interpreted as a projection of inter-omics inference observed in biological networks. Experimental results on six cancer datasets demonstrate that our approach significantly improves data fit and subtyping performance in fewer high-dimensional cancer instances. Moreover, our framework incorporates various medical evaluations as the final component, providing high interpretability in medical analysis.

{{</citation>}}


## eess.SY (1)



### (94/97) Federated Reinforcement Learning for Electric Vehicles Charging Control on Distribution Networks (Junkai Qian et al., 2023)

{{<citation>}}

Junkai Qian, Yuning Jiang, Xin Liu, Qing Wang, Ting Wang, Yuanming Shi, Wei Chen. (2023)  
**Federated Reinforcement Learning for Electric Vehicles Charging Control on Distribution Networks**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-MA, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.08792v1)  

---


**ABSTRACT**  
With the growing popularity of electric vehicles (EVs), maintaining power grid stability has become a significant challenge. To address this issue, EV charging control strategies have been developed to manage the switch between vehicle-to-grid (V2G) and grid-to-vehicle (G2V) modes for EVs. In this context, multi-agent deep reinforcement learning (MADRL) has proven its effectiveness in EV charging control. However, existing MADRL-based approaches fail to consider the natural power flow of EV charging/discharging in the distribution network and ignore driver privacy. To deal with these problems, this paper proposes a novel approach that combines multi-EV charging/discharging with a radial distribution network (RDN) operating under optimal power flow (OPF) to distribute power flow in real time. A mathematical model is developed to describe the RDN load. The EV charging control problem is formulated as a Markov Decision Process (MDP) to find an optimal charging control strategy that balances V2G profits, RDN load, and driver anxiety. To effectively learn the optimal EV charging control strategy, a federated deep reinforcement learning algorithm named FedSAC is further proposed. Comprehensive simulation results demonstrate the effectiveness and superiority of our proposed algorithm in terms of the diversity of the charging control strategy, the power fluctuations on RDN, the convergence efficiency, and the generalization ability.

{{</citation>}}


## quant-ph (1)



### (95/97) A Feasibility-Preserved Quantum Approximate Solver for the Capacitated Vehicle Routing Problem (Ningyi Xie et al., 2023)

{{<citation>}}

Ningyi Xie, Xinwei Lee, Dongsheng Cai, Yoshiyuki Saito, Nobuyoshi Asai, Hoong Chuin Lau. (2023)  
**A Feasibility-Preserved Quantum Approximate Solver for the Capacitated Vehicle Routing Problem**  

---
Primary Category: quant-ph  
Categories: cs-DS, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.08785v1)  

---


**ABSTRACT**  
The Capacitated Vehicle Routing Problem (CVRP) is an NP-optimization problem (NPO) that arises in various fields including transportation and logistics. The CVRP extends from the Vehicle Routing Problem (VRP), aiming to determine the most efficient plan for a fleet of vehicles to deliver goods to a set of customers, subject to the limited carrying capacity of each vehicle. As the number of possible solutions skyrockets when the number of customers increases, finding the optimal solution remains a significant challenge. Recently, a quantum-classical hybrid algorithm known as Quantum Approximate Optimization Algorithm (QAOA) can provide better solutions in some cases of combinatorial optimization problems, compared to classical heuristics. However, the QAOA exhibits a diminished ability to produce high-quality solutions for some constrained optimization problems including the CVRP. One potential approach for improvement involves a variation of the QAOA known as the Grover-Mixer Quantum Alternating Operator Ansatz (GM-QAOA). In this work, we attempt to use GM-QAOA to solve the CVRP. We present a new binary encoding for the CVRP, with an alternative objective function of minimizing the shortest path that bypasses the vehicle capacity constraint of the CVRP. The search space is further restricted by the Grover-Mixer. We examine and discuss the effectiveness of the proposed solver through its application to several illustrative examples.

{{</citation>}}


## econ.GN (1)



### (96/97) Large Language Models at Work in China's Labor Market (Qin Chen et al., 2023)

{{<citation>}}

Qin Chen, Jinfeng Ge, Huaqing Xie, Xingcheng Xu, Yanqing Yang. (2023)  
**Large Language Models at Work in China's Labor Market**  

---
Primary Category: econ.GN  
Categories: cs-AI, cs-CY, econ-GN, econ.GN, q-fin-EC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.08776v1)  

---


**ABSTRACT**  
This paper explores the potential impacts of large language models (LLMs) on the Chinese labor market. We analyze occupational exposure to LLM capabilities by incorporating human expertise and LLM classifications, following Eloundou et al. (2023)'s methodology. We then aggregate occupation exposure to the industry level to obtain industry exposure scores. The results indicate a positive correlation between occupation exposure and wage levels/experience premiums, suggesting higher-paying and experience-intensive jobs may face greater displacement risks from LLM-powered software. The industry exposure scores align with expert assessments and economic intuitions. We also develop an economic growth model incorporating industry exposure to quantify the productivity-employment trade-off from AI adoption. Overall, this study provides an analytical basis for understanding the labor market impacts of increasingly capable AI systems in China. Key innovations include the occupation-level exposure analysis, industry aggregation approach, and economic modeling incorporating AI adoption and labor market effects. The findings will inform policymakers and businesses on strategies for maximizing the benefits of AI while mitigating adverse disruption risks.

{{</citation>}}


## cs.RO (1)



### (97/97) ReProHRL: Towards Multi-Goal Navigation in the Real World using Hierarchical Agents (Tejaswini Manjunath et al., 2023)

{{<citation>}}

Tejaswini Manjunath, Mozhgan Navardi, Prakhar Dixit, Bharat Prakash, Tinoosh Mohsenin. (2023)  
**ReProHRL: Towards Multi-Goal Navigation in the Real World using Hierarchical Agents**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.08737v1)  

---


**ABSTRACT**  
Robots have been successfully used to perform tasks with high precision. In real-world environments with sparse rewards and multiple goals, learning is still a major challenge and Reinforcement Learning (RL) algorithms fail to learn good policies. Training in simulation environments and then fine-tuning in the real world is a common approach. However, adapting to the real-world setting is a challenge. In this paper, we present a method named Ready for Production Hierarchical RL (ReProHRL) that divides tasks with hierarchical multi-goal navigation guided by reinforcement learning. We also use object detectors as a pre-processing step to learn multi-goal navigation and transfer it to the real world. Empirical results show that the proposed ReProHRL method outperforms the state-of-the-art baseline in simulation and real-world environments in terms of both training time and performance. Although both methods achieve a 100% success rate in a simple environment for single goal-based navigation, in a more complex environment and multi-goal setting, the proposed method outperforms the baseline by 18% and 5%, respectively. For the real-world implementation and proof of concept demonstration, we deploy the proposed method on a nano-drone named Crazyflie with a front camera to perform multi-goal navigation experiments.

{{</citation>}}
