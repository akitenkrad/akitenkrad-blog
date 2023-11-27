---
draft: false
title: "arXiv @ 2023.11.23"
date: 2023-11-23
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.23"
    identifier: arxiv_20231123
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AI (16)](#csai-16)
- [cs.CL (23)](#cscl-23)
- [hep-ex (1)](#hep-ex-1)
- [cs.SD (2)](#cssd-2)
- [cs.HC (3)](#cshc-3)
- [cs.CV (36)](#cscv-36)
- [cs.CY (4)](#cscy-4)
- [cs.LG (22)](#cslg-22)
- [cs.CE (1)](#csce-1)
- [math.OC (1)](#mathoc-1)
- [cs.NI (2)](#csni-2)
- [cs.RO (2)](#csro-2)
- [cs.SE (2)](#csse-2)
- [eess.IV (3)](#eessiv-3)
- [cs.CR (3)](#cscr-3)
- [quant-ph (1)](#quant-ph-1)
- [cs.IT (1)](#csit-1)
- [cs.NE (1)](#csne-1)
- [q-bio.GN (1)](#q-biogn-1)
- [math.NA (1)](#mathna-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [eess.AS (1)](#eessas-1)
- [cs.IR (3)](#csir-3)
- [cs.SI (1)](#cssi-1)
- [cs.DB (1)](#csdb-1)
- [cond-mat.stat-mech (1)](#cond-matstat-mech-1)
- [eess.SY (1)](#eesssy-1)

## cs.AI (16)



### (1/135) From Classification to Clinical Insights: Towards Analyzing and Reasoning About Mobile and Behavioral Health Data With Large Language Models (Zachary Englhardt et al., 2023)

{{<citation>}}

Zachary Englhardt, Chengqian Ma, Margaret E. Morris, Xuhai "Orson" Xu, Chun-Cheng Chang, Lianhui Qin, Xin Liu, Shwetak Patel, Vikram Iyer. (2023)  
**From Classification to Clinical Insights: Towards Analyzing and Reasoning About Mobile and Behavioral Health Data With Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Clinical, GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.13063v1)  

---


**ABSTRACT**  
Passively collected behavioral health data from ubiquitous sensors holds significant promise to provide mental health professionals insights from patient's daily lives; however, developing analysis tools to use this data in clinical practice requires addressing challenges of generalization across devices and weak or ambiguous correlations between the measured signals and an individual's mental health. To address these challenges, we take a novel approach that leverages large language models (LLMs) to synthesize clinically useful insights from multi-sensor data. We develop chain of thought prompting methods that use LLMs to generate reasoning about how trends in data such as step count and sleep relate to conditions like depression and anxiety. We first demonstrate binary depression classification with LLMs achieving accuracies of 61.1% which exceed the state of the art. While it is not robust for clinical use, this leads us to our key finding: even more impactful and valued than classification is a new human-AI collaboration approach in which clinician experts interactively query these tools and combine their domain expertise and context about the patient with AI generated reasoning to support clinical decision-making. We find models like GPT-4 correctly reference numerical data 75% of the time, and clinician participants express strong interest in using this approach to interpret self-tracking data.

{{</citation>}}


### (2/135) Latent Lab: Large Language Models for Knowledge Exploration (Kevin Dunnell et al., 2023)

{{<citation>}}

Kevin Dunnell, Trudy Painter, Andrew Stoddard, Andy Lippman. (2023)  
**Latent Lab: Large Language Models for Knowledge Exploration**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13051v1)  

---


**ABSTRACT**  
This paper investigates the potential of AI models, particularly large language models (LLMs), to support knowledge exploration and augment human creativity during ideation. We present "Latent Lab" an interactive tool for discovering connections among MIT Media Lab research projects, emphasizing "exploration" over search. The work offers insights into collaborative AI systems by addressing the challenges of organizing, searching, and synthesizing content. In a user study, the tool's success was evaluated based on its ability to introduce users to an unfamiliar knowledge base, ultimately setting the groundwork for the ongoing advancement of human-AI knowledge exploration systems.

{{</citation>}}


### (3/135) RLIF: Interactive Imitation Learning as Reinforcement Learning (Jianlan Luo et al., 2023)

{{<citation>}}

Jianlan Luo, Perry Dong, Yuexiang Zhai, Yi Ma, Sergey Levine. (2023)  
**RLIF: Interactive Imitation Learning as Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.12996v1)  

---


**ABSTRACT**  
Although reinforcement learning methods offer a powerful framework for automatic skill acquisition, for practical learning-based control problems in domains such as robotics, imitation learning often provides a more convenient and accessible alternative. In particular, an interactive imitation learning method such as DAgger, which queries a near-optimal expert to intervene online to collect correction data for addressing the distributional shift challenges that afflict na\"ive behavioral cloning, can enjoy good performance both in theory and practice without requiring manually specified reward functions and other components of full reinforcement learning methods. In this paper, we explore how off-policy reinforcement learning can enable improved performance under assumptions that are similar but potentially even more practical than those of interactive imitation learning. Our proposed method uses reinforcement learning with user intervention signals themselves as rewards. This relaxes the assumption that intervening experts in interactive imitation learning should be near-optimal and enables the algorithm to learn behaviors that improve over the potential suboptimal human expert. We also provide a unified framework to analyze our RL method and DAgger; for which we present the asymptotic analysis of the suboptimal gap for both methods as well as the non-asymptotic sample complexity bound of our method. We then evaluate our method on challenging high-dimensional continuous control simulation benchmarks as well as real-world robotic vision-based manipulation tasks. The results show that it strongly outperforms DAgger-like approaches across the different tasks, especially when the intervening experts are suboptimal. Code and videos can be found on the project website: rlif-page.github.io

{{</citation>}}


### (4/135) NERIF: GPT-4V for Automatic Scoring of Drawn Models (Gyeong-Geon Lee et al., 2023)

{{<citation>}}

Gyeong-Geon Lee, Xiaoming Zhai. (2023)  
**NERIF: GPT-4V for Automatic Scoring of Drawn Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, GPT-4, NER  
[Paper Link](http://arxiv.org/abs/2311.12990v1)  

---


**ABSTRACT**  
Scoring student-drawn models is time-consuming. Recently released GPT-4V provides a unique opportunity to advance scientific modeling practices by leveraging the powerful image processing capability. To test this ability specifically for automatic scoring, we developed a method NERIF (Notation-Enhanced Rubric Instruction for Few-shot Learning) employing instructional note and rubrics to prompt GPT-4V to score students' drawn models for science phenomena. We randomly selected a set of balanced data (N = 900) that includes student-drawn models for six modeling assessment tasks. Each model received a score from GPT-4V ranging at three levels: 'Beginning,' 'Developing,' or 'Proficient' according to scoring rubrics. GPT-4V scores were compared with human experts' scores to calculate scoring accuracy. Results show that GPT-4V's average scoring accuracy was mean =.51, SD = .037. Specifically, average scoring accuracy was .64 for the 'Beginning' class, .62 for the 'Developing' class, and .26 for the 'Proficient' class, indicating that more proficient models are more challenging to score. Further qualitative study reveals how GPT-4V retrieves information from image input, including problem context, example evaluations provided by human coders, and students' drawing models. We also uncovered how GPT-4V catches the characteristics of student-drawn models and narrates them in natural language. At last, we demonstrated how GPT-4V assigns scores to student-drawn models according to the given scoring rubric and instructional notes. Our findings suggest that the NERIF is an effective approach for employing GPT-4V to score drawn models. Even though there is space for GPT-4V to improve scoring accuracy, some mis-assigned scores seemed interpretable to experts. The results of this study show that utilizing GPT-4V for automatic scoring of student-drawn models is promising.

{{</citation>}}


### (5/135) Digital Twin Framework for Optimal and Autonomous Decision-Making in Cyber-Physical Systems: Enhancing Reliability and Adaptability in the Oil and Gas Industry (Carine Menezes Rebello et al., 2023)

{{<citation>}}

Carine Menezes Rebello, Johannes Jäschkea, Idelfonso B. R. Nogueira. (2023)  
**Digital Twin Framework for Optimal and Autonomous Decision-Making in Cyber-Physical Systems: Enhancing Reliability and Adaptability in the Oil and Gas Industry**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12755v1)  

---


**ABSTRACT**  
The concept of creating a virtual copy of a complete Cyber-Physical System opens up numerous possibilities, including real-time assessments of the physical environment and continuous learning from the system to provide reliable and precise information. This process, known as the twinning process or the development of a digital twin (DT), has been widely adopted across various industries. However, challenges arise when considering the computational demands of implementing AI models, such as those employed in digital twins, in real-time information exchange scenarios. This work proposes a digital twin framework for optimal and autonomous decision-making applied to a gas-lift process in the oil and gas industry, focusing on enhancing the robustness and adaptability of the DT. The framework combines Bayesian inference, Monte Carlo simulations, transfer learning, online learning, and novel strategies to confer cognition to the DT, including model hyperdimensional reduction and cognitive tack. Consequently, creating a framework for efficient, reliable, and trustworthy DT identification was possible. The proposed approach addresses the current gap in the literature regarding integrating various learning techniques and uncertainty management in digital twin strategies. This digital twin framework aims to provide a reliable and efficient system capable of adapting to changing environments and incorporating prediction uncertainty, thus enhancing the overall decision-making process in complex, real-world scenarios. Additionally, this work lays the foundation for further developments in digital twins for process systems engineering, potentially fostering new advancements and applications across various industrial sectors.

{{</citation>}}


### (6/135) Development of a Legal Document AI-Chatbot (Pranav Nataraj Devaraj et al., 2023)

{{<citation>}}

Pranav Nataraj Devaraj, Rakesh Teja P V, Aaryav Gangrade, Manoj Kumar R. (2023)  
**Development of a Legal Document AI-Chatbot**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Legal  
[Paper Link](http://arxiv.org/abs/2311.12719v1)  

---


**ABSTRACT**  
With the exponential growth of digital data and the increasing complexity of legal documentation, there is a pressing need for efficient and intelligent tools to streamline the handling of legal documents.With the recent developments in the AI field, especially in chatbots, it cannot be ignored as a very compelling solution to this problem.An insight into the process of creating a Legal Documentation AI Chatbot with as many relevant features as possible within the given time frame is presented.The development of each component of the chatbot is presented in detail.Each component's workings and functionality has been discussed.Starting from the build of the Android app and the Langchain query processing code till the integration of both through a Flask backend and REST API methods.

{{</citation>}}


### (7/135) From Concept to Manufacturing: Evaluating Vision-Language Models for Engineering Design (Cyril Picard et al., 2023)

{{<citation>}}

Cyril Picard, Kristen M. Edwards, Anna C. Doris, Brandon Man, Giorgio Giannone, Md Ferdous Alam, Faez Ahmed. (2023)  
**From Concept to Manufacturing: Evaluating Vision-Language Models for Engineering Design**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs.AI  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12668v1)  

---


**ABSTRACT**  
Engineering Design is undergoing a transformative shift with the advent of AI, marking a new era in how we approach product, system, and service planning. Large language models have demonstrated impressive capabilities in enabling this shift. Yet, with text as their only input modality, they cannot leverage the large body of visual artifacts that engineers have used for centuries and are accustomed to. This gap is addressed with the release of multimodal vision language models, such as GPT-4V, enabling AI to impact many more types of tasks. In light of these advancements, this paper presents a comprehensive evaluation of GPT-4V, a vision language model, across a wide spectrum of engineering design tasks, categorized into four main areas: Conceptual Design, System-Level and Detailed Design, Manufacturing and Inspection, and Engineering Education Tasks. Our study assesses GPT-4V's capabilities in design tasks such as sketch similarity analysis, concept selection using Pugh Charts, material selection, engineering drawing analysis, CAD generation, topology optimization, design for additive and subtractive manufacturing, spatial reasoning challenges, and textbook problems. Through this structured evaluation, we not only explore GPT-4V's proficiency in handling complex design and manufacturing challenges but also identify its limitations in complex engineering design applications. Our research establishes a foundation for future assessments of vision language models, emphasizing their immense potential for innovating and enhancing the engineering design and manufacturing landscape. It also contributes a set of benchmark testing datasets, with more than 1000 queries, for ongoing advancements and applications in this field.

{{</citation>}}


### (8/135) Trustworthy AI: Deciding What to Decide (Caesar Wu et al., 2023)

{{<citation>}}

Caesar Wu, Yuan-Fang Li, Jian Li, Jingjing Xu, Bouvry Pascal. (2023)  
**Trustworthy AI: Deciding What to Decide**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12604v1)  

---


**ABSTRACT**  
When engaging in strategic decision-making, we are frequently confronted with overwhelming information and data. The situation can be further complicated when certain pieces of evidence contradict each other or become paradoxical. The primary challenge is how to determine which information can be trusted when we adopt Artificial Intelligence (AI) systems for decision-making. This issue is known as deciding what to decide or Trustworthy AI. However, the AI system itself is often considered an opaque black box. We propose a new approach to address this issue by introducing a novel framework of Trustworthy AI (TAI) encompassing three crucial components of AI: representation space, loss function, and optimizer. Each component is loosely coupled with four TAI properties. Altogether, the framework consists of twelve TAI properties. We aim to use this framework to conduct the TAI experiments by quantitive and qualitative research methods to satisfy TAI properties for the decision-making context. The framework allows us to formulate an optimal prediction model trained by the given dataset for applying the strategic investment decision of credit default swaps (CDS) in the technology sector. Finally, we provide our view of the future direction of TAI research

{{</citation>}}


### (9/135) Classification of Tabular Data by Text Processing (Keshav Ramani et al., 2023)

{{<citation>}}

Keshav Ramani, Daniel Borrajo. (2023)  
**Classification of Tabular Data by Text Processing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.12521v1)  

---


**ABSTRACT**  
Natural Language Processing technology has advanced vastly in the past decade. Text processing has been successfully applied to a wide variety of domains. In this paper, we propose a novel framework, Text Based Classification(TBC), that uses state of the art text processing techniques to solve classification tasks on tabular data. We provide a set of controlled experiments where we present the benefits of using this approach against other classification methods. Experimental results on several data sets also show that this framework achieves comparable performance to that of several state of the art models in accuracy, precision and recall of predicted classes.

{{</citation>}}


### (10/135) Self-Supervised Deconfounding Against Spatio-Temporal Shifts: Theory and Modeling (Jiahao Ji et al., 2023)

{{<citation>}}

Jiahao Ji, Wentao Zhang, Jingyuan Wang, Yue He, Chao Huang. (2023)  
**Self-Supervised Deconfounding Against Spatio-Temporal Shifts: Theory and Modeling**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.12472v1)  

---


**ABSTRACT**  
As an important application of spatio-temporal (ST) data, ST traffic forecasting plays a crucial role in improving urban travel efficiency and promoting sustainable development. In practice, the dynamics of traffic data frequently undergo distributional shifts attributed to external factors such as time evolution and spatial differences. This entails forecasting models to handle the out-of-distribution (OOD) issue where test data is distributed differently from training data. In this work, we first formalize the problem by constructing a causal graph of past traffic data, future traffic data, and external ST contexts. We reveal that the failure of prior arts in OOD traffic data is due to ST contexts acting as a confounder, i.e., the common cause for past data and future ones. Then, we propose a theoretical solution named Disentangled Contextual Adjustment (DCA) from a causal lens. It differentiates invariant causal correlations against variant spurious ones and deconfounds the effect of ST contexts. On top of that, we devise a Spatio-Temporal sElf-superVised dEconfounding (STEVE) framework. It first encodes traffic data into two disentangled representations for associating invariant and variant ST contexts. Then, we use representative ST contexts from three conceptually different perspectives (i.e., temporal, spatial, and semantic) as self-supervised signals to inject context information into both representations. In this way, we improve the generalization ability of the learned context-oriented representations to OOD ST traffic forecasting. Comprehensive experiments on four large-scale benchmark datasets demonstrate that our STEVE consistently outperforms the state-of-the-art baselines across various ST OOD scenarios.

{{</citation>}}


### (11/135) Towards a Gateway for Knowledge Graph Schemas Collection, Analysis, and Embedding (Mattia Fumagalli et al., 2023)

{{<citation>}}

Mattia Fumagalli, Marco Boffo, Daqian Shi, Mayukh Bagchi, Fausto Giunchiglia. (2023)  
**Towards a Gateway for Knowledge Graph Schemas Collection, Analysis, and Embedding**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.12465v1)  

---


**ABSTRACT**  
One of the significant barriers to the training of statistical models on knowledge graphs is the difficulty that scientists have in finding the best input data to address their prediction goal. In addition to this, a key challenge is to determine how to manipulate these relational data, which are often in the form of particular triples (i.e., subject, predicate, object), to enable the learning process. Currently, many high-quality catalogs of knowledge graphs, are available. However, their primary goal is the re-usability of these resources, and their interconnection, in the context of the Semantic Web. This paper describes the LiveSchema initiative, namely, a first version of a gateway that has the main scope of leveraging the gold mine of data collected by many existing catalogs collecting relational data like ontologies and knowledge graphs. At the current state, LiveSchema contains - 1000 datasets from 4 main sources and offers some key facilities, which allow to: i) evolving LiveSchema, by aggregating other source catalogs and repositories as input sources; ii) querying all the collected resources; iii) transforming each given dataset into formal concept analysis matrices that enable analysis and visualization services; iv) generating models and tensors from each given dataset.

{{</citation>}}


### (12/135) Extracting Definienda in Mathematical Scholarly Articles with Transformers (Shufan Jiang et al., 2023)

{{<citation>}}

Shufan Jiang, Pierre Senellart. (2023)  
**Extracting Definienda in Mathematical Scholarly Articles with Transformers**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.12448v1)  

---


**ABSTRACT**  
We consider automatically identifying the defined term within a mathematical definition from the text of an academic article. Inspired by the development of transformer-based natural language processing applications, we pose the problem as (a) a token-level classification task using fine-tuned pre-trained transformers; and (b) a question-answering task using a generalist large language model (GPT). We also propose a rule-based approach to build a labeled dataset from the LATEX source of papers. Experimental results show that it is possible to reach high levels of precision and recall using either recent (and expensive) GPT 4 or simpler pre-trained models fine-tuned on our task.

{{</citation>}}


### (13/135) How Far Have We Gone in Vulnerability Detection Using Large Language Models (Zeyu Gao et al., 2023)

{{<citation>}}

Zeyu Gao, Hao Wang, Yuchen Zhou, Wenyu Zhu, Chao Zhang. (2023)  
**How Far Have We Gone in Vulnerability Detection Using Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CR, cs.AI  
Keywords: Language Model, Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2311.12420v1)  

---


**ABSTRACT**  
As software becomes increasingly complex and prone to vulnerabilities, automated vulnerability detection is critically important, yet challenging. Given the significant successes of Large Language Models (LLMs) in various tasks, there is growing anticipation of their efficacy in vulnerability detection. However, a quantitative understanding of their potential in vulnerability detection is still missing. To bridge this gap, we introduce a comprehensive vulnerability benchmark VulBench. This benchmark aggregates high-quality data from a wide range of CTF (Capture-the-Flag) challenges and real-world applications, with annotations for each vulnerable function detailing the vulnerability type and its root cause. Through our experiments encompassing 16 LLMs and 6 state-of-the-art (SOTA) deep learning-based models and static analyzers, we find that several LLMs outperform traditional deep learning approaches in vulnerability detection, revealing an untapped potential in LLMs. This work contributes to the understanding and utilization of LLMs for enhanced software security.

{{</citation>}}


### (14/135) A Survey on Multimodal Large Language Models for Autonomous Driving (Can Cui et al., 2023)

{{<citation>}}

Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, Tianren Gao, Erlong Li, Kun Tang, Zhipeng Cao, Tong Zhou, Ao Liu, Xinrui Yan, Shuqi Mei, Jianguo Cao, Ziran Wang, Chao Zheng. (2023)  
**A Survey on Multimodal Large Language Models for Autonomous Driving**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12320v1)  

---


**ABSTRACT**  
With the emergence of Large Language Models (LLMs) and Vision Foundation Models (VFMs), multimodal AI systems benefiting from large models have the potential to equally perceive the real world, make decisions, and control tools as humans. In recent months, LLMs have shown widespread attention in autonomous driving and map systems. Despite its immense potential, there is still a lack of a comprehensive understanding of key challenges, opportunities, and future endeavors to apply in LLM driving systems. In this paper, we present a systematic investigation in this field. We first introduce the background of Multimodal Large Language Models (MLLMs), the multimodal models development using LLMs, and the history of autonomous driving. Then, we overview existing MLLM tools for driving, transportation, and map systems together with existing datasets and benchmarks. Moreover, we summarized the works in The 1st WACV Workshop on Large Language and Vision Models for Autonomous Driving (LLVM-AD), which is the first workshop of its kind regarding LLMs in autonomous driving. To further promote the development of this field, we also discuss several important problems regarding using MLLMs in autonomous driving systems that need to be solved by both academia and industry.

{{</citation>}}


### (15/135) IEKM: A Model Incorporating External Keyword Matrices (Cheng Luo et al., 2023)

{{<citation>}}

Cheng Luo, Qin Li, Zhao Yan, Mengliang Rao, Yunbo Cao. (2023)  
**IEKM: A Model Incorporating External Keyword Matrices**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.12310v1)  

---


**ABSTRACT**  
A customer service platform system with a core text semantic similarity (STS) task faces two urgent challenges: Firstly, one platform system needs to adapt to different domains of customers, i.e., different domains adaptation (DDA). Secondly, it is difficult for the model of the platform system to distinguish sentence pairs that are literally close but semantically different, i.e., hard negative samples. In this paper, we propose an incorporation external keywords matrices model (IEKM) to address these challenges. The model uses external tools or dictionaries to construct external matrices and fuses them to the self-attention layers of the Transformer structure through gating units, thus enabling flexible corrections to the model results. We evaluate the method on multiple datasets and the results show that our method has improved performance on all datasets. To demonstrate that our method can effectively solve all the above challenges, we conduct a flexible correction experiment, which results in an increase in the F1 value from 56.61 to 73.53. Our code will be publicly available.

{{</citation>}}


### (16/135) Causality is all you need (Ning Xu et al., 2023)

{{<citation>}}

Ning Xu, Yifei Gao, Hongshuo Tian, Yongdong Zhang, An-An Liu. (2023)  
**Causality is all you need**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2311.12307v1)  

---


**ABSTRACT**  
In the fundamental statistics course, students are taught to remember the well-known saying: "Correlation is not Causation". Till now, statistics (i.e., correlation) have developed various successful frameworks, such as Transformer and Pre-training large-scale models, which have stacked multiple parallel self-attention blocks to imitate a wide range of tasks. However, in the causation community, how to build an integrated causal framework still remains an untouched domain despite its excellent intervention capabilities. In this paper, we propose the Causal Graph Routing (CGR) framework, an integrated causal scheme relying entirely on the intervention mechanisms to reveal the cause-effect forces hidden in data. Specifically, CGR is composed of a stack of causal layers. Each layer includes a set of parallel deconfounding blocks from different causal graphs. We combine these blocks via the concept of the proposed sufficient cause, which allows the model to dynamically select the suitable deconfounding methods in each layer. CGR is implemented as the stacked networks, integrating no confounder, back-door adjustment, front-door adjustment, and probability of sufficient cause. We evaluate this framework on two classical tasks of CV and NLP. Experiments show CGR can surpass the current state-of-the-art methods on both Visual Question Answer and Long Document Classification tasks. In particular, CGR has great potential in building the "causal" pre-training large-scale model that effectively generalizes to diverse tasks. It will improve the machines' comprehension of causal relationships within a broader semantic space.

{{</citation>}}


## cs.CL (23)



### (17/135) Attribution and Alignment: Effects of Local Context Repetition on Utterance Production and Comprehension in Dialogue (Aron Molnar et al., 2023)

{{<citation>}}

Aron Molnar, Jaap Jumelet, Mario Giulianelli, Arabella Sinclair. (2023)  
**Attribution and Alignment: Effects of Local Context Repetition on Utterance Production and Comprehension in Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2311.13061v1)  

---


**ABSTRACT**  
Language models are often used as the backbone of modern dialogue systems. These models are pre-trained on large amounts of written fluent language. Repetition is typically penalised when evaluating language model generations. However, it is a key component of dialogue. Humans use local and partner specific repetitions; these are preferred by human users and lead to more successful communication in dialogue. In this study, we evaluate (a) whether language models produce human-like levels of repetition in dialogue, and (b) what are the processing mechanisms related to lexical re-use they use during comprehension. We believe that such joint analysis of model production and comprehension behaviour can inform the development of cognitively inspired dialogue generation systems.

{{</citation>}}


### (18/135) Beyond Text: Unveiling Multimodal Proficiency of Large Language Models with MultiAPI Benchmark (Xiao Liu et al., 2023)

{{<citation>}}

Xiao Liu, Jianfeng Lin, Jiawei Zhang. (2023)  
**Beyond Text: Unveiling Multimodal Proficiency of Large Language Models with MultiAPI Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13053v1)  

---


**ABSTRACT**  
The proliferation of Large Language Models like ChatGPT has significantly advanced language understanding and generation, impacting a broad spectrum of applications. However, these models predominantly excel in text-based tasks, overlooking the complexity of real-world multimodal information. This study introduces MultiAPI, a pioneering comprehensive large-scale API benchmark dataset aimed at expanding LLMs' proficiency in multimodal contexts. Developed collaboratively through ChatGPT, MultiAPI consists of 235 diverse API calls and 2,038 contextual prompts, offering a unique platform evaluation of tool-augmented LLMs handling multimodal tasks. Through comprehensive experiments, our findings reveal that while LLMs demonstrate proficiency in API call decision-making, they face challenges in domain identification, function selection, and argument generation. What's more, we surprisingly notice that auxiliary context can actually impair the performance. An in-depth error analysis paves the way for a new paradigm to address these challenges, suggesting a potential direction for future LLM research.

{{</citation>}}


### (19/135) Unsupervised Graph Attention Autoencoder for Attributed Networks using K-means Loss (Abdelfateh Bekkaira et al., 2023)

{{<citation>}}

Abdelfateh Bekkaira, Slimane Bellaouar, Slimane Oulad-Naoui. (2023)  
**Unsupervised Graph Attention Autoencoder for Attributed Networks using K-means Loss**  

---
Primary Category: cs.CL  
Categories: 68T07, I-2-4, cs-AI, cs-CL, cs.CL  
Keywords: Attention, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.12986v1)  

---


**ABSTRACT**  
Multimodal Sentiment Analysis (MSA) has recently become a centric research direction for many real-world applications. This proliferation is due to the fact that opinions are central to almost all human activities and are key influencers of our behaviors. In addition, the recent deployment of Deep Learning-based (DL) models has proven their high efficiency for a wide range of Western languages. In contrast, Arabic DL-based multimodal sentiment analysis (MSA) is still in its infantile stage due, mainly, to the lack of standard datasets. % The contribution In this paper, our investigation is twofold. First, we design a pipeline that helps building our Arabic Multimodal dataset leveraging both state-of-the-art transformers and feature extraction tools within word alignment techniques. Thereafter, we validate our dataset using state-of-the-art transformer-based model dealing with multimodality. Despite the small size of the outcome dataset, experiments show that Arabic multimodality is very promising.

{{</citation>}}


### (20/135) GAIA: a benchmark for General AI Assistants (Grégoire Mialon et al., 2023)

{{<citation>}}

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, Thomas Scialom. (2023)  
**GAIA: a benchmark for General AI Assistants**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.12983v1)  

---


**ABSTRACT**  
We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92\% vs. 15\% for GPT-4 equipped with plugins. This notable performance disparity contrasts with the recent trend of LLMs outperforming humans on tasks requiring professional skills in e.g. law or chemistry. GAIA's philosophy departs from the current trend in AI benchmarks suggesting to target tasks that are ever more difficult for humans. We posit that the advent of Artificial General Intelligence (AGI) hinges on a system's capability to exhibit similar robustness as the average human does on such questions. Using GAIA's methodology, we devise 466 questions and their answer. We release our questions while retaining answers to 300 of them to power a leader-board available at https://huggingface.co/gaia-benchmark.

{{</citation>}}


### (21/135) LowResource at BLP-2023 Task 2: Leveraging BanglaBert for Low Resource Sentiment Analysis of Bangla Language (Aunabil Chakma et al., 2023)

{{<citation>}}

Aunabil Chakma, Masum Hasan. (2023)  
**LowResource at BLP-2023 Task 2: Leveraging BanglaBert for Low Resource Sentiment Analysis of Bangla Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Sentiment Analysis, T5  
[Paper Link](http://arxiv.org/abs/2311.12735v1)  

---


**ABSTRACT**  
This paper describes the system of the LowResource Team for Task 2 of BLP-2023, which involves conducting sentiment analysis on a dataset composed of public posts and comments from diverse social media platforms. Our primary aim is to utilize BanglaBert, a BERT model pre-trained on a large Bangla corpus, using various strategies including fine-tuning, dropping random tokens, and using several external datasets. Our final model is an ensemble of the three best BanglaBert variations. Our system has achieved overall 3rd in the Test Set among 30 participating teams with a score of 0.718. Additionally, we discuss the promising systems that didn't perform well namely task-adaptive pertaining and paraphrasing using BanglaT5. Training codes and external datasets which are used for our system are publicly available at https://github.com/Aunabil4602/bnlp-workshop-task2-2023

{{</citation>}}


### (22/135) Can Large Language Models Understand Content and Propagation for Misinformation Detection: An Empirical Study (Mengyang Chen et al., 2023)

{{<citation>}}

Mengyang Chen, Lingwei Wei, Han Cao, Wei Zhou, Songlin Hu. (2023)  
**Can Large Language Models Understand Content and Propagation for Misinformation Detection: An Empirical Study**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12699v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have garnered significant attention for their powerful ability in natural language understanding and reasoning. In this paper, we present a comprehensive empirical study to explore the performance of LLMs on misinformation detection tasks. This study stands as the pioneering investigation into the understanding capabilities of multiple LLMs regarding both content and propagation across social media platforms. Our empirical studies on five misinformation detection datasets show that LLMs with diverse prompts achieve comparable performance in text-based misinformation detection but exhibit notably constrained capabilities in comprehending propagation structure compared to existing models in propagation-based misinformation detection. Besides, we further design four instruction-tuned strategies to enhance LLMs for both content and propagation-based misinformation detection. These strategies boost LLMs to actively learn effective features from multiple instances or hard instances, and eliminate irrelevant propagation structures, thereby achieving better detection performance. Extensive experiments further demonstrate LLMs would play a better capacity in content and propagation structure under these proposed strategies and achieve promising detection performance. These findings highlight the potential ability of LLMs to detect misinformation.

{{</citation>}}


### (23/135) Fair Text Classification with Wasserstein Independence (Thibaud Leteno et al., 2023)

{{<citation>}}

Thibaud Leteno, Antoine Gourru, Charlotte Laclau, Rémi Emonet, Christophe Gravier. (2023)  
**Fair Text Classification with Wasserstein Independence**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Text Classification  
[Paper Link](http://arxiv.org/abs/2311.12689v1)  

---


**ABSTRACT**  
Group fairness is a central research topic in text classification, where reaching fair treatment between sensitive groups (e.g. women vs. men) remains an open challenge. This paper presents a novel method for mitigating biases in neural text classification, agnostic to the model architecture. Considering the difficulty to distinguish fair from unfair information in a text encoder, we take inspiration from adversarial training to induce Wasserstein independence between representations learned to predict our target label and the ones learned to predict some sensitive attribute. Our approach provides two significant advantages. Firstly, it does not require annotations of sensitive attributes in both testing and training data. This is more suitable for real-life scenarios compared to existing methods that require annotations of sensitive attributes at train time. Second, our approach exhibits a comparable or better fairness-accuracy trade-off compared to existing methods.

{{</citation>}}


### (24/135) MathGloss: Building mathematical glossaries from text (Lucy Horowitz et al., 2023)

{{<citation>}}

Lucy Horowitz, Valeria de Paiva. (2023)  
**MathGloss: Building mathematical glossaries from text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.12649v1)  

---


**ABSTRACT**  
MathGloss is a project to create a knowledge graph (KG) for undergraduate mathematics from text, automatically, using modern natural language processing (NLP) tools and resources already available on the web. MathGloss is a linked database of undergraduate concepts in mathematics. So far, it combines five resources: (i) Wikidata, a collaboratively edited, multilingual knowledge graph hosted by the Wikimedia Foundation, (ii) terms covered in mathematics courses at the University of Chicago, (iii) the syllabus of the French undergraduate mathematics curriculum which includes hyperlinks to the automated theorem prover Lean 4, (iv) MuLiMa, a multilingual dictionary of mathematics curated by mathematicians, and (v) the nLab, a wiki for category theory also curated by mathematicians. MathGloss's goal is to bring together resources for learning mathematics and to allow every mathematician to tailor their learning to their own preferences. Moreover, by organizing different resources for learning undergraduate mathematics alongside those for learning formal mathematics, we hope to make it easier for mathematicians and formal tools (theorem provers, computer algebra systems, etc) experts to "understand" each other and break down some of the barriers to formal math.

{{</citation>}}


### (25/135) Oasis: Data Curation and Assessment System for Pretraining of Large Language Models (Tong Zhou et al., 2023)

{{<citation>}}

Tong Zhou, Yubo Chen, Pengfei Cao, Kang Liu, Jun Zhao, Shengping Liu. (2023)  
**Oasis: Data Curation and Assessment System for Pretraining of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12537v1)  

---


**ABSTRACT**  
Data is one of the most critical elements in building a large language model. However, existing systems either fail to customize a corpus curation pipeline or neglect to leverage comprehensive corpus assessment for iterative optimization of the curation. To this end, we present a pretraining corpus curation and assessment platform called Oasis -- a one-stop system for data quality improvement and quantification with user-friendly interactive interfaces. Specifically, the interactive modular rule filter module can devise customized rules according to explicit feedback. The debiased neural filter module builds the quality classification dataset in a negative-centric manner to remove the undesired bias. The adaptive document deduplication module could execute large-scale deduplication with limited memory resources. These three parts constitute the customized data curation module. And in the holistic data assessment module, a corpus can be assessed in local and global views, with three evaluation means including human, GPT-4, and heuristic metrics. We exhibit a complete process to use Oasis for the curation and assessment of pretraining data. In addition, an 800GB bilingual corpus curated by Oasis is publicly released.

{{</citation>}}


### (26/135) Evaluation Metrics of Language Generation Models for Synthetic Traffic Generation Tasks (Simone Filice et al., 2023)

{{<citation>}}

Simone Filice, Jason Ingyu Choi, Giuseppe Castellucci, Eugene Agichtein, Oleg Rokhlenko. (2023)  
**Evaluation Metrics of Language Generation Models for Synthetic Traffic Generation Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Natural Language Generation, QA, Question Generation  
[Paper Link](http://arxiv.org/abs/2311.12534v1)  

---


**ABSTRACT**  
Many Natural Language Generation (NLG) tasks aim to generate a single output text given an input prompt. Other settings require the generation of multiple texts, e.g., for Synthetic Traffic Generation (STG). This generation task is crucial for training and evaluating QA systems as well as conversational agents, where the goal is to generate multiple questions or utterances resembling the linguistic variability of real users. In this paper, we show that common NLG metrics, like BLEU, are not suitable for evaluating STG. We propose and evaluate several metrics designed to compare the generated traffic to the distribution of real user texts. We validate our metrics with an automatic procedure to verify whether they capture different types of quality issues of generated data; we also run human annotations to verify the correlation with human judgements. Experiments on three tasks, i.e., Shopping Utterance Generation, Product Question Generation and Query Auto Completion, demonstrate that our metrics are effective for evaluating STG tasks, and improve the agreement with human judgement up to 20% with respect to common NLG metrics. We believe these findings can pave the way towards better solutions for estimating the representativeness of synthetic text data.

{{</citation>}}


### (27/135) Multilingual Word Embeddings for Low-Resource Languages using Anchors and a Chain of Related Languages (Viktor Hangya et al., 2023)

{{<citation>}}

Viktor Hangya, Silvia Severini, Radoslav Ralev, Alexander Fraser, Hinrich Schütze. (2023)  
**Multilingual Word Embeddings for Low-Resource Languages using Anchors and a Chain of Related Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Low-Resource, Multilingual, NLP, Word Embedding  
[Paper Link](http://arxiv.org/abs/2311.12489v1)  

---


**ABSTRACT**  
Very low-resource languages, having only a few million tokens worth of data, are not well-supported by multilingual NLP approaches due to poor quality cross-lingual word representations. Recent work showed that good cross-lingual performance can be achieved if a source language is related to the low-resource target language. However, not all language pairs are related. In this paper, we propose to build multilingual word embeddings (MWEs) via a novel language chain-based approach, that incorporates intermediate related languages to bridge the gap between the distant source and target. We build MWEs one language at a time by starting from the resource rich source and sequentially adding each language in the chain till we reach the target. We extend a semi-joint bilingual approach to multiple languages in order to eliminate the main weakness of previous works, i.e., independently trained monolingual embeddings, by anchoring the target language around the multilingual space. We evaluate our method on bilingual lexicon induction for 4 language families, involving 4 very low-resource (<5M tokens) and 4 moderately low-resource (<50M) target languages, showing improved performance in both categories. Additionally, our analysis reveals the importance of good quality embeddings for intermediate languages as well as the importance of leveraging anchor points from all languages in the multilingual space.

{{</citation>}}


### (28/135) PhayaThaiBERT: Enhancing a Pretrained Thai Language Model with Unassimilated Loanwords (Panyut Sriwirote et al., 2023)

{{<citation>}}

Panyut Sriwirote, Jalinee Thapiang, Vasan Timtong, Attapol T. Rutherford. (2023)  
**PhayaThaiBERT: Enhancing a Pretrained Thai Language Model with Unassimilated Loanwords**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12475v1)  

---


**ABSTRACT**  
While WangchanBERTa has become the de facto standard in transformer-based Thai language modeling, it still has shortcomings in regard to the understanding of foreign words, most notably English words, which are often borrowed without orthographic assimilation into Thai in many contexts. We identify the lack of foreign vocabulary in WangchanBERTa's tokenizer as the main source of these shortcomings. We then expand WangchanBERTa's vocabulary via vocabulary transfer from XLM-R's pretrained tokenizer and pretrain a new model using the expanded tokenizer, starting from WangchanBERTa's checkpoint, on a new dataset that is larger than the one used to train WangchanBERTa. Our results show that our new pretrained model, PhayaThaiBERT, outperforms WangchanBERTa in many downstream tasks and datasets.

{{</citation>}}


### (29/135) Visual Analytics for Generative Transformer Models (Raymond Li et al., 2023)

{{<citation>}}

Raymond Li, Ruixin Yang, Wen Xiao, Ahmed AbuRaed, Gabriel Murray, Giuseppe Carenini. (2023)  
**Visual Analytics for Generative Transformer Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2311.12418v1)  

---


**ABSTRACT**  
While transformer-based models have achieved state-of-the-art results in a variety of classification and generation tasks, their black-box nature makes them challenging for interpretability. In this work, we present a novel visual analytical framework to support the analysis of transformer-based generative networks. In contrast to previous work, which has mainly focused on encoder-based models, our framework is one of the first dedicated to supporting the analysis of transformer-based encoder-decoder models and decoder-only models for generative and classification tasks. Hence, we offer an intuitive overview that allows the user to explore different facets of the model through interactive visualization. To demonstrate the feasibility and usefulness of our framework, we present three detailed case studies based on real-world NLP research problems.

{{</citation>}}


### (30/135) nach0: Multimodal Natural and Chemical Languages Foundation Model (Micha Livne et al., 2023)

{{<citation>}}

Micha Livne, Zulfat Miftahutdinov, Elena Tutubalina, Maksim Kuznetsov, Daniil Polykovskiy, Annika Brundyn, Aastha Jhunjhunwala, Anthony Costa, Alex Aliper, Alex Zhavoronkov. (2023)  
**nach0: Multimodal Natural and Chemical Languages Foundation Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, q-bio-QM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12410v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have substantially driven scientific progress in various domains, and many papers have demonstrated their ability to tackle complex problems with creative solutions. Our paper introduces a new foundation model, nach0, capable of solving various chemical and biological tasks: biomedical question answering, named entity recognition, molecular generation, molecular synthesis, attributes prediction, and others. nach0 is a multi-domain and multi-task encoder-decoder LLM pre-trained on unlabeled text from scientific literature, patents, and molecule strings to incorporate a range of chemical and linguistic knowledge. We employed instruction tuning, where specific task-related instructions are utilized to fine-tune nach0 for the final set of tasks. To train nach0 effectively, we leverage the NeMo framework, enabling efficient parallel optimization of both base and large model versions. Extensive experiments demonstrate that our model outperforms state-of-the-art baselines on single-domain and cross-domain tasks. Furthermore, it can generate high-quality outputs in molecular and textual formats, showcasing its effectiveness in multi-domain setups.

{{</citation>}}


### (31/135) IndoRobusta: Towards Robustness Against Diverse Code-Mixed Indonesian Local Languages (Muhammad Farid Adilazuarda et al., 2023)

{{<citation>}}

Muhammad Farid Adilazuarda, Samuel Cahyawijaya, Genta Indra Winata, Pascale Fung, Ayu Purwarianti. (2023)  
**IndoRobusta: Towards Robustness Against Diverse Code-Mixed Indonesian Local Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.12405v1)  

---


**ABSTRACT**  
Significant progress has been made on Indonesian NLP. Nevertheless, exploration of the code-mixing phenomenon in Indonesian is limited, despite many languages being frequently mixed with Indonesian in daily conversation. In this work, we explore code-mixing in Indonesian with four embedded languages, i.e., English, Sundanese, Javanese, and Malay; and introduce IndoRobusta, a framework to evaluate and improve the code-mixing robustness. Our analysis shows that the pre-training corpus bias affects the model's ability to better handle Indonesian-English code-mixing when compared to other local languages, despite having higher language diversity.

{{</citation>}}


### (32/135) InterPrompt: Interpretable Prompting for Interrelated Interpersonal Risk Factors in Reddit Posts (MSVPJ Sathvik et al., 2023)

{{<citation>}}

MSVPJ Sathvik, Surjodeep Sarkar, Chandni Saxena, Sunghwan Sohn, Muskan Garg. (2023)  
**InterPrompt: Interpretable Prompting for Interrelated Interpersonal Risk Factors in Reddit Posts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.12404v1)  

---


**ABSTRACT**  
Mental health professionals and clinicians have observed the upsurge of mental disorders due to Interpersonal Risk Factors (IRFs). To simulate the human-in-the-loop triaging scenario for early detection of mental health disorders, we recognized textual indications to ascertain these IRFs : Thwarted Belongingness (TBe) and Perceived Burdensomeness (PBu) within personal narratives. In light of this, we use N-shot learning with GPT-3 model on the IRF dataset, and underscored the importance of fine-tuning GPT-3 model to incorporate the context-specific sensitivity and the interconnectedness of textual cues that represent both IRFs.   In this paper, we introduce an Interpretable Prompting (InterPrompt)} method to boost the attention mechanism by fine-tuning the GPT-3 model. This allows a more sophisticated level of language modification by adjusting the pre-trained weights. Our model learns to detect usual patterns and underlying connections across both the IRFs, which leads to better system-level explainability and trustworthiness. The results of our research demonstrate that all four variants of GPT-3 model, when fine-tuned with InterPrompt, perform considerably better as compared to the baseline methods, both in terms of classification and explanation generation.

{{</citation>}}


### (33/135) The Obscure Limitation of Modular Multilingual Language Models (Muhammad Farid Adilazuarda et al., 2023)

{{<citation>}}

Muhammad Farid Adilazuarda, Samuel Cahyawijaya, Ayu Purwarianti. (2023)  
**The Obscure Limitation of Modular Multilingual Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2311.12375v1)  

---


**ABSTRACT**  
We expose the limitation of modular multilingual language models (MLMs) in multilingual inference scenarios with unknown languages. Existing evaluations of modular MLMs exclude the involvement of language identification (LID) modules, which obscures the performance of real-case multilingual scenarios of modular MLMs. In this work, we showcase the effect of adding LID on the multilingual evaluation of modular MLMs and provide discussions for closing the performance gap of caused by the pipelined approach of LID and modular MLMs.

{{</citation>}}


### (34/135) Beyond Turing: A Comparative Analysis of Approaches for Detecting Machine-Generated Text (Muhammad Farid Adilazuarda et al., 2023)

{{<citation>}}

Muhammad Farid Adilazuarda, Nikolaos Nektarios Arkoulis, Oleksii Chumakov. (2023)  
**Beyond Turing: A Comparative Analysis of Approaches for Detecting Machine-Generated Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2311.12373v1)  

---


**ABSTRACT**  
Significant progress has been made on text generation by pre-trained language models (PLMs), yet distinguishing between human and machine-generated text poses an escalating challenge. This paper offers an in-depth evaluation of three distinct methods used to address this task: traditional shallow learning, Language Model (LM) fine-tuning, and Multilingual Model fine-tuning. These approaches are rigorously tested on a wide range of machine-generated texts, providing a benchmark of their competence in distinguishing between human-authored and machine-authored linguistic constructs. The results reveal considerable differences in performance across methods, thus emphasizing the continued need for advancement in this crucial area of NLP. This study offers valuable insights and paves the way for future research aimed at creating robust and highly discriminative models.

{{</citation>}}


### (35/135) Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey (Yunpeng Huang et al., 2023)

{{<citation>}}

Yunpeng Huang, Jingwei Xu, Zixu Jiang, Junyu Lai, Zenan Li, Yuan Yao, Taolue Chen, Lijuan Yang, Zhou Xin, Xiaoxing Ma. (2023)  
**Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2311.12351v1)  

---


**ABSTRACT**  
With the bomb ignited by ChatGPT, Transformer-based Large Language Models (LLMs) have paved a revolutionary path toward Artificial General Intelligence (AGI) and have been applied in diverse areas as knowledge bases, human interfaces, and dynamic agents. However, a prevailing limitation exists: many current LLMs, constrained by resources, are primarily pre-trained on shorter texts, rendering them less effective for longer-context prompts, commonly encountered in real-world settings. In this paper, we present a comprehensive survey focusing on the advancement of model architecture in Transformer-based LLMs to optimize long-context capabilities across all stages from pre-training to inference. We firstly delineate and analyze the problems of handling long-context input and output with the current Transformer-based models. Then, we mainly offer a holistic taxonomy to navigate the landscape of Transformer upgrades on architecture to solve these problems. Afterward, we provide the investigation on wildly used evaluation necessities tailored for long-context LLMs, including datasets, metrics, and baseline models, as well as some amazing optimization toolkits like libraries, systems, and compilers to augment LLMs' efficiency and efficacy across different stages. Finally, we further discuss the predominant challenges and potential avenues for future research in this domain. Additionally, we have established a repository where we curate relevant literature with real-time updates at https://github.com/Strivin0311/long-llms-learning.

{{</citation>}}


### (36/135) Do Smaller Language Models Answer Contextualised Questions Through Memorisation Or Generalisation? (Tim Hartill et al., 2023)

{{<citation>}}

Tim Hartill, Joshua Bensemann, Michael Witbrock, Patricia J. Riddle. (2023)  
**Do Smaller Language Models Answer Contextualised Questions Through Memorisation Or Generalisation?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12337v1)  

---


**ABSTRACT**  
A distinction is often drawn between a model's ability to predict a label for an evaluation sample that is directly memorised from highly similar training samples versus an ability to predict the label via some method of generalisation. In the context of using Language Models for question-answering, discussion continues to occur as to the extent to which questions are answered through memorisation. We consider this issue for questions that would ideally be answered through reasoning over an associated context. We propose a method of identifying evaluation samples for which it is very unlikely our model would have memorised the answers. Our method is based on semantic similarity of input tokens and label tokens between training and evaluation samples. We show that our method offers advantages upon some prior approaches in that it is able to surface evaluation-train pairs that have overlap in either contiguous or discontiguous sequences of tokens. We use this method to identify unmemorisable subsets of our evaluation datasets. We train two Language Models in a multitask fashion whereby the second model differs from the first only in that it has two additional datasets added to the training regime that are designed to impart simple numerical reasoning strategies of a sort known to improve performance on some of our evaluation datasets but not on others. We then show that there is performance improvement between the two models on the unmemorisable subsets of the evaluation datasets that were expected to benefit from the additional training datasets. Specifically, performance on unmemorisable subsets of two of our evaluation datasets, DROP and ROPES significantly improves by 9.0%, and 25.7% respectively while other evaluation datasets have no significant change in performance.

{{</citation>}}


### (37/135) AcademicGPT: Empowering Academic Research (Shufa Wei et al., 2023)

{{<citation>}}

Shufa Wei, Xiaolong Xu, Xianbiao Qi, Xi Yin, Jun Xia, Jingyi Ren, Peijun Tang, Yuxiang Zhong, Yihao Chen, Xiaoqin Ren, Yuxin Liang, Liankai Huang, Kai Xie, Weikang Gui, Wei Tan, Shuanglong Sun, Yongquan Hu, Qinxian Liu, Nanjin Li, Chihao Dai, Lihua Wang, Xiaohui Liu, Lei Zhang, Yutao Xie. (2023)  
**AcademicGPT: Empowering Academic Research**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, LLaMA, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.12315v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated exceptional capabilities across various natural language processing tasks. Yet, many of these advanced LLMs are tailored for broad, general-purpose applications. In this technical report, we introduce AcademicGPT, designed specifically to empower academic research. AcademicGPT is a continual training model derived from LLaMA2-70B. Our training corpus mainly consists of academic papers, thesis, content from some academic domain, high-quality Chinese data and others. While it may not be extensive in data scale, AcademicGPT marks our initial venture into a domain-specific GPT tailored for research area. We evaluate AcademicGPT on several established public benchmarks such as MMLU and CEval, as well as on some specialized academic benchmarks like PubMedQA, SCIEval, and our newly-created ComputerScienceQA, to demonstrate its ability from general knowledge ability, to Chinese ability, and to academic ability. Building upon AcademicGPT's foundation model, we also developed several applications catered to the academic area, including General Academic Question Answering, AI-assisted Paper Reading, Paper Review, and AI-assisted Title and Abstract Generation.

{{</citation>}}


### (38/135) ATLANTIC: Structure-Aware Retrieval-Augmented Language Model for Interdisciplinary Science (Sai Munikoti et al., 2023)

{{<citation>}}

Sai Munikoti, Anurag Acharya, Sridevi Wagle, Sameera Horawalavithana. (2023)  
**ATLANTIC: Structure-Aware Retrieval-Augmented Language Model for Interdisciplinary Science**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12289v1)  

---


**ABSTRACT**  
Large language models record impressive performance on many natural language processing tasks. However, their knowledge capacity is limited to the pretraining corpus. Retrieval augmentation offers an effective solution by retrieving context from external knowledge sources to complement the language model. However, existing retrieval augmentation techniques ignore the structural relationships between these documents. Furthermore, retrieval models are not explored much in scientific tasks, especially in regard to the faithfulness of retrieved documents. In this paper, we propose a novel structure-aware retrieval augmented language model that accommodates document structure during retrieval augmentation. We create a heterogeneous document graph capturing multiple types of relationships (e.g., citation, co-authorship, etc.) that connect documents from more than 15 scientific disciplines (e.g., Physics, Medicine, Chemistry, etc.). We train a graph neural network on the curated document graph to act as a structural encoder for the corresponding passages retrieved during the model pretraining. Particularly, along with text embeddings of the retrieved passages, we obtain structural embeddings of the documents (passages) and fuse them together before feeding them to the language model. We evaluate our model extensively on various scientific benchmarks that include science question-answering and scientific document classification tasks. Experimental results demonstrate that structure-aware retrieval improves retrieving more coherent, faithful and contextually relevant passages, while showing a comparable performance in the overall accuracy.

{{</citation>}}


### (39/135) Enabling On-Device Large Language Model Personalization with Self-Supervised Data Selection and Synthesis (Ruiyang Qin et al., 2023)

{{<citation>}}

Ruiyang Qin, Jun Xia, Zhenge Jia, Meng Jiang, Ahmed Abbasi, Peipei Zhou, Jingtong Hu, Yiyu Shi. (2023)  
**Enabling On-Device Large Language Model Personalization with Self-Supervised Data Selection and Synthesis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.12275v1)  

---


**ABSTRACT**  
After a large language model (LLM) is deployed on edge devices, it is desirable for these devices to learn from user-generated conversation data to generate user-specific and personalized responses in real-time. However, user-generated data usually contains sensitive and private information, and uploading such data to the cloud for annotation is not preferred if not prohibited. While it is possible to obtain annotation locally by directly asking users to provide preferred responses, such annotations have to be sparse to not affect user experience. In addition, the storage of edge devices is usually too limited to enable large-scale fine-tuning with full user-generated data. It remains an open question how to enable on-device LLM personalization, considering sparse annotation and limited on-device storage. In this paper, we propose a novel framework to select and store the most representative data online in a self-supervised way. Such data has a small memory footprint and allows infrequent requests of user annotations for further fine-tuning. To enhance fine-tuning quality, multiple semantically similar pairs of question texts and expected responses are generated using the LLM. Our experiments show that the proposed framework achieves the best user-specific content-generating capability (accuracy) and fine-tuning speed (performance) compared with vanilla baselines. To the best of our knowledge, this is the very first on-device LLM personalization framework.

{{</citation>}}


## hep-ex (1)



### (40/135) Training Deep 3D Convolutional Neural Networks to Extract BSM Physics Parameters Directly from HEP Data: a Proof-of-Concept Study Using Monte Carlo Simulations (S. Dubey et al., 2023)

{{<citation>}}

S. Dubey, T. E. Browder, S. Kohani, R. Mandal, A. Sibidanov, R. Sinha. (2023)  
**Training Deep 3D Convolutional Neural Networks to Extract BSM Physics Parameters Directly from HEP Data: a Proof-of-Concept Study Using Monte Carlo Simulations**  

---
Primary Category: hep-ex  
Categories: cs-LG, hep-ex, hep-ex, hep-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13060v1)  

---


**ABSTRACT**  
We report on a novel application of computer vision techniques to extract beyond the Standard Model (BSM) parameters directly from high energy physics (HEP) flavor data. We develop a method of transforming angular and kinematic distributions into "quasi-images" that can be used to train a convolutional neural network to perform regression tasks, similar to fitting. This contrasts with the usual classification functions performed using ML/AI in HEP. As a proof-of-concept, we train a 34-layer Residual Neural Network to regress on these images and determine the Wilson Coefficient $C_{9}$ in MC (Monte Carlo) simulations of $B \rightarrow K^{*}\mu^{+}\mu^{-}$ decays. The technique described here can be generalized and may find applicability across various HEP experiments and elsewhere.

{{</citation>}}


## cs.SD (2)



### (41/135) Self-Supervised Music Source Separation Using Vector-Quantized Source Category Estimates (Marco Pasini et al., 2023)

{{<citation>}}

Marco Pasini, Stefan Lattner, George Fazekas. (2023)  
**Self-Supervised Music Source Separation Using Vector-Quantized Source Category Estimates**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.13058v1)  

---


**ABSTRACT**  
Music source separation is focused on extracting distinct sonic elements from composite tracks. Historically, many methods have been grounded in supervised learning, necessitating labeled data, which is occasionally constrained in its diversity. More recent methods have delved into N-shot techniques that utilize one or more audio samples to aid in the separation. However, a challenge with some of these methods is the necessity for an audio query during inference, making them less suited for genres with varied timbres and effects. This paper offers a proof-of-concept for a self-supervised music source separation system that eliminates the need for audio queries at inference time. In the training phase, while it adopts a query-based approach, we introduce a modification by substituting the continuous embedding of query audios with Vector Quantized (VQ) representations. Trained end-to-end with up to N classes as determined by the VQ's codebook size, the model seeks to effectively categorise instrument classes. During inference, the input is partitioned into N sources, with some potentially left unutilized based on the mix's instrument makeup. This methodology suggests an alternative avenue for considering source separation across diverse music genres. We provide examples and additional results online.

{{</citation>}}


### (42/135) Equipping Pretrained Unconditional Music Transformers with Instrument and Genre Controls (Weihan Xu et al., 2023)

{{<citation>}}

Weihan Xu, Julian McAuley, Shlomo Dubnov, Hao-Wen Dong. (2023)  
**Equipping Pretrained Unconditional Music Transformers with Instrument and Genre Controls**  

---
Primary Category: cs.SD  
Categories: cs-IR, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.12257v1)  

---


**ABSTRACT**  
The ''pretraining-and-finetuning'' paradigm has become a norm for training domain-specific models in natural language processing and computer vision. In this work, we aim to examine this paradigm for symbolic music generation through leveraging the largest ever symbolic music dataset sourced from the MuseScore forum. We first pretrain a large unconditional transformer model using 1.5 million songs. We then propose a simple technique to equip this pretrained unconditional music transformer model with instrument and genre controls by finetuning the model with additional control tokens. Our proposed representation offers improved high-level controllability and expressiveness against two existing representations. The experimental results show that the proposed model can successfully generate music with user-specified instruments and genre. In a subjective listening test, the proposed model outperforms the pretrained baseline model in terms of coherence, harmony, arrangement and overall quality.

{{</citation>}}


## cs.HC (3)



### (43/135) The HaLLMark Effect: Supporting Provenance and Transparent Use of Large Language Models in Writing through Interactive Visualization (Md Naimul Hoque et al., 2023)

{{<citation>}}

Md Naimul Hoque, Tasfia Mashiat, Bhavya Ghai, Cecilia Shelton, Fanny Chevalier, Kari Kraus, Niklas Elmqvist. (2023)  
**The HaLLMark Effect: Supporting Provenance and Transparent Use of Large Language Models in Writing through Interactive Visualization**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13057v1)  

---


**ABSTRACT**  
The use of Large Language Models (LLMs) for writing has sparked controversy both among readers and writers. On one hand, writers are concerned that LLMs will deprive them of agency and ownership, and readers are concerned about spending their time on text generated by soulless machines. On the other hand, writers who genuinely want to use LLMs must conform to publisher policies for AI-assisted writing, and readers need assurance that a text has been verified by a human. We argue that a system that captures the provenance of interaction with an LLM can help writers retain their agency, conform to policies, and communicate their use of AI to publishers and readers transparently. Thus we propose HaLLMark, a tool for facilitating and visualizing writers' interaction with LLMs. We evaluated HaLLMark with 13 creative writers, and found that it helped them retain a sense of control and ownership of the written text.

{{</citation>}}


### (44/135) Keeping Users Engaged During Repeated Administration of the Same Questionnaire: Using Large Language Models to Reliably Diversify Questions (Hye Sun Yun et al., 2023)

{{<citation>}}

Hye Sun Yun, Mehdi Arjmand, Phillip Raymond Sherlock, Michael Paasche-Orlow, James W. Griffith, Timothy Bickmore. (2023)  
**Keeping Users Engaged During Repeated Administration of the Same Questionnaire: Using Large Language Models to Reliably Diversify Questions**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12707v1)  

---


**ABSTRACT**  
Standardized, validated questionnaires are vital tools in HCI research and healthcare, offering dependable self-report data. However, their repeated use in longitudinal or pre-post studies can induce respondent fatigue, impacting data quality via response biases and decreased response rates. We propose utilizing large language models (LLMs) to generate diverse questionnaire versions while retaining good psychometric properties. In a longitudinal study, participants engaged with our agent system and responded daily for two weeks to either a standardized depression questionnaire or one of two LLM-generated questionnaire variants, alongside a validated depression questionnaire. Psychometric testing revealed consistent covariation between the external criterion and the focal measure administered across the three conditions, demonstrating the reliability and validity of the LLM-generated variants. Participants found the repeated administration of the standardized questionnaire significantly more repetitive compared to the variants. Our findings highlight the potential of LLM-generated variants to invigorate questionnaires, fostering engagement and interest without compromising validity.

{{</citation>}}


### (45/135) Interpretability is in the eye of the beholder: Human versus artificial classification of image segments generated by humans versus XAI (Romy Müller et al., 2023)

{{<citation>}}

Romy Müller, Marius Thoß, Julian Ullrich, Steffen Seitz, Carsten Knoll. (2023)  
**Interpretability is in the eye of the beholder: Human versus artificial classification of image segments generated by humans versus XAI**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12481v1)  

---


**ABSTRACT**  
Evaluating methods of explainable artificial intelligence (XAI) is challenging, because the fidelity of an explanation to the AI model does not necessarily go hand in hand with its interpretability for humans. For instance, when classifying images with Convolutional Neural Networks (CNN), XAI algorithms can explain which image areas had an impact on the CNN decision. However, it is unclear whether the areas that best reflect the CNN internal data processing will also make most sense to humans. Thus, the present study investigated whether the image classification of humans and CNN is supported by the same explanations. To assess such differences in interpretability, human participants and a CNN classified image segments that were considered most informative either by other humans (as revealed by eye movements and manual selection) or by two XAI methods (Grad-CAM and XRAI). In three experiments, humans classified and rated these segments, and we also had a CNN classify them. The results indicated that the respective interpretability of the two XAI methods strongly depended on image type, both for humans and CNN. Moreover, human classification performance was highest with human segments, regardless of how they were generated (i.e., from eye movements or manual selection), whereas the type of human segment had major impacts on CNN classification performance. Our results caution against general statements about the interpretability of explanations, as this interpretability varies with the explanation method, the explanations to be interpreted, and the agent who needs to perform the interpretation.

{{</citation>}}


## cs.CV (36)



### (46/135) Descriptor and Word Soups: Overcoming the Parameter Efficiency Accuracy Tradeoff for Out-of-Distribution Few-shot Learning (Christopher Liao et al., 2023)

{{<citation>}}

Christopher Liao, Theodoros Tsiligkaridis, Brian Kulis. (2023)  
**Descriptor and Word Soups: Overcoming the Parameter Efficiency Accuracy Tradeoff for Out-of-Distribution Few-shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.13612v1)  

---


**ABSTRACT**  
Over the past year, a large body of multimodal research has emerged around zero-shot evaluation using GPT descriptors. These studies boost the zero-shot accuracy of pretrained VL models with an ensemble of label-specific text generated by GPT. A recent study, WaffleCLIP, demonstrated that similar zero-shot accuracy can be achieved with an ensemble of random descriptors. However, both zero-shot methods are un-trainable and consequently sub-optimal when some few-shot out-of-distribution (OOD) training data is available. Inspired by these prior works, we present two more flexible methods called descriptor and word soups, which do not require an LLM at test time and can leverage training data to increase OOD target accuracy. Descriptor soup greedily selects a small set of textual descriptors using generic few-shot training data, then calculates robust class embeddings using the selected descriptors. Word soup greedily assembles a chain of words in a similar manner. Compared to existing few-shot soft prompt tuning methods, word soup requires fewer parameters by construction and less GPU memory, since it does not require backpropagation. Both soups outperform current published few-shot methods, even when combined with SoTA zero-shot methods, on cross-dataset and domain generalization benchmarks. Compared with SoTA prompt and descriptor ensembling methods, such as ProDA and WaffleCLIP, word soup achieves higher OOD accuracy with fewer ensemble members. Please checkout our code: github.com/Chris210634/word_soups

{{</citation>}}


### (47/135) AI for Agriculture: the Comparison of Semantic Segmentation Methods for Crop Mapping with Sentinel-2 Imagery (Irina Korotkova et al., 2023)

{{<citation>}}

Irina Korotkova, Natalia Efremova. (2023)  
**AI for Agriculture: the Comparison of Semantic Segmentation Methods for Crop Mapping with Sentinel-2 Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.12993v1)  

---


**ABSTRACT**  
Crop mapping is one of the most common tasks in artificial intelligence for agriculture due to higher food demands from a growing population and increased awareness of climate change. In case of vineyards, the texture is very important for crop segmentation: with higher resolution satellite imagery the texture is easily detected by majority of state-of-the-art algorithms. However, this task becomes increasingly more difficult as the resolution of satellite imagery decreases and the information about the texture becomes unavailable. In this paper we aim to explore the main machine learning methods that can be used with freely available satellite imagery and discuss how and when they can be applied for vineyard segmentation problem. We assess the effectiveness of various widely-used machine learning techniques and offer guidance on selecting the most suitable model for specific scenarios.

{{</citation>}}


### (48/135) Innovative Horizons in Aerial Imagery: LSKNet Meets DiffusionDet for Advanced Object Detection (Ahmed Sharshar et al., 2023)

{{<citation>}}

Ahmed Sharshar, Aleksandr Matsun. (2023)  
**Innovative Horizons in Aerial Imagery: LSKNet Meets DiffusionDet for Advanced Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2311.12956v1)  

---


**ABSTRACT**  
In the realm of aerial image analysis, object detection plays a pivotal role, with significant implications for areas such as remote sensing, urban planning, and disaster management. This study addresses the inherent challenges in this domain, notably the detection of small objects, managing densely packed elements, and accounting for diverse orientations. We present an in-depth evaluation of an object detection model that integrates the Large Selective Kernel Network (LSKNet)as its backbone with the DiffusionDet head, utilizing the iSAID dataset for empirical analysis. Our approach encompasses the introduction of novel methodologies and extensive ablation studies. These studies critically assess various aspects such as loss functions, box regression techniques, and classification strategies to refine the model's precision in object detection. The paper details the experimental application of the LSKNet backbone in synergy with the DiffusionDet heads, a combination tailored to meet the specific challenges in aerial image object detection. The findings of this research indicate a substantial enhancement in the model's performance, especially in the accuracy-time tradeoff. The proposed model achieves a mean average precision (MAP) of approximately 45.7%, which is a significant improvement, outperforming the RCNN model by 4.7% on the same dataset. This advancement underscores the effectiveness of the proposed modifications and sets a new benchmark in aerial image analysis, paving the way for more accurate and efficient object detection methodologies. The code is publicly available at https://github.com/SashaMatsun/LSKDiffDet

{{</citation>}}


### (49/135) ShareGPT4V: Improving Large Multi-Modal Models with Better Captions (Lin Chen et al., 2023)

{{<citation>}}

Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, Dahua Lin. (2023)  
**ShareGPT4V: Improving Large Multi-Modal Models with Better Captions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.12793v1)  

---


**ABSTRACT**  
In the realm of large multi-modal models (LMMs), efficient modality alignment is crucial yet often constrained by the scarcity of high-quality image-text data. To address this bottleneck, we introduce the ShareGPT4V dataset, a pioneering large-scale resource featuring 1.2 million highly descriptive captions, which surpasses existing datasets in diversity and information content, covering world knowledge, object properties, spatial relationships, and aesthetic evaluations. Specifically, ShareGPT4V originates from a curated 100K high-quality captions collected from advanced GPT4-Vision and has been expanded to 1.2M with a superb caption model trained on this subset. ShareGPT4V first demonstrates its effectiveness for the Supervised Fine-Tuning (SFT) phase, by substituting an equivalent quantity of detailed captions in existing SFT datasets with a subset of our high-quality captions, significantly enhancing the LMMs like LLaVA-7B, LLaVA-1.5-13B, and Qwen-VL-Chat-7B on the MME and MMBench benchmarks, with respective gains of 222.8/22.0/22.3 and 2.7/1.3/1.5. We further incorporate ShareGPT4V data into both the pre-training and SFT phases, obtaining ShareGPT4V-7B, a superior LMM based on a simple architecture that has remarkable performance across a majority of the multi-modal benchmarks. This project is available at https://ShareGPT4V.github.io to serve as a pivotal resource for advancing the LMMs community.

{{</citation>}}


### (50/135) SPOT! Revisiting Video-Language Models for Event Understanding (Gengyuan Zhang et al., 2023)

{{<citation>}}

Gengyuan Zhang, Jinhe Bi, Jindong Gu, Volker Tresp. (2023)  
**SPOT! Revisiting Video-Language Models for Event Understanding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12919v1)  

---


**ABSTRACT**  
Understanding videos is an important research topic for multimodal learning. Leveraging large-scale datasets of web-crawled video-text pairs as weak supervision has become a pre-training paradigm for learning joint representations and showcased remarkable potential in video understanding tasks. However, videos can be multi-event and multi-grained, while these video-text pairs usually contain only broad-level video captions. This raises a question: with such weak supervision, can video representation in video-language models gain the ability to distinguish even factual discrepancies in textual description and understand fine-grained events? To address this, we introduce SPOT Prober, to benchmark existing video-language models's capacities of distinguishing event-level discrepancies as an indicator of models' event understanding ability. Our approach involves extracting events as tuples (<Subject, Predicate, Object, Attribute, Timestamps>) from videos and generating false event tuples by manipulating tuple components systematically. We reevaluate the existing video-language models with these positive and negative captions and find they fail to distinguish most of the manipulated events. Based on our findings, we propose to plug in these manipulated event captions as hard negative samples and find them effective in enhancing models for event understanding.

{{</citation>}}


### (51/135) Breathing Life Into Sketches Using Text-to-Video Priors (Rinon Gal et al., 2023)

{{<citation>}}

Rinon Gal, Yael Vinker, Yuval Alaluf, Amit H. Bermano, Daniel Cohen-Or, Ariel Shamir, Gal Chechik. (2023)  
**Breathing Life Into Sketches Using Text-to-Video Priors**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2311.13608v1)  

---


**ABSTRACT**  
A sketch is one of the most intuitive and versatile tools humans use to convey their ideas visually. An animated sketch opens another dimension to the expression of ideas and is widely used by designers for a variety of purposes. Animating sketches is a laborious process, requiring extensive experience and professional design skills. In this work, we present a method that automatically adds motion to a single-subject sketch (hence, "breathing life into it"), merely by providing a text prompt indicating the desired motion. The output is a short animation provided in vector representation, which can be easily edited. Our method does not require extensive training, but instead leverages the motion prior of a large pretrained text-to-video diffusion model using a score-distillation loss to guide the placement of strokes. To promote natural and smooth motion and to better preserve the sketch's appearance, we model the learned motion through two components. The first governs small local deformations and the second controls global affine transformations. Surprisingly, we find that even models that struggle to generate sketch videos on their own can still serve as a useful backbone for animating abstract representations.

{{</citation>}}


### (52/135) SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction (Yuanhui Huang et al., 2023)

{{<citation>}}

Yuanhui Huang, Wenzhao Zheng, Borui Zhang, Jie Zhou, Jiwen Lu. (2023)  
**SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.12754v1)  

---


**ABSTRACT**  
3D occupancy prediction is an important task for the robustness of vision-centric autonomous driving, which aims to predict whether each point is occupied in the surrounding 3D space. Existing methods usually require 3D occupancy labels to produce meaningful results. However, it is very laborious to annotate the occupancy status of each voxel. In this paper, we propose SelfOcc to explore a self-supervised way to learn 3D occupancy using only video sequences. We first transform the images into the 3D space (e.g., bird's eye view) to obtain 3D representation of the scene. We directly impose constraints on the 3D representations by treating them as signed distance fields. We can then render 2D images of previous and future frames as self-supervision signals to learn the 3D representations. We propose an MVS-embedded strategy to directly optimize the SDF-induced weights with multiple depth proposals. Our SelfOcc outperforms the previous best method SceneRF by 58.7% using a single frame as input on SemanticKITTI and is the first self-supervised work that produces reasonable 3D occupancy for surround cameras on Occ3D. SelfOcc produces high-quality depth and achieves state-of-the-art results on novel depth synthesis, monocular depth estimation, and surround-view depth estimation on the SemanticKITTI, KITTI-2015, and nuScenes, respectively. Code: https://github.com/huang-yh/SelfOcc.

{{</citation>}}


### (53/135) Attention Deficit is Ordered! Fooling Deformable Vision Transformers with Collaborative Adversarial Patches (Quazi Mishkatul Alam et al., 2023)

{{<citation>}}

Quazi Mishkatul Alam, Bilel Tarchoun, Ihsen Alouani, Nael Abu-Ghazaleh. (2023)  
**Attention Deficit is Ordered! Fooling Deformable Vision Transformers with Collaborative Adversarial Patches**  

---
Primary Category: cs.CV  
Categories: I-4, cs-CR, cs-CV, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.12914v1)  

---


**ABSTRACT**  
The latest generation of transformer-based vision models have proven to be superior to Convolutional Neural Network (CNN)-based models across several vision tasks, largely attributed to their remarkable prowess in relation modeling. Deformable vision transformers significantly reduce the quadratic complexity of modeling attention by using sparse attention structures, enabling them to be used in larger scale applications such as multi-view vision systems. Recent work demonstrated adversarial attacks against transformers; we show that these attacks do not transfer to deformable transformers due to their sparse attention structure. Specifically, attention in deformable transformers is modeled using pointers to the most relevant other tokens. In this work, we contribute for the first time adversarial attacks that manipulate the attention of deformable transformers, distracting them to focus on irrelevant parts of the image. We also develop new collaborative attacks where a source patch manipulates attention to point to a target patch that adversarially attacks the system. In our experiments, we find that only 1% patched area of the input field can lead to 0% AP. We also show that the attacks provide substantial versatility to support different attacker scenarios because of their ability to redirect attention under the attacker control.

{{</citation>}}


### (54/135) Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatially Relation Matching (Meng Chu et al., 2023)

{{<citation>}}

Meng Chu, Zhedong Zheng, Wei Ji, Tat-Seng Chua. (2023)  
**Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatially Relation Matching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Drone, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12751v1)  

---


**ABSTRACT**  
Drone navigation through natural language commands remains a significant challenge due to the lack of publicly available multi-modal datasets and the intricate demands of fine-grained visual-text alignment. In response to this pressing need, we present a new human-computer interaction annotation benchmark called GeoText-1652, meticulously curated through a robust Large Language Model (LLM)-based data generation framework and the expertise of pre-trained vision models. This new dataset seamlessly extends the existing image dataset, \ie, University-1652, with spatial-aware text annotations, encompassing intricate image-text-bounding box associations. Besides, we introduce a new optimization objective to leverage fine-grained spatial associations, called blending spatial matching, for region-level spatial relation matching. Extensive experiments reveal that our approach maintains an exceptional recall rate under varying description complexities. This underscores the promising potential of our approach in elevating drone control and navigation through the seamless integration of natural language commands in real-world scenarios.

{{</citation>}}


### (55/135) Diffusion Model Alignment Using Direct Preference Optimization (Bram Wallace et al., 2023)

{{<citation>}}

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, Nikhil Naik. (2023)  
**Diffusion Model Alignment Using Direct Preference Optimization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.12908v1)  

---


**ABSTRACT**  
Large language models (LLMs) are fine-tuned using human comparison data with Reinforcement Learning from Human Feedback (RLHF) methods to make them better aligned with users' preferences. In contrast to LLMs, human preference learning has not been widely explored in text-to-image diffusion models; the best existing approach is to fine-tune a pretrained model using carefully curated high quality images and captions to improve visual appeal and text alignment. We propose Diffusion-DPO, a method to align diffusion models to human preferences by directly optimizing on human comparison data. Diffusion-DPO is adapted from the recently developed Direct Preference Optimization (DPO), a simpler alternative to RLHF which directly optimizes a policy that best satisfies human preferences under a classification objective. We re-formulate DPO to account for a diffusion model notion of likelihood, utilizing the evidence lower bound to derive a differentiable objective. Using the Pick-a-Pic dataset of 851K crowdsourced pairwise preferences, we fine-tune the base model of the state-of-the-art Stable Diffusion XL (SDXL)-1.0 model with Diffusion-DPO. Our fine-tuned base model significantly outperforms both base SDXL-1.0 and the larger SDXL-1.0 model consisting of an additional refinement model in human evaluation, improving visual appeal and prompt alignment. We also develop a variant that uses AI feedback and has comparable performance to training on human preferences, opening the door for scaling of diffusion model alignment methods.

{{</citation>}}


### (56/135) Similar Document Template Matching Algorithm (Harshitha Yenigalla et al., 2023)

{{<citation>}}

Harshitha Yenigalla, Bommareddy Revanth Srinivasa Reddy, Batta Venkata Rahul, Nannapuraju Hemanth Raju. (2023)  
**Similar Document Template Matching Algorithm**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2311.12663v1)  

---


**ABSTRACT**  
This study outlines a comprehensive methodology for verifying medical documents, integrating advanced techniques in template extraction, comparison, and fraud detection. It begins with template extraction using sophisticated region-of-interest (ROI) methods, incorporating contour analysis and edge identification. Pre-processing steps ensure template clarity through morphological operations and adaptive thresholding. The template comparison algorithm utilizes advanced feature matching with key points and descriptors, enhancing robustness through histogram-based analysis for accounting variations. Fraud detection involves the SSIM computation and OCR for textual information extraction. The SSIM quantifies structural similarity, aiding in potential match identification. OCR focuses on critical areas like patient details, provider information, and billing amounts. Extracted information is compared with a reference dataset, and confidence thresholding ensures reliable fraud detection. Adaptive parameters enhance system flexibility for dynamic adjustments to varying document layouts. This methodology provides a robust approach to medical document verification, addressing complexities in template extraction, comparison, fraud detection, and adaptability to diverse document structures.

{{</citation>}}


### (57/135) Mobile-Seed: Joint Semantic Segmentation and Boundary Detection for Mobile Robots (Youqi Liao et al., 2023)

{{<citation>}}

Youqi Liao, Shuhao Kang, Jianping Li, Yang Liu, Yun Liu, Zhen Dong, Bisheng Yang, Xieyuanli Chen. (2023)  
**Mobile-Seed: Joint Semantic Segmentation and Boundary Detection for Mobile Robots**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.12651v2)  

---


**ABSTRACT**  
Precise and rapid delineation of sharp boundaries and robust semantics is essential for numerous downstream robotic tasks, such as robot grasping and manipulation, real-time semantic mapping, and online sensor calibration performed on edge computing units. Although boundary detection and semantic segmentation are complementary tasks, most studies focus on lightweight models for semantic segmentation but overlook the critical role of boundary detection. In this work, we introduce Mobile-Seed, a lightweight, dual-task framework tailored for simultaneous semantic segmentation and boundary detection. Our framework features a two-stream encoder, an active fusion decoder (AFD) and a dual-task regularization approach. The encoder is divided into two pathways: one captures category-aware semantic information, while the other discerns boundaries from multi-scale features. The AFD module dynamically adapts the fusion of semantic and boundary information by learning channel-wise relationships, allowing for precise weight assignment of each channel. Furthermore, we introduce a regularization loss to mitigate the conflicts in dual-task learning and deep diversity supervision. Compared to existing methods, the proposed Mobile-Seed offers a lightweight framework to simultaneously improve semantic segmentation performance and accurately locate object boundaries. Experiments on the Cityscapes dataset have shown that Mobile-Seed achieves notable improvement over the state-of-the-art (SOTA) baseline by 2.2 percentage points (pp) in mIoU and 4.2 pp in mF-score, while maintaining an online inference speed of 23.9 frames-per-second (FPS) with 1024x2048 resolution input on an RTX 2080 Ti GPU. Additional experiments on CamVid and PASCAL Context datasets confirm our method's generalizability. Code and additional results are publicly available at https://whu-usi3dv.github.io/Mobile-Seed/.

{{</citation>}}


### (58/135) KNVQA: A Benchmark for evaluation knowledge-based VQA (Sirui Cheng et al., 2023)

{{<citation>}}

Sirui Cheng, Siyu Zhang, Jiayi Wu, Muchen Lan. (2023)  
**KNVQA: A Benchmark for evaluation knowledge-based VQA**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2311.12639v1)  

---


**ABSTRACT**  
Within the multimodal field, large vision-language models (LVLMs) have made significant progress due to their strong perception and reasoning capabilities in the visual and language systems. However, LVLMs are still plagued by the two critical issues of object hallucination and factual accuracy, which limit the practicality of LVLMs in different scenarios. Furthermore, previous evaluation methods focus more on the comprehension and reasoning of language content but lack a comprehensive evaluation of multimodal interactions, thereby resulting in potential limitations. To this end, we propose a novel KNVQA-Eval, which is devoted to knowledge-based VQA task evaluation to reflect the factuality of multimodal LVLMs. To ensure the robustness and scalability of the evaluation, we develop a new KNVQA dataset by incorporating human judgment and perception, aiming to evaluate the accuracy of standard answers relative to AI-generated answers in knowledge-based VQA. This work not only comprehensively evaluates the contextual information of LVLMs using reliable human annotations, but also further analyzes the fine-grained capabilities of current methods to reveal potential avenues for subsequent optimization of LVLMs-based estimators. Our proposed VQA-Eval and corresponding dataset KNVQA will facilitate the development of automatic evaluation tools with the advantages of low cost, privacy protection, and reproducibility. Our code will be released upon publication.

{{</citation>}}


### (59/135) GPT4Motion: Scripting Physical Motions in Text-to-Video Generation via Blender-Oriented GPT Planning (Jiaxi Lv et al., 2023)

{{<citation>}}

Jiaxi Lv, Yi Huang, Mingfu Yan, Jiancheng Huang, Jianzhuang Liu, Yifan Liu, Yafei Wen, Xiaoxin Chen, Shifeng Chen. (2023)  
**GPT4Motion: Scripting Physical Motions in Text-to-Video Generation via Blender-Oriented GPT Planning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.12631v1)  

---


**ABSTRACT**  
Recent advances in text-to-video generation have harnessed the power of diffusion models to create visually compelling content conditioned on text prompts. However, they usually encounter high computational costs and often struggle to produce videos with coherent physical motions. To tackle these issues, we propose GPT4Motion, a training-free framework that leverages the planning capability of large language models such as GPT, the physical simulation strength of Blender, and the excellent image generation ability of text-to-image diffusion models to enhance the quality of video synthesis. Specifically, GPT4Motion employs GPT-4 to generate a Blender script based on a user textual prompt, which commands Blender's built-in physics engine to craft fundamental scene components that encapsulate coherent physical motions across frames. Then these components are inputted into Stable Diffusion to generate a video aligned with the textual prompt. Experimental results on three basic physical motion scenarios, including rigid object drop and collision, cloth draping and swinging, and liquid flow, demonstrate that GPT4Motion can generate high-quality videos efficiently in maintaining motion coherency and entity consistency. GPT4Motion offers new insights in text-to-video research, enhancing its quality and broadening its horizon for future explorations.

{{</citation>}}


### (60/135) Bridging Generalization Gaps in High Content Imaging Through Online Self-Supervised Domain Adaptation (Johan Fredin Haslum et al., 2023)

{{<citation>}}

Johan Fredin Haslum, Christos Matsoukas, Karl-Johan Leuchowius, Kevin Smith. (2023)  
**Bridging Generalization Gaps in High Content Imaging Through Online Self-Supervised Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.12623v1)  

---


**ABSTRACT**  
High Content Imaging (HCI) plays a vital role in modern drug discovery and development pipelines, facilitating various stages from hit identification to candidate drug characterization. Applying machine learning models to these datasets can prove challenging as they typically consist of multiple batches, affected by experimental variation, especially if different imaging equipment have been used. Moreover, as new data arrive, it is preferable that they are analyzed in an online fashion. To overcome this, we propose CODA, an online self-supervised domain adaptation approach. CODA divides the classifier's role into a generic feature extractor and a task-specific model. We adapt the feature extractor's weights to the new domain using cross-batch self-supervision while keeping the task-specific model unchanged. Our results demonstrate that this strategy significantly reduces the generalization gap, achieving up to a 300% improvement when applied to data from different labs utilizing different microscopes. CODA can be applied to new, unlabeled out-of-domain data sources of different sizes, from a single plate to multiple experimental batches.

{{</citation>}}


### (61/135) Crowd management, crime detection, work monitoring using aiml (P. R. Adithya et al., 2023)

{{<citation>}}

P. R. Adithya, Dheepak. S, B. Akash, Harshini. V, Sai Lakshana. (2023)  
**Crowd management, crime detection, work monitoring using aiml**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12621v1)  

---


**ABSTRACT**  
This research endeavors to harness the potential of existing Closed-Circuit Television (CCTV) networks for a comprehensive approach to crowd management, crime prevention, and workplace monitoring through the integration of Artificial Intelligence (AI) and Machine Learning (ML) technologies. The primary objective is to develop and implement advanced algorithms capable of real-time analysis of video feeds, enabling the identification and assessment of crowd dynamics, early detection of potential criminal activities, and continuous monitoring of workplace environments. By leveraging AI/ML, the project aims to optimize surveillance capabilities, thereby enhancing public safety measures and improving organizational productivity. This initiative underscores the transformative impact that intelligent video analytics can have on existing infrastructure, mitigating the need for extensive system overhauls while significantly advancing security and operational efficiency.

{{</citation>}}


### (62/135) Leveraging Unlabeled Data for 3D Medical Image Segmentation through Self-Supervised Contrastive Learning (Sanaz Karimijafarbigloo et al., 2023)

{{<citation>}}

Sanaz Karimijafarbigloo, Reza Azad, Yury Velichko, Ulas Bagci, Dorit Merhof. (2023)  
**Leveraging Unlabeled Data for 3D Medical Image Segmentation through Self-Supervised Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.12617v1)  

---


**ABSTRACT**  
Current 3D semi-supervised segmentation methods face significant challenges such as limited consideration of contextual information and the inability to generate reliable pseudo-labels for effective unsupervised data use. To address these challenges, we introduce two distinct subnetworks designed to explore and exploit the discrepancies between them, ultimately correcting the erroneous prediction results. More specifically, we identify regions of inconsistent predictions and initiate a targeted verification training process. This procedure strategically fine-tunes and harmonizes the predictions of the subnetworks, leading to enhanced utilization of contextual information. Furthermore, to adaptively fine-tune the network's representational capacity and reduce prediction uncertainty, we employ a self-supervised contrastive learning paradigm. For this, we use the network's confidence to distinguish between reliable and unreliable predictions. The model is then trained to effectively minimize unreliable predictions. Our experimental results for organ segmentation, obtained from clinical MRI and CT scans, demonstrate the effectiveness of our approach when compared to state-of-the-art methods. The codebase is accessible on \href{https://github.com/xmindflow/SSL-contrastive}{GitHub}.

{{</citation>}}


### (63/135) Adaptive Dense Pseudo Label Selection for Semi-supervised Oriented Object Detection (Tong Zhao et al., 2023)

{{<citation>}}

Tong Zhao, Qiang Fang, Shuohao Shi, Xin Xu. (2023)  
**Adaptive Dense Pseudo Label Selection for Semi-supervised Oriented Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.12608v1)  

---


**ABSTRACT**  
Recently, dense pseudo-label, which directly selects pseudo labels from the original output of the teacher model without any complicated post-processing steps, has received considerable attention in semi-supervised object detection (SSOD). However, for the multi-oriented and dense objects that are common in aerial scenes, existing dense pseudo-label selection methods are inefficient and impede the performance in semi-supervised oriented object detection. Therefore, we propose Adaptive Dense Pseudo Label Selection (ADPLS) for semi-supervised oriented object detection. In ADPLS, we design a simple but effective adaptive mechanism to guide the selection of dense pseudo labels. Specifically, we propose the mean Feature-Richness Score (mFRS) to estimate the density of potential objects and use this score to adjust the number of dense pseudo labels. On the DOTA-v1.5 benchmark, the proposed method outperforms previous methods especially when labeled data are scarce. For example, it achieves 49.78 mAP given only 5% of annotated data, which surpasses previous state-of-the-art method given 10% of annotated data by 1.15 mAP. Our codes will be available soon.

{{</citation>}}


### (64/135) Improving Source-Free Target Adaptation with Vision Transformers Leveraging Domain Representation Images (Gauransh Sawhney et al., 2023)

{{<citation>}}

Gauransh Sawhney, Daksh Dave, Adeel Ahmed, Jiechao Gao, Khalid Saleem. (2023)  
**Improving Source-Free Target Adaptation with Vision Transformers Leveraging Domain Representation Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.12589v1)  

---


**ABSTRACT**  
Unsupervised Domain Adaptation (UDA) methods facilitate knowledge transfer from a labeled source domain to an unlabeled target domain, navigating the obstacle of domain shift. While Convolutional Neural Networks (CNNs) are a staple in UDA, the rise of Vision Transformers (ViTs) provides new avenues for domain generalization. This paper presents an innovative method to bolster ViT performance in source-free target adaptation, beginning with an evaluation of how key, query, and value elements affect ViT outcomes. Experiments indicate that altering the key component has negligible effects on Transformer performance. Leveraging this discovery, we introduce Domain Representation Images (DRIs), feeding embeddings through the key element. DRIs act as domain-specific markers, effortlessly merging with the training regimen. To assess our method, we perform target adaptation tests on the Cross Instance DRI source-only (SO) control. We measure the efficacy of target adaptation with and without DRIs, against existing benchmarks like SHOT-B* and adaptations via CDTrans. Findings demonstrate that excluding DRIs offers limited gains over SHOT-B*, while their inclusion in the key segment boosts average precision promoting superior domain generalization. This research underscores the vital role of DRIs in enhancing ViT efficiency in UDA scenarios, setting a precedent for further domain adaptation explorations.

{{</citation>}}


### (65/135) HiPose: Hierarchical Binary Surface Encoding and Correspondence Pruning for RGB-D 6DoF Object Pose Estimation (Yongliang Lin et al., 2023)

{{<citation>}}

Yongliang Lin, Yongzhi Su, Praveen Nathan, Sandeep Inuganti, Yan Di, Martin Sundermeyer, Fabian Manhardt, Didier Stricke, Jason Rambach, Yu Zhang. (2023)  
**HiPose: Hierarchical Binary Surface Encoding and Correspondence Pruning for RGB-D 6DoF Object Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.12588v1)  

---


**ABSTRACT**  
In this work, we present a novel dense-correspondence method for 6DoF object pose estimation from a single RGB-D image. While many existing data-driven methods achieve impressive performance, they tend to be time-consuming due to their reliance on rendering-based refinement approaches. To circumvent this limitation, we present HiPose, which establishes 3D-3D correspondences in a coarse-to-fine manner with a hierarchical binary surface encoding. Unlike previous dense-correspondence methods, we estimate the correspondence surface by employing point-to-surface matching and iteratively constricting the surface until it becomes a correspondence point while gradually removing outliers. Extensive experiments on public benchmarks LM-O, YCB-V, and T-Less demonstrate that our method surpasses all refinement-free methods and is even on par with expensive refinement-based approaches. Crucially, our approach is computationally efficient and enables real-time critical applications with high accuracy requirements. Code and models will be released.

{{</citation>}}


### (66/135) Benchmarking bias: Expanding clinical AI model card to incorporate bias reporting of social and non-social factors (Carolina A. M. Heming et al., 2023)

{{<citation>}}

Carolina A. M. Heming, Mohamed Abdalla, Monish Ahluwalia, Linglin Zhang, Hari Trivedi, MinJae Woo, Benjamin Fine, Judy Wawira Gichoya, Leo Anthony Celi, Laleh Seyyed-Kalantari. (2023)  
**Benchmarking bias: Expanding clinical AI model card to incorporate bias reporting of social and non-social factors**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2311.12560v1)  

---


**ABSTRACT**  
Clinical AI model reporting cards should be expanded to incorporate a broad bias reporting of both social and non-social factors. Non-social factors consider the role of other factors, such as disease dependent, anatomic, or instrument factors on AI model bias, which are essential to ensure safe deployment.

{{</citation>}}


### (67/135) HCA-Net: Hierarchical Context Attention Network for Intervertebral Disc Semantic Labeling (Afshin Bozorgpour et al., 2023)

{{<citation>}}

Afshin Bozorgpour, Bobby Azad, Reza Azad, Yury Velichko, Ulas Bagci, Dorit Merhof. (2023)  
**HCA-Net: Hierarchical Context Attention Network for Intervertebral Disc Semantic Labeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.12486v1)  

---


**ABSTRACT**  
Accurate and automated segmentation of intervertebral discs (IVDs) in medical images is crucial for assessing spine-related disorders, such as osteoporosis, vertebral fractures, or IVD herniation. We present HCA-Net, a novel contextual attention network architecture for semantic labeling of IVDs, with a special focus on exploiting prior geometric information. Our approach excels at processing features across different scales and effectively consolidating them to capture the intricate spatial relationships within the spinal cord. To achieve this, HCA-Net models IVD labeling as a pose estimation problem, aiming to minimize the discrepancy between each predicted IVD location and its corresponding actual joint location. In addition, we introduce a skeletal loss term to reinforce the model's geometric dependence on the spine. This loss function is designed to constrain the model's predictions to a range that matches the general structure of the human vertebral skeleton. As a result, the network learns to reduce the occurrence of false predictions and adaptively improves the accuracy of IVD location estimation. Through extensive experimental evaluation on multi-center spine datasets, our approach consistently outperforms previous state-of-the-art methods on both MRI T1w and T2w modalities. The codebase is accessible to the public on \href{https://github.com/xmindflow/HCA-Net}{GitHub}.

{{</citation>}}


### (68/135) Speaker-Adapted End-to-End Visual Speech Recognition for Continuous Spanish (David Gimeno-Gómez et al., 2023)

{{<citation>}}

David Gimeno-Gómez, Carlos-D. Martínez-Hinarejos. (2023)  
**Speaker-Adapted End-to-End Visual Speech Recognition for Continuous Spanish**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-SD, cs.CV, eess-AS  
Keywords: Attention, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2311.12480v1)  

---


**ABSTRACT**  
Different studies have shown the importance of visual cues throughout the speech perception process. In fact, the development of audiovisual approaches has led to advances in the field of speech technologies. However, although noticeable results have recently been achieved, visual speech recognition remains an open research problem. It is a task in which, by dispensing with the auditory sense, challenges such as visual ambiguities and the complexity of modeling silence must be faced. Nonetheless, some of these challenges can be alleviated when the problem is approached from a speaker-dependent perspective. Thus, this paper studies, using the Spanish LIP-RTVE database, how the estimation of specialized end-to-end systems for a specific person could affect the quality of speech recognition. First, different adaptation strategies based on the fine-tuning technique were proposed. Then, a pre-trained CTC/Attention architecture was used as a baseline throughout our experiments. Our findings showed that a two-step fine-tuning process, where the VSR system is first adapted to the task domain, provided significant improvements when the speaker adaptation was addressed. Furthermore, results comparable to the current state of the art were reached even when only a limited amount of data was available.

{{</citation>}}


### (69/135) LIP-RTVE: An Audiovisual Database for Continuous Spanish in the Wild (David Gimeno-Gómez et al., 2023)

{{<citation>}}

David Gimeno-Gómez, Carlos-D. Martínez-Hinarejos. (2023)  
**LIP-RTVE: An Audiovisual Database for Continuous Spanish in the Wild**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2311.12457v1)  

---


**ABSTRACT**  
Speech is considered as a multi-modal process where hearing and vision are two fundamentals pillars. In fact, several studies have demonstrated that the robustness of Automatic Speech Recognition systems can be improved when audio and visual cues are combined to represent the nature of speech. In addition, Visual Speech Recognition, an open research problem whose purpose is to interpret speech by reading the lips of the speaker, has been a focus of interest in the last decades. Nevertheless, in order to estimate these systems in the currently Deep Learning era, large-scale databases are required. On the other hand, while most of these databases are dedicated to English, other languages lack sufficient resources. Thus, this paper presents a semi-automatically annotated audiovisual database to deal with unconstrained natural Spanish, providing 13 hours of data extracted from Spanish television. Furthermore, baseline results for both speaker-dependent and speaker-independent scenarios are reported using Hidden Markov Models, a traditional paradigm that has been widely used in the field of Speech Technologies.

{{</citation>}}


### (70/135) AR Visualization System for Ship Detection and Recognition Based on AI (Ziqi Ye et al., 2023)

{{<citation>}}

Ziqi Ye, Limin Huang, Yongji Wu, Min Hu. (2023)  
**AR Visualization System for Ship Detection and Recognition Based on AI**  

---
Primary Category: cs.CV  
Categories: 68T07, I-2; I-4, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12430v1)  

---


**ABSTRACT**  
Augmented reality technology has been widely used in industrial design interaction, exhibition guide, information retrieval and other fields. The combination of artificial intelligence and augmented reality technology has also become a future development trend. This project is an AR visualization system for ship detection and recognition based on AI, which mainly includes three parts: artificial intelligence module, Unity development module and Hololens2AR module. This project is based on R3Det algorithm to complete the detection and recognition of ships in remote sensing images. The recognition rate of model detection trained on RTX 2080Ti can reach 96%. Then, the 3D model of the ship is obtained by ship categories and information and generated in the virtual scene. At the same time, voice module and UI interaction module are added. Finally, we completed the deployment of the project on Hololens2 through MRTK. The system realizes the fusion of computer vision and augmented reality technology, which maps the results of object detection to the AR field, and makes a brave step toward the future technological trend and intelligent application.

{{</citation>}}


### (71/135) Rich and Poor Texture Contrast: A Simple yet Effective Approach for AI-generated Image Detection (Nan Zhong et al., 2023)

{{<citation>}}

Nan Zhong, Yiran Xu, Zhenxing Qian, Xinpeng Zhang. (2023)  
**Rich and Poor Texture Contrast: A Simple yet Effective Approach for AI-generated Image Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12397v1)  

---


**ABSTRACT**  
Recent generative models show impressive performance in generating photographic images. Humans can hardly distinguish such incredibly realistic-looking AI-generated images from real ones. AI-generated images may lead to ubiquitous disinformation dissemination. Therefore, it is of utmost urgency to develop a detector to identify AI-generated images. Most existing detectors suffer from sharp performance drops over unseen generative models. In this paper, we propose a novel AI-generated image detector capable of identifying fake images created by a wide range of generative models. Our approach leverages the inter-pixel correlation contrast between rich and poor texture regions within an image. Pixels in rich texture regions exhibit more significant fluctuations than those in poor texture regions. This discrepancy reflects that the entropy of rich texture regions is larger than that of poor ones. Consequently, synthesizing realistic rich texture regions proves to be more challenging for existing generative models. Based on this principle, we divide an image into multiple patches and reconstruct them into two images, comprising rich-texture and poor-texture patches respectively. Subsequently, we extract the inter-pixel correlation discrepancy feature between rich and poor texture regions. This feature serves as a universal fingerprint used for AI-generated image forensics across different generative models. In addition, we build a comprehensive AI-generated image detection benchmark, which includes 16 kinds of prevalent generative models, to evaluate the effectiveness of existing baselines and our approach. Our benchmark provides a leaderboard for follow-up studies. Extensive experimental results show that our approach outperforms state-of-the-art baselines by a significant margin. Our project: https://fdmas.github.io/AIGCDetect/

{{</citation>}}


### (72/135) From Wrong To Right: A Recursive Approach Towards Vision-Language Explanation (Jiaxin Ge et al., 2023)

{{<citation>}}

Jiaxin Ge, Sanjay Subramanian, Trevor Darrell, Boyi Li. (2023)  
**From Wrong To Right: A Recursive Approach Towards Vision-Language Explanation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BLEU, QA  
[Paper Link](http://arxiv.org/abs/2311.12391v1)  

---


**ABSTRACT**  
Addressing the challenge of adapting pre-trained vision-language models for generating insightful explanations for visual reasoning tasks with limited annotations, we present ReVisE: a $\textbf{Re}$cursive $\textbf{Vis}$ual $\textbf{E}$xplanation algorithm. Our method iteratively computes visual features (conditioned on the text input), an answer, and an explanation, to improve the explanation quality step by step until the answer converges. We find that this multi-step approach guides the model to correct its own answers and outperforms single-step explanation generation. Furthermore, explanations generated by ReVisE also serve as valuable annotations for few-shot self-training. Our approach outperforms previous methods while utilizing merely 5% of the human-annotated explanations across 10 metrics, demonstrating up to a 4.2 and 1.3 increase in BLEU-1 score on the VCR and VQA-X datasets, underscoring the efficacy and data-efficiency of our method.

{{</citation>}}


### (73/135) Enhancing Scene Graph Generation with Hierarchical Relationships and Commonsense Knowledge (Bowen Jiang et al., 2023)

{{<citation>}}

Bowen Jiang, Zhijun Zhuang, Camillo Jose Taylor. (2023)  
**Enhancing Scene Graph Generation with Hierarchical Relationships and Commonsense Knowledge**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Commonsense Knowledge  
[Paper Link](http://arxiv.org/abs/2311.12889v1)  

---


**ABSTRACT**  
This work presents an enhanced approach to generating scene graphs by incorporating a relationship hierarchy and commonsense knowledge. Specifically, we propose a Bayesian classification head that exploits an informative hierarchical structure. It jointly predicts the super-category or type of relationship between the two objects, along with the detailed relationship under each super-category. We design a commonsense validation pipeline that uses a large language model to critique the results from the scene graph prediction system and then use that feedback to enhance the model performance. The system requires no external large language model assistance at test time, making it more convenient for practical applications. Experiments on the Visual Genome and the OpenImage V6 datasets demonstrate that harnessing hierarchical relationships enhances the model performance by a large margin. The proposed Bayesian head can also be incorporated as a portable module in existing scene graph generation algorithms to improve their results. In addition, the commonsense validation enables the model to generate an extensive set of reasonable predictions beyond dataset annotations.

{{</citation>}}


### (74/135) Post-Training Quantization with Low-precision Minifloats and Integers on FPGAs (Shivam Aggarwal et al., 2023)

{{<citation>}}

Shivam Aggarwal, Alessandro Pappalardo, Hans Jakob Damsgaard, Giuseppe Franco, Thomas B. Preußer, Michaela Blott, Tulika Mitra. (2023)  
**Post-Training Quantization with Low-precision Minifloats and Integers on FPGAs**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-AR, cs-CV, cs-LG, cs-PF, cs.CV  
Keywords: GPT, Quantization  
[Paper Link](http://arxiv.org/abs/2311.12359v1)  

---


**ABSTRACT**  
Post-Training Quantization (PTQ) is a powerful technique for model compression, reducing the precision of neural networks without additional training overhead. Recent works have investigated adopting 8-bit floating-point quantization (FP8) in the context of PTQ for model inference. However, the exploration of floating-point formats smaller than 8 bits and their comparison with integer quantization remains relatively limited. In this work, we present minifloats, which are reduced-precision floating-point formats capable of further reducing the memory footprint, latency, and energy cost of a model while approaching full-precision model accuracy. Our work presents a novel PTQ design-space exploration, comparing minifloat and integer quantization schemes across a range of 3 to 8 bits for both weights and activations. We examine the applicability of various PTQ techniques to minifloats, including weight equalization, bias correction, SmoothQuant, gradient-based learned rounding, and the GPTQ method. Our experiments validate the effectiveness of low-precision minifloats when compared to their integer counterparts across a spectrum of accuracy-precision trade-offs on a set of reference deep learning vision workloads. Finally, we evaluate our results against an FPGA-based hardware cost model, showing that integer quantization often remains the Pareto-optimal option, given its relatively smaller hardware resource footprint.

{{</citation>}}


### (75/135) Stable Diffusion For Aerial Object Detection (Yanan Jian et al., 2023)

{{<citation>}}

Yanan Jian, Fuxun Yu, Simranjit Singh, Dimitrios Stamoulis. (2023)  
**Stable Diffusion For Aerial Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.12345v1)  

---


**ABSTRACT**  
Aerial object detection is a challenging task, in which one major obstacle lies in the limitations of large-scale data collection and the long-tail distribution of certain classes. Synthetic data offers a promising solution, especially with recent advances in diffusion-based methods like stable diffusion (SD). However, the direct application of diffusion methods to aerial domains poses unique challenges: stable diffusion's optimization for rich ground-level semantics doesn't align with the sparse nature of aerial objects, and the extraction of post-synthesis object coordinates remains problematic. To address these challenges, we introduce a synthetic data augmentation framework tailored for aerial images. It encompasses sparse-to-dense region of interest (ROI) extraction to bridge the semantic gap, fine-tuning the diffusion model with low-rank adaptation (LORA) to circumvent exhaustive retraining, and finally, a Copy-Paste method to compose synthesized objects with backgrounds, providing a nuanced approach to aerial object detection through synthetic data.

{{</citation>}}


### (76/135) LoCo: Locally Constrained Training-Free Layout-to-Image Synthesis (Peiang Zhao et al., 2023)

{{<citation>}}

Peiang Zhao, Han Li, Ruiyang Jin, S. Kevin Zhou. (2023)  
**LoCo: Locally Constrained Training-Free Layout-to-Image Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.12342v1)  

---


**ABSTRACT**  
Recent text-to-image diffusion models have reached an unprecedented level in generating high-quality images. However, their exclusive reliance on textual prompts often falls short in accurately conveying fine-grained spatial compositions. In this paper, we propose LoCo, a training-free approach for layout-to-image synthesis that excels in producing high-quality images aligned with both textual prompts and spatial layouts. Our method introduces a Localized Attention Constraint to refine cross-attention for individual objects, ensuring their precise placement in designated regions. We further propose a Padding Token Constraint to leverage the semantic information embedded in previously neglected padding tokens, thereby preventing the undesired fusion of synthesized objects. LoCo seamlessly integrates into existing text-to-image and layout-to-image models, significantly amplifying their performance and effectively addressing semantic failures observed in prior methods. Through extensive experiments, we showcase the superiority of our approach, surpassing existing state-of-the-art training-free layout-to-image methods both qualitatively and quantitatively across multiple benchmarks.

{{</citation>}}


### (77/135) ViLaM: A Vision-Language Model with Enhanced Visual Grounding and Generalization Capability (Xiaoyu Yang et al., 2023)

{{<citation>}}

Xiaoyu Yang, Lijian Xu, Hongsheng Li, Shaoting Zhang. (2023)  
**ViLaM: A Vision-Language Model with Enhanced Visual Grounding and Generalization Capability**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12327v1)  

---


**ABSTRACT**  
Vision-language models have revolutionized human-computer interaction and shown significant progress in multi-modal tasks. However, applying these models to complex visual tasks like medical image analysis remains challenging. In this study, we propose ViLaM, a unified Vision-Language transformer model that integrates instruction tuning predicated on a large language model. This approach enables us to optimally utilize the knowledge and reasoning capacities of large pre-trained language models for an array of tasks encompassing both language and vision. We employ frozen pre-trained encoders to encode and align both image and text features, enabling ViLaM to handle a variety of visual tasks following textual instructions. Besides, we've designed cycle training for referring expressions to address the need for high-quality, paired referring expression datasets for training large models in terms of both quantity and quality. We evaluated ViLaM's exceptional performance on public general datasets and further confirmed its generalizability on medical datasets. Importantly, we've observed the model's impressive zero-shot learning ability, indicating the potential future application of ViLaM in the medical field.

{{</citation>}}


### (78/135) Long-MIL: Scaling Long Contextual Multiple Instance Learning for Histopathology Whole Slide Image Analysis (Honglin Li et al., 2023)

{{<citation>}}

Honglin Li, Yunlong Zhang, Chenglu Zhu, Jiatong Cai, Sunyi Zheng, Lin Yang. (2023)  
**Long-MIL: Scaling Long Contextual Multiple Instance Learning for Histopathology Whole Slide Image Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Bias, Transformer  
[Paper Link](http://arxiv.org/abs/2311.12885v1)  

---


**ABSTRACT**  
Histopathology image analysis is the golden standard of clinical diagnosis for Cancers. In doctors daily routine and computer-aided diagnosis, the Whole Slide Image (WSI) of histopathology tissue is used for analysis. Because of the extremely large scale of resolution, previous methods generally divide the WSI into a large number of patches, then aggregate all patches within a WSI by Multi-Instance Learning (MIL) to make the slide-level prediction when developing computer-aided diagnosis tools. However, most previous WSI-MIL models using global-attention without pairwise interaction and any positional information, or self-attention with absolute position embedding can not well handle shape varying large WSIs, e.g. testing WSIs after model deployment may be larger than training WSIs, since the model development set is always limited due to the difficulty of histopathology WSIs collection. To deal with the problem, in this paper, we propose to amend position embedding for shape varying long-contextual WSI by introducing Linear Bias into Attention, and adapt it from 1-d long sequence into 2-d long-contextual WSI which helps model extrapolate position embedding to unseen or under-fitted positions. We further utilize Flash-Attention module to tackle the computational complexity of Transformer, which also keep full self-attention performance compared to previous attention approximation work. Our method, Long-contextual MIL (Long-MIL) are evaluated on extensive experiments including 4 dataset including WSI classification and survival prediction tasks to validate the superiority on shape varying WSIs. The source code will be open-accessed soon.

{{</citation>}}


### (79/135) ABFL: Angular Boundary Discontinuity Free Loss for Arbitrary Oriented Object Detection in Aerial Images (Zifei Zhao et al., 2023)

{{<citation>}}

Zifei Zhao, Shengyang Li. (2023)  
**ABFL: Angular Boundary Discontinuity Free Loss for Arbitrary Oriented Object Detection in Aerial Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.12311v1)  

---


**ABSTRACT**  
Arbitrary oriented object detection (AOOD) in aerial images is a widely concerned and highly challenging task, and plays an important role in many scenarios. The core of AOOD involves the representation, encoding, and feature augmentation of oriented bounding-boxes (Bboxes). Existing methods lack intuitive modeling of angle difference measurement in oriented Bbox representations. Oriented Bboxes under different representations exhibit rotational symmetry with varying periods due to angle periodicity. The angular boundary discontinuity (ABD) problem at periodic boundary positions is caused by rotational symmetry in measuring angular differences. In addition, existing methods also use additional encoding-decoding structures for oriented Bboxes. In this paper, we design an angular boundary free loss (ABFL) based on the von Mises distribution. The ABFL aims to solve the ABD problem when detecting oriented objects. Specifically, ABFL proposes to treat angles as circular data rather than linear data when measuring angle differences, aiming to introduce angle periodicity to alleviate the ABD problem and improve the accuracy of angle difference measurement. In addition, ABFL provides a simple and effective solution for various periodic boundary discontinuities caused by rotational symmetry in AOOD tasks, as it does not require additional encoding-decoding structures for oriented Bboxes. Extensive experiments on the DOTA and HRSC2016 datasets show that the proposed ABFL loss outperforms some state-of-the-art methods focused on addressing the ABD problem.

{{</citation>}}


### (80/135) Instance-aware 3D Semantic Segmentation powered by Shape Generators and Classifiers (Bo Sun et al., 2023)

{{<citation>}}

Bo Sun, Qixing Huang, Xiangru Huang. (2023)  
**Instance-aware 3D Semantic Segmentation powered by Shape Generators and Classifiers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.12291v1)  

---


**ABSTRACT**  
Existing 3D semantic segmentation methods rely on point-wise or voxel-wise feature descriptors to output segmentation predictions. However, these descriptors are often supervised at point or voxel level, leading to segmentation models that can behave poorly at instance-level. In this paper, we proposed a novel instance-aware approach for 3D semantic segmentation. Our method combines several geometry processing tasks supervised at instance-level to promote the consistency of the learned feature representation. Specifically, our methods use shape generators and shape classifiers to perform shape reconstruction and classification tasks for each shape instance. This enforces the feature representation to faithfully encode both structural and local shape information, with an awareness of shape instances. In the experiments, our method significantly outperform existing approaches in 3D semantic segmentation on several public benchmarks, such as Waymo Open Dataset, SemanticKITTI and ScanNetV2.

{{</citation>}}


### (81/135) Boosting Audio-visual Zero-shot Learning with Large Language Models (Haoxing Chen et al., 2023)

{{<citation>}}

Haoxing Chen, Yaohui Li, Yan Hong, Zizheng Huang, Zhuoer Xu, Zhangxuan Gu, Jun Lan, Huijia Zhu, Weiqiang Wang. (2023)  
**Boosting Audio-visual Zero-shot Learning with Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12268v1)  

---


**ABSTRACT**  
Audio-visual zero-shot learning aims to recognize unseen categories based on paired audio-visual sequences. Recent methods mainly focus on learning aligned and discriminative multi-modal features to boost generalization towards unseen categories. However, these approaches ignore the obscure action concepts in category names and may inevitably introduce complex network structures with difficult training objectives. In this paper, we propose a simple yet effective framework named Knowledge-aware Distribution Adaptation (KDA) to help the model better grasp the novel action contents with an external knowledge base. Specifically, we first propose using large language models to generate rich descriptions from category names, which leads to a better understanding of unseen categories. Additionally, we propose a distribution alignment loss as well as a knowledge-aware adaptive margin loss to further improve the generalization ability towards unseen categories. Extensive experimental results demonstrate that our proposed KDA can outperform state-of-the-art methods on three popular audio-visual zero-shot learning datasets. Our code will be avaliable at \url{https://github.com/chenhaoxing/KDA}.

{{</citation>}}


## cs.CY (4)



### (82/135) Attention: Large Multimodal Model is Watching your Geo-privacy (Yifan Yang et al., 2023)

{{<citation>}}

Yifan Yang, Yixian Zhang, Daoyang Li, Shuju Sun, Junhong Duan, Junzhou He, Qingyang Wu, Hao Liu. (2023)  
**Attention: Large Multimodal Model is Watching your Geo-privacy**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CV, cs-CY, cs-SI, cs.CY  
Keywords: AI, Attention, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.13018v1)  

---


**ABSTRACT**  
Geographic privacy, a crucial aspect of personal security, often goes unnoticed in daily activities. This paper addresses the underestimation of this privacy in the context of increasing online data sharing and the advancements in information gathering technologies. With the surge in the use of Large Multimodal Models, such as GPT-4, for Open Source Intelligence (OSINT), the potential risks associated with geographic privacy breaches have intensified. This study highlights the criticality of these developments, focusing on their implications for individual privacy. The primary objective is to demonstrate the capabilities of advanced AI tools, specifically a GPT-4 based model named "Dr. Watson," in identifying and potentially compromising geographic privacy through online shared content. We developed "Dr. Watson" to analyze and extract geographic information from publicly available data sources. The study involved five experimental cases, each offering different perspectives on the tool's application in extracting precise location data from partial images and social media content. The experiments revealed that "Dr. Watson" could successfully identify specific geographic details, thereby exposing the vulnerabilities in current geo-privacy measures. These findings underscore the ease with which geographic information can be unintentionally disclosed. The paper concludes with a discussion on the broader implications of these findings for individuals and the community at large. It emphasizes the urgency for enhanced awareness and protective measures against geo-privacy leakage in the era of advanced AI and widespread social media usage.

{{</citation>}}


### (83/135) Not Just Training, Also Testing: High School Youths' Perspective-Taking through Peer Testing Machine Learning-Powered Applications (L. Morales-Navarro et al., 2023)

{{<citation>}}

L. Morales-Navarro, M. Shah, Y. B. Kafai. (2023)  
**Not Just Training, Also Testing: High School Youths' Perspective-Taking through Peer Testing Machine Learning-Powered Applications**  

---
Primary Category: cs.CY  
Categories: K-3-2, cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12733v1)  

---


**ABSTRACT**  
Most attention in K-12 artificial intelligence and machine learning (AI/ML) education has been given to having youths train models, with much less attention to the equally important testing of models when creating machine learning applications. Testing ML applications allows for the evaluation of models against predictions and can help creators of applications identify and address failure and edge cases that could negatively impact user experiences. We investigate how testing each other's projects supported youths to take perspective about functionality, performance, and potential issues in their own projects. We analyzed testing worksheets, audio and video recordings collected during a two week workshop in which 11 high school youths created physical computing projects that included (audio, pose, and image) ML classifiers. We found that through peer-testing youths reflected on the size of their training datasets, the diversity of their training data, the design of their classes and the contexts in which they produced training data. We discuss future directions for research on peer-testing in AI/ML education and current limitations for these kinds of activities.

{{</citation>}}


### (84/135) 'There Has To Be a Lot That We're Missing': Moderating AI-Generated Content on Reddit (Travis Lloyd et al., 2023)

{{<citation>}}

Travis Lloyd, Joseph Reagle, Mor Naaman. (2023)  
**'There Has To Be a Lot That We're Missing': Moderating AI-Generated Content on Reddit**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-SI, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.12702v1)  

---


**ABSTRACT**  
Generative AI threatens to disrupt how we work, learn, communicate, and participate in online communities. We performed a qualitative interview study to understand how online communities on the social sharing site Reddit are challenged by AI-generated content (AIGC) and how they are adapting. We conducted fifteen in-depth, semi-structured interviews with subreddit moderators about their experiences moderating AIGC. Though our participants see both legitimate and illegitimate motivations for using AIGC, on the whole they view it as detrimental to their communities, with a level of concern that is dependent on the purpose and size of their subreddits. Moderators reported developing rules and using a variety of strategies that may help communities prevent or curb AIGC, but without foolproof detection tools, enforcement is challenging and relies on heuristics. Overall, for online communities, the threat of Generative AI is not speculative: the disruption has already begun.

{{</citation>}}


### (85/135) Moderating Model Marketplaces: Platform Governance Puzzles for AI Intermediaries (Robert Gorwa et al., 2023)

{{<citation>}}

Robert Gorwa, Michael Veale. (2023)  
**Moderating Model Marketplaces: Platform Governance Puzzles for AI Intermediaries**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-LG, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12573v1)  

---


**ABSTRACT**  
The AI development community is increasingly making use of hosting intermediaries such as Hugging Face provide easy access to user-uploaded models and training data. These model marketplaces lower technical deployment barriers for hundreds of thousands of users, yet can be used in numerous potentially harmful and illegal ways. In this article, we explain ways in which AI systems, which can both `contain' content and be open-ended tools, present one of the trickiest platform governance challenges seen to date. We provide case studies of several incidents across three illustrative platforms -- Hugging Face, GitHub and Civitai -- to examine how model marketplaces moderate models. Building on this analysis, we outline important (and yet nevertheless limited) practices that industry has been developing to respond to moderation demands: licensing, access and use restrictions, automated content moderation, and open policy development. While the policy challenge at hand is a considerable one, we conclude with some ideas as to how platforms could better mobilize resources to act as a careful, fair, and proportionate regulatory access point.

{{</citation>}}


## cs.LG (22)



### (86/135) CovarNav: Machine Unlearning via Model Inversion and Covariance Navigation (Ali Abbasi et al., 2023)

{{<citation>}}

Ali Abbasi, Chayne Thrash, Elaheh Akbari, Daniel Zhang, Soheil Kolouri. (2023)  
**CovarNav: Machine Unlearning via Model Inversion and Covariance Navigation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12999v1)  

---


**ABSTRACT**  
The rapid progress of AI, combined with its unprecedented public adoption and the propensity of large neural networks to memorize training data, has given rise to significant data privacy concerns. To address these concerns, machine unlearning has emerged as an essential technique to selectively remove the influence of specific training data points on trained models. In this paper, we approach the machine unlearning problem through the lens of continual learning. Given a trained model and a subset of training data designated to be forgotten (i.e., the "forget set"), we introduce a three-step process, named CovarNav, to facilitate this forgetting. Firstly, we derive a proxy for the model's training data using a model inversion attack. Secondly, we mislabel the forget set by selecting the most probable class that deviates from the actual ground truth. Lastly, we deploy a gradient projection method to minimize the cross-entropy loss on the modified forget set (i.e., learn incorrect labels for this set) while preventing forgetting of the inverted samples. We rigorously evaluate CovarNav on the CIFAR-10 and Vggface2 datasets, comparing our results with recent benchmarks in the field and demonstrating the efficacy of our proposed approach.

{{</citation>}}


### (87/135) How Capable Can a Transformer Become? A Study on Synthetic, Interpretable Tasks (Rahul Ramesh et al., 2023)

{{<citation>}}

Rahul Ramesh, Mikail Khona, Robert P. Dick, Hidenori Tanaka, Ekdeep Singh Lubana. (2023)  
**How Capable Can a Transformer Become? A Study on Synthetic, Interpretable Tasks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.12997v1)  

---


**ABSTRACT**  
Transformers trained on huge text corpora exhibit a remarkable set of capabilities, e.g., performing simple logical operations. Given the inherent compositional nature of language, one can expect the model to learn to compose these capabilities, potentially yielding a combinatorial explosion of what operations it can perform on an input. Motivated by the above, we aim to assess in this paper "how capable can a transformer become?". Specifically, we train autoregressive Transformer models on a data-generating process that involves compositions of a set of well-defined monolithic capabilities. Through a series of extensive and systematic experiments on this data-generating process, we show that: (1) autoregressive Transformers can learn compositional structures from the training data and generalize to exponentially or even combinatorially many functions; (2) composing functions by generating intermediate outputs is more effective at generalizing to unseen compositions, compared to generating no intermediate outputs; (3) the training data has a significant impact on the model's ability to compose unseen combinations of functions; and (4) the attention layers in the latter half of the model are critical to compositionality.

{{</citation>}}


### (88/135) Quantifying Impairment and Disease Severity Using AI Models Trained on Healthy Subjects (Boyang Yu et al., 2023)

{{<citation>}}

Boyang Yu, Aakash Kaku, Kangning Liu, Avinash Parnandi, Emily Fokas, Anita Venkatesan, Natasha Pandit, Rajesh Ranganath, Heidi Schambra, Carlos Fernandez-Granda. (2023)  
**Quantifying Impairment and Disease Severity Using AI Models Trained on Healthy Subjects**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-QM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12781v1)  

---


**ABSTRACT**  
Automatic assessment of impairment and disease severity is a key challenge in data-driven medicine. We propose a novel framework to address this challenge, which leverages AI models trained exclusively on healthy individuals. The COnfidence-Based chaRacterization of Anomalies (COBRA) score exploits the decrease in confidence of these models when presented with impaired or diseased patients to quantify their deviation from the healthy population. We applied the COBRA score to address a key limitation of current clinical evaluation of upper-body impairment in stroke patients. The gold-standard Fugl-Meyer Assessment (FMA) requires in-person administration by a trained assessor for 30-45 minutes, which restricts monitoring frequency and precludes physicians from adapting rehabilitation protocols to the progress of each patient. The COBRA score, computed automatically in under one minute, is shown to be strongly correlated with the FMA on an independent test cohort for two different data modalities: wearable sensors ($\rho = 0.845$, 95% CI [0.743,0.908]) and video ($\rho = 0.746$, 95% C.I [0.594, 0.847]). To demonstrate the generalizability of the approach to other conditions, the COBRA score was also applied to quantify severity of knee osteoarthritis from magnetic-resonance imaging scans, again achieving significant correlation with an independent clinical assessment ($\rho = 0.644$, 95% C.I [0.585,0.696]).

{{</citation>}}


### (89/135) Learning to Optimise Wind Farms with Graph Transformers (Siyi Li et al., 2023)

{{<citation>}}

Siyi Li, Arnaud Robert, A. Aldo Faisal, Matthew D. Piggott. (2023)  
**Learning to Optimise Wind Farms with Graph Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.12750v1)  

---


**ABSTRACT**  
This work proposes a novel data-driven model capable of providing accurate predictions for the power generation of all wind turbines in wind farms of arbitrary layout, yaw angle configurations and wind conditions. The proposed model functions by encoding a wind farm into a fully-connected graph and processing the graph representation through a graph transformer. The graph transformer surrogate is shown to generalise well and is able to uncover latent structural patterns within the graph representation of wind farms. It is demonstrated how the resulting surrogate model can be used to optimise yaw angle configurations using genetic algorithms, achieving similar levels of accuracy to industrially-standard wind farm simulation tools while only taking a fraction of the computational cost.

{{</citation>}}


### (90/135) Content Augmented Graph Neural Networks (Fatemeh Gholamzadeh Nasrabadi et al., 2023)

{{<citation>}}

Fatemeh Gholamzadeh Nasrabadi, AmirHossein Kashani, Pegah Zahedi, Mostafa Haghir Chehreghani. (2023)  
**Content Augmented Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.12741v1)  

---


**ABSTRACT**  
In recent years, graph neural networks (GNNs) have become a popular tool for solving various problems over graphs. In these models, the link structure of the graph is typically exploited and nodes' embeddings are iteratively updated based on adjacent nodes. Nodes' contents are used solely in the form of feature vectors, served as nodes' first-layer embeddings. However, the filters or convolutions, applied during iterations/layers to these initial embeddings lead to their impact diminish and contribute insignificantly to the final embeddings. In order to address this issue, in this paper we propose augmenting nodes' embeddings by embeddings generating from their content, at higher GNN layers. More precisely, we propose models wherein a structural embedding using a GNN and a content embedding are computed for each node. These two are combined using a combination layer to form the embedding of a node at a given layer. We suggest methods such as using an auto-encoder or building a content graph, to generate content embeddings. In the end, by conducting experiments over several real-world datasets, we demonstrate the high accuracy and performance of our models.

{{</citation>}}


### (91/135) Exploring Graph Classification Techniques Under Low Data Constraints: A Comprehensive Study (Kush Kothari et al., 2023)

{{<citation>}}

Kush Kothari, Bhavya Mehta, Reshmika Nambiar, Seema Shrawne. (2023)  
**Exploring Graph Classification Techniques Under Low Data Constraints: A Comprehensive Study**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.12737v1)  

---


**ABSTRACT**  
This survey paper presents a brief overview of recent research on graph data augmentation and few-shot learning. It covers various techniques for graph data augmentation, including node and edge perturbation, graph coarsening, and graph generation, as well as the latest developments in few-shot learning, such as meta-learning and model-agnostic meta-learning. The paper explores these areas in depth and delves into further sub classifications. Rule based approaches and learning based approaches are surveyed under graph augmentation techniques. Few-Shot Learning on graphs is also studied in terms of metric learning techniques and optimization-based techniques. In all, this paper provides an extensive array of techniques that can be employed in solving graph processing problems faced in low-data scenarios.

{{</citation>}}


### (92/135) Adversarial Reweighting Guided by Wasserstein Distance for Bias Mitigation (Xuan Zhao et al., 2023)

{{<citation>}}

Xuan Zhao, Simone Fabbrizzi, Paula Reyero Lobo, Siamak Ghodsi, Klaus Broelemann, Steffen Staab, Gjergji Kasneci. (2023)  
**Adversarial Reweighting Guided by Wasserstein Distance for Bias Mitigation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.12684v1)  

---


**ABSTRACT**  
The unequal representation of different groups in a sample population can lead to discrimination of minority groups when machine learning models make automated decisions. To address these issues, fairness-aware machine learning jointly optimizes two (or more) metrics aiming at predictive effectiveness and low unfairness. However, the inherent under-representation of minorities in the data makes the disparate treatment of subpopulations less noticeable and difficult to deal with during learning. In this paper, we propose a novel adversarial reweighting method to address such \emph{representation bias}. To balance the data distribution between the majority and the minority groups, our approach deemphasizes samples from the majority group. To minimize empirical risk, our method prefers samples from the majority group that are close to the minority group as evaluated by the Wasserstein distance. Our theoretical analysis shows the effectiveness of our adversarial reweighting approach. Experiments demonstrate that our approach mitigates bias without sacrificing classification accuracy, outperforming related state-of-the-art methods on image and tabular benchmark datasets.

{{</citation>}}


### (93/135) Interpretation of the Transformer and Improvement of the Extractor (Zhe Chen, 2023)

{{<citation>}}

Zhe Chen. (2023)  
**Interpretation of the Transformer and Improvement of the Extractor**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.12678v1)  

---


**ABSTRACT**  
It has been over six years since the Transformer architecture was put forward. Surprisingly, the vanilla Transformer architecture is still widely used today. One reason is that the lack of deep understanding and comprehensive interpretation of the Transformer architecture makes it more challenging to improve the Transformer architecture. In this paper, we first interpret the Transformer architecture comprehensively in plain words based on our understanding and experiences. The interpretations are further proved and verified. These interpretations also cover the Extractor, a family of drop-in replacements for the multi-head self-attention in the Transformer architecture. Then, we propose an improvement on a type of the Extractor that outperforms the self-attention, without introducing additional trainable parameters. Experimental results demonstrate that the improved Extractor performs even better, showing a way to improve the Transformer architecture.

{{</citation>}}


### (94/135) Careful Selection and Thoughtful Discarding: Graph Explicit Pooling Utilizing Discarded Nodes (Chuang Liu et al., 2023)

{{<citation>}}

Chuang Liu, Wenhang Yu, Kuang Gao, Xueqi Ma, Yibing Zhan, Jia Wu, Bo Du, Wenbin Hu. (2023)  
**Careful Selection and Thoughtful Discarding: Graph Explicit Pooling Utilizing Discarded Nodes**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Convolutional Network, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.12644v1)  

---


**ABSTRACT**  
Graph pooling has been increasingly recognized as crucial for Graph Neural Networks (GNNs) to facilitate hierarchical graph representation learning. Existing graph pooling methods commonly consist of two stages: selecting top-ranked nodes and discarding the remaining to construct coarsened graph representations. However, this paper highlights two key issues with these methods: 1) The process of selecting nodes to discard frequently employs additional Graph Convolutional Networks or Multilayer Perceptrons, lacking a thorough evaluation of each node's impact on the final graph representation and subsequent prediction tasks. 2) Current graph pooling methods tend to directly discard the noise segment (dropped) of the graph without accounting for the latent information contained within these elements. To address the first issue, we introduce a novel Graph Explicit Pooling (GrePool) method, which selects nodes by explicitly leveraging the relationships between the nodes and final representation vectors crucial for classification. The second issue is addressed using an extended version of GrePool (i.e., GrePool+), which applies a uniform loss on the discarded nodes. This addition is designed to augment the training process and improve classification accuracy. Furthermore, we conduct comprehensive experiments across 12 widely used datasets to validate our proposed method's effectiveness, including the Open Graph Benchmark datasets. Our experimental results uniformly demonstrate that GrePool outperforms 14 baseline methods for most datasets. Likewise, implementing GrePool+ enhances GrePool's performance without incurring additional computational costs.

{{</citation>}}


### (95/135) Hierarchical Joint Graph Learning and Multivariate Time Series Forecasting (Juhyeon Kim et al., 2023)

{{<citation>}}

Juhyeon Kim, Hyungeun Lee, Seungwon Yu, Ung Hwang, Wooyul Jung, Miseon Park, Kijung Yoon. (2023)  
**Hierarchical Joint Graph Learning and Multivariate Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Time Series  
[Paper Link](http://arxiv.org/abs/2311.12630v1)  

---


**ABSTRACT**  
Multivariate time series is prevalent in many scientific and industrial domains. Modeling multivariate signals is challenging due to their long-range temporal dependencies and intricate interactions--both direct and indirect. To confront these complexities, we introduce a method of representing multivariate signals as nodes in a graph with edges indicating interdependency between them. Specifically, we leverage graph neural networks (GNN) and attention mechanisms to efficiently learn the underlying relationships within the time series data. Moreover, we suggest employing hierarchical signal decompositions running over the graphs to capture multiple spatial dependencies. The effectiveness of our proposed model is evaluated across various real-world benchmark datasets designed for long-term forecasting tasks. The results consistently showcase the superiority of our model, achieving an average 23\% reduction in mean squared error (MSE) compared to existing models.

{{</citation>}}


### (96/135) Bridging Algorithmic Information Theory and Machine Learning: A New Approach to Kernel Learning (Boumediene Hamzi et al., 2023)

{{<citation>}}

Boumediene Hamzi, Marcus Hutter, Houman Owhadi. (2023)  
**Bridging Algorithmic Information Theory and Machine Learning: A New Approach to Kernel Learning**  

---
Primary Category: cs.LG  
Categories: cs-IT, cs-LG, cs.LG, math-IT, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12624v1)  

---


**ABSTRACT**  
Machine Learning (ML) and Algorithmic Information Theory (AIT) look at Complexity from different points of view. We explore the interface between AIT and Kernel Methods (that are prevalent in ML) by adopting an AIT perspective on the problem of learning kernels from data, in kernel ridge regression, through the method of Sparse Kernel Flows. In particular, by looking at the differences and commonalities between Minimal Description Length (MDL) and Regularization in Machine Learning (RML), we prove that the method of Sparse Kernel Flows is the natural approach to adopt to learn kernels from data. This paper shows that it is not necessary to use the statistical route to derive Sparse Kernel Flows and that one can directly work with code-lengths and complexities that are concepts that show up in AIT.

{{</citation>}}


### (97/135) Explainable Anomaly Detection using Masked Latent Generative Modeling (Daesoo Lee et al., 2023)

{{<citation>}}

Daesoo Lee, Sara Malacarne, Erlend Aune. (2023)  
**Explainable Anomaly Detection using Masked Latent Generative Modeling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2311.12550v2)  

---


**ABSTRACT**  
We present a novel time series anomaly detection method that achieves excellent detection accuracy while offering a superior level of explainability. Our proposed method, TimeVQVAE-AD, leverages masked generative modeling adapted from the cutting-edge time series generation method known as TimeVQVAE. The prior model is trained on the discrete latent space of a time-frequency domain. Notably, the dimensional semantics of the time-frequency domain are preserved in the latent space, enabling us to compute anomaly scores across different frequency bands, which provides a better insight into the detected anomalies. Additionally, the generative nature of the prior model allows for sampling likely normal states for detected anomalies, enhancing the explainability of the detected anomalies through counterfactuals. Our experimental evaluation on the UCR Time Series Anomaly archive demonstrates that TimeVQVAE-AD significantly surpasses the existing methods in terms of detection accuracy and explainability.

{{</citation>}}


### (98/135) In-Context Learning Functions with Varying Number of Minima (David Oniani et al., 2023)

{{<citation>}}

David Oniani, Yanshan Wang. (2023)  
**In-Context Learning Functions with Varying Number of Minima**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12538v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have proven effective at In-Context Learning (ICL), an ability that allows them to create predictors from labeled examples. Few studies have explored the interplay between ICL and specific properties of functions it attempts to approximate. In our study, we use a formal framework to explore ICL and propose a new task of approximating functions with varying number of minima. We implement a method that allows for producing functions with given inputs as minima. We find that increasing the number of minima degrades ICL performance. At the same time, our evaluation shows that ICL outperforms 2-layer Neural Network (2NN) model. Furthermore, ICL learns faster than 2NN in all settings. We validate the findings through a set of few-shot experiments across various hyperparameter configurations.

{{</citation>}}


### (99/135) Neural Network Pruning by Gradient Descent (Zhang Zhang et al., 2023)

{{<citation>}}

Zhang Zhang, Ruyi Tao, Jiang Zhang. (2023)  
**Neural Network Pruning by Gradient Descent**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.12526v2)  

---


**ABSTRACT**  
The rapid increase in the parameters of deep learning models has led to significant costs, challenging computational efficiency and model interpretability. In this paper, we introduce a novel and straightforward neural network pruning framework that incorporates the Gumbel-Softmax technique. This framework enables the simultaneous optimization of a network's weights and topology in an end-to-end process using stochastic gradient descent. Empirical results demonstrate its exceptional compression capability, maintaining high accuracy on the MNIST dataset with only 0.15\% of the original network parameters. Moreover, our framework enhances neural network interpretability, not only by allowing easy extraction of feature importance directly from the pruned network but also by enabling visualization of feature symmetry and the pathways of information propagation from features to outcomes. Although the pruning strategy is learned through deep learning, it is surprisingly intuitive and understandable, focusing on selecting key representative features and exploiting data patterns to achieve extreme sparse pruning. We believe our method opens a promising new avenue for deep learning pruning and the creation of interpretable machine learning systems.

{{</citation>}}


### (100/135) ALPHA: AnomaLous Physiological Health Assessment Using Large Language Models (Jiankai Tang et al., 2023)

{{<citation>}}

Jiankai Tang, Kegang Wang, Hongming Hu, Xiyuxing Zhang, Peiyu Wang, Xin Liu, Yuntao Wang. (2023)  
**ALPHA: AnomaLous Physiological Health Assessment Using Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12524v1)  

---


**ABSTRACT**  
This study concentrates on evaluating the efficacy of Large Language Models (LLMs) in healthcare, with a specific focus on their application in personal anomalous health monitoring. Our research primarily investigates the capabilities of LLMs in interpreting and analyzing physiological data obtained from FDA-approved devices. We conducted an extensive analysis using anomalous physiological data gathered in a simulated low-air-pressure plateau environment. This allowed us to assess the precision and reliability of LLMs in understanding and evaluating users' health status with notable specificity. Our findings reveal that LLMs exhibit exceptional performance in determining medical indicators, including a Mean Absolute Error (MAE) of less than 1 beat per minute for heart rate and less than 1% for oxygen saturation (SpO2). Furthermore, the Mean Absolute Percentage Error (MAPE) for these evaluations remained below 1%, with the overall accuracy of health assessments surpassing 85%. In image analysis tasks, such as interpreting photoplethysmography (PPG) data, our specially adapted GPT models demonstrated remarkable proficiency, achieving less than 1 bpm error in cycle count and 7.28 MAE for heart rate estimation. This study highlights LLMs' dual role as health data analysis tools and pivotal elements in advanced AI health assistants, offering personalized health insights and recommendations within the future health assistant framework.

{{</citation>}}


### (101/135) Multi-Objective Reinforcement Learning based on Decomposition: A taxonomy and framework (Florian Felten et al., 2023)

{{<citation>}}

Florian Felten, El-Ghazali Talbi, Grégoire Danoy. (2023)  
**Multi-Objective Reinforcement Learning based on Decomposition: A taxonomy and framework**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.12495v1)  

---


**ABSTRACT**  
Multi-objective reinforcement learning (MORL) extends traditional RL by seeking policies making different compromises among conflicting objectives. The recent surge of interest in MORL has led to diverse studies and solving methods, often drawing from existing knowledge in multi-objective optimization based on decomposition (MOO/D). Yet, a clear categorization based on both RL and MOO/D is lacking in the existing literature. Consequently, MORL researchers face difficulties when trying to classify contributions within a broader context due to the absence of a standardized taxonomy. To tackle such an issue, this paper introduces Multi-Objective Reinforcement Learning based on Decomposition (MORL/D), a novel methodology bridging RL and MOO literature. A comprehensive taxonomy for MORL/D is presented, providing a structured foundation for categorizing existing and potential MORL works. The introduced taxonomy is then used to scrutinize MORL research, enhancing clarity and conciseness through well-defined categorization. Moreover, a flexible framework derived from the taxonomy is introduced. This framework accommodates diverse instantiations using tools from both RL and MOO/D. Implementation across various configurations demonstrates its versatility, assessed against benchmark problems. Results indicate MORL/D instantiations achieve comparable performance with significantly greater versatility than current state-of-the-art approaches. By presenting the taxonomy and framework, this paper offers a comprehensive perspective and a unified vocabulary for MORL. This not only facilitates the identification of algorithmic contributions but also lays the groundwork for novel research avenues in MORL, contributing to the continued advancement of this field.

{{</citation>}}


### (102/135) Harnessing FPGA Technology for Enhanced Biomedical Computation (Nisanur Alici et al., 2023)

{{<citation>}}

Nisanur Alici, Kayode Inadagbo, Murat Isik. (2023)  
**Harnessing FPGA Technology for Enhanced Biomedical Computation**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.12439v1)  

---


**ABSTRACT**  
This research delves into sophisticated neural network frameworks like Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Long Short-Term Memory Networks (LSTMs), and Deep Belief Networks (DBNs) for improved analysis of ECG signals via Field Programmable Gate Arrays (FPGAs). The MIT-BIH Arrhythmia Database serves as the foundation for training and evaluating our models, with added Gaussian noise to heighten the algorithms' resilience. The developed architectures incorporate various layers for specific processing and categorization functions, employing strategies such as the EarlyStopping callback and Dropout layer to prevent overfitting. Additionally, this paper details the creation of a tailored Tensor Compute Unit (TCU) accelerator for the PYNQ Z1 platform. It provides a thorough methodology for implementing FPGA-based machine learning, encompassing the configuration of the Tensil toolchain in Docker, selection of architectures, PS-PL configuration, and the compilation and deployment of models. By evaluating performance indicators like latency and throughput, we showcase the efficacy of FPGAs in advanced biomedical computing. This study ultimately serves as a comprehensive guide to optimizing neural network operations on FPGAs across various fields.

{{</citation>}}


### (103/135) Looped Transformers are Better at Learning Learning Algorithms (Liu Yang et al., 2023)

{{<citation>}}

Liu Yang, Kangwook Lee, Robert Nowak, Dimitris Papailiopoulos. (2023)  
**Looped Transformers are Better at Learning Learning Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.12424v1)  

---


**ABSTRACT**  
Transformers have demonstrated effectiveness in \emph{in-context solving} data-fitting problems from various (latent) models, as reported by Garg et al. However, the absence of an inherent iterative structure in the transformer architecture presents a challenge in emulating the iterative algorithms, which are commonly employed in traditional machine learning methods. To address this, we propose the utilization of \emph{looped} transformer architecture and its associated training methodology, with the aim of incorporating iterative characteristics into the transformer architectures. Experimental results suggest that the looped transformer achieves performance comparable to the standard transformer in solving various data-fitting problems, while utilizing less than 10\% of the parameter count.

{{</citation>}}


### (104/135) A Survey of Graph Meets Large Language Model: Progress and Future Directions (Yuhan Li et al., 2023)

{{<citation>}}

Yuhan Li, Zhixun Li, Peisong Wang, Jia Li, Xiangguo Sun, Hong Cheng, Jeffrey Xu Yu. (2023)  
**A Survey of Graph Meets Large Language Model: Progress and Future Directions**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12399v1)  

---


**ABSTRACT**  
Graph plays a significant role in representing and analyzing complex relationships in real-world applications such as citation networks, social networks, and biological data. Recently, Large Language Models (LLMs), which have achieved tremendous success in various domains, have also been leveraged in graph-related tasks to surpass traditional Graph Neural Networks (GNNs) based methods and yield state-of-the-art performance. In this survey, we first present a comprehensive review and analysis of existing methods that integrate LLMs with graphs. First of all, we propose a new taxonomy, which organizes existing methods into three categories based on the role (i.e., enhancer, predictor, and alignment component) played by LLMs in graph-related tasks. Then we systematically survey the representative methods along the three categories of the taxonomy. Finally, we discuss the remaining limitations of existing studies and highlight promising avenues for future research. The relevant papers are summarized and will be consistently updated at: https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks.

{{</citation>}}


### (105/135) Power grid operational risk assessment using graph neural network surrogates (Yadong Zhang et al., 2023)

{{<citation>}}

Yadong Zhang, Pranav M Karve, Sankaran Mahadevan. (2023)  
**Power grid operational risk assessment using graph neural network surrogates**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.12309v1)  

---


**ABSTRACT**  
We investigate the utility of graph neural networks (GNNs) as proxies of power grid operational decision-making algorithms (optimal power flow (OPF) and security-constrained unit commitment (SCUC)) to enable rigorous quantification of the operational risk. To conduct principled risk analysis, numerous Monte Carlo (MC) samples are drawn from the (foretasted) probability distributions of spatio-temporally correlated stochastic grid variables. The corresponding OPF and SCUC solutions, which are needed to quantify the risk, are generated using traditional OPF and SCUC solvers to generate data for training GNN model(s). The GNN model performance is evaluated in terms of the accuracy of predicting quantities of interests (QoIs) derived from the decision variables in OPF and SCUC. Specifically, we focus on thermal power generation and load shedding at system and individual zone level. We also perform reliability and risk quantification based on GNN predictions and compare with that obtained from OPF/SCUC solutions. Our results demonstrate that GNNs are capable of providing fast and accurate prediction of QoIs and thus can be good surrogate models for OPF and SCUC. The excellent accuracy of GNN-based reliability and risk assessment further suggests that GNN surrogate has the potential to be applied in real-time and hours-ahead risk quantification.

{{</citation>}}


### (106/135) A Supervised Contrastive Learning Pretrain-Finetune Approach for Time Series (Trang H. Tran et al., 2023)

{{<citation>}}

Trang H. Tran, Lam M. Nguyen, Kyongmin Yeo, Nam Nguyen, Roman Vaculin. (2023)  
**A Supervised Contrastive Learning Pretrain-Finetune Approach for Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2311.12290v1)  

---


**ABSTRACT**  
Foundation models have recently gained attention within the field of machine learning thanks to its efficiency in broad data processing. While researchers had attempted to extend this success to time series models, the main challenge is effectively extracting representations and transferring knowledge from pretraining datasets to the target finetuning dataset. To tackle this issue, we introduce a novel pretraining procedure that leverages supervised contrastive learning to distinguish features within each pretraining dataset. This pretraining phase enables a probabilistic similarity metric, which assesses the likelihood of a univariate sample being closely related to one of the pretraining datasets. Subsequently, using this similarity metric as a guide, we propose a fine-tuning procedure designed to enhance the accurate prediction of the target data by aligning it more closely with the learned dynamics of the pretraining datasets. Our experiments have shown promising results which demonstrate the efficacy of our approach.

{{</citation>}}


### (107/135) Exploring Time Granularity on Temporal Graphs for Dynamic Link Prediction in Real-world Networks (Xiangjian Jiang et al., 2023)

{{<citation>}}

Xiangjian Jiang, Yanyi Pu. (2023)  
**Exploring Time Granularity on Temporal Graphs for Dynamic Link Prediction in Real-world Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.12255v2)  

---


**ABSTRACT**  
Dynamic Graph Neural Networks (DGNNs) have emerged as the predominant approach for processing dynamic graph-structured data. However, the influence of temporal information on model performance and robustness remains insufficiently explored, particularly regarding how models address prediction tasks with different time granularities. In this paper, we explore the impact of time granularity when training DGNNs on dynamic graphs through extensive experiments. We examine graphs derived from various domains and compare three different DGNNs to the baseline model across four varied time granularities. We mainly consider the interplay between time granularities, model architectures, and negative sampling strategies to obtain general conclusions. Our results reveal that a sophisticated memory mechanism and proper time granularity are crucial for a DGNN to deliver competitive and robust performance in the dynamic link prediction task. We also discuss drawbacks in considered models and datasets and propose promising directions for future research on the time granularity of temporal graphs.

{{</citation>}}


## cs.CE (1)



### (108/135) Volatility and irregularity Capturing in stock price indices using time series Generative adversarial networks (TimeGAN) (Leonard Mushunje et al., 2023)

{{<citation>}}

Leonard Mushunje, David Allen, Shelton Peiris. (2023)  
**Volatility and irregularity Capturing in stock price indices using time series Generative adversarial networks (TimeGAN)**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.12987v1)  

---


**ABSTRACT**  
This paper captures irregularities in financial time series data, particularly stock prices, in the presence of COVID-19 shock. We conjectured that jumps and irregularities are embedded in stock data due to the pandemic shock, which brings forth irregular trends in the time series data. We put forward that efficient and robust forecasting methods are needed to predict stock closing prices in the presence of the pandemic shock. This piece of information is helpful to investors as far as confidence risk and return boost are concerned. Generative adversarial networks of a time series nature are used to provide new ways of modeling and learning the proper and suitable distribution for the financial time series data under complex setups. Ideally, these traditional models are liable to producing high forecasting errors, and they need to be more robust to capture dependency structures and other stylized facts like volatility in stock markets. The TimeGAN model is used, effectively dealing with this risk of poor forecasts. Using the DAX stock index from January 2010 to November 2022, we trained the LSTM, GRU, WGAN, and TimeGAN models as benchmarks and forecasting errors were noted, and our TimeGAN outperformed them all as indicated by a small forecasting error.

{{</citation>}}


## math.OC (1)



### (109/135) Neural Approximate Dynamic Programming for the Ultra-fast Order Dispatching Problem (Arash Dehghan et al., 2023)

{{<citation>}}

Arash Dehghan, Mucahit Cevik, Merve Bodur. (2023)  
**Neural Approximate Dynamic Programming for the Ultra-fast Order Dispatching Problem**  

---
Primary Category: math.OC  
Categories: cs-AI, math-OC, math.OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.12975v1)  

---


**ABSTRACT**  
Same-Day Delivery (SDD) services aim to maximize the fulfillment of online orders while minimizing delivery delays but are beset by operational uncertainties such as those in order volumes and courier planning. Our work aims to enhance the operational efficiency of SDD by focusing on the ultra-fast Order Dispatching Problem (ODP), which involves matching and dispatching orders to couriers within a centralized warehouse setting, and completing the delivery within a strict timeline (e.g., within minutes). We introduce important extensions to ultra-fast ODP such as order batching and explicit courier assignments to provide a more realistic representation of dispatching operations and improve delivery efficiency. As a solution method, we primarily focus on NeurADP, a methodology that combines Approximate Dynamic Programming (ADP) and Deep Reinforcement Learning (DRL), and our work constitutes the first application of NeurADP outside of the ride-pool matching problem. NeurADP is particularly suitable for ultra-fast ODP as it addresses complex one-to-many matching and routing intricacies through a neural network-based VFA that captures high-dimensional problem dynamics without requiring manual feature engineering as in generic ADP methods. We test our proposed approach using four distinct realistic datasets tailored for ODP and compare the performance of NeurADP against myopic and DRL baselines by also making use of non-trivial bounds to assess the quality of the policies. Our numerical results indicate that the inclusion of order batching and courier queues enhances the efficiency of delivery operations and that NeurADP significantly outperforms other methods. Detailed sensitivity analysis with important parameters confirms the robustness of NeurADP under different scenarios, including variations in courier numbers, spatial setup, vehicle capacity, and permitted delay time.

{{</citation>}}


## cs.NI (2)



### (110/135) DroneOptiNet: A Framework for Optimal Drone-based Load Redistribution Mechanism for 5G and Beyond Solar Small Cell Networks (Daksh Dave et al., 2023)

{{<citation>}}

Daksh Dave, Vinay Chamola, Sandeep Joshi, Sherali Zeadally. (2023)  
**DroneOptiNet: A Framework for Optimal Drone-based Load Redistribution Mechanism for 5G and Beyond Solar Small Cell Networks**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-LG, cs-NE, cs-NI, cs.NI  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2311.12944v1)  

---


**ABSTRACT**  
The power requirements posed by the fifth-generation and beyond cellular networks are an important constraint in network deployment and require energy-efficient solutions. In this work, we propose a novel user load transfer approach using airborne base stations (BS), mounted on drones, for reliable and secure power redistribution across the micro-grid network comprising green small cell BSs. Depending on the user density and the availability of an aerial BS, the energy requirement of a cell with an energy deficit is accommodated by migrating the aerial BS from a high-energy to a low-energy cell. The proposed hybrid drone-based framework integrates long short-term memory with unique cost functions using an evolutionary neural network for drones and BSs, and efficiently manages energy and load redistribution. The proposed algorithm reduces power outages at BSs and maintains consistent throughput stability, thereby demonstrating its capability to boost the reliability and robustness of wireless communication systems.

{{</citation>}}


### (111/135) How AI-driven Digital Twins Can Empower Mobile Networks (Tong Li et al., 2023)

{{<citation>}}

Tong Li, Fenyu Jiang, Qiaohong Yu, Wenzhen Huang, Tao Jiang, Depeng Jin. (2023)  
**How AI-driven Digital Twins Can Empower Mobile Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs-SY, cs.NI, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12273v1)  

---


**ABSTRACT**  
The growing complexity of next-generation networks exacerbates the modeling and algorithmic flaws of conventional network optimization methodology. In this paper, we propose a mobile network digital twin (MNDT) architecture for 6G networks. To address the modeling and algorithmic shortcomings, the MNDT uses a simulation-optimization structure. The feedback from the network simulation engine, which serves as validation for the optimizer's decision outcomes, is used explicitly to train artificial intelligence (AI) empowered optimizers iteratively. In practice, we develop a network digital twin prototype system leveraging data-driven technology to accurately model the behaviors of mobile network elements (e.g., mobile users and base stations), wireless environments, and network performance. An AI-powered network optimizer has been developed based on the deployed MNDT prototype system for providing reliable and optimized network configurations. The results of the experiments demonstrate that the proposed MNDT infrastructure can provide practical network optimization solutions while adapting to the more complex environment.

{{</citation>}}


## cs.RO (2)



### (112/135) InteRACT: Transformer Models for Human Intent Prediction Conditioned on Robot Actions (Kushal Kedia et al., 2023)

{{<citation>}}

Kushal Kedia, Atiksh Bhardwaj, Prithwish Dan, Sanjiban Choudhury. (2023)  
**InteRACT: Transformer Models for Human Intent Prediction Conditioned on Robot Actions**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-MA, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.12943v1)  

---


**ABSTRACT**  
In collaborative human-robot manipulation, a robot must predict human intents and adapt its actions accordingly to smoothly execute tasks. However, the human's intent in turn depends on actions the robot takes, creating a chicken-or-egg problem. Prior methods ignore such inter-dependency and instead train marginal intent prediction models independent of robot actions. This is because training conditional models is hard given a lack of paired human-robot interaction datasets.   Can we instead leverage large-scale human-human interaction data that is more easily accessible? Our key insight is to exploit a correspondence between human and robot actions that enables transfer learning from human-human to human-robot data. We propose a novel architecture, InteRACT, that pre-trains a conditional intent prediction model on large human-human datasets and fine-tunes on a small human-robot dataset. We evaluate on a set of real-world collaborative human-robot manipulation tasks and show that our conditional model improves over various marginal baselines. We also introduce new techniques to tele-operate a 7-DoF robot arm and collect a diverse range of human-robot collaborative manipulation data, which we open-source.

{{</citation>}}


### (113/135) A Safer Vision-based Autonomous Planning System for Quadrotor UAVs with Dynamic Obstacle Trajectory Prediction and Its Application with LLMs (Jiageng Zhong et al., 2023)

{{<citation>}}

Jiageng Zhong, Ming Li, Yinliang Chen, Zihang Wei, Fan Yang, Haoran Shen. (2023)  
**A Safer Vision-based Autonomous Planning System for Quadrotor UAVs with Dynamic Obstacle Trajectory Prediction and Its Application with LLMs**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-RO, cs.RO  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.12893v1)  

---


**ABSTRACT**  
For intelligent quadcopter UAVs, a robust and reliable autonomous planning system is crucial. Most current trajectory planning methods for UAVs are suitable for static environments but struggle to handle dynamic obstacles, which can pose challenges and even dangers to flight. To address this issue, this paper proposes a vision-based planning system that combines tracking and trajectory prediction of dynamic obstacles to achieve efficient and reliable autonomous flight. We use a lightweight object detection algorithm to identify dynamic obstacles and then use Kalman Filtering to track and estimate their motion states. During the planning phase, we not only consider static obstacles but also account for the potential movements of dynamic obstacles. For trajectory generation, we use a B-spline-based trajectory search algorithm, which is further optimized with various constraints to enhance safety and alignment with the UAV's motion characteristics. We conduct experiments in both simulation and real-world environments, and the results indicate that our approach can successfully detect and avoid obstacles in dynamic environments in real-time, offering greater reliability compared to existing approaches. Furthermore, with the advancements in Natural Language Processing (NLP) technology demonstrating exceptional zero-shot generalization capabilities, more user-friendly human-machine interactions have become feasible, and this study also explores the integration of autonomous planning systems with Large Language Models (LLMs).

{{</citation>}}


## cs.SE (2)



### (114/135) Prompting Frameworks for Large Language Models: A Survey (Xiaoxia Liu et al., 2023)

{{<citation>}}

Xiaoxia Liu, Jingyi Wang, Jun Sun, Xiaohan Yuan, Guoliang Dong, Peng Di, Wenhai Wang, Dongxia Wang. (2023)  
**Prompting Frameworks for Large Language Models: A Survey**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12785v1)  

---


**ABSTRACT**  
Since the launch of ChatGPT, a powerful AI Chatbot developed by OpenAI, large language models (LLMs) have made significant advancements in both academia and industry, bringing about a fundamental engineering paradigm shift in many areas. While LLMs are powerful, it is also crucial to best use their power where "prompt'' plays a core role. However, the booming LLMs themselves, including excellent APIs like ChatGPT, have several inherent limitations: 1) temporal lag of training data, and 2) the lack of physical capabilities to perform external actions. Recently, we have observed the trend of utilizing prompt-based tools to better utilize the power of LLMs for downstream tasks, but a lack of systematic literature and standardized terminology, partly due to the rapid evolution of this field. Therefore, in this work, we survey related prompting tools and promote the concept of the "Prompting Framework" (PF), i.e. the framework for managing, simplifying, and facilitating interaction with large language models. We define the lifecycle of the PF as a hierarchical structure, from bottom to top, namely: Data Level, Base Level, Execute Level, and Service Level. We also systematically depict the overall landscape of the emerging PF field and discuss potential future research and challenges. To continuously track the developments in this area, we maintain a repository at https://github.com/lxx0628/Prompting-Framework-Survey, which can be a useful resource sharing platform for both academic and industry in this field.

{{</citation>}}


### (115/135) Pricing4APIs: A Rigorous Model for RESTful API Pricings (Rafael Fresno-Aranda et al., 2023)

{{<citation>}}

Rafael Fresno-Aranda, Pablo Fernandez, Antonio Gamez-Diaz, Amador Duran, Antonio Ruiz-Cortes. (2023)  
**Pricing4APIs: A Rigorous Model for RESTful API Pricings**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12485v1)  

---


**ABSTRACT**  
APIs are increasingly becoming new business assets for organizations and consequently, API functionality and its pricing should be precisely defined for customers. Pricing is typically composed by different plans that specify a range of limitations, e.g., a Free plan allows 100 monthly requests while a Gold plan has 10000 requests per month. In this context, the OpenAPI Specification (OAS) has emerged to model the functional part of an API, becoming a de facto industry standard and boosting a rich ecosystem of vendor-neutral tools to assist API providers and consumers. In contrast, there is no proposal for modeling API pricings (i.e. their plans and limitations) and this lack hinders the creation of tools that can leverage this information. To deal with this gap, this paper presents a pricing modeling framework that includes: (a) Pricing4APIs model, a comprehensive and rigorous model of API pricings, along SLA4OAI, a serialization that extends OAS; (b) an operation to validate the description of API pricings, with a toolset (sla4oai-analyzer) that has been developed to automate this operation. Additionally, we analyzed 268 real-world APIs to assess the expressiveness of our proposal and created a representative dataset of 54 pricing models to validate our framework.

{{</citation>}}


## eess.IV (3)



### (116/135) Swift Parameter-free Attention Network for Efficient Super-Resolution (Cheng Wan et al., 2023)

{{<citation>}}

Cheng Wan, Hongyuan Yu, Zhiqi Li, Yihang Chen, Yajun Zou, Yuqing Liu, Xuanwu Yin, Kunlong Zuo. (2023)  
**Swift Parameter-free Attention Network for Efficient Super-Resolution**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.12770v1)  

---


**ABSTRACT**  
Single Image Super-Resolution (SISR) is a crucial task in low-level computer vision, aiming to reconstruct high-resolution images from low-resolution counterparts. Conventional attention mechanisms have significantly improved SISR performance but often result in complex network structures and large number of parameters, leading to slow inference speed and large model size. To address this issue, we propose the Swift Parameter-free Attention Network (SPAN), a highly efficient SISR model that balances parameter count, inference speed, and image quality. SPAN employs a novel parameter-free attention mechanism, which leverages symmetric activation functions and residual connections to enhance high-contribution information and suppress redundant information. Our theoretical analysis demonstrates the effectiveness of this design in achieving the attention mechanism's purpose. We evaluate SPAN on multiple benchmarks, showing that it outperforms existing efficient super-resolution models in terms of both image quality and inference speed, achieving a significant quality-speed trade-off. This makes SPAN highly suitable for real-world applications, particularly in resource-constrained scenarios. Notably, our model attains the best PSNR of 27.09 dB, and the test runtime of our team is reduced by 7.08ms in the NTIRE 2023 efficient super-resolution challenge. Our code and models are made publicly available at \url{https://github.com/hongyuanyu/SPAN}.

{{</citation>}}


### (117/135) Echocardiogram Foundation Model -- Application 1: Estimating Ejection Fraction (Adil Dahlan et al., 2023)

{{<citation>}}

Adil Dahlan, Cyril Zakka, Abhinav Kumar, Laura Tang, Rohan Shad, Robyn Fong, William Hiesinger. (2023)  
**Echocardiogram Foundation Model -- Application 1: Estimating Ejection Fraction**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12582v1)  

---


**ABSTRACT**  
Cardiovascular diseases stand as the primary global cause of mortality. Among the various imaging techniques available for visualising the heart and evaluating its function, echocardiograms emerge as the preferred choice due to their safety and low cost. Quantifying cardiac function based on echocardiograms is very laborious, time-consuming and subject to high interoperator variability. In this work, we introduce EchoAI, an echocardiogram foundation model, that is trained using self-supervised learning (SSL) on 1.5 million echocardiograms. We evaluate our approach by fine-tuning EchoAI to estimate the ejection fraction achieving a mean absolute percentage error of 9.40%. This level of accuracy aligns with the performance of expert sonographers.

{{</citation>}}


### (118/135) 'HoVer-UNet': Accelerating HoVerNet with UNet-based multi-class nuclei segmentation via knowledge distillation (Cristian Tommasino et al., 2023)

{{<citation>}}

Cristian Tommasino, Cristiano Russo, Antonio Maria Rinaldi, Francesco Ciompi. (2023)  
**'HoVer-UNet': Accelerating HoVerNet with UNet-based multi-class nuclei segmentation via knowledge distillation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.12553v2)  

---


**ABSTRACT**  
We present "HoVer-UNet", an approach to distill the knowledge of the multi-branch HoVerNet framework for nuclei instance segmentation and classification in histopathology. We propose a compact, streamlined single UNet network with a Mix Vision Transformer backbone, and equip it with a custom loss function to optimally encode the distilled knowledge of HoVerNet, reducing computational requirements without compromising performances. We show that our model achieved results comparable to HoVerNet on the public PanNuke and Consep datasets with a three-fold reduction in inference time. We make the code of our model publicly available at https://github.com/DIAGNijmegen/HoVer-UNet.

{{</citation>}}


## cs.CR (3)



### (119/135) High-resolution Image-based Malware Classification using Multiple Instance Learning (Tim Peters et al., 2023)

{{<citation>}}

Tim Peters, Hikmat Farhat. (2023)  
**High-resolution Image-based Malware Classification using Multiple Instance Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2311.12760v1)  

---


**ABSTRACT**  
This paper proposes a novel method of classifying malware into families using high-resolution greyscale images and multiple instance learning to overcome adversarial binary enlargement. Current methods of visualisation-based malware classification largely rely on lossy transformations of inputs such as resizing to handle the large, variable-sized images. Through empirical analysis and experimentation, it is shown that these approaches cause crucial information loss that can be exploited. The proposed solution divides the images into patches and uses embedding-based multiple instance learning with a convolutional neural network and an attention aggregation function for classification. The implementation is evaluated on the Microsoft Malware Classification dataset and achieves accuracies of up to $96.6\%$ on adversarially enlarged samples compared to the baseline of $22.8\%$. The Python code is available online at https://github.com/timppeters/MIL-Malware-Images .

{{</citation>}}


### (120/135) Hyena: Optimizing Homomorphically Encrypted Convolution for Private CNN Inference (Hyeri Roh et al., 2023)

{{<citation>}}

Hyeri Roh, Woo-Seok Choi. (2023)  
**Hyena: Optimizing Homomorphically Encrypted Convolution for Private CNN Inference**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.12519v1)  

---


**ABSTRACT**  
Processing convolution layers remains a huge bottleneck for private deep convolutional neural network (CNN) inference for large datasets. To solve this issue, this paper presents a novel homomorphic convolution algorithm that provides speedup, communication cost, and storage saving. We first note that padded convolution provides the advantage of model storage saving, but it does not support channel packing, thereby increasing the amount of computation and communication. We address this limitation by proposing a novel plaintext multiplication algorithm using the Walsh-Hadamard matrix. Furthermore, we propose the optimization techniques to significantly reduce the latency of the proposed convolution by selecting the optimal encryption parameters and applying lazy reduction. It achieves 1.6-3.8x speedup and reduces the weight storage by 2000-8000x compared to the conventional convolution. When the proposed convolution is employed for CNNs like VGG-16, ResNet-20, and MobileNetV1 on ImageNet, it reduces the end-to-end latency by 1.3-2.6x, the memory usage by 2.1-7.9x and communication cost by 1.7-2.0x compared to conventional method.

{{</citation>}}


### (121/135) Malicious URL Detection via Pretrained Language Model Guided Multi-Level Feature Attention Network (Ruitong Liu et al., 2023)

{{<citation>}}

Ruitong Liu, Yanbin Wang, Haitao Xu, Zhan Qin, Yiwei Liu, Zheng Cao. (2023)  
**Malicious URL Detection via Pretrained Language Model Guided Multi-Level Feature Attention Network**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Attention, BERT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2311.12372v1)  

---


**ABSTRACT**  
The widespread use of the Internet has revolutionized information retrieval methods. However, this transformation has also given rise to a significant cybersecurity challenge: the rapid proliferation of malicious URLs, which serve as entry points for a wide range of cyber threats. In this study, we present an efficient pre-training model-based framework for malicious URL detection. Leveraging the subword and character-aware pre-trained model, CharBERT, as our foundation, we further develop three key modules: hierarchical feature extraction, layer-aware attention, and spatial pyramid pooling. The hierarchical feature extraction module follows the pyramid feature learning principle, extracting multi-level URL embeddings from the different Transformer layers of CharBERT. Subsequently, the layer-aware attention module autonomously learns connections among features at various hierarchical levels and allocates varying weight coefficients to each level of features. Finally, the spatial pyramid pooling module performs multiscale downsampling on the weighted multi-level feature pyramid, achieving the capture of local features as well as the aggregation of global features. The proposed method has been extensively validated on multiple public datasets, demonstrating a significant improvement over prior works, with the maximum accuracy gap reaching 8.43% compared to the previous state-of-the-art method. Additionally, we have assessed the model's generalization and robustness in scenarios such as cross-dataset evaluation and adversarial attacks. Finally, we conducted real-world case studies on the active phishing URLs.

{{</citation>}}


## quant-ph (1)



### (122/135) Tight Lieb-Robinson Bound for approximation ratio in Quantum Annealing (Arthur Braida et al., 2023)

{{<citation>}}

Arthur Braida, Simon Martiel, Ioan Todinca. (2023)  
**Tight Lieb-Robinson Bound for approximation ratio in Quantum Annealing**  

---
Primary Category: quant-ph  
Categories: cs-DS, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.12732v1)  

---


**ABSTRACT**  
Quantum annealing (QA) holds promise for optimization problems in quantum computing, especially for combinatorial optimization. This analog framework attracts attention for its potential to address complex problems. Its gate-based homologous, QAOA with proven performance, has brought lots of attention to the NISQ era. Several numerical benchmarks try to classify these two metaheuristics however, classical computational power highly limits the performance insights. In this work, we introduce a new parametrized version of QA enabling a precise 1-local analysis of the algorithm. We develop a tight Lieb-Robinson bound for regular graphs, achieving the best-known numerical value to analyze QA locally. Studying MaxCut over cubic graph as a benchmark optimization problem, we show that a linear-schedule QA with a 1-local analysis achieves an approximation ratio over 0.7020, outperforming any known 1-local algorithms.

{{</citation>}}


## cs.IT (1)



### (123/135) Reinforcement Learning for the Near-Optimal Design of Zero-Delay Codes for Markov Sources (Liam Cregg et al., 2023)

{{<citation>}}

Liam Cregg, Tamas Linder, Serdar Yuksel. (2023)  
**Reinforcement Learning for the Near-Optimal Design of Zero-Delay Codes for Markov Sources**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.12609v1)  

---


**ABSTRACT**  
In the classical lossy source coding problem, one encodes long blocks of source symbols that enables the distortion to approach the ultimate Shannon limit. Such a block-coding approach introduces large delays, which is undesirable in many delay-sensitive applications. We consider the zero-delay case, where the goal is to encode and decode a finite-alphabet Markov source without any delay. It has been shown that this problem lends itself to stochastic control techniques, which lead to existence, structural, and general structural approximation results. However, these techniques so far have resulted only in computationally prohibitive algorithmic implementations for code design. To address this problem, we present a reinforcement learning design algorithm and rigorously prove its asymptotic optimality. In particular, we show that a quantized Q-learning algorithm can be used to obtain a near-optimal coding policy for this problem. The proof builds on recent results on quantized Q-learning for weakly Feller controlled Markov chains whose application necessitates the development of supporting technical results on regularity and stability properties, and relating the optimal solutions for discounted and average cost infinite horizon criteria problems. These theoretical results are supported by simulations.

{{</citation>}}


## cs.NE (1)



### (124/135) Scheduling Distributed Flexible Assembly Lines using Safe Reinforcement Learning with Soft Shielding (Lele Li et al., 2023)

{{<citation>}}

Lele Li, Liyong Lin. (2023)  
**Scheduling Distributed Flexible Assembly Lines using Safe Reinforcement Learning with Soft Shielding**  

---
Primary Category: cs.NE  
Categories: 68T20, 68T07, I-2-8; F-2-2, cs-AI, cs-NE, cs.NE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.12572v1)  

---


**ABSTRACT**  
Highly automated assembly lines enable significant productivity gains in the manufacturing industry, particularly in mass production condition. Nonetheless, challenges persist in job scheduling for make-to-job and mass customization, necessitating further investigation to improve efficiency, reduce tardiness, promote safety and reliability. In this contribution, an advantage actor-critic based reinforcement learning method is proposed to address scheduling problems of distributed flexible assembly lines in a real-time manner. To enhance the performance, a more condensed environment representation approach is proposed, which is designed to work with the masks made by priority dispatching rules to generate fixed and advantageous action space. Moreover, a Monte-Carlo tree search based soft shielding component is developed to help address long-sequence dependent unsafe behaviors and monitor the risk of overdue scheduling. Finally, the proposed algorithm and its soft shielding component are validated in performance evaluation.

{{</citation>}}


## q-bio.GN (1)



### (125/135) BEND: Benchmarking DNA Language Models on biologically meaningful tasks (Frederikke Isa Marin et al., 2023)

{{<citation>}}

Frederikke Isa Marin, Felix Teufel, Marc Horrender, Dennis Madsen, Dennis Pultz, Ole Winther, Wouter Boomsma. (2023)  
**BEND: Benchmarking DNA Language Models on biologically meaningful tasks**  

---
Primary Category: q-bio.GN  
Categories: cs-LG, q-bio-GN, q-bio.GN  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12570v1)  

---


**ABSTRACT**  
The genome sequence contains the blueprint for governing cellular processes. While the availability of genomes has vastly increased over the last decades, experimental annotation of the various functional, non-coding and regulatory elements encoded in the DNA sequence remains both expensive and challenging. This has sparked interest in unsupervised language modeling of genomic DNA, a paradigm that has seen great success for protein sequence data. Although various DNA language models have been proposed, evaluation tasks often differ between individual works, and might not fully recapitulate the fundamental challenges of genome annotation, including the length, scale and sparsity of the data. In this study, we introduce BEND, a Benchmark for DNA language models, featuring a collection of realistic and biologically meaningful downstream tasks defined on the human genome. We find that embeddings from current DNA LMs can approach performance of expert methods on some tasks, but only capture limited information about long-range features. BEND is available at https://github.com/frederikkemarin/BEND.

{{</citation>}}


## math.NA (1)



### (126/135) The $β$ maps and the strong clustering at the complex unit circle (Alec Schiavoni Piazza et al., 2023)

{{<citation>}}

Alec Schiavoni Piazza, David Meadon, Stefano Serra-Capizzano. (2023)  
**The $β$ maps and the strong clustering at the complex unit circle**  

---
Primary Category: math.NA  
Categories: 15A18, 15B05, 37A30, 37E05, 30B10, cs-NA, math-DS, math-NA, math.NA  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.12568v1)  

---


**ABSTRACT**  
In this work, we study eigenvalue distribution results of a class of highly non-normal matrix-sequences, which can be viewed as a low rank perturbation depending on a parameter $\beta>1$ of the basic Toeplitz matrix-sequence $\{T_n(e^{\mathbf{i}\theta})\}_{n\in\N}$, $\mathbf{i}^2=-1$. The latter has obviously all eigenvalues equal to zero for any matrix order $n$, while for the matrix-sequence under consideration we will show a strong clustering at the range of the generating function $e^{\mathbf{i}\theta}$ i.e. at the complex unit circle. For $\beta\ge 2$ no outliers show up, while for $\beta \in (1,2)$ only two outliers are present, which are both real, positive and have finite limits equal to $\beta-1$ and $(\beta-1)^{-1}$, respectively. The problem looks mathematically innocent, but indeed it is quite challenging since all the sophisticated machinery for deducing the eigenvalue clustering is not easy to apply in the current setting and at most we may hope for weak clustering results. In the derivations, we resort to a trick already used for the spectral analysis of the Google matrix plus several tools from complex analysis. We only mention that the problem is not an academical curiosity and in fact it stems from problems in dynamical systems and number theory. Numerical experiments in high precision are provided and more results are sketched for limit case of $\beta=1$.

{{</citation>}}


## q-bio.QM (1)



### (127/135) From Microbes to Methane: AI-Based Predictive Modeling of Feed Additive Efficacy in Dairy Cows (Yaniv Altshuler et al., 2023)

{{<citation>}}

Yaniv Altshuler, Tzruya Calvao Chebach, Shalom Cohen. (2023)  
**From Microbes to Methane: AI-Based Predictive Modeling of Feed Additive Efficacy in Dairy Cows**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, q-bio-QM, q-bio.QM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12901v1)  

---


**ABSTRACT**  
In an era of increasing pressure to achieve sustainable agriculture, the optimization of livestock feed for enhancing yield and minimizing environmental impact is a paramount objective. This study presents a pioneering approach towards this goal, using rumen microbiome data to predict the efficacy of feed additives in dairy cattle.   We collected an extensive dataset that includes methane emissions from 2,190 Holstein cows distributed across 34 distinct sites. The cows were divided into control and experimental groups in a double-blind, unbiased manner, accounting for variables such as age, days in lactation, and average milk yield. The experimental groups were administered one of four leading commercial feed additives: Agolin, Kexxtone, Allimax, and Relyon. Methane emissions were measured individually both before the administration of additives and over a subsequent 12-week period. To develop our predictive model for additive efficacy, rumen microbiome samples were collected from 510 cows from the same herds prior to the study's onset. These samples underwent deep metagenomic shotgun sequencing, yielding an average of 15.7 million reads per sample. Utilizing innovative artificial intelligence techniques we successfully estimated the efficacy of these feed additives across different farms. The model's robustness was further confirmed through validation with independent cohorts, affirming its generalizability and reliability.   Our results underscore the transformative capability of using targeted feed additive strategies to both optimize dairy yield and milk composition, and to significantly reduce methane emissions. Specifically, our predictive model demonstrates a scenario where its application could guide the assignment of additives to farms where they are most effective. In doing so, we could achieve an average potential reduction of over 27\% in overall emissions.

{{</citation>}}


## eess.AS (1)



### (128/135) HPCNeuroNet: Advancing Neuromorphic Audio Signal Processing with Transformer-Enhanced Spiking Neural Networks (Murat Isik et al., 2023)

{{<citation>}}

Murat Isik, Hiruna Vishwamith, Kayode Inadagbo, I. Can Dikmen. (2023)  
**HPCNeuroNet: Advancing Neuromorphic Audio Signal Processing with Transformer-Enhanced Spiking Neural Networks**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.12449v1)  

---


**ABSTRACT**  
This paper presents a novel approach to neuromorphic audio processing by integrating the strengths of Spiking Neural Networks (SNNs), Transformers, and high-performance computing (HPC) into the HPCNeuroNet architecture. Utilizing the Intel N-DNS dataset, we demonstrate the system's capability to process diverse human vocal recordings across multiple languages and noise backgrounds. The core of our approach lies in the fusion of the temporal dynamics of SNNs with the attention mechanisms of Transformers, enabling the model to capture intricate audio patterns and relationships. Our architecture, HPCNeuroNet, employs the Short-Time Fourier Transform (STFT) for time-frequency representation, Transformer embeddings for dense vector generation, and SNN encoding/decoding mechanisms for spike train conversions. The system's performance is further enhanced by leveraging the computational capabilities of NVIDIA's GeForce RTX 3060 GPU and Intel's Core i9 12900H CPU. Additionally, we introduce a hardware implementation on the Xilinx VU37P HBM FPGA platform, optimizing for energy efficiency and real-time processing. The proposed accelerator achieves a throughput of 71.11 Giga-Operations Per Second (GOP/s) with a 3.55 W on-chip power consumption at 100 MHz. The comparison results with off-the-shelf devices and recent state-of-the-art implementations illustrate that the proposed accelerator has obvious advantages in terms of energy efficiency and design flexibility. Through design-space exploration, we provide insights into optimizing core capacities for audio tasks. Our findings underscore the transformative potential of integrating SNNs, Transformers, and HPC for neuromorphic audio processing, setting a new benchmark for future research and applications.

{{</citation>}}


## cs.IR (3)



### (129/135) Utilizing Language Models for Tour Itinerary Recommendation (Ngai Lam Ho et al., 2023)

{{<citation>}}

Ngai Lam Ho, Kwan Hui Lim. (2023)  
**Utilizing Language Models for Tour Itinerary Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12355v1)  

---


**ABSTRACT**  
Tour itinerary recommendation involves planning a sequence of relevant Point-of-Interest (POIs), which combines challenges from the fields of both Operations Research (OR) and Recommendation Systems (RS). As an OR problem, there is the need to maximize a certain utility (e.g., popularity of POIs in the tour) while adhering to some constraints (e.g., maximum time for the tour). As a RS problem, it is heavily related to problem or filtering or ranking a subset of POIs that are relevant to a user and recommending it as part of an itinerary. In this paper, we explore the use of language models for the task of tour itinerary recommendation and planning. This task has the unique requirement of recommending personalized POIs relevant to users and planning these POIs as an itinerary that satisfies various constraints. We discuss some approaches in this area, such as using word embedding techniques like Word2Vec and GloVe for learning POI embeddings and transformer-based techniques like BERT for generating   itineraries.

{{</citation>}}


### (130/135) A Survey on Large Language Models for Personalized and Explainable Recommendations (Junyi Chen, 2023)

{{<citation>}}

Junyi Chen. (2023)  
**A Survey on Large Language Models for Personalized and Explainable Recommendations**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: AI, GPT, GPT-3.5, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.12338v1)  

---


**ABSTRACT**  
In recent years, Recommender Systems(RS) have witnessed a transformative shift with the advent of Large Language Models(LLMs) in the field of Natural Language Processing(NLP). These models such as OpenAI's GPT-3.5/4, Llama from Meta, have demonstrated unprecedented capabilities in understanding and generating human-like text. This has led to a paradigm shift in the realm of personalized and explainable recommendations, as LLMs offer a versatile toolset for processing vast amounts of textual data to enhance user experiences. To provide a comprehensive understanding of the existing LLM-based recommendation systems, this survey aims to analyze how RS can benefit from LLM-based methodologies. Furthermore, we describe major challenges in Personalized Explanation Generating(PEG) tasks, which are cold-start problems, unfairness and bias problems in RS.

{{</citation>}}


### (131/135) Adapting LLMs for Efficient, Personalized Information Retrieval: Methods and Implications (Samira Ghodratnama et al., 2023)

{{<citation>}}

Samira Ghodratnama, Mehrdad Zakershahrak. (2023)  
**Adapting LLMs for Efficient, Personalized Information Retrieval: Methods and Implications**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Information Retrieval, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12287v1)  

---


**ABSTRACT**  
The advent of Large Language Models (LLMs) heralds a pivotal shift in online user interactions with information. Traditional Information Retrieval (IR) systems primarily relied on query-document matching, whereas LLMs excel in comprehending and generating human-like text, thereby enriching the IR experience significantly. While LLMs are often associated with chatbot functionalities, this paper extends the discussion to their explicit application in information retrieval. We explore methodologies to optimize the retrieval process, select optimal models, and effectively scale and orchestrate LLMs, aiming for cost-efficiency and enhanced result accuracy. A notable challenge, model hallucination-where the model yields inaccurate or misinterpreted data-is addressed alongside other model-specific hurdles. Our discourse extends to crucial considerations including user privacy, data optimization, and the necessity for system clarity and interpretability. Through a comprehensive examination, we unveil not only innovative strategies for integrating Language Models (LLMs) with Information Retrieval (IR) systems, but also the consequential considerations that underline the need for a balanced approach aligned with user-centric principles.

{{</citation>}}


## cs.SI (1)



### (132/135) Modeling Political Orientation of Social Media Posts: An Extended Analysis (Sadia Kamal et al., 2023)

{{<citation>}}

Sadia Kamal, Brenner Little, Jade Gullic, Trevor Harms, Kristin Olofsson, Arunkumar Bagavathi. (2023)  
**Modeling Political Orientation of Social Media Posts: An Extended Analysis**  

---
Primary Category: cs.SI  
Categories: cs-CL, cs-LG, cs-SI, cs.SI  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2311.12323v1)  

---


**ABSTRACT**  
Developing machine learning models to characterize political polarization on online social media presents significant challenges. These challenges mainly stem from various factors such as the lack of annotated data, presence of noise in social media datasets, and the sheer volume of data. The common research practice typically examines the biased structure of online user communities for a given topic or qualitatively measuring the impacts of polarized topics on social media. However, there is limited work focusing on analyzing polarization at the ground-level, specifically in the social media posts themselves. Such existing analysis heavily relies on annotated data, which often requires laborious human labeling, offers labels only to specific problems, and lacks the ability to determine the near-future bias state of a social media conversations. Understanding the degree of political orientation conveyed in social media posts is crucial for quantifying the bias of online user communities and investigating the spread of polarized content. In this work, we first introduce two heuristic methods that leverage on news media bias and post content to label social media posts. Next, we compare the efficacy and quality of heuristically labeled dataset with a randomly sampled human-annotated dataset. Additionally, we demonstrate that current machine learning models can exhibit improved performance in predicting political orientation of social media posts, employing both traditional supervised learning and few-shot learning setups. We conduct experiments using the proposed heuristic methods and machine learning approaches to predict the political orientation of posts collected from two social media forums with diverse political ideologies: Gab and Twitter.

{{</citation>}}


## cs.DB (1)



### (133/135) Demystifying Graph Sparsification Algorithms in Graph Properties Preservation (Yuhan Chen et al., 2023)

{{<citation>}}

Yuhan Chen, Haojie Ye, Sanketh Vedula, Alex Bronstein, Ronald Dreslinski, Trevor Mudge, Nishil Talati. (2023)  
**Demystifying Graph Sparsification Algorithms in Graph Properties Preservation**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.12314v1)  

---


**ABSTRACT**  
Graph sparsification is a technique that approximates a given graph by a sparse graph with a subset of vertices and/or edges. The goal of an effective sparsification algorithm is to maintain specific graph properties relevant to the downstream task while minimizing the graph's size. Graph algorithms often suffer from long execution time due to the irregularity and the large real-world graph size. Graph sparsification can be applied to greatly reduce the run time of graph algorithms by substituting the full graph with a much smaller sparsified graph, without significantly degrading the output quality. However, the interaction between numerous sparsifiers and graph properties is not widely explored, and the potential of graph sparsification is not fully understood.   In this work, we cover 16 widely-used graph metrics, 12 representative graph sparsification algorithms, and 14 real-world input graphs spanning various categories, exhibiting diverse characteristics, sizes, and densities. We developed a framework to extensively assess the performance of these sparsification algorithms against graph metrics, and provide insights to the results. Our study shows that there is no one sparsifier that performs the best in preserving all graph properties, e.g. sparsifiers that preserve distance-related graph properties (eccentricity) struggle to perform well on Graph Neural Networks (GNN). This paper presents a comprehensive experimental study evaluating the performance of sparsification algorithms in preserving essential graph metrics. The insights inform future research in incorporating matching graph sparsification to graph algorithms to maximize benefits while minimizing quality degradation. Furthermore, we provide a framework to facilitate the future evaluation of evolving sparsification algorithms, graph metrics, and ever-growing graph data.

{{</citation>}}


## cond-mat.stat-mech (1)



### (134/135) Detecting subtle macroscopic changes in a finite temperature classical scalar field with machine learning (Jiming Yang et al., 2023)

{{<citation>}}

Jiming Yang, Yutong Zheng, Jiahong Zhou, Huiyu Li, Jun Yin. (2023)  
**Detecting subtle macroscopic changes in a finite temperature classical scalar field with machine learning**  

---
Primary Category: cond-mat.stat-mech  
Categories: cond-mat-stat-mech, cond-mat.stat-mech, cs-AI, cs-LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12303v1)  

---


**ABSTRACT**  
The ability to detect macroscopic changes is important for probing the behaviors of experimental many-body systems from the classical to the quantum realm. Although abrupt changes near phase boundaries can easily be detected, subtle macroscopic changes are much more difficult to detect as the changes can be obscured by noise. In this study, as a toy model for detecting subtle macroscopic changes in many-body systems, we try to differentiate scalar field samples at varying temperatures. We compare different methods for making such differentiations, from physics method, statistics method, to AI method. Our finding suggests that the AI method outperforms both the statistical method and the physics method in its sensitivity. Our result provides a proof-of-concept that AI can potentially detect macroscopic changes in many-body systems that elude physical measures.

{{</citation>}}


## eess.SY (1)



### (135/135) Resilient Control of Networked Microgrids using Vertical Federated Reinforcement Learning: Designs and Real-Time Test-Bed Validations (Sayak Mukherjee et al., 2023)

{{<citation>}}

Sayak Mukherjee, Ramij R. Hossain, Sheik M. Mohiuddin, Yuan Liu, Wei Du, Veronica Adetola, Rohit A. Jinsiwale, Qiuhua Huang, Tianzhixi Yin, Ankit Singhal. (2023)  
**Resilient Control of Networked Microgrids using Vertical Federated Reinforcement Learning: Designs and Real-Time Test-Bed Validations**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.12264v1)  

---


**ABSTRACT**  
Improving system-level resiliency of networked microgrids is an important aspect with increased population of inverter-based resources (IBRs). This paper (1) presents resilient control design in presence of adversarial cyber-events, and proposes a novel federated reinforcement learning (Fed-RL) approach to tackle (a) model complexities, unknown dynamical behaviors of IBR devices, (b) privacy issues regarding data sharing in multi-party-owned networked grids, and (2) transfers learned controls from simulation to hardware-in-the-loop test-bed, thereby bridging the gap between simulation and real world. With these multi-prong objectives, first, we formulate a reinforcement learning (RL) training setup generating episodic trajectories with adversaries (attack signal) injected at the primary controllers of the grid forming (GFM) inverters where RL agents (or controllers) are being trained to mitigate the injected attacks. For networked microgrids, the horizontal Fed-RL method involving distinct independent environments is not appropriate, leading us to develop vertical variant Federated Soft Actor-Critic (FedSAC) algorithm to grasp the interconnected dynamics of networked microgrid. Next, utilizing OpenAI Gym interface, we built a custom simulation set-up in GridLAB-D/HELICS co-simulation platform, named Resilient RL Co-simulation (ResRLCoSIM), to train the RL agents with IEEE 123-bus benchmark test systems comprising 3 interconnected microgrids. Finally, the learned policies in simulation world are transferred to the real-time hardware-in-the-loop test-bed set-up developed using high-fidelity Hypersim platform. Experiments show that the simulator-trained RL controllers produce convincing results with the real-time test-bed set-up, validating the minimization of sim-to-real gap.

{{</citation>}}
