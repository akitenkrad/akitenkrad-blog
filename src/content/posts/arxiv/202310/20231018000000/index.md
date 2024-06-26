---
draft: false
title: "arXiv @ 2023.10.18"
date: 2023-10-18
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.18"
    identifier: arxiv_20231018
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [q-bio.QM (1)](#q-bioqm-1)
- [cs.CL (65)](#cscl-65)
- [cs.RO (9)](#csro-9)
- [cs.CV (33)](#cscv-33)
- [eess.SY (1)](#eesssy-1)
- [cs.LG (32)](#cslg-32)
- [cs.GT (1)](#csgt-1)
- [cs.IR (3)](#csir-3)
- [eess.IV (6)](#eessiv-6)
- [cs.PL (1)](#cspl-1)
- [eess.AS (2)](#eessas-2)
- [cs.CR (3)](#cscr-3)
- [cs.AI (3)](#csai-3)
- [cs.NI (2)](#csni-2)
- [q-bio.NC (1)](#q-bionc-1)
- [q-fin.TR (1)](#q-fintr-1)
- [cs.SD (3)](#cssd-3)
- [stat.ML (4)](#statml-4)
- [cs.CY (1)](#cscy-1)
- [cs.DB (1)](#csdb-1)
- [cs.NE (1)](#csne-1)
- [cs.DC (1)](#csdc-1)
- [cs.SI (1)](#cssi-1)

## q-bio.QM (1)



### (1/176) Active Learning Framework for Cost-Effective TCR-Epitope Binding Affinity Prediction (Pengfei Zhang et al., 2023)

{{<citation>}}

Pengfei Zhang, Seojin Bang, Heewook Lee. (2023)  
**Active Learning Framework for Cost-Effective TCR-Epitope Binding Affinity Prediction**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, q-bio-QM, q-bio.QM  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2310.10893v1)  

---


**ABSTRACT**  
T cell receptors (TCRs) are critical components of adaptive immune systems, responsible for responding to threats by recognizing epitope sequences presented on host cell surface. Computational prediction of binding affinity between TCRs and epitope sequences using machine/deep learning has attracted intense attention recently. However, its success is hindered by the lack of large collections of annotated TCR-epitope pairs. Annotating their binding affinity requires expensive and time-consuming wet-lab evaluation. To reduce annotation cost, we present ActiveTCR, a framework that incorporates active learning and TCR-epitope binding affinity prediction models. Starting with a small set of labeled training pairs, ActiveTCR iteratively searches for unlabeled TCR-epitope pairs that are ''worth'' for annotation. It aims to maximize performance gains while minimizing the cost of annotation. We compared four query strategies with a random sampling baseline and demonstrated that ActiveTCR reduces annotation costs by approximately 40%. Furthermore, we showed that providing ground truth labels of TCR-epitope pairs to query strategies can help identify and reduce more than 40% redundancy among already annotated pairs without compromising model performance, enabling users to train equally powerful prediction models with less training data. Our work is the first systematic investigation of data optimization for TCR-epitope binding affinity prediction.

{{</citation>}}


## cs.CL (65)



### (2/176) IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models (Shaokun Zhang et al., 2023)

{{<citation>}}

Shaokun Zhang, Xiaobo Xia, Zhaoqing Wang, Ling-Hao Chen, Jiale Liu, Qingyun Wu, Tongliang Liu. (2023)  
**IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10873v1)  

---


**ABSTRACT**  
In-context learning is a promising paradigm that utilizes in-context examples as prompts for the predictions of large language models. These prompts are crucial for achieving strong performance. However, since the prompts need to be sampled from a large volume of annotated examples, finding the right prompt may result in high annotation costs. To address this challenge, this paper introduces an influence-driven selective annotation method that aims to minimize annotation costs while improving the quality of in-context examples. The essence of our method is to select a pivotal subset from a large-scale unlabeled data pool to annotate for the subsequent sampling of prompts. Specifically, a directed graph is first constructed to represent unlabeled data. Afterward, the influence of candidate unlabeled subsets is quantified with a diffusion process. A simple yet effective greedy algorithm for unlabeled data selection is lastly introduced. It iteratively selects the data if it provides a maximum marginal gain with respect to quantified influence. Compared with previous efforts on selective annotations, our influence-driven method works in an end-to-end manner, avoids an intractable explicit balance between data diversity and representativeness, and enjoys theoretical support. Experiments confirm the superiority of the proposed method on various benchmarks, achieving better performance under lower time consumption during subset selection. The project page is available at https://skzhang1.github.io/IDEAL/.

{{</citation>}}


### (3/176) Will the Prince Get True Love's Kiss? On the Model Sensitivity to Gender Perturbation over Fairytale Texts (Christina Chance et al., 2023)

{{<citation>}}

Christina Chance, Da Yin, Dakuo Wang, Kai-Wei Chang. (2023)  
**Will the Prince Get True Love's Kiss? On the Model Sensitivity to Gender Perturbation over Fairytale Texts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.10865v1)  

---


**ABSTRACT**  
Recent studies show that traditional fairytales are rife with harmful gender biases. To help mitigate these gender biases in fairytales, this work aims to assess learned biases of language models by evaluating their robustness against gender perturbations. Specifically, we focus on Question Answering (QA) tasks in fairytales. Using counterfactual data augmentation to the FairytaleQA dataset, we evaluate model robustness against swapped gender character information, and then mitigate learned biases by introducing counterfactual gender stereotypes during training time. We additionally introduce a novel approach that utilizes the massive vocabulary of language models to support text genres beyond fairytales. Our experimental results suggest that models are sensitive to gender perturbations, with significant performance drops compared to the original testing set. However, when first fine-tuned on a counterfactual training dataset, models are less sensitive to the later introduced anti-gender stereotyped text.

{{</citation>}}


### (4/176) CoTFormer: More Tokens With Attention Make Up For Less Depth (Amirkeivan Mohtashami et al., 2023)

{{<citation>}}

Amirkeivan Mohtashami, Matteo Pagliardini, Martin Jaggi. (2023)  
**CoTFormer: More Tokens With Attention Make Up For Less Depth**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.10845v1)  

---


**ABSTRACT**  
The race to continually develop ever larger and deeper foundational models is underway. However, techniques like the Chain-of-Thought (CoT) method continue to play a pivotal role in achieving optimal downstream performance. In this work, we establish an approximate parallel between using chain-of-thought and employing a deeper transformer. Building on this insight, we introduce CoTFormer, a transformer variant that employs an implicit CoT-like mechanism to achieve capacity comparable to a deeper model. Our empirical findings demonstrate the effectiveness of CoTFormers, as they significantly outperform larger standard transformers.

{{</citation>}}


### (5/176) Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks (Erfan Shayegani et al., 2023)

{{<citation>}}

Erfan Shayegani, Md Abdullah Al Mamun, Yu Fu, Pedram Zaree, Yue Dong, Nael Abu-Ghazaleh. (2023)  
**Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: AI, Adversarial Attack, ChatGPT, GPT, Language Model, Natural Language Processing, Security  
[Paper Link](http://arxiv.org/abs/2310.10844v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are swiftly advancing in architecture and capability, and as they integrate more deeply into complex systems, the urgency to scrutinize their security properties grows. This paper surveys research in the emerging interdisciplinary field of adversarial attacks on LLMs, a subfield of trustworthy ML, combining the perspectives of Natural Language Processing and Security. Prior work has shown that even safety-aligned LLMs (via instruction tuning and reinforcement learning through human feedback) can be susceptible to adversarial attacks, which exploit weaknesses and mislead AI systems, as evidenced by the prevalence of `jailbreak' attacks on models like ChatGPT and Bard. In this survey, we first provide an overview of large language models, describe their safety alignment, and categorize existing research based on various learning structures: textual-only attacks, multi-modal attacks, and additional attack methods specifically targeting complex systems, such as federated learning or multi-agent systems. We also offer comprehensive remarks on works that focus on the fundamental sources of vulnerabilities and potential defenses. To make this field more accessible to newcomers, we present a systematic review of existing works, a structured typology of adversarial attack concepts, and additional resources, including slides for presentations on related topics at the 62nd Annual Meeting of the Association for Computational Linguistics (ACL'24).

{{</citation>}}


### (6/176) Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks (Jiaying Wu et al., 2023)

{{<citation>}}

Jiaying Wu, Bryan Hooi. (2023)  
**Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Fake News, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10830v1)  

---


**ABSTRACT**  
It is commonly perceived that online fake news and reliable news exhibit stark differences in writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the rise of powerful Large Language Models (LLMs) has enabled malicious users to mimic the style of trustworthy news outlets at minimal cost. Our analysis reveals that LLM-camouflaged fake news content leads to substantial performance degradation of state-of-the-art text-based detectors (up to 38% decrease in F1 Score), posing a significant challenge for automated detection in online ecosystems. To address this, we introduce SheepDog, a style-agnostic fake news detector robust to news writing styles. SheepDog achieves this adaptability through LLM-empowered news reframing, which customizes each article to match different writing styles using style-oriented reframing prompts. By employing style-agnostic training, SheepDog enhances its resilience to stylistic variations by maximizing prediction consistency across these diverse reframings. Furthermore, SheepDog extracts content-focused veracity attributions from LLMs, where the news content is evaluated against a set of fact-checking rationales. These attributions provide supplementary information and potential interpretability that assist veracity prediction. On three benchmark datasets, empirical results show that SheepDog consistently yields significant improvements over competitive baselines and enhances robustness against LLM-empowered style attacks.

{{</citation>}}


### (7/176) SD-HuBERT: Self-Distillation Induces Syllabic Organization in HuBERT (Cheol Jun Cho et al., 2023)

{{<citation>}}

Cheol Jun Cho, Abdelrahman Mohamed, Shang-Wen Li, Alan W Black, Gopala K. Anumanchipalli. (2023)  
**SD-HuBERT: Self-Distillation Induces Syllabic Organization in HuBERT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.10803v1)  

---


**ABSTRACT**  
Data-driven unit discovery in self-supervised learning (SSL) of speech has embarked on a new era of spoken language processing. Yet, the discovered units often remain in phonetic space, limiting the utility of SSL representations. Here, we demonstrate that a syllabic organization emerges in learning sentence-level representation of speech. In particular, we adopt "self-distillation" objective to fine-tune the pretrained HuBERT with an aggregator token that summarizes the entire sentence. Without any supervision, the resulting model draws definite boundaries in speech, and the representations across frames show salient syllabic structures. We demonstrate that this emergent structure largely corresponds to the ground truth syllables. Furthermore, we propose a new benchmark task, Spoken Speech ABX, for evaluating sentence-level representation of speech. When compared to previous models, our model outperforms in both unsupervised syllable discovery and learning sentence-level representation. Together, we demonstrate that the self-distillation of HuBERT gives rise to syllabic organization without relying on external labels or modalities, and potentially provides novel data-driven units for spoken language modeling.

{{</citation>}}


### (8/176) BanglaNLP at BLP-2023 Task 1: Benchmarking different Transformer Models for Violence Inciting Text Detection in Bengali (Saumajit Saha et al., 2023)

{{<citation>}}

Saumajit Saha, Albert Nanda. (2023)  
**BanglaNLP at BLP-2023 Task 1: Benchmarking different Transformer Models for Violence Inciting Text Detection in Bengali**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2310.10781v1)  

---


**ABSTRACT**  
This paper presents the system that we have developed while solving this shared task on violence inciting text detection in Bangla. We explain both the traditional and the recent approaches that we have used to make our models learn. Our proposed system helps to classify if the given text contains any threat. We studied the impact of data augmentation when there is a limited dataset available. Our quantitative results show that finetuning a multilingual-e5-base model performed the best in our task compared to other transformer-based architectures. We obtained a macro F1 of 68.11\% in the test set and our performance in this shared task is ranked at 23 in the leaderboard.

{{</citation>}}


### (9/176) Towards reducing hallucination in extracting information from financial reports using Large Language Models (Bhaskarjit Sarmah et al., 2023)

{{<citation>}}

Bhaskarjit Sarmah, Tianjie Zhu, Dhagash Mehta, Stefano Pasquali. (2023)  
**Towards reducing hallucination in extracting information from financial reports using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, q-fin-PM, q-fin-ST, stat-AP  
Keywords: Language Model, OCR  
[Paper Link](http://arxiv.org/abs/2310.10760v1)  

---


**ABSTRACT**  
For a financial analyst, the question and answer (Q\&A) segment of the company financial report is a crucial piece of information for various analysis and investment decisions. However, extracting valuable insights from the Q\&A section has posed considerable challenges as the conventional methods such as detailed reading and note-taking lack scalability and are susceptible to human errors, and Optical Character Recognition (OCR) and similar techniques encounter difficulties in accurately processing unstructured transcript text, often missing subtle linguistic nuances that drive investor decisions. Here, we demonstrate the utilization of Large Language Models (LLMs) to efficiently and rapidly extract information from earnings report transcripts while ensuring high accuracy transforming the extraction process as well as reducing hallucination by combining retrieval-augmented generation technique as well as metadata. We evaluate the outcomes of various LLMs with and without using our proposed approach based on various objective metrics for evaluating Q\&A systems, and empirically demonstrate superiority of our method.

{{</citation>}}


### (10/176) Building Persona Consistent Dialogue Agents with Offline Reinforcement Learning (Ryan Shea et al., 2023)

{{<citation>}}

Ryan Shea, Zhou Yu. (2023)  
**Building Persona Consistent Dialogue Agents with Offline Reinforcement Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10735v1)  

---


**ABSTRACT**  
Maintaining a consistent persona is a key quality for any open domain dialogue system. Current state-of-the-art systems do this by training agents with supervised learning or online reinforcement learning (RL). However, systems trained with supervised learning often lack consistency as they are never punished for uttering contradictions. Additional training with RL can alleviate some of these issues, however the training process is expensive. Instead, we propose an offline RL framework to improve the persona consistency of dialogue systems. Our framework allows us to combine the advantages of previous methods as we can inexpensively train our model on existing data as in supervised learning, while punishing and rewarding specific utterances as in RL. We also introduce a simple importance sampling method to reduce the variance of importance weights in offline RL training which we call Variance-Reducing MLE-Initialized (VaRMI) importance sampling. Our automatic and human evaluations show that our framework improves both the persona consistency and dialogue quality of a state-of-the-art social chatbot.

{{</citation>}}


### (11/176) In-Context Pretraining: Language Modeling Beyond Document Boundaries (Weijia Shi et al., 2023)

{{<citation>}}

Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Victoria Lin, Noah A. Smith, Luke Zettlemoyer, Scott Yih, Mike Lewis. (2023)  
**In-Context Pretraining: Language Modeling Beyond Document Boundaries**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10638v2)  

---


**ABSTRACT**  
Large language models (LMs) are currently trained to predict tokens given document prefixes, enabling them to directly perform long-form generation and prompting-style tasks which can be reduced to document completion. Existing pretraining pipelines train LMs by concatenating random sets of short documents to create input contexts but the prior documents provide no signal for predicting the next document. We instead present In-Context Pretraining, a new approach where language models are pretrained on a sequence of related documents, thereby explicitly encouraging them to read and reason across document boundaries. We can do In-Context Pretraining by simply changing the document ordering so that each context contains related documents, and directly applying existing pretraining pipelines. However, this document sorting problem is challenging. There are billions of documents and we would like the sort to maximize contextual similarity for every document without repeating any data. To do this, we introduce approximate algorithms for finding related documents with efficient nearest neighbor search and constructing coherent input contexts with a graph traversal algorithm. Our experiments show In-Context Pretraining offers a simple and scalable approach to significantly enhance LMs'performance: we see notable improvements in tasks that require more complex contextual reasoning, including in-context learning (+8%), reading comprehension (+15%), faithfulness to previous contexts (+16%), long-context reasoning (+5%), and retrieval augmentation (+9%).

{{</citation>}}


### (12/176) BioPlanner: Automatic Evaluation of LLMs on Protocol Planning in Biology (Odhran O'Donoghue et al., 2023)

{{<citation>}}

Odhran O'Donoghue, Aleksandar Shtedritski, John Ginger, Ralph Abboud, Ali Essa Ghareeb, Justin Booth, Samuel G Rodriques. (2023)  
**BioPlanner: Automatic Evaluation of LLMs on Protocol Planning in Biology**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-RO, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10632v1)  

---


**ABSTRACT**  
The ability to automatically generate accurate protocols for scientific experiments would represent a major step towards the automation of science. Large Language Models (LLMs) have impressive capabilities on a wide range of tasks, such as question answering and the generation of coherent text and code. However, LLMs can struggle with multi-step problems and long-term planning, which are crucial for designing scientific experiments. Moreover, evaluation of the accuracy of scientific protocols is challenging, because experiments can be described correctly in many different ways, require expert knowledge to evaluate, and cannot usually be executed automatically. Here we present an automatic evaluation framework for the task of planning experimental protocols, and we introduce BioProt: a dataset of biology protocols with corresponding pseudocode representations. To measure performance on generating scientific protocols, we use an LLM to convert a natural language protocol into pseudocode, and then evaluate an LLM's ability to reconstruct the pseudocode from a high-level description and a list of admissible pseudocode functions. We evaluate GPT-3 and GPT-4 on this task and explore their robustness. We externally validate the utility of pseudocode representations of text by generating accurate novel protocols using retrieved pseudocode, and we run a generated protocol successfully in our biological laboratory. Our framework is extensible to the evaluation and improvement of language model planning abilities in other areas of science or other areas that lack automatic evaluation.

{{</citation>}}


### (13/176) Llemma: An Open Language Model For Mathematics (Zhangir Azerbayev et al., 2023)

{{<citation>}}

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, Sean Welleck. (2023)  
**Llemma: An Open Language Model For Mathematics**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LO, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10631v1)  

---


**ABSTRACT**  
We present Llemma, a large language model for mathematics. We continue pretraining Code Llama on the Proof-Pile-2, a mixture of scientific papers, web data containing mathematics, and mathematical code, yielding Llemma. On the MATH benchmark Llemma outperforms all known open base models, as well as the unreleased Minerva model suite on an equi-parameter basis. Moreover, Llemma is capable of tool use and formal theorem proving without any further finetuning. We openly release all artifacts, including 7 billion and 34 billion parameter models, the Proof-Pile-2, and code to replicate our experiments.

{{</citation>}}


### (14/176) Data Contamination Through the Lens of Time (Manley Roberts et al., 2023)

{{<citation>}}

Manley Roberts, Himanshu Thakur, Christine Herlihy, Colin White, Samuel Dooley. (2023)  
**Data Contamination Through the Lens of Time**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.10628v1)  

---


**ABSTRACT**  
Recent claims about the impressive abilities of large language models (LLMs) are often supported by evaluating publicly available benchmarks. Since LLMs train on wide swaths of the internet, this practice raises concerns of data contamination, i.e., evaluating on examples that are explicitly or implicitly included in the training data. Data contamination remains notoriously challenging to measure and mitigate, even with partial attempts like controlled experimentation of training data, canary strings, or embedding similarities. In this work, we conduct the first thorough longitudinal analysis of data contamination in LLMs by using the natural experiment of training cutoffs in GPT models to look at benchmarks released over time. Specifically, we consider two code/mathematical problem-solving datasets, Codeforces and Project Euler, and find statistically significant trends among LLM pass rate vs. GitHub popularity and release date that provide strong evidence of contamination. By open-sourcing our dataset, raw results, and evaluation framework, our work paves the way for rigorous analyses of data contamination in modern models. We conclude with a discussion of best practices and future steps for publicly releasing benchmarks in the age of LLMs that train on webscale data.

{{</citation>}}


### (15/176) Factored Verification: Detecting and Reducing Hallucination in Summaries of Academic Papers (Charlie George et al., 2023)

{{<citation>}}

Charlie George, Andreas Stuhlmüller. (2023)  
**Factored Verification: Detecting and Reducing Hallucination in Summaries of Academic Papers**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.10627v1)  

---


**ABSTRACT**  
Hallucination plagues even frontier LLMs--but how bad is it really for summarizing academic papers? We evaluate Factored Verification, a simple automated method for detecting hallucinations in abstractive summaries. This method sets a new SotA on hallucination detection in the summarization task of the HaluEval benchmark, achieving 76.2% accuracy. We then use this method to estimate how often language models hallucinate when summarizing across multiple academic papers and find 0.62 hallucinations in the average ChatGPT (16k) summary, 0.84 for GPT-4, and 1.55 for Claude 2. We ask models to self-correct using Factored Critiques and find that this lowers the number of hallucinations to 0.49 for ChatGPT, 0.46 for GPT-4, and 0.95 for Claude 2. The hallucinations we find are often subtle, so we advise caution when using models to synthesize academic papers.

{{</citation>}}


### (16/176) Mastering the Task of Open Information Extraction with Large Language Models and Consistent Reasoning Environment (Ji Qi et al., 2023)

{{<citation>}}

Ji Qi, Kaixuan Ji, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Lei Hou, Juanzi Li, Bin Xu. (2023)  
**Mastering the Task of Open Information Extraction with Large Language Models and Consistent Reasoning Environment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.10590v1)  

---


**ABSTRACT**  
Open Information Extraction (OIE) aims to extract objective structured knowledge from natural texts, which has attracted growing attention to build dedicated models with human experience. As the large language models (LLMs) have exhibited remarkable in-context learning capabilities, a question arises as to whether the task of OIE can be effectively tackled with this paradigm? In this paper, we explore solving the OIE problem by constructing an appropriate reasoning environment for LLMs. Specifically, we first propose a method to effectively estimate the discrepancy of syntactic distribution between a LLM and test samples, which can serve as correlation evidence for preparing positive demonstrations. Upon the evidence, we introduce a simple yet effective mechanism to establish the reasoning environment for LLMs on specific tasks. Without bells and whistles, experimental results on the standard CaRB benchmark demonstrate that our $6$-shot approach outperforms state-of-the-art supervised method, achieving an $55.3$ $F_1$ score. Further experiments on TACRED and ACE05 show that our method can naturally generalize to other information extraction tasks, resulting in improvements of $5.7$ and $6.8$ $F_1$ scores, respectively.

{{</citation>}}


### (17/176) Who Are All The Stochastic Parrots Imitating? They Should Tell Us! (Sagi Shaier et al., 2023)

{{<citation>}}

Sagi Shaier, Lawrence E. Hunter, Katharina von der Wense. (2023)  
**Who Are All The Stochastic Parrots Imitating? They Should Tell Us!**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.10583v1)  

---


**ABSTRACT**  
Both standalone language models (LMs) as well as LMs within downstream-task systems have been shown to generate statements which are factually untrue. This problem is especially severe for low-resource languages, where training data is scarce and of worse quality than for high-resource languages. In this opinion piece, we argue that LMs in their current state will never be fully trustworthy in critical settings and suggest a possible novel strategy to handle this issue: by building LMs such that can cite their sources - i.e., point a user to the parts of their training data that back up their outputs. We first discuss which current NLP tasks would or would not benefit from such models. We then highlight the expected benefits such models would bring, e.g., quick verifiability of statements. We end by outlining the individual tasks that would need to be solved on the way to developing LMs with the ability to cite. We hope to start a discussion about the field's current approach to building LMs, especially for low-resource languages, and the role of the training data in explaining model generations.

{{</citation>}}


### (18/176) Emerging Challenges in Personalized Medicine: Assessing Demographic Effects on Biomedical Question Answering Systems (Sagi Shaier et al., 2023)

{{<citation>}}

Sagi Shaier, Kevin Bennett, Lawrence Hunter, Katharina von der Wense. (2023)  
**Emerging Challenges in Personalized Medicine: Assessing Demographic Effects on Biomedical Question Answering Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.10571v1)  

---


**ABSTRACT**  
State-of-the-art question answering (QA) models exhibit a variety of social biases (e.g., with respect to sex or race), generally explained by similar issues in their training data. However, what has been overlooked so far is that in the critical domain of biomedicine, any unjustified change in model output due to patient demographics is problematic: it results in the unfair treatment of patients. Selecting only questions on biomedical topics whose answers do not depend on ethnicity, sex, or sexual orientation, we ask the following research questions: (RQ1) Do the answers of QA models change when being provided with irrelevant demographic information? (RQ2) Does the answer of RQ1 differ between knowledge graph (KG)-grounded and text-based QA systems? We find that irrelevant demographic information change up to 15% of the answers of a KG-grounded system and up to 23% of the answers of a text-based system, including changes that affect accuracy. We conclude that unjustified answer changes caused by patient demographics are a frequent phenomenon, which raises fairness concerns and should be paid more attention to.

{{</citation>}}


### (19/176) On Position Bias in Summarization with Large Language Models (Mathieu Ravaut et al., 2023)

{{<citation>}}

Mathieu Ravaut, Shafiq Joty, Aixin Sun, Nancy F. Chen. (2023)  
**On Position Bias in Summarization with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2310.10570v1)  

---


**ABSTRACT**  
Large language models (LLMs) excel in zero-shot abstractive summarization tasks, delivering fluent and pertinent summaries. Recent advancements have extended their capabilities to handle long-input contexts, surpassing token limits of 32k or more. However, in the realm of multi-document question answering, language models exhibit uneven utilization of their input context. They tend to favor the initial and final segments, resulting in a U-shaped performance pattern concerning where the answer is located within the input. This bias raises concerns, particularly in summarization tasks where crucial content may be dispersed throughout the source document(s). This paper presents a comprehensive investigation encompassing 10 datasets, 4 LLMs, and 5 evaluation metrics to analyze how these models leverage their input for abstractive summarization. Our findings reveal a pronounced bias towards the introductory content (and to a lesser extent, the final content), posing challenges for LLM performance across a range of diverse summarization benchmarks.

{{</citation>}}


### (20/176) RegaVAE: A Retrieval-Augmented Gaussian Mixture Variational Auto-Encoder for Language Modeling (Jingcheng Deng et al., 2023)

{{<citation>}}

Jingcheng Deng, Liang Pang, Huawei Shen, Xueqi Cheng. (2023)  
**RegaVAE: A Retrieval-Augmented Gaussian Mixture Variational Auto-Encoder for Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10567v1)  

---


**ABSTRACT**  
Retrieval-augmented language models show promise in addressing issues like outdated information and hallucinations in language models (LMs). However, current research faces two main problems: 1) determining what information to retrieve, and 2) effectively combining retrieved information during generation. We argue that valuable retrieved information should not only be related to the current source text but also consider the future target text, given the nature of LMs that model future tokens. Moreover, we propose that aggregation using latent variables derived from a compact latent space is more efficient than utilizing explicit raw text, which is limited by context length and susceptible to noise. Therefore, we introduce RegaVAE, a retrieval-augmented language model built upon the variational auto-encoder (VAE). It encodes the text corpus into a latent space, capturing current and future information from both source and target text. Additionally, we leverage the VAE to initialize the latent space and adopt the probabilistic form of the retrieval generation paradigm by expanding the Gaussian prior distribution into a Gaussian mixture distribution. Theoretical analysis provides an optimizable upper bound for RegaVAE. Experimental results on various datasets demonstrate significant improvements in text generation quality and hallucination removal.

{{</citation>}}


### (21/176) ViPE: Visualise Pretty-much Everything (Hassan Shahmohammadi et al., 2023)

{{<citation>}}

Hassan Shahmohammadi, Adhiraj Ghosh, Hendrik P. A. Lensch. (2023)  
**ViPE: Visualise Pretty-much Everything**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.10543v1)  

---


**ABSTRACT**  
Figurative and non-literal expressions are profoundly integrated in human communication. Visualising such expressions allow us to convey our creative thoughts, and evoke nuanced emotions. Recent text-to-image models like Stable Diffusion, on the other hand, struggle to depict non-literal expressions. Recent works primarily deal with this issue by compiling humanly annotated datasets on a small scale, which not only demands specialised expertise but also proves highly inefficient. To address this issue, we introduce ViPE: Visualise Pretty-much Everything. ViPE offers a series of lightweight and robust language models that have been trained on a large-scale set of lyrics with noisy visual descriptions that represent their implicit meaning. The synthetic visual descriptions are generated by GPT3.5 relying on neither human annotations nor images. ViPE effectively expresses any arbitrary piece of text into a visualisable description, enabling meaningful and high-quality image generation. We provide compelling evidence that ViPE is more robust than GPT3.5 in synthesising visual elaborations. ViPE also exhibits an understanding of figurative expressions comparable to human experts, providing a powerful and open-source backbone to many downstream applications such as music video and caption generation.

{{</citation>}}


### (22/176) One For All & All For One: Bypassing Hyperparameter Tuning with Model Averaging For Cross-Lingual Transfer (Fabian David Schmidt et al., 2023)

{{<citation>}}

Fabian David Schmidt, Ivan Vulić, Goran Glavaš. (2023)  
**One For All & All For One: Bypassing Hyperparameter Tuning with Model Averaging For Cross-Lingual Transfer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, NER, NLI, QA  
[Paper Link](http://arxiv.org/abs/2310.10532v1)  

---


**ABSTRACT**  
Multilingual language models enable zero-shot cross-lingual transfer (ZS-XLT): fine-tuned on sizable source-language task data, they perform the task in target languages without labeled instances. The effectiveness of ZS-XLT hinges on the linguistic proximity between languages and the amount of pretraining data for a language. Because of this, model selection based on source-language validation is unreliable: it picks model snapshots with suboptimal target-language performance. As a remedy, some work optimizes ZS-XLT by extensively tuning hyperparameters: the follow-up work then routinely struggles to replicate the original results. Other work searches over narrower hyperparameter grids, reporting substantially lower performance. In this work, we therefore propose an unsupervised evaluation protocol for ZS-XLT that decouples performance maximization from hyperparameter tuning. As a robust and more transparent alternative to extensive hyperparameter tuning, we propose to accumulatively average snapshots from different runs into a single model. We run broad ZS-XLT experiments on both higher-level semantic tasks (NLI, extractive QA) and a lower-level token classification task (NER) and find that conventional model selection based on source-language validation quickly plateaus to suboptimal ZS-XLT performance. On the other hand, our accumulative run-by-run averaging of models trained with different hyperparameters boosts ZS-XLT performance and closely correlates with "oracle" ZS-XLT, i.e., model selection based on target-language validation performance.

{{</citation>}}


### (23/176) Semantic Parsing by Large Language Models for Intricate Updating Strategies of Zero-Shot Dialogue State Tracking (Yuxiang Wu et al., 2023)

{{<citation>}}

Yuxiang Wu, Guanting Dong, Weiran Xu. (2023)  
**Semantic Parsing by Large Language Models for Intricate Updating Strategies of Zero-Shot Dialogue State Tracking**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Dialog, Dialogue, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.10520v2)  

---


**ABSTRACT**  
Zero-shot Dialogue State Tracking (DST) addresses the challenge of acquiring and annotating task-oriented dialogues, which can be time consuming and costly. However, DST extends beyond simple slot-filling and requires effective updating strategies for tracking dialogue state as conversations progress. In this paper, we propose ParsingDST, a new In-Context Learning (ICL) method, to introduce additional intricate updating strategies in zero-shot DST. Our approach reformulates the DST task by leveraging powerful Large Language Models (LLMs) and translating the original dialogue text to JSON through semantic parsing as an intermediate state. We also design a novel framework that includes more modules to ensure the effectiveness of updating strategies in the text-to-JSON process. Experimental results demonstrate that our approach outperforms existing zero-shot DST methods on MultiWOZ, exhibiting significant improvements in Joint Goal Accuracy (JGA) and slot accuracy compared to existing ICL methods.

{{</citation>}}


### (24/176) UNO-DST: Leveraging Unlabelled Data in Zero-Shot Dialogue State Tracking (Chuang Li et al., 2023)

{{<citation>}}

Chuang Li, Yan Zhang, Min-Yen Kan, Haizhou Li. (2023)  
**UNO-DST: Leveraging Unlabelled Data in Zero-Shot Dialogue State Tracking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.10492v1)  

---


**ABSTRACT**  
Previous zero-shot dialogue state tracking (DST) methods only apply transfer learning, but ignore unlabelled data in the target domain. We transform zero-shot DST into few-shot DST by utilising such unlabelled data via joint and self-training methods. Our method incorporates auxiliary tasks that generate slot types as inverse prompts for main tasks, creating slot values during joint training. Cycle consistency between these two tasks enables the generation and selection of quality samples in unknown target domains for subsequent fine-tuning. This approach also facilitates automatic label creation, thereby optimizing the training and fine-tuning of DST models. We demonstrate this method's effectiveness on large language models in zero-shot scenarios, improving average joint goal accuracy by $8\%$ across all domains in MultiWOZ.

{{</citation>}}


### (25/176) Harnessing the Power of LLMs: Evaluating Human-AI Text Co-Creation through the Lens of News Headline Generation (Zijian Ding et al., 2023)

{{<citation>}}

Zijian Ding, Alison Smith-Renner, Wenjuan Zhang, Joel R. Tetreault, Alejandro Jaimes. (2023)  
**Harnessing the Power of LLMs: Evaluating Human-AI Text Co-Creation through the Lens of News Headline Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10706v2)  

---


**ABSTRACT**  
To explore how humans can best leverage LLMs for writing and how interacting with these models affects feelings of ownership and trust in the writing process, we compared common human-AI interaction types (e.g., guiding system, selecting from system outputs, post-editing outputs) in the context of LLM-assisted news headline generation. While LLMs alone can generate satisfactory news headlines, on average, human control is needed to fix undesirable model outputs. Of the interaction methods, guiding and selecting model output added the most benefit with the lowest cost (in time and effort). Further, AI assistance did not harm participants' perception of control compared to freeform editing.

{{</citation>}}


### (26/176) Type-aware Decoding via Explicitly Aggregating Event Information for Document-level Event Extraction (Gang Zhao et al., 2023)

{{<citation>}}

Gang Zhao, Yidong Shi, Shudong Lu, Xinjie Yang, Guanting Dong, Jian Xu, Xiaocheng Gong, Si Li. (2023)  
**Type-aware Decoding via Explicitly Aggregating Event Information for Document-level Event Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Event Extraction  
[Paper Link](http://arxiv.org/abs/2310.10487v1)  

---


**ABSTRACT**  
Document-level event extraction (DEE) faces two main challenges: arguments-scattering and multi-event. Although previous methods attempt to address these challenges, they overlook the interference of event-unrelated sentences during event detection and neglect the mutual interference of different event roles during argument extraction. Therefore, this paper proposes a novel Schema-based Explicitly Aggregating~(SEA) model to address these limitations. SEA aggregates event information into event type and role representations, enabling the decoding of event records based on specific type-aware representations. By detecting each event based on its event type representation, SEA mitigates the interference caused by event-unrelated information. Furthermore, SEA extracts arguments for each role based on its role-aware representations, reducing mutual interference between different roles. Experimental results on the ChFinAnn and DuEE-fin datasets show that SEA outperforms the SOTA methods.

{{</citation>}}


### (27/176) xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection (Nuno M. Guerreiro et al., 2023)

{{<citation>}}

Nuno M. Guerreiro, Ricardo Rei, Daan van Stigt, Luisa Coheur, Pierre Colombo, André F. T. Martins. (2023)  
**xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.10482v1)  

---


**ABSTRACT**  
Widely used learned metrics for machine translation evaluation, such as COMET and BLEURT, estimate the quality of a translation hypothesis by providing a single sentence-level score. As such, they offer little insight into translation errors (e.g., what are the errors and what is their severity). On the other hand, generative large language models (LLMs) are amplifying the adoption of more granular strategies to evaluation, attempting to detail and categorize translation errors. In this work, we introduce xCOMET, an open-source learned metric designed to bridge the gap between these approaches. xCOMET integrates both sentence-level evaluation and error span detection capabilities, exhibiting state-of-the-art performance across all types of evaluation (sentence-level, system-level, and error span detection). Moreover, it does so while highlighting and categorizing error spans, thus enriching the quality assessment. We also provide a robustness analysis with stress tests, and show that xCOMET is largely capable of identifying localized critical errors and hallucinations.

{{</citation>}}


### (28/176) DemoSG: Demonstration-enhanced Schema-guided Generation for Low-resource Event Extraction (Gang Zhao et al., 2023)

{{<citation>}}

Gang Zhao, Xiaocheng Gong, Xinjie Yang, Guanting Dong, Shudong Lu, Si Li. (2023)  
**DemoSG: Demonstration-enhanced Schema-guided Generation for Low-resource Event Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Event Extraction  
[Paper Link](http://arxiv.org/abs/2310.10481v1)  

---


**ABSTRACT**  
Most current Event Extraction (EE) methods focus on the high-resource scenario, which requires a large amount of annotated data and can hardly be applied to low-resource domains. To address EE more effectively with limited resources, we propose the Demonstration-enhanced Schema-guided Generation (DemoSG) model, which benefits low-resource EE from two aspects: Firstly, we propose the demonstration-based learning paradigm for EE to fully use the annotated data, which transforms them into demonstrations to illustrate the extraction process and help the model learn effectively. Secondly, we formulate EE as a natural language generation task guided by schema-based prompts, thereby leveraging label semantics and promoting knowledge transfer in low-resource scenarios. We conduct extensive experiments under in-domain and domain adaptation low-resource settings on three datasets, and study the robustness of DemoSG. The results show that DemoSG significantly outperforms current methods in low-resource scenarios.

{{</citation>}}


### (29/176) G-SPEED: General SParse Efficient Editing MoDel (Haoke Zhang et al., 2023)

{{<citation>}}

Haoke Zhang, Yue Wang, Juntao Li, Xiabing Zhou, Min Zhang. (2023)  
**G-SPEED: General SParse Efficient Editing MoDel**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10480v1)  

---


**ABSTRACT**  
Large Language Models~(LLMs) have demonstrated incredible capabilities in understanding, generating, and manipulating languages. Through human-model interactions, LLMs can automatically understand human-issued instructions and output the expected contents, which can significantly increase working efficiency. In various types of real-world demands, editing-oriented tasks account for a considerable proportion, which involves an interactive process that entails the continuous refinement of existing texts to meet specific criteria. Due to the need for multi-round human-model interaction and the generation of complicated editing tasks, there is an emergent need for efficient general editing models. In this paper, we propose \underline{\textbf{G}}eneral \underline{\textbf{SP}}arse \underline{\textbf{E}}fficient \underline{\textbf{E}}diting Mo\underline{\textbf{D}}el~(\textbf{G-SPEED}), which can fulfill diverse editing requirements through a single model while maintaining low computational costs. Specifically, we first propose a novel unsupervised text editing data clustering algorithm to deal with the data scarcity problem. Subsequently, we introduce a sparse editing model architecture to mitigate the inherently limited learning capabilities of small language models. The experimental outcomes indicate that G-SPEED, with its 508M parameters, can surpass LLMs equipped with 175B parameters. Our code and model checkpoints are available at \url{https://github.com/Banner-Z/G-SPEED}.

{{</citation>}}


### (30/176) Gaining Wisdom from Setbacks: Aligning Large Language Models via Mistake Analysis (Kai Chen et al., 2023)

{{<citation>}}

Kai Chen, Chunwei Wang, Kuo Yang, Jianhua Han, Lanqing Hong, Fei Mi, Hang Xu, Zhengying Liu, Wenyong Huang, Zhenguo Li, Dit-Yan Yeung, Lifeng Shang, Xin Jiang, Qun Liu. (2023)  
**Gaining Wisdom from Setbacks: Aligning Large Language Models via Mistake Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10477v1)  

---


**ABSTRACT**  
The rapid advancement of large language models (LLMs) presents both opportunities and challenges, particularly concerning unintentional generation of harmful and toxic responses. While the traditional alignment methods strive to steer LLMs towards desired performance and shield them from malicious content, this study proposes a novel alignment strategy rooted in mistake analysis by exposing LLMs to flawed outputs purposefully and then conducting a thorough assessment to fully comprehend internal reasons via natural language analysis. Thus, toxic responses can be transformed into instruction tuning corpus for model alignment, and LLMs can not only be deterred from generating flawed responses but also trained to self-criticize, leveraging its innate ability to discriminate toxic content. Experimental results demonstrate that the proposed method outperforms conventional alignment techniques for safety instruction following, while maintaining superior efficiency.

{{</citation>}}


### (31/176) Stance Detection with Collaborative Role-Infused LLM-Based Agents (Xiaochong Lan et al., 2023)

{{<citation>}}

Xiaochong Lan, Chen Gao, Depeng Jin, Yong Li. (2023)  
**Stance Detection with Collaborative Role-Infused LLM-Based Agents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Stance Detection  
[Paper Link](http://arxiv.org/abs/2310.10467v1)  

---


**ABSTRACT**  
Stance detection automatically detects the stance in a text towards a target, vital for content analysis in web and social media research. Despite their promising capabilities, LLMs encounter challenges when directly applied to stance detection. First, stance detection demands multi-aspect knowledge, from deciphering event-related terminologies to understanding the expression styles in social media platforms. Second, stance detection requires advanced reasoning to infer authors' implicit viewpoints, as stance are often subtly embedded rather than overtly stated in the text. To address these challenges, we design a three-stage framework COLA (short for Collaborative rOle-infused LLM-based Agents) in which LLMs are designated distinct roles, creating a collaborative system where each role contributes uniquely. Initially, in the multidimensional text analysis stage, we configure the LLMs to act as a linguistic expert, a domain specialist, and a social media veteran to get a multifaceted analysis of texts, thus overcoming the first challenge. Next, in the reasoning-enhanced debating stage, for each potential stance, we designate a specific LLM-based agent to advocate for it, guiding the LLM to detect logical connections between text features and stance, tackling the second challenge. Finally, in the stance conclusion stage, a final decision maker agent consolidates prior insights to determine the stance. Our approach avoids extra annotated data and model training and is highly usable. We achieve state-of-the-art performance across multiple datasets. Ablation studies validate the effectiveness of each design role in handling stance detection. Further experiments have demonstrated the explainability and the versatility of our approach. Our approach excels in usability, accuracy, effectiveness, explainability and versatility, highlighting its value.

{{</citation>}}


### (32/176) Text Summarization Using Large Language Models: A Comparative Study of MPT-7b-instruct, Falcon-7b-instruct, and OpenAI Chat-GPT Models (Lochan Basyal et al., 2023)

{{<citation>}}

Lochan Basyal, Mihir Sanghvi. (2023)  
**Text Summarization Using Large Language Models: A Comparative Study of MPT-7b-instruct, Falcon-7b-instruct, and OpenAI Chat-GPT Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, BERT, BLEU, ChatGPT, Falcon, GPT, Generative AI, Language Model, NLP, Natural Language Processing, Summarization, Text Summarization, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.10449v2)  

---


**ABSTRACT**  
Text summarization is a critical Natural Language Processing (NLP) task with applications ranging from information retrieval to content generation. Leveraging Large Language Models (LLMs) has shown remarkable promise in enhancing summarization techniques. This paper embarks on an exploration of text summarization with a diverse set of LLMs, including MPT-7b-instruct, falcon-7b-instruct, and OpenAI ChatGPT text-davinci-003 models. The experiment was performed with different hyperparameters and evaluated the generated summaries using widely accepted metrics such as the Bilingual Evaluation Understudy (BLEU) Score, Recall-Oriented Understudy for Gisting Evaluation (ROUGE) Score, and Bidirectional Encoder Representations from Transformers (BERT) Score. According to the experiment, text-davinci-003 outperformed the others. This investigation involved two distinct datasets: CNN Daily Mail and XSum. Its primary objective was to provide a comprehensive understanding of the performance of Large Language Models (LLMs) when applied to different datasets. The assessment of these models' effectiveness contributes valuable insights to researchers and practitioners within the NLP domain. This work serves as a resource for those interested in harnessing the potential of LLMs for text summarization and lays the foundation for the development of advanced Generative AI applications aimed at addressing a wide spectrum of business challenges.

{{</citation>}}


### (33/176) MechGPT, a language-based strategy for mechanics and materials modeling that connects knowledge across scales, disciplines and modalities (Markus J. Buehler, 2023)

{{<citation>}}

Markus J. Buehler. (2023)  
**MechGPT, a language-based strategy for mechanics and materials modeling that connects knowledge across scales, disciplines and modalities**  

---
Primary Category: cs.CL  
Categories: cond-mat-mtrl-sci, cs-CL, cs.CL  
Keywords: GPT, Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10445v1)  

---


**ABSTRACT**  
For centuries, researchers have sought out ways to connect disparate areas of knowledge. While early scholars (Galileo, da Vinci, etc.) were experts across fields, specialization has taken hold later. With the advent of Artificial Intelligence, we can now explore relationships across areas (e.g., mechanics-biology) or disparate domains (e.g., failure mechanics-art). To achieve this, we use a fine-tuned Large Language Model (LLM), here for a subset of knowledge in multiscale materials failure. The approach includes the use of a general-purpose LLM to distill question-answer pairs from raw sources followed by LLM fine-tuning. The resulting MechGPT LLM foundation model is used in a series of computational experiments to explore its capacity for knowledge retrieval, various language tasks, hypothesis generation, and connecting knowledge across disparate areas. While the model has some ability to recall knowledge from training, we find that LLMs are particularly useful to extract structural insights through Ontological Knowledge Graphs. These interpretable graph structures provide explanatory insights, frameworks for new research questions, and visual representations of knowledge that also can be used in retrieval-augmented generation. Three versions of MechGPT are discussed, featuring different sizes from 13 billion to 70 billion parameters, and reaching context lengths of more than 10,000 tokens. This provides ample capacity for sophisticated retrieval augmented strategies, as well as agent-based modeling where multiple LLMs interact collaboratively and/or adversarially, the incorporation of new data from the literature or web searches, as well as multimodality.

{{</citation>}}


### (34/176) Exploiting User Comments for Early Detection of Fake News Prior to Users' Commenting (Qiong Nan et al., 2023)

{{<citation>}}

Qiong Nan, Qiang Sheng, Juan Cao, Yongchun Zhu, Danding Wang, Guang Yang, Jintao Li, Kai Shu. (2023)  
**Exploiting User Comments for Early Detection of Fake News Prior to Users' Commenting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-SI, cs.CL  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2310.10429v1)  

---


**ABSTRACT**  
Both accuracy and timeliness are key factors in detecting fake news on social media. However, most existing methods encounter an accuracy-timeliness dilemma: Content-only methods guarantee timeliness but perform moderately because of limited available information, while social context-based ones generally perform better but inevitably lead to latency because of social context accumulation needs. To break such a dilemma, a feasible but not well-studied solution is to leverage social contexts (e.g., comments) from historical news for training a detection model and apply it to newly emerging news without social contexts. This requires the model to (1) sufficiently learn helpful knowledge from social contexts, and (2) be well compatible with situations that social contexts are available or not. To achieve this goal, we propose to absorb and parameterize useful knowledge from comments in historical news and then inject it into a content-only detection model. Specifically, we design the Comments Assisted Fake News Detection method (CAS-FEND), which transfers useful knowledge from a comments-aware teacher model to a content-only student model during training. The student model is further used to detect newly emerging fake news. Experiments show that the CAS-FEND student model outperforms all content-only methods and even those with 1/4 comments as inputs, demonstrating its superiority for early detection.

{{</citation>}}


### (35/176) Can Word Sense Distribution Detect Semantic Changes of Words? (Xiaohang Tang et al., 2023)

{{<citation>}}

Xiaohang Tang, Yi Zhou, Taichi Aida, Procheta Sen, Danushka Bollegala. (2023)  
**Can Word Sense Distribution Detect Semantic Changes of Words?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP, Word Sense Disambiguation  
[Paper Link](http://arxiv.org/abs/2310.10400v1)  

---


**ABSTRACT**  
Semantic Change Detection (SCD) of words is an important task for various NLP applications that must make time-sensitive predictions. Some words are used over time in novel ways to express new meanings, and these new meanings establish themselves as novel senses of existing words. On the other hand, Word Sense Disambiguation (WSD) methods associate ambiguous words with sense ids, depending on the context in which they occur. Given this relationship between WSD and SCD, we explore the possibility of predicting whether a target word has its meaning changed between two corpora collected at different time steps, by comparing the distributions of senses of that word in each corpora. For this purpose, we use pretrained static sense embeddings to automatically annotate each occurrence of the target word in a corpus with a sense id. Next, we compute the distribution of sense ids of a target word in a given corpus. Finally, we use different divergence or distance measures to quantify the semantic change of the target word across the two given corpora. Our experimental results on SemEval 2020 Task 1 dataset show that word sense distributions can be accurately used to predict semantic changes of words in English, German, Swedish and Latin.

{{</citation>}}


### (36/176) $\textit{Swap and Predict}$ -- Predicting the Semantic Changes in Words across Corpora by Context Swapping (Taichi Aida et al., 2023)

{{<citation>}}

Taichi Aida, Danushka Bollegala. (2023)  
**$\textit{Swap and Predict}$ -- Predicting the Semantic Changes in Words across Corpora by Context Swapping**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.10397v1)  

---


**ABSTRACT**  
Meanings of words change over time and across domains. Detecting the semantic changes of words is an important task for various NLP applications that must make time-sensitive predictions. We consider the problem of predicting whether a given target word, $w$, changes its meaning between two different text corpora, $\mathcal{C}_1$ and $\mathcal{C}_2$. For this purpose, we propose $\textit{Swapping-based Semantic Change Detection}$ (SSCD), an unsupervised method that randomly swaps contexts between $\mathcal{C}_1$ and $\mathcal{C}_2$ where $w$ occurs. We then look at the distribution of contextualised word embeddings of $w$, obtained from a pretrained masked language model (MLM), representing the meaning of $w$ in its occurrence contexts in $\mathcal{C}_1$ and $\mathcal{C}_2$. Intuitively, if the meaning of $w$ does not change between $\mathcal{C}_1$ and $\mathcal{C}_2$, we would expect the distributions of contextualised word embeddings of $w$ to remain the same before and after this random swapping process. Despite its simplicity, we demonstrate that even by using pretrained MLMs without any fine-tuning, our proposed context swapping method accurately predicts the semantic changes of words in four languages (English, German, Swedish, and Latin) and across different time spans (over 50 years and about five years). Moreover, our method achieves significant performance improvements compared to strong baselines for the English semantic change prediction task. Source code is available at https://github.com/a1da4/svp-swap .

{{</citation>}}


### (37/176) Towards a Better Understanding of Variations in Zero-Shot Neural Machine Translation Performance (Shaomu Tan et al., 2023)

{{<citation>}}

Shaomu Tan, Christof Monz. (2023)  
**Towards a Better Understanding of Variations in Zero-Shot Neural Machine Translation Performance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Machine Translation, Multilingual, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.10385v1)  

---


**ABSTRACT**  
Multilingual Neural Machine Translation (MNMT) facilitates knowledge sharing but often suffers from poor zero-shot (ZS) translation qualities. While prior work has explored the causes of overall low ZS performance, our work introduces a fresh perspective: the presence of high variations in ZS performance. This suggests that MNMT does not uniformly exhibit poor ZS capability; instead, certain translation directions yield reasonable results. Through systematic experimentation involving 1,560 language directions spanning 40 languages, we identify three key factors contributing to high variations in ZS NMT performance: 1) target side translation capability 2) vocabulary overlap 3) linguistic properties. Our findings highlight that the target side translation quality is the most influential factor, with vocabulary overlap consistently impacting ZS performance. Additionally, linguistic properties, such as language family and writing system, play a role, particularly with smaller models. Furthermore, we suggest that the off-target issue is a symptom of inadequate ZS performance, emphasizing that zero-shot translation challenges extend beyond addressing the off-target problem. We release the data and models serving as a benchmark to study zero-shot for future research at https://github.com/Smu-Tan/ZS-NMT-Variations

{{</citation>}}


### (38/176) Privacy in Large Language Models: Attacks, Defenses and Future Directions (Haoran Li et al., 2023)

{{<citation>}}

Haoran Li, Yulin Chen, Jinglong Luo, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, Yangqiu Song. (2023)  
**Privacy in Large Language Models: Attacks, Defenses and Future Directions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.10383v1)  

---


**ABSTRACT**  
The advancement of large language models (LLMs) has significantly enhanced the ability to effectively tackle various downstream NLP tasks and unify these tasks into generative pipelines. On the one hand, powerful language models, trained on massive textual data, have brought unparalleled accessibility and usability for both models and users. On the other hand, unrestricted access to these models can also introduce potential malicious and unintentional privacy risks. Despite ongoing efforts to address the safety and privacy concerns associated with LLMs, the problem remains unresolved. In this paper, we provide a comprehensive analysis of the current privacy attacks targeting LLMs and categorize them according to the adversary's assumed capabilities to shed light on the potential vulnerabilities present in LLMs. Then, we present a detailed overview of prominent defense strategies that have been developed to counter these privacy attacks. Beyond existing works, we identify upcoming privacy concerns as LLMs evolve. Lastly, we point out several potential avenues for future exploration.

{{</citation>}}


### (39/176) Contextual Data Augmentation for Task-Oriented Dialog Systems (Dustin Axman et al., 2023)

{{<citation>}}

Dustin Axman, Avik Ray, Shubham Garg, Jing Huang. (2023)  
**Contextual Data Augmentation for Task-Oriented Dialog Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Dialog  
[Paper Link](http://arxiv.org/abs/2310.10380v1)  

---


**ABSTRACT**  
Collection of annotated dialogs for training task-oriented dialog systems have been one of the key bottlenecks in improving current models. While dialog response generation has been widely studied on the agent side, it is not evident if similar generative models can be used to generate a large variety of, and often unexpected, user inputs that real dialog systems encounter in practice. Existing data augmentation techniques such as paraphrase generation do not take the dialog context into consideration. In this paper, we develop a novel dialog augmentation model that generates a user turn, conditioning on full dialog context. Additionally, with a new prompt design for language model, and output re-ranking, the dialogs generated from our model can be directly used to train downstream dialog systems. On common benchmark datasets MultiWoZ and SGD, we show that our dialog augmentation model generates high quality dialogs and improves dialog success rate by as much as $8\%$ over baseline.

{{</citation>}}


### (40/176) Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models (Jirui Qi et al., 2023)

{{<citation>}}

Jirui Qi, Raquel Fernández, Arianna Bisazza. (2023)  
**Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs-LG, cs.CL  
Keywords: Language Model, Multilingual, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2310.10378v2)  

---


**ABSTRACT**  
Multilingual large-scale Pretrained Language Models (PLMs) have been shown to store considerable amounts of factual knowledge, but large variations are observed across languages. With the ultimate goal of ensuring that users with different language backgrounds obtain consistent feedback from the same model, we study the cross-lingual consistency (CLC) of factual knowledge in various multilingual PLMs. To this end, we propose a Ranking-based Consistency (RankC) metric to evaluate knowledge consistency across languages independently from accuracy. Using this metric, we conduct an in-depth analysis of the determining factors for CLC, both at model level and at language-pair level. Among other results, we find that increasing model size leads to higher factual probing accuracy in most languages, but does not improve cross-lingual consistency. Finally, we conduct a case study on CLC when new factual associations are inserted in the PLMs via model editing. Results on a small sample of facts inserted in English reveal a clear pattern whereby the new piece of knowledge transfers only to languages with which English has a high RankC score.

{{</citation>}}


### (41/176) Untying the Reversal Curse via Bidirectional Language Model Editing (Jun-Yu Ma et al., 2023)

{{<citation>}}

Jun-Yu Ma, Jia-Chen Gu, Zhen-Hua Ling, Quan Liu, Cong Liu. (2023)  
**Untying the Reversal Curse via Bidirectional Language Model Editing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10322v1)  

---


**ABSTRACT**  
Recent studies have demonstrated that large language models (LLMs) store massive factual knowledge within their parameters. But existing LLMs are prone to hallucinate unintended text due to false or outdated knowledge. Since retraining LLMs is resource intensive, there has been a growing interest in the concept of model editing. Despite the emergence of benchmarks and approaches, these unidirectional editing and evaluation have failed to explore the reversal curse. Intuitively, if "The capital of France is" is edited to be a counterfact "London" within a model, then it should be able to naturally reason and recall the reverse fact, i.e., "London is the capital of" followed by "France" instead of "England". In this paper, we study bidirectional language model editing, aiming to provide rigorous model editing evaluation to assess if edited LLMs can recall the editing knowledge bidirectionally. A new evaluation metric of reversibility is introduced, and a benchmark dubbed as Bidirectional Assessment for Knowledge Editing (BAKE) is constructed to evaluate the reversibility of edited models in recalling knowledge in the reverse direction of editing. We surprisingly observe that while current editing methods and LLMs can effectively recall editing facts in the direction of editing, they suffer serious deficiencies when evaluated in the reverse direction. To mitigate the reversal curse, a method named Bidirectionally Inversible Relationship moDeling (BIRD) is proposed. A set of editing objectives that incorporate bidirectional relationships between subject and object into the updated model weights are designed. Experiments show that BIRD improves the performance of four representative LLMs of different sizes via question answering and judgement.

{{</citation>}}


### (42/176) Interpreting and Exploiting Functional Specialization in Multi-Head Attention under Multi-task Learning (Chong Li et al., 2023)

{{<citation>}}

Chong Li, Shaonan Wang, Yunhao Zhang, Jiajun Zhang, Chengqing Zong. (2023)  
**Interpreting and Exploiting Functional Specialization in Multi-Head Attention under Multi-task Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.10318v1)  

---


**ABSTRACT**  
Transformer-based models, even though achieving super-human performance on several downstream tasks, are often regarded as a black box and used as a whole. It is still unclear what mechanisms they have learned, especially their core module: multi-head attention. Inspired by functional specialization in the human brain, which helps to efficiently handle multiple tasks, this work attempts to figure out whether the multi-head attention module will evolve similar function separation under multi-tasking training. If it is, can this mechanism further improve the model performance? To investigate these questions, we introduce an interpreting method to quantify the degree of functional specialization in multi-head attention. We further propose a simple multi-task training method to increase functional specialization and mitigate negative information transfer in multi-task learning. Experimental results on seven pre-trained transformer models have demonstrated that multi-head attention does evolve functional specialization phenomenon after multi-task training which is affected by the similarity of tasks. Moreover, the multi-task training strategy based on functional specialization boosts performance in both multi-task learning and transfer learning without adding any parameters.

{{</citation>}}


### (43/176) Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques (Manon Reusens et al., 2023)

{{<citation>}}

Manon Reusens, Philipp Borchert, Margot Mieskes, Jochen De Weerdt, Bart Baesens. (2023)  
**Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Bias, Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2310.10310v1)  

---


**ABSTRACT**  
This paper investigates the transferability of debiasing techniques across different languages within multilingual models. We examine the applicability of these techniques in English, French, German, and Dutch. Using multilingual BERT (mBERT), we demonstrate that cross-lingual transfer of debiasing techniques is not only feasible but also yields promising results. Surprisingly, our findings reveal no performance disadvantages when applying these techniques to non-English languages. Using translations of the CrowS-Pairs dataset, our analysis identifies SentenceDebias as the best technique across different languages, reducing bias in mBERT by an average of 13%. We also find that debiasing techniques with additional pretraining exhibit enhanced cross-lingual effectiveness for the languages included in the analyses, particularly in lower-resource languages. These novel insights contribute to a deeper understanding of bias mitigation in multilingual language models and provide practical guidance for debiasing techniques in different language contexts.

{{</citation>}}


### (44/176) Key-phrase boosted unsupervised summary generation for FinTech organization (Aadit Deshpande et al., 2023)

{{<citation>}}

Aadit Deshpande, Shreya Goyal, Prateek Nagwanshi, Avinash Tripathy. (2023)  
**Key-phrase boosted unsupervised summary generation for FinTech organization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.10294v1)  

---


**ABSTRACT**  
With the recent advances in social media, the use of NLP techniques in social media data analysis has become an emerging research direction. Business organizations can particularly benefit from such an analysis of social media discourse, providing an external perspective on consumer behavior. Some of the NLP applications such as intent detection, sentiment classification, text summarization can help FinTech organizations to utilize the social media language data to find useful external insights and can be further utilized for downstream NLP tasks. Particularly, a summary which highlights the intents and sentiments of the users can be very useful for these organizations to get an external perspective. This external perspective can help organizations to better manage their products, offers, promotional campaigns, etc. However, certain challenges, such as a lack of labeled domain-specific datasets impede further exploration of these tasks in the FinTech domain. To overcome these challenges, we design an unsupervised phrase-based summary generation from social media data, using 'Action-Object' pairs (intent phrases). We evaluated the proposed method with other key-phrase based summary generation methods in the direction of contextual information of various Reddit discussion threads, available in the different summaries. We introduce certain "Context Metrics" such as the number of Unique words, Action-Object pairs, and Noun chunks to evaluate the contextual information retrieved from the source text in these phrase-based summaries. We demonstrate that our methods significantly outperform the baseline on these metrics, thus providing a qualitative and quantitative measure of their efficacy. Proposed framework has been leveraged as a web utility portal hosted within Amex.

{{</citation>}}


### (45/176) Multi-Stage Pre-training Enhanced by ChatGPT for Multi-Scenario Multi-Domain Dialogue Summarization (Weixiao Zhou et al., 2023)

{{<citation>}}

Weixiao Zhou, Gengyao Li, Xianfu Cheng, Xinnian Liang, Junnan Zhu, Feifei Zhai, Zhoujun Li. (2023)  
**Multi-Stage Pre-training Enhanced by ChatGPT for Multi-Scenario Multi-Domain Dialogue Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT, Summarization  
[Paper Link](http://arxiv.org/abs/2310.10285v1)  

---


**ABSTRACT**  
Dialogue summarization involves a wide range of scenarios and domains. However, existing methods generally only apply to specific scenarios or domains. In this study, we propose a new pre-trained model specifically designed for multi-scenario multi-domain dialogue summarization. It adopts a multi-stage pre-training strategy to reduce the gap between the pre-training objective and fine-tuning objective. Specifically, we first conduct domain-aware pre-training using large-scale multi-scenario multi-domain dialogue data to enhance the adaptability of our pre-trained model. Then, we conduct task-oriented pre-training using large-scale multi-scenario multi-domain "dialogue-summary" parallel data annotated by ChatGPT to enhance the dialogue summarization ability of our pre-trained model. Experimental results on three dialogue summarization datasets from different scenarios and domains indicate that our pre-trained model significantly outperforms previous state-of-the-art models in full fine-tuning, zero-shot, and few-shot settings.

{{</citation>}}


### (46/176) Enhancing Interpretability using Human Similarity Judgements to Prune Word Embeddings (Natalia Flechas Manrique et al., 2023)

{{<citation>}}

Natalia Flechas Manrique, Wanqian Bao, Aurelie Herbelot, Uri Hasson. (2023)  
**Enhancing Interpretability using Human Similarity Judgements to Prune Word Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Embedding, NLP, Word Embedding  
[Paper Link](http://arxiv.org/abs/2310.10262v1)  

---


**ABSTRACT**  
Interpretability methods in NLP aim to provide insights into the semantics underlying specific system architectures. Focusing on word embeddings, we present a supervised-learning method that, for a given domain (e.g., sports, professions), identifies a subset of model features that strongly improve prediction of human similarity judgments. We show this method keeps only 20-40% of the original embeddings, for 8 independent semantic domains, and that it retains different feature sets across domains. We then present two approaches for interpreting the semantics of the retained features. The first obtains the scores of the domain words (co-hyponyms) on the first principal component of the retained embeddings, and extracts terms whose co-occurrence with the co-hyponyms tracks these scores' profile. This analysis reveals that humans differentiate e.g. sports based on how gender-inclusive and international they are. The second approach uses the retained sets as variables in a probing task that predicts values along 65 semantically annotated dimensions for a dataset of 535 words. The features retained for professions are best at predicting cognitive, emotional and social dimensions, whereas features retained for fruits or vegetables best predict the gustation (taste) dimension. We discuss implications for alignment between AI systems and human knowledge.

{{</citation>}}


### (47/176) Prediction of Arabic Legal Rulings using Large Language Models (Adel Ammar et al., 2023)

{{<citation>}}

Adel Ammar, Anis Koubaa, Bilel Benjdira, Omar Najar, Serry Sibaee. (2023)  
**Prediction of Arabic Legal Rulings using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, BLEU, GPT, GPT-3.5, LLaMA, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2310.10260v1)  

---


**ABSTRACT**  
In the intricate field of legal studies, the analysis of court decisions is a cornerstone for the effective functioning of the judicial system. The ability to predict court outcomes helps judges during the decision-making process and equips lawyers with invaluable insights, enhancing their strategic approaches to cases. Despite its significance, the domain of Arabic court analysis remains under-explored. This paper pioneers a comprehensive predictive analysis of Arabic court decisions on a dataset of 10,813 commercial court real cases, leveraging the advanced capabilities of the current state-of-the-art large language models. Through a systematic exploration, we evaluate three prevalent foundational models (LLaMA-7b, JAIS-13b, and GPT3.5-turbo) and three training paradigms: zero-shot, one-shot, and tailored fine-tuning. Besides, we assess the benefit of summarizing and/or translating the original Arabic input texts. This leads to a spectrum of 14 model variants, for which we offer a granular performance assessment with a series of different metrics (human assessment, GPT evaluation, ROUGE, and BLEU scores). We show that all variants of LLaMA models yield limited performance, whereas GPT-3.5-based models outperform all other models by a wide margin, surpassing the average score of the dedicated Arabic-centric JAIS model by 50%. Furthermore, we show that all scores except human evaluation are inconsistent and unreliable for assessing the performance of large language models on court decision predictions. This study paves the way for future research, bridging the gap between computational linguistics and Arabic legal analytics.

{{</citation>}}


### (48/176) VIBE: Topic-Driven Temporal Adaptation for Twitter Classification (Yuji Zhang et al., 2023)

{{<citation>}}

Yuji Zhang, Jing Li, Wenjie Li. (2023)  
**VIBE: Topic-Driven Temporal Adaptation for Twitter Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.10191v2)  

---


**ABSTRACT**  
Language features are evolving in real-world social media, resulting in the deteriorating performance of text classification in dynamics. To address this challenge, we study temporal adaptation, where models trained on past data are tested in the future. Most prior work focused on continued pretraining or knowledge updating, which may compromise their performance on noisy social media data. To tackle this issue, we reflect feature change via modeling latent topic evolution and propose a novel model, VIBE: Variational Information Bottleneck for Evolutions. Concretely, we first employ two Information Bottleneck (IB) regularizers to distinguish past and future topics. Then, the distinguished topics work as adaptive features via multi-task training with timestamp and class label prediction. In adaptive learning, VIBE utilizes retrieved unlabeled data from online streams created posterior to training data time. Substantial Twitter experiments on three classification tasks show that our model, with only 3% of data, significantly outperforms previous state-of-the-art continued-pretraining methods.

{{</citation>}}


### (49/176) Battle of the Large Language Models: Dolly vs LLaMA vs Vicuna vs Guanaco vs Bard vs ChatGPT -- A Text-to-SQL Parsing Comparison (Shuo Sun et al., 2023)

{{<citation>}}

Shuo Sun, Yuchen Zhang, Jiahuan Yan, Yuze Gao, Donovan Ong, Bin Chen, Jian Su. (2023)  
**Battle of the Large Language Models: Dolly vs LLaMA vs Vicuna vs Guanaco vs Bard vs ChatGPT -- A Text-to-SQL Parsing Comparison**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-3.5, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10190v1)  

---


**ABSTRACT**  
The success of ChatGPT has ignited an AI race, with researchers striving to develop new large language models (LLMs) that can match or surpass the language understanding and generation abilities of commercial ones. In recent times, a number of models have emerged, claiming performance near that of GPT-3.5 or GPT-4 through various instruction-tuning methods. As practitioners of Text-to-SQL parsing, we are grateful for their valuable contributions to open-source research. However, it is important to approach these claims with a sense of scrutiny and ascertain the actual effectiveness of these models. Therefore, we pit six popular large language models against each other, systematically evaluating their Text-to-SQL parsing capability on nine benchmark datasets with five different prompting strategies, covering both zero-shot and few-shot scenarios. Regrettably, the open-sourced models fell significantly short of the performance achieved by closed-source models like GPT-3.5, highlighting the need for further work to bridge the performance gap between these models.

{{</citation>}}


### (50/176) Continual Generalized Intent Discovery: Marching Towards Dynamic and Open-world Intent Recognition (Xiaoshuai Song et al., 2023)

{{<citation>}}

Xiaoshuai Song, Yutao Mou, Keqing He, Yueyan Qiu, Pei Wang, Weiran Xu. (2023)  
**Continual Generalized Intent Discovery: Marching Towards Dynamic and Open-world Intent Recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Intent Recognition  
[Paper Link](http://arxiv.org/abs/2310.10184v1)  

---


**ABSTRACT**  
In a practical dialogue system, users may input out-of-domain (OOD) queries. The Generalized Intent Discovery (GID) task aims to discover OOD intents from OOD queries and extend them to the in-domain (IND) classifier. However, GID only considers one stage of OOD learning, and needs to utilize the data in all previous stages for joint training, which limits its wide application in reality. In this paper, we introduce a new task, Continual Generalized Intent Discovery (CGID), which aims to continuously and automatically discover OOD intents from dynamic OOD data streams and then incrementally add them to the classifier with almost no previous data, thus moving towards dynamic intent recognition in an open world. Next, we propose a method called Prototype-guided Learning with Replay and Distillation (PLRD) for CGID, which bootstraps new intent discovery through class prototypes and balances new and old intents through data replay and feature distillation. Finally, we conduct detailed experiments and analysis to verify the effectiveness of PLRD and understand the key challenges of CGID for future research.

{{</citation>}}


### (51/176) TRIGO: Benchmarking Formal Mathematical Proof Reduction for Generative Language Models (Jing Xiong et al., 2023)

{{<citation>}}

Jing Xiong, Jianhao Shen, Ye Yuan, Haiming Wang, Yichun Yin, Zhengying Liu, Lin Li, Zhijiang Guo, Qingxing Cao, Yinya Huang, Chuanyang Zheng, Xiaodan Liang, Ming Zhang, Qun Liu. (2023)  
**TRIGO: Benchmarking Formal Mathematical Proof Reduction for Generative Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10180v1)  

---


**ABSTRACT**  
Automated theorem proving (ATP) has become an appealing domain for exploring the reasoning ability of the recent successful generative language models. However, current ATP benchmarks mainly focus on symbolic inference, but rarely involve the understanding of complex number combination reasoning. In this work, we propose TRIGO, an ATP benchmark that not only requires a model to reduce a trigonometric expression with step-by-step proofs but also evaluates a generative LM's reasoning ability on formulas and its capability to manipulate, group, and factor number terms. We gather trigonometric expressions and their reduced forms from the web, annotate the simplification process manually, and translate it into the ``Lean'' formal language system. We then automatically generate additional examples from the annotated samples to expand the dataset. Furthermore, we develop an automatic generator based on Lean-Gym to create dataset splits of varying difficulties and distributions in order to thoroughly analyze the model's generalization ability. Our extensive experiments show our proposed TRIGO poses a new challenge for advanced generative LM's including GPT-4 which is pre-trained on a considerable amount of open-source formal theorem-proving language data, and provide a new tool to study the generative LM's ability on both formal and mathematical reasoning.

{{</citation>}}


### (52/176) Large Language Models Meet Open-World Intent Discovery and Recognition: An Evaluation of ChatGPT (Xiaoshuai Song et al., 2023)

{{<citation>}}

Xiaoshuai Song, Keqing He, Pei Wang, Guanting Dong, Yutao Mou, Jingang Wang, Yunsen Xian, Xunliang Cai, Weiran Xu. (2023)  
**Large Language Models Meet Open-World Intent Discovery and Recognition: An Evaluation of ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10176v1)  

---


**ABSTRACT**  
The tasks of out-of-domain (OOD) intent discovery and generalized intent discovery (GID) aim to extend a closed intent classifier to open-world intent sets, which is crucial to task-oriented dialogue (TOD) systems. Previous methods address them by fine-tuning discriminative models. Recently, although some studies have been exploring the application of large language models (LLMs) represented by ChatGPT to various downstream tasks, it is still unclear for the ability of ChatGPT to discover and incrementally extent OOD intents. In this paper, we comprehensively evaluate ChatGPT on OOD intent discovery and GID, and then outline the strengths and weaknesses of ChatGPT. Overall, ChatGPT exhibits consistent advantages under zero-shot settings, but is still at a disadvantage compared to fine-tuned models. More deeply, through a series of analytical experiments, we summarize and discuss the challenges faced by LLMs including clustering, domain-specific understanding, and cross-domain in-context learning scenarios. Finally, we provide empirical guidance for future directions to address these challenges.

{{</citation>}}


### (53/176) Character-LLM: A Trainable Agent for Role-Playing (Yunfan Shao et al., 2023)

{{<citation>}}

Yunfan Shao, Linyang Li, Junqi Dai, Xipeng Qiu. (2023)  
**Character-LLM: A Trainable Agent for Role-Playing**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.10158v1)  

---


**ABSTRACT**  
Large language models (LLMs) can be used to serve as agents to simulate human behaviors, given the powerful ability to understand human instructions and provide high-quality generated texts. Such ability stimulates us to wonder whether LLMs can simulate a person in a higher form than simple human behaviors. Therefore, we aim to train an agent with the profile, experience, and emotional states of a specific person instead of using limited prompts to instruct ChatGPT API. In this work, we introduce Character-LLM that teach LLMs to act as specific people such as Beethoven, Queen Cleopatra, Julius Caesar, etc. Our method focuses on editing profiles as experiences of a certain character and training models to be personal simulacra with these experiences. To assess the effectiveness of our approach, we build a test playground that interviews trained agents and evaluates whether the agents \textit{memorize} their characters and experiences. Experimental results show interesting observations that help build future simulacra of humankind.

{{</citation>}}


### (54/176) Theory of Mind for Multi-Agent Collaboration via Large Language Models (Huao Li et al., 2023)

{{<citation>}}

Huao Li, Yu Quan Chong, Simon Stepputtis, Joseph Campbell, Dana Hughes, Michael Lewis, Katia Sycara. (2023)  
**Theory of Mind for Multi-Agent Collaboration via Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10701v1)  

---


**ABSTRACT**  
While Large Language Models (LLMs) have demonstrated impressive accomplishments in both reasoning and planning, their abilities in multi-agent collaborations remains largely unexplored. This study evaluates LLM-based agents in a multi-agent cooperative text game with Theory of Mind (ToM) inference tasks, comparing their performance with Multi-Agent Reinforcement Learning (MARL) and planning-based baselines. We observed evidence of emergent collaborative behaviors and high-order Theory of Mind capabilities among LLM-based agents. Our results reveal limitations in LLM-based agents' planning optimization due to systematic failures in managing long-horizon contexts and hallucination about the task state. We explore the use of explicit belief state representations to mitigate these issues, finding that it enhances task performance and the accuracy of ToM inferences for LLM-based agents.

{{</citation>}}


### (55/176) Learning to Rank Context for Named Entity Recognition Using a Synthetic Dataset (Arthur Amalvy et al., 2023)

{{<citation>}}

Arthur Amalvy, Vincent Labatut, Richard Dufour. (2023)  
**Learning to Rank Context for Named Entity Recognition Using a Synthetic Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2310.10118v1)  

---


**ABSTRACT**  
While recent pre-trained transformer-based models can perform named entity recognition (NER) with great accuracy, their limited range remains an issue when applied to long documents such as whole novels. To alleviate this issue, a solution is to retrieve relevant context at the document level. Unfortunately, the lack of supervision for such a task means one has to settle for unsupervised approaches. Instead, we propose to generate a synthetic context retrieval training dataset using Alpaca, an instructiontuned large language model (LLM). Using this dataset, we train a neural context retriever based on a BERT model that is able to find relevant context for NER. We show that our method outperforms several retrieval baselines for the NER task on an English literary dataset composed of the first chapter of 40 books.

{{</citation>}}


### (56/176) End-to-end Multichannel Speaker-Attributed ASR: Speaker Guided Decoder and Input Feature Analysis (Can Cui et al., 2023)

{{<citation>}}

Can Cui, Imran Ahamad Sheikh, Mostafa Sadeghi, Emmanuel Vincent. (2023)  
**End-to-end Multichannel Speaker-Attributed ASR: Speaker Guided Decoder and Input Feature Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.10106v1)  

---


**ABSTRACT**  
We present an end-to-end multichannel speaker-attributed automatic speech recognition (MC-SA-ASR) system that combines a Conformer-based encoder with multi-frame crosschannel attention and a speaker-attributed Transformer-based decoder. To the best of our knowledge, this is the first model that efficiently integrates ASR and speaker identification modules in a multichannel setting. On simulated mixtures of LibriSpeech data, our system reduces the word error rate (WER) by up to 12% and 16% relative compared to previously proposed single-channel and multichannel approaches, respectively. Furthermore, we investigate the impact of different input features, including multichannel magnitude and phase information, on the ASR performance. Finally, our experiments on the AMI corpus confirm the effectiveness of our system for real-world multichannel meeting transcription.

{{</citation>}}


### (57/176) Decomposed Prompt Tuning via Low-Rank Reparameterization (Yao Xiao et al., 2023)

{{<citation>}}

Yao Xiao, Lu Xu, Jiaxi Li, Wei Lu, Xiaoli Li. (2023)  
**Decomposed Prompt Tuning via Low-Rank Reparameterization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE, SuperGLUE  
[Paper Link](http://arxiv.org/abs/2310.10094v1)  

---


**ABSTRACT**  
While prompt tuning approaches have achieved competitive performance with high efficiency, we observe that they invariably employ the same initialization process, wherein the soft prompt is either randomly initialized or derived from an existing embedding vocabulary. In contrast to these conventional methods, this study aims to investigate an alternative way to derive soft prompt. Our empirical studies show that the soft prompt typically exhibits a low intrinsic rank characteristic. With such observations, we propose decomposed prompt tuning, a novel approach that utilizes low-rank matrices to initialize the soft prompt. Through the low-rank reparameterization, our method significantly reduces the number of trainable parameters while maintaining effectiveness. Experimental results on the SuperGLUE benchmark in both high-resource and low-resource scenarios demonstrate the effectiveness of the proposed method.

{{</citation>}}


### (58/176) JMedLoRA:Medical Domain Adaptation on Japanese Large Language Models using Instruction-tuning (Issey Sukeda et al., 2023)

{{<citation>}}

Issey Sukeda, Masahiro Suzuki, Hiroki Sakaji, Satoshi Kodera. (2023)  
**JMedLoRA:Medical Domain Adaptation on Japanese Large Language Models using Instruction-tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10083v1)  

---


**ABSTRACT**  
In the ongoing wave of impact driven by large language models (LLMs) like ChatGPT, the adaptation of LLMs to medical domain has emerged as a crucial research frontier. Since mainstream LLMs tend to be designed for general-purpose applications, constructing a medical LLM through domain adaptation is a huge challenge. While instruction-tuning is used to fine-tune some LLMs, its precise roles in domain adaptation remain unknown. Here we show the contribution of LoRA-based instruction-tuning to performance in Japanese medical question-answering tasks. In doing so, we employ a multifaceted evaluation for multiple-choice questions, including scoring based on "Exact match" and "Gestalt distance" in addition to the conventional accuracy. Our findings suggest that LoRA-based instruction-tuning can partially incorporate domain-specific knowledge into LLMs, with larger models demonstrating more pronounced effects. Furthermore, our results underscore the potential of adapting English-centric models for Japanese applications in domain adaptation, while also highlighting the persisting limitations of Japanese-centric models. This initiative represents a pioneering effort in enabling medical institutions to fine-tune and operate models without relying on external services.

{{</citation>}}


### (59/176) Let's reward step by step: Step-Level reward model as the Navigators for Reasoning (Qianli Ma et al., 2023)

{{<citation>}}

Qianli Ma, Haotian Zhou, Tingkai Liu, Jianbo Yuan, Pengfei Liu, Yang You, Hongxia Yang. (2023)  
**Let's reward step by step: Step-Level reward model as the Navigators for Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.10080v1)  

---


**ABSTRACT**  
Recent years have seen considerable advancements in multi-step reasoning with Large Language Models (LLMs). The previous studies have elucidated the merits of integrating feedback or search mechanisms during model inference to improve the reasoning accuracy. The Process-Supervised Reward Model (PRM), typically furnishes LLMs with step-by-step feedback during the training phase, akin to Proximal Policy Optimization (PPO) or reject sampling. Our objective is to examine the efficacy of PRM in the inference phase to help discern the optimal solution paths for multi-step tasks such as mathematical reasoning and code generation. To this end, we propose a heuristic greedy search algorithm that employs the step-level feedback from PRM to optimize the reasoning pathways explored by LLMs. This tailored PRM demonstrated enhanced results compared to the Chain of Thought (CoT) on mathematical benchmarks like GSM8K and MATH. Additionally, to explore the versatility of our approach, we develop a novel method to automatically generate step-level reward dataset for coding tasks and observed similar improved performance in the code generation tasks. Thus highlighting the robust nature of our reward-model-based approach to inference for reasoning tasks.

{{</citation>}}


### (60/176) Prompt Packer: Deceiving LLMs through Compositional Instruction with Hidden Attacks (Shuyu Jiang et al., 2023)

{{<citation>}}

Shuyu Jiang, Xingshu Chen, Rui Tang. (2023)  
**Prompt Packer: Deceiving LLMs through Compositional Instruction with Hidden Attacks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GLM, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.10077v1)  

---


**ABSTRACT**  
Recently, Large language models (LLMs) with powerful general capabilities have been increasingly integrated into various Web applications, while undergoing alignment training to ensure that the generated content aligns with user intent and ethics. Unfortunately, they remain the risk of generating harmful content like hate speech and criminal activities in practical applications. Current approaches primarily rely on detecting, collecting, and training against harmful prompts to prevent such risks. However, they typically focused on the "superficial" harmful prompts with a solitary intent, ignoring composite attack instructions with multiple intentions that can easily elicit harmful content in real-world scenarios. In this paper, we introduce an innovative technique for obfuscating harmful instructions: Compositional Instruction Attacks (CIA), which refers to attacking by combination and encapsulation of multiple instructions. CIA hides harmful prompts within instructions of harmless intentions, making it impossible for the model to identify underlying malicious intentions. Furthermore, we implement two transformation methods, known as T-CIA and W-CIA, to automatically disguise harmful instructions as talking or writing tasks, making them appear harmless to LLMs. We evaluated CIA on GPT-4, ChatGPT, and ChatGLM2 with two safety assessment datasets and two harmful prompt datasets. It achieves an attack success rate of 95%+ on safety assessment datasets, and 83%+ for GPT-4, 91%+ for ChatGPT (gpt-3.5-turbo backed) and ChatGLM2-6B on harmful prompt datasets. Our approach reveals the vulnerability of LLMs to such compositional instruction attacks that harbor underlying harmful intentions, contributing significantly to LLM security development. Warning: this paper may contain offensive or upsetting content!

{{</citation>}}


### (61/176) Verbosity Bias in Preference Labeling by Large Language Models (Keita Saito et al., 2023)

{{<citation>}}

Keita Saito, Akifumi Wachi, Koki Wataoka, Youhei Akimoto. (2023)  
**Verbosity Bias in Preference Labeling by Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Bias, GPT, GPT-4, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10076v1)  

---


**ABSTRACT**  
In recent years, Large Language Models (LLMs) have witnessed a remarkable surge in prevalence, altering the landscape of natural language processing and machine learning. One key factor in improving the performance of LLMs is alignment with humans achieved with Reinforcement Learning from Human Feedback (RLHF), as for many LLMs such as GPT-4, Bard, etc. In addition, recent studies are investigating the replacement of human feedback with feedback from other LLMs named Reinforcement Learning from AI Feedback (RLAIF). We examine the biases that come along with evaluating LLMs with other LLMs and take a closer look into verbosity bias -- a bias where LLMs sometimes prefer more verbose answers even if they have similar qualities. We see that in our problem setting, GPT-4 prefers longer answers more than humans. We also propose a metric to measure this bias.

{{</citation>}}


### (62/176) Bridging Code Semantic and LLMs: Semantic Chain-of-Thought Prompting for Code Generation (Yingwei Ma et al., 2023)

{{<citation>}}

Yingwei Ma, Yue Yu, Shanshan Li, Yu Jiang, Yong Guo, Yuanliang Zhang, Yutao Xie, Xiangke Liao. (2023)  
**Bridging Code Semantic and LLMs: Semantic Chain-of-Thought Prompting for Code Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.10698v1)  

---


**ABSTRACT**  
Large language models (LLMs) have showcased remarkable prowess in code generation. However, automated code generation is still challenging since it requires a high-level semantic mapping between natural language requirements and codes. Most existing LLMs-based approaches for code generation rely on decoder-only causal language models often treate codes merely as plain text tokens, i.e., feeding the requirements as a prompt input, and outputing code as flat sequence of tokens, potentially missing the rich semantic features inherent in source code. To bridge this gap, this paper proposes the "Semantic Chain-of-Thought" approach to intruduce semantic information of code, named SeCoT. Our motivation is that the semantic information of the source code (\eg data flow and control flow) describes more precise program execution behavior, intention and function. By guiding LLM consider and integrate semantic information, we can achieve a more granular understanding and representation of code, enhancing code generation accuracy. Meanwhile, while traditional techniques leveraging such semantic information require complex static or dynamic code analysis to obtain features such as data flow and control flow, SeCoT demonstrates that this process can be fully automated via the intrinsic capabilities of LLMs (i.e., in-context learning), while being generalizable and applicable to challenging domains. While SeCoT can be applied with different LLMs, this paper focuses on the powerful GPT-style models: ChatGPT(close-source model) and WizardCoder(open-source model). The experimental study on three popular DL benchmarks (i.e., HumanEval, HumanEval-ET and MBPP) shows that SeCoT can achieves state-of-the-art performance, greatly improving the potential for large models and code generation.

{{</citation>}}


### (63/176) Fine-tuning ChatGPT for Automatic Scoring (Ehsan Latif et al., 2023)

{{<citation>}}

Ehsan Latif, Xiaoming Zhai. (2023)  
**Fine-tuning ChatGPT for Automatic Scoring**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, BERT, ChatGPT, GPT, GPT-3.5, Google  
[Paper Link](http://arxiv.org/abs/2310.10072v1)  

---


**ABSTRACT**  
This study highlights the potential of fine-tuned ChatGPT (GPT-3.5) for automatically scoring student written constructed responses using example assessment tasks in science education. Recent studies on OpenAI's generative model GPT-3.5 proved its superiority in predicting the natural language with high accuracy and human-like responses. GPT-3.5 has been trained over enormous online language materials such as journals and Wikipedia; therefore, more than direct usage of pre-trained GPT-3.5 is required for automatic scoring as students utilize a different language than trained material. These imply that a domain-specific model, fine-tuned over data for specific tasks, can enhance model performance. In this study, we fine-tuned GPT-3.5 on six assessment tasks with a diverse dataset of middle-school and high-school student responses and expert scoring. The six tasks comprise two multi-label and four multi-class assessment tasks. We compare the performance of fine-tuned GPT-3.5 with the fine-tuned state-of-the-art Google's generated language model, BERT. The results show that in-domain training corpora constructed from science questions and responses for BERT achieved average accuracy = 0.838, SD = 0.069. GPT-3.5 shows a remarkable average increase (9.1%) in automatic scoring accuracy (mean = 9.15, SD = 0.042) for the six tasks, p =0.001 < 0.05. Specifically, for multi-label tasks (item 1 with 5 labels; item 2 with 10 labels), GPT-3.5 achieved significantly higher scoring accuracy than BERT across all the labels, with the second item achieving a 7.1% increase. The average scoring increase for the four multi-class items for GPT-3.5 was 10.6% compared to BERT. Our study confirmed the effectiveness of fine-tuned GPT-3.5 for automatic scoring of student responses on domain-specific data in education with high accuracy. We have released fine-tuned models for public use and community engagement.

{{</citation>}}


### (64/176) NASH: A Simple Unified Framework of Structured Pruning for Accelerating Encoder-Decoder Language Models (Jongwoo Ko et al., 2023)

{{<citation>}}

Jongwoo Ko, Seungjoon Park, Yujin Kim, Sumyeong Ahn, Du-Seong Chang, Euijai Ahn, Se-Young Yun. (2023)  
**NASH: A Simple Unified Framework of Structured Pruning for Accelerating Encoder-Decoder Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP, Pruning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.10054v1)  

---


**ABSTRACT**  
Structured pruning methods have proven effective in reducing the model size and accelerating inference speed in various network architectures such as Transformers. Despite the versatility of encoder-decoder models in numerous NLP tasks, the structured pruning methods on such models are relatively less explored compared to encoder-only models. In this study, we investigate the behavior of the structured pruning of the encoder-decoder models in the decoupled pruning perspective of the encoder and decoder component, respectively. Our findings highlight two insights: (1) the number of decoder layers is the dominant factor of inference speed, and (2) low sparsity in the pruned encoder network enhances generation quality. Motivated by these findings, we propose a simple and effective framework, NASH, that narrows the encoder and shortens the decoder networks of encoder-decoder models. Extensive experiments on diverse generation and inference tasks validate the effectiveness of our method in both speedup and output quality.

{{</citation>}}


### (65/176) Improving Large Language Model Fine-tuning for Solving Math Problems (Yixin Liu et al., 2023)

{{<citation>}}

Yixin Liu, Avi Singh, C. Daniel Freeman, John D. Co-Reyes, Peter J. Liu. (2023)  
**Improving Large Language Model Fine-tuning for Solving Math Problems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2310.10047v1)  

---


**ABSTRACT**  
Despite their success in many natural language tasks, solving math problems remains a significant challenge for large language models (LLMs). A large gap exists between LLMs' pass-at-one and pass-at-N performance in solving math problems, suggesting LLMs might be close to finding correct solutions, motivating our exploration of fine-tuning methods to unlock LLMs' performance. Using the challenging MATH dataset, we investigate three fine-tuning strategies: (1) solution fine-tuning, where we fine-tune to generate a detailed solution for a given math problem; (2) solution-cluster re-ranking, where the LLM is fine-tuned as a solution verifier/evaluator to choose among generated candidate solution clusters; (3) multi-task sequential fine-tuning, which integrates both solution generation and evaluation tasks together efficiently to enhance the LLM performance. With these methods, we present a thorough empirical study on a series of PaLM 2 models and find: (1) The quality and style of the step-by-step solutions used for fine-tuning can make a significant impact on the model performance; (2) While solution re-ranking and majority voting are both effective for improving the model performance when used separately, they can also be used together for an even greater performance boost; (3) Multi-task fine-tuning that sequentially separates the solution generation and evaluation tasks can offer improved performance compared with the solution fine-tuning baseline. Guided by these insights, we design a fine-tuning recipe that yields approximately 58.8% accuracy on the MATH dataset with fine-tuned PaLM 2-L models, an 11.2% accuracy improvement over the few-shot performance of pre-trained PaLM 2-L model with majority voting.

{{</citation>}}


### (66/176) Empirical Study of Zero-Shot NER with ChatGPT (Tingyu Xie et al., 2023)

{{<citation>}}

Tingyu Xie, Qi Li, Jian Zhang, Yan Zhang, Zuozhu Liu, Hongwei Wang. (2023)  
**Empirical Study of Zero-Shot NER with ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, NER, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.10035v1)  

---


**ABSTRACT**  
Large language models (LLMs) exhibited powerful capability in various natural language processing tasks. This work focuses on exploring LLM performance on zero-shot information extraction, with a focus on the ChatGPT and named entity recognition (NER) task. Inspired by the remarkable reasoning capability of LLM on symbolic and arithmetic reasoning, we adapt the prevalent reasoning methods to NER and propose reasoning strategies tailored for NER. First, we explore a decomposed question-answering paradigm by breaking down the NER task into simpler subproblems by labels. Second, we propose syntactic augmentation to stimulate the model's intermediate thinking in two ways: syntactic prompting, which encourages the model to analyze the syntactic structure itself, and tool augmentation, which provides the model with the syntactic information generated by a parsing tool. Besides, we adapt self-consistency to NER by proposing a two-stage majority voting strategy, which first votes for the most consistent mentions, then the most consistent types. The proposed methods achieve remarkable improvements for zero-shot NER across seven benchmarks, including Chinese and English datasets, and on both domain-specific and general-domain scenarios. In addition, we present a comprehensive analysis of the error types with suggestions for optimization directions. We also verify the effectiveness of the proposed methods on the few-shot setting and other LLMs.

{{</citation>}}


## cs.RO (9)



### (67/176) Greedy Perspectives: Multi-Drone View Planning for Collaborative Coverage in Cluttered Environments (Krishna Suresh et al., 2023)

{{<citation>}}

Krishna Suresh, Aditya Rauniyar, Micah Corah, Sebastian Scherer. (2023)  
**Greedy Perspectives: Multi-Drone View Planning for Collaborative Coverage in Cluttered Environments**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.10863v1)  

---


**ABSTRACT**  
Deployment of teams of aerial robots could enable large-scale filming of dynamic groups of people (actors) in complex environments for novel applications in areas such as team sports and cinematography. Toward this end, methods for submodular maximization via sequential greedy planning can be used for scalable optimization of camera views across teams of robots but face challenges with efficient coordination in cluttered environments. Obstacles can produce occlusions and increase chances of inter-robot collision which can violate requirements for near-optimality guarantees. To coordinate teams of aerial robots in filming groups of people in dense environments, a more general view-planning approach is required. We explore how collision and occlusion impact performance in filming applications through the development of a multi-robot multi-actor view planner with an occlusion-aware objective for filming groups of people and compare with a greedy formation planner. To evaluate performance, we plan in five test environments with complex multiple-actor behaviors. Compared with a formation planner, our sequential planner generates 14% greater view reward over the actors for three scenarios and comparable performance to formation planning on two others. We also observe near identical performance of sequential planning both with and without inter-robot collision constraints. Overall, we demonstrate effective coordination of teams of aerial robots for filming groups that may split, merge, or spread apart and in environments cluttered with obstacles that may cause collisions or occlusions.

{{</citation>}}


### (68/176) Interactive Task Planning with Language Models (Boyi Li et al., 2023)

{{<citation>}}

Boyi Li, Philipp Wu, Pieter Abbeel, Jitendra Malik. (2023)  
**Interactive Task Planning with Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-HC, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10645v1)  

---


**ABSTRACT**  
An interactive robot framework accomplishes long-horizon task planning and can easily generalize to new goals or distinct tasks, even during execution. However, most traditional methods require predefined module design, which makes it hard to generalize to different goals. Recent large language model based approaches can allow for more open-ended planning but often require heavy prompt engineering or domain-specific pretrained models. To tackle this, we propose a simple framework that achieves interactive task planning with language models. Our system incorporates both high-level planning and low-level function execution via language. We verify the robustness of our system in generating novel high-level instructions for unseen objectives and its ease of adaptation to different tasks by merely substituting the task guidelines, without the need for additional complex prompt engineering. Furthermore, when the user sends a new request, our system is able to replan accordingly with precision based on the new request, task guidelines and previously executed steps. Please check more details on our https://wuphilipp.github.io/itp_site and https://youtu.be/TrKLuyv26_g.

{{</citation>}}


### (69/176) Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models (Kevin Black et al., 2023)

{{<citation>}}

Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, Sergey Levine. (2023)  
**Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.10639v1)  

---


**ABSTRACT**  
If generalist robots are to operate in truly unstructured environments, they need to be able to recognize and reason about novel objects and scenarios. Such objects and scenarios might not be present in the robot's own training data. We propose SuSIE, a method that leverages an image-editing diffusion model to act as a high-level planner by proposing intermediate subgoals that a low-level controller can accomplish. Specifically, we finetune InstructPix2Pix on video data, consisting of both human videos and robot rollouts, such that it outputs hypothetical future "subgoal" observations given the robot's current observation and a language command. We also use the robot data to train a low-level goal-conditioned policy to act as the aforementioned low-level controller. We find that the high-level subgoal predictions can utilize Internet-scale pretraining and visual understanding to guide the low-level goal-conditioned policy, achieving significantly better generalization and precision than conventional language-conditioned policies. We achieve state-of-the-art results on the CALVIN benchmark, and also demonstrate robust generalization on real-world manipulation tasks, beating strong baselines that have access to privileged information or that utilize orders of magnitude more compute and training data. The project website can be found at http://rail-berkeley.github.io/susie .

{{</citation>}}


### (70/176) Adaptive Robot Assistance: Expertise and Influence in Multi-User Task Planning (Abhinav Dahiya et al., 2023)

{{<citation>}}

Abhinav Dahiya, Stephen L. Smith. (2023)  
**Adaptive Robot Assistance: Expertise and Influence in Multi-User Task Planning**  

---
Primary Category: cs.RO  
Categories: cs-MA, cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.10502v1)  

---


**ABSTRACT**  
This paper addresses the challenge of enabling a single robot to effectively assist multiple humans in decision-making for task planning domains. We introduce a comprehensive framework designed to enhance overall team performance by considering both human expertise in making the optimal decisions and robot influence on human decision-making. Our model integrates these factors seamlessly within the task-planning domain, formulating the problem as a partially observable Markov decision process (POMDP) while treating expertise and influence as unobservable components of the system state. To solve for the robot's actions in such systems, we propose an efficient Attention-Switching policy. This policy capitalizes on the inherent structure of such systems, solving multiple smaller POMDPs to generate heuristics for prioritizing interactions with different human teammates, thereby reducing the state space and improving scalability. Our empirical results on a simulated kit fulfillment task demonstrate improved team performance when the robot's policy accounts for both expertise and influence. This research represents a significant step forward in the field of adaptive robot assistance, paving the way for integration into cost-effective small and mid-scale industries, where substantial investments in robotic infrastructure may not be economically viable.

{{</citation>}}


### (71/176) BEVGPT: Generative Pre-trained Large Model for Autonomous Driving Prediction, Decision-Making, and Planning (Pengqin Wang et al., 2023)

{{<citation>}}

Pengqin Wang, Meixin Zhu, Hongliang Lu, Hui Zhong, Xianda Chen, Shaojie Shen, Xuesong Wang, Yinhai Wang. (2023)  
**BEVGPT: Generative Pre-trained Large Model for Autonomous Driving Prediction, Decision-Making, and Planning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.10357v1)  

---


**ABSTRACT**  
Prediction, decision-making, and motion planning are essential for autonomous driving. In most contemporary works, they are considered as individual modules or combined into a multi-task learning paradigm with a shared backbone but separate task heads. However, we argue that they should be integrated into a comprehensive framework. Although several recent approaches follow this scheme, they suffer from complicated input representations and redundant framework designs. More importantly, they can not make long-term predictions about future driving scenarios. To address these issues, we rethink the necessity of each module in an autonomous driving task and incorporate only the required modules into a minimalist autonomous driving framework. We propose BEVGPT, a generative pre-trained large model that integrates driving scenario prediction, decision-making, and motion planning. The model takes the bird's-eye-view (BEV) images as the only input source and makes driving decisions based on surrounding traffic scenarios. To ensure driving trajectory feasibility and smoothness, we develop an optimization-based motion planning method. We instantiate BEVGPT on Lyft Level 5 Dataset and use Woven Planet L5Kit for realistic driving simulation. The effectiveness and robustness of the proposed framework are verified by the fact that it outperforms previous methods in 100% decision-making metrics and 66% motion planning metrics. Furthermore, the ability of our framework to accurately generate BEV images over the long term is demonstrated through the task of driving scenario prediction. To the best of our knowledge, this is the first generative pre-trained large model for autonomous driving prediction, decision-making, and motion planning with only BEV images as input.

{{</citation>}}


### (72/176) Learning visual-based deformable object rearrangement with local graph neural networks (Yuhong Deng et al., 2023)

{{<citation>}}

Yuhong Deng, Xueqian Wang, Lipeng chen. (2023)  
**Learning visual-based deformable object rearrangement with local graph neural networks**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.10307v1)  

---


**ABSTRACT**  
Goal-conditioned rearrangement of deformable objects (e.g. straightening a rope and folding a cloth) is one of the most common deformable manipulation tasks, where the robot needs to rearrange a deformable object into a prescribed goal configuration with only visual observations. These tasks are typically confronted with two main challenges: the high dimensionality of deformable configuration space and the underlying complexity, nonlinearity and uncertainty inherent in deformable dynamics. To address these challenges, we propose a novel representation strategy that can efficiently model the deformable object states with a set of keypoints and their interactions. We further propose local-graph neural network (GNN), a light local GNN learning to jointly model the deformable rearrangement dynamics and infer the optimal manipulation actions (e.g. pick and place) by constructing and updating two dynamic graphs. Both simulated and real experiments have been conducted to demonstrate that the proposed dynamic graph representation shows superior expressiveness in modeling deformable rearrangement dynamics. Our method reaches much higher success rates on a variety of deformable rearrangement tasks (96.3% on average) than state-of-the-art method in simulation experiments. Besides, our method is much more lighter and has a 60% shorter inference time than state-of-the-art methods. We also demonstrate that our method performs well in the multi-task learning scenario and can be transferred to real-world applications with an average success rate of 95% by solely fine tuning a keypoint detector.

{{</citation>}}


### (73/176) RoboLLM: Robotic Vision Tasks Grounded on Multimodal Large Language Models (Zijun Long et al., 2023)

{{<citation>}}

Zijun Long, George Killick, Richard McCreadie, Gerardo Aragon Camarasa. (2023)  
**RoboLLM: Robotic Vision Tasks Grounded on Multimodal Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10221v1)  

---


**ABSTRACT**  
Robotic vision applications often necessitate a wide range of visual perception tasks, such as object detection, segmentation, and identification. While there have been substantial advances in these individual tasks, integrating specialized models into a unified vision pipeline presents significant engineering challenges and costs. Recently, Multimodal Large Language Models (MLLMs) have emerged as novel backbones for various downstream tasks. We argue that leveraging the pre-training capabilities of MLLMs enables the creation of a simplified framework, thus mitigating the need for task-specific encoders. Specifically, the large-scale pretrained knowledge in MLLMs allows for easier fine-tuning to downstream robotic vision tasks and yields superior performance. We introduce the RoboLLM framework, equipped with a BEiT-3 backbone, to address all visual perception tasks in the ARMBench challenge-a large-scale robotic manipulation dataset about real-world warehouse scenarios. RoboLLM not only outperforms existing baselines but also substantially reduces the engineering burden associated with model selection and tuning. The source code is publicly available at https://github.com/longkukuhi/armbench.

{{</citation>}}


### (74/176) Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning (Dhruv Shah et al., 2023)

{{<citation>}}

Dhruv Shah, Michael Equi, Blazej Osinski, Fei Xia, Brian Ichter, Sergey Levine. (2023)  
**Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10103v1)  

---


**ABSTRACT**  
Navigation in unfamiliar environments presents a major challenge for robots: while mapping and planning techniques can be used to build up a representation of the world, quickly discovering a path to a desired goal in unfamiliar settings with such methods often requires lengthy mapping and exploration. Humans can rapidly navigate new environments, particularly indoor environments that are laid out logically, by leveraging semantics -- e.g., a kitchen often adjoins a living room, an exit sign indicates the way out, and so forth. Language models can provide robots with such knowledge, but directly using language models to instruct a robot how to reach some destination can also be impractical: while language models might produce a narrative about how to reach some goal, because they are not grounded in real-world observations, this narrative might be arbitrarily wrong. Therefore, in this paper we study how the ``semantic guesswork'' produced by language models can be utilized as a guiding heuristic for planning algorithms. Our method, Language Frontier Guide (LFG), uses the language model to bias exploration of novel real-world environments by incorporating the semantic knowledge stored in language models as a search heuristic for planning with either topological or metric maps. We evaluate LFG in challenging real-world environments and simulated benchmarks, outperforming uninformed exploration and other ways of using language models.

{{</citation>}}


### (75/176) Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance (Jesse Zhang et al., 2023)

{{<citation>}}

Jesse Zhang, Jiahui Zhang, Karl Pertsch, Ziyi Liu, Xiang Ren, Minsuk Chang, Shao-Hua Sun, Joseph J. Lim. (2023)  
**Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10021v2)  

---


**ABSTRACT**  
We propose BOSS, an approach that automatically learns to solve new long-horizon, complex, and meaningful tasks by growing a learned skill library with minimal supervision. Prior work in reinforcement learning require expert supervision, in the form of demonstrations or rich reward functions, to learn long-horizon tasks. Instead, our approach BOSS (BOotStrapping your own Skills) learns to accomplish new tasks by performing "skill bootstrapping," where an agent with a set of primitive skills interacts with the environment to practice new skills without receiving reward feedback for tasks outside of the initial skill set. This bootstrapping phase is guided by large language models (LLMs) that inform the agent of meaningful skills to chain together. Through this process, BOSS builds a wide range of complex and useful behaviors from a basic set of primitive skills. We demonstrate through experiments in realistic household environments that agents trained with our LLM-guided bootstrapping procedure outperform those trained with naive bootstrapping as well as prior unsupervised skill acquisition methods on zero-shot execution of unseen, long-horizon tasks in new environments. Website at clvrai.com/boss.

{{</citation>}}


## cs.CV (33)



### (76/176) SoybeanNet: Transformer-Based Convolutional Neural Network for Soybean Pod Counting from Unmanned Aerial Vehicle (UAV) Images (Jiajia Li et al., 2023)

{{<citation>}}

Jiajia Li, Raju Thada Magar, Dong Chen, Feng Lin, Dechun Wang, Xiang Yin, Weichao Zhuang, Zhaojian Li. (2023)  
**SoybeanNet: Transformer-Based Convolutional Neural Network for Soybean Pod Counting from Unmanned Aerial Vehicle (UAV) Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.10861v1)  

---


**ABSTRACT**  
Soybeans are a critical source of food, protein and oil, and thus have received extensive research aimed at enhancing their yield, refining cultivation practices, and advancing soybean breeding techniques. Within this context, soybean pod counting plays an essential role in understanding and optimizing production. Despite recent advancements, the development of a robust pod-counting algorithm capable of performing effectively in real-field conditions remains a significant challenge This paper presents a pioneering work of accurate soybean pod counting utilizing unmanned aerial vehicle (UAV) images captured from actual soybean fields in Michigan, USA. Specifically, this paper presents SoybeanNet, a novel point-based counting network that harnesses powerful transformer backbones for simultaneous soybean pod counting and localization with high accuracy. In addition, a new dataset of UAV-acquired images for soybean pod counting was created and open-sourced, consisting of 113 drone images with more than 260k manually annotated soybean pods captured under natural lighting conditions. Through comprehensive evaluations, SoybeanNet demonstrated superior performance over five state-of-the-art approaches when tested on the collected images. Remarkably, SoybeanNet achieved a counting accuracy of $84.51\%$ when tested on the testing dataset, attesting to its efficacy in real-world scenarios. The publication also provides both the source code (\url{https://github.com/JiajiaLi04/Soybean-Pod-Counting-from-UAV-Images}) and the labeled soybean dataset (\url{https://www.kaggle.com/datasets/jiajiali/uav-based-soybean-pod-images}), offering a valuable resource for future research endeavors in soybean pod counting and related fields.

{{</citation>}}


### (77/176) LAMP: Learn A Motion Pattern for Few-Shot-Based Video Generation (Ruiqi Wu et al., 2023)

{{<citation>}}

Ruiqi Wu, Liangyu Chen, Tong Yang, Chunle Guo, Chongyi Li, Xiangyu Zhang. (2023)  
**LAMP: Learn A Motion Pattern for Few-Shot-Based Video Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.10769v1)  

---


**ABSTRACT**  
With the impressive progress in diffusion-based text-to-image generation, extending such powerful generative ability to text-to-video raises enormous attention. Existing methods either require large-scale text-video pairs and a large number of training resources or learn motions that are precisely aligned with template videos. It is non-trivial to balance a trade-off between the degree of generation freedom and the resource costs for video generation. In our study, we present a few-shot-based tuning framework, LAMP, which enables text-to-image diffusion model Learn A specific Motion Pattern with 8~16 videos on a single GPU. Specifically, we design a first-frame-conditioned pipeline that uses an off-the-shelf text-to-image model for content generation so that our tuned video diffusion model mainly focuses on motion learning. The well-developed text-to-image techniques can provide visually pleasing and diverse content as generation conditions, which highly improves video quality and generation freedom. To capture the features of temporal dimension, we expand the pretrained 2D convolution layers of the T2I model to our novel temporal-spatial motion learning layers and modify the attention blocks to the temporal level. Additionally, we develop an effective inference trick, shared-noise sampling, which can improve the stability of videos with computational costs. Our method can also be flexibly applied to other tasks, e.g. real-world image animation and video editing. Extensive experiments demonstrate that LAMP can effectively learn the motion pattern on limited data and generate high-quality videos. The code and models are available at https://rq-wu.github.io/projects/LAMP.

{{</citation>}}


### (78/176) BiomedJourney: Counterfactual Biomedical Image Generation by Instruction-Learning from Multimodal Patient Journeys (Yu Gu et al., 2023)

{{<citation>}}

Yu Gu, Jianwei Yang, Naoto Usuyama, Chunyuan Li, Sheng Zhang, Matthew P. Lungren, Jianfeng Gao, Hoifung Poon. (2023)  
**BiomedJourney: Counterfactual Biomedical Image Generation by Instruction-Learning from Multimodal Patient Journeys**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.10765v2)  

---


**ABSTRACT**  
Rapid progress has been made in instruction-learning for image editing with natural-language instruction, as exemplified by InstructPix2Pix. In biomedicine, such methods can be applied to counterfactual image generation, which helps differentiate causal structure from spurious correlation and facilitate robust image interpretation for disease progression modeling. However, generic image-editing models are ill-suited for the biomedical domain, and counterfactual biomedical image generation is largely underexplored. In this paper, we present BiomedJourney, a novel method for counterfactual biomedical image generation by instruction-learning from multimodal patient journeys. Given a patient with two biomedical images taken at different time points, we use GPT-4 to process the corresponding imaging reports and generate a natural language description of disease progression. The resulting triples (prior image, progression description, new image) are then used to train a latent diffusion model for counterfactual biomedical image generation. Given the relative scarcity of image time series data, we introduce a two-stage curriculum that first pretrains the denoising network using the much more abundant single image-report pairs (with dummy prior image), and then continues training using the counterfactual triples. Experiments using the standard MIMIC-CXR dataset demonstrate the promise of our method. In a comprehensive battery of tests on counterfactual medical image generation, BiomedJourney substantially outperforms prior state-of-the-art methods in instruction image editing and medical image generation such as InstructPix2Pix and RoentGen. To facilitate future study in counterfactual medical generation, we plan to release our instruction-learning code and pretrained models.

{{</citation>}}


### (79/176) IDRNet: Intervention-Driven Relation Network for Semantic Segmentation (Zhenchao Jin et al., 2023)

{{<citation>}}

Zhenchao Jin, Xiaowei Hu, Lingting Zhu, Luchuan Song, Li Yuan, Lequan Yu. (2023)  
**IDRNet: Intervention-Driven Relation Network for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.10755v1)  

---


**ABSTRACT**  
Co-occurrent visual patterns suggest that pixel relation modeling facilitates dense prediction tasks, which inspires the development of numerous context modeling paradigms, \emph{e.g.}, multi-scale-driven and similarity-driven context schemes. Despite the impressive results, these existing paradigms often suffer from inadequate or ineffective contextual information aggregation due to reliance on large amounts of predetermined priors. To alleviate the issues, we propose a novel \textbf{I}ntervention-\textbf{D}riven \textbf{R}elation \textbf{Net}work (\textbf{IDRNet}), which leverages a deletion diagnostics procedure to guide the modeling of contextual relations among different pixels. Specifically, we first group pixel-level representations into semantic-level representations with the guidance of pseudo labels and further improve the distinguishability of the grouped representations with a feature enhancement module. Next, a deletion diagnostics procedure is conducted to model relations of these semantic-level representations via perceiving the network outputs and the extracted relations are utilized to guide the semantic-level representations to interact with each other. Finally, the interacted representations are utilized to augment original pixel-level representations for final predictions. Extensive experiments are conducted to validate the effectiveness of IDRNet quantitatively and qualitatively. Notably, our intervention-driven context scheme brings consistent performance improvements to state-of-the-art segmentation frameworks and achieves competitive results on popular benchmark datasets, including ADE20K, COCO-Stuff, PASCAL-Context, LIP, and Cityscapes. Code is available at \url{https://github.com/SegmentationBLWX/sssegmentation}.

{{</citation>}}


### (80/176) A Survey on Video Diffusion Models (Zhen Xing et al., 2023)

{{<citation>}}

Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, Yu-Gang Jiang. (2023)  
**A Survey on Video Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.10647v1)  

---


**ABSTRACT**  
The recent wave of AI-generated content (AIGC) has witnessed substantial success in computer vision, with the diffusion model playing a crucial role in this achievement. Due to their impressive generative capabilities, diffusion models are gradually superseding methods based on GANs and auto-regressive Transformers, demonstrating exceptional performance not only in image generation and editing, but also in the realm of video-related research. However, existing surveys mainly focus on diffusion models in the context of image generation, with few up-to-date reviews on their application in the video domain. To address this gap, this paper presents a comprehensive review of video diffusion models in the AIGC era. Specifically, we begin with a concise introduction to the fundamentals and evolution of diffusion models. Subsequently, we present an overview of research on diffusion models in the video domain, categorizing the work into three key areas: video generation, video editing, and other video understanding tasks. We conduct a thorough review of the literature in these three key areas, including further categorization and practical contributions in the field. Finally, we discuss the challenges faced by research in this domain and outline potential future developmental trends. A comprehensive list of video diffusion models studied in this survey is available at https://github.com/ChenHsing/Awesome-Video-Diffusion-Models.

{{</citation>}}


### (81/176) LLM Blueprint: Enabling Text-to-Image Generation with Complex and Detailed Prompts (Hanan Gani et al., 2023)

{{<citation>}}

Hanan Gani, Shariq Farooq Bhat, Muzammal Naseer, Salman Khan, Peter Wonka. (2023)  
**LLM Blueprint: Enabling Text-to-Image Generation with Complex and Detailed Prompts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10640v1)  

---


**ABSTRACT**  
Diffusion-based generative models have significantly advanced text-to-image generation but encounter challenges when processing lengthy and intricate text prompts describing complex scenes with multiple objects. While excelling in generating images from short, single-object descriptions, these models often struggle to faithfully capture all the nuanced details within longer and more elaborate textual inputs. In response, we present a novel approach leveraging Large Language Models (LLMs) to extract critical components from text prompts, including bounding box coordinates for foreground objects, detailed textual descriptions for individual objects, and a succinct background context. These components form the foundation of our layout-to-image generation model, which operates in two phases. The initial Global Scene Generation utilizes object layouts and background context to create an initial scene but often falls short in faithfully representing object characteristics as specified in the prompts. To address this limitation, we introduce an Iterative Refinement Scheme that iteratively evaluates and refines box-level content to align them with their textual descriptions, recomposing objects as needed to ensure consistency. Our evaluation on complex prompts featuring multiple objects demonstrates a substantial improvement in recall compared to baseline diffusion models. This is further validated by a user study, underscoring the efficacy of our approach in generating coherent and detailed scenes from intricate textual inputs.

{{</citation>}}


### (82/176) Motion2Language, Unsupervised learning of synchronized semantic motion segmentation (Karim Radouane et al., 2023)

{{<citation>}}

Karim Radouane, Andon Tchechmedjiev, Sylvie Ranwez, Julien Lagarde. (2023)  
**Motion2Language, Unsupervised learning of synchronized semantic motion segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2310.10594v1)  

---


**ABSTRACT**  
In this paper, we investigate building a sequence to sequence architecture for motion to language translation and synchronization. The aim is to translate motion capture inputs into English natural-language descriptions, such that the descriptions are generated synchronously with the actions performed, enabling semantic segmentation as a byproduct, but without requiring synchronized training data. We propose a new recurrent formulation of local attention that is suited for synchronous/live text generation, as well as an improved motion encoder architecture better suited to smaller data and for synchronous generation. We evaluate both contributions in individual experiments, using the standard BLEU4 metric, as well as a simple semantic equivalence measure, on the KIT motion language dataset. In a follow-up experiment, we assess the quality of the synchronization of generated text in our proposed approaches through multiple evaluation metrics. We find that both contributions to the attention mechanism and the encoder architecture additively improve the quality of generated text (BLEU and semantic equivalence), but also of synchronization. Our code will be made available at \url{https://github.com/rd20karim/M2T-Segmentation/tree/main}

{{</citation>}}


### (83/176) BiLL-VTG: Bridging Large Language Models and Lightweight Visual Tools for Video-based Texts Generation (Ji Qi et al., 2023)

{{<citation>}}

Ji Qi, Kaixuan Ji, Jifan Yu, Duokang Wang, Bin Xu, Lei Hou, Juanzi Li. (2023)  
**BiLL-VTG: Bridging Large Language Models and Lightweight Visual Tools for Video-based Texts Generation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10586v1)  

---


**ABSTRACT**  
Building models that generate textual responses to user instructions for videos is a practical and challenging topic, as it requires both vision understanding and knowledge reasoning. Compared to language and image modalities, training efficiency remains a serious problem as existing studies train models on massive sparse videos aligned with brief descriptions. In this paper, we introduce BiLL-VTG, a fast adaptive framework that leverages large language models (LLMs) to reasoning on videos based on essential lightweight visual tools. Specifically, we reveal the key to response specific instructions is the concentration on relevant video events, and utilize two visual tools of structured scene graph generation and descriptive image caption generation to gather and represent the events information. Thus, a LLM equipped with world knowledge is adopted as the reasoning agent to achieve the response by performing multiple reasoning steps on specified video events.To address the difficulty of specifying events from agent, we further propose an Instruction-oriented Video Events Recognition (InsOVER) algorithm based on the efficient Hungarian matching to localize corresponding video events using linguistic instructions, enabling LLMs to interact with long videos. Extensive experiments on two typical video-based texts generations tasks show that our tuning-free framework outperforms the pre-trained models including Flamingo-80B, to achieve the state-of-the-art performance.

{{</citation>}}


### (84/176) RefConv: Re-parameterized Refocusing Convolution for Powerful ConvNets (Zhicheng Cai et al., 2023)

{{<citation>}}

Zhicheng Cai, Xiaohan Ding, Qiu Shen, Xun Cao. (2023)  
**RefConv: Re-parameterized Refocusing Convolution for Powerful ConvNets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.10563v1)  

---


**ABSTRACT**  
We propose Re-parameterized Refocusing Convolution (RefConv) as a replacement for regular convolutional layers, which is a plug-and-play module to improve the performance without any inference costs. Specifically, given a pre-trained model, RefConv applies a trainable Refocusing Transformation to the basis kernels inherited from the pre-trained model to establish connections among the parameters. For example, a depth-wise RefConv can relate the parameters of a specific channel of convolution kernel to the parameters of the other kernel, i.e., make them refocus on the other parts of the model they have never attended to, rather than focus on the input features only. From another perspective, RefConv augments the priors of existing model structures by utilizing the representations encoded in the pre-trained parameters as the priors and refocusing on them to learn novel representations, thus further enhancing the representational capacity of the pre-trained model. Experimental results validated that RefConv can improve multiple CNN-based models by a clear margin on image classification (up to 1.47% higher top-1 accuracy on ImageNet), object detection and semantic segmentation without introducing any extra inference costs or altering the original model structure. Further studies demonstrated that RefConv can reduce the redundancy of channels and smooth the loss landscape, which explains its effectiveness.

{{</citation>}}


### (85/176) Unifying Image Processing as Visual Prompting Question Answering (Yihao Liu et al., 2023)

{{<citation>}}

Yihao Liu, Xiangyu Chen, Xianzheng Ma, Xintao Wang, Jiantao Zhou, Yu Qiao, Chao Dong. (2023)  
**Unifying Image Processing as Visual Prompting Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: NLP, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.10513v1)  

---


**ABSTRACT**  
Image processing is a fundamental task in computer vision, which aims at enhancing image quality and extracting essential features for subsequent vision applications. Traditionally, task-specific models are developed for individual tasks and designing such models requires distinct expertise. Building upon the success of large language models (LLMs) in natural language processing (NLP), there is a similar trend in computer vision, which focuses on developing large-scale models through pretraining and in-context learning. This paradigm shift reduces the reliance on task-specific models, yielding a powerful unified model to deal with various tasks. However, these advances have predominantly concentrated on high-level vision tasks, with less attention paid to low-level vision tasks. To address this issue, we propose a universal model for general image processing that covers image restoration, image enhancement, image feature extraction tasks, \textit{etc}. Our proposed framework, named PromptGIP, unifies these diverse image processing tasks within a universal framework. Inspired by NLP question answering (QA) techniques, we employ a visual prompting question answering paradigm. Specifically, we treat the input-output image pair as a structured question-answer sentence, thereby reprogramming the image processing task as a prompting QA problem. PromptGIP can undertake diverse \textbf{cross-domain} tasks using provided visual prompts, eliminating the need for task-specific finetuning. Our methodology offers a universal and adaptive solution to general image processing. While PromptGIP has demonstrated a certain degree of out-of-domain task generalization capability, further research is expected to fully explore its more powerful emergent generalization.

{{</citation>}}


### (86/176) On the Transferability of Learning Models for Semantic Segmentation for Remote Sensing Data (Rongjun Qin et al., 2023)

{{<citation>}}

Rongjun Qin, Guixiang Zhang, Yang Tang. (2023)  
**On the Transferability of Learning Models for Semantic Segmentation for Remote Sensing Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.10490v1)  

---


**ABSTRACT**  
Recent deep learning-based methods outperform traditional learning methods on remote sensing (RS) semantic segmentation/classification tasks. However, they require large training datasets and are generally known for lack of transferability due to the highly disparate RS image content across different geographical regions. Yet, there is no comprehensive analysis of their transferability, i.e., to which extent a model trained on a source domain can be readily applicable to a target domain. Therefore, in this paper, we aim to investigate the raw transferability of traditional and deep learning (DL) models, as well as the effectiveness of domain adaptation (DA) approaches in enhancing the transferability of the DL models (adapted transferability). By utilizing four highly diverse RS datasets, we train six models with and without three DA approaches to analyze their transferability between these datasets quantitatively. Furthermore, we developed a straightforward method to quantify the transferability of a model using the spectral indices as a medium and have demonstrated its effectiveness in evaluating the model transferability at the target domain when the labels are unavailable. Our experiments yield several generally important yet not well-reported observations regarding the raw and adapted transferability. Moreover, our proposed label-free transferability assessment method is validated to be better than posterior model confidence. The findings can guide the future development of generalized RS learning models. The trained models are released under this link: https://github.com/GDAOSU/Transferability-Remote-Sensing

{{</citation>}}


### (87/176) Object Detection in Aerial Images in Scarce Data Regimes (Pierre Le Jeune, 2023)

{{<citation>}}

Pierre Le Jeune. (2023)  
**Object Detection in Aerial Images in Scarce Data Regimes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Few-Shot, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.10433v1)  

---


**ABSTRACT**  
Most contributions on Few-Shot Object Detection (FSOD) evaluate their methods on natural images only, yet the transferability of the announced performance is not guaranteed for applications on other kinds of images. We demonstrate this with an in-depth analysis of existing FSOD methods on aerial images and observed a large performance gap compared to natural images. Small objects, more numerous in aerial images, are the cause for the apparent performance gap between natural and aerial images. As a consequence, we improve FSOD performance on small objects with a carefully designed attention mechanism. In addition, we also propose a scale-adaptive box similarity criterion, that improves the training and evaluation of FSOD methods, particularly for small objects. We also contribute to generic FSOD with two distinct approaches based on metric learning and fine-tuning. Impressive results are achieved with the fine-tuning method, which encourages tackling more complex scenarios such as Cross-Domain FSOD. We conduct preliminary experiments in this direction and obtain promising results. Finally, we address the deployment of the detection models inside COSE's systems. Detection must be done in real-time in extremely large images (more than 100 megapixels), with limited computation power. Leveraging existing optimization tools such as TensorRT, we successfully tackle this engineering challenge.

{{</citation>}}


### (88/176) LLM4SGG: Large Language Model for Weakly Supervised Scene Graph Generation (Kibum Kim et al., 2023)

{{<citation>}}

Kibum Kim, Kanghoon Yoon, Jaehyeong Jeon, Yeonjun In, Jinyoung Moon, Donghyun Kim, Chanyoung Park. (2023)  
**LLM4SGG: Large Language Model for Weakly Supervised Scene Graph Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.10404v3)  

---


**ABSTRACT**  
Weakly-Supervised Scene Graph Generation (WSSGG) research has recently emerged as an alternative to the fully-supervised approach that heavily relies on costly annotations. In this regard, studies on WSSGG have utilized image captions to obtain unlocalized triplets while primarily focusing on grounding the unlocalized triplets over image regions. However, they have overlooked the two issues involved in the triplet formation process from the captions: 1) Semantic over-simplification issue arises when extracting triplets from captions, where fine-grained predicates in captions are undesirably converted into coarse-grained predicates, resulting in a long-tailed predicate distribution, and 2) Low-density scene graph issue arises when aligning the triplets in the caption with entity/predicate classes of interest, where many triplets are discarded and not used in training, leading to insufficient supervision. To tackle the two issues, we propose a new approach, i.e., Large Language Model for weakly-supervised SGG (LLM4SGG), where we mitigate the two issues by leveraging the LLM's in-depth understanding of language and reasoning ability during the extraction of triplets from captions and alignment of entity/predicate classes with target data. To further engage the LLM in these processes, we adopt the idea of Chain-of-Thought and the in-context few-shot learning strategy. To validate the effectiveness of LLM4SGG, we conduct extensive experiments on Visual Genome and GQA datasets, showing significant improvements in both Recall@K and mean Recall@K compared to the state-of-the-art WSSGG methods. A further appeal is that LLM4SGG is data-efficient, enabling effective model training with a small amount of training images.

{{</citation>}}


### (89/176) Towards Open World Active Learning for 3D Object Detection (Zhuoxiao Chen et al., 2023)

{{<citation>}}

Zhuoxiao Chen, Yadan Luo, Zixin Wang, Zijian Wang, Xin Yu, Zi Huang. (2023)  
**Towards Open World Active Learning for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Active Learning, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.10391v1)  

---


**ABSTRACT**  
Significant strides have been made in closed world 3D object detection, testing systems in environments with known classes. However, the challenge arises in open world scenarios where new object classes appear. Existing efforts sequentially learn novel classes from streams of labeled data at a significant annotation cost, impeding efficient deployment to the wild. To seek effective solutions, we investigate a more practical yet challenging research task: Open World Active Learning for 3D Object Detection (OWAL-3D), aiming at selecting a small number of 3D boxes to annotate while maximizing detection performance on both known and unknown classes. The core difficulty centers on striking a balance between mining more unknown instances and minimizing the labeling expenses of point clouds. Empirically, our study finds the harmonious and inverse relationship between box quantities and their confidences can help alleviate the dilemma, avoiding the repeated selection of common known instances and focusing on uncertain objects that are potentially unknown. We unify both relational constraints into a simple and effective AL strategy namely OpenCRB, which guides to acquisition of informative point clouds with the least amount of boxes to label. Furthermore, we develop a comprehensive codebase for easy reproducing and future research, supporting 15 baseline methods (i.e., active learning, out-of-distribution detection and open world detection), 2 types of modern 3D detectors (i.e., one-stage SECOND and two-stage PV-RCNN) and 3 benchmark 3D datasets (i.e., KITTI, nuScenes and Waymo). Extensive experiments evidence that the proposed Open-CRB demonstrates superiority and flexibility in recognizing both novel and shared categories with very limited labeling costs, compared to state-of-the-art baselines.

{{</citation>}}


### (90/176) GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers (Takeru Miyato et al., 2023)

{{<citation>}}

Takeru Miyato, Bernhard Jaeger, Max Welling, Andreas Geiger. (2023)  
**GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, stat-ML  
Keywords: Attention, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.10375v1)  

---


**ABSTRACT**  
As transformers are equivariant to the permutation of input tokens, encoding the positional information of tokens is necessary for many tasks. However, since existing positional encoding schemes have been initially designed for NLP tasks, their suitability for vision tasks, which typically exhibit different structural properties in their data, is questionable. We argue that existing positional encoding schemes are suboptimal for 3D vision tasks, as they do not respect their underlying 3D geometric structure. Based on this hypothesis, we propose a geometry-aware attention mechanism that encodes the geometric structure of tokens as relative transformation determined by the geometric relationship between queries and key-value pairs. By evaluating on multiple novel view synthesis (NVS) datasets in the sparse wide-baseline multi-view setting, we show that our attention, called Geometric Transform Attention (GTA), improves learning efficiency and performance of state-of-the-art transformer-based NVS models without any additional learned parameters and only minor computational overhead.

{{</citation>}}


### (91/176) Multimodal Object Query Initialization for 3D Object Detection (Mathijs R. van Geerenstein et al., 2023)

{{<citation>}}

Mathijs R. van Geerenstein, Felicia Ruppel, Klaus Dietmayer, Dariu M. Gavrila. (2023)  
**Multimodal Object Query Initialization for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.10353v1)  

---


**ABSTRACT**  
3D object detection models that exploit both LiDAR and camera sensor features are top performers in large-scale autonomous driving benchmarks. A transformer is a popular network architecture used for this task, in which so-called object queries act as candidate objects. Initializing these object queries based on current sensor inputs is a common practice. For this, existing methods strongly rely on LiDAR data however, and do not fully exploit image features. Besides, they introduce significant latency. To overcome these limitations we propose EfficientQ3M, an efficient, modular, and multimodal solution for object query initialization for transformer-based 3D object detection models. The proposed initialization method is combined with a "modality-balanced" transformer decoder where the queries can access all sensor modalities throughout the decoder. In experiments, we outperform the state of the art in transformer-based LiDAR object detection on the competitive nuScenes benchmark and showcase the benefits of input-dependent multimodal query initialization, while being more efficient than the available alternatives for LiDAR-camera initialization. The proposed method can be applied with any combination of sensor modalities as input, demonstrating its modularity.

{{</citation>}}


### (92/176) Semi-Supervised Crowd Counting with Contextual Modeling: Facilitating Holistic Understanding of Crowd Scenes (Yifei Qian et al., 2023)

{{<citation>}}

Yifei Qian, Xiaopeng Hong, Ognjen Arandjelović, Zhongliang Guo, Carl R. Donovan. (2023)  
**Semi-Supervised Crowd Counting with Contextual Modeling: Facilitating Holistic Understanding of Crowd Scenes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.10352v1)  

---


**ABSTRACT**  
To alleviate the heavy annotation burden for training a reliable crowd counting model and thus make the model more practicable and accurate by being able to benefit from more data, this paper presents a new semi-supervised method based on the mean teacher framework. When there is a scarcity of labeled data available, the model is prone to overfit local patches. Within such contexts, the conventional approach of solely improving the accuracy of local patch predictions through unlabeled data proves inadequate. Consequently, we propose a more nuanced approach: fostering the model's intrinsic 'subitizing' capability. This ability allows the model to accurately estimate the count in regions by leveraging its understanding of the crowd scenes, mirroring the human cognitive process. To achieve this goal, we apply masking on unlabeled data, guiding the model to make predictions for these masked patches based on the holistic cues. Furthermore, to help with feature learning, herein we incorporate a fine-grained density classification task. Our method is general and applicable to most existing crowd counting methods as it doesn't have strict structural or loss constraints. In addition, we observe that the model trained with our framework exhibits a 'subitizing'-like behavior. It accurately predicts low-density regions with only a 'glance', while incorporating local details to predict high-density regions. Our method achieves the state-of-the-art performance, surpassing previous approaches by a large margin on challenging benchmarks such as ShanghaiTech A and UCF-QNRF. The code is available at: https://github.com/cha15yq/MRC-Crowd.

{{</citation>}}


### (93/176) Scene Graph Conditioning in Latent Diffusion (Frank Fundel, 2023)

{{<citation>}}

Frank Fundel. (2023)  
**Scene Graph Conditioning in Latent Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2310.10338v1)  

---


**ABSTRACT**  
Diffusion models excel in image generation but lack detailed semantic control using text prompts. Additional techniques have been developed to address this limitation. However, conditioning diffusion models solely on text-based descriptions is challenging due to ambiguity and lack of structure. In contrast, scene graphs offer a more precise representation of image content, making them superior for fine-grained control and accurate synthesis in image generation models. The amount of image and scene-graph data is sparse, which makes fine-tuning large diffusion models challenging. We propose multiple approaches to tackle this problem using ControlNet and Gated Self-Attention. We were able to show that using out proposed methods it is possible to generate images from scene graphs with much higher quality, outperforming previous methods. Our source code is publicly available on https://github.com/FrankFundel/SGCond

{{</citation>}}


### (94/176) Towards Open-World Co-Salient Object Detection with Generative Uncertainty-aware Group Selective Exchange-Masking (Yang Wu et al., 2023)

{{<citation>}}

Yang Wu, Shenglong Hu, Huihui Song, Kaihua Zhang, Bo Liu, Dong Liu. (2023)  
**Towards Open-World Co-Salient Object Detection with Generative Uncertainty-aware Group Selective Exchange-Masking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.10264v1)  

---


**ABSTRACT**  
The traditional definition of co-salient object detection (CoSOD) task is to segment the common salient objects in a group of relevant images. This definition is based on an assumption of group consensus consistency that is not always reasonable in the open-world setting, which results in robustness issue in the model when dealing with irrelevant images in the inputting image group under the open-word scenarios. To tackle this problem, we introduce a group selective exchange-masking (GSEM) approach for enhancing the robustness of the CoSOD model. GSEM takes two groups of images as input, each containing different types of salient objects. Based on the mixed metric we designed, GSEM selects a subset of images from each group using a novel learning-based strategy, then the selected images are exchanged. To simultaneously consider the uncertainty introduced by irrelevant images and the consensus features of the remaining relevant images in the group, we designed a latent variable generator branch and CoSOD transformer branch. The former is composed of a vector quantised-variational autoencoder to generate stochastic global variables that model uncertainty. The latter is designed to capture correlation-based local features that include group consensus. Finally, the outputs of the two branches are merged and passed to a transformer-based decoder to generate robust predictions. Taking into account that there are currently no benchmark datasets specifically designed for open-world scenarios, we constructed three open-world benchmark datasets, namely OWCoSal, OWCoSOD, and OWCoCA, based on existing datasets. By breaking the group-consistency assumption, these datasets provide effective simulations of real-world scenarios and can better evaluate the robustness and practicality of models.

{{</citation>}}


### (95/176) Mask wearing object detection algorithm based on improved YOLOv5 (Peng Wen et al., 2023)

{{<citation>}}

Peng Wen, Junhu Zhang, Haitao Li. (2023)  
**Mask wearing object detection algorithm based on improved YOLOv5**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, stat-ML  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.10245v1)  

---


**ABSTRACT**  
Wearing a mask is one of the important measures to prevent infectious diseases. However, it is difficult to detect people's mask-wearing situation in public places with high traffic flow. To address the above problem, this paper proposes a mask-wearing face detection model based on YOLOv5l. Firstly, Multi-Head Attentional Self-Convolution not only improves the convergence speed of the model but also enhances the accuracy of the model detection. Secondly, the introduction of Swin Transformer Block is able to extract more useful feature information, enhance the detection ability of small targets, and improve the overall accuracy of the model. Our designed I-CBAM module can improve target detection accuracy. In addition, using enhanced feature fusion enables the model to better adapt to object detection tasks of different scales. In the experimentation on the MASK dataset, the results show that the model proposed in this paper achieved a 1.1% improvement in mAP(0.5) and a 1.3% improvement in mAP(0.5:0.95) compared to the YOLOv5l model. Our proposed method significantly enhances the detection capability of mask-wearing.

{{</citation>}}


### (96/176) MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete Representations (Heyuan Yao et al., 2023)

{{<citation>}}

Heyuan Yao, Zhenhua Song, Yuyang Zhou, Tenglong Ao, Baoquan Chen, Libin Liu. (2023)  
**MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete Representations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.10198v2)  

---


**ABSTRACT**  
In this work, we present MoConVQ, a novel unified framework for physics-based motion control leveraging scalable discrete representations. Building upon vector quantized variational autoencoders (VQ-VAE) and model-based reinforcement learning, our approach effectively learns motion embeddings from a large, unstructured dataset spanning tens of hours of motion examples. The resultant motion representation not only captures diverse motion skills but also offers a robust and intuitive interface for various applications. We demonstrate the versatility of MoConVQ through several applications: universal tracking control from various motion sources, interactive character control with latent motion representations using supervised learning, physics-based motion generation from natural language descriptions using the GPT framework, and, most interestingly, seamless integration with large language models (LLMs) with in-context learning to tackle complex and abstract tasks.

{{</citation>}}


### (97/176) The Road to On-board Change Detection: A Lightweight Patch-Level Change Detection Network via Exploring the Potential of Pruning and Pooling (Lihui Xue et al., 2023)

{{<citation>}}

Lihui Xue, Zhihao Wang, Xueqian Wang, Gang Li. (2023)  
**The Road to On-board Change Detection: A Lightweight Patch-Level Change Detection Network via Exploring the Potential of Pruning and Pooling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2310.10166v1)  

---


**ABSTRACT**  
Existing satellite remote sensing change detection (CD) methods often crop original large-scale bi-temporal image pairs into small patch pairs and then use pixel-level CD methods to fairly process all the patch pairs. However, due to the sparsity of change in large-scale satellite remote sensing images, existing pixel-level CD methods suffer from a waste of computational cost and memory resources on lots of unchanged areas, which reduces the processing efficiency of on-board platform with extremely limited computation and memory resources. To address this issue, we propose a lightweight patch-level CD network (LPCDNet) to rapidly remove lots of unchanged patch pairs in large-scale bi-temporal image pairs. This is helpful to accelerate the subsequent pixel-level CD processing stage and reduce its memory costs. In our LPCDNet, a sensitivity-guided channel pruning method is proposed to remove unimportant channels and construct the lightweight backbone network on basis of ResNet18 network. Then, the multi-layer feature compression (MLFC) module is designed to compress and fuse the multi-level feature information of bi-temporal image patch. The output of MLFC module is fed into the fully-connected decision network to generate the predicted binary label. Finally, a weighted cross-entropy loss is utilized in the training process of network to tackle the change/unchange class imbalance problem. Experiments on two CD datasets demonstrate that our LPCDNet achieves more than 1000 frames per second on an edge computation platform, i.e., NVIDIA Jetson AGX Orin, which is more than 3 times that of the existing methods without noticeable CD performance loss. In addition, our method reduces more than 60% memory costs of the subsequent pixel-level CD processing stage.

{{</citation>}}


### (98/176) Recursive Segmentation Living Image: An eXplainable AI (XAI) Approach for Computing Structural Beauty of Images or the Livingness of Space (Yao Qianxiang et al., 2023)

{{<citation>}}

Yao Qianxiang, Bin Jiang. (2023)  
**Recursive Segmentation Living Image: An eXplainable AI (XAI) Approach for Computing Structural Beauty of Images or the Livingness of Space**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10149v1)  

---


**ABSTRACT**  
This study introduces the concept of "structural beauty" as an objective computational approach for evaluating the aesthetic appeal of images. Through the utilization of the Segment anything model (SAM), we propose a method that leverages recursive segmentation to extract finer-grained substructures. Additionally, by reconstructing the hierarchical structure, we obtain a more accurate representation of substructure quantity and hierarchy. This approach reproduces and extends our previous research, allowing for the simultaneous assessment of Livingness in full-color images without the need for grayscale conversion or separate computations for foreground and background Livingness. Furthermore, the application of our method to the Scenic or Not dataset, a repository of subjective scenic ratings, demonstrates a high degree of consistency with subjective ratings in the 0-6 score range. This underscores that structural beauty is not solely a subjective perception, but a quantifiable attribute accessible through objective computation. Through our case studies, we have arrived at three significant conclusions. 1) our method demonstrates the capability to accurately segment meaningful objects, including trees, buildings, and windows, as well as abstract substructures within paintings. 2) we observed that the clarity of an image impacts our computational results; clearer images tend to yield higher Livingness scores. However, for equally blurry images, Livingness does not exhibit a significant reduction, aligning with human visual perception. 3) our approach fundamentally differs from methods employing Convolutional Neural Networks (CNNs) for predicting image scores. Our method not only provides computational results but also offers transparency and interpretability, positioning it as a novel avenue in the realm of Explainable AI (XAI).

{{</citation>}}


### (99/176) A Search for Prompts: Generating Structured Answers from Contracts (Adam Roegiest et al., 2023)

{{<citation>}}

Adam Roegiest, Radha Chitta, Jonathan Donnelly, Maya Lash, Alexandra Vtyurina, François Longtin. (2023)  
**A Search for Prompts: Generating Structured Answers from Contracts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2310.10141v1)  

---


**ABSTRACT**  
In many legal processes being able to action on the concrete implication of a legal question can be valuable to automating human review or signalling certain conditions (e.g., alerts around automatic renewal). To support such tasks, we present a form of legal question answering that seeks to return one (or more) fixed answers for a question about a contract clause. After showing that unstructured generative question answering can have questionable outcomes for such a task, we discuss our exploration methodology for legal question answering prompts using OpenAI's \textit{GPT-3.5-Turbo} and provide a summary of insights.   Using insights gleaned from our qualitative experiences, we compare our proposed template prompts against a common semantic matching approach and find that our prompt templates are far more accurate despite being less reliable in the exact response return. With some additional tweaks to prompts and the use of in-context learning, we are able to further improve the performance of our proposed strategy while maximizing the reliability of responses as best we can.

{{</citation>}}


### (100/176) PELA: Learning Parameter-Efficient Models with Low-Rank Approximation (Yangyang Guo et al., 2023)

{{<citation>}}

Yangyang Guo, Guangzhi Wang, Mohan Kankanhalli. (2023)  
**PELA: Learning Parameter-Efficient Models with Low-Rank Approximation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.10700v1)  

---


**ABSTRACT**  
Applying a pre-trained large model to downstream tasks is prohibitive under resource-constrained conditions. Recent dominant approaches for addressing efficiency issues involve adding a few learnable parameters to the fixed backbone model. This strategy, however, leads to more challenges in loading large models for downstream fine-tuning with limited resources. In this paper, we propose a novel method for increasing the parameter efficiency of pre-trained models by introducing an intermediate pre-training stage. To this end, we first employ low-rank approximation to compress the original large model and then devise a feature distillation module and a weight perturbation regularization module. These modules are specifically designed to enhance the low-rank model. Concretely, we update only the low-rank model while freezing the backbone parameters during pre-training. This allows for direct and efficient utilization of the low-rank model for downstream tasks. The proposed method achieves both efficiencies in terms of required parameters and computation time while maintaining comparable results with minimal modifications to the base architecture. Specifically, when applied to three vision-only and one vision-language Transformer models, our approach often demonstrates a $\sim$0.6 point decrease in performance while reducing the original parameter size by 1/3 to 2/3.

{{</citation>}}


### (101/176) Few-shot Action Recognition with Captioning Foundation Models (Xiang Wang et al., 2023)

{{<citation>}}

Xiang Wang, Shiwei Zhang, Hangjie Yuan, Yingya Zhang, Changxin Gao, Deli Zhao, Nong Sang. (2023)  
**Few-shot Action Recognition with Captioning Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.10125v1)  

---


**ABSTRACT**  
Transferring vision-language knowledge from pretrained multimodal foundation models to various downstream tasks is a promising direction. However, most current few-shot action recognition methods are still limited to a single visual modality input due to the high cost of annotating additional textual descriptions. In this paper, we develop an effective plug-and-play framework called CapFSAR to exploit the knowledge of multimodal models without manually annotating text. To be specific, we first utilize a captioning foundation model (i.e., BLIP) to extract visual features and automatically generate associated captions for input videos. Then, we apply a text encoder to the synthetic captions to obtain representative text embeddings. Finally, a visual-text aggregation module based on Transformer is further designed to incorporate cross-modal spatio-temporal complementary information for reliable few-shot matching. In this way, CapFSAR can benefit from powerful multimodal knowledge of pretrained foundation models, yielding more comprehensive classification in the low-shot regime. Extensive experiments on multiple standard few-shot benchmarks demonstrate that the proposed CapFSAR performs favorably against existing methods and achieves state-of-the-art performance. The code will be made publicly available.

{{</citation>}}


### (102/176) AutoDIR: Automatic All-in-One Image Restoration with Latent Diffusion (Yitong Jiang et al., 2023)

{{<citation>}}

Yitong Jiang, Zhaoyang Zhang, Tianfan Xue, Jinwei Gu. (2023)  
**AutoDIR: Automatic All-in-One Image Restoration with Latent Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2310.10123v2)  

---


**ABSTRACT**  
In this paper, we aim to solve complex real-world image restoration situations, in which, one image may have a variety of unknown degradations. To this end, we propose an all-in-one image restoration framework with latent diffusion (AutoDIR), which can automatically detect and address multiple unknown degradations. Our framework first utilizes a Blind Image Quality Assessment Module (BIQA) to automatically detect and identify the unknown dominant image degradation type of the image. Then, an All-in-One Image Refinement (AIR) Module handles multiple kinds of degradation image restoration with the guidance of BIQA. Finally, a Structure Correction Module (SCM) is proposed to recover the image details distorted by AIR. Our comprehensive evaluation demonstrates that AutoDIR outperforms state-of-the-art approaches by achieving superior restoration results while supporting a wider range of tasks. Notably, AutoDIR is also the first method to automatically handle real-scenario images with multiple unknown degradations.

{{</citation>}}


### (103/176) A computational model of serial and parallel processing in visual search (Rachel F. Heaton, 2023)

{{<citation>}}

Rachel F. Heaton. (2023)  
**A computational model of serial and parallel processing in visual search**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-SC, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.10061v1)  

---


**ABSTRACT**  
The following is a dissertation aimed at understanding what the various phenomena in visual search teach us about the nature of human visual representations and processes. I first review some of the major empirical findings in the study of visual search. I next present a theory of visual search in terms of what I believe these findings suggest about the representations and processes underlying ventral visual processing. These principles are instantiated in a computational model called CASPER (Concurrent Attention: Serial and Parallel Evaluation with Relations), originally developed by Hummel, that I have adapted to account for a range of phenomena in visual search. I then describe an extension of the CASPER model to account for our ability to search for visual items defined not simply by the features composing those items but by the spatial relations among those features. Seven experiments (four main experiments and three replications) are described that test CASPER's predictions about relational search. Finally, I evaluate the fit between CASPER's predictions and the empirical findings and show with three additional simulations that CASPER can account for negative acceleration in search functions for relational stimuli if one postulates that the visual system is leveraging an emergent feature that bypasses relational processing.

{{</citation>}}


### (104/176) EfficientOCR: An Extensible, Open-Source Package for Efficiently Digitizing World Knowledge (Tom Bryan et al., 2023)

{{<citation>}}

Tom Bryan, Jacob Carlson, Abhishek Arora, Melissa Dell. (2023)  
**EfficientOCR: An Extensible, Open-Source Package for Efficiently Digitizing World Knowledge**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV, econ-GN, q-fin-EC  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2310.10050v1)  

---


**ABSTRACT**  
Billions of public domain documents remain trapped in hard copy or lack an accurate digitization. Modern natural language processing methods cannot be used to index, retrieve, and summarize their texts; conduct computational textual analyses; or extract information for statistical analyses, and these texts cannot be incorporated into language model training. Given the diversity and sheer quantity of public domain texts, liberating them at scale requires optical character recognition (OCR) that is accurate, extremely cheap to deploy, and sample-efficient to customize to novel collections, languages, and character sets. Existing OCR engines, largely designed for small-scale commercial applications in high resource languages, often fall short of these requirements. EffOCR (EfficientOCR), a novel open-source OCR package, meets both the computational and sample efficiency requirements for liberating texts at scale by abandoning the sequence-to-sequence architecture typically used for OCR, which takes representations from a learned vision model as inputs to a learned language model. Instead, EffOCR models OCR as a character or word-level image retrieval problem. EffOCR is cheap and sample efficient to train, as the model only needs to learn characters' visual appearance and not how they are used in sequence to form language. Models in the EffOCR model zoo can be deployed off-the-shelf with only a few lines of code. Importantly, EffOCR also allows for easy, sample efficient customization with a simple model training interface and minimal labeling requirements due to its sample efficiency. We illustrate the utility of EffOCR by cheaply and accurately digitizing 20 million historical U.S. newspaper scans, evaluating zero-shot performance on randomly selected documents from the U.S. National Archives, and accurately digitizing Japanese documents for which all other OCR solutions failed.

{{</citation>}}


### (105/176) Smart City Transportation: Deep Learning Ensemble Approach for Traffic Accident Detection (Victor Adewopo et al., 2023)

{{<citation>}}

Victor Adewopo, Nelly Elsayed. (2023)  
**Smart City Transportation: Deep Learning Ensemble Approach for Traffic Accident Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.10038v1)  

---


**ABSTRACT**  
The dynamic and unpredictable nature of road traffic necessitates effective accident detection methods for enhancing safety and streamlining traffic management in smart cities. This paper offers a comprehensive exploration study of prevailing accident detection techniques, shedding light on the nuances of other state-of-the-art methodologies while providing a detailed overview of distinct traffic accident types like rear-end collisions, T-bone collisions, and frontal impact accidents. Our novel approach introduces the I3D-CONVLSTM2D model architecture, a lightweight solution tailored explicitly for accident detection in smart city traffic surveillance systems by integrating RGB frames with optical flow information. Our experimental study's empirical analysis underscores our approach's efficacy, with the I3D-CONVLSTM2D RGB + Optical-Flow (Trainable) model outperforming its counterparts, achieving an impressive 87\% Mean Average Precision (MAP). Our findings further elaborate on the challenges posed by data imbalances, particularly when working with a limited number of datasets, road structures, and traffic scenarios. Ultimately, our research illuminates the path towards a sophisticated vision-based accident detection system primed for real-time integration into edge IoT devices within smart urban infrastructures.

{{</citation>}}


### (106/176) Black-box Targeted Adversarial Attack on Segment Anything (SAM) (Sheng Zheng et al., 2023)

{{<citation>}}

Sheng Zheng, Chaoning Zhang. (2023)  
**Black-box Targeted Adversarial Attack on Segment Anything (SAM)**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2310.10010v1)  

---


**ABSTRACT**  
Deep recognition models are widely vulnerable to adversarial examples, which change the model output by adding quasi-imperceptible perturbation to the image input. Recently, Segment Anything Model (SAM) has emerged to become a popular foundation model in computer vision due to its impressive generalization to unseen data and tasks. Realizing flexible attacks on SAM is beneficial for understanding the robustness of SAM in the adversarial context. To this end, this work aims to achieve a targeted adversarial attack (TAA) on SAM. Specifically, under a certain prompt, the goal is to make the predicted mask of an adversarial example resemble that of a given target image. The task of TAA on SAM has been realized in a recent arXiv work in the white-box setup by assuming access to prompt and model, which is thus less practical. To address the issue of prompt dependence, we propose a simple yet effective approach by only attacking the image encoder. Moreover, we propose a novel regularization loss to enhance the cross-model transferability by increasing the feature dominance of adversarial images over random natural images. Extensive experiments verify the effectiveness of our proposed simple techniques to conduct a successful black-box TAA on SAM.

{{</citation>}}


### (107/176) Towards Unified and Effective Domain Generalization (Yiyuan Zhang et al., 2023)

{{<citation>}}

Yiyuan Zhang, Kaixiong Gong, Xiaohan Ding, Kaipeng Zhang, Fangrui Lv, Kurt Keutzer, Xiangyu Yue. (2023)  
**Towards Unified and Effective Domain Generalization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.10008v1)  

---


**ABSTRACT**  
We propose $\textbf{UniDG}$, a novel and $\textbf{Uni}$fied framework for $\textbf{D}$omain $\textbf{G}$eneralization that is capable of significantly enhancing the out-of-distribution generalization performance of foundation models regardless of their architectures. The core idea of UniDG is to finetune models during the inference stage, which saves the cost of iterative training. Specifically, we encourage models to learn the distribution of test data in an unsupervised manner and impose a penalty regarding the updating step of model parameters. The penalty term can effectively reduce the catastrophic forgetting issue as we would like to maximally preserve the valuable knowledge in the original model. Empirically, across 12 visual backbones, including CNN-, MLP-, and Transformer-based models, ranging from 1.89M to 303M parameters, UniDG shows an average accuracy improvement of +5.4% on DomainBed. These performance results demonstrate the superiority and versatility of UniDG. The code is publicly available at https://github.com/invictus717/UniDG

{{</citation>}}


### (108/176) A Survey of Graph and Attention Based Hyperspectral Image Classification Methods for Remote Sensing Data (Aryan Vats et al., 2023)

{{<citation>}}

Aryan Vats, Manan Suri. (2023)  
**A Survey of Graph and Attention Based Hyperspectral Image Classification Methods for Remote Sensing Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Image Classification  
[Paper Link](http://arxiv.org/abs/2310.09994v1)  

---


**ABSTRACT**  
The use of Deep Learning techniques for classification in Hyperspectral Imaging (HSI) is rapidly growing and achieving improved performances. Due to the nature of the data captured by sensors that produce HSI images, a common issue is the dimensionality of the bands that may or may not contribute to the label class distinction. Due to the widespread nature of class labels, Principal Component Analysis is a common method used for reducing the dimensionality. However,there may exist methods that incorporate all bands of the Hyperspectral image with the help of the Attention mechanism. Furthermore, to yield better spectral spatial feature extraction, recent methods have also explored the usage of Graph Convolution Networks and their unique ability to use node features in prediction, which is akin to the pixel spectral makeup. In this survey we present a comprehensive summary of Graph based and Attention based methods to perform Hyperspectral Image Classification for remote sensing and aerial HSI images. We also summarize relevant datasets on which these techniques have been evaluated and benchmark the processing techniques.

{{</citation>}}


## eess.SY (1)



### (109/176) Joint Optimization of Traffic Signal Control and Vehicle Routing in Signalized Road Networks using Multi-Agent Deep Reinforcement Learning (Xianyue Peng et al., 2023)

{{<citation>}}

Xianyue Peng, Hang Gao, Gengyue Han, Hao Wang, Michael Zhang. (2023)  
**Joint Optimization of Traffic Signal Control and Vehicle Routing in Signalized Road Networks using Multi-Agent Deep Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-MA, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10856v1)  

---


**ABSTRACT**  
Urban traffic congestion is a critical predicament that plagues modern road networks. To alleviate this issue and enhance traffic efficiency, traffic signal control and vehicle routing have proven to be effective measures. In this paper, we propose a joint optimization approach for traffic signal control and vehicle routing in signalized road networks. The objective is to enhance network performance by simultaneously controlling signal timings and route choices using Multi-Agent Deep Reinforcement Learning (MADRL). Signal control agents (SAs) are employed to establish signal timings at intersections, whereas vehicle routing agents (RAs) are responsible for selecting vehicle routes. By establishing relevance between agents and enabling them to share observations and rewards, interaction and cooperation among agents are fostered, which enhances individual training. The Multi-Agent Advantage Actor-Critic algorithm is used to handle multi-agent environments, and Deep Neural Network (DNN) structures are designed to facilitate the algorithm's convergence. Notably, our work is the first to utilize MADRL in determining the optimal joint policy for signal control and vehicle routing. Numerical experiments conducted on the modified Sioux network demonstrate that our integration of signal control and vehicle routing outperforms controlling signal timings or vehicles' routes alone in enhancing traffic efficiency.

{{</citation>}}


## cs.LG (32)



### (110/176) A Machine Learning-based Algorithm for Automated Detection of Frequency-based Events in Recorded Time Series of Sensor Data (Bahareh Medghalchi et al., 2023)

{{<citation>}}

Bahareh Medghalchi, Andreas Vogel. (2023)  
**A Machine Learning-based Algorithm for Automated Detection of Frequency-based Events in Recorded Time Series of Sensor Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-ST, stat-TH  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.10841v1)  

---


**ABSTRACT**  
Automated event detection has emerged as one of the fundamental practices to monitor the behavior of technical systems by means of sensor data. In the automotive industry, these methods are in high demand for tracing events in time series data. For assessing the active vehicle safety systems, a diverse range of driving scenarios is conducted. These scenarios involve the recording of the vehicle's behavior using external sensors, enabling the evaluation of operational performance. In such setting, automated detection methods not only accelerate but also standardize and objectify the evaluation by avoiding subjective, human-based appraisals in the data inspection. This work proposes a novel event detection method that allows to identify frequency-based events in time series data. To this aim, the time series data is mapped to representations in the time-frequency domain, known as scalograms. After filtering scalograms to enhance relevant parts of the signal, an object detection model is trained to detect the desired event objects in the scalograms. For the analysis of unseen time series data, events can be detected in their scalograms with the trained object detection model and are thereafter mapped back to the time series data to mark the corresponding time interval. The algorithm, evaluated on unseen datasets, achieves a precision rate of 0.97 in event detection, providing sharp time interval boundaries whose accurate indication by human visual inspection is challenging. Incorporating this method into the vehicle development process enhances the accuracy and reliability of event detection, which holds major importance for rapid testing analysis.

{{</citation>}}


### (111/176) Approximating Two-Layer Feedforward Networks for Efficient Transformers (Róbert Csordás et al., 2023)

{{<citation>}}

Róbert Csordás, Kazuki Irie, Jürgen Schmidhuber. (2023)  
**Approximating Two-Layer Feedforward Networks for Efficient Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.10837v1)  

---


**ABSTRACT**  
How to reduce compute and memory requirements of neural networks (NNs) without sacrificing performance? Many recent works use sparse Mixtures of Experts (MoEs) to build resource-efficient large language models (LMs). Here we introduce several novel perspectives on MoEs, presenting a general framework that unifies various methods to approximate two-layer NNs (e.g., feedforward blocks of Transformers), including product-key memories (PKMs). Leveraging insights from this framework, we propose methods to improve both MoEs and PKMs. Unlike prior work that compares MoEs with dense baselines under the compute-equal condition, our evaluation condition is parameter-equal, which is crucial to properly evaluate LMs. We show that our MoEs are competitive with the dense Transformer-XL on both the WikiText-103 and enwiki8 datasets at two different scales, while being much more resource efficient. This demonstrates that MoEs are relevant not only to extremely large LMs but also to any-scale resource-efficient LMs. Our code is public.

{{</citation>}}


### (112/176) Proper Laplacian Representation Learning (Diego Gomez et al., 2023)

{{<citation>}}

Diego Gomez, Michael Bowling, Marlos C. Machado. (2023)  
**Proper Laplacian Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.10833v1)  

---


**ABSTRACT**  
The ability to learn good representations of states is essential for solving large reinforcement learning problems, where exploration, generalization, and transfer are particularly challenging. The Laplacian representation is a promising approach to address these problems by inducing intrinsic rewards for temporally-extended action discovery and reward shaping, and informative state encoding. To obtain the Laplacian representation one needs to compute the eigensystem of the graph Laplacian, which is often approximated through optimization objectives compatible with deep learning approaches. These approximations, however, depend on hyperparameters that are impossible to tune efficiently, converge to arbitrary rotations of the desired eigenvectors, and are unable to accurately recover the corresponding eigenvalues. In this paper we introduce a theoretically sound objective and corresponding optimization algorithm for approximating the Laplacian representation. Our approach naturally recovers both the true eigenvectors and eigenvalues while eliminating the hyperparameter dependence of previous approximations. We provide theoretical guarantees for our method and we show that those results translate empirically into robust learning across multiple environments.

{{</citation>}}


### (113/176) Robust Multi-Agent Reinforcement Learning via Adversarial Regularization: Theoretical Foundation and Stable Algorithms (Alexander Bukharin et al., 2023)

{{<citation>}}

Alexander Bukharin, Yan Li, Yue Yu, Qingru Zhang, Zhehui Chen, Simiao Zuo, Chao Zhang, Songan Zhang, Tuo Zhao. (2023)  
**Robust Multi-Agent Reinforcement Learning via Adversarial Regularization: Theoretical Foundation and Stable Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10810v1)  

---


**ABSTRACT**  
Multi-Agent Reinforcement Learning (MARL) has shown promising results across several domains. Despite this promise, MARL policies often lack robustness and are therefore sensitive to small changes in their environment. This presents a serious concern for the real world deployment of MARL algorithms, where the testing environment may slightly differ from the training environment. In this work we show that we can gain robustness by controlling a policy's Lipschitz constant, and under mild conditions, establish the existence of a Lipschitz and close-to-optimal policy. Based on these insights, we propose a new robust MARL framework, ERNIE, that promotes the Lipschitz continuity of the policies with respect to the state observations and actions by adversarial regularization. The ERNIE framework provides robustness against noisy observations, changing transition dynamics, and malicious actions of agents. However, ERNIE's adversarial regularization may introduce some training instability. To reduce this instability, we reformulate adversarial regularization as a Stackelberg game. We demonstrate the effectiveness of the proposed framework with extensive experiments in traffic light control and particle environments. In addition, we extend ERNIE to mean-field MARL with a formulation based on distributionally robust optimization that outperforms its non-robust counterpart and is of independent interest. Our code is available at https://github.com/abukharin3/ERNIE.

{{</citation>}}


### (114/176) Neural Tangent Kernels Motivate Graph Neural Networks with Cross-Covariance Graphs (Shervin Khalafi et al., 2023)

{{<citation>}}

Shervin Khalafi, Saurabh Sihag, Alejandro Ribeiro. (2023)  
**Neural Tangent Kernels Motivate Graph Neural Networks with Cross-Covariance Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.10791v1)  

---


**ABSTRACT**  
Neural tangent kernels (NTKs) provide a theoretical regime to analyze the learning and generalization behavior of over-parametrized neural networks. For a supervised learning task, the association between the eigenvectors of the NTK kernel and given data (a concept referred to as alignment in this paper) can govern the rate of convergence of gradient descent, as well as generalization to unseen data. Building upon this concept, we investigate NTKs and alignment in the context of graph neural networks (GNNs), where our analysis reveals that optimizing alignment translates to optimizing the graph representation or the graph shift operator in a GNN. Our results further establish the theoretical guarantees on the optimality of the alignment for a two-layer GNN and these guarantees are characterized by the graph shift operator being a function of the cross-covariance between the input and the output data. The theoretical insights drawn from the analysis of NTKs are validated by our experiments focused on a multi-variate time series prediction task for a publicly available dataset. Specifically, they demonstrate that GNNs with cross-covariance as the graph shift operator indeed outperform those that operate on the covariance matrix from only the input data.

{{</citation>}}


### (115/176) Gotta be SAFE: A New Framework for Molecular Design (Emmanuel Noutahi et al., 2023)

{{<citation>}}

Emmanuel Noutahi, Cristian Gabellini, Michael Craig, Jonathan S. C Lim, Prudencio Tossou. (2023)  
**Gotta be SAFE: A New Framework for Molecular Design**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: AI, Embedding, GPT  
[Paper Link](http://arxiv.org/abs/2310.10773v1)  

---


**ABSTRACT**  
Traditional molecular string representations, such as SMILES, often pose challenges for AI-driven molecular design due to their non-sequential depiction of molecular substructures. To address this issue, we introduce Sequential Attachment-based Fragment Embedding (SAFE), a novel line notation for chemical structures. SAFE reimagines SMILES strings as an unordered sequence of interconnected fragment blocks while maintaining full compatibility with existing SMILES parsers. It streamlines complex generative tasks, including scaffold decoration, fragment linking, polymer generation, and scaffold hopping, while facilitating autoregressive generation for fragment-constrained design, thereby eliminating the need for intricate decoding or graph-based models. We demonstrate the effectiveness of SAFE by training an 87-million-parameter GPT2-like model on a dataset containing 1.1 billion SAFE representations. Through extensive experimentation, we show that our SAFE-GPT model exhibits versatile and robust optimization performance. SAFE opens up new avenues for the rapid exploration of chemical space under various constraints, promising breakthroughs in AI-driven molecular design.

{{</citation>}}


### (116/176) Towards Scenario-based Safety Validation for Autonomous Trains with Deep Generative Models (Thomas Decker et al., 2023)

{{<citation>}}

Thomas Decker, Ananta R. Bhattarai, Michael Lebacher. (2023)  
**Towards Scenario-based Safety Validation for Autonomous Trains with Deep Generative Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10635v1)  

---


**ABSTRACT**  
Modern AI techniques open up ever-increasing possibilities for autonomous vehicles, but how to appropriately verify the reliability of such systems remains unclear. A common approach is to conduct safety validation based on a predefined Operational Design Domain (ODD) describing specific conditions under which a system under test is required to operate properly. However, collecting sufficient realistic test cases to ensure comprehensive ODD coverage is challenging. In this paper, we report our practical experiences regarding the utility of data simulation with deep generative models for scenario-based ODD validation. We consider the specific use case of a camera-based rail-scene segmentation system designed to support autonomous train operation. We demonstrate the capabilities of semantically editing railway scenes with deep generative models to make a limited amount of test data more representative. We also show how our approach helps to analyze the degree to which a system complies with typical ODD requirements. Specifically, we focus on evaluating proper operation under different lighting and weather conditions as well as while transitioning between them.

{{</citation>}}


### (117/176) How Do Transformers Learn In-Context Beyond Simple Functions? A Case Study on Learning with Representations (Tianyu Guo et al., 2023)

{{<citation>}}

Tianyu Guo, Wei Hu, Song Mei, Huan Wang, Caiming Xiong, Silvio Savarese, Yu Bai. (2023)  
**How Do Transformers Learn In-Context Beyond Simple Functions? A Case Study on Learning with Representations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.10616v1)  

---


**ABSTRACT**  
While large language models based on the transformer architecture have demonstrated remarkable in-context learning (ICL) capabilities, understandings of such capabilities are still in an early stage, where existing theory and mechanistic understanding focus mostly on simple scenarios such as learning simple function classes. This paper takes initial steps on understanding ICL in more complex scenarios, by studying learning with representations. Concretely, we construct synthetic in-context learning problems with a compositional structure, where the label depends on the input through a possibly complex but fixed representation function, composed with a linear function that differs in each instance. By construction, the optimal ICL algorithm first transforms the inputs by the representation function, and then performs linear ICL on top of the transformed dataset. We show theoretically the existence of transformers that approximately implement such algorithms with mild depth and size. Empirically, we find trained transformers consistently achieve near-optimal ICL performance in this setting, and exhibit the desired dissection where lower layers transforms the dataset and upper layers perform linear ICL. Through extensive probing and a new pasting experiment, we further reveal several mechanisms within the trained transformers, such as concrete copying behaviors on both the inputs and the representations, linear ICL capability of the upper layers alone, and a post-ICL representation selection mechanism in a harder mixture setting. These observed mechanisms align well with our theory and may shed light on how transformers perform ICL in more realistic scenarios.

{{</citation>}}


### (118/176) IW-GAE: Importance weighted group accuracy estimation for improved calibration and model selection in unsupervised domain adaptation (Taejong Joo et al., 2023)

{{<citation>}}

Taejong Joo, Diego Klabjan. (2023)  
**IW-GAE: Importance weighted group accuracy estimation for improved calibration and model selection in unsupervised domain adaptation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.10611v1)  

---


**ABSTRACT**  
Reasoning about a model's accuracy on a test sample from its confidence is a central problem in machine learning, being connected to important applications such as uncertainty representation, model selection, and exploration. While these connections have been well-studied in the i.i.d. settings, distribution shifts pose significant challenges to the traditional methods. Therefore, model calibration and model selection remain challenging in the unsupervised domain adaptation problem--a scenario where the goal is to perform well in a distribution shifted domain without labels. In this work, we tackle difficulties coming from distribution shifts by developing a novel importance weighted group accuracy estimator. Specifically, we formulate an optimization problem for finding an importance weight that leads to an accurate group accuracy estimation in the distribution shifted domain with theoretical analyses. Extensive experiments show the effectiveness of group accuracy estimation on model calibration and model selection. Our results emphasize the significance of group accuracy estimation for addressing challenges in unsupervised domain adaptation, as an orthogonal improvement direction with improving transferability of accuracy.

{{</citation>}}


### (119/176) Exploring the Power of Graph Neural Networks in Solving Linear Optimization Problems (Chendi Qian et al., 2023)

{{<citation>}}

Chendi Qian, Didier Chételat, Christopher Morris. (2023)  
**Exploring the Power of Graph Neural Networks in Solving Linear Optimization Problems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG, math-OC, stat-ML  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.10603v1)  

---


**ABSTRACT**  
Recently, machine learning, particularly message-passing graph neural networks (MPNNs), has gained traction in enhancing exact optimization algorithms. For example, MPNNs speed up solving mixed-integer optimization problems by imitating computational intensive heuristics like strong branching, which entails solving multiple linear optimization problems (LPs). Despite the empirical success, the reasons behind MPNNs' effectiveness in emulating linear optimization remain largely unclear. Here, we show that MPNNs can simulate standard interior-point methods for LPs, explaining their practical success. Furthermore, we highlight how MPNNs can serve as a lightweight proxy for solving LPs, adapting to a given problem instance distribution. Empirically, we show that MPNNs solve LP relaxations of standard combinatorial optimization problems close to optimality, often surpassing conventional solvers and competing approaches in solving time.

{{</citation>}}


### (120/176) Towards the Imagenets of ML4EDA (Animesh Basak Chowdhury et al., 2023)

{{<citation>}}

Animesh Basak Chowdhury, Shailja Thakur, Hammond Pearce, Ramesh Karri, Siddharth Garg. (2023)  
**Towards the Imagenets of ML4EDA**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-AR, cs-LG, cs-PL, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10560v1)  

---


**ABSTRACT**  
Despite the growing interest in ML-guided EDA tools from RTL to GDSII, there are no standard datasets or prototypical learning tasks defined for the EDA problem domain. Experience from the computer vision community suggests that such datasets are crucial to spur further progress in ML for EDA. Here we describe our experience curating two large-scale, high-quality datasets for Verilog code generation and logic synthesis. The first, VeriGen, is a dataset of Verilog code collected from GitHub and Verilog textbooks. The second, OpenABC-D, is a large-scale, labeled dataset designed to aid ML for logic synthesis tasks. The dataset consists of 870,000 And-Inverter-Graphs (AIGs) produced from 1500 synthesis runs on a large number of open-source hardware projects. In this paper we will discuss challenges in curating, maintaining and growing the size and scale of these datasets. We will also touch upon questions of dataset quality and security, and the use of novel data augmentation tools that are tailored for the hardware domain.

{{</citation>}}


### (121/176) TacticAI: an AI assistant for football tactics (Zhe Wang et al., 2023)

{{<citation>}}

Zhe Wang, Petar Veličković, Daniel Hennes, Nenad Tomašev, Laurel Prince, Michael Kaisers, Yoram Bachrach, Romuald Elie, Li Kevin Wenliang, Federico Piccinini, William Spearman, Ian Graham, Jerome Connor, Yi Yang, Adrià Recasens, Mina Khan, Nathalie Beauguerlange, Pablo Sprechmann, Pol Moreno, Nicolas Heess, Michael Bowling, Demis Hassabis, Karl Tuyls. (2023)  
**TacticAI: an AI assistant for football tactics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10553v2)  

---


**ABSTRACT**  
Identifying key patterns of tactics implemented by rival teams, and developing effective responses, lies at the heart of modern football. However, doing so algorithmically remains an open research challenge. To address this unmet need, we propose TacticAI, an AI football tactics assistant developed and evaluated in close collaboration with domain experts from Liverpool FC. We focus on analysing corner kicks, as they offer coaches the most direct opportunities for interventions and improvements. TacticAI incorporates both a predictive and a generative component, allowing the coaches to effectively sample and explore alternative player setups for each corner kick routine and to select those with the highest predicted likelihood of success. We validate TacticAI on a number of relevant benchmark tasks: predicting receivers and shot attempts and recommending player position adjustments. The utility of TacticAI is validated by a qualitative study conducted with football domain experts at Liverpool FC. We show that TacticAI's model suggestions are not only indistinguishable from real tactics, but also favoured over existing tactics 90% of the time, and that TacticAI offers an effective corner kick retrieval system. TacticAI achieves these results despite the limited availability of gold-standard data, achieving data efficiency through geometric deep learning.

{{</citation>}}


### (122/176) Microscaling Data Formats for Deep Learning (Bita Darvish Rouhani et al., 2023)

{{<citation>}}

Bita Darvish Rouhani, Ritchie Zhao, Ankit More, Mathew Hall, Alireza Khodamoradi, Summer Deng, Dhruv Choudhary, Marius Cornea, Eric Dellinger, Kristof Denolf, Stosic Dusan, Venmugil Elango, Maximilian Golub, Alexander Heinecke, Phil James-Roxby, Dharmesh Jani, Gaurav Kolhe, Martin Langhammer, Ada Li, Levi Melnick, Maral Mesmakhosroshahi, Andres Rodriguez, Michael Schulte, Rasoul Shafipour, Lei Shao, Michael Siu, Pradeep Dubey, Paulius Micikevicius, Maxim Naumov, Colin Verrilli, Ralph Wittig, Doug Burger, Eric Chung. (2023)  
**Microscaling Data Formats for Deep Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10537v3)  

---


**ABSTRACT**  
Narrow bit-width data formats are key to reducing the computational and storage costs of modern deep learning applications. This paper evaluates Microscaling (MX) data formats that combine a per-block scaling factor with narrow floating-point and integer types for individual elements. MX formats balance the competing needs of hardware efficiency, model accuracy, and user friction. Empirical results on over two dozen benchmarks demonstrate practicality of MX data formats as a drop-in replacement for baseline FP32 for AI inference and training with low user friction. We also show the first instance of training generative language models at sub-8-bit weights, activations, and gradients with minimal accuracy loss and no modifications to the training recipe.

{{</citation>}}


### (123/176) ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models (Ziniu Li et al., 2023)

{{<citation>}}

Ziniu Li, Tian Xu, Yushun Zhang, Yang Yu, Ruoyu Sun, Zhi-Quan Luo. (2023)  
**ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10505v2)  

---


**ABSTRACT**  
Alignment is of critical importance for training large language models (LLMs). The predominant strategy to address this is through Reinforcement Learning from Human Feedback (RLHF), where PPO serves as the de-facto algorithm. Yet, PPO is known to suffer from computational inefficiency, which is a challenge that this paper aims to address. We identify three important properties in RLHF tasks: fast simulation, deterministic transitions, and trajectory-level rewards, which are not leveraged in PPO. Based on such observations, we develop a new algorithm tailored for RLHF, called ReMax. The algorithm design of ReMax is built on a celebrated algorithm REINFORCE but is equipped with a new variance-reduction technique.   Our method has three-fold advantages over PPO: first, ReMax is simple to implement and removes many hyper-parameters in PPO, which are scale-sensitive and laborious to tune. Second, ReMax saves about 50% memory usage in principle. As a result, PPO runs out-of-memory when fine-tuning a Llama2 (7B) model on 8xA100-40GB GPUs, whereas ReMax can afford training. This memory improvement is achieved by removing the value model in PPO. Third, based on our calculations, we find that even assuming PPO can afford the training of Llama2 (7B), it would still run about 2x slower than ReMax. This is due to the computational overhead of the value model, which does not exist in ReMax. Importantly, the above computational improvements do not sacrifice the performance. We hypothesize these advantages can be maintained in larger-scaled models. Our implementation of ReMax is available at https://github.com/liziniu/ReMax

{{</citation>}}


### (124/176) Reading Books is Great, But Not if You Are Driving! Visually Grounded Reasoning about Defeasible Commonsense Norms (Seungju Han et al., 2023)

{{<citation>}}

Seungju Han, Junhyeok Kim, Jack Hessel, Liwei Jiang, Jiwan Chung, Yejin Son, Yejin Choi, Youngjae Yu. (2023)  
**Reading Books is Great, But Not if You Are Driving! Visually Grounded Reasoning about Defeasible Commonsense Norms**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.10418v1)  

---


**ABSTRACT**  
Commonsense norms are defeasible by context: reading books is usually great, but not when driving a car. While contexts can be explicitly described in language, in embodied scenarios, contexts are often provided visually. This type of visually grounded reasoning about defeasible commonsense norms is generally easy for humans, but (as we show) poses a challenge for machines, as it necessitates both visual understanding and reasoning about commonsense norms. We construct a new multimodal benchmark for studying visual-grounded commonsense norms: NORMLENS. NORMLENS consists of 10K human judgments accompanied by free-form explanations covering 2K multimodal situations, and serves as a probe to address two questions: (1) to what extent can models align with average human judgment? and (2) how well can models explain their predicted judgments? We find that state-of-the-art model judgments and explanations are not well-aligned with human annotation. Additionally, we present a new approach to better align models with humans by distilling social commonsense knowledge from large language models. The data and code are released at https://seungjuhan.me/normlens.

{{</citation>}}


### (125/176) Revisiting Logistic-softmax Likelihood in Bayesian Meta-Learning for Few-Shot Classification (Tianjun Ke et al., 2023)

{{<citation>}}

Tianjun Ke, Haoqun Cao, Zenan Ling, Feng Zhou. (2023)  
**Revisiting Logistic-softmax Likelihood in Bayesian Meta-Learning for Few-Shot Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.10379v1)  

---


**ABSTRACT**  
Meta-learning has demonstrated promising results in few-shot classification (FSC) by learning to solve new problems using prior knowledge. Bayesian methods are effective at characterizing uncertainty in FSC, which is crucial in high-risk fields. In this context, the logistic-softmax likelihood is often employed as an alternative to the softmax likelihood in multi-class Gaussian process classification due to its conditional conjugacy property. However, the theoretical property of logistic-softmax is not clear and previous research indicated that the inherent uncertainty of logistic-softmax leads to suboptimal performance. To mitigate these issues, we revisit and redesign the logistic-softmax likelihood, which enables control of the \textit{a priori} confidence level through a temperature parameter. Furthermore, we theoretically and empirically show that softmax can be viewed as a special case of logistic-softmax and logistic-softmax induces a larger family of data distribution than softmax. Utilizing modified logistic-softmax, we integrate the data augmentation technique into the deep kernel based Gaussian process meta-learning framework, and derive an analytical mean-field approximation for task-specific updates. Our approach yields well-calibrated uncertainty estimates and achieves comparable or superior results on standard benchmark datasets. Code is publicly available at \url{https://github.com/keanson/revisit-logistic-softmax}.

{{</citation>}}


### (126/176) Prompt Tuning for Multi-View Graph Contrastive Learning (Chenghua Gong et al., 2023)

{{<citation>}}

Chenghua Gong, Xiang Li, Jianxiang Yu, Cheng Yao, Jiaqi Tan, Chengcheng Yu, Dawei Yin. (2023)  
**Prompt Tuning for Multi-View Graph Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning, GNN  
[Paper Link](http://arxiv.org/abs/2310.10362v1)  

---


**ABSTRACT**  
In recent years, "pre-training and fine-tuning" has emerged as a promising approach in addressing the issues of label dependency and poor generalization performance in traditional GNNs. To reduce labeling requirement, the "pre-train, fine-tune" and "pre-train, prompt" paradigms have become increasingly common. In particular, prompt tuning is a popular alternative to "pre-training and fine-tuning" in natural language processing, which is designed to narrow the gap between pre-training and downstream objectives. However, existing study of prompting on graphs is still limited, lacking a framework that can accommodate commonly used graph pre-training methods and downstream tasks. In this paper, we propose a multi-view graph contrastive learning method as pretext and design a prompting tuning for it. Specifically, we first reformulate graph pre-training and downstream tasks into a common format. Second, we construct multi-view contrasts to capture relevant information of graphs by GNN. Third, we design a prompting tuning method for our multi-view graph contrastive learning method to bridge the gap between pretexts and downsteam tasks. Finally, we conduct extensive experiments on benchmark datasets to evaluate and analyze our proposed method.

{{</citation>}}


### (127/176) Transparent Anomaly Detection via Concept-based Explanations (Laya Rafiee Sevyeri et al., 2023)

{{<citation>}}

Laya Rafiee Sevyeri, Ivaxi Sheth, Farhood Farahnak, Shirin Abbasinejad Enger. (2023)  
**Transparent Anomaly Detection via Concept-based Explanations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.10702v1)  

---


**ABSTRACT**  
Advancements in deep learning techniques have given a boost to the performance of anomaly detection. However, real-world and safety-critical applications demand a level of transparency and reasoning beyond accuracy. The task of anomaly detection (AD) focuses on finding whether a given sample follows the learned distribution. Existing methods lack the ability to reason with clear explanations for their outcomes. Hence to overcome this challenge, we propose Transparent {A}nomaly Detection {C}oncept {E}xplanations (ACE). ACE is able to provide human interpretable explanations in the form of concepts along with anomaly prediction. To the best of our knowledge, this is the first paper that proposes interpretable by-design anomaly detection. In addition to promoting transparency in AD, it allows for effective human-model interaction. Our proposed model shows either higher or comparable results to black-box uninterpretable models. We validate the performance of ACE across three realistic datasets - bird classification on CUB-200-2011, challenging histopathology slide image classification on TIL-WSI-TCGA, and gender classification on CelebA. We further demonstrate that our concept learning paradigm can be seamlessly integrated with other classification-based AD methods.

{{</citation>}}


### (128/176) Mimicking the Maestro: Exploring the Efficacy of a Virtual AI Teacher in Fine Motor Skill Acquisition (Hadar Mulian et al., 2023)

{{<citation>}}

Hadar Mulian, Segev Shlomov, Lior Limonad. (2023)  
**Mimicking the Maestro: Exploring the Efficacy of a Virtual AI Teacher in Fine Motor Skill Acquisition**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10280v1)  

---


**ABSTRACT**  
Motor skills, especially fine motor skills like handwriting, play an essential role in academic pursuits and everyday life. Traditional methods to teach these skills, although effective, can be time-consuming and inconsistent. With the rise of advanced technologies like robotics and artificial intelligence, there is increasing interest in automating such teaching processes using these technologies, via human-robot and human-computer interactions. In this study, we examine the potential of a virtual AI teacher in emulating the techniques of human educators for motor skill acquisition. We introduce an AI teacher model that captures the distinct characteristics of human instructors. Using a Reinforcement Learning environment tailored to mimic teacher-learner interactions, we tested our AI model against four guiding hypotheses, emphasizing improved learner performance, enhanced rate of skill acquisition, and reduced variability in learning outcomes. Our findings, validated on synthetic learners, revealed significant improvements across all tested hypotheses. Notably, our model showcased robustness across different learners and settings and demonstrated adaptability to handwriting. This research underscores the potential of integrating Reinforcement Learning and Imitation Learning models with robotics in revolutionizing the teaching of critical motor skills.

{{</citation>}}


### (129/176) Leveraging Topological Maps in Deep Reinforcement Learning for Multi-Object Navigation (Simon Hakenes et al., 2023)

{{<citation>}}

Simon Hakenes, Tobias Glasmachers. (2023)  
**Leveraging Topological Maps in Deep Reinforcement Learning for Multi-Object Navigation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10250v1)  

---


**ABSTRACT**  
This work addresses the challenge of navigating expansive spaces with sparse rewards through Reinforcement Learning (RL). Using topological maps, we elevate elementary actions to object-oriented macro actions, enabling a simple Deep Q-Network (DQN) agent to solve otherwise practically impossible environments.

{{</citation>}}


### (130/176) Bongard-OpenWorld: Few-Shot Reasoning for Free-form Visual Concepts in the Real World (Rujie Wu et al., 2023)

{{<citation>}}

Rujie Wu, Xiaojian Ma, Qing Li, Wei Wang, Zhenliang Zhang, Song-Chun Zhu, Yizhou Wang. (2023)  
**Bongard-OpenWorld: Few-Shot Reasoning for Free-form Visual Concepts in the Real World**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Few-Shot, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.10207v1)  

---


**ABSTRACT**  
We introduce Bongard-OpenWorld, a new benchmark for evaluating real-world few-shot reasoning for machine vision. It originates from the classical Bongard Problems (BPs): Given two sets of images (positive and negative), the model needs to identify the set that query images belong to by inducing the visual concepts, which is exclusively depicted by images from the positive set. Our benchmark inherits the few-shot concept induction of the original BPs while adding the two novel layers of challenge: 1) open-world free-form concepts, as the visual concepts in Bongard-OpenWorld are unique compositions of terms from an open vocabulary, ranging from object categories to abstract visual attributes and commonsense factual knowledge; 2) real-world images, as opposed to the synthetic diagrams used by many counterparts. In our exploration, Bongard-OpenWorld already imposes a significant challenge to current few-shot reasoning algorithms. We further investigate to which extent the recently introduced Large Language Models (LLMs) and Vision-Language Models (VLMs) can solve our task, by directly probing VLMs, and combining VLMs and LLMs in an interactive reasoning scheme. We even designed a neuro-symbolic reasoning approach that reconciles LLMs & VLMs with logical reasoning to emulate the human problem-solving process for Bongard Problems. However, none of these approaches manage to close the human-machine gap, as the best learner achieves 64% accuracy while human participants easily reach 91%. We hope Bongard-OpenWorld can help us better understand the limitations of current visual intelligence and facilitate future research on visual agents with stronger few-shot visual reasoning capabilities.

{{</citation>}}


### (131/176) Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook (Ming Jin et al., 2023)

{{<citation>}}

Ming Jin, Qingsong Wen, Yuxuan Liang, Chaoli Zhang, Siqiao Xue, Xue Wang, James Zhang, Yi Wang, Haifeng Chen, Xiaoli Li, Shirui Pan, Vincent S. Tseng, Yu Zheng, Lei Chen, Hui Xiong. (2023)  
**Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.10196v1)  

---


**ABSTRACT**  
Temporal data, notably time series and spatio-temporal data, are prevalent in real-world applications. They capture dynamic system measurements and are produced in vast quantities by both physical and virtual sensors. Analyzing these data types is vital to harnessing the rich information they encompass and thus benefits a wide range of downstream tasks. Recent advances in large language and other foundational models have spurred increased use of these models in time series and spatio-temporal data mining. Such methodologies not only enable enhanced pattern recognition and reasoning across diverse domains but also lay the groundwork for artificial general intelligence capable of comprehending and processing common temporal data. In this survey, we offer a comprehensive and up-to-date review of large models tailored (or adapted) for time series and spatio-temporal data, spanning four key facets: data types, model categories, model scopes, and application areas/tasks. Our objective is to equip practitioners with the knowledge to develop applications and further research in this underexplored domain. We primarily categorize the existing literature into two major clusters: large models for time series analysis (LM4TS) and spatio-temporal data mining (LM4STD). On this basis, we further classify research based on model scopes (i.e., general vs. domain-specific) and application areas/tasks. We also provide a comprehensive collection of pertinent resources, including datasets, model assets, and useful tools, categorized by mainstream applications. This survey coalesces the latest strides in large model-centric research on time series and spatio-temporal data, underscoring the solid foundations, current advances, practical applications, abundant resources, and future research opportunities.

{{</citation>}}


### (132/176) An Interpretable Deep-Learning Framework for Predicting Hospital Readmissions From Electronic Health Records (Fabio Azzalini et al., 2023)

{{<citation>}}

Fabio Azzalini, Tommaso Dolci, Marco Vagaggini. (2023)  
**An Interpretable Deep-Learning Framework for Predicting Hospital Readmissions From Electronic Health Records**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: LSTM, NLP  
[Paper Link](http://arxiv.org/abs/2310.10187v1)  

---


**ABSTRACT**  
With the increasing availability of patients' data, modern medicine is shifting towards prospective healthcare. Electronic health records contain a variety of information useful for clinical patient description and can be exploited for the construction of predictive models, given that similar medical histories will likely lead to similar progressions. One example is unplanned hospital readmission prediction, an essential task for reducing hospital costs and improving patient health. Despite predictive models showing very good performances especially with deep-learning models, they are often criticized for the poor interpretability of their results, a fundamental characteristic in the medical field, where incorrect predictions might have serious consequences for the patient health. In this paper we propose a novel, interpretable deep-learning framework for predicting unplanned hospital readmissions, supported by NLP findings on word embeddings and by neural-network models (ConvLSTM) for better handling temporal data. We validate our system on the two predictive tasks of hospital readmission within 30 and 180 days, using real-world data. In addition, we introduce and test a model-dependent technique to make the representation of results easily interpretable by the medical staff. Our solution achieves better performances compared to traditional models based on machine learning, while providing at the same time more interpretable results.

{{</citation>}}


### (133/176) Leveraging Knowledge Distillation for Efficient Deep Reinforcement Learning in Resource-Constrained Environments (Guanlin Meng, 2023)

{{<citation>}}

Guanlin Meng. (2023)  
**Leveraging Knowledge Distillation for Efficient Deep Reinforcement Learning in Resource-Constrained Environments**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Knowledge Distillation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10170v1)  

---


**ABSTRACT**  
This paper aims to explore the potential of combining Deep Reinforcement Learning (DRL) with Knowledge Distillation (KD) by distilling various DRL algorithms and studying their distillation effects. By doing so, the computational burden of deep models could be reduced while maintaining the performance. The primary objective is to provide a benchmark for evaluating the performance of different DRL algorithms that have been refined using KD techniques. By distilling these algorithms, the goal is to develop efficient and fast DRL models. This research is expected to provide valuable insights that can facilitate further advancements in this promising direction. By exploring the combination of DRL and KD, this work aims to promote the development of models that require fewer GPU resources, learn more quickly, and make faster decisions in complex environments. The results of this research have the capacity to significantly advance the field of DRL and pave the way for the future deployment of resource-efficient, decision-making intelligent systems.

{{</citation>}}


### (134/176) From Continuous Dynamics to Graph Neural Networks: Neural Diffusion and Beyond (Andi Han et al., 2023)

{{<citation>}}

Andi Han, Dai Shi, Lequan Lin, Junbin Gao. (2023)  
**From Continuous Dynamics to Graph Neural Networks: Neural Diffusion and Beyond**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.10121v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have demonstrated significant promise in modelling relational data and have been widely applied in various fields of interest. The key mechanism behind GNNs is the so-called message passing where information is being iteratively aggregated to central nodes from their neighbourhood. Such a scheme has been found to be intrinsically linked to a physical process known as heat diffusion, where the propagation of GNNs naturally corresponds to the evolution of heat density. Analogizing the process of message passing to the heat dynamics allows to fundamentally understand the power and pitfalls of GNNs and consequently informs better model design. Recently, there emerges a plethora of works that proposes GNNs inspired from the continuous dynamics formulation, in an attempt to mitigate the known limitations of GNNs, such as oversmoothing and oversquashing. In this survey, we provide the first systematic and comprehensive review of studies that leverage the continuous perspective of GNNs. To this end, we introduce foundational ingredients for adapting continuous dynamics to GNNs, along with a general framework for the design of graph neural dynamics. We then review and categorize existing works based on their driven mechanisms and underlying dynamics. We also summarize how the limitations of classic GNNs can be addressed under the continuous framework. We conclude by identifying multiple open research directions.

{{</citation>}}


### (135/176) Regret Analysis of the Posterior Sampling-based Learning Algorithm for Episodic POMDPs (Dengwang Tang et al., 2023)

{{<citation>}}

Dengwang Tang, Rahul Jain, Ashutosh Nayyar, Pierluigi Nuzzo. (2023)  
**Regret Analysis of the Posterior Sampling-based Learning Algorithm for Episodic POMDPs**  

---
Primary Category: cs.LG  
Categories: 93E35, cs-AI, cs-LG, cs-SY, cs.LG, eess-SY, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10107v1)  

---


**ABSTRACT**  
Compared to Markov Decision Processes (MDPs), learning in Partially Observable Markov Decision Processes (POMDPs) can be significantly harder due to the difficulty of interpreting observations. In this paper, we consider episodic learning problems in POMDPs with unknown transition and observation models. We consider the Posterior Sampling-based Reinforcement Learning (PSRL) algorithm for POMDPs and show that its Bayesian regret scales as the square root of the number of episodes. In general, the regret scales exponentially with the horizon length $H$, and we show that this is inevitable by providing a lower bound. However, under the condition that the POMDP is undercomplete and weakly revealing, we establish a polynomial Bayesian regret bound that improves the regret bound by a factor of $\Omega(H^2\sqrt{SA})$ over the recent result by arXiv:2204.08967.

{{</citation>}}


### (136/176) Reusing Pretrained Models by Multi-linear Operators for Efficient Training (Yu Pan et al., 2023)

{{<citation>}}

Yu Pan, Ye Yuan, Yichun Yin, Zenglin Xu, Lifeng Shang, Xin Jiang, Qun Liu. (2023)  
**Reusing Pretrained Models by Multi-linear Operators for Efficient Training**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.10699v1)  

---


**ABSTRACT**  
Training large models from scratch usually costs a substantial amount of resources. Towards this problem, recent studies such as bert2BERT and LiGO have reused small pretrained models to initialize a large model (termed the ``target model''), leading to a considerable acceleration in training. Despite the successes of these previous studies, they grew pretrained models by mapping partial weights only, ignoring potential correlations across the entire model. As we show in this paper, there are inter- and intra-interactions among the weights of both the pretrained and the target models. As a result, the partial mapping may not capture the complete information and lead to inadequate growth. In this paper, we propose a method that linearly correlates each weight of the target model to all the weights of the pretrained model to further enhance acceleration ability. We utilize multi-linear operators to reduce computational and spacial complexity, enabling acceptable resource requirements. Experiments demonstrate that our method can save 76\% computational costs on DeiT-base transferred from DeiT-small, which outperforms bert2BERT by +12.0\% and LiGO by +20.7\%, respectively.

{{</citation>}}


### (137/176) Learning Graph Filters for Spectral GNNs via Newton Interpolation (Junjie Xu et al., 2023)

{{<citation>}}

Junjie Xu, Enyan Dai, Dongsheng Luo, Xiang Zhang, Suhang Wang. (2023)  
**Learning Graph Filters for Spectral GNNs via Newton Interpolation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.10064v1)  

---


**ABSTRACT**  
Spectral Graph Neural Networks (GNNs) are gaining attention because they can surpass the limitations of message-passing GNNs by learning spectral filters that capture essential frequency information in graph data through task supervision. However, previous research suggests that the choice of filter frequency is tied to the graph's homophily level, a connection that hasn't been thoroughly explored in existing spectral GNNs. To address this gap, the study conducts both theoretical and empirical analyses, revealing that low-frequency filters have a positive correlation with homophily, while high-frequency filters have a negative correlation. This leads to the introduction of a shape-aware regularization technique applied to a Newton Interpolation-based spectral filter, enabling the customization of polynomial spectral filters that align with desired homophily levels. Extensive experiments demonstrate that NewtonNet successfully achieves the desired filter shapes and exhibits superior performance on both homophilous and heterophilous datasets.

{{</citation>}}


### (138/176) Data Augmentation for Time-Series Classification: An Extensive Empirical Study and Comprehensive Survey (Zijun Gao et al., 2023)

{{<citation>}}

Zijun Gao, Lingbo Li, Tianhua Xu. (2023)  
**Data Augmentation for Time-Series Classification: An Extensive Empirical Study and Comprehensive Survey**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation, Time Series  
[Paper Link](http://arxiv.org/abs/2310.10060v2)  

---


**ABSTRACT**  
Data Augmentation (DA) has emerged as an indispensable strategy in Time Series Classification (TSC), primarily due to its capacity to amplify training samples, thereby bolstering model robustness, diversifying datasets, and curtailing overfitting. However, the current landscape of DA in TSC is plagued with fragmented literature reviews, nebulous methodological taxonomies, inadequate evaluative measures, and a dearth of accessible, user-oriented tools. In light of these challenges, this study embarks on an exhaustive dissection of DA methodologies within the TSC realm. Our initial approach involved an extensive literature review spanning a decade, revealing that contemporary surveys scarcely capture the breadth of advancements in DA for TSC, prompting us to meticulously analyze over 100 scholarly articles to distill more than 60 unique DA techniques. This rigorous analysis precipitated the formulation of a novel taxonomy, purpose-built for the intricacies of DA in TSC, categorizing techniques into five principal echelons: Transformation-Based, Pattern-Based, Generative, Decomposition-Based, and Automated Data Augmentation. Our taxonomy promises to serve as a robust navigational aid for scholars, offering clarity and direction in method selection. Addressing the conspicuous absence of holistic evaluations for prevalent DA techniques, we executed an all-encompassing empirical assessment, wherein upwards of 15 DA strategies were subjected to scrutiny across 8 UCR time-series datasets, employing ResNet and a multi-faceted evaluation paradigm encompassing Accuracy, Method Ranking, and Residual Analysis, yielding a benchmark accuracy of 88.94 +- 11.83%. Our investigation underscored the inconsistent efficacies of DA techniques, with...

{{</citation>}}


### (139/176) FATE-LLM: A Industrial Grade Federated Learning Framework for Large Language Models (Tao Fan et al., 2023)

{{<citation>}}

Tao Fan, Yan Kang, Guoqiang Ma, Weijing Chen, Wenbin Wei, Lixin Fan, Qiang Yang. (2023)  
**FATE-LLM: A Industrial Grade Federated Learning Framework for Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, ChatGPT, GLM, GPT, LLaMA, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2310.10049v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), such as ChatGPT, LLaMA, GLM, and PaLM, have exhibited remarkable performances across various tasks in recent years. However, LLMs face two main challenges in real-world applications. One challenge is that training LLMs consumes vast computing resources, preventing LLMs from being adopted by small and medium-sized enterprises with limited computing resources. Another is that training LLM requires a large amount of high-quality data, which are often scattered among enterprises. To address these challenges, we propose FATE-LLM, an industrial-grade federated learning framework for large language models. FATE-LLM (1) facilitates federated learning for large language models (coined FedLLM); (2) promotes efficient training of FedLLM using parameter-efficient fine-tuning methods; (3) protects the intellectual property of LLMs; (4) preserves data privacy during training and inference through privacy-preserving mechanisms. We release the code of FATE-LLM at https://github.com/FederatedAI/FATE-LLM to facilitate the research of FedLLM and enable a broad range of industrial applications.

{{</citation>}}


### (140/176) Symmetrical SyncMap for Imbalanced General Chunking Problems (Heng Zhang et al., 2023)

{{<citation>}}

Heng Zhang, Danilo Vasconcellos Vargas. (2023)  
**Symmetrical SyncMap for Imbalanced General Chunking Problems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: GCP  
[Paper Link](http://arxiv.org/abs/2310.10045v1)  

---


**ABSTRACT**  
Recently, SyncMap pioneered an approach to learn complex structures from sequences as well as adapt to any changes in underlying structures. This is achieved by using only nonlinear dynamical equations inspired by neuron group behaviors, i.e., without loss functions. Here we propose Symmetrical SyncMap that goes beyond the original work to show how to create dynamical equations and attractor-repeller points which are stable over the long run, even dealing with imbalanced continual general chunking problems (CGCPs). The main idea is to apply equal updates from negative and positive feedback loops by symmetrical activation. We then introduce the concept of memory window to allow for more positive updates. Our algorithm surpasses or ties other unsupervised state-of-the-art baselines in all 12 imbalanced CGCPs with various difficulties, including dynamically changing ones. To verify its performance in real-world scenarios, we conduct experiments on several well-studied structure learning problems. The proposed method surpasses substantially other methods in 3 out of 4 scenarios, suggesting that symmetrical activation plays a critical role in uncovering topological structures and even hierarchies encoded in temporal data.

{{</citation>}}


### (141/176) Personalization of CTC-based End-to-End Speech Recognition Using Pronunciation-Driven Subword Tokenization (Zhihong Lei et al., 2023)

{{<citation>}}

Zhihong Lei, Ernest Pusateri, Shiyi Han, Leo Liu, Mingbin Xu, Tim Ng, Ruchir Travadi, Youyuan Zhang, Mirko Hannemann, Man-Hung Siu, Zhen Huang. (2023)  
**Personalization of CTC-based End-to-End Speech Recognition Using Pronunciation-Driven Subword Tokenization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.09988v1)  

---


**ABSTRACT**  
Recent advances in deep learning and automatic speech recognition have improved the accuracy of end-to-end speech recognition systems, but recognition of personal content such as contact names remains a challenge. In this work, we describe our personalization solution for an end-to-end speech recognition system based on connectionist temporal classification. Building on previous work, we present a novel method for generating additional subword tokenizations for personal entities from their pronunciations. We show that using this technique in combination with two established techniques, contextual biasing and wordpiece prior normalization, we are able to achieve personal named entity accuracy on par with a competitive hybrid system.

{{</citation>}}


## cs.GT (1)



### (142/176) Mechanism Design for Large Language Models (Paul Duetting et al., 2023)

{{<citation>}}

Paul Duetting, Vahab Mirrokni, Renato Paes Leme, Haifeng Xu, Song Zuo. (2023)  
**Mechanism Design for Large Language Models**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT, econ-TH  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10826v1)  

---


**ABSTRACT**  
We investigate auction mechanisms to support the emerging format of AI-generated content. We in particular study how to aggregate several LLMs in an incentive compatible manner. In this problem, the preferences of each agent over stochastically generated contents are described/encoded as an LLM. A key motivation is to design an auction format for AI-generated ad creatives to combine inputs from different advertisers. We argue that this problem, while generally falling under the umbrella of mechanism design, has several unique features. We propose a general formalism -- the token auction model -- for studying this problem. A key feature of this model is that it acts on a token-by-token basis and lets LLM agents influence generated contents through single dimensional bids.   We first explore a robust auction design approach, in which all we assume is that agent preferences entail partial orders over outcome distributions. We formulate two natural incentive properties, and show that these are equivalent to a monotonicity condition on distribution aggregation. We also show that for such aggregation functions, it is possible to design a second-price auction, despite the absence of bidder valuation functions. We then move to designing concrete aggregation functions by focusing on specific valuation forms based on KL-divergence, a commonly used loss function in LLM. The welfare-maximizing aggregation rules turn out to be the weighted (log-space) convex combination of the target distributions from all participants. We conclude with experimental results in support of the token auction formulation.

{{</citation>}}


## cs.IR (3)



### (143/176) If the Sources Could Talk: Evaluating Large Language Models for Research Assistance in History (Giselle Gonzalez Garcia et al., 2023)

{{<citation>}}

Giselle Gonzalez Garcia, Christian Weilbach. (2023)  
**If the Sources Could Talk: Evaluating Large Language Models for Research Assistance in History**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10808v1)  

---


**ABSTRACT**  
The recent advent of powerful Large-Language Models (LLM) provides a new conversational form of inquiry into historical memory (or, training data, in this case). We show that by augmenting such LLMs with vector embeddings from highly specialized academic sources, a conversational methodology can be made accessible to historians and other researchers in the Humanities. Concretely, we evaluate and demonstrate how LLMs have the ability of assisting researchers while they examine a customized corpora of different types of documents, including, but not exclusive to: (1). primary sources, (2). secondary sources written by experts, and (3). the combination of these two. Compared to established search interfaces for digital catalogues, such as metadata and full-text search, we evaluate the richer conversational style of LLMs on the performance of two main types of tasks: (1). question-answering, and (2). extraction and organization of data. We demonstrate that LLMs semantic retrieval and reasoning abilities on problem-specific tasks can be applied to large textual archives that have not been part of the its training data. Therefore, LLMs can be augmented with sources relevant to specific research projects, and can be queried privately by researchers.

{{</citation>}}


### (144/176) Rethinking Financial Service Promotion With Hybrid Recommender Systems at PicPay (Gabriel Mendonça et al., 2023)

{{<citation>}}

Gabriel Mendonça, Matheus Santos, André Gonçalves, Yan Almeida. (2023)  
**Rethinking Financial Service Promotion With Hybrid Recommender Systems at PicPay**  

---
Primary Category: cs.IR  
Categories: J-1, cs-AI, cs-IR, cs.IR  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2310.10268v1)  

---


**ABSTRACT**  
The fintech PicPay offers a wide range of financial services to its 30 million monthly active users, with more than 50 thousand items recommended in the PicPay mobile app. In this scenario, promoting specific items that are strategic to the company can be very challenging. In this work, we present a Switching Hybrid Recommender System that combines two algorithms to effectively promote items without negatively impacting the user's experience. The results of our A/B tests show an uplift of up to 3.2\% when compared to a default recommendation strategy.

{{</citation>}}


### (145/176) On Generative Agents in Recommendation (An Zhang et al., 2023)

{{<citation>}}

An Zhang, Leheng Sheng, Yuxin Chen, Hao Li, Yang Deng, Xiang Wang, Tat-Seng Chua. (2023)  
**On Generative Agents in Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10108v1)  

---


**ABSTRACT**  
Recommender systems are the cornerstone of today's information dissemination, yet a disconnect between offline metrics and online performance greatly hinders their development. Addressing this challenge, we envision a recommendation simulator, capitalizing on recent breakthroughs in human-level intelligence exhibited by Large Language Models (LLMs). We propose Agent4Rec, a novel movie recommendation simulator, leveraging LLM-empowered generative agents equipped with user profile, memory, and actions modules specifically tailored for the recommender system. In particular, these agents' profile modules are initialized using the MovieLens dataset, capturing users' unique tastes and social traits; memory modules log both factual and emotional memories and are integrated with an emotion-driven reflection mechanism; action modules support a wide variety of behaviors, spanning both taste-driven and emotion-driven actions. Each agent interacts with personalized movie recommendations in a page-by-page manner, relying on a pre-implemented collaborative filtering-based recommendation algorithm. We delve into both the capabilities and limitations of Agent4Rec, aiming to explore an essential research question: to what extent can LLM-empowered generative agents faithfully simulate the behavior of real, autonomous humans in recommender systems? Extensive and multi-faceted evaluations of Agent4Rec highlight both the alignment and deviation between agents and user-personalized preferences. Beyond mere performance comparison, we explore insightful experiments, such as emulating the filter bubble effect and discovering the underlying causal relationships in recommendation tasks. Our codes are available at https://github.com/LehengTHU/Agent4Rec.

{{</citation>}}


## eess.IV (6)



### (146/176) Convolutional Neural Network Model for Diabetic Retinopathy Feature Extraction and Classification (Sharan Subramanian et al., 2023)

{{<citation>}}

Sharan Subramanian, Leilani H. Gilpin. (2023)  
**Convolutional Neural Network Model for Diabetic Retinopathy Feature Extraction and Classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10806v1)  

---


**ABSTRACT**  
The application of Artificial Intelligence in the medical market brings up increasing concerns but aids in more timely diagnosis of silent progressing diseases like Diabetic Retinopathy. In order to diagnose Diabetic Retinopathy (DR), ophthalmologists use color fundus images, or pictures of the back of the retina, to identify small distinct features through a difficult and time-consuming process. Our work creates a novel CNN model and identifies the severity of DR through fundus image input. We classified 4 known DR features, including micro-aneurysms, cotton wools, exudates, and hemorrhages, through convolutional layers and were able to provide an accurate diagnostic without additional user input. The proposed model is more interpretable and robust to overfitting. We present initial results with a sensitivity of 97% and an accuracy of 71%. Our contribution is an interpretable model with similar accuracy to more complex models. With that, our model advances the field of DR detection and proves to be a key step towards AI-focused medical diagnosis.

{{</citation>}}


### (147/176) A cross Transformer for image denoising (Chunwei Tian et al., 2023)

{{<citation>}}

Chunwei Tian, Menghua Zheng, Wangmeng Zuo, Shichao Zhang, Yanning Zhang, Chia-Wen Ling. (2023)  
**A cross Transformer for image denoising**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.10408v1)  

---


**ABSTRACT**  
Deep convolutional neural networks (CNNs) depend on feedforward and feedback ways to obtain good performance in image denoising. However, how to obtain effective structural information via CNNs to efficiently represent given noisy images is key for complex scenes. In this paper, we propose a cross Transformer denoising CNN (CTNet) with a serial block (SB), a parallel block (PB), and a residual block (RB) to obtain clean images for complex scenes. A SB uses an enhanced residual architecture to deeply search structural information for image denoising. To avoid loss of key information, PB uses three heterogeneous networks to implement multiple interactions of multi-level features to broadly search for extra information for improving the adaptability of an obtained denoiser for complex scenes. Also, to improve denoising performance, Transformer mechanisms are embedded into the SB and PB to extract complementary salient features for effectively removing noise in terms of pixel relations. Finally, a RB is applied to acquire clean images. Experiments illustrate that our CTNet is superior to some popular denoising methods in terms of real and synthetic image denoising. It is suitable to mobile digital devices, i.e., phones. Codes can be obtained at https://github.com/hellloxiaotian/CTNet.

{{</citation>}}


### (148/176) A Multi-Scale Spatial Transformer U-Net for Simultaneously Automatic Reorientation and Segmentation of 3D Nuclear Cardiac Images (Yangfan Ni et al., 2023)

{{<citation>}}

Yangfan Ni, Duo Zhang, Gege Ma, Lijun Lu, Zhongke Huang, Wentao Zhu. (2023)  
**A Multi-Scale Spatial Transformer U-Net for Simultaneously Automatic Reorientation and Segmentation of 3D Nuclear Cardiac Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.10095v1)  

---


**ABSTRACT**  
Accurate reorientation and segmentation of the left ventricular (LV) is essential for the quantitative analysis of myocardial perfusion imaging (MPI), in which one critical step is to reorient the reconstructed transaxial nuclear cardiac images into standard short-axis slices for subsequent image processing. Small-scale LV myocardium (LV-MY) region detection and the diverse cardiac structures of individual patients pose challenges to LV segmentation operation. To mitigate these issues, we propose an end-to-end model, named as multi-scale spatial transformer UNet (MS-ST-UNet), that involves the multi-scale spatial transformer network (MSSTN) and multi-scale UNet (MSUNet) modules to perform simultaneous reorientation and segmentation of LV region from nuclear cardiac images. The proposed method is trained and tested using two different nuclear cardiac image modalities: 13N-ammonia PET and 99mTc-sestamibi SPECT. We use a multi-scale strategy to generate and extract image features with different scales. Our experimental results demonstrate that the proposed method significantly improves the reorientation and segmentation performance. This joint learning framework promotes mutual enhancement between reorientation and segmentation tasks, leading to cutting edge performance and an efficient image processing workflow. The proposed end-to-end deep network has the potential to reduce the burden of manual delineation for cardiac images, thereby providing multimodal quantitative analysis assistance for physicists.

{{</citation>}}


### (149/176) PUCA: Patch-Unshuffle and Channel Attention for Enhanced Self-Supervised Image Denoising (Hyemi Jang et al., 2023)

{{<citation>}}

Hyemi Jang, Junsung Park, Dahuin Jung, Jaihyun Lew, Ho Bae, Sungroh Yoon. (2023)  
**PUCA: Patch-Unshuffle and Channel Attention for Enhanced Self-Supervised Image Denoising**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.10088v1)  

---


**ABSTRACT**  
Although supervised image denoising networks have shown remarkable performance on synthesized noisy images, they often fail in practice due to the difference between real and synthesized noise. Since clean-noisy image pairs from the real world are extremely costly to gather, self-supervised learning, which utilizes noisy input itself as a target, has been studied. To prevent a self-supervised denoising model from learning identical mapping, each output pixel should not be influenced by its corresponding input pixel; This requirement is known as J-invariance. Blind-spot networks (BSNs) have been a prevalent choice to ensure J-invariance in self-supervised image denoising. However, constructing variations of BSNs by injecting additional operations such as downsampling can expose blinded information, thereby violating J-invariance. Consequently, convolutions designed specifically for BSNs have been allowed only, limiting architectural flexibility. To overcome this limitation, we propose PUCA, a novel J-invariant U-Net architecture, for self-supervised denoising. PUCA leverages patch-unshuffle/shuffle to dramatically expand receptive fields while maintaining J-invariance and dilated attention blocks (DABs) for global context incorporation. Experimental results demonstrate that PUCA achieves state-of-the-art performance, outperforming existing methods in self-supervised image denoising.

{{</citation>}}


### (150/176) Assessing Encoder-Decoder Architectures for Robust Coronary Artery Segmentation (Shisheng Zhang et al., 2023)

{{<citation>}}

Shisheng Zhang, Ramtin Gharleghi, Sonit Singh, Arcot Sowmya, Susann Beier. (2023)  
**Assessing Encoder-Decoder Architectures for Robust Coronary Artery Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10002v1)  

---


**ABSTRACT**  
Coronary artery diseases are among the leading causes of mortality worldwide. Timely and accurate diagnosis, facilitated by precise coronary artery segmentation, is pivotal in changing patient outcomes. In the realm of biomedical imaging, convolutional neural networks, especially the U-Net architecture, have revolutionised segmentation processes. However, one of the primary challenges remains the lack of benchmarking datasets specific to coronary arteries. However through the use of the recently published public dataset ASOCA, the potential of deep learning for accurate coronary segmentation can be improved. This paper delves deep into examining the performance of 25 distinct encoder-decoder combinations. Through analysis of the 40 cases provided to ASOCA participants, it is revealed that the EfficientNet-LinkNet combination, serving as encoder and decoder, stands out. It achieves a Dice coefficient of 0.882 and a 95th percentile Hausdorff distance of 4.753. These findings not only underscore the superiority of our model in comparison to those presented at the MICCAI 2020 challenge but also set the stage for future advancements in coronary artery segmentation, opening doors to enhanced diagnostic and treatment strategies.

{{</citation>}}


### (151/176) SeUNet-Trans: A Simple yet Effective UNet-Transformer Model for Medical Image Segmentation (Tan-Hanh Pham et al., 2023)

{{<citation>}}

Tan-Hanh Pham, Xianqi Li, Kim-Doang Nguyen. (2023)  
**SeUNet-Trans: A Simple yet Effective UNet-Transformer Model for Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.09998v1)  

---


**ABSTRACT**  
Automated medical image segmentation is becoming increasingly crucial in modern clinical practice, driven by the growing demand for precise diagnoses, the push towards personalized treatment plans, and advancements in machine learning algorithms, especially the incorporation of deep learning methods. While convolutional neural networks (CNNs) have been prevalent among these methods, the remarkable potential of Transformer-based models for computer vision tasks is gaining more acknowledgment. To harness the advantages of both CNN-based and Transformer-based models, we propose a simple yet effective UNet-Transformer (seUNet-Trans) model for medical image segmentation. In our approach, the UNet model is designed as a feature extractor to generate multiple feature maps from the input images, and these maps are propagated into a bridge layer, which sequentially connects the UNet and the Transformer. In this stage, we employ the pixel-level embedding technique without position embedding vectors to make the model more efficient. Moreover, we applied spatial-reduction attention in the Transformer to reduce the computational/memory overhead. By leveraging the UNet architecture and the self-attention mechanism, our model not only preserves both local and global context information but also captures long-range dependencies between input elements. The proposed model is extensively experimented on five medical image segmentation datasets, including polyp segmentation, to demonstrate its efficacy. A comparison with several state-of-the-art segmentation models on these datasets shows the superior performance of seUNet-Trans.

{{</citation>}}


## cs.PL (1)



### (152/176) Three Quantum Programming Language Parser Implementations for the Web (Marcus Edwards, 2023)

{{<citation>}}

Marcus Edwards. (2023)  
**Three Quantum Programming Language Parser Implementations for the Web**  

---
Primary Category: cs.PL  
Categories: cs-PL, cs.PL, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.10802v1)  

---


**ABSTRACT**  
IBM has developed a quantum assembly (QASM) language particular to gate model quantum computing since 2017 [CBSG17]. Version 3.0 which adds timing, pulse control, and gate modifiers is currently undergoing finalization in 2023 [CJA+21]. In a similar vein, Pakin of Los Alamos National Laboratory published a quantum macro assembler (QMASM) for D-Wave quantum annealers in 2016 [Pak16]. This assembler specifically targets quantum annealers like D-Wave's. A comparable technology that targets continuous-variable (CV) quantum computing is the Blackbird language developed by Xanadu since 2018 [KIQ+19]. We implement parsers for each of these languages in TypeScript with a singular approach. In the cases of Blackbird and QMASM these are the first parser implementations that are web compatible and so bring these languages to a new audience and to new runtimes. This makes the parsing and execution of QMASM, QASM and Blackbird possible in web and mobile environments that don't have access to heavy compile toolchains, enabling adoption and scientific research.

{{</citation>}}


## eess.AS (2)



### (153/176) Self-Supervised Models of Speech Infer Universal Articulatory Kinematics (Cheol Jun Cho et al., 2023)

{{<citation>}}

Cheol Jun Cho, Abdelrahman Mohamed, Alan W Black, Gopala K. Anumanchipalli. (2023)  
**Self-Supervised Models of Speech Infer Universal Articulatory Kinematics**  

---
Primary Category: eess.AS  
Categories: cs-CL, eess-AS, eess.AS  
Keywords: AI, BERT, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.10788v1)  

---


**ABSTRACT**  
Self-Supervised Learning (SSL) based models of speech have shown remarkable performance on a range of downstream tasks. These state-of-the-art models have remained blackboxes, but many recent studies have begun "probing" models like HuBERT, to correlate their internal representations to different aspects of speech. In this paper, we show "inference of articulatory kinematics" as fundamental property of SSL models, i.e., the ability of these models to transform acoustics into the causal articulatory dynamics underlying the speech signal. We also show that this abstraction is largely overlapping across the language of the data used to train the model, with preference to the language with similar phonological system. Furthermore, we show that with simple affine transformations, Acoustic-to-Articulatory inversion (AAI) is transferrable across speakers, even across genders, languages, and dialects, showing the generalizability of this property. Together, these results shed new light on the internals of SSL models that are critical to their superior performance, and open up new avenues into language-agnostic universal models for speech engineering, that are interpretable and grounded in speech science.

{{</citation>}}


### (154/176) Advancing Audio Emotion and Intent Recognition with Large Pre-Trained Models and Bayesian Inference (Dejan Porjazovski et al., 2023)

{{<citation>}}

Dejan Porjazovski, Yaroslav Getman, Tamás Grósz, Mikko Kurimo. (2023)  
**Advancing Audio Emotion and Intent Recognition with Large Pre-Trained Models and Bayesian Inference**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Intent Recognition, Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2310.10179v1)  

---


**ABSTRACT**  
Large pre-trained models are essential in paralinguistic systems, demonstrating effectiveness in tasks like emotion recognition and stuttering detection. In this paper, we employ large pre-trained models for the ACM Multimedia Computational Paralinguistics Challenge, addressing the Requests and Emotion Share tasks. We explore audio-only and hybrid solutions leveraging audio and text modalities. Our empirical results consistently show the superiority of the hybrid approaches over the audio-only models. Moreover, we introduce a Bayesian layer as an alternative to the standard linear output layer. The multimodal fusion approach achieves an 85.4% UAR on HC-Requests and 60.2% on HC-Complaints. The ensemble model for the Emotion Share task yields the best rho value of .614. The Bayesian wav2vec2 approach, explored in this study, allows us to easily build ensembles, at the cost of fine-tuning only one model. Moreover, we can have usable confidence values instead of the usual overconfident posterior probabilities.

{{</citation>}}


## cs.CR (3)



### (155/176) Security in Cryptocurrency (Chelsea Medina et al., 2023)

{{<citation>}}

Chelsea Medina, Lily Shaw, Dissy Vargas, Sundar Krishnan. (2023)  
**Security in Cryptocurrency**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.10768v1)  

---


**ABSTRACT**  
This paper discusses the mechanisms of cryptocurrency, the idea of using security in the system, and the popularity of it. To begin, the authors provide a background on cryptocurrency and how it works. The authors understand that while most people may be familiar with the concept, they may not know how it works. Next, the authors discuss the security of cryptocurrency in-depth within the paper. The authors also provide examples of attacks on cryptocurrency systems to show the vulnerabilities within the system. Lastly, the authors discuss the popularity of the system to further express the need for security in cryptocurrency.

{{</citation>}}


### (156/176) A Multilayered Security Infrastructure for Connected Vehicles -- First Lessons from the Field (Timo Häckel et al., 2023)

{{<citation>}}

Timo Häckel, Philipp Meyer, Lukas Stahlbock, Falk Langer, Sebastian A. Eckhardt, Franz Korf, Thomas C. Schmidt. (2023)  
**A Multilayered Security Infrastructure for Connected Vehicles -- First Lessons from the Field**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.10336v1)  

---


**ABSTRACT**  
Connected vehicles are vulnerable to manipulation and a broad attack surface can be used to intrude in-vehicle networks from anywhere on earth. In this work, we present an integrated security infrastructure comprising network protection, monitoring, incident management, and counteractions, which we built into a prototype based on a production car. Our vehicle implements a Software-Defined Networking Ethernet backbone to restrict communication routes, network anomaly detection to make misbehavior evident, virtual controller functions to enable agile countermeasures, and an automotive cloud defense center to analyse and manage incidents on vehicle fleets. We present first measurements and lessons learned from operating the prototype: many network attacks can be prevented through software-defined access control in the backbone; anomaly detection can reliably detect misbehavior but needs to improve on false positive rate; controller virtualization needs tailored frameworks to meet in-car requirements; and cloud defence enables fleet management and advanced countermeasures. Our findings indicate attack mitigation times in the vehicle from 257 ms to 328 ms and from 2,168 ms to 2,713 ms traversing the cloud.

{{</citation>}}


### (157/176) A Comprehensive Study of Privacy Risks in Curriculum Learning (Joann Qiongna Chen et al., 2023)

{{<citation>}}

Joann Qiongna Chen, Xinlei He, Zheng Li, Yang Zhang, Zhou Li. (2023)  
**A Comprehensive Study of Privacy Risks in Curriculum Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10124v1)  

---


**ABSTRACT**  
Training a machine learning model with data following a meaningful order, i.e., from easy to hard, has been proven to be effective in accelerating the training process and achieving better model performance. The key enabling technique is curriculum learning (CL), which has seen great success and has been deployed in areas like image and text classification. Yet, how CL affects the privacy of machine learning is unclear. Given that CL changes the way a model memorizes the training data, its influence on data privacy needs to be thoroughly evaluated. To fill this knowledge gap, we perform the first study and leverage membership inference attack (MIA) and attribute inference attack (AIA) as two vectors to quantify the privacy leakage caused by CL.   Our evaluation of nine real-world datasets with attack methods (NN-based, metric-based, label-only MIA, and NN-based AIA) revealed new insights about CL. First, MIA becomes slightly more effective when CL is applied, but the impact is much more prominent to a subset of training samples ranked as difficult. Second, a model trained under CL is less vulnerable under AIA, compared to MIA. Third, the existing defense techniques like DP-SGD, MemGuard, and MixupMMD are still effective under CL, though DP-SGD has a significant impact on target model accuracy. Finally, based on our insights into CL, we propose a new MIA, termed Diff-Cali, which exploits the difficulty scores for result calibration and is demonstrated to be effective against all CL methods and the normal training method. With this study, we hope to draw the community's attention to the unintended privacy risks of emerging machine-learning techniques and develop new attack benchmarks and defense solutions.

{{</citation>}}


## cs.AI (3)



### (158/176) Quantifying Assistive Robustness Via the Natural-Adversarial Frontier (Jerry Zhi-Yang He et al., 2023)

{{<citation>}}

Jerry Zhi-Yang He, Zackory Erickson, Daniel S. Brown, Anca D. Dragan. (2023)  
**Quantifying Assistive Robustness Via the Natural-Adversarial Frontier**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10610v1)  

---


**ABSTRACT**  
Our ultimate goal is to build robust policies for robots that assist people. What makes this hard is that people can behave unexpectedly at test time, potentially interacting with the robot outside its training distribution and leading to failures. Even just measuring robustness is a challenge. Adversarial perturbations are the default, but they can paint the wrong picture: they can correspond to human motions that are unlikely to occur during natural interactions with people. A robot policy might fail under small adversarial perturbations but work under large natural perturbations. We propose that capturing robustness in these interactive settings requires constructing and analyzing the entire natural-adversarial frontier: the Pareto-frontier of human policies that are the best trade-offs between naturalness and low robot performance. We introduce RIGID, a method for constructing this frontier by training adversarial human policies that trade off between minimizing robot reward and acting human-like (as measured by a discriminator). On an Assistive Gym task, we use RIGID to analyze the performance of standard collaborative Reinforcement Learning, as well as the performance of existing methods meant to increase robustness. We also compare the frontier RIGID identifies with the failures identified in expert adversarial interaction, and with naturally-occurring failures during user interaction. Overall, we find evidence that RIGID can provide a meaningful measure of robustness predictive of deployment performance, and uncover failure cases in human-robot interaction that are difficult to find manually. https://ood-human.github.io.

{{</citation>}}


### (159/176) Large Language Model-Empowered Agents for Simulating Macroeconomic Activities (Nian Li et al., 2023)

{{<citation>}}

Nian Li, Chen Gao, Yong Li, Qingmin Liao. (2023)  
**Large Language Model-Empowered Agents for Simulating Macroeconomic Activities**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10436v1)  

---


**ABSTRACT**  
The advent of the Web has brought about a paradigm shift in traditional economics, particularly in the digital economy era, enabling the precise recording and analysis of individual economic behavior. This has led to a growing emphasis on data-driven modeling in macroeconomics. In macroeconomic research, Agent-based modeling (ABM) emerged as an alternative, evolving through rule-based agents, machine learning-enhanced decision-making, and, more recently, advanced AI agents. However, the existing works are suffering from three main challenges when endowing agents with human-like decision-making, including agent heterogeneity, the influence of macroeconomic trends, and multifaceted economic factors. Large language models (LLMs) have recently gained prominence in offering autonomous human-like characteristics. Therefore, leveraging LLMs in macroeconomic simulation presents an opportunity to overcome traditional limitations. In this work, we take an early step in introducing a novel approach that leverages LLMs in macroeconomic simulation. We design prompt-engineering-driven LLM agents to exhibit human-like decision-making and adaptability in the economic environment, with the abilities of perception, reflection, and decision-making to address the abovementioned challenges. Simulation experiments on macroeconomic activities show that LLM-empowered agents can make realistic work and consumption decisions and emerge more reasonable macroeconomic phenomena than existing rule-based or AI agents. Our work demonstrates the promising potential to simulate macroeconomics based on LLM and its human-like characteristics.

{{</citation>}}


### (160/176) End-to-end Offline Reinforcement Learning for Glycemia Control (Tristan Beolet et al., 2023)

{{<citation>}}

Tristan Beolet, Alice Adenis, Erik Huneker, Maxime Louis. (2023)  
**End-to-end Offline Reinforcement Learning for Glycemia Control**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI, q-bio-QM  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10312v1)  

---


**ABSTRACT**  
The development of closed-loop systems for glycemia control in type I diabetes relies heavily on simulated patients. Improving the performances and adaptability of these close-loops raises the risk of over-fitting the simulator. This may have dire consequences, especially in unusual cases which were not faithfully-if at all-captured by the simulator. To address this, we propose to use offline RL agents, trained on real patient data, to perform the glycemia control. To further improve the performances, we propose an end-to-end personalization pipeline, which leverages offline-policy evaluation methods to remove altogether the need of a simulator, while still enabling an estimation of clinically relevant metrics for diabetes.

{{</citation>}}


## cs.NI (2)



### (161/176) Applications of Distributed Machine Learning for the Internet-of-Things: A Comprehensive Survey (Mai Le et al., 2023)

{{<citation>}}

Mai Le, Thien Huynh-The, Tan Do-Duy, Thai-Hoc Vu, Won-Joo Hwang, Quoc-Viet Pham. (2023)  
**Applications of Distributed Machine Learning for the Internet-of-Things: A Comprehensive Survey**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10549v1)  

---


**ABSTRACT**  
The emergence of new services and applications in emerging wireless networks (e.g., beyond 5G and 6G) has shown a growing demand for the usage of artificial intelligence (AI) in the Internet of Things (IoT). However, the proliferation of massive IoT connections and the availability of computing resources distributed across future IoT systems have strongly demanded the development of distributed AI for better IoT services and applications. Therefore, existing AI-enabled IoT systems can be enhanced by implementing distributed machine learning (aka distributed learning) approaches. This work aims to provide a comprehensive survey on distributed learning for IoT services and applications in emerging networks. In particular, we first provide a background of machine learning and present a preliminary to typical distributed learning approaches, such as federated learning, multi-agent reinforcement learning, and distributed inference. Then, we provide an extensive review of distributed learning for critical IoT services (e.g., data sharing and computation offloading, localization, mobile crowdsensing, and security and privacy) and IoT applications (e.g., smart healthcare, smart grid, autonomous vehicle, aerial IoT networks, and smart industry). From the reviewed literature, we also present critical challenges of distributed learning for IoT and propose several promising solutions and research directions in this emerging area.

{{</citation>}}


### (162/176) Unlocking Metasurface Practicality for B5G Networks: AI-assisted RIS Planning (Guillermo Encinas-Lago et al., 2023)

{{<citation>}}

Guillermo Encinas-Lago, Antonio Albanese, Vincenzo Sciancalepore, Marco Di Renzo, Xavier Costa-Pérez. (2023)  
**Unlocking Metasurface Practicality for B5G Networks: AI-assisted RIS Planning**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10330v1)  

---


**ABSTRACT**  
The advent of reconfigurable intelligent surfaces(RISs) brings along significant improvements for wireless technology on the verge of beyond-fifth-generation networks (B5G).The proven flexibility in influencing the propagation environment opens up the possibility of programmatically altering the wireless channel to the advantage of network designers, enabling the exploitation of higher-frequency bands for superior throughput overcoming the challenging electromagnetic (EM) propagation properties at these frequency bands.   However, RISs are not magic bullets. Their employment comes with significant complexity, requiring ad-hoc deployments and management operations to come to fruition. In this paper, we tackle the open problem of bringing RISs to the field, focusing on areas with little or no coverage. In fact, we present a first-of-its-kind deep reinforcement learning (DRL) solution, dubbed as D-RISA, which trains a DRL agent and, in turn, obtain san optimal RIS deployment. We validate our framework in the indoor scenario of the Rennes railway station in France, assessing the performance of our algorithm against state-of-the-art (SOA) approaches. Our benchmarks showcase better coverage, i.e., 10-dB increase in minimum signal-to-noise ratio (SNR), at lower computational time (up to -25 percent) while improving scalability towards denser network deployments.

{{</citation>}}


## q-bio.NC (1)



### (163/176) Use of probabilistic phrases in a coordination game: human versus GPT-4 (Laurence T Maloney et al., 2023)

{{<citation>}}

Laurence T Maloney, Maria F Dal Martello, Vivian Fei, Valerie Ma. (2023)  
**Use of probabilistic phrases in a coordination game: human versus GPT-4**  

---
Primary Category: q-bio.NC  
Categories: cs-AI, q-bio-NC, q-bio.NC  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10544v1)  

---


**ABSTRACT**  
English speakers use probabilistic phrases such as likely to communicate information about the probability or likelihood of events. Communication is successful to the extent that the listener grasps what the speaker means to convey and, if communication is successful, two individuals can potentially coordinate their actions based on shared knowledge about uncertainty. We first assessed human ability to estimate the probability and the ambiguity (imprecision) of 23 probabilistic phrases in two different contexts, investment advice and medical advice. We then had GPT4 (OpenAI), a recent Large Language Model, complete the same tasks as the human participants. We found that the median human participant and GPT4 assigned probability estimates that were in good agreement (proportions of variance accounted were close to .90). GPT4's estimates of probability both in the investment and Medical contexts were as close or closer to that of the human participants as the human participants were to one another. Estimates of probability for both the human participants and GPT4 were little affected by context. In contrast, human and GPT4 estimates of ambiguity were not in as good agreement. We repeated some of the GPT4 estimates to assess their stability: does GPT4, if run twice, produce the same or similar estimates? There is some indication that it does not.

{{</citation>}}


## q-fin.TR (1)



### (164/176) Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies (Kieran Wood et al., 2023)

{{<citation>}}

Kieran Wood, Samuel Kessler, Stephen J. Roberts, Stefan Zohren. (2023)  
**Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies**  

---
Primary Category: q-fin.TR  
Categories: cs-LG, q-fin-PM, q-fin-TR, q-fin.TR  
Keywords: Few-Shot, Financial  
[Paper Link](http://arxiv.org/abs/2310.10500v1)  

---


**ABSTRACT**  
Forecasting models for systematic trading strategies do not adapt quickly when financial market conditions change, as was seen in the advent of the COVID-19 pandemic in 2020, when market conditions changed dramatically causing many forecasting models to take loss-making positions. To deal with such situations, we propose a novel time-series trend-following forecaster that is able to quickly adapt to new market conditions, referred to as regimes. We leverage recent developments from the deep learning community and use few-shot learning. We propose the Cross Attentive Time-Series Trend Network - X-Trend - which takes positions attending over a context set of financial time-series regimes. X-Trend transfers trends from similar patterns in the context set to make predictions and take positions for a new distinct target regime. X-Trend is able to quickly adapt to new financial regimes with a Sharpe ratio increase of 18.9% over a neural forecaster and 10-fold over a conventional Time-series Momentum strategy during the turbulent market period from 2018 to 2023. Our strategy recovers twice as quickly from the COVID-19 drawdown compared to the neural-forecaster. X-Trend can also take zero-shot positions on novel unseen financial assets obtaining a 5-fold Sharpe ratio increase versus a neural time-series trend forecaster over the same period. X-Trend both forecasts next-day prices and outputs a trading signal. Furthermore, the cross-attention mechanism allows us to interpret the relationship between forecasts and patterns in the context set.

{{</citation>}}


## cs.SD (3)



### (165/176) LocSelect: Target Speaker Localization with an Auditory Selective Hearing Mechanism (Yu Chen et al., 2023)

{{<citation>}}

Yu Chen, Xinyuan Qian, Zexu Pan, Kainan Chen, Haizhou Li. (2023)  
**LocSelect: Target Speaker Localization with an Auditory Selective Hearing Mechanism**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.10497v2)  

---


**ABSTRACT**  
The prevailing noise-resistant and reverberation-resistant localization algorithms primarily emphasize separating and providing directional output for each speaker in multi-speaker scenarios, without association with the identity of speakers. In this paper, we present a target speaker localization algorithm with a selective hearing mechanism. Given a reference speech of the target speaker, we first produce a speaker-dependent spectrogram mask to eliminate interfering speakers' speech. Subsequently, a Long short-term memory (LSTM) network is employed to extract the target speaker's location from the filtered spectrogram. Experiments validate the superiority of our proposed method over the existing algorithms for different scale invariant signal-to-noise ratios (SNR) conditions. Specifically, at SNR = -10 dB, our proposed network LocSelect achieves a mean absolute error (MAE) of 3.55 and an accuracy (ACC) of 87.40%.

{{</citation>}}


### (166/176) BeatDance: A Beat-Based Model-Agnostic Contrastive Learning Framework for Music-Dance Retrieval (Kaixing Yang et al., 2023)

{{<citation>}}

Kaixing Yang, Xukun Zhou, Xulong Tang, Ran Diao, Hongyan Liu, Jun He, Zhaoxin Fan. (2023)  
**BeatDance: A Beat-Based Model-Agnostic Contrastive Learning Framework for Music-Dance Retrieval**  

---
Primary Category: cs.SD  
Categories: cs-IR, cs-SD, cs.SD, eess-AS  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.10300v1)  

---


**ABSTRACT**  
Dance and music are closely related forms of expression, with mutual retrieval between dance videos and music being a fundamental task in various fields like education, art, and sports. However, existing methods often suffer from unnatural generation effects or fail to fully explore the correlation between music and dance. To overcome these challenges, we propose BeatDance, a novel beat-based model-agnostic contrastive learning framework. BeatDance incorporates a Beat-Aware Music-Dance InfoExtractor, a Trans-Temporal Beat Blender, and a Beat-Enhanced Hubness Reducer to improve dance-music retrieval performance by utilizing the alignment between music beats and dance movements. We also introduce the Music-Dance (MD) dataset, a large-scale collection of over 10,000 music-dance video pairs for training and testing. Experimental results on the MD dataset demonstrate the superiority of our method over existing baselines, achieving state-of-the-art performance. The code and dataset will be made public available upon acceptance.

{{</citation>}}


### (167/176) Joint Music and Language Attention Models for Zero-shot Music Tagging (Xingjian Du et al., 2023)

{{<citation>}}

Xingjian Du, Zhesong Yu, Jiaju Lin, Bilei Zhu, Qiuqiang Kong. (2023)  
**Joint Music and Language Attention Models for Zero-shot Music Tagging**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Attention, ChatGPT, Falcon, GPT  
[Paper Link](http://arxiv.org/abs/2310.10159v1)  

---


**ABSTRACT**  
Music tagging is a task to predict the tags of music recordings. However, previous music tagging research primarily focuses on close-set music tagging tasks which can not be generalized to new tags. In this work, we propose a zero-shot music tagging system modeled by a joint music and language attention (JMLA) model to address the open-set music tagging problem. The JMLA model consists of an audio encoder modeled by a pretrained masked autoencoder and a decoder modeled by a Falcon7B. We introduce preceiver resampler to convert arbitrary length audio into fixed length embeddings. We introduce dense attention connections between encoder and decoder layers to improve the information flow between the encoder and decoder layers. We collect a large-scale music and description dataset from the internet. We propose to use ChatGPT to convert the raw descriptions into formalized and diverse descriptions to train the JMLA models. Our proposed JMLA system achieves a zero-shot audio tagging accuracy of $ 64.82\% $ on the GTZAN dataset, outperforming previous zero-shot systems and achieves comparable results to previous systems on the FMA and the MagnaTagATune datasets.

{{</citation>}}


## stat.ML (4)



### (168/176) A Geometric Insight into Equivariant Message Passing Neural Networks on Riemannian Manifolds (Ilyes Batatia, 2023)

{{<citation>}}

Ilyes Batatia. (2023)  
**A Geometric Insight into Equivariant Message Passing Neural Networks on Riemannian Manifolds**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.10448v1)  

---


**ABSTRACT**  
This work proposes a geometric insight into equivariant message passing on Riemannian manifolds. As previously proposed, numerical features on Riemannian manifolds are represented as coordinate-independent feature fields on the manifold. To any coordinate-independent feature field on a manifold comes attached an equivariant embedding of the principal bundle to the space of numerical features. We argue that the metric this embedding induces on the numerical feature space should optimally preserve the principal bundle's original metric. This optimality criterion leads to the minimization of a twisted form of the Polyakov action with respect to the graph of this embedding, yielding an equivariant diffusion process on the associated vector bundle. We obtain a message passing scheme on the manifold by discretizing the diffusion equation flow for a fixed time step. We propose a higher-order equivariant diffusion process equivalent to diffusion on the cartesian product of the base manifold. The discretization of the higher-order diffusion process on a graph yields a new general class of equivariant GNN, generalizing the ACE and MACE formalism to data on Riemannian manifolds.

{{</citation>}}


### (169/176) Equivariant Matrix Function Neural Networks (Ilyes Batatia et al., 2023)

{{<citation>}}

Ilyes Batatia, Lars L. Schaaf, Huajie Chen, Gábor Csányi, Christoph Ortner, Felix A. Faber. (2023)  
**Equivariant Matrix Function Neural Networks**  

---
Primary Category: stat.ML  
Categories: cond-mat-mtrl-sci, cs-LG, physics-chem-ph, stat-ML, stat.ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.10434v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs), especially message-passing neural networks (MPNNs), have emerged as powerful architectures for learning on graphs in diverse applications. However, MPNNs face challenges when modeling non-local interactions in systems such as large conjugated molecules, metals, or amorphous materials. Although Spectral GNNs and traditional neural networks such as recurrent neural networks and transformers mitigate these challenges, they often lack extensivity, adaptability, generalizability, computational efficiency, or fail to capture detailed structural relationships or symmetries in the data. To address these concerns, we introduce Matrix Function Neural Networks (MFNs), a novel architecture that parameterizes non-local interactions through analytic matrix equivariant functions. Employing resolvent expansions offers a straightforward implementation and the potential for linear scaling with system size. The MFN architecture achieves state-of-the-art performance in standard graph benchmarks, such as the ZINC and TU datasets, and is able to capture intricate non-local interactions in quantum systems, paving the way to new state-of-the-art force fields.

{{</citation>}}


### (170/176) An Anytime Algorithm for Good Arm Identification (Marc Jourdan et al., 2023)

{{<citation>}}

Marc Jourdan, Clémence Réda. (2023)  
**An Anytime Algorithm for Good Arm Identification**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10359v1)  

---


**ABSTRACT**  
In good arm identification (GAI), the goal is to identify one arm whose average performance exceeds a given threshold, referred to as good arm, if it exists. Few works have studied GAI in the fixed-budget setting, when the sampling budget is fixed beforehand, or the anytime setting, when a recommendation can be asked at any time. We propose APGAI, an anytime and parameter-free sampling rule for GAI in stochastic bandits. APGAI can be straightforwardly used in fixed-confidence and fixed-budget settings. First, we derive upper bounds on its probability of error at any time. They show that adaptive strategies are more efficient in detecting the absence of good arms than uniform sampling. Second, when APGAI is combined with a stopping rule, we prove upper bounds on the expected sampling complexity, holding at any confidence level. Finally, we show good empirical performance of APGAI on synthetic and real-world data. Our work offers an extensive overview of the GAI problem in all settings.

{{</citation>}}


### (171/176) An Empirical Study of Simplicial Representation Learning with Wasserstein Distance (Makoto Yamada et al., 2023)

{{<citation>}}

Makoto Yamada, Yuki Takezawa, Guillaume Houry, Kira Michaela Dusterwald, Deborah Sulem, Han Zhao, Yao-Hung Hubert Tsai. (2023)  
**An Empirical Study of Simplicial Representation Learning with Wasserstein Distance**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.10143v1)  

---


**ABSTRACT**  
In this paper, we delve into the problem of simplicial representation learning utilizing the 1-Wasserstein distance on a tree structure (a.k.a., Tree-Wasserstein distance (TWD)), where TWD is defined as the L1 distance between two tree-embedded vectors. Specifically, we consider a framework for simplicial representation estimation employing a self-supervised learning approach based on SimCLR with a negative TWD as a similarity measure. In SimCLR, the cosine similarity with real-vector embeddings is often utilized; however, it has not been well studied utilizing L1-based measures with simplicial embeddings. A key challenge is that training the L1 distance is numerically challenging and often yields unsatisfactory outcomes, and there are numerous choices for probability models. Thus, this study empirically investigates a strategy for optimizing self-supervised learning with TWD and find a stable training procedure. More specifically, we evaluate the combination of two types of TWD (total variation and ClusterTree) and several simplicial models including the softmax function, the ArcFace probability model, and simplicial embedding. Moreover, we propose a simple yet effective Jeffrey divergence-based regularization method to stabilize the optimization. Through empirical experiments on STL10, CIFAR10, CIFAR100, and SVHN, we first found that the simple combination of softmax function and TWD can obtain significantly lower results than the standard SimCLR (non-simplicial model and cosine similarity). We found that the model performance depends on the combination of TWD and the simplicial model, and the Jeffrey divergence regularization usually helps model training. Finally, we inferred that the appropriate choice of combination of TWD and simplicial models outperformed cosine similarity based representation learning.

{{</citation>}}


## cs.CY (1)



### (172/176) NLP for Crypto-Asset Regulation: A Roadmap (Carolina Camassa, 2023)

{{<citation>}}

Carolina Camassa. (2023)  
**NLP for Crypto-Asset Regulation: A Roadmap**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY, q-fin-GN  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.10333v2)  

---


**ABSTRACT**  
In the rapidly evolving field of crypto-assets, white papers are essential documents for investor guidance, and are now subject to unprecedented content requirements under the EU's Markets in Crypto-Assets Regulation (MiCAR). Natural Language Processing can serve as a powerful tool for both analyzing these documents and assisting in regulatory compliance. This paper delivers two contributions to the topic. First, we survey existing applications of textual analysis to unregulated crypto-asset white papers, uncovering a research gap that could be bridged with interdisciplinary collaboration. We then conduct an analysis of the changes introduced by MiCAR, highlighting the opportunities and challenges of integrating NLP within the new regulatory framework. The findings set the stage for further research, with the potential to benefit regulators, crypto-asset issuers, and investors.

{{</citation>}}


## cs.DB (1)



### (173/176) Node-based Knowledge Graph Contrastive Learning for Medical Relationship Prediction (Zhiguang Fan et al., 2023)

{{<citation>}}

Zhiguang Fan, Yuedong Yang, Mingyuan Xu, Hongming Chen. (2023)  
**Node-based Knowledge Graph Contrastive Learning for Medical Relationship Prediction**  

---
Primary Category: cs.DB  
Categories: cs-CL, cs-DB, cs.DB, q-bio-QM  
Keywords: Contrastive Learning, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.10138v1)  

---


**ABSTRACT**  
The embedding of Biomedical Knowledge Graphs (BKGs) generates robust representations, valuable for a variety of artificial intelligence applications, including predicting drug combinations and reasoning disease-drug relationships. Meanwhile, contrastive learning (CL) is widely employed to enhance the distinctiveness of these representations. However, constructing suitable contrastive pairs for CL, especially within Knowledge Graphs (KGs), has been challenging. In this paper, we proposed a novel node-based contrastive learning method for knowledge graph embedding, NC-KGE. NC-KGE enhances knowledge extraction in embeddings and speeds up training convergence by constructing appropriate contrastive node pairs on KGs. This scheme can be easily integrated with other knowledge graph embedding (KGE) methods. For downstream task such as biochemical relationship prediction, we have incorporated a relation-aware attention mechanism into NC-KGE, focusing on the semantic relationships and node interactions. Extensive experiments show that NC-KGE performs competitively with state-of-the-art models on public datasets like FB15k-237 and WN18RR. Particularly in biomedical relationship prediction tasks, NC-KGE outperforms all baselines on datasets such as PharmKG8k-28, DRKG17k-21, and BioKG72k-14, especially in predicting drug combination relationships. We release our code at https://github.com/zhi520/NC-KGE.

{{</citation>}}


## cs.NE (1)



### (174/176) Solution to Advanced Manufacturing Process Problems using Cohort Intelligence Algorithm with Improved Constraint Handling Approaches (Aniket Nargundkar et al., 2023)

{{<citation>}}

Aniket Nargundkar, Madhav Rawal, Aryaman Patel, Anand J Kulkarni, Apoorva S Shastri. (2023)  
**Solution to Advanced Manufacturing Process Problems using Cohort Intelligence Algorithm with Improved Constraint Handling Approaches**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10085v1)  

---


**ABSTRACT**  
Recently, various Artificial Intelligence (AI) based optimization metaheuristics are proposed and applied for a variety of problems. Cohort Intelligence (CI) algorithm is a socio inspired optimization technique which is successfully applied for solving several unconstrained & constrained real-world problems from the domains such as design, manufacturing, supply chain, healthcare, etc. Generally, real-world problems are constrained in nature. Even though most of the Evolutionary Algorithms (EAs) can efficiently solve unconstrained problems, their performance degenerates when the constraints are involved. In this paper, two novel constraint handling approaches based on modulus and hyperbolic tangent probability distributions are proposed. Constrained CI algorithm with constraint handling approaches based on triangular, modulus and hyperbolic tangent is presented and applied for optimizing advanced manufacturing processes such as Water Jet Machining (WJM), Abrasive Jet Machining (AJM), Ultrasonic Machining (USM) and Grinding process. The solutions obtained using proposed CI algorithm are compared with contemporary algorithms such as Genetic Algorithm, Simulated Annealing, Teaching Learning Based Optimization, etc. The proposed approaches achieved 2%-127% maximization of material removal rate satisfying hard constraints. As compared to the GA, CI with Hyperbolic tangent probability distribution achieved 15%, 2%, 2%, 127%, and 4% improvement in MRR for AJMB, AJMD, WJM, USM, and Grinding processes, respectively contributing to the productivity improvement. The contributions in this paper have opened several avenues for further applicability of the proposed constraint handling approaches for solving complex constrained problems.

{{</citation>}}


## cs.DC (1)



### (175/176) TRANSOM: An Efficient Fault-Tolerant System for Training LLMs (Baodong Wu et al., 2023)

{{<citation>}}

Baodong Wu, Lei Xia, Qingping Li, Kangyu Li, Xu Chen, Yongqiang Guo, Tieyao Xiang, Yuheng Chen, Shigang Li. (2023)  
**TRANSOM: An Efficient Fault-Tolerant System for Training LLMs**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs.DC  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.10046v3)  

---


**ABSTRACT**  
Large language models (LLMs) with hundreds of billions or trillions of parameters, represented by chatGPT, have achieved profound impact on various fields. However, training LLMs with super-large-scale parameters requires large high-performance GPU clusters and long training periods lasting for months. Due to the inevitable hardware and software failures in large-scale clusters, maintaining uninterrupted and long-duration training is extremely challenging. As a result, A substantial amount of training time is devoted to task checkpoint saving and loading, task rescheduling and restart, and task manual anomaly checks, which greatly harms the overall training efficiency. To address these issues, we propose TRANSOM, a novel fault-tolerant LLM training system. In this work, we design three key subsystems: the training pipeline automatic fault tolerance and recovery mechanism named Transom Operator and Launcher (TOL), the training task multi-dimensional metric automatic anomaly detection system named Transom Eagle Eye (TEE), and the training checkpoint asynchronous access automatic fault tolerance and recovery technology named Transom Checkpoint Engine (TCE). Here, TOL manages the lifecycle of training tasks, while TEE is responsible for task monitoring and anomaly reporting. TEE detects training anomalies and reports them to TOL, who automatically enters the fault tolerance strategy to eliminate abnormal nodes and restart the training task. And the asynchronous checkpoint saving and loading functionality provided by TCE greatly shorten the fault tolerance overhead. The experimental results indicate that TRANSOM significantly enhances the efficiency of large-scale LLM training on clusters. Specifically, the pre-training time for GPT3-175B has been reduced by 28%, while checkpoint saving and loading performance have improved by a factor of 20.

{{</citation>}}


## cs.SI (1)



### (176/176) Auditing Targeted Political Advertising on Social Media During the 2021 German Election (Dominik Bär et al., 2023)

{{<citation>}}

Dominik Bär, Francesco Pierri, Gianmarco De Francisci Morales, Stefan Feuerriegel. (2023)  
**Auditing Targeted Political Advertising on Social Media During the 2021 German Election**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2310.10001v1)  

---


**ABSTRACT**  
Political advertising on social media has become a central element in election campaigns. However, granular information about political advertising on social media was previously unavailable, thus raising concerns regarding fairness, accountability, and transparency in electoral processes. In this paper, we analyze targeted political advertising on social media using a unique, large-scale dataset of over 80000 political ads from Meta during the 2021 German federal election, with more than 1.1 billion impressions. For each political ad, our dataset records granular information about targeting strategies, spending, and actual impressions. We then study (i) the prevalence of targeted ads across the political spectrum; (ii) the discrepancies between targeted and actual audiences due to algorithmic distribution; and (iii) what makes an efficient targeting strategy on social media. We find that targeted ads are prevalent across the entire political spectrum, with considerable differences in strategies and efficiency between the political left and right. Furthermore, there are significant discrepancies between the targeted and actual audience, which vary across parties. Notably, the efficiency of political ads (as measured by impressions per EUR) is particularly high when ads are targeted at a broad audience, or published by far-right parties - which raises important fairness concerns. Overall, our work contributes to a better understanding of targeted political advertising on social media and informs policymakers about the design of effective regulatory frameworks to promote fairness, accountability, and transparency.

{{</citation>}}
