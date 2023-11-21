---
draft: false
title: "arXiv @ 2023.11.18"
date: 2023-11-18
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.18"
    identifier: arxiv_20231118
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (98)](#cscl-98)
- [eess.SY (2)](#eesssy-2)
- [cs.AI (10)](#csai-10)
- [eess.IV (8)](#eessiv-8)
- [cs.SI (1)](#cssi-1)
- [cs.CY (6)](#cscy-6)
- [cs.CR (5)](#cscr-5)
- [cs.RO (5)](#csro-5)
- [cs.LG (20)](#cslg-20)
- [cs.CV (20)](#cscv-20)
- [stat.ML (1)](#statml-1)
- [cs.HC (2)](#cshc-2)
- [cs.DS (1)](#csds-1)
- [cs.MM (1)](#csmm-1)
- [cond-mat.other (1)](#cond-matother-1)
- [cs.IT (1)](#csit-1)
- [cs.SE (1)](#csse-1)
- [cs.LO (1)](#cslo-1)
- [cs.GT (1)](#csgt-1)
- [cs.AR (1)](#csar-1)
- [cs.SD (3)](#cssd-3)
- [physics.optics (1)](#physicsoptics-1)
- [cs.DC (1)](#csdc-1)
- [cs.DL (1)](#csdl-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.IR (3)](#csir-3)
- [physics.soc-ph (1)](#physicssoc-ph-1)

## cs.CL (98)



### (1/197) Latent Feature-based Data Splits to Improve Generalisation Evaluation: A Hate Speech Detection Case Study (Maike Züfle et al., 2023)

{{<citation>}}

Maike Züfle, Verna Dankers, Ivan Titov. (2023)  
**Latent Feature-based Data Splits to Improve Generalisation Evaluation: A Hate Speech Detection Case Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Hate Speech Detection  
[Paper Link](http://arxiv.org/abs/2311.10236v1)  

---


**ABSTRACT**  
With the ever-growing presence of social media platforms comes the increased spread of harmful content and the need for robust hate speech detection systems. Such systems easily overfit to specific targets and keywords, and evaluating them without considering distribution shifts that might occur between train and test data overestimates their benefit. We challenge hate speech models via new train-test splits of existing datasets that rely on the clustering of models' hidden representations. We present two split variants (Subset-Sum-Split and Closest-Split) that, when applied to two datasets using four pretrained models, reveal how models catastrophically fail on blind spots in the latent space. This result generalises when developing a split with one model and evaluating it on another. Our analysis suggests that there is no clear surface-level property of the data split that correlates with the decreased performance, which underscores that task difficulty is not always humanly interpretable. We recommend incorporating latent feature-based splits in model development and release two splits via the GenBench benchmark.

{{</citation>}}


### (2/197) Predictive Minds: LLMs As Atypical Active Inference Agents (Jan Kulveit et al., 2023)

{{<citation>}}

Jan Kulveit, Clem von Stengel, Roman Leventov. (2023)  
**Predictive Minds: LLMs As Atypical Active Inference Agents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.10215v1)  

---


**ABSTRACT**  
Large language models (LLMs) like GPT are often conceptualized as passive predictors, simulators, or even stochastic parrots. We instead conceptualize LLMs by drawing on the theory of active inference originating in cognitive science and neuroscience. We examine similarities and differences between traditional active inference systems and LLMs, leading to the conclusion that, currently, LLMs lack a tight feedback loop between acting in the world and perceiving the impacts of their actions, but otherwise fit in the active inference paradigm. We list reasons why this loop may soon be closed, and possible consequences of this including enhanced model self-awareness and the drive to minimize prediction error by changing the world.

{{</citation>}}


### (3/197) JWSign: A Highly Multilingual Corpus of Bible Translations for more Diversity in Sign Language Processing (Shester Gueuwou et al., 2023)

{{<citation>}}

Shester Gueuwou, Sophie Siake, Colin Leong, Mathias Müller. (2023)  
**JWSign: A Highly Multilingual Corpus of Bible Translations for more Diversity in Sign Language Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2311.10174v1)  

---


**ABSTRACT**  
Advancements in sign language processing have been hindered by a lack of sufficient data, impeding progress in recognition, translation, and production tasks. The absence of comprehensive sign language datasets across the world's sign languages has widened the gap in this field, resulting in a few sign languages being studied more than others, making this research area extremely skewed mostly towards sign languages from high-income countries. In this work we introduce a new large and highly multilingual dataset for sign language translation: JWSign. The dataset consists of 2,530 hours of Bible translations in 98 sign languages, featuring more than 1,500 individual signers. On this dataset, we report neural machine translation experiments. Apart from bilingual baseline systems, we also train multilingual systems, including some that take into account the typological relatedness of signed or spoken languages. Our experiments highlight that multilingual systems are superior to bilingual baselines, and that in higher-resource scenarios, clustering language pairs that are related improves translation quality.

{{</citation>}}


### (4/197) Text Sanitization Beyond Specific Domains: Zero-Shot Redaction & Substitution with Large Language Models (Federico Albanese et al., 2023)

{{<citation>}}

Federico Albanese, Daniel Ciolek, Nicolas D'Ippolito. (2023)  
**Text Sanitization Beyond Specific Domains: Zero-Shot Redaction & Substitution with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.10785v1)  

---


**ABSTRACT**  
In the context of information systems, text sanitization techniques are used to identify and remove sensitive data to comply with security and regulatory requirements. Even though many methods for privacy preservation have been proposed, most of them are focused on the detection of entities from specific domains (e.g., credit card numbers, social security numbers), lacking generality and requiring customization for each desirable domain. Moreover, removing words is, in general, a drastic measure, as it can degrade text coherence and contextual information. Less severe measures include substituting a word for a safe alternative, yet it can be challenging to automatically find meaningful substitutions. We present a zero-shot text sanitization technique that detects and substitutes potentially sensitive information using Large Language Models. Our evaluation shows that our method excels at protecting privacy while maintaining text coherence and contextual information, preserving data utility for downstream tasks.

{{</citation>}}


### (5/197) Characterizing Tradeoffs in Language Model Decoding with Informational Interpretations (Chung-Ching Chang et al., 2023)

{{<citation>}}

Chung-Ching Chang, William W. Cohen, Yun-Hsuan Sung. (2023)  
**Characterizing Tradeoffs in Language Model Decoding with Informational Interpretations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.10083v1)  

---


**ABSTRACT**  
We propose a theoretical framework for formulating language model decoder algorithms with dynamic programming and information theory. With dynamic programming, we lift the design of decoder algorithms from the logit space to the action-state value function space, and show that the decoding algorithms are consequences of optimizing the action-state value functions. Each component in the action-state value function space has an information theoretical interpretation. With the lifting and interpretation, it becomes evident what the decoder algorithm is optimized for, and hence facilitating the arbitration of the tradeoffs in sensibleness, diversity, and attribution.

{{</citation>}}


### (6/197) ChatGPT-3.5, ChatGPT-4, Google Bard, and Microsoft Bing to Improve Health Literacy and Communication in Pediatric Populations and Beyond (Kanhai S. Amin et al., 2023)

{{<citation>}}

Kanhai S. Amin, Linda Mayes, Pavan Khosla, Rushabh Doshi. (2023)  
**ChatGPT-3.5, ChatGPT-4, Google Bard, and Microsoft Bing to Improve Health Literacy and Communication in Pediatric Populations and Beyond**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Google, Microsoft  
[Paper Link](http://arxiv.org/abs/2311.10075v1)  

---


**ABSTRACT**  
Purpose: Enhanced health literacy has been linked to better health outcomes; however, few interventions have been studied. We investigate whether large language models (LLMs) can serve as a medium to improve health literacy in children and other populations.   Methods: We ran 288 conditions using 26 different prompts through ChatGPT-3.5, Microsoft Bing, and Google Bard. Given constraints imposed by rate limits, we tested a subset of 150 conditions through ChatGPT-4. The primary outcome measurements were the reading grade level (RGL) and word counts of output.   Results: Across all models, output for basic prompts such as "Explain" and "What is (are)" were at, or exceeded, a 10th-grade RGL. When prompts were specified to explain conditions from the 1st to 12th RGL, we found that LLMs had varying abilities to tailor responses based on RGL. ChatGPT-3.5 provided responses that ranged from the 7th-grade to college freshmen RGL while ChatGPT-4 outputted responses from the 6th-grade to the college-senior RGL. Microsoft Bing provided responses from the 9th to 11th RGL while Google Bard provided responses from the 7th to 10th RGL.   Discussion: ChatGPT-3.5 and ChatGPT-4 did better in achieving lower-grade level outputs. Meanwhile Bard and Bing tended to consistently produce an RGL that is at the high school level regardless of prompt. Additionally, Bard's hesitancy in providing certain outputs indicates a cautious approach towards health information. LLMs demonstrate promise in enhancing health communication, but future research should verify the accuracy and effectiveness of such tools in this context.   Implications: LLMs face challenges in crafting outputs below a sixth-grade reading level. However, their capability to modify outputs above this threshold provides a potential mechanism to improve health literacy and communication in a pediatric population and beyond.

{{</citation>}}


### (7/197) Is 'A Helpful Assistant' the Best Role for Large Language Models? A Systematic Evaluation of Social Roles in System Prompts (Mingqian Zheng et al., 2023)

{{<citation>}}

Mingqian Zheng, Jiaxin Pei, David Jurgens. (2023)  
**Is 'A Helpful Assistant' the Best Role for Large Language Models? A Systematic Evaluation of Social Roles in System Prompts**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-HC, cs-LG, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.10054v1)  

---


**ABSTRACT**  
Prompting serves as the major way humans interact with Large Language Models (LLM). Commercial AI systems commonly define the role of the LLM in system prompts. For example, ChatGPT uses "You are a helpful assistant" as part of the default system prompt. But is "a helpful assistant" the best role for LLMs? In this study, we present a systematic evaluation of how social roles in system prompts affect model performance. We curate a list of 162 roles covering 6 types of interpersonal relationships and 8 types of occupations. Through extensive analysis of 3 popular LLMs and 2457 questions, we show that adding interpersonal roles in prompts consistently improves the models' performance over a range of questions. Moreover, while we find that using gender-neutral roles and specifying the role as the audience leads to better performances, predicting which role leads to the best performance remains a challenging task, and that frequency, similarity, and perplexity do not fully explain the effect of social roles on model performances. Our results can help inform the design of system prompts for AI systems. Code and data are available at https://github.com/Jiaxin-Pei/Prompting-with-Social-Roles.

{{</citation>}}


### (8/197) Generative AI for Hate Speech Detection: Evaluation and Findings (Sagi Pendzel et al., 2023)

{{<citation>}}

Sagi Pendzel, Tomer Wullach, Amir Adler, Einat Minkov. (2023)  
**Generative AI for Hate Speech Detection: Evaluation and Findings**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, BERT, GPT, GPT-3.5, Generative AI, Hate Speech Detection  
[Paper Link](http://arxiv.org/abs/2311.09993v1)  

---


**ABSTRACT**  
Automatic hate speech detection using deep neural models is hampered by the scarcity of labeled datasets, leading to poor generalization. To mitigate this problem, generative AI has been utilized to generate large amounts of synthetic hate speech sequences from available labeled examples, leveraging the generated data in finetuning large pre-trained language models (LLMs). In this chapter, we provide a review of relevant methods, experimental setups and evaluation of this approach. In addition to general LLMs, such as BERT, RoBERTa and ALBERT, we apply and evaluate the impact of train set augmentation with generated data using LLMs that have been already adapted for hate detection, including RoBERTa-Toxicity, HateBERT, HateXplain, ToxDect, and ToxiGen. An empirical study corroborates our previous findings, showing that this approach improves hate speech generalization, boosting recall performance across data distributions. In addition, we explore and compare the performance of the finetuned LLMs with zero-shot hate detection using a GPT-3.5 model. Our results demonstrate that while better generalization is achieved using the GPT-3.5 model, it achieves mediocre recall and low precision on most datasets. It is an open question whether the sensitivity of models such as GPT-3.5, and onward, can be improved using similar techniques of text generation.

{{</citation>}}


### (9/197) ExFake: Towards an Explainable Fake News Detection Based on Content and Social Context Information (Sabrine Amri et al., 2023)

{{<citation>}}

Sabrine Amri, Henri-Cedric Mputu Boleilanga, Esma Aïmeur. (2023)  
**ExFake: Towards an Explainable Fake News Detection Based on Content and Social Context Information**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-SI, cs.CL  
Keywords: AI, Fake News  
[Paper Link](http://arxiv.org/abs/2311.10784v1)  

---


**ABSTRACT**  
ExFake is an explainable fake news detection system based on content and context-level information. It is concerned with the veracity analysis of online posts based on their content, social context (i.e., online users' credibility and historical behaviour), and data coming from trusted entities such as fact-checking websites and named entities. Unlike state-of-the-art systems, an Explainable AI (XAI) assistant is also adopted to help online social networks (OSN) users develop good reflexes when faced with any doubted information that spreads on social networks. The trustworthiness of OSN users is also addressed by assigning a credibility score to OSN users, as OSN users are one of the main culprits for spreading fake news. Experimental analysis on a real-world dataset demonstrates that ExFake significantly outperforms other baseline methods for fake news detection.

{{</citation>}}


### (10/197) A BERT based Ensemble Approach for Sentiment Classification of Customer Reviews and its Application to Nudge Marketing in e-Commerce (Sayan Putatunda et al., 2023)

{{<citation>}}

Sayan Putatunda, Anwesha Bhowmik, Girish Thiruvenkadam, Rahul Ghosh. (2023)  
**A BERT based Ensemble Approach for Sentiment Classification of Customer Reviews and its Application to Nudge Marketing in e-Commerce**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2311.10782v1)  

---


**ABSTRACT**  
According to the literature, Product reviews are an important source of information for customers to support their buying decision. Product reviews improve customer trust and loyalty. Reviews help customers in understanding what other customers think about a particular product and helps in driving purchase decisions. Therefore, for an e-commerce platform it is important to understand the sentiments in customer reviews to understand their products and services, and it also allows them to potentially create positive consumer interaction as well as long lasting relationships. Reviews also provide innovative ways to market the products for an ecommerce company. One such approach is Nudge Marketing. Nudge marketing is a subtle way for an ecommerce company to help their customers make better decisions without hesitation.

{{</citation>}}


### (11/197) Which Modality should I use -- Text, Motif, or Image? : Understanding Graphs with Large Language Models (Debarati Das et al., 2023)

{{<citation>}}

Debarati Das, Ishaan Gupta, Jaideep Srivastava, Dongyeop Kang. (2023)  
**Which Modality should I use -- Text, Motif, or Image? : Understanding Graphs with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09862v1)  

---


**ABSTRACT**  
Large language models (LLMs) are revolutionizing various fields by leveraging large text corpora for context-aware intelligence. Due to the context size, however, encoding an entire graph with LLMs is fundamentally limited. This paper explores how to better integrate graph data with LLMs and presents a novel approach using various encoding modalities (e.g., text, image, and motif) and approximation of global connectivity of a graph using different prompting methods to enhance LLMs' effectiveness in handling complex graph structures. The study also introduces GraphTMI, a new benchmark for evaluating LLMs in graph structure analysis, focusing on factors such as homophily, motif presence, and graph difficulty. Key findings reveal that image modality, supported by advanced vision-language models like GPT-4V, is more effective than text in managing token limits while retaining critical information. The research also examines the influence of different factors on each encoding modality's performance. This study highlights the current limitations and charts future directions for LLMs in graph understanding and reasoning tasks.

{{</citation>}}


### (12/197) PsyBench: a balanced and in-depth Psychological Chinese Evaluation Benchmark for Foundation Models (Junlei Zhang et al., 2023)

{{<citation>}}

Junlei Zhang, Hongliang He, Nirui Song, Shuyuan He, Shuai Zhang, Huachuan Qiu, Anqi Li, Lizhi Ma, Zhenzhong Lan. (2023)  
**PsyBench: a balanced and in-depth Psychological Chinese Evaluation Benchmark for Foundation Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.09861v2)  

---


**ABSTRACT**  
As Large Language Models (LLMs) are becoming prevalent in various fields, there is an urgent need for improved NLP benchmarks that encompass all the necessary knowledge of individual discipline. Many contemporary benchmarks for foundational models emphasize a broad range of subjects but often fall short in presenting all the critical subjects and encompassing necessary professional knowledge of them. This shortfall has led to skewed results, given that LLMs exhibit varying performance across different subjects and knowledge areas. To address this issue, we present psybench, the first comprehensive Chinese evaluation suite that covers all the necessary knowledge required for graduate entrance exams. psybench offers a deep evaluation of a model's strengths and weaknesses in psychology through multiple-choice questions. Our findings show significant differences in performance across different sections of a subject, highlighting the risk of skewed results when the knowledge in test sets is not balanced. Notably, only the ChatGPT model reaches an average accuracy above $70\%$, indicating that there is still plenty of room for improvement. We expect that psybench will help to conduct thorough evaluations of base models' strengths and weaknesses and assist in practical application in the field of psychology.

{{</citation>}}


### (13/197) GSAP-NER: A Novel Task, Corpus, and Baseline for Scholarly Entity Extraction Focused on Machine Learning Models and Datasets (Wolfgang Otto et al., 2023)

{{<citation>}}

Wolfgang Otto, Matthäus Zloch, Lu Gan, Saurav Karmakar, Stefan Dietze. (2023)  
**GSAP-NER: A Novel Task, Corpus, and Baseline for Scholarly Entity Extraction Focused on Machine Learning Models and Datasets**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NER, NLP, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2311.09860v1)  

---


**ABSTRACT**  
Named Entity Recognition (NER) models play a crucial role in various NLP tasks, including information extraction (IE) and text understanding. In academic writing, references to machine learning models and datasets are fundamental components of various computer science publications and necessitate accurate models for identification. Despite the advancements in NER, existing ground truth datasets do not treat fine-grained types like ML model and model architecture as separate entity types, and consequently, baseline models cannot recognize them as such. In this paper, we release a corpus of 100 manually annotated full-text scientific publications and a first baseline model for 10 entity types centered around ML models and datasets. In order to provide a nuanced understanding of how ML models and datasets are mentioned and utilized, our dataset also contains annotations for informal mentions like "our BERT-based model" or "an image CNN". You can find the ground truth dataset and code to replicate model training at https://data.gesis.org/gsap/gsap-ner.

{{</citation>}}


### (14/197) Leveraging LLMs in Scholarly Knowledge Graph Question Answering (Tilahun Abedissa Taffa et al., 2023)

{{<citation>}}

Tilahun Abedissa Taffa, Ricardo Usbeck. (2023)  
**Leveraging LLMs in Scholarly Knowledge Graph Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DB, cs-LG, cs.CL  
Keywords: BERT, Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.09841v1)  

---


**ABSTRACT**  
This paper presents a scholarly Knowledge Graph Question Answering (KGQA) that answers bibliographic natural language questions by leveraging a large language model (LLM) in a few-shot manner. The model initially identifies the top-n similar training questions related to a given test question via a BERT-based sentence encoder and retrieves their corresponding SPARQL. Using the top-n similar question-SPARQL pairs as an example and the test question creates a prompt. Then pass the prompt to the LLM and generate a SPARQL. Finally, runs the SPARQL against the underlying KG - ORKG (Open Research KG) endpoint and returns an answer. Our system achieves an F1 score of 99.0%, on SciQA - one of the Scholarly-QALD-23 challenge benchmarks.

{{</citation>}}


### (15/197) PELMS: Pre-training for Effective Low-Shot Multi-Document Summarization (Joseph J. Peper et al., 2023)

{{<citation>}}

Joseph J. Peper, Wenzhao Qiu, Lu Wang. (2023)  
**PELMS: Pre-training for Effective Low-Shot Multi-Document Summarization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.09836v1)  

---


**ABSTRACT**  
We investigate pre-training techniques for abstractive multi-document summarization (MDS), which is much less studied than summarizing single documents. Though recent work has demonstrated the effectiveness of highlighting information salience for pre-training strategy design, it struggles to generate abstractive and reflective summaries, which are critical properties for MDS. To this end, we present PELMS, a pre-trained model that uses objectives based on semantic coherence heuristics and faithfulness constraints with un-labeled multi-document inputs, to promote the generation of concise, fluent, and faithful summaries. To support the training of PELMS, we compile MultiPT, a multi-document pre-training corpus containing over 93 million documents to form more than 3 million unlabeled topic-centric document clusters, covering diverse genres such as product reviews, news, and general knowledge. We perform extensive evaluation of PELMS in low-shot settings on a wide range of MDS datasets. Our approach consistently outperforms competitive comparisons with respect to overall informativeness, abstractiveness, coherence, and faithfulness.

{{</citation>}}


### (16/197) ML-Bench: Large Language Models Leverage Open-source Libraries for Machine Learning Tasks (Yuliang Liu et al., 2023)

{{<citation>}}

Yuliang Liu, Xiangru Tang, Zefan Cai, Junjie Lu, Yichi Zhang, Yanjun Shao, Zexuan Deng, Helan Hu, Zengxian Yang, Kaikai An, Ruijun Huang, Shuzheng Si, Sheng Chen, Haozhe Zhao, Zhengliang Li, Liang Chen, Yiming Zong, Yan Wang, Tianyu Liu, Zhiwei Jiang, Baobao Chang, Yujia Qin, Wangchunshu Zhou, Yilun Zhao, Arman Cohan, Mark Gerstein. (2023)  
**ML-Bench: Large Language Models Leverage Open-source Libraries for Machine Learning Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09835v1)  

---


**ABSTRACT**  
Large language models have shown promising performance in code generation benchmarks. However, a considerable divide exists between these benchmark achievements and their practical applicability, primarily attributed to real-world programming's reliance on pre-existing libraries. Instead of evaluating LLMs to code from scratch, this work aims to propose a new evaluation setup where LLMs use open-source libraries to finish machine learning tasks. Therefore, we propose ML-Bench, an expansive benchmark developed to assess the effectiveness of LLMs in leveraging existing functions in open-source libraries. Consisting of 10044 samples spanning 130 tasks over 14 notable machine learning GitHub repositories. In this setting, given a specific machine learning task instruction and the accompanying README in a codebase, an LLM is tasked to generate code to accomplish the task. This necessitates the comprehension of long and language-code interleaved documents, as well as the understanding of complex cross-file code structures, introducing new challenges. Notably, while GPT-4 exhibits remarkable improvement over other LLMs, it manages to accomplish only 39.73\% of the tasks, leaving a huge space for improvement. We address these challenges by proposing ML-Agent, designed to effectively navigate the codebase, locate documentation, retrieve code, and generate executable code. Empirical results demonstrate that ML-Agent, built upon GPT-4, results in further improvements. Code, data, and models are available at \url{https://ml-bench.github.io/}.

{{</citation>}}


### (17/197) FollowEval: A Multi-Dimensional Benchmark for Assessing the Instruction-Following Capability of Large Language Models (Yimin Jing et al., 2023)

{{<citation>}}

Yimin Jing, Renren Jin, Jiahao Hu, Huishi Qiu, Xiaohua Wang, Peng Wang, Deyi Xiong. (2023)  
**FollowEval: A Multi-Dimensional Benchmark for Assessing the Instruction-Following Capability of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09829v1)  

---


**ABSTRACT**  
The effective assessment of the instruction-following ability of large language models (LLMs) is of paramount importance. A model that cannot adhere to human instructions might be not able to provide reliable and helpful responses. In pursuit of this goal, various benchmarks have been constructed to evaluate the instruction-following capacity of these models. However, these benchmarks are limited to a single language and are constructed using automated approaches, which restricts their applicability and the quality of the test examples they contain. To bridge this gap, we introduce the FollowEval benchmark in this paper. This benchmark is composed of instances in both English and Chinese, and all test examples are crafted by human experts. Furthermore, the FollowEval benchmark is designed to assess LLMs across five critical dimensions of instruction following: string manipulation, commonsense reasoning, logical reasoning, spatial reasoning, and response constraints. To enhance the complexity and present a sufficient challenge, each test example is designed to evaluate more than one dimension. We have evaluated various LLMs using the FollowEval benchmark and found that their performance significantly lags behind that of humans. This highlights the considerable room for improvement in the instruction-following ability of these models.

{{</citation>}}


### (18/197) AfriMTE and AfriCOMET: Empowering COMET to Embrace Under-resourced African Languages (Jiayi Wang et al., 2023)

{{<citation>}}

Jiayi Wang, David Ifeoluwa Adelani, Sweta Agrawal, Ricardo Rei, Eleftheria Briakou, Marine Carpuat, Marek Masiak, Xuanli He, Sofia Bourhim, Andiswa Bukula, Muhidin Mohamed, Temitayo Olatoye, Hamam Mokayede, Christine Mwase, Wangui Kimotho, Foutse Yuehgoh, Anuoluwapo Aremu, Jessica Ojo, Shamsuddeen Hassan Muhammad, Salomey Osei, Abdul-Hakeem Omotayo, Chiamaka Chukwuneke, Perez Ogayo, Oumaima Hourrane, Salma El Anigri, Lolwethu Ndolela, Thabiso Mangwana, Shafie Abdi Mohamed, Ayinde Hassan, Oluwabusayo Olufunke Awoyomi, Lama Alkhaled, Sana Al-Azzawi, Naome A. Etori, Millicent Ochieng, Clemencia Siro, Samuel Njoroge, Eric Muchiri, Wangari Kimotho, Lyse Naomi Wamba Momo, Daud Abolade, Simbiat Ajao, Tosin Adewumi, Iyanuoluwa Shode, Ricky Macharm, Ruqayya Nasir Iro, Saheed S. Abdullahi, Stephen E. Moore, Bernard Opoku, Zainab Akinjobi, Abeeb Afolabi, Nnaemeka Obiefuna, Onyekachi Raphael Ogbu, Sam Brian, Verrah Akinyi Otiende, Chinedu Emmanuel Mbonu, Sakayo Toadoum Sari, Pontus Stenetorp. (2023)  
**AfriMTE and AfriCOMET: Empowering COMET to Embrace Under-resourced African Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Embedding  
[Paper Link](http://arxiv.org/abs/2311.09828v1)  

---


**ABSTRACT**  
Despite the progress we have recorded in scaling multilingual machine translation (MT) models and evaluation data to several under-resourced African languages, it is difficult to measure accurately the progress we have made on these languages because evaluation is often performed on n-gram matching metrics like BLEU that often have worse correlation with human judgments. Embedding-based metrics such as COMET correlate better; however, lack of evaluation data with human ratings for under-resourced languages, complexity of annotation guidelines like Multidimensional Quality Metrics (MQM), and limited language coverage of multilingual encoders have hampered their applicability to African languages. In this paper, we address these challenges by creating high-quality human evaluation data with a simplified MQM guideline for error-span annotation and direct assessment (DA) scoring for 13 typologically diverse African languages. Furthermore, we develop AfriCOMET, a COMET evaluation metric for African languages by leveraging DA training data from high-resource languages and African-centric multilingual encoder (AfroXLM-Roberta) to create the state-of-the-art evaluation metric for African languages MT with respect to Spearman-rank correlation with human judgments (+0.406).

{{</citation>}}


### (19/197) Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking (Nan Xu et al., 2023)

{{<citation>}}

Nan Xu, Fei Wang, Ben Zhou, Bang Zheng Li, Chaowei Xiao, Muhao Chen. (2023)  
**Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09827v1)  

---


**ABSTRACT**  
While large language models (LLMs) have demonstrated increasing power, they have also given rise to a wide range of harmful behaviors. As representatives, jailbreak attacks can provoke harmful or unethical responses from LLMs, even after safety alignment. In this paper, we investigate a novel category of jailbreak attacks specifically designed to target the cognitive structure and processes of LLMs. Specifically, we analyze the safety vulnerability of LLMs in the face of (1) multilingual cognitive overload, (2) veiled expression, and (3) effect-to-cause reasoning. Different from previous jailbreak attacks, our proposed cognitive overload is a black-box attack with no need for knowledge of model architecture or access to model weights. Experiments conducted on AdvBench and MasterKey reveal that various LLMs, including both popular open-source model Llama 2 and the proprietary model ChatGPT, can be compromised through cognitive overload. Motivated by cognitive psychology work on managing cognitive load, we further investigate defending cognitive overload attack from two perspectives. Empirical studies show that our cognitive overload from three perspectives can jailbreak all studied LLMs successfully, while existing defense strategies can hardly mitigate the caused malicious uses effectively.

{{</citation>}}


### (20/197) Human Still Wins over LLM: An Empirical Study of Active Learning on Domain-Specific Annotation Tasks (Yuxuan Lu et al., 2023)

{{<citation>}}

Yuxuan Lu, Bingsheng Yao, Shao Zhang, Yun Wang, Peng Zhang, Tun Lu, Toby Jia-Jun Li, Dakuo Wang. (2023)  
**Human Still Wins over LLM: An Empirical Study of Active Learning on Domain-Specific Annotation Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Active Learning, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09825v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated considerable advances, and several claims have been made about their exceeding human performance. However, in real-world tasks, domain knowledge is often required. Low-resource learning methods like Active Learning (AL) have been proposed to tackle the cost of domain expert annotation, raising this question: Can LLMs surpass compact models trained with expert annotations in domain-specific tasks? In this work, we conduct an empirical experiment on four datasets from three different domains comparing SOTA LLMs with small models trained on expert annotations with AL. We found that small models can outperform GPT-3.5 with a few hundreds of labeled data, and they achieve higher or similar performance with GPT-4 despite that they are hundreds time smaller. Based on these findings, we posit that LLM predictions can be used as a warmup method in real-world applications and human experts remain indispensable in tasks involving data annotation driven by domain-specific knowledge.

{{</citation>}}


### (21/197) Towards Robust Temporal Reasoning of Large Language Models via a Multi-Hop QA Dataset and Pseudo-Instruction Tuning (Qingyu Tan et al., 2023)

{{<citation>}}

Qingyu Tan, Hwee Tou Ng, Lidong Bing. (2023)  
**Towards Robust Temporal Reasoning of Large Language Models via a Multi-Hop QA Dataset and Pseudo-Instruction Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09821v1)  

---


**ABSTRACT**  
Knowledge in the real world is being updated constantly. However, it is costly to frequently update large language models (LLMs). Therefore, it is crucial for LLMs to understand the concept of temporal knowledge. However, prior works on temporal question answering did not emphasize multi-answer and multi-hop types of temporal reasoning. In this paper, we propose a complex temporal question-answering (QA) dataset Complex-TR that focuses on multi-answer and multi-hop temporal reasoning. Besides, we also propose a novel data augmentation strategy to improve the complex temporal reasoning capability and robustness of LLMs. We conducted experiments on multiple temporal QA datasets. Experimental results show that our method is able to improve LLMs' performance on temporal QA benchmarks by significant margins.

{{</citation>}}


### (22/197) SUQL: Conversational Search over Structured and Unstructured Data with Large Language Models (Shicheng Liu et al., 2023)

{{<citation>}}

Shicheng Liu, Jialiang Xu, Wesley Tjangnaka, Sina J. Semnani, Chen Jie Yu, Gui Dávid, Monica S. Lam. (2023)  
**SUQL: Conversational Search over Structured and Unstructured Data with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-PL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09818v1)  

---


**ABSTRACT**  
Many knowledge sources consist of both structured information such as relational databases as well as unstructured free text. Building a conversational interface to such data sources is challenging.   This paper introduces SUQL, Structured and Unstructured Query Language, the first formal executable representation that naturally covers compositions of structured and unstructured data queries. Specifically, it augments SQL with several free-text primitives to form a precise, succinct, and expressive representation. This paper also presents a conversational search agent based on large language models, including a few-shot contextual semantic parser for SUQL.   To validate our approach, we introduce a dataset consisting of crowdsourced questions and conversations about real restaurants. Over 51% of the questions in the dataset require both structured and unstructured data, suggesting that it is a common phenomenon. We show that our few-shot conversational agent based on SUQL finds an entity satisfying all user requirements 89.3% of the time, compared to just 65.0% for a strong and commonly used baseline.

{{</citation>}}


### (23/197) MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning (Xiangru Tang et al., 2023)

{{<citation>}}

Xiangru Tang, Anni Zou, Zhuosheng Zhang, Yilun Zhao, Xingyao Zhang, Arman Cohan, Mark Gerstein. (2023)  
**MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.10537v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), despite their remarkable progress across various general domains, encounter significant barriers in medicine and healthcare. This field faces unique challenges such as domain-specific terminologies and the reasoning over specialized knowledge. To address these obstinate issues, we propose a novel Multi-disciplinary Collaboration (MC) framework for the medical domain that leverages role-playing LLM-based agents who participate in a collaborative multi-round discussion, thereby enhancing LLM proficiency and reasoning capabilities. This training-free and interpretable framework encompasses five critical steps: gathering domain experts, proposing individual analyses, summarising these analyses into a report, iterating over discussions until a consensus is reached, and ultimately making a decision. Our work particularly focuses on the zero-shot scenario, our results on nine data sets (MedQA, MedMCQA, PubMedQA, and six subtasks from MMLU) establish that our proposed MC framework excels at mining and harnessing the medical expertise in LLMs, as well as extending its reasoning abilities. Based on these outcomes, we further conduct a human evaluation to pinpoint and categorize common errors within our method, as well as ablation studies aimed at understanding the impact of various factors on overall performance. Our code can be found at \url{https://github.com/gersteinlab/MedAgents}.

{{</citation>}}


### (24/197) Performance Trade-offs of Watermarking Large Language Models (Anirudh Ajith et al., 2023)

{{<citation>}}

Anirudh Ajith, Sameer Singh, Danish Pruthi. (2023)  
**Performance Trade-offs of Watermarking Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09816v1)  

---


**ABSTRACT**  
Amidst growing concerns of large language models (LLMs) being misused for generating misinformation or completing homework assignments, watermarking has emerged as an effective solution for distinguishing human-written and LLM-generated text. A prominent watermarking strategy is to embed a signal into generated text by upsampling a (pseudorandomly-chosen) subset of tokens at every generation step. Although this signal is imperceptible to a human reader, it is detectable through statistical testing. However, implanting such signals alters the model's output distribution and can have unintended effects when watermarked LLMs are used for downstream applications. In this work, we evaluate the performance of watermarked LLMs on a diverse suite of tasks, including text classification, textual entailment, reasoning, question answering, translation, summarization, and language modeling. We find that watermarking has negligible impact on the performance of tasks posed as k-class classification problems in the average case. However, the accuracy can plummet to that of a random classifier for some scenarios (that occur with non-negligible probability). Tasks that are cast as multiple-choice questions and short-form generation are surprisingly unaffected by watermarking. For long-form generation tasks, including summarization and translation, we see a drop of 15-20% in the performance due to watermarking. Our findings highlight the trade-offs that users should be cognizant of when using watermarked models, and point to cases where future research could improve existing trade-offs.

{{</citation>}}


### (25/197) Large Language Models for Propaganda Span Annotation (Maram Hasanain et al., 2023)

{{<citation>}}

Maram Hasanain, Fatema Ahmed, Firoj Alam. (2023)  
**Large Language Models for Propaganda Span Annotation**  

---
Primary Category: cs.CL  
Categories: 68T50, F-2-2; I-2-7, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09812v1)  

---


**ABSTRACT**  
The use of propagandistic techniques in online communication has increased in recent years, aiming to manipulate online audiences. Efforts to automatically detect and debunk such content have been made, addressing various modeling scenarios. These include determining whether the content (text, image, or multimodal) (i) is propagandistic, (ii) employs one or more techniques, and (iii) includes techniques with identifiable spans. Significant research efforts have been devoted to the first two scenarios compared to the latter. Therefore, in this study, we focus on the task of detecting propagandistic textual spans. We investigate whether large language models such as GPT-4 can be utilized to perform the task of an annotator. For the experiments, we used an in-house developed dataset consisting of annotations from multiple annotators. Our results suggest that providing more information to the model as prompts improves the annotation agreement and performance compared to human annotations. We plan to make the annotated labels from multiple annotators, including GPT-4, available for the community.

{{</citation>}}


### (26/197) The Curious Decline of Linguistic Diversity: Training Language Models on Synthetic Text (Yanzhu Guo et al., 2023)

{{<citation>}}

Yanzhu Guo, Guokan Shang, Michalis Vazirgiannis, Chloé Clavel. (2023)  
**The Curious Decline of Linguistic Diversity: Training Language Models on Synthetic Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09807v1)  

---


**ABSTRACT**  
This study investigates the consequences of training large language models (LLMs) on synthetic data generated by their predecessors, an increasingly prevalent practice aimed at addressing the limited supply of human-generated training data. Diverging from the usual emphasis on performance metrics, we focus on the impact of this training methodology on linguistic diversity, especially when conducted recursively over time. To assess this, we developed a set of novel metrics targeting lexical, syntactic, and semantic diversity, applying them in recursive fine-tuning experiments across various natural language generation tasks. Our findings reveal a marked decrease in the diversity of the models' outputs through successive iterations. This trend underscores the potential risks of training LLMs on predecessor-generated text, particularly concerning the preservation of linguistic richness. Our study highlights the need for careful consideration of the long-term effects of such training approaches on the linguistic capabilities of LLMs.

{{</citation>}}


### (27/197) DocMath-Eval: Evaluating Numerical Reasoning Capabilities of LLMs in Understanding Long Documents with Tabular Data (Yilun Zhao et al., 2023)

{{<citation>}}

Yilun Zhao, Yitao Long, Hongjun Liu, Linyong Nan, Lyuhao Chen, Ryo Kamoi, Yixin Liu, Xiangru Tang, Rui Zhang, Arman Cohan. (2023)  
**DocMath-Eval: Evaluating Numerical Reasoning Capabilities of LLMs in Understanding Long Documents with Tabular Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09805v1)  

---


**ABSTRACT**  
Recent LLMs have demonstrated remarkable performance in solving exam-like math word problems. However, the degree to which these numerical reasoning skills are effective in real-world scenarios, particularly in expert domains, is still largely unexplored. This paper introduces DocMath-Eval, a comprehensive benchmark specifically designed to evaluate the numerical reasoning and problem-solving capabilities of LLMs in the context of understanding and analyzing financial documents containing both text and tables. We evaluate a wide spectrum of 19 LLMs, including those specialized in coding and finance. We also incorporate different prompting strategies (i.e., Chain-of-Thoughts and Program-of-Thoughts) to comprehensively assess the capabilities and limitations of existing LLMs in DocMath-Eval. We found that, although the current best-performing system (i.e., GPT-4), can perform well on simple problems such as calculating the rate of increase in a financial metric within a short document context, it significantly lags behind human experts in more complex problems grounded in longer contexts. We believe DocMath-Eval can be used as a valuable benchmark to evaluate LLMs' capabilities to solve challenging numerical reasoning problems in expert domains. We will release the benchmark and code at https://github.com/yale-nlp/DocMath-Eval.

{{</citation>}}


### (28/197) $\textit{Dial BeInfo for Faithfulness}$: Improving Factuality of Information-Seeking Dialogue via Behavioural Fine-Tuning (Evgeniia Razumovskaia et al., 2023)

{{<citation>}}

Evgeniia Razumovskaia, Ivan Vulić, Pavle Marković, Tomasz Cichy, Qian Zheng, Tsung-Hsien Wen, Paweł Budzianowski. (2023)  
**$\textit{Dial BeInfo for Faithfulness}$: Improving Factuality of Information-Seeking Dialogue via Behavioural Fine-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT, T5  
[Paper Link](http://arxiv.org/abs/2311.09800v1)  

---


**ABSTRACT**  
Factuality is a crucial requirement in information seeking dialogue: the system should respond to the user's queries so that the responses are meaningful and aligned with the knowledge provided to the system. However, most modern large language models suffer from hallucinations, that is, they generate responses not supported by or contradicting the knowledge source. To mitigate the issue and increase faithfulness of information-seeking dialogue systems, we introduce BeInfo, a simple yet effective method that applies behavioural tuning to aid information-seeking dialogue. Relying on three standard datasets, we show that models tuned with BeInfo} become considerably more faithful to the knowledge source both for datasets and domains seen during BeInfo-tuning, as well as on unseen domains, when applied in a zero-shot manner. In addition, we show that the models with 3B parameters (e.g., Flan-T5) tuned with BeInfo demonstrate strong performance on data from real `production' conversations and outperform GPT4 when tuned on a limited amount of such realistic in-domain dialogues.

{{</citation>}}


### (29/197) How Far Can We Extract Diverse Perspectives from Large Language Models? Criteria-Based Diversity Prompting! (Shirley Anugrah Hayati et al., 2023)

{{<citation>}}

Shirley Anugrah Hayati, Minhwa Lee, Dheeraj Rajagopal, Dongyeop Kang. (2023)  
**How Far Can We Extract Diverse Perspectives from Large Language Models? Criteria-Based Diversity Prompting!**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.09799v1)  

---


**ABSTRACT**  
Collecting diverse human data on subjective NLP topics is costly and challenging. As Large Language Models (LLMs) have developed human-like capabilities, there is a recent trend in collaborative efforts between humans and LLMs for generating diverse data, offering potential scalable and efficient solutions. However, the extent of LLMs' capability to generate diverse perspectives on subjective topics remains an unexplored question. In this study, we investigate LLMs' capacity for generating diverse perspectives and rationales on subjective topics, such as social norms and argumentative texts. We formulate this problem as diversity extraction in LLMs and propose a criteria-based prompting technique to ground diverse opinions and measure perspective diversity from the generated criteria words. Our results show that measuring semantic diversity through sentence embeddings and distance metrics is not enough to measure perspective diversity. To see how far we can extract diverse perspectives from LLMs, or called diversity coverage, we employ a step-by-step recall prompting for generating more outputs from the model in an iterative manner. As we apply our prompting method to other tasks (hate speech labeling and story continuation), indeed we find that LLMs are able to generate diverse opinions according to the degree of task subjectivity.

{{</citation>}}


### (30/197) KnowledgeMath: Knowledge-Intensive Math Word Problem Solving in Finance Domains (Yilun Zhao et al., 2023)

{{<citation>}}

Yilun Zhao, Hongjun Liu, Yitao Long, Rui Zhang, Chen Zhao, Arman Cohan. (2023)  
**KnowledgeMath: Knowledge-Intensive Math Word Problem Solving in Finance Domains**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.09797v1)  

---


**ABSTRACT**  
We introduce KnowledgeMath, a novel benchmark designed to evaluate LLMs' capabilities in applying financial knowledge to solve complex math word problems. Compared to prior works, this study features three core advancements. First, KnowledgeMath includes 1,259 problems with a hybrid of textual and tabular content and require college-level knowledge in the finance domain for effective resolution. Second, we provide expert-annotated, detailed solution references in Python program format, ensuring a high-quality benchmark for LLM assessment. Finally, we evaluate a wide spectrum of 14 LLMs with different prompting strategies like Chain-of-Thoughts and Program-of-Thoughts. The current best-performing system (i.e., GPT-4 with Program-of-Thoughts) achieves only 45.4% accuracy, leaving substantial room for improvement. While knowledge-augmented LLMs can improve the performance (e.g., from 23.9% to 32.0% for GPT-3.5), it is still significantly lower the estimated human expert performance of 94%. We believe that KnowledgeMath can facilitate future research on domain-specific knowledge retrieval and augmentation into the math word problem-solving process. We will release the benchmark and code at https://github.com/yale-nlp/KnowledgeMath.

{{</citation>}}


### (31/197) Interpreting User Requests in the Context of Natural Language Standing Instructions (Nikita Moghe et al., 2023)

{{<citation>}}

Nikita Moghe, Patrick Xia, Jacob Andreas, Jason Eisner, Benjamin Van Durme, Harsh Jhamtani. (2023)  
**Interpreting User Requests in the Context of Natural Language Standing Instructions**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09796v1)  

---


**ABSTRACT**  
Users of natural language interfaces, generally powered by Large Language Models (LLMs),often must repeat their preferences each time they make a similar request. To alleviate this, we propose including some of a user's preferences and instructions in natural language -- collectively termed standing instructions -- as additional context for such interfaces. For example, when a user states I'm hungry, their previously expressed preference for Persian food will be automatically added to the LLM prompt, so as to influence the search for relevant restaurants. We develop NLSI, a language-to-program dataset consisting of over 2.4K dialogues spanning 17 domains, where each dialogue is paired with a user profile (a set of users specific standing instructions) and corresponding structured representations (API calls). A key challenge in NLSI is to identify which subset of the standing instructions is applicable to a given dialogue. NLSI contains diverse phenomena, from simple preferences to interdependent instructions such as triggering a hotel search whenever the user is booking tickets to an event. We conduct experiments on NLSI using prompting with large language models and various retrieval approaches, achieving a maximum of 44.7% exact match on API prediction. Our results demonstrate the challenges in identifying the relevant standing instructions and their interpretation into API calls.

{{</citation>}}


### (32/197) Can Language Model Moderators Improve the Health of Online Discourse? (Hyundong Cho et al., 2023)

{{<citation>}}

Hyundong Cho, Shuai Liu, Taiwei Shi, Darpan Jain, Basem Rizk, Yuyang Huang, Zixun Lu, Nuan Wen, Jonathan Gratch, Emilio Ferrara, Jonathan May. (2023)  
**Can Language Model Moderators Improve the Health of Online Discourse?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.10781v1)  

---


**ABSTRACT**  
Human moderation of online conversation is essential to maintaining civility and focus in a dialogue, but is challenging to scale and harmful to moderators. The inclusion of sophisticated natural language generation modules as a force multiplier aid moderators is a tantalizing prospect, but adequate evaluation approaches have so far been elusive. In this paper, we establish a systematic definition of conversational moderation effectiveness through a multidisciplinary lens that incorporates insights from social science. We then propose a comprehensive evaluation framework that uses this definition to asses models' moderation capabilities independently of human intervention. With our framework, we conduct the first known study of conversational dialogue models as moderators, finding that appropriately prompted models can provide specific and fair feedback on toxic behavior but struggle to influence users to increase their levels of respect and cooperation.

{{</citation>}}


### (33/197) Investigating Data Contamination in Modern Benchmarks for Large Language Models (Chunyuan Deng et al., 2023)

{{<citation>}}

Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, Arman Cohan. (2023)  
**Investigating Data Contamination in Modern Benchmarks for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.09783v1)  

---


**ABSTRACT**  
Recent observations have underscored a disparity between the inflated benchmark scores and the actual performance of LLMs, raising concerns about potential contamination of evaluation benchmarks. This issue is especially critical for closed-source models and certain open-source models where training data transparency is lacking. In this paper we study data contamination by proposing two methods tailored for both open-source and proprietary LLMs. We first introduce a retrieval-based system to explore potential overlaps between evaluation benchmarks and pretraining corpora. We further present a novel investigation protocol named \textbf{T}estset \textbf{S}lot Guessing (\textit{TS-Guessing}), applicable to both open and proprietary models. This approach entails masking a wrong answer in a multiple-choice question and prompting the model to fill in the gap. Additionally, it involves obscuring an unlikely word in an evaluation example and asking the model to produce it. We find that certain commercial LLMs could surprisingly guess the missing option in various test sets. Specifically, in the TruthfulQA benchmark, we find that LLMs exhibit notable performance improvement when provided with additional metadata in the benchmark. Further, in the MMLU benchmark, ChatGPT and GPT-4 demonstrated an exact match rate of 52\% and 57\%, respectively, in guessing the missing options in benchmark test data. We hope these results underscore the need for more robust evaluation methodologies and benchmarks in the field.

{{</citation>}}


### (34/197) More Samples or More Prompt Inputs? Exploring Effective In-Context Sampling for LLM Few-Shot Prompt Engineering (Bingsheng Yao et al., 2023)

{{<citation>}}

Bingsheng Yao, Guiming Chen, Ruishi Zou, Yuxuan Lu, Jiachen Li, Shao Zhang, Sijia Liu, James Hendler, Dakuo Wang. (2023)  
**More Samples or More Prompt Inputs? Exploring Effective In-Context Sampling for LLM Few-Shot Prompt Engineering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot, NLI, T5  
[Paper Link](http://arxiv.org/abs/2311.09782v1)  

---


**ABSTRACT**  
While most existing works on LLM prompt-engineering focus only on how to select a better set of data samples inside one single prompt input (In-Context Learning or ICL), why can't we design and leverage multiple prompt inputs together to further improve the LLM performance? In this work, we propose In-Context Sampling (ICS), a low-resource LLM prompt-engineering technique to produce the most confident prediction results by optimizing the construction of multiple ICL prompt inputs. Extensive experiments with two SOTA LLMs (FlanT5-XL and Mistral-7B) on three NLI datasets (e-SNLI, Multi-NLI, and ANLI) illustrate that ICS can consistently enhance LLM's prediction performance and confidence. An ablation study suggests that a diversity-based ICS strategy may further improve LLM's performance, which sheds light on a new yet promising future research direction.

{{</citation>}}


### (35/197) HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs (Junying Chen et al., 2023)

{{<citation>}}

Junying Chen, Xidong Wang, Anningzhe Gao, Feng Jiang, Shunian Chen, Hongbo Zhang, Dingjie Song, Wenya Xie, Chuyi Kong, Jianquan Li, Xiang Wan, Haizhou Li, Benyou Wang. (2023)  
**HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.09774v1)  

---


**ABSTRACT**  
Adapting a language model into a specific domain, a.k.a `domain adaption', is a common practice when specialized knowledge, e.g. medicine, is not encapsulated in a general language model like Llama2. The challenge lies in the heterogeneity of data across the two training stages, as it varies in languages, genres, or formats. To tackle this and simplify the learning protocol, we propose to transform heterogeneous data, from the both pre-training and supervised stages, into a unified, simple input-output pair format. We validate the new protocol in the domains where proprietary LLMs like ChatGPT perform relatively poorly, such as Traditional Chinese Medicine. The developed model, HuatuoGPT-II, has shown state-of-the-art performance in Chinese medicine domain on a number of benchmarks, e.g. medical licensing exams. It even outperforms proprietary models like ChatGPT and GPT-4 in some aspects, especially in Traditional Chinese Medicine. Expert manual evaluations further validate HuatuoGPT-II's advantages over existing LLMs. Notably, HuatuoGPT-II was benchmarked in a fresh Chinese National Medical Licensing Examination where it achieved the best performance, showcasing not only its effectiveness but also its generalization capabilities.

{{</citation>}}


### (36/197) LLMs as Narcissistic Evaluators: When Ego Inflates Evaluation Scores (Yiqi Liu et al., 2023)

{{<citation>}}

Yiqi Liu, Nafise Sadat Moosavi, Chenghua Lin. (2023)  
**LLMs as Narcissistic Evaluators: When Ego Inflates Evaluation Scores**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, NLP, T5  
[Paper Link](http://arxiv.org/abs/2311.09766v1)  

---


**ABSTRACT**  
Automatic evaluation of generated textual content presents an ongoing challenge within the field of NLP. Given the impressive capabilities of modern language models (LMs) across diverse NLP tasks, there is a growing trend to employ these models in creating innovative evaluation metrics for automated assessment of generation tasks. This paper investigates a pivotal question: Do language model-driven evaluation metrics inherently exhibit bias favoring texts generated by the same underlying language model? Specifically, we assess whether prominent LM-based evaluation metrics--namely, BARTScore, T5Score, and GPTScore--demonstrate a favorable bias toward their respective underlying LMs in the context of summarization tasks. Our findings unveil a latent bias, particularly pronounced when such evaluation metrics are used in an reference-free manner without leveraging gold summaries. These results underscore that assessments provided by generative evaluation models can be influenced by factors beyond the inherent text quality, highlighting the necessity of developing more dependable evaluation protocols in the future.

{{</citation>}}


### (37/197) Test-time Backdoor Mitigation for Black-Box Large Language Models with Defensive Demonstrations (Wenjie Mo et al., 2023)

{{<citation>}}

Wenjie Mo, Jiashu Xu, Qin Liu, Jiongxiao Wang, Jun Yan, Chaowei Xiao, Muhao Chen. (2023)  
**Test-time Backdoor Mitigation for Black-Box Large Language Models with Defensive Demonstrations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09763v1)  

---


**ABSTRACT**  
Existing studies in backdoor defense have predominantly focused on the training phase, overlooking the critical aspect of testing time defense. This gap becomes particularly pronounced in the context of Large Language Models (LLMs) deployed as Web Services, which typically offer only black-box access, rendering training-time defenses impractical. To bridge this gap, our work introduces defensive demonstrations, an innovative backdoor defense strategy for blackbox large language models. Our method involves identifying the task and retrieving task-relevant demonstrations from an uncontaminated pool. These demonstrations are then combined with user queries and presented to the model during testing, without requiring any modifications/tuning to the black-box model or insights into its internal mechanisms. Defensive demonstrations are designed to counteract the adverse effects of triggers, aiming to recalibrate and correct the behavior of poisoned models during test-time evaluations. Extensive experiments show that defensive demonstrations are effective in defending both instance-level and instruction-level backdoor attacks, not only rectifying the behavior of poisoned models but also surpassing existing baselines in most scenarios.

{{</citation>}}


### (38/197) Graph-Guided Reasoning for Multi-Hop Question Answering in Large Language Models (Jinyoung Park et al., 2023)

{{<citation>}}

Jinyoung Park, Ameen Patel, Omar Zia Khan, Hyunwoo J. Kim, Joo-Kyung Kim. (2023)  
**Graph-Guided Reasoning for Multi-Hop Question Answering in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09762v1)  

---


**ABSTRACT**  
Chain-of-Thought (CoT) prompting has boosted the multi-step reasoning capabilities of Large Language Models (LLMs) by generating a series of rationales before the final answer. We analyze the reasoning paths generated by CoT and find two issues in multi-step reasoning: (i) Generating rationales irrelevant to the question, (ii) Unable to compose subquestions or queries for generating/retrieving all the relevant information. To address them, we propose a graph-guided CoT prompting method, which guides the LLMs to reach the correct answer with graph representation/verification steps. Specifically, we first leverage LLMs to construct a "question/rationale graph" by using knowledge extraction prompting given the initial question and the rationales generated in the previous steps. Then, the graph verification step diagnoses the current rationale triplet by comparing it with the existing question/rationale graph to filter out irrelevant rationales and generate follow-up questions to obtain relevant information. Additionally, we generate CoT paths that exclude the extracted graph information to represent the context information missed from the graph extraction. Our graph-guided reasoning method shows superior performance compared to previous CoT prompting and the variants on multi-hop question answering benchmark datasets.

{{</citation>}}


### (39/197) MAFALDA: A Benchmark and Comprehensive Study of Fallacy Detection and Classification (Chadi Helwe et al., 2023)

{{<citation>}}

Chadi Helwe, Tom Calamai, Pierre-Henri Paris, Chloé Clavel, Fabian Suchanek. (2023)  
**MAFALDA: A Benchmark and Comprehensive Study of Fallacy Detection and Classification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.09761v1)  

---


**ABSTRACT**  
Fallacies can be used to spread disinformation, fake news, and propaganda, underlining the importance of their detection. Automated detection and classification of fallacies, however, remain challenging, mainly because of the innate subjectivity of the task and the need for a comprehensive, unified approach in existing research. Addressing these limitations, our study introduces a novel taxonomy of fallacies that aligns and refines previous classifications, a new annotation scheme tailored for subjective NLP tasks, and a new evaluation method designed to handle subjectivity, adapted to precision, recall, and F1-Score metrics. Using our annotation scheme, the paper introduces MAFALDA (Multi-level Annotated FALlacy DAtaset), a gold standard dataset. MAFALDA is based on examples from various previously existing fallacy datasets under our unified taxonomy across three levels of granularity. We then evaluate several language models under a zero-shot learning setting using MAFALDA to assess their fallacy detection and classification capability. Our comprehensive evaluation not only benchmarks the performance of these models but also provides valuable insights into their strengths and limitations in addressing fallacious reasoning.

{{</citation>}}


### (40/197) OrchestraLLM: Efficient Orchestration of Language Models for Dialogue State Tracking (Chia-Hsuan Lee et al., 2023)

{{<citation>}}

Chia-Hsuan Lee, Hao Cheng, Mari Ostendorf. (2023)  
**OrchestraLLM: Efficient Orchestration of Language Models for Dialogue State Tracking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.09758v1)  

---


**ABSTRACT**  
Large language models (LLMs) have revolutionized the landscape of Natural Language Processing systems, but are computationally expensive. To reduce the cost without sacrificing performance, previous studies have explored various approaches to harness the potential of Small Language Models (SLMs) as cost-effective alternatives to their larger counterparts. Driven by findings that SLMs and LLMs exhibit complementary strengths in a structured knowledge extraction task, this work presents a novel SLM/LLM routing framework designed to improve computational efficiency and enhance task performance. First, exemplar pools are created to represent the types of contexts where each LM provides a more reliable answer, leveraging a sentence embedding fine-tuned so that context similarity is close to dialogue state similarity. Then, during inference, the k-nearest exemplars to the testing instance are retrieved, and the instance is routed according to majority vote. In dialogue state tracking tasks, the proposed routing framework enhances performance substantially compared to relying solely on LLMs, while reducing the computational costs by over 50%.

{{</citation>}}


### (41/197) FairytaleCQA: Integrating a Commonsense Knowledge Graph into Children's Storybook Narratives (Jiaju Chen et al., 2023)

{{<citation>}}

Jiaju Chen, Yuxuan Lu, Shao Zhang, Bingsheng Yao, Yuanzhe Dong, Ying Xu, Yunyao Li, Qianwen Wang, Dakuo Wang, Yuling Sun. (2023)  
**FairytaleCQA: Integrating a Commonsense Knowledge Graph into Children's Storybook Narratives**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Commonsense Knowledge, GPT, GPT-4, Knowledge Graph, QA, T5  
[Paper Link](http://arxiv.org/abs/2311.09756v1)  

---


**ABSTRACT**  
AI models (including LLM) often rely on narrative question-answering (QA) datasets to provide customized QA functionalities to support downstream children education applications; however, existing datasets only include QA pairs that are grounded within the given storybook content, but children can learn more when teachers refer the storybook content to real-world knowledge (e.g., commonsense knowledge). We introduce the FairytaleCQA dataset, which is annotated by children education experts, to supplement 278 storybook narratives with educationally appropriate commonsense knowledge. The dataset has 5,868 QA pairs that not only originate from the storybook narrative but also contain the commonsense knowledge grounded by an external knowledge graph (i.e., ConceptNet). A follow-up experiment shows that a smaller model (T5-large) fine-tuned with FairytaleCQA reliably outperforms much larger prompt-engineered LLM (e.g., GPT-4) in this new QA-pair generation task (QAG). This result suggests that: 1) our dataset brings novel challenges to existing LLMs, and 2) human experts' data annotation are still critical as they have much nuanced knowledge that LLMs do not know in the children educational domain.

{{</citation>}}


### (42/197) How Does Calibration Data Affect the Post-training Pruning and Quantization of Large Language Models? (Miles Williams et al., 2023)

{{<citation>}}

Miles Williams, Nikolaos Aletras. (2023)  
**How Does Calibration Data Affect the Post-training Pruning and Quantization of Large Language Models?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Pruning, Quantization  
[Paper Link](http://arxiv.org/abs/2311.09755v1)  

---


**ABSTRACT**  
Pruning and quantization form the foundation of model compression for neural networks, enabling efficient inference for large language models (LLMs). Recently, various quantization and pruning techniques have demonstrated state-of-the-art performance in a post-training setting. They rely upon calibration data, a small set of unlabeled examples, to generate layer activations. However, no prior work has systematically investigated how the calibration data impacts the effectiveness of model compression methods. In this paper, we present the first extensive empirical study on the effect of calibration data upon LLM performance. We trial a variety of pruning and quantization methods, tasks, models, and datasets. Surprisingly, we find substantial variations in downstream task performance, contrasting existing work that suggests a greater level of robustness to the calibration data. Finally, we make a series of recommendations for the effective use of calibration data in LLM quantization and pruning.

{{</citation>}}


### (43/197) Translation Aligned Sentence Embeddings for Turkish Language (Eren Unlu et al., 2023)

{{<citation>}}

Eren Unlu, Unver Ciftci. (2023)  
**Translation Aligned Sentence Embeddings for Turkish Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2311.09748v1)  

---


**ABSTRACT**  
Due to the limited availability of high quality datasets for training sentence embeddings in Turkish, we propose a training methodology and a regimen to develop a sentence embedding model. The central idea is simple but effective : is to fine-tune a pretrained encoder-decoder model in two consecutive stages, where the first stage involves aligning the embedding space with translation pairs. Thanks to this alignment, the prowess of the main model can be better projected onto the target language in a sentence embedding setting where it can be fine-tuned with high accuracy in short duration with limited target language dataset.

{{</citation>}}


### (44/197) What Constitutes a Faithful Summary? Preserving Author Perspectives in News Summarization (Yuhan Liu et al., 2023)

{{<citation>}}

Yuhan Liu, Shangbin Feng, Xiaochuang Han, Vidhisha Balachandran, Chan Young Park, Sachin Kumar, Yulia Tsvetkov. (2023)  
**What Constitutes a Faithful Summary? Preserving Author Perspectives in News Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.09741v1)  

---


**ABSTRACT**  
In this work, we take a first step towards designing summarization systems that are faithful to the author's opinions and perspectives. Focusing on a case study of preserving political perspectives in news summarization, we find that existing approaches alter the political opinions and stances of news articles in more than 50% of summaries, misrepresenting the intent and perspectives of the news authors. We thus propose P^3Sum, a diffusion model-based summarization approach controlled by political perspective classifiers. In P^3Sum, the political leaning of a generated summary is iteratively evaluated at each decoding step, and any drift from the article's original stance incurs a loss back-propagated to the embedding layers, steering the political stance of the summary at inference time. Extensive experiments on three news summarization datasets demonstrate that P^3Sum outperforms state-of-the-art summarization systems and large language models by up to 11.4% in terms of the success rate of stance preservation, with on-par performance on standard summarization utility metrics. These findings highlight the lacunae that even for state-of-the-art models it is still challenging to preserve author perspectives in news summarization, while P^3Sum presents an important first step towards evaluating and developing summarization systems that are faithful to author intent and perspectives.

{{</citation>}}


### (45/197) CARE: Extracting Experimental Findings From Clinical Literature (Aakanksha Naik et al., 2023)

{{<citation>}}

Aakanksha Naik, Bailey Kuehl, Erin Bransom, Doug Downey, Tom Hope. (2023)  
**CARE: Extracting Experimental Findings From Clinical Literature**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Clinical, GPT  
[Paper Link](http://arxiv.org/abs/2311.09736v1)  

---


**ABSTRACT**  
Extracting fine-grained experimental findings from literature can provide massive utility for scientific applications. Prior work has focused on developing annotation schemas and datasets for limited aspects of this problem, leading to simpler information extraction datasets which do not capture the real-world complexity and nuance required for this task. Focusing on biomedicine, this work presents CARE (Clinical Aggregation-oriented Result Extraction) -- a new IE dataset for the task of extracting clinical findings. We develop a new annotation schema capturing fine-grained findings as n-ary relations between entities and attributes, which includes phenomena challenging for current IE systems such as discontinuous entity spans, nested relations, and variable arity n-ary relations. Using this schema, we collect extensive annotations for 700 abstracts from two sources: clinical trials and case reports. We also benchmark the performance of various state-of-the-art IE systems on our dataset, including extractive models and generative LLMs in fully supervised and limited data settings. Our results demonstrate the difficulty of our dataset -- even SOTA models such as GPT4 struggle, particularly on relation extraction. We release our annotation schema and CARE to encourage further research on extracting and aggregating scientific findings from literature.

{{</citation>}}


### (46/197) MOKA: Moral Knowledge Augmentation for Moral Event Extraction (Xinliang Frederick Zhang et al., 2023)

{{<citation>}}

Xinliang Frederick Zhang, Winston Wu, Nick Beauchamp, Lu Wang. (2023)  
**MOKA: Moral Knowledge Augmentation for Moral Event Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Event Extraction, NLP  
[Paper Link](http://arxiv.org/abs/2311.09733v1)  

---


**ABSTRACT**  
News media employ moral language to create memorable stories, and readers often engage with the content that align with their values. Moral theories have been applied to news analysis studying moral values in isolation, while the intricate dynamics among participating entities in shaping moral events have been overlooked. This is mainly due to the use of obscure language to conceal evident ideology and values, coupled with the insufficient moral reasoning capability in most existing NLP systems, where LLMs are no exception. To study this phenomenon, we first annotate a new dataset, MORAL EVENTS, consisting of 5,494 structured annotations on 474 news articles by diverse US media across the political spectrum. We further propose MOKA, a moral event extraction framework with MOral Knowledge Augmentation, that leverages knowledge derived from moral words and moral scenarios. Experimental results show that MOKA outperforms competitive baselines across three moral event understanding tasks. Further analyses illuminate the selective reporting of moral events by media outlets of different ideological leanings, suggesting the significance of event-level morality analysis in news. Our datasets and codebase are available at https://github.com/launchnlp/MOKA.

{{</citation>}}


### (47/197) Source Prompt: Coordinated Pre-training of Language Models on Diverse Corpora from Multiple Sources (Yipei Xu et al., 2023)

{{<citation>}}

Yipei Xu, Dakuan Lu, Jiaqing Liang, Xintao Wang, Yipeng Geng, Yingsi Xin, Hengkui Wu, Ken Chen, ruiji zhang, Yanghua Xiao. (2023)  
**Source Prompt: Coordinated Pre-training of Language Models on Diverse Corpora from Multiple Sources**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.09732v1)  

---


**ABSTRACT**  
Pre-trained language models (PLMs) have established the new paradigm in the field of NLP. For more powerful PLMs, one of the most popular and successful way is to continuously scale up sizes of the models and the pre-training corpora. These large corpora are generally obtained by converging smaller ones from multiple sources, they are thus growing increasingly diverse. However, the side-effects of these colossal converged corpora remain understudied. In this paper, we identify the disadvantage of heterogeneous corpora from multiple sources for pre-training PLMs. Towards coordinated pre-training on diverse corpora, we further propose source prompts (SP), which explicitly prompt the model of the data source at the pre-training and fine-tuning stages. Results of extensive experiments demonstrate that PLMs pre-trained with SP on diverse corpora gain significant improvement in various downstream tasks.

{{</citation>}}


### (48/197) Prudent Silence or Foolish Babble? Examining Large Language Models' Responses to the Unknown (Genglin Liu et al., 2023)

{{<citation>}}

Genglin Liu, Xingyao Wang, Lifan Yuan, Yangyi Chen, Hao Peng. (2023)  
**Prudent Silence or Foolish Babble? Examining Large Language Models' Responses to the Unknown**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09731v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) often struggle when faced with situations where they lack the prerequisite knowledge to generate a sensical response. In these cases, models tend to fabricate and hallucinate, rather than appropriately signaling uncertainty as humans would. This behavior misaligns with human conversational norms and presents challenges surrounding responsible and ethical AI development. This work aims to systematically investigate LLMs' behaviors in such situations. We curate an adversarial question-answering benchmark containing unanswerable questions targeting information absent from the LLM's training data. Concretely, these unanswerable questions contain non-existent concepts or false premises. When presented with such unanswerable questions, an LLM should appropriately convey uncertainty, and be able to challenge the premise and refuse to generate a response. While facing answerable valid questions, a model should demonstrate a positive correlation between accuracy and confidence. Using a model-agnostic unified confidence elicitation approach, we observe that LLMs that have gone through instruction finetuning and reinforcement learning from human feedback (RLHF) perform significantly better than their counterparts that do not. Moreover, uncertainty expression 1 through our elicitation method does not always stay consistent with the perceived confidence of the direct response of an LLM. Our findings call for further research into teaching LLMs to proactively and reliably express uncertainty.

{{</citation>}}


### (49/197) Aligning with Whom? Large Language Models Have Gender and Racial Biases in Subjective NLP Tasks (Huaman Sun et al., 2023)

{{<citation>}}

Huaman Sun, Jiaxin Pei, Minje Choi, David Jurgens. (2023)  
**Aligning with Whom? Large Language Models Have Gender and Racial Biases in Subjective NLP Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-HC, cs-LG, cs.CL  
Keywords: Bias, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.09730v1)  

---


**ABSTRACT**  
Human perception of language depends on personal backgrounds like gender and ethnicity. While existing studies have shown that large language models (LLMs) hold values that are closer to certain societal groups, it is unclear whether their prediction behaviors on subjective NLP tasks also exhibit a similar bias. In this study, leveraging the POPQUORN dataset which contains annotations of diverse demographic backgrounds, we conduct a series of experiments on four popular LLMs to investigate their capability to understand group differences and potential biases in their predictions for politeness and offensiveness. We find that for both tasks, model predictions are closer to the labels from White and female participants. We further explore prompting with the target demographic labels and show that including the target demographic in the prompt actually worsens the model's performance. More specifically, when being prompted to respond from the perspective of "Black" and "Asian" individuals, models show lower performance in predicting both overall scores as well as the scores from corresponding groups. Our results suggest that LLMs hold gender and racial biases for subjective NLP tasks and that demographic-infused prompts alone may be insufficient to mitigate such effects. Code and data are available at https://github.com/Jiaxin-Pei/LLM-Group-Bias.

{{</citation>}}


### (50/197) On Evaluating the Integration of Reasoning and Action in LLM Agents with Database Question Answering (Linyong Nan et al., 2023)

{{<citation>}}

Linyong Nan, Ellen Zhang, Weijin Zou, Yilun Zhao, Wenfei Zhou, Arman Cohan. (2023)  
**On Evaluating the Integration of Reasoning and Action in LLM Agents with Database Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09721v1)  

---


**ABSTRACT**  
This study introduces a new long-form database question answering dataset designed to evaluate how Large Language Models (LLMs) interact with a SQL interpreter. The task necessitates LLMs to strategically generate multiple SQL queries to retrieve sufficient data from a database, to reason with the acquired context, and to synthesize them into a comprehensive analytical narrative. Our findings highlight that this task poses great challenges even for the state-of-the-art GPT-4 model. We propose and evaluate two interaction strategies, and provide a fine-grained analysis of the individual stages within the interaction. A key discovery is the identification of two primary bottlenecks hindering effective interaction: the capacity for planning and the ability to generate multiple SQL queries. To address the challenge of accurately assessing answer quality, we introduce a multi-agent evaluation framework that simulates the academic peer-review process, enhancing the precision and reliability of our evaluations. This framework allows for a more nuanced understanding of the strengths and limitations of current LLMs in complex retrieval and reasoning tasks.

{{</citation>}}


### (51/197) You don't need a personality test to know these models are unreliable: Assessing the Reliability of Large Language Models on Psychometric Instruments (Bangzhao Shu et al., 2023)

{{<citation>}}

Bangzhao Shu, Lechen Zhang, Minje Choi, Lavinia Dunagan, Dallas Card, David Jurgens. (2023)  
**You don't need a personality test to know these models are unreliable: Assessing the Reliability of Large Language Models on Psychometric Instruments**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09718v1)  

---


**ABSTRACT**  
The versatility of Large Language Models (LLMs) on natural language understanding tasks has made them popular for research in social sciences. In particular, to properly understand the properties and innate personas of LLMs, researchers have performed studies that involve using prompts in the form of questions that ask LLMs of particular opinions. In this study, we take a cautionary step back and examine whether the current format of prompting enables LLMs to provide responses in a consistent and robust manner. We first construct a dataset that contains 693 questions encompassing 39 different instruments of persona measurement on 115 persona axes. Additionally, we design a set of prompts containing minor variations and examine LLM's capabilities to generate accurate answers, as well as consistency variations to examine their consistency towards simple perturbations such as switching the option order. Our experiments on 15 different open-source LLMs reveal that even simple perturbations are sufficient to significantly downgrade a model's question-answering ability, and that most LLMs have low negation consistency. Our results suggest that the currently widespread practice of prompting is insufficient to accurately capture model perceptions, and we discuss potential alternatives to improve such issues.

{{</citation>}}


### (52/197) Regularized Conventions: Equilibrium Computation as a Model of Pragmatic Reasoning (Athul Paul Jacob et al., 2023)

{{<citation>}}

Athul Paul Jacob, Gabriele Farina, Jacob Andreas. (2023)  
**Regularized Conventions: Equilibrium Computation as a Model of Pragmatic Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09712v1)  

---


**ABSTRACT**  
We present a model of pragmatic language understanding, where utterances are produced and understood by searching for regularized equilibria of signaling games. In this model (which we call ReCo, for Regularized Conventions), speakers and listeners search for contextually appropriate utterance--meaning mappings that are both close to game-theoretically optimal conventions and close to a shared, ''default'' semantics. By characterizing pragmatic communication as equilibrium search, we obtain principled sampling algorithms and formal guarantees about the trade-off between communicative success and naturalness. Across several datasets capturing real and idealized human judgments about pragmatic implicatures, ReCo matches or improves upon predictions made by best response and rational speech act models of language understanding.

{{</citation>}}


### (53/197) Large Language Model Inference with Lexical Shortlisting (Nikolay Bogoychev et al., 2023)

{{<citation>}}

Nikolay Bogoychev, Pinzhen Chen, Barry Haddow, Alexandra Birch. (2023)  
**Large Language Model Inference with Lexical Shortlisting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09709v1)  

---


**ABSTRACT**  
Large language model (LLM) inference is computation and memory intensive, so we adapt lexical shortlisting to it hoping to improve both. While lexical shortlisting is well-explored in tasks like machine translation, it requires modifications before being suitable for LLMs as the intended applications vary significantly. Our work studies two heuristics to shortlist sub-vocabulary at LLM inference time: Unicode-based script filtering and corpus-based selection. We explore different LLM families and sizes, and we find that lexical shortlisting can reduce the memory usage of some models by nearly 50\% and has an upper bound of 25\% improvement in generation speed. In this pilot study, we also identify the drawbacks of such vocabulary selection methods and propose avenues for future research.

{{</citation>}}


### (54/197) GenCodeSearchNet: A Benchmark Test Suite for Evaluating Generalization in Programming Language Understanding (Andor Diera et al., 2023)

{{<citation>}}

Andor Diera, Abdelhalim Dahou, Lukas Galke, Fabian Karl, Florian Sihler, Ansgar Scherp. (2023)  
**GenCodeSearchNet: A Benchmark Test Suite for Evaluating Generalization in Programming Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-PL, cs.CL  
Keywords: BERT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2311.09707v1)  

---


**ABSTRACT**  
Language models can serve as a valuable tool for software developers to increase productivity. Large generative models can be used for code generation and code completion, while smaller encoder-only models are capable of performing code search tasks using natural language queries.These capabilities are heavily influenced by the quality and diversity of the available training data. Source code datasets used for training usually focus on the most popular languages and testing is mostly conducted on the same distributions, often overlooking low-resource programming languages. Motivated by the NLP generalization taxonomy proposed by Hupkes et.\,al., we propose a new benchmark dataset called GenCodeSearchNet (GeCS) which builds upon existing natural language code search datasets to systemically evaluate the programming language understanding generalization capabilities of language models. As part of the full dataset, we introduce a new, manually curated subset StatCodeSearch that focuses on R, a popular but so far underrepresented programming language that is often used by researchers outside the field of computer science. For evaluation and comparison, we collect several baseline results using fine-tuned BERT-style models and GPT-style large language models in a zero-shot setting.

{{</citation>}}


### (55/197) Deceiving Semantic Shortcuts on Reasoning Chains: How Far Can Models Go without Hallucination? (Bangzheng Li et al., 2023)

{{<citation>}}

Bangzheng Li, Ben Zhou, Fei Wang, Xingyu Fu, Dan Roth, Muhao Chen. (2023)  
**Deceiving Semantic Shortcuts on Reasoning Chains: How Far Can Models Go without Hallucination?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09702v1)  

---


**ABSTRACT**  
Despite the recent advancement in large language models (LLMs) and their high performances across numerous benchmarks, recent research has unveiled that LLMs suffer from hallucinations and unfaithful reasoning. This work studies a specific type of hallucination induced by semantic associations. Specifically, we investigate to what extent LLMs take shortcuts from certain keyword/entity biases in the prompt instead of following the correct reasoning path. To quantify this phenomenon, we propose a novel probing method and benchmark called EureQA. We start from questions that LLMs will answer correctly with utmost certainty, and mask the important entity with evidence sentence recursively, asking models to find masked entities according to a chain of evidence before answering the question.   During the construction of the evidence, we purposefully replace semantic clues (entities) that may lead to the correct answer with distractor clues (evidence) that will not directly lead to the correct answer but require a chain-like reasoning process. We evaluate if models can follow the correct reasoning chain instead of short-cutting through distractor clues. We find that existing LLMs lack the necessary capabilities to follow correct reasoning paths and resist the attempt of greedy shortcuts. We show that the distractor semantic associations often lead to model hallucination, which is strong evidence that questions the validity of current LLM reasoning.

{{</citation>}}


### (56/197) Fumbling in Babel: An Investigation into ChatGPT's Language Identification Ability (Wei-Rui Chen et al., 2023)

{{<citation>}}

Wei-Rui Chen, Ife Adebara, Khai Duy Doan, Qisheng Liao, Muhammad Abdul-Mageed. (2023)  
**Fumbling in Babel: An Investigation into ChatGPT's Language Identification Ability**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Language Identification, NLP  
[Paper Link](http://arxiv.org/abs/2311.09696v1)  

---


**ABSTRACT**  
Recently, ChatGPT has emerged as a powerful NLP tool that can carry out several tasks. However, the range of languages ChatGPT can handle remains largely a mystery. In this work, we investigate ChatGPT's language identification abilities. For this purpose, we compile Babel-670, a benchmark comprising $670$ languages representing $23$ language families. Languages in Babel-670 run the gamut between the very high-resource to the very low-resource and are spoken in five continents. We then study ChatGPT's (both GPT-3.5 and GPT-4) ability to (i) identify both language names and language codes (ii) under both zero- and few-shot conditions (iii) with and without provision of label set. When compared to smaller finetuned language identification tools, we find that ChatGPT lags behind. Our empirical analysis shows the reality that ChatGPT still resides in a state of potential enhancement before it can sufficiently serve diverse communities.

{{</citation>}}


### (57/197) Whispers of Doubt Amidst Echoes of Triumph in NLP Robustness (Ashim Gupta et al., 2023)

{{<citation>}}

Ashim Gupta, Rishanth Rajendhran, Nathan Stringham, Vivek Srikumar, Ana Marasović. (2023)  
**Whispers of Doubt Amidst Echoes of Triumph in NLP Robustness**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.09694v1)  

---


**ABSTRACT**  
Are the longstanding robustness issues in NLP resolved by today's larger and more performant models? To address this question, we conduct a thorough investigation using 19 models of different sizes spanning different architectural choices and pretraining objectives. We conduct evaluations using (a) OOD and challenge test sets, (b) CheckLists, (c) contrast sets, and (d) adversarial inputs. Our analysis reveals that not all OOD tests provide further insight into robustness. Evaluating with CheckLists and contrast sets shows significant gaps in model performance; merely scaling models does not make them sufficiently robust. Finally, we point out that current approaches for adversarial evaluations of models are themselves problematic: they can be easily thwarted, and in their current forms, do not represent a sufficiently deep probe of model robustness. We conclude that not only is the question of robustness in NLP as yet unresolved, but even some of the approaches to measure robustness need to be reassessed.

{{</citation>}}


### (58/197) BLT: Can Large Language Models Handle Basic Legal Text? (Andrew Blair-Stanek et al., 2023)

{{<citation>}}

Andrew Blair-Stanek, Nils Holzenberger, Benjamin Van Durme. (2023)  
**BLT: Can Large Language Models Handle Basic Legal Text?**  

---
Primary Category: cs.CL  
Categories: I-2-1; I-2-7; J-7, cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Legal, PaLM  
[Paper Link](http://arxiv.org/abs/2311.09693v1)  

---


**ABSTRACT**  
We find that the best publicly available LLMs like GPT-4 and PaLM 2 currently perform poorly at basic text handling required of lawyers or paralegals, such as looking up the text at a line of a witness deposition or at a subsection of a contract. We introduce a benchmark to quantify this poor performance, which casts into doubt LLMs' current reliability as-is for legal practice. Finetuning for these tasks brings an older LLM to near-perfect performance on our test set and also raises performance on a related legal task. This stark result highlights the need for more domain expertise in LLM training.

{{</citation>}}


### (59/197) Inducing Political Bias Allows Language Models Anticipate Partisan Reactions to Controversies (Zihao He et al., 2023)

{{<citation>}}

Zihao He, Siyi Guo, Ashwin Rao, Kristina Lerman. (2023)  
**Inducing Political Bias Allows Language Models Anticipate Partisan Reactions to Controversies**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.09687v1)  

---


**ABSTRACT**  
Social media platforms are rife with politically charged discussions. Therefore, accurately deciphering and predicting partisan biases using Large Language Models (LLMs) is increasingly critical. In this study, we address the challenge of understanding political bias in digitized discourse using LLMs. While traditional approaches often rely on finetuning separate models for each political faction, our work innovates by employing a singular, instruction-tuned LLM to reflect a spectrum of political ideologies. We present a comprehensive analytical framework, consisting of Partisan Bias Divergence Assessment and Partisan Class Tendency Prediction, to evaluate the model's alignment with real-world political ideologies in terms of stances, emotions, and moral foundations. Our findings reveal the model's effectiveness in capturing emotional and moral nuances, albeit with some challenges in stance detection, highlighting the intricacies and potential for refinement in NLP tools for politically sensitive contexts. This research contributes significantly to the field by demonstrating the feasibility and importance of nuanced political understanding in LLMs, particularly for applications requiring acute awareness of political bias.

{{</citation>}}


### (60/197) Do Physicians Know How to Prompt? The Need for Automatic Prompt Optimization Help in Clinical Note Generation (Zonghai Yao et al., 2023)

{{<citation>}}

Zonghai Yao, Ahmed Jaafar, Beining Wang, Yue Zhu, Zhichao Yang, Hong Yu. (2023)  
**Do Physicians Know How to Prompt? The Need for Automatic Prompt Optimization Help in Clinical Note Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Clinical, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09684v1)  

---


**ABSTRACT**  
This study examines the effect of prompt engineering on the performance of Large Language Models (LLMs) in clinical note generation. We introduce an Automatic Prompt Optimization (APO) framework to refine initial prompts and compare the outputs of medical experts, non-medical experts, and APO-enhanced GPT3.5 and GPT4. Results highlight GPT4 APO's superior performance in standardizing prompt quality across clinical note sections. A human-in-the-loop approach shows that experts maintain content quality post-APO, with a preference for their own modifications, suggesting the value of expert customization. We recommend a two-phase optimization process, leveraging APO-GPT4 for consistency and expert input for personalization.

{{</citation>}}


### (61/197) MacGyver: Are Large Language Models Creative Problem Solvers? (Yufei Tian et al., 2023)

{{<citation>}}

Yufei Tian, Abhilasha Ravichander, Lianhui Qin, Ronan Le Bras, Raja Marjieh, Nanyun Peng, Yejin Choi, Thomas L. Griffiths, Faeze Brahman. (2023)  
**MacGyver: Are Large Language Models Creative Problem Solvers?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09682v1)  

---


**ABSTRACT**  
We explore the creative problem-solving capabilities of modern large language models (LLMs) in a constrained setting. The setting requires circumventing a cognitive bias known in psychology as ''functional fixedness'' to use familiar objects in innovative or unconventional ways. To this end, we create MacGyver, an automatically generated dataset consisting of 1,600 real-world problems that deliberately trigger functional fixedness and require thinking 'out-of-the-box'. We then present our collection of problems to both LLMs and humans to compare and contrast their problem-solving abilities. We show that MacGyver is challenging for both groups, but in unique and complementary ways. For example, humans typically excel in solving problems that they are familiar with but may struggle with tasks requiring domain-specific knowledge, leading to a higher variance. On the other hand, LLMs, being exposed to a variety of highly specialized knowledge, attempt broader problems but are prone to overconfidence and propose actions that are physically infeasible or inefficient. We also provide a detailed error analysis of LLMs, and demonstrate the potential of enhancing their problem-solving ability with novel prompting techniques such as iterative step-wise reflection and divergent-convergent thinking. This work provides insight into the creative problem-solving capabilities of humans and AI and illustrates how psychological paradigms can be extended into large-scale tasks for comparing humans and machines.

{{</citation>}}


### (62/197) R-Tuning: Teaching Large Language Models to Refuse Unknown Questions (Hanning Zhang et al., 2023)

{{<citation>}}

Hanning Zhang, Shizhe Diao, Yong Lin, Yi R. Fung, Qing Lian, Xingyao Wang, Yangyi Chen, Heng Ji, Tong Zhang. (2023)  
**R-Tuning: Teaching Large Language Models to Refuse Unknown Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09677v1)  

---


**ABSTRACT**  
Large language models (LLMs) have revolutionized numerous domains with their impressive performance but still face their challenges. A predominant issue is the propensity for these models to generate non-existent facts, a concern termed hallucination. Our research is motivated by the observation that previous instruction tuning methods force the model to complete a sentence no matter whether the model knows the knowledge or not. When the question is out of the parametric knowledge, it will try to make up something and fail to indicate when it lacks knowledge. In this paper, we present a new approach called Refusal-Aware Instruction Tuning (R-Tuning). This approach is formalized by first identifying the knowledge gap between parametric knowledge and the instruction tuning data. Then, we construct the refusal-aware data based on the knowledge intersection, to tune LLMs to refrain from responding to questions beyond its parametric knowledge. Experimental results demonstrate this new instruction tuning approach effectively improves a model's ability to answer known questions and refrain from answering unknown questions. Furthermore, when tested on out-of-domain datasets, the refusal ability was found to be a meta-skill that could be generalized to other tasks. Further analysis surprisingly finds that learning the uncertainty during training displays a better ability to estimate uncertainty than uncertainty-based testing. Our code will be released at https://github.com/shizhediao/R-Tuning.

{{</citation>}}


### (63/197) Improving the Generation Quality of Watermarked Large Language Models via Word Importance Scoring (Yuhang Li et al., 2023)

{{<citation>}}

Yuhang Li, Yihan Wang, Zhouxing Shi, Cho-Jui Hsieh. (2023)  
**Improving the Generation Quality of Watermarked Large Language Models via Word Importance Scoring**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09668v1)  

---


**ABSTRACT**  
The strong general capabilities of Large Language Models (LLMs) bring potential ethical risks if they are unrestrictedly accessible to malicious users. Token-level watermarking inserts watermarks in the generated texts by altering the token probability distributions with a private random number generator seeded by its prefix tokens. However, this watermarking algorithm alters the logits during generation, which can lead to a downgraded text quality if it chooses to promote tokens that are less relevant given the input. In this work, we propose to improve the quality of texts generated by a watermarked language model by Watermarking with Importance Scoring (WIS). At each generation step, we estimate the importance of the token to generate, and prevent it from being impacted by watermarking if it is important for the semantic correctness of the output. We further propose three methods to predict importance scoring, including a perturbation-based method and two model-based methods. Empirical experiments show that our method can generate texts with better quality with comparable level of detection rate.

{{</citation>}}


### (64/197) Evaluating LLM Agent Group Dynamics against Human Group Dynamics: A Case Study on Wisdom of Partisan Crowds (Yun-Shiuan Chuang et al., 2023)

{{<citation>}}

Yun-Shiuan Chuang, Siddharth Suresh, Nikunj Harlalka, Agam Goyal, Robert Hawkins, Sijia Yang, Dhavan Shah, Junjie Hu, Timothy T. Rogers. (2023)  
**Evaluating LLM Agent Group Dynamics against Human Group Dynamics: A Case Study on Wisdom of Partisan Crowds**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09665v1)  

---


**ABSTRACT**  
This study investigates the potential of Large Language Models (LLMs) to simulate human group dynamics, particularly within politically charged contexts. We replicate the Wisdom of Partisan Crowds phenomenon using LLMs to role-play as Democrat and Republican personas, engaging in a structured interaction akin to human group study. Our approach evaluates how agents' responses evolve through social influence. Our key findings indicate that LLM agents role-playing detailed personas and without Chain-of-Thought (CoT) reasoning closely align with human behaviors, while having CoT reasoning hurts the alignment. However, incorporating explicit biases into agent prompts does not necessarily enhance the wisdom of partisan crowds. Moreover, fine-tuning LLMs with human data shows promise in achieving human-like behavior but poses a risk of overfitting certain behaviors. These findings show the potential and limitations of using LLM agents in modeling human group phenomena.

{{</citation>}}


### (65/197) Evolving Domain Adaptation of Pretrained Language Models for Text Classification (Yun-Shiuan Chuang et al., 2023)

{{<citation>}}

Yun-Shiuan Chuang, Yi Wu, Dhruv Gupta, Rheeya Uppaal, Ananya Kumar, Luhang Sun, Makesh Narsimhan Sreedhar, Sijia Yang, Timothy T. Rogers, Junjie Hu. (2023)  
**Evolving Domain Adaptation of Pretrained Language Models for Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Pretrained Language Models, Text Classification  
[Paper Link](http://arxiv.org/abs/2311.09661v1)  

---


**ABSTRACT**  
Adapting pre-trained language models (PLMs) for time-series text classification amidst evolving domain shifts (EDS) is critical for maintaining accuracy in applications like stance detection. This study benchmarks the effectiveness of evolving domain adaptation (EDA) strategies, notably self-training, domain-adversarial training, and domain-adaptive pretraining, with a focus on an incremental self-training method. Our analysis across various datasets reveals that this incremental method excels at adapting PLMs to EDS, outperforming traditional domain adaptation techniques. These findings highlight the importance of continually updating PLMs to ensure their effectiveness in real-world applications, paving the way for future research into PLM robustness against the natural temporal evolution of language.

{{</citation>}}


### (66/197) Structured Chemistry Reasoning with Large Language Models (Siru Ouyang et al., 2023)

{{<citation>}}

Siru Ouyang, Zhuosheng Zhang, Bing Yan, Xuan Liu, Jiawei Han, Lianhui Qin. (2023)  
**Structured Chemistry Reasoning with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09656v1)  

---


**ABSTRACT**  
This paper studies the problem of solving complex chemistry problems with large language models (LLMs). Despite the extensive general knowledge in LLMs (such as GPT-4), they struggle with chemistry reasoning that requires faithful grounded reasoning with diverse chemical knowledge and an integrative understanding of chemical interactions. We propose InstructChem, a new structured reasoning approach that substantially boosts the LLMs' chemical reasoning capabilities. InstructChem explicitly decomposes the reasoning into three critical phrases, including chemical formulae generation by LLMs that offers the basis for subsequent grounded reasoning, step-by-step reasoning that makes multi-step derivations with the identified formulae for a preliminary answer, and iterative review-and-refinement that steers LLMs to progressively revise the previous phases for increasing confidence, leading to the final high-confidence answer. We conduct extensive experiments on four different chemistry challenges, including quantum chemistry, quantum mechanics, physical chemistry, and chemistry kinetics. Our approach significantly enhances GPT-4 on chemistry reasoning, yielding an 8% average absolute improvement and a 30% peak improvement. We further use the generated reasoning by GPT-4 to fine-tune smaller LMs (e.g., Vicuna) and observe strong improvement of the smaller LMs. This validates our approach and enables LLMs to generate high-quality reasoning.

{{</citation>}}


### (67/197) Event Causality Is Key to Computational Story Understanding (Yidan Sun et al., 2023)

{{<citation>}}

Yidan Sun, Qin Chao, Boyang Li. (2023)  
**Event Causality Is Key to Computational Story Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.09648v1)  

---


**ABSTRACT**  
Psychological research suggests the central role of event causality in human story understanding. Further, event causality has been heavily utilized in symbolic story generation. However, few machine learning systems for story understanding employ event causality, partially due to the lack of reliable methods for identifying open-world causal event relations. Leveraging recent progress in large language models (LLMs), we present the first method for event causality identification that leads to material improvements in computational story understanding. We design specific prompts for extracting event causal relations from GPT. Against human-annotated event causal relations in the GLUCOSE dataset, our technique performs on par with supervised models, while being easily generalizable to stories of different types and lengths. The extracted causal relations lead to 5.7\% improvements on story quality evaluation and 8.7\% on story video-text alignment. Our findings indicate enormous untapped potential for event causality in computational story understanding.

{{</citation>}}


### (68/197) Evaluating In-Context Learning of Libraries for Code Generation (Arkil Patel et al., 2023)

{{<citation>}}

Arkil Patel, Siva Reddy, Dzmitry Bahdanau, Pradeep Dasigi. (2023)  
**Evaluating In-Context Learning of Libraries for Code Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09635v1)  

---


**ABSTRACT**  
Contemporary Large Language Models (LLMs) exhibit a high degree of code generation and comprehension capability. A particularly promising area is their ability to interpret code modules from unfamiliar libraries for solving user-instructed tasks. Recent work has shown that large proprietary LLMs can learn novel library usage in-context from demonstrations. These results raise several open questions: whether demonstrations of library usage is required, whether smaller (and more open) models also possess such capabilities, etc. In this work, we take a broader approach by systematically evaluating a diverse array of LLMs across three scenarios reflecting varying levels of domain specialization to understand their abilities and limitations in generating code based on libraries defined in-context. Our results show that even smaller open-source LLMs like Llama-2 and StarCoder demonstrate an adept understanding of novel code libraries based on specification presented in-context. Our findings further reveal that LLMs exhibit a surprisingly high proficiency in learning novel library modules even when provided with just natural language descriptions or raw code implementations of the functions, which are often cheaper to obtain than demonstrations. Overall, our results pave the way for harnessing LLMs in more adaptable and dynamic coding environments.

{{</citation>}}


### (69/197) Online Continual Knowledge Learning for Language Models (Yuhao Wu et al., 2023)

{{<citation>}}

Yuhao Wu, Tongjun Shi, Karthick Sharma, Chun Wei Seah, Shuhao Zhang. (2023)  
**Online Continual Knowledge Learning for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09632v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) serve as repositories of extensive world knowledge, enabling them to perform tasks such as question-answering and fact-checking. However, this knowledge can become obsolete as global contexts change. In this paper, we introduce a novel problem in the realm of continual learning: Online Continual Knowledge Learning (OCKL). This problem formulation aims to manage the dynamic nature of world knowledge in LMs under real-time constraints. We propose a new benchmark and evaluation metric designed to measure both the rate of new knowledge acquisition and the retention of previously learned knowledge. Our empirical evaluation, conducted using a variety of state-of-the-art methods, establishes robust base-lines for OCKL. Our results reveal that existing continual learning approaches are unfortunately insufficient for tackling the unique challenges posed by OCKL. We identify key factors that influence the trade-off between knowledge acquisition and retention, thereby advancing our understanding of how to train LMs in a continually evolving environment.

{{</citation>}}


### (70/197) From Scroll to Misbelief: Modeling the Unobservable Susceptibility to Misinformation on Social Media (Yanchen Liu et al., 2023)

{{<citation>}}

Yanchen Liu, Mingyu Derek Ma, Wenna Qin, Azure Zhou, Jiaao Chen, Weiyan Shi, Wei Wang, Diyi Yang. (2023)  
**From Scroll to Misbelief: Modeling the Unobservable Susceptibility to Misinformation on Social Media**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-SI, cs.CL  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2311.09630v1)  

---


**ABSTRACT**  
Susceptibility to misinformation describes the extent to believe unverifiable claims, which is hidden in people's mental process and infeasible to observe. Existing susceptibility studies heavily rely on the self-reported beliefs, making any downstream applications on susceptability hard to scale. To address these limitations, in this work, we propose a computational model to infer users' susceptibility levels given their activities. Since user's susceptibility is a key indicator for their reposting behavior, we utilize the supervision from the observable sharing behavior to infer the underlying susceptibility tendency. The evaluation shows that our model yields estimations that are highly aligned with human judgment on users' susceptibility level comparisons. Building upon such large-scale susceptibility labeling, we further conduct a comprehensive analysis of how different social factors relate to susceptibility. We find that political leanings and psychological factors are associated with susceptibility in varying degrees.

{{</citation>}}


### (71/197) Take One Step at a Time to Know Incremental Utility of Demonstration: An Analysis on Reranking for Few-Shot In-Context Learning (Kazuma Hashimoto et al., 2023)

{{<citation>}}

Kazuma Hashimoto, Karthik Raman, Michael Bendersky. (2023)  
**Take One Step at a Time to Know Incremental Utility of Demonstration: An Analysis on Reranking for Few-Shot In-Context Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09619v1)  

---


**ABSTRACT**  
In-Context Learning (ICL) is an emergent capability of Large Language Models (LLMs). Only a few demonstrations enable LLMs to be used as blackbox for new tasks. Previous studies have shown that using LLMs' outputs as labels is effective in training models to select demonstrations. Such a label is expected to estimate utility of a demonstration in ICL; however, it has not been well understood how different labeling strategies affect results on target tasks. This paper presents an analysis on different utility functions by focusing on LLMs' output probability given ground-truth output, and task-specific reward given LLMs' prediction. Unlike the previous work, we introduce a novel labeling method, incremental utility, which estimates how much incremental knowledge is brought into the LLMs by a demonstration. We conduct experiments with instruction-tuned LLMs on binary/multi-class classification, segmentation, and translation across Arabic, English, Finnish, Japanese, and Spanish. Our results show that (1) the probability is effective when the probability values are distributed across the whole value range (on the classification tasks), and (2) the downstream metric is more robust when nuanced reward values are provided with long outputs (on the segmentation and translation tasks). We then show that the proposed incremental utility further helps ICL by contrasting how the LLMs perform with and without the demonstrations.

{{</citation>}}


### (72/197) On Retrieval Augmentation and the Limitations of Language Model Training (Ting-Rui Chiang et al., 2023)

{{<citation>}}

Ting-Rui Chiang, Xinyan Velocity Yu, Joshua Robinson, Ollie Liu, Isabelle Lee, Dani Yogatama. (2023)  
**On Retrieval Augmentation and the Limitations of Language Model Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09615v1)  

---


**ABSTRACT**  
Augmenting a language model (LM) with $k$-nearest neighbors (kNN) retrieval on its training data alone can decrease its perplexity, though the underlying reasons for this remains elusive. In this work, we first rule out one previously posited possibility -- the "softmax bottleneck." We further identify the MLP hurdle phenomenon, where the final MLP layer in LMs may impede LM optimization early on. We explore memorization and generalization in language models with two new datasets, where advanced model like GPT-3.5-turbo find generalizing to irrelevant information in the training data challenging. However, incorporating kNN retrieval to vanilla GPT-2 117M can consistently improve performance in this setting.

{{</citation>}}


### (73/197) Measuring and Improving Attentiveness to Partial Inputs with Counterfactuals (Yanai Elazar et al., 2023)

{{<citation>}}

Yanai Elazar, Bhargavi Paranjape, Hao Peng, Sarah Wiegreffe, Khyathi Raghavi, Vivek Srikumar, Sameer Singh, Noah A. Smith. (2023)  
**Measuring and Improving Attentiveness to Partial Inputs with Counterfactuals**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, NLI, NLP  
[Paper Link](http://arxiv.org/abs/2311.09605v1)  

---


**ABSTRACT**  
The inevitable appearance of spurious correlations in training datasets hurts the generalization of NLP models on unseen data. Previous work has found that datasets with paired inputs are prone to correlations between a specific part of the input (e.g., the hypothesis in NLI) and the label; consequently, models trained only on those outperform chance. Are these correlations picked up by models trained on the full input data? To address this question, we propose a new evaluation method, Counterfactual Attentiveness Test (CAT). CAT uses counterfactuals by replacing part of the input with its counterpart from a different example (subject to some restrictions), expecting an attentive model to change its prediction. Using CAT, we systematically investigate established supervised and in-context learning models on ten datasets spanning four tasks: natural language inference, reading comprehension, paraphrase detection, and visual & language reasoning. CAT reveals that reliance on such correlations is mainly data-dependent. Surprisingly, we find that GPT3 becomes less attentive with an increased number of demonstrations, while its accuracy on the test data improves. Our results demonstrate that augmenting training or demonstration data with counterfactuals is effective in improving models' attentiveness. We show that models' attentiveness measured by CAT reveals different conclusions from solely measuring correlations in data.

{{</citation>}}


### (74/197) SCORE: A framework for Self-Contradictory Reasoning Evaluation (Ziyi Liu et al., 2023)

{{<citation>}}

Ziyi Liu, Isabelle Lee, Yongkang Du, Soumya Sanyal, Jieyu Zhao. (2023)  
**SCORE: A framework for Self-Contradictory Reasoning Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09603v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated impressive reasoning ability in various language-based tasks. Despite many proposed reasoning methods aimed at enhancing performance in downstream tasks, two fundamental questions persist: Does reasoning genuinely support predictions, and how reliable is the quality of reasoning? In this paper, we propose a framework \textsc{SCORE} to analyze how well LLMs can reason. Specifically, we focus on self-contradictory reasoning, where reasoning does not support the prediction. We find that LLMs often contradict themselves when performing reasoning tasks that involve contextual information and commonsense. The model may miss evidence or use shortcuts, thereby exhibiting self-contradictory behaviors. We also employ the Point-of-View (POV) method, which probes models to generate reasoning from multiple perspectives, as a diagnostic tool for further analysis. We find that though LLMs may appear to perform well in one-perspective settings, they fail to stabilize such behavior in multi-perspectives settings. Even for correct predictions, the reasoning may be messy and incomplete, and LLMs can easily be led astray from good reasoning. \textsc{SCORE}'s results underscore the lack of robustness required for trustworthy reasoning and the urgency for further research to establish best practices for a comprehensive evaluation of reasoning beyond accuracy-based metrics.

{{</citation>}}


### (75/197) Language Models (Mostly) Do Not Consider Emotion Triggers When Predicting Emotion (Smriti Singh et al., 2023)

{{<citation>}}

Smriti Singh, Cornelia Caragea, Junyi Jessy Li. (2023)  
**Language Models (Mostly) Do Not Consider Emotion Triggers When Predicting Emotion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09602v1)  

---


**ABSTRACT**  
Situations and events evoke emotions in humans, but to what extent do they inform the prediction of emotion detection models? Prior work in emotion trigger or cause identification focused on training models to recognize events that trigger an emotion. Instead, this work investigates how well human-annotated emotion triggers correlate with features that models deemed salient in their prediction of emotions. First, we introduce a novel dataset EmoTrigger, consisting of 900 social media posts sourced from three different datasets; these were annotated by experts for emotion triggers with high agreement. Using EmoTrigger, we evaluate the ability of large language models (LLMs) to identify emotion triggers, and conduct a comparative analysis of the features considered important for these tasks between LLMs and fine-tuned models. Our analysis reveals that emotion triggers are largely not considered salient features for emotion prediction models, instead there is intricate interplay between various features and the task of emotion detection.

{{</citation>}}


### (76/197) Multi-Step Dialogue Workflow Action Prediction (Ramya Ramakrishnan et al., 2023)

{{<citation>}}

Ramya Ramakrishnan, Ethan Elenberg, Hashan Narangodage, Ryan McDonald. (2023)  
**Multi-Step Dialogue Workflow Action Prediction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2311.09593v1)  

---


**ABSTRACT**  
In task-oriented dialogue, a system often needs to follow a sequence of actions, called a workflow, that complies with a set of guidelines in order to complete a task. In this paper, we propose the novel problem of multi-step workflow action prediction, in which the system predicts multiple future workflow actions. Accurate prediction of multiple steps allows for multi-turn automation, which can free up time to focus on more complex tasks. We propose three modeling approaches that are simple to implement yet lead to more action automation: 1) fine-tuning on a training dataset, 2) few-shot in-context learning leveraging retrieval and large language model prompting, and 3) zero-shot graph traversal, which aggregates historical action sequences into a graph for prediction. We show that multi-step action prediction produces features that improve accuracy on downstream dialogue tasks like predicting task success, and can increase automation of steps by 20% without requiring as much feedback from a human overseeing the system.

{{</citation>}}


### (77/197) A Systematic Review of Aspect-based Sentiment Analysis (ABSA): Domains, Methods, and Trends (Yan Cathy Hua et al., 2023)

{{<citation>}}

Yan Cathy Hua, Paul Denny, Katerina Taskova, Jöerg Wicker. (2023)  
**A Systematic Review of Aspect-based Sentiment Analysis (ABSA): Domains, Methods, and Trends**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.10777v1)  

---


**ABSTRACT**  
Aspect-based Sentiment Analysis (ABSA) is a type of fine-grained sentiment analysis (SA) that identifies aspects and the associated opinions from a given text. In the digital era, ABSA gained increasing popularity and applications in mining opinionated text data to obtain insights and support decisions. ABSA research employs linguistic, statistical, and machine-learning approaches and utilises resources such as labelled datasets, aspect and sentiment lexicons and ontology. By its nature, ABSA is domain-dependent and can be sensitive to the impact of misalignment between the resource and application domains. However, to our knowledge, this topic has not been explored by the existing ABSA literature reviews. In this paper, we present a Systematic Literature Review (SLR) of ABSA studies with a focus on the research application domain, dataset domain, and the research methods to examine their relationships and identify trends over time. Our results suggest a number of potential systemic issues in the ABSA research literature, including the predominance of the ``product/service review'' dataset domain among the majority of studies that did not have a specific research application domain, coupled with the prevalence of dataset-reliant methods such as supervised machine learning. This review makes a number of unique contributions to the ABSA research field: 1) To our knowledge, it is the first SLR that links the research domain, dataset domain, and research method through a systematic perspective; 2) it is one of the largest scoped SLR on ABSA, with 519 eligible studies filtered from 4191 search results without time constraint; and 3) our review methodology adopted an innovative automatic filtering process based on PDF-mining, which enhanced screening quality and reliability. Suggestions and our review limitations are also discussed.

{{</citation>}}


### (78/197) LifeTox: Unveiling Implicit Toxicity in Life Advice (Minbeom Kim et al., 2023)

{{<citation>}}

Minbeom Kim, Jahyun Koo, Hwanhee Lee, Joonsuk Park, Hwaran Lee, Kyomin Jung. (2023)  
**LifeTox: Unveiling Implicit Toxicity in Life Advice**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2311.09585v1)  

---


**ABSTRACT**  
As large language models become increasingly integrated into daily life, detecting implicit toxicity across diverse contexts is crucial. To this end, we introduce LifeTox, a dataset designed for identifying implicit toxicity within a broad range of advice-seeking scenarios. Unlike existing safety datasets, LifeTox comprises diverse contexts derived from personal experiences through open-ended questions. Experiments demonstrate that RoBERTa fine-tuned on LifeTox matches or surpasses the zero-shot performance of large language models in toxicity classification tasks. These results underscore the efficacy of LifeTox in addressing the complex challenges inherent in implicit toxicity.

{{</citation>}}


### (79/197) Enhancing Medical Text Evaluation with GPT-4 (Yiqing Xie et al., 2023)

{{<citation>}}

Yiqing Xie, Sheng Zhang, Hao Cheng, Zelalem Gero, Cliff Wong, Tristan Naumann, Hoifung Poon. (2023)  
**Enhancing Medical Text Evaluation with GPT-4**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.09581v1)  

---


**ABSTRACT**  
In the evaluation of medical text generation, it is essential to scrutinize each piece of information and ensure the utmost accuracy of the evaluation. Existing evaluation metrics either focus on coarse-level evaluation that assigns one score for the whole generated output or rely on evaluation models trained on general domain, resulting in inaccuracies when adapted to the medical domain. To address these issues, we propose a set of factuality-centric evaluation aspects and design corresponding GPT-4-based metrics for medical text generation. We systematically compare these metrics with existing ones on clinical note generation and medical report summarization tasks, revealing low inter-metric correlation. A comprehensive human evaluation confirms that the proposed GPT-4-based metrics exhibit substantially higher agreement with human judgments than existing evaluation metrics. Our study contributes to the understanding of medical text generation evaluation and offers a more reliable alternative to existing metrics.

{{</citation>}}


### (80/197) Work State-Centric AI Agents: Design, Implementation, and Management of Cognitive Work Threads (Chen Zhang, 2023)

{{<citation>}}

Chen Zhang. (2023)  
**Work State-Centric AI Agents: Design, Implementation, and Management of Cognitive Work Threads**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09576v1)  

---


**ABSTRACT**  
AI agents excel in executing predefined tasks, but the dynamic management of work state information during task execution remains an underexplored area. We propose a work state-centric AI agent model employing "work notes" to record and reflect the state throughout task execution. This paper details the model's architecture, featuring worker threads for task oversight, planner modules for task decomposition and planning, and executor modules for performing subtasks using a ReAct-inspired thought-action loop. We provide an exhaustive work state record incorporating plans and outcomes, constituting a comprehensive work journal. Our results show that this model not only improves task execution efficiency but also lays a solid foundation for subsequent task analysis and auditing.

{{</citation>}}


### (81/197) LongBoX: Evaluating Transformers on Long-Sequence Clinical Tasks (Mihir Parmar et al., 2023)

{{<citation>}}

Mihir Parmar, Aakanksha Naik, Himanshu Gupta, Disha Agrawal, Chitta Baral. (2023)  
**LongBoX: Evaluating Transformers on Long-Sequence Clinical Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Clinical, GPT, T5, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.09564v1)  

---


**ABSTRACT**  
Many large language models (LLMs) for medicine have largely been evaluated on short texts, and their ability to handle longer sequences such as a complete electronic health record (EHR) has not been systematically explored. Assessing these models on long sequences is crucial since prior work in the general domain has demonstrated performance degradation of LLMs on longer texts. Motivated by this, we introduce LongBoX, a collection of seven medical datasets in text-to-text format, designed to investigate model performance on long sequences. Preliminary experiments reveal that both medical LLMs (e.g., BioGPT) and strong general domain LLMs (e.g., FLAN-T5) struggle on this benchmark. We further evaluate two techniques designed for long-sequence handling: (i) local-global attention, and (ii) Fusion-in-Decoder (FiD). Our results demonstrate mixed results with long-sequence handling - while scores on some datasets increase, there is substantial room for improvement. We hope that LongBoX facilitates the development of more effective long-sequence techniques for the medical domain. Data and source code are available at https://github.com/Mihir3009/LongBoX.

{{</citation>}}


### (82/197) A Reevaluation of Event Extraction: Past, Present, and Future Challenges (Kuan-Hao Huang et al., 2023)

{{<citation>}}

Kuan-Hao Huang, I-Hung Hsu, Tanmay Parekh, Zhiyu Xie, Zixuan Zhang, Premkumar Natarajan, Kai-Wei Chang, Nanyun Peng, Heng Ji. (2023)  
**A Reevaluation of Event Extraction: Past, Present, and Future Challenges**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Event Extraction  
[Paper Link](http://arxiv.org/abs/2311.09562v1)  

---


**ABSTRACT**  
Event extraction has attracted much attention in recent years due to its potential for many applications. However, recent studies observe some evaluation challenges, suggesting that reported scores might not reflect the true performance. In this work, we first identify and discuss these evaluation challenges, including the unfair comparisons resulting from different assumptions about data or different data preprocessing steps, the incompleteness of the current evaluation framework leading to potential dataset bias or data split bias, and low reproducibility of prior studies. To address these challenges, we propose TextEE, a standardized, fair, and reproducible benchmark for event extraction. TextEE contains standardized data preprocessing scripts and splits for more than ten datasets across different domains. In addition, we aggregate and re-implement over ten event extraction approaches published in recent years and conduct a comprehensive reevaluation. Finally, we explore the capability of large language models in event extraction and discuss some future challenges. We expect TextEE will serve as a reliable benchmark for event extraction, facilitating future research in the field.

{{</citation>}}


### (83/197) Enchancing Semi-Supervised Learning for Extractive Summarization with an LLM-based pseudolabeler (Gaurav Sahu et al., 2023)

{{<citation>}}

Gaurav Sahu, Olga Vechtomova, Issam H. Laradji. (2023)  
**Enchancing Semi-Supervised Learning for Extractive Summarization with an LLM-based pseudolabeler**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Semi-Supervised, Summarization  
[Paper Link](http://arxiv.org/abs/2311.09559v1)  

---


**ABSTRACT**  
This work tackles the task of extractive text summarization in a limited labeled data scenario using a semi-supervised approach. Specifically, we propose a prompt-based pseudolabel selection strategy using GPT-4. We evaluate our method on three text summarization datasets: TweetSumm, WikiHow, and ArXiv/PubMed. Our experiments show that by using an LLM to evaluate and generate pseudolabels, we can improve the ROUGE-1 by 10-20\% on the different datasets, which is akin to enhancing pretrained models. We also show that such a method needs a smaller pool of unlabeled examples to perform better.

{{</citation>}}


### (84/197) Pachinko: Patching Interpretable QA Models through Natural Language Feedback (Chaitanya Malaviya et al., 2023)

{{<citation>}}

Chaitanya Malaviya, Subin Lee, Dan Roth, Mark Yatskar. (2023)  
**Pachinko: Patching Interpretable QA Models through Natural Language Feedback**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, QA  
[Paper Link](http://arxiv.org/abs/2311.09558v1)  

---


**ABSTRACT**  
Eliciting feedback from end users of NLP models can be beneficial for improving models. However, how should we present model responses to users so they are most amenable to be corrected from user feedback? Further, what properties do users value to understand and trust responses? We answer these questions by analyzing the effect of rationales generated by QA models to support their answers. We specifically consider decomposed question-answering models that first extract an intermediate rationale based on a context and a question and then use solely this rationale to answer the question. A rationale outlines the approach followed by the model to answer the question. Our work considers various formats of these rationales that vary according to well-defined properties of interest. We sample these rationales from large language models using few-shot prompting for two reading comprehension datasets, and then perform two user studies. In the first one, we present users with incorrect answers and corresponding rationales of various formats and ask them to provide natural language feedback to revise the rationale. We then measure the effectiveness of this feedback in patching these rationales through in-context learning. The second study evaluates how well different rationale formats enable users to understand and trust model answers, when they are correct. We find that rationale formats significantly affect how easy it is (1) for users to give feedback for rationales, and (2) for models to subsequently execute this feedback. In addition to influencing critiquablity, certain formats significantly enhance user reported understanding and trust of model outputs.

{{</citation>}}


### (85/197) Large Language Models are Few-Shot Training Example Generators: A Case Study in Fallacy Recognition (Tariq Alhindi et al., 2023)

{{<citation>}}

Tariq Alhindi, Smaranda Muresan, Preslav Nakov. (2023)  
**Large Language Models are Few-Shot Training Example Generators: A Case Study in Fallacy Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09552v1)  

---


**ABSTRACT**  
Recognizing fallacies is crucial for ensuring the quality and validity of arguments across various domains. However, computational fallacy recognition faces challenges due to the diverse genres, domains, and types of fallacies found in datasets. This leads to a highly multiclass, and even multi-label, setup with substantial class imbalance. In this study, we aim to enhance existing models for fallacy recognition by incorporating additional context and by leveraging large language models to generate synthetic data, thus increasing the representation of the infrequent classes. We experiment with GPT3.5 to generate synthetic examples and we examine the impact of prompt settings for this. Moreover, we explore zero-shot and few-shot scenarios to evaluate the effectiveness of using the generated examples for training smaller models within a unified fallacy recognition framework. Furthermore, we analyze the overlap between the synthetic data and existing fallacy datasets. Finally, we investigate the usefulness of providing supplementary context for detecting fallacy types that need such context, e.g., diversion fallacies. Our evaluation results demonstrate consistent improvements across fallacy types, datasets, and generators.

{{</citation>}}


### (86/197) Towards Pragmatic Awareness in Question Answering: A Case Study in Maternal and Infant Health (Neha Srikanth et al., 2023)

{{<citation>}}

Neha Srikanth, Rupak Sarkar, Rachel Rudinger, Jordan Boyd-Graber. (2023)  
**Towards Pragmatic Awareness in Question Answering: A Case Study in Maternal and Infant Health**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.09542v1)  

---


**ABSTRACT**  
Questions posed by information-seeking users often contain implicit false or potentially harmful assumptions. In a high-risk domain such as maternal and infant health, a question-answering system must recognize these pragmatic constraints and go beyond simply answering user questions, examining them in context to respond helpfully. To achieve this, we study pragmatic inferences made when mothers ask questions about pregnancy and infant care. Some of the inferences in these questions evade detection by existing methods, risking the possibility of QA systems failing to address them which can have dangerous health and policy implications. We explore the viability of detecting inferences from questions using large language models and illustrate that informing existing QA pipelines with pragmatic inferences produces responses that can mitigate the propagation of harmful beliefs.

{{</citation>}}


### (87/197) Reducing Privacy Risks in Online Self-Disclosures with Language Models (Yao Dou et al., 2023)

{{<citation>}}

Yao Dou, Isadora Krsek, Tarek Naous, Anubha Kabra, Sauvik Das, Alan Ritter, Wei Xu. (2023)  
**Reducing Privacy Risks in Online Self-Disclosures with Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09538v1)  

---


**ABSTRACT**  
Self-disclosure, while being common and rewarding in social media interaction, also poses privacy risks. In this paper, we take the initiative to protect the user-side privacy associated with online self-disclosure through identification and abstraction. We develop a taxonomy of 19 self-disclosure categories, and curate a large corpus consisting of 4.8K annotated disclosure spans. We then fine-tune a language model for identification, achieving over 75% in Token F$_1$. We further conduct a HCI user study, with 82\% of participants viewing the model positively, highlighting its real world applicability. Motivated by the user feedback, we introduce the task of self-disclosure abstraction. We experiment with both one-span abstraction and three-span abstraction settings, and explore multiple fine-tuning strategies. Our best model can generate diverse abstractions that moderately reduce privacy risks while maintaining high utility according to human evaluation.

{{</citation>}}


### (88/197) Effective Large Language Model Adaptation for Improved Grounding (Xi Ye et al., 2023)

{{<citation>}}

Xi Ye, Ruoxi Sun, Sercan Ö. Arik, Tomas Pfister. (2023)  
**Effective Large Language Model Adaptation for Improved Grounding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09533v1)  

---


**ABSTRACT**  
Large language models (LLMs) have achieved remarkable advancements in natural language understanding, generation, and manipulation of text-based data. However, one major issue towards their widespread deployment in the real world is that they can generate "hallucinated" answers that are not factual. Towards this end, this paper focuses on improving grounding from a holistic perspective with a novel framework, AGREE, Adaptation of LLMs for GRounding EnhancEment. We start with the design of an iterative test-time adaptation (TTA) capability that takes into account the support information generated in self-grounded responses. To effectively enable this capability, we tune LLMs to ground the claims in their responses to retrieved documents by providing citations. This tuning on top of the pre-trained LLMs requires a small amount of data that needs to be constructed in a particular way to learn the grounding information, for which we introduce a data construction method. Our results show that the tuning-based AGREE framework generates better grounded responses with more accurate citations compared to prompting-based approaches.

{{</citation>}}


### (89/197) HelpSteer: Multi-attribute Helpfulness Dataset for SteerLM (Zhilin Wang et al., 2023)

{{<citation>}}

Zhilin Wang, Yi Dong, Jiaqi Zeng, Virginia Adams, Makesh Narsimhan Sreedhar, Daniel Egert, Olivier Delalleau, Jane Polak Scowcroft, Neel Kant, Aidan Swope, Oleksii Kuchaiev. (2023)  
**HelpSteer: Multi-attribute Helpfulness Dataset for SteerLM**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.09528v1)  

---


**ABSTRACT**  
Existing open-source helpfulness preference datasets do not specify what makes some responses more helpful and others less so. Models trained on these datasets can incidentally learn to model dataset artifacts (e.g. preferring longer but unhelpful responses only due to their length). To alleviate this problem, we collect HelpSteer, a multi-attribute helpfulness dataset annotated for the various aspects that make responses helpful. Specifically, our 37k-sample dataset has annotations for correctness, coherence, complexity, and verbosity in addition to overall helpfulness of responses. Training Llama 2 70B using the HelpSteer dataset with SteerLM technique produces a model that scores 7.54 on MT Bench, which is currently the highest score for open models that do not require training data from more powerful models (e.g. GPT4). We release this dataset with CC-BY-4.0 license at https://huggingface.co/datasets/nvidia/HelpSteer

{{</citation>}}


### (90/197) AMRFact: Enhancing Summarization Factuality Evaluation with AMR-driven Training Data Generation (Haoyi Qiu et al., 2023)

{{<citation>}}

Haoyi Qiu, Kung-Hsiang Huang, Jingnong Qu, Nanyun Peng. (2023)  
**AMRFact: Enhancing Summarization Factuality Evaluation with AMR-driven Training Data Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Abstract Meaning Representation, Summarization  
[Paper Link](http://arxiv.org/abs/2311.09521v1)  

---


**ABSTRACT**  
Ensuring factual consistency is crucial in various natural language processing tasks, particularly in abstractive summarization, where preserving the integrity of information is paramount. Prior entailment-based approaches often generate factually inconsistent summaries and then train a classifier on the generated data. However, summaries produced by these approaches are either of low coherence or lack error-type coverage. To address these issues, we propose AMRFact, a novel framework that generates factually inconsistent summaries using Abstract Meaning Representation (AMR). Our approach parses factually correct summaries into AMR graphs and injects controlled factual inconsistencies to create negative examples, allowing for coherent factually inconsistent summaries to be generated with high error-type coverage. Additionally, we present a data selection module NegFilter based on natural language inference and BARTScore to ensure the quality of the generated negative samples. Experimental results demonstrate that our approach significantly outperforms previous systems on the AggreFact-SOTA dataset, showcasing its efficacy in assessing factuality in abstractive summarization.

{{</citation>}}


### (91/197) GEE! Grammar Error Explanation with Large Language Models (Yixiao Song et al., 2023)

{{<citation>}}

Yixiao Song, Kalpesh Krishna, Rajesh Bhatt, Kevin Gimpel, Mohit Iyyer. (2023)  
**GEE! Grammar Error Explanation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09517v1)  

---


**ABSTRACT**  
Grammatical error correction tools are effective at correcting grammatical errors in users' input sentences but do not provide users with \textit{natural language} explanations about their errors. Such explanations are essential for helping users learn the language by gaining a deeper understanding of its grammatical rules (DeKeyser, 2003; Ellis et al., 2006). To address this gap, we propose the task of grammar error explanation, where a system needs to provide one-sentence explanations for each grammatical error in a pair of erroneous and corrected sentences. We analyze the capability of GPT-4 in grammar error explanation, and find that it only produces explanations for 60.2% of the errors using one-shot prompting. To improve upon this performance, we develop a two-step pipeline that leverages fine-tuned and prompted large language models to perform structured atomic token edit extraction, followed by prompting GPT-4 to generate explanations. We evaluate our pipeline on German and Chinese grammar error correction data sampled from language learners with a wide range of proficiency levels. Human evaluation reveals that our pipeline produces 93.9% and 98.0% correct explanations for German and Chinese data, respectively. To encourage further research in this area, we will open-source our data and code.

{{</citation>}}


### (92/197) Sequencing Matters: A Generate-Retrieve-Generate Model for Building Conversational Agents (Quinn Patwardhan et al., 2023)

{{<citation>}}

Quinn Patwardhan, Grace Hui Yang. (2023)  
**Sequencing Matters: A Generate-Retrieve-Generate Model for Building Conversational Agents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09513v1)  

---


**ABSTRACT**  
This paper contains what the Georgetown InfoSense group has done in regard to solving the challenges presented by TREC iKAT 2023. Our submitted runs outperform the median runs by a significant margin, exhibiting superior performance in nDCG across various cut numbers and in overall success rate. Our approach uses a Generate-Retrieve-Generate method, which we've found to greatly outpace Retrieve-Then-Generate approaches for the purposes of iKAT. Our solution involves the use of Large Language Models (LLMs) for initial answers, answer grounding by BM25, passage quality filtering by logistic regression, and answer generation by LLMs again. We leverage several purpose-built Language Models, including BERT, Chat-based, and text-to-transfer-based models, for text understanding, classification, generation, and summarization. The official results of the TREC evaluation contradict our initial self-evaluation, which may suggest that a decrease in the reliance on our retrieval and classification methods is better. Nonetheless, our findings suggest that the sequence of involving these different components matters, where we see an essentiality of using LLMs before using search engines.

{{</citation>}}


### (93/197) SegMix: A Simple Structure-Aware Data Augmentation Method (Yuxin Pei et al., 2023)

{{<citation>}}

Yuxin Pei, Pushkar Bhuse, Zhengzhong Liu, Eric Xing. (2023)  
**SegMix: A Simple Structure-Aware Data Augmentation Method**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Augmentation, NER, NLP, Named Entity Recognition, Natural Language Processing, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2311.09505v1)  

---


**ABSTRACT**  
Interpolation-based Data Augmentation (DA) methods (Mixup) linearly interpolate the inputs and labels of two or more training examples. Mixup has more recently been adapted to the field of Natural Language Processing (NLP), mainly for sequence labeling tasks. However, such a simple adoption yields mixed or unstable improvements over the baseline models. We argue that the direct-adoption methods do not account for structures in NLP tasks. To this end, we propose SegMix, a collection of interpolation-based DA algorithms that can adapt to task-specific structures. SegMix poses fewer constraints on data structures, is robust to various hyperparameter settings, applies to more task settings, and adds little computational overhead. In the algorithm's core, we apply interpolation methods on task-specific meaningful segments, in contrast to applying them on sequences as in prior work. We find SegMix to be a flexible framework that combines rule-based DA methods with interpolation-based methods, creating interesting mixtures of DA techniques. We show that SegMix consistently improves performance over strong baseline models in Named Entity Recognition (NER) and Relation Extraction (RE) tasks, especially under data-scarce settings. Furthermore, this method is easy to implement and adds negligible training overhead.

{{</citation>}}


### (94/197) SQATIN: Supervised Instruction Tuning Meets Question Answering for Improved Dialogue NLU (Evgeniia Razumovskaia et al., 2023)

{{<citation>}}

Evgeniia Razumovskaia, Goran Glavaš, Anna Korhonen, Ivan Vulić. (2023)  
**SQATIN: Supervised Instruction Tuning Meets Question Answering for Improved Dialogue NLU**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Intent Detection, NLU, Natural Language Understanding, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.09502v1)  

---


**ABSTRACT**  
Task-oriented dialogue (ToD) systems help users execute well-defined tasks across a variety of domains (e.g., $\textit{flight booking}$ or $\textit{food ordering}$), with their Natural Language Understanding (NLU) components being dedicated to the analysis of user utterances, predicting users' intents ($\textit{Intent Detection}$, ID) and extracting values for informational slots ($\textit{Value Extraction}$, VE). In most domains, labelled NLU data is scarce, making sample-efficient learning -- enabled with effective transfer paradigms -- paramount. In this work, we introduce SQATIN, a new framework for dialog NLU based on (i) instruction tuning and (ii) question-answering-based formulation of ID and VE tasks. According to the evaluation on established NLU benchmarks, SQATIN sets the new state of the art in dialogue NLU, substantially surpassing the performance of current models based on standard fine-tuning objectives in both in-domain training and cross-domain transfer. SQATIN yields particularly large performance gains in cross-domain transfer, owing to the fact that our QA-based instruction tuning leverages similarities between natural language descriptions of classes (i.e., slots and intents) across domains.

{{</citation>}}


### (95/197) Personalized Jargon Identification for Enhanced Interdisciplinary Communication (Yue Guo et al., 2023)

{{<citation>}}

Yue Guo, Joseph Chee Chang, Maria Antoniak, Erin Bransom, Trevor Cohen, Lucy Lu Wang, Tal August. (2023)  
**Personalized Jargon Identification for Enhanced Interdisciplinary Communication**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.09481v1)  

---


**ABSTRACT**  
Scientific jargon can impede researchers when they read materials from other domains. Current methods of jargon identification mainly use corpus-level familiarity indicators (e.g., Simple Wikipedia represents plain language). However, researchers' familiarity of a term can vary greatly based on their own background. We collect a dataset of over 10K term familiarity annotations from 11 computer science researchers for terms drawn from 100 paper abstracts. Analysis of this data reveals that jargon familiarity and information needs vary widely across annotators, even within the same sub-domain (e.g., NLP). We investigate features representing individual, sub-domain, and domain knowledge to predict individual jargon familiarity. We compare supervised and prompt-based approaches, finding that prompt-based methods including personal publications yields the highest accuracy, though zero-shot prompting provides a strong baseline. This research offers insight into features and methods to integrate personal data into scientific jargon identification.

{{</citation>}}


### (96/197) ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems (Jon Saad-Falcon et al., 2023)

{{<citation>}}

Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia. (2023)  
**ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: GLUE, SuperGLUE  
[Paper Link](http://arxiv.org/abs/2311.09476v1)  

---


**ABSTRACT**  
Evaluating retrieval-augmented generation (RAG) systems traditionally relies on hand annotations for input queries, passages to retrieve, and responses to generate. We introduce ARES, an Automated RAG Evaluation System, for evaluating RAG systems along the dimensions of context relevance, answer faithfulness, and answer relevance. Using synthetic training data, ARES finetunes lightweight LM judges to assess the quality of individual RAG components. To mitigate potential prediction errors, ARES utilizes a small set of human-annotated datapoints for prediction-powered inference (PPI). Across six different knowledge-intensive tasks in KILT and SuperGLUE, ARES accurately evaluates RAG systems while using a few hundred human annotations during evaluation. Furthermore, ARES judges remain effective across domain shifts, proving accurate even after changing the type of queries and/or documents used in the evaluated RAG systems. We make our datasets and code for replication and deployment available at https://github.com/stanford-futuredata/ARES.

{{</citation>}}


### (97/197) Clarify When Necessary: Resolving Ambiguity Through Interaction with LMs (Michael J. Q. Zhang et al., 2023)

{{<citation>}}

Michael J. Q. Zhang, Eunsol Choi. (2023)  
**Clarify When Necessary: Resolving Ambiguity Through Interaction with LMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2311.09469v1)  

---


**ABSTRACT**  
Resolving ambiguities through interaction is a hallmark of natural language, and modeling this behavior is a core challenge in crafting AI assistants. In this work, we study such behavior in LMs by proposing a task-agnostic framework for resolving ambiguity by asking users clarifying questions. Our framework breaks down this objective into three subtasks: (1) determining when clarification is needed, (2) determining what clarifying question to ask, and (3) responding accurately with the new information gathered through clarification. We evaluate systems across three NLP applications: question answering, machine translation and natural language inference. For the first subtask, we present a novel uncertainty estimation approach, intent-sim, that determines the utility of querying for clarification by estimating the entropy over user intents. Our method consistently outperforms existing uncertainty estimation approaches at identifying predictions that will benefit from clarification. When only allowed to ask for clarification on 10% of examples, our system is able to double the performance gains over randomly selecting examples to clarify. Furthermore, we find that intent-sim is robust, demonstrating improvements across a wide range of NLP tasks and LMs. Together, our work lays foundation for studying clarifying interactions with LMs.

{{</citation>}}


### (98/197) Think While You Write: Hypothesis Verification Promotes Faithful Knowledge-to-Text Generation (Yifu Qiu et al., 2023)

{{<citation>}}

Yifu Qiu, Varun Embar, Shay B. Cohen, Benjamin Han. (2023)  
**Think While You Write: Hypothesis Verification Promotes Faithful Knowledge-to-Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, NLI, Natural Language Inference, Text Generation, Textual Entailment  
[Paper Link](http://arxiv.org/abs/2311.09467v1)  

---


**ABSTRACT**  
Neural knowledge-to-text generation models often struggle to faithfully generate descriptions for the input facts: they may produce hallucinations that contradict the given facts, or describe facts not present in the input. To reduce hallucinations, we propose a novel decoding method, TWEAK (Think While Effectively Articulating Knowledge). TWEAK treats the generated sequences at each decoding step and its future sequences as hypotheses, and ranks each generation candidate based on how well their corresponding hypotheses support the input facts using a Hypothesis Verification Model (HVM). We first demonstrate the effectiveness of TWEAK by using a Natural Language Inference (NLI) model as the HVM and report improved faithfulness with minimal impact on the quality. We then replace the NLI model with our task-specific HVM trained with a first-of-a-kind dataset, FATE (Fact-Aligned Textual Entailment), which pairs input facts with their faithful and hallucinated descriptions with the hallucinated spans marked. The new HVM improves the faithfulness and the quality further and runs faster. Overall the best TWEAK variants improve on average 2.22/7.17 points on faithfulness measured by FactKB over WebNLG and TekGen/GenWiki, respectively, with only 0.14/0.32 points degradation on quality measured by BERTScore over the same datasets. Since TWEAK is a decoding-only approach, it can be integrated with any neural generative model without retraining.

{{</citation>}}


## eess.SY (2)



### (99/197) Data-Driven LQR using Reinforcement Learning and Quadratic Neural Networks (Soroush Asri et al., 2023)

{{<citation>}}

Soroush Asri, Luis Rodrigues. (2023)  
**Data-Driven LQR using Reinforcement Learning and Quadratic Neural Networks**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.10235v1)  

---


**ABSTRACT**  
This paper introduces a novel data-driven approach to design a linear quadratic regulator (LQR) using a reinforcement learning (RL) algorithm that does not require a system model. The key contribution is to perform policy iteration (PI) by designing the policy evaluator as a two-layer quadratic neural network (QNN). This network is trained through convex optimization. To the best of our knowledge, this is the first time that a QNN trained through convex optimization is employed as the Q-function approximator (QFA). The main advantage is that the QNN's input-output mapping has an analytical expression as a quadratic form, which can then be used to obtain an analytical expression for policy improvement. This is in stark contrast to the available techniques in the literature that must train a second neural network to obtain policy improvement. The article establishes the convergence of the learning algorithm to the optimal control, provided the system is controllable and one starts from a stabilitzing policy. A quadrotor example demonstrates the effectiveness of the proposed approach.

{{</citation>}}


### (100/197) Guaranteeing Control Requirements via Reward Shaping in Reinforcement Learning (Francesco De Lellis et al., 2023)

{{<citation>}}

Francesco De Lellis, Marco Coraggio, Giovanni Russo, Mirco Musolesi, Mario di Bernardo. (2023)  
**Guaranteeing Control Requirements via Reward Shaping in Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.10026v1)  

---


**ABSTRACT**  
In addressing control problems such as regulation and tracking through reinforcement learning, it is often required to guarantee that the acquired policy meets essential performance and stability criteria such as a desired settling time and steady-state error prior to deployment. Motivated by this necessity, we present a set of results and a systematic reward shaping procedure that (i) ensures the optimal policy generates trajectories that align with specified control requirements and (ii) allows to assess whether any given policy satisfies them. We validate our approach through comprehensive numerical experiments conducted in two representative environments from OpenAI Gym: the Inverted Pendulum swing-up problem and the Lunar Lander. Utilizing both tabular and deep reinforcement learning methods, our experiments consistently affirm the efficacy of our proposed framework, highlighting its effectiveness in ensuring policy adherence to the prescribed control requirements.

{{</citation>}}


## cs.AI (10)



### (101/197) Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities (Alex Wilf et al., 2023)

{{<citation>}}

Alex Wilf, Sihyun Shawn Lee, Paul Pu Liang, Louis-Philippe Morency. (2023)  
**Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.10227v1)  

---


**ABSTRACT**  
Human interactions are deeply rooted in the interplay of thoughts, beliefs, and desires made possible by Theory of Mind (ToM): our cognitive ability to understand the mental states of ourselves and others. Although ToM may come naturally to us, emulating it presents a challenge to even the most advanced Large Language Models (LLMs). Recent improvements to LLMs' reasoning capabilities from simple yet effective prompting techniques such as Chain-of-Thought have seen limited applicability to ToM. In this paper, we turn to the prominent cognitive science theory "Simulation Theory" to bridge this gap. We introduce SimToM, a novel two-stage prompting framework inspired by Simulation Theory's notion of perspective-taking. To implement this idea on current ToM benchmarks, SimToM first filters context based on what the character in question knows before answering a question about their mental state. Our approach, which requires no additional training and minimal prompt-tuning, shows substantial improvement over existing methods, and our analysis reveals the importance of perspective-taking to Theory-of-Mind capabilities. Our findings suggest perspective-taking as a promising direction for future research into improving LLMs' ToM capabilities.

{{</citation>}}


### (102/197) Learning interactions to boost human creativity with bandits and GPT-4 (Ara Vartanian et al., 2023)

{{<citation>}}

Ara Vartanian, Xiaoxi Sun, Yun-Shiuan Chuang, Siddharth Suresh, Xiaojin Zhu, Timothy T. Rogers. (2023)  
**Learning interactions to boost human creativity with bandits and GPT-4**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs.AI  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.10127v1)  

---


**ABSTRACT**  
This paper considers how interactions with AI algorithms can boost human creative thought. We employ a psychological task that demonstrates limits on human creativity, namely semantic feature generation: given a concept name, respondents must list as many of its features as possible. Human participants typically produce only a fraction of the features they know before getting "stuck." In experiments with humans and with a language AI (GPT-4) we contrast behavior in the standard task versus a variant in which participants can ask for algorithmically-generated hints. Algorithm choice is administered by a multi-armed bandit whose reward indicates whether the hint helped generating more features. Humans and the AI show similar benefits from hints, and remarkably, bandits learning from AI responses prefer the same prompting strategy as those learning from human behavior. The results suggest that strategies for boosting human creativity via computer interactions can be learned by bandits run on groups of simulated participants.

{{</citation>}}


### (103/197) Towards Formal Fault Injection for Safety Assessment of Automated Systems (Ashfaq Farooqui et al., 2023)

{{<citation>}}

Ashfaq Farooqui, Behrooz Sangchoolie. (2023)  
**Towards Formal Fault Injection for Safety Assessment of Automated Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09810v1)  

---


**ABSTRACT**  
Reasoning about safety, security, and other dependability attributes of autonomous systems is a challenge that needs to be addressed before the adoption of such systems in day-to-day life. Formal methods is a class of methods that mathematically reason about a system's behavior. Thus, a correctness proof is sufficient to conclude the system's dependability. However, these methods are usually applied to abstract models of the system, which might not fully represent the actual system. Fault injection, on the other hand, is a testing method to evaluate the dependability of systems. However, the amount of testing required to evaluate the system is rather large and often a problem. This vision paper introduces formal fault injection, a fusion of these two techniques throughout the development lifecycle to enhance the dependability of autonomous systems. We advocate for a more cohesive approach by identifying five areas of mutual support between formal methods and fault injection. By forging stronger ties between the two fields, we pave the way for developing safe and dependable autonomous systems. This paper delves into the integration's potential and outlines future research avenues, addressing open challenges along the way.

{{</citation>}}


### (104/197) Neuro-Symbolic Integration Brings Causal and Reliable Reasoning Proofs (Sen Yang et al., 2023)

{{<citation>}}

Sen Yang, Xin Li, Leyang Cui, Lidong Bing, Wai Lam. (2023)  
**Neuro-Symbolic Integration Brings Causal and Reliable Reasoning Proofs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09802v1)  

---


**ABSTRACT**  
Though prompting LLMs with various reasoning structures produces reasoning proofs along with answers, these proofs are not ensured to be causal and reliable due to the inherent defects of LLMs. Tracking such deficiencies, we present a neuro-symbolic integration method, in which a neural LLM is used to represent the knowledge of the problem while an LLM-free symbolic solver is adopted to do deliberative reasoning using the knowledge. Specifically, our customized meta-interpreters allow the production of reasoning proofs and support flexible search strategies. These reasoning proofs are ensured to be causal and reliable because of the deterministic executing nature of the symbolic solvers. Empirically, on ProofWriter, our method surpasses the CoT baseline by nearly double in accuracy and more than triple in proof similarity. On GSM8K, our method also shows accuracy improvements and nearly doubled proof similarity. Our code is released at https://github.com/DAMO-NLP-SG/CaRing

{{</citation>}}


### (105/197) Outcome-supervised Verifiers for Planning in Mathematical Reasoning (Fei Yu et al., 2023)

{{<citation>}}

Fei Yu, Anningzhe Gao, Benyou Wang. (2023)  
**Outcome-supervised Verifiers for Planning in Mathematical Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.09724v1)  

---


**ABSTRACT**  
Large language models (LLMs) often struggle with maintaining accuracy across a sequence of intermediate reasoning steps in mathematical reasoning, leading to error propagation that undermines the final result. The current methodology to mitigate this issue primarily involves using a verifier model to assess the correctness of generated solution candidates, focusing either on the overall reasoning path or on an incomplete reasoning path. By rethinking this approach, we argue that assessing potentials of incomplete reasoning paths could be more advantageous as it guides towards correct final answers, transforming the task into a \textit{planning} problem. Our proposed verifier, the Outcome-supervision Value Model (OVM), employs outcome supervision for training, offering an efficient and intuitive method for \textit{planning} by prioritizing steps that lead to accurate conclusions over mere per-step correctness. Furthermore, the OVM eschews the need for labor-intensive annotations on step-level correctness, enhancing its scalability. Our experiments on two multi-step mathematical reasoning datasets, GSM8K and Game of 24, demonstrate the superior performance of the OVM model. Notably, in GSM8K, our \textbf{OVM-7B model achieves state-of-the-art results among LLMs up to 13B parameters}; especially it does not utilize GPT-4 or code execution. These findings offer a novel perspective on the role of outcome supervision in training verifiers for multi-step reasoning tasks and provide theoretical justification for its advantage in value estimation for planning.

{{</citation>}}


### (106/197) Towards Autonomous Hypothesis Verification via Language Models with Minimal Guidance (Shiro Takagi et al., 2023)

{{<citation>}}

Shiro Takagi, Ryutaro Yamauchi, Wataru Kumagai. (2023)  
**Towards Autonomous Hypothesis Verification via Language Models with Minimal Guidance**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs.AI  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09706v1)  

---


**ABSTRACT**  
Research automation efforts usually employ AI as a tool to automate specific tasks within the research process. To create an AI that truly conduct research themselves, it must independently generate hypotheses, design verification plans, and execute verification. Therefore, we investigated if an AI itself could autonomously generate and verify hypothesis for a toy machine learning research problem. We prompted GPT-4 to generate hypotheses and Python code for hypothesis verification with limited methodological guidance. Our findings suggest that, in some instances, GPT-4 can autonomously generate and validate hypotheses without detailed guidance. While this is a promising result, we also found that none of the verifications were flawless, and there remain significant challenges in achieving autonomous, human-level research using only generic instructions. These findings underscore the need for continued exploration to develop a general and autonomous AI researcher.

{{</citation>}}


### (107/197) On the Exploitability of Reinforcement Learning with Human Feedback for Large Language Models (Jiongxiao Wang et al., 2023)

{{<citation>}}

Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, Chaowei Xiao. (2023)  
**On the Exploitability of Reinforcement Learning with Human Feedback for Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CR, cs-HC, cs.AI  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.09641v1)  

---


**ABSTRACT**  
Reinforcement Learning with Human Feedback (RLHF) is a methodology designed to align Large Language Models (LLMs) with human preferences, playing an important role in LLMs alignment. Despite its advantages, RLHF relies on human annotators to rank the text, which can introduce potential security vulnerabilities if any adversarial annotator (i.e., attackers) manipulates the ranking score by up-ranking any malicious text to steer the LLM adversarially. To assess the red-teaming of RLHF against human preference data poisoning, we propose RankPoison, a poisoning attack method on candidates' selection of preference rank flipping to reach certain malicious behaviors (e.g., generating longer sequences, which can increase the computational cost). With poisoned dataset generated by RankPoison, we can perform poisoning attacks on LLMs to generate longer tokens without hurting the original safety alignment performance. Moreover, applying RankPoison, we also successfully implement a backdoor attack where LLMs can generate longer answers under questions with the trigger word. Our findings highlight critical security challenges in RLHF, underscoring the necessity for more robust alignment methods for LLMs.

{{</citation>}}


### (108/197) CRISPR: Eliminating Bias Neurons from an Instruction-following Language Model (Nakyeong Yang et al., 2023)

{{<citation>}}

Nakyeong Yang, Taegwan Kang, Kyomin Jung. (2023)  
**CRISPR: Eliminating Bias Neurons from an Instruction-following Language Model**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09627v1)  

---


**ABSTRACT**  
Large language models (LLMs) executing tasks through instruction-based prompts often face challenges stemming from distribution differences between user instructions and training instructions. This leads to distractions and biases, especially when dealing with inconsistent dynamic labels. In this paper, we introduces a novel bias mitigation method, CRISPR, designed to alleviate instruction-label biases in LLMs. CRISPR utilizes attribution methods to identify bias neurons influencing biased outputs and employs pruning to eliminate the bias neurons. Experimental results demonstrate the method's effectiveness in mitigating biases in instruction-based prompting, enhancing language model performance on social bias benchmarks without compromising pre-existing knowledge. CRISPR proves highly practical, model-agnostic, offering flexibility in adapting to evolving social biases.

{{</citation>}}


### (109/197) Program-Aided Reasoners (better) Know What They Know (Anubha Kabra et al., 2023)

{{<citation>}}

Anubha Kabra, Sanketh Rangreji, Yash Mathur, Aman Madaan, Emmy Liu, Graham Neubig. (2023)  
**Program-Aided Reasoners (better) Know What They Know**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09553v1)  

---


**ABSTRACT**  
Prior work shows that program-aided reasoning, in which large language models (LLMs) are combined with programs written in programming languages such as Python, can significantly improve accuracy on various reasoning tasks. However, while accuracy is essential, it is also important for such reasoners to "know what they know", which can be quantified through the calibration of the model. In this paper, we compare the calibration of Program Aided Language Models (PAL) and text-based Chain-of-thought (COT) prompting techniques over 5 datasets and 2 model types: LLaMA models and OpenAI models. Our results indicate that PAL leads to improved calibration in 75% of the instances. Our analysis uncovers that prompting styles that produce lesser diversity in generations also have more calibrated results, and thus we also experiment with inducing lower generation diversity using temperature scaling and find that for certain temperatures, PAL is not only more accurate but is also more calibrated than COT. Overall, we demonstrate that, in the majority of cases, program-aided reasoners better know what they know than text-based counterparts.

{{</citation>}}


### (110/197) JAB: Joint Adversarial Prompting and Belief Augmentation (Ninareh Mehrabi et al., 2023)

{{<citation>}}

Ninareh Mehrabi, Palash Goyal, Anil Ramakrishna, Jwala Dhamala, Shalini Ghosh, Richard Zemel, Kai-Wei Chang, Aram Galstyan, Rahul Gupta. (2023)  
**JAB: Joint Adversarial Prompting and Belief Augmentation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.09473v1)  

---


**ABSTRACT**  
With the recent surge of language models in different applications, attention to safety and robustness of these models has gained significant importance. Here we introduce a joint framework in which we simultaneously probe and improve the robustness of a black-box target model via adversarial prompting and belief augmentation using iterative feedback loops. This framework utilizes an automated red teaming approach to probe the target model, along with a belief augmenter to generate instructions for the target model to improve its robustness to those adversarial probes. Importantly, the adversarial model and the belief generator leverage the feedback from past interactions to improve the effectiveness of the adversarial prompts and beliefs, respectively. In our experiments, we demonstrate that such a framework can reduce toxic content generation both in dynamic cases where an adversary directly interacts with a target model and static cases where we use a static benchmark dataset to evaluate our model.

{{</citation>}}


## eess.IV (8)



### (111/197) CV-Attention UNet: Attention-based UNet for 3D Cerebrovascular Segmentation of Enhanced TOF-MRA Images (Syed Farhan Abbas et al., 2023)

{{<citation>}}

Syed Farhan Abbas, Nguyen Thanh Duc, Yoonguu Song, Kyungwon Kim, Boreom Lee. (2023)  
**CV-Attention UNet: Attention-based UNet for 3D Cerebrovascular Segmentation of Enhanced TOF-MRA Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.10224v1)  

---


**ABSTRACT**  
Due to the lack of automated methods, to diagnose cerebrovascular disease, time-of-flight magnetic resonance angiography (TOF-MRA) is assessed visually, making it time-consuming. The commonly used encoder-decoder architectures for cerebrovascular segmentation utilize redundant features, eventually leading to the extraction of low-level features multiple times. Additionally, convolutional neural networks (CNNs) suffer from performance degradation when the batch size is small, and deeper networks experience the vanishing gradient problem. Methods: In this paper, we attempt to solve these limitations and propose the 3D cerebrovascular attention UNet method, named CV-AttentionUNet, for precise extraction of brain vessel images. We proposed a sequence of preprocessing techniques followed by deeply supervised UNet to improve the accuracy of segmentation of the brain vessels leading to a stroke. To combine the low and high semantics, we applied the attention mechanism. This mechanism focuses on relevant associations and neglects irrelevant anatomical information. Furthermore, the inclusion of deep supervision incorporates different levels of features that prove to be beneficial for network convergence. Results: We demonstrate the efficiency of the proposed method by cross-validating with an unlabeled dataset, which was further labeled by us. We believe that the novelty of this algorithm lies in its ability to perform well on both labeled and unlabeled data with image processing-based enhancement. The results indicate that our method performed better than the existing state-of-the-art methods on the TubeTK dataset. Conclusion: The proposed method will help in accurate segmentation of cerebrovascular structure leading to stroke

{{</citation>}}


### (112/197) VertDetect: Fully End-to-End 3D Vertebral Instance Segmentation Model (Geoff Klein et al., 2023)

{{<citation>}}

Geoff Klein, Michael Hardisty, Cari Whyne, Anne L. Martel. (2023)  
**VertDetect: Fully End-to-End 3D Vertebral Instance Segmentation Model**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2311.09958v1)  

---


**ABSTRACT**  
Vertebral detection and segmentation are critical steps for treatment planning in spine surgery and radiation therapy. Accurate identification and segmentation are complicated in imaging that does not include the full spine, in cases with variations in anatomy (T13 and/or L6 vertebrae), and in the presence of fracture or hardware. This paper proposes VertDetect, a fully automated end-to-end 3D vertebral instance segmentation Convolutional Neural Network (CNN) model to predict vertebral level labels and segmentations for all vertebrae present in a CT scan. The utilization of a shared CNN backbone provides the detection and segmentation branches of the network with feature maps containing both spinal and vertebral level information. A Graph Convolutional Network (GCN) layer is used to improve vertebral labelling by using the known structure of the spine. This model achieved a Dice Similarity Coefficient (DSC) of 0.883 (95% CI, 0.843-0.906) and 0.882 (95% CI, 0.835-0.909) in the VerSe 2019 and 0.868 (95\% CI, 0.834-0.890) and 0.869 (95\% CI, 0.832-0.891) in the VerSe 2020 public and hidden test sets, respectively. This model achieved state-of-the-art performance for an end-to-end architecture, whose design facilitates the extraction of features that can be subsequently used for downstream tasks.

{{</citation>}}


### (113/197) Harnessing Transformers: A Leap Forward in Lung Cancer Image Detection (Amine Bechar et al., 2023)

{{<citation>}}

Amine Bechar, Youssef Elmir, Rafik Medjoudj, Yassine Himeur, Abbes Amira. (2023)  
**Harnessing Transformers: A Leap Forward in Lung Cancer Image Detection**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.09942v1)  

---


**ABSTRACT**  
This paper discusses the role of Transfer Learning (TL) and transformers in cancer detection based on image analysis. With the enormous evolution of cancer patients, the identification of cancer cells in a patient's body has emerged as a trend in the field of Artificial Intelligence (AI). This process involves analyzing medical images, such as Computed Tomography (CT) scans and Magnetic Resonance Imaging (MRIs), to identify abnormal growths that may help in cancer detection. Many techniques and methods have been realized to improve the quality and performance of cancer classification and detection, such as TL, which allows the transfer of knowledge from one task to another with the same task or domain. TL englobes many methods, particularly those used in image analysis, such as transformers and Convolutional Neural Network (CNN) models trained on the ImageNet dataset. This paper analyzes and criticizes each method of TL based on image analysis and compares the results of each method, showing that transformers have achieved the best results with an accuracy of 97.41% for colon cancer detection and 94.71% for Histopathological Lung cancer. Future directions for cancer detection based on image analysis are also discussed.

{{</citation>}}


### (114/197) GroupMixer: Patch-based Group Convolutional Neural Network for Breast Cancer Detection from Histopathological Images (Ardavan Modarres et al., 2023)

{{<citation>}}

Ardavan Modarres, Erfan Ebrahim Esfahani, Mahsa Bahrami. (2023)  
**GroupMixer: Patch-based Group Convolutional Neural Network for Breast Cancer Detection from Histopathological Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Computer Vision, Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2311.09846v1)  

---


**ABSTRACT**  
Diagnosis of breast cancer malignancy at the early stages is a crucial step for controlling its side effects. Histopathological analysis provides a unique opportunity for malignant breast cancer detection. However, such a task would be tedious and time-consuming for the histopathologists. Deep Neural Networks enable us to learn informative features directly from raw histopathological images without manual feature extraction. Although Convolutional Neural Networks (CNNs) have been the dominant architectures in the computer vision realm, Transformer-based architectures have shown promising results in different computer vision tasks. Although harnessing the capability of Transformer-based architectures for medical image analysis seems interesting, these architectures are large, have a significant number of trainable parameters, and require large datasets to be trained on, which are usually rare in the medical domain. It has been claimed and empirically proved that at least part of the superior performance of Transformer-based architectures in Computer Vision domain originates from patch embedding operation. In this paper, we borrowed the previously introduced idea of integrating a fully Convolutional Neural Network architecture with Patch Embedding operation and presented an efficient CNN architecture for breast cancer malignancy detection from histopathological images. Despite the number of parameters that is significantly smaller than other methods, the accuracy performance metrics achieved 97.65%, 98.92%, 99.21%, and 98.01% for 40x, 100x, 200x, and 400x magnifications respectively. We took a step forward and modified the architecture using Group Convolution and Channel Shuffling ideas and reduced the number of trainable parameters even more with a negligible decline in performance and achieved 95.42%, 98.16%, 96.05%, and 97.92% accuracy for the mentioned magnifications respectively.

{{</citation>}}


### (115/197) Weakly Supervised Anomaly Detection for Chest X-Ray Image (Haoqi Ni et al., 2023)

{{<citation>}}

Haoqi Ni, Ximiao Zhang, Min Xu, Ning Lang, Xiuzhuang Zhou. (2023)  
**Weakly Supervised Anomaly Detection for Chest X-Ray Image**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.09642v2)  

---


**ABSTRACT**  
Chest X-Ray (CXR) examination is a common method for assessing thoracic diseases in clinical applications. While recent advances in deep learning have enhanced the significance of visual analysis for CXR anomaly detection, current methods often miss key cues in anomaly images crucial for identifying disease regions, as they predominantly rely on unsupervised training with normal images. This letter focuses on a more practical setup in which few-shot anomaly images with only image-level labels are available during training. For this purpose, we propose WSCXR, a weakly supervised anomaly detection framework for CXR. WSCXR firstly constructs sets of normal and anomaly image features respectively. It then refines the anomaly image features by eliminating normal region features through anomaly feature mining, thus fully leveraging the scarce yet crucial features of diseased areas. Additionally, WSCXR employs a linear mixing strategy to augment the anomaly features, facilitating the training of anomaly detector with few-shot anomaly images. Experiments on two CXR datasets demonstrate the effectiveness of our approach.

{{</citation>}}


### (116/197) Apoptosis classification using attention based spatio temporal graph convolution neural network (Akash Awasthi, 2023)

{{<citation>}}

Akash Awasthi. (2023)  
**Apoptosis classification using attention based spatio temporal graph convolution neural network**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.09623v1)  

---


**ABSTRACT**  
Accurate classification of apoptosis plays an important role in cell biology research. There are many state-of-the-art approaches which use deep CNNs to perform the apoptosis classification but these approaches do not account for the cell interaction. Our paper proposes the Attention Graph spatio-temporal graph convolutional network to classify the cell death based on the target cells in the video. This method considers the interaction of multiple target cells at each time stamp. We model the whole video sequence as a set of graphs and classify the target cell in the video as dead or alive. Our method encounters both spatial and temporal relationships.

{{</citation>}}


### (117/197) Multi-Task Learning Approach for Unified Biometric Estimation from Fetal Ultrasound Anomaly Scans (Mohammad Areeb Qazi et al., 2023)

{{<citation>}}

Mohammad Areeb Qazi, Mohammed Talha Alam, Ibrahim Almakky, Werner Gerhard Diehl, Leanne Bricker, Mohammad Yaqub. (2023)  
**Multi-Task Learning Approach for Unified Biometric Estimation from Fetal Ultrasound Anomaly Scans**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09607v1)  

---


**ABSTRACT**  
Precise estimation of fetal biometry parameters from ultrasound images is vital for evaluating fetal growth, monitoring health, and identifying potential complications reliably. However, the automated computerized segmentation of the fetal head, abdomen, and femur from ultrasound images, along with the subsequent measurement of fetal biometrics, remains challenging. In this work, we propose a multi-task learning approach to classify the region into head, abdomen and femur as well as estimate the associated parameters. We were able to achieve a mean absolute error (MAE) of 1.08 mm on head circumference, 1.44 mm on abdomen circumference and 1.10 mm on femur length with a classification accuracy of 99.91\% on a dataset of fetal Ultrasound images. To achieve this, we leverage a weighted joint classification and segmentation loss function to train a U-Net architecture with an added classification head. The code can be accessed through \href{https://github.com/BioMedIA-MBZUAI/Multi-Task-Learning-Approach-for-Unified-Biometric-Estimation-from-Fetal-Ultrasound-Anomaly-Scans.git}{\texttt{Github}

{{</citation>}}


### (118/197) MARformer: An Efficient Metal Artifact Reduction Transformer for Dental CBCT Images (Yuxuan Shi et al., 2023)

{{<citation>}}

Yuxuan Shi, Jun Xu, Dinggang Shen. (2023)  
**MARformer: An Efficient Metal Artifact Reduction Transformer for Dental CBCT Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.09590v1)  

---


**ABSTRACT**  
Cone Beam Computed Tomography (CBCT) plays a key role in dental diagnosis and surgery. However, the metal teeth implants could bring annoying metal artifacts during the CBCT imaging process, interfering diagnosis and downstream processing such as tooth segmentation. In this paper, we develop an efficient Transformer to perform metal artifacts reduction (MAR) from dental CBCT images. The proposed MAR Transformer (MARformer) reduces computation complexity in the multihead self-attention by a new Dimension-Reduced Self-Attention (DRSA) module, based on that the CBCT images have globally similar structure. A Patch-wise Perceptive Feed Forward Network (P2FFN) is also proposed to perceive local image information for fine-grained restoration. Experimental results on CBCT images with synthetic and real-world metal artifacts show that our MARformer is efficient and outperforms previous MAR methods and two restoration Transformers.

{{</citation>}}


## cs.SI (1)



### (119/197) Measuring Moral Dimensions in Social Media with Mformer (Tuan Dung Nguyen et al., 2023)

{{<citation>}}

Tuan Dung Nguyen, Ziyu Chen, Nicholas George Carroll, Alasdair Tran, Colin Klein, Lexing Xie. (2023)  
**Measuring Moral Dimensions in Social Media with Mformer**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2311.10219v1)  

---


**ABSTRACT**  
The ever-growing textual records of contemporary social issues, often discussed online with moral rhetoric, present both an opportunity and a challenge for studying how moral concerns are debated in real life. Moral foundations theory is a taxonomy of intuitions widely used in data-driven analyses of online content, but current computational tools to detect moral foundations suffer from the incompleteness and fragility of their lexicons and from poor generalization across data domains. In this paper, we fine-tune a large language model to measure moral foundations in text based on datasets covering news media and long- and short-form online discussions. The resulting model, called Mformer, outperforms existing approaches on the same domains by 4--12% in AUC and further generalizes well to four commonly used moral text datasets, improving by up to 17% in AUC. We present case studies using Mformer to analyze everyday moral dilemmas on Reddit and controversies on Twitter, showing that moral foundations can meaningfully describe people's stance on social issues and such variations are topic-dependent. Pre-trained model and datasets are released publicly. We posit that Mformer will help the research community quantify moral dimensions for a range of tasks and data domains, and eventually contribute to the understanding of moral situations faced by humans and machines.

{{</citation>}}


## cs.CY (6)



### (120/197) Generative AI in Undergraduate Information Technology Education -- Insights from nine courses (Anh Nguyen Duc et al., 2023)

{{<citation>}}

Anh Nguyen Duc, Tor Lønnestad, Ingrid Sundbø, Marius Rohde Johannessen, Veralia Gabriela, Salah Uddin Ahmed, Rania El-Gazzar. (2023)  
**Generative AI in Undergraduate Information Technology Education -- Insights from nine courses**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-SE, cs.CY  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.10199v1)  

---


**ABSTRACT**  
The increasing use of digital teaching and emerging technologies, particularly AI-based tools, such as ChatGPT, is presenting an inevitable and significant impact on higher education. The capability of processing and generating text could bring change to several areas, such as learning assessments or learning experiences. Besides the negative impact, i.e exam cheating, we also see a positive side that ChatGPT can bring to education. This research article aims to contribute to the current debate on ChatGPT by systematic reflection and experience reported from nine bachelor IT courses at a Norwegian university. We conducted inductive empirical research with reflective notes and focused groups of lecturers from nine different IT courses. The findings were thematically organized with numerous use cases in teaching IT subjects. Our discussion highlights the disruptive implications of AI assistant usage in higher education and emphasizes the need for educators to shape this transformation.

{{</citation>}}


### (121/197) Examining bias perpetuation in academic search engines: an algorithm audit of Google and Semantic Scholar (Celina Kacperski et al., 2023)

{{<citation>}}

Celina Kacperski, Mona Bielig, Mykola Makorthyk, Maryna Sydorova, Roberto Ulloa. (2023)  
**Examining bias perpetuation in academic search engines: an algorithm audit of Google and Semantic Scholar**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.09969v1)  

---


**ABSTRACT**  
Researchers rely on academic web search engines to find scientific sources, but search engine mechanisms may selectively present content that aligns with biases embedded in the queries. This study examines whether confirmation-biased queries prompted into Google Scholar and Semantic Scholar will yield skewed results. Six queries (topics across health and technology domains such as "vaccines" or "internet use") were analyzed for disparities in search results. We confirm that biased queries (targeting "benefits" or "risks") affect search results in line with the bias, with technology-related queries displaying more significant disparities. Overall, Semantic Scholar exhibited fewer disparities than Google Scholar. Topics rated as more polarizing did not consistently show more skewed results. Academic search results that perpetuate confirmation bias have strong implications for both researchers and citizens searching for evidence. More research is needed to explore how scientific inquiry and academic search engines interact.

{{</citation>}}


### (122/197) An Attention-Based Denoising Framework for Personality Detection in Social Media Texts (Qirui Tang et al., 2023)

{{<citation>}}

Qirui Tang, Wenkang Jiang, Yihua Du, Lei Lin. (2023)  
**An Attention-Based Denoising Framework for Personality Detection in Social Media Texts**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY  
Keywords: AI, Attention, Personality Detection, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2311.09945v1)  

---


**ABSTRACT**  
In social media networks, users produce a large amount of text content anytime, providing researchers with a valuable approach to digging for personality-related information. Personality detection based on user-generated texts is a universal method that can be used to build user portraits. The presence of noise in social media texts hinders personality detection. However, previous studies have not fully addressed this challenge. Inspired by the scanning reading technique, we propose an attention-based information extraction mechanism (AIEM) for long texts, which is applied to quickly locate valuable pieces of information, and focus more attention on the deep semantics of key pieces. Then, we provide a novel attention-based denoising framework (ADF) for personality detection tasks and achieve state-of-the-art performance on two commonly used datasets. Notably, we obtain an average accuracy improvement of 10.2% on the gold standard Twitter-Myers-Briggs Type Indicator (Twitter-MBTI) dataset. We made our code publicly available on GitHub. We shed light on how AIEM works to magnify personality-related signals.

{{</citation>}}


### (123/197) Echo Chambers within the Russo-Ukrainian War: The Role of Bipartisan Users (Peixian Zhang et al., 2023)

{{<citation>}}

Peixian Zhang, Ehsan-Ul Haq, Yiming Zhu, Pan Hui, Gareth Tyson. (2023)  
**Echo Chambers within the Russo-Ukrainian War: The Role of Bipartisan Users**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2311.09934v1)  

---


**ABSTRACT**  
The ongoing Russia-Ukraine war has been extensively discussed on social media. One commonly observed problem in such discussions is the emergence of echo chambers, where users are rarely exposed to opinions outside their worldview. Prior literature on this topic has assumed that such users hold a single consistent view. However, recent work has revealed that complex topics (such as the war) often trigger bipartisanship among certain people. With this in mind, we study the presence of echo chambers on Twitter related to the Russo-Ukrainian war. We measure their presence and identify an important subset of bipartisan users who vary their opinions during the invasion. We explore the role they play in the communications graph and identify features that distinguish them from remaining users. We conclude by discussing their importance and how they can improve the quality of discourse surrounding the war.

{{</citation>}}


### (124/197) TransCrimeNet: A Transformer-Based Model for Text-Based Crime Prediction in Criminal Networks (Chen Yang, 2023)

{{<citation>}}

Chen Yang. (2023)  
**TransCrimeNet: A Transformer-Based Model for Text-Based Crime Prediction in Criminal Networks**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2311.09529v1)  

---


**ABSTRACT**  
This paper presents TransCrimeNet, a novel transformer-based model for predicting future crimes in criminal networks from textual data. Criminal network analysis has become vital for law enforcement agencies to prevent crimes. However, existing graph-based methods fail to effectively incorporate crucial textual data like social media posts and interrogation transcripts that provide valuable insights into planned criminal activities. To address this limitation, we develop TransCrimeNet which leverages the representation learning capabilities of transformer models like BERT to extract features from unstructured text data. These text-derived features are fused with graph embeddings of the criminal network for accurate prediction of future crimes. Extensive experiments on real-world criminal network datasets demonstrate that TransCrimeNet outperforms previous state-of-the-art models by 12.7\% in F1 score for crime prediction. The results showcase the benefits of combining textual and graph-based features for actionable insights to disrupt criminal enterprises.

{{</citation>}}


### (125/197) From GPT-3 to GPT-4: On the Evolving Efficacy of LLMs to Answer Multiple-choice Questions for Programming Classes in Higher Education (Jaromir Savelka et al., 2023)

{{<citation>}}

Jaromir Savelka, Arav Agarwal, Christopher Bogart, Majd Sakr. (2023)  
**From GPT-3 to GPT-4: On the Evolving Efficacy of LLMs to Answer Multiple-choice Questions for Programming Classes in Higher Education**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.09518v1)  

---


**ABSTRACT**  
We explore the evolving efficacy of three generative pre-trained transformer (GPT) models in generating answers for multiple-choice questions (MCQ) from introductory and intermediate Python programming courses in higher education. We focus on the differences in capabilities of the models prior to the release of ChatGPT (Nov '22), at the time of the release, and today (i.e., Aug '23). Recent studies have established that the abilities of the OpenAI's GPT models to handle assessments originally designed for humans keep increasing as the newer more capable models are released. However, the qualitative differences in the capabilities and limitations of these models to reason about and/or analyze programming MCQs have been under-explored. We evaluated three OpenAI's GPT models on formative and summative MCQ assessments from three Python courses (530 questions) focusing on the qualitative differences in the evolving efficacy of the subsequent models. This study provides further evidence and insight into the trajectory of the current developments where there already exists a technology that can be utilized by students to collect passing scores, with no effort whatsoever, on what today counts as viable programming knowledge and skills assessments. This study could be leveraged by educators and institutions to better understand the recent technological developments in order to adapt the design of programming assessments as well as to fuel the necessary discussions into how assessments in future programming classes should be updated.

{{</citation>}}


## cs.CR (5)



### (126/197) You Cannot Escape Me: Detecting Evasions of SIEM Rules in Enterprise Networks (Rafael Uetz et al., 2023)

{{<citation>}}

Rafael Uetz, Marco Herzog, Louis Hackländer, Simon Schwarz, Martin Henze. (2023)  
**You Cannot Escape Me: Detecting Evasions of SIEM Rules in Enterprise Networks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.10197v1)  

---


**ABSTRACT**  
Cyberattacks have grown into a major risk for organizations, with common consequences being data theft, sabotage, and extortion. Since preventive measures do not suffice to repel attacks, timely detection of successful intruders is crucial to stop them from reaching their final goals. For this purpose, many organizations utilize Security Information and Event Management (SIEM) systems to centrally collect security-related events and scan them for attack indicators using expert-written detection rules. However, as we show by analyzing a set of widespread SIEM detection rules, adversaries can evade almost half of them easily, allowing them to perform common malicious actions within an enterprise network without being detected. To remedy these critical detection blind spots, we propose the idea of adaptive misuse detection, which utilizes machine learning to compare incoming events to SIEM rules on the one hand and known-benign events on the other hand to discover successful evasions. Based on this idea, we present AMIDES, an open-source proof-of-concept adaptive misuse detection system. Using four weeks of SIEM events from a large enterprise network and more than 500 hand-crafted evasions, we show that AMIDES successfully detects a majority of these evasions without any false alerts. In addition, AMIDES eases alert analysis by assessing which rules were evaded. Its computational efficiency qualifies AMIDES for real-world operation and hence enables organizations to significantly reduce detection blind spots with moderate effort.

{{</citation>}}


### (127/197) Practical Cybersecurity Ethics: Mapping CyBOK to Ethical Concerns (Ivan Flechais et al., 2023)

{{<citation>}}

Ivan Flechais, George Chalhoub. (2023)  
**Practical Cybersecurity Ethics: Mapping CyBOK to Ethical Concerns**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-HC, cs.CR  
Keywords: AI, Cyber Security, Security  
[Paper Link](http://arxiv.org/abs/2311.10165v1)  

---


**ABSTRACT**  
Research into the ethics of cybersecurity is an established and growing topic of investigation, however the translation of this research into practice is lacking: there exists a small number of professional codes of ethics or codes of practice in cybersecurity, however these are very broad and do not offer much insight into the ethical dilemmas that can be faced while performing specific cybersecurity activities. In order to address this gap, we leverage ongoing work on the Cyber Security Body of Knowledge (CyBOK) to help elicit and document the responsibilities and ethics of the profession. Based on a literature review of the ethics of cybersecurity, we use CyBOK to frame the exploration of ethical challenges in the cybersecurity profession through a series of 15 interviews with cybersecurity experts. Our approach is qualitative and exploratory, aiming to answer the research question "What ethical challenges, insights, and solutions arise in different areas of cybersecurity?". Our findings indicate that there are broad ethical challenges across the whole of cybersecurity, but also that different areas of cybersecurity can face specific ethical considerations for which more detailed guidance can help professionals in those areas. In particular, our findings indicate that security decision-making is expected of all security professionals, but that this requires them to balance a complex mix of technical, objective and subjective points of view, and that resolving conflicts raises challenging ethical dilemmas. We conclude that more work is needed to explore, map, and integrate ethical considerations into cybersecurity practice; the urgent need to conduct further research into the ethics of cybersecurity AI; and highlight the importance of this work for individuals and professional bodies who seek to develop and mature the cybersecurity profession in a responsible manner.

{{</citation>}}


### (128/197) Towards more Practical Threat Models in Artificial Intelligence Security (Kathrin Grosse et al., 2023)

{{<citation>}}

Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Alexandre Alahi. (2023)  
**Towards more Practical Threat Models in Artificial Intelligence Security**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2311.09994v1)  

---


**ABSTRACT**  
Recent works have identified a gap between research and practice in artificial intelligence security: threats studied in academia do not always reflect the practical use and security risks of AI. For example, while models are often studied in isolation, they form part of larger ML pipelines in practice. Recent works also brought forward that adversarial manipulations introduced by academic attacks are impractical. We take a first step towards describing the full extent of this disparity. To this end, we revisit the threat models of the six most studied attacks in AI security research and match them to AI usage in practice via a survey with \textbf{271} industrial practitioners. On the one hand, we find that all existing threat models are indeed applicable. On the other hand, there are significant mismatches: research is often too generous with the attacker, assuming access to information not frequently available in real-world settings. Our paper is thus a call for action to study more practical threat models in artificial intelligence security.

{{</citation>}}


### (129/197) SynDiffix: More accurate synthetic structured data (Paul Francis et al., 2023)

{{<citation>}}

Paul Francis, Cristian Berneanu, Edon Gashi. (2023)  
**SynDiffix: More accurate synthetic structured data**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09628v1)  

---


**ABSTRACT**  
This paper introduces SynDiffix, a mechanism for generating statistically accurate, anonymous synthetic data for structured data. Recent open source and commercial systems use Generative Adversarial Networks or Transformed Auto Encoders to synthesize data, and achieve anonymity through overfitting-avoidance. By contrast, SynDiffix exploits traditional mechanisms of aggregation, noise addition, and suppression among others. Compared to CTGAN, ML models generated from SynDiffix are twice as accurate, marginal and column pairs data quality is one to two orders of magnitude more accurate, and execution time is two orders of magnitude faster. Compared to the best commercial product we measured (MostlyAI), ML model accuracy is comparable, marginal and pairs accuracy is 5 to 10 times better, and execution time is an order of magnitude faster. Similar to the other approaches, SynDiffix anonymization is very strong. This paper describes SynDiffix and compares its performance with other popular open source and commercial systems.

{{</citation>}}


### (130/197) FunctionMarker: Watermarking Language Datasets via Knowledge Injection (Shuai Li et al., 2023)

{{<citation>}}

Shuai Li, Kejiang Chen, Kunsheng Tang, Wen Huang, Jie Zhang, Weiming Zhang, Nenghai Yu. (2023)  
**FunctionMarker: Watermarking Language Datasets via Knowledge Injection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09535v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated superior performance in various natural language processing tasks. Meanwhile, they require extensive training data, raising concerns related to dataset copyright protection. Backdoor-based watermarking is a viable approach to protect the copyright of classification datasets. However, these methods may introduce malicious misclassification behaviors into watermarked LLMs by attackers and also affect the semantic information of the watermarked text. To address these issues, we propose FunctionMarker, a novel copyright protection method for language datasets via knowledge injection. FunctionMarker enables LLMs to learn specific knowledge through fine-tuning on watermarked datasets, and we can extract the embedded watermark by obtaining the responses of LLMs to specific knowledge-related queries. Considering watermark capacity and stealthness, we select customizable functions as specific knowledge for LLMs to learn and embed the watermark into them. Moreover, FunctionMarker can embed multi-bit watermarks while preserving the original semantic information, thereby increasing the difficulty of adaptive attacks. We take mathematical functions as an instance to evaluate the effectiveness of FunctionMarker, and experiments show that only 0.3% of watermarked text achieves a 90% watermark extraction accuracy in most cases, validating our method's effectiveness.

{{</citation>}}


## cs.RO (5)



### (131/197) Utility AI for Dynamic Task Offloading in the Multi-Edge Infrastructure (Nazish Tahir et al., 2023)

{{<citation>}}

Nazish Tahir, Ramviyas Parasuraman. (2023)  
**Utility AI for Dynamic Task Offloading in the Multi-Edge Infrastructure**  

---
Primary Category: cs.RO  
Categories: cs-NI, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.10196v1)  

---


**ABSTRACT**  
To circumvent persistent connectivity to the cloud infrastructure, the current emphasis on computing at network edge devices in the multi-robot domain is a promising enabler for delay-sensitive jobs, yet its adoption is rife with challenges. This paper proposes a novel utility-aware dynamic task offloading strategy based on a multi-edge-robot system that takes into account computation, communication, and task execution load to minimize the overall service time for delay-sensitive applications. Prior to task offloading, continuous device, network, and task profiling are performed, and for each task assigned, an edge with maximum utility is derived using a weighted utility maximization technique, and a system reward assignment for task connectivity or sensitivity is performed. A scheduler is in charge of task assignment, whereas an executor is responsible for task offloading on edge devices. Experimental comparisons of the proposed approach with conventional offloading methods indicate better performance in terms of optimizing resource utilization and minimizing task latency.

{{</citation>}}


### (132/197) Interpretable Reinforcement Learning for Robotics and Continuous Control (Rohan Paleja et al., 2023)

{{<citation>}}

Rohan Paleja, Letian Chen, Yaru Niu, Andrew Silva, Zhaoxin Li, Songan Zhang, Chace Ritchie, Sugju Choi, Kimberlee Chestnut Chang, Hongtei Eric Tseng, Yan Wang, Subramanya Nageshrao, Matthew Gombolay. (2023)  
**Interpretable Reinforcement Learning for Robotics and Continuous Control**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.10041v1)  

---


**ABSTRACT**  
Interpretability in machine learning is critical for the safe deployment of learned policies across legally-regulated and safety-critical domains. While gradient-based approaches in reinforcement learning have achieved tremendous success in learning policies for continuous control problems such as robotics and autonomous driving, the lack of interpretability is a fundamental barrier to adoption. We propose Interpretable Continuous Control Trees (ICCTs), a tree-based model that can be optimized via modern, gradient-based, reinforcement learning approaches to produce high-performing, interpretable policies. The key to our approach is a procedure for allowing direct optimization in a sparse decision-tree-like representation. We validate ICCTs against baselines across six domains, showing that ICCTs are capable of learning policies that parity or outperform baselines by up to 33% in autonomous driving scenarios while achieving a 300x-600x reduction in the number of parameters against deep learning baselines. We prove that ICCTs can serve as universal function approximators and display analytically that ICCTs can be verified in linear time. Furthermore, we deploy ICCTs in two realistic driving domains, based on interstate Highway-94 and 280 in the US. Finally, we verify ICCT's utility with end-users and find that ICCTs are rated easier to simulate, quicker to validate, and more interpretable than neural networks.

{{</citation>}}


### (133/197) Dynamic modeling of wing-assisted inclined running with a morphing multi-modal robot (Eric Sihite et al., 2023)

{{<citation>}}

Eric Sihite, Alireza Ramezani, Morteza Gharib. (2023)  
**Dynamic modeling of wing-assisted inclined running with a morphing multi-modal robot**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09963v1)  

---


**ABSTRACT**  
Robot designs can take many inspirations from nature, where there are many examples of highly resilient and fault-tolerant locomotion strategies to navigate complex terrains by using multi-functional appendages. For example, Chukar and Hoatzin birds can repurpose their wings for quadrupedal walking and wing-assisted incline running (WAIR) to climb steep surfaces. We took inspiration from nature and designed a morphing robot with multi-functional thruster-wheel appendages that allows the robot to change its mode of locomotion by transforming into a rover, quad-rotor, mobile inverted pendulum (MIP), and other modes. In this work, we derive a dynamic model and formulate a nonlinear model predictive controller to perform WAIR to showcase the unique capabilities of our robot. We implemented the model and controller in a numerical simulation and experiments to show their feasibility and the capabilities of our transforming multi-modal robot.

{{</citation>}}


### (134/197) Short vs. Long-term Coordination of Drones: When Distributed Optimization Meets Deep Reinforcement Learning (Chuhao Qin et al., 2023)

{{<citation>}}

Chuhao Qin, Evangelos Pournaras. (2023)  
**Short vs. Long-term Coordination of Drones: When Distributed Optimization Meets Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-MA, cs-RO, cs.RO  
Keywords: Drone, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.09852v1)  

---


**ABSTRACT**  
Swarms of smart drones, with the support of charging technology, can provide completing sensing capabilities in Smart Cities, such as traffic monitoring and disaster response. Existing approaches, including distributed optimization and deep reinforcement learning (DRL), aim to coordinate drones to achieve cost-effective, high-quality navigation, sensing, and recharging. However, they have distinct challenges: short-term optimization struggles to provide sustained benefits, while long-term DRL lacks scalability, resilience, and flexibility. To bridge this gap, this paper introduces a new progressive approach that encompasses the planning and selection based on distributed optimization, as well as DRL-based flying direction scheduling. Extensive experiment with datasets generated from realisitic urban mobility demonstrate the outstanding performance of the proposed solution in traffic monitoring compared to three baseline methods.

{{</citation>}}


### (135/197) Soft and Rigid Object Grasping With Cross-Structure Hand Using Bilateral Control-Based Imitation Learning (Koki Yamane et al., 2023)

{{<citation>}}

Koki Yamane, Sho Sakaino, Toshiaki Tsuji. (2023)  
**Soft and Rigid Object Grasping With Cross-Structure Hand Using Bilateral Control-Based Imitation Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09555v1)  

---


**ABSTRACT**  
Object grasping is an important ability required for various robot tasks. In particular, tasks that require precise force adjustments during operation, such as grasping an unknown object or using a grasped tool, are difficult for humans to program in advance. Recently, AI-based algorithms that can imitate human force skills have been actively explored as a solution. In particular, bilateral control-based imitation learning achieves human-level motion speeds with environmental adaptability, only requiring human demonstration and without programming. However, owing to hardware limitations, its grasping performance remains limited, and tasks that involves grasping various objects are yet to be achieved. Here, we developed a cross-structure hand to grasp various objects. We experimentally demonstrated that the integration of bilateral control-based imitation learning and the cross-structure hand is effective for grasping various objects and harnessing tools.

{{</citation>}}


## cs.LG (20)



### (136/197) Improving Unimodal Inference with Multimodal Transformers (Kateryna Chumachenko et al., 2023)

{{<citation>}}

Kateryna Chumachenko, Alexandros Iosifidis, Moncef Gabbouj. (2023)  
**Improving Unimodal Inference with Multimodal Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.10170v1)  

---


**ABSTRACT**  
This paper proposes an approach for improving performance of unimodal models with multimodal training. Our approach involves a multi-branch architecture that incorporates unimodal models with a multimodal transformer-based branch. By co-training these branches, the stronger multimodal branch can transfer its knowledge to the weaker unimodal branches through a multi-task objective, thereby improving the performance of the resulting unimodal models. We evaluate our approach on tasks of dynamic hand gesture recognition based on RGB and Depth, audiovisual emotion recognition based on speech and facial video, and audio-video-text based sentiment analysis. Our approach outperforms the conventionally trained unimodal counterparts. Interestingly, we also observe that optimization of the unimodal branches improves the multimodal branch, compared to a similar multimodal model trained from scratch.

{{</citation>}}


### (137/197) Tabular Few-Shot Generalization Across Heterogeneous Feature Spaces (Max Zhu et al., 2023)

{{<citation>}}

Max Zhu, Katarzyna Kobalczyk, Andrija Petrovic, Mladen Nikolic, Mihaela van der Schaar, Boris Delibasic, Petro Lio. (2023)  
**Tabular Few-Shot Generalization Across Heterogeneous Feature Spaces**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Few-Shot, Graph Attention Network  
[Paper Link](http://arxiv.org/abs/2311.10051v1)  

---


**ABSTRACT**  
Despite the prevalence of tabular datasets, few-shot learning remains under-explored within this domain. Existing few-shot methods are not directly applicable to tabular datasets due to varying column relationships, meanings, and permutational invariance. To address these challenges, we propose FLAT-a novel approach to tabular few-shot learning, encompassing knowledge sharing between datasets with heterogeneous feature spaces. Utilizing an encoder inspired by Dataset2Vec, FLAT learns low-dimensional embeddings of datasets and their individual columns, which facilitate knowledge transfer and generalization to previously unseen datasets. A decoder network parametrizes the predictive target network, implemented as a Graph Attention Network, to accommodate the heterogeneous nature of tabular datasets. Experiments on a diverse collection of 118 UCI datasets demonstrate FLAT's successful generalization to new tabular datasets and a considerable improvement over the baselines.

{{</citation>}}


### (138/197) Inherently Interpretable Time Series Classification via Multiple Instance Learning (Joseph Early et al., 2023)

{{<citation>}}

Joseph Early, Gavin KC Cheung, Kurt Cutajar, Hanting Xie, Jas Kandola, Niall Twomey. (2023)  
**Inherently Interpretable Time Series Classification via Multiple Instance Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.10049v1)  

---


**ABSTRACT**  
Conventional Time Series Classification (TSC) methods are often black boxes that obscure inherent interpretation of their decision-making processes. In this work, we leverage Multiple Instance Learning (MIL) to overcome this issue, and propose a new framework called MILLET: Multiple Instance Learning for Locally Explainable Time series classification. We apply MILLET to existing deep learning TSC models and show how they become inherently interpretable without compromising (and in some cases, even improving) predictive performance. We evaluate MILLET on 85 UCR TSC datasets and also present a novel synthetic dataset that is specially designed to facilitate interpretability evaluation. On these datasets, we show MILLET produces sparse explanations quickly that are of higher quality than other well-known interpretability methods. To the best of our knowledge, our work with MILLET, which is available on GitHub (https://github.com/JAEarly/MILTimeSeriesClassification), is the first to develop general MIL methods for TSC and apply them to an extensive variety of domains

{{</citation>}}


### (139/197) DeepEMD: A Transformer-based Fast Estimation of the Earth Mover's Distance (Atul Kumar Sinha et al., 2023)

{{<citation>}}

Atul Kumar Sinha, Francois Fleuret. (2023)  
**DeepEMD: A Transformer-based Fast Estimation of the Earth Mover's Distance**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.09998v1)  

---


**ABSTRACT**  
The Earth Mover's Distance (EMD) is the measure of choice between point clouds. However the computational cost to compute it makes it prohibitive as a training loss, and the standard approach is to use a surrogate such as the Chamfer distance. We propose an attention-based model to compute an accurate approximation of the EMD that can be used as a training loss for generative models. To get the necessary accurate estimation of the gradients we train our model to explicitly compute the matching between point clouds instead of EMD itself. We cast this new objective as the estimation of an attention matrix that approximates the ground truth matching matrix. Experiments show that this model provides an accurate estimate of the EMD and its gradient with a wall clock speed-up of more than two orders of magnitude with respect to the exact Hungarian matching algorithm and one order of magnitude with respect to the standard approximate Sinkhorn algorithm, allowing in particular to train a point cloud VAE with the EMD itself. Extensive evaluation show the remarkable behaviour of this model when operating out-of-distribution, a key requirement for a distance surrogate. Finally, the model generalizes very well to point clouds during inference several times larger than during training.

{{</citation>}}


### (140/197) Self-supervised learning of multi-omics embeddings in the low-label, high-data regime (Christian John Hurry et al., 2023)

{{<citation>}}

Christian John Hurry, Emma Slade. (2023)  
**Self-supervised learning of multi-omics embeddings in the low-label, high-data regime**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.09962v1)  

---


**ABSTRACT**  
Contrastive, self-supervised learning (SSL) is used to train a model that predicts cancer type from miRNA, mRNA or RPPA expression data. This model, a pretrained FT-Transformer, is shown to outperform XGBoost and CatBoost, standard benchmarks for tabular data, when labelled samples are scarce but the number of unlabelled samples is high. This is despite the fact that the datasets we use have $\mathcal{O}(10^{1})$ classes and $\mathcal{O}(10^{2})-\mathcal{O}(10^{4})$ features. After demonstrating the efficacy of our chosen method of self-supervised pretraining, we investigate SSL for multi-modal models. A late-fusion model is proposed, where each omics is passed through its own sub-network, the outputs of which are averaged and passed to the pretraining or downstream objective function. Multi-modal pretraining is shown to improve predictions from a single omics, and we argue that this is useful for datasets with many unlabelled multi-modal samples, but few labelled unimodal samples. Additionally, we show that pretraining each omics-specific module individually is highly effective. This enables the application of the proposed model in a variety of contexts where a large amount of unlabelled data is available from each omics, but only a few labelled samples.

{{</citation>}}


### (141/197) Hijacking Large Language Models via Adversarial In-Context Learning (Yao Qiang et al., 2023)

{{<citation>}}

Yao Qiang, Xiangyu Zhou, Dongxiao Zhu. (2023)  
**Hijacking Large Language Models via Adversarial In-Context Learning**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09948v1)  

---


**ABSTRACT**  
In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific tasks by utilizing labeled examples as demonstrations in the precondition prompts. Despite its promising performance, ICL suffers from instability with the choice and arrangement of examples. Additionally, crafted adversarial attacks pose a notable threat to the robustness of ICL. However, existing attacks are either easy to detect, rely on external models, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable attack for ICL, aiming to hijack LLMs to generate the targeted response. The proposed LLM hijacking attack leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demonstrations. Extensive experimental results on various tasks and datasets demonstrate the effectiveness of our LLM hijacking attack, resulting in a distracted attention towards adversarial tokens, consequently leading to the targeted unwanted outputs.

{{</citation>}}


### (142/197) Natural Disaster Analysis using Satellite Imagery and Social-Media Data for Emergency Response Situations (Sukeerthi Mandyam et al., 2023)

{{<citation>}}

Sukeerthi Mandyam, Shanmuga Priya MG, Shalini Suresh, Kavitha Srinivasan. (2023)  
**Natural Disaster Analysis using Satellite Imagery and Social-Media Data for Emergency Response Situations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.09947v1)  

---


**ABSTRACT**  
Disaster Management is one of the most promising research areas because of its significant economic, environmental and social repercussions. This research focuses on analyzing different types of data (pre and post satellite images and twitter data) related to disaster management for in-depth analysis of location-wise emergency requirements. This research has been divided into two stages, namely, satellite image analysis and twitter data analysis followed by integration using location. The first stage involves pre and post disaster satellite image analysis of the location using multi-class land cover segmentation technique based on U-Net architecture. The second stage focuses on mapping the region with essential information about the disaster situation and immediate requirements for relief operations. The severely affected regions are demarcated and twitter data is extracted using keywords respective to that location. The extraction of situational information from a large corpus of raw tweets adopts Content Word based Tweet Summarization (COWTS) technique. An integration of these modules using real-time location-based mapping and frequency analysis technique gathers multi-dimensional information in the advent of disaster occurrence such as the Kerala and Mississippi floods that were analyzed and validated as test cases. The novelty of this research lies in the application of segmented satellite images for disaster relief using highlighted land cover changes and integration of twitter data by mapping these region-specific filters for obtaining a complete overview of the disaster.

{{</citation>}}


### (143/197) A Framework for Monitoring and Retraining Language Models in Real-World Applications (Jaykumar Kasundra et al., 2023)

{{<citation>}}

Jaykumar Kasundra, Claudia Schulz, Melicaalsadat Mirsafian, Stavroula Skylaki. (2023)  
**A Framework for Monitoring and Retraining Language Models in Real-World Applications**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09930v2)  

---


**ABSTRACT**  
In the Machine Learning (ML) model development lifecycle, training candidate models using an offline holdout dataset and identifying the best model for the given task is only the first step. After the deployment of the selected model, continuous model monitoring and model retraining is required in many real-world applications. There are multiple reasons for retraining, including data or concept drift, which may be reflected on the model performance as monitored by an appropriate metric. Another motivation for retraining is the acquisition of increasing amounts of data over time, which may be used to retrain and improve the model performance even in the absence of drifts. We examine the impact of various retraining decision points on crucial factors, such as model performance and resource utilization, in the context of Multilabel Classification models. We explain our key decision points and propose a reference framework for designing an effective model retraining strategy.

{{</citation>}}


### (144/197) Safety Aware Autonomous Path Planning Using Model Predictive Reinforcement Learning for Inland Waterways (Astrid Vanneste et al., 2023)

{{<citation>}}

Astrid Vanneste, Simon Vanneste, Olivier Vasseur, Robin Janssens, Mattias Billast, Ali Anwar, Kevin Mets, Tom De Schepper, Siegfried Mercelis, Peter Hellinckx. (2023)  
**Safety Aware Autonomous Path Planning Using Model Predictive Reinforcement Learning for Inland Waterways**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.09878v1)  

---


**ABSTRACT**  
In recent years, interest in autonomous shipping in urban waterways has increased significantly due to the trend of keeping cars and trucks out of city centers. Classical approaches such as Frenet frame based planning and potential field navigation often require tuning of many configuration parameters and sometimes even require a different configuration depending on the situation. In this paper, we propose a novel path planning approach based on reinforcement learning called Model Predictive Reinforcement Learning (MPRL). MPRL calculates a series of waypoints for the vessel to follow. The environment is represented as an occupancy grid map, allowing us to deal with any shape of waterway and any number and shape of obstacles. We demonstrate our approach on two scenarios and compare the resulting path with path planning using a Frenet frame and path planning based on a proximal policy optimization (PPO) agent. Our results show that MPRL outperforms both baselines in both test scenarios. The PPO based approach was not able to reach the goal in either scenario while the Frenet frame approach failed in the scenario consisting of a corner with obstacles. MPRL was able to safely (collision free) navigate to the goal in both of the test scenarios.

{{</citation>}}


### (145/197) SurvTimeSurvival: Survival Analysis On The Patient With Multiple Visits/Records (Hung Le et al., 2023)

{{<citation>}}

Hung Le, Ong Eng-Jon, Bober Miroslaw. (2023)  
**SurvTimeSurvival: Survival Analysis On The Patient With Multiple Visits/Records**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NA, cs.LG, math-NA  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.09854v1)  

---


**ABSTRACT**  
The accurate prediction of survival times for patients with severe diseases remains a critical challenge despite recent advances in artificial intelligence. This study introduces "SurvTimeSurvival: Survival Analysis On Patients With Multiple Visits/Records", utilizing the Transformer model to not only handle the complexities of time-varying covariates but also covariates data. We also tackle the data sparsity issue common to survival analysis datasets by integrating synthetic data generation into the learning process of our model. We show that our method outperforms state-of-the-art deep learning approaches on both covariates and time-varying covariates datasets. Our approach aims not only to enhance the understanding of individual patient survival trajectories across various medical conditions, thereby improving prediction accuracy, but also to play a pivotal role in designing clinical trials and creating new treatments.

{{</citation>}}


### (146/197) Runtime Verification of Learning Properties for Reinforcement Learning Algorithms (Tommaso Mannucci et al., 2023)

{{<citation>}}

Tommaso Mannucci, Julio de Oliveira Filho. (2023)  
**Runtime Verification of Learning Properties for Reinforcement Learning Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.09811v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) algorithms interact with their environment in a trial-and-error fashion. Such interactions can be expensive, inefficient, and timely when learning on a physical system rather than in a simulation. This work develops new runtime verification techniques to predict when the learning phase has not met or will not meet qualitative and timely expectations. This paper presents three verification properties concerning the quality and timeliness of learning in RL algorithms. With each property, we propose design steps for monitoring and assessing the properties during the system's operation.

{{</citation>}}


### (147/197) GEO: Generative Engine Optimization (Pranjal Aggarwal et al., 2023)

{{<citation>}}

Pranjal Aggarwal, Vishvak Murahari, Tanmay Rajpurohit, Ashwin Kalyan, Karthik R Narasimhan, Ameet Deshpande. (2023)  
**GEO: Generative Engine Optimization**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.09735v1)  

---


**ABSTRACT**  
The advent of large language models (LLMs) has ushered in a new paradigm of search engines that use generative models to gather and summarize information to answer user queries. This emerging technology, which we formalize under the unified framework of Generative Engines (GEs), has the potential to generate accurate and personalized responses, and is rapidly replacing traditional search engines like Google and Bing. Generative Engines typically satisfy queries by synthesizing information from multiple sources and summarizing them with the help of LLMs. While this shift significantly improves \textit{user} utility and \textit{generative search engine} traffic, it results in a huge challenge for the third stakeholder -- website and content creators. Given the black-box and fast-moving nature of Generative Engines, content creators have little to no control over when and how their content is displayed. With generative engines here to stay, the right tools should be provided to ensure that creator economy is not severely disadvantaged. To address this, we introduce Generative Engine Optimization (GEO), a novel paradigm to aid content creators in improving the visibility of their content in Generative Engine responses through a black-box optimization framework for optimizing and defining visibility metrics. We facilitate systematic evaluation in this new paradigm by introducing GEO-bench, a benchmark of diverse user queries across multiple domains, coupled with sources required to answer these queries. Through rigorous evaluation, we show that GEO can boost visibility by up to 40\% in generative engine responses. Moreover, we show the efficacy of these strategies varies across domains, underscoring the need for domain-specific methods. Our work opens a new frontier in the field of information discovery systems, with profound implications for generative engines and content creators.

{{</citation>}}


### (148/197) Accommodating Missing Modalities in Time-Continuous Multimodal Emotion Recognition (Juan Vazquez-Rodriguez et al., 2023)

{{<citation>}}

Juan Vazquez-Rodriguez, Grégoire Lefebvre, Julien Cumin, James L. Crowley. (2023)  
**Accommodating Missing Modalities in Time-Continuous Multimodal Emotion Recognition**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Emotion Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2311.10119v1)  

---


**ABSTRACT**  
Decades of research indicate that emotion recognition is more effective when drawing information from multiple modalities. But what if some modalities are sometimes missing? To address this problem, we propose a novel Transformer-based architecture for recognizing valence and arousal in a time-continuous manner even with missing input modalities. We use a coupling of cross-attention and self-attention mechanisms to emphasize relationships between modalities during time and enhance the learning process on weak salient inputs. Experimental results on the Ulm-TSST dataset show that our model exhibits an improvement of the concordance correlation coefficient evaluation of 37% when predicting arousal values and 30% when predicting valence values, compared to a late-fusion baseline approach.

{{</citation>}}


### (149/197) Augmenting Unsupervised Reinforcement Learning with Self-Reference (Andrew Zhao et al., 2023)

{{<citation>}}

Andrew Zhao, Erle Zhu, Rui Lu, Matthieu Lin, Yong-Jin Liu, Gao Huang. (2023)  
**Augmenting Unsupervised Reinforcement Learning with Self-Reference**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.09692v1)  

---


**ABSTRACT**  
Humans possess the ability to draw on past experiences explicitly when learning new tasks and applying them accordingly. We believe this capacity for self-referencing is especially advantageous for reinforcement learning agents in the unsupervised pretrain-then-finetune setting. During pretraining, an agent's past experiences can be explicitly utilized to mitigate the nonstationarity of intrinsic rewards. In the finetuning phase, referencing historical trajectories prevents the unlearning of valuable exploratory behaviors. Motivated by these benefits, we propose the Self-Reference (SR) approach, an add-on module explicitly designed to leverage historical information and enhance agent performance within the pretrain-finetune paradigm. Our approach achieves state-of-the-art results in terms of Interquartile Mean (IQM) performance and Optimality Gap reduction on the Unsupervised Reinforcement Learning Benchmark for model-free methods, recording an 86% IQM and a 16% Optimality Gap. Additionally, it improves current algorithms by up to 17% IQM and reduces the Optimality Gap by 31%. Beyond performance enhancement, the Self-Reference add-on also increases sample efficiency, a crucial attribute for real-world applications.

{{</citation>}}


### (150/197) Robust Contrastive Learning With Theory Guarantee (Ngoc N. Tran et al., 2023)

{{<citation>}}

Ngoc N. Tran, Lam Tran, Hoang Phan, Anh Bui, Tung Pham, Toan Tran, Dinh Phung, Trung Le. (2023)  
**Robust Contrastive Learning With Theory Guarantee**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.09671v1)  

---


**ABSTRACT**  
Contrastive learning (CL) is a self-supervised training paradigm that allows us to extract meaningful features without any label information. A typical CL framework is divided into two phases, where it first tries to learn the features from unlabelled data, and then uses those features to train a linear classifier with the labeled data. While a fair amount of existing theoretical works have analyzed how the unsupervised loss in the first phase can support the supervised loss in the second phase, none has examined the connection between the unsupervised loss and the robust supervised loss, which can shed light on how to construct an effective unsupervised loss for the first phase of CL. To fill this gap, our work develops rigorous theories to dissect and identify which components in the unsupervised loss can help improve the robust supervised loss and conduct proper experiments to verify our findings.

{{</citation>}}


### (151/197) ICXML: An In-Context Learning Framework for Zero-Shot Extreme Multi-Label Classification (Yaxin Zhu et al., 2023)

{{<citation>}}

Yaxin Zhu, Hamed Zamani. (2023)  
**ICXML: An In-Context Learning Framework for Zero-Shot Extreme Multi-Label Classification**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.09649v1)  

---


**ABSTRACT**  
This paper focuses on the task of Extreme Multi-Label Classification (XMC) whose goal is to predict multiple labels for each instance from an extremely large label space. While existing research has primarily focused on fully supervised XMC, real-world scenarios often lack complete supervision signals, highlighting the importance of zero-shot settings. Given the large label space, utilizing in-context learning approaches is not trivial. We address this issue by introducing In-Context Extreme Multilabel Learning (ICXML), a two-stage framework that cuts down the search space by generating a set of candidate labels through incontext learning and then reranks them. Extensive experiments suggest that ICXML advances the state of the art on two diverse public benchmarks.

{{</citation>}}


### (152/197) GAIA: Delving into Gradient-based Attribution Abnormality for Out-of-distribution Detection (Jinggang Chen et al., 2023)

{{<citation>}}

Jinggang Chen, Junjie Li, Xiaoyang Qu, Jianzong Wang, Jiguang Wan, Jing Xiao. (2023)  
**GAIA: Delving into Gradient-based Attribution Abnormality for Out-of-distribution Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.09620v1)  

---


**ABSTRACT**  
Detecting out-of-distribution (OOD) examples is crucial to guarantee the reliability and safety of deep neural networks in real-world settings. In this paper, we offer an innovative perspective on quantifying the disparities between in-distribution (ID) and OOD data -- analyzing the uncertainty that arises when models attempt to explain their predictive decisions. This perspective is motivated by our observation that gradient-based attribution methods encounter challenges in assigning feature importance to OOD data, thereby yielding divergent explanation patterns. Consequently, we investigate how attribution gradients lead to uncertain explanation outcomes and introduce two forms of abnormalities for OOD detection: the zero-deflation abnormality and the channel-wise average abnormality. We then propose GAIA, a simple and effective approach that incorporates Gradient Abnormality Inspection and Aggregation. The effectiveness of GAIA is validated on both commonly utilized (CIFAR) and large-scale (ImageNet-1k) benchmarks. Specifically, GAIA reduces the average FPR95 by 23.10% on CIFAR10 and by 45.41% on CIFAR100 compared to advanced post-hoc methods.

{{</citation>}}


### (153/197) A Knowledge Distillation Approach for Sepsis Outcome Prediction from Multivariate Clinical Time Series (Anna Wong et al., 2023)

{{<citation>}}

Anna Wong, Shu Ge, Nassim Oufattole, Adam Dejl, Megan Su, Ardavan Saeedi, Li-wei H. Lehman. (2023)  
**A Knowledge Distillation Approach for Sepsis Outcome Prediction from Multivariate Clinical Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Clinical, Knowledge Distillation, LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2311.09566v1)  

---


**ABSTRACT**  
Sepsis is a life-threatening condition triggered by an extreme infection response. Our objective is to forecast sepsis patient outcomes using their medical history and treatments, while learning interpretable state representations to assess patients' risks in developing various adverse outcomes. While neural networks excel in outcome prediction, their limited interpretability remains a key issue. In this work, we use knowledge distillation via constrained variational inference to distill the knowledge of a powerful "teacher" neural network model with high predictive power to train a "student" latent variable model to learn interpretable hidden state representations to achieve high predictive performance for sepsis outcome prediction. Using real-world data from the MIMIC-IV database, we trained an LSTM as the "teacher" model to predict mortality for sepsis patients, given information about their recent history of vital signs, lab values and treatments. For our student model, we use an autoregressive hidden Markov model (AR-HMM) to learn interpretable hidden states from patients' clinical time series, and use the posterior distribution of the learned state representations to predict various downstream outcomes, including hospital mortality, pulmonary edema, need for diuretics, dialysis, and mechanical ventilation. Our results show that our approach successfully incorporates the constraint to achieve high predictive power similar to the teacher model, while maintaining the generative performance.

{{</citation>}}


### (154/197) A Speed Odyssey for Deployable Quantization of LLMs (Qingyuan Li et al., 2023)

{{<citation>}}

Qingyuan Li, Ran Meng, Yiduo Li, Bo Zhang, Liang Li, Yifan Lu, Xiangxiang Chu, Yerui Sun, Yuchen Xie. (2023)  
**A Speed Odyssey for Deployable Quantization of LLMs**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.09550v1)  

---


**ABSTRACT**  
The large language model era urges faster and less costly inference. Prior model compression works on LLMs tend to undertake a software-centric approach primarily focused on the simulated quantization performance. By neglecting the feasibility of deployment, these approaches are typically disabled in real practice. They used to drastically push down the quantization bit range for a reduced computation which might not be supported by the mainstream hardware, or involve sophisticated algorithms that introduce extra computation or memory access overhead. We argue that pursuing a hardware-centric approach in the construction of quantization algorithms is crucial. In this regard, we are driven to build our compression method on top of hardware awareness, eliminating impractical algorithm choices while maximizing the benefit of hardware acceleration. Our method, OdysseyLLM, comes with a novel W4A8 kernel implementation called FastGEMM and a combined recipe of quantization strategies. Extensive experiments manifest the superiority of our W4A8 method which brings the actual speed boosting up to \textbf{4$\times$} compared to Hugging Face FP16 inference and \textbf{2.23$\times$} vs. the state-of-the-art inference engine TensorRT-LLM in FP16, and \textbf{1.45$\times$} vs. TensorRT-LLM in INT8, yet without substantially harming the performance.

{{</citation>}}


### (155/197) Investigating the Impact of Weight Sharing Decisions on Knowledge Transfer in Continual Learning (Josh Andle et al., 2023)

{{<citation>}}

Josh Andle, Ali Payani, Salimeh Yasaei-Sekeh. (2023)  
**Investigating the Impact of Weight Sharing Decisions on Knowledge Transfer in Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.09506v1)  

---


**ABSTRACT**  
Continual Learning (CL) has generated attention as a method of avoiding Catastrophic Forgetting (CF) in the sequential training of neural networks, improving network efficiency and adaptability to different tasks. Additionally, CL serves as an ideal setting for studying network behavior and Forward Knowledge Transfer (FKT) between tasks. Pruning methods for CL train subnetworks to handle the sequential tasks which allows us to take a structured approach to investigating FKT. Sharing prior subnetworks' weights leverages past knowledge for the current task through FKT. Understanding which weights to share is important as sharing all weights can yield sub-optimal accuracy. This paper investigates how different sharing decisions affect the FKT between tasks. Through this lens we demonstrate how task complexity and similarity influence the optimal weight sharing decisions, giving insights into the relationships between tasks and helping inform decision making in similar CL methods. We implement three sequential datasets designed to emphasize variation in task complexity and similarity, reporting results for both ResNet-18 and VGG-16. By sharing in accordance with the decisions supported by our findings, we show that we can improve task accuracy compared to other sharing decisions.

{{</citation>}}


## cs.CV (20)



### (156/197) Traffic Video Object Detection using Motion Prior (Lihao Liu et al., 2023)

{{<citation>}}

Lihao Liu, Yanqi Cheng, Dongdong Chen, Jing He, Pietro Liò, Carola-Bibiane Schönlieb, Angelica I Aviles-Rivero. (2023)  
**Traffic Video Object Detection using Motion Prior**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.10092v1)  

---


**ABSTRACT**  
Traffic videos inherently differ from generic videos in their stationary camera setup, thus providing a strong motion prior where objects often move in a specific direction over a short time interval. Existing works predominantly employ generic video object detection framework for traffic video object detection, which yield certain advantages such as broad applicability and robustness to diverse scenarios. However, they fail to harness the strength of motion prior to enhance detection accuracy. In this work, we propose two innovative methods to exploit the motion prior and boost the performance of both fully-supervised and semi-supervised traffic video object detection. Firstly, we introduce a new self-attention module that leverages the motion prior to guide temporal information integration in the fully-supervised setting. Secondly, we utilise the motion prior to develop a pseudo-labelling mechanism to eliminate noisy pseudo labels for the semi-supervised setting. Both of our motion-prior-centred methods consistently demonstrates superior performance, outperforming existing state-of-the-art approaches by a margin of 2% in terms of mAP.

{{</citation>}}


### (157/197) Emu Edit: Precise Image Editing via Recognition and Generation Tasks (Shelly Sheynin et al., 2023)

{{<citation>}}

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, Yaniv Taigman. (2023)  
**Emu Edit: Precise Image Editing via Recognition and Generation Tasks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2311.10089v1)  

---


**ABSTRACT**  
Instruction-based image editing holds immense potential for a variety of applications, as it enables users to perform any editing operation using a natural language instruction. However, current models in this domain often struggle with accurately executing user instructions. We present Emu Edit, a multi-task image editing model which sets state-of-the-art results in instruction-based image editing. To develop Emu Edit we train it to multi-task across an unprecedented range of tasks, such as region-based editing, free-form editing, and Computer Vision tasks, all of which are formulated as generative tasks. Additionally, to enhance Emu Edit's multi-task learning abilities, we provide it with learned task embeddings which guide the generation process towards the correct edit type. Both these elements are essential for Emu Edit's outstanding performance. Furthermore, we show that Emu Edit can generalize to new tasks, such as image inpainting, super-resolution, and compositions of editing tasks, with just a few labeled examples. This capability offers a significant advantage in scenarios where high-quality samples are scarce. Lastly, to facilitate a more rigorous and informed assessment of instructable image editing models, we release a new challenging and versatile benchmark that includes seven different image editing tasks.

{{</citation>}}


### (158/197) DRESS: Instructing Large Vision-Language Models to Align and Interact with Humans via Natural Language Feedback (Yangyi Chen et al., 2023)

{{<citation>}}

Yangyi Chen, Karan Sikka, Michael Cogswell, Heng Ji, Ajay Divakaran. (2023)  
**DRESS: Instructing Large Vision-Language Models to Align and Interact with Humans via Natural Language Feedback**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.10081v1)  

---


**ABSTRACT**  
We present DRESS, a large vision language model (LVLM) that innovatively exploits Natural Language feedback (NLF) from Large Language Models to enhance its alignment and interactions by addressing two key limitations in the state-of-the-art LVLMs. First, prior LVLMs generally rely only on the instruction finetuning stage to enhance alignment with human preferences. Without incorporating extra feedback, they are still prone to generate unhelpful, hallucinated, or harmful responses. Second, while the visual instruction tuning data is generally structured in a multi-turn dialogue format, the connections and dependencies among consecutive conversational turns are weak. This reduces the capacity for effective multi-turn interactions. To tackle these, we propose a novel categorization of the NLF into two key types: critique and refinement. The critique NLF identifies the strengths and weaknesses of the responses and is used to align the LVLMs with human preferences. The refinement NLF offers concrete suggestions for improvement and is adopted to improve the interaction ability of the LVLMs-- which focuses on LVLMs' ability to refine responses by incorporating feedback in multi-turn interactions. To address the non-differentiable nature of NLF, we generalize conditional reinforcement learning for training. Our experimental results demonstrate that DRESS can generate more helpful (9.76%), honest (11.52%), and harmless (21.03%) responses, and more effectively learn from feedback during multi-turn interactions compared to SOTA LVMLs.

{{</citation>}}


### (159/197) Match and Locate: low-frequency monocular odometry based on deep feature matching (Stepan Konev et al., 2023)

{{<citation>}}

Stepan Konev, Yuriy Biktairov. (2023)  
**Match and Locate: low-frequency monocular odometry based on deep feature matching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.10034v1)  

---


**ABSTRACT**  
Accurate and robust pose estimation plays a crucial role in many robotic systems. Popular algorithms for pose estimation typically rely on high-fidelity and high-frequency signals from various sensors. Inclusion of these sensors makes the system less affordable and much more complicated. In this work we introduce a novel approach for the robotic odometry which only requires a single camera and, importantly, can produce reliable estimates given even extremely low-frequency signal of around one frame per second. The approach is based on matching image features between the consecutive frames of the video stream using deep feature matching models. The resulting coarse estimate is then adjusted by a convolutional neural network, which is also responsible for estimating the scale of the transition, otherwise irretrievable using only the feature matching information. We evaluate the performance of the approach in the AISG-SLA Visual Localisation Challenge and find that while being computationally efficient and easy to implement our method shows competitive results with only around $3^{\circ}$ of orientation estimation error and $2m$ of translation estimation error taking the third place in the challenge.

{{</citation>}}


### (160/197) SQLNet: Scale-Modulated Query and Localization Network for Few-Shot Class-Agnostic Counting (Hefeng Wu et al., 2023)

{{<citation>}}

Hefeng Wu, Yandong Chen, Lingbo Liu, Tianshui Chen, Keze Wang, Liang Lin. (2023)  
**SQLNet: Scale-Modulated Query and Localization Network for Few-Shot Class-Agnostic Counting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.10011v1)  

---


**ABSTRACT**  
The class-agnostic counting (CAC) task has recently been proposed to solve the problem of counting all objects of an arbitrary class with several exemplars given in the input image. To address this challenging task, existing leading methods all resort to density map regression, which renders them impractical for downstream tasks that require object locations and restricts their ability to well explore the scale information of exemplars for supervision. To address the limitations, we propose a novel localization-based CAC approach, termed Scale-modulated Query and Localization Network (SQLNet). It fully explores the scales of exemplars in both the query and localization stages and achieves effective counting by accurately locating each object and predicting its approximate size. Specifically, during the query stage, rich discriminative representations of the target class are acquired by the Hierarchical Exemplars Collaborative Enhancement (HECE) module from the few exemplars through multi-scale exemplar cooperation with equifrequent size prompt embedding. These representations are then fed into the Exemplars-Unified Query Correlation (EUQC) module to interact with the query features in a unified manner and produce the correlated query tensor. In the localization stage, the Scale-aware Multi-head Localization (SAML) module utilizes the query tensor to predict the confidence, location, and size of each potential object. Moreover, a scale-aware localization loss is introduced, which exploits flexible location associations and exemplar scales for supervision to optimize the model performance. Extensive experiments demonstrate that SQLNet outperforms state-of-the-art methods on popular CAC benchmarks, achieving excellent performance not only in counting accuracy but also in localization and bounding box generation. Our codes will be available at https://github.com/HCPLab-SYSU/SQLNet

{{</citation>}}


### (161/197) TransFusion -- A Transparency-Based Diffusion Model for Anomaly Detection (Matic Fučka et al., 2023)

{{<citation>}}

Matic Fučka, Vitjan Zavrtanik, Danijel Skočaj. (2023)  
**TransFusion -- A Transparency-Based Diffusion Model for Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.09999v1)  

---


**ABSTRACT**  
Surface anomaly detection is a vital component in manufacturing inspection. Reconstructive anomaly detection methods restore the normal appearance of an object, ideally modifying only the anomalous regions. Due to the limitations of commonly used reconstruction architectures, the produced reconstructions are often poor and either still contain anomalies or lack details in anomaly-free regions. Recent reconstructive methods adopt diffusion models, however with the standard diffusion process the problems are not adequately addressed. We propose a novel transparency-based diffusion process, where the transparency of anomalous regions is progressively increased, restoring their normal appearance accurately and maintaining the appearance of anomaly-free regions without loss of detail. We propose TRANSparency DifFUSION (TransFusion), a discriminative anomaly detection method that implements the proposed diffusion process, enabling accurate downstream anomaly detection. TransFusion achieves state-of-the-art performance on both the VisA and the MVTec AD datasets, with an image-level AUROC of 98.5% and 99.2%, respectively.

{{</citation>}}


### (162/197) From Pretext to Purpose: Batch-Adaptive Self-Supervised Learning (Jiansong Zhang et al., 2023)

{{<citation>}}

Jiansong Zhang, Peizhong Liu. (2023)  
**From Pretext to Purpose: Batch-Adaptive Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, ImageNet, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.09974v1)  

---


**ABSTRACT**  
In recent years, self-supervised contrastive learning has emerged as a distinguished paradigm in the artificial intelligence landscape. It facilitates unsupervised feature learning through contrastive delineations at the instance level. However, crafting an effective self-supervised paradigm remains a pivotal challenge within this field. This paper delves into two crucial factors impacting self-supervised contrastive learning-bach size and pretext tasks, and from a data processing standpoint, proposes an adaptive technique of batch fusion. The proposed method, via dimensionality reduction and reconstruction of batch data, enables formerly isolated individual data to partake in intra-batch communication through the Embedding Layer. Moreover, it adaptively amplifies the self-supervised feature encoding capability as the training progresses. We conducted a linear classification test of this method based on the classic contrastive learning framework on ImageNet-1k. The empirical findings illustrate that our approach achieves state-of-the-art performance under equitable comparisons. Benefiting from its "plug-and-play" characteristics, we further explored other contrastive learning methods. On the ImageNet-100, compared to the original performance, the top1 has seen a maximum increase of 1.25%. We suggest that the proposed method may contribute to the advancement of data-driven self-supervised learning research, bringing a fresh perspective to this community.

{{</citation>}}


### (163/197) I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization (Yunshan Zhong et al., 2023)

{{<citation>}}

Yunshan Zhong, Jiawei Hu, Mingbao Lin, Mengzhao Chen, Rongrong Ji. (2023)  
**I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.10126v1)  

---


**ABSTRACT**  
Albeit the scalable performance of vision transformers (ViTs), the dense computational costs (training & inference) undermine their position in industrial applications. Post-training quantization (PTQ), tuning ViTs with a tiny dataset and running in a low-bit format, well addresses the cost issue but unluckily bears more performance drops in lower-bit cases. In this paper, we introduce I&S-ViT, a novel method that regulates the PTQ of ViTs in an inclusive and stable fashion. I&S-ViT first identifies two issues in the PTQ of ViTs: (1) Quantization inefficiency in the prevalent log2 quantizer for post-Softmax activations; (2) Rugged and magnified loss landscape in coarse-grained quantization granularity for post-LayerNorm activations. Then, I&S-ViT addresses these issues by introducing: (1) A novel shift-uniform-log2 quantizer (SULQ) that incorporates a shift mechanism followed by uniform quantization to achieve both an inclusive domain representation and accurate distribution approximation; (2) A three-stage smooth optimization strategy (SOS) that amalgamates the strengths of channel-wise and layer-wise quantization to enable stable learning. Comprehensive evaluations across diverse vision tasks validate I&S-ViT' superiority over existing PTQ of ViTs methods, particularly in low-bit scenarios. For instance, I&S-ViT elevates the performance of 3-bit ViT-B by an impressive 50.68%.

{{</citation>}}


### (164/197) UnifiedVisionGPT: Streamlining Vision-Oriented AI through Generalized Multimodal Framework (Chris Kelly et al., 2023)

{{<citation>}}

Chris Kelly, Luhui Hu, Cindy Yang, Yu Tian, Deshun Yang, Bang Yang, Zaoshan Huang, Zihao Li, Yuexian Zou. (2023)  
**UnifiedVisionGPT: Streamlining Vision-Oriented AI through Generalized Multimodal Framework**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.10125v1)  

---


**ABSTRACT**  
In the current landscape of artificial intelligence, foundation models serve as the bedrock for advancements in both language and vision domains. OpenAI GPT-4 has emerged as the pinnacle in large language models (LLMs), while the computer vision (CV) domain boasts a plethora of state-of-the-art (SOTA) models such as Meta's SAM and DINO, and YOLOS. However, the financial and computational burdens of training new models from scratch remain a significant barrier to progress. In response to this challenge, we introduce UnifiedVisionGPT, a novel framework designed to consolidate and automate the integration of SOTA vision models, thereby facilitating the development of vision-oriented AI. UnifiedVisionGPT distinguishes itself through four key features: (1) provides a versatile multimodal framework adaptable to a wide range of applications, building upon the strengths of multimodal foundation models; (2) seamlessly integrates various SOTA vision models to create a comprehensive multimodal platform, capitalizing on the best components of each model; (3) prioritizes vision-oriented AI, ensuring a more rapid progression in the CV domain compared to the current trajectory of LLMs; and (4) introduces automation in the selection of SOTA vision models, generating optimal results based on diverse multimodal inputs such as text prompts and images. This paper outlines the architecture and capabilities of UnifiedVisionGPT, demonstrating its potential to revolutionize the field of computer vision through enhanced efficiency, versatility, generalization, and performance. Our implementation, along with the unified multimodal framework and comprehensive dataset, is made publicly available at https://github.com/LHBuilder/SA-Segment-Anything.

{{</citation>}}


### (165/197) Overcoming Data Scarcity in Biomedical Imaging with a Foundational Multi-Task Model (Raphael Schäfer et al., 2023)

{{<citation>}}

Raphael Schäfer, Till Nicke, Henning Höfener, Annkristin Lange, Dorit Merhof, Friedrich Feuerhake, Volkmar Schulz, Johannes Lotz, Fabian Kiessling. (2023)  
**Overcoming Data Scarcity in Biomedical Imaging with a Foundational Multi-Task Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.09847v1)  

---


**ABSTRACT**  
Foundational models, pretrained on a large scale, have demonstrated substantial success across non-medical domains. However, training these models typically requires large, comprehensive datasets, which contrasts with the smaller and more heterogeneous datasets common in biomedical imaging. Here, we propose a multi-task learning strategy that decouples the number of training tasks from memory requirements. We trained a Universal bioMedical PreTrained model (UMedPT) on a multi-task database including tomographic, microscopic, and X-ray images, with various labelling strategies such as classification, segmentation, and object detection. The UMedPT foundational model outperformed ImageNet pretraining and the previous state-of-the-art models. For tasks related to the pretraining database, it maintained its performance with only 1% of the original training data and without fine-tuning. For out-of-domain tasks it required not more than 50% of the original training data. In an external independent validation imaging features extracted using UMedPT proved to be a new standard for cross-center transferability.

{{</citation>}}


### (166/197) Neural-Logic Human-Object Interaction Detection (Liulei Li et al., 2023)

{{<citation>}}

Liulei Li, Jianan Wei, Wenguan Wang, Yi Yang. (2023)  
**Neural-Logic Human-Object Interaction Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.09817v1)  

---


**ABSTRACT**  
The interaction decoder utilized in prevalent Transformer-based HOI detectors typically accepts pre-composed human-object pairs as inputs. Though achieving remarkable performance, such paradigm lacks feasibility and cannot explore novel combinations over entities during decoding. We present L OGIC HOI, a new HOI detector that leverages neural-logic reasoning and Transformer to infer feasible interactions between entities. Specifically, we modify the self-attention mechanism in vanilla Transformer, enabling it to reason over the <human, action, object> triplet and constitute novel interactions. Meanwhile, such reasoning process is guided by two crucial properties for understanding HOI: affordances (the potential actions an object can facilitate) and proxemics (the spatial relations between humans and objects). We formulate these two properties in first-order logic and ground them into continuous space to constrain the learning process of our approach, leading to improved performance and zero-shot generalization capabilities. We evaluate L OGIC HOI on V-COCO and HICO-DET under both normal and zero-shot setups, achieving significant improvements over existing methods.

{{</citation>}}


### (167/197) Certified Control for Train Sign Classification (Jan Roßbach et al., 2023)

{{<citation>}}

Jan Roßbach, Michael Leuschel. (2023)  
**Certified Control for Train Sign Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09778v1)  

---


**ABSTRACT**  
There is considerable industrial interest in integrating AI techniques into railway systems, notably for fully autonomous train systems. The KI-LOK research project is involved in developing new methods for certifying such AI-based systems. Here we explore the utility of a certified control architecture for a runtime monitor that prevents false positive detection of traffic signs in an AI-based perception system. The monitor uses classical computer vision algorithms to check if the signs -- detected by an AI object detection model -- fit predefined specifications. We provide such specifications for some critical signs and integrate a Python prototype of the monitor with a popular object detection model to measure relevant performance metrics on generated data. Our initial results are promising, achieving considerable precision gains with only minor recall reduction; however, further investigation into generalization possibilities will be necessary.

{{</citation>}}


### (168/197) Video-LLaVA: Learning United Visual Representation by Alignment Before Projection (Bin Lin et al., 2023)

{{<citation>}}

Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, Li Yuan. (2023)  
**Video-LLaVA: Learning United Visual Representation by Alignment Before Projection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.10122v1)  

---


**ABSTRACT**  
The Large Vision-Language Model (LVLM) has enhanced the performance of various downstream tasks in visual-language understanding. Most existing approaches encode images and videos into separate feature spaces, which are then fed as inputs to large language models. However, due to the lack of unified tokenization for images and videos, namely misalignment before projection, it becomes challenging for a Large Language Model (LLM) to learn multi-modal interactions from several poor projection layers. In this work, we unify visual representation into the language feature space to advance the foundational LLM towards a unified LVLM. As a result, we establish a simple but robust LVLM baseline, Video-LLaVA, which learns from a mixed dataset of images and videos, mutually enhancing each other. Video-LLaVA achieves superior performances on a broad range of 9 image benchmarks across 5 image question-answering datasets and 4 image benchmark toolkits. Additionally, our Video-LLaVA also outperforms Video-ChatGPT by 5.8%, 9.9%, 18.6%, and 10.1% on MSRVTT, MSVD, TGIF, and ActivityNet, respectively. Notably, extensive experiments demonstrate that Video-LLaVA mutually benefits images and videos within a unified visual representation, outperforming models designed specifically for images or videos.

{{</citation>}}


### (169/197) DIFFNAT: Improving Diffusion Image Quality Using Natural Image Statistics (Aniket Roy et al., 2023)

{{<citation>}}

Aniket Roy, Maiterya Suin, Anshul Shah, Ketul Shah, Jiang Liu, Rama Chellappa. (2023)  
**DIFFNAT: Improving Diffusion Image Quality Using Natural Image Statistics**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09753v1)  

---


**ABSTRACT**  
Diffusion models have advanced generative AI significantly in terms of editing and creating naturalistic images. However, efficiently improving generated image quality is still of paramount interest. In this context, we propose a generic "naturalness" preserving loss function, viz., kurtosis concentration (KC) loss, which can be readily applied to any standard diffusion model pipeline to elevate the image quality. Our motivation stems from the projected kurtosis concentration property of natural images, which states that natural images have nearly constant kurtosis values across different band-pass versions of the image. To retain the "naturalness" of the generated images, we enforce reducing the gap between the highest and lowest kurtosis values across the band-pass versions (e.g., Discrete Wavelet Transform (DWT)) of images. Note that our approach does not require any additional guidance like classifier or classifier-free guidance to improve the image quality. We validate the proposed approach for three diverse tasks, viz., (1) personalized few-shot finetuning using text guidance, (2) unconditional image generation, and (3) image super-resolution. Integrating the proposed KC loss has improved the perceptual quality across all these tasks in terms of both FID, MUSIQ score, and user evaluation.

{{</citation>}}


### (170/197) Redefining the Laparoscopic Spatial Sense: AI-based Intra- and Postoperative Measurement from Stereoimages (Leopold Müller et al., 2023)

{{<citation>}}

Leopold Müller, Patrick Hemmer, Moritz Queisner, Igor Sauer, Simeon Allmendinger, Johannes Jakubik, Michael Vössing, Niklas Kühl. (2023)  
**Redefining the Laparoscopic Spatial Sense: AI-based Intra- and Postoperative Measurement from Stereoimages**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09744v1)  

---


**ABSTRACT**  
A significant challenge in image-guided surgery is the accurate measurement task of relevant structures such as vessel segments, resection margins, or bowel lengths. While this task is an essential component of many surgeries, it involves substantial human effort and is prone to inaccuracies. In this paper, we develop a novel human-AI-based method for laparoscopic measurements utilizing stereo vision that has been guided by practicing surgeons. Based on a holistic qualitative requirements analysis, this work proposes a comprehensive measurement method, which comprises state-of-the-art machine learning architectures, such as RAFT-Stereo and YOLOv8. The developed method is assessed in various realistic experimental evaluation environments. Our results outline the potential of our method achieving high accuracies in distance measurements with errors below 1 mm. Furthermore, on-surface measurements demonstrate robustness when applied in challenging environments with textureless regions. Overall, by addressing the inherent challenges of image-guided surgery, we lay the foundation for a more robust and accurate solution for intra- and postoperative measurements, enabling more precise, safe, and efficient surgical procedures.

{{</citation>}}


### (171/197) MS-Former: Memory-Supported Transformer for Weakly Supervised Change Detection with Patch-Level Annotations (Zhenglai Li et al., 2023)

{{<citation>}}

Zhenglai Li, Chang Tang, Xinwang Liu, Changdong Li, Xianju Li, Wei Zhang. (2023)  
**MS-Former: Memory-Supported Transformer for Weakly Supervised Change Detection with Patch-Level Annotations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.09726v1)  

---


**ABSTRACT**  
Fully supervised change detection methods have achieved significant advancements in performance, yet they depend severely on acquiring costly pixel-level labels. Considering that the patch-level annotations also contain abundant information corresponding to both changed and unchanged objects in bi-temporal images, an intuitive solution is to segment the changes with patch-level annotations. How to capture the semantic variations associated with the changed and unchanged regions from the patch-level annotations to obtain promising change results is the critical challenge for the weakly supervised change detection task. In this paper, we propose a memory-supported transformer (MS-Former), a novel framework consisting of a bi-directional attention block (BAB) and a patch-level supervision scheme (PSS) tailored for weakly supervised change detection with patch-level annotations. More specifically, the BAM captures contexts associated with the changed and unchanged regions from the temporal difference features to construct informative prototypes stored in the memory bank. On the other hand, the BAM extracts useful information from the prototypes as supplementary contexts to enhance the temporal difference features, thereby better distinguishing changed and unchanged regions. After that, the PSS guides the network learning valuable knowledge from the patch-level annotations, thus further elevating the performance. Experimental results on three benchmark datasets demonstrate the effectiveness of our proposed method in the change detection task. The demo code for our work will be publicly available at \url{https://github.com/guanyuezhen/MS-Former}.

{{</citation>}}


### (172/197) Trustworthy Large Models in Vision: A Survey (Ziyan Guo et al., 2023)

{{<citation>}}

Ziyan Guo, Jun Liu. (2023)  
**Trustworthy Large Models in Vision: A Survey**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.09680v2)  

---


**ABSTRACT**  
The rapid progress of Large Models (LMs) has recently revolutionized various fields of deep learning with remarkable grades, ranging from Natural Language Processing (NLP) to Computer Vision (CV). However, LMs are increasingly challenged and criticized by academia and industry due to their powerful performance but untrustworthy behavior, which urgently needs to be alleviated by reliable methods. Despite the abundance of literature on trustworthy LMs in NLP, a systematic survey specifically delving into the trustworthiness of LMs in CV remains absent. In order to mitigate this gap, we summarize four relevant concerns that obstruct the trustworthy usage in vision of LMs in this survey, including 1) human misuse, 2) vulnerability, 3) inherent issue and 4) interpretability. By highlighting corresponding challenge, countermeasures, and discussion in each topic, we hope this survey will facilitate readers' understanding of this field, promote alignment of LMs with human expectations and enable trustworthy LMs to serve as welfare rather than disaster for human society.

{{</citation>}}


### (173/197) DECDM: Document Enhancement using Cycle-Consistent Diffusion Models (Jiaxin Zhang et al., 2023)

{{<citation>}}

Jiaxin Zhang, Joy Rimchala, Lalla Mouatadid, Kamalika Das, Sricharan Kumar. (2023)  
**DECDM: Document Enhancement using Cycle-Consistent Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2311.09625v1)  

---


**ABSTRACT**  
The performance of optical character recognition (OCR) heavily relies on document image quality, which is crucial for automatic document processing and document intelligence. However, most existing document enhancement methods require supervised data pairs, which raises concerns about data separation and privacy protection, and makes it challenging to adapt these methods to new domain pairs. To address these issues, we propose DECDM, an end-to-end document-level image translation method inspired by recent advances in diffusion models. Our method overcomes the limitations of paired training by independently training the source (noisy input) and target (clean output) models, making it possible to apply domain-specific diffusion models to other pairs. DECDM trains on one dataset at a time, eliminating the need to scan both datasets concurrently, and effectively preserving data privacy from the source or target domain. We also introduce simple data augmentation strategies to improve character-glyph conservation during translation. We compare DECDM with state-of-the-art methods on multiple synthetic data and benchmark datasets, such as document denoising and {\color{black}shadow} removal, and demonstrate the superiority of performance quantitatively and qualitatively.

{{</citation>}}


### (174/197) Wildfire Smoke Detection with Cross Contrast Patch Embedding (Chong Wang et al., 2023)

{{<citation>}}

Chong Wang, Cheng Xu, Adeel Akram, Zhilin Shan, Qixing Zhang. (2023)  
**Wildfire Smoke Detection with Cross Contrast Patch Embedding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2311.10116v1)  

---


**ABSTRACT**  
The Transformer-based deep networks have increasingly shown significant advantages over CNNs. Some existing work has applied it in the field of wildfire recognition or detection. However, we observed that the vanilla Transformer is not friendly for extracting smoke features. Because low-level information such as color, transparency and texture is very important for smoke recognition, and transformer pays more attention to the semantic relevance between middle- or high-level features, and is not sensitive to the subtle changes of low-level features along the space. To solve this problem, we propose the Cross Contrast Patch Embedding(CCPE) module based on the Swin Transformer, which uses the multi-scales spatial frequency contrast information in both vertical and horizontal directions to improve the discrimination of the network on the underlying details. The fuzzy boundary of smoke makes the positive and negative label assignment for instances in a dilemma, which is another challenge for wildfires detection. To solve this problem, a Separable Negative Sampling Mechanism(SNSM) is proposed. By using two different negative instance sampling strategies on positive images and negative images respectively, the problem of supervision signal confusion caused by label diversity in the process of network training is alleviated. This paper also releases the RealFire Test, the largest real wildfire test set so far, to evaluate the proposed method and promote future research. It contains 50,535 images from 3,649 video clips. The proposed method has been extensively tested and evaluated on RealFire Test dataset, and has a significant performance improvement compared with the baseline detection models.

{{</citation>}}


### (175/197) Efficient End-to-End Visual Document Understanding with Rationale Distillation (Wang Zhu et al., 2023)

{{<citation>}}

Wang Zhu, Alekh Agarwal, Mandar Joshi, Robin Jia, Jesse Thomason, Kristina Toutanova. (2023)  
**Efficient End-to-End Visual Document Understanding with Rationale Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2311.09612v1)  

---


**ABSTRACT**  
Understanding visually situated language requires recognizing text and visual elements, and interpreting complex layouts. State-of-the-art methods commonly use specialized pre-processing tools, such as optical character recognition (OCR) systems, that map document image inputs to extracted information in the space of textual tokens, and sometimes also employ large language models (LLMs) to reason in text token space. However, the gains from external tools and LLMs come at the cost of increased computational and engineering complexity. In this paper, we ask whether small pretrained image-to-text models can learn selective text or layout recognition and reasoning as an intermediate inference step in an end-to-end model for pixel-level visual language understanding. We incorporate the outputs of such OCR tools, LLMs, and larger multimodal models as intermediate ``rationales'' on training data, and train a small student model to predict both rationales and answers for input questions based on those training examples. A student model based on Pix2Struct (282M parameters) achieves consistent improvements on three visual document understanding benchmarks representing infographics, scanned documents, and figures, with improvements of more than 4\% absolute over a comparable Pix2Struct model that predicts answers directly.

{{</citation>}}


## stat.ML (1)



### (176/197) Online Optimization for Network Resource Allocation and Comparison with Reinforcement Learning Techniques (Ahmed Sid-Ali et al., 2023)

{{<citation>}}

Ahmed Sid-Ali, Ioannis Lambadaris, Yiqiang Q. Zhao, Gennady Shaikhet, Amirhossein Asgharnia. (2023)  
**Online Optimization for Network Resource Allocation and Comparison with Reinforcement Learning Techniques**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.10023v1)  

---


**ABSTRACT**  
We tackle in this paper an online network resource allocation problem with job transfers. The network is composed of many servers connected by communication links. The system operates in discrete time; at each time slot, the administrator reserves resources at servers for future job requests, and a cost is incurred for the reservations made. Then, after receptions, the jobs may be transferred between the servers to best accommodate the demands. This incurs an additional transport cost. Finally, if a job request cannot be satisfied, there is a violation that engenders a cost to pay for the blocked job. We propose a randomized online algorithm based on the exponentially weighted method. We prove that our algorithm enjoys a sub-linear in time regret, which indicates that the algorithm is adapting and learning from its experiences and is becoming more efficient in its decision-making as it accumulates more data. Moreover, we test the performance of our algorithm on artificial data and compare it against a reinforcement learning method where we show that our proposed method outperforms the latter.

{{</citation>}}


## cs.HC (2)



### (177/197) Revolutionizing Customer Interactions: Insights and Challenges in Deploying ChatGPT and Generative Chatbots for FAQs (Feriel Khennouche et al., 2023)

{{<citation>}}

Feriel Khennouche, Youssef Elmir, Nabil Djebari, Yassine Himeur, Abbes Amira. (2023)  
**Revolutionizing Customer Interactions: Insights and Challenges in Deploying ChatGPT and Generative Chatbots for FAQs**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, GPT, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.09976v1)  

---


**ABSTRACT**  
In the rapidly evolving domain of artificial intelligence, chatbots have emerged as a potent tool for various applications ranging from e-commerce to healthcare. This research delves into the intricacies of chatbot technology, from its foundational concepts to advanced generative models like ChatGPT. We present a comprehensive taxonomy of existing chatbot approaches, distinguishing between rule-based, retrieval-based, generative, and hybrid models. A specific emphasis is placed on ChatGPT, elucidating its merits for frequently asked questions (FAQs)-based chatbots, coupled with an exploration of associated Natural Language Processing (NLP) techniques such as named entity recognition, intent classification, and sentiment analysis. The paper further delves into the customization and fine-tuning of ChatGPT, its integration with knowledge bases, and the consequent challenges and ethical considerations that arise. Through real-world applications in domains such as online shopping, healthcare, and education, we underscore the transformative potential of chatbots. However, we also spotlight open challenges and suggest future research directions, emphasizing the need for optimizing conversational flow, advancing dialogue mechanics, improving domain adaptability, and enhancing ethical considerations. The research culminates in a call for further exploration in ensuring transparent, ethical, and user-centric chatbot systems.

{{</citation>}}


### (178/197) 'It's not like Jarvis, but it's pretty close!' -- Examining ChatGPT's Usage among Undergraduate Students in Computer Science (Ishika Joshi et al., 2023)

{{<citation>}}

Ishika Joshi, Ritvik Budhiraja, Harshal D Akolekar, Jagat Sesh Challa, Dhruv Kumar. (2023)  
**'It's not like Jarvis, but it's pretty close!' -- Examining ChatGPT's Usage among Undergraduate Students in Computer Science**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT, Google  
[Paper Link](http://arxiv.org/abs/2311.09651v1)  

---


**ABSTRACT**  
Large language models (LLMs) such as ChatGPT and Google Bard have garnered significant attention in the academic community. Previous research has evaluated these LLMs for various applications such as generating programming exercises and solutions. However, these evaluations have predominantly been conducted by instructors and researchers, not considering the actual usage of LLMs by students. This study adopts a student-first approach to comprehensively understand how undergraduate computer science students utilize ChatGPT, a popular LLM, released by OpenAI. We employ a combination of student surveys and interviews to obtain valuable insights into the benefits, challenges, and suggested improvements related to ChatGPT. Our findings suggest that a majority of students (over 57%) have a convincingly positive outlook towards adopting ChatGPT as an aid in coursework-related tasks. However, our research also highlights various challenges that must be resolved for long-term acceptance of ChatGPT amongst students. The findings from this investigation have broader implications and may be applicable to other LLMs and their role in computing education.

{{</citation>}}


## cs.DS (1)



### (179/197) Ghost Value Augmentation for $k$-ECSS and $k$-ECSM (D Ellis Hershkowitz et al., 2023)

{{<citation>}}

D Ellis Hershkowitz, Nathan Klein, Rico Zenklusen. (2023)  
**Ghost Value Augmentation for $k$-ECSS and $k$-ECSM**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs.DS, math-CO  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.09941v2)  

---


**ABSTRACT**  
We give a poly-time algorithm for the $k$-edge-connected spanning subgraph ($k$-ECSS) problem that returns a solution of cost no greater than the cheapest $(k+10)$-ECSS on the same graph. Our approach enhances the iterative relaxation framework with a new ingredient, which we call ghost values, that allows for high sparsity in intermediate problems.   Our guarantees improve upon the best-known approximation factor of $2$ for $k$-ECSS whenever the optimal value of $(k+10)$-ECSS is close to that of $k$-ECSS. This is a property that holds for the closely related problem $k$-edge-connected spanning multi-subgraph ($k$-ECSM), which is identical to $k$-ECSS except edges can be selected multiple times at the same cost. As a consequence, we obtain a $\left(1+O\left(\frac{1}{k}\right)\right)$-approximation algorithm for $k$-ECSM, which resolves a conjecture of Pritchard and improves upon a recent $\left(1+O\left(\frac{1}{\sqrt{k}}\right)\right)$-approximation algorithm of Karlin, Klein, Oveis Gharan, and Zhang. Moreover, we present a matching lower bound for $k$-ECSM, showing that our approximation ratio is tight up to the constant factor in $O\left(\frac{1}{k}\right)$, unless $P=NP$.

{{</citation>}}


## cs.MM (1)



### (180/197) RED-DOT: Multimodal Fact-checking via Relevant Evidence Detection (Stefanos-Iordanis Papadopoulos et al., 2023)

{{<citation>}}

Stefanos-Iordanis Papadopoulos, Christos Koutlis, Symeon Papadopoulos, Panagiotis C. Petrantonakis. (2023)  
**RED-DOT: Multimodal Fact-checking via Relevant Evidence Detection**  

---
Primary Category: cs.MM  
Categories: cs-CV, cs-MM, cs.MM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.09939v1)  

---


**ABSTRACT**  
Online misinformation is often multimodal in nature, i.e., it is caused by misleading associations between texts and accompanying images. To support the fact-checking process, researchers have been recently developing automatic multimodal methods that gather and analyze external information, evidence, related to the image-text pairs under examination. However, prior works assumed all collected evidence to be relevant. In this study, we introduce a "Relevant Evidence Detection" (RED) module to discern whether each piece of evidence is relevant, to support or refute the claim. Specifically, we develop the "Relevant Evidence Detection Directed Transformer" (RED-DOT) and explore multiple architectural variants (e.g., single or dual-stage) and mechanisms (e.g., "guided attention"). Extensive ablation and comparative experiments demonstrate that RED-DOT achieves significant improvements over the state-of-the-art on the VERITE benchmark by up to 28.5%. Furthermore, our evidence re-ranking and element-wise modality fusion led to RED-DOT achieving competitive and even improved performance on NewsCLIPings+, without the need for numerous evidence or multiple backbone encoders. Finally, our qualitative analysis demonstrates that the proposed "guided attention" module has the potential to enhance the architecture's interpretability. We release our code at: https://github.com/stevejpapad/relevant-evidence-detection

{{</citation>}}


## cond-mat.other (1)



### (181/197) On some elusive aspects of databases hindering AI based discovery: A case study on superconducting materials (Giovanni Trezza et al., 2023)

{{<citation>}}

Giovanni Trezza, Eliodoro Chiavazzo. (2023)  
**On some elusive aspects of databases hindering AI based discovery: A case study on superconducting materials**  

---
Primary Category: cond-mat.other  
Categories: cond-mat-other, cond-mat.other, cs-LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09891v1)  

---


**ABSTRACT**  
It stands to reason that the amount and the quality of big data is of key importance for setting up accurate AI-driven models. Nonetheless, we believe there are still critical roadblocks in the inherent generation of databases, that are often underestimated and poorly discussed in the literature. In our view, such issues can seriously hinder the AI-based discovery process, even when high quality, sufficiently large and highly reputable data sources are available. Here, considering superconducting and thermoelectric materials as two representative case studies, we specifically discuss three aspects, namely intrinsically biased sample selection, possible hidden variables, disparate data age. Importantly, to our knowledge, we suggest and test a first strategy capable of detecting and quantifying the presence of the intrinsic data bias.

{{</citation>}}


## cs.IT (1)



### (182/197) Cross-Layer Optimization for Statistical QoS Provision in C-RAN with Finite-Length Coding (Chang Wu et al., 2023)

{{<citation>}}

Chang Wu, Hancheng Lu, Yuang Chen, Langtian Qin. (2023)  
**Cross-Layer Optimization for Statistical QoS Provision in C-RAN with Finite-Length Coding**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-NI, cs.IT, math-IT  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.09879v1)  

---


**ABSTRACT**  
The cloud radio access network (C-RAN) has become the foundational structure for various emerging communication paradigms, leveraging the flexible deployment of distributed access points (APs) and centralized task processing. In this paper, we propose a cross-layer optimization framework based on a practical finite-length coding communication system in C-RAN, aiming at maximizing bandwidth efficiency while providing statistical quality of service (QoS) for individual services. Based on the theoretical results from effective capacity and finite-length coding, we formulate a joint optimization problem involving modulation and coding schemes (MCS), retransmission count, initial bandwidth allocation and AP selection, which reflects the coordinated decision of parameters across the physical layer, data link layer and transport layer. To tackle such a mixed-integer nonlinear programming (MINLP) problem, we firstly decompose it into a transmission parameter decision (TPD) sub-problem and a user association (UA) sub-problem, which can be solved by a binary search-based algorithm and an auction-based algorithm respectively. Simulation results demonstrate that the proposed model can accurately capture the impact of QoS requirements and channel quality on the optimal transmission parameters. Furthermore, compared with fixed transmission parameter setting, the proposed algorithms achieve the bandwidth efficiency gain up to 27.87% under various traffic and channel scenarios.

{{</citation>}}


## cs.SE (1)



### (183/197) INTERVENOR: Prompt the Coding Ability of Large Language Models with the Interactive Chain of Repairing (Hanbin Wang et al., 2023)

{{<citation>}}

Hanbin Wang, Zhenghao Liu, Shuo Wang, Ganqu Cui, Ning Ding, Zhiyuan Liu, Ge Yu. (2023)  
**INTERVENOR: Prompt the Coding Ability of Large Language Models with the Interactive Chain of Repairing**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09868v1)  

---


**ABSTRACT**  
This paper proposes INTERactiVE chaiN Of Repairing (INTERVENOR), which mimics human code repairing behavior (iteratively judging, rethinking, and repairing) and prompts the coding ability of regard Large Language Models (LLMs). Specifically, INTERVENOR employs two LLM based agents, Code Learner and Code Teacher, to play different roles in code repairing and work interactively to repair the generated codes. The Code Learner is asked to generate and repair code according to the instructions from the Code Teacher. The Code Teacher rethinks the code errors according to the corresponding feedback from compilers and iteratively generates the chain-of-repairing (CoR) to guide the code repairing process for Code Learner. Our experiments show that INTERVENOR outperforms the state-of-the-art methods and achieves about 13% and 4.5% improvements over the GPT-3.5 model in code generation and code translation tasks, respectively. Our further analyses show that CoR can illuminate the bug reasons and solution plans via natural language. Thanks to the feedback of code compilers, INTERVENOR can accurately identify the syntax errors and assertion errors in the code and provide precise instructions to repair codes, making LLMs achieve the plateau performance with only three repairing turns. All data and codes are available at https://github.com/NEUIR/INTERVENOR

{{</citation>}}


## cs.LO (1)



### (184/197) Automatic Generation of Scenarios for System-level Simulation-based Verification of Autonomous Driving Systems (Srajan Goyal et al., 2023)

{{<citation>}}

Srajan Goyal, Alberto Griggio, Jacob Kimblad, Stefano Tonetta. (2023)  
**Automatic Generation of Scenarios for System-level Simulation-based Verification of Autonomous Driving Systems**  

---
Primary Category: cs.LO  
Categories: cs-AI, cs-LO, cs-SE, cs.LO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09784v1)  

---


**ABSTRACT**  
With increasing complexity of Automated Driving Systems (ADS), ensuring their safety and reliability has become a critical challenge. The Verification and Validation (V&V) of these systems are particularly demanding when AI components are employed to implement perception and/or control functions. In ESA-funded project VIVAS, we developed a generic framework for system-level simulation-based V&V of autonomous systems. The approach is based on a simulation model of the system, an abstract model that describes symbolically the system behavior, and formal methods to generate scenarios and verify the simulation executions. Various coverage criteria can be defined to guide the automated generation of the scenarios.   In this paper, we describe the instantiation of the VIVAS framework for an ADS case study. This is based on the integration of CARLA, a widely-used driving simulator, and its ScenarioRunner tool, which enables the creation of diverse and complex driving scenarios. This is also used in the CARLA Autonomous Driving Challenge to validate different ADS agents for perception and control based on AI, shared by the CARLA community. We describe the development of an abstract ADS model and the formulation of a coverage criterion that focuses on the behaviors of vehicles relative to the vehicle with ADS under verification. Leveraging the VIVAS framework, we generate and execute various driving scenarios, thus testing the capabilities of the AI components. The results show the effectiveness of VIVAS in automatically generating scenarios for system-level simulation-based V&V of an automated driving system using CARLA and ScenarioRunner. Therefore, they highlight the potential of the approach as a powerful tool in the future of ADS V&V methodologies.

{{</citation>}}


## cs.GT (1)



### (185/197) Trust Modelling and Verification Using Event-B (Asieh Salehi Fathabadi et al., 2023)

{{<citation>}}

Asieh Salehi Fathabadi, Vahid Yazdanpanah. (2023)  
**Trust Modelling and Verification Using Event-B**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs-MA, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09777v1)  

---


**ABSTRACT**  
Trust is a crucial component in collaborative multiagent systems (MAS) involving humans and autonomous AI agents. Rather than assuming trust based on past system behaviours, it is important to formally verify trust by modelling the current state and capabilities of agents. We argue for verifying actual trust relations based on agents abilities to deliver intended outcomes in specific contexts. To enable reasoning about different notions of trust, we propose using the refinement-based formal method Event-B. Refinement allows progressively introducing new aspects of trust from abstract to concrete models incorporating knowledge and runtime states. We demonstrate modelling three trust concepts and verifying associated trust properties in MAS. The formal, correctness-by-construction approach allows to deduce guarantees about trustworthy autonomy in human-AI partnerships. Overall, our contribution facilitates rigorous verification of trust in multiagent systems.

{{</citation>}}


## cs.AR (1)



### (186/197) MEGA: A Memory-Efficient GNN Accelerator Exploiting Degree-Aware Mixed-Precision Quantization (Zeyu Zhu et al., 2023)

{{<citation>}}

Zeyu Zhu, Fanrong Li, Gang Li, Zejian Liu, Zitao Mo, Qinghao Hu, Xiaoyao Liang, Jian Cheng. (2023)  
**MEGA: A Memory-Efficient GNN Accelerator Exploiting Degree-Aware Mixed-Precision Quantization**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR, eess-SP  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Quantization  
[Paper Link](http://arxiv.org/abs/2311.09775v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are becoming a promising technique in various domains due to their excellent capabilities in modeling non-Euclidean data. Although a spectrum of accelerators has been proposed to accelerate the inference of GNNs, our analysis demonstrates that the latency and energy consumption induced by DRAM access still significantly impedes the improvement of performance and energy efficiency. To address this issue, we propose a Memory-Efficient GNN Accelerator (MEGA) through algorithm and hardware co-design in this work. Specifically, at the algorithm level, through an in-depth analysis of the node property, we observe that the data-independent quantization in previous works is not optimal in terms of accuracy and memory efficiency. This motivates us to propose the Degree-Aware mixed-precision quantization method, in which a proper bitwidth is learned and allocated to a node according to its in-degree to compress GNNs as much as possible while maintaining accuracy. At the hardware level, we employ a heterogeneous architecture design in which the aggregation and combination phases are implemented separately with different dataflows. In order to boost the performance and energy efficiency, we also present an Adaptive-Package format to alleviate the storage overhead caused by the fine-grained bitwidth and diverse sparsity, and a Condense-Edge scheduling method to enhance the data locality and further alleviate the access irregularity induced by the extremely sparse adjacency matrix in the graph. We implement our MEGA accelerator in a 28nm technology node. Extensive experiments demonstrate that MEGA can achieve an average speedup of 38.3x, 7.1x, 4.0x, 3.6x and 47.6x, 7.2x, 5.4x, 4.5x energy savings over four state-of-the-art GNN accelerators, HyGCN, GCNAX, GROW, and SGCN, respectively, while retaining task accuracy.

{{</citation>}}


## cs.SD (3)



### (187/197) DINO-VITS: Data-Efficient Noise-Robust Zero-Shot Voice Cloning via Multi-Tasking with Self-Supervised Speaker Verification Loss (Vikentii Pankov et al., 2023)

{{<citation>}}

Vikentii Pankov, Valeria Pronina, Alexander Kuzmin, Maksim Borisov, Nikita Usoltsev, Xingshan Zeng, Alexander Golubkov, Nikolai Ermolenko, Aleksandra Shirshova, Yulia Matveeva. (2023)  
**DINO-VITS: Data-Efficient Noise-Robust Zero-Shot Voice Cloning via Multi-Tasking with Self-Supervised Speaker Verification Loss**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: BERT, Self-Supervised, Speaker Verification, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.09770v2)  

---


**ABSTRACT**  
Recent progress in self-supervised representation learning has opened up new opportunities for training from unlabeled data and has been a growing trend in voice conversion. However, unsupervised training of voice cloning seems to remain a challenging task. In this paper we propose a semi-supervised zero-shot voice cloning approach that works by adapting a HuBERT-based voice conversion system to the voice cloning task and shows the robustness of such a system to noises both in training data (we add noises resulting in up to 0db signal-to-noise-ratio to 35% of training data with no significant degradation of evaluation metrics) and in the target speaker reference audio at inference. Moreover, such a method does not require any type of denoising or noise-labeling of training data. Finally, we introduce a novel multi-tasking approach by incorporating self-supervised DINO loss into joint training of a CAM++ based speaker verification system and a unit-based VITS cloning system. We show that it significantly improves the quality of generated audio over baselines, especially for noisy target speaker references.

{{</citation>}}


### (188/197) Multi-View Spectrogram Transformer for Respiratory Sound Classification (Wentao He et al., 2023)

{{<citation>}}

Wentao He, Yuchen Yan, Jianfeng Ren, Ruibin Bai, Xudong Jiang. (2023)  
**Multi-View Spectrogram Transformer for Respiratory Sound Classification**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.09655v1)  

---


**ABSTRACT**  
Deep neural networks have been applied to audio spectrograms for respiratory sound classification. Existing models often treat the spectrogram as a synthetic image while overlooking its physical characteristics. In this paper, a Multi-View Spectrogram Transformer (MVST) is proposed to embed different views of time-frequency characteristics into the vision transformer. Specifically, the proposed MVST splits the mel-spectrogram into different sized patches, representing the multi-view acoustic elements of a respiratory sound. These patches and positional embeddings are then fed into transformer encoders to extract the attentional information among patches through a self-attention mechanism. Finally, a gated fusion scheme is designed to automatically weigh the multi-view features to highlight the best one in a specific scenario. Experimental results on the ICBHI dataset demonstrate that the proposed MVST significantly outperforms state-of-the-art methods for classifying respiratory sounds.

{{</citation>}}


### (189/197) Future Full-Ocean Deep SSPs Prediction based on Hierarchical Long Short-Term Memory Neural Networks (Jiajun Lu et al., 2023)

{{<citation>}}

Jiajun Lu, Hao Zhang, Pengfei Wu, Sijia Li, Wei Huang. (2023)  
**Future Full-Ocean Deep SSPs Prediction based on Hierarchical Long Short-Term Memory Neural Networks**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.09537v1)  

---


**ABSTRACT**  
The spatial-temporal distribution of underwater sound velocity affects the propagation mode of underwater acoustic signals. Therefore, rapid estimation and prediction of underwater sound velocity distribution is crucial for providing underwater positioning, navigation and timing (PNT) services. Currently, sound speed profile (SSP) inversion methods have a faster time response rate compared to direct measurement methods, however, most SSP inversion methods focus on constructing spatial dimensional sound velocity fields and are highly dependent on sonar observation data, thus high requirements have been placed on observation data sources. To explore the distribution pattern of sound velocity in the time dimension and achieve future SSP prediction without sonar observation data, we propose a hierarchical long short-term memory (H-LSTM) neural network for SSP prediction. By our SSP prediction method, the sound speed distribution could be estimated without any on-site data measurement process, so that the time efficiency could be greatly improved. Through comparing with other state-of-the-art methods, H-LSTM has better accuracy performance on prediction of monthly average sound velocity distribution, which is less than 1 m/s in different depth layers.

{{</citation>}}


## physics.optics (1)



### (190/197) New advancements, challenges and opportunities of nanophotonics for neuromorphic computing: A state-of-the-art review (Renjie Li et al., 2023)

{{<citation>}}

Renjie Li, Yuanhao Gong, Hai Huang, Yuze Zhou, Sixuan Mao, Connie Chang-Hasnain, Zhaoyu Zhang. (2023)  
**New advancements, challenges and opportunities of nanophotonics for neuromorphic computing: A state-of-the-art review**  

---
Primary Category: physics.optics  
Categories: cs-ET, physics-optics, physics.optics  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09767v1)  

---


**ABSTRACT**  
The expansion of optoelectronic devices on photonic integration platforms has led to significant growth in the field of photonic computing. Photonic integrated circuits have facilitated the creation of ultrafast artificial neural networks, forming the basis for a novel category of information processing devices. Their application extends to diverse domains such as medical diagnosis, language models, telecommunications, quantum computing, and the metaverse, addressing the escalating demands of machine learning and artificial intelligence (AI). In contrast, conventional electronics faces challenges in latency, crosstalk, and energy consumption. Neuromorphic photonics emerges as a compelling solution, featuring sub-nanosecond latencies, minimal heat dissipation, and high parallelism, expanding the scope of AI and Optical Neural Networks. This review explores recent advances in integrated photonic neuromorphic systems, focusing on materials and device engineering breakthroughs needed to overcome existing challenges. Examining various technologies in AI accelerators, from traditional optics to PICs, we assess energy efficiency through operations per joule and compute density in operations per squared millimeter per second. A comparative analysis highlights crucial technical aspects, emphasizing nanophotonic components like VCSEL lasers, optical interconnects, nanocavity resonators, and frequency microcombs. These components showcase recent breakthroughs in photonic engineering and materials science, enabling the creation of customized neuromorphic systems for AI tasks. Despite progress, current technologies face obstacles in achieving photonic AI accelerators with computing speed and energy efficiencies reaching the petaOPS range. The review explores potential future approaches in new devices, fabrication, materials, scalability, and integration to enhance critical performance metrics.

{{</citation>}}


## cs.DC (1)



### (191/197) Application-Centric Benchmarking of Distributed FaaS Platforms using BeFaaS (Martin Grambow et al., 2023)

{{<citation>}}

Martin Grambow, Tobias Pfandzelter, David Bermbach. (2023)  
**Application-Centric Benchmarking of Distributed FaaS Platforms using BeFaaS**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AWS, Azure, Google  
[Paper Link](http://arxiv.org/abs/2311.09745v1)  

---


**ABSTRACT**  
Due to the popularity of the FaaS programming model, there is now a wide variety of commercial and open-source FaaS systems. Hence, for comparison of different FaaS systems and their configuration options, FaaS application developers rely on FaaS benchmarking frameworks. Existing frameworks, however, tend to evaluate only single isolated aspects, a more holistic application-centric benchmarking framework is still missing. In previous work, we proposed BeFaaS, an extensible application-centric benchmarking framework for FaaS environments that focuses on the evaluation of FaaS platforms through realistic and typical examples of FaaS applications. In this extended paper, we (i) enhance our benchmarking framework with additional features for distributed FaaS setups, (ii) design application benchmarks reflecting typical FaaS use cases, and (iii) use them to run extensive experiments with commercial cloud FaaS platforms (AWS Lambda, Azure Functions, Google Cloud Functions) and the tinyFaaS edge serverless platform. BeFaaS now includes four FaaS application-centric benchmarks, is extensible for additional workload profiles and platforms, and supports federated benchmark runs in which the benchmark application is distributed over multiple FaaS systems while collecting fine-grained measurement results for drill-down analysis. Our experiment results show that (i) network transmission is a major contributor to response latency for function chains, (ii) this effect is exacerbated in hybrid edge-cloud deployments, (iii) the trigger delay between a published event and the start of the triggered function ranges from about 100ms for AWS Lambda to 800ms for Google Cloud Functions, and (iv) Azure Functions shows the best cold start behavior for our workloads.

{{</citation>}}


## cs.DL (1)



### (192/197) Open Access in Ukraine: characteristics and evolution from 2012 to 2021 (Nataliia Kaliuzhna et al., 2023)

{{<citation>}}

Nataliia Kaliuzhna, Christian Hauschke. (2023)  
**Open Access in Ukraine: characteristics and evolution from 2012 to 2021**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09657v1)  

---


**ABSTRACT**  
This study investigates development of open access (OA) to publications produced by authors affiliated with Ukrainian universities and research organisations in the period 2012-2021. In order to get a comprehensive overview we assembled data from three popular databases: Dimensions, Web of Science (WoS) and Scopus. Our final dataset consisted of 187,135 records. To determine the OA status of each article, this study utilised Unpaywall data which was obtained via API. It was determined that 71.5% of all considered articles during the observed period were openly available at the time of analysis. Our findings show that gold OA was the most prevalent type of OA through a 10 years studied period. We also took a look at how OA varies by research fields, how dominant large commercial publishers are in disseminating national research and the preferences of authors regarding where to self-archive articles versions. We concluded that Ukraine needs to be thoughtful with engagement with large publishers and make sure academics control publishing, not for profit companies, which would monopolise research output distribution, leaving national publishers behind. Beyond that we put a special emphasis on the importance of FAIRness of national scholarly communication infrastructure in monitoring OA uptake.

{{</citation>}}


## quant-ph (1)



### (193/197) On the Pauli Spectrum of QAC0 (Shivam Nadimpalli et al., 2023)

{{<citation>}}

Shivam Nadimpalli, Natalie Parham, Francisca Vasconcelos, Henry Yuen. (2023)  
**On the Pauli Spectrum of QAC0**  

---
Primary Category: quant-ph  
Categories: cs-CC, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.09631v2)  

---


**ABSTRACT**  
The circuit class $\mathsf{QAC}^0$ was introduced by Moore (1999) as a model for constant depth quantum circuits where the gate set includes many-qubit Toffoli gates. Proving lower bounds against such circuits is a longstanding challenge in quantum circuit complexity; in particular, showing that polynomial-size $\mathsf{QAC}^0$ cannot compute the parity function has remained an open question for over 20 years.   In this work, we identify a notion of the Pauli spectrum of $\mathsf{QAC}^0$ circuits, which can be viewed as the quantum analogue of the Fourier spectrum of classical $\mathsf{AC}^0$ circuits. We conjecture that the Pauli spectrum of $\mathsf{QAC}^0$ circuits satisfies low-degree concentration, in analogy to the famous Linial, Nisan, Mansour theorem on the low-degree Fourier concentration of $\mathsf{AC}^0$ circuits. If true, this conjecture immediately implies that polynomial-size $\mathsf{QAC}^0$ circuits cannot compute parity.   We prove this conjecture for the class of depth-$d$, polynomial-size $\mathsf{QAC}^0$ circuits with at most $n^{O(1/d)}$ auxiliary qubits. We obtain new circuit lower bounds and learning results as applications: this class of circuits cannot correctly compute   - the $n$-bit parity function on more than $(\frac{1}{2} + 2^{-\Omega(n^{1/d})})$-fraction of inputs, and   - the $n$-bit majority function on more than $(1 - 1/\mathrm{poly}(n))$-fraction of inputs.   Additionally we show that this class of $\mathsf{QAC}^0$ circuits with limited auxiliary qubits can be learned with quasipolynomial sample complexity, giving the first learning result for $\mathsf{QAC}^0$ circuits.   More broadly, our results add evidence that "Pauli-analytic" techniques can be a powerful tool in studying quantum circuits.

{{</citation>}}


## cs.IR (3)



### (194/197) AI Recommendation System for Enhanced Customer Experience: A Novel Image-to-Text Method (Mohamaed Foued Ayedi et al., 2023)

{{<citation>}}

Mohamaed Foued Ayedi, Hiba Ben Salem, Soulaimen Hammami, Ahmed Ben Said, Rateb Jabbar, Achraf CHabbouh. (2023)  
**AI Recommendation System for Enhanced Customer Experience: A Novel Image-to-Text Method**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.09624v1)  

---


**ABSTRACT**  
Existing fashion recommendation systems encounter difficulties in using visual data for accurate and personalized recommendations. This research describes an innovative end-to-end pipeline that uses artificial intelligence to provide fine-grained visual interpretation for fashion recommendations. When customers upload images of desired products or outfits, the system automatically generates meaningful descriptions emphasizing stylistic elements. These captions guide retrieval from a global fashion product catalogue to offer similar alternatives that fit the visual characteristics of the original image. On a dataset of over 100,000 categorized fashion photos, the pipeline was trained and evaluated. The F1-score for the object detection model was 0.97, exhibiting exact fashion object recognition capabilities optimized for recommendation. This visually aware system represents a key advancement in customer engagement through personalized fashion recommendations

{{</citation>}}


### (195/197) Knowledge Plugins: Enhancing Large Language Models for Domain-Specific Recommendations (Jing Yao et al., 2023)

{{<citation>}}

Jing Yao, Wei Xu, Jianxun Lian, Xiting Wang, Xiaoyuan Yi, Xing Xie. (2023)  
**Knowledge Plugins: Enhancing Large Language Models for Domain-Specific Recommendations**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.10779v1)  

---


**ABSTRACT**  
The significant progress of large language models (LLMs) provides a promising opportunity to build human-like systems for various practical applications. However, when applied to specific task domains, an LLM pre-trained on a general-purpose corpus may exhibit a deficit or inadequacy in two types of domain-specific knowledge. One is a comprehensive set of domain data that is typically large-scale and continuously evolving. The other is specific working patterns of this domain reflected in the data. The absence or inadequacy of such knowledge impacts the performance of the LLM. In this paper, we propose a general paradigm that augments LLMs with DOmain-specific KnowledgE to enhance their performance on practical applications, namely DOKE. This paradigm relies on a domain knowledge extractor, working in three steps: 1) preparing effective knowledge for the task; 2) selecting the knowledge for each specific sample; and 3) expressing the knowledge in an LLM-understandable way. Then, the extracted knowledge is incorporated through prompts, without any computational cost of model fine-tuning. We instantiate the general paradigm on a widespread application, i.e. recommender systems, where critical item attributes and collaborative filtering signals are incorporated. Experimental results demonstrate that DOKE can substantially improve the performance of LLMs in specific domains.

{{</citation>}}


### (196/197) Towards an Automatic AI Agent for Reaction Condition Recommendation in Chemical Synthesis (Kexin Chen et al., 2023)

{{<citation>}}

Kexin Chen, Junyou Li, Kunyi Wang, Yuyang Du, Jiahui Yu, Jiamin Lu, Guangyong Chen, Lanqing Li, Jiezhong Qiu, Qun Fang, Pheng Ann Heng. (2023)  
**Towards an Automatic AI Agent for Reaction Condition Recommendation in Chemical Synthesis**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: AI, Contrastive Learning, Language Model  
[Paper Link](http://arxiv.org/abs/2311.10776v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) for reaction condition optimization has become an important topic in the pharmaceutical industry, given that a data-driven AI model can assist drug discovery and accelerate reaction design. However, existing AI models lack the chemical insights and real-time knowledge acquisition abilities of experienced human chemists. This paper proposes a Large Language Model (LLM) empowered AI agent to bridge this gap. We put forth a novel three-phase paradigm and applied advanced intelligence-enhancement methods like in-context learning and multi-LLM debate so that the AI agent can borrow human insight and update its knowledge by searching the latest chemical literature. Additionally, we introduce a novel Coarse-label Contrastive Learning (CCL) based chemical fingerprint that greatly enhances the agent's performance in optimizing the reaction condition. With the above efforts, the proposed AI agent can autonomously generate the optimal reaction condition recommendation without any human interaction. Further, the agent is highly professional in terms of chemical reactions. It demonstrates close-to-human performance and strong generalization capability in both dry-lab and wet-lab experiments. As the first attempt in the chemical AI agent, this work goes a step further in the field of "AI for chemistry" and opens up new possibilities for computer-aided synthesis planning.

{{</citation>}}


## physics.soc-ph (1)



### (197/197) Simulating Opinion Dynamics with Networks of LLM-based Agents (Yun-Shiuan Chuang et al., 2023)

{{<citation>}}

Yun-Shiuan Chuang, Agam Goyal, Nikunj Harlalka, Siddharth Suresh, Robert Hawkins, Sijia Yang, Dhavan Shah, Junjie Hu, Timothy T. Rogers. (2023)  
**Simulating Opinion Dynamics with Networks of LLM-based Agents**  

---
Primary Category: physics.soc-ph  
Categories: cs-CL, physics-soc-ph, physics.soc-ph  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.09618v1)  

---


**ABSTRACT**  
Accurately simulating human opinion dynamics is crucial for understanding a variety of societal phenomena, including polarization and the spread of misinformation. However, the agent-based models (ABMs) commonly used for such simulations lack fidelity to human behavior. We propose a new approach to simulating opinion dynamics based on populations of Large Language Models (LLMs). Our findings reveal a strong inherent bias in LLM agents towards accurate information, leading to consensus in line with scientific reality. However, this bias limits the simulation of individuals with resistant views on issues like climate change. After inducing confirmation bias through prompt engineering, we observed opinion fragmentation in line with existing agent-based research. These insights highlight the promise and limitations of LLM agents in this domain and suggest a path forward: refining LLMs with real-world discourse to better simulate the evolution of human beliefs.

{{</citation>}}
