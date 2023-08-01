---
draft: false
title: "arXiv @ 2023.07.28"
date: 2023-07-28
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.28"
    identifier: arxiv_20230728
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (18)](#cscl-18)
- [cs.LG (11)](#cslg-11)
- [cs.CV (20)](#cscv-20)
- [cs.HC (5)](#cshc-5)
- [cs.RO (4)](#csro-4)
- [cs.MM (3)](#csmm-3)
- [eess.IV (4)](#eessiv-4)
- [math.CT (1)](#mathct-1)
- [cs.IR (6)](#csir-6)
- [cs.SD (2)](#cssd-2)
- [q-fin.CP (1)](#q-fincp-1)
- [eess.SY (1)](#eesssy-1)
- [cs.AI (3)](#csai-3)
- [cs.NI (1)](#csni-1)
- [cs.CY (6)](#cscy-6)
- [cs.CR (4)](#cscr-4)
- [math.OC (1)](#mathoc-1)
- [cs.IT (2)](#csit-2)
- [cs.DC (1)](#csdc-1)

## cs.CL (18)



### (1/94) Speed Reading Tool Powered by Artificial Intelligence for Students with ADHD, Dyslexia, or Short Attention Span (Megat Irfan Zackry Bin Ismail Ahmad Nazran bin Yusri Muhammad Hafizzul Bin Abdul Manap Muhammad Muizzuddin Bin Kamarozaman, 2023)

{{<citation>}}

Megat Irfan Zackry Bin Ismail Ahmad Nazran bin Yusri Muhammad Hafizzul Bin Abdul Manap Muhammad Muizzuddin Bin Kamarozaman. (2023)  
**Speed Reading Tool Powered by Artificial Intelligence for Students with ADHD, Dyslexia, or Short Attention Span**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Attention, NLP, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2307.14544v1)  

---


**ABSTRACT**  
This paper presents a novel approach to assist students with dyslexia, ADHD, and short attention span in digesting any text-based information more efficiently. The proposed solution utilizes the Multilayer Perceptron (MLP) algorithm for complex text processing and summarization tasks. The tool leverages the T5 (Text-to-Text Transfer Transformer) model from Hugging Face, which treats every NLP task as a text generation task. The model is fine-tuned on specific tasks using a smaller dataset. The NLTK's Punkt Sentence Tokenizer is used to divide a text into a list of sentences. The application is served using Flask, a lightweight web server and framework. The tool also applies principles from Bionic Reading to enhance readability, which includes a bolding function and adjustments to line, word, and character spacing. The paper discusses the methodology, implementation, and results of the AI-based speed reading tool.

{{</citation>}}


### (2/94) CliniDigest: A Case Study in Large Language Model Based Large-Scale Summarization of Clinical Trial Descriptions (Renee D. White et al., 2023)

{{<citation>}}

Renee D. White, Tristan Peng, Pann Sripitak, Alexander Rosenberg Johansen, Michael Snyder, Stanford University. (2023)  
**CliniDigest: A Case Study in Large Language Model Based Large-Scale Summarization of Clinical Trial Descriptions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Clinical, GPT, GPT-3.5, Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2307.14522v1)  

---


**ABSTRACT**  
A clinical trial is a study that evaluates new biomedical interventions. To design new trials, researchers draw inspiration from those current and completed. In 2022, there were on average more than 100 clinical trials submitted to ClinicalTrials.gov every day, with each trial having a mean of approximately 1500 words [1]. This makes it nearly impossible to keep up to date. To mitigate this issue, we have created a batch clinical trial summarizer called CliniDigest using GPT-3.5. CliniDigest is, to our knowledge, the first tool able to provide real-time, truthful, and comprehensive summaries of clinical trials. CliniDigest can reduce up to 85 clinical trial descriptions (approximately 10,500 words) into a concise 200-word summary with references and limited hallucinations. We have tested CliniDigest on its ability to summarize 457 trials divided across 27 medical subdomains. For each field, CliniDigest generates summaries of $\mu=153,\ \sigma=69 $ words, each of which utilizes $\mu=54\%,\ \sigma=30\% $ of the sources. A more comprehensive evaluation is planned and outlined in this paper.

{{</citation>}}


### (3/94) Controllable Generation of Dialogue Acts for Dialogue Systems via Few-Shot Response Generation and Ranking (Angela Ramirez et al., 2023)

{{<citation>}}

Angela Ramirez, Karik Agarwal, Juraj Juraska, Utkarsh Garg, Marilyn A. Walker. (2023)  
**Controllable Generation of Dialogue Acts for Dialogue Systems via Few-Shot Response Generation and Ranking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.14440v1)  

---


**ABSTRACT**  
Dialogue systems need to produce responses that realize multiple types of dialogue acts (DAs) with high semantic fidelity. In the past, natural language generators (NLGs) for dialogue were trained on large parallel corpora that map from a domain-specific DA and its semantic attributes to an output utterance. Recent work shows that pretrained language models (LLMs) offer new possibilities for controllable NLG using prompt-based learning. Here we develop a novel few-shot overgenerate-and-rank approach that achieves the controlled generation of DAs. We compare eight few-shot prompt styles that include a novel method of generating from textual pseudo-references using a textual style transfer approach. We develop six automatic ranking functions that identify outputs with both the correct DA and high semantic accuracy at generation time. We test our approach on three domains and four LLMs. To our knowledge, this is the first work on NLG for dialogue that automatically ranks outputs using both DA and attribute accuracy. For completeness, we compare our results to fine-tuned few-shot models trained with 5 to 100 instances per DA. Our results show that several prompt settings achieve perfect DA accuracy, and near perfect semantic accuracy (99.81%) and perform better than few-shot fine-tuning.

{{</citation>}}


### (4/94) Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models (Mayee F. Chen et al., 2023)

{{<citation>}}

Mayee F. Chen, Nicholas Roberts, Kush Bhatia, Jue Wang, Ce Zhang, Frederic Sala, Christopher Ré. (2023)  
**Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.14430v1)  

---


**ABSTRACT**  
The quality of training data impacts the performance of pre-trained large language models (LMs). Given a fixed budget of tokens, we study how to best select data that leads to good downstream model performance across tasks. We develop a new framework based on a simple hypothesis: just as humans acquire interdependent skills in a deliberate order, language models also follow a natural order when learning a set of skills from their training data. If such an order exists, it can be utilized for improved understanding of LMs and for data-efficient training. Using this intuition, our framework formalizes the notion of a skill and of an ordered set of skills in terms of the associated data. First, using both synthetic and real data, we demonstrate that these ordered skill sets exist, and that their existence enables more advanced skills to be learned with less data when we train on their prerequisite skills. Second, using our proposed framework, we introduce an online data sampling algorithm, Skill-It, over mixtures of skills for both continual pre-training and fine-tuning regimes, where the objective is to efficiently learn multiple skills in the former and an individual skill in the latter. On the LEGO synthetic in the continual pre-training setting, Skill-It obtains 36.5 points higher accuracy than random sampling. On the Natural Instructions dataset in the fine-tuning setting, Skill-It reduces the validation loss on the target skill by 13.6% versus training on data associated with the target skill itself. We apply our skills framework on the recent RedPajama dataset to continually pre-train a 3B-parameter LM, achieving higher accuracy on the LM Evaluation Harness with 1B tokens than the baseline approach of sampling uniformly over data sources with 3B tokens.

{{</citation>}}


### (5/94) Towards Generalist Biomedical AI (Tao Tu et al., 2023)

{{<citation>}}

Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Chuck Lau, Ryutaro Tanno, Ira Ktena, Basil Mustafa, Aakanksha Chowdhery, Yun Liu, Simon Kornblith, David Fleet, Philip Mansfield, Sushant Prakash, Renee Wong, Sunny Virmani, Christopher Semturs, S Sara Mahdavi, Bradley Green, Ewa Dominowska, Blaise Aguera y Arcas, Joelle Barral, Dale Webster, Greg S. Corrado, Yossi Matias, Karan Singhal, Pete Florence, Alan Karthikesalingam, Vivek Natarajan. (2023)  
**Towards Generalist Biomedical AI**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: AI, PaLM  
[Paper Link](http://arxiv.org/abs/2307.14334v1)  

---


**ABSTRACT**  
Medicine is inherently multimodal, with rich data modalities spanning text, imaging, genomics, and more. Generalist biomedical artificial intelligence (AI) systems that flexibly encode, integrate, and interpret this data at scale can potentially enable impactful applications ranging from scientific discovery to care delivery. To enable the development of these models, we first curate MultiMedBench, a new multimodal biomedical benchmark. MultiMedBench encompasses 14 diverse tasks such as medical question answering, mammography and dermatology image interpretation, radiology report generation and summarization, and genomic variant calling. We then introduce Med-PaLM Multimodal (Med-PaLM M), our proof of concept for a generalist biomedical AI system. Med-PaLM M is a large multimodal generative model that flexibly encodes and interprets biomedical data including clinical language, imaging, and genomics with the same set of model weights. Med-PaLM M reaches performance competitive with or exceeding the state of the art on all MultiMedBench tasks, often surpassing specialist models by a wide margin. We also report examples of zero-shot generalization to novel medical concepts and tasks, positive transfer learning across tasks, and emergent zero-shot medical reasoning. To further probe the capabilities and limitations of Med-PaLM M, we conduct a radiologist evaluation of model-generated (and human) chest X-ray reports and observe encouraging performance across model scales. In a side-by-side ranking on 246 retrospective chest X-rays, clinicians express a pairwise preference for Med-PaLM M reports over those produced by radiologists in up to 40.50% of cases, suggesting potential clinical utility. While considerable work is needed to validate these models in real-world use cases, our results represent a milestone towards the development of generalist biomedical AI systems.

{{</citation>}}


### (6/94) Utilizing Large Language Models for Natural Interface to Pharmacology Databases (Hong Lu et al., 2023)

{{<citation>}}

Hong Lu, Chuan Li, Yinheng Li, Jie Zhao. (2023)  
**Utilizing Large Language Models for Natural Interface to Pharmacology Databases**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DB, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.15717v1)  

---


**ABSTRACT**  
The drug development process necessitates that pharmacologists undertake various tasks, such as reviewing literature, formulating hypotheses, designing experiments, and interpreting results. Each stage requires accessing and querying vast amounts of information. In this abstract, we introduce a Large Language Model (LLM)-based Natural Language Interface designed to interact with structured information stored in databases. Our experiments demonstrate the feasibility and effectiveness of the proposed framework. This framework can generalize to query a wide range of pharmaceutical data and knowledge bases.

{{</citation>}}


### (7/94) Comparative Analysis of Libraries for the Sentimental Analysis (Wendy Ccoya et al., 2023)

{{<citation>}}

Wendy Ccoya, Edson Pinto. (2023)  
**Comparative Analysis of Libraries for the Sentimental Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.14311v1)  

---


**ABSTRACT**  
This study is main goal is to provide a comparative comparison of libraries using machine learning methods. Experts in natural language processing (NLP) are becoming more and more interested in sentiment analysis (SA) of text changes. The objective of employing NLP text analysis techniques is to recognize and categorize feelings related to twitter users utterances. In this examination, issues with SA and the libraries utilized are also looked at. provides a number of cooperative methods to classify emotional polarity. The Naive Bayes Classifier, Decision Tree Classifier, Maxent Classifier, Sklearn Classifier, Sklearn Classifier MultinomialNB, and other conjoint learning algorithms, according to recent research, are very effective. In the project will use Five Python and R libraries NLTK, TextBlob, Vader, Transformers (GPT and BERT pretrained), and Tidytext will be used in the study to apply sentiment analysis techniques. Four machine learning models Tree of Decisions (DT), Support Vector Machine (SVM), Naive Bayes (NB), and K-Nearest Neighbor (KNN) will also be used. To evaluate how well libraries for SA operate in the social network environment, comparative study was also carried out. The measures to assess the best algorithms in this experiment, which used a single data set for each method, were precision, recall, and F1 score. We conclude that the BERT transformer method with an Accuracy: 0.973 is recommended for sentiment analysis.

{{</citation>}}


### (8/94) Automatically Evaluating Opinion Prevalence in Opinion Summarization (Christopher Malon, 2023)

{{<citation>}}

Christopher Malon. (2023)  
**Automatically Evaluating Opinion Prevalence in Opinion Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Amazon, Summarization  
[Paper Link](http://arxiv.org/abs/2307.14305v1)  

---


**ABSTRACT**  
When faced with a large number of product reviews, it is not clear that a human can remember all of them and weight opinions representatively to write a good reference summary. We propose an automatic metric to test the prevalence of the opinions that a summary expresses, based on counting the number of reviews that are consistent with each statement in the summary, while discrediting trivial or redundant statements. To formulate this opinion prevalence metric, we consider several existing methods to score the factual consistency of a summary statement with respect to each individual source review. On a corpus of Amazon product reviews, we gather multiple human judgments of the opinion consistency, to determine which automatic metric best expresses consistency in product reviews. Using the resulting opinion prevalence metric, we show that a human authored summary has only slightly better opinion prevalence than randomly selected extracts from the source reviews, and previous extractive and abstractive unsupervised opinion summarization methods perform worse than humans. We demonstrate room for improvement with a greedy construction of extractive summaries with twice the opinion prevalence achieved by humans. Finally, we show that preprocessing source reviews by simplification can raise the opinion prevalence achieved by existing abstractive opinion summarization systems to the level of human performance.

{{</citation>}}


### (9/94) Developing and Evaluating Tiny to Medium-Sized Turkish BERT Models (Himmet Toprak Kesgin et al., 2023)

{{<citation>}}

Himmet Toprak Kesgin, Muzaffer Kaan Yuce, Mehmet Fatih Amasyali. (2023)  
**Developing and Evaluating Tiny to Medium-Sized Turkish BERT Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.14134v1)  

---


**ABSTRACT**  
This study introduces and evaluates tiny, mini, small, and medium-sized uncased Turkish BERT models, aiming to bridge the research gap in less-resourced languages. We trained these models on a diverse dataset encompassing over 75GB of text from multiple sources and tested them on several tasks, including mask prediction, sentiment analysis, news classification, and, zero-shot classification. Despite their smaller size, our models exhibited robust performance, including zero-shot task, while ensuring computational efficiency and faster execution times. Our findings provide valuable insights into the development and application of smaller language models, especially in the context of the Turkish language.

{{</citation>}}


### (10/94) Leveraging Implicit Feedback from Deployment Data in Dialogue (Richard Yuanzhe Pang et al., 2023)

{{<citation>}}

Richard Yuanzhe Pang, Stephen Roller, Kyunghyun Cho, He He, Jason Weston. (2023)  
**Leveraging Implicit Feedback from Deployment Data in Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.14117v1)  

---


**ABSTRACT**  
We study improving social conversational agents by learning from natural dialogue between users and a deployed model, without extra annotations. To implicitly measure the quality of a machine-generated utterance, we leverage signals like user response length, sentiment and reaction of the future human utterances in the collected dialogue episodes. Our experiments use the publicly released deployment data from BlenderBot (Xu et al., 2023). Human evaluation indicates improvements in our new models over baseline responses; however, we find that some proxy signals can lead to more generations with undesirable properties as well. For example, optimizing for conversation length can lead to more controversial or unfriendly generations compared to the baseline, whereas optimizing for positive sentiment or reaction can decrease these behaviors.

{{</citation>}}


### (11/94) Decoding ChatGPT: A Taxonomy of Existing Research, Current Challenges, and Possible Future Directions (Shahab Saquib Sohail et al., 2023)

{{<citation>}}

Shahab Saquib Sohail, Faiza Farhat, Yassine Himeur, Mohammad Nadeem, Dag Øivind Madsen, Yashbir Singh, Shadi Atalla, Wathiq Mansoor. (2023)  
**Decoding ChatGPT: A Taxonomy of Existing Research, Current Challenges, and Possible Future Directions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: AI, ChatGPT, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2307.14107v1)  

---


**ABSTRACT**  
Chat Generative Pre-trained Transformer (ChatGPT) has gained significant interest and attention since its launch in November 2022. It has shown impressive performance in various domains, including passing exams and creative writing. However, challenges and concerns related to biases and trust persist. In this work, we present a comprehensive review of over 100 Scopus-indexed publications on ChatGPT, aiming to provide a taxonomy of ChatGPT research and explore its applications. We critically analyze the existing literature, identifying common approaches employed in the studies. Additionally, we investigate diverse application areas where ChatGPT has found utility, such as healthcare, marketing and financial services, software engineering, academic and scientific writing, research and education, environmental science, and natural language processing. Through examining these applications, we gain valuable insights into the potential of ChatGPT in addressing real-world challenges. We also discuss crucial issues related to ChatGPT, including biases and trustworthiness, emphasizing the need for further research and development in these areas. Furthermore, we identify potential future directions for ChatGPT research, proposing solutions to current challenges and speculating on expected advancements. By fully leveraging the capabilities of ChatGPT, we can unlock its potential across various domains, leading to advancements in conversational AI and transformative impacts in society.

{{</citation>}}


### (12/94) Multi3WOZ: A Multilingual, Multi-Domain, Multi-Parallel Dataset for Training and Evaluating Culturally Adapted Task-Oriented Dialog Systems (Songbo Hu et al., 2023)

{{<citation>}}

Songbo Hu, Han Zhou, Mete Hergul, Milan Gritta, Guchun Zhang, Ignacio Iacobacci, Ivan Vulić, Anna Korhonen. (2023)  
**Multi3WOZ: A Multilingual, Multi-Domain, Multi-Parallel Dataset for Training and Evaluating Culturally Adapted Task-Oriented Dialog Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Multilingual  
[Paper Link](http://arxiv.org/abs/2307.14031v1)  

---


**ABSTRACT**  
Creating high-quality annotated data for task-oriented dialog (ToD) is known to be notoriously difficult, and the challenges are amplified when the goal is to create equitable, culturally adapted, and large-scale ToD datasets for multiple languages. Therefore, the current datasets are still very scarce and suffer from limitations such as translation-based non-native dialogs with translation artefacts, small scale, or lack of cultural adaptation, among others. In this work, we first take stock of the current landscape of multilingual ToD datasets, offering a systematic overview of their properties and limitations. Aiming to reduce all the detected limitations, we then introduce Multi3WOZ, a novel multilingual, multi-domain, multi-parallel ToD dataset. It is large-scale and offers culturally adapted dialogs in 4 languages to enable training and evaluation of multilingual and cross-lingual ToD systems. We describe a complex bottom-up data collection process that yielded the final dataset, and offer the first sets of baseline scores across different ToD-related tasks for future reference, also highlighting its challenging nature.

{{</citation>}}


### (13/94) Affective Natural Language Generation of Event Descriptions through Fine-grained Appraisal Conditions (Yarik Menchaca Resendiz et al., 2023)

{{<citation>}}

Yarik Menchaca Resendiz, Roman Klinger. (2023)  
**Affective Natural Language Generation of Event Descriptions through Fine-grained Appraisal Conditions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Natural Language Generation, T5  
[Paper Link](http://arxiv.org/abs/2307.14004v1)  

---


**ABSTRACT**  
Models for affective text generation have shown a remarkable progress, but they commonly rely only on basic emotion theories or valance/arousal values as conditions. This is appropriate when the goal is to create explicit emotion statements ("The kid is happy."). Emotions are, however, commonly communicated implicitly. For instance, the emotional interpretation of an event ("Their dog died.") does often not require an explicit emotion statement. In psychology, appraisal theories explain the link between a cognitive evaluation of an event and the potentially developed emotion. They put the assessment of the situation on the spot, for instance regarding the own control or the responsibility for what happens. We hypothesize and subsequently show that including appraisal variables as conditions in a generation framework comes with two advantages. (1) The generation model is informed in greater detail about what makes a specific emotion and what properties it has. This leads to text generation that better fulfills the condition. (2) The variables of appraisal allow a user to perform a more fine-grained control of the generated text, by stating properties of a situation instead of only providing the emotion category. Our Bart and T5-based experiments with 7 emotions (Anger, Disgust, Fear, Guilt, Joy, Sadness, Shame), and 7 appraisals (Attention, Responsibility, Control, Circumstance, Pleasantness, Effort, Certainty) show that (1) adding appraisals during training improves the accurateness of the generated texts by 10 pp in F1. Further, (2) the texts with appraisal variables are longer and contain more details. This exemplifies the greater control for users.

{{</citation>}}


### (14/94) This is not correct! Negation-aware Evaluation of Language Generation Systems (Miriam Anschütz et al., 2023)

{{<citation>}}

Miriam Anschütz, Diego Miguel Lozano, Georg Groh. (2023)  
**This is not correct! Negation-aware Evaluation of Language Generation Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2307.13989v1)  

---


**ABSTRACT**  
Large language models underestimate the impact of negations on how much they change the meaning of a sentence. Therefore, learned evaluation metrics based on these models are insensitive to negations. In this paper, we propose NegBLEURT, a negation-aware version of the BLEURT evaluation metric. For that, we designed a rule-based sentence negation tool and used it to create the CANNOT negation evaluation dataset. Based on this dataset, we fine-tuned a sentence transformer and an evaluation metric to improve their negation sensitivity. Evaluating these models on existing benchmarks shows that our fine-tuned models outperform existing metrics on the negated sentences by far while preserving their base models' performances on other perturbations.

{{</citation>}}


### (15/94) How Does Diffusion Influence Pretrained Language Models on Out-of-Distribution Data? (Huazheng Wang et al., 2023)

{{<citation>}}

Huazheng Wang, Daixuan Cheng, Haifeng Sun, Jingyu Wang, Qi Qi, Jianxin Liao, Jing Wang, Cong Liu. (2023)  
**How Does Diffusion Influence Pretrained Language Models on Out-of-Distribution Data?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP, Pretrained Language Models, Transformer  
[Paper Link](http://arxiv.org/abs/2307.13949v1)  

---


**ABSTRACT**  
Transformer-based pretrained language models (PLMs) have achieved great success in modern NLP. An important advantage of PLMs is good out-of-distribution (OOD) robustness. Recently, diffusion models have attracted a lot of work to apply diffusion to PLMs. It remains under-explored how diffusion influences PLMs on OOD data. The core of diffusion models is a forward diffusion process which gradually applies Gaussian noise to inputs, and a reverse denoising process which removes noise. The noised input reconstruction is a fundamental ability of diffusion models. We directly analyze OOD robustness by measuring the reconstruction loss, including testing the abilities to reconstruct OOD data, and to detect OOD samples. Experiments are conducted by analyzing different training parameters and data statistical features on eight datasets. It shows that finetuning PLMs with diffusion degrades the reconstruction ability on OOD data. The comparison also shows that diffusion models can effectively detect OOD samples, achieving state-of-the-art performance in most of the datasets with an absolute accuracy improvement up to 18%. These results indicate that diffusion reduces OOD robustness of PLMs.

{{</citation>}}


### (16/94) GrammarGPT: Exploring Open-Source LLMs for Native Chinese Grammatical Error Correction with Supervised Fine-Tuning (Yaxin Fan et al., 2023)

{{<citation>}}

Yaxin Fan, Feng Jiang, Peifeng Li, Haizhou Li. (2023)  
**GrammarGPT: Exploring Open-Source LLMs for Native Chinese Grammatical Error Correction with Supervised Fine-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2307.13923v1)  

---


**ABSTRACT**  
Grammatical error correction aims to correct ungrammatical sentences automatically. Recently, some work has demonstrated the excellent capabilities of closed-source Large Language Models (LLMs, e.g., ChatGPT) in grammatical error correction. However, the potential of open-source LLMs remains unexplored. In this paper, we introduced GrammarGPT, an open-source LLM, to preliminary explore its potential for native Chinese grammatical error correction. The core recipe of GrammarGPT is to leverage the hybrid dataset of ChatGPT-generated and human-annotated. For grammatical errors with clues, we proposed a heuristic method to guide ChatGPT to generate ungrammatical sentences by providing those clues. For grammatical errors without clues, we collected ungrammatical sentences from publicly available websites and manually corrected them. In addition, we employed an error-invariant augmentation method to enhance the ability of the model to correct native Chinese grammatical errors. We ultimately constructed about 1k parallel data and utilized these data to fine-tune open-source LLMs (e.g., Phoenix, released by The Chinese University of Hong Kong, Shenzhen) with instruction tuning. The experimental results show that GrammarGPT outperforms the existing SOTA system significantly. Although model parameters are 20x larger than the SOTA baseline, the required amount of data for instruction tuning is 1200x smaller, illustrating the potential of open-source LLMs on native CGEC. Our GrammarGPT ranks $3^{rd}$ on NLPCC2023 SharedTask1, demonstrating our approach's effectiveness. The code and data are available at \url{https://github.com/FreedomIntelligence/GrammarGPT}.

{{</citation>}}


### (17/94) Data Augmentation for Neural Machine Translation using Generative Language Model (Seokjin Oh et al., 2023)

{{<citation>}}

Seokjin Oh, Su ah Lee, Woohwan Jung. (2023)  
**Data Augmentation for Neural Machine Translation using Generative Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation, BLEU, ChatGPT, GPT, Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2307.16833v1)  

---


**ABSTRACT**  
Despite the rapid growth in model architecture, the scarcity of large parallel corpora remains the main bottleneck in Neural Machine Translation. Data augmentation is a technique that enhances the performance of data-hungry models by generating synthetic data instead of collecting new ones. We explore prompt-based data augmentation approaches that leverage large-scale language models such as ChatGPT. To create a synthetic parallel corpus, we compare 3 methods using different prompts. We employ two assessment metrics to measure the diversity of the generated synthetic data. This approach requires no further model training cost, which is mandatory in other augmentation methods like back-translation. The proposed method improves the unaugmented baseline by 0.68 BLEU score.

{{</citation>}}


### (18/94) FinTree: Financial Dataset Pretrain Transformer Encoder for Relation Extraction (Hyunjong Ok, 2023)

{{<citation>}}

Hyunjong Ok. (2023)  
**FinTree: Financial Dataset Pretrain Transformer Encoder for Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Financial, Relation Extraction, Transformer  
[Paper Link](http://arxiv.org/abs/2307.13900v1)  

---


**ABSTRACT**  
We present FinTree, Financial Dataset Pretrain Transformer Encoder for Relation Extraction. Utilizing an encoder language model, we further pretrain FinTree on the financial dataset, adapting the model in financial domain tasks. FinTree stands out with its novel structure that predicts a masked token instead of the conventional [CLS] token, inspired by the Pattern Exploiting Training methodology. This structure allows for more accurate relation predictions between two given entities. The model is trained with a unique input pattern to provide contextual and positional information about the entities of interest, and a post-processing step ensures accurate predictions in line with the entity types. Our experiments demonstrate that FinTree outperforms on the REFinD, a large-scale financial relation extraction dataset. The code and pretrained models are available at https://github.com/HJ-Ok/FinTree.

{{</citation>}}


## cs.LG (11)



### (19/94) Controlling the Inductive Bias of Wide Neural Networks by Modifying the Kernel's Spectrum (Amnon Geifman et al., 2023)

{{<citation>}}

Amnon Geifman, Daniel Barzilai, Ronen Basri, Meirav Galun. (2023)  
**Controlling the Inductive Bias of Wide Neural Networks by Modifying the Kernel's Spectrum**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.14531v1)  

---


**ABSTRACT**  
Wide neural networks are biased towards learning certain functions, influencing both the rate of convergence of gradient descent (GD) and the functions that are reachable with GD in finite training time. As such, there is a great need for methods that can modify this bias according to the task at hand. To that end, we introduce Modified Spectrum Kernels (MSKs), a novel family of constructed kernels that can be used to approximate kernels with desired eigenvalues for which no closed form is known. We leverage the duality between wide neural networks and Neural Tangent Kernels and propose a preconditioned gradient descent method, which alters the trajectory of GD. As a result, this allows for a polynomial and, in some cases, exponential training speedup without changing the final solution. Our method is both computationally efficient and simple to implement.

{{</citation>}}


### (20/94) HUGE: Huge Unsupervised Graph Embeddings with TPUs (Brandon Mayer et al., 2023)

{{<citation>}}

Brandon Mayer, Anton Tsitsulin, Hendrik Fichtenberger, Jonathan Halcrow, Bryan Perozzi. (2023)  
**HUGE: Huge Unsupervised Graph Embeddings with TPUs**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs-SI, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.14490v1)  

---


**ABSTRACT**  
Graphs are a representation of structured data that captures the relationships between sets of objects. With the ubiquity of available network data, there is increasing industrial and academic need to quickly analyze graphs with billions of nodes and trillions of edges. A common first step for network understanding is Graph Embedding, the process of creating a continuous representation of nodes in a graph. A continuous representation is often more amenable, especially at scale, for solving downstream machine learning tasks such as classification, link prediction, and clustering. A high-performance graph embedding architecture leveraging Tensor Processing Units (TPUs) with configurable amounts of high-bandwidth memory is presented that simplifies the graph embedding problem and can scale to graphs with billions of nodes and trillions of edges. We verify the embedding space quality on real and synthetic large-scale datasets.

{{</citation>}}


### (21/94) Reinforcement Learning by Guided Safe Exploration (Qisong Yang et al., 2023)

{{<citation>}}

Qisong Yang, Thiago D. Simão, Nils Jansen, Simon H. Tindemans, Matthijs T. J. Spaan. (2023)  
**Reinforcement Learning by Guided Safe Exploration**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.14316v1)  

---


**ABSTRACT**  
Safety is critical to broadening the application of reinforcement learning (RL). Often, we train RL agents in a controlled environment, such as a laboratory, before deploying them in the real world. However, the real-world target task might be unknown prior to deployment. Reward-free RL trains an agent without the reward to adapt quickly once the reward is revealed. We consider the constrained reward-free setting, where an agent (the guide) learns to explore safely without the reward signal. This agent is trained in a controlled environment, which allows unsafe interactions and still provides the safety signal. After the target task is revealed, safety violations are not allowed anymore. Thus, the guide is leveraged to compose a safe behaviour policy. Drawing from transfer learning, we also regularize a target policy (the student) towards the guide while the student is unreliable and gradually eliminate the influence of the guide as training progresses. The empirical analysis shows that this method can achieve safe transfer learning and helps the student solve the target task faster.

{{</citation>}}


### (22/94) Unraveling the Complexity of Splitting Sequential Data: Tackling Challenges in Video and Time Series Analysis (Diego Botache et al., 2023)

{{<citation>}}

Diego Botache, Kristina Dingel, Rico Huhnstock, Arno Ehresmann, Bernhard Sick. (2023)  
**Unraveling the Complexity of Splitting Sequential Data: Tackling Challenges in Video and Time Series Analysis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.14294v1)  

---


**ABSTRACT**  
Splitting of sequential data, such as videos and time series, is an essential step in various data analysis tasks, including object tracking and anomaly detection. However, splitting sequential data presents a variety of challenges that can impact the accuracy and reliability of subsequent analyses. This concept article examines the challenges associated with splitting sequential data, including data acquisition, data representation, split ratio selection, setting up quality criteria, and choosing suitable selection strategies. We explore these challenges through two real-world examples: motor test benches and particle tracking in liquids.

{{</citation>}}


### (23/94) A comparison of machine learning surrogate models of street-scale flooding in Norfolk, Virginia (Diana McSpadden et al., 2023)

{{<citation>}}

Diana McSpadden, Steven Goldenberg, Binata Roy, Malachi Schram, Jonathan L. Goodall, Heather Richter. (2023)  
**A comparison of machine learning surrogate models of street-scale flooding in Norfolk, Virginia**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.14185v1)  

---


**ABSTRACT**  
Low-lying coastal cities, exemplified by Norfolk, Virginia, face the challenge of street flooding caused by rainfall and tides, which strain transportation and sewer systems and can lead to property damage. While high-fidelity, physics-based simulations provide accurate predictions of urban pluvial flooding, their computational complexity renders them unsuitable for real-time applications. Using data from Norfolk rainfall events between 2016 and 2018, this study compares the performance of a previous surrogate model based on a random forest algorithm with two deep learning models: Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU). This investigation underscores the importance of using a model architecture that supports the communication of prediction uncertainty and the effective integration of relevant, multi-modal features.

{{</citation>}}


### (24/94) Actions Speak What You Want: Provably Sample-Efficient Reinforcement Learning of the Quantal Stackelberg Equilibrium from Strategic Feedbacks (Siyu Chen et al., 2023)

{{<citation>}}

Siyu Chen, Mengdi Wang, Zhuoran Yang. (2023)  
**Actions Speak What You Want: Provably Sample-Efficient Reinforcement Learning of the Quantal Stackelberg Equilibrium from Strategic Feedbacks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-ST, stat-ML, stat-TH  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.14085v1)  

---


**ABSTRACT**  
We study reinforcement learning (RL) for learning a Quantal Stackelberg Equilibrium (QSE) in an episodic Markov game with a leader-follower structure. In specific, at the outset of the game, the leader announces her policy to the follower and commits to it. The follower observes the leader's policy and, in turn, adopts a quantal response policy by solving an entropy-regularized policy optimization problem induced by leader's policy. The goal of the leader is to find her optimal policy, which yields the optimal expected total return, by interacting with the follower and learning from data. A key challenge of this problem is that the leader cannot observe the follower's reward, and needs to infer the follower's quantal response model from his actions against leader's policies. We propose sample-efficient algorithms for both the online and offline settings, in the context of function approximation. Our algorithms are based on (i) learning the quantal response model via maximum likelihood estimation and (ii) model-free or model-based RL for solving the leader's decision making problem, and we show that they achieve sublinear regret upper bounds. Moreover, we quantify the uncertainty of these estimators and leverage the uncertainty to implement optimistic and pessimistic algorithms for online and offline settings. Besides, when specialized to the linear and myopic setting, our algorithms are also computationally efficient. Our theoretical analysis features a novel performance-difference lemma which incorporates the error of quantal response model, which might be of independent interest.

{{</citation>}}


### (25/94) Are Transformers with One Layer Self-Attention Using Low-Rank Weight Matrices Universal Approximators? (Tokio Kajitsuka et al., 2023)

{{<citation>}}

Tokio Kajitsuka, Issei Sato. (2023)  
**Are Transformers with One Layer Self-Attention Using Low-Rank Weight Matrices Universal Approximators?**  

---
Primary Category: cs.LG  
Categories: 68T07, I-2-0, cs-LG, cs.LG  
Keywords: Attention, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.14023v1)  

---


**ABSTRACT**  
Existing analyses of the expressive capacity of Transformer models have required excessively deep layers for data memorization, leading to a discrepancy with the Transformers actually used in practice. This is primarily due to the interpretation of the softmax function as an approximation of the hardmax function. By clarifying the connection between the softmax function and the Boltzmann operator, we prove that a single layer of self-attention with low-rank weight matrices possesses the capability to perfectly capture the context of an entire input sequence. As a consequence, we show that single-layer Transformer has a memorization capacity for finite samples, and that Transformers consisting of one self-attention layer with two feed-forward neural networks are universal approximators for continuous functions on a compact domain.

{{</citation>}}


### (26/94) Controlling the Latent Space of GANs through Reinforcement Learning: A Case Study on Task-based Image-to-Image Translation (Mahyar Abbasian et al., 2023)

{{<citation>}}

Mahyar Abbasian, Taha Rajabzadeh, Ahmadreza Moradipari, Seyed Amir Hossein Aqajari, Hongsheng Lu, Amir Rahmani. (2023)  
**Controlling the Latent Space of GANs through Reinforcement Learning: A Case Study on Task-based Image-to-Image Translation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13978v1)  

---


**ABSTRACT**  
Generative Adversarial Networks (GAN) have emerged as a formidable AI tool to generate realistic outputs based on training datasets. However, the challenge of exerting control over the generation process of GANs remains a significant hurdle. In this paper, we propose a novel methodology to address this issue by integrating a reinforcement learning (RL) agent with a latent-space GAN (l-GAN), thereby facilitating the generation of desired outputs. More specifically, we have developed an actor-critic RL agent with a meticulously designed reward policy, enabling it to acquire proficiency in navigating the latent space of the l-GAN and generating outputs based on specified tasks. To substantiate the efficacy of our approach, we have conducted a series of experiments employing the MNIST dataset, including arithmetic addition as an illustrative task. The outcomes of these experiments serve to validate our methodology. Our pioneering integration of an RL agent with a GAN model represents a novel advancement, holding great potential for enhancing generative networks in the future.

{{</citation>}}


### (27/94) Entropy Neural Estimation for Graph Contrastive Learning (Yixuan Ma et al., 2023)

{{<citation>}}

Yixuan Ma, Xiaolin Zhang, Peng Zhang, Kun Zhan. (2023)  
**Entropy Neural Estimation for Graph Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.13944v1)  

---


**ABSTRACT**  
Contrastive learning on graphs aims at extracting distinguishable high-level representations of nodes. In this paper, we theoretically illustrate that the entropy of a dataset can be approximated by maximizing the lower bound of the mutual information across different views of a graph, \ie, entropy is estimated by a neural network. Based on this finding, we propose a simple yet effective subset sampling strategy to contrast pairwise representations between views of a dataset. In particular, we randomly sample nodes and edges from a given graph to build the input subset for a view. Two views are fed into a parameter-shared Siamese network to extract the high-dimensional embeddings and estimate the information entropy of the entire graph. For the learning process, we propose to optimize the network using two objectives, simultaneously. Concretely, the input of the contrastive loss function consists of positive and negative pairs. Our selection strategy of pairs is different from previous works and we present a novel strategy to enhance the representation ability of the graph encoder by selecting nodes based on cross-view similarities. We enrich the diversity of the positive and negative pairs by selecting highly similar samples and totally different data with the guidance of cross-view similarity scores, respectively. We also introduce a cross-view consistency constraint on the representations generated from the different views. This objective guarantees the learned representations are consistent across views from the perspective of the entire graph. We conduct extensive experiments on seven graph benchmarks, and the proposed approach achieves competitive performance compared to the current state-of-the-art methods. The source code will be publicly released once this paper is accepted.

{{</citation>}}


### (28/94) Graph Neural Networks-based Hybrid Framework For Predicting Particle Crushing Strength (Tongya Zheng et al., 2023)

{{<citation>}}

Tongya Zheng, Tianli Zhang, Qingzheng Guan, Wenjie Huang, Zunlei Feng, Mingli Song, Chun Chen. (2023)  
**Graph Neural Networks-based Hybrid Framework For Predicting Particle Crushing Strength**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.13909v1)  

---


**ABSTRACT**  
Graph Neural Networks have emerged as an effective machine learning tool for multi-disciplinary tasks such as pharmaceutical molecule classification and chemical reaction prediction, because they can model non-euclidean relationships between different entities. Particle crushing, as a significant field of civil engineering, describes the breakage of granular materials caused by the breakage of particle fragment bonds under the modeling of numerical simulations, which motivates us to characterize the mechanical behaviors of particle crushing through the connectivity of particle fragments with Graph Neural Networks (GNNs). However, there lacks an open-source large-scale particle crushing dataset for research due to the expensive costs of laboratory tests or numerical simulations. Therefore, we firstly generate a dataset with 45,000 numerical simulations and 900 particle types to facilitate the research progress of machine learning for particle crushing. Secondly, we devise a hybrid framework based on GNNs to predict particle crushing strength in a particle fragment view with the advances of state of the art GNNs. Finally, we compare our hybrid framework against traditional machine learning methods and the plain MLP to verify its effectiveness. The usefulness of different features is further discussed through the gradient attribution explanation w.r.t the predictions. Our data and code are released at https://github.com/doujiang-zheng/GNN-For-Particle-Crushing.

{{</citation>}}


### (29/94) Robustness Verification of Deep Neural Networks using Star-Based Reachability Analysis with Variable-Length Time Series Input (Neelanjana Pal et al., 2023)

{{<citation>}}

Neelanjana Pal, Diego Manzanas Lopez, Taylor T Johnson. (2023)  
**Robustness Verification of Deep Neural Networks using Star-Based Reachability Analysis with Variable-Length Time Series Input**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs-NE, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.13907v1)  

---


**ABSTRACT**  
Data-driven, neural network (NN) based anomaly detection and predictive maintenance are emerging research areas. NN-based analytics of time-series data offer valuable insights into past behaviors and estimates of critical parameters like remaining useful life (RUL) of equipment and state-of-charge (SOC) of batteries. However, input time series data can be exposed to intentional or unintentional noise when passing through sensors, necessitating robust validation and verification of these NNs. This paper presents a case study of the robustness verification approach for time series regression NNs (TSRegNN) using set-based formal methods. It focuses on utilizing variable-length input data to streamline input manipulation and enhance network architecture generalizability. The method is applied to two data sets in the Prognostics and Health Management (PHM) application areas: (1) SOC estimation of a Lithium-ion battery and (2) RUL estimation of a turbine engine. The NNs' robustness is checked using star-based reachability analysis, and several performance measures evaluate the effect of bounded perturbations in the input on network outputs, i.e., future outcomes. Overall, the paper offers a comprehensive case study for validating and verifying NN-based analytics of time-series data in real-world applications, emphasizing the importance of robustness testing for accurate and reliable predictions, especially considering the impact of noise on future outcomes.

{{</citation>}}


## cs.CV (20)



### (30/94) Open Problems in Computer Vision for Wilderness SAR and The Search for Patricia Wu-Murad (Thomas Manzini et al., 2023)

{{<citation>}}

Thomas Manzini, Robin Murphy. (2023)  
**Open Problems in Computer Vision for Wilderness SAR and The Search for Patricia Wu-Murad**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2307.14527v1)  

---


**ABSTRACT**  
This paper details the challenges in applying two computer vision systems, an EfficientDET supervised learning model and the unsupervised RX spectral classifier, to 98.9 GB of drone imagery from the Wu-Murad wilderness search and rescue (WSAR) effort in Japan and identifies 3 directions for future research. There have been at least 19 proposed approaches and 3 datasets aimed at locating missing persons in drone imagery, but only 3 approaches (2 unsupervised and 1 of an unknown structure) are referenced in the literature as having been used in an actual WSAR operation. Of these proposed approaches, the EfficientDET architecture and the unsupervised spectral RX classifier were selected as the most appropriate for this setting. The EfficientDET model was applied to the HERIDAL dataset and despite achieving performance that is statistically equivalent to the state-of-the-art, the model fails to translate to the real world in terms of false positives (e.g., identifying tree limbs and rocks as people), and false negatives (e.g., failing to identify members of the search team). The poor results in practice for algorithms that showed good results on datasets suggest 3 areas of future research: more realistic datasets for wilderness SAR, computer vision models that are capable of seamlessly handling the variety of imagery that can be collected during actual WSAR operations, and better alignment on performance measures.

{{</citation>}}


### (31/94) SuperInpaint: Learning Detail-Enhanced Attentional Implicit Representation for Super-resolutional Image Inpainting (Canyu Zhang et al., 2023)

{{<citation>}}

Canyu Zhang, Qing Guo, Xiaoguang Li, Renjie Wan, Hongkai Yu, Ivor Tsang, Song Wang. (2023)  
**SuperInpaint: Learning Detail-Enhanced Attentional Implicit Representation for Super-resolutional Image Inpainting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.14489v1)  

---


**ABSTRACT**  
In this work, we introduce a challenging image restoration task, referred to as SuperInpaint, which aims to reconstruct missing regions in low-resolution images and generate completed images with arbitrarily higher resolutions. We have found that this task cannot be effectively addressed by stacking state-of-the-art super-resolution and image inpainting methods as they amplify each other's flaws, leading to noticeable artifacts. To overcome these limitations, we propose the detail-enhanced attentional implicit representation (DEAR) that can achieve SuperInpaint with a single model, resulting in high-quality completed images with arbitrary resolutions. Specifically, we use a deep convolutional network to extract the latent embedding of an input image and then enhance the high-frequency components of the latent embedding via an adaptive high-pass filter. This leads to detail-enhanced semantic embedding. We further feed the semantic embedding into an unmask-attentional module that suppresses embeddings from ineffective masked pixels. Additionally, we extract a pixel-wise importance map that indicates which pixels should be used for image reconstruction. Given the coordinates of a pixel we want to reconstruct, we first collect its neighboring pixels in the input image and extract their detail-enhanced semantic embeddings, unmask-attentional semantic embeddings, importance values, and spatial distances to the desired pixel. Then, we feed all the above terms into an implicit representation and generate the color of the specified pixel. To evaluate our method, we extend three existing datasets for this new task and build 18 meaningful baselines using SOTA inpainting and super-resolution methods. Extensive experimental results demonstrate that our method outperforms all existing methods by a significant margin on four widely used metrics.

{{</citation>}}


### (32/94) Self-supervised Few-shot Learning for Semantic Segmentation: An Annotation-free Approach (Sanaz Karimijafarbigloo et al., 2023)

{{<citation>}}

Sanaz Karimijafarbigloo, Reza Azad, Dorit Merhof. (2023)  
**Self-supervised Few-shot Learning for Semantic Segmentation: An Annotation-free Approach**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.14446v1)  

---


**ABSTRACT**  
Few-shot semantic segmentation (FSS) offers immense potential in the field of medical image analysis, enabling accurate object segmentation with limited training data. However, existing FSS techniques heavily rely on annotated semantic classes, rendering them unsuitable for medical images due to the scarcity of annotations. To address this challenge, multiple contributions are proposed: First, inspired by spectral decomposition methods, the problem of image decomposition is reframed as a graph partitioning task. The eigenvectors of the Laplacian matrix, derived from the feature affinity matrix of self-supervised networks, are analyzed to estimate the distribution of the objects of interest from the support images. Secondly, we propose a novel self-supervised FSS framework that does not rely on any annotation. Instead, it adaptively estimates the query mask by leveraging the eigenvectors obtained from the support images. This approach eliminates the need for manual annotation, making it particularly suitable for medical images with limited annotated data. Thirdly, to further enhance the decoding of the query image based on the information provided by the support image, we introduce a multi-scale large kernel attention module. By selectively emphasizing relevant features and details, this module improves the segmentation process and contributes to better object delineation. Evaluations on both natural and medical image datasets demonstrate the efficiency and effectiveness of our method. Moreover, the proposed approach is characterized by its generality and model-agnostic nature, allowing for seamless integration with various deep architectures. The code is publicly available at \href{https://github.com/mindflow-institue/annotation_free_fewshot}{\textcolor{magenta}{GitHub}}.

{{</citation>}}


### (33/94) MAMo: Leveraging Memory and Attention for Monocular Video Depth Estimation (Rajeev Yasarla et al., 2023)

{{<citation>}}

Rajeev Yasarla, Hong Cai, Jisoo Jeong, Yunxiao Shi, Risheek Garrepalli, Fatih Porikli. (2023)  
**MAMo: Leveraging Memory and Attention for Monocular Video Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.14336v1)  

---


**ABSTRACT**  
We propose MAMo, a novel memory and attention frame-work for monocular video depth estimation. MAMo can augment and improve any single-image depth estimation networks into video depth estimation models, enabling them to take advantage of the temporal information to predict more accurate depth. In MAMo, we augment model with memory which aids the depth prediction as the model streams through the video. Specifically, the memory stores learned visual and displacement tokens of the previous time instances. This allows the depth network to cross-reference relevant features from the past when predicting depth on the current frame. We introduce a novel scheme to continuously update the memory, optimizing it to keep tokens that correspond with both the past and the present visual information. We adopt attention-based approach to process memory features where we first learn the spatio-temporal relation among the resultant visual and displacement memory tokens using self-attention module. Further, the output features of self-attention are aggregated with the current visual features through cross-attention. The cross-attended features are finally given to a decoder to predict depth on the current frame. Through extensive experiments on several benchmarks, including KITTI, NYU-Depth V2, and DDAD, we show that MAMo consistently improves monocular depth estimation networks and sets new state-of-the-art (SOTA) accuracy. Notably, our MAMo video depth estimation provides higher accuracy with lower latency, when omparing to SOTA cost-volume-based video depth models.

{{</citation>}}


### (34/94) Event-based Vision for Early Prediction of Manipulation Actions (Daniel Deniz et al., 2023)

{{<citation>}}

Daniel Deniz, Cornelia Fermuller, Eduardo Ros, Manuel Rodriguez-Alvarez, Francisco Barranco. (2023)  
**Event-based Vision for Early Prediction of Manipulation Actions**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.14332v1)  

---


**ABSTRACT**  
Neuromorphic visual sensors are artificial retinas that output sequences of asynchronous events when brightness changes occur in the scene. These sensors offer many advantages including very high temporal resolution, no motion blur and smart data compression ideal for real-time processing. In this study, we introduce an event-based dataset on fine-grained manipulation actions and perform an experimental study on the use of transformers for action prediction with events. There is enormous interest in the fields of cognitive robotics and human-robot interaction on understanding and predicting human actions as early as possible. Early prediction allows anticipating complex stages for planning, enabling effective and real-time interaction. Our Transformer network uses events to predict manipulation actions as they occur, using online inference. The model succeeds at predicting actions early on, building up confidence over time and achieving state-of-the-art classification. Moreover, the attention-based transformer architecture allows us to study the role of the spatio-temporal patterns selected by the model. Our experiments show that the Transformer network captures action dynamic features outperforming video-based approaches and succeeding with scenarios where the differences between actions lie in very subtle cues. Finally, we release the new event dataset, which is the first in the literature for manipulation action recognition. Code will be available at https://github.com/DaniDeniz/EventVisionTransformer.

{{</citation>}}


### (35/94) Sparse Double Descent in Vision Transformers: real or phantom threat? (Victor Quétu et al., 2023)

{{<citation>}}

Victor Quétu, Marta Milovanovic, Enzo Tartaglione. (2023)  
**Sparse Double Descent in Vision Transformers: real or phantom threat?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.14253v1)  

---


**ABSTRACT**  
Vision transformers (ViT) have been of broad interest in recent theoretical and empirical works. They are state-of-the-art thanks to their attention-based approach, which boosts the identification of key features and patterns within images thanks to the capability of avoiding inductive bias, resulting in highly accurate image analysis. Meanwhile, neoteric studies have reported a ``sparse double descent'' phenomenon that can occur in modern deep-learning models, where extremely over-parametrized models can generalize well. This raises practical questions about the optimal size of the model and the quest over finding the best trade-off between sparsity and performance is launched: are Vision Transformers also prone to sparse double descent? Can we find a way to avoid such a phenomenon? Our work tackles the occurrence of sparse double descent on ViTs. Despite some works that have shown that traditional architectures, like Resnet, are condemned to the sparse double descent phenomenon, for ViTs we observe that an optimally-tuned $\ell_2$ regularization relieves such a phenomenon. However, everything comes at a cost: optimal lambda will sacrifice the potential compression of the ViT.

{{</citation>}}


### (36/94) ADAPT: Efficient Multi-Agent Trajectory Prediction with Adaptation (Görkay Aydemir et al., 2023)

{{<citation>}}

Görkay Aydemir, Adil Kaan Akan, Fatma Güney. (2023)  
**ADAPT: Efficient Multi-Agent Trajectory Prediction with Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14187v1)  

---


**ABSTRACT**  
Forecasting future trajectories of agents in complex traffic scenes requires reliable and efficient predictions for all agents in the scene. However, existing methods for trajectory prediction are either inefficient or sacrifice accuracy. To address this challenge, we propose ADAPT, a novel approach for jointly predicting the trajectories of all agents in the scene with dynamic weight learning. Our approach outperforms state-of-the-art methods in both single-agent and multi-agent settings on the Argoverse and Interaction datasets, with a fraction of their computational overhead. We attribute the improvement in our performance: first, to the adaptive head augmenting the model capacity without increasing the model size; second, to our design choices in the endpoint-conditioned prediction, reinforced by gradient stopping. Our analyses show that ADAPT can focus on each agent with adaptive prediction, allowing for accurate predictions efficiently. https://KUIS-AI.github.io/adapt

{{</citation>}}


### (37/94) Resolution-Aware Design of Atrous Rates for Semantic Segmentation Networks (Bum Jun Kim et al., 2023)

{{<citation>}}

Bum Jun Kim, Hyeyeon Choi, Hyeonah Jang, Sang Woo Kim. (2023)  
**Resolution-Aware Design of Atrous Rates for Semantic Segmentation Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.14179v1)  

---


**ABSTRACT**  
DeepLab is a widely used deep neural network for semantic segmentation, whose success is attributed to its parallel architecture called atrous spatial pyramid pooling (ASPP). ASPP uses multiple atrous convolutions with different atrous rates to extract both local and global information. However, fixed values of atrous rates are used for the ASPP module, which restricts the size of its field of view. In principle, atrous rate should be a hyperparameter to change the field of view size according to the target task or dataset. However, the manipulation of atrous rate is not governed by any guidelines. This study proposes practical guidelines for obtaining an optimal atrous rate. First, an effective receptive field for semantic segmentation is introduced to analyze the inner behavior of segmentation networks. We observed that the use of ASPP module yielded a specific pattern in the effective receptive field, which was traced to reveal the module's underlying mechanism. Accordingly, we derive practical guidelines for obtaining the optimal atrous rate, which should be controlled based on the size of input image. Compared to other values, using the optimal atrous rate consistently improved the segmentation results across multiple datasets, including the STARE, CHASE_DB1, HRF, Cityscapes, and iSAID datasets.

{{</citation>}}


### (38/94) LOIS: Looking Out of Instance Semantics for Visual Question Answering (Siyu Zhang et al., 2023)

{{<citation>}}

Siyu Zhang, Yeming Chen, Yaoru Sun, Fang Wang, Haibo Shi, Haoran Wang. (2023)  
**LOIS: Looking Out of Instance Semantics for Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.14142v1)  

---


**ABSTRACT**  
Visual question answering (VQA) has been intensively studied as a multimodal task that requires effort in bridging vision and language to infer answers correctly. Recent attempts have developed various attention-based modules for solving VQA tasks. However, the performance of model inference is largely bottlenecked by visual processing for semantics understanding. Most existing detection methods rely on bounding boxes, remaining a serious challenge for VQA models to understand the causal nexus of object semantics in images and correctly infer contextual information. To this end, we propose a finer model framework without bounding boxes in this work, termed Looking Out of Instance Semantics (LOIS) to tackle this important issue. LOIS enables more fine-grained feature descriptions to produce visual facts. Furthermore, to overcome the label ambiguity caused by instance masks, two types of relation attention modules: 1) intra-modality and 2) inter-modality, are devised to infer the correct answers from the different multi-view features. Specifically, we implement a mutual relation attention module to model sophisticated and deeper visual semantic relations between instance objects and background information. In addition, our proposed attention model can further analyze salient image regions by focusing on important word-related questions. Experimental results on four benchmark VQA datasets prove that our proposed method has favorable performance in improving visual reasoning capability.

{{</citation>}}


### (39/94) Creative Birds: Self-Supervised Single-View 3D Style Transfer (Renke Wang et al., 2023)

{{<citation>}}

Renke Wang, Guimin Que, Shuo Chen, Xiang Li, Jun Li, Jian Yang. (2023)  
**Creative Birds: Self-Supervised Single-View 3D Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised, Style Transfer  
[Paper Link](http://arxiv.org/abs/2307.14127v2)  

---


**ABSTRACT**  
In this paper, we propose a novel method for single-view 3D style transfer that generates a unique 3D object with both shape and texture transfer. Our focus lies primarily on birds, a popular subject in 3D reconstruction, for which no existing single-view 3D transfer methods have been developed.The method we propose seeks to generate a 3D mesh shape and texture of a bird from two single-view images. To achieve this, we introduce a novel shape transfer generator that comprises a dual residual gated network (DRGNet), and a multi-layer perceptron (MLP). DRGNet extracts the features of source and target images using a shared coordinate gate unit, while the MLP generates spatial coordinates for building a 3D mesh. We also introduce a semantic UV texture transfer module that implements textural style transfer using semantic UV segmentation, which ensures consistency in the semantic meaning of the transferred regions. This module can be widely adapted to many existing approaches. Finally, our method constructs a novel 3D bird using a differentiable renderer. Experimental results on the CUB dataset verify that our method achieves state-of-the-art performance on the single-view 3D style transfer task. Code is available in https://github.com/wrk226/creative_birds.

{{</citation>}}


### (40/94) Memory-Efficient Graph Convolutional Networks for Object Classification and Detection with Event Cameras (Kamil Jeziorek et al., 2023)

{{<citation>}}

Kamil Jeziorek, Andrea Pinna, Tomasz Kryjak. (2023)  
**Memory-Efficient Graph Convolutional Networks for Object Classification and Detection with Event Cameras**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2307.14124v1)  

---


**ABSTRACT**  
Recent advances in event camera research emphasize processing data in its original sparse form, which allows the use of its unique features such as high temporal resolution, high dynamic range, low latency, and resistance to image blur. One promising approach for analyzing event data is through graph convolutional networks (GCNs). However, current research in this domain primarily focuses on optimizing computational costs, neglecting the associated memory costs. In this paper, we consider both factors together in order to achieve satisfying results and relatively low model complexity. For this purpose, we performed a comparative analysis of different graph convolution operations, considering factors such as execution time, the number of trainable model parameters, data format requirements, and training outcomes. Our results show a 450-fold reduction in the number of parameters for the feature extraction module and a 4.5-fold reduction in the size of the data representation while maintaining a classification accuracy of 52.3%, which is 6.3% higher compared to the operation used in state-of-the-art approaches. To further evaluate performance, we implemented the object detection architecture and evaluated its performance on the N-Caltech101 dataset. The results showed an accuracy of 53.7 % mAP@0.5 and reached an execution rate of 82 graphs per second.

{{</citation>}}


### (41/94) A semantics-driven methodology for high-quality image annotation (Fausto Giunchiglia et al., 2023)

{{<citation>}}

Fausto Giunchiglia, Mayukh Bagchi, Xiaolei Diao. (2023)  
**A semantics-driven methodology for high-quality image annotation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MM, cs.CV  
Keywords: Computer Vision, ImageNet, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.14119v1)  

---


**ABSTRACT**  
Recent work in Machine Learning and Computer Vision has highlighted the presence of various types of systematic flaws inside ground truth object recognition benchmark datasets. Our basic tenet is that these flaws are rooted in the many-to-many mappings which exist between the visual information encoded in images and the intended semantics of the labels annotating them. The net consequence is that the current annotation process is largely under-specified, thus leaving too much freedom to the subjective judgment of annotators. In this paper, we propose vTelos, an integrated Natural Language Processing, Knowledge Representation, and Computer Vision methodology whose main goal is to make explicit the (otherwise implicit) intended annotation semantics, thus minimizing the number and role of subjective choices. A key element of vTelos is the exploitation of the WordNet lexico-semantic hierarchy as the main means for providing the meaning of natural language labels and, as a consequence, for driving the annotation of images based on the objects and the visual properties they depict. The methodology is validated on images populating a subset of the ImageNet hierarchy.

{{</citation>}}


### (42/94) ECO: Ensembling Context Optimization for Vision-Language Models (Lorenzo Agnolucci et al., 2023)

{{<citation>}}

Lorenzo Agnolucci, Alberto Baldrati, Francesco Todino, Federico Becattini, Marco Bertini, Alberto Del Bimbo. (2023)  
**ECO: Ensembling Context Optimization for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.14063v1)  

---


**ABSTRACT**  
Image recognition has recently witnessed a paradigm shift, where vision-language models are now used to perform few-shot classification based on textual prompts. Among these, the CLIP model has shown remarkable capabilities for zero-shot transfer by matching an image and a custom textual prompt in its latent space. This has paved the way for several works that focus on engineering or learning textual contexts for maximizing CLIP's classification capabilities. In this paper, we follow this trend by learning an ensemble of prompts for image classification. We show that learning diverse and possibly shorter contexts improves considerably and consistently the results rather than relying on a single trainable prompt. In particular, we report better few-shot capabilities with no additional cost at inference time. We demonstrate the capabilities of our approach on 11 different benchmarks.

{{</citation>}}


### (43/94) ESSAformer: Efficient Transformer for Hyperspectral Image Super-resolution (Mingjin Zhang et al., 2023)

{{<citation>}}

Mingjin Zhang, Chi Zhang, Qiming Zhang, Jie Guo, Xinbo Gao, Jing Zhang. (2023)  
**ESSAformer: Efficient Transformer for Hyperspectral Image Super-resolution**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.14010v1)  

---


**ABSTRACT**  
Single hyperspectral image super-resolution (single-HSI-SR) aims to restore a high-resolution hyperspectral image from a low-resolution observation. However, the prevailing CNN-based approaches have shown limitations in building long-range dependencies and capturing interaction information between spectral features. This results in inadequate utilization of spectral information and artifacts after upsampling. To address this issue, we propose ESSAformer, an ESSA attention-embedded Transformer network for single-HSI-SR with an iterative refining structure. Specifically, we first introduce a robust and spectral-friendly similarity metric, \ie, the spectral correlation coefficient of the spectrum (SCC), to replace the original attention matrix and incorporates inductive biases into the model to facilitate training. Built upon it, we further utilize the kernelizable attention technique with theoretical support to form a novel efficient SCC-kernel-based self-attention (ESSA) and reduce attention computation to linear complexity. ESSA enlarges the receptive field for features after upsampling without bringing much computation and allows the model to effectively utilize spatial-spectral information from different scales, resulting in the generation of more natural high-resolution images. Without the need for pretraining on large-scale datasets, our experiments demonstrate ESSA's effectiveness in both visual quality and quantitative results.

{{</citation>}}


### (44/94) Learning Snippet-to-Motion Progression for Skeleton-based Human Motion Prediction (Xinshun Wang et al., 2023)

{{<citation>}}

Xinshun Wang, Qiongjie Cui, Chen Chen, Shen Zhao, Mengyuan Liu. (2023)  
**Learning Snippet-to-Motion Progression for Skeleton-based Human Motion Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2307.14006v1)  

---


**ABSTRACT**  
Existing Graph Convolutional Networks to achieve human motion prediction largely adopt a one-step scheme, which output the prediction straight from history input, failing to exploit human motion patterns. We observe that human motions have transitional patterns and can be split into snippets representative of each transition. Each snippet can be reconstructed from its starting and ending poses referred to as the transitional poses. We propose a snippet-to-motion multi-stage framework that breaks motion prediction into sub-tasks easier to accomplish. Each sub-task integrates three modules: transitional pose prediction, snippet reconstruction, and snippet-to-motion prediction. Specifically, we propose to first predict only the transitional poses. Then we use them to reconstruct the corresponding snippets, obtaining a close approximation to the true motion sequence. Finally we refine them to produce the final prediction output. To implement the network, we propose a novel unified graph modeling, which allows for direct and effective feature propagation compared to existing approaches which rely on separate space-time modeling. Extensive experiments on Human 3.6M, CMU Mocap and 3DPW datasets verify the effectiveness of our method which achieves state-of-the-art performance.

{{</citation>}}


### (45/94) Analysis of Video Quality Datasets via Design of Minimalistic Video Quality Models (Wei Sun et al., 2023)

{{<citation>}}

Wei Sun, Wen Wen, Xiongkuo Min, Long Lan, Guangtao Zhai, Kede Ma. (2023)  
**Analysis of Video Quality Datasets via Design of Minimalistic Video Quality Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.13981v1)  

---


**ABSTRACT**  
Blind video quality assessment (BVQA) plays an indispensable role in monitoring and improving the end-users' viewing experience in various real-world video-enabled media applications. As an experimental field, the improvements of BVQA models have been measured primarily on a few human-rated VQA datasets. Thus, it is crucial to gain a better understanding of existing VQA datasets in order to properly evaluate the current progress in BVQA. Towards this goal, we conduct a first-of-its-kind computational analysis of VQA datasets via designing minimalistic BVQA models. By minimalistic, we restrict our family of BVQA models to build only upon basic blocks: a video preprocessor (for aggressive spatiotemporal downsampling), a spatial quality analyzer, an optional temporal quality analyzer, and a quality regressor, all with the simplest possible instantiations. By comparing the quality prediction performance of different model variants on eight VQA datasets with realistic distortions, we find that nearly all datasets suffer from the easy dataset problem of varying severity, some of which even admit blind image quality assessment (BIQA) solutions. We additionally justify our claims by contrasting our model generalizability on these VQA datasets, and by ablating a dizzying set of BVQA design choices related to the basic building blocks. Our results cast doubt on the current progress in BVQA, and meanwhile shed light on good practices of constructing next-generation VQA datasets and models.

{{</citation>}}


### (46/94) Improving Semi-Supervised Semantic Segmentation with Dual-Level Siamese Structure Network (Zhibo Tain et al., 2023)

{{<citation>}}

Zhibo Tain, Xiaolin Zhang, Peng Zhang, Kun Zhan. (2023)  
**Improving Semi-Supervised Semantic Segmentation with Dual-Level Siamese Structure Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.13938v1)  

---


**ABSTRACT**  
Semi-supervised semantic segmentation (SSS) is an important task that utilizes both labeled and unlabeled data to reduce expenses on labeling training examples. However, the effectiveness of SSS algorithms is limited by the difficulty of fully exploiting the potential of unlabeled data. To address this, we propose a dual-level Siamese structure network (DSSN) for pixel-wise contrastive learning. By aligning positive pairs with a pixel-wise contrastive loss using strong augmented views in both low-level image space and high-level feature space, the proposed DSSN is designed to maximize the utilization of available unlabeled data. Additionally, we introduce a novel class-aware pseudo-label selection strategy for weak-to-strong supervision, which addresses the limitations of most existing methods that do not perform selection or apply a predefined threshold for all classes. Specifically, our strategy selects the top high-confidence prediction of the weak view for each class to generate pseudo labels that supervise the strong augmented views. This strategy is capable of taking into account the class imbalance and improving the performance of long-tailed classes. Our proposed method achieves state-of-the-art results on two datasets, PASCAL VOC 2012 and Cityscapes, outperforming other SSS algorithms by a significant margin.

{{</citation>}}


### (47/94) AIDE: A Vision-Driven Multi-View, Multi-Modal, Multi-Tasking Dataset for Assistive Driving Perception (Dingkang Yang et al., 2023)

{{<citation>}}

Dingkang Yang, Shuai Huang, Zhi Xu, Zhenpeng Li, Shunli Wang, Mingcheng Li, Yuzheng Wang, Yang Liu, Kun Yang, Zhaoyu Chen, Yan Wang, Jing Liu, Peixuan Zhang, Peng Zhai, Lihua Zhang. (2023)  
**AIDE: A Vision-Driven Multi-View, Multi-Modal, Multi-Tasking Dataset for Assistive Driving Perception**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13933v1)  

---


**ABSTRACT**  
Driver distraction has become a significant cause of severe traffic accidents over the past decade. Despite the growing development of vision-driven driver monitoring systems, the lack of comprehensive perception datasets restricts road safety and traffic security. In this paper, we present an AssIstive Driving pErception dataset (AIDE) that considers context information both inside and outside the vehicle in naturalistic scenarios. AIDE facilitates holistic driver monitoring through three distinctive characteristics, including multi-view settings of driver and scene, multi-modal annotations of face, body, posture, and gesture, and four pragmatic task designs for driving understanding. To thoroughly explore AIDE, we provide experimental benchmarks on three kinds of baseline frameworks via extensive methods. Moreover, two fusion strategies are introduced to give new insights into learning effective multi-stream/modal representations. We also systematically investigate the importance and rationality of the key components in AIDE and benchmarks. The project link is https://github.com/ydk122024/AIDE.

{{</citation>}}


### (48/94) EasyNet: An Easy Network for 3D Industrial Anomaly Detection (Ruitao Chen et al., 2023)

{{<citation>}}

Ruitao Chen, Guoyang Xie, Jiaqi Liu, Jinbao Wang, Ziqi Luo, Jinfan Wang, Feng Zheng. (2023)  
**EasyNet: An Easy Network for 3D Industrial Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.13925v2)  

---


**ABSTRACT**  
3D anomaly detection is an emerging and vital computer vision task in industrial manufacturing (IM). Recently many advanced algorithms have been published, but most of them cannot meet the needs of IM. There are several disadvantages: i) difficult to deploy on production lines since their algorithms heavily rely on large pre-trained models; ii) hugely increase storage overhead due to overuse of memory banks; iii) the inference speed cannot be achieved in real-time. To overcome these issues, we propose an easy and deployment-friendly network (called EasyNet) without using pre-trained models and memory banks: firstly, we design a multi-scale multi-modality feature encoder-decoder to accurately reconstruct the segmentation maps of anomalous regions and encourage the interaction between RGB images and depth images; secondly, we adopt a multi-modality anomaly segmentation network to achieve a precise anomaly map; thirdly, we propose an attention-based information entropy fusion module for feature fusion during inference, making it suitable for real-time deployment. Extensive experiments show that EasyNet achieves an anomaly detection AUROC of 92.6% without using pre-trained models and memory banks. In addition, EasyNet is faster than existing methods, with a high frame rate of 94.55 FPS on a Tesla V100 GPU.

{{</citation>}}


### (49/94) AViT: Adapting Vision Transformers for Small Skin Lesion Segmentation Datasets (Siyi Du et al., 2023)

{{<citation>}}

Siyi Du, Nourhan Bayasi, Ghassan Harmarneh, Rafeef Garbi. (2023)  
**AViT: Adapting Vision Transformers for Small Skin Lesion Segmentation Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.13897v1)  

---


**ABSTRACT**  
Skin lesion segmentation (SLS) plays an important role in skin lesion analysis. Vision transformers (ViTs) are considered an auspicious solution for SLS, but they require more training data compared to convolutional neural networks (CNNs) due to their inherent parameter-heavy structure and lack of some inductive biases. To alleviate this issue, current approaches fine-tune pre-trained ViT backbones on SLS datasets, aiming to leverage the knowledge learned from a larger set of natural images to lower the amount of skin training data needed. However, fully fine-tuning all parameters of large backbones is computationally expensive and memory intensive. In this paper, we propose AViT, a novel efficient strategy to mitigate ViTs' data-hunger by transferring any pre-trained ViTs to the SLS task. Specifically, we integrate lightweight modules (adapters) within the transformer layers, which modulate the feature representation of a ViT without updating its pre-trained weights. In addition, we employ a shallow CNN as a prompt generator to create a prompt embedding from the input image, which grasps fine-grained information and CNN's inductive biases to guide the segmentation task on small datasets. Our quantitative experiments on 4 skin lesion datasets demonstrate that AViT achieves competitive, and at times superior, performance to SOTA but with significantly fewer trainable parameters. Our code is available at https://github.com/siyi-wind/AViT.

{{</citation>}}


## cs.HC (5)



### (50/94) Words That Stick: Predicting Decision Making and Synonym Engagement Using Cognitive Biases and Computational Linguistics (Nimrod Dvir et al., 2023)

{{<citation>}}

Nimrod Dvir, Elaine Friedman, Suraj Commuri, Fan Yang, Jennifer Romano. (2023)  
**Words That Stick: Predicting Decision Making and Synonym Engagement Using Cognitive Biases and Computational Linguistics**  

---
Primary Category: cs.HC  
Categories: 03B65, H-5; I-7, cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: Bias, NLP  
[Paper Link](http://arxiv.org/abs/2307.14511v1)  

---


**ABSTRACT**  
This research draws upon cognitive psychology and information systems studies to anticipate user engagement and decision-making on digital platforms. By employing natural language processing (NLP) techniques and insights from cognitive bias research, we delve into user interactions with synonyms within digital content. Our methodology synthesizes four cognitive biasesRepresentativeness, Ease-of-use, Affect, and Distributioninto the READ model. Through a comprehensive user survey, we assess the model's ability to predict user engagement, discovering that synonyms that accurately represent core ideas, are easy to understand, elicit emotional responses, and are commonly encountered, promote greater user engagement. Crucially, our work offers a fresh lens on human-computer interaction, digital behaviors, and decision-making processes. Our results highlight the promise of cognitive biases as potent indicators of user engagement, underscoring their significance in designing effective digital content across fields like education and marketing.

{{</citation>}}


### (51/94) A Predictive Model of Digital Information Engagement: Forecasting User Engagement With English Words by Incorporating Cognitive Biases, Computational Linguistics and Natural Language Processing (Nimrod Dvir et al., 2023)

{{<citation>}}

Nimrod Dvir, Elaine Friedman, Suraj Commuri, Fan yang, Jennifer Romano. (2023)  
**A Predictive Model of Digital Information Engagement: Forecasting User Engagement With English Words by Incorporating Cognitive Biases, Computational Linguistics and Natural Language Processing**  

---
Primary Category: cs.HC  
Categories: 68U15, H-5; H-5-1; H-5-2; D-3-2, cs-CL, cs-HC, cs-LG, cs.HC  
Keywords: AI, Bias, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.14500v1)  

---


**ABSTRACT**  
This study introduces and empirically tests a novel predictive model for digital information engagement (IE) - the READ model, an acronym for the four pivotal attributes of engaging information: Representativeness, Ease-of-use, Affect, and Distribution. Conceptualized within the theoretical framework of Cumulative Prospect Theory, the model integrates key cognitive biases with computational linguistics and natural language processing to develop a multidimensional perspective on information engagement. A rigorous testing protocol was implemented, involving 50 randomly selected pairs of synonymous words (100 words in total) from the WordNet database. These words' engagement levels were evaluated through a large-scale online survey (n = 80,500) to derive empirical IE metrics. The READ attributes for each word were then computed and their predictive efficacy examined. The findings affirm the READ model's robustness, accurately predicting a word's IE level and distinguishing the more engaging word from a pair of synonyms with an 84% accuracy rate. The READ model's potential extends across various domains, including business, education, government, and healthcare, where it could enhance content engagement and inform AI language model development and generative text work. Future research should address the model's scalability and adaptability across different domains and languages, thereby broadening its applicability and efficacy.

{{</citation>}}


### (52/94) AI and Education: An Investigation into the Use of ChatGPT for Systems Thinking (Holger Arndt, 2023)

{{<citation>}}

Holger Arndt. (2023)  
**AI and Education: An Investigation into the Use of ChatGPT for Systems Thinking**  

---
Primary Category: cs.HC  
Categories: 91, cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.14206v1)  

---


**ABSTRACT**  
This exploratory study investigates the potential of the artificial intelligence tool, ChatGPT, to support systems thinking (ST) in various subjects. Using both general and subject specific prompts, the study assesses the accuracy, helpfulness, and reliability of ChatGPT's responses across different versions of the tool. The results indicate that ChatGPT can provide largely correct and very helpful responses in various subjects, demonstrating its potential as a tool for enhancing ST skills. However, occasional inaccuracies highlight the need for users to remain critical of ChatGPT's responses. Despite some limitations, this study suggests that with careful use and attention to its idiosyncrasies, ChatGPT can be a valuable tool for teaching and learning ST.

{{</citation>}}


### (53/94) Leveraging Large Language Models for Mental Health Prediction via Online Text Data (Xuhai Xu et al., 2023)

{{<citation>}}

Xuhai Xu, Bingshen Yao, Yuanzhe Dong, Hong Yu, James Hendler, Anind K. Dey, Dakuo Wang. (2023)  
**Leveraging Large Language Models for Mental Health Prediction via Online Text Data**  

---
Primary Category: cs.HC  
Categories: 68U35, H-5-2; I-2-m, cs-CL, cs-HC, cs.HC  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2307.14385v1)  

---


**ABSTRACT**  
The recent technology boost of large language models (LLMs) has empowered a variety of applications. However, there is very little research on understanding and improving LLMs' capability for the mental health domain. In this work, we present the first comprehensive evaluation of multiple LLMs, including Alpaca, Alpaca-LoRA, and GPT-3.5, on various mental health prediction tasks via online text data. We conduct a wide range of experiments, covering zero-shot prompting, few-shot prompting, and instruction finetuning. The results indicate the promising yet limited performance of LLMs with zero-shot and few-shot prompt designs for mental health tasks. More importantly, our experiments show that instruction finetuning can significantly boost the performance of LLMs for all tasks simultaneously. Our best-finetuned model, Mental-Alpaca, outperforms GPT-3.5 (25 times bigger) by 16.7\% on balanced accuracy and performs on par with the state-of-the-art task-specific model. We summarize our findings into a set of action guidelines for future researchers, engineers, and practitioners on how to empower LLMs with better mental health domain knowledge and become an expert in mental health prediction tasks.

{{</citation>}}


### (54/94) Embedding Democratic Values into Social Media AIs via Societal Objective Functions (Chenyan Jia et al., 2023)

{{<citation>}}

Chenyan Jia, Michelle S. Lam, Minh Chau Mai, Jeff Hancock, Michael S. Bernstein. (2023)  
**Embedding Democratic Values into Social Media AIs via Societal Objective Functions**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, Embedding, Social Media  
[Paper Link](http://arxiv.org/abs/2307.13912v1)  

---


**ABSTRACT**  
Can we design artificial intelligence (AI) systems that rank our social media feeds to consider democratic values such as mitigating partisan animosity as part of their objective functions? We introduce a method for translating established, vetted social scientific constructs into AI objective functions, which we term societal objective functions, and demonstrate the method with application to the political science construct of anti-democratic attitudes. Traditionally, we have lacked observable outcomes to use to train such models, however, the social sciences have developed survey instruments and qualitative codebooks for these constructs, and their precision facilitates translation into detailed prompts for large language models. We apply this method to create a democratic attitude model that estimates the extent to which a social media post promotes anti-democratic attitudes, and test this democratic attitude model across three studies. In Study 1, we first test the attitudinal and behavioral effectiveness of the intervention among US partisans (N=1,380) by manually annotating (alpha=.895) social media posts with anti-democratic attitude scores and testing several feed ranking conditions based on these scores. Removal (d=.20) and downranking feeds (d=.25) reduced participants' partisan animosity without compromising their experience and engagement. In Study 2, we scale up the manual labels by creating the democratic attitude model, finding strong agreement with manual labels (rho=.75). Finally, in Study 3, we replicate Study 1 using the democratic attitude model instead of manual labels to test its attitudinal and behavioral impact (N=558), and again find that the feed downranking using the societal objective function reduced partisan animosity (d=.25). This method presents a novel strategy to draw on social science theory and methods to mitigate societal harms in social media AIs.

{{</citation>}}


## cs.RO (4)



### (55/94) Attention of Robot Touch: Tactile Saliency Prediction for Robust Sim-to-Real Tactile Control (Yijiong Lin et al., 2023)

{{<citation>}}

Yijiong Lin, Mauro Comi, Alex Church, Dandan Zhang, Nathan F. Lepora. (2023)  
**Attention of Robot Touch: Tactile Saliency Prediction for Robust Sim-to-Real Tactile Control**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.14510v1)  

---


**ABSTRACT**  
High-resolution tactile sensing can provide accurate information about local contact in contact-rich robotic tasks. However, the deployment of such tasks in unstructured environments remains under-investigated. To improve the robustness of tactile robot control in unstructured environments, we propose and study a new concept: \textit{tactile saliency} for robot touch, inspired by the human touch attention mechanism from neuroscience and the visual saliency prediction problem from computer vision. In analogy to visual saliency, this concept involves identifying key information in tactile images captured by a tactile sensor. While visual saliency datasets are commonly annotated by humans, manually labelling tactile images is challenging due to their counterintuitive patterns. To address this challenge, we propose a novel approach comprised of three interrelated networks: 1) a Contact Depth Network (ConDepNet), which generates a contact depth map to localize deformation in a real tactile image that contains target and noise features; 2) a Tactile Saliency Network (TacSalNet), which predicts a tactile saliency map to describe the target areas for an input contact depth map; 3) and a Tactile Noise Generator (TacNGen), which generates noise features to train the TacSalNet. Experimental results in contact pose estimation and edge-following in the presence of distractors showcase the accurate prediction of target features from real tactile images. Overall, our tactile saliency prediction approach gives robust sim-to-real tactile control in environments with unknown distractors. Project page: https://sites.google.com/view/tactile-saliency/.

{{</citation>}}


### (56/94) Sim-to-Real Model-Based and Model-Free Deep Reinforcement Learning for Tactile Pushing (Max Yang et al., 2023)

{{<citation>}}

Max Yang, Yijiong Lin, Alex Church, John Lloyd, Dandan Zhang, David A. W. Barton, Nathan F. Lepora. (2023)  
**Sim-to-Real Model-Based and Model-Free Deep Reinforcement Learning for Tactile Pushing**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.14272v1)  

---


**ABSTRACT**  
Object pushing presents a key non-prehensile manipulation problem that is illustrative of more complex robotic manipulation tasks. While deep reinforcement learning (RL) methods have demonstrated impressive learning capabilities using visual input, a lack of tactile sensing limits their capability for fine and reliable control during manipulation. Here we propose a deep RL approach to object pushing using tactile sensing without visual input, namely tactile pushing. We present a goal-conditioned formulation that allows both model-free and model-based RL to obtain accurate policies for pushing an object to a goal. To achieve real-world performance, we adopt a sim-to-real approach. Our results demonstrate that it is possible to train on a single object and a limited sample of goals to produce precise and reliable policies that can generalize to a variety of unseen objects and pushing scenarios without domain randomization. We experiment with the trained agents in harsh pushing conditions, and show that with significantly more training samples, a model-free policy can outperform a model-based planner, generating shorter and more reliable pushing trajectories despite large disturbances. The simplicity of our training environment and effective real-world performance highlights the value of rich tactile information for fine manipulation. Code and videos are available at https://sites.google.com/view/tactile-rl-pushing/.

{{</citation>}}


### (57/94) MorphoLander: Reinforcement Learning Based Landing of a Group of Drones on the Adaptive Morphogenetic UAV (Sausar Karaf et al., 2023)

{{<citation>}}

Sausar Karaf, Aleksey Fedoseev, Mikhail Martynov, Zhanibek Darush, Aleksei Shcherbak, Dzmitry Tsetserukou. (2023)  
**MorphoLander: Reinforcement Learning Based Landing of a Group of Drones on the Adaptive Morphogenetic UAV**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.14147v2)  

---


**ABSTRACT**  
This paper focuses on a novel robotic system MorphoLander representing heterogeneous swarm of drones for exploring rough terrain environments. The morphogenetic leader drone is capable of landing on uneven terrain, traversing it, and maintaining horizontal position to deploy smaller drones for extensive area exploration. After completing their tasks, these drones return and land back on the landing pads of MorphoGear. The reinforcement learning algorithm was developed for a precise landing of drones on the leader robot that either remains static during their mission or relocates to the new position. Several experiments were conducted to evaluate the performance of the developed landing algorithm under both even and uneven terrain conditions. The experiments revealed that the proposed system results in high landing accuracy of 0.5 cm when landing on the leader drone under even terrain conditions and 2.35 cm under uneven terrain conditions. MorphoLander has the potential to significantly enhance the efficiency of the industrial inspections, seismic surveys, and rescue missions in highly cluttered and unstructured environments.

{{</citation>}}


### (58/94) Research on Inertial Navigation Technology of Unmanned Aerial Vehicles with Integrated Reinforcement Learning Algorithm (Longcheng Guo, 2023)

{{<citation>}}

Longcheng Guo. (2023)  
**Research on Inertial Navigation Technology of Unmanned Aerial Vehicles with Integrated Reinforcement Learning Algorithm**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.14038v1)  

---


**ABSTRACT**  
We first define appropriate state representation and action space, and then design an adjustment mechanism based on the actions selected by the intelligent agent. The adjustment mechanism outputs the next state and reward value of the agent. Additionally, the adjustment mechanism calculates the error between the adjusted state and the unadjusted state. Furthermore, the intelligent agent stores the acquired experience samples containing states and reward values in a buffer and replays the experiences during each iteration to learn the dynamic characteristics of the environment. We name the improved algorithm as the DQM algorithm. Experimental results demonstrate that the intelligent agent using our proposed algorithm effectively reduces the accumulated errors of inertial navigation in dynamic environments. Although our research provides a basis for achieving autonomous navigation of unmanned aerial vehicles, there is still room for significant optimization. Further research can include testing unmanned aerial vehicles in simulated environments, testing unmanned aerial vehicles in real-world environments, optimizing the design of reward functions, improving the algorithm workflow to enhance convergence speed and performance, and enhancing the algorithm's generalization ability.

{{</citation>}}


## cs.MM (3)



### (59/94) Modality-Agnostic Audio-Visual Deepfake Detection (Cai Yu et al., 2023)

{{<citation>}}

Cai Yu, Peng Chen, Jiahe Tian, Jin Liu, Jiao Dai, Xi Wang, Yesheng Chai, Jizhong Han. (2023)  
**Modality-Agnostic Audio-Visual Deepfake Detection**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs-SD, cs.MM, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14491v1)  

---


**ABSTRACT**  
As AI-generated content (AIGC) thrives, Deepfakes have expanded from single-modality falsification to cross-modal fake content creation, where either audio or visual components can be manipulated. While using two unimodal detectors can detect audio-visual deepfakes, cross-modal forgery clues could be overlooked. Existing multimodal deepfake detection methods typically establish correspondence between the audio and visual modalities for binary real/fake classification, and require the co-occurrence of both modalities. However, in real-world multi-modal applications, missing modality scenarios may occur where either modality is unavailable. In such cases, audio-visual detection methods are less practical than two independent unimodal methods. Consequently, the detector can not always obtain the number or type of manipulated modalities beforehand, necessitating a fake-modality-agnostic audio-visual detector. In this work, we propose a unified fake-modality-agnostic scenarios framework that enables the detection of multimodal deepfakes and handles missing modalities cases, no matter the manipulation hidden in audio, video, or even cross-modal forms. To enhance the modeling of cross-modal forgery clues, we choose audio-visual speech recognition (AVSR) as a preceding task, which effectively extracts speech correlation across modalities, which is difficult for deepfakes to reproduce. Additionally, we propose a dual-label detection approach that follows the structure of AVSR to support the independent detection of each modality. Extensive experiments show that our scheme not only outperforms other state-of-the-art binary detection methods across all three audio-visual datasets but also achieves satisfying performance on detection modality-agnostic audio/video fakes. Moreover, it even surpasses the joint use of two unimodal methods in the presence of missing modality cases.

{{</citation>}}


### (60/94) Neural-based Cross-modal Search and Retrieval of Artwork (Yan Gong et al., 2023)

{{<citation>}}

Yan Gong, Georgina Cosma, Axel Finke. (2023)  
**Neural-based Cross-modal Search and Retrieval of Artwork**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.14244v1)  

---


**ABSTRACT**  
Creating an intelligent search and retrieval system for artwork images, particularly paintings, is crucial for documenting cultural heritage, fostering wider public engagement, and advancing artistic analysis and interpretation. Visual-Semantic Embedding (VSE) networks are deep learning models used for information retrieval, which learn joint representations of textual and visual data, enabling 1) cross-modal search and retrieval tasks, such as image-to-text and text-to-image retrieval; and 2) relation-focused retrieval to capture entity relationships and provide more contextually relevant search results. Although VSE networks have played a significant role in cross-modal information retrieval, their application to painting datasets, such as ArtUK, remains unexplored. This paper introduces BoonArt, a VSE-based cross-modal search engine that allows users to search for images using textual queries, and to obtain textual descriptions along with the corresponding images when using image queries. The performance of BoonArt was evaluated using the ArtUK dataset. Experimental evaluations revealed that BoonArt achieved 97% Recall@10 for image-to-text retrieval, and 97.4% Recall@10 for text-to-image Retrieval. By bridging the gap between textual and visual modalities, BoonArt provides a much-improved search performance compared to traditional search engines, such as the one provided by the ArtUK website. BoonArt can be utilised to work with other artwork datasets.

{{</citation>}}


### (61/94) Boon: A Neural Search Engine for Cross-Modal Information Retrieval (Yan Gong et al., 2023)

{{<citation>}}

Yan Gong, Georgina Cosma. (2023)  
**Boon: A Neural Search Engine for Cross-Modal Information Retrieval**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: AI, Embedding, GPT, GPT-3.5, Google, Information Retrieval, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.14240v1)  

---


**ABSTRACT**  
Visual-Semantic Embedding (VSE) networks can help search engines better understand the meaning behind visual content and associate it with relevant textual information, leading to more accurate search results. VSE networks can be used in cross-modal search engines to embed image and textual descriptions in a shared space, enabling image-to-text and text-to-image retrieval tasks. However, the full potential of VSE networks for search engines has yet to be fully explored. This paper presents Boon, a novel cross-modal search engine that combines two state-of-the-art networks: the GPT-3.5-turbo large language model, and the VSE network VITR (VIsion Transformers with Relation-focused learning) to enhance the engine's capabilities in extracting and reasoning with regional relationships in images. VITR employs encoders from CLIP that were trained with 400 million image-description pairs and it was fine-turned on the RefCOCOg dataset. Boon's neural-based components serve as its main functionalities: 1) a 'cross-modal search engine' that enables end-users to perform image-to-text and text-to-image retrieval. 2) a 'multi-lingual conversational AI' component that enables the end-user to converse about one or more images selected by the end-user. Such a feature makes the search engine accessible to a wide audience, including those with visual impairments. 3) Boon is multi-lingual and can take queries and handle conversations about images in multiple languages. Boon was implemented using the Django and PyTorch frameworks. The interface and capabilities of the Boon search engine are demonstrated using the RefCOCOg dataset, and the engine's ability to search for multimedia through the web is facilitated by Google's API.

{{</citation>}}


## eess.IV (4)



### (62/94) Role of Image Acquisition and Patient Phenotype Variations in Automatic Segmentation Model Generalization (Timothy L. Kline et al., 2023)

{{<citation>}}

Timothy L. Kline, Sumana Ramanathan, Harrison C. Gottlich, Panagiotis Korfiatis, Adriana V. Gregory. (2023)  
**Role of Image Acquisition and Patient Phenotype Variations in Automatic Segmentation Model Generalization**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14482v1)  

---


**ABSTRACT**  
Purpose: This study evaluated the out-of-domain performance and generalization capabilities of automated medical image segmentation models, with a particular focus on adaptation to new image acquisitions and disease type.   Materials: Datasets from both non-contrast and contrast-enhanced abdominal CT scans of healthy patients and those with polycystic kidney disease (PKD) were used. A total of 400 images (100 non-contrast controls, 100 contrast controls, 100 non-contrast PKD, 100 contrast PKD) were utilized for training/validation of models to segment kidneys, livers, and spleens, and the final models were then tested on 100 non-contrast CT images of patients affected by PKD. Performance was evaluated using Dice, Jaccard, TPR, and Precision.   Results: Models trained on a diverse range of data showed no worse performance than models trained exclusively on in-domain data when tested on in-domain data. For instance, the Dice similarity of the model trained on 25% from each dataset was found to be non-inferior to the model trained purely on in-domain data.   Conclusions: The results indicate that broader training examples significantly enhances model generalization and out-of-domain performance, thereby improving automated segmentation tools' applicability in clinical settings. The study's findings provide a roadmap for future research to adopt a data-centric approach in medical image AI model development.

{{</citation>}}


### (63/94) Artifact Restoration in Histology Images with Diffusion Probabilistic Models (Zhenqi He et al., 2023)

{{<citation>}}

Zhenqi He, Junjun He, Jin Ye, Yiqing Shen. (2023)  
**Artifact Restoration in Histology Images with Diffusion Probabilistic Models**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.14262v1)  

---


**ABSTRACT**  
Histological whole slide images (WSIs) can be usually compromised by artifacts, such as tissue folding and bubbles, which will increase the examination difficulty for both pathologists and Computer-Aided Diagnosis (CAD) systems. Existing approaches to restoring artifact images are confined to Generative Adversarial Networks (GANs), where the restoration process is formulated as an image-to-image transfer. Those methods are prone to suffer from mode collapse and unexpected mistransfer in the stain style, leading to unsatisfied and unrealistic restored images. Innovatively, we make the first attempt at a denoising diffusion probabilistic model for histological artifact restoration, namely ArtiFusion.Specifically, ArtiFusion formulates the artifact region restoration as a gradual denoising process, and its training relies solely on artifact-free images to simplify the training complexity.Furthermore, to capture local-global correlations in the regional artifact restoration, a novel Swin-Transformer denoising architecture is designed, along with a time token scheme. Our extensive evaluations demonstrate the effectiveness of ArtiFusion as a pre-processing method for histology analysis, which can successfully preserve the tissue structures and stain style in artifact-free regions during the restoration. Code is available at https://github.com/zhenqi-he/ArtiFusion.

{{</citation>}}


### (64/94) Non-Linear Self Augmentation Deep Pipeline for Cancer Treatment outcome Prediction (Francesco Rundo et al., 2023)

{{<citation>}}

Francesco Rundo, Concetto Spampinato, Michael Rundo. (2023)  
**Non-Linear Self Augmentation Deep Pipeline for Cancer Treatment outcome Prediction**  

---
Primary Category: eess.IV  
Categories: cs-AI, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.14398v1)  

---


**ABSTRACT**  
Immunotherapy emerges as promising approach for treating cancer. Encouraging findings have validated the efficacy of immunotherapy medications in addressing tumors, resulting in prolonged survival rates and notable reductions in toxicity compared to conventional chemotherapy methods. However, the pool of eligible patients for immunotherapy remains relatively small, indicating a lack of comprehensive understanding regarding the physiological mechanisms responsible for favorable treatment response in certain individuals while others experience limited benefits. To tackle this issue, the authors present an innovative strategy that harnesses a non-linear cellular architecture in conjunction with a deep downstream classifier. This approach aims to carefully select and enhance 2D features extracted from chest-abdomen CT images, thereby improving the prediction of treatment outcomes. The proposed pipeline has been meticulously designed to seamlessly integrate with an advanced embedded Point of Care system. In this context, the authors present a compelling case study focused on Metastatic Urothelial Carcinoma (mUC), a particularly aggressive form of cancer. Performance evaluation of the proposed approach underscores its effectiveness, with an impressive overall accuracy of approximately 93%

{{</citation>}}


### (65/94) Hybrid Representation-Enhanced Sampling for Bayesian Active Learning in Musculoskeletal Segmentation of Lower Extremities (Ganping Li et al., 2023)

{{<citation>}}

Ganping Li, Yoshito Otake, Mazen Soufi, Masashi Taniguchi, Masahide Yagi, Noriaki Ichihashi, Keisuke Uemura, Masaki Takao, Nobuhiko Sugano, Yoshinobu Sato. (2023)  
**Hybrid Representation-Enhanced Sampling for Bayesian Active Learning in Musculoskeletal Segmentation of Lower Extremities**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2307.13986v1)  

---


**ABSTRACT**  
Purpose: Obtaining manual annotations to train deep learning (DL) models for auto-segmentation is often time-consuming. Uncertainty-based Bayesian active learning (BAL) is a widely-adopted method to reduce annotation efforts. Based on BAL, this study introduces a hybrid representation-enhanced sampling strategy that integrates density and diversity criteria to save manual annotation costs by efficiently selecting the most informative samples.   Methods: The experiments are performed on two lower extremity (LE) datasets of MRI and CT images by a BAL framework based on Bayesian U-net. Our method selects uncertain samples with high density and diversity for manual revision, optimizing for maximal similarity to unlabeled instances and minimal similarity to existing training data. We assess the accuracy and efficiency using Dice and a proposed metric called reduced annotation cost (RAC), respectively. We further evaluate the impact of various acquisition rules on BAL performance and design an ablation study for effectiveness estimation.   Results: The proposed method showed superiority or non-inferiority to other methods on both datasets across two acquisition rules, and quantitative results reveal the pros and cons of the acquisition rules. Our ablation study in volume-wise acquisition shows that the combination of density and diversity criteria outperforms solely using either of them in musculoskeletal segmentation.   Conclusion: Our sampling method is proven efficient in reducing annotation costs in image segmentation tasks. The combination of the proposed method and our BAL framework provides a semi-automatic way for efficient annotation of medical image datasets.

{{</citation>}}


## math.CT (1)



### (66/94) Obstructions to Compositionality (Caterina Puca et al., 2023)

{{<citation>}}

Caterina Puca, Amar Hadzihasanovic, Fabrizio Genovese, Bob Coecke. (2023)  
**Obstructions to Compositionality**  

---
Primary Category: math.CT  
Categories: cs-LO, math-CT, math.CT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14461v1)  

---


**ABSTRACT**  
Compositionality is at the heart of computer science and several other areas of applied category theory such as computational linguistics, categorical quantum mechanics, interpretable AI, dynamical systems, compositional game theory, and Petri nets. However, the meaning of the term seems to vary across the many different applications. This work contributes to understanding, and in particular qualifying, different kinds of compositionality. Formally, we introduce invariants of categories that we call zeroth and first homotopy posets, generalising in a precise sense the $\pi_0$ and $\pi_1$ of a groupoid. These posets can be used to obtain a qualitative description of how far an object is from being terminal and a morphism is from being iso. In the context of applied category theory, this formal machinery gives us a way to qualitatively describe the "failures of compositionality", seen as failures of certain (op)lax functors to be strong, by classifying obstructions to the (op)laxators being isomorphisms. Failure of compositionality, for example for the interpretation of a categorical syntax in a semantic universe, can both be a bad thing and a good thing, which we illustrate by respective examples in graph theory and quantum theory.

{{</citation>}}


## cs.IR (6)



### (67/94) Integrating Offline Reinforcement Learning with Transformers for Sequential Recommendation (Xumei Xi et al., 2023)

{{<citation>}}

Xumei Xi, Yuke Zhao, Quan Liu, Liwen Ouyang, Yang Wu. (2023)  
**Integrating Offline Reinforcement Learning with Transformers for Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.14450v1)  

---


**ABSTRACT**  
We consider the problem of sequential recommendation, where the current recommendation is made based on past interactions. This recommendation task requires efficient processing of the sequential data and aims to provide recommendations that maximize the long-term reward. To this end, we train a farsighted recommender by using an offline RL algorithm with the policy network in our model architecture that has been initialized from a pre-trained transformer model. The pre-trained model leverages the superb ability of the transformer to process sequential information. Compared to prior works that rely on online interaction via simulation, we focus on implementing a fully offline RL framework that is able to converge in a fast and stable way. Through extensive experiments on public datasets, we show that our method is robust across various recommendation regimes, including e-commerce and movie suggestions. Compared to state-of-the-art supervised learning algorithms, our algorithm yields recommendations of higher quality, demonstrating the clear advantage of combining RL and transformers.

{{</citation>}}


### (68/94) ChatGPT and Persuasive Technologies for the Management and Delivery of Personalized Recommendations in Hotel Hospitality (Manolis Remountakis et al., 2023)

{{<citation>}}

Manolis Remountakis, Konstantinos Kotis, Babis Kourtzis, George E. Tsekouras. (2023)  
**ChatGPT and Persuasive Technologies for the Management and Delivery of Personalized Recommendations in Hotel Hospitality**  

---
Primary Category: cs.IR  
Categories: 68T01, I-2-1, cs-AI, cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.14298v1)  

---


**ABSTRACT**  
Recommender systems have become indispensable tools in the hotel hospitality industry, enabling personalized and tailored experiences for guests. Recent advancements in large language models (LLMs), such as ChatGPT, and persuasive technologies, have opened new avenues for enhancing the effectiveness of those systems. This paper explores the potential of integrating ChatGPT and persuasive technologies for automating and improving hotel hospitality recommender systems. First, we delve into the capabilities of ChatGPT, which can understand and generate human-like text, enabling more accurate and context-aware recommendations. We discuss the integration of ChatGPT into recommender systems, highlighting the ability to analyze user preferences, extract valuable insights from online reviews, and generate personalized recommendations based on guest profiles. Second, we investigate the role of persuasive technology in influencing user behavior and enhancing the persuasive impact of hotel recommendations. By incorporating persuasive techniques, such as social proof, scarcity and personalization, recommender systems can effectively influence user decision-making and encourage desired actions, such as booking a specific hotel or upgrading their room. To investigate the efficacy of ChatGPT and persuasive technologies, we present a pilot experi-ment with a case study involving a hotel recommender system. We aim to study the impact of integrating ChatGPT and persua-sive techniques on user engagement, satisfaction, and conversion rates. The preliminary results demonstrate the potential of these technologies in enhancing the overall guest experience and business performance. Overall, this paper contributes to the field of hotel hospitality by exploring the synergistic relationship between LLMs and persuasive technology in recommender systems, ultimately influencing guest satisfaction and hotel revenue.

{{</citation>}}


### (69/94) Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences (Scott Sanner et al., 2023)

{{<citation>}}

Scott Sanner, Krisztian Balog, Filip Radlinski, Ben Wedin, Lucas Dixon. (2023)  
**Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.14225v1)  

---


**ABSTRACT**  
Traditional recommender systems leverage users' item preference history to recommend novel content that users may like. However, modern dialog interfaces that allow users to express language-based preferences offer a fundamentally different modality for preference input. Inspired by recent successes of prompting paradigms for large language models (LLMs), we study their use for making recommendations from both item-based and language-based preferences in comparison to state-of-the-art item-based collaborative filtering (CF) methods. To support this investigation, we collect a new dataset consisting of both item-based and language-based preferences elicited from users along with their ratings on a variety of (biased) recommended items and (unbiased) random items. Among numerous experimental results, we find that LLMs provide competitive recommendation performance for pure language-based preferences (no item preferences) in the near cold-start case in comparison to item-based CF methods, despite having no supervised training for this specific task (zero-shot) or only a few labels (few-shot). This is particularly promising as language-based preference representations are more explainable and scrutable than item-based or vector-based representations.

{{</citation>}}


### (70/94) A Probabilistic Position Bias Model for Short-Video Recommendation Feeds (Olivier Jeunen, 2023)

{{<citation>}}

Olivier Jeunen. (2023)  
**A Probabilistic Position Bias Model for Short-Video Recommendation Feeds**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.14059v1)  

---


**ABSTRACT**  
Modern web-based platforms show ranked lists of recommendations to users, attempting to maximise user satisfaction or business metrics. Typically, the goal of such systems boils down to maximising the exposure probability for items that are deemed "reward-maximising" according to a metric of interest. This general framing comprises streaming applications, as well as e-commerce or job recommendations, and even web search. Position bias or user models can be used to estimate exposure probabilities for each use-case, specifically tailored to how users interact with the presented rankings. A unifying factor in these diverse problem settings is that typically only one or several items will be engaged with (clicked, streamed,...) before a user leaves the ranked list. Short-video feeds on social media platforms diverge from this general framing in several ways, most notably that users do not tend to leave the feed after e.g. liking a post. Indeed, seemingly infinite feeds invite users to scroll further down the ranked list. For this reason, existing position bias or user models tend to fall short in such settings, as they do not accurately capture users' interaction modalities.   In this work, we propose a novel and probabilistically sound personalised position bias model for feed recommendations. We focus on a 1st-level feed in a hierarchical structure, where users may enter a 2nd-level feed via any given 1st-level item. We posit that users come to the platform with a scrolling budget drawn according to some distribution, and show how the survival function of said distribution can be used to obtain closed-form estimates for personalised exposure probabilities. Empirical insights from a large-scale social media platform show how our probabilistic position bias model more accurately captures empirical exposure than existing models, and paves the way for unbiased evaluation and learning-to-rank.

{{</citation>}}


### (71/94) Multi-view Hypergraph Contrastive Policy Learning for Conversational Recommendation (Sen Zhao et al., 2023)

{{<citation>}}

Sen Zhao, Wei Wei, Xian-Ling Mao, Shuai Zhu, Minghui Yang, Zujie Wen, Dangyang Chen, Feida Zhu. (2023)  
**Multi-view Hypergraph Contrastive Policy Learning for Conversational Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Conversational Recommendation  
[Paper Link](http://arxiv.org/abs/2307.14024v1)  

---


**ABSTRACT**  
Conversational recommendation systems (CRS) aim to interactively acquire user preferences and accordingly recommend items to users. Accurately learning the dynamic user preferences is of crucial importance for CRS. Previous works learn the user preferences with pairwise relations from the interactive conversation and item knowledge, while largely ignoring the fact that factors for a relationship in CRS are multiplex. Specifically, the user likes/dislikes the items that satisfy some attributes (Like/Dislike view). Moreover social influence is another important factor that affects user preference towards the item (Social view), while is largely ignored by previous works in CRS. The user preferences from these three views are inherently different but also correlated as a whole. The user preferences from the same views should be more similar than that from different views. The user preferences from Like View should be similar to Social View while different from Dislike View. To this end, we propose a novel model, namely Multi-view Hypergraph Contrastive Policy Learning (MHCPL). Specifically, MHCPL timely chooses useful social information according to the interactive history and builds a dynamic hypergraph with three types of multiplex relations from different views. The multiplex relations in each view are successively connected according to their generation order.

{{</citation>}}


### (72/94) Domain Disentanglement with Interpolative Data Augmentation for Dual-Target Cross-Domain Recommendation (Jiajie Zhu et al., 2023)

{{<citation>}}

Jiajie Zhu, Yan Wang, Feng Zhu, Zhu Sun. (2023)  
**Domain Disentanglement with Interpolative Data Augmentation for Dual-Target Cross-Domain Recommendation**  

---
Primary Category: cs.IR  
Categories: 68T07 (Primary), H-3-3; I-2-6, cs-IR, cs.IR  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.13910v1)  

---


**ABSTRACT**  
The conventional single-target Cross-Domain Recommendation (CDR) aims to improve the recommendation performance on a sparser target domain by transferring the knowledge from a source domain that contains relatively richer information. By contrast, in recent years, dual-target CDR has been proposed to improve the recommendation performance on both domains simultaneously. However, to this end, there are two challenges in dual-target CDR: (1) how to generate both relevant and diverse augmented user representations, and (2) how to effectively decouple domain-independent information from domain-specific information, in addition to domain-shared information, to capture comprehensive user preferences. To address the above two challenges, we propose a Disentanglement-based framework with Interpolative Data Augmentation for dual-target Cross-Domain Recommendation, called DIDA-CDR. In DIDA-CDR, we first propose an interpolative data augmentation approach to generating both relevant and diverse augmented user representations to augment sparser domain and explore potential user preferences. We then propose a disentanglement module to effectively decouple domain-specific and domain-independent information to capture comprehensive user preferences. Both steps significantly contribute to capturing more comprehensive user preferences, thereby improving the recommendation performance on each domain. Extensive experiments conducted on five real-world datasets show the significant superiority of DIDA-CDR over the state-of-the-art methods.

{{</citation>}}


## cs.SD (2)



### (73/94) WavJourney: Compositional Audio Creation with Large Language Models (Xubo Liu et al., 2023)

{{<citation>}}

Xubo Liu, Zhongkai Zhu, Haohe Liu, Yi Yuan, Meng Cui, Qiushi Huang, Jinhua Liang, Yin Cao, Qiuqiang Kong, Mark D. Plumbley, Wenwu Wang. (2023)  
**WavJourney: Compositional Audio Creation with Large Language Models**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2307.14335v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown great promise in integrating diverse expert models to tackle intricate language and vision tasks. Despite their significance in advancing the field of Artificial Intelligence Generated Content (AIGC), their potential in intelligent audio content creation remains unexplored. In this work, we tackle the problem of creating audio content with storylines encompassing speech, music, and sound effects, guided by text instructions. We present WavJourney, a system that leverages LLMs to connect various audio models for audio content generation. Given a text description of an auditory scene, WavJourney first prompts LLMs to generate a structured script dedicated to audio storytelling. The audio script incorporates diverse audio elements, organized based on their spatio-temporal relationships. As a conceptual representation of audio, the audio script provides an interactive and interpretable rationale for human engagement. Afterward, the audio script is fed into a script compiler, converting it into a computer program. Each line of the program calls a task-specific audio generation model or computational operation function (e.g., concatenate, mix). The computer program is then executed to obtain an explainable solution for audio generation. We demonstrate the practicality of WavJourney across diverse real-world scenarios, including science fiction, education, and radio play. The explainable and interactive design of WavJourney fosters human-machine co-creation in multi-round dialogues, enhancing creative control and adaptability in audio production. WavJourney audiolizes the human imagination, opening up new avenues for creativity in multimedia content creation.

{{</citation>}}


### (74/94) Say Goodbye to RNN-T Loss: A Novel CIF-based Transducer Architecture for Automatic Speech Recognition (Tian-Hao Zhang et al., 2023)

{{<citation>}}

Tian-Hao Zhang, Dinghao Zhou, Guiping Zhong, Baoxiang Li. (2023)  
**Say Goodbye to RNN-T Loss: A Novel CIF-based Transducer Architecture for Automatic Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: AI, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.14132v2)  

---


**ABSTRACT**  
RNN-T models are widely used in ASR, which rely on the RNN-T loss to achieve length alignment between input audio and target sequence. However, the implementation complexity and the alignment-based optimization target of RNN-T loss lead to computational redundancy and a reduced role for predictor network, respectively. In this paper, we propose a novel model named CIF-Transducer (CIF-T) which incorporates the Continuous Integrate-and-Fire (CIF) mechanism with the RNN-T model to achieve efficient alignment. In this way, the RNN-T loss is abandoned, thus bringing a computational reduction and allowing the predictor network a more significant role. We also introduce Funnel-CIF, Context Blocks, Unified Gating and Bilinear Pooling joint network, and auxiliary training strategy to further improve performance. Experiments on the 178-hour AISHELL-1 and 10000-hour WenetSpeech datasets show that CIF-T achieves state-of-the-art results with lower computational overhead compared to RNN-T models.

{{</citation>}}


## q-fin.CP (1)



### (75/94) Modeling Inverse Demand Function with Explainable Dual Neural Networks (Zhiyu Cao et al., 2023)

{{<citation>}}

Zhiyu Cao, Zihan Chen, Prerna Mishra, Hamed Amini, Zachary Feinstein. (2023)  
**Modeling Inverse Demand Function with Explainable Dual Neural Networks**  

---
Primary Category: q-fin.CP  
Categories: J-1; I-2-6, cs-CE, cs-NE, q-fin-CP, q-fin.CP  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2307.14322v1)  

---


**ABSTRACT**  
Financial contagion has been widely recognized as a fundamental risk to the financial system. Particularly potent is price-mediated contagion, wherein forced liquidations by firms depress asset prices and propagate financial stress, enabling crises to proliferate across a broad spectrum of seemingly unrelated entities. Price impacts are currently modeled via exogenous inverse demand functions. However, in real-world scenarios, only the initial shocks and the final equilibrium asset prices are typically observable, leaving actual asset liquidations largely obscured. This missing data presents significant limitations to calibrating the existing models. To address these challenges, we introduce a novel dual neural network structure that operates in two sequential stages: the first neural network maps initial shocks to predicted asset liquidations, and the second network utilizes these liquidations to derive resultant equilibrium prices. This data-driven approach can capture both linear and non-linear forms without pre-specifying an analytical structure; furthermore, it functions effectively even in the absence of observable liquidation data. Experiments with simulated datasets demonstrate that our model can accurately predict equilibrium asset prices based solely on initial shocks, while revealing a strong alignment between predicted and true liquidations. Our explainable framework contributes to the understanding and modeling of price-mediated contagion and provides valuable insights for financial authorities to construct effective stress tests and regulatory policies.

{{</citation>}}


## eess.SY (1)



### (76/94) A Constraint Enforcement Deep Reinforcement Learning Framework for Optimal Energy Storage Systems Dispatch (Shengren Hou et al., 2023)

{{<citation>}}

Shengren Hou, Edgar Mauricio Salazar Duque, Peter Palensky, Pedro P. Vergara. (2023)  
**A Constraint Enforcement Deep Reinforcement Learning Framework for Optimal Energy Storage Systems Dispatch**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.14304v1)  

---


**ABSTRACT**  
The optimal dispatch of energy storage systems (ESSs) presents formidable challenges due to the uncertainty introduced by fluctuations in dynamic prices, demand consumption, and renewable-based energy generation. By exploiting the generalization capabilities of deep neural networks (DNNs), deep reinforcement learning (DRL) algorithms can learn good-quality control models that adaptively respond to distribution networks' stochastic nature. However, current DRL algorithms lack the capabilities to enforce operational constraints strictly, often even providing unfeasible control actions. To address this issue, we propose a DRL framework that effectively handles continuous action spaces while strictly enforcing the environments and action space operational constraints during online operation. Firstly, the proposed framework trains an action-value function modeled using DNNs. Subsequently, this action-value function is formulated as a mixed-integer programming (MIP) formulation enabling the consideration of the environment's operational constraints. Comprehensive numerical simulations show the superior performance of the proposed MIP-DRL framework, effectively enforcing all constraints while delivering high-quality dispatch decisions when compared with state-of-the-art DRL algorithms and the optimal solution obtained with a perfect forecast of the stochastic variables.

{{</citation>}}


## cs.AI (3)



### (77/94) General Purpose Artificial Intelligence Systems (GPAIS): Properties, Definition, Taxonomy, Open Challenges and Implications (Isaac Triguero et al., 2023)

{{<citation>}}

Isaac Triguero, Daniel Molina, Javier Poyatos, Javier Del Ser, Francisco Herrera. (2023)  
**General Purpose Artificial Intelligence Systems (GPAIS): Properties, Definition, Taxonomy, Open Challenges and Implications**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14283v1)  

---


**ABSTRACT**  
Most applications of Artificial Intelligence (AI) are designed for a confined and specific task. However, there are many scenarios that call for a more general AI, capable of solving a wide array of tasks without being specifically designed for them. The term General-Purpose Artificial Intelligence Systems (GPAIS) has been defined to refer to these AI systems. To date, the possibility of an Artificial General Intelligence, powerful enough to perform any intellectual task as if it were human, or even improve it, has remained an aspiration, fiction, and considered a risk for our society. Whilst we might still be far from achieving that, GPAIS is a reality and sitting at the forefront of AI research.   This work discusses existing definitions for GPAIS and proposes a new definition that allows for a gradual differentiation among types of GPAIS according to their properties and limitations. We distinguish between closed-world and open-world GPAIS, characterising their degree of autonomy and ability based on several factors such as adaptation to new tasks, competence in domains not intentionally trained for, ability to learn from few data, or proactive acknowledgment of their own limitations. We then propose a taxonomy of approaches to realise GPAIS, describing research trends such as the use of AI techniques to improve another AI or foundation models. As a prime example, we delve into generative AI, aligning them with the terms and concepts presented in the taxonomy. Through the proposed definition and taxonomy, our aim is to facilitate research collaboration across different areas that are tackling general-purpose tasks, as they share many common aspects. Finally, we discuss the current state of GPAIS, its challenges and prospects, implications for our society, and the need for responsible and trustworthy AI systems and regulation, with the goal of providing a holistic view of GPAIS.

{{</citation>}}


### (78/94) A New Perspective on Evaluation Methods for Explainable Artificial Intelligence (XAI) (Timo Speith et al., 2023)

{{<citation>}}

Timo Speith, Markus Langer. (2023)  
**A New Perspective on Evaluation Methods for Explainable Artificial Intelligence (XAI)**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SE, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14246v1)  

---


**ABSTRACT**  
Within the field of Requirements Engineering (RE), the increasing significance of Explainable Artificial Intelligence (XAI) in aligning AI-supported systems with user needs, societal expectations, and regulatory standards has garnered recognition. In general, explainability has emerged as an important non-functional requirement that impacts system quality. However, the supposed trade-off between explainability and performance challenges the presumed positive influence of explainability. If meeting the requirement of explainability entails a reduction in system performance, then careful consideration must be given to which of these quality aspects takes precedence and how to compromise between them. In this paper, we critically examine the alleged trade-off. We argue that it is best approached in a nuanced way that incorporates resource availability, domain characteristics, and considerations of risk. By providing a foundation for future research and best practices, this work aims to advance the field of RE for AI.

{{</citation>}}


### (79/94) Revisiting the Performance-Explainability Trade-Off in Explainable Artificial Intelligence (XAI) (Barnaby Crook et al., 2023)

{{<citation>}}

Barnaby Crook, Maximilian Schlüter, Timo Speith. (2023)  
**Revisiting the Performance-Explainability Trade-Off in Explainable Artificial Intelligence (XAI)**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SE, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14239v1)  

---


**ABSTRACT**  
Within the field of Requirements Engineering (RE), the increasing significance of Explainable Artificial Intelligence (XAI) in aligning AI-supported systems with user needs, societal expectations, and regulatory standards has garnered recognition. In general, explainability has emerged as an important non-functional requirement that impacts system quality. However, the supposed trade-off between explainability and performance challenges the presumed positive influence of explainability. If meeting the requirement of explainability entails a reduction in system performance, then careful consideration must be given to which of these quality aspects takes precedence and how to compromise between them. In this paper, we critically examine the alleged trade-off. We argue that it is best approached in a nuanced way that incorporates resource availability, domain characteristics, and considerations of risk. By providing a foundation for future research and best practices, this work aims to advance the field of RE for AI.

{{</citation>}}


## cs.NI (1)



### (80/94) A Clustering Strategy for Enhanced FL-Based Intrusion Detection in IoT Networks (Jacopo Talpini et al., 2023)

{{<citation>}}

Jacopo Talpini, Fabio Sartori, Marco Savi. (2023)  
**A Clustering Strategy for Enhanced FL-Based Intrusion Detection in IoT Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2307.14268v1)  

---


**ABSTRACT**  
The Internet of Things (IoT) is growing rapidly and so the need of ensuring protection against cybersecurity attacks to IoT devices. In this scenario, Intrusion Detection Systems (IDSs) play a crucial role and data-driven IDSs based on machine learning (ML) have recently attracted more and more interest by the research community. While conventional ML-based IDSs are based on a centralized architecture where IoT devices share their data with a central server for model training, we propose a novel approach that is based on federated learning (FL). However, conventional FL is ineffective in the considered scenario, due to the high statistical heterogeneity of data collected by IoT devices. To overcome this limitation, we propose a three-tier FL-based architecture where IoT devices are clustered together based on their statistical properties. Clustering decisions are taken by means of a novel entropy-based strategy, which helps improve model training performance. We tested our solution on the CIC-ToN-IoT dataset: our clustering strategy increases intrusion detection performance with respect to a conventional FL approach up to +17% in terms of F1-score, along with a significant reduction of the number of training rounds.

{{</citation>}}


## cs.CY (6)



### (81/94) Improving International Climate Policy via Mutually Conditional Binding Commitments (Jobst Heitzig et al., 2023)

{{<citation>}}

Jobst Heitzig, Jörg Oechssler, Christoph Pröschel, Niranjana Ragavan, Yat Long Lo. (2023)  
**Improving International Climate Policy via Mutually Conditional Binding Commitments**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14267v1)  

---


**ABSTRACT**  
The Paris Agreement, considered a significant milestone in climate negotiations, has faced challenges in effectively addressing climate change due to the unconditional nature of most Nationally Determined Contributions (NDCs). This has resulted in a prevalence of free-riding behavior among major polluters and a lack of concrete conditionality in NDCs. To address this issue, we propose the implementation of a decentralized, bottom-up approach called the Conditional Commitment Mechanism. This mechanism, inspired by the National Popular Vote Interstate Compact, offers flexibility and incentives for early adopters, aiming to formalize conditional cooperation in international climate policy. In this paper, we provide an overview of the mechanism, its performance in the AI4ClimateCooperation challenge, and discuss potential real-world implementation aspects. Prior knowledge of the climate mitigation collective action problem, basic economic principles, and game theory concepts are assumed.

{{</citation>}}


### (82/94) Improving International Climate Policy via Mutually Conditional Binding Commitments (Jobst Heitzig et al., 2023)

{{<citation>}}

Jobst Heitzig, Jörg Oechssler, Christoph Pröschel, Niranjana Ragavan, Richie YatLong Lo. (2023)  
**Improving International Climate Policy via Mutually Conditional Binding Commitments**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.14266v1)  

---


**ABSTRACT**  
This paper proposes enhancements to the RICE-N simulation and multi-agent reinforcement learning framework to improve the realism of international climate policy negotiations. Acknowledging the framework's value, we highlight the necessity of significant enhancements to address the diverse array of factors in modeling climate negotiations. Building upon our previous work on the "Conditional Commitments Mechanism" (CCF mechanism) we discuss ways to bridge the gap between simulation and reality. We suggest the inclusion of a recommender or planner agent to enhance coordination, address the Real2Sim gap by incorporating social factors and non-party stakeholder sub-agents, and propose enhancements to the underlying Reinforcement Learning solution algorithm. These proposed improvements aim to advance the evaluation and formulation of negotiation protocols for more effective international climate policy decision-making in Rice-N. However, further experimentation and testing are required to determine the implications and effectiveness of these suggestions.

{{</citation>}}


### (83/94) AI4GCC - Team: Below Sea Level: Critiques and Improvements (Bram Renting et al., 2023)

{{<citation>}}

Bram Renting, Phillip Wozny, Robert Loftin, Claudia Wieners, Erman Acar. (2023)  
**AI4GCC - Team: Below Sea Level: Critiques and Improvements**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13894v1)  

---


**ABSTRACT**  
We present a critical analysis of the simulation framework RICE-N, an integrated assessment model (IAM) for evaluating the impacts of climate change on the economy. We identify key issues with RICE-N, including action masking and irrelevant actions, and suggest improvements such as utilizing tariff revenue and penalizing overproduction. We also critically engage with features of IAMs in general, namely overly optimistic damage functions and unrealistic abatement cost functions. Our findings contribute to the ongoing efforts to further develop the RICE-N framework in an effort to improve the simulation, making it more useful as an inspiration for policymakers.

{{</citation>}}


### (84/94) Dynamic Grouping for Climate Change Negotiation: Facilitating Cooperation and Balancing Interests through Effective Strategies (Yu Qin et al., 2023)

{{<citation>}}

Yu Qin, Duo Zhang, Yuren Pang. (2023)  
**Dynamic Grouping for Climate Change Negotiation: Facilitating Cooperation and Balancing Interests through Effective Strategies**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13893v1)  

---


**ABSTRACT**  
In this paper, we propose a dynamic grouping negotiation model for climate mitigation based on real-world business and political negotiation protocols. Within the AI4GCC competition framework, we develop a three-stage process: group formation and updates, intra-group negotiation, and inter-group negotiation. Our model promotes efficient and effective cooperation between various stakeholders to achieve global climate change objectives. By implementing a group-forming method and group updating strategy, we address the complexities and imbalances in multi-region climate negotiations. Intra-group negotiations ensure that all members contribute to mitigation efforts, while inter-group negotiations use the proposal-evaluation framework to set mitigation and savings rates. We demonstrate our negotiation model within the RICE-N framework, illustrating a promising approach for facilitating international cooperation on climate change mitigation.

{{</citation>}}


### (85/94) AI4GCC - Team: Below Sea Level: Score and Real World Relevance (Phillip Wozny et al., 2023)

{{<citation>}}

Phillip Wozny, Bram Renting, Robert Loftin, Claudia Wieners, Erman Acar. (2023)  
**AI4GCC - Team: Below Sea Level: Score and Real World Relevance**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13892v1)  

---


**ABSTRACT**  
As our submission for track three of the AI for Global Climate Cooperation (AI4GCC) competition, we propose a negotiation protocol for use in the RICE-N climate-economic simulation. Our proposal seeks to address the challenges of carbon leakage through methods inspired by the Carbon Border Adjustment Mechanism (CBAM) and Climate Clubs (CC). We demonstrate the effectiveness of our approach by comparing simulated outcomes to representative concentration pathways (RCP) and shared socioeconomic pathways (SSP). Our protocol results in a temperature rise comparable to RCP 3.4/4.5 and SSP 2. Furthermore, we provide an analysis of our protocol's World Trade Organization compliance, administrative and political feasibility, and ethical concerns. We recognize that our proposal risks hurting the least developing countries, and we suggest specific corrective measures to avoid exacerbating existing inequalities, such as technology sharing and wealth redistribution. Future research should improve the RICE-N tariff mechanism and implement actions allowing for the aforementioned corrective measures.

{{</citation>}}


### (86/94) Human Culture: A History Irrelevant and Predictable Experience (Hao Wang, 2023)

{{<citation>}}

Hao Wang. (2023)  
**Human Culture: A History Irrelevant and Predictable Experience**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13882v1)  

---


**ABSTRACT**  
Human culture research has witnessed an opportunity of revolution thanks to the big data and social network revolution. Websites such as Douban.com, Goodreads.com, Pandora and IMDB become the new gold mine for cultural researchers. In 2021 and 2022, the author of this paper invented 2 data-free recommender systems for AI cold-start problem. The algorithms can recommend cultural and commercial products to users without reference to users' past preferences. The social implications of the new inventions are human cultural tastes can be predicted very precisely without any information related to human individuals. In this paper, we analyze the AI technologies and its cultural implications together with other AI algorithms. We show that human culture is (mostly) a history irrelevant and predictable experience.

{{</citation>}}


## cs.CR (4)



### (87/94) Unveiling Security, Privacy, and Ethical Concerns of ChatGPT (Xiaodong Wu et al., 2023)

{{<citation>}}

Xiaodong Wu, Ran Duan, Jianbing Ni. (2023)  
**Unveiling Security, Privacy, and Ethical Concerns of ChatGPT**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT, GPT-4, Security  
[Paper Link](http://arxiv.org/abs/2307.14192v1)  

---


**ABSTRACT**  
This paper delves into the realm of ChatGPT, an AI-powered chatbot that utilizes topic modeling and reinforcement learning to generate natural responses. Although ChatGPT holds immense promise across various industries, such as customer service, education, mental health treatment, personal productivity, and content creation, it is essential to address its security, privacy, and ethical implications. By exploring the upgrade path from GPT-1 to GPT-4, discussing the model's features, limitations, and potential applications, this study aims to shed light on the potential risks of integrating ChatGPT into our daily lives. Focusing on security, privacy, and ethics issues, we highlight the challenges these concerns pose for widespread adoption. Finally, we analyze the open problems in these areas, calling for concerted efforts to ensure the development of secure and ethically sound large language models.

{{</citation>}}


### (88/94) Enhanced Security against Adversarial Examples Using a Random Ensemble of Encrypted Vision Transformer Models (Ryota Iijima et al., 2023)

{{<citation>}}

Ryota Iijima, Miki Tanaka, Sayaka Shiota, Hitoshi Kiya. (2023)  
**Enhanced Security against Adversarial Examples Using a Random Ensemble of Encrypted Vision Transformer Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs.CR  
Keywords: Security, Transformer  
[Paper Link](http://arxiv.org/abs/2307.13985v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) are well known to be vulnerable to adversarial examples (AEs). In addition, AEs have adversarial transferability, which means AEs generated for a source model can fool another black-box model (target model) with a non-trivial probability. In previous studies, it was confirmed that the vision transformer (ViT) is more robust against the property of adversarial transferability than convolutional neural network (CNN) models such as ConvMixer, and moreover encrypted ViT is more robust than ViT without any encryption. In this article, we propose a random ensemble of encrypted ViT models to achieve much more robust models. In experiments, the proposed scheme is verified to be more robust against not only black-box attacks but also white-box ones than convention methods.

{{</citation>}}


### (89/94) Dual-Space Attacks against Random-Walk-based Anomaly Detection (Yuni Lai et al., 2023)

{{<citation>}}

Yuni Lai, Marcin Waniek, Yulin Zhu, Liying Li, Jingwen Wu, Tomasz P. Michalak, Talal Rahwan, Kai Zhou. (2023)  
**Dual-Space Attacks against Random-Walk-based Anomaly Detection**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.14387v1)  

---


**ABSTRACT**  
Random Walks-based Anomaly Detection (RWAD) is commonly used to identify anomalous patterns in various applications. An intriguing characteristic of RWAD is that the input graph can either be pre-existing or constructed from raw features. Consequently, there are two potential attack surfaces against RWAD: graph-space attacks and feature-space attacks. In this paper, we explore this vulnerability by designing practical dual-space attacks, investigating the interplay between graph-space and feature-space attacks. To this end, we conduct a thorough complexity analysis, proving that attacking RWAD is NP-hard. Then, we proceed to formulate the graph-space attack as a bi-level optimization problem and propose two strategies to solve it: alternative iteration (alterI-attack) or utilizing the closed-form solution of the random walk model (cf-attack). Finally, we utilize the results from the graph-space attacks as guidance to design more powerful feature-space attacks (i.e., graph-guided attacks). Comprehensive experiments demonstrate that our proposed attacks are effective in enabling the target nodes from RWAD with a limited attack budget. In addition, we conduct transfer attack experiments in a black-box setting, which show that our feature attack significantly decreases the anomaly scores of target nodes. Our study opens the door to studying the dual-space attack against graph anomaly detection in which the graph space relies on the feature space.

{{</citation>}}


### (90/94) Security Weaknesses in IoT Management Platforms (Bhaskar Tejaswi et al., 2023)

{{<citation>}}

Bhaskar Tejaswi, Mohammad Mannan, Amr Youssef. (2023)  
**Security Weaknesses in IoT Management Platforms**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.13952v1)  

---


**ABSTRACT**  
A diverse set of Internet of Things (IoT) devices are becoming an integrated part of daily lives, and playing an increasingly vital role in various industry, enterprise and agricultural settings. The current IoT ecosystem relies on several IoT management platforms to manage and operate a large number of IoT devices, their data, and their connectivity. Considering their key role, these platforms must be properly secured against cyber attacks. In this work, we first explore the core operations/features of leading platforms to design a framework to perform a systematic security evaluation of these platforms. Subsequently, we use our framework to analyze a representative set of 52 IoT management platforms, including 42 web-hosted and 10 locally-deployable platforms. We discover a number of high severity unauthorized access vulnerabilities in 9/52 evaluated IoT management platforms, which could be abused to perform attacks such as remote IoT SIM deactivation, IoT SIM overcharging and IoT device data forgery. More seriously, we also uncover instances of broken authentication in 13/52 platforms, including complete account takeover on 8/52 platforms along with remote code execution on 2/52 platforms. In effect, 17/52 platforms were affected by vulnerabilities that could lead to platform-wide attacks. Overall, vulnerabilities were uncovered in 33 platforms, out of which 28 platforms responded to our responsible disclosure. We were also assigned 11 CVEs and awarded bounty for our findings.

{{</citation>}}


## math.OC (1)



### (91/94) Improving Conflict Analysis in MIP Solvers by Pseudo-Boolean Reasoning (Gioni Mexi et al., 2023)

{{<citation>}}

Gioni Mexi, Timo Berthold, Ambros Gleixner, Jakob Nordström. (2023)  
**Improving Conflict Analysis in MIP Solvers by Pseudo-Boolean Reasoning**  

---
Primary Category: math.OC  
Categories: cs-DM, math-OC, math.OC  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2307.14166v1)  

---


**ABSTRACT**  
Conflict analysis has been successfully generalized from Boolean satisfiability (SAT) solving to mixed integer programming (MIP) solvers, but although MIP solvers operate with general linear inequalities, the conflict analysis in MIP has been limited to reasoning with the more restricted class of clausal constraint. This is in contrast to how conflict analysis is performed in so-called pseudo-Boolean solving, where solvers can reason directly with 0-1 integer linear inequalities rather than with clausal constraints extracted from such inequalities. In this work, we investigate how pseudo-Boolean conflict analysis can be integrated in MIP solving, focusing on 0-1 integer linear programs (0-1 ILPs). Phrased in MIP terminology, conflict analysis can be understood as a sequence of linear combinations and cuts. We leverage this perspective to design a new conflict analysis algorithm based on mixed integer rounding (MIR) cuts, which theoretically dominates the state-of-the-art division-based method in pseudo-Boolean solving. We also report results from a first proof-of-concept implementation of different pseudo-Boolean conflict analysis methods in the open-source MIP solver SCIP. When evaluated on a large and diverse set of 0-1 ILP instances from MIPLIB 2017, our new MIR-based conflict analysis outperforms both previous pseudo-Boolean methods and the clause-based method used in MIP. Our conclusion is that pseudo-Boolean conflict analysis in MIP is a promising research direction that merits further study, and that it might also make sense to investigate the use of such conflict analysis to generate stronger no-goods in constraint programming.

{{</citation>}}


## cs.IT (2)



### (92/94) Is the Performance of NOMA-aided Integrated Sensing and Multicast-Unicast Communications Improved by IRS? (Yang Gou et al., 2023)

{{<citation>}}

Yang Gou, Yinghui Ye, Guangyue Lu, Lu Lv, Rose Qingyang Hu. (2023)  
**Is the Performance of NOMA-aided Integrated Sensing and Multicast-Unicast Communications Improved by IRS?**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2307.14050v1)  

---


**ABSTRACT**  
In this paper, we consider intelligent reflecting surface (IRS) in a non-orthogonal multiple access (NOMA)-aided Integrated Sensing and Multicast-Unicast Communication (ISMUC) system, where the multicast signal is used for sensing and communications while the unicast signal is used only for communications. Our goal is to depict whether the IRS improves the performance of NOMA-ISMUC system or not under the imperfect/perfect successive interference cancellation (SIC) scenario. Towards this end, we formulate a non-convex problem to maximize the unicast rate while ensuring the minimum target illumination power and multicast rate. To settle this problem, we employ the Dinkelbach method to transform this original problem into an equivalent one, which is then solved via alternating optimization algorithm and semidefinite relaxation (SDR) with Sequential Rank-One Constraint Relaxation (SROCR). Based on this, an iterative algorithm is devised to obtain a near-optimal solution. Computer simulations verify the quick convergence of the devised iterative algorithm, and provide insightful results. Compared to NOMA-ISMUC without IRS, IRS-aided NOMA-ISMUC achieves a higher rate with perfect SIC but keeps the almost same rate in the case of imperfect SIC.

{{</citation>}}


### (93/94) Reinforcement Learning for Sequential Decoding of Generalized LDPC Codes (Salman Habib et al., 2023)

{{<citation>}}

Salman Habib, David G. M. Mitchell. (2023)  
**Reinforcement Learning for Sequential Decoding of Generalized LDPC Codes**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13905v1)  

---


**ABSTRACT**  
In this work, we propose reinforcement learning (RL) for sequential decoding of moderate length generalized low-density parity-check (GLDPC) codes. Here, sequential decoding refers to scheduling all the generalized constraint nodes (GCNs) and single parity-check nodes (SPCNs) of a GLDPC code serially in each iteration. A GLDPC decoding environment is modeled as a finite Markov decision process (MDP) in which the state-space comprises of all possible sequences of hard-decision values of the variables nodes (VNs) connected to the scheduled GCN or SPCN, and the action-space of the MDP consists of all possible actions (GCN and SPCN scheduling). The goal of RL is to determine an optimized scheduling policy, i.e., one that results in a decoded codeword by minimizing the complexity of the belief propagation (BP) decoder. For training, we consider the proportion of correct bits at the output of the GCN or SPCN as a reward once it is scheduled. The expected rewards for scheduling all the GCNs/SPCNs in the code's Tanner graph are earned via BP decoding during the RL phase. The proposed RL-based decoding scheme is shown to significantly outperform the standard BP flooding decoder, as well as a sequential decoder in which the GCNs/SPCNs are scheduled randomly.

{{</citation>}}


## cs.DC (1)



### (94/94) Low-Parameter Federated Learning with Large Language Models (Jingang Jiang et al., 2023)

{{<citation>}}

Jingang Jiang, Xiangyang Liu, Chenyou Fan. (2023)  
**Low-Parameter Federated Learning with Large Language Models**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Language Model, NLU, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2307.13896v1)  

---


**ABSTRACT**  
We study few-shot Natural Language Understanding (NLU) tasks with Large Language Models (LLMs) in federated learning (FL) scenarios. It is a challenging task due to limited labeled data and communication capacities in FL, especially with mobile devices. Recent studies show LLMs can be prompted to perform few-shot NLU tasks like sentiment analysis and arithmetic reasoning. However, the huge sizes of LLMs result in high computation and communication costs, making classical FL schemes impractical. To address these challenges, we propose Low-Parameter Federated Learning (LP-FL). LP-FL combines few-shot prompt learning from LLMs with efficient communication and federating techniques. Our approach enables federated clients to assign soft labels to unlabeled data using gradually learned knowledge from the global model. Through iterative soft-label assigning, we continually expand the labeled set during the FL process. Additionally, to reduce computation and communication costs, LP-FL utilizes the Low-Rank Adaptation (LoRA) technique for compact learnable parameter construction, efficient local model fine-tuning, and affordable global model federation. LP-FL consistently outperforms Full-Parameter Federated Learning (FP-FL) in sentiment analysis tasks across various FL settings. Its resistance to overfitting allows LP-FL to equal or surpass centralized training in few-shot scenarios.

{{</citation>}}
