---
draft: false
title: "arXiv @ 2023.10.22"
date: 2023-10-22
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.22"
    identifier: arxiv_20231022
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (62)](#cscl-62)
- [cs.LG (26)](#cslg-26)
- [cs.IR (6)](#csir-6)
- [cs.RO (5)](#csro-5)
- [cs.CV (18)](#cscv-18)
- [cs.NI (1)](#csni-1)
- [cs.SE (2)](#csse-2)
- [eess.IV (2)](#eessiv-2)
- [cs.CY (2)](#cscy-2)
- [cs.DB (1)](#csdb-1)
- [eess.SY (2)](#eesssy-2)
- [cs.SD (3)](#cssd-3)
- [cs.CR (1)](#cscr-1)
- [cs.GR (1)](#csgr-1)
- [hep-lat (1)](#hep-lat-1)
- [cs.SI (1)](#cssi-1)
- [cs.PF (1)](#cspf-1)
- [eess.SP (1)](#eesssp-1)

## cs.CL (62)



### (1/136) Not all Fake News is Written: A Dataset and Analysis of Misleading Video Headlines (Yoo Yeon Sung et al., 2023)

{{<citation>}}

Yoo Yeon Sung, Jordan Boyd-Graber, Naeemul Hassan. (2023)  
**Not all Fake News is Written: A Dataset and Analysis of Misleading Video Headlines**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2310.13859v1)  

---


**ABSTRACT**  
Polarization and the marketplace for impressions have conspired to make navigating information online difficult for users, and while there has been a significant effort to detect false or misleading text, multimodal datasets have received considerably less attention. To complement existing resources, we present multimodal Video Misleading Headline (VMH), a dataset that consists of videos and whether annotators believe the headline is representative of the video's contents. After collecting and annotating this dataset, we analyze multimodal baselines for detecting misleading headlines. Our annotation process also focuses on why annotators view a video as misleading, allowing us to better understand the interplay of annotators' background and the content of the videos.

{{</citation>}}


### (2/136) Ecologically Valid Explanations for Label Variation in NLI (Nan-Jiang Jiang et al., 2023)

{{<citation>}}

Nan-Jiang Jiang, Chenhao Tan, Marie-Catherine de Marneffe. (2023)  
**Ecologically Valid Explanations for Label Variation in NLI**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLI, NLP  
[Paper Link](http://arxiv.org/abs/2310.13850v1)  

---


**ABSTRACT**  
Human label variation, or annotation disagreement, exists in many natural language processing (NLP) tasks, including natural language inference (NLI). To gain direct evidence of how NLI label variation arises, we build LiveNLI, an English dataset of 1,415 ecologically valid explanations (annotators explain the NLI labels they chose) for 122 MNLI items (at least 10 explanations per item). The LiveNLI explanations confirm that people can systematically vary on their interpretation and highlight within-label variation: annotators sometimes choose the same label for different reasons. This suggests that explanations are crucial for navigating label interpretations in general. We few-shot prompt large language models to generate explanations but the results are inconsistent: they sometimes produces valid and informative explanations, but it also generates implausible ones that do not support the label, highlighting directions for improvement.

{{</citation>}}


### (3/136) Plausibility Processing in Transformer Language Models: Focusing on the Role of Attention Heads in GPT (Soo Hyun Ryu, 2023)

{{<citation>}}

Soo Hyun Ryu. (2023)  
**Plausibility Processing in Transformer Language Models: Focusing on the Role of Attention Heads in GPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.13824v1)  

---


**ABSTRACT**  
The goal of this paper is to explore how Transformer language models process semantic knowledge, especially regarding the plausibility of noun-verb relations. First, I demonstrate GPT2 exhibits a higher degree of similarity with humans in plausibility processing compared to other Transformer language models. Next, I delve into how knowledge of plausibility is contained within attention heads of GPT2 and how these heads causally contribute to GPT2's plausibility processing ability. Through several experiments, it was found that: i) GPT2 has a number of attention heads that detect plausible noun-verb relationships; ii) these heads collectively contribute to the Transformer's ability to process plausibility, albeit to varying degrees; and iii) attention heads' individual performance in detecting plausibility does not necessarily correlate with how much they contribute to GPT2's plausibility processing ability.

{{</citation>}}


### (4/136) Evaluation Metrics in the Era of GPT-4: Reliably Evaluating Large Language Models on Sequence to Sequence Tasks (Andrea Sottana et al., 2023)

{{<citation>}}

Andrea Sottana, Bin Liang, Kai Zou, Zheng Yuan. (2023)  
**Evaluation Metrics in the Era of GPT-4: Reliably Evaluating Large Language Models on Sequence to Sequence Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.13800v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) evaluation is a patchy and inconsistent landscape, and it is becoming clear that the quality of automatic evaluation metrics is not keeping up with the pace of development of generative models. We aim to improve the understanding of current models' performance by providing a preliminary and hybrid evaluation on a range of open and closed-source generative LLMs on three NLP benchmarks: text summarisation, text simplification and grammatical error correction (GEC), using both automatic and human evaluation. We also explore the potential of the recently released GPT-4 to act as an evaluator. We find that ChatGPT consistently outperforms many other popular models according to human reviewers on the majority of metrics, while scoring much more poorly when using classic automatic evaluation metrics. We also find that human reviewers rate the gold reference as much worse than the best models' outputs, indicating the poor quality of many popular benchmarks. Finally, we find that GPT-4 is capable of ranking models' outputs in a way which aligns reasonably closely to human judgement despite task-specific variations, with a lower alignment in the GEC task.

{{</citation>}}


### (5/136) Specific versus General Principles for Constitutional AI (Sandipan Kundu et al., 2023)

{{<citation>}}

Sandipan Kundu, Yuntao Bai, Saurav Kadavath, Amanda Askell, Andrew Callahan, Anna Chen, Anna Goldie, Avital Balwit, Azalia Mirhoseini, Brayden McLean, Catherine Olsson, Cassie Evraets, Eli Tran-Johnson, Esin Durmus, Ethan Perez, Jackson Kernion, Jamie Kerr, Kamal Ndousse, Karina Nguyen, Nelson Elhage, Newton Cheng, Nicholas Schiefer, Nova DasSarma, Oliver Rausch, Robin Larson, Shannon Yang, Shauna Kravec, Timothy Telleen-Lawton, Thomas I. Liao, Tom Henighan, Tristan Hume, Zac Hatfield-Dodds, Sören Mindermann, Nicholas Joseph, Sam McCandlish, Jared Kaplan. (2023)  
**Specific versus General Principles for Constitutional AI**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13798v1)  

---


**ABSTRACT**  
Human feedback can prevent overtly harmful utterances in conversational models, but may not automatically mitigate subtle problematic behaviors such as a stated desire for self-preservation or power. Constitutional AI offers an alternative, replacing human feedback with feedback from AI models conditioned only on a list of written principles. We find this approach effectively prevents the expression of such behaviors. The success of simple principles motivates us to ask: can models learn general ethical behaviors from only a single written principle? To test this, we run experiments using a principle roughly stated as "do what's best for humanity". We find that the largest dialogue models can generalize from this short constitution, resulting in harmless assistants with no stated interest in specific motivations like power. A general principle may thus partially avoid the need for a long list of constitutions targeting potentially harmful behaviors. However, more detailed constitutions still improve fine-grained control over specific types of harms. This suggests both general and specific principles have value for steering AI safely.

{{</citation>}}


### (6/136) Copyright Violations and Large Language Models (Antonia Karamolegkou et al., 2023)

{{<citation>}}

Antonia Karamolegkou, Jiaang Li, Li Zhou, Anders Søgaard. (2023)  
**Copyright Violations and Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13771v1)  

---


**ABSTRACT**  
Language models may memorize more than just facts, including entire chunks of texts seen during training. Fair use exemptions to copyright laws typically allow for limited use of copyrighted material without permission from the copyright holder, but typically for extraction of information from copyrighted materials, rather than {\em verbatim} reproduction. This work explores the issue of copyright violations and large language models through the lens of verbatim memorization, focusing on possible redistribution of copyrighted text. We present experiments with a range of language models over a collection of popular books and coding problems, providing a conservative characterization of the extent to which language models can redistribute these materials. Overall, this research highlights the need for further examination and the potential impact on future developments in natural language processing to ensure adherence to copyright regulations. Code is at \url{https://github.com/coastalcph/CopyrightLLMs}.

{{</citation>}}


### (7/136) Enhancing Abstractiveness of Summarization Models through Calibrated Distillation (Hwanjun Song et al., 2023)

{{<citation>}}

Hwanjun Song, Igor Shalyminov, Hang Su, Siffi Singh, Kaisheng Yao, Saab Mansour. (2023)  
**Enhancing Abstractiveness of Summarization Models through Calibrated Distillation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Seq2Seq, Summarization  
[Paper Link](http://arxiv.org/abs/2310.13760v1)  

---


**ABSTRACT**  
Sequence-level knowledge distillation reduces the size of Seq2Seq models for more efficient abstractive summarization. However, it often leads to a loss of abstractiveness in summarization. In this paper, we propose a novel approach named DisCal to enhance the level of abstractiveness (measured by n-gram overlap) without sacrificing the informativeness (measured by ROUGE) of generated summaries. DisCal exposes diverse pseudo summaries with two supervision to the student model. Firstly, the best pseudo summary is identified in terms of abstractiveness and informativeness and used for sequence-level distillation. Secondly, their ranks are used to ensure the student model to assign higher prediction scores to summaries with higher ranks. Our experiments show that DisCal outperforms prior methods in abstractive summarization distillation, producing highly abstractive and informative summaries.

{{</citation>}}


### (8/136) ALDi: Quantifying the Arabic Level of Dialectness of Text (Amr Keleg et al., 2023)

{{<citation>}}

Amr Keleg, Sharon Goldwater, Walid Magdy. (2023)  
**ALDi: Quantifying the Arabic Level of Dialectness of Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.13747v1)  

---


**ABSTRACT**  
Transcribed speech and user-generated text in Arabic typically contain a mixture of Modern Standard Arabic (MSA), the standardized language taught in schools, and Dialectal Arabic (DA), used in daily communications. To handle this variation, previous work in Arabic NLP has focused on Dialect Identification (DI) on the sentence or the token level. However, DI treats the task as binary, whereas we argue that Arabic speakers perceive a spectrum of dialectness, which we operationalize at the sentence level as the Arabic Level of Dialectness (ALDi), a continuous linguistic variable. We introduce the AOC-ALDi dataset (derived from the AOC dataset), containing 127,835 sentences (17% from news articles and 83% from user comments on those articles) which are manually labeled with their level of dialectness. We provide a detailed analysis of AOC-ALDi and show that a model trained on it can effectively identify levels of dialectness on a range of other corpora (including dialects and genres not included in AOC-ALDi), providing a more nuanced picture than traditional DI systems. Through case studies, we illustrate how ALDi can reveal Arabic speakers' stylistic choices in different situations, a useful property for sociolinguistic analyses.

{{</citation>}}


### (9/136) Long-Form Speech Translation through Segmentation with Finite-State Decoding Constraints on Large Language Models (Arya D. McCarthy et al., 2023)

{{<citation>}}

Arya D. McCarthy, Hao Zhang, Shankar Kumar, Felix Stahlberg, Ke Wu. (2023)  
**Long-Form Speech Translation through Segmentation with Finite-State Decoding Constraints on Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13678v2)  

---


**ABSTRACT**  
One challenge in speech translation is that plenty of spoken content is long-form, but short units are necessary for obtaining high-quality translations. To address this mismatch, we adapt large language models (LLMs) to split long ASR transcripts into segments that can be independently translated so as to maximize the overall translation quality. We overcome the tendency of hallucination in LLMs by incorporating finite-state constraints during decoding; these eliminate invalid outputs without requiring additional training. We discover that LLMs are adaptable to transcripts containing ASR errors through prompt-tuning or fine-tuning. Relative to a state-of-the-art automatic punctuation baseline, our best LLM improves the average BLEU by 2.9 points for English-German, English-Spanish, and English-Arabic TED talk translation in 9 test sets, just by improving segmentation.

{{</citation>}}


### (10/136) StereoMap: Quantifying the Awareness of Human-like Stereotypes in Large Language Models (Sullam Jeoung et al., 2023)

{{<citation>}}

Sullam Jeoung, Yubin Ge, Jana Diesner. (2023)  
**StereoMap: Quantifying the Awareness of Human-like Stereotypes in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13673v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have been observed to encode and perpetuate harmful associations present in the training data. We propose a theoretically grounded framework called StereoMap to gain insights into their perceptions of how demographic groups have been viewed by society. The framework is grounded in the Stereotype Content Model (SCM); a well-established theory from psychology. According to SCM, stereotypes are not all alike. Instead, the dimensions of Warmth and Competence serve as the factors that delineate the nature of stereotypes. Based on the SCM theory, StereoMap maps LLMs' perceptions of social groups (defined by socio-demographic features) using the dimensions of Warmth and Competence. Furthermore, the framework enables the investigation of keywords and verbalizations of reasoning of LLMs' judgments to uncover underlying factors influencing their perceptions. Our results show that LLMs exhibit a diverse range of perceptions towards these groups, characterized by mixed evaluations along the dimensions of Warmth and Competence. Furthermore, analyzing the reasonings of LLMs, our findings indicate that LLMs demonstrate an awareness of social disparities, often stating statistical data and research findings to support their reasoning. This study contributes to the understanding of how LLMs perceive and represent social groups, shedding light on their potential biases and the perpetuation of harmful associations.

{{</citation>}}


### (11/136) Let's Synthesize Step by Step: Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models (Ruida Wang et al., 2023)

{{<citation>}}

Ruida Wang, Wangchunshu Zhou, Mrinmaya Sachan. (2023)  
**Let's Synthesize Step by Step: Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.13671v1)  

---


**ABSTRACT**  
*Data Synthesis* is a promising way to train a small model with very little labeled data. One approach for data synthesis is to leverage the rich knowledge from large language models to synthesize pseudo training examples for small models, making it possible to achieve both data and compute efficiency at the same time. However, a key challenge in data synthesis is that the synthesized dataset often suffers from a large distributional discrepancy from the *real task* data distribution. Thus, in this paper, we propose *Synthesis Step by Step* (**S3**), a data synthesis framework that shrinks this distribution gap by iteratively extrapolating the errors made by a small model trained on the synthesized dataset on a small real-world validation dataset using a large language model. Extensive experiments on multiple NLP tasks show that our approach improves the performance of a small model by reducing the gap between the synthetic dataset and the real data, resulting in significant improvement compared to several baselines: 9.48% improvement compared to ZeroGen and 2.73% compared to GoldGen, and at most 15.17% improvement compared to the small model trained on human-annotated data.

{{</citation>}}


### (12/136) Explainable Depression Symptom Detection in Social Media (Eliseo Bao Souto et al., 2023)

{{<citation>}}

Eliseo Bao Souto, Anxo Pérez, Javier Parapar. (2023)  
**Explainable Depression Symptom Detection in Social Media**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2310.13664v2)  

---


**ABSTRACT**  
Users of social platforms often perceive these sites as supportive spaces to post about their mental health issues. Those conversations contain important traces about individuals' health risks. Recently, researchers have exploited this online information to construct mental health detection models, which aim to identify users at risk on platforms like Twitter, Reddit or Facebook. Most of these models are centred on achieving good classification results, ignoring the explainability and interpretability of the decisions. Recent research has pointed out the importance of using clinical markers, such as the use of symptoms, to improve trust in the computational models by health professionals. In this paper, we propose using transformer-based architectures to detect and explain the appearance of depressive symptom markers in the users' writings. We present two approaches: i) train a model to classify, and another one to explain the classifier's decision separately and ii) unify the two tasks simultaneously using a single model. Additionally, for this latter manner, we also investigated the performance of recent conversational LLMs when using in-context learning. Our natural language explanations enable clinicians to interpret the models' decisions based on validated symptoms, enhancing trust in the automated process. We evaluate our approach using recent symptom-based datasets, employing both offline and expert-in-the-loop metrics to assess the quality of the explanations generated by our models. The experimental results show that it is possible to achieve good classification results while generating interpretable symptom-based explanations.

{{</citation>}}


### (13/136) Benchmarking and Improving Text-to-SQL Generation under Ambiguity (Adithya Bhaskar et al., 2023)

{{<citation>}}

Adithya Bhaskar, Tushar Tomar, Ashutosh Sathe, Sunita Sarawagi. (2023)  
**Benchmarking and Improving Text-to-SQL Generation under Ambiguity**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.13659v1)  

---


**ABSTRACT**  
Research in Text-to-SQL conversion has been largely benchmarked against datasets where each text query corresponds to one correct SQL. However, natural language queries over real-life databases frequently involve significant ambiguity about the intended SQL due to overlapping schema names and multiple confusing relationship paths. To bridge this gap, we develop a novel benchmark called AmbiQT with over 3000 examples where each text is interpretable as two plausible SQLs due to lexical and/or structural ambiguity.   When faced with ambiguity, an ideal top-$k$ decoder should generate all valid interpretations for possible disambiguation by the user. We evaluate several Text-to-SQL systems and decoding algorithms, including those employing state-of-the-art LLMs, and find them to be far from this ideal. The primary reason is that the prevalent beam search algorithm and its variants, treat SQL queries as a string and produce unhelpful token-level diversity in the top-$k$.   We propose LogicalBeam, a new decoding algorithm that navigates the SQL logic space using a blend of plan-based template generation and constrained infilling. Counterfactually generated plans diversify templates while in-filling with a beam-search that branches solely on schema names provides value diversity. LogicalBeam is up to $2.5$ times more effective than state-of-the-art models at generating all candidate SQLs in the top-$k$ ranked outputs. It also enhances the top-$5$ Exact and Execution Match Accuracies on SPIDER and Kaggle DBQA.

{{</citation>}}


### (14/136) BotChat: Evaluating LLMs' Capabilities of Having Multi-Turn Dialogues (Haodong Duan et al., 2023)

{{<citation>}}

Haodong Duan, Jueqi Wei, Chonghua Wang, Hongwei Liu, Yixiao Fang, Songyang Zhang, Dahua Lin, Kai Chen. (2023)  
**BotChat: Evaluating LLMs' Capabilities of Having Multi-Turn Dialogues**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.13650v1)  

---


**ABSTRACT**  
Interacting with human via high-quality multi-turn dialogues is a key feature of large language models (LLMs). However, human-based evaluation of such capability involves intensive manual labor. This report provides a preliminary evaluation of existing large language models for human-style multi-turn chatting, through an LLM-based approach. We start from real-world human dialogues and keep the very first utterances as the ChatSEED. Then we prompt LLMs to generate a full multi-turn dialogue (tens of utterances) based on the ChatSEED, utterance by utterance. Finally, we adopt state-of-the-art LLMs (GPT-4, \etc) as the judge to evaluate the generated dialogues. With different evaluation protocols, we come to substantially identical conclusions. We find that GPT-4 can generate human-style multi-turn dialogues with impressive quality, significantly outperforms its counterparts. It's difficult for a discriminator to distinguish between GPT-4 generated dialogues and human dialogues. In contrast, other LLMs struggle to generate multi-turn dialogues of satisfactory quality due to poor instruction-following capability, tendency to generate lengthy utterances, or limited general capability. All data and codes will be provided in https://github.com/open-compass/BotChat/ and we hope they can serve as a valuable resource for evaluating multi-turn chatting capabilities of LLMs.

{{</citation>}}


### (15/136) Bridging Information-Theoretic and Geometric Compression in Language Models (Emily Cheng et al., 2023)

{{<citation>}}

Emily Cheng, Corentin Kervadec, Marco Baroni. (2023)  
**Bridging Information-Theoretic and Geometric Compression in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13620v1)  

---


**ABSTRACT**  
For a language model (LM) to faithfully model human language, it must compress vast, potentially infinite information into relatively few dimensions. We propose analyzing compression in (pre-trained) LMs from two points of view: geometric and information-theoretic. We demonstrate that the two views are highly correlated, such that the intrinsic geometric dimension of linguistic data predicts their coding length under the LM. We then show that, in turn, high compression of a linguistic dataset predicts rapid adaptation to that dataset, confirming that being able to compress linguistic information is an important part of successful LM performance. As a practical byproduct of our analysis, we evaluate a battery of intrinsic dimension estimators for the first time on linguistic data, showing that only some encapsulate the relationship between information-theoretic compression, geometric compression, and ease-of-adaptation.

{{</citation>}}


### (16/136) Three Questions Concerning the Use of Large Language Models to Facilitate Mathematics Learning (An-Zi Yen et al., 2023)

{{<citation>}}

An-Zi Yen, Wei-Ling Hsu. (2023)  
**Three Questions Concerning the Use of Large Language Models to Facilitate Mathematics Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13615v1)  

---


**ABSTRACT**  
Due to the remarkable language understanding and generation abilities of large language models (LLMs), their use in educational applications has been explored. However, little work has been done on investigating the pedagogical ability of LLMs in helping students to learn mathematics. In this position paper, we discuss the challenges associated with employing LLMs to enhance students' mathematical problem-solving skills by providing adaptive feedback. Apart from generating the wrong reasoning processes, LLMs can misinterpret the meaning of the question, and also exhibit difficulty in understanding the given questions' rationales when attempting to correct students' answers. Three research questions are formulated.

{{</citation>}}


### (17/136) Hunayn: Elevating Translation Beyond the Literal (Nasser Almousa et al., 2023)

{{<citation>}}

Nasser Almousa, Nasser Alzamil, Abdullah Alshehri, Ahmad Sait. (2023)  
**Hunayn: Elevating Translation Beyond the Literal**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.13613v2)  

---


**ABSTRACT**  
This project introduces an advanced English-to-Arabic translator surpassing conventional tools. Leveraging the Helsinki transformer (MarianMT), our approach involves fine-tuning on a self-scraped, purely literary Arabic dataset. Evaluations against Google Translate show consistent outperformance in qualitative assessments. Notably, it excels in cultural sensitivity and context accuracy. This research underscores the Helsinki transformer's superiority for English-to-Arabic translation using a Fusha dataset.

{{</citation>}}


### (18/136) Make Your Decision Convincing! A Unified Two-Stage Framework: Self-Attribution and Decision-Making (Yanrui Du et al., 2023)

{{<citation>}}

Yanrui Du, Sendong Zhao, Haochun Wang, Yuhan Chen, Rui Bai, Zewen Qiang, Muzhen Cai, Bing Qin. (2023)  
**Make Your Decision Convincing! A Unified Two-Stage Framework: Self-Attribution and Decision-Making**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.13610v1)  

---


**ABSTRACT**  
Explaining black-box model behavior with natural language has achieved impressive results in various NLP tasks. Recent research has explored the utilization of subsequences from the input text as a rationale, providing users with evidence to support the model decision. Although existing frameworks excel in generating high-quality rationales while achieving high task performance, they neglect to account for the unreliable link between the generated rationale and model decision. In simpler terms, a model may make correct decisions while attributing wrong rationales, or make poor decisions while attributing correct rationales. To mitigate this issue, we propose a unified two-stage framework known as Self-Attribution and Decision-Making (SADM). Through extensive experiments on five reasoning datasets from the ERASER benchmark, we demonstrate that our framework not only establishes a more reliable link between the generated rationale and model decision but also achieves competitive results in task performance and the quality of rationale. Furthermore, we explore the potential of our framework in semi-supervised scenarios.

{{</citation>}}


### (19/136) MULTITuDE: Large-Scale Multilingual Machine-Generated Text Detection Benchmark (Dominik Macko et al., 2023)

{{<citation>}}

Dominik Macko, Robert Moro, Adaku Uchendu, Jason Samuel Lucas, Michiharu Yamashita, Matúš Pikuliak, Ivan Srba, Thai Le, Dongwon Lee, Jakub Simko, Maria Bielikova. (2023)  
**MULTITuDE: Large-Scale Multilingual Machine-Generated Text Detection Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2310.13606v1)  

---


**ABSTRACT**  
There is a lack of research into capabilities of recent LLMs to generate convincing text in languages other than English and into performance of detectors of machine-generated text in multilingual settings. This is also reflected in the available benchmarks which lack authentic texts in languages other than English and predominantly cover older generators. To fill this gap, we introduce MULTITuDE, a novel benchmarking dataset for multilingual machine-generated text detection comprising of 74,081 authentic and machine-generated texts in 11 languages (ar, ca, cs, de, en, es, nl, pt, ru, uk, and zh) generated by 8 multilingual LLMs. Using this benchmark, we compare the performance of zero-shot (statistical and black-box) and fine-tuned detectors. Considering the multilinguality, we evaluate 1) how these detectors generalize to unseen languages (linguistically similar as well as dissimilar) and unseen LLMs and 2) whether the detectors improve their performance when trained on multiple languages.

{{</citation>}}


### (20/136) MarineGPT: Unlocking Secrets of Ocean to the Public (Ziqiang Zheng et al., 2023)

{{<citation>}}

Ziqiang Zheng, Jipeng Zhang, Tuan-Anh Vu, Shizhe Diao, Yue Him Wong Tim, Sai-Kit Yeung. (2023)  
**MarineGPT: Unlocking Secrets of Ocean to the Public**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.13596v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as ChatGPT/GPT-4, have proven to be powerful tools in promoting the user experience as an AI assistant. The continuous works are proposing multi-modal large language models (MLLM), empowering LLMs with the ability to sense multiple modality inputs through constructing a joint semantic space (e.g. visual-text space). Though significant success was achieved in LLMs and MLLMs, exploring LLMs and MLLMs in domain-specific applications that required domain-specific knowledge and expertise has been less conducted, especially for \textbf{marine domain}. Different from general-purpose MLLMs, the marine-specific MLLM is required to yield much more \textbf{sensitive}, \textbf{informative}, and \textbf{scientific} responses. In this work, we demonstrate that the existing MLLMs optimized on huge amounts of readily available general-purpose training data show a minimal ability to understand domain-specific intents and then generate informative and satisfactory responses. To address these issues, we propose \textbf{MarineGPT}, the first vision-language model specially designed for the marine domain, unlocking the secrets of the ocean to the public. We present our \textbf{Marine-5M} dataset with more than 5 million marine image-text pairs to inject domain-specific marine knowledge into our model and achieve better marine vision and language alignment. Our MarineGPT not only pushes the boundaries of marine understanding to the general public but also offers a standard protocol for adapting a general-purpose assistant to downstream domain-specific experts. We pave the way for a wide range of marine applications while setting valuable data and pre-trained models for future research in both academic and industrial communities.

{{</citation>}}


### (21/136) Simultaneous Machine Translation with Tailored Reference (Shoutao Guo et al., 2023)

{{<citation>}}

Shoutao Guo, Shaolei Zhang, Yang Feng. (2023)  
**Simultaneous Machine Translation with Tailored Reference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.13588v1)  

---


**ABSTRACT**  
Simultaneous machine translation (SiMT) generates translation while reading the whole source sentence. However, existing SiMT models are typically trained using the same reference disregarding the varying amounts of available source information at different latency. Training the model with ground-truth at low latency may introduce forced anticipations, whereas utilizing reference consistent with the source word order at high latency results in performance degradation. Consequently, it is crucial to train the SiMT model with appropriate reference that avoids forced anticipations during training while maintaining high quality. In this paper, we propose a novel method that provides tailored reference for the SiMT models trained at different latency by rephrasing the ground-truth. Specifically, we introduce the tailor, induced by reinforcement learning, to modify ground-truth to the tailored reference. The SiMT model is trained with the tailored reference and jointly optimized with the tailor to enhance performance. Importantly, our method is applicable to a wide range of current SiMT approaches. Experiments on three translation tasks demonstrate that our method achieves state-of-the-art performance in both fixed and adaptive policies.

{{</citation>}}


### (22/136) Improving Cross-Lingual Transfer through Subtree-Aware Word Reordering (Ofir Arviv et al., 2023)

{{<citation>}}

Ofir Arviv, Dmitry Nikolaev, Taelin Karidi, Omri Abend. (2023)  
**Improving Cross-Lingual Transfer through Subtree-Aware Word Reordering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2310.13583v1)  

---


**ABSTRACT**  
Despite the impressive growth of the abilities of multilingual language models, such as XLM-R and mT5, it has been shown that they still face difficulties when tackling typologically-distant languages, particularly in the low-resource setting. One obstacle for effective cross-lingual transfer is variability in word-order patterns. It can be potentially mitigated via source- or target-side word reordering, and numerous approaches to reordering have been proposed. However, they rely on language-specific rules, work on the level of POS tags, or only target the main clause, leaving subordinate clauses intact. To address these limitations, we present a new powerful reordering method, defined in terms of Universal Dependencies, that is able to learn fine-grained word-order patterns conditioned on the syntactic context from a small amount of annotated data and can be applied at all levels of the syntactic tree. We conduct experiments on a diverse set of tasks and show that our method consistently outperforms strong baselines over different language pairs and model architectures. This performance advantage holds true in both zero-shot and few-shot scenarios.

{{</citation>}}


### (23/136) Why Can Large Language Models Generate Correct Chain-of-Thoughts? (Rasul Tutunov et al., 2023)

{{<citation>}}

Rasul Tutunov, Antoine Grosnit, Juliusz Ziomek, Jun Wang, Haitham Bou-Ammar. (2023)  
**Why Can Large Language Models Generate Correct Chain-of-Thoughts?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13571v1)  

---


**ABSTRACT**  
This paper delves into the capabilities of large language models (LLMs), specifically focusing on advancing the theoretical comprehension of chain-of-thought prompting. We investigate how LLMs can be effectively induced to generate a coherent chain of thoughts. To achieve this, we introduce a two-level hierarchical graphical model tailored for natural language generation. Within this framework, we establish a compelling geometrical convergence rate that gauges the likelihood of an LLM-generated chain of thoughts compared to those originating from the true language. Our findings provide a theoretical justification for the ability of LLMs to produce the correct sequence of thoughts (potentially) explaining performance gains in tasks demanding reasoning skills.

{{</citation>}}


### (24/136) Retrieval-Augmented Neural Response Generation Using Logical Reasoning and Relevance Scoring (Nicholas Thomas Walker et al., 2023)

{{<citation>}}

Nicholas Thomas Walker, Stefan Ultes, Pierre Lison. (2023)  
**Retrieval-Augmented Neural Response Generation Using Logical Reasoning and Relevance Scoring**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.13566v1)  

---


**ABSTRACT**  
Constructing responses in task-oriented dialogue systems typically relies on information sources such the current dialogue state or external databases. This paper presents a novel approach to knowledge-grounded response generation that combines retrieval-augmented language models with logical reasoning. The approach revolves around a knowledge graph representing the current dialogue state and background information, and proceeds in three steps. The knowledge graph is first enriched with logically derived facts inferred using probabilistic logical programming. A neural model is then employed at each turn to score the conversational relevance of each node and edge of this extended graph. Finally, the elements with highest relevance scores are converted to a natural language form, and are integrated into the prompt for the neural conversational model employed to generate the system response.   We investigate the benefits of the proposed approach on two datasets (KVRET and GraphWOZ) along with a human evaluation. Experimental results show that the combination of (probabilistic) logical reasoning with conversational relevance scoring does increase both the factuality and fluency of the responses.

{{</citation>}}


### (25/136) Cache & Distil: Optimising API Calls to Large Language Models (Guillem Ramírez et al., 2023)

{{<citation>}}

Guillem Ramírez, Matthias Lindemann, Alexandra Birch, Ivan Titov. (2023)  
**Cache & Distil: Optimising API Calls to Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13561v1)  

---


**ABSTRACT**  
Large-scale deployment of generative AI tools often depends on costly API calls to a Large Language Model (LLM) to fulfil user queries. To curtail the frequency of these calls, one can employ a smaller language model -- a student -- which is continuously trained on the responses of the LLM. This student gradually gains proficiency in independently handling an increasing number of user requests, a process we term neural caching. The crucial element in neural caching is a policy that decides which requests should be processed by the student alone and which should be redirected to the LLM, subsequently aiding the student's learning. In this study, we focus on classification tasks, and we consider a range of classic active learning-based selection criteria as the policy. Our experiments suggest that Margin Sampling and Query by Committee bring consistent benefits across tasks and budgets.

{{</citation>}}


### (26/136) Self-prompted Chain-of-Thought on Large Language Models for Open-domain Multi-hop Reasoning (Jinyuan Wang et al., 2023)

{{<citation>}}

Jinyuan Wang, Junlong Li, Hai Zhao. (2023)  
**Self-prompted Chain-of-Thought on Large Language Models for Open-domain Multi-hop Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.13552v2)  

---


**ABSTRACT**  
In open-domain question-answering (ODQA), most existing questions require single-hop reasoning on commonsense. To further extend this task, we officially introduce open-domain multi-hop reasoning (ODMR) by answering multi-hop questions with explicit reasoning steps in open-domain setting. Recently, large language models (LLMs) have found significant utility in facilitating ODQA without external corpus. Furthermore, chain-of-thought (CoT) prompting boosts the reasoning capability of LLMs to a greater extent with manual or automated paradigms. However, existing automated methods lack of quality assurance, while manual approaches suffer from limited scalability and poor diversity, hindering the capabilities of LLMs. In this paper, we propose Self-prompted Chain-of-Thought (SP-CoT), an automated framework to mass-produce high quality CoTs of LLMs, by LLMs and for LLMs. SP-CoT introduces an automated generation pipeline of high quality ODMR datasets, an adaptive sampler for in-context CoT selection and self-prompted inference via in-context learning. Extensive experiments on four multi-hop question-answering benchmarks show that our proposed SP-CoT not only significantly surpasses the previous SOTA methods on large-scale (175B) LLMs, but also nearly doubles the zero-shot performance of small-scale (13B) LLMs. Further analysis reveals the remarkable capability of SP-CoT to elicit direct and concise intermediate reasoning steps by recalling $\sim$50\% of intermediate answers on MuSiQue-Ans dataset.

{{</citation>}}


### (27/136) The Perils & Promises of Fact-checking with Large Language Models (Dorian Quelle et al., 2023)

{{<citation>}}

Dorian Quelle, Alexandre Bovet. (2023)  
**The Perils & Promises of Fact-checking with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-HC, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13549v1)  

---


**ABSTRACT**  
Autonomous fact-checking, using machine learning to verify claims, has grown vital as misinformation spreads beyond human fact-checking capacity. Large Language Models (LLMs) like GPT-4 are increasingly trusted to verify information and write academic papers, lawsuits, and news articles, emphasizing their role in discerning truth from falsehood and the importance of being able to verify their outputs. Here, we evaluate the use of LLM agents in fact-checking by having them phrase queries, retrieve contextual data, and make decisions. Importantly, in our framework, agents explain their reasoning and cite the relevant sources from the retrieved context. Our results show the enhanced prowess of LLMs when equipped with contextual information. GPT-4 outperforms GPT-3, but accuracy varies based on query language and claim veracity. While LLMs show promise in fact-checking, caution is essential due to inconsistent accuracy. Our investigation calls for further research, fostering a deeper comprehension of when agents succeed and when they fail.

{{</citation>}}


### (28/136) Towards Understanding Sycophancy in Language Models (Mrinank Sharma et al., 2023)

{{<citation>}}

Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, Ethan Perez. (2023)  
**Towards Understanding Sycophancy in Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-6, cs-AI, cs-CL, cs-LG, cs.CL, stat-ML  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13548v2)  

---


**ABSTRACT**  
Reinforcement learning from human feedback (RLHF) is a popular technique for training high-quality AI assistants. However, RLHF may also encourage model responses that match user beliefs over truthful responses, a behavior known as sycophancy. We investigate the prevalence of sycophancy in RLHF-trained models and whether human preference judgements are responsible. We first demonstrate that five state-of-the-art AI assistants consistently exhibit sycophantic behavior across four varied free-form text-generation tasks. To understand if human preferences drive this broadly observed behavior of RLHF models, we analyze existing human preference data. We find that when a response matches a user's views, it is more likely to be preferred. Moreover, both humans and preference models (PMs) prefer convincingly-written sycophantic responses over correct ones a non-negligible fraction of the time. Optimizing model outputs against PMs also sometimes sacrifices truthfulness in favor of sycophancy. Overall, our results indicate that sycophancy is a general behavior of RLHF models, likely driven in part by human preference judgements favoring sycophantic responses.

{{</citation>}}


### (29/136) A Diachronic Perspective on User Trust in AI under Uncertainty (Shehzaad Dhuliawala et al., 2023)

{{<citation>}}

Shehzaad Dhuliawala, Vilém Zouhar, Mennatallah El-Assady, Mrinmaya Sachan. (2023)  
**A Diachronic Perspective on User Trust in AI under Uncertainty**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2310.13544v1)  

---


**ABSTRACT**  
In a human-AI collaboration, users build a mental model of the AI system based on its reliability and how it presents its decision, e.g. its presentation of system confidence and an explanation of the output. Modern NLP systems are often uncalibrated, resulting in confidently incorrect predictions that undermine user trust. In order to build trustworthy AI, we must understand how user trust is developed and how it can be regained after potential trust-eroding events. We study the evolution of user trust in response to these trust-eroding events using a betting game. We find that even a few incorrect instances with inaccurate confidence estimates damage user trust and performance, with very slow recovery. We also show that this degradation in trust reduces the success of human-AI collaboration and that different types of miscalibration -- unconfidently correct and confidently incorrect -- have different negative effects on user trust. Our findings highlight the importance of calibration in user-facing AI applications and shed light on what aspects help users decide whether to trust the AI system.

{{</citation>}}


### (30/136) Controlled Randomness Improves the Performance of Transformer Models (Tobias Deußer et al., 2023)

{{<citation>}}

Tobias Deußer, Cong Zhao, Wolfgang Krämer, David Leonhard, Christian Bauckhage, Rafet Sifa. (2023)  
**Controlled Randomness Improves the Performance of Transformer Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.13526v1)  

---


**ABSTRACT**  
During the pre-training step of natural language models, the main objective is to learn a general representation of the pre-training dataset, usually requiring large amounts of textual data to capture the complexity and diversity of natural language. Contrasting this, in most cases, the size of the data available to solve the specific downstream task is often dwarfed by the aforementioned pre-training dataset, especially in domains where data is scarce. We introduce controlled randomness, i.e. noise, into the training process to improve fine-tuning language models and explore the performance of targeted noise in addition to the parameters of these models. We find that adding such noise can improve the performance in our two downstream tasks of joint named entity recognition and relation extraction and text summarization.

{{</citation>}}


### (31/136) Teaching Language Models to Self-Improve through Interactive Demonstrations (Xiao Yu et al., 2023)

{{<citation>}}

Xiao Yu, Baolin Peng, Michel Galley, Jianfeng Gao, Zhou Yu. (2023)  
**Teaching Language Models to Self-Improve through Interactive Demonstrations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13522v1)  

---


**ABSTRACT**  
The self-improving ability of large language models (LLMs), enabled by prompting them to analyze and revise their own outputs, has garnered significant interest in recent research. However, this ability has been shown to be absent and difficult to learn for smaller models, thus widening the performance gap between state-of-the-art LLMs and more cost-effective and faster ones. To reduce this gap, we introduce TriPosT, a training algorithm that endows smaller models with such self-improvement ability, and show that our approach can improve a LLaMA-7b's performance on math and reasoning tasks by up to 7.13%. In contrast to prior work, we achieve this by using the smaller model to interact with LLMs to collect feedback and improvements on its own generations. We then replay this experience to train the small model. Our experiments on four math and reasoning datasets show that the interactive experience of learning from and correcting its own mistakes is crucial for small models to improve their performance.

{{</citation>}}


### (32/136) Improving Question Generation with Multi-level Content Planning (Zehua Xia et al., 2023)

{{<citation>}}

Zehua Xia, Qi Gou, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li, Cam-Tu Nguyen. (2023)  
**Improving Question Generation with Multi-level Content Planning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Question Generation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.13512v2)  

---


**ABSTRACT**  
This paper addresses the problem of generating questions from a given context and an answer, specifically focusing on questions that require multi-hop reasoning across an extended context. Previous studies have suggested that key phrase selection is essential for question generation (QG), yet it is still challenging to connect such disjointed phrases into meaningful questions, particularly for long context. To mitigate this issue, we propose MultiFactor, a novel QG framework based on multi-level content planning. Specifically, MultiFactor includes two components: FA-model, which simultaneously selects key phrases and generates full answers, and Q-model which takes the generated full answer as an additional input to generate questions. Here, full answer generation is introduced to connect the short answer with the selected key phrases, thus forming an answer-aware summary to facilitate QG. Both FA-model and Q-model are formalized as simple-yet-effective Phrase-Enhanced Transformers, our joint model for phrase selection and text generation. Experimental results show that our method outperforms strong baselines on two popular QG datasets. Our code is available at https://github.com/zeaver/MultiFactor.

{{</citation>}}


### (33/136) Explaining Interactions Between Text Spans (Sagnik Ray Choudhury et al., 2023)

{{<citation>}}

Sagnik Ray Choudhury, Pepa Atanasova, Isabelle Augenstein. (2023)  
**Explaining Interactions Between Text Spans**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: NLI, NLU, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.13506v1)  

---


**ABSTRACT**  
Reasoning over spans of tokens from different parts of the input is essential for natural language understanding (NLU) tasks such as fact-checking (FC), machine reading comprehension (MRC) or natural language inference (NLI). However, existing highlight-based explanations primarily focus on identifying individual important tokens or interactions only between adjacent tokens or tuples of tokens. Most notably, there is a lack of annotations capturing the human decision-making process w.r.t. the necessary interactions for informed decision-making in such tasks. To bridge this gap, we introduce SpanEx, a multi-annotator dataset of human span interaction explanations for two NLU tasks: NLI and FC. We then investigate the decision-making processes of multiple fine-tuned large language models in terms of the employed connections between spans in separate parts of the input and compare them to the human reasoning processes. Finally, we present a novel community detection based unsupervised method to extract such interaction explanations from a model's inner workings.

{{</citation>}}


### (34/136) Robust Training for Conversational Question Answering Models with Reinforced Reformulation Generation (Magdalena Kaiser et al., 2023)

{{<citation>}}

Magdalena Kaiser, Rishiraj Saha Roy, Gerhard Weikum. (2023)  
**Robust Training for Conversational Question Answering Models with Reinforced Reformulation Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: GPT, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.13505v1)  

---


**ABSTRACT**  
Models for conversational question answering (ConvQA) over knowledge graphs (KGs) are usually trained and tested on benchmarks of gold QA pairs. This implies that training is limited to surface forms seen in the respective datasets, and evaluation is on a small set of held-out questions. Through our proposed framework REIGN, we take several steps to remedy this restricted learning setup. First, we systematically generate reformulations of training questions to increase robustness of models to surface form variations. This is a particularly challenging problem, given the incomplete nature of such questions. Second, we guide ConvQA models towards higher performance by feeding it only those reformulations that help improve their answering quality, using deep reinforcement learning. Third, we demonstrate the viability of training major model components on one benchmark and applying them zero-shot to another. Finally, for a rigorous evaluation of robustness for trained models, we use and release large numbers of diverse reformulations generated by prompting GPT for benchmark test sets (resulting in 20x increase in sizes). Our findings show that ConvQA models with robust training via reformulations, significantly outperform those with standard training from gold QA pairs only.

{{</citation>}}


### (35/136) DistillCSE: Distilled Contrastive Learning for Sentence Embeddings (Jiahao Xu et al., 2023)

{{<citation>}}

Jiahao Xu, Wei Shao, Lihui Chen, Lemao Liu. (2023)  
**DistillCSE: Distilled Contrastive Learning for Sentence Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, Embedding, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2310.13499v1)  

---


**ABSTRACT**  
This paper proposes the DistillCSE framework, which performs contrastive learning under the self-training paradigm with knowledge distillation. The potential advantage of DistillCSE is its self-enhancing feature: using a base model to provide additional supervision signals, a stronger model may be learned through knowledge distillation. However, the vanilla DistillCSE through the standard implementation of knowledge distillation only achieves marginal improvements due to severe overfitting. The further quantitative analyses demonstrate the reason that the standard knowledge distillation exhibits a relatively large variance of the teacher model's logits due to the essence of contrastive learning. To mitigate the issue induced by high variance, this paper accordingly proposed two simple yet effective solutions for knowledge distillation: a Group-P shuffling strategy as an implicit regularization and the averaging logits from multiple teacher components. Experiments on standard benchmarks demonstrate that the proposed DistillCSE outperforms many strong baseline methods and yields a new state-of-the-art performance.

{{</citation>}}


### (36/136) Mind the instructions: a holistic evaluation of consistency and interactions in prompt-based learning (Lucas Weber et al., 2023)

{{<citation>}}

Lucas Weber, Elia Bruni, Dieuwke Hupkes. (2023)  
**Mind the instructions: a holistic evaluation of consistency and interactions in prompt-based learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.13486v1)  

---


**ABSTRACT**  
Finding the best way of adapting pre-trained language models to a task is a big challenge in current NLP. Just like the previous generation of task-tuned models (TT), models that are adapted to tasks via in-context-learning (ICL) are robust in some setups but not in others. Here, we present a detailed analysis of which design choices cause instabilities and inconsistencies in LLM predictions. First, we show how spurious correlations between input distributions and labels -- a known issue in TT models -- form only a minor problem for prompted models. Then, we engage in a systematic, holistic evaluation of different factors that have been found to influence predictions in a prompting setup. We test all possible combinations of a range of factors on both vanilla and instruction-tuned (IT) LLMs of different scale and statistically analyse the results to show which factors are the most influential, interactive or stable. Our results show which factors can be used without precautions and which should be avoided or handled with care in most settings.

{{</citation>}}


### (37/136) Ask Language Model to Clean Your Noisy Translation Data (Quinten Bolding et al., 2023)

{{<citation>}}

Quinten Bolding, Baohao Liao, Brandon James Denis, Jun Luo, Christof Monz. (2023)  
**Ask Language Model to Clean Your Noisy Translation Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.13469v3)  

---


**ABSTRACT**  
Transformer models have demonstrated remarkable performance in neural machine translation (NMT). However, their vulnerability to noisy input poses a significant challenge in practical implementation, where generating clean output from noisy input is crucial. The MTNT dataset is widely used as a benchmark for evaluating the robustness of NMT models against noisy input. Nevertheless, its utility is limited due to the presence of noise in both the source and target sentences. To address this limitation, we focus on cleaning the noise from the target sentences in MTNT, making it more suitable as a benchmark for noise evaluation. Leveraging the capabilities of large language models (LLMs), we observe their impressive abilities in noise removal. For example, they can remove emojis while considering their semantic meaning. Additionally, we show that LLM can effectively rephrase slang, jargon, and profanities. The resulting datasets, called C-MTNT, exhibit significantly less noise in the target sentences while preserving the semantic integrity of the original sentences. Our human and GPT-4 evaluations also lead to a consistent conclusion that LLM performs well on this task. Lastly, experiments on C-MTNT showcased its effectiveness in evaluating the robustness of NMT models, highlighting the potential of advanced language models for data cleaning and emphasizing C-MTNT as a valuable resource.

{{</citation>}}


### (38/136) Steering Large Language Models for Machine Translation with Finetuning and In-Context Learning (Duarte M. Alves et al., 2023)

{{<citation>}}

Duarte M. Alves, Nuno M. Guerreiro, João Alves, José Pombal, Ricardo Rei, José G. C. de Souza, Pierre Colombo, André F. T. Martins. (2023)  
**Steering Large Language Models for Machine Translation with Finetuning and In-Context Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.13448v1)  

---


**ABSTRACT**  
Large language models (LLMs) are a promising avenue for machine translation (MT). However, current LLM-based MT systems are brittle: their effectiveness highly depends on the choice of few-shot examples and they often require extra post-processing due to overgeneration. Alternatives such as finetuning on translation instructions are computationally expensive and may weaken in-context learning capabilities, due to overspecialization. In this paper, we provide a closer look at this problem. We start by showing that adapter-based finetuning with LoRA matches the performance of traditional finetuning while reducing the number of training parameters by a factor of 50. This method also outperforms few-shot prompting and eliminates the need for post-processing or in-context examples. However, we show that finetuning generally degrades few-shot performance, hindering adaptation capabilities. Finally, to obtain the best of both worlds, we propose a simple approach that incorporates few-shot examples during finetuning. Experiments on 10 language pairs show that our proposed approach recovers the original few-shot capabilities while keeping the added benefits of finetuning.

{{</citation>}}


### (39/136) The Past, Present, and Future of Typological Databases in NLP (Emi Baylor et al., 2023)

{{<citation>}}

Emi Baylor, Esther Ploeger, Johannes Bjerva. (2023)  
**The Past, Present, and Future of Typological Databases in NLP**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.13440v1)  

---


**ABSTRACT**  
Typological information has the potential to be beneficial in the development of NLP models, particularly for low-resource languages. Unfortunately, current large-scale typological databases, notably WALS and Grambank, are inconsistent both with each other and with other sources of typological information, such as linguistic grammars. Some of these inconsistencies stem from coding errors or linguistic variation, but many of the disagreements are due to the discrete categorical nature of these databases. We shed light on this issue by systematically exploring disagreements across typological databases and resources, and their uses in NLP, covering the past and present. We next investigate the future of such work, offering an argument that a continuous view of typological features is clearly beneficial, echoing recommendations from linguistics. We propose that such a view of typology has significant potential in the future, including in language modeling in low-resource scenarios.

{{</citation>}}


### (40/136) Self-Consistency of Large Language Models under Ambiguity (Henning Bartsch et al., 2023)

{{<citation>}}

Henning Bartsch, Ole Jorgensen, Domenic Rosati, Jason Hoelscher-Obermaier, Jacob Pfau. (2023)  
**Self-Consistency of Large Language Models under Ambiguity**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13439v1)  

---


**ABSTRACT**  
Large language models (LLMs) that do not give consistent answers across contexts are problematic when used for tasks with expectations of consistency, e.g., question-answering, explanations, etc. Our work presents an evaluation benchmark for self-consistency in cases of under-specification where two or more answers can be correct. We conduct a series of behavioral experiments on the OpenAI model suite using an ambiguous integer sequence completion task. We find that average consistency ranges from 67\% to 82\%, far higher than would be predicted if a model's consistency was random, and increases as model capability improves. Furthermore, we show that models tend to maintain self-consistency across a series of robustness checks, including prompting speaker changes and sequence length changes. These results suggest that self-consistency arises as an emergent capability without specifically training for it. Despite this, we find that models are uncalibrated when judging their own consistency, with models displaying both over- and under-confidence. We also propose a nonparametric test for determining from token output distribution whether a model assigns non-trivial probability to alternative answers. Using this test, we find that despite increases in self-consistency, models usually place significant weight on alternative, inconsistent answers. This distribution of probability mass provides evidence that even highly self-consistent models internally compute multiple possible responses.

{{</citation>}}


### (41/136) Towards Enhancing Relational Rules for Knowledge Graph Link Prediction (Shuhan Wu et al., 2023)

{{<citation>}}

Shuhan Wu, Huaiyu Wan, Wei Chen, Yuting Wu, Junfeng Shen, Youfang Lin. (2023)  
**Towards Enhancing Relational Rules for Knowledge Graph Link Prediction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GNN, Graph Neural Network, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.13411v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have shown promising performance for knowledge graph reasoning. A recent variant of GNN called progressive relational graph neural network (PRGNN), utilizes relational rules to infer missing knowledge in relational digraphs and achieves notable results. However, during reasoning with PRGNN, two important properties are often overlooked: (1) the sequentiality of relation composition, where the order of combining different relations affects the semantics of the relational rules, and (2) the lagged entity information propagation, where the transmission speed of required information lags behind the appearance speed of new entities. Ignoring these properties leads to incorrect relational rule learning and decreased reasoning accuracy. To address these issues, we propose a novel knowledge graph reasoning approach, the Relational rUle eNhanced Graph Neural Network (RUN-GNN). Specifically, RUN-GNN employs a query related fusion gate unit to model the sequentiality of relation composition and utilizes a buffering update mechanism to alleviate the negative effect of lagged entity information propagation, resulting in higher-quality relational rule learning. Experimental results on multiple datasets demonstrate the superiority of RUN-GNN is superior on both transductive and inductive link prediction tasks.

{{</citation>}}


### (42/136) Explicit Alignment and Many-to-many Entailment Based Reasoning for Conversational Machine Reading (Yangyang Luo et al., 2023)

{{<citation>}}

Yangyang Luo, Shiyu Tian, Caixia Yuan, Xiaojie Wang. (2023)  
**Explicit Alignment and Many-to-many Entailment Based Reasoning for Conversational Machine Reading**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.13409v1)  

---


**ABSTRACT**  
Conversational Machine Reading (CMR) requires answering a user's initial question through multi-turn dialogue interactions based on a given document. Although there exist many effective methods, they largely neglected the alignment between the document and the user-provided information, which significantly affects the intermediate decision-making and subsequent follow-up question generation. To address this issue, we propose a pipeline framework that (1) aligns the aforementioned two sides in an explicit way, (2)makes decisions using a lightweight many-to-many entailment reasoning module, and (3) directly generates follow-up questions based on the document and previously asked questions. Our proposed method achieves state-of-the-art in micro-accuracy and ranks the first place on the public leaderboard of the CMR benchmark dataset ShARC.

{{</citation>}}


### (43/136) Cache me if you Can: an Online Cost-aware Teacher-Student framework to Reduce the Calls to Large Language Models (Ilias Stogiannidis et al., 2023)

{{<citation>}}

Ilias Stogiannidis, Stavros Vassos, Prodromos Malakasiotis, Ion Androutsopoulos. (2023)  
**Cache me if you Can: an Online Cost-aware Teacher-Student framework to Reduce the Calls to Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13395v1)  

---


**ABSTRACT**  
Prompting Large Language Models (LLMs) performs impressively in zero- and few-shot settings. Hence, small and medium-sized enterprises (SMEs) that cannot afford the cost of creating large task-specific training datasets, but also the cost of pretraining their own LLMs, are increasingly turning to third-party services that allow them to prompt LLMs. However, such services currently require a payment per call, which becomes a significant operating expense (OpEx). Furthermore, customer inputs are often very similar over time, hence SMEs end-up prompting LLMs with very similar instances. We propose a framework that allows reducing the calls to LLMs by caching previous LLM responses and using them to train a local inexpensive model on the SME side. The framework includes criteria for deciding when to trust the local model or call the LLM, and a methodology to tune the criteria and measure the tradeoff between performance and cost. For experimental purposes, we instantiate our framework with two LLMs, GPT-3.5 or GPT-4, and two inexpensive students, a k-NN classifier or a Multi-Layer Perceptron, using two common business tasks, intent recognition and sentiment analysis. Experimental results indicate that significant OpEx savings can be obtained with only slightly lower performance.

{{</citation>}}


### (44/136) POSQA: Probe the World Models of LLMs with Size Comparisons (Chang Shu et al., 2023)

{{<citation>}}

Chang Shu, Jiuzhou Han, Fangyu Liu, Ehsan Shareghi, Nigel Collier. (2023)  
**POSQA: Probe the World Models of LLMs with Size Comparisons**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.13394v1)  

---


**ABSTRACT**  
Embodied language comprehension emphasizes that language understanding is not solely a matter of mental processing in the brain but also involves interactions with the physical and social environment. With the explosive growth of Large Language Models (LLMs) and their already ubiquitous presence in our daily lives, it is becoming increasingly necessary to verify their real-world understanding. Inspired by cognitive theories, we propose POSQA: a Physical Object Size Question Answering dataset with simple size comparison questions to examine the extremity and analyze the potential mechanisms of the embodied comprehension of the latest LLMs.   We show that even the largest LLMs today perform poorly under the zero-shot setting. We then push their limits with advanced prompting techniques and external knowledge augmentation. Furthermore, we investigate whether their real-world comprehension primarily derives from contextual information or internal weights and analyse the impact of prompt formats and report bias of different objects. Our results show that real-world understanding that LLMs shaped from textual data can be vulnerable to deception and confusion by the surface form of prompts, which makes it less aligned with human behaviours.

{{</citation>}}


### (45/136) Tuna: Instruction Tuning using Feedback from Large Language Models (Haoran Li et al., 2023)

{{<citation>}}

Haoran Li, Yiran Liu, Xingxing Zhang, Wei Lu, Furu Wei. (2023)  
**Tuna: Instruction Tuning using Feedback from Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, LLaMA, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.13385v1)  

---


**ABSTRACT**  
Instruction tuning of open-source large language models (LLMs) like LLaMA, using direct outputs from more powerful LLMs such as Instruct-GPT and GPT-4, has proven to be a cost-effective way to align model behaviors with human preferences. However, the instruction-tuned model has only seen one response per instruction, lacking the knowledge of potentially better responses. In this paper, we propose finetuning an instruction-tuned LLM using our novel \textit{probabilistic ranking} and \textit{contextual ranking} approaches to increase the likelihood of generating better responses. Probabilistic ranking enables the instruction-tuned model to inherit the relative rankings of high-quality and low-quality responses from the teacher LLM. On the other hand, learning with contextual ranking allows the model to refine its own response distribution using the contextual understanding ability of stronger LLMs. Furthermore, we apply probabilistic ranking and contextual ranking sequentially to the instruction-tuned LLM. The resulting model, which we call \textbf{Tuna}, consistently improves the performance on Super Natural Instructions (119 test tasks), LMentry (25 test tasks), Vicuna QA, and can even obtain better results than several strong reinforcement learning baselines. Our code and data are available at \url{ https://github.com/microsoft/LMOps}.

{{</citation>}}


### (46/136) Towards General Error Diagnosis via Behavioral Testing in Machine Translation (Junjie Wu et al., 2023)

{{<citation>}}

Junjie Wu, Lemao Liu, Dit-Yan Yeung. (2023)  
**Towards General Error Diagnosis via Behavioral Testing in Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-SE, cs.CL  
Keywords: Machine Translation, NLP  
[Paper Link](http://arxiv.org/abs/2310.13362v1)  

---


**ABSTRACT**  
Behavioral testing offers a crucial means of diagnosing linguistic errors and assessing capabilities of NLP models. However, applying behavioral testing to machine translation (MT) systems is challenging as it generally requires human efforts to craft references for evaluating the translation quality of such systems on newly generated test cases. Existing works in behavioral testing of MT systems circumvent this by evaluating translation quality without references, but this restricts diagnosis to specific types of errors, such as incorrect translation of single numeric or currency words. In order to diagnose general errors, this paper proposes a new Bilingual Translation Pair Generation based Behavior Testing (BTPGBT) framework for conducting behavioral testing of MT systems. The core idea of BTPGBT is to employ a novel bilingual translation pair generation (BTPG) approach that automates the construction of high-quality test cases and their pseudoreferences. Experimental results on various MT systems demonstrate that BTPGBT could provide comprehensive and accurate behavioral testing results for general error diagnosis, which further leads to several insightful findings. Our code and data are available at https: //github.com/wujunjie1998/BTPGBT.

{{</citation>}}


### (47/136) Challenges and Contributing Factors in the Utilization of Large Language Models (LLMs) (Xiaoliang Chen et al., 2023)

{{<citation>}}

Xiaoliang Chen, Liangbin Li, Le Chang, Yunhe Huang, Yuxuan Zhao, Yuxiao Zhang, Dinuo Li. (2023)  
**Challenges and Contributing Factors in the Utilization of Large Language Models (LLMs)**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13343v1)  

---


**ABSTRACT**  
With the development of large language models (LLMs) like the GPT series, their widespread use across various application scenarios presents a myriad of challenges. This review initially explores the issue of domain specificity, where LLMs may struggle to provide precise answers to specialized questions within niche fields. The problem of knowledge forgetting arises as these LLMs might find it hard to balance old and new information. The knowledge repetition phenomenon reveals that sometimes LLMs might deliver overly mechanized responses, lacking depth and originality. Furthermore, knowledge illusion describes situations where LLMs might provide answers that seem insightful but are actually superficial, while knowledge toxicity focuses on harmful or biased information outputs. These challenges underscore problems in the training data and algorithmic design of LLMs. To address these issues, it's suggested to diversify training data, fine-tune models, enhance transparency and interpretability, and incorporate ethics and fairness training. Future technological trends might lean towards iterative methodologies, multimodal learning, model personalization and customization, and real-time learning and feedback mechanisms. In conclusion, future LLMs should prioritize fairness, transparency, and ethics, ensuring they uphold high moral and ethical standards when serving humanity.

{{</citation>}}


### (48/136) Large-Scale and Multi-Perspective Opinion Summarization with Diverse Review Subsets (Han Jiang et al., 2023)

{{<citation>}}

Han Jiang, Rui Wang, Zhihua Wei, Yu Li, Xinpeng Wang. (2023)  
**Large-Scale and Multi-Perspective Opinion Summarization with Diverse Review Subsets**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2310.13340v1)  

---


**ABSTRACT**  
Opinion summarization is expected to digest larger review sets and provide summaries from different perspectives. However, most existing solutions are deficient in epitomizing extensive reviews and offering opinion summaries from various angles due to the lack of designs for information selection. To this end, we propose SUBSUMM, a supervised summarization framework for large-scale multi-perspective opinion summarization. SUBSUMM consists of a review sampling strategy set and a two-stage training scheme. The sampling strategies take sentiment orientation and contrastive information value into consideration, with which the review subsets from different perspectives and quality levels can be selected. Subsequently, the summarizer is encouraged to learn from the sub-optimal and optimal subsets successively in order to capitalize on the massive input. Experimental results on AmaSum and Rotten Tomatoes datasets demonstrate that SUBSUMM is adept at generating pros, cons, and verdict summaries from hundreds of input reviews. Furthermore, our in-depth analysis verifies that the advanced selection of review subsets and the two-stage training scheme are vital to boosting the summarization performance.

{{</citation>}}


### (49/136) Democratizing Reasoning Ability: Tailored Learning from Large Language Model (Zhaoyang Wang et al., 2023)

{{<citation>}}

Zhaoyang Wang, Shaohan Huang, Yuxuan Liu, Jiahai Wang, Minghui Song, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, Qi Zhang. (2023)  
**Democratizing Reasoning Ability: Tailored Learning from Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.13332v1)  

---


**ABSTRACT**  
Large language models (LLMs) exhibit impressive emergent abilities in natural language processing, but their democratization is hindered due to huge computation requirements and closed-source nature. Recent research on advancing open-source smaller LMs by distilling knowledge from black-box LLMs has obtained promising results in the instruction-following ability. However, the reasoning ability which is more challenging to foster, is relatively rarely explored. In this paper, we propose a tailored learning approach to distill such reasoning ability to smaller LMs to facilitate the democratization of the exclusive reasoning ability. In contrast to merely employing LLM as a data annotator, we exploit the potential of LLM as a reasoning teacher by building an interactive multi-round learning paradigm. This paradigm enables the student to expose its deficiencies to the black-box teacher who then can provide customized training data in return. Further, to exploit the reasoning potential of the smaller LM, we propose self-reflection learning to motivate the student to learn from self-made mistakes. The learning from self-reflection and LLM are all tailored to the student's learning status, thanks to the seamless integration with the multi-round learning paradigm. Comprehensive experiments and analysis on mathematical and commonsense reasoning tasks demonstrate the effectiveness of our method. The code will be available at https://github.com/Raibows/Learn-to-Reason.

{{</citation>}}


### (50/136) Zero-Shot Sharpness-Aware Quantization for Pre-trained Language Models (Miaoxi Zhu et al., 2023)

{{<citation>}}

Miaoxi Zhu, Qihuang Zhong, Li Shen, Liang Ding, Juhua Liu, Bo Du, Dacheng Tao. (2023)  
**Zero-Shot Sharpness-Aware Quantization for Pre-trained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Quantization, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.13315v1)  

---


**ABSTRACT**  
Quantization is a promising approach for reducing memory overhead and accelerating inference, especially in large pre-trained language model (PLM) scenarios. While having no access to original training data due to security and privacy concerns has emerged the demand for zero-shot quantization. Most of the cutting-edge zero-shot quantization methods primarily 1) apply to computer vision tasks, and 2) neglect of overfitting problem in the generative adversarial learning process, leading to sub-optimal performance. Motivated by this, we propose a novel zero-shot sharpness-aware quantization (ZSAQ) framework for the zero-shot quantization of various PLMs. The key algorithm in solving ZSAQ is the SAM-SGA optimization, which aims to improve the quantization accuracy and model generalization via optimizing a minimax problem. We theoretically prove the convergence rate for the minimax optimization problem and this result can be applied to other nonconvex-PL minimax optimization frameworks. Extensive experiments on 11 tasks demonstrate that our method brings consistent and significant performance gains on both discriminative and generative PLMs, i.e., up to +6.98 average score. Furthermore, we empirically validate that our method can effectively improve the model generalization.

{{</citation>}}


### (51/136) Exploring the Impact of Corpus Diversity on Financial Pretrained Language Models (Jaeyoung Choe et al., 2023)

{{<citation>}}

Jaeyoung Choe, Keonwoong Noh, Nayeon Kim, Seyun Ahn, Woohwan Jung. (2023)  
**Exploring the Impact of Corpus Diversity on Financial Pretrained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Financial, Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2310.13312v1)  

---


**ABSTRACT**  
Over the past few years, various domain-specific pretrained language models (PLMs) have been proposed and have outperformed general-domain PLMs in specialized areas such as biomedical, scientific, and clinical domains. In addition, financial PLMs have been studied because of the high economic impact of financial data analysis. However, we found that financial PLMs were not pretrained on sufficiently diverse financial data. This lack of diverse training data leads to a subpar generalization performance, resulting in general-purpose PLMs, including BERT, often outperforming financial PLMs on many downstream tasks. To address this issue, we collected a broad range of financial corpus and trained the Financial Language Model (FiLM) on these diverse datasets. Our experimental results confirm that FiLM outperforms not only existing financial PLMs but also general domain PLMs. Furthermore, we provide empirical evidence that this improvement can be achieved even for unseen corpus groups.

{{</citation>}}


### (52/136) Test-Time Self-Adaptive Small Language Models for Question Answering (Soyeong Jeong et al., 2023)

{{<citation>}}

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park. (2023)  
**Test-Time Self-Adaptive Small Language Models for Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.13307v1)  

---


**ABSTRACT**  
Recent instruction-finetuned large language models (LMs) have achieved notable performances in various tasks, such as question-answering (QA). However, despite their ability to memorize a vast amount of general knowledge across diverse tasks, they might be suboptimal on specific tasks due to their limited capacity to transfer and adapt knowledge to target tasks. Moreover, further finetuning LMs with labeled datasets is often infeasible due to their absence, but it is also questionable if we can transfer smaller LMs having limited knowledge only with unlabeled test data. In this work, we show and investigate the capabilities of smaller self-adaptive LMs, only with unlabeled test data. In particular, we first stochastically generate multiple answers, and then ensemble them while filtering out low-quality samples to mitigate noise from inaccurate labels. Our proposed self-adaption strategy demonstrates significant performance improvements on benchmark QA datasets with higher robustness across diverse prompts, enabling LMs to stay stable. Code is available at: https://github.com/starsuzi/T-SAS.

{{</citation>}}


### (53/136) Decoding the Silent Majority: Inducing Belief Augmented Social Graph with Large Language Model for Response Forecasting (Chenkai Sun et al., 2023)

{{<citation>}}

Chenkai Sun, Jinning Li, Yi R. Fung, Hou Pong Chan, Tarek Abdelzaher, ChengXiang Zhai, Heng Ji. (2023)  
**Decoding the Silent Majority: Inducing Belief Augmented Social Graph with Large Language Model for Response Forecasting**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13297v1)  

---


**ABSTRACT**  
Automatic response forecasting for news media plays a crucial role in enabling content producers to efficiently predict the impact of news releases and prevent unexpected negative outcomes such as social conflict and moral injury. To effectively forecast responses, it is essential to develop measures that leverage the social dynamics and contextual information surrounding individuals, especially in cases where explicit profiles or historical actions of the users are limited (referred to as lurkers). As shown in a previous study, 97% of all tweets are produced by only the most active 25% of users. However, existing approaches have limited exploration of how to best process and utilize these important features. To address this gap, we propose a novel framework, named SocialSense, that leverages a large language model to induce a belief-centered graph on top of an existent social network, along with graph-based propagation to capture social dynamics. We hypothesize that the induced graph that bridges the gap between distant users who share similar beliefs allows the model to effectively capture the response patterns. Our method surpasses existing state-of-the-art in experimental evaluations for both zero-shot and supervised settings, demonstrating its effectiveness in response forecasting. Moreover, the analysis reveals the framework's capability to effectively handle unseen user and lurker scenarios, further highlighting its robustness and practical applicability.

{{</citation>}}


### (54/136) Assessing Privacy Risks in Language Models: A Case Study on Summarization Tasks (Ruixiang Tang et al., 2023)

{{<citation>}}

Ruixiang Tang, Gord Lueck, Rodolfo Quispe, Huseyin A Inan, Janardhan Kulkarni, Xia Hu. (2023)  
**Assessing Privacy Risks in Language Models: A Case Study on Summarization Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP, Summarization  
[Paper Link](http://arxiv.org/abs/2310.13291v1)  

---


**ABSTRACT**  
Large language models have revolutionized the field of NLP by achieving state-of-the-art performance on various tasks. However, there is a concern that these models may disclose information in the training data. In this study, we focus on the summarization task and investigate the membership inference (MI) attack: given a sample and black-box access to a model's API, it is possible to determine if the sample was part of the training data. We exploit text similarity and the model's resistance to document modifications as potential MI signals and evaluate their effectiveness on widely used datasets. Our results demonstrate that summarization models are at risk of exposing data membership, even in cases where the reference summary is not available. Furthermore, we discuss several safeguards for training summarization models to protect against MI attacks and discuss the inherent trade-off between privacy and utility.

{{</citation>}}


### (55/136) MoqaGPT : Zero-Shot Multi-modal Open-domain Question Answering with Large Language Model (Le Zhang et al., 2023)

{{<citation>}}

Le Zhang, Yihong Wu, Fengran Mo, Jian-Yun Nie, Aishwarya Agrawal. (2023)  
**MoqaGPT : Zero-Shot Multi-modal Open-domain Question Answering with Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, QA, Question Answering, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.13265v1)  

---


**ABSTRACT**  
Multi-modal open-domain question answering typically requires evidence retrieval from databases across diverse modalities, such as images, tables, passages, etc. Even Large Language Models (LLMs) like GPT-4 fall short in this task. To enable LLMs to tackle the task in a zero-shot manner, we introduce MoqaGPT, a straightforward and flexible framework. Using a divide-and-conquer strategy that bypasses intricate multi-modality ranking, our framework can accommodate new modalities and seamlessly transition to new models for the task. Built upon LLMs, MoqaGPT retrieves and extracts answers from each modality separately, then fuses this multi-modal information using LLMs to produce a final answer. Our methodology boosts performance on the MMCoQA dataset, improving F1 by +37.91 points and EM by +34.07 points over the supervised baseline. On the MultiModalQA dataset, MoqaGPT surpasses the zero-shot baseline, improving F1 by 9.5 points and EM by 10.1 points, and significantly closes the gap with supervised methods. Our codebase is available at https://github.com/lezhang7/MOQAGPT.

{{</citation>}}


### (56/136) Anomaly Detection of Command Shell Sessions based on DistilBERT: Unsupervised and Supervised Approaches (Zefang Liu et al., 2023)

{{<citation>}}

Zefang Liu, John Buford. (2023)  
**Anomaly Detection of Command Shell Sessions based on DistilBERT: Unsupervised and Supervised Approaches**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs.CL  
Keywords: Anomaly Detection, BERT  
[Paper Link](http://arxiv.org/abs/2310.13247v1)  

---


**ABSTRACT**  
Anomaly detection in command shell sessions is a critical aspect of computer security. Recent advances in deep learning and natural language processing, particularly transformer-based models, have shown great promise for addressing complex security challenges. In this paper, we implement a comprehensive approach to detect anomalies in Unix shell sessions using a pretrained DistilBERT model, leveraging both unsupervised and supervised learning techniques to identify anomalous activity while minimizing data labeling. The unsupervised method captures the underlying structure and syntax of Unix shell commands, enabling the detection of session deviations from normal behavior. Experiments on a large-scale enterprise dataset collected from production systems demonstrate the effectiveness of our approach in detecting anomalous behavior in Unix shell sessions. This work highlights the potential of leveraging recent advances in transformers to address important computer security challenges.

{{</citation>}}


### (57/136) Multi-level Contrastive Learning for Script-based Character Understanding (Dawei Li et al., 2023)

{{<citation>}}

Dawei Li, Hengyuan Zhang, Yanran Li, Shiping Yang. (2023)  
**Multi-level Contrastive Learning for Script-based Character Understanding**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, ChatGPT, Contrastive Learning, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2310.13231v1)  

---


**ABSTRACT**  
In this work, we tackle the scenario of understanding characters in scripts, which aims to learn the characters' personalities and identities from their utterances. We begin by analyzing several challenges in this scenario, and then propose a multi-level contrastive learning framework to capture characters' global information in a fine-grained manner. To validate the proposed framework, we conduct extensive experiments on three character understanding sub-tasks by comparing with strong pre-trained language models, including SpanBERT, Longformer, BigBird and ChatGPT-3.5. Experimental results demonstrate that our method improves the performances by a considerable margin. Through further in-depth analysis, we show the effectiveness of our method in addressing the challenges and provide more hints on the scenario of character understanding. We will open-source our work on github at https://github.com/David-Li0406/Script-based-Character-Understanding.

{{</citation>}}


### (58/136) The Less the Merrier? Investigating Language Representation in Multilingual Models (Hellina Hailu Nigatu et al., 2023)

{{<citation>}}

Hellina Hailu Nigatu, Atnafu Lambebo Tonja, Jugal Kalita. (2023)  
**The Less the Merrier? Investigating Language Representation in Multilingual Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Multilingual, NLP, Named Entity Recognition, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.13228v1)  

---


**ABSTRACT**  
Multilingual Language Models offer a way to incorporate multiple languages in one model and utilize cross-language transfer learning to improve performance for different Natural Language Processing (NLP) tasks. Despite progress in multilingual models, not all languages are supported as well, particularly in low-resource settings. In this work, we investigate the linguistic representation of different languages in multilingual models. We start by asking the question which languages are supported in popular multilingual models and which languages are left behind. Then, for included languages, we look at models' learned representations based on language family and dialect and try to understand how models' learned representations for~(1) seen and~(2) unseen languages vary across different language groups. In addition, we test and analyze performance on downstream tasks such as text generation and Named Entity Recognition. We observe from our experiments that community-centered models -- models that focus on languages of a given family or geographical location and are built by communities who speak them -- perform better at distinguishing between languages in the same family for low-resource languages. Our paper contributes to the literature in understanding multilingual models and their shortcomings and offers insights on potential ways to improve them.

{{</citation>}}


### (59/136) ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search (Yuchen Zhuang et al., 2023)

{{<citation>}}

Yuchen Zhuang, Xiang Chen, Tong Yu, Saayan Mitra, Victor Bursztyn, Ryan A. Rossi, Somdeb Sarkhel, Chao Zhang. (2023)  
**ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13227v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated powerful decision-making and planning capabilities in solving complicated real-world problems. LLM-based autonomous agents can interact with diverse tools (e.g., functional APIs) and generate solution plans that execute a series of API function calls in a step-by-step manner. The multitude of candidate API function calls significantly expands the action space, amplifying the critical need for efficient action space navigation. However, existing methods either struggle with unidirectional exploration in expansive action spaces, trapped into a locally optimal solution, or suffer from exhaustively traversing all potential actions, causing inefficient navigation. To address these issues, we propose ToolChain*, an efficient tree search-based planning algorithm for LLM-based agents. It formulates the entire action space as a decision tree, where each node represents a possible API function call involved in a solution plan. By incorporating the A* search algorithm with task-specific cost function design, it efficiently prunes high-cost branches that may involve incorrect actions, identifying the most low-cost valid path as the solution. Extensive experiments on multiple tool-use and reasoning tasks demonstrate that ToolChain* efficiently balances exploration and exploitation within an expansive action space. It outperforms state-of-the-art baselines on planning and reasoning tasks by 3.1% and 3.5% on average while requiring 7.35x and 2.31x less time, respectively.

{{</citation>}}


### (60/136) Enhancing Zero-Shot Crypto Sentiment with Fine-tuned Language Model and Prompt Engineering (Rahman S M Wahidur et al., 2023)

{{<citation>}}

Rahman S M Wahidur, Ishmam Tashdeed, Manjit Kaur, Heung-No-Lee. (2023)  
**Enhancing Zero-Shot Crypto Sentiment with Fine-tuned Language Model and Prompt Engineering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.13226v1)  

---


**ABSTRACT**  
Blockchain technology has revolutionized the financial landscape, with cryptocurrencies gaining widespread adoption for their decentralized and transparent nature. As the sentiment expressed on social media platforms can significantly influence cryptocurrency discussions and market movements, sentiment analysis has emerged as a crucial tool for understanding public opinion and predicting market trends. Motivated by the aim to enhance sentiment analysis accuracy in the cryptocurrency domain, this paper investigates fine-tuning techniques on large language models. This paper also investigates the efficacy of supervised fine-tuning and instruction-based fine-tuning on large language models for unseen tasks. Experimental results demonstrate a significant average zero-shot performance gain of 40% after fine-tuning, highlighting the potential of this technique in optimizing pre-trained language model efficiency. Additionally, the impact of instruction tuning on models of varying scales is examined, revealing that larger models benefit from instruction tuning, achieving the highest average accuracy score of 75.16%. In contrast, smaller-scale models may experience reduced generalization due to the complete utilization of model capacity. To gain deeper insight about how instruction works with these language models, this paper presents an experimental investigation into the response of an instruction-based model under different instruction tuning setups. The investigation demonstrates that the model achieves an average accuracy score of 72.38% for short and simple instructions. This performance significantly outperforms its accuracy under long and complex instructions by over 12%, thereby effectively highlighting the profound significance of instruction characteristics in maximizing model performance.

{{</citation>}}


### (61/136) MultiCoNER v2: a Large Multilingual dataset for Fine-grained and Noisy Named Entity Recognition (Besnik Fetahu et al., 2023)

{{<citation>}}

Besnik Fetahu, Zhiyu Chen, Sudipta Kar, Oleg Rokhlenko, Shervin Malmasi. (2023)  
**MultiCoNER v2: a Large Multilingual dataset for Fine-grained and Noisy Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Multilingual, NER, Named Entity Recognition, OCR  
[Paper Link](http://arxiv.org/abs/2310.13213v1)  

---


**ABSTRACT**  
We present MULTICONER V2, a dataset for fine-grained Named Entity Recognition covering 33 entity classes across 12 languages, in both monolingual and multilingual settings. This dataset aims to tackle the following practical challenges in NER: (i) effective handling of fine-grained classes that include complex entities like movie titles, and (ii) performance degradation due to noise generated from typing mistakes or OCR errors. The dataset is compiled from open resources like Wikipedia and Wikidata, and is publicly available. Evaluation based on the XLM-RoBERTa baseline highlights the unique challenges posed by MULTICONER V2: (i) the fine-grained taxonomy is challenging, where the scores are low with macro-F1=0.63 (across all languages), and (ii) the corruption strategy significantly impairs performance, with entity corruption resulting in 9% lower performance relative to non-entity corruptions across all languages. This highlights the greater impact of entity noise in contrast to context noise.

{{</citation>}}


### (62/136) Primacy Effect of ChatGPT (Yiwei Wang et al., 2023)

{{<citation>}}

Yiwei Wang, Yujun Cai, Muhao Chen, Yuxuan Liang, Bryan Hooi. (2023)  
**Primacy Effect of ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, NLU  
[Paper Link](http://arxiv.org/abs/2310.13206v1)  

---


**ABSTRACT**  
Instruction-tuned large language models (LLMs), such as ChatGPT, have led to promising zero-shot performance in discriminative natural language understanding (NLU) tasks. This involves querying the LLM using a prompt containing the question, and the candidate labels to choose from. The question-answering capabilities of ChatGPT arise from its pre-training on large amounts of human-written text, as well as its subsequent fine-tuning on human preferences, which motivates us to ask: Does ChatGPT also inherits humans' cognitive biases? In this paper, we study the primacy effect of ChatGPT: the tendency of selecting the labels at earlier positions as the answer. We have two main findings: i) ChatGPT's decision is sensitive to the order of labels in the prompt; ii) ChatGPT has a clearly higher chance to select the labels at earlier positions as the answer. We hope that our experiments and analyses provide additional insights into building more reliable ChatGPT-based solutions. We release the source code at https://github.com/wangywUST/PrimacyEffectGPT.

{{</citation>}}


## cs.LG (26)



### (63/136) Towards Subject Agnostic Affective Emotion Recognition (Amit Kumar Jaiswal et al., 2023)

{{<citation>}}

Amit Kumar Jaiswal, Haiming Liu, Prayag Tiwari. (2023)  
**Towards Subject Agnostic Affective Emotion Recognition**  

---
Primary Category: cs.LG  
Categories: cs-HC, cs-LG, cs-MM, cs.LG  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2310.15189v1)  

---


**ABSTRACT**  
This paper focuses on affective emotion recognition, aiming to perform in the subject-agnostic paradigm based on EEG signals. However, EEG signals manifest subject instability in subject-agnostic affective Brain-computer interfaces (aBCIs), which led to the problem of distributional shift. Furthermore, this problem is alleviated by approaches such as domain generalisation and domain adaptation. Typically, methods based on domain adaptation confer comparatively better results than the domain generalisation methods but demand more computational resources given new subjects. We propose a novel framework, meta-learning based augmented domain adaptation for subject-agnostic aBCIs. Our domain adaptation approach is augmented through meta-learning, which consists of a recurrent neural network, a classifier, and a distributional shift controller based on a sum-decomposable function. Also, we present that a neural network explicating a sum-decomposable function can effectively estimate the divergence between varied domains. The network setting for augmented domain adaptation follows meta-learning and adversarial learning, where the controller promptly adapts to new domains employing the target data via a few self-adaptation steps in the test phase. Our proposed approach is shown to be effective in experiments on a public aBICs dataset and achieves similar performance to state-of-the-art domain adaptation methods while avoiding the use of additional computational resources.

{{</citation>}}


### (64/136) Augment with Care: Enhancing Graph Contrastive Learning with Selective Spectrum Perturbation (Kaiqi Yang et al., 2023)

{{<citation>}}

Kaiqi Yang, Haoyu Han, Wei Jin, Hui Liu. (2023)  
**Augment with Care: Enhancing Graph Contrastive Learning with Selective Spectrum Perturbation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.13845v1)  

---


**ABSTRACT**  
In recent years, Graph Contrastive Learning (GCL) has shown remarkable effectiveness in learning representations on graphs. As a component of GCL, good augmentation views are supposed to be invariant to the important information while discarding the unimportant part. Existing augmentation views with perturbed graph structures are usually based on random topology corruption in the spatial domain; however, from perspectives of the spectral domain, this approach may be ineffective as it fails to pose tailored impacts on the information of different frequencies, thus weakening the agreement between the augmentation views. By a preliminary experiment, we show that the impacts caused by spatial random perturbation are approximately evenly distributed among frequency bands, which may harm the invariance of augmentations required by contrastive learning frameworks. To address this issue, we argue that the perturbation should be selectively posed on the information concerning different frequencies. In this paper, we propose GASSER which poses tailored perturbation on the specific frequencies of graph structures in spectral domain, and the edge perturbation is selectively guided by the spectral hints. As shown by extensive experiments and theoretical analysis, the augmentation views are adaptive and controllable, as well as heuristically fitting the homophily ratios and spectrum of graph structures.

{{</citation>}}


### (65/136) Foundation Model's Embedded Representations May Detect Distribution Shift (Adam Tsou et al., 2023)

{{<citation>}}

Adam Tsou, Max Vargas, Andrew Engel, Tony Chiang. (2023)  
**Foundation Model's Embedded Representations May Detect Distribution Shift**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.13836v1)  

---


**ABSTRACT**  
Distribution shifts between train and test datasets obscure our ability to understand the generalization capacity of neural network models. This topic is especially relevant given the success of pre-trained foundation models as starting points for transfer learning (TL) models across tasks and contexts. We present a case study for TL on a pre-trained GPT-2 model onto the Sentiment140 dataset for sentiment classification. We show that Sentiment140's test dataset $M$ is not sampled from the same distribution as the training dataset $P$, and hence training on $P$ and measuring performance on $M$ does not actually account for the model's generalization on sentiment classification.

{{</citation>}}


### (66/136) Adversarial Attacks on Fairness of Graph Neural Networks (Binchi Zhang et al., 2023)

{{<citation>}}

Binchi Zhang, Yushun Dong, Chen Chen, Yada Zhu, Minnan Luo, Jundong Li. (2023)  
**Adversarial Attacks on Fairness of Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Attack, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.13822v1)  

---


**ABSTRACT**  
Fairness-aware graph neural networks (GNNs) have gained a surge of attention as they can reduce the bias of predictions on any demographic group (e.g., female) in graph-based applications. Although these methods greatly improve the algorithmic fairness of GNNs, the fairness can be easily corrupted by carefully designed adversarial attacks. In this paper, we investigate the problem of adversarial attacks on fairness of GNNs and propose G-FairAttack, a general framework for attacking various types of fairness-aware GNNs in terms of fairness with an unnoticeable effect on prediction utility. In addition, we propose a fast computation technique to reduce the time complexity of G-FairAttack. The experimental study demonstrates that G-FairAttack successfully corrupts the fairness of different types of GNNs while keeping the attack unnoticeable. Our study on fairness attacks sheds light on potential vulnerabilities in fairness-aware GNNs and guides further research on the robustness of GNNs in terms of fairness. The open-source code is available at https://github.com/zhangbinchi/G-FairAttack.

{{</citation>}}


### (67/136) FATA-Trans: Field And Time-Aware Transformer for Sequential Tabular Data (Dongyu Zhang et al., 2023)

{{<citation>}}

Dongyu Zhang, Liang Wang, Xin Dai, Shubham Jain, Junpeng Wang, Yujie Fan, Chin-Chia Michael Yeh, Yan Zheng, Zhongfang Zhuang, Wei Zhang. (2023)  
**FATA-Trans: Field And Time-Aware Transformer for Sequential Tabular Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.13818v1)  

---


**ABSTRACT**  
Sequential tabular data is one of the most commonly used data types in real-world applications. Different from conventional tabular data, where rows in a table are independent, sequential tabular data contains rich contextual and sequential information, where some fields are dynamically changing over time and others are static. Existing transformer-based approaches analyzing sequential tabular data overlook the differences between dynamic and static fields by replicating and filling static fields into each transformer, and ignore temporal information between rows, which leads to three major disadvantages: (1) computational overhead, (2) artificially simplified data for masked language modeling pre-training task that may yield less meaningful representations, and (3) disregarding the temporal behavioral patterns implied by time intervals. In this work, we propose FATA-Trans, a model with two field transformers for modeling sequential tabular data, where each processes static and dynamic field information separately. FATA-Trans is field- and time-aware for sequential tabular data. The field-type embedding in the method enables FATA-Trans to capture differences between static and dynamic fields. The time-aware position embedding exploits both order and time interval information between rows, which helps the model detect underlying temporal behavior in a sequence. Our experiments on three benchmark datasets demonstrate that the learned representations from FATA-Trans consistently outperform state-of-the-art solutions in the downstream tasks. We also present visualization studies to highlight the insights captured by the learned representations, enhancing our understanding of the underlying data. Our codes are available at https://github.com/zdy93/FATA-Trans.

{{</citation>}}


### (68/136) A Better Match for Drivers and Riders: Reinforcement Learning at Lyft (Xabi Azagirre et al., 2023)

{{<citation>}}

Xabi Azagirre, Akshay Balwally, Guillaume Candeli, Nicholas Chamandy, Benjamin Han, Alona King, Hyungjun Lee, Martin Loncaric, Sébastien Martin, Vijay Narasiman, Zhiwei, Qin, Baptiste Richard, Sara Smoot, Sean Taylor, Garrett van Ryzin, Di Wu, Fei Yu, Alex Zamoshchin. (2023)  
**A Better Match for Drivers and Riders: Reinforcement Learning at Lyft**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13810v1)  

---


**ABSTRACT**  
To better match drivers to riders in our ridesharing application, we revised Lyft's core matching algorithm. We use a novel online reinforcement learning approach that estimates the future earnings of drivers in real time and use this information to find more efficient matches. This change was the first documented implementation of a ridesharing matching algorithm that can learn and improve in real time. We evaluated the new approach during weeks of switchback experimentation in most Lyft markets, and estimated how it benefited drivers, riders, and the platform. In particular, it enabled our drivers to serve millions of additional riders each year, leading to more than $30 million per year in incremental revenue. Lyft rolled out the algorithm globally in 2021.

{{</citation>}}


### (69/136) Learning to (Learn at Test Time) (Yu Sun et al., 2023)

{{<citation>}}

Yu Sun, Xinhao Li, Karan Dalal, Chloe Hsu, Sanmi Koyejo, Carlos Guestrin, Xiaolong Wang, Tatsunori Hashimoto, Xinlei Chen. (2023)  
**Learning to (Learn at Test Time)**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.13807v1)  

---


**ABSTRACT**  
We reformulate the problem of supervised learning as learning to learn with two nested loops (i.e. learning problems). The inner loop learns on each individual instance with self-supervision before final prediction. The outer loop learns the self-supervised task used by the inner loop, such that its final prediction improves. Our inner loop turns out to be equivalent to linear attention when the inner-loop learner is only a linear model, and to self-attention when it is a kernel estimator. For practical comparison with linear or self-attention layers, we replace each of them in a transformer with an inner loop, so our outer loop is equivalent to training the architecture. When each inner-loop learner is a neural network, our approach vastly outperforms transformers with linear attention on ImageNet from 224 x 224 raw pixels in both accuracy and FLOPs, while (regular) transformers cannot run.

{{</citation>}}


### (70/136) Improving Molecular Properties Prediction Through Latent Space Fusion (Eduardo Soares et al., 2023)

{{<citation>}}

Eduardo Soares, Akihiro Kishimoto, Emilio Vital Brazil, Seiji Takeda, Hiroshi Kajino, Renato Cerqueira. (2023)  
**Improving Molecular Properties Prediction Through Latent Space Fusion**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-LG, cs.LG, q-bio-QM  
Keywords: GNN, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13802v1)  

---


**ABSTRACT**  
Pre-trained Language Models have emerged as promising tools for predicting molecular properties, yet their development is in its early stages, necessitating further research to enhance their efficacy and address challenges such as generalization and sample efficiency. In this paper, we present a multi-view approach that combines latent spaces derived from state-of-the-art chemical models. Our approach relies on two pivotal elements: the embeddings derived from MHG-GNN, which represent molecular structures as graphs, and MoLFormer embeddings rooted in chemical language. The attention mechanism of MoLFormer is able to identify relations between two atoms even when their distance is far apart, while the GNN of MHG-GNN can more precisely capture relations among multiple atoms closely located. In this work, we demonstrate the superior performance of our proposed multi-view approach compared to existing state-of-the-art methods, including MoLFormer-XL, which was trained on 1.1 billion molecules, particularly in intricate tasks such as predicting clinical trial drug toxicity and inhibiting HIV replication. We assessed our approach using six benchmark datasets from MoleculeNet, where it outperformed competitors in five of them. Our study highlights the potential of latent space fusion and feature integration for advancing molecular property prediction. In this work, we use small versions of MHG-GNN and MoLFormer, which opens up an opportunity for further improvement when our approach uses a larger-scale dataset.

{{</citation>}}


### (71/136) Enhancing Illicit Activity Detection using XAI: A Multimodal Graph-LLM Framework (Jack Nicholls et al., 2023)

{{<citation>}}

Jack Nicholls, Aditya Kuppa, Nhien-An Le-Khac. (2023)  
**Enhancing Illicit Activity Detection using XAI: A Multimodal Graph-LLM Framework**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Financial  
[Paper Link](http://arxiv.org/abs/2310.13787v1)  

---


**ABSTRACT**  
Financial cybercrime prevention is an increasing issue with many organisations and governments. As deep learning models have progressed to identify illicit activity on various financial and social networks, the explainability behind the model decisions has been lacklustre with the investigative analyst at the heart of any deep learning platform. In our paper, we present a state-of-the-art, novel multimodal proactive approach to addressing XAI in financial cybercrime detection.   We leverage a triad of deep learning models designed to distill essential representations from transaction sequencing, subgraph connectivity, and narrative generation to significantly streamline the analyst's investigative process. Our narrative generation proposal leverages LLM to ingest transaction details and output contextual narrative for an analyst to understand a transaction and its metadata much further.

{{</citation>}}


### (72/136) Graph AI in Medicine (Ruth Johnson et al., 2023)

{{<citation>}}

Ruth Johnson, Michelle M. Li, Ayush Noori, Owen Queen, Marinka Zitnik. (2023)  
**Graph AI in Medicine**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, GNN  
[Paper Link](http://arxiv.org/abs/2310.13767v1)  

---


**ABSTRACT**  
In clinical artificial intelligence (AI), graph representation learning, mainly through graph neural networks (GNNs), stands out for its capability to capture intricate relationships within structured clinical datasets. With diverse data -- from patient records to imaging -- GNNs process data holistically by viewing modalities as nodes interconnected by their relationships. Graph AI facilitates model transfer across clinical tasks, enabling models to generalize across patient populations without additional parameters or minimal re-training. However, the importance of human-centered design and model interpretability in clinical decision-making cannot be overstated. Since graph AI models capture information through localized neural transformations defined on graph relationships, they offer both an opportunity and a challenge in elucidating model rationale. Knowledge graphs can enhance interpretability by aligning model-driven insights with medical knowledge. Emerging graph models integrate diverse data modalities through pre-training, facilitate interactive feedback loops, and foster human-AI collaboration, paving the way to clinically meaningful predictions.

{{</citation>}}


### (73/136) CAPIVARA: Cost-Efficient Approach for Improving Multilingual CLIP Performance on Low-Resource Languages (Gabriel Oliveira dos Santos et al., 2023)

{{<citation>}}

Gabriel Oliveira dos Santos, Diego A. B. Moreira, Alef Iury Ferreira, Jhessica Silva, Luiz Pereira, Pedro Bueno, Thiago Sousa, Helena Maia, Nádia Da Silva, Esther Colombini, Helio Pedrini, Sandra Avila. (2023)  
**CAPIVARA: Cost-Efficient Approach for Improving Multilingual CLIP Performance on Low-Resource Languages**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Low-Resource, Multilingual  
[Paper Link](http://arxiv.org/abs/2310.13683v2)  

---


**ABSTRACT**  
This work introduces CAPIVARA, a cost-efficient framework designed to enhance the performance of multilingual CLIP models in low-resource languages. While CLIP has excelled in zero-shot vision-language tasks, the resource-intensive nature of model training remains challenging. Many datasets lack linguistic diversity, featuring solely English descriptions for images. CAPIVARA addresses this by augmenting text data using image captioning and machine translation to generate multiple synthetic captions in low-resource languages. We optimize the training pipeline with LiT, LoRA, and gradient checkpointing to alleviate the computational cost. Through extensive experiments, CAPIVARA emerges as state of the art in zero-shot tasks involving images and Portuguese texts. We show the potential for significant improvements in other low-resource languages, achieved by fine-tuning the pre-trained multilingual CLIP using CAPIVARA on a single GPU for 2 hours. Our model and code is available at https://github.com/hiaac-nlp/CAPIVARA.

{{</citation>}}


### (74/136) Automatic Unit Test Data Generation and Actor-Critic Reinforcement Learning for Code Synthesis (Philip John Gorinski et al., 2023)

{{<citation>}}

Philip John Gorinski, Matthieu Zimmer, Gerasimos Lampouras, Derrick Goh Xin Deik, Ignacio Iacobacci. (2023)  
**Automatic Unit Test Data Generation and Actor-Critic Reinforcement Learning for Code Synthesis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs-PL, cs.LG  
Keywords: Language Model, Natural Language Generation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13669v1)  

---


**ABSTRACT**  
The advent of large pre-trained language models in the domain of Code Synthesis has shown remarkable performance on various benchmarks, treating the problem of Code Generation in a fashion similar to Natural Language Generation, trained with a Language Modelling (LM) objective. In addition, the property of programming language code being precisely evaluable with respect to its semantics -- through the use of Unit Tests to check its functional correctness -- lends itself to using Reinforcement Learning (RL) as a further training paradigm. Previous work has shown that RL can be applied as such to improve models' coding capabilities; however, such RL-based methods rely on a reward signal based on defined Unit Tests, which are much harder to obtain compared to the huge crawled code datasets used in LM objectives. In this work, we present a novel approach to automatically obtain data consisting of function signatures and associated Unit Tests, suitable for RL training of Code Synthesis models. We also introduce a straightforward, simple yet effective Actor-Critic RL training scheme and show that it, in conjunction with automatically generated training data, leads to improvement of a pre-trained code language model's performance by up to 9.9% improvement over the original underlying code synthesis LM, and up to 4.3% over RL-based models trained with standard PPO or CodeRL.

{{</citation>}}


### (75/136) Contrastive Preference Learning: Learning from Human Feedback without RL (Joey Hejna et al., 2023)

{{<citation>}}

Joey Hejna, Rafael Rafailov, Harshit Sikchi, Chelsea Finn, Scott Niekum, W. Bradley Knox, Dorsa Sadigh. (2023)  
**Contrastive Preference Learning: Learning from Human Feedback without RL**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13639v2)  

---


**ABSTRACT**  
Reinforcement Learning from Human Feedback (RLHF) has emerged as a popular paradigm for aligning models with human intent. Typically RLHF algorithms operate in two phases: first, use human preferences to learn a reward function and second, align the model by optimizing the learned reward via reinforcement learning (RL). This paradigm assumes that human preferences are distributed according to reward, but recent work suggests that they instead follow the regret under the user's optimal policy. Thus, learning a reward function from feedback is not only based on a flawed assumption of human preference, but also leads to unwieldy optimization challenges that stem from policy gradients or bootstrapping in the RL phase. Because of these optimization challenges, contemporary RLHF methods restrict themselves to contextual bandit settings (e.g., as in large language models) or limit observation dimensionality (e.g., state-based robotics). We overcome these limitations by introducing a new family of algorithms for optimizing behavior from human feedback using the regret-based model of human preferences. Using the principle of maximum entropy, we derive Contrastive Preference Learning (CPL), an algorithm for learning optimal policies from preferences without learning reward functions, circumventing the need for RL. CPL is fully off-policy, uses only a simple contrastive objective, and can be applied to arbitrary MDPs. This enables CPL to elegantly scale to high-dimensional and sequential RLHF problems while being simpler than prior methods.

{{</citation>}}


### (76/136) ReLM: Leveraging Language Models for Enhanced Chemical Reaction Prediction (Yaorui Shi et al., 2023)

{{<citation>}}

Yaorui Shi, An Zhang, Enzhi Zhang, Zhiyuan Liu, Xiang Wang. (2023)  
**ReLM: Leveraging Language Models for Enhanced Chemical Reaction Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13590v1)  

---


**ABSTRACT**  
Predicting chemical reactions, a fundamental challenge in chemistry, involves forecasting the resulting products from a given reaction process. Conventional techniques, notably those employing Graph Neural Networks (GNNs), are often limited by insufficient training data and their inability to utilize textual information, undermining their applicability in real-world applications. In this work, we propose ReLM, a novel framework that leverages the chemical knowledge encoded in language models (LMs) to assist GNNs, thereby enhancing the accuracy of real-world chemical reaction predictions. To further enhance the model's robustness and interpretability, we incorporate the confidence score strategy, enabling the LMs to self-assess the reliability of their predictions. Our experimental results demonstrate that ReLM improves the performance of state-of-the-art GNN-based methods across various chemical reaction datasets, especially in out-of-distribution settings. Codes are available at https://github.com/syr-cn/ReLM.

{{</citation>}}


### (77/136) Tree Search in DAG Space with Model-based Reinforcement Learning for Causal Discovery (Victor-Alexandru Darvariu et al., 2023)

{{<citation>}}

Victor-Alexandru Darvariu, Stephen Hailes, Mirco Musolesi. (2023)  
**Tree Search in DAG Space with Model-based Reinforcement Learning for Causal Discovery**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13576v1)  

---


**ABSTRACT**  
Identifying causal structure is central to many fields ranging from strategic decision-making to biology and economics. In this work, we propose a model-based reinforcement learning method for causal discovery based on tree search, which builds directed acyclic graphs incrementally. We also formalize and prove the correctness of an efficient algorithm for excluding edges that would introduce cycles, which enables deeper discrete search and sampling in DAG space. We evaluate our approach on two real-world tasks, achieving substantially better performance than the state-of-the-art model-free method and greedy search, constituting a promising advancement for combinatorial methods.

{{</citation>}}


### (78/136) Reward Shaping for Happier Autonomous Cyber Security Agents (Elizabeth Bates et al., 2023)

{{<citation>}}

Elizabeth Bates, Vasilios Mavroudis, Chris Hicks. (2023)  
**Reward Shaping for Happier Autonomous Cyber Security Agents**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Cyber Security, Security  
[Paper Link](http://arxiv.org/abs/2310.13565v1)  

---


**ABSTRACT**  
As machine learning models become more capable, they have exhibited increased potential in solving complex tasks. One of the most promising directions uses deep reinforcement learning to train autonomous agents in computer network defense tasks. This work studies the impact of the reward signal that is provided to the agents when training for this task. Due to the nature of cybersecurity tasks, the reward signal is typically 1) in the form of penalties (e.g., when a compromise occurs), and 2) distributed sparsely across each defense episode. Such reward characteristics are atypical of classic reinforcement learning tasks where the agent is regularly rewarded for progress (cf. to getting occasionally penalized for failures). We investigate reward shaping techniques that could bridge this gap so as to enable agents to train more sample-efficiently and potentially converge to a better performance. We first show that deep reinforcement learning algorithms are sensitive to the magnitude of the penalties and their relative size. Then, we combine penalties with positive external rewards and study their effect compared to penalty-only training. Finally, we evaluate intrinsic curiosity as an internal positive reward mechanism and discuss why it might not be as advantageous for high-level network monitoring tasks.

{{</citation>}}


### (79/136) Learning Successor Representations with Distributed Hebbian Temporal Memory (Evgenii Dzhivelikian et al., 2023)

{{<citation>}}

Evgenii Dzhivelikian, Petr Kuderov, Aleksandr I. Panov. (2023)  
**Learning Successor Representations with Distributed Hebbian Temporal Memory**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.13391v1)  

---


**ABSTRACT**  
This paper presents a novel approach to address the challenge of online hidden representation learning for decision-making under uncertainty in non-stationary, partially observable environments. The proposed algorithm, Distributed Hebbian Temporal Memory (DHTM), is based on factor graph formalism and a multicomponent neuron model. DHTM aims to capture sequential data relationships and make cumulative predictions about future observations, forming Successor Representation (SR). Inspired by neurophysiological models of the neocortex, the algorithm utilizes distributed representations, sparse transition matrices, and local Hebbian-like learning rules to overcome the instability and slow learning process of traditional temporal memory algorithms like RNN and HMM. Experimental results demonstrate that DHTM outperforms classical LSTM and performs comparably to more advanced RNN-like algorithms, speeding up Temporal Difference learning for SR in changing environments. Additionally, we compare the SRs produced by DHTM to another biologically inspired HMM-like algorithm, CSCG. Our findings suggest that DHTM is a promising approach for addressing the challenges of online hidden representation learning in dynamic environments.

{{</citation>}}


### (80/136) Physics-Informed Graph Convolutional Networks: Towards a generalized framework for complex geometries (Marien Chenaud et al., 2023)

{{<citation>}}

Marien Chenaud, José Alves, Frédéric Magoulès. (2023)  
**Physics-Informed Graph Convolutional Networks: Towards a generalized framework for complex geometries**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-AP, math-MP, math-ph  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2310.14948v2)  

---


**ABSTRACT**  
Since the seminal work of [9] and their Physics-Informed neural networks (PINNs), many efforts have been conducted towards solving partial differential equations (PDEs) with Deep Learning models. However, some challenges remain, for instance the extension of such models to complex three-dimensional geometries, and a study on how such approaches could be combined to classical numerical solvers. In this work, we justify the use of graph neural networks for these problems, based on the similarity between these architectures and the meshes used in traditional numerical techniques for solving partial differential equations. After proving an issue with the Physics-Informed framework for complex geometries, during the computation of PDE residuals, an alternative procedure is proposed, by combining classical numerical solvers and the Physics-Informed framework. Finally, we propose an implementation of this approach, that we test on a three-dimensional problem on an irregular geometry.

{{</citation>}}


### (81/136) SigFormer: Signature Transformers for Deep Hedging (Anh Tong et al., 2023)

{{<citation>}}

Anh Tong, Thanh Nguyen-Tang, Dongeun Lee, Toan Tran, Jaesik Choi. (2023)  
**SigFormer: Signature Transformers for Deep Hedging**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.13369v1)  

---


**ABSTRACT**  
Deep hedging is a promising direction in quantitative finance, incorporating models and techniques from deep learning research. While giving excellent hedging strategies, models inherently requires careful treatment in designing architectures for neural networks. To mitigate such difficulties, we introduce SigFormer, a novel deep learning model that combines the power of path signatures and transformers to handle sequential data, particularly in cases with irregularities. Path signatures effectively capture complex data patterns, while transformers provide superior sequential attention. Our proposed model is empirically compared to existing methods on synthetic data, showcasing faster learning and enhanced robustness, especially in the presence of irregular underlying price data. Additionally, we validate our model performance through a real-world backtest on hedging the SP 500 index, demonstrating positive outcomes.

{{</citation>}}


### (82/136) Dissecting Causal Biases (Rūta Binkytė et al., 2023)

{{<citation>}}

Rūta Binkytė, Sami Zhioua, Yassine Turki. (2023)  
**Dissecting Causal Biases**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.13364v1)  

---


**ABSTRACT**  
Accurately measuring discrimination in machine learning-based automated decision systems is required to address the vital issue of fairness between subpopulations and/or individuals. Any bias in measuring discrimination can lead to either amplification or underestimation of the true value of discrimination. This paper focuses on a class of bias originating in the way training data is generated and/or collected. We call such class causal biases and use tools from the field of causality to formally define and analyze such biases. Four sources of bias are considered, namely, confounding, selection, measurement, and interaction. The main contribution of this paper is to provide, for each source of bias, a closed-form expression in terms of the model parameters. This makes it possible to analyze the behavior of each source of bias, in particular, in which cases they are absent and in which other cases they are maximized. We hope that the provided characterizations help the community better understand the sources of bias in machine learning applications.

{{</citation>}}


### (83/136) DIG-MILP: a Deep Instance Generator for Mixed-Integer Linear Programming with Feasibility Guarantee (Haoyu Wang et al., 2023)

{{<citation>}}

Haoyu Wang, Jialin Liu, Xiaohan Chen, Xinshang Wang, Pan Li, Wotao Yin. (2023)  
**DIG-MILP: a Deep Instance Generator for Mixed-Integer Linear Programming with Feasibility Guarantee**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.13261v1)  

---


**ABSTRACT**  
Mixed-integer linear programming (MILP) stands as a notable NP-hard problem pivotal to numerous crucial industrial applications. The development of effective algorithms, the tuning of solvers, and the training of machine learning models for MILP resolution all hinge on access to extensive, diverse, and representative data. Yet compared to the abundant naturally occurring data in image and text realms, MILP is markedly data deficient, underscoring the vital role of synthetic MILP generation. We present DIG-MILP, a deep generative framework based on variational auto-encoder (VAE), adept at extracting deep-level structural features from highly limited MILP data and producing instances that closely mirror the target data. Notably, by leveraging the MILP duality, DIG-MILP guarantees a correct and complete generation space as well as ensures the boundedness and feasibility of the generated instances. Our empirical study highlights the novelty and quality of the instances generated by DIG-MILP through two distinct downstream tasks: (S1) Data sharing, where solver solution times correlate highly positive between original and DIG-MILP-generated instances, allowing data sharing for solver tuning without publishing the original data; (S2) Data Augmentation, wherein the DIG-MILP-generated instances bolster the generalization performance of machine learning models tasked with resolving MILP problems.

{{</citation>}}


### (84/136) FLEE-GNN: A Federated Learning System for Edge-Enhanced Graph Neural Network in Analyzing Geospatial Resilience of Multicommodity Food Flows (Yuxiao Qu et al., 2023)

{{<citation>}}

Yuxiao Qu, Jinmeng Rao, Song Gao, Qianheng Zhang, Wei-Lun Chao, Yu Su, Michelle Miller, Alfonso Morales, Patrick Huber. (2023)  
**FLEE-GNN: A Federated Learning System for Edge-Enhanced Graph Neural Network in Analyzing Geospatial Resilience of Multicommodity Food Flows**  

---
Primary Category: cs.LG  
Categories: I-2, cs-AI, cs-CY, cs-LG, cs-SI, cs.LG  
Keywords: AI, GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.13248v1)  

---


**ABSTRACT**  
Understanding and measuring the resilience of food supply networks is a global imperative to tackle increasing food insecurity. However, the complexity of these networks, with their multidimensional interactions and decisions, presents significant challenges. This paper proposes FLEE-GNN, a novel Federated Learning System for Edge-Enhanced Graph Neural Network, designed to overcome these challenges and enhance the analysis of geospatial resilience of multicommodity food flow network, which is one type of spatial networks. FLEE-GNN addresses the limitations of current methodologies, such as entropy-based methods, in terms of generalizability, scalability, and data privacy. It combines the robustness and adaptability of graph neural networks with the privacy-conscious and decentralized aspects of federated learning on food supply network resilience analysis across geographical regions. This paper also discusses FLEE-GNN's innovative data generation techniques, experimental designs, and future directions for improvement. The results show the advancements of this approach to quantifying the resilience of multicommodity food flow networks, contributing to efforts towards ensuring global food security using AI methods. The developed FLEE-GNN has the potential to be applied in other spatial networks with spatially heterogeneous sub-network distributions.

{{</citation>}}


### (85/136) Transparency challenges in policy evaluation with causal machine learning -- improving usability and accountability (Patrick Rehill et al., 2023)

{{<citation>}}

Patrick Rehill, Nicholas Biddle. (2023)  
**Transparency challenges in policy evaluation with causal machine learning -- improving usability and accountability**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, econ-EM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13240v1)  

---


**ABSTRACT**  
Causal machine learning tools are beginning to see use in real-world policy evaluation tasks to flexibly estimate treatment effects. One issue with these methods is that the machine learning models used are generally black boxes, i.e., there is no globally interpretable way to understand how a model makes estimates. This is a clear problem in policy evaluation applications, particularly in government, because it is difficult to understand whether such models are functioning in ways that are fair, based on the correct interpretation of evidence and transparent enough to allow for accountability if things go wrong. However, there has been little discussion of transparency problems in the causal machine learning literature and how these might be overcome. This paper explores why transparency issues are a problem for causal machine learning in public policy evaluation applications and considers ways these problems might be addressed through explainable AI tools and by simplifying models in line with interpretable AI principles. It then applies these ideas to a case-study using a causal forest model to estimate conditional average treatment effects for a hypothetical change in the school leaving age in Australia. It shows that existing tools for understanding black-box predictive models are poorly suited to causal machine learning and that simplifying the model to make it interpretable leads to an unacceptable increase in error (in this application). It concludes that new tools are needed to properly understand causal machine learning models and the algorithms that fit them.

{{</citation>}}


### (86/136) Interpretable Deep Reinforcement Learning for Optimizing Heterogeneous Energy Storage Systems (Luolin Xiong et al., 2023)

{{<citation>}}

Luolin Xiong, Yang Tang, Chensheng Liu, Shuai Mao, Ke Meng, Zhaoyang Dong, Feng Qian. (2023)  
**Interpretable Deep Reinforcement Learning for Optimizing Heterogeneous Energy Storage Systems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.14783v1)  

---


**ABSTRACT**  
Energy storage systems (ESS) are pivotal component in the energy market, serving as both energy suppliers and consumers. ESS operators can reap benefits from energy arbitrage by optimizing operations of storage equipment. To further enhance ESS flexibility within the energy market and improve renewable energy utilization, a heterogeneous photovoltaic-ESS (PV-ESS) is proposed, which leverages the unique characteristics of battery energy storage (BES) and hydrogen energy storage (HES). For scheduling tasks of the heterogeneous PV-ESS, cost description plays a crucial role in guiding operator's strategies to maximize benefits. We develop a comprehensive cost function that takes into account degradation, capital, and operation/maintenance costs to reflect real-world scenarios. Moreover, while numerous methods excel in optimizing ESS energy arbitrage, they often rely on black-box models with opaque decision-making processes, limiting practical applicability. To overcome this limitation and enable transparent scheduling strategies, a prototype-based policy network with inherent interpretability is introduced. This network employs human-designed prototypes to guide decision-making by comparing similarities between prototypical situations and encountered situations, which allows for naturally explained scheduling strategies. Comparative results across four distinct cases underscore the effectiveness and practicality of our proposed pre-hoc interpretable optimization method when contrasted with black-box models.

{{</citation>}}


### (87/136) Scalable Neural Network Kernels (Arijit Sehanobish et al., 2023)

{{<citation>}}

Arijit Sehanobish, Krzysztof Choromanski, Yunfan Zhao, Avinava Dubey, Valerii Likhosherstov. (2023)  
**Scalable Neural Network Kernels**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.13225v1)  

---


**ABSTRACT**  
We introduce the concept of scalable neural network kernels (SNNKs), the replacements of regular feedforward layers (FFLs), capable of approximating the latter, but with favorable computational properties. SNNKs effectively disentangle the inputs from the parameters of the neural network in the FFL, only to connect them in the final computation via the dot-product kernel. They are also strictly more expressive, as allowing to model complicated relationships beyond the functions of the dot-products of parameter-input vectors. We also introduce the neural network bundling process that applies SNNKs to compactify deep neural network architectures, resulting in additional compression gains. In its extreme version, it leads to the fully bundled network whose optimal parameters can be expressed via explicit formulae for several loss functions (e.g. mean squared error), opening a possibility to bypass backpropagation. As a by-product of our analysis, we introduce the mechanism of the universal random features (or URFs), applied to instantiate several SNNK variants, and interesting on its own in the context of scalable kernel methods. We provide rigorous theoretical analysis of all these concepts as well as an extensive empirical evaluation, ranging from point-wise kernel estimation to Transformers' fine-tuning with novel adapter layers inspired by SNNKs. Our mechanism provides up to 5x reduction in the number of trainable parameters, while maintaining competitive accuracy.

{{</citation>}}


### (88/136) In-context Learning with Transformer Is Really Equivalent to a Contrastive Learning Pattern (Ruifeng Ren et al., 2023)

{{<citation>}}

Ruifeng Ren, Yong Liu. (2023)  
**In-context Learning with Transformer Is Really Equivalent to a Contrastive Learning Pattern**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.13220v1)  

---


**ABSTRACT**  
Pre-trained large language models based on Transformers have demonstrated amazing in-context learning (ICL) abilities. Given several demonstration examples, the models can implement new tasks without any parameter updates. However, it is still an open question to understand the mechanism of ICL. In this paper, we interpret the inference process of ICL as a gradient descent process in a contrastive learning pattern. Firstly, leveraging kernel methods, we establish the relationship between gradient descent and self-attention mechanism under generally used softmax attention setting instead of linear attention setting. Then, we analyze the corresponding gradient descent process of ICL from the perspective of contrastive learning without negative samples and discuss possible improvements of this contrastive learning pattern, based on which the self-attention layer can be further modified. Finally, we design experiments to support our opinions. To the best of our knowledge, our work is the first to provide the understanding of ICL from the perspective of contrastive learning and has the potential to facilitate future model design by referring to related works on contrastive learning.

{{</citation>}}


## cs.IR (6)



### (89/136) FABULA: Intelligence Report Generation Using Retrieval-Augmented Narrative Construction (Priyanka Ranade et al., 2023)

{{<citation>}}

Priyanka Ranade, Anupam Joshi. (2023)  
**FABULA: Intelligence Report Generation Using Retrieval-Augmented Narrative Construction**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13848v1)  

---


**ABSTRACT**  
Narrative construction is the process of representing disparate event information into a logical plot structure that models an end to end story. Intelligence analysis is an example of a domain that can benefit tremendously from narrative construction techniques, particularly in aiding analysts during the largely manual and costly process of synthesizing event information into comprehensive intelligence reports. Manual intelligence report generation is often prone to challenges such as integrating dynamic event information, writing fine-grained queries, and closing information gaps. This motivates the development of a system that retrieves and represents critical aspects of events in a form that aids in automatic generation of intelligence reports.   We introduce a Retrieval Augmented Generation (RAG) approach to augment prompting of an autoregressive decoder by retrieving structured information asserted in a knowledge graph to generate targeted information based on a narrative plot model. We apply our approach to the problem of neural intelligence report generation and introduce FABULA, framework to augment intelligence analysis workflows using RAG. An analyst can use FABULA to query an Event Plot Graph (EPG) to retrieve relevant event plot points, which can be used to augment prompting of a Large Language Model (LLM) during intelligence report generation. Our evaluation studies show that the plot points included in the generated intelligence reports have high semantic relevance, high coherency, and low data redundancy.

{{</citation>}}


### (90/136) Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language (Zekai Qu et al., 2023)

{{<citation>}}

Zekai Qu, Ruobing Xie, Chaojun Xiao, Yuan Yao, Zhiyuan Liu, Fengzong Lian, Zhanhui Kang, Jie Zhou. (2023)  
**Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.13540v1)  

---


**ABSTRACT**  
With the thriving of pre-trained language model (PLM) widely verified in various of NLP tasks, pioneer efforts attempt to explore the possible cooperation of the general textual information in PLM with the personalized behavioral information in user historical behavior sequences to enhance sequential recommendation (SR). However, despite the commonalities of input format and task goal, there are huge gaps between the behavioral and textual information, which obstruct thoroughly modeling SR as language modeling via PLM. To bridge the gap, we propose a novel Unified pre-trained language model enhanced sequential recommendation (UPSR), aiming to build a unified pre-trained recommendation model for multi-domain recommendation tasks. We formally design five key indicators, namely naturalness, domain consistency, informativeness, noise & ambiguity, and text length, to guide the text->item adaptation and behavior sequence->text sequence adaptation differently for pre-training and fine-tuning stages, which are essential but under-explored by previous works. In experiments, we conduct extensive evaluations on seven datasets with both tuning and zero-shot settings and achieve the overall best performance. Comprehensive model analyses also provide valuable insights for behavior modeling via PLM, shedding light on large pre-trained recommendation models. The source codes will be released in the future.

{{</citation>}}


### (91/136) Towards Multi-Subsession Conversational Recommendation (Yu Ji et al., 2023)

{{<citation>}}

Yu Ji, Qi Shen, Shixuan Zhu, Hang Yu, Yiming Zhang, Chuan Cui, Zhihua Wei. (2023)  
**Towards Multi-Subsession Conversational Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Conversational Recommendation  
[Paper Link](http://arxiv.org/abs/2310.13365v1)  

---


**ABSTRACT**  
Conversational recommendation systems (CRS) could acquire dynamic user preferences towards desired items through multi-round interactive dialogue. Previous CRS mainly focuses on the single conversation (subsession) that user quits after a successful recommendation, neglecting the common scenario where user has multiple conversations (multi-subsession) over a short period. Therefore, we propose a novel conversational recommendation scenario named Multi-Subsession Multi-round Conversational Recommendation (MSMCR), where user would still resort to CRS after several subsessions and might preserve vague interests, and system would proactively ask attributes to activate user interests in the current subsession. To fill the gap in this new CRS scenario, we devise a novel framework called Multi-Subsession Conversational Recommender with Activation Attributes (MSCAA). Specifically, we first develop a context-aware recommendation module, comprehensively modeling user interests from historical interactions, previous subsessions, and feedback in the current subsession. Furthermore, an attribute selection policy module is proposed to learn a flexible strategy for asking appropriate attributes to elicit user interests. Finally, we design a conversation policy module to manage the above two modules to decide actions between asking and recommending. Extensive experiments on four datasets verify the effectiveness of our MSCAA framework for the MSMCR setting.

{{</citation>}}


### (92/136) Knowledge Graph Context-Enhanced Diversified Recommendation (Xiaolong Liu et al., 2023)

{{<citation>}}

Xiaolong Liu, Liangwei Yang, Zhiwei Liu, Mingdai Yang, Chen Wang, Hao Peng, Philip S. Yu. (2023)  
**Knowledge Graph Context-Enhanced Diversified Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.13253v1)  

---


**ABSTRACT**  
The field of Recommender Systems (RecSys) has been extensively studied to enhance accuracy by leveraging users' historical interactions. Nonetheless, this persistent pursuit of accuracy frequently engenders diminished diversity, culminating in the well-recognized "echo chamber" phenomenon. Diversified RecSys has emerged as a countermeasure, placing diversity on par with accuracy and garnering noteworthy attention from academic circles and industry practitioners. This research explores the realm of diversified RecSys within the intricate context of knowledge graphs (KG). These KGs act as repositories of interconnected information concerning entities and items, offering a propitious avenue to amplify recommendation diversity through the incorporation of insightful contextual information. Our contributions include introducing an innovative metric, Entity Coverage, and Relation Coverage, which effectively quantifies diversity within the KG domain. Additionally, we introduce the Diversified Embedding Learning (DEL) module, meticulously designed to formulate user representations that possess an innate awareness of diversity. In tandem with this, we introduce a novel technique named Conditional Alignment and Uniformity (CAU). It adeptly encodes KG item embeddings while preserving contextual integrity. Collectively, our contributions signify a substantial stride towards augmenting the panorama of recommendation diversity within the realm of KG-informed RecSys paradigms.

{{</citation>}}


### (93/136) TempGNN: Temporal Graph Neural Networks for Dynamic Session-Based Recommendations (Eunkyu Oh et al., 2023)

{{<citation>}}

Eunkyu Oh, Taehun Kim. (2023)  
**TempGNN: Temporal Graph Neural Networks for Dynamic Session-Based Recommendations**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.13249v1)  

---


**ABSTRACT**  
Session-based recommendations which predict the next action by understanding a user's interaction behavior with items within a relatively short ongoing session have recently gained increasing popularity. Previous research has focused on capturing the dynamics of sequential dependencies from complicated item transitions in a session by means of recurrent neural networks, self-attention models, and recently, mostly graph neural networks. Despite the plethora of different models relying on the order of items in a session, few approaches have been proposed for dealing better with the temporal implications between interactions. We present Temporal Graph Neural Networks (TempGNN), a generic framework for capturing the structural and temporal dynamics in complex item transitions utilizing temporal embedding operators on nodes and edges on dynamic session graphs, represented as sequences of timed events. Extensive experimental results show the effectiveness and adaptability of the proposed method by plugging it into existing state-of-the-art models. Finally, TempGNN achieved state-of-the-art performance on two real-world e-commerce datasets.

{{</citation>}}


### (94/136) Open-source Large Language Models are Strong Zero-shot Query Likelihood Models for Document Ranking (Shengyao Zhuang et al., 2023)

{{<citation>}}

Shengyao Zhuang, Bing Liu, Bevan Koopman, Guido Zuccon. (2023)  
**Open-source Large Language Models are Strong Zero-shot Query Likelihood Models for Document Ranking**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13243v1)  

---


**ABSTRACT**  
In the field of information retrieval, Query Likelihood Models (QLMs) rank documents based on the probability of generating the query given the content of a document. Recently, advanced large language models (LLMs) have emerged as effective QLMs, showcasing promising ranking capabilities. This paper focuses on investigating the genuine zero-shot ranking effectiveness of recent LLMs, which are solely pre-trained on unstructured text data without supervised instruction fine-tuning. Our findings reveal the robust zero-shot ranking ability of such LLMs, highlighting that additional instruction fine-tuning may hinder effectiveness unless a question generation task is present in the fine-tuning dataset. Furthermore, we introduce a novel state-of-the-art ranking system that integrates LLM-based QLMs with a hybrid zero-shot retriever, demonstrating exceptional effectiveness in both zero-shot and few-shot scenarios. We make our codebase publicly available at https://github.com/ielab/llm-qlm.

{{</citation>}}


## cs.RO (5)



### (95/136) Transformers for Trajectory Optimization with Application to Spacecraft Rendezvous (Tommaso Guffanti et al., 2023)

{{<citation>}}

Tommaso Guffanti, Daniele Gammelli, Simone D'Amico, Marco Pavone. (2023)  
**Transformers for Trajectory Optimization with Application to Spacecraft Rendezvous**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.13831v1)  

---


**ABSTRACT**  
Reliable and efficient trajectory optimization methods are a fundamental need for autonomous dynamical systems, effectively enabling applications including rocket landing, hypersonic reentry, spacecraft rendezvous, and docking. Within such safety-critical application areas, the complexity of the emerging trajectory optimization problems has motivated the application of AI-based techniques to enhance the performance of traditional approaches. However, current AI-based methods either attempt to fully replace traditional control algorithms, thus lacking constraint satisfaction guarantees and incurring in expensive simulation, or aim to solely imitate the behavior of traditional methods via supervised learning. To address these limitations, this paper proposes the Autonomous Rendezvous Transformer (ART) and assesses the capability of modern generative models to solve complex trajectory optimization problems, both from a forecasting and control standpoint. Specifically, this work assesses the capabilities of Transformers to (i) learn near-optimal policies from previously collected data, and (ii) warm-start a sequential optimizer for the solution of non-convex optimal control problems, thus guaranteeing hard constraint satisfaction. From a forecasting perspective, results highlight how ART outperforms other learning-based architectures at predicting known fuel-optimal trajectories. From a control perspective, empirical analyses show how policies learned through Transformers are able to generate near-optimal warm-starts, achieving trajectories that are (i) more fuel-efficient, (ii) obtained in fewer sequential optimizer iterations, and (iii) computed with an overall runtime comparable to benchmarks based on convex optimization.

{{</citation>}}


### (96/136) Enhanced Low-Dimensional Sensing Mapless Navigation of Terrestrial Mobile Robots Using Double Deep Reinforcement Learning Techniques (Linda Dotto de Moraes et al., 2023)

{{<citation>}}

Linda Dotto de Moraes, Victor Augusto Kich, Alisson Henrique Kolling, Jair Augusto Bottega, Ricardo Bedin Grando, Anselmo Rafael Cukla, Daniel Fernando Tello Gamarra. (2023)  
**Enhanced Low-Dimensional Sensing Mapless Navigation of Terrestrial Mobile Robots Using Double Deep Reinforcement Learning Techniques**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13809v1)  

---


**ABSTRACT**  
In this study, we present two distinct approaches within the realm of Deep Reinforcement Learning (Deep-RL) aimed at enhancing mapless navigation for a ground-based mobile robot. The research methodology primarily involves a comparative analysis between a Deep-RL strategy grounded in the foundational Deep Q-Network (DQN) algorithm, and an alternative approach based on the Double Deep Q-Network (DDQN) algorithm. The agents in these approaches leverage 24 measurements from laser range sampling, coupled with the agent's positional differentials and orientation relative to the target. This amalgamation of data influences the agents' determinations regarding navigation, ultimately dictating the robot's velocities. By embracing this parsimonious sensory framework as proposed, we successfully showcase the training of an agent for proficiently executing navigation tasks and adeptly circumventing obstacles. Notably, this accomplishment is attained without a dependency on intricate sensory inputs like those inherent to image-centric methodologies. The proposed methodology is evaluated in three different real environments, revealing that Double Deep structures significantly enhance the navigation capabilities of mobile robots compared to simple Q structures.

{{</citation>}}


### (97/136) RL-X: A Deep Reinforcement Learning Library (not only) for RoboCup (Nico Bohlinger et al., 2023)

{{<citation>}}

Nico Bohlinger, Klaus Dorer. (2023)  
**RL-X: A Deep Reinforcement Learning Library (not only) for RoboCup**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13396v1)  

---


**ABSTRACT**  
This paper presents the new Deep Reinforcement Learning (DRL) library RL-X and its application to the RoboCup Soccer Simulation 3D League and classic DRL benchmarks. RL-X provides a flexible and easy-to-extend codebase with self-contained single directory algorithms. Through the fast JAX-based implementations, RL-X can reach up to 4.5x speedups compared to well-known frameworks like Stable-Baselines3.

{{</citation>}}


### (98/136) PathRL: An End-to-End Path Generation Method for Collision Avoidance via Deep Reinforcement Learning (Wenhao Yu et al., 2023)

{{<citation>}}

Wenhao Yu, Jie Peng, Quecheng Qiu, Hanyu Wang, Lu Zhang, Jianmin Ji. (2023)  
**PathRL: An End-to-End Path Generation Method for Collision Avoidance via Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13295v1)  

---


**ABSTRACT**  
Robot navigation using deep reinforcement learning (DRL) has shown great potential in improving the performance of mobile robots. Nevertheless, most existing DRL-based navigation methods primarily focus on training a policy that directly commands the robot with low-level controls, like linear and angular velocities, which leads to unstable speeds and unsmooth trajectories of the robot during the long-term execution. An alternative method is to train a DRL policy that outputs the navigation path directly. However, two roadblocks arise for training a DRL policy that outputs paths: (1) The action space for potential paths often involves higher dimensions comparing to low-level commands, which increases the difficulties of training; (2) It takes multiple time steps to track a path instead of a single time step, which requires the path to predicate the interactions of the robot w.r.t. the dynamic environment in multiple time steps. This, in turn, amplifies the challenges associated with training. In response to these challenges, we propose PathRL, a novel DRL method that trains the policy to generate the navigation path for the robot. Specifically, we employ specific action space discretization techniques and tailored state space representation methods to address the associated challenges. In our experiments, PathRL achieves better success rates and reduces angular rotation variability compared to other DRL navigation methods, facilitating stable and smooth robot movement. We demonstrate the competitive edge of PathRL in both real-world scenarios and multiple challenging simulation environments.

{{</citation>}}


### (99/136) Dynamic Object Detection in Range data using Spatiotemporal Normals (Raphael Falque et al., 2023)

{{<citation>}}

Raphael Falque, Cedric Le Gentil, Fouad Sukkar. (2023)  
**Dynamic Object Detection in Range data using Spatiotemporal Normals**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.13273v1)  

---


**ABSTRACT**  
On the journey to enable robots to interact with the real world where humans, animals, and unpredictable elements are acting as independent agents; it is crucial for robots to have the capability to detect dynamic objects. In this paper, we argue that the detection of dynamic objects can be solved by computing the spatiotemporal normals of a point cloud. In our experiments, we demonstrate that this simple method can be used robustly for LiDAR and depth cameras with performances similar to the state of the art while offering a significantly simpler method.

{{</citation>}}


## cs.CV (18)



### (100/136) Data-Free Knowledge Distillation Using Adversarially Perturbed OpenGL Shader Images (Logan Frank et al., 2023)

{{<citation>}}

Logan Frank, Jim Davis. (2023)  
**Data-Free Knowledge Distillation Using Adversarially Perturbed OpenGL Shader Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.13782v1)  

---


**ABSTRACT**  
Knowledge distillation (KD) has been a popular and effective method for model compression. One important assumption of KD is that the original training dataset is always available. However, this is not always the case due to privacy concerns and more. In recent years, "data-free" KD has emerged as a growing research topic which focuses on the scenario of performing KD when no data is provided. Many methods rely on a generator network to synthesize examples for distillation (which can be difficult to train) and can frequently produce images that are visually similar to the original dataset, which raises questions surrounding whether privacy is completely preserved. In this work, we propose a new approach to data-free KD that utilizes unnatural OpenGL images, combined with large amounts of data augmentation and adversarial attacks, to train a student network. We demonstrate that our approach achieves state-of-the-art results for a variety of datasets/networks and is more stable than existing generator-based data-free KD methods. Source code will be available in the future.

{{</citation>}}


### (101/136) TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models (Tianshi Cao et al., 2023)

{{<citation>}}

Tianshi Cao, Karsten Kreis, Sanja Fidler, Nicholas Sharp, Kangxue Yin. (2023)  
**TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models**  

---
Primary Category: cs.CV  
Categories: I-3-3, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13772v1)  

---


**ABSTRACT**  
We present TexFusion (Texture Diffusion), a new method to synthesize textures for given 3D geometries, using large-scale text-guided image diffusion models. In contrast to recent works that leverage 2D text-to-image diffusion models to distill 3D objects using a slow and fragile optimization process, TexFusion introduces a new 3D-consistent generation technique specifically designed for texture synthesis that employs regular diffusion model sampling on different 2D rendered views. Specifically, we leverage latent diffusion models, apply the diffusion model's denoiser on a set of 2D renders of the 3D object, and aggregate the different denoising predictions on a shared latent texture map. Final output RGB textures are produced by optimizing an intermediate neural color field on the decodings of 2D renders of the latent texture. We thoroughly validate TexFusion and show that we can efficiently generate diverse, high quality and globally coherent textures. We achieve state-of-the-art text-guided texture synthesis performance using only image diffusion models, while avoiding the pitfalls of previous distillation-based methods. The text-conditioning offers detailed control and we also do not rely on any ground truth 3D textures for training. This makes our method versatile and applicable to a broad range of geometry and texture types. We hope that TexFusion will advance AI-based texturing of 3D assets for applications in virtual reality, game design, simulation, and more.

{{</citation>}}


### (102/136) Using Human-like Mechanism to Weaken Effect of Pre-training Weight Bias in Face-Recognition Convolutional Neural Network (Haojiang Ying et al., 2023)

{{<citation>}}

Haojiang Ying, Yi-Fan Li, Yiyang Chen. (2023)  
**Using Human-like Mechanism to Weaken Effect of Pre-training Weight Bias in Face-Recognition Convolutional Neural Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, q-bio-NC  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.13674v1)  

---


**ABSTRACT**  
Convolutional neural network (CNN), as an important model in artificial intelligence, has been widely used and studied in different disciplines. The computational mechanisms of CNNs are still not fully revealed due to the their complex nature. In this study, we focused on 4 extensively studied CNNs (AlexNet, VGG11, VGG13, and VGG16) which has been analyzed as human-like models by neuroscientists with ample evidence. We trained these CNNs to emotion valence classification task by transfer learning. Comparing their performance with human data, the data unveiled that these CNNs would partly perform as human does. We then update the object-based AlexNet using self-attention mechanism based on neuroscience and behavioral data. The updated FE-AlexNet outperformed all the other tested CNNs and closely resembles human perception. The results further unveil the computational mechanisms of these CNNs. Moreover, this study offers a new paradigm to better understand and improve CNN performance via human data.

{{</citation>}}


### (103/136) ARNIQA: Learning Distortion Manifold for Image Quality Assessment (Lorenzo Agnolucci et al., 2023)

{{<citation>}}

Lorenzo Agnolucci, Leonardo Galteri, Marco Bertini, Alberto Del Bimbo. (2023)  
**ARNIQA: Learning Distortion Manifold for Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.14918v1)  

---


**ABSTRACT**  
No-Reference Image Quality Assessment (NR-IQA) aims to develop methods to measure image quality in alignment with human perception without the need for a high-quality reference image. In this work, we propose a self-supervised approach named ARNIQA (leArning distoRtion maNifold for Image Quality Assessment) for modeling the image distortion manifold to obtain quality representations in an intrinsic manner. First, we introduce an image degradation model that randomly composes ordered sequences of consecutively applied distortions. In this way, we can synthetically degrade images with a large variety of degradation patterns. Second, we propose to train our model by maximizing the similarity between the representations of patches of different images distorted equally, despite varying content. Therefore, images degraded in the same manner correspond to neighboring positions within the distortion manifold. Finally, we map the image representations to the quality scores with a simple linear regressor, thus without fine-tuning the encoder weights. The experiments show that our approach achieves state-of-the-art performance on several datasets. In addition, ARNIQA demonstrates improved data efficiency, generalization capabilities, and robustness compared to competing methods. The code and the model are publicly available at https://github.com/miccunifi/ARNIQA.

{{</citation>}}


### (104/136) FMRT: Learning Accurate Feature Matching with Reconciliatory Transformer (Xinyu Zhang et al., 2023)

{{<citation>}}

Xinyu Zhang, Li Wang, Zhiqiang Jiang, Kun Dai, Tao Xie, Lei Yang, Wenhao Yu, Yang Shen, Jun Li. (2023)  
**FMRT: Learning Accurate Feature Matching with Reconciliatory Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.13605v1)  

---


**ABSTRACT**  
Local Feature Matching, an essential component of several computer vision tasks (e.g., structure from motion and visual localization), has been effectively settled by Transformer-based methods. However, these methods only integrate long-range context information among keypoints with a fixed receptive field, which constrains the network from reconciling the importance of features with different receptive fields to realize complete image perception, hence limiting the matching accuracy. In addition, these methods utilize a conventional handcrafted encoding approach to integrate the positional information of keypoints into the visual descriptors, which limits the capability of the network to extract reliable positional encoding message. In this study, we propose Feature Matching with Reconciliatory Transformer (FMRT), a novel Transformer-based detector-free method that reconciles different features with multiple receptive fields adaptively and utilizes parallel networks to realize reliable positional encoding. Specifically, FMRT proposes a dedicated Reconciliatory Transformer (RecFormer) that consists of a Global Perception Attention Layer (GPAL) to extract visual descriptors with different receptive fields and integrate global context information under various scales, Perception Weight Layer (PWL) to measure the importance of various receptive fields adaptively, and Local Perception Feed-forward Network (LPFFN) to extract deep aggregated multi-scale local feature representation. Extensive experiments demonstrate that FMRT yields extraordinary performance on multiple benchmarks, including pose estimation, visual localization, homography estimation, and image matching.

{{</citation>}}


### (105/136) Longer-range Contextualized Masked Autoencoder (Taekyung Kim et al., 2023)

{{<citation>}}

Taekyung Kim, Sanghyuk Chun, Byeongho Heo, Dongyoon Han. (2023)  
**Longer-range Contextualized Masked Autoencoder**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.13593v1)  

---


**ABSTRACT**  
Masked image modeling (MIM) has emerged as a promising self-supervised learning (SSL) strategy. The MIM pre-training facilitates learning powerful representations using an encoder-decoder framework by randomly masking some input pixels and reconstructing the masked pixels from the remaining ones. However, as the encoder is trained with partial pixels, the MIM pre-training can suffer from a low capability of understanding long-range dependency. This limitation may hinder its capability to fully understand multiple-range dependencies, resulting in narrow highlighted regions in the attention map that may incur accuracy drops. To mitigate the limitation, We propose a self-supervised learning framework, named Longer-range Contextualized Masked Autoencoder (LC-MAE). LC-MAE effectively leverages a global context understanding of visual representations while simultaneously reducing the spatial redundancy of input at the same time. Our method steers the encoder to learn from entire pixels in multiple views while also learning local representation from sparse pixels. As a result, LC-MAE learns more discriminative representations, leading to a performance improvement of achieving 84.2% top-1 accuracy with ViT-B on ImageNet-1K with 0.6%p gain. We attribute the success to the enhanced pre-training method, as evidenced by the singular value spectrum and attention analyses. Finally, LC-MAE achieves significant performance gains at the downstream semantic segmentation and fine-grained visual classification tasks; and on diverse robust evaluation metrics. Our code will be publicly available.

{{</citation>}}


### (106/136) POTLoc: Pseudo-Label Oriented Transformer for Point-Supervised Temporal Action Localization (Elahe Vahdani et al., 2023)

{{<citation>}}

Elahe Vahdani, Yingli Tian. (2023)  
**POTLoc: Pseudo-Label Oriented Transformer for Point-Supervised Temporal Action Localization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.13585v1)  

---


**ABSTRACT**  
This paper tackles the challenge of point-supervised temporal action detection, wherein only a single frame is annotated for each action instance in the training set. Most of the current methods, hindered by the sparse nature of annotated points, struggle to effectively represent the continuous structure of actions or the inherent temporal and semantic dependencies within action instances. Consequently, these methods frequently learn merely the most distinctive segments of actions, leading to the creation of incomplete action proposals. This paper proposes POTLoc, a Pseudo-label Oriented Transformer for weakly-supervised Action Localization utilizing only point-level annotation. POTLoc is designed to identify and track continuous action structures via a self-training strategy. The base model begins by generating action proposals solely with point-level supervision. These proposals undergo refinement and regression to enhance the precision of the estimated action boundaries, which subsequently results in the production of `pseudo-labels' to serve as supplementary supervisory signals. The architecture of the model integrates a transformer with a temporal feature pyramid to capture video snippet dependencies and model actions of varying duration. The pseudo-labels, providing information about the coarse locations and boundaries of actions, assist in guiding the transformer for enhanced learning of action dynamics. POTLoc outperforms the state-of-the-art point-supervised methods on THUMOS'14 and ActivityNet-v1.2 datasets, showing a significant improvement of 5% average mAP on the former.

{{</citation>}}


### (107/136) A Simple Baseline for Knowledge-Based Visual Question Answering (Alexandros Xenos et al., 2023)

{{<citation>}}

Alexandros Xenos, Themos Stafylakis, Ioannis Patras, Georgios Tzimiropoulos. (2023)  
**A Simple Baseline for Knowledge-Based Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, LLaMA, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.13570v2)  

---


**ABSTRACT**  
This paper is on the problem of Knowledge-Based Visual Question Answering (KB-VQA). Recent works have emphasized the significance of incorporating both explicit (through external databases) and implicit (through LLMs) knowledge to answer questions requiring external knowledge effectively. A common limitation of such approaches is that they consist of relatively complicated pipelines and often heavily rely on accessing GPT-3 API. Our main contribution in this paper is to propose a much simpler and readily reproducible pipeline which, in a nutshell, is based on efficient in-context learning by prompting LLaMA (1 and 2) using question-informative captions as contextual information. Contrary to recent approaches, our method is training-free, does not require access to external databases or APIs, and yet achieves state-of-the-art accuracy on the OK-VQA and A-OK-VQA datasets. Finally, we perform several ablation studies to understand important aspects of our method. Our code is publicly available at https://github.com/alexandrosXe/ASimple-Baseline-For-Knowledge-Based-VQA

{{</citation>}}


### (108/136) ROSS: Radar Off-road Semantic Segmentation (Peng Jiang et al., 2023)

{{<citation>}}

Peng Jiang, Srikanth Saripalli. (2023)  
**ROSS: Radar Off-road Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.13551v1)  

---


**ABSTRACT**  
As the demand for autonomous navigation in off-road environments increases, the need for effective solutions to understand these surroundings becomes essential. In this study, we confront the inherent complexities of semantic segmentation in RADAR data for off-road scenarios. We present a novel pipeline that utilizes LIDAR data and an existing annotated off-road LIDAR dataset for generating RADAR labels, in which the RADAR data are represented as images. Validated with real-world datasets, our pragmatic approach underscores the potential of RADAR technology for navigation applications in off-road environments.

{{</citation>}}


### (109/136) Technical Report for ICCV 2023 Visual Continual Learning Challenge: Continuous Test-time Adaptation for Semantic Segmentation (Damian Sójka et al., 2023)

{{<citation>}}

Damian Sójka, Yuyang Liu, Dipam Goswami, Sebastian Cygert, Bartłomiej Twardowski, Joost van de Weijer. (2023)  
**Technical Report for ICCV 2023 Visual Continual Learning Challenge: Continuous Test-time Adaptation for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.13533v1)  

---


**ABSTRACT**  
The goal of the challenge is to develop a test-time adaptation (TTA) method, which could adapt the model to gradually changing domains in video sequences for semantic segmentation task. It is based on a synthetic driving video dataset - SHIFT. The source model is trained on images taken during daytime in clear weather. Domain changes at test-time are mainly caused by varying weather conditions and times of day. The TTA methods are evaluated in each image sequence (video) separately, meaning the model is reset to the source model state before the next sequence. Images come one by one and a prediction has to be made at the arrival of each frame. Each sequence is composed of 401 images and starts with the source domain, then gradually drifts to a different one (changing weather or time of day) until the middle of the sequence. In the second half of the sequence, the domain gradually shifts back to the source one. Ground truth data is available only for the validation split of the SHIFT dataset, in which there are only six sequences that start and end with the source domain. We conduct an analysis specifically on those sequences. Ground truth data for test split, on which the developed TTA methods are evaluated for leader board ranking, are not publicly available.   The proposed solution secured a 3rd place in a challenge and received an innovation award. Contrary to the solutions that scored better, we did not use any external pretrained models or specialized data augmentations, to keep the solutions as general as possible. We have focused on analyzing the distributional shift and developing a method that could adapt to changing data dynamics and generalize across different scenarios.

{{</citation>}}


### (110/136) Application of deep learning for livestock behaviour recognition: A systematic literature review (Ali Rohan et al., 2023)

{{<citation>}}

Ali Rohan, Muhammad Saad Rafaq, Md. Junayed Hasan, Furqan Asghar, Ali Kashif Bashir, Tania Dottorini. (2023)  
**Application of deep learning for livestock behaviour recognition: A systematic literature review**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13483v1)  

---


**ABSTRACT**  
Livestock health and welfare monitoring has traditionally been a labor-intensive task performed manually. Recent advances have led to the adoption of AI and computer vision techniques, particularly deep learning models, as decision-making tools within the livestock industry. These models have been employed for tasks like animal identification, tracking, body part recognition, and species classification. In the past decade, there has been a growing interest in using these models to explore the connection between livestock behaviour and health issues. While previous review studies have been rather generic, there is currently no review study specifically focusing on DL for livestock behaviour recognition. Hence, this systematic literature review (SLR) was conducted. The SLR involved an initial search across electronic databases, resulting in 1101 publications. After applying defined selection criteria, 126 publications were shortlisted. These publications were further filtered based on quality criteria, resulting in the selection of 44 high-quality primary studies. These studies were analysed to address the research questions. The results showed that DL successfully addressed 13 behaviour recognition problems encompassing 44 different behaviour classes. A variety of DL models and networks were employed, with CNN, Faster R-CNN, YOLOv5, and YOLOv4 being among the most common models, and VGG16, CSPDarknet53, GoogLeNet, ResNet101, and ResNet50 being popular networks. Performance evaluation involved ten different matrices, with precision and accuracy being the most frequently used. Primary studies identified challenges, including occlusion, adhesion, data imbalance, and the complexities of the livestock environment. The SLR study also discussed potential solutions and research directions to facilitate the development of autonomous livestock behaviour recognition systems.

{{</citation>}}


### (111/136) Benchmarking Sequential Visual Input Reasoning and Prediction in Multimodal Large Language Models (Mingwei Zhu et al., 2023)

{{<citation>}}

Mingwei Zhu, Leigang Sha, Yu Shu, Kangjia Zhao, Tiancheng Zhao, Jianwei Yin. (2023)  
**Benchmarking Sequential Visual Input Reasoning and Prediction in Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.13473v1)  

---


**ABSTRACT**  
Multimodal large language models (MLLMs) have shown great potential in perception and interpretation tasks, but their capabilities in predictive reasoning remain under-explored. To address this gap, we introduce a novel benchmark that assesses the predictive reasoning capabilities of MLLMs across diverse scenarios. Our benchmark targets three important domains: abstract pattern reasoning, human activity prediction, and physical interaction prediction. We further develop three evaluation methods powered by large language model to robustly quantify a model's performance in predicting and reasoning the future based on multi-visual context. Empirical experiments confirm the soundness of the proposed benchmark and evaluation methods via rigorous testing and reveal pros and cons of current popular MLLMs in the task of predictive reasoning. Lastly, our proposed benchmark provides a standardized evaluation framework for MLLMs and can facilitate the development of more advanced models that can reason and predict over complex long sequence of multimodal input.

{{</citation>}}


### (112/136) Dance Your Latents: Consistent Dance Generation through Spatial-temporal Subspace Attention Guided by Motion Flow (Haipeng Fang et al., 2023)

{{<citation>}}

Haipeng Fang, Zhihao Sun, Ziyao Huang, Fan Tang, Juan Cao, Sheng Tang. (2023)  
**Dance Your Latents: Consistent Dance Generation through Spatial-temporal Subspace Attention Guided by Motion Flow**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2310.14780v1)  

---


**ABSTRACT**  
The advancement of generative AI has extended to the realm of Human Dance Generation, demonstrating superior generative capacities. However, current methods still exhibit deficiencies in achieving spatiotemporal consistency, resulting in artifacts like ghosting, flickering, and incoherent motions. In this paper, we present Dance-Your-Latents, a framework that makes latents dance coherently following motion flow to generate consistent dance videos. Firstly, considering that each constituent element moves within a confined space, we introduce spatial-temporal subspace-attention blocks that decompose the global space into a combination of regular subspaces and efficiently model the spatiotemporal consistency within these subspaces. This module enables each patch pay attention to adjacent areas, mitigating the excessive dispersion of long-range attention. Furthermore, observing that body part's movement is guided by pose control, we design motion flow guided subspace align & restore. This method enables the attention to be computed on the irregular subspace along the motion flow. Experimental results in TikTok dataset demonstrate that our approach significantly enhances spatiotemporal consistency of the generated videos.

{{</citation>}}


### (113/136) Multiscale Superpixel Structured Difference Graph Convolutional Network for VL Representation (Siyu Zhang et al., 2023)

{{<citation>}}

Siyu Zhang, Yeming Chen, Sirui Cheng, Yaoru Sun, Jun Yang, Lizhi Bai. (2023)  
**Multiscale Superpixel Structured Difference Graph Convolutional Network for VL Representation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2310.13447v2)  

---


**ABSTRACT**  
Within the multimodal field, the key to integrating vision and language lies in establishing a good alignment strategy. Recently, benefiting from the success of self-supervised learning, significant progress has been made in multimodal semantic representation based on pre-trained models for vision and language. However, there is still room for improvement in visual semantic representation. The lack of spatial semantic coherence and vulnerability to noise makes it challenging for current pixel or patch-based methods to accurately extract complex scene boundaries. To this end, this paper develops superpixel as a comprehensive compact representation of learnable image data, which effectively reduces the number of visual primitives for subsequent processing by clustering perceptually similar pixels. To mine more precise topological relations, we propose a Multiscale Difference Graph Convolutional Network (MDGCN). It parses the entire image as a fine-to-coarse hierarchical structure of constituent visual patterns, and captures multiscale features by progressively merging adjacent superpixels as graph nodes. Moreover, we predict the differences between adjacent nodes through the graph structure, facilitating key information aggregation of graph nodes to reason actual semantic relations. Afterward, we design a multi-level fusion rule in a bottom-up manner to avoid understanding deviation by learning complementary spatial information at different regional scales. Our proposed method can be well applied to multiple downstream task learning. Extensive experiments demonstrate that our method is competitive with other state-of-the-art methods in visual reasoning. Our code will be released upon publication.

{{</citation>}}


### (114/136) OpenAnnotate3D: Open-Vocabulary Auto-Labeling System for Multi-modal 3D Data (Yijie Zhou et al., 2023)

{{<citation>}}

Yijie Zhou, Likun Cai, Xianhui Cheng, Zhongxue Gan, Xiangyang Xue, Wenchao Ding. (2023)  
**OpenAnnotate3D: Open-Vocabulary Auto-Labeling System for Multi-modal 3D Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13398v1)  

---


**ABSTRACT**  
In the era of big data and large models, automatic annotating functions for multi-modal data are of great significance for real-world AI-driven applications, such as autonomous driving and embodied AI. Unlike traditional closed-set annotation, open-vocabulary annotation is essential to achieve human-level cognition capability. However, there are few open-vocabulary auto-labeling systems for multi-modal 3D data. In this paper, we introduce OpenAnnotate3D, an open-source open-vocabulary auto-labeling system that can automatically generate 2D masks, 3D masks, and 3D bounding box annotations for vision and point cloud data. Our system integrates the chain-of-thought capabilities of Large Language Models (LLMs) and the cross-modality capabilities of vision-language models (VLMs). To the best of our knowledge, OpenAnnotate3D is one of the pioneering works for open-vocabulary multi-modal 3D auto-labeling. We conduct comprehensive evaluations on both public and in-house real-world datasets, which demonstrate that the system significantly improves annotation efficiency compared to manual annotation while providing accurate open-vocabulary auto-annotating results.

{{</citation>}}


### (115/136) Bridging the Gap between Synthetic and Authentic Images for Multimodal Machine Translation (Wenyu Guo et al., 2023)

{{<citation>}}

Wenyu Guo, Qingkai Fang, Dong Yu, Yang Feng. (2023)  
**Bridging the Gap between Synthetic and Authentic Images for Multimodal Machine Translation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Machine Translation, Transformer  
[Paper Link](http://arxiv.org/abs/2310.13361v1)  

---


**ABSTRACT**  
Multimodal machine translation (MMT) simultaneously takes the source sentence and a relevant image as input for translation. Since there is no paired image available for the input sentence in most cases, recent studies suggest utilizing powerful text-to-image generation models to provide image inputs. Nevertheless, synthetic images generated by these models often follow different distributions compared to authentic images. Consequently, using authentic images for training and synthetic images for inference can introduce a distribution shift, resulting in performance degradation during inference. To tackle this challenge, in this paper, we feed synthetic and authentic images to the MMT model, respectively. Then we minimize the gap between the synthetic and authentic images by drawing close the input image representations of the Transformer Encoder and the output distributions of the Transformer Decoder. Therefore, we mitigate the distribution disparity introduced by the synthetic images during inference, thereby freeing the authentic images from the inference process.Experimental results show that our approach achieves state-of-the-art performance on the Multi30K En-De and En-Fr datasets, while remaining independent of authentic images during inference.

{{</citation>}}


### (116/136) FLAIR: a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery (Anatol Garioud et al., 2023)

{{<citation>}}

Anatol Garioud, Nicolas Gonthier, Loic Landrieu, Apolline De Wit, Marion Valette, Marc Poupée, Sébastien Giordano, Boris Wattrelos. (2023)  
**FLAIR: a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.13336v1)  

---


**ABSTRACT**  
We introduce the French Land cover from Aerospace ImageRy (FLAIR), an extensive dataset from the French National Institute of Geographical and Forest Information (IGN) that provides a unique and rich resource for large-scale geospatial analysis. FLAIR contains high-resolution aerial imagery with a ground sample distance of 20 cm and over 20 billion individually labeled pixels for precise land-cover classification. The dataset also integrates temporal and spectral data from optical satellite time series. FLAIR thus combines data with varying spatial, spectral, and temporal resolutions across over 817 km2 of acquisitions representing the full landscape diversity of France. This diversity makes FLAIR a valuable resource for the development and evaluation of novel methods for large-scale land-cover semantic segmentation and raises significant challenges in terms of computer vision, data fusion, and geospatial analysis. We also provide powerful uni- and multi-sensor baseline models that can be employed to assess algorithm's performance and for downstream applications. Through its extent and the quality of its annotation, FLAIR aims to spur improvements in monitoring and understanding key anthropogenic development indicators such as urban growth, deforestation, and soil artificialization. Dataset and codes can be accessed at https://ignf.github.io/FLAIR/

{{</citation>}}


### (117/136) Zone Evaluation: Revealing Spatial Bias in Object Detection (Zhaohui Zheng et al., 2023)

{{<citation>}}

Zhaohui Zheng, Yuming Chen, Qibin Hou, Xiang Li, Ping Wang, Ming-Ming Cheng. (2023)  
**Zone Evaluation: Revealing Spatial Bias in Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.13215v1)  

---


**ABSTRACT**  
A fundamental limitation of object detectors is that they suffer from "spatial bias", and in particular perform less satisfactorily when detecting objects near image borders. For a long time, there has been a lack of effective ways to measure and identify spatial bias, and little is known about where it comes from and what degree it is. To this end, we present a new zone evaluation protocol, extending from the traditional evaluation to a more generalized one, which measures the detection performance over zones, yielding a series of Zone Precisions (ZPs). For the first time, we provide numerical results, showing that the object detectors perform quite unevenly across the zones. Surprisingly, the detector's performance in the 96\% border zone of the image does not reach the AP value (Average Precision, commonly regarded as the average detection performance in the entire image zone). To better understand spatial bias, a series of heuristic experiments are conducted. Our investigation excludes two intuitive conjectures about spatial bias that the object scale and the absolute positions of objects barely influence the spatial bias. We find that the key lies in the human-imperceptible divergence in data patterns between objects in different zones, thus eventually forming a visible performance gap between the zones. With these findings, we finally discuss a future direction for object detection, namely, spatial disequilibrium problem, aiming at pursuing a balanced detection ability over the entire image zone. By broadly evaluating 10 popular object detectors and 5 detection datasets, we shed light on the spatial bias of object detectors. We hope this work could raise a focus on detection robustness. The source codes, evaluation protocols, and tutorials are publicly available at \url{https://github.com/Zzh-tju/ZoneEval}.

{{</citation>}}


## cs.NI (1)



### (118/136) EXPLORA: AI/ML EXPLainability for the Open RAN (Claudio Fiandrino et al., 2023)

{{<citation>}}

Claudio Fiandrino, Leonardo Bonati, Salvatore D'Oro, Michele Polese, Tommaso Melodia, Joerg Widmer. (2023)  
**EXPLORA: AI/ML EXPLainability for the Open RAN**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13667v1)  

---


**ABSTRACT**  
The Open Radio Access Network (RAN) paradigm is transforming cellular networks into a system of disaggregated, virtualized, and software-based components. These self-optimize the network through programmable, closed-loop control, leveraging Artificial Intelligence (AI) and Machine Learning (ML) routines. In this context, Deep Reinforcement Learning (DRL) has shown great potential in addressing complex resource allocation problems. However, DRL -based solutions are inherently hard to explain, which hinders their deployment and use in practice. In this paper, we propose EXPLORA, a framework that provides explainability of DRL-based control solutions for the Open RAN ecosystem. EXPLORA synthesizes network-oriented explanations based on an attributed graph that produces a link between the actions taken by a DRL agent (i.e., the nodes of the graph) and the input state space (i.e., the attributes of each node). This novel approach allows EXPLORA to explain models by providing information on the wireless context in which the DRL agent operates. EXPLORA is also designed to be lightweight for real-time operation. We prototype EXPLORA and test it experimentally on an O-RAN-compliant near-real-time RIC deployed on the Colosseum wireless network emulator. We evaluate EXPLORA for agents trained for different purposes and showcase how it generates clear network-oriented explanations. We also show how explanations can be used to perform informative and targeted intent-based action steering and achieve median transmission bitrate improvements of 4% and tail improvements of 10%.

{{</citation>}}


## cs.SE (2)



### (119/136) Using ChatGPT throughout the Software Development Life Cycle by Novice Developers (Muhammad Waseem et al., 2023)

{{<citation>}}

Muhammad Waseem, Teerath Das, Aakash Ahmad, Mahdi Fehmideh, Peng Liang, Tommi Mikkonen. (2023)  
**Using ChatGPT throughout the Software Development Life Cycle by Novice Developers**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.13648v1)  

---


**ABSTRACT**  
This study investigates the impact of ChatGPT -- a generative AI-based tool -- on undergraduate students' software development experiences. Through a three-month project involving seven undergraduate students, ChatGPT was employed as a supporting tool, and their experiences were systematically surveyed before and after the projects. The research aims to answer four key questions related to ChatGPT's effectiveness, advantages, limitations, impact on learning, and challenges faced. The findings revealed significant skill gaps among undergraduate students, underscoring the importance of addressing educational deficiencies in software development. ChatGPT was found to have a positive influence on various phases of the software development life cycle, leading to enhanced efficiency, accuracy, and collaboration. ChatGPT also consistently improved participants' foundational understanding and soft skills in software development. These findings underscore the significance of integrating AI tools like ChatGPT into undergraduate students education, particularly to bridge skill gaps and enhance productivity. However, a nuanced approach to technology reliance is essential, acknowledging the variability in opinions and the need for customization. Future research should explore strategies to optimize ChatGPT's application across development contexts, ensuring it maximizes learning while addressing specific challenges.

{{</citation>}}


### (120/136) The GitHub Recent Bugs Dataset for Evaluating LLM-based Debugging Applications (Jae Yong Lee et al., 2023)

{{<citation>}}

Jae Yong Lee, Sungmin Kang, Juyeon Yoon, Shin Yoo. (2023)  
**The GitHub Recent Bugs Dataset for Evaluating LLM-based Debugging Applications**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13229v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated strong natural language processing and code synthesis capabilities, which has led to their rapid adoption in software engineering applications. However, details about LLM training data are often not made public, which has caused concern as to whether existing bug benchmarks are included. In lieu of the training data for the popular GPT models, we examine the training data of the open-source LLM StarCoder, and find it likely that data from the widely used Defects4J benchmark was included, raising the possibility of its inclusion in GPT training data as well. This makes it difficult to tell how well LLM-based results on Defects4J would generalize, as for any results it would be unclear whether a technique's performance is due to LLM generalization or memorization. To remedy this issue and facilitate continued research on LLM-based SE, we present the GitHub Recent Bugs (GHRB) dataset, which includes 76 real-world Java bugs that were gathered after the OpenAI data cut-off point.

{{</citation>}}


## eess.IV (2)



### (121/136) Inter-Scale Dependency Modeling for Skin Lesion Segmentation with Transformer-based Networks (Sania Eskandari et al., 2023)

{{<citation>}}

Sania Eskandari, Janet Lumpp. (2023)  
**Inter-Scale Dependency Modeling for Skin Lesion Segmentation with Transformer-based Networks**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.13727v1)  

---


**ABSTRACT**  
Melanoma is a dangerous form of skin cancer caused by the abnormal growth of skin cells. Fully Convolutional Network (FCN) approaches, including the U-Net architecture, can automatically segment skin lesions to aid diagnosis. The symmetrical U-Net model has shown outstanding results, but its use of a convolutional operation limits its ability to capture long-range dependencies, which are essential for accurate medical image segmentation. In addition, the U-shaped structure suffers from the semantic gaps between the encoder and decoder. In this study, we developed and evaluated a U-shaped hierarchical Transformer-based structure for skin lesion segmentation while we proposed an Inter-scale Context Fusion (ISCF) to utilize the attention correlations in each stage of the encoder to adaptively combine the contexts coming from each stage to hinder the semantic gaps. The preliminary results of the skin lesion segmentation benchmark endorse the applicability and efficacy of the ISCF module.

{{</citation>}}


### (122/136) Skin Lesion Segmentation Improved by Transformer-based Networks with Inter-scale Dependency Modeling (Sania Eskandari et al., 2023)

{{<citation>}}

Sania Eskandari, Janet Lumpp, Luis Sanchez Giraldo. (2023)  
**Skin Lesion Segmentation Improved by Transformer-based Networks with Inter-scale Dependency Modeling**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.13604v1)  

---


**ABSTRACT**  
Melanoma, a dangerous type of skin cancer resulting from abnormal skin cell growth, can be treated if detected early. Various approaches using Fully Convolutional Networks (FCNs) have been proposed, with the U-Net architecture being prominent To aid in its diagnosis through automatic skin lesion segmentation. However, the symmetrical U-Net model's reliance on convolutional operations hinders its ability to capture long-range dependencies crucial for accurate medical image segmentation. Several Transformer-based U-Net topologies have recently been created to overcome this limitation by replacing CNN blocks with different Transformer modules to capture local and global representations. Furthermore, the U-shaped structure is hampered by semantic gaps between the encoder and decoder. This study intends to increase the network's feature re-usability by carefully building the skip connection path. Integrating an already calculated attention affinity within the skip connection path improves the typical concatenation process utilized in the conventional skip connection path. As a result, we propose a U-shaped hierarchical Transformer-based structure for skin lesion segmentation and an Inter-scale Context Fusion (ISCF) method that uses attention correlations in each stage of the encoder to adaptively combine the contexts from each stage to mitigate semantic gaps. The findings from two skin lesion segmentation benchmarks support the ISCF module's applicability and effectiveness. The code is publicly available at \url{https://github.com/saniaesk/skin-lesion-segmentation}

{{</citation>}}


## cs.CY (2)



### (123/136) Oversight for Frontier AI through a Know-Your-Customer Scheme for Compute Providers (Janet Egan et al., 2023)

{{<citation>}}

Janet Egan, Lennart Heim. (2023)  
**Oversight for Frontier AI through a Know-Your-Customer Scheme for Compute Providers**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13625v1)  

---


**ABSTRACT**  
To address security and safety risks stemming from highly capable artificial intelligence (AI) models, we propose that the US government should ensure compute providers implement Know-Your-Customer (KYC) schemes. Compute - the computational power and infrastructure required to train and run these AI models - is emerging as a node for oversight. KYC, a standard developed by the banking sector to identify and verify client identity, could provide a mechanism for greater public oversight of frontier AI development and close loopholes in existing export controls. Such a scheme has the potential to identify and warn stakeholders of potentially problematic and/or sudden advancements in AI capabilities, build government capacity for AI regulation, and allow for the development and implementation of more nuanced and targeted export controls. Unlike the strategy of limiting access to AI chip purchases, regulating the digital access to compute offers more precise controls, allowing regulatory control over compute quantities, as well as the flexibility to suspend access at any time. To enact a KYC scheme, the US government will need to work closely with industry to (1) establish a dynamic threshold of compute that effectively captures high-risk frontier model development, while minimizing imposition on developers not engaged in frontier AI; (2) set requirements and guidance for compute providers to keep records and report high-risk entities; (3) establish government capacity that allows for co-design, implementation, administration and enforcement of the scheme; and (4) engage internationally to promote international alignment with the scheme and support its long-term efficacy. While the scheme will not address all AI risks, it complements proposed solutions by allowing for a more precise and flexible approach to controlling the development of frontier AI models and unwanted AI proliferation.

{{</citation>}}


### (124/136) Entangled Preferences: The History and Risks of Reinforcement Learning and Human Feedback (Nathan Lambert et al., 2023)

{{<citation>}}

Nathan Lambert, Thomas Krendl Gilbert, Tom Zick. (2023)  
**Entangled Preferences: The History and Risks of Reinforcement Learning and Human Feedback**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13595v1)  

---


**ABSTRACT**  
Reinforcement learning from human feedback (RLHF) has emerged as a powerful technique to make large language models (LLMs) easier to use and more effective. A core piece of the RLHF process is the training and utilization of a model of human preferences that acts as a reward function for optimization. This approach, which operates at the intersection of many stakeholders and academic disciplines, remains poorly understood. RLHF reward models are often cited as being central to achieving performance, yet very few descriptors of capabilities, evaluations, training methods, or open-source models exist. Given this lack of information, further study and transparency is needed for learned RLHF reward models. In this paper, we illustrate the complex history of optimizing preferences, and articulate lines of inquiry to understand the sociotechnical context of reward models. In particular, we highlight the ontological differences between costs, rewards, and preferences at stake in RLHF's foundations, related methodological tensions, and possible research directions to improve general understanding of how reward models function.

{{</citation>}}


## cs.DB (1)



### (125/136) SPARE: A Single-Pass Neural Model for Relational Databases (Benjamin Hilprecht et al., 2023)

{{<citation>}}

Benjamin Hilprecht, Kristian Kersting, Carsten Binnig. (2023)  
**SPARE: A Single-Pass Neural Model for Relational Databases**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-DB, cs.DB  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.13581v1)  

---


**ABSTRACT**  
While there has been extensive work on deep neural networks for images and text, deep learning for relational databases (RDBs) is still a rather unexplored field.   One direction that recently gained traction is to apply Graph Neural Networks (GNNs) to RBDs. However, training GNNs on large relational databases (i.e., data stored in multiple database tables) is rather inefficient due to multiple rounds of training and potentially large and inefficient representations. Hence, in this paper we propose SPARE (Single-Pass Relational models), a new class of neural models that can be trained efficiently on RDBs while providing similar accuracies as GNNs. For enabling efficient training, different from GNNs, SPARE makes use of the fact that data in RDBs has a regular structure, which allows one to train these models in a single pass while exploiting symmetries at the same time. Our extensive empirical evaluation demonstrates that SPARE can significantly speedup both training and inference while offering competitive predictive performance over numerous baselines.

{{</citation>}}


## eess.SY (2)



### (126/136) Cooperative Multi-Agent Deep Reinforcement Learning for Adaptive Decentralized Emergency Voltage Control (Ying Zhang et al., 2023)

{{<citation>}}

Ying Zhang, Meng Yue. (2023)  
**Cooperative Multi-Agent Deep Reinforcement Learning for Adaptive Decentralized Emergency Voltage Control**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13577v1)  

---


**ABSTRACT**  
Under voltage load shedding (UVLS) for power grid emergency control builds the last defensive perimeter to prevent cascade outages and blackouts in case of contingencies. This letter proposes a novel cooperative multi-agent deep reinforcement learning (MADRL)-based UVLS algorithm in an adaptive decentralized way. With well-designed input signals reflecting the voltage deviation, newly structured neural networks are developed as intelligent agents to obtain control actions and their probabilities to accommodate high uncertainties in volatile power system operations. Moreover, the interaction among the agents for coordinated control is implemented and refined by a state-of-the-art attention mechanism, which helps agents concentratively learn effective interacted information. The proposed method realizes decentralized coordinated control, adapting to extremely high uncertainties. Case studies on an IEEE benchmark system indicate the superior performance of the proposed algorithm.

{{</citation>}}


### (127/136) Deep Reinforcement Learning-Enabled Adaptive Forecasting-Aided State Estimation in Distribution Systems with Multi-Source Multi-Rate Data (Ying Zhang et al., 2023)

{{<citation>}}

Ying Zhang, Junbo Zhao, Di Shi, Sungjoo Chung. (2023)  
**Deep Reinforcement Learning-Enabled Adaptive Forecasting-Aided State Estimation in Distribution Systems with Multi-Source Multi-Rate Data**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13218v1)  

---


**ABSTRACT**  
Distribution system state estimation (DSSE) is paramount for effective state monitoring and control. However, stochastic outputs of renewables and asynchronous streaming of multi-rate measurements in practical systems largely degrade the estimation performance. This paper proposes a deep reinforcement learning (DRL)-enabled adaptive DSSE algorithm in unbalanced distribution systems, which tackles hybrid measurements with different time scales efficiently. We construct a three-step forecasting-aided state estimation framework, including DRL-based parameter identification, prediction, and state estimation, with multi-rate measurements incorporating limited synchrophasor data. Furthermore, a DRL-based adaptive parameter identification mechanism is embedded in the prediction step. As a novel attempt at utilizing DRL to enable DSSE adaptive to varying operating conditions, this method improves the prediction performance and further facilitates accurate state estimation. Case studies in two unbalanced feeders indicate that our method captures state variation with multi-source multi-rate data efficiently, outperforming the traditional methods.

{{</citation>}}


## cs.SD (3)



### (128/136) Two-Stage Triplet Loss Training with Curriculum Augmentation for Audio-Visual Retrieval (Donghuo Zeng et al., 2023)

{{<citation>}}

Donghuo Zeng, Kazushi Ikeda. (2023)  
**Two-Stage Triplet Loss Training with Curriculum Augmentation for Audio-Visual Retrieval**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-IR, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.13451v1)  

---


**ABSTRACT**  
The cross-modal retrieval model leverages the potential of triple loss optimization to learn robust embedding spaces. However, existing methods often train these models in a singular pass, overlooking the distinction between semi-hard and hard triples in the optimization process. The oversight of not distinguishing between semi-hard and hard triples leads to suboptimal model performance. In this paper, we introduce a novel approach rooted in curriculum learning to address this problem. We propose a two-stage training paradigm that guides the model's learning process from semi-hard to hard triplets. In the first stage, the model is trained with a set of semi-hard triplets, starting from a low-loss base. Subsequently, in the second stage, we augment the embeddings using an interpolation technique. This process identifies potential hard negatives, alleviating issues arising from high-loss functions due to a scarcity of hard triples. Our approach then applies hard triplet mining in the augmented embedding space to further optimize the model. Extensive experimental results conducted on two audio-visual datasets show a significant improvement of approximately 9.8% in terms of average Mean Average Precision (MAP) over the current state-of-the-art method, MSNSCA, for the Audio-Visual Cross-Modal Retrieval (AV-CMR) task on the AVE dataset, indicating the effectiveness of our proposed method.

{{</citation>}}


### (129/136) Music Augmentation and Denoising For Peak-Based Audio Fingerprinting (Kamil Akesbi et al., 2023)

{{<citation>}}

Kamil Akesbi, Dorian Desblancs, Benjamin Martin. (2023)  
**Music Augmentation and Denoising For Peak-Based Audio Fingerprinting**  

---
Primary Category: cs.SD  
Categories: cs-IR, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.13388v1)  

---


**ABSTRACT**  
Audio fingerprinting is a well-established solution for song identification from short recording excerpts. Popular methods rely on the extraction of sparse representations, generally spectral peaks, and have proven to be accurate, fast, and scalable to large collections. However, real-world applications of audio identification often happen in noisy environments, which can cause these systems to fail. In this work, we tackle this problem by introducing and releasing a new audio augmentation pipeline that adds noise to music snippets in a realistic way, by stochastically mimicking real-world scenarios. We then propose and release a deep learning model that removes noisy components from spectrograms in order to improve peak-based fingerprinting systems' accuracy. We show that the addition of our model improves the identification performance of commonly used audio fingerprinting systems, even under noisy conditions.

{{</citation>}}


### (130/136) SALMONN: Towards Generic Hearing Abilities for Large Language Models (Changli Tang et al., 2023)

{{<citation>}}

Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Chao Zhang. (2023)  
**SALMONN: Towards Generic Hearing Abilities for Large Language Models**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13289v1)  

---


**ABSTRACT**  
Hearing is arguably an essential ability of artificial intelligence (AI) agents in the physical world, which refers to the perception and understanding of general auditory information consisting of at least three types of sounds: speech, audio events, and music. In this paper, we propose SALMONN, a speech audio language music open neural network, built by integrating a pre-trained text-based large language model (LLM) with speech and audio encoders into a single multimodal model. SALMONN enables the LLM to directly process and understand general audio inputs and achieve competitive performances on a number of speech and audio tasks used in training, such as automatic speech recognition and translation, auditory-information-based question answering, emotion recognition, speaker verification, and music and audio captioning \textit{etc.} SALMONN also has a diverse set of emergent abilities unseen in the training, which includes but is not limited to speech translation to untrained languages, speech-based slot filling, spoken-query-based question answering, audio-based storytelling, and speech audio co-reasoning \textit{etc}. The presence of the cross-modal emergent abilities is studied, and a novel few-shot activation tuning approach is proposed to activate such abilities of SALMONN. To our knowledge, SALMONN is the first model of its type and can be regarded as a step towards AI with generic hearing abilities. An interactive demo of SALMONN is available at \texttt{\url{https://github.com/bytedance/SALMONN}}, and the training code and model checkpoints will be released upon acceptance.

{{</citation>}}


## cs.CR (1)



### (131/136) An LLM can Fool Itself: A Prompt-Based Adversarial Attack (Xilie Xu et al., 2023)

{{<citation>}}

Xilie Xu, Keyi Kong, Ning Liu, Lizhen Cui, Di Wang, Jingfeng Zhang, Mohan Kankanhalli. (2023)  
**An LLM can Fool Itself: A Prompt-Based Adversarial Attack**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Adversarial Attack, GLUE, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2310.13345v1)  

---


**ABSTRACT**  
The wide-ranging applications of large language models (LLMs), especially in safety-critical domains, necessitate the proper evaluation of the LLM's adversarial robustness. This paper proposes an efficient tool to audit the LLM's adversarial robustness via a prompt-based adversarial attack (PromptAttack). PromptAttack converts adversarial textual attacks into an attack prompt that can cause the victim LLM to output the adversarial sample to fool itself. The attack prompt is composed of three important components: (1) original input (OI) including the original sample and its ground-truth label, (2) attack objective (AO) illustrating a task description of generating a new sample that can fool itself without changing the semantic meaning, and (3) attack guidance (AG) containing the perturbation instructions to guide the LLM on how to complete the task by perturbing the original sample at character, word, and sentence levels, respectively. Besides, we use a fidelity filter to ensure that PromptAttack maintains the original semantic meanings of the adversarial examples. Further, we enhance the attack power of PromptAttack by ensembling adversarial examples at different perturbation levels. Comprehensive empirical results using Llama2 and GPT-3.5 validate that PromptAttack consistently yields a much higher attack success rate compared to AdvGLUE and AdvGLUE++. Interesting findings include that a simple emoji can easily mislead GPT-3.5 to make wrong predictions.

{{</citation>}}


## cs.GR (1)



### (132/136) Auxiliary Features-Guided Super Resolution for Monte Carlo Rendering (Qiqi Hou et al., 2023)

{{<citation>}}

Qiqi Hou, Feng Liu. (2023)  
**Auxiliary Features-Guided Super Resolution for Monte Carlo Rendering**  

---
Primary Category: cs.GR  
Categories: cs-CV, cs-GR, cs.GR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.13235v1)  

---


**ABSTRACT**  
This paper investigates super resolution to reduce the number of pixels to render and thus speed up Monte Carlo rendering algorithms. While great progress has been made to super resolution technologies, it is essentially an ill-posed problem and cannot recover high-frequency details in renderings. To address this problem, we exploit high-resolution auxiliary features to guide super resolution of low-resolution renderings. These high-resolution auxiliary features can be quickly rendered by a rendering engine and at the same time provide valuable high-frequency details to assist super resolution. To this end, we develop a cross-modality Transformer network that consists of an auxiliary feature branch and a low-resolution rendering branch. These two branches are designed to fuse high-resolution auxiliary features with the corresponding low-resolution rendering. Furthermore, we design residual densely-connected Swin Transformer groups to learn to extract representative features to enable high-quality super-resolution. Our experiments show that our auxiliary features-guided super-resolution method outperforms both super-resolution methods and Monte Carlo denoising methods in producing high-quality renderings.

{{</citation>}}


## hep-lat (1)



### (133/136) Equivariant Transformer is all you need (Akio Tomiya et al., 2023)

{{<citation>}}

Akio Tomiya, Yuki Nagai. (2023)  
**Equivariant Transformer is all you need**  

---
Primary Category: hep-lat  
Categories: cond-mat-dis-nn, cs-LG, hep-lat, hep-lat  
Keywords: Attention, GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.13222v1)  

---


**ABSTRACT**  
Machine learning, deep learning, has been accelerating computational physics, which has been used to simulate systems on a lattice. Equivariance is essential to simulate a physical system because it imposes a strong induction bias for the probability distribution described by a machine learning model. This reduces the risk of erroneous extrapolation that deviates from data symmetries and physical laws. However, imposing symmetry on the model sometimes occur a poor acceptance rate in self-learning Monte-Carlo (SLMC). On the other hand, Attention used in Transformers like GPT realizes a large model capacity. We introduce symmetry equivariant attention to SLMC. To evaluate our architecture, we apply it to our proposed new architecture on a spin-fermion model on a two-dimensional lattice. We find that it overcomes poor acceptance rates for linear models and observe the scaling law of the acceptance rate as in the large language models with Transformers.

{{</citation>}}


## cs.SI (1)



### (134/136) HierCas: Hierarchical Temporal Graph Attention Networks for Popularity Prediction in Information Cascades (Zhizhen Zhang et al., 2023)

{{<citation>}}

Zhizhen Zhang, Xiaohui Xie, Yishuo Zhang, Lanshan Zhang, Yong Jiang. (2023)  
**HierCas: Hierarchical Temporal Graph Attention Networks for Popularity Prediction in Information Cascades**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Attention, Graph Attention Network  
[Paper Link](http://arxiv.org/abs/2310.13219v1)  

---


**ABSTRACT**  
Information cascade popularity prediction is critical for many applications, including but not limited to identifying fake news and accurate recommendations. Traditional feature-based methods heavily rely on handcrafted features, which are domain-specific and lack generalizability to new domains. To address this problem, researchers have turned to neural network-based approaches. However, existing methods follow a sampling-based modeling approach, potentially losing continuous dynamic information and structural-temporal dependencies that emerge during the information diffusion process. In this paper, we propose a novel framework called Hierarchical Temporal Graph Attention Networks for cascade popularity prediction (HierCas). Unlike existing methods, HierCas operates on the entire cascade graph by a dynamic graph modeling approach, enabling it to capture the full range of continuous dynamic information and explicitly model the interplay between structural and temporal factors. By leveraging time-aware node embedding, graph attention mechanisms and hierarchical pooling structures, HierCas effectively captures the popularity trend implicit in the complex cascade. Extensive experiments conducted on two real-world datasets in different scenarios demonstrate that our HierCas significantly outperforms the state-of-the-art approaches.

{{</citation>}}


## cs.PF (1)



### (135/136) Facile: Fast, Accurate, and Interpretable Basic-Block Throughput Prediction (Andreas Abel et al., 2023)

{{<citation>}}

Andreas Abel, Shrey Sharma, Jan Reineke. (2023)  
**Facile: Fast, Accurate, and Interpretable Basic-Block Throughput Prediction**  

---
Primary Category: cs.PF  
Categories: cs-PF, cs.PF  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.13212v1)  

---


**ABSTRACT**  
Basic-block throughput models such as uiCA, IACA, GRANITE, Ithemal, llvm-mca, OSACA, or CQA guide optimizing compilers and help performance engineers identify and eliminate bottlenecks. For this purpose, basic-block throughput models should ideally be fast, accurate, and interpretable.   Recent advances have significantly improved accuracy: uiCA, the state-of-the-art model, achieves an error of about 1% relative to measurements across a wide range of microarchitectures. The computational efficiency of throughput models, which is equally important for widespread adoption, especially in compilers, has so far received little attention.   In this paper, we introduce Facile, an analytical throughput model that is fast, accurate, and interpretable. Facile analyzes different potential bottlenecks independently and analytically. Due to its compositional nature, Facile's predictions directly pinpoint the bottlenecks. We evaluate Facile on a wide range of microarchitectures and show that it is almost two orders of magnitude faster than existing models while achieving state-of-the-art accuracy.

{{</citation>}}


## eess.SP (1)



### (136/136) Foundational Techniques for Wireless Communications: Channel Coding, Modulation, and Equalization (Solomon McKiernan, 2023)

{{<citation>}}

Solomon McKiernan. (2023)  
**Foundational Techniques for Wireless Communications: Channel Coding, Modulation, and Equalization**  

---
Primary Category: eess.SP  
Categories: cs-NI, eess-SP, eess.SP  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.13209v1)  

---


**ABSTRACT**  
This paper analyses foundational techniques for improving wireless communication systems, including coding methods, modulation schemes, and channel equalization. Using industry-standard simulation tools, the paper evaluates the performance of these techniques under different channel conditions. Convolutional codes, punctured and unpunctured, are assessed for reliable data transfer. The suitability of various modulation schemes, such as Phase Shift Keying (PSK) and Quadrature Amplitude Modulation (QAM), are examined. Linear and decision-feedback equalization techniques are evaluated for mitigating the effects of channel impairments. The paper provides practical insights into the implementation of these techniques, emphasizing their importance in modern wireless communication systems.

{{</citation>}}
