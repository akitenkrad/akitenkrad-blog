---
draft: false
title: "arXiv @ 2024.01.11"
date: 2024-01-11
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.11"
    identifier: arxiv_20240111
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (24)](#cscl-24)
- [cs.IR (5)](#csir-5)
- [cs.MA (1)](#csma-1)
- [econ.EM (1)](#econem-1)
- [cs.LG (15)](#cslg-15)
- [cs.CR (1)](#cscr-1)
- [cs.HC (3)](#cshc-3)
- [cs.CV (21)](#cscv-21)
- [eess.IV (2)](#eessiv-2)
- [eess.SP (1)](#eesssp-1)
- [math.OC (1)](#mathoc-1)
- [cs.SE (4)](#csse-4)
- [cs.NE (2)](#csne-2)
- [cs.AI (2)](#csai-2)
- [cs.SD (1)](#cssd-1)
- [cs.IT (1)](#csit-1)
- [stat.ML (1)](#statml-1)
- [eess.AS (1)](#eessas-1)
- [q-bio.BM (1)](#q-biobm-1)
- [q-fin.ST (1)](#q-finst-1)
- [cs.MM (1)](#csmm-1)
- [cs.RO (2)](#csro-2)
- [cs.GT (1)](#csgt-1)

## cs.CL (24)



### (1/93) Entity Recognition from Colloquial Text (Tamara Babaian et al., 2024)

{{<citation>}}

Tamara Babaian, Jennifer Xu. (2024)  
**Entity Recognition from Colloquial Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.04853v1)  

---


**ABSTRACT**  
Extraction of concepts and entities of interest from non-formal texts such as social media posts and informal communication is an important capability for decision support systems in many domains, including healthcare, customer relationship management, and others. Despite the recent advances in training large language models for a variety of natural language processing tasks, the developed models and techniques have mainly focused on formal texts and do not perform as well on colloquial data, which is characterized by a number of distinct challenges. In our research, we focus on the healthcare domain and investigate the problem of symptom recognition from colloquial texts by designing and evaluating several training strategies for BERT-based model fine-tuning. These strategies are distinguished by the choice of the base model, the training corpora, and application of term perturbations in the training data. The best-performing models trained using these strategies outperform the state-of-the-art specialized symptom recognizer by a large margin. Through a series of experiments, we have found specific patterns of model behavior associated with the training strategies we designed. We present design principles for training strategies for effective entity recognition in colloquial texts based on our findings.

{{</citation>}}


### (2/93) Arabic Text Diacritization In The Age Of Transfer Learning: Token Classification Is All You Need (Abderrahman Skiredj et al., 2024)

{{<citation>}}

Abderrahman Skiredj, Ismail Berrada. (2024)  
**Arabic Text Diacritization In The Age Of Transfer Learning: Token Classification Is All You Need**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.04848v1)  

---


**ABSTRACT**  
Automatic diacritization of Arabic text involves adding diacritical marks (diacritics) to the text. This task poses a significant challenge with noteworthy implications for computational processing and comprehension. In this paper, we introduce PTCAD (Pre-FineTuned Token Classification for Arabic Diacritization, a novel two-phase approach for the Arabic Text Diacritization task. PTCAD comprises a pre-finetuning phase and a finetuning phase, treating Arabic Text Diacritization as a token classification task for pre-trained models. The effectiveness of PTCAD is demonstrated through evaluations on two benchmark datasets derived from the Tashkeela dataset, where it achieves state-of-the-art results, including a 20\% reduction in Word Error Rate (WER) compared to existing benchmarks and superior performance over GPT-4 in ATD tasks.

{{</citation>}}


### (3/93) MoSECroT: Model Stitching with Static Word Embeddings for Crosslingual Zero-shot Transfer (Haotian Ye et al., 2024)

{{<citation>}}

Haotian Ye, Yihong Liu, Chunlan Ma, Hinrich Schütze. (2024)  
**MoSECroT: Model Stitching with Static Word Embeddings for Crosslingual Zero-shot Transfer**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding, NLP, Transformer, Word Embedding  
[Paper Link](http://arxiv.org/abs/2401.04821v1)  

---


**ABSTRACT**  
Transformer-based pre-trained language models (PLMs) have achieved remarkable performance in various natural language processing (NLP) tasks. However, pre-training such models can take considerable resources that are almost only available to high-resource languages. On the contrary, static word embeddings are easier to train in terms of computing resources and the amount of data required. In this paper, we introduce MoSECroT Model Stitching with Static Word Embeddings for Crosslingual Zero-shot Transfer), a novel and challenging task that is especially relevant to low-resource languages for which static word embeddings are available. To tackle the task, we present the first framework that leverages relative representations to construct a common space for the embeddings of a source language PLM and the static word embeddings of a target language. In this way, we can train the PLM on source-language training data and perform zero-shot transfer to the target language by simply swapping the embedding layer. However, through extensive experiments on two classification datasets, we show that although our proposed framework is competitive with weak baselines when addressing MoSECroT, it fails to achieve competitive results compared with some strong baselines. In this paper, we attempt to explain this negative result and provide several thoughts on possible improvement.

{{</citation>}}


### (4/93) Model Editing Can Hurt General Abilities of Large Language Models (Jia-Chen Gu et al., 2024)

{{<citation>}}

Jia-Chen Gu, Hao-Xiang Xu, Jun-Yu Ma, Pan Lu, Zhen-Hua Ling, Kai-Wei Chang, Nanyun Peng. (2024)  
**Model Editing Can Hurt General Abilities of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04700v1)  

---


**ABSTRACT**  
Recent advances in large language models (LLMs) have opened up new paradigms for accessing the knowledge stored in their parameters. One critical challenge that has emerged is the presence of hallucinations in LLM outputs due to false or outdated knowledge. Since retraining LLMs with updated information is resource-intensive, there has been a growing interest in model editing. However, many model editing methods, while effective in various scenarios, tend to overemphasize aspects such as efficacy, generalization, and locality in editing performance, often overlooking potential side effects on the general abilities of LLMs. In this paper, we raise concerns that the improvement of model factuality may come at the cost of a significant degradation of these general abilities, which is not conducive to the sustainable development of LLMs. Systematically, we analyze side effects by evaluating four popular editing methods on two LLMs across eight representative task categories. Extensive empirical research reveals that model editing does improve model factuality but at the expense of substantially impairing general abilities. Therefore, we advocate for more research efforts to minimize the loss of general abilities acquired during LLM pre-training and to ultimately preserve them during model editing.

{{</citation>}}


### (5/93) Narrowing the Knowledge Evaluation Gap: Open-Domain Question Answering with Multi-Granularity Answers (Gal Yona et al., 2024)

{{<citation>}}

Gal Yona, Roee Aharoni, Mor Geva. (2024)  
**Narrowing the Knowledge Evaluation Gap: Open-Domain Question Answering with Multi-Granularity Answers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.04695v1)  

---


**ABSTRACT**  
Factual questions typically can be answered correctly at different levels of granularity. For example, both ``August 4, 1961'' and ``1961'' are correct answers to the question ``When was Barack Obama born?''. Standard question answering (QA) evaluation protocols, however, do not explicitly take this into account and compare a predicted answer against answers of a single granularity level. In this work, we propose GRANOLA QA, a novel evaluation setting where a predicted answer is evaluated in terms of accuracy and informativeness against a set of multi-granularity answers. We present a simple methodology for enriching existing datasets with multi-granularity answers, and create GRANOLA-EQ, a multi-granularity version of the EntityQuestions dataset. We evaluate a range of decoding methods on GRANOLA-EQ, including a new algorithm, called Decoding with Response Aggregation (DRAG), that is geared towards aligning the response granularity with the model's uncertainty. Our experiments show that large language models with standard decoding tend to generate specific answers, which are often incorrect. In contrast, when evaluated on multi-granularity answers, DRAG yields a nearly 20 point increase in accuracy on average, which further increases for rare entities. Overall, this reveals that standard evaluation and decoding schemes may significantly underestimate the knowledge encapsulated in LMs.

{{</citation>}}


### (6/93) Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models (Zhen Qin et al., 2024)

{{<citation>}}

Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, Yiran Zhong. (2024)  
**Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.04658v1)  

---


**ABSTRACT**  
Linear attention is an efficient attention mechanism that has recently emerged as a promising alternative to conventional softmax attention. With its ability to process tokens in linear computational complexities, linear attention, in theory, can handle sequences of unlimited length without sacrificing speed, i.e., maintaining a constant training speed for various sequence lengths with a fixed memory consumption. However, due to the issue with cumulative summation (cumsum), current linear attention algorithms cannot demonstrate their theoretical advantage in a causal setting. In this paper, we present Lightning Attention-2, the first linear attention implementation that enables linear attention to realize its theoretical computational benefits. To achieve this, we leverage the thought of tiling, separately handling the intra-block and inter-block components in linear attention calculation. Specifically, we utilize the conventional attention computation mechanism for the intra-blocks and apply linear attention kernel tricks for the inter-blocks. A tiling technique is adopted through both forward and backward procedures to take full advantage of the GPU hardware. We implement our algorithm in Triton to make it IO-aware and hardware-friendly. Various experiments are conducted on different model sizes and sequence lengths. Lightning Attention-2 retains consistent training and inference speed regardless of input sequence length and is significantly faster than other attention mechanisms. The source code is available at https://github.com/OpenNLPLab/lightning-attention.

{{</citation>}}


### (7/93) DepressionEmo: A novel dataset for multilabel classification of depression emotions (Abu Bakar Siddiqur Rahman et al., 2024)

{{<citation>}}

Abu Bakar Siddiqur Rahman, Hoang-Thang Ta, Lotfollah Najjar, Azad Azadmanesh, Ali Saffet Gönül. (2024)  
**DepressionEmo: A novel dataset for multilabel classification of depression emotions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.04655v1)  

---


**ABSTRACT**  
Emotions are integral to human social interactions, with diverse responses elicited by various situational contexts. Particularly, the prevalence of negative emotional states has been correlated with negative outcomes for mental health, necessitating a comprehensive analysis of their occurrence and impact on individuals. In this paper, we introduce a novel dataset named DepressionEmo designed to detect 8 emotions associated with depression by 6037 examples of long Reddit user posts. This dataset was created through a majority vote over inputs by zero-shot classifications from pre-trained models and validating the quality by annotators and ChatGPT, exhibiting an acceptable level of interrater reliability between annotators. The correlation between emotions, their distribution over time, and linguistic analysis are conducted on DepressionEmo. Besides, we provide several text classification methods classified into two groups: machine learning methods such as SVM, XGBoost, and Light GBM; and deep learning methods such as BERT, GAN-BERT, and BART. The pretrained BART model, bart-base allows us to obtain the highest F1- Macro of 0.76, showing its outperformance compared to other methods evaluated in our analysis. Across all emotions, the highest F1-Macro value is achieved by suicide intent, indicating a certain value of our dataset in identifying emotions in individuals with depression symptoms through text analysis. The curated dataset is publicly available at: https://github.com/abuBakarSiddiqurRahman/DepressionEmo.

{{</citation>}}


### (8/93) Agent Alignment in Evolving Social Norms (Shimin Li et al., 2024)

{{<citation>}}

Shimin Li, Tianxiang Sun, Xipeng Qiu. (2024)  
**Agent Alignment in Evolving Social Norms**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.04620v2)  

---


**ABSTRACT**  
Agents based on Large Language Models (LLMs) are increasingly permeating various domains of human production and life, highlighting the importance of aligning them with human values. The current alignment of AI systems primarily focuses on passively aligning LLMs through human intervention. However, agents possess characteristics like receiving environmental feedback and self-evolution, rendering the LLM alignment methods inadequate. In response, we propose an evolutionary framework for agent evolution and alignment, named EvolutionaryAgent, which transforms agent alignment into a process of evolution and selection under the principle of survival of the fittest. In an environment where social norms continuously evolve, agents better adapted to the current social norms will have a higher probability of survival and proliferation, while those inadequately aligned dwindle over time. Experimental results assessing the agents from multiple perspectives in aligning with social norms demonstrate that EvolutionaryAgent possesses the capability to align progressively better with the evolving social norms while maintaining its proficiency in general tasks. Effectiveness tests conducted on various open and closed-source LLMs as the foundation for agents also prove the applicability of our approach.

{{</citation>}}


### (9/93) Language Detection for Transliterated Content (Selva Kumar S et al., 2024)

{{<citation>}}

Selva Kumar S, Afifah Khan Mohammed Ajmal Khan, Chirag Manjeshwar, Imadh Ajaz Banday. (2024)  
**Language Detection for Transliterated Content**  

---
Primary Category: cs.CL  
Categories: C-m; I-2, cs-CL, cs.CL  
Keywords: BERT, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2401.04619v1)  

---


**ABSTRACT**  
In the contemporary digital era, the Internet functions as an unparalleled catalyst, dismantling geographical and linguistic barriers particularly evident in texting. This evolution facilitates global communication, transcending physical distances and fostering dynamic cultural exchange. A notable trend is the widespread use of transliteration, where the English alphabet is employed to convey messages in native languages, posing a unique challenge for language technology in accurately detecting the source language. This paper addresses this challenge through a dataset of phone text messages in Hindi and Russian transliterated into English utilizing BERT for language classification and Google Translate API for transliteration conversion. The research pioneers innovative approaches to identify and convert transliterated text, navigating challenges in the diverse linguistic landscape of digital communication. Emphasizing the pivotal role of comprehensive datasets for training Large Language Models LLMs like BERT, our model showcases exceptional proficiency in accurately identifying and classifying languages from transliterated text. With a validation accuracy of 99% our models robust performance underscores its reliability. The comprehensive exploration of transliteration dynamics supported by innovative approaches and cutting edge technologies like BERT, positions our research at the forefront of addressing unique challenges in the linguistic landscape of digital communication. Beyond contributing to language identification and transliteration capabilities this work holds promise for applications in content moderation, analytics and fostering a globally connected community engaged in meaningful dialogue.

{{</citation>}}


### (10/93) An Assessment on Comprehending Mental Health through Large Language Models (Mihael Arcan et al., 2024)

{{<citation>}}

Mihael Arcan, Paul-David Niland, Fionn Delahunty. (2024)  
**An Assessment on Comprehending Mental Health through Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, BERT, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.04592v1)  

---


**ABSTRACT**  
Mental health challenges pose considerable global burdens on individuals and communities. Recent data indicates that more than 20% of adults may encounter at least one mental disorder in their lifetime. On the one hand, the advancements in large language models have facilitated diverse applications, yet a significant research gap persists in understanding and enhancing the potential of large language models within the domain of mental health. On the other hand, across various applications, an outstanding question involves the capacity of large language models to comprehend expressions of human mental health conditions in natural language. This study presents an initial evaluation of large language models in addressing this gap. Due to this, we compare the performance of Llama-2 and ChatGPT with classical Machine as well as Deep learning models. Our results on the DAIC-WOZ dataset show that transformer-based models, like BERT or XLNet, outperform the large language models.

{{</citation>}}


### (11/93) Evaluating Language Model Agency through Negotiations (Tim R. Davidson et al., 2024)

{{<citation>}}

Tim R. Davidson, Veniamin Veselovsky, Martin Josifoski, Maxime Peyrard, Antoine Bosselut, Michal Kosinski, Robert West. (2024)  
**Evaluating Language Model Agency through Negotiations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04536v1)  

---


**ABSTRACT**  
Companies, organizations, and governments increasingly exploit Language Models' (LM) remarkable capability to display agent-like behavior. As LMs are adopted to perform tasks with growing autonomy, there exists an urgent need for reliable and scalable evaluation benchmarks. Current, predominantly static LM benchmarks are ill-suited to evaluate such dynamic applications. Thus, we propose jointly evaluating LM performance and alignment through the lenses of negotiation games. We argue that this common task better reflects real-world deployment conditions while offering insights into LMs' decision-making processes. Crucially, negotiation games allow us to study multi-turn, and cross-model interactions, modulate complexity, and side-step accidental data leakage in evaluation. We report results for six publicly accessible LMs from several major providers on a variety of negotiation games, evaluating both self-play and cross-play performance. Noteworthy findings include: (i) open-source models are currently unable to complete these tasks; (ii) cooperative bargaining games prove challenging; and (iii) the most powerful models do not always "win".

{{</citation>}}


### (12/93) MERA: A Comprehensive LLM Evaluation in Russian (Alena Fenogenova et al., 2024)

{{<citation>}}

Alena Fenogenova, Artem Chervyakov, Nikita Martynov, Anastasia Kozlova, Maria Tikhonova, Albina Akhmetgareeva, Anton Emelyanov, Denis Shevelev, Pavel Lebedev, Leonid Sinev, Ulyana Isaeva, Katerina Kolomeytseva, Daniil Moskovskiy, Elizaveta Goncharova, Nikita Savushkin, Polina Mikhailova, Denis Dimitrov, Alexander Panchenko, Sergei Markov. (2024)  
**MERA: A Comprehensive LLM Evaluation in Russian**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04531v2)  

---


**ABSTRACT**  
Over the past few years, one of the most notable advancements in AI research has been in foundation models (FMs), headlined by the rise of language models (LMs). As the models' size increases, LMs demonstrate enhancements in measurable aspects and the development of new qualitative features. However, despite researchers' attention and the rapid growth in LM application, the capabilities, limitations, and associated risks still need to be better understood. To address these issues, we introduce an open Multimodal Evaluation of Russian-language Architectures (MERA), a new instruction benchmark for evaluating foundation models oriented towards the Russian language. The benchmark encompasses 21 evaluation tasks for generative models in 11 skill domains and is designed as a black-box test to ensure the exclusion of data leakage. The paper introduces a methodology to evaluate FMs and LMs in zero- and few-shot fixed instruction settings that can be extended to other modalities. We propose an evaluation methodology, an open-source code base for the MERA assessment, and a leaderboard with a submission system. We evaluate open LMs as baselines and find that they are still far behind the human level. We publicly release MERA to guide forthcoming research, anticipate groundbreaking model features, standardize the evaluation procedure, and address potential societal drawbacks.

{{</citation>}}


### (13/93) LUNA: A Framework for Language Understanding and Naturalness Assessment (Marat Saidov et al., 2024)

{{<citation>}}

Marat Saidov, Aleksandra Bakalova, Ekaterina Taktasheva, Vladislav Mikhailov, Ekaterina Artemova. (2024)  
**LUNA: A Framework for Language Understanding and Naturalness Assessment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2401.04522v1)  

---


**ABSTRACT**  
The evaluation of Natural Language Generation (NLG) models has gained increased attention, urging the development of metrics that evaluate various aspects of generated text. LUNA addresses this challenge by introducing a unified interface for 20 NLG evaluation metrics. These metrics are categorized based on their reference-dependence and the type of text representation they employ, from string-based n-gram overlap to the utilization of static embeddings and pre-trained language models.   The straightforward design of LUNA allows for easy extension with novel metrics, requiring just a few lines of code. LUNA offers a user-friendly tool for evaluating generated texts.

{{</citation>}}


### (14/93) The Critique of Critique (Shichao Sun et al., 2024)

{{<citation>}}

Shichao Sun, Junlong Li, Weizhe Yuan, Ruifeng Yuan, Wenjie Li, Pengfei Liu. (2024)  
**The Critique of Critique**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.04518v1)  

---


**ABSTRACT**  
Critique, as a natural language description for assessing the quality of model-generated content, has been proven to play an essential role in the training, evaluation, and refinement of Large Language Models (LLMs). However, there is a lack of principled understanding in evaluating the quality of the critique itself. In this paper, we pioneer the critique of critique, termed MetaCritique, which is a framework to evaluate the critique from two aspects, i.e., factuality as precision score and comprehensiveness as recall score. We calculate the harmonic mean of precision and recall as the overall rating called F1 score. To obtain a reliable evaluation outcome, we propose Atomic Information Units (AIUs), which describe the critique in a more fine-grained manner. MetaCritique takes each AIU into account and aggregates each AIU's judgment for the overall score. Moreover, given the evaluation process involves intricate reasoning, our MetaCritique provides a natural language rationale to support each judgment. We construct a meta-evaluation dataset containing 300 critiques (2653 AIUs) across four tasks (question answering, reasoning, entailment, and summarization), and we conduct a comparative study to demonstrate the feasibility and effectiveness. Experiments also show superior critique judged by MetaCritique leads to better refinement, indicating generative artificial intelligence indeed has the potential to be significantly advanced with our MetaCritique. We will release relevant code and meta-evaluation datasets at https://github.com/GAIR-NLP/MetaCritique.

{{</citation>}}


### (15/93) Exploring Prompt-Based Methods for Zero-Shot Hypernym Prediction with Large Language Models (Mikhail Tikhomirov et al., 2024)

{{<citation>}}

Mikhail Tikhomirov, Natalia Loukachevitch. (2024)  
**Exploring Prompt-Based Methods for Zero-Shot Hypernym Prediction with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.04515v1)  

---


**ABSTRACT**  
This article investigates a zero-shot approach to hypernymy prediction using large language models (LLMs). The study employs a method based on text probability calculation, applying it to various generated prompts. The experiments demonstrate a strong correlation between the effectiveness of language model prompts and classic patterns, indicating that preliminary prompt selection can be carried out using smaller models before moving to larger ones. We also explore prompts for predicting co-hyponyms and improving hypernymy predictions by augmenting prompts with additional information through automatically identified co-hyponyms. An iterative approach is developed for predicting higher-level concepts, which further improves the quality on the BLESS dataset (MAP = 0.8).

{{</citation>}}


### (16/93) TechGPT-2.0: A large language model project to solve the task of knowledge graph construction (Jiaqi Wang et al., 2024)

{{<citation>}}

Jiaqi Wang, Yuying Chang, Zhong Li, Ning An, Qi Ma, Lei Hei, Haibo Luo, Yifei Lu, Feiliang Ren. (2024)  
**TechGPT-2.0: A large language model project to solve the task of knowledge graph construction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, NER, NLP  
[Paper Link](http://arxiv.org/abs/2401.04507v1)  

---


**ABSTRACT**  
Large language models have exhibited robust performance across diverse natural language processing tasks. This report introduces TechGPT-2.0, a project designed to enhance the capabilities of large language models specifically in knowledge graph construction tasks, including named entity recognition (NER) and relationship triple extraction (RTE) tasks in NLP applications. Additionally, it serves as a LLM accessible for research within the Chinese open-source model community. We offer two 7B large language model weights and a QLoRA weight specialized for processing lengthy texts.Notably, TechGPT-2.0 is trained on Huawei's Ascend server. Inheriting all functionalities from TechGPT-1.0, it exhibits robust text processing capabilities, particularly in the domains of medicine and law. Furthermore, we introduce new capabilities to the model, enabling it to process texts in various domains such as geographical areas, transportation, organizations, literary works, biology, natural sciences, astronomical objects, and architecture. These enhancements also fortified the model's adeptness in handling hallucinations, unanswerable queries, and lengthy texts. This report provides a comprehensive and detailed introduction to the full fine-tuning process on Huawei's Ascend servers, encompassing experiences in Ascend server debugging, instruction fine-tuning data processing, and model training. Our code is available at https://github.com/neukg/TechGPT-2.0

{{</citation>}}


### (17/93) Continuously Learning New Words in Automatic Speech Recognition (Christian Huber et al., 2024)

{{<citation>}}

Christian Huber, Alexander Waibel. (2024)  
**Continuously Learning New Words in Automatic Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.04482v1)  

---


**ABSTRACT**  
Despite recent advances, Automatic Speech Recognition (ASR) systems are still far from perfect. Typical errors include acronyms, named entities and domain-specific special words for which little or no data is available. To address the problem of recognizing these words, we propose an self-supervised continual learning approach. Given the audio of a lecture talk with corresponding slides, we bias the model towards decoding new words from the slides by using a memory-enhanced ASR model from previous work. Then, we perform inference on the talk, collecting utterances that contain detected new words into an adaptation dataset. Continual learning is then performed on this set by adapting low-rank matrix weights added to each weight matrix of the model. The whole procedure is iterated for many talks. We show that with this approach, we obtain increasing performance on the new words when they occur more frequently (more than 80% recall) while preserving the general performance of the model.

{{</citation>}}


### (18/93) Fighting Fire with Fire: Adversarial Prompting to Generate a Misinformation Detection Dataset (Shrey Satapara et al., 2024)

{{<citation>}}

Shrey Satapara, Parth Mehta, Debasis Ganguly, Sandip Modha. (2024)  
**Fighting Fire with Fire: Adversarial Prompting to Generate a Misinformation Detection Dataset**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.04481v1)  

---


**ABSTRACT**  
The recent success in language generation capabilities of large language models (LLMs), such as GPT, Bard, Llama etc., can potentially lead to concerns about their possible misuse in inducing mass agitation and communal hatred via generating fake news and spreading misinformation. Traditional means of developing a misinformation ground-truth dataset does not scale well because of the extensive manual effort required to annotate the data. In this paper, we propose an LLM-based approach of creating silver-standard ground-truth datasets for identifying misinformation. Specifically speaking, given a trusted news article, our proposed approach involves prompting LLMs to automatically generate a summarised version of the original article. The prompts in our proposed approach act as a controlling mechanism to generate specific types of factual incorrectness in the generated summaries, e.g., incorrect quantities, false attributions etc. To investigate the usefulness of this dataset, we conduct a set of experiments where we train a range of supervised models for the task of misinformation detection.

{{</citation>}}


### (19/93) TransportationGames: Benchmarking Transportation Knowledge of (Multimodal) Large Language Models (Xue Zhang et al., 2024)

{{<citation>}}

Xue Zhang, Xiangyu Shi, Xinyue Lou, Rui Qi, Yufeng Chen, Jinan Xu, Wenjuan Han. (2024)  
**TransportationGames: Benchmarking Transportation Knowledge of (Multimodal) Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04471v1)  

---


**ABSTRACT**  
Large language models (LLMs) and multimodal large language models (MLLMs) have shown excellent general capabilities, even exhibiting adaptability in many professional domains such as law, economics, transportation, and medicine. Currently, many domain-specific benchmarks have been proposed to verify the performance of (M)LLMs in specific fields. Among various domains, transportation plays a crucial role in modern society as it impacts the economy, the environment, and the quality of life for billions of people. However, it is unclear how much traffic knowledge (M)LLMs possess and whether they can reliably perform transportation-related tasks. To address this gap, we propose TransportationGames, a carefully designed and thorough evaluation benchmark for assessing (M)LLMs in the transportation domain. By comprehensively considering the applications in real-world scenarios and referring to the first three levels in Bloom's Taxonomy, we test the performance of various (M)LLMs in memorizing, understanding, and applying transportation knowledge by the selected tasks. The experimental results show that although some models perform well in some tasks, there is still much room for improvement overall. We hope the release of TransportationGames can serve as a foundation for future research, thereby accelerating the implementation and application of (M)LLMs in the transportation domain.

{{</citation>}}


### (20/93) Estimating Text Similarity based on Semantic Concept Embeddings (Tim vor der Brück et al., 2024)

{{<citation>}}

Tim vor der Brück, Marc Pouly. (2024)  
**Estimating Text Similarity based on Semantic Concept Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.04422v1)  

---


**ABSTRACT**  
Due to their ease of use and high accuracy, Word2Vec (W2V) word embeddings enjoy great success in the semantic representation of words, sentences, and whole documents as well as for semantic similarity estimation. However, they have the shortcoming that they are directly extracted from a surface representation, which does not adequately represent human thought processes and also performs poorly for highly ambiguous words. Therefore, we propose Semantic Concept Embeddings (CE) based on the MultiNet Semantic Network (SN) formalism, which addresses both shortcomings. The evaluation on a marketing target group distribution task showed that the accuracy of predicted target groups can be increased by combining traditional word embeddings with semantic CEs.

{{</citation>}}


### (21/93) Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding (Zilong Wang et al., 2024)

{{<citation>}}

Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee, Tomas Pfister. (2024)  
**Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.04398v1)  

---


**ABSTRACT**  
Table-based reasoning with large language models (LLMs) is a promising direction to tackle many table understanding tasks, such as table-based question answering and fact verification. Compared with generic reasoning, table-based reasoning requires the extraction of underlying semantics from both free-form questions and semi-structured tabular data. Chain-of-Thought and its similar approaches incorporate the reasoning chain in the form of textual context, but it is still an open question how to effectively leverage tabular data in the reasoning chain. We propose the Chain-of-Table framework, where tabular data is explicitly used in the reasoning chain as a proxy for intermediate thoughts. Specifically, we guide LLMs using in-context learning to iteratively generate operations and update the table to represent a tabular reasoning chain. LLMs can therefore dynamically plan the next operation based on the results of the previous ones. This continuous evolution of the table forms a chain, showing the reasoning process for a given tabular problem. The chain carries structured information of the intermediate results, enabling more accurate and reliable predictions. Chain-of-Table achieves new state-of-the-art performance on WikiTQ, FeTaQA, and TabFact benchmarks across multiple LLM choices.

{{</citation>}}


### (22/93) Improving the Robustness of Knowledge-Grounded Dialogue via Contrastive Learning (Jiaan Wang et al., 2024)

{{<citation>}}

Jiaan Wang, Jianfeng Qu, Kexin Wang, Zhixu Li, Wen Hua, Ximing Li, An Liu. (2024)  
**Improving the Robustness of Knowledge-Grounded Dialogue via Contrastive Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Contrastive Learning, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2401.04361v1)  

---


**ABSTRACT**  
Knowledge-grounded dialogue (KGD) learns to generate an informative response based on a given dialogue context and external knowledge (\emph{e.g.}, knowledge graphs; KGs). Recently, the emergence of large language models (LLMs) and pre-training techniques has brought great success to knowledge-grounded dialogue. However, when building KGD systems in real applications, there are various real-world noises that are inevitable to face. For example, the dialogue context might involve perturbations such as misspellings and abbreviations. In addition, KGs typically suffer from incompletion and also might contain erroneous and outdated facts. Such real-world noises pose a challenge to the robustness of KGD systems and hinder their applications in the real world. In this paper, we propose an entity-based contrastive learning framework for improving the robustness of KGD. Specifically, we make use of the entity information in a KGD sample to create both its positive and negative samples which involve semantic-irrelevant and semantic-relevant perturbations, respectively. The contrastive learning framework ensures the KGD model is aware of these two types of perturbations, thus generating informative responses with the potentially noisy inputs in real applications. Experimental results on three benchmark datasets show that our method achieves new state-of-the-art performance in terms of automatic evaluation scores, verifying its effectiveness and potentiality. Furthermore, we show that our method can generate better responses than comparison models in both the noisy and the few-shot settings.

{{</citation>}}


### (23/93) LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training (Khoi M. Le et al., 2024)

{{<citation>}}

Khoi M. Le, Trinh Pham, Tho Quan, Anh Tuan Luu. (2024)  
**LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Adversarial Training, Multilingual, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.04348v1)  

---


**ABSTRACT**  
Paraphrases are texts that convey the same meaning while using different words or sentence structures. It can be used as an automatic data augmentation tool for many Natural Language Processing tasks, especially when dealing with low-resource languages, where data shortage is a significant problem. To generate a paraphrase in multilingual settings, previous studies have leveraged the knowledge from the machine translation field, i.e., forming a paraphrase through zero-shot machine translation in the same language. Despite good performance on human evaluation, those methods still require parallel translation datasets, thus making them inapplicable to languages that do not have parallel corpora. To mitigate that problem, we proposed the first unsupervised multilingual paraphrasing model, LAMPAT ($\textbf{L}$ow-rank $\textbf{A}$daptation for $\textbf{M}$ultilingual $\textbf{P}$araphrasing using $\textbf{A}$dversarial $\textbf{T}$raining), by which monolingual dataset is sufficient enough to generate a human-like and diverse sentence. Throughout the experiments, we found out that our method not only works well for English but can generalize on unseen languages as well. Data and code are available at https://github.com/phkhanhtrinh23/LAMPAT.

{{</citation>}}


### (24/93) Know Your Needs Better: Towards Structured Understanding of Marketer Demands with Analogical Reasoning Augmented LLMs (Junjie Wang et al., 2024)

{{<citation>}}

Junjie Wang, Dan Yang, Binbin Hu, Yue Shen, Ziqi Liu, Wen Zhang, Jinjie Gu, Zhiqiang Zhang. (2024)  
**Know Your Needs Better: Towards Structured Understanding of Marketer Demands with Analogical Reasoning Augmented LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Model Distillation, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.04319v1)  

---


**ABSTRACT**  
In this paper, we explore a new way for user targeting, where non-expert marketers could select their target users solely given demands in natural language form. The key to this issue is how to transform natural languages into practical structured logical languages, i.e., the structured understanding of marketer demands. Considering the impressive natural language processing ability of large language models (LLMs), we try to leverage LLMs to solve this issue. Past research indicates that the reasoning ability of LLMs can be effectively enhanced through chain-of-thought (CoT) prompting. But existing methods still have some limitations: (1) Previous methods either use simple "Let's think step by step" spells or provide fixed examples in demonstrations without considering compatibility between prompts and questions, making LLMs ineffective in some complex reasoning tasks such as structured language transformation. (2) Previous methods are often implemented in closed-source models or excessively large models, which is not suitable in industrial practical scenarios. Based on these, we propose ARALLM (i.e., Analogical Reasoning Augmented Large Language Models) consisting of two modules: Analogical Reasoning based Prompting and Reasoning-Augmented Multi-Task Model Distillation.

{{</citation>}}


## cs.IR (5)



### (25/93) Answer Retrieval in Legal Community Question Answering (Arian Askari et al., 2024)

{{<citation>}}

Arian Askari, Zihui Yang, Zhaochun Ren, Suzan Verberne. (2024)  
**Answer Retrieval in Legal Community Question Answering**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Legal, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.04852v1)  

---


**ABSTRACT**  
The task of answer retrieval in the legal domain aims to help users to seek relevant legal advice from massive amounts of professional responses. Two main challenges hinder applying existing answer retrieval approaches in other domains to the legal domain: (1) a huge knowledge gap between lawyers and non-professionals; and (2) a mix of informal and formal content on legal QA websites. To tackle these challenges, we propose CE_FS, a novel cross-encoder (CE) re-ranker based on the fine-grained structured inputs. CE_FS uses additional structured information in the CQA data to improve the effectiveness of cross-encoder re-rankers. Furthermore, we propose LegalQA: a real-world benchmark dataset for evaluating answer retrieval in the legal domain. Experiments conducted on LegalQA show that our proposed method significantly outperforms strong cross-encoder re-rankers fine-tuned on MS MARCO. Our novel finding is that adding the question tags of each question besides the question description and title into the input of cross-encoder re-rankers structurally boosts the rankers' effectiveness. While we study our proposed method in the legal domain, we believe that our method can be applied in similar applications in other domains.

{{</citation>}}


### (26/93) Adapting Standard Retrieval Benchmarks to Evaluate Generated Answers (Negar Arabzadeh et al., 2024)

{{<citation>}}

Negar Arabzadeh, Amin Bigdeli, Charles L. A. Clarke. (2024)  
**Adapting Standard Retrieval Benchmarks to Evaluate Generated Answers**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.04842v1)  

---


**ABSTRACT**  
Large language models can now directly generate answers to many factual questions without referencing external sources. Unfortunately, relatively little attention has been paid to methods for evaluating the quality and correctness of these answers, for comparing the performance of one model to another, or for comparing one prompt to another. In addition, the quality of generated answers are rarely directly compared to the quality of retrieved answers. As models evolve and prompts are modified, we have no systematic way to measure improvements without resorting to expensive human judgments. To address this problem we adapt standard retrieval benchmarks to evaluate answers generated by large language models. Inspired by the BERTScore metric for summarization, we explore two approaches. In the first, we base our evaluation on the benchmark relevance judgments. We empirically run experiments on how information retrieval relevance judgments can be utilized as an anchor to evaluating the generated answers. In the second, we compare generated answers to the top results retrieved by a diverse set of retrieval models, ranging from traditional approaches to advanced methods, allowing us to measure improvements without human judgments. In both cases, we measure the similarity between an embedded representation of the generated answer and an embedded representation of a known, or assumed, relevant passage from the retrieval benchmark.

{{</citation>}}


### (27/93) Translate-Distill: Learning Cross-Language Dense Retrieval by Translation and Distillation (Eugene Yang et al., 2024)

{{<citation>}}

Eugene Yang, Dawn Lawrie, James Mayfield, Douglas W. Oard, Scott Miller. (2024)  
**Translate-Distill: Learning Cross-Language Dense Retrieval by Translation and Distillation**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2401.04810v1)  

---


**ABSTRACT**  
Prior work on English monolingual retrieval has shown that a cross-encoder trained using a large number of relevance judgments for query-document pairs can be used as a teacher to train more efficient, but similarly effective, dual-encoder student models. Applying a similar knowledge distillation approach to training an efficient dual-encoder model for Cross-Language Information Retrieval (CLIR), where queries and documents are in different languages, is challenging due to the lack of a sufficiently large training collection when the query and document languages differ. The state of the art for CLIR thus relies on translating queries, documents, or both from the large English MS MARCO training set, an approach called Translate-Train. This paper proposes an alternative, Translate-Distill, in which knowledge distillation from either a monolingual cross-encoder or a CLIR cross-encoder is used to train a dual-encoder CLIR student model. This richer design space enables the teacher model to perform inference in an optimized setting, while training the student model directly for CLIR. Trained models and artifacts are publicly available on Huggingface.

{{</citation>}}


### (28/93) Combining Embedding-Based and Semantic-Based Models for Post-hoc Explanations in Recommender Systems (Ngoc Luyen Le et al., 2024)

{{<citation>}}

Ngoc Luyen Le, Marie-Hélène Abel, Philippe Gouspillou. (2024)  
**Combining Embedding-Based and Semantic-Based Models for Post-hoc Explanations in Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.04474v1)  

---


**ABSTRACT**  
In today's data-rich environment, recommender systems play a crucial role in decision support systems. They provide to users personalized recommendations and explanations about these recommendations. Embedding-based models, despite their widespread use, often suffer from a lack of interpretability, which can undermine trust and user engagement. This paper presents an approach that combines embedding-based and semantic-based models to generate post-hoc explanations in recommender systems, leveraging ontology-based knowledge graphs to improve interpretability and explainability. By organizing data within a structured framework, ontologies enable the modeling of intricate relationships between entities, which is essential for generating explanations. By combining embedding-based and semantic based models for post-hoc explanations in recommender systems, the framework we defined aims at producing meaningful and easy-to-understand explanations, enhancing user trust and satisfaction, and potentially promoting the adoption of recommender systems across the e-commerce sector.

{{</citation>}}


### (29/93) Fine-Grained Embedding Dimension Optimization During Training for Recommender Systems (Qinyi Luo et al., 2024)

{{<citation>}}

Qinyi Luo, Penghan Wang, Wei Zhang, Fan Lai, Jiachen Mao, Xiaohan Wei, Jun Song, Wei-Yu Tsai, Shuai Yang, Yuxi Hu, Xuehai Qian. (2024)  
**Fine-Grained Embedding Dimension Optimization During Training for Recommender Systems**  

---
Primary Category: cs.IR  
Categories: I-2-6; H-3-3, cs-IR, cs-LG, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.04408v1)  

---


**ABSTRACT**  
Huge embedding tables in modern Deep Learning Recommender Models (DLRM) require prohibitively large memory during training and inference. Aiming to reduce the memory footprint of training, this paper proposes FIne-grained In-Training Embedding Dimension optimization (FIITED). Given the observation that embedding vectors are not equally important, FIITED adjusts the dimension of each individual embedding vector continuously during training, assigning longer dimensions to more important embeddings while adapting to dynamic changes in data. A novel embedding storage system based on virtually-hashed physically-indexed hash tables is designed to efficiently implement the embedding dimension adjustment and effectively enable memory saving. Experiments on two industry models show that FIITED is able to reduce the size of embeddings by more than 65% while maintaining the trained model's quality, saving significantly more memory than a state-of-the-art in-training embedding pruning method. On public click-through rate prediction datasets, FIITED is able to prune up to 93.75%-99.75% embeddings without significant accuracy loss.

{{</citation>}}


## cs.MA (1)



### (30/93) Graph Learning-based Fleet Scheduling for Urban Air Mobility under Operational Constraints, Varying Demand & Uncertainties (Steve Paul et al., 2024)

{{<citation>}}

Steve Paul, Jhoel Witter, Souma Chowdhury. (2024)  
**Graph Learning-based Fleet Scheduling for Urban Air Mobility under Operational Constraints, Varying Demand & Uncertainties**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-LG, cs-MA, cs.MA  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.04851v1)  

---


**ABSTRACT**  
This paper develops a graph reinforcement learning approach to online planning of the schedule and destinations of electric aircraft that comprise an urban air mobility (UAM) fleet operating across multiple vertiports. This fleet scheduling problem is formulated to consider time-varying demand, constraints related to vertiport capacity, aircraft capacity and airspace safety guidelines, uncertainties related to take-off delay, weather-induced route closures, and unanticipated aircraft downtime. Collectively, such a formulation presents greater complexity, and potentially increased realism, than in existing UAM fleet planning implementations. To address these complexities, a new policy architecture is constructed, primary components of which include: graph capsule conv-nets for encoding vertiport and aircraft-fleet states both abstracted as graphs; transformer layers encoding time series information on demand and passenger fare; and a Multi-head Attention-based decoder that uses the encoded information to compute the probability of selecting each available destination for an aircraft. Trained with Proximal Policy Optimization, this policy architecture shows significantly better performance in terms of daily averaged profits on unseen test scenarios involving 8 vertiports and 40 aircraft, when compared to a random baseline and genetic algorithm-derived optimal solutions, while being nearly 1000 times faster in execution than the latter.

{{</citation>}}


## econ.EM (1)



### (31/93) A Deep Learning Representation of Spatial Interaction Model for Resilient Spatial Planning of Community Business Clusters (Haiyan Hao et al., 2024)

{{<citation>}}

Haiyan Hao, Yan Wang. (2024)  
**A Deep Learning Representation of Spatial Interaction Model for Resilient Spatial Planning of Community Business Clusters**  

---
Primary Category: econ.EM  
Categories: cs-AI, econ-EM, econ.EM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04849v1)  

---


**ABSTRACT**  
Existing Spatial Interaction Models (SIMs) are limited in capturing the complex and context-aware interactions between business clusters and trade areas. To address the limitation, we propose a SIM-GAT model to predict spatiotemporal visitation flows between community business clusters and their trade areas. The model innovatively represents the integrated system of business clusters, trade areas, and transportation infrastructure within an urban region using a connected graph. Then, a graph-based deep learning model, i.e., Graph AttenTion network (GAT), is used to capture the complexity and interdependencies of business clusters. We developed this model with data collected from the Miami metropolitan area in Florida. We then demonstrated its effectiveness in capturing varying attractiveness of business clusters to different residential neighborhoods and across scenarios with an eXplainable AI approach. We contribute a novel method supplementing conventional SIMs to predict and analyze the dynamics of inter-connected community business clusters. The analysis results can inform data-evidenced and place-specific planning strategies helping community business clusters better accommodate their customers across scenarios, and hence improve the resilience of community businesses.

{{</citation>}}


## cs.LG (15)



### (32/93) T-PRIME: Transformer-based Protocol Identification for Machine-learning at the Edge (Mauro Belgiovine et al., 2024)

{{<citation>}}

Mauro Belgiovine, Joshua Groen, Miquel Sirera, Chinenye Tassie, Ayberk Yarkin Yildiz, Sage Trudeau, Stratis Ioannidis, Kaushik Chowdhury. (2024)  
**T-PRIME: Transformer-based Protocol Identification for Machine-learning at the Edge**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs-SY, cs.LG, eess-SY  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2401.04837v1)  

---


**ABSTRACT**  
Spectrum sharing allows different protocols of the same standard (e.g., 802.11 family) or different standards (e.g., LTE and DVB) to coexist in overlapping frequency bands. As this paradigm continues to spread, wireless systems must also evolve to identify active transmitters and unauthorized waveforms in real time under intentional distortion of preambles, extremely low signal-to-noise ratios and challenging channel conditions. We overcome limitations of correlation-based preamble matching methods in such conditions through the design of T-PRIME: a Transformer-based machine learning approach. T-PRIME learns the structural design of transmitted frames through its attention mechanism, looking at sequence patterns that go beyond the preamble alone. The paper makes three contributions: First, it compares Transformer models and demonstrates their superiority over traditional methods and state-of-the-art neural networks. Second, it rigorously analyzes T-PRIME's real-time feasibility on DeepWave's AIR-T platform. Third, it utilizes an extensive 66 GB dataset of over-the-air (OTA) WiFi transmissions for training, which is released along with the code for community use. Results reveal nearly perfect (i.e. $>98\%$) classification accuracy under simulated scenarios, showing $100\%$ detection improvement over legacy methods in low SNR ranges, $97\%$ classification accuracy for OTA single-protocol transmissions and up to $75\%$ double-protocol classification accuracy in interference scenarios.

{{</citation>}}


### (33/93) GNNShap: Fast and Accurate GNN Explanations using Shapley Values (Selahattin Akkas et al., 2024)

{{<citation>}}

Selahattin Akkas, Ariful Azad. (2024)  
**GNNShap: Fast and Accurate GNN Explanations using Shapley Values**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.04829v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) are popular machine learning models for graphs with many applications across scientific domains. However, GNNs are considered black box models, and it is challenging to understand how the model makes predictions. Game theory-based Shapley value approaches are popular explanation methods in other domains but are not well-studied for graphs. Some studies have proposed Shapley value-based GNN explanations, yet they have several limitations: they consider limited samples to approximate Shapley values; some mainly focus on small and large coalition sizes, and they are an order of magnitude slower than other explanation methods, making them inapplicable to even moderate-size graphs. In this work, we propose GNNShap, which provides explanations for edges since they provide more natural explanations for graphs and more fine-grained explanations. We overcome the limitations by sampling from all coalition sizes, parallelizing the sampling on GPUs, and speeding up model predictions by batching. GNNShap gives better fidelity scores and faster explanations than baselines on real-world datasets.

{{</citation>}}


### (34/93) AI-based Mapping of the Conservation Status of Orchid Assemblages at Global Scale (Joaquim Estopinan et al., 2024)

{{<citation>}}

Joaquim Estopinan, Maximilien Servajean, Pierre Bonnet, Alexis Joly, François Munoz. (2024)  
**AI-based Mapping of the Conservation Status of Orchid Assemblages at Global Scale**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04691v1)  

---


**ABSTRACT**  
Although increasing threats on biodiversity are now widely recognised, there are no accurate global maps showing whether and where species assemblages are at risk. We hereby assess and map at kilometre resolution the conservation status of the iconic orchid family, and discuss the insights conveyed at multiple scales. We introduce a new Deep Species Distribution Model trained on 1M occurrences of 14K orchid species to predict their assemblages at global scale and at kilometre resolution. We propose two main indicators of the conservation status of the assemblages: (i) the proportion of threatened species, and (ii) the status of the most threatened species in the assemblage. We show and analyze the variation of these indicators at World scale and in relation to currently protected areas in Sumatra island. Global and interactive maps available online show the indicators of conservation status of orchid assemblages, with sharp spatial variations at all scales. The highest level of threat is found at Madagascar and the neighbouring islands. In Sumatra, we found good correspondence of protected areas with our indicators, but supplementing current IUCN assessments with status predictions results in alarming levels of species threat across the island. Recent advances in deep learning enable reliable mapping of the conservation status of species assemblages on a global scale. As an umbrella taxon, orchid family provides a reference for identifying vulnerable ecosystems worldwide, and prioritising conservation actions both at international and local levels.

{{</citation>}}


### (35/93) How predictable is language model benchmark performance? (David Owen, 2024)

{{<citation>}}

David Owen. (2024)  
**How predictable is language model benchmark performance?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04757v1)  

---


**ABSTRACT**  
We investigate large language model performance across five orders of magnitude of compute scaling in eleven recent model architectures. We show that average benchmark performance, aggregating over many individual tasks and evaluations as in the commonly-used BIG-Bench dataset, is decently predictable as a function of training compute scale. Specifically, when extrapolating BIG-Bench Hard performance across one order of magnitude in compute, we observe average absolute errors of 6 percentage points (pp). By contrast, extrapolation for individual BIG-Bench tasks across an order of magnitude in compute yields higher average errors of 18pp. Nonetheless, individual task performance remains significantly more predictable than chance. Overall, our work suggests compute scaling provides a promising basis to forecast AI capabilities in diverse benchmarks, though predicting performance in specific tasks poses challenges.

{{</citation>}}


### (36/93) Identifying Best Practice Melting Patterns in Induction Furnaces: A Data-Driven Approach Using Time Series KMeans Clustering and Multi-Criteria Decision Making (Daniel Anthony Howard et al., 2024)

{{<citation>}}

Daniel Anthony Howard, Bo Nørregaard Jørgensen, Zheng Ma. (2024)  
**Identifying Best Practice Melting Patterns in Induction Furnaces: A Data-Driven Approach Using Time Series KMeans Clustering and Multi-Criteria Decision Making**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NA, cs-PF, cs.LG, math-NA  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.04751v1)  

---


**ABSTRACT**  
Improving energy efficiency in industrial production processes is crucial for competitiveness, and compliance with climate policies. This paper introduces a data-driven approach to identify optimal melting patterns in induction furnaces. Through time-series K-means clustering the melting patterns could be classified into distinct clusters based on temperature profiles. Using the elbow method, 12 clusters were identified, representing the range of melting patterns. Performance parameters such as melting time, energy-specific performance, and carbon cost were established for each cluster, indicating furnace efficiency and environmental impact. Multiple criteria decision-making methods including Simple Additive Weighting, Multiplicative Exponential Weighting, Technique for Order of Preference by Similarity to Ideal Solution, modified TOPSIS, and VlseKriterijumska Optimizacija I Kompromisno Resenje were utilized to determine the best-practice cluster. The study successfully identified the cluster with the best performance. Implementing the best practice operation resulted in an 8.6 % reduction in electricity costs, highlighting the potential energy savings in the foundry.

{{</citation>}}


### (37/93) LogFormer: A Pre-train and Tuning Pipeline for Log Anomaly Detection (Hongcheng Guo et al., 2024)

{{<citation>}}

Hongcheng Guo, Jian Yang, Jiaheng Liu, Jiaqi Bai, Boyang Wang, Zhoujun Li, Tieqiao Zheng, Bo Zhang, Junran peng, Qi Tian. (2024)  
**LogFormer: A Pre-train and Tuning Pipeline for Log Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SE, cs.LG  
Keywords: AI, Anomaly Detection, Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2401.04749v1)  

---


**ABSTRACT**  
Log anomaly detection is a key component in the field of artificial intelligence for IT operations (AIOps). Considering log data of variant domains, retraining the whole network for unknown domains is inefficient in real industrial scenarios. However, previous deep models merely focused on extracting the semantics of log sequences in the same domain, leading to poor generalization on multi-domain logs. To alleviate this issue, we propose a unified Transformer-based framework for Log anomaly detection (LogFormer) to improve the generalization ability across different domains, where we establish a two-stage process including the pre-training and adapter-based tuning stage. Specifically, our model is first pre-trained on the source domain to obtain shared semantic knowledge of log data. Then, we transfer such knowledge to the target domain via shared parameters. Besides, the Log-Attention module is proposed to supplement the information ignored by the log-paring. The proposed method is evaluated on three public and one real-world datasets. Experimental results on multiple benchmarks demonstrate the effectiveness of our LogFormer with fewer trainable parameters and lower training costs.

{{</citation>}}


### (38/93) AI Competitions and Benchmarks, Practical issues: Proposals, grant money, sponsors, prizes, dissemination, publicity (Magali Richard et al., 2024)

{{<citation>}}

Magali Richard, Yuna Blum, Justin Guinney, Gustavo Stolovitzky, Adrien Pavão. (2024)  
**AI Competitions and Benchmarks, Practical issues: Proposals, grant money, sponsors, prizes, dissemination, publicity**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04452v1)  

---


**ABSTRACT**  
This chapter provides a comprehensive overview of the pragmatic aspects involved in organizing AI competitions. We begin by discussing strategies to incentivize participation, touching upon effective communication techniques, aligning with trending topics in the field, structuring awards, potential recruitment opportunities, and more. We then shift to the essence of community engagement, and into organizational best practices and effective means of disseminating challenge outputs. Lastly, the chapter addresses the logistics, exposing on costs, required manpower, and resource allocation for effectively managing and executing a challenge. By examining these practical problems, readers will gain actionable insights to navigate the multifaceted landscape of AI competition organization, from inception to completion.

{{</citation>}}


### (39/93) The Role of Higher-Order Cognitive Models in Active Learning (Oskar Keurulainen et al., 2024)

{{<citation>}}

Oskar Keurulainen, Gokhan Alcan, Ville Kyrki. (2024)  
**The Role of Higher-Order Cognitive Models in Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2401.04397v1)  

---


**ABSTRACT**  
Building machines capable of efficiently collaborating with humans has been a longstanding goal in artificial intelligence. Especially in the presence of uncertainties, optimal cooperation often requires that humans and artificial agents model each other's behavior and use these models to infer underlying goals, beliefs or intentions, potentially involving multiple levels of recursion. Empirical evidence for such higher-order cognition in human behavior is also provided by previous works in cognitive science, linguistics, and robotics. We advocate for a new paradigm for active learning for human feedback that utilises humans as active data sources while accounting for their higher levels of agency. In particular, we discuss how increasing level of agency results in qualitatively different forms of rational communication between an active learning system and a teacher. Additionally, we provide a practical example of active learning using a higher-order cognitive model. This is accompanied by a computational study that underscores the unique behaviors that this model produces.

{{</citation>}}


### (40/93) Air Quality Forecasting Using Machine Learning: A Global perspective with Relevance to Low-Resource Settings (Mulomba Mukendi Christian et al., 2024)

{{<citation>}}

Mulomba Mukendi Christian, Hyebong Choi. (2024)  
**Air Quality Forecasting Using Machine Learning: A Global perspective with Relevance to Low-Resource Settings**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Low-Resource  
[Paper Link](http://arxiv.org/abs/2401.04369v1)  

---


**ABSTRACT**  
Air pollution stands as the fourth leading cause of death globally. While extensive research has been conducted in this domain, most approaches rely on large datasets when it comes to prediction. This limits their applicability in low-resource settings though more vulnerable. This study addresses this gap by proposing a novel machine learning approach for accurate air quality prediction using two months of air quality data. By leveraging the World Weather Repository, the meteorological, air pollutant, and Air Quality Index features from 197 capital cities were considered to predict air quality for the next day. The evaluation of several machine learning models demonstrates the effectiveness of the Random Forest algorithm in generating reliable predictions, particularly when applied to classification rather than regression, approach which enhances the model's generalizability by 42%, achieving a cross-validation score of 0.38 for regression and 0.89 for classification. To instill confidence in the predictions, interpretable machine learning was considered. Finally, a cost estimation comparing the implementation of this solution in high-resource and low-resource settings is presented including a tentative of technology licensing business model. This research highlights the potential for resource-limited countries to independently predict air quality while awaiting larger datasets to further refine their predictions.

{{</citation>}}


### (41/93) A Change Point Detection Integrated Remaining Useful Life Estimation Model under Variable Operating Conditions (Anushiya Arunan et al., 2024)

{{<citation>}}

Anushiya Arunan, Yan Qin, Xiaoli Li, Chau Yuen. (2024)  
**A Change Point Detection Integrated Remaining Useful Life Estimation Model under Variable Operating Conditions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.04351v1)  

---


**ABSTRACT**  
By informing the onset of the degradation process, health status evaluation serves as a significant preliminary step for reliable remaining useful life (RUL) estimation of complex equipment. This paper proposes a novel temporal dynamics learning-based model for detecting change points of individual devices, even under variable operating conditions, and utilises the learnt change points to improve the RUL estimation accuracy. During offline model development, the multivariate sensor data are decomposed to learn fused temporal correlation features that are generalisable and representative of normal operation dynamics across multiple operating conditions. Monitoring statistics and control limit thresholds for normal behaviour are dynamically constructed from these learnt temporal features for the unsupervised detection of device-level change points. The detected change points then inform the degradation data labelling for training a long short-term memory (LSTM)-based RUL estimation model. During online monitoring, the temporal correlation dynamics of a query device is monitored for breach of the control limit derived in offline training. If a change point is detected, the device's RUL is estimated with the well-trained offline model for early preventive action. Using C-MAPSS turbofan engines as the case study, the proposed method improved the accuracy by 5.6\% and 7.5\% for two scenarios with six operating conditions, when compared to existing LSTM-based RUL estimation models that do not consider heterogeneous change points.

{{</citation>}}


### (42/93) Private Fine-tuning of Large Language Models with Zeroth-order Optimization (Xinyu Tang et al., 2024)

{{<citation>}}

Xinyu Tang, Ashwinee Panda, Milad Nasr, Saeed Mahloujifar, Prateek Mittal. (2024)  
**Private Fine-tuning of Large Language Models with Zeroth-order Optimization**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04343v1)  

---


**ABSTRACT**  
Fine-tuning large pretrained models on private datasets may run the risk of violating privacy. Differential privacy is a framework for mitigating privacy risks by enforcing algorithmic stability. DP-SGD enables training models with private data in a privacy-preserving manner, but raises new obstacles in the form of performance loss and significant engineering challenges. We introduce DP-ZO, a new method for fine-tuning large language models that preserves the privacy of training data by privatizing zeroth-order optimization. A key insight into the design of our method is that the direction of the gradient in SPSA, the zeroth-order algorithm we use, is always random and the only information that depends on private data is the step size, i.e., a scalar. Therefore, we only need to privatize the scalar step size, which is memory-efficient. DP-ZO, which can be instantiated with either Laplace or Gaussian noise, provides a strong privacy-utility trade-off across different tasks, and model sizes, under conservative privacy budgets. One noteworthy result is that DP-ZO exhibits just $1.86\%$ performance degradation due to privacy at $(1,10^{-5})$-DP when fine-tuning OPT-66B on 1000 training samples from SQuAD.

{{</citation>}}


### (43/93) Deep Efficient Private Neighbor Generation for Subgraph Federated Learning (Ke Zhang et al., 2024)

{{<citation>}}

Ke Zhang, Lichao Sun, Bolin Ding, Siu Ming Yiu, Carl Yang. (2024)  
**Deep Efficient Private Neighbor Generation for Subgraph Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.04336v2)  

---


**ABSTRACT**  
Behemoth graphs are often fragmented and separately stored by multiple data owners as distributed subgraphs in many realistic applications. Without harming data privacy, it is natural to consider the subgraph federated learning (subgraph FL) scenario, where each local client holds a subgraph of the entire global graph, to obtain globally generalized graph mining models. To overcome the unique challenge of incomplete information propagation on local subgraphs due to missing cross-subgraph neighbors, previous works resort to the augmentation of local neighborhoods through the joint FL of missing neighbor generators and GNNs. Yet their technical designs have profound limitations regarding the utility, efficiency, and privacy goals of FL. In this work, we propose FedDEP to comprehensively tackle these challenges in subgraph FL. FedDEP consists of a series of novel technical designs: (1) Deep neighbor generation through leveraging the GNN embeddings of potential missing neighbors; (2) Efficient pseudo-FL for neighbor generation through embedding prototyping; and (3) Privacy protection through noise-less edge-local-differential-privacy. We analyze the correctness and efficiency of FedDEP, and provide theoretical guarantees on its privacy. Empirical results on four real-world datasets justify the clear benefits of proposed techniques.

{{</citation>}}


### (44/93) Coupling Graph Neural Networks with Fractional Order Continuous Dynamics: A Robustness Study (Qiyu Kang et al., 2024)

{{<citation>}}

Qiyu Kang, Kai Zhao, Yang Song, Yihang Xie, Yanan Zhao, Sijie Wang, Rui She, Wee Peng Tay. (2024)  
**Coupling Graph Neural Networks with Fractional Order Continuous Dynamics: A Robustness Study**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.04331v1)  

---


**ABSTRACT**  
In this work, we rigorously investigate the robustness of graph neural fractional-order differential equation (FDE) models. This framework extends beyond traditional graph neural (integer-order) ordinary differential equation (ODE) models by implementing the time-fractional Caputo derivative. Utilizing fractional calculus allows our model to consider long-term memory during the feature updating process, diverging from the memoryless Markovian updates seen in traditional graph neural ODE models. The superiority of graph neural FDE models over graph neural ODE models has been established in environments free from attacks or perturbations. While traditional graph neural ODE models have been verified to possess a degree of stability and resilience in the presence of adversarial attacks in existing literature, the robustness of graph neural FDE models, especially under adversarial conditions, remains largely unexplored. This paper undertakes a detailed assessment of the robustness of graph neural FDE models. We establish a theoretical foundation outlining the robustness characteristics of graph neural FDE models, highlighting that they maintain more stringent output perturbation bounds in the face of input and graph topology disturbances, compared to their integer-order counterparts. Our empirical evaluations further confirm the enhanced robustness of graph neural FDE models, highlighting their potential in adversarially robust applications.

{{</citation>}}


### (45/93) Advancing Deep Active Learning & Data Subset Selection: Unifying Principles with Information-Theory Intuitions (Andreas Kirsch, 2024)

{{<citation>}}

Andreas Kirsch. (2024)  
**Advancing Deep Active Learning & Data Subset Selection: Unifying Principles with Information-Theory Intuitions**  

---
Primary Category: cs.LG  
Categories: cs-IT, cs-LG, cs.LG, math-IT  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2401.04305v1)  

---


**ABSTRACT**  
At its core, this thesis aims to enhance the practicality of deep learning by improving the label and training efficiency of deep learning models. To this end, we investigate data subset selection techniques, specifically active learning and active sampling, grounded in information-theoretic principles. Active learning improves label efficiency, while active sampling enhances training efficiency. Supervised deep learning models often require extensive training with labeled data. Label acquisition can be expensive and time-consuming, and training large models is resource-intensive, hindering the adoption outside academic research and ``big tech.'' Existing methods for data subset selection in deep learning often rely on heuristics or lack a principled information-theoretic foundation. In contrast, this thesis examines several objectives for data subset selection and their applications within deep learning, striving for a more principled approach inspired by information theory. We begin by disentangling epistemic and aleatoric uncertainty in single forward-pass deep neural networks, which provides helpful intuitions and insights into different forms of uncertainty and their relevance for data subset selection. We then propose and investigate various approaches for active learning and data subset selection in (Bayesian) deep learning. Finally, we relate various existing and proposed approaches to approximations of information quantities in weight or prediction space. Underpinning this work is a principled and practical notation for information-theoretic quantities that includes both random variables and observed outcomes. This thesis demonstrates the benefits of working from a unified perspective and highlights the potential impact of our contributions to the practical application of deep learning.

{{</citation>}}


### (46/93) Setting the Record Straight on Transformer Oversmoothing (Gbètondji J-S Dovonon et al., 2024)

{{<citation>}}

Gbètondji J-S Dovonon, Michael M. Bronstein, Matt J. Kusner. (2024)  
**Setting the Record Straight on Transformer Oversmoothing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.04301v1)  

---


**ABSTRACT**  
Transformer-based models have recently become wildly successful across a diverse set of domains. At the same time, recent work has shown that Transformers are inherently low-pass filters that gradually oversmooth the inputs, reducing the expressivity of their representations. A natural question is: How can Transformers achieve these successes given this shortcoming? In this work we show that in fact Transformers are not inherently low-pass filters. Instead, whether Transformers oversmooth or not depends on the eigenspectrum of their update equations. Our analysis extends prior work in oversmoothing and in the closely-related phenomenon of rank collapse. We show that many successful Transformer models have attention and weights which satisfy conditions that avoid oversmoothing. Based on this analysis, we derive a simple way to parameterize the weights of the Transformer update equations that allows for control over its spectrum, ensuring that oversmoothing does not occur. Compared to a recent solution for oversmoothing, our approach improves generalization, even when training with more layers, fewer datapoints, and data that is corrupted.

{{</citation>}}


## cs.CR (1)



### (47/93) Phishing Website Detection through Multi-Model Analysis of HTML Content (Furkan Çolhak et al., 2024)

{{<citation>}}

Furkan Çolhak, Mert İlhan Ecevit, Bilal Emir Uçar, Reiner Creutzburg, Hasan Dağ. (2024)  
**Phishing Website Detection through Multi-Model Analysis of HTML Content**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: BERT, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.04820v1)  

---


**ABSTRACT**  
The way we communicate and work has changed significantly with the rise of the Internet. While it has opened up new opportunities, it has also brought about an increase in cyber threats. One common and serious threat is phishing, where cybercriminals employ deceptive methods to steal sensitive information.This study addresses the pressing issue of phishing by introducing an advanced detection model that meticulously focuses on HTML content. Our proposed approach integrates a specialized Multi-Layer Perceptron (MLP) model for structured tabular data and two pretrained Natural Language Processing (NLP) models for analyzing textual features such as page titles and content. The embeddings from these models are harmoniously combined through a novel fusion process. The resulting fused embeddings are then input into a linear classifier. Recognizing the scarcity of recent datasets for comprehensive phishing research, our contribution extends to the creation of an up-to-date dataset, which we openly share with the community. The dataset is meticulously curated to reflect real-life phishing conditions, ensuring relevance and applicability. The research findings highlight the effectiveness of the proposed approach, with the CANINE demonstrating superior performance in analyzing page titles and the RoBERTa excelling in evaluating page content. The fusion of two NLP and one MLP model,termed MultiText-LP, achieves impressive results, yielding a 96.80 F1 score and a 97.18 accuracy score on our research dataset. Furthermore, our approach outperforms existing methods on the CatchPhish HTML dataset, showcasing its efficacies.

{{</citation>}}


## cs.HC (3)



### (48/93) On the Effect of Contextual Information on Human Delegation Behavior in Human-AI collaboration (Philipp Spitzer et al., 2024)

{{<citation>}}

Philipp Spitzer, Joshua Holstein, Patrick Hemmer, Michael Vössing, Niklas Kühl, Dominik Martin, Gerhard Satzger. (2024)  
**On the Effect of Contextual Information on Human Delegation Behavior in Human-AI collaboration**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-LG, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04729v1)  

---


**ABSTRACT**  
The constantly increasing capabilities of artificial intelligence (AI) open new possibilities for human-AI collaboration. One promising approach to leverage existing complementary capabilities is allowing humans to delegate individual instances to the AI. However, enabling humans to delegate instances effectively requires them to assess both their own and the AI's capabilities in the context of the given task. In this work, we explore the effects of providing contextual information on human decisions to delegate instances to an AI. We find that providing participants with contextual information significantly improves the human-AI team performance. Additionally, we show that the delegation behavior changes significantly when participants receive varying types of contextual information. Overall, this research advances the understanding of human-AI interaction in human delegation and provides actionable insights for designing more effective collaborative systems.

{{</citation>}}


### (49/93) Imagining Computing Education Assessment after Generative AI (Stephen MacNeil et al., 2024)

{{<citation>}}

Stephen MacNeil, Scott Spurlock, Ian Applebaum. (2024)  
**Imagining Computing Education Assessment after Generative AI**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.04601v1)  

---


**ABSTRACT**  
In the contemporary landscape of computing education, the ubiquity of Generative Artificial Intelligence has significantly disrupted traditional assessment methods, rendering them obsolete and prompting educators to seek innovative alternatives. This research paper explores the challenges posed by Generative AI in the assessment domain and the persistent attempts to circumvent its impact. Despite various efforts to devise workarounds, the academic community is yet to find a comprehensive solution. Amidst this struggle, ungrading emerges as a potential yet under-appreciated solution to the assessment dilemma. Ungrading, a pedagogical approach that involves moving away from traditional grading systems, has faced resistance due to its perceived complexity and the reluctance of educators to depart from conventional assessment practices. However, as the inadequacies of current assessment methods become increasingly evident in the face of Generative AI, the time is ripe to reconsider and embrace ungrading.

{{</citation>}}


### (50/93) Healthcare Voice AI Assistants: Factors Influencing Trust and Intention to Use (Xiao Zhan et al., 2024)

{{<citation>}}

Xiao Zhan, Noura Abdi, William Seymour, Jose Such. (2024)  
**Healthcare Voice AI Assistants: Factors Influencing Trust and Intention to Use**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2401.04543v2)  

---


**ABSTRACT**  
AI assistants such as Alexa, Google Assistant, and Siri, are making their way into the healthcare sector, offering a convenient way for users to access different healthcare services. Trust is a vital factor in the uptake of healthcare services, but the factors affecting trust in voice assistants used for healthcare are under-explored and this specialist domain introduces additional requirements. This study explores the effects of different functional, personal, and risk factors on trust in and adoption of healthcare voice AI assistants (HVAs), generating a partial least squares structural model from a survey of 300 voice assistant users. Our results indicate that trust in HVAs can be significantly explained by functional factors (usefulness, content credibility, quality of service relative to a healthcare professional), together with security, and privacy risks and personal stance in technology. We also discuss differences in terms of trust between HVAs and general-purpose voice assistants as well as implications that are unique to HVAs.

{{</citation>}}


## cs.CV (21)



### (51/93) Revisiting Adversarial Training at Scale (Zeyu Wang et al., 2024)

{{<citation>}}

Zeyu Wang, Xianhang Li, Hongru Zhu, Cihang Xie. (2024)  
**Revisiting Adversarial Training at Scale**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Training, ImageNet  
[Paper Link](http://arxiv.org/abs/2401.04727v1)  

---


**ABSTRACT**  
The machine learning community has witnessed a drastic change in the training pipeline, pivoted by those ''foundation models'' with unprecedented scales. However, the field of adversarial training is lagging behind, predominantly centered around small model sizes like ResNet-50, and tiny and low-resolution datasets like CIFAR-10. To bridge this transformation gap, this paper provides a modern re-examination with adversarial training, investigating its potential benefits when applied at scale. Additionally, we introduce an efficient and effective training strategy to enable adversarial training with giant models and web-scale data at an affordable computing cost. We denote this newly introduced framework as AdvXL.   Empirical results demonstrate that AdvXL establishes new state-of-the-art robust accuracy records under AutoAttack on ImageNet-1K. For example, by training on DataComp-1B dataset, our AdvXL empowers a vanilla ViT-g model to substantially surpass the previous records of $l_{\infty}$-, $l_{2}$-, and $l_{1}$-robust accuracy by margins of 11.4%, 14.2% and 12.9%, respectively. This achievement posits AdvXL as a pioneering approach, charting a new trajectory for the efficient training of robust visual representations at significantly larger scales. Our code is available at https://github.com/UCSC-VLAA/AdvXL.

{{</citation>}}


### (52/93) Low-Resource Vision Challenges for Foundation Models (Yunhua Zhang et al., 2024)

{{<citation>}}

Yunhua Zhang, Hazel Doughty, Cees G. M. Snoek. (2024)  
**Low-Resource Vision Challenges for Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Low-Resource  
[Paper Link](http://arxiv.org/abs/2401.04716v2)  

---


**ABSTRACT**  
Low-resource settings are well-established in natural language processing, where many languages lack sufficient data for machine learning at scale. However, low-resource problems are under-explored in computer vision. In this paper, we strive to address this gap and explore the challenges of low-resource image tasks with vision foundation models. Thus, we first collect a benchmark of genuinely low-resource image data, covering historic maps, circuit diagrams, and mechanical drawings. These low-resource settings all share the three challenges of data scarcity, fine-grained differences, and the distribution shift from natural images to the specialized domain of interest. While existing foundation models have shown impressive generalizability, we find they cannot transfer well to our low-resource tasks. To begin to tackle the challenges of low-resource vision, we introduce one simple baseline per challenge. Specifically, we propose to i) enlarge the data space by generative models, ii) adopt the best sub-kernels to encode local regions for fine-grained difference discovery and iii) learn attention for specialized domains. Experiments on the three low-resource data sources in our benchmark demonstrate our proposals already provide a better baseline than common transfer learning, data augmentation, and fine-grained methods. This highlights the unique characteristics and challenges of low-resource vision for foundation models that warrant further investigation. Project website: https://xiaobai1217.github.io/Low-Resource-Vision/.

{{</citation>}}


### (53/93) Advancing Ante-Hoc Explainable Models through Generative Adversarial Networks (Tanmay Garg et al., 2024)

{{<citation>}}

Tanmay Garg, Deepika Vemuri, Vineeth N Balasubramanian. (2024)  
**Advancing Ante-Hoc Explainable Models through Generative Adversarial Networks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04647v1)  

---


**ABSTRACT**  
This paper presents a novel concept learning framework for enhancing model interpretability and performance in visual classification tasks. Our approach appends an unsupervised explanation generator to the primary classifier network and makes use of adversarial training. During training, the explanation module is optimized to extract visual concepts from the classifier's latent representations, while the GAN-based module aims to discriminate images generated from concepts, from true images. This joint training scheme enables the model to implicitly align its internally learned concepts with human-interpretable visual properties. Comprehensive experiments demonstrate the robustness of our approach, while producing coherent concept activations. We analyse the learned concepts, showing their semantic concordance with object parts and visual attributes. We also study how perturbations in the adversarial training protocol impact both classification and concept acquisition. In summary, this work presents a significant step towards building inherently interpretable deep vision models with task-aligned concept representations - a key enabler for developing trustworthy AI for real-world perception tasks.

{{</citation>}}


### (54/93) Generic Knowledge Boosted Pre-training For Remote Sensing Images (Ziyue Huang et al., 2024)

{{<citation>}}

Ziyue Huang, Mingming Zhang, Yuan Gong, Qingjie Liu, Yunhong Wang. (2024)  
**Generic Knowledge Boosted Pre-training For Remote Sensing Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.04614v1)  

---


**ABSTRACT**  
Deep learning models are essential for scene classification, change detection, land cover segmentation, and other remote sensing image understanding tasks. Most backbones of existing remote sensing deep learning models are typically initialized by pre-trained weights obtained from ImageNet pre-training (IMP). However, domain gaps exist between remote sensing images and natural images (e.g., ImageNet), making deep learning models initialized by pre-trained weights of IMP perform poorly for remote sensing image understanding. Although some pre-training methods are studied in the remote sensing community, current remote sensing pre-training methods face the problem of vague generalization by only using remote sensing images. In this paper, we propose a novel remote sensing pre-training framework, Generic Knowledge Boosted Remote Sensing Pre-training (GeRSP), to learn robust representations from remote sensing and natural images for remote sensing understanding tasks. GeRSP contains two pre-training branches: (1) A self-supervised pre-training branch is adopted to learn domain-related representations from unlabeled remote sensing images. (2) A supervised pre-training branch is integrated into GeRSP for general knowledge learning from labeled natural images. Moreover, GeRSP combines two pre-training branches using a teacher-student architecture to simultaneously learn representations with general and special knowledge, which generates a powerful pre-trained model for deep learning model initialization. Finally, we evaluate GeRSP and other remote sensing pre-training methods on three downstream tasks, i.e., object detection, semantic segmentation, and scene classification. The extensive experimental results consistently demonstrate that GeRSP can effectively learn robust representations in a unified manner, improving the performance of remote sensing downstream tasks.

{{</citation>}}


### (55/93) Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models (Xuewen Liu et al., 2024)

{{<citation>}}

Xuewen Liu, Zhikai Li, Junrui Xiao, Qingyi Gu. (2024)  
**Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2401.04585v1)  

---


**ABSTRACT**  
Diffusion models have achieved great success in image generation tasks through iterative noise estimation. However, the heavy denoising process and complex neural networks hinder their low-latency applications in real-world scenarios. Quantization can effectively reduce model complexity, and post-training quantization (PTQ), which does not require fine-tuning, is highly promising in accelerating the denoising process. Unfortunately, we find that due to the highly dynamic distribution of activations in different denoising steps, existing PTQ methods for diffusion models suffer from distribution mismatch issues at both calibration sample level and reconstruction output level, which makes the performance far from satisfactory, especially in low-bit cases. In this paper, we propose Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models (EDA-DM) to address the above issues. Specifically, at the calibration sample level, we select calibration samples based on the density and diversity in the latent space, thus facilitating the alignment of their distribution with the overall samples; and at the reconstruction output level, we propose Fine-grained Block Reconstruction, which can align the outputs of the quantized model and the full-precision model at different network granularity. Extensive experiments demonstrate that EDA-DM outperforms the existing post-training quantization frameworks in both unconditional and conditional generation scenarios. At low-bit precision, the quantized models with our method even outperform the full-precision models on most datasets.

{{</citation>}}


### (56/93) Effective pruning of web-scale datasets based on complexity of concept clusters (Amro Abbas et al., 2024)

{{<citation>}}

Amro Abbas, Evgenia Rusak, Kushal Tirumala, Wieland Brendel, Kamalika Chaudhuri, Ari S. Morcos. (2024)  
**Effective pruning of web-scale datasets based on complexity of concept clusters**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2401.04578v1)  

---


**ABSTRACT**  
Utilizing massive web-scale datasets has led to unprecedented performance gains in machine learning models, but also imposes outlandish compute requirements for their training. In order to improve training and data efficiency, we here push the limits of pruning large-scale multimodal datasets for training CLIP-style models. Today's most effective pruning method on ImageNet clusters data samples into separate concepts according to their embedding and prunes away the most prototypical samples. We scale this approach to LAION and improve it by noting that the pruning rate should be concept-specific and adapted to the complexity of the concept. Using a simple and intuitive complexity measure, we are able to reduce the training cost to a quarter of regular training. By filtering from the LAION dataset, we find that training on a smaller set of high-quality data can lead to higher performance with significantly lower training costs. More specifically, we are able to outperform the LAION-trained OpenCLIP-ViT-B32 model on ImageNet zero-shot accuracy by 1.1p.p. while only using 27.7% of the data and training compute. Despite a strong reduction in training cost, we also see improvements on ImageNet dist. shifts, retrieval tasks and VTAB. On the DataComp Medium benchmark, we achieve a new state-of-the-art ImageNet zero-shot accuracy and a competitive average zero-shot accuracy on 38 evaluation tasks.

{{</citation>}}


### (57/93) WaveletFormerNet: A Transformer-based Wavelet Network for Real-world Non-homogeneous and Dense Fog Removal (Shengli Zhang et al., 2024)

{{<citation>}}

Shengli Zhang, Zhiyong Tao, Sen Lin. (2024)  
**WaveletFormerNet: A Transformer-based Wavelet Network for Real-world Non-homogeneous and Dense Fog Removal**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.04550v1)  

---


**ABSTRACT**  
Although deep convolutional neural networks have achieved remarkable success in removing synthetic fog, it is essential to be able to process images taken in complex foggy conditions, such as dense or non-homogeneous fog, in the real world. However, the haze distribution in the real world is complex, and downsampling can lead to color distortion or loss of detail in the output results as the resolution of a feature map or image resolution decreases. In addition to the challenges of obtaining sufficient training data, overfitting can also arise in deep learning techniques for foggy image processing, which can limit the generalization abilities of the model, posing challenges for its practical applications in real-world scenarios. Considering these issues, this paper proposes a Transformer-based wavelet network (WaveletFormerNet) for real-world foggy image recovery. We embed the discrete wavelet transform into the Vision Transformer by proposing the WaveletFormer and IWaveletFormer blocks, aiming to alleviate texture detail loss and color distortion in the image due to downsampling. We introduce parallel convolution in the Transformer block, which allows for the capture of multi-frequency information in a lightweight mechanism. Additionally, we have implemented a feature aggregation module (FAM) to maintain image resolution and enhance the feature extraction capacity of our model, further contributing to its impressive performance in real-world foggy image recovery tasks. Extensive experiments demonstrate that our WaveletFormerNet performs better than state-of-the-art methods, as shown through quantitative and qualitative evaluations of minor model complexity. Additionally, our satisfactory results on real-world dust removal and application tests showcase the superior generalization ability and improved performance of WaveletFormerNet in computer vision-related applications.

{{</citation>}}


### (58/93) DedustNet: A Frequency-dominated Swin Transformer-based Wavelet Network for Agricultural Dust Removal (Shengli Zhang et al., 2024)

{{<citation>}}

Shengli Zhang, Zhiyong Tao, Sen Lin. (2024)  
**DedustNet: A Frequency-dominated Swin Transformer-based Wavelet Network for Agricultural Dust Removal**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.04750v1)  

---


**ABSTRACT**  
While dust significantly affects the environmental perception of automated agricultural machines, the existing deep learning-based methods for dust removal require further research and improvement in this area to improve the performance and reliability of automated agricultural machines in agriculture. We propose an end-to-end trainable learning network (DedustNet) to solve the real-world agricultural dust removal task. To our knowledge, DedustNet is the first time Swin Transformer-based units have been used in wavelet networks for agricultural image dusting. Specifically, we present the frequency-dominated block (DWTFormer block and IDWTFormer block) by adding a spatial features aggregation scheme (SFAS) to the Swin Transformer and combining it with the wavelet transform, the DWTFormer block and IDWTFormer block, alleviating the limitation of the global receptive field of Swin Transformer when dealing with complex dusty backgrounds. Furthermore, We propose a cross-level information fusion module to fuse different levels of features and effectively capture global and long-range feature relationships. In addition, we present a dilated convolution module to capture contextual information guided by wavelet transform at multiple scales, which combines the advantages of wavelet transform and dilated convolution. Our algorithm leverages deep learning techniques to effectively remove dust from images while preserving the original structural and textural features. Compared to existing state-of-the-art methods, DedustNet achieves superior performance and more reliable results in agricultural image dedusting, providing strong support for the application of agricultural machinery in dusty environments. Additionally, the impressive performance on real-world hazy datasets and application tests highlights DedustNet superior generalization ability and computer vision-related application performance.

{{</citation>}}


### (59/93) Convolutional Neural Network Ensemble Learning for Hyperspectral Imaging-based Blackberry Fruit Ripeness Detection in Uncontrolled Farm Environment (Chollette C. Olisah et al., 2024)

{{<citation>}}

Chollette C. Olisah, Ben Trewhella, Bo Li, Melvyn L. Smith, Benjamin Winstone, E. Charles Whitfield, Felicidad Fernández Fernández, Harriet Duncalfe. (2024)  
**Convolutional Neural Network Ensemble Learning for Hyperspectral Imaging-based Blackberry Fruit Ripeness Detection in Uncontrolled Farm Environment**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.04748v1)  

---


**ABSTRACT**  
Fruit ripeness estimation models have for decades depended on spectral index features or colour-based features, such as mean, standard deviation, skewness, colour moments, and/or histograms for learning traits of fruit ripeness. Recently, few studies have explored the use of deep learning techniques to extract features from images of fruits with visible ripeness cues. However, the blackberry (Rubus fruticosus) fruit does not show obvious and reliable visible traits of ripeness when mature and therefore poses great difficulty to fruit pickers. The mature blackberry, to the human eye, is black before, during, and post-ripening. To address this engineering application challenge, this paper proposes a novel multi-input convolutional neural network (CNN) ensemble classifier for detecting subtle traits of ripeness in blackberry fruits. The multi-input CNN was created from a pre-trained visual geometry group 16-layer deep convolutional network (VGG16) model trained on the ImageNet dataset. The fully connected layers were optimized for learning traits of ripeness of mature blackberry fruits. The resulting model served as the base for building homogeneous ensemble learners that were ensemble using the stack generalization ensemble (SGE) framework. The input to the network is images acquired with a stereo sensor using visible and near-infrared (VIS-NIR) spectral filters at wavelengths of 700 nm and 770 nm. Through experiments, the proposed model achieved 95.1% accuracy on unseen sets and 90.2% accuracy with in-field conditions. Further experiments reveal that machine sensory is highly and positively correlated to human sensory over blackberry fruit skin texture.

{{</citation>}}


### (60/93) PhilEO Bench: Evaluating Geo-Spatial Foundation Models (Casper Fibaek et al., 2024)

{{<citation>}}

Casper Fibaek, Luke Camilleri, Andreas Luyts, Nikolaos Dionelis, Bertrand Le Saux. (2024)  
**PhilEO Bench: Evaluating Geo-Spatial Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.04464v1)  

---


**ABSTRACT**  
Massive amounts of unlabelled data are captured by Earth Observation (EO) satellites, with the Sentinel-2 constellation generating 1.6 TB of data daily. This makes Remote Sensing a data-rich domain well suited to Machine Learning (ML) solutions. However, a bottleneck in applying ML models to EO is the lack of annotated data as annotation is a labour-intensive and costly process. As a result, research in this domain has focused on Self-Supervised Learning and Foundation Model approaches. This paper addresses the need to evaluate different Foundation Models on a fair and uniform benchmark by introducing the PhilEO Bench, a novel evaluation framework for EO Foundation Models. The framework comprises of a testbed and a novel 400 GB Sentinel-2 dataset containing labels for three downstream tasks, building density estimation, road segmentation, and land cover classification. We present experiments using our framework evaluating different Foundation Models, including Prithvi and SatMAE, at multiple n-shots and convergence rates.

{{</citation>}}


### (61/93) D3AD: Dynamic Denoising Diffusion Probabilistic Model for Anomaly Detection (Justin Tebbe et al., 2024)

{{<citation>}}

Justin Tebbe, Jawad Tayyub. (2024)  
**D3AD: Dynamic Denoising Diffusion Probabilistic Model for Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.04463v1)  

---


**ABSTRACT**  
Diffusion models have found valuable applications in anomaly detection by capturing the nominal data distribution and identifying anomalies via reconstruction. Despite their merits, they struggle to localize anomalies of varying scales, especially larger anomalies like entire missing components. Addressing this, we present a novel framework that enhances the capability of diffusion models, by extending the previous introduced implicit conditioning approach Meng et al. (2022) in three significant ways. First, we incorporate a dynamic step size computation that allows for variable noising steps in the forward process guided by an initial anomaly prediction. Second, we demonstrate that denoising an only scaled input, without any added noise, outperforms conventional denoising process. Third, we project images in a latent space to abstract away from fine details that interfere with reconstruction of large missing components. Additionally, we propose a fine-tuning mechanism that facilitates the model to effectively grasp the nuances of the target domain. Our method undergoes rigorous evaluation on two prominent anomaly detection datasets VISA and BTAD, yielding state-of-the-art performance. Importantly, our framework effectively localizes anomalies regardless of their scale, marking a pivotal advancement in diffusion-based anomaly detection.

{{</citation>}}


### (62/93) Empirical Analysis of Anomaly Detection on Hyperspectral Imaging Using Dimension Reduction Methods (Dongeon Kim et al., 2024)

{{<citation>}}

Dongeon Kim, YeongHyeon Park. (2024)  
**Empirical Analysis of Anomaly Detection on Hyperspectral Imaging Using Dimension Reduction Methods**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.04437v1)  

---


**ABSTRACT**  
Recent studies try to use hyperspectral imaging (HSI) to detect foreign matters in products because it enables to visualize the invisible wavelengths including ultraviolet and infrared. Considering the enormous image channels of the HSI, several dimension reduction methods-e.g., PCA or UMAP-can be considered to reduce but those cannot ease the fundamental limitations, as follows: (1) latency of HSI capturing. (2) less explanation ability of the important channels. In this paper, to circumvent the aforementioned methods, one of the ways to channel reduction, on anomaly detection proposed HSI. Different from feature extraction methods (i.e., PCA or UMAP), feature selection can sort the feature by impact and show better explainability so we might redesign the task-optimized and cost-effective spectroscopic camera. Via the extensive experiment results with synthesized MVTec AD dataset, we confirm that the feature selection method shows 6.90x faster at the inference phase compared with feature extraction-based approaches while preserving anomaly detection performance. Ultimately, we conclude the advantage of feature selection which is effective yet fast.

{{</citation>}}


### (63/93) MapAI: Precision in Building Segmentation (Sander Riisøen Jyhne et al., 2024)

{{<citation>}}

Sander Riisøen Jyhne, Morten Goodwin, Per Arne Andersen, Ivar Oveland, Alexander Salveson Nossum, Karianne Ormseth, Mathilde Ørstavik, Andrew C. Flatman. (2024)  
**MapAI: Precision in Building Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04406v1)  

---


**ABSTRACT**  
MapAI: Precision in Building Segmentation is a competition arranged with the Norwegian Artificial Intelligence Research Consortium (NORA) in collaboration with Centre for Artificial Intelligence Research at the University of Agder (CAIR), the Norwegian Mapping Authority, AI:Hub, Norkart, and the Danish Agency for Data Supply and Infrastructure. The competition will be held in the fall of 2022. It will be concluded at the Northern Lights Deep Learning conference focusing on the segmentation of buildings using aerial images and laser data. We propose two different tasks to segment buildings, where the first task can only utilize aerial images, while the second must use laser data (LiDAR) with or without aerial images. Furthermore, we use IoU and Boundary IoU to properly evaluate the precision of the models, with the latter being an IoU measure that evaluates the results' boundaries. We provide the participants with a training dataset and keep a test dataset for evaluation.

{{</citation>}}


### (64/93) Content-Conditioned Generation of Stylized Free hand Sketches (Jiajun Liu et al., 2024)

{{<citation>}}

Jiajun Liu, Siyuan Wang, Guangming Zhu, Liang Zhang, Ning Li, Eryang Gao. (2024)  
**Content-Conditioned Generation of Stylized Free hand Sketches**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2401.04739v1)  

---


**ABSTRACT**  
In recent years, the recognition of free-hand sketches has remained a popular task. However, in some special fields such as the military field, free-hand sketches are difficult to sample on a large scale. Common data augmentation and image generation techniques are difficult to produce images with various free-hand sketching styles. Therefore, the recognition and segmentation tasks in related fields are limited. In this paper, we propose a novel adversarial generative network that can accurately generate realistic free-hand sketches with various styles. We explore the performance of the model, including using styles randomly sampled from a prior normal distribution to generate images with various free-hand sketching styles, disentangling the painters' styles from known free-hand sketches to generate images with specific styles, and generating images of unknown classes that are not in the training set. We further demonstrate with qualitative and quantitative evaluations our advantages in visual quality, content accuracy, and style imitation on SketchIME.

{{</citation>}}


### (65/93) Representative Feature Extraction During Diffusion Process for Sketch Extraction with One Example (Kwan Yun et al., 2024)

{{<citation>}}

Kwan Yun, Youngseo Kim, Kwanggyoon Seo, Chang Wook Seo, Junyong Noh. (2024)  
**Representative Feature Extraction During Diffusion Process for Sketch Extraction with One Example**  

---
Primary Category: cs.CV  
Categories: 68T01, I-4-9, cs-AI, cs-CV, cs-GR, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2401.04362v1)  

---


**ABSTRACT**  
We introduce DiffSketch, a method for generating a variety of stylized sketches from images. Our approach focuses on selecting representative features from the rich semantics of deep features within a pretrained diffusion model. This novel sketch generation method can be trained with one manual drawing. Furthermore, efficient sketch extraction is ensured by distilling a trained generator into a streamlined extractor. We select denoising diffusion features through analysis and integrate these selected features with VAE features to produce sketches. Additionally, we propose a sampling scheme for training models using a conditional generative approach. Through a series of comparisons, we verify that distilled DiffSketch not only outperforms existing state-of-the-art sketch extraction methods but also surpasses diffusion-based stylization methods in the task of extracting sketches.

{{</citation>}}


### (66/93) Iterative Feedback Network for Unsupervised Point Cloud Registration (Yifan Xie et al., 2024)

{{<citation>}}

Yifan Xie, Boyu Wang, Shiqi Li, Jihua Zhu. (2024)  
**Iterative Feedback Network for Unsupervised Point Cloud Registration**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.04357v1)  

---


**ABSTRACT**  
As a fundamental problem in computer vision, point cloud registration aims to seek the optimal transformation for aligning a pair of point clouds. In most existing methods, the information flows are usually forward transferring, thus lacking the guidance from high-level information to low-level information. Besides, excessive high-level information may be overly redundant, and directly using it may conflict with the original low-level information. In this paper, we propose a novel Iterative Feedback Network (IFNet) for unsupervised point cloud registration, in which the representation of low-level features is efficiently enriched by rerouting subsequent high-level features. Specifically, our IFNet is built upon a series of Feedback Registration Block (FRB) modules, with each module responsible for generating the feedforward rigid transformation and feedback high-level features. These FRB modules are cascaded and recurrently unfolded over time. Further, the Feedback Transformer is designed to efficiently select relevant information from feedback high-level features, which is utilized to refine the low-level features. What's more, we incorporate a geometry-awareness descriptor to empower the network for making full use of most geometric information, which leads to more precise registration results. Extensive experiments on various benchmark datasets demonstrate the superior registration performance of our IFNet.

{{</citation>}}


### (67/93) Knowledge-enhanced Multi-perspective Video Representation Learning for Scene Recognition (Xuzheng Yu et al., 2024)

{{<citation>}}

Xuzheng Yu, Chen Jiang, Wei Zhang, Tian Gan, Linlin Chao, Jianan Zhao, Yuan Cheng, Qingpei Guo, Wei Chu. (2024)  
**Knowledge-enhanced Multi-perspective Video Representation Learning for Scene Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.04354v1)  

---


**ABSTRACT**  
With the explosive growth of video data in real-world applications, a comprehensive representation of videos becomes increasingly important. In this paper, we address the problem of video scene recognition, whose goal is to learn a high-level video representation to classify scenes in videos. Due to the diversity and complexity of video contents in realistic scenarios, this task remains a challenge. Most existing works identify scenes for videos only from visual or textual information in a temporal perspective, ignoring the valuable information hidden in single frames, while several earlier studies only recognize scenes for separate images in a non-temporal perspective. We argue that these two perspectives are both meaningful for this task and complementary to each other, meanwhile, externally introduced knowledge can also promote the comprehension of videos. We propose a novel two-stream framework to model video representations from multiple perspectives, i.e. temporal and non-temporal perspectives, and integrate the two perspectives in an end-to-end manner by self-distillation. Besides, we design a knowledge-enhanced feature fusion and label prediction method that contributes to naturally introducing knowledge into the task of video scene recognition. Experiments conducted on a real-world dataset demonstrate the effectiveness of our proposed method.

{{</citation>}}


### (68/93) Pre-trained Model Guided Fine-Tuning for Zero-Shot Adversarial Robustness (Sibo Wang et al., 2024)

{{<citation>}}

Sibo Wang, Jie Zhang, Zheng Yuan, Shiguang Shan. (2024)  
**Pre-trained Model Guided Fine-Tuning for Zero-Shot Adversarial Robustness**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.04350v1)  

---


**ABSTRACT**  
Large-scale pre-trained vision-language models like CLIP have demonstrated impressive performance across various tasks, and exhibit remarkable zero-shot generalization capability, while they are also vulnerable to imperceptible adversarial examples. Existing works typically employ adversarial training (fine-tuning) as a defense method against adversarial examples. However, direct application to the CLIP model may result in overfitting, compromising the model's capacity for generalization. In this paper, we propose Pre-trained Model Guided Adversarial Fine-Tuning (PMG-AFT) method, which leverages supervision from the original pre-trained model by carefully designing an auxiliary branch, to enhance the model's zero-shot adversarial robustness. Specifically, PMG-AFT minimizes the distance between the features of adversarial examples in the target model and those in the pre-trained model, aiming to preserve the generalization features already captured by the pre-trained model. Extensive Experiments on 15 zero-shot datasets demonstrate that PMG-AFT significantly outperforms the state-of-the-art method, improving the top-1 robust accuracy by an average of 4.99%. Furthermore, our approach consistently improves clean accuracy by an average of 8.72%.

{{</citation>}}


### (69/93) Memory-Efficient Personalization using Quantized Diffusion Model (Hyogon Ryu et al., 2024)

{{<citation>}}

Hyogon Ryu, Seohyun Lim, Hyunjung Shim. (2024)  
**Memory-Efficient Personalization using Quantized Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2401.04339v1)  

---


**ABSTRACT**  
The rise of billion-parameter diffusion models like Stable Diffusion XL, Imagen, and Dall-E3 markedly advances the field of generative AI. However, their large-scale nature poses challenges in fine-tuning and deployment due to high resource demands and slow inference speed. This paper ventures into the relatively unexplored yet promising realm of fine-tuning quantized diffusion models. We establish a strong baseline by customizing three models: PEQA for fine-tuning quantization parameters, Q-Diffusion for post-training quantization, and DreamBooth for personalization. Our analysis reveals a notable trade-off between subject and prompt fidelity within the baseline model. To address these issues, we introduce two strategies, inspired by the distinct roles of different timesteps in diffusion models: S1 optimizing a single set of fine-tuning parameters exclusively at selected intervals, and S2 creating multiple fine-tuning parameter sets, each specialized for different timestep intervals. Our approach not only enhances personalization but also upholds prompt fidelity and image quality, significantly outperforming the baseline qualitatively and quantitatively. The code will be made publicly available.

{{</citation>}}


### (70/93) Vision Reimagined: AI-Powered Breakthroughs in WiFi Indoor Imaging (Jianyang Shi et al., 2024)

{{<citation>}}

Jianyang Shi, Bowen Zhang, Amartansh Dubey, Ross Murch, Liwen Jing. (2024)  
**Vision Reimagined: AI-Powered Breakthroughs in WiFi Indoor Imaging**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04317v1)  

---


**ABSTRACT**  
Indoor imaging is a critical task for robotics and internet-of-things. WiFi as an omnipresent signal is a promising candidate for carrying out passive imaging and synchronizing the up-to-date information to all connected devices. This is the first research work to consider WiFi indoor imaging as a multi-modal image generation task that converts the measured WiFi power into a high-resolution indoor image. Our proposed WiFi-GEN network achieves a shape reconstruction accuracy that is 275% of that achieved by physical model-based inversion methods. Additionally, the Frechet Inception Distance score has been significantly reduced by 82%. To examine the effectiveness of models for this task, the first large-scale dataset is released containing 80,000 pairs of WiFi signal and imaging target. Our model absorbs challenges for the model-based methods including the non-linearity, ill-posedness and non-certainty into massive parameters of our generative AI network. The network is also designed to best fit measured WiFi signals and the desired imaging output. For reproducibility, we will release the data and code upon acceptance.

{{</citation>}}


### (71/93) StarCraftImage: A Dataset For Prototyping Spatial Reasoning Methods For Multi-Agent Environments (Sean Kulinski et al., 2024)

{{<citation>}}

Sean Kulinski, Nicholas R. Waytowich, James Z. Hare, David I. Inouye. (2024)  
**StarCraftImage: A Dataset For Prototyping Spatial Reasoning Methods For Multi-Agent Environments**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MA, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.04290v1)  

---


**ABSTRACT**  
Spatial reasoning tasks in multi-agent environments such as event prediction, agent type identification, or missing data imputation are important for multiple applications (e.g., autonomous surveillance over sensor networks and subtasks for reinforcement learning (RL)). StarCraft II game replays encode intelligent (and adversarial) multi-agent behavior and could provide a testbed for these tasks; however, extracting simple and standardized representations for prototyping these tasks is laborious and hinders reproducibility. In contrast, MNIST and CIFAR10, despite their extreme simplicity, have enabled rapid prototyping and reproducibility of ML methods. Following the simplicity of these datasets, we construct a benchmark spatial reasoning dataset based on StarCraft II replays that exhibit complex multi-agent behaviors, while still being as easy to use as MNIST and CIFAR10. Specifically, we carefully summarize a window of 255 consecutive game states to create 3.6 million summary images from 60,000 replays, including all relevant metadata such as game outcome and player races. We develop three formats of decreasing complexity: Hyperspectral images that include one channel for every unit type (similar to multispectral geospatial images), RGB images that mimic CIFAR10, and grayscale images that mimic MNIST. We show how this dataset can be used for prototyping spatial reasoning methods. All datasets, code for extraction, and code for dataset loading can be found at https://starcraftdata.davidinouye.com

{{</citation>}}


## eess.IV (2)



### (72/93) U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation (Jun Ma et al., 2024)

{{<citation>}}

Jun Ma, Feifei Li, Bo Wang. (2024)  
**U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.04722v1)  

---


**ABSTRACT**  
Convolutional Neural Networks (CNNs) and Transformers have been the most popular architectures for biomedical image segmentation, but both of them have limited ability to handle long-range dependencies because of inherent locality or computational complexity. To address this challenge, we introduce U-Mamba, a general-purpose network for biomedical image segmentation. Inspired by the State Space Sequence Models (SSMs), a new family of deep sequence models known for their strong capability in handling long sequences, we design a hybrid CNN-SSM block that integrates the local feature extraction power of convolutional layers with the abilities of SSMs for capturing the long-range dependency. Moreover, U-Mamba enjoys a self-configuring mechanism, allowing it to automatically adapt to various datasets without manual intervention. We conduct extensive experiments on four diverse tasks, including the 3D abdominal organ segmentation in CT and MR images, instrument segmentation in endoscopy images, and cell segmentation in microscopy images. The results reveal that U-Mamba outperforms state-of-the-art CNN-based and Transformer-based segmentation networks across all tasks. This opens new avenues for efficient long-range dependency modeling in biomedical image analysis. The code, models, and data are publicly available at https://wanglab.ai/u-mamba.html.

{{</citation>}}


### (73/93) Skin Cancer Segmentation and Classification Using Vision Transformer for Automatic Analysis in Dermatoscopy-based Non-invasive Digital System (Galib Muhammad Shahriar Himel et al., 2024)

{{<citation>}}

Galib Muhammad Shahriar Himel, Md. Masudul Islam, Kh Abdullah Al-Aff, Shams Ibne Karim, Md. Kabir Uddin Sikder. (2024)  
**Skin Cancer Segmentation and Classification Using Vision Transformer for Automatic Analysis in Dermatoscopy-based Non-invasive Digital System**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Google, Transformer  
[Paper Link](http://arxiv.org/abs/2401.04746v1)  

---


**ABSTRACT**  
Skin cancer is a global health concern, necessitating early and accurate diagnosis for improved patient outcomes. This study introduces a groundbreaking approach to skin cancer classification, employing the Vision Transformer, a state-of-the-art deep learning architecture renowned for its success in diverse image analysis tasks. Utilizing the HAM10000 dataset of 10,015 meticulously annotated skin lesion images, the model undergoes preprocessing for enhanced robustness. The Vision Transformer, adapted to the skin cancer classification task, leverages the self-attention mechanism to capture intricate spatial dependencies, achieving superior performance over traditional deep learning architectures. Segment Anything Model aids in precise segmentation of cancerous areas, attaining high IOU and Dice Coefficient. Extensive experiments highlight the model's supremacy, particularly the Google-based ViT patch-32 variant, which achieves 96.15% accuracy and showcases potential as an effective tool for dermatologists in skin cancer diagnosis, contributing to advancements in dermatological practices.

{{</citation>}}


## eess.SP (1)



### (74/93) Cuff-less Arterial Blood Pressure Waveform Synthesis from Single-site PPG using Transformer & Frequency-domain Learning (Muhammad Ahmad Tahir et al., 2024)

{{<citation>}}

Muhammad Ahmad Tahir, Ahsan Mehmood, Muhammad Mahboob Ur Rahman, Muhammad Wasim Nawaz, Kashif Riaz, Qammer H. Abbasi. (2024)  
**Cuff-less Arterial Blood Pressure Waveform Synthesis from Single-site PPG using Transformer & Frequency-domain Learning**  

---
Primary Category: eess.SP  
Categories: cs-IT, cs-LG, eess-SP, eess.SP, math-IT  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.05452v1)  

---


**ABSTRACT**  
We propose two novel purpose-built deep learning (DL) models for synthesis of the arterial blood pressure (ABP) waveform in a cuff-less manner, using a single-site photoplethysmography (PPG) signal. We utilize the public UCI dataset on cuff-less blood pressure (CLBP) estimation to train and evaluate our DL models. Firstly, we implement a transformer model that incorporates positional encoding, multi-head attention, layer normalization, and dropout techniques, and synthesizes the ABP waveform with a mean absolute error (MAE) of 14. Secondly, we implement a frequency-domain (FD) learning approach where we first obtain the discrete cosine transform (DCT) coefficients of the PPG and ABP signals corresponding to two cardiac cycles, and then learn a linear/non-linear (L/NL) regression between them. We learn that the FD L/NL regression model outperforms the transformer model by achieving an MAE of 11.87 and 8.01, for diastolic blood pressure (DBP) and systolic blood pressure (SBP), respectively. Our FD L/NL regression model also fulfills the AAMI criterion of utilizing data from more than 85 subjects, and achieves grade B by the BHS criterion.

{{</citation>}}


## math.OC (1)



### (75/93) Time-certified Input-constrained NMPC via Koopman Operator (Liang Wu et al., 2024)

{{<citation>}}

Liang Wu, Krystian Ganko, Richard D. Braatz. (2024)  
**Time-certified Input-constrained NMPC via Koopman Operator**  

---
Primary Category: math.OC  
Categories: cs-SY, eess-SY, math-OC, math.OC  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.04653v1)  

---


**ABSTRACT**  
Determining solving-time certificates of nonlinear model predictive control (NMPC) implementations is a pressing requirement when deploying NMPC in production environments. Such a certificate guarantees that the NMPC controller returns a solution before the next sampling time. However, NMPC formulations produce nonlinear programs (NLPs) for which it is very difficult to derive their solving-time certificates. Our previous work, Wu and Braatz (2023), challenged this limitation with a proposed input-constrained MPC algorithm having exact iteration complexity but was restricted to linear MPC formulations. This work extends the algorithm to solve input-constrained NMPC problems, by using the Koopman operator and a condensing MPC technique. We illustrate the algorithm performance on a high-dimensional, nonlinear partial differential equation (PDE) control case study, in which we theoretically and numerically certify the solving time to be less than the sampling time.

{{</citation>}}


## cs.SE (4)



### (76/93) Applying Large Language Models API to Issue Classification Problem (Gabriel Aracena et al., 2024)

{{<citation>}}

Gabriel Aracena, Kyle Luster, Fabio Santos, Igor Steinmacher, Marco A. Gerosa. (2024)  
**Applying Large Language Models API to Issue Classification Problem**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-LG, cs-SE, cs.SE  
Keywords: GPT, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.04637v1)  

---


**ABSTRACT**  
Effective prioritization of issue reports is crucial in software engineering to optimize resource allocation and address critical problems promptly. However, the manual classification of issue reports for prioritization is laborious and lacks scalability. Alternatively, many open source software (OSS) projects employ automated processes for this task, albeit relying on substantial datasets for adequate training. This research seeks to devise an automated approach that ensures reliability in issue prioritization, even when trained on smaller datasets. Our proposed methodology harnesses the power of Generative Pre-trained Transformers (GPT), recognizing their potential to efficiently handle this task. By leveraging the capabilities of such models, we aim to develop a robust system for prioritizing issue reports accurately, mitigating the necessity for extensive training data while maintaining reliability. In our research, we have developed a reliable GPT-based approach to accurately label and prioritize issue reports with a reduced training dataset. By reducing reliance on massive data requirements and focusing on few-shot fine-tuning, our methodology offers a more accessible and efficient solution for issue prioritization in software engineering. Our model predicted issue types in individual projects up to 93.2% in precision, 95% in recall, and 89.3% in F1-score.

{{</citation>}}


### (77/93) DebugBench: Evaluating Debugging Capability of Large Language Models (Runchu Tian et al., 2024)

{{<citation>}}

Runchu Tian, Yining Ye, Yujia Qin, Xin Cong, Yankai Lin, Yinxu Pan, Yesai Wu, Zhiyuan Liu, Maosong Sun. (2024)  
**DebugBench: Evaluating Debugging Capability of Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.04621v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated exceptional coding capability. However, as another critical component of programming proficiency, the debugging capability of LLMs remains relatively unexplored. Previous evaluations of LLMs' debugging ability are significantly limited by the risk of data leakage, the scale of the dataset, and the variety of tested bugs. To overcome these deficiencies, we introduce `DebugBench', an LLM debugging benchmark consisting of 4,253 instances. It covers four major bug categories and 18 minor types in C++, Java, and Python. To construct DebugBench, we collect code snippets from the LeetCode community, implant bugs into source data with GPT-4, and assure rigorous quality checks. We evaluate two commercial and three open-source models in a zero-shot scenario. We find that (1) while closed-source models like GPT-4 exhibit inferior debugging performance compared to humans, open-source models such as Code Llama fail to attain any pass rate scores; (2) the complexity of debugging notably fluctuates depending on the bug category; (3) incorporating runtime feedback has a clear impact on debugging performance which is not always helpful. As an extension, we also compare LLM debugging and code generation, revealing a strong correlation between them for closed-source models. These findings will benefit the development of LLMs in debugging.

{{</citation>}}


### (78/93) Rewriting the Code: A Simple Method for Large Language Model Augmented Code Search (Haochen Li et al., 2024)

{{<citation>}}

Haochen Li, Xin Zhou, Zhiqi Shen. (2024)  
**Rewriting the Code: A Simple Method for Large Language Model Augmented Code Search**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-IR, cs-LG, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04514v1)  

---


**ABSTRACT**  
In code search, the Generation-Augmented Retrieval (GAR) framework, which generates exemplar code snippets to augment queries, has emerged as a promising strategy to address the principal challenge of modality misalignment between code snippets and natural language queries, particularly with the demonstrated code generation capabilities of Large Language Models (LLMs). Nevertheless, our preliminary investigations indicate that the improvements conferred by such an LLM-augmented framework are somewhat constrained. This limitation could potentially be ascribed to the fact that the generated codes, albeit functionally accurate, frequently display a pronounced stylistic deviation from the ground truth code in the codebase. In this paper, we extend the foundational GAR framework and propose a simple yet effective method that additionally Rewrites the Code (ReCo) within the codebase for style normalization. Experimental results demonstrate that ReCo significantly boosts retrieval accuracy across sparse (up to 35.7%), zero-shot dense (up to 27.6%), and fine-tuned dense (up to 23.6%) retrieval settings in diverse search scenarios. To further elucidate the advantages of ReCo and stimulate research in code style normalization, we introduce Code Style Similarity, the first metric tailored to quantify stylistic similarities in code. Notably, our empirical findings reveal the inadequacy of existing metrics in capturing stylistic nuances.

{{</citation>}}


### (79/93) How Dataflow Diagrams Impact Software Security Analysis: an Empirical Experiment (Simon Schneider et al., 2024)

{{<citation>}}

Simon Schneider, Nicolás E. Díaz Ferreyra, Pierre-Jean Quéval, Georg Simhandl, Uwe Zdun, Riccardo Scandariato. (2024)  
**How Dataflow Diagrams Impact Software Security Analysis: an Empirical Experiment**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.04446v1)  

---


**ABSTRACT**  
Models of software systems are used throughout the software development lifecycle. Dataflow diagrams (DFDs), in particular, are well-established resources for security analysis. Many techniques, such as threat modelling, are based on DFDs of the analysed application. However, their impact on the performance of analysts in a security analysis setting has not been explored before. In this paper, we present the findings of an empirical experiment conducted to investigate this effect. Following a within-groups design, participants were asked to solve security-relevant tasks for a given microservice application. In the control condition, the participants had to examine the source code manually. In the model-supported condition, they were additionally provided a DFD of the analysed application and traceability information linking model items to artefacts in source code. We found that the participants (n = 24) performed significantly better in answering the analysis tasks correctly in the model-supported condition (41% increase in analysis correctness). Further, participants who reported using the provided traceability information performed better in giving evidence for their answers (315% increase in correctness of evidence). Finally, we identified three open challenges of using DFDs for security analysis based on the insights gained in the experiment.

{{</citation>}}


## cs.NE (2)



### (80/93) Hypercomplex neural network in time series forecasting of stock data (Radosław Kycia et al., 2024)

{{<citation>}}

Radosław Kycia, Agnieszka Niemczynowicz. (2024)  
**Hypercomplex neural network in time series forecasting of stock data**  

---
Primary Category: cs.NE  
Categories: cs-LG, cs-NE, cs.NE  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.04632v1)  

---


**ABSTRACT**  
The three classes of architectures for time series prediction were tested. They differ by input layers which contain either convolutional, LSTM, or dense hypercomplex layers for 4D algebras. The input was four related Stock Market time series, and the prediction of one of them is expected. The optimization of hyperparameters related to the classes of architectures was performed in order to compare the best neural networks within the class. The results show that in most cases, the architecture with a hypercomplex dense layer provides similar MAE accuracy to other architectures, however, with considerably less trainable parameters. Thanks to it, hypercomplex neural networks can be learned and process data faster than the other tested architectures. Moreover, the order of the input time series has an impact on effectively.

{{</citation>}}


### (81/93) Fully Spiking Actor Network with Intra-layer Connections for Reinforcement Learning (Ding Chen et al., 2024)

{{<citation>}}

Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian. (2024)  
**Fully Spiking Actor Network with Intra-layer Connections for Reinforcement Learning**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-LG, cs-NE, cs.NE  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.05444v1)  

---


**ABSTRACT**  
With the help of special neuromorphic hardware, spiking neural networks (SNNs) are expected to realize artificial intelligence (AI) with less energy consumption. It provides a promising energy-efficient way for realistic control tasks by combining SNNs with deep reinforcement learning (DRL). In this paper, we focus on the task where the agent needs to learn multi-dimensional deterministic policies to control, which is very common in real scenarios. Recently, the surrogate gradient method has been utilized for training multi-layer SNNs, which allows SNNs to achieve comparable performance with the corresponding deep networks in this task. Most existing spike-based RL methods take the firing rate as the output of SNNs, and convert it to represent continuous action space (i.e., the deterministic policy) through a fully-connected (FC) layer. However, the decimal characteristic of the firing rate brings the floating-point matrix operations to the FC layer, making the whole SNN unable to deploy on the neuromorphic hardware directly. To develop a fully spiking actor network without any floating-point matrix operations, we draw inspiration from the non-spiking interneurons found in insects and employ the membrane voltage of the non-spiking neurons to represent the action. Before the non-spiking neurons, multiple population neurons are introduced to decode different dimensions of actions. Since each population is used to decode a dimension of action, we argue that the neurons in each population should be connected in time domain and space domain. Hence, the intra-layer connections are used in output populations to enhance the representation capacity. Finally, we propose a fully spiking actor network with intra-layer connections (ILC-SAN).

{{</citation>}}


## cs.AI (2)



### (82/93) Deep Reinforcement Multi-agent Learning framework for Information Gathering with Local Gaussian Processes for Water Monitoring (Samuel Yanes Luis et al., 2024)

{{<citation>}}

Samuel Yanes Luis, Dmitriy Shutin, Juan Marchal Gómez, Daniel Gutiérrez Reina, Sergio Toral Marín. (2024)  
**Deep Reinforcement Multi-agent Learning framework for Information Gathering with Local Gaussian Processes for Water Monitoring**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.04631v1)  

---


**ABSTRACT**  
The conservation of hydrological resources involves continuously monitoring their contamination. A multi-agent system composed of autonomous surface vehicles is proposed in this paper to efficiently monitor the water quality. To achieve a safe control of the fleet, the fleet policy should be able to act based on measurements and to the the fleet state. It is proposed to use Local Gaussian Processes and Deep Reinforcement Learning to jointly obtain effective monitoring policies. Local Gaussian processes, unlike classical global Gaussian processes, can accurately model the information in a dissimilar spatial correlation which captures more accurately the water quality information. A Deep convolutional policy is proposed, that bases the decisions on the observation on the mean and variance of this model, by means of an information gain reward. Using a Double Deep Q-Learning algorithm, agents are trained to minimize the estimation error in a safe manner thanks to a Consensus-based heuristic. Simulation results indicate an improvement of up to 24% in terms of the mean absolute error with the proposed models. Also, training results with 1-3 agents indicate that our proposed approach returns 20% and 24% smaller average estimation errors for, respectively, monitoring water quality variables and monitoring algae blooms, as compared to state-of-the-art approaches

{{</citation>}}


### (83/93) Towards Explainable Artificial Intelligence (XAI): A Data Mining Perspective (Haoyi Xiong et al., 2024)

{{<citation>}}

Haoyi Xiong, Xuhong L, Xiaofei Zhang, Jiamin Chen, Xinhao Sun, Yuchen Li, Zeyi Sun, Mengnan Du. (2024)  
**Towards Explainable Artificial Intelligence (XAI): A Data Mining Perspective**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04374v1)  

---


**ABSTRACT**  
Given the complexity and lack of transparency in deep neural networks (DNNs), extensive efforts have been made to make these systems more interpretable or explain their behaviors in accessible terms. Unlike most reviews, which focus on algorithmic and model-centric perspectives, this work takes a "data-centric" view, examining how data collection, processing, and analysis contribute to explainable AI (XAI). We categorize existing work into three categories subject to their purposes: interpretations of deep models, referring to feature attributions and reasoning processes that correlate data points with model outputs; influences of training data, examining the impact of training data nuances, such as data valuation and sample anomalies, on decision-making processes; and insights of domain knowledge, discovering latent patterns and fostering new knowledge from data and models to advance social values and scientific discovery. Specifically, we distill XAI methodologies into data mining operations on training and testing data across modalities, such as images, text, and tabular data, as well as on training logs, checkpoints, models and other DNN behavior descriptors. In this way, our study offers a comprehensive, data-centric examination of XAI from a lens of data mining methods and applications.

{{</citation>}}


## cs.SD (1)



### (84/93) Masked Audio Generation using a Single Non-Autoregressive Transformer (Alon Ziv et al., 2024)

{{<citation>}}

Alon Ziv, Itai Gat, Gael Le Lan, Tal Remez, Felix Kreuk, Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi. (2024)  
**Masked Audio Generation using a Single Non-Autoregressive Transformer**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.04577v1)  

---


**ABSTRACT**  
We introduce MAGNeT, a masked generative sequence modeling method that operates directly over several streams of audio tokens. Unlike prior work, MAGNeT is comprised of a single-stage, non-autoregressive transformer. During training, we predict spans of masked tokens obtained from a masking scheduler, while during inference we gradually construct the output sequence using several decoding steps. To further enhance the quality of the generated audio, we introduce a novel rescoring method in which, we leverage an external pre-trained model to rescore and rank predictions from MAGNeT, which will be then used for later decoding steps. Lastly, we explore a hybrid version of MAGNeT, in which we fuse between autoregressive and non-autoregressive models to generate the first few seconds in an autoregressive manner while the rest of the sequence is being decoded in parallel. We demonstrate the efficiency of MAGNeT for the task of text-to-music and text-to-audio generation and conduct an extensive empirical evaluation, considering both objective metrics and human studies. The proposed approach is comparable to the evaluated baselines, while being significantly faster (x7 faster than the autoregressive baseline). Through ablation studies and analysis, we shed light on the importance of each of the components comprising MAGNeT, together with pointing to the trade-offs between autoregressive and non-autoregressive modeling, considering latency, throughput, and generation quality. Samples are available on our demo page https://pages.cs.huji.ac.il/adiyoss-lab/MAGNeT.

{{</citation>}}


## cs.IT (1)



### (85/93) A Novel Framework of K-repetition Grant-free Access via Diversity Slotted Aloha (DSA) (Haoran Mei et al., 2024)

{{<citation>}}

Haoran Mei, Limei Peng, Pin-Han Ho. (2024)  
**A Novel Framework of K-repetition Grant-free Access via Diversity Slotted Aloha (DSA)**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-NI, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04539v1)  

---


**ABSTRACT**  
This article introduces a novel framework of multi-user detection (MUD) for K-repetition grant-free non-orthogonal multiple access (K-GF-NOMA), called $\alpha$ iterative interference cancellation diversity slotted aloha ($\alpha$-IIC-DSA). The proposed framework targets at a simple yet effective decoding process where the AP can intelligently exploit the correlation among signals received at different resource blocks (RBs) so as to generate required multi-access interference (MAI) for realizing the signal-interference cancellation (SIC) based MUD. By keeping all operation and hardware complexity at the access point (AP), the proposed framework is applicable to the scenarios with random and uncoordinated access by numerous miniature mMTC devices (MTCDs). Numerical experiments are conducted to gain deep understanding on the performance of launching the proposed framework for K-GF-NOMA.

{{</citation>}}


## stat.ML (1)



### (86/93) Semi-Supervised Deep Sobolev Regression: Estimation, Variable Selection and Beyond (Zhao Ding et al., 2024)

{{<citation>}}

Zhao Ding, Chenguang Duan, Yuling Jiao, Jerry Zhijian Yang. (2024)  
**Semi-Supervised Deep Sobolev Regression: Estimation, Variable Selection and Beyond**  

---
Primary Category: stat.ML  
Categories: 62G05, 62G08, 65N21, cs-LG, stat-ML, stat.ML  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.04535v1)  

---


**ABSTRACT**  
We propose SDORE, a semi-supervised deep Sobolev regressor, for the nonparametric estimation of the underlying regression function and its gradient. SDORE employs deep neural networks to minimize empirical risk with gradient norm regularization, allowing computation of the gradient norm on unlabeled data. We conduct a comprehensive analysis of the convergence rates of SDORE and establish a minimax optimal rate for the regression function. Crucially, we also derive a convergence rate for the associated plug-in gradient estimator, even in the presence of significant domain shift. These theoretical findings offer valuable prior guidance for selecting regularization parameters and determining the size of the neural network, while showcasing the provable advantage of leveraging unlabeled data in semi-supervised learning. To the best of our knowledge, SDORE is the first provable neural network-based approach that simultaneously estimates the regression function and its gradient, with diverse applications including nonparametric variable selection and inverse problems. The effectiveness of SDORE is validated through an extensive range of numerical simulations and real data analysis.

{{</citation>}}


## eess.AS (1)



### (87/93) Zero Shot Audio to Audio Emotion Transfer With Speaker Disentanglement (Soumya Dutta et al., 2024)

{{<citation>}}

Soumya Dutta, Sriram Ganapathy. (2024)  
**Zero Shot Audio to Audio Emotion Transfer With Speaker Disentanglement**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2401.04511v1)  

---


**ABSTRACT**  
The problem of audio-to-audio (A2A) style transfer involves replacing the style features of the source audio with those from the target audio while preserving the content related attributes of the source audio. In this paper, we propose an efficient approach, termed as Zero-shot Emotion Style Transfer (ZEST), that allows the transfer of emotional content present in the given source audio with the one embedded in the target audio while retaining the speaker and speech content from the source. The proposed system builds upon decomposing speech into semantic tokens, speaker representations and emotion embeddings. Using these factors, we propose a framework to reconstruct the pitch contour of the given speech signal and train a decoder that reconstructs the speech signal. The model is trained using a self-supervision based reconstruction loss. During conversion, the emotion embedding is alone derived from the target audio, while rest of the factors are derived from the source audio. In our experiments, we show that, even without using parallel training data or labels from the source or target audio, we illustrate zero shot emotion transfer capabilities of the proposed ZEST model using objective and subjective quality evaluations.

{{</citation>}}


## q-bio.BM (1)



### (88/93) TwinBooster: Synergising Large Language Models with Barlow Twins and Gradient Boosting for Enhanced Molecular Property Prediction (Maximilian G. Schuh et al., 2024)

{{<citation>}}

Maximilian G. Schuh, Davide Boldini, Stephan A. Sieber. (2024)  
**TwinBooster: Synergising Large Language Models with Barlow Twins and Gradient Boosting for Enhanced Molecular Property Prediction**  

---
Primary Category: q-bio.BM  
Categories: cs-AI, cs-CL, cs-LG, q-bio-BM, q-bio.BM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04478v1)  

---


**ABSTRACT**  
The success of drug discovery and development relies on the precise prediction of molecular activities and properties. While in silico molecular property prediction has shown remarkable potential, its use has been limited so far to assays for which large amounts of data are available. In this study, we use a fine-tuned large language model to integrate biological assays based on their textual information, coupled with Barlow Twins, a Siamese neural network using a novel self-supervised learning approach. This architecture uses both assay information and molecular fingerprints to extract the true molecular information. TwinBooster enables the prediction of properties of unseen bioassays and molecules by providing state-of-the-art zero-shot learning tasks. Remarkably, our artificial intelligence pipeline shows excellent performance on the FS-Mol benchmark. This breakthrough demonstrates the application of deep learning to critical property prediction tasks where data is typically scarce. By accelerating the early identification of active molecules in drug discovery and development, this method has the potential to help streamline the identification of novel therapeutics.

{{</citation>}}


## q-fin.ST (1)



### (89/93) Can ChatGPT Compute Trustworthy Sentiment Scores from Bloomberg Market Wraps? (Baptiste Lefort et al., 2024)

{{<citation>}}

Baptiste Lefort, Eric Benhamou, Jean-Jacques Ohana, David Saltiel, Beatrice Guez, Damien Challet. (2024)  
**Can ChatGPT Compute Trustworthy Sentiment Scores from Bloomberg Market Wraps?**  

---
Primary Category: q-fin.ST  
Categories: cs-AI, q-fin-ST, q-fin.ST  
Keywords: ChatGPT, Financial, GPT  
[Paper Link](http://arxiv.org/abs/2401.05447v1)  

---


**ABSTRACT**  
We used a dataset of daily Bloomberg Financial Market Summaries from 2010 to 2023, reposted on large financial media, to determine how global news headlines may affect stock market movements using ChatGPT and a two-stage prompt approach. We document a statistically significant positive correlation between the sentiment score and future equity market returns over short to medium term, which reverts to a negative correlation over longer horizons. Validation of this correlation pattern across multiple equity markets indicates its robustness across equity regions and resilience to non-linearity, evidenced by comparison of Pearson and Spearman correlations. Finally, we provide an estimate of the optimal horizon that strikes a balance between reactivity to new information and correlation.

{{</citation>}}


## cs.MM (1)



### (90/93) SonicVisionLM: Playing Sound with Vision Language Models (Zhifeng Xie et al., 2024)

{{<citation>}}

Zhifeng Xie, Shengye Yu, Mengtian Li, Qile He, Chaofeng Chen, Yu-Gang Jiang. (2024)  
**SonicVisionLM: Playing Sound with Vision Language Models**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs-SD, cs.MM, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04394v1)  

---


**ABSTRACT**  
There has been a growing interest in the task of generating sound for silent videos, primarily because of its practicality in streamlining video post-production. However, existing methods for video-sound generation attempt to directly create sound from visual representations, which can be challenging due to the difficulty of aligning visual representations with audio representations. In this paper, we present SonicVisionLM, a novel framework aimed at generating a wide range of sound effects by leveraging vision language models. Instead of generating audio directly from video, we use the capabilities of powerful vision language models (VLMs). When provided with a silent video, our approach first identifies events within the video using a VLM to suggest possible sounds that match the video content. This shift in approach transforms the challenging task of aligning image and audio into more well-studied sub-problems of aligning image-to-text and text-to-audio through the popular diffusion models. To improve the quality of audio recommendations with LLMs, we have collected an extensive dataset that maps text descriptions to specific sound effects and developed temporally controlled audio adapters. Our approach surpasses current state-of-the-art methods for converting video to audio, resulting in enhanced synchronization with the visuals and improved alignment between audio and video components. Project page: https://yusiissy.github.io/SonicVisionLM.github.io/

{{</citation>}}


## cs.RO (2)



### (91/93) Towards Real-World Aerial Vision Guidance with Categorical 6D Pose Tracker (Jingtao Sun et al., 2024)

{{<citation>}}

Jingtao Sun, Yaonan Wang, Danwei Wang. (2024)  
**Towards Real-World Aerial Vision Guidance with Categorical 6D Pose Tracker**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.04377v1)  

---


**ABSTRACT**  
Tracking the object 6-DoF pose is crucial for various downstream robot tasks and real-world applications. In this paper, we investigate the real-world robot task of aerial vision guidance for aerial robotics manipulation, utilizing category-level 6-DoF pose tracking. Aerial conditions inevitably introduce special challenges, such as rapid viewpoint changes in pitch and roll. To support this task and challenge, we firstly introduce a robust category-level 6-DoF pose tracker (Robust6DoF). This tracker leverages shape and temporal prior knowledge to explore optimal inter-frame keypoint pairs, generated under a priori structural adaptive supervision in a coarse-to-fine manner. Notably, our Robust6DoF employs a Spatial-Temporal Augmentation module to deal with the problems of the inter-frame differences and intra-class shape variations through both temporal dynamic filtering and shape-similarity filtering. We further present a Pose-Aware Discrete Servo strategy (PAD-Servo), serving as a decoupling approach to implement the final aerial vision guidance task. It contains two servo action policies to better accommodate the structural properties of aerial robotics manipulation. Exhaustive experiments on four well-known public benchmarks demonstrate the superiority of our Robust6DoF. Real-world tests directly verify that our Robust6DoF along with PAD-Servo can be readily used in real-world aerial robotic applications.

{{</citation>}}


### (92/93) Large Language Models for Robotics: Opportunities, Challenges, and Perspectives (Jiaqi Wang et al., 2024)

{{<citation>}}

Jiaqi Wang, Zihao Wu, Yiwei Li, Hanqi Jiang, Peng Shu, Enze Shi, Huawen Hu, Chong Ma, Yiheng Liu, Xuhui Wang, Yincheng Yao, Xuan Liu, Huaqin Zhao, Zhengliang Liu, Haixing Dai, Lin Zhao, Bao Ge, Xiang Li, Tianming Liu, Shu Zhang. (2024)  
**Large Language Models for Robotics: Opportunities, Challenges, and Perspectives**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.04334v1)  

---


**ABSTRACT**  
Large language models (LLMs) have undergone significant expansion and have been increasingly integrated across various domains. Notably, in the realm of robot task planning, LLMs harness their advanced reasoning and language comprehension capabilities to formulate precise and efficient action plans based on natural language instructions. However, for embodied tasks, where robots interact with complex environments, text-only LLMs often face challenges due to a lack of compatibility with robotic visual perception. This study provides a comprehensive overview of the emerging integration of LLMs and multimodal LLMs into various robotic tasks. Additionally, we propose a framework that utilizes multimodal GPT-4V to enhance embodied task planning through the combination of natural language instructions and robot visual perceptions. Our results, based on diverse datasets, indicate that GPT-4V effectively enhances robot performance in embodied tasks. This extensive survey and evaluation of LLMs and multimodal LLMs across a variety of robotic tasks enriches the understanding of LLM-centric embodied intelligence and provides forward-looking insights toward bridging the gap in Human-Robot-Environment interaction.

{{</citation>}}


## cs.GT (1)



### (93/93) Online Allocation with Replenishable Budgets: Worst Case and Beyond (Jianyi Yang et al., 2024)

{{<citation>}}

Jianyi Yang, Pengfei Li, Mohammad Jaminur Islam, Shaolei Ren. (2024)  
**Online Allocation with Replenishable Budgets: Worst Case and Beyond**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs-PF, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04340v1)  

---


**ABSTRACT**  
This paper studies online resource allocation with replenishable budgets, where budgets can be replenished on top of the initial budget and an agent sequentially chooses online allocation decisions without violating the available budget constraint at each round. We propose a novel online algorithm, called OACP (Opportunistic Allocation with Conservative Pricing), that conservatively adjusts dual variables while opportunistically utilizing available resources. OACP achieves a bounded asymptotic competitive ratio in adversarial settings as the number of decision rounds T gets large. Importantly, the asymptotic competitive ratio of OACP is optimal in the absence of additional assumptions on budget replenishment. To further improve the competitive ratio, we make a mild assumption that there is budget replenishment every T^* >= 1 decision rounds and propose OACP+ to dynamically adjust the total budget assignment for online allocation. Next, we move beyond the worst-case and propose LA-OACP (Learning-Augmented OACP/OACP+), a novel learning-augmented algorithm for online allocation with replenishable budgets. We prove that LA-OACP can improve the average utility compared to OACP/OACP+ when the ML predictor is properly trained, while still offering worst-case utility guarantees when the ML predictions are arbitrarily wrong. Finally, we run simulation studies of sustainable AI inference powered by renewables, validating our analysis and demonstrating the empirical benefits of LA-OACP.

{{</citation>}}
