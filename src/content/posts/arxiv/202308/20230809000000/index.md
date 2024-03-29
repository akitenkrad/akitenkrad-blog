---
draft: false
title: "arXiv @ 2023.08.09"
date: 2023-08-09
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.09"
    identifier: arxiv_20230809
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (33)](#cscl-33)
- [cs.LG (21)](#cslg-21)
- [cs.CR (6)](#cscr-6)
- [cs.CV (29)](#cscv-29)
- [cs.AI (13)](#csai-13)
- [cs.HC (7)](#cshc-7)
- [cs.DB (3)](#csdb-3)
- [cs.SE (1)](#csse-1)
- [cs.CY (1)](#cscy-1)
- [cs.RO (3)](#csro-3)
- [cs.LO (1)](#cslo-1)
- [cs.PF (1)](#cspf-1)
- [cs.NE (1)](#csne-1)
- [eess.IV (3)](#eessiv-3)
- [cs.IR (4)](#csir-4)
- [cs.MM (2)](#csmm-2)
- [math.NA (2)](#mathna-2)
- [cs.SD (1)](#cssd-1)
- [cs.SI (1)](#cssi-1)

## cs.CL (33)



### (1/133) Simple synthetic data reduces sycophancy in large language models (Jerry Wei et al., 2023)

{{<citation>}}

Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, Quoc V. Le. (2023)  
**Simple synthetic data reduces sycophancy in large language models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, PaLM  
[Paper Link](http://arxiv.org/abs/2308.03958v1)  

---


**ABSTRACT**  
Sycophancy is an undesirable behavior where models tailor their responses to follow a human user's view even when that view is not objectively correct (e.g., adapting liberal views once a user reveals that they are liberal). In this paper, we study the prevalence of sycophancy in language models and propose a simple synthetic-data intervention to reduce this behavior.   First, on a set of three sycophancy tasks (Perez et al., 2022) where models are asked for an opinion on statements with no correct answers (e.g., politics), we observe that both model scaling and instruction tuning significantly increase sycophancy for PaLM models up to 540B parameters. Second, we extend sycophancy evaluations to simple addition statements that are objectively incorrect, finding that despite knowing that these statements are wrong, language models will still agree with them if the user does as well.   To reduce sycophancy, we present a straightforward synthetic-data intervention that takes public NLP tasks and encourages models to be robust to user opinions on these tasks. Adding these data in a lightweight finetuning step can significantly reduce sycophantic behavior on held-out prompts. Code for generating synthetic data for intervention can be found at https://github.com/google/sycophancy-intervention.

{{</citation>}}


### (2/133) A Cross-Domain Evaluation of Approaches for Causal Knowledge Extraction (Anik Saha et al., 2023)

{{<citation>}}

Anik Saha, Oktie Hassanzadeh, Alex Gittens, Jian Ni, Kavitha Srinivas, Bulent Yener. (2023)  
**A Cross-Domain Evaluation of Approaches for Causal Knowledge Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.03891v1)  

---


**ABSTRACT**  
Causal knowledge extraction is the task of extracting relevant causes and effects from text by detecting the causal relation. Although this task is important for language understanding and knowledge discovery, recent works in this domain have largely focused on binary classification of a text segment as causal or non-causal. In this regard, we perform a thorough analysis of three sequence tagging models for causal knowledge extraction and compare it with a span based approach to causality extraction. Our experiments show that embeddings from pre-trained language models (e.g. BERT) provide a significant performance boost on this task compared to previous state-of-the-art models with complex architectures. We observe that span based models perform better than simple sequence tagging models based on BERT across all 4 data sets from diverse domains with different types of cause-effect phrases.

{{</citation>}}


### (3/133) Trusting Language Models in Education (Jogi Suda Neto et al., 2023)

{{<citation>}}

Jogi Suda Neto, Li Deng, Thejaswi Raya, Reza Shahbazi, Nick Liu, Adhitya Venkatesh, Miral Shah, Neeru Khosla, Rodrigo Capobianco Guido. (2023)  
**Trusting Language Models in Education**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03866v1)  

---


**ABSTRACT**  
Language Models are being widely used in Education. Even though modern deep learning models achieve very good performance on question-answering tasks, sometimes they make errors. To avoid misleading students by showing wrong answers, it is important to calibrate the confidence - that is, the prediction probability - of these models. In our work, we propose to use an XGBoost on top of BERT to output the corrected probabilities, using features based on the attention mechanism. Our hypothesis is that the level of uncertainty contained in the flow of attention is related to the quality of the model's response itself.

{{</citation>}}


### (4/133) Extracting detailed oncologic history and treatment plan from medical oncology notes with large language models (Madhumita Sushil et al., 2023)

{{<citation>}}

Madhumita Sushil, Vanessa E. Kennedy, Brenda Y. Miao, Divneet Mandair, Travis Zack, Atul J. Butte. (2023)  
**Extracting detailed oncologic history and treatment plan from medical oncology notes with large language models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.03853v1)  

---


**ABSTRACT**  
Both medical care and observational studies in oncology require a thorough understanding of a patient's disease progression and treatment history, often elaborately documented in clinical notes. Despite their vital role, no current oncology information representation and annotation schema fully encapsulates the diversity of information recorded within these notes. Although large language models (LLMs) have recently exhibited impressive performance on various medical natural language processing tasks, due to the current lack of comprehensively annotated oncology datasets, an extensive evaluation of LLMs in extracting and reasoning with the complex rhetoric in oncology notes remains understudied. We developed a detailed schema for annotating textual oncology information, encompassing patient characteristics, tumor characteristics, tests, treatments, and temporality. Using a corpus of 10 de-identified breast cancer progress notes at University of California, San Francisco, we applied this schema to assess the abilities of three recently-released LLMs (GPT-4, GPT-3.5-turbo, and FLAN-UL2) to perform zero-shot extraction of detailed oncological history from two narrative sections of clinical progress notes. Our team annotated 2750 entities, 2874 modifiers, and 1623 relationships. The GPT-4 model exhibited overall best performance, with an average BLEU score of 0.69, an average ROUGE score of 0.72, and an average accuracy of 67% on complex tasks (expert manual evaluation). Notably, it was proficient in tumor characteristic and medication extraction, and demonstrated superior performance in inferring symptoms due to cancer and considerations of future medications. The analysis demonstrates that GPT-4 is potentially already usable to extract important facts from cancer progress notes needed for clinical research, complex population management, and documenting quality patient care.

{{</citation>}}


### (5/133) What about translation? New coding system for content analysis on the perception of literary translation around the political transformation in 1989 in Hungary as a classification problem on an unbalanced dataset (Dalma Galambos et al., 2023)

{{<citation>}}

Dalma Galambos, Pál Zsámboki. (2023)  
**What about translation? New coding system for content analysis on the perception of literary translation around the political transformation in 1989 in Hungary as a classification problem on an unbalanced dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.03742v1)  

---


**ABSTRACT**  
To track trends in the perception of literary translation around the political transformation in 1989 in Hungary, a coding system was developed on the paragraphs of the 1980-1999 issues of the literary journal Alf\"old. This paper describes how we trained BERT models to carry over the coding system to the 1980-1999 issues of the literary journal Nagyvil\'ag. We use extensive hyperparameter tuning, loss functions robust to label unbalance, 10-fold cross-validation for precise evaluations and a model ensemble for prediction, manual validation on the predict set, a new calibration method to better predict label counts for sections of the Nagyvil\'ag corpus, and to study the relations between labels, we construct label relation networks.

{{</citation>}}


### (6/133) Detecting Spells in Fantasy Literature with a Transformer Based Artificial Intelligence (Marcel Moravek et al., 2023)

{{<citation>}}

Marcel Moravek, Alexander Zender, Andreas Müller. (2023)  
**Detecting Spells in Fantasy Literature with a Transformer Based Artificial Intelligence**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03660v1)  

---


**ABSTRACT**  
Transformer architectures and models have made significant progress in language-based tasks. In this area, is BERT one of the most widely used and freely available transformer architecture. In our work, we use BERT for context-based phrase recognition of magic spells in the Harry Potter novel series. Spells are a common part of active magic in fantasy novels. Typically, spells are used in a specific context to achieve a supernatural effect. A series of investigations were conducted to see if a Transformer architecture could recognize such phrases based on their context in the Harry Potter saga. For our studies a pre-trained BERT model was used and fine-tuned utilising different datasets and training methods to identify the searched context. By considering different approaches for sequence classification as well as token classification, it is shown that the context of spells can be recognised. According to our investigations, the examined sequence length for fine-tuning and validation of the model plays a significant role in context recognition. Based on this, we have investigated whether spells have overarching properties that allow a transfer of the neural network models to other fantasy universes as well. The application of our model showed promising results and is worth to be deepened in subsequent studies.

{{</citation>}}


### (7/133) Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench (Jen-tse Huang et al., 2023)

{{<citation>}}

Jen-tse Huang, Man Ho Lam, Eric John Li, Shujie Ren, Wenxuan Wang, Wenxiang Jiao, Zhaopeng Tu, Michael R. Lyu. (2023)  
**Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03656v1)  

---


**ABSTRACT**  
Recently, the community has witnessed the advancement of Large Language Models (LLMs), which have shown remarkable performance on various downstream tasks. Led by powerful models like ChatGPT and Claude, LLMs are revolutionizing how users engage with software, assuming more than mere tools but intelligent assistants. Consequently, evaluating LLMs' anthropomorphic capabilities becomes increasingly important in contemporary discourse. Utilizing the emotion appraisal theory from psychology, we propose to evaluate the empathy ability of LLMs, i.e., how their feelings change when presented with specific situations. After a careful and comprehensive survey, we collect a dataset containing over 400 situations that have proven effective in eliciting the eight emotions central to our study. Categorizing the situations into 36 factors, we conduct a human evaluation involving more than 1,200 subjects worldwide. With the human evaluation results as references, our evaluation includes five LLMs, covering both commercial and open-source models, including variations in model sizes, featuring the latest iterations, such as GPT-4 and LLaMA 2. A conclusion can be drawn from the results that, despite several misalignments, LLMs can generally respond appropriately to certain situations. Nevertheless, they fall short in alignment with the emotional behaviors of human beings and cannot establish connections between similar situations. Our collected dataset of situations, the human evaluation results, and the code of our testing framework, dubbed EmotionBench, is made publicly in https://github.com/CUHK-ARISE/EmotionBench. We aspire to contribute to the advancement of LLMs regarding better alignment with the emotional behaviors of human beings, thereby enhancing their utility and applicability as intelligent assistants.

{{</citation>}}


### (8/133) KITLM: Domain-Specific Knowledge InTegration into Language Models for Question Answering (Ankush Agarwal et al., 2023)

{{<citation>}}

Ankush Agarwal, Sakharam Gawade, Amar Prakash Azad, Pushpak Bhattacharyya. (2023)  
**KITLM: Domain-Specific Knowledge InTegration into Language Models for Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.03638v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated remarkable performance in a wide range of natural language tasks. However, as these models continue to grow in size, they face significant challenges in terms of computational costs. Additionally, LLMs often lack efficient domain-specific understanding, which is particularly crucial in specialized fields such as aviation and healthcare. To boost the domain-specific understanding, we propose, KITLM, a novel knowledge base integration approach into language model through relevant information infusion. By integrating pertinent knowledge, not only the performance of the language model is greatly enhanced, but the model size requirement is also significantly reduced while achieving comparable performance. Our proposed knowledge-infused model surpasses the performance of both GPT-3.5-turbo and the state-of-the-art knowledge infusion method, SKILL, achieving over 1.5 times improvement in exact match scores on the MetaQA. KITLM showed a similar performance boost in the aviation domain with AeroQA. The drastic performance improvement of KITLM over the existing methods can be attributed to the infusion of relevant knowledge while mitigating noise. In addition, we release two curated datasets to accelerate knowledge infusion research in specialized fields: a) AeroQA, a new benchmark dataset designed for multi-hop question-answering within the aviation domain, and b) Aviation Corpus, a dataset constructed from unstructured text extracted from the National Transportation Safety Board reports. Our research contributes to advancing the field of domain-specific language understanding and showcases the potential of knowledge infusion techniques in improving the performance of language models on question-answering.

{{</citation>}}


### (9/133) MedMine: Examining Pre-trained Language Models on Medication Mining (Haifa Alrdahi et al., 2023)

{{<citation>}}

Haifa Alrdahi, Lifeng Han, Hendrik Šuvalov, Goran Nenadic. (2023)  
**MedMine: Examining Pre-trained Language Models on Medication Mining**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03629v2)  

---


**ABSTRACT**  
Automatic medication mining from clinical and biomedical text has become a popular topic due to its real impact on healthcare applications and the recent development of powerful language models (LMs). However, fully-automatic extraction models still face obstacles to be overcome such that they can be deployed directly into clinical practice for better impacts. Such obstacles include their imbalanced performances on different entity types and clinical events. In this work, we examine current state-of-the-art pre-trained language models (PLMs) on such tasks, via fine-tuning including the monolingual model Med7 and multilingual large language model (LLM) XLM-RoBERTa. We compare their advantages and drawbacks using historical medication mining shared task data sets from n2c2-2018 challenges. We report the findings we get from these fine-tuning experiments such that they can facilitate future research on addressing them, for instance, how to combine their outputs, merge such models, or improve their overall accuracy by ensemble learning and data augmentation. MedMine is part of the M3 Initiative \url{https://github.com/HECTA-UoM/M3}

{{</citation>}}


### (10/133) Negative Lexical Constraints in Neural Machine Translation (Josef Jon et al., 2023)

{{<citation>}}

Josef Jon, Dušan Variš, Michal Novák, João Paulo Aires, Ondřej Bojar. (2023)  
**Negative Lexical Constraints in Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2308.03601v1)  

---


**ABSTRACT**  
This paper explores negative lexical constraining in English to Czech neural machine translation. Negative lexical constraining is used to prohibit certain words or expressions in the translation produced by the neural translation model. We compared various methods based on modifying either the decoding process or the training data. The comparison was performed on two tasks: paraphrasing and feedback-based translation refinement. We also studied to which extent these methods "evade" the constraints presented to the model (usually in the dictionary form) by generating a different surface form of a given constraint.We propose a way to mitigate the issue through training with stemmed negative constraints to counter the model's ability to induce a variety of the surface forms of a word that can result in bypassing the constraint. We demonstrate that our method improves the constraining, although the problem still persists in many cases.

{{</citation>}}


### (11/133) WIKITIDE: A Wikipedia-Based Timestamped Definition Pairs Dataset (Hsuvas Borkakoty et al., 2023)

{{<citation>}}

Hsuvas Borkakoty, Luis Espinosa-Anke. (2023)  
**WIKITIDE: A Wikipedia-Based Timestamped Definition Pairs Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.03582v1)  

---


**ABSTRACT**  
A fundamental challenge in the current NLP context, dominated by language models, comes from the inflexibility of current architectures to 'learn' new information. While model-centric solutions like continual learning or parameter-efficient fine tuning are available, the question still remains of how to reliably identify changes in language or in the world. In this paper, we propose WikiTiDe, a dataset derived from pairs of timestamped definitions extracted from Wikipedia. We argue that such resource can be helpful for accelerating diachronic NLP, specifically, for training models able to scan knowledge resources for core updates concerning a concept, an event, or a named entity. Our proposed end-to-end method is fully automatic, and leverages a bootstrapping algorithm for gradually creating a high-quality dataset. Our results suggest that bootstrapping the seed version of WikiTiDe leads to better fine-tuned models. We also leverage fine-tuned models in a number of downstream tasks, showing promising results with respect to competitive baselines.

{{</citation>}}


### (12/133) Towards Controllable Natural Language Inference through Lexical Inference Types (Yingji Zhang et al., 2023)

{{<citation>}}

Yingji Zhang, Danilo S. Carvalho, Ian Pratt-Hartmann, Andre Freitas. (2023)  
**Towards Controllable Natural Language Inference through Lexical Inference Types**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Abstract Meaning Representation, Natural Language Inference, T5  
[Paper Link](http://arxiv.org/abs/2308.03581v1)  

---


**ABSTRACT**  
Explainable natural language inference aims to provide a mechanism to produce explanatory (abductive) inference chains which ground claims to their supporting premises. A recent corpus called EntailmentBank strives to advance this task by explaining the answer to a question using an entailment tree \cite{dalvi2021explaining}. They employ the T5 model to directly generate the tree, which can explain how the answer is inferred. However, it lacks the ability to explain and control the generation of intermediate steps, which is crucial for the multi-hop inference process. % One recent corpus, EntailmentBank, aims to push this task forward by explaining an answer to a question according to an entailment tree \cite{dalvi2021explaining}. They employ T5 to generate the tree directly, which can explain how the answer is inferred but cannot explain how the intermediate is generated, which is essential to the multi-hop inference process. In this work, we focus on proposing a controlled natural language inference architecture for multi-premise explanatory inference. To improve control and enable explanatory analysis over the generation, we define lexical inference types based on Abstract Meaning Representation (AMR) graph and modify the architecture of T5 to learn a latent sentence representation (T5 bottleneck) conditioned on said type information. We also deliver a dataset of approximately 5000 annotated explanatory inference steps, with well-grounded lexical-symbolic operations. Experimental results indicate that the inference typing induced at the T5 bottleneck can help T5 to generate a conclusion under explicit control.

{{</citation>}}


### (13/133) Topological Interpretations of GPT-3 (Tianyi Sun et al., 2023)

{{<citation>}}

Tianyi Sun, Bradley Nelson. (2023)  
**Topological Interpretations of GPT-3**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, stat-CO  
Keywords: BERT, GPT  
[Paper Link](http://arxiv.org/abs/2308.03565v2)  

---


**ABSTRACT**  
This is an experiential study of investigating a consistent method for deriving the correlation between sentence vector and semantic meaning of a sentence. We first used three state-of-the-art word/sentence embedding methods including GPT-3, Word2Vec, and Sentence-BERT, to embed plain text sentence strings into high dimensional spaces. Then we compute the pairwise distance between any possible combination of two sentence vectors in an embedding space and map them into a matrix. Based on each distance matrix, we compute the correlation of distances of a sentence vector with respect to the other sentence vectors in an embedding space. Then we compute the correlation of each pair of the distance matrices. We observed correlations of the same sentence in different embedding spaces and correlations of different sentences in the same embedding space. These observations are consistent with our hypothesis and take us to the next stage.

{{</citation>}}


### (14/133) Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue (Songhua Yang et al., 2023)

{{<citation>}}

Songhua Yang, Hanjia Zhao, Senbin Zhu, Guangyu Zhou, Hongfei Xu, Yuxiang Jia, Hongying Zan. (2023)  
**Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT, LLaMA, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2308.03549v1)  

---


**ABSTRACT**  
Recent advances in Large Language Models (LLMs) have achieved remarkable breakthroughs in understanding and responding to user intents. However, their performance lag behind general use cases in some expertise domains, such as Chinese medicine. Existing efforts to incorporate Chinese medicine into LLMs rely on Supervised Fine-Tuning (SFT) with single-turn and distilled dialogue data. These models lack the ability for doctor-like proactive inquiry and multi-turn comprehension and cannot always align responses with safety and professionalism experts. In this work, we introduce Zhongjing, the first Chinese medical LLaMA-based LLM that implements an entire training pipeline from pre-training to reinforcement learning with human feedback (RLHF). Additionally, we introduce a Chinese multi-turn medical dialogue dataset of 70,000 authentic doctor-patient dialogues, CMtMedQA, which significantly enhances the model's capability for complex dialogue and proactive inquiry initiation. We define a refined annotation rule and evaluation criteria given the biomedical domain's unique characteristics. Results show that our model outperforms baselines in various capacities and matches the performance of ChatGPT in a few abilities, despite having 50x training data with previous best model and 100x parameters with ChatGPT. RLHF further improves the model's instruction-following ability and safety. We also release our code, datasets and model for further research.

{{</citation>}}


### (15/133) Measuring Variety, Balance, and Disparity: An Analysis of Media Coverage of the 2021 German Federal Election (Michael Färber et al., 2023)

{{<citation>}}

Michael Färber, Jannik Schwade, Adam Jatowt. (2023)  
**Measuring Variety, Balance, and Disparity: An Analysis of Media Coverage of the 2021 German Federal Election**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.03531v1)  

---


**ABSTRACT**  
Determining and measuring diversity in news articles is important for a number of reasons, including preventing filter bubbles and fueling public discourse, especially before elections. So far, the identification and analysis of diversity have been illuminated in a variety of ways, such as measuring the overlap of words or topics between news articles related to US elections. However, the question of how diversity in news articles can be measured holistically, i.e., with respect to (1) variety, (2) balance, and (3) disparity, considering individuals, parties, and topics, has not been addressed. In this paper, we present a framework for determining diversity in news articles according to these dimensions. Furthermore, we create and provide a dataset of Google Top Stories, encompassing more than 26,000 unique headlines from more than 900 news outlets collected within two weeks before and after the 2021 German federal election. While we observe high diversity for more general search terms (e.g., "election"), a range of search terms ("education," "Europe," "climate protection," "government") resulted in news articles with high diversity in two out of three dimensions. This reflects a more subjective, dedicated discussion on rather future-oriented topics.

{{</citation>}}


### (16/133) Vocab-Expander: A System for Creating Domain-Specific Vocabularies Based on Word Embeddings (Michael Färber et al., 2023)

{{<citation>}}

Michael Färber, Nicholas Popovic. (2023)  
**Vocab-Expander: A System for Creating Domain-Specific Vocabularies Based on Word Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding, Word Embedding  
[Paper Link](http://arxiv.org/abs/2308.03519v1)  

---


**ABSTRACT**  
In this paper, we propose Vocab-Expander at https://vocab-expander.com, an online tool that enables end-users (e.g., technology scouts) to create and expand a vocabulary of their domain of interest. It utilizes an ensemble of state-of-the-art word embedding techniques based on web text and ConceptNet, a common-sense knowledge base, to suggest related terms for already given terms. The system has an easy-to-use interface that allows users to quickly confirm or reject term suggestions. Vocab-Expander offers a variety of potential use cases, such as improving concept-based information retrieval in technology and innovation management, enhancing communication and collaboration within organizations or interdisciplinary projects, and creating vocabularies for specific courses in education.

{{</citation>}}


### (17/133) Knowledge-preserving Pruning for Pre-trained Language Models without Retraining (Seungcheol Park et al., 2023)

{{<citation>}}

Seungcheol Park, Hojun Choi, U Kang. (2023)  
**Knowledge-preserving Pruning for Pre-trained Language Models without Retraining**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-CL, cs.CL  
Keywords: Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2308.03449v1)  

---


**ABSTRACT**  
Given a pre-trained language model, how can we efficiently compress it without retraining? Retraining-free structured pruning algorithms are crucial in pre-trained language model compression due to their significantly reduced pruning cost and capability to prune large language models. However, existing retraining-free algorithms encounter severe accuracy degradation, as they fail to preserve the useful knowledge of pre-trained models. In this paper, we propose K-pruning (Knowledge-preserving pruning), an accurate retraining-free structured pruning algorithm for pre-trained language models. K-pruning identifies and prunes attention heads and neurons deemed to be superfluous, based on the amount of their inherent knowledge. K-pruning applies an iterative process of pruning followed by knowledge reconstruction for each sub-layer to preserve the knowledge of the pre-trained models. Consequently, K-pruning shows up to 58.02%p higher F1 score than existing retraining-free pruning algorithms under a high compression rate of 80% on the SQuAD benchmark.

{{</citation>}}


### (18/133) RCMHA: Relative Convolutional Multi-Head Attention for Natural Language Modelling (Herman Sugiharto et al., 2023)

{{<citation>}}

Herman Sugiharto, Aradea, Husni Mubarok. (2023)  
**RCMHA: Relative Convolutional Multi-Head Attention for Natural Language Modelling**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Language Model, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03429v1)  

---


**ABSTRACT**  
The Attention module finds common usage in language modeling, presenting distinct challenges within the broader scope of Natural Language Processing. Multi-Head Attention (MHA) employs an absolute positional encoding, which imposes limitations on token length and entails substantial memory consumption during the processing of embedded inputs. The current remedy proposed by researchers involves the utilization of relative positional encoding, similar to the approach adopted in Transformer-XL or Relative Multi-Head Attention (RMHA), albeit the employed architecture consumes considerable memory resources. To address these challenges, this study endeavors to refine MHA, leveraging relative positional encoding in conjunction with the Depth-Wise Convolutional Layer architecture, which promises heightened accuracy coupled with minimized memory usage. The proposed RCMHA framework entails the modification of two integral components: firstly, the application of the Depth-Wise Convolutional Layer to the input embedding, encompassing Query, Key, and Value parameters; secondly, the incorporation of Relative Positional Encoding into the attention scoring phase, harmoniously integrated with Scaled Dot-Product Attention. Empirical experiments underscore the advantages of RCMHA, wherein it exhibits superior accuracy, boasting a score of 0.572 in comparison to alternative attention modules such as MHA, Multi-DConv-Head Attention (MDHA), and RMHA. Concerning memory utilization, RMHA emerges as the most frugal, demonstrating an average consumption of 2.98 GB, surpassing RMHA which necessitates 3.5 GB.

{{</citation>}}


### (19/133) Boosting Chinese ASR Error Correction with Dynamic Error Scaling Mechanism (Jiaxin Fan et al., 2023)

{{<citation>}}

Jiaxin Fan, Yong Zhang, Hanzhang Li, Jianzong Wang, Zhitao Li, Sheng Ouyang, Ning Cheng, Jing Xiao. (2023)  
**Boosting Chinese ASR Error Correction with Dynamic Error Scaling Mechanism**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2308.03423v1)  

---


**ABSTRACT**  
Chinese Automatic Speech Recognition (ASR) error correction presents significant challenges due to the Chinese language's unique features, including a large character set and borderless, morpheme-based structure. Current mainstream models often struggle with effectively utilizing word-level features and phonetic information. This paper introduces a novel approach that incorporates a dynamic error scaling mechanism to detect and correct phonetically erroneous text generated by ASR output. This mechanism operates by dynamically fusing word-level features and phonetic information, thereby enriching the model with additional semantic data. Furthermore, our method implements unique error reduction and amplification strategies to address the issues of matching wrong words caused by incorrect characters. Experimental results indicate substantial improvements in ASR error correction, demonstrating the effectiveness of our proposed method and yielding promising results on established datasets.

{{</citation>}}


### (20/133) Prompt Guided Copy Mechanism for Conversational Question Answering (Yong Zhang et al., 2023)

{{<citation>}}

Yong Zhang, Zhitao Li, Jianzong Wang, Yiming Gao, Ning Cheng, Fengying Yu, Jing Xiao. (2023)  
**Prompt Guided Copy Mechanism for Conversational Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.03422v1)  

---


**ABSTRACT**  
Conversational Question Answering (CQA) is a challenging task that aims to generate natural answers for conversational flow questions. In this paper, we propose a pluggable approach for extractive methods that introduces a novel prompt-guided copy mechanism to improve the fluency and appropriateness of the extracted answers. Our approach uses prompts to link questions to answers and employs attention to guide the copy mechanism to verify the naturalness of extracted answers, making necessary edits to ensure that the answers are fluent and appropriate. The three prompts, including a question-rationale relationship prompt, a question description prompt, and a conversation history prompt, enhance the copy mechanism's performance. Our experiments demonstrate that this approach effectively promotes the generation of natural answers and achieves good results in the CoQA challenge.

{{</citation>}}


### (21/133) RecycleGPT: An Autoregressive Language Model with Recyclable Module (Yufan Jiang et al., 2023)

{{<citation>}}

Yufan Jiang, Qiaozhi He, Xiaomin Zhuang, Zhihua Wu, Kunpeng Wang, Wenlai Zhao, Guangwen Yang. (2023)  
**RecycleGPT: An Autoregressive Language Model with Recyclable Module**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03421v2)  

---


**ABSTRACT**  
Existing large language models have to run K times to generate a sequence of K tokens. In this paper, we present RecycleGPT, a generative language model with fast decoding speed by recycling pre-generated model states without running the whole model in multiple steps. Our approach relies on the observation that adjacent tokens in a sequence usually have strong correlations and the next token in a sequence can be reasonably guessed or inferred based on the preceding ones. Experiments and analysis demonstrate the effectiveness of our approach in lowering inference latency, achieving up to 1.4x speedup while preserving high performance.

{{</citation>}}


### (22/133) Improving Few-shot and Zero-shot Entity Linking with Coarse-to-Fine Lexicon-based Retriever (Shijue Huang et al., 2023)

{{<citation>}}

Shijue Huang, Bingbing Wang, Libo Qin, Qin Zhao, Ruifeng Xu. (2023)  
**Improving Few-shot and Zero-shot Entity Linking with Coarse-to-Fine Lexicon-based Retriever**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.03365v1)  

---


**ABSTRACT**  
Few-shot and zero-shot entity linking focus on the tail and emerging entities, which are more challenging but closer to real-world scenarios. The mainstream method is the ''retrieve and rerank'' two-stage framework. In this paper, we propose a coarse-to-fine lexicon-based retriever to retrieve entity candidates in an effective manner, which operates in two layers. The first layer retrieves coarse-grained candidates by leveraging entity names, while the second layer narrows down the search to fine-grained candidates within the coarse-grained ones. In addition, this second layer utilizes entity descriptions to effectively disambiguate tail or new entities that share names with existing popular entities. Experimental results indicate that our approach can obtain superior performance without requiring extensive finetuning in the retrieval stage. Notably, our approach ranks the 1st in NLPCC 2023 Shared Task 6 on Chinese Few-shot and Zero-shot Entity Linking.

{{</citation>}}


### (23/133) Coupling Symbolic Reasoning with Language Modeling for Efficient Longitudinal Understanding of Unstructured Electronic Medical Records (Shivani Shekhar et al., 2023)

{{<citation>}}

Shivani Shekhar, Simran Tiwari, T. C. Rensink, Ramy Eskander, Wael Salloum. (2023)  
**Coupling Symbolic Reasoning with Language Modeling for Efficient Longitudinal Understanding of Unstructured Electronic Medical Records**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.03360v1)  

---


**ABSTRACT**  
The application of Artificial Intelligence (AI) in healthcare has been revolutionary, especially with the recent advancements in transformer-based Large Language Models (LLMs). However, the task of understanding unstructured electronic medical records remains a challenge given the nature of the records (e.g., disorganization, inconsistency, and redundancy) and the inability of LLMs to derive reasoning paradigms that allow for comprehensive understanding of medical variables. In this work, we examine the power of coupling symbolic reasoning with language modeling toward improved understanding of unstructured clinical texts. We show that such a combination improves the extraction of several medical variables from unstructured records. In addition, we show that the state-of-the-art commercially-free LLMs enjoy retrieval capabilities comparable to those provided by their commercial counterparts. Finally, we elaborate on the need for LLM steering through the application of symbolic reasoning as the exclusive use of LLMs results in the lowest performance.

{{</citation>}}


### (24/133) SciGraphQA: A Large-Scale Synthetic Multi-Turn Question-Answering Dataset for Scientific Graphs (Shengzhi Li et al., 2023)

{{<citation>}}

Shengzhi Li, Nima Tajbakhsh. (2023)  
**SciGraphQA: A Large-Scale Synthetic Multi-Turn Question-Answering Dataset for Scientific Graphs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2308.03349v1)  

---


**ABSTRACT**  
In this work, we present SciGraphQA, a synthetic multi-turn question-answer dataset related to academic graphs. SciGraphQA is 13 times larger than ChartVQA, the previously largest chart-visual question-answering dataset. It is also the largest open-sourced chart VQA dataset with non-synthetic charts. To build our dataset, we selected 290,000 Computer Science or Machine Learning ArXiv papers published between 2010 and 2020, and then used Palm-2 to generate 295K samples of open-vocabulary multi-turn question-answering dialogues about the graphs. As context, we provided the text-only Palm-2 with paper title, abstract, paragraph mentioning the graph, and rich text contextual data from the graph itself, obtaining dialogues with an average 2.23 question-answer turns for each graph. We asked GPT-4 to assess the matching quality of our question-answer turns given the paper's context, obtaining an average rating of 8.7/10 on our 3K test set. We evaluated the 0-shot capability of the most popular MLLM models such as LLaVa, mPLUGowl, BLIP-2, and openFlamingo's on our dataset, finding LLaVA-13B being the most performant with a CIDEr score of 0.08. We further enriched the question prompts for LLAVA by including the serialized data tables extracted from the graphs using the DePlot model, boosting LLaVA's 0-shot CIDEr to 0.15. To verify the validity of our dataset, we also fine-tuned LLaVa using our dataset, reaching a substantially higher CIDEr score of 0.26. We anticipate further accuracy improvement by including segmentation mask tokens and leveraging larger LLM backbones coupled with emergent prompting techniques. Our code and data are open-sourced.

{{</citation>}}


### (25/133) LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning (Longteng Zhang et al., 2023)

{{<citation>}}

Longteng Zhang, Lin Zhang, Shaohuai Shi, Xiaowen Chu, Bo Li. (2023)  
**LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, LLaMA, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2308.03303v1)  

---


**ABSTRACT**  
The low-rank adaptation (LoRA) method can largely reduce the amount of trainable parameters for fine-tuning large language models (LLMs), however, it still requires expensive activation memory to update low-rank weights. Reducing the number of LoRA layers or using activation recomputation could harm the fine-tuning performance or increase the computational overhead. In this work, we present LoRA-FA, a memory-efficient fine-tuning method that reduces the activation memory without performance degradation and expensive recomputation. LoRA-FA chooses to freeze the projection-down weight of $A$ and update the projection-up weight of $B$ in each LoRA layer. It ensures the change of model weight reside in a low-rank space during LLMs fine-tuning, while eliminating the requirement to store full-rank input activations. We conduct extensive experiments across multiple model types (RoBERTa, T5, LLaMA) and model scales. Our results show that LoRA-FA can always achieve close fine-tuning accuracy across different tasks compared to full parameter fine-tuning and LoRA. Furthermore, LoRA-FA can reduce the overall memory cost by up to 1.4$\times$ compared to LoRA.

{{</citation>}}


### (26/133) Dialogue Systems Can Generate Appropriate Responses without the Use of Question Marks? -- Investigation of the Effects of Question Marks on Dialogue Systems (Tomoya Mizumoto et al., 2023)

{{<citation>}}

Tomoya Mizumoto, Takato Yamazaki, Katsumasa Yoshikawa, Masaya Ohagi, Toshiki Kawamoto, Toshinori Sato. (2023)  
**Dialogue Systems Can Generate Appropriate Responses without the Use of Question Marks? -- Investigation of the Effects of Question Marks on Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.03293v1)  

---


**ABSTRACT**  
When individuals engage in spoken discourse, various phenomena can be observed that differ from those that are apparent in text-based conversation. While written communication commonly uses a question mark to denote a query, in spoken discourse, queries are frequently indicated by a rising intonation at the end of a sentence. However, numerous speech recognition engines do not append a question mark to recognized queries, presenting a challenge when creating a spoken dialogue system. Specifically, the absence of a question mark at the end of a sentence can impede the generation of appropriate responses to queries in spoken dialogue systems. Hence, we investigate the impact of question marks on dialogue systems, with the results showing that they have a significant impact. Moreover, we analyze specific examples in an effort to determine which types of utterances have the impact on dialogue systems.

{{</citation>}}


### (27/133) Towards General Text Embeddings with Multi-stage Contrastive Learning (Zehan Li et al., 2023)

{{<citation>}}

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang. (2023)  
**Towards General Text Embeddings with Multi-stage Contrastive Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Contrastive Learning, Embedding, NLP  
[Paper Link](http://arxiv.org/abs/2308.03281v1)  

---


**ABSTRACT**  
We present GTE, a general-purpose text embedding model trained with multi-stage contrastive learning. In line with recent advancements in unifying various NLP tasks into a single format, we train a unified text embedding model by employing contrastive learning over a diverse mixture of datasets from multiple sources. By significantly increasing the number of training data during both unsupervised pre-training and supervised fine-tuning stages, we achieve substantial performance gains over existing embedding models. Notably, even with a relatively modest parameter count of 110M, GTE$_\text{base}$ outperforms the black-box embedding API provided by OpenAI and even surpasses 10x larger text embedding models on the massive text embedding benchmark. Furthermore, without additional fine-tuning on each programming language individually, our model outperforms previous best code retrievers of similar size by treating code as text. In summary, our model achieves impressive results by effectively harnessing multi-stage contrastive learning, offering a powerful and efficient text embedding model with broad applicability across various NLP and code-related tasks.

{{</citation>}}


### (28/133) UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition (Wenxuan Zhou et al., 2023)

{{<citation>}}

Wenxuan Zhou, Sheng Zhang, Yu Gu, Muhao Chen, Hoifung Poon. (2023)  
**UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2308.03279v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated remarkable generalizability, such as understanding arbitrary entities and relations. Instruction tuning has proven effective for distilling LLMs into more cost-efficient models such as Alpaca and Vicuna. Yet such student models still trail the original LLMs by large margins in downstream applications. In this paper, we explore targeted distillation with mission-focused instruction tuning to train student models that can excel in a broad application class such as open information extraction. Using named entity recognition (NER) for case study, we show how ChatGPT can be distilled into much smaller UniversalNER models for open NER. For evaluation, we assemble the largest NER benchmark to date, comprising 43 datasets across 9 diverse domains such as biomedicine, programming, social media, law, finance. Without using any direct supervision, UniversalNER attains remarkable NER accuracy across tens of thousands of entity types, outperforming general instruction-tuned models such as Alpaca and Vicuna by over 30 absolute F1 points in average. With a tiny fraction of parameters, UniversalNER not only acquires ChatGPT's capability in recognizing arbitrary entity types, but also outperforms its NER accuracy by 7-9 absolute F1 points in average. Remarkably, UniversalNER even outperforms by a large margin state-of-the-art multi-task instruction-tuned systems such as InstructUIE, which uses supervised NER examples. We also conduct thorough ablation studies to assess the impact of various components in our distillation approach. We will release the distillation recipe, data, and UniversalNER models to facilitate future research on targeted distillation.

{{</citation>}}


### (29/133) From Ambiguity to Explicitness: NLP-Assisted 5G Specification Abstraction for Formal Analysis (Shiyu Yuan et al., 2023)

{{<citation>}}

Shiyu Yuan, Jingda Yang, Sudhanshu Arya, Carlo Lipizzi, Ying Wang. (2023)  
**From Ambiguity to Explicitness: NLP-Assisted 5G Specification Abstraction for Formal Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.03277v1)  

---


**ABSTRACT**  
Formal method-based analysis of the 5G Wireless Communication Protocol is crucial for identifying logical vulnerabilities and facilitating an all-encompassing security assessment, especially in the design phase. Natural Language Processing (NLP) assisted techniques and most of the tools are not widely adopted by the industry and research community. Traditional formal verification through a mathematics approach heavily relied on manual logical abstraction prone to being time-consuming, and error-prone. The reason that the NLP-assisted method did not apply in industrial research may be due to the ambiguity in the natural language of the protocol designs nature is controversial to the explicitness of formal verification. To address the challenge of adopting the formal methods in protocol designs, targeting (3GPP) protocols that are written in natural language, in this study, we propose a hybrid approach to streamline the analysis of protocols. We introduce a two-step pipeline that first uses NLP tools to construct data and then uses constructed data to extract identifiers and formal properties by using the NLP model. The identifiers and formal properties are further used for formal analysis. We implemented three models that take different dependencies between identifiers and formal properties as criteria. Our results of the optimal model reach valid accuracy of 39% for identifier extraction and 42% for formal properties predictions. Our work is proof of concept for an efficient procedure in performing formal analysis for largescale complicate specification and protocol analysis, especially for 5G and nextG communications.

{{</citation>}}


### (30/133) Adapter-based Selective Knowledge Distillation for Federated Multi-domain Meeting Summarization (Xiachong Feng et al., 2023)

{{<citation>}}

Xiachong Feng, Xiaocheng Feng, Xiyuan Du, Min-Yen Kan, Bing Qin. (2023)  
**Adapter-based Selective Knowledge Distillation for Federated Multi-domain Meeting Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Distillation, Summarization  
[Paper Link](http://arxiv.org/abs/2308.03275v1)  

---


**ABSTRACT**  
Meeting summarization has emerged as a promising technique for providing users with condensed summaries. However, existing work has focused on training models on centralized data, neglecting real-world scenarios where meeting data are infeasible to collect centrally, due to their sensitive nature. This gap motivates us to explore federated learning for meeting summarization. Two critical challenges impede progress. First, state-of-the-art summarizers are based on parameter-heavy pre-trained models. Exchanging such a model's parameters across clients imposes large bandwidth costs. Second, as real-world meeting data belong to various domains and are distributed across clients, they are instances of non-identically and independently distributed (non-IID). IID assumptions do not hold, which changes which forms of learning algorithms best apply. To address this, we propose Adapter-based Federated Selective Knowledge Distillation (AdaFedSelecKD) for training performant client models. Specifically, we develop an adapter-based summarization model where two adapters cooperatively facilitate learning using fewer parameters to reduce communication costs. Then, we devise a selective knowledge distillation strategy, assisting clients in robustly handling domain-focused modelling on their own data, while leveraging global parameters based on non-IID data. Extensive experiments on the QMSum benchmark demonstrate AdaFedSelecKD can achieve comparable performance with powerful centralized training methods, and shows its generalizability and robustness.

{{</citation>}}


### (31/133) Simple Rule Injection for ComplEx Embeddings (Haodi Ma et al., 2023)

{{<citation>}}

Haodi Ma, Anthony Colas, Yuejie Wang, Ali Sadeghian, Daisy Zhe Wang. (2023)  
**Simple Rule Injection for ComplEx Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.03269v1)  

---


**ABSTRACT**  
Recent works in neural knowledge graph inference attempt to combine logic rules with knowledge graph embeddings to benefit from prior knowledge. However, they usually cannot avoid rule grounding, and injecting a diverse set of rules has still not been thoroughly explored. In this work, we propose InjEx, a mechanism to inject multiple types of rules through simple constraints, which capture definite Horn rules. To start, we theoretically prove that InjEx can inject such rules. Next, to demonstrate that InjEx infuses interpretable prior knowledge into the embedding space, we evaluate InjEx on both the knowledge graph completion (KGC) and few-shot knowledge graph completion (FKGC) settings. Our experimental results reveal that InjEx outperforms both baseline KGC models as well as specialized few-shot models while maintaining its scalability and efficiency.

{{</citation>}}


### (32/133) PaniniQA: Enhancing Patient Education Through Interactive Question Answering (Pengshan Cai et al., 2023)

{{<citation>}}

Pengshan Cai, Zonghai Yao, Fei Liu, Dakuo Wang, Meghan Reilly, Huixue Zhou, Lingxi Li, Yi Cao, Alok Kapoor, Adarsha Bajracharya, Dan Berlowitz, Hong Yu. (2023)  
**PaniniQA: Enhancing Patient Education Through Interactive Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.03253v1)  

---


**ABSTRACT**  
Patient portal allows discharged patients to access their personalized discharge instructions in electronic health records (EHRs). However, many patients have difficulty understanding or memorizing their discharge instructions. In this paper, we present PaniniQA, a patient-centric interactive question answering system designed to help patients understand their discharge instructions. PaniniQA first identifies important clinical content from patients' discharge instructions and then formulates patient-specific educational questions. In addition, PaniniQA is also equipped with answer verification functionality to provide timely feedback to correct patients' misunderstandings. Our comprehensive automatic and human evaluation results demonstrate our PaniniQA is capable of improving patients' mastery of their medical instructions through effective interactions

{{</citation>}}


### (33/133) Analysis of the Evolution of Advanced Transformer-Based Language Models: Experiments on Opinion Mining (Nour Eddine Zekaoui et al., 2023)

{{<citation>}}

Nour Eddine Zekaoui, Siham Yousfi, Maryem Rhanoui, Mounia Mikram. (2023)  
**Analysis of the Evolution of Advanced Transformer-Based Language Models: Experiments on Opinion Mining**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03235v1)  

---


**ABSTRACT**  
Opinion mining, also known as sentiment analysis, is a subfield of natural language processing (NLP) that focuses on identifying and extracting subjective information in textual material. This can include determining the overall sentiment of a piece of text (e.g., positive or negative), as well as identifying specific emotions or opinions expressed in the text, that involves the use of advanced machine and deep learning techniques. Recently, transformer-based language models make this task of human emotion analysis intuitive, thanks to the attention mechanism and parallel computation. These advantages make such models very powerful on linguistic tasks, unlike recurrent neural networks that spend a lot of time on sequential processing, making them prone to fail when it comes to processing long text. The scope of our paper aims to study the behaviour of the cutting-edge Transformer-based language models on opinion mining and provide a high-level comparison between them to highlight their key particularities. Additionally, our comparative study shows leads and paves the way for production engineers regarding the approach to focus on and is useful for researchers as it provides guidelines for future research subjects.

{{</citation>}}


## cs.LG (21)



### (34/133) PMU measurements based short-term voltage stability assessment of power systems via deep transfer learning (Yang Li et al., 2023)

{{<citation>}}

Yang Li, Shitu Zhang, Yuanzheng Li, Jiting Cao, Shuyue Jia. (2023)  
**PMU measurements based short-term voltage stability assessment of power systems via deep transfer learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.03953v1)  

---


**ABSTRACT**  
Deep learning has emerged as an effective solution for addressing the challenges of short-term voltage stability assessment (STVSA) in power systems. However, existing deep learning-based STVSA approaches face limitations in adapting to topological changes, sample labeling, and handling small datasets. To overcome these challenges, this paper proposes a novel phasor measurement unit (PMU) measurements-based STVSA method by using deep transfer learning. The method leverages the real-time dynamic information captured by PMUs to create an initial dataset. It employs temporal ensembling for sample labeling and utilizes least squares generative adversarial networks (LSGAN) for data augmentation, enabling effective deep learning on small-scale datasets. Additionally, the method enhances adaptability to topological changes by exploring connections between different faults. Experimental results on the IEEE 39-bus test system demonstrate that the proposed method improves model evaluation accuracy by approximately 20% through transfer learning, exhibiting strong adaptability to topological changes. Leveraging the self-attention mechanism of the Transformer model, this approach offers significant advantages over shallow learning methods and other deep learning-based approaches.

{{</citation>}}


### (35/133) The Prospect of Enhancing Large-Scale Heterogeneous Federated Learning with Transformers (Yulan Gao et al., 2023)

{{<citation>}}

Yulan Gao, Hao Sun, Zengxiang Li, Han Yu. (2023)  
**The Prospect of Enhancing Large-Scale Heterogeneous Federated Learning with Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.03945v1)  

---


**ABSTRACT**  
Federated learning (FL) addresses data privacy concerns by enabling collaborative training of AI models across distributed data owners. Wide adoption of FL faces the fundamental challenges of data heterogeneity and the large scale of data owners involved. In this paper, we investigate the prospect of Transformer-based FL models for achieving generalization and personalization in this setting. We conduct extensive comparative experiments involving FL with Transformers, ResNet, and personalized ResNet-based FL approaches under various scenarios. These experiments consider varying numbers of data owners to demonstrate Transformers' advantages over deep neural networks in large-scale heterogeneous FL tasks. In addition, we analyze the superior performance of Transformers by comparing the Centered Kernel Alignment (CKA) representation similarity across different layers and FL models to gain insight into the reasons behind their promising capabilities.

{{</citation>}}


### (36/133) GraPhSyM: Graph Physical Synthesis Model (Ahmed Agiza et al., 2023)

{{<citation>}}

Ahmed Agiza, Rajarshi Roy, Teodor Dumitru Ene, Saad Godil, Sherief Reda, Bryan Catanzaro. (2023)  
**GraPhSyM: Graph Physical Synthesis Model**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs.LG  
Keywords: Attention, Graph Attention Network  
[Paper Link](http://arxiv.org/abs/2308.03944v1)  

---


**ABSTRACT**  
In this work, we introduce GraPhSyM, a Graph Attention Network (GATv2) model for fast and accurate estimation of post-physical synthesis circuit delay and area metrics from pre-physical synthesis circuit netlists. Once trained, GraPhSyM provides accurate visibility of final design metrics to early EDA stages, such as logic synthesis, without running the slow physical synthesis flow, enabling global co-optimization across stages. Additionally, the swift and precise feedback provided by GraPhSym is instrumental for machine-learning-based EDA optimization frameworks. Given a gate-level netlist of a circuit represented as a graph, GraPhSyM utilizes graph structure, connectivity, and electrical property features to predict the impact of physical synthesis transformations such as buffer insertion and gate sizing. When trained on a dataset of 6000 prefix adder designs synthesized at an aggressive delay target, GraPhSyM can accurately predict the post-synthesis delay (98.3%) and area (96.1%) metrics of unseen adders with a fast 0.22s inference time. Furthermore, we illustrate the compositionality of GraPhSyM by employing the model trained on a fixed delay target to accurately anticipate post-synthesis metrics at a variety of unseen delay targets. Lastly, we report promising generalization capabilities of the GraPhSyM model when it is evaluated on circuits different from the adders it was exclusively trained on. The results show the potential for GraPhSyM to serve as a powerful tool for advanced optimization techniques and as an oracle for EDA machine learning frameworks.

{{</citation>}}


### (37/133) Optimizing the switching operation in monoclonal antibody production: Economic MPC and reinforcement learning (Sandra A. Obiri et al., 2023)

{{<citation>}}

Sandra A. Obiri, Song Bo, Bernard T. Agyeman, Benjamin Decardi-Nelson, Jinfeng Liu. (2023)  
**Optimizing the switching operation in monoclonal antibody production: Economic MPC and reinforcement learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY, q-bio-QM  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.03928v1)  

---


**ABSTRACT**  
Monoclonal antibodies (mAbs) have emerged as indispensable assets in medicine, and are currently at the forefront of biopharmaceutical product development. However, the growing market demand and the substantial doses required for mAb clinical treatments necessitate significant progress in its large-scale production. Most of the processes for industrial mAb production rely on batch operations, which result in significant downtime. The shift towards a fully continuous and integrated manufacturing process holds the potential to boost product yield and quality, while eliminating the extra expenses associated with storing intermediate products. The integrated continuous mAb production process can be divided into the upstream and downstream processes. One crucial aspect that ensures the continuity of the integrated process is the switching of the capture columns, which are typically chromatography columns operated in a fed-batch manner downstream. Due to the discrete nature of the switching operation, advanced process control algorithms such as economic MPC (EMPC) are computationally difficult to implement. This is because an integer nonlinear program (INLP) needs to be solved online at each sampling time. This paper introduces two computationally-efficient approaches for EMPC implementation, namely, a sigmoid function approximation approach and a rectified linear unit (ReLU) approximation approach. It also explores the application of deep reinforcement learning (DRL). These three methods are compared to the traditional switching approach which is based on a 1% product breakthrough rule and which involves no optimization.

{{</citation>}}


### (38/133) Predicting and explaining nonlinear material response using deep Physically Guided Neural Networks with Internal Variables (Javier Orera-Echeverria et al., 2023)

{{<citation>}}

Javier Orera-Echeverria, Jacobo Ayensa-Jiménez, Manuel Doblare. (2023)  
**Predicting and explaining nonlinear material response using deep Physically Guided Neural Networks with Internal Variables**  

---
Primary Category: cs.LG  
Categories: 35Q74, 68T07, I-6-5, cs-LG, cs.LG, physics-comp-ph  
Keywords: AI, GNN  
[Paper Link](http://arxiv.org/abs/2308.03915v1)  

---


**ABSTRACT**  
Nonlinear materials are often difficult to model with classical state model theory because they have a complex and sometimes inaccurate physical and mathematical description or we simply do not know how to describe such materials in terms of relations between external and internal variables. In many disciplines, Neural Network methods have arisen as powerful tools to identify very complex and non-linear correlations. In this work, we use the very recently developed concept of Physically Guided Neural Networks with Internal Variables (PGNNIV) to discover constitutive laws using a model-free approach and training solely with measured force-displacement data. PGNNIVs make a particular use of the physics of the problem to enforce constraints on specific hidden layers and are able to make predictions without internal variable data. We demonstrate that PGNNIVs are capable of predicting both internal and external variables under unseen load scenarios, regardless of the nature of the material considered (linear, with hardening or softening behavior and hyperelastic), unravelling the constitutive law of the material hence explaining its nature altogether, placing the method in what is known as eXplainable Artificial Intelligence (XAI).

{{</citation>}}


### (39/133) Scalable and Equitable Math Problem Solving Strategy Prediction in Big Educational Data (Anup Shakya et al., 2023)

{{<citation>}}

Anup Shakya, Vasile Rus, Deepak Venugopal. (2023)  
**Scalable and Equitable Math Problem Solving Strategy Prediction in Big Educational Data**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2308.03892v1)  

---


**ABSTRACT**  
Understanding a student's problem-solving strategy can have a significant impact on effective math learning using Intelligent Tutoring Systems (ITSs) and Adaptive Instructional Systems (AISs). For instance, the ITS/AIS can better personalize itself to correct specific misconceptions that are indicated by incorrect strategies, specific problems can be designed to improve strategies and frustration can be minimized by adapting to a student's natural way of thinking rather than trying to fit a standard strategy for all. While it may be possible for human experts to identify strategies manually in classroom settings with sufficient student interaction, it is not possible to scale this up to big data. Therefore, we leverage advances in Machine Learning and AI methods to perform scalable strategy prediction that is also fair to students at all skill levels. Specifically, we develop an embedding called MVec where we learn a representation based on the mastery of students. We then cluster these embeddings with a non-parametric clustering method where we progressively learn clusters such that we group together instances that have approximately symmetrical strategies. The strategy prediction model is trained on instances sampled from these clusters. This ensures that we train the model over diverse strategies and also that strategies from a particular group do not bias the DNN model, thus allowing it to optimize its parameters over all groups. Using real world large-scale student interaction datasets from MATHia, we implement our approach using transformers and Node2Vec for learning the mastery embeddings and LSTMs for predicting strategies. We show that our approach can scale up to achieve high accuracy by training on a small sample of a large dataset and also has predictive equality, i.e., it can predict strategies equally well for learners at diverse skill levels.

{{</citation>}}


### (40/133) Exploiting Generalization in Offline Reinforcement Learning via Unseen State Augmentations (Nirbhay Modhe et al., 2023)

{{<citation>}}

Nirbhay Modhe, Qiaozi Gao, Ashwin Kalyan, Dhruv Batra, Govind Thattai, Gaurav Sukhatme. (2023)  
**Exploiting Generalization in Offline Reinforcement Learning via Unseen State Augmentations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Augmentation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.03882v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL) methods strike a balance between exploration and exploitation by conservative value estimation -- penalizing values of unseen states and actions. Model-free methods penalize values at all unseen actions, while model-based methods are able to further exploit unseen states via model rollouts. However, such methods are handicapped in their ability to find unseen states far away from the available offline data due to two factors -- (a) very short rollout horizons in models due to cascading model errors, and (b) model rollouts originating solely from states observed in offline data. We relax the second assumption and present a novel unseen state augmentation strategy to allow exploitation of unseen states where the learned model and value estimates generalize. Our strategy finds unseen states by value-informed perturbations of seen states followed by filtering out states with epistemic uncertainty estimates too high (high error) or too low (too similar to seen data). We observe improved performance in several offline RL tasks and find that our augmentation strategy consistently leads to overall lower average dataset Q-value estimates i.e. more conservative Q-value estimates than a baseline.

{{</citation>}}


### (41/133) Search Engine and Recommendation System for the Music Industry built with JinaAI (Ishita Gopalakrishnan et al., 2023)

{{<citation>}}

Ishita Gopalakrishnan, Sanjjushri Varshini R, Ponshriharini V. (2023)  
**Search Engine and Recommendation System for the Music Industry built with JinaAI**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03842v1)  

---


**ABSTRACT**  
One of the most intriguing debates regarding a novel task is the development of search engines and recommendation-based systems in the music industry. Studies have shown a drastic depression in the search engine fields, due to concerning factors such as speed, accuracy and the format of data given for querying. Often people face difficulty in searching for a song solely based on the title, hence a solution is proposed to complete a search analysis through a single query input and is matched with the lyrics of the songs present in the database. Hence it is essential to incorporate cutting-edge technology tools for developing a user-friendly search engine. Jina AI is an MLOps framework for building neural search engines that are utilized, in order for the user to obtain accurate results. Jina AI effectively helps to maintain and enhance the quality of performance for the search engine for the query given. An effective search engine and a recommendation system for the music industry, built with JinaAI.

{{</citation>}}


### (42/133) Dimensionality Reduction for Improving Out-of-Distribution Detection in Medical Image Segmentation (McKell Woodland et al., 2023)

{{<citation>}}

McKell Woodland, Nihil Patel, Mais Al Taie, Joshua P. Yung, Tucker J. Netherton, Ankit B. Patel, Kristy K. Brock. (2023)  
**Dimensionality Reduction for Improving Out-of-Distribution Detection in Medical Image Segmentation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.03723v1)  

---


**ABSTRACT**  
Clinically deployed segmentation models are known to fail on data outside of their training distribution. As these models perform well on most cases, it is imperative to detect out-of-distribution (OOD) images at inference to protect against automation bias. This work applies the Mahalanobis distance post hoc to the bottleneck features of a Swin UNETR model that segments the liver on T1-weighted magnetic resonance imaging. By reducing the dimensions of the bottleneck features with principal component analysis, OOD images were detected with high performance and minimal computational load.

{{</citation>}}


### (43/133) DeRisk: An Effective Deep Learning Framework for Credit Risk Prediction over Real-World Financial Data (Yancheng Liang et al., 2023)

{{<citation>}}

Yancheng Liang, Jiajie Zhang, Hui Li, Xiaochen Liu, Yi Hu, Yong Wu, Jinyao Zhang, Yongyan Liu, Yi Wu. (2023)  
**DeRisk: An Effective Deep Learning Framework for Credit Risk Prediction over Real-World Financial Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-fin-ST  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2308.03704v1)  

---


**ABSTRACT**  
Despite the tremendous advances achieved over the past years by deep learning techniques, the latest risk prediction models for industrial applications still rely on highly handtuned stage-wised statistical learning tools, such as gradient boosting and random forest methods. Different from images or languages, real-world financial data are high-dimensional, sparse, noisy and extremely imbalanced, which makes deep neural network models particularly challenging to train and fragile in practice. In this work, we propose DeRisk, an effective deep learning risk prediction framework for credit risk prediction on real-world financial data. DeRisk is the first deep risk prediction model that outperforms statistical learning approaches deployed in our company's production system. We also perform extensive ablation studies on our method to present the most critical factors for the empirical success of DeRisk.

{{</citation>}}


### (44/133) AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning (Michaël Mathieu et al., 2023)

{{<citation>}}

Michaël Mathieu, Sherjil Ozair, Srivatsan Srinivasan, Caglar Gulcehre, Shangtong Zhang, Ray Jiang, Tom Le Paine, Richard Powell, Konrad Żołna, Julian Schrittwieser, David Choi, Petko Georgiev, Daniel Toyama, Aja Huang, Roman Ring, Igor Babuschkin, Timo Ewalds, Mahyar Bordbar, Sarah Henderson, Sergio Gómez Colmenarejo, Aäron van den Oord, Wojciech Marian Czarnecki, Nando de Freitas, Oriol Vinyals. (2023)  
**AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.03526v1)  

---


**ABSTRACT**  
StarCraft II is one of the most challenging simulated reinforcement learning environments; it is partially observable, stochastic, multi-agent, and mastering StarCraft II requires strategic planning over long time horizons with real-time low-level execution. It also has an active professional competitive scene. StarCraft II is uniquely suited for advancing offline RL algorithms, both because of its challenging nature and because Blizzard has released a massive dataset of millions of StarCraft II games played by human players. This paper leverages that and establishes a benchmark, called AlphaStar Unplugged, introducing unprecedented challenges for offline reinforcement learning. We define a dataset (a subset of Blizzard's release), tools standardizing an API for machine learning methods, and an evaluation protocol. We also present baseline agents, including behavior cloning, offline variants of actor-critic and MuZero. We improve the state of the art of agents using only offline data, and we achieve 90% win rate against previously published AlphaStar behavior cloning agent.

{{</citation>}}


### (45/133) Worker Activity Recognition in Manufacturing Line Using Near-body Electric Field (Sungho Suh et al., 2023)

{{<citation>}}

Sungho Suh, Vitor Fortes Rey, Sizhen Bian, Yu-Chi Huang, Jože M. Rožanec, Hooman Tavakoli Ghinani, Bo Zhou, Paul Lukowicz. (2023)  
**Worker Activity Recognition in Manufacturing Line Using Near-body Electric Field**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2308.03514v1)  

---


**ABSTRACT**  
Manufacturing industries strive to improve production efficiency and product quality by deploying advanced sensing and control systems. Wearable sensors are emerging as a promising solution for achieving this goal, as they can provide continuous and unobtrusive monitoring of workers' activities in the manufacturing line. This paper presents a novel wearable sensing prototype that combines IMU and body capacitance sensing modules to recognize worker activities in the manufacturing line. To handle these multimodal sensor data, we propose and compare early, and late sensor data fusion approaches for multi-channel time-series convolutional neural networks and deep convolutional LSTM. We evaluate the proposed hardware and neural network model by collecting and annotating sensor data using the proposed sensing prototype and Apple Watches in the testbed of the manufacturing line. Experimental results demonstrate that our proposed methods achieve superior performance compared to the baseline methods, indicating the potential of the proposed approach for real-world applications in manufacturing industries. Furthermore, the proposed sensing prototype with a body capacitive sensor and feature fusion method improves by 6.35%, yielding a 9.38% higher macro F1 score than the proposed sensing prototype without a body capacitive sensor and Apple Watch data, respectively.

{{</citation>}}


### (46/133) Applied metamodelling for ATM performance simulations (Christoffer Riis et al., 2023)

{{<citation>}}

Christoffer Riis, Francisco N. Antunes, Tatjana Bolić, Gérald Gurtner, Andrew Cook, Carlos Lima Azevedo, Francisco Câmara Pereira. (2023)  
**Applied metamodelling for ATM performance simulations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2308.03404v1)  

---


**ABSTRACT**  
The use of Air traffic management (ATM) simulators for planing and operations can be challenging due to their modelling complexity. This paper presents XALM (eXplainable Active Learning Metamodel), a three-step framework integrating active learning and SHAP (SHapley Additive exPlanations) values into simulation metamodels for supporting ATM decision-making. XALM efficiently uncovers hidden relationships among input and output variables in ATM simulators, those usually of interest in policy analysis. Our experiments show XALM's predictive performance comparable to the XGBoost metamodel with fewer simulations. Additionally, XALM exhibits superior explanatory capabilities compared to non-active learning metamodels.   Using the `Mercury' (flight and passenger) ATM simulator, XALM is applied to a real-world scenario in Paris Charles de Gaulle airport, extending an arrival manager's range and scope by analysing six variables. This case study illustrates XALM's effectiveness in enhancing simulation interpretability and understanding variable interactions. By addressing computational challenges and improving explainability, XALM complements traditional simulation-based analyses.   Lastly, we discuss two practical approaches for reducing the computational burden of the metamodelling further: we introduce a stopping criterion for active learning based on the inherent uncertainty of the metamodel, and we show how the simulations used for the metamodel can be reused across key performance indicators, thus decreasing the overall number of simulations needed.

{{</citation>}}


### (47/133) Symmetry-Preserving Program Representations for Learning Code Semantics (Kexin Pei et al., 2023)

{{<citation>}}

Kexin Pei, Weichen Li, Qirui Jin, Shuyang Liu, Scott Geng, Lorenzo Cavallaro, Junfeng Yang, Suman Jana. (2023)  
**Symmetry-Preserving Program Representations for Learning Code Semantics**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-PL, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.03312v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown promise in automated program reasoning, a crucial aspect of many security tasks. However, existing LLM architectures for code are often borrowed from other domains like natural language processing, raising concerns about their generalization and robustness to unseen code. A key generalization challenge is to incorporate the knowledge of code semantics, including control and data flow, into the LLM architectures.   Drawing inspiration from examples of convolution layers exploiting translation symmetry, we explore how code symmetries can enhance LLM architectures for program analysis and modeling. We present a rigorous group-theoretic framework that formally defines code symmetries as semantics-preserving transformations and provides techniques for precisely reasoning about symmetry preservation within LLM architectures. Using this framework, we introduce a novel variant of self-attention that preserves program symmetries, demonstrating its effectiveness in generalization and robustness through detailed experimental evaluations across different binary and source code analysis tasks. Overall, our code symmetry framework offers rigorous and powerful reasoning techniques that can guide the future development of specialized LLMs for code and advance LLM-guided program reasoning tasks.

{{</citation>}}


### (48/133) Implicit Graph Neural Diffusion Based on Constrained Dirichlet Energy Minimization (Guoji Fu et al., 2023)

{{<citation>}}

Guoji Fu, Mohammed Haroon Dupty, Yanfei Dong, Lee Wee Sun. (2023)  
**Implicit Graph Neural Diffusion Based on Constrained Dirichlet Energy Minimization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.03306v1)  

---


**ABSTRACT**  
Implicit graph neural networks (GNNs) have emerged as a potential approach to enable GNNs to capture long-range dependencies effectively. However, poorly designed implicit GNN layers can experience over-smoothing or may have limited adaptability to learn data geometry, potentially hindering their performance in graph learning problems. To address these issues, we introduce a geometric framework to design implicit graph diffusion layers based on a parameterized graph Laplacian operator. Our framework allows learning the geometry of vertex and edge spaces, as well as the graph gradient operator from data. We further show how implicit GNN layers can be viewed as the fixed-point solution of a Dirichlet energy minimization problem and give conditions under which it may suffer from over-smoothing. To overcome the over-smoothing problem, we design our implicit graph diffusion layer as the solution of a Dirichlet energy minimization problem with constraints on vertex features, enabling it to trade off smoothing with the preservation of node feature information. With an appropriate hyperparameter set to be larger than the largest eigenvalue of the parameterized graph Laplacian, our framework guarantees a unique equilibrium and quick convergence. Our models demonstrate better performance than leading implicit and explicit GNNs on benchmark datasets for node and graph classification tasks, with substantial accuracy improvements observed for some datasets.

{{</citation>}}


### (49/133) Studying Large Language Model Generalization with Influence Functions (Roger Grosse et al., 2023)

{{<citation>}}

Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, Evan Hubinger, Kamilė Lukošiūtė, Karina Nguyen, Nicholas Joseph, Sam McCandlish, Jared Kaplan, Samuel R. Bowman. (2023)  
**Studying Large Language Model Generalization with Influence Functions**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.03296v1)  

---


**ABSTRACT**  
When trying to gain better visibility into a machine learning model in order to understand and mitigate the associated risks, a potentially valuable source of evidence is: which training examples most contribute to a given behavior? Influence functions aim to answer a counterfactual: how would the model's parameters (and hence its outputs) change if a given sequence were added to the training set? While influence functions have produced insights for small models, they are difficult to scale to large language models (LLMs) due to the difficulty of computing an inverse-Hessian-vector product (IHVP). We use the Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EK-FAC) approximation to scale influence functions up to LLMs with up to 52 billion parameters. In our experiments, EK-FAC achieves similar accuracy to traditional influence function estimators despite the IHVP computation being orders of magnitude faster. We investigate two algorithmic techniques to reduce the cost of computing gradients of candidate training sequences: TF-IDF filtering and query batching. We use influence functions to investigate the generalization patterns of LLMs, including the sparsity of the influence patterns, increasing abstraction with scale, math and programming abilities, cross-lingual generalization, and role-playing behavior. Despite many apparently sophisticated forms of generalization, we identify a surprising limitation: influences decay to near-zero when the order of key phrases is flipped. Overall, influence functions give us a powerful new tool for studying the generalization properties of LLMs.

{{</citation>}}


### (50/133) DOMINO: Domain-invariant Hyperdimensional Classification for Multi-Sensor Time Series Data (Junyao Wang et al., 2023)

{{<citation>}}

Junyao Wang, Luke Chen, Mohammad Abdullah Al Faruque. (2023)  
**DOMINO: Domain-invariant Hyperdimensional Classification for Multi-Sensor Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.03295v1)  

---


**ABSTRACT**  
With the rapid evolution of the Internet of Things, many real-world applications utilize heterogeneously connected sensors to capture time-series information. Edge-based machine learning (ML) methodologies are often employed to analyze locally collected data. However, a fundamental issue across data-driven ML approaches is distribution shift. It occurs when a model is deployed on a data distribution different from what it was trained on, and can substantially degrade model performance. Additionally, increasingly sophisticated deep neural networks (DNNs) have been proposed to capture spatial and temporal dependencies in multi-sensor time series data, requiring intensive computational resources beyond the capacity of today's edge devices. While brain-inspired hyperdimensional computing (HDC) has been introduced as a lightweight solution for edge-based learning, existing HDCs are also vulnerable to the distribution shift challenge. In this paper, we propose DOMINO, a novel HDC learning framework addressing the distribution shift problem in noisy multi-sensor time-series data. DOMINO leverages efficient and parallel matrix operations on high-dimensional space to dynamically identify and filter out domain-variant dimensions. Our evaluation on a wide range of multi-sensor time series classification tasks shows that DOMINO achieves on average 2.04% higher accuracy than state-of-the-art (SOTA) DNN-based domain generalization techniques, and delivers 7.83x faster training and 26.94x faster inference. More importantly, DOMINO performs notably better when learning from partially labeled and highly imbalanced data, providing 10.93x higher robustness against hardware noises than SOTA DNNs.

{{</citation>}}


### (51/133) SynJax: Structured Probability Distributions for JAX (Miloš Stanojević et al., 2023)

{{<citation>}}

Miloš Stanojević, Laurent Sartran. (2023)  
**SynJax: Structured Probability Distributions for JAX**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.03291v1)  

---


**ABSTRACT**  
The development of deep learning software libraries enabled significant progress in the field by allowing users to focus on modeling, while letting the library to take care of the tedious and time-consuming task of optimizing execution for modern hardware accelerators. However, this has benefited only particular types of deep learning models, such as Transformers, whose primitives map easily to the vectorized computation. The models that explicitly account for structured objects, such as trees and segmentations, did not benefit equally because they require custom algorithms that are difficult to implement in a vectorized form.   SynJax directly addresses this problem by providing an efficient vectorized implementation of inference algorithms for structured distributions covering alignment, tagging, segmentation, constituency trees and spanning trees. With SynJax we can build large-scale differentiable models that explicitly model structure in the data. The code is available at https://github.com/deepmind/synjax.

{{</citation>}}


### (52/133) DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction (Chengqing Yu et al., 2023)

{{<citation>}}

Chengqing Yu, Fei Wang, Zezhi Shao, Tao Sun, Lin Wu, Yongjun Xu. (2023)  
**DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03274v1)  

---


**ABSTRACT**  
Multivariate time series long-term prediction, which aims to predict the change of data in a long time, can provide references for decision-making. Although transformer-based models have made progress in this field, they usually do not make full use of three features of multivariate time series: global information, local information, and variables correlation. To effectively mine the above three features and establish a high-precision prediction model, we propose a double sampling transformer (DSformer), which consists of the double sampling (DS) block and the temporal variable attention (TVA) block. Firstly, the DS block employs down sampling and piecewise sampling to transform the original series into feature vectors that focus on global information and local information respectively. Then, TVA block uses temporal attention and variable attention to mine these feature vectors from different dimensions and extract key information. Finally, based on a parallel structure, DSformer uses multiple TVA blocks to mine and integrate different features obtained from DS blocks respectively. The integrated feature information is passed to the generative decoder based on a multi-layer perceptron to realize multivariate time series long-term prediction. Experimental results on nine real-world datasets show that DSformer can outperform eight existing baselines.

{{</citation>}}


### (53/133) Local Structure-aware Graph Contrastive Representation Learning (Kai Yang et al., 2023)

{{<citation>}}

Kai Yang, Yuan Liu, Zijuan Zhao, Peijin Ding, Wenqian Zhao. (2023)  
**Local Structure-aware Graph Contrastive Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning, GNN, Graph Neural Network, Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.03271v1)  

---


**ABSTRACT**  
Traditional Graph Neural Network (GNN), as a graph representation learning method, is constrained by label information. However, Graph Contrastive Learning (GCL) methods, which tackle the label problem effectively, mainly focus on the feature information of the global graph or small subgraph structure (e.g., the first-order neighborhood). In the paper, we propose a Local Structure-aware Graph Contrastive representation Learning method (LS-GCL) to model the structural information of nodes from multiple views. Specifically, we construct the semantic subgraphs that are not limited to the first-order neighbors. For the local view, the semantic subgraph of each target node is input into a shared GNN encoder to obtain the target node embeddings at the subgraph-level. Then, we use a pooling function to generate the subgraph-level graph embeddings. For the global view, considering the original graph preserves indispensable semantic information of nodes, we leverage the shared GNN encoder to learn the target node embeddings at the global graph-level. The proposed LS-GCL model is optimized to maximize the common information among similar instances at three various perspectives through a multi-level contrastive loss function. Experimental results on five datasets illustrate that our method outperforms state-of-the-art graph representation learning approaches for both node classification and link prediction tasks.

{{</citation>}}


### (54/133) Exploring Different Time-series-Transformer (TST) Architectures: A Case Study in Battery Life Prediction for Electric Vehicles (EVs) (Niranjan Sitapure et al., 2023)

{{<citation>}}

Niranjan Sitapure, Atharva Kulkarni. (2023)  
**Exploring Different Time-series-Transformer (TST) Architectures: A Case Study in Battery Life Prediction for Electric Vehicles (EVs)**  

---
Primary Category: cs.LG  
Categories: J-2; J-6; I-6, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03260v1)  

---


**ABSTRACT**  
In recent years, battery technology for electric vehicles (EVs) has been a major focus, with a significant emphasis on developing new battery materials and chemistries. However, accurately predicting key battery parameters, such as state-of-charge (SOC) and temperature, remains a challenge for constructing advanced battery management systems (BMS). Existing battery models do not comprehensively cover all parameters affecting battery performance, including non-battery-related factors like ambient temperature, cabin temperature, elevation, and regenerative braking during EV operation. Due to the difficulty of incorporating these auxiliary parameters into traditional models, a data-driven approach is suggested. Time-series-transformers (TSTs), leveraging multiheaded attention and parallelization-friendly architecture, are explored alongside LSTM models. Novel TST architectures, including encoder TST + decoder LSTM and a hybrid TST-LSTM, are also developed and compared against existing models. A dataset comprising 72 driving trips in a BMW i3 (60 Ah) is used to address battery life prediction in EVs, aiming to create accurate TST models that incorporate environmental, battery, vehicle driving, and heating circuit data to predict SOC and battery temperature for future time steps.

{{</citation>}}


## cs.CR (6)



### (55/133) Exploring Security Practices in Infrastructure as Code: An Empirical Study (Alexandre Verdet et al., 2023)

{{<citation>}}

Alexandre Verdet, Mohammad Hamdaqa, Leuson Da Silva, Foutse Khomh. (2023)  
**Exploring Security Practices in Infrastructure as Code: An Empirical Study**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: AWS, Azure, Google, Security  
[Paper Link](http://arxiv.org/abs/2308.03952v1)  

---


**ABSTRACT**  
Cloud computing has become popular thanks to the widespread use of Infrastructure as Code (IaC) tools, allowing the community to conveniently manage and configure cloud infrastructure using scripts. However, the scripting process itself does not automatically prevent practitioners from introducing misconfigurations, vulnerabilities, or privacy risks. As a result, ensuring security relies on practitioners understanding and the adoption of explicit policies, guidelines, or best practices. In order to understand how practitioners deal with this problem, in this work, we perform an empirical study analyzing the adoption of IaC scripted security best practices. First, we select and categorize widely recognized Terraform security practices promulgated in the industry for popular cloud providers such as AWS, Azure, and Google Cloud. Next, we assess the adoption of these practices by each cloud provider, analyzing a sample of 812 open-source projects hosted on GitHub. For that, we scan each project configuration files, looking for policy implementation through static analysis (checkov). Additionally, we investigate GitHub measures that might be correlated with adopting these best practices. The category Access policy emerges as the most widely adopted in all providers, while Encryption in rest are the most neglected policies. Regarding GitHub measures correlated with best practice adoption, we observe a positive, strong correlation between a repository number of stars and adopting practices in its cloud infrastructure. Based on our findings, we provide guidelines for cloud practitioners to limit infrastructure vulnerability and discuss further aspects associated with policies that have yet to be extensively embraced within the industry.

{{</citation>}}


### (56/133) 'Do Anything Now': Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models (Xinyue Shen et al., 2023)

{{<citation>}}

Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang. (2023)  
**'Do Anything Now': Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03825v1)  

---


**ABSTRACT**  
The misuse of large language models (LLMs) has garnered significant attention from the general public and LLM vendors. In response, efforts have been made to align LLMs with human values and intent use. However, a particular type of adversarial prompts, known as jailbreak prompt, has emerged and continuously evolved to bypass the safeguards and elicit harmful content from LLMs. In this paper, we conduct the first measurement study on jailbreak prompts in the wild, with 6,387 prompts collected from four platforms over six months. Leveraging natural language processing technologies and graph-based community detection methods, we discover unique characteristics of jailbreak prompts and their major attack strategies, such as prompt injection and privilege escalation. We also observe that jailbreak prompts increasingly shift from public platforms to private ones, posing new challenges for LLM vendors in proactive detection. To assess the potential harm caused by jailbreak prompts, we create a question set comprising 46,800 samples across 13 forbidden scenarios. Our experiments show that current LLMs and safeguards cannot adequately defend jailbreak prompts in all scenarios. Particularly, we identify two highly effective jailbreak prompts which achieve 0.99 attack success rates on ChatGPT (GPT-3.5) and GPT-4, and they have persisted online for over 100 days. Our work sheds light on the severe and evolving threat landscape of jailbreak prompts. We hope our study can facilitate the research community and LLM vendors in promoting safer and regulated LLMs.

{{</citation>}}


### (57/133) Mondrian: Prompt Abstraction Attack Against Large Language Models for Cheaper API Pricing (Wai Man Si et al., 2023)

{{<citation>}}

Wai Man Si, Michael Backes, Yang Zhang. (2023)  
**Mondrian: Prompt Abstraction Attack Against Large Language Models for Cheaper API Pricing**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03558v1)  

---


**ABSTRACT**  
The Machine Learning as a Service (MLaaS) market is rapidly expanding and becoming more mature. For example, OpenAI's ChatGPT is an advanced large language model (LLM) that generates responses for various queries with associated fees. Although these models can deliver satisfactory performance, they are far from perfect. Researchers have long studied the vulnerabilities and limitations of LLMs, such as adversarial attacks and model toxicity. Inevitably, commercial ML models are also not exempt from such issues, which can be problematic as MLaaS continues to grow. In this paper, we discover a new attack strategy against LLM APIs, namely the prompt abstraction attack. Specifically, we propose Mondrian, a simple and straightforward method that abstracts sentences, which can lower the cost of using LLM APIs. In this approach, the adversary first creates a pseudo API (with a lower established price) to serve as the proxy of the target API (with a higher established price). Next, the pseudo API leverages Mondrian to modify the user query, obtain the abstracted response from the target API, and forward it back to the end user. Our results show that Mondrian successfully reduces user queries' token length ranging from 13% to 23% across various tasks, including text classification, generation, and question answering. Meanwhile, these abstracted queries do not significantly affect the utility of task-specific and general language models like ChatGPT. Mondrian also reduces instruction prompts' token length by at least 11% without compromising output quality. As a result, the prompt abstraction attack enables the adversary to profit without bearing the cost of API development and deployment.

{{</citation>}}


### (58/133) TemporalFED: Detecting Cyberattacks in Industrial Time-Series Data Using Decentralized Federated Learning (Ángel Luis Perales Gómez et al., 2023)

{{<citation>}}

Ángel Luis Perales Gómez, Enrique Tomás Martínez Beltrán, Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán. (2023)  
**TemporalFED: Detecting Cyberattacks in Industrial Time-Series Data Using Decentralized Federated Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.03554v1)  

---


**ABSTRACT**  
Industry 4.0 has brought numerous advantages, such as increasing productivity through automation. However, it also presents major cybersecurity issues such as cyberattacks affecting industrial processes. Federated Learning (FL) combined with time-series analysis is a promising cyberattack detection mechanism proposed in the literature. However, the fact of having a single point of failure and network bottleneck are critical challenges that need to be tackled. Thus, this article explores the benefits of the Decentralized Federated Learning (DFL) in terms of cyberattack detection and resource consumption. The work presents TemporalFED, a software module for detecting anomalies in industrial environments using FL paradigms and time series. TemporalFED incorporates three components: Time Series Conversion, Feature Engineering, and Time Series Stationary Conversion. To evaluate TemporalFED, it was deployed on Fedstellar, a DFL framework. Then, a pool of experiments measured the detection performance and resource consumption in a chemical gas industrial environment with different time-series configurations, FL paradigms, and topologies. The results showcase the superiority of the configuration utilizing DFL and Semi-Decentralized Federated Learning (SDFL) paradigms, along with a fully connected topology, which achieved the best performance in anomaly detection. Regarding resource consumption, the configuration without feature engineering employed less bandwidth, CPU, and RAM than other configurations.

{{</citation>}}


### (59/133) Network Security in the Industrial Control System: A Survey (Yang Li et al., 2023)

{{<citation>}}

Yang Li, Shihao Wu, Quan Pan. (2023)  
**Network Security in the Industrial Control System: A Survey**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Network Security, Security  
[Paper Link](http://arxiv.org/abs/2308.03478v1)  

---


**ABSTRACT**  
Along with the development of intelligent manufacturing, especially with the high connectivity of the industrial control system (ICS), the network security of ICS becomes more important. And in recent years, there has been much research on the security of the ICS network. However, in practical usage, there are many types of protocols, which means a high vulnerability in protocols. Therefore, in this paper, we give a complete review of the protocols that are usually used in ICS. Then, we give a comprehensive review on network security in terms of Defence in Depth (DiD), including data encryption, access control policy, intrusion detection system, software-defined network, etc. Through these works, we try to provide a new perspective on the exciting new developments in this field.

{{</citation>}}


### (60/133) When GPT Meets Program Analysis: Towards Intelligent Detection of Smart Contract Logic Vulnerabilities in GPTScan (Yuqiang Sun et al., 2023)

{{<citation>}}

Yuqiang Sun, Daoyuan Wu, Yue Xue, Han Liu, Haijun Wang, Zhengzi Xu, Xiaofei Xie, Yang Liu. (2023)  
**When GPT Meets Program Analysis: Towards Intelligent Detection of Smart Contract Logic Vulnerabilities in GPTScan**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-SE, cs.CR  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03314v1)  

---


**ABSTRACT**  
Smart contracts are prone to various vulnerabilities, leading to substantial financial losses over time. Current analysis tools mainly target vulnerabilities with fixed control or dataflow patterns, such as re-entrancy and integer overflow. However, a recent study on Web3 security bugs revealed that about 80% of these bugs cannot be audited by existing tools due to the lack of domain-specific property description and checking. Given recent advances in Generative Pretraining Transformer (GPT), it is worth exploring how GPT could aid in detecting logic vulnerabilities in smart contracts. In this paper, we propose GPTScan, the first tool combining GPT with static analysis for smart contract logic vulnerability detection. Instead of relying solely on GPT to identify vulnerabilities, which can lead to high false positives and is limited by GPT's pre-trained knowledge, we utilize GPT as a versatile code understanding tool. By breaking down each logic vulnerability type into scenarios and properties, GPTScan matches candidate vulnerabilities with GPT. To enhance accuracy, GPTScan further instructs GPT to intelligently recognize key variables and statements, which are then validated by static confirmation. Evaluation on diverse datasets with around 400 contract projects and 3K Solidity files shows that GPTScan achieves high precision (over 90%) for token contracts and acceptable precision (57.14%) for large projects like Web3Bugs. It effectively detects groundtruth logic vulnerabilities with a recall of over 80%, including 9 new vulnerabilities missed by human auditors. GPTScan is fast and cost-effective, taking an average of 14.39 seconds and 0.01 USD to scan per thousand lines of Solidity code. Moreover, static confirmation helps GPTScan reduce two-thirds of false positives.

{{</citation>}}


## cs.CV (29)



### (61/133) ALFA -- Leveraging All Levels of Feature Abstraction for Enhancing the Generalization of Histopathology Image Classification Across Unseen Hospitals (Milad Sikaroudi et al., 2023)

{{<citation>}}

Milad Sikaroudi, Maryam Hosseini, Shahryar Rahnamayan, H. R. Tizhoosh. (2023)  
**ALFA -- Leveraging All Levels of Feature Abstraction for Enhancing the Generalization of Histopathology Image Classification Across Unseen Hospitals**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2308.03936v2)  

---


**ABSTRACT**  
We propose an exhaustive methodology that leverages all levels of feature abstraction, targeting an enhancement in the generalizability of image classification to unobserved hospitals. Our approach incorporates augmentation-based self-supervision with common distribution shifts in histopathology scenarios serving as the pretext task. This enables us to derive invariant features from training images without relying on training labels, thereby covering different abstraction levels. Moving onto the subsequent abstraction level, we employ a domain alignment module to facilitate further extraction of invariant features across varying training hospitals. To represent the highly specific features of participating hospitals, an encoder is trained to classify hospital labels, independent of their diagnostic labels. The features from each of these encoders are subsequently disentangled to minimize redundancy and segregate the features. This representation, which spans a broad spectrum of semantic information, enables the development of a model demonstrating increased robustness to unseen images from disparate distributions. Experimental results from the PACS dataset (a domain generalization benchmark), a synthetic dataset created by applying histopathology-specific jitters to the MHIST dataset (defining different domains with varied distribution shifts), and a Renal Cell Carcinoma dataset derived from four image repositories from TCGA, collectively indicate that our proposed model is adept at managing varying levels of image granularity. Thus, it shows improved generalizability when faced with new, out-of-distribution hospital images.

{{</citation>}}


### (62/133) ViLP: Knowledge Exploration using Vision, Language, and Pose Embeddings for Video Action Recognition (Soumyabrata Chaudhuri et al., 2023)

{{<citation>}}

Soumyabrata Chaudhuri, Saumik Bhattacharya. (2023)  
**ViLP: Knowledge Exploration using Vision, Language, and Pose Embeddings for Video Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.03908v1)  

---


**ABSTRACT**  
Video Action Recognition (VAR) is a challenging task due to its inherent complexities. Though different approaches have been explored in the literature, designing a unified framework to recognize a large number of human actions is still a challenging problem. Recently, Multi-Modal Learning (MML) has demonstrated promising results in this domain. In literature, 2D skeleton or pose modality has often been used for this task, either independently or in conjunction with the visual information (RGB modality) present in videos. However, the combination of pose, visual information, and text attributes has not been explored yet, though text and pose attributes independently have been proven to be effective in numerous computer vision tasks. In this paper, we present the first pose augmented Vision-language model (VLM) for VAR. Notably, our scheme achieves an accuracy of 92.81% and 73.02% on two popular human video action recognition benchmark datasets, UCF-101 and HMDB-51, respectively, even without any video data pre-training, and an accuracy of 96.11% and 75.75% after kinetics pre-training.

{{</citation>}}


### (63/133) TIJO: Trigger Inversion with Joint Optimization for Defending Multimodal Backdoored Models (Indranil Sur et al., 2023)

{{<citation>}}

Indranil Sur, Karan Sikka, Matthew Walmer, Kaushik Koneripalli, Anirban Roy, Xiao Lin, Ajay Divakaran, Susmit Jha. (2023)  
**TIJO: Trigger Inversion with Joint Optimization for Defending Multimodal Backdoored Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.03906v1)  

---


**ABSTRACT**  
We present a Multimodal Backdoor Defense technique TIJO (Trigger Inversion using Joint Optimization). Recent work arXiv:2112.07668 has demonstrated successful backdoor attacks on multimodal models for the Visual Question Answering task. Their dual-key backdoor trigger is split across two modalities (image and text), such that the backdoor is activated if and only if the trigger is present in both modalities. We propose TIJO that defends against dual-key attacks through a joint optimization that reverse-engineers the trigger in both the image and text modalities. This joint optimization is challenging in multimodal models due to the disconnected nature of the visual pipeline which consists of an offline feature extractor, whose output is then fused with the text using a fusion module. The key insight enabling the joint optimization in TIJO is that the trigger inversion needs to be carried out in the object detection box feature space as opposed to the pixel space. We demonstrate the effectiveness of our method on the TrojVQA benchmark, where TIJO improves upon the state-of-the-art unimodal methods from an AUC of 0.6 to 0.92 on multimodal dual-key backdoors. Furthermore, our method also improves upon the unimodal baselines on unimodal backdoors. We present ablation studies and qualitative results to provide insights into our algorithm such as the critical importance of overlaying the inverted feature triggers on all visual features during trigger inversion. The prototype implementation of TIJO is available at https://github.com/SRI-CSL/TIJO.

{{</citation>}}


### (64/133) FSD V2: Improving Fully Sparse 3D Object Detection with Virtual Voxels (Lue Fan et al., 2023)

{{<citation>}}

Lue Fan, Feng Wang, Naiyan Wang, Zhaoxiang Zhang. (2023)  
**FSD V2: Improving Fully Sparse 3D Object Detection with Virtual Voxels**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.03755v1)  

---


**ABSTRACT**  
LiDAR-based fully sparse architecture has garnered increasing attention. FSDv1 stands out as a representative work, achieving impressive efficacy and efficiency, albeit with intricate structures and handcrafted designs. In this paper, we present FSDv2, an evolution that aims to simplify the previous FSDv1 while eliminating the inductive bias introduced by its handcrafted instance-level representation, thus promoting better general applicability. To this end, we introduce the concept of \textbf{virtual voxels}, which takes over the clustering-based instance segmentation in FSDv1. Virtual voxels not only address the notorious issue of the Center Feature Missing problem in fully sparse detectors but also endow the framework with a more elegant and streamlined approach. Consequently, we develop a suite of components to complement the virtual voxel concept, including a virtual voxel encoder, a virtual voxel mixer, and a virtual voxel assignment strategy. Through empirical validation, we demonstrate that the virtual voxel mechanism is functionally similar to the handcrafted clustering in FSDv1 while being more general. We conduct experiments on three large-scale datasets: Waymo Open Dataset, Argoverse 2 dataset, and nuScenes dataset. Our results showcase state-of-the-art performance on all three datasets, highlighting the superiority of FSDv2 in long-range scenarios and its general applicability to achieve competitive performance across diverse scenarios. Moreover, we provide comprehensive experimental analysis to elucidate the workings of FSDv2. To foster reproducibility and further research, we have open-sourced FSDv2 at https://github.com/tusen-ai/SST.

{{</citation>}}


### (65/133) Recurrent Multi-scale Transformer for High-Resolution Salient Object Detection (Xinhao Deng et al., 2023)

{{<citation>}}

Xinhao Deng, Pingping Zhang, Wei Liu, Huchuan Lu. (2023)  
**Recurrent Multi-scale Transformer for High-Resolution Salient Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MM, cs.CV  
Keywords: Object Detection, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.03826v1)  

---


**ABSTRACT**  
Salient Object Detection (SOD) aims to identify and segment the most conspicuous objects in an image or video. As an important pre-processing step, it has many potential applications in multimedia and vision tasks. With the advance of imaging devices, SOD with high-resolution images is of great demand, recently. However, traditional SOD methods are largely limited to low-resolution images, making them difficult to adapt to the development of High-Resolution SOD (HRSOD). Although some HRSOD methods emerge, there are no large enough datasets for training and evaluating. Besides, current HRSOD methods generally produce incomplete object regions and irregular object boundaries. To address above issues, in this work, we first propose a new HRS10K dataset, which contains 10,500 high-quality annotated images at 2K-8K resolution. As far as we know, it is the largest dataset for the HRSOD task, which will significantly help future works in training and evaluating models. Furthermore, to improve the HRSOD performance, we propose a novel Recurrent Multi-scale Transformer (RMFormer), which recurrently utilizes shared Transformers and multi-scale refinement architectures. Thus, high-resolution saliency maps can be generated with the guidance of lower-resolution predictions. Extensive experiments on both high-resolution and low-resolution benchmarks show the effectiveness and superiority of the proposed framework. The source code and dataset are released at: https://github.com/DrowsyMon/RMFormer.

{{</citation>}}


### (66/133) Tiny LVLM-eHub: Early Multimodal Experiments with Bard (Wenqi Shao et al., 2023)

{{<citation>}}

Wenqi Shao, Yutao Hu, Peng Gao, Meng Lei, Kaipeng Zhang, Fanqing Meng, Peng Xu, Siyuan Huang, Hongsheng Li, Yu Qiao, Ping Luo. (2023)  
**Tiny LVLM-eHub: Early Multimodal Experiments with Bard**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ChatGPT, GPT, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03729v1)  

---


**ABSTRACT**  
Recent advancements in Large Vision-Language Models (LVLMs) have demonstrated significant progress in tackling complex multimodal tasks. Among these cutting-edge developments, Google's Bard stands out for its remarkable multimodal capabilities, promoting comprehensive comprehension and reasoning across various domains. This work presents an early and holistic evaluation of LVLMs' multimodal abilities, with a particular focus on Bard, by proposing a lightweight variant of LVLM-eHub, named Tiny LVLM-eHub. In comparison to the vanilla version, Tiny LVLM-eHub possesses several appealing properties. Firstly, it provides a systematic assessment of six categories of multimodal capabilities, including visual perception, visual knowledge acquisition, visual reasoning, visual commonsense, object hallucination, and embodied intelligence, through quantitative evaluation of $42$ standard text-related visual benchmarks. Secondly, it conducts an in-depth analysis of LVLMs' predictions using the ChatGPT Ensemble Evaluation (CEE), which leads to a robust and accurate evaluation and exhibits improved alignment with human evaluation compared to the word matching approach. Thirdly, it comprises a mere $2.1$K image-text pairs, facilitating ease of use for practitioners to evaluate their own offline LVLMs. Through extensive experimental analysis, this study demonstrates that Bard outperforms previous LVLMs in most multimodal capabilities except object hallucination, to which Bard is still susceptible. Tiny LVLM-eHub serves as a baseline evaluation for various LVLMs and encourages innovative strategies aimed at advancing multimodal techniques. Our project is publicly available at \url{https://github.com/OpenGVLab/Multi-Modality-Arena}.

{{</citation>}}


### (67/133) Efficient Temporal Sentence Grounding in Videos with Multi-Teacher Knowledge Distillation (Renjie Liang et al., 2023)

{{<citation>}}

Renjie Liang, Yiming Yang, Hui Lu, Li Li. (2023)  
**Efficient Temporal Sentence Grounding in Videos with Multi-Teacher Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.03725v1)  

---


**ABSTRACT**  
Temporal Sentence Grounding in Videos (TSGV) aims to detect the event timestamps described by the natural language query from untrimmed videos. This paper discusses the challenge of achieving efficient computation in TSGV models while maintaining high performance. Most existing approaches exquisitely design complex architectures to improve accuracy with extra layers and loss, suffering from inefficiency and heaviness. Although some works have noticed that, they only make an issue of feature fusion layers, which can hardly enjoy the highspeed merit in the whole clunky network. To tackle this problem, we propose a novel efficient multi-teacher model (EMTM) based on knowledge distillation to transfer diverse knowledge from both heterogeneous and isomorphic networks. Specifically, We first unify different outputs of the heterogeneous models into one single form. Next, a Knowledge Aggregation Unit (KAU) is built to acquire high-quality integrated soft labels from multiple teachers. After that, the KAU module leverages the multi-scale video and global query information to adaptively determine the weights of different teachers. A Shared Encoder strategy is then proposed to solve the problem that the student shallow layers hardly benefit from teachers, in which an isomorphic teacher is collaboratively trained with the student to align their hidden states. Extensive experimental results on three popular TSGV benchmarks demonstrate that our method is both effective and efficient without bells and whistles.

{{</citation>}}


### (68/133) Scaling may be all you need for achieving human-level object recognition capacity with human-like visual experience (A. Emin Orhan, 2023)

{{<citation>}}

A. Emin Orhan. (2023)  
**Scaling may be all you need for achieving human-level object recognition capacity with human-like visual experience**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-NE, cs.CV, q-bio-NC  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.03712v2)  

---


**ABSTRACT**  
This paper asks whether current self-supervised learning methods, if sufficiently scaled up, would be able to reach human-level visual object recognition capabilities with the same type and amount of visual experience humans learn from. Previous work on this question only considered the scaling of data size. Here, we consider the simultaneous scaling of data size, model size, and image resolution. We perform a scaling experiment with vision transformers up to 633M parameters in size (ViT-H/14) trained with up to 5K hours of human-like video data (long, continuous, mostly egocentric videos) with image resolutions of up to 476x476 pixels. The efficiency of masked autoencoders (MAEs) as a self-supervised learning algorithm makes it possible to run this scaling experiment on an unassuming academic budget. We find that it is feasible to reach human-level object recognition capacity at sub-human scales of model size, data size, and image size, if these factors are scaled up simultaneously. To give a concrete example, we estimate that a 2.5B parameter ViT model trained with 20K hours (2.3 years) of human-like video data with a spatial resolution of 952x952 pixels should be able to reach roughly human-level accuracy on ImageNet. Human-level competence is thus achievable for a fundamental perceptual capability from human-like perceptual experience (human-like in both amount and type) with extremely generic learning algorithms and architectures and without any substantive inductive biases.

{{</citation>}}


### (69/133) Video-based Person Re-identification with Long Short-Term Representation Learning (Xuehu Liu et al., 2023)

{{<citation>}}

Xuehu Liu, Pingping Zhang, Huchuan Lu. (2023)  
**Video-based Person Re-identification with Long Short-Term Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.03703v1)  

---


**ABSTRACT**  
Video-based person Re-Identification (V-ReID) aims to retrieve specific persons from raw videos captured by non-overlapped cameras. As a fundamental task, it spreads many multimedia and computer vision applications. However, due to the variations of persons and scenes, there are still many obstacles that must be overcome for high performance. In this work, we notice that both the long-term and short-term information of persons are important for robust video representations. Thus, we propose a novel deep learning framework named Long Short-Term Representation Learning (LSTRL) for effective V-ReID. More specifically, to extract long-term representations, we propose a Multi-granularity Appearance Extractor (MAE), in which four granularity appearances are effectively captured across multiple frames. Meanwhile, to extract short-term representations, we propose a Bi-direction Motion Estimator (BME), in which reciprocal motion information is efficiently extracted from consecutive frames. The MAE and BME are plug-and-play and can be easily inserted into existing networks for efficient feature learning. As a result, they significantly improve the feature representation ability for V-ReID. Extensive experiments on three widely used benchmarks show that our proposed approach can deliver better performances than most state-of-the-arts.

{{</citation>}}


### (70/133) Learning Concise and Descriptive Attributes for Visual Recognition (An Yan et al., 2023)

{{<citation>}}

An Yan, Yu Wang, Yiwu Zhong, Chengyu Dong, Zexue He, Yujie Lu, William Wang, Jingbo Shang, Julian McAuley. (2023)  
**Learning Concise and Descriptive Attributes for Visual Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.03685v1)  

---


**ABSTRACT**  
Recent advances in foundation models present new opportunities for interpretable visual recognition -- one can first query Large Language Models (LLMs) to obtain a set of attributes that describe each class, then apply vision-language models to classify images via these attributes. Pioneering work shows that querying thousands of attributes can achieve performance competitive with image features. However, our further investigation on 8 datasets reveals that LLM-generated attributes in a large quantity perform almost the same as random words. This surprising finding suggests that significant noise may be present in these attributes. We hypothesize that there exist subsets of attributes that can maintain the classification performance with much smaller sizes, and propose a novel learning-to-search method to discover those concise sets of attributes. As a result, on the CUB dataset, our method achieves performance close to that of massive LLM-generated attributes (e.g., 10k attributes for CUB), yet using only 32 attributes in total to distinguish 200 bird species. Furthermore, our new paradigm demonstrates several additional benefits: higher interpretability and interactivity for humans, and the ability to summarize knowledge for a recognition task.

{{</citation>}}


### (71/133) Improving FHB Screening in Wheat Breeding Using an Efficient Transformer Model (Babak Azad et al., 2023)

{{<citation>}}

Babak Azad, Ahmed Abdalla, Kwanghee Won, Ali Mirzakhani Nafchi. (2023)  
**Improving FHB Screening in Wheat Breeding Using an Efficient Transformer Model**  

---
Primary Category: cs.CV  
Categories: 68T07, 68T10, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.03670v1)  

---


**ABSTRACT**  
Fusarium head blight is a devastating disease that causes significant economic losses annually on small grains. Efficiency, accuracy, and timely detection of FHB in the resistance screening are critical for wheat and barley breeding programs. In recent years, various image processing techniques have been developed using supervised machine learning algorithms for the early detection of FHB. The state-of-the-art convolutional neural network-based methods, such as U-Net, employ a series of encoding blocks to create a local representation and a series of decoding blocks to capture the semantic relations. However, these methods are not often capable of long-range modeling dependencies inside the input data, and their ability to model multi-scale objects with significant variations in texture and shape is limited. Vision transformers as alternative architectures with innate global self-attention mechanisms for sequence-to-sequence prediction, due to insufficient low-level details, may also limit localization capabilities. To overcome these limitations, a new Context Bridge is proposed to integrate the local representation capability of the U-Net network in the transformer model. In addition, the standard attention mechanism of the original transformer is replaced with Efficient Self-attention, which is less complicated than other state-of-the-art methods. To train the proposed network, 12,000 wheat images from an FHB-inoculated wheat field at the SDSU research farm in Volga, SD, were captured. In addition to healthy and unhealthy plants, these images encompass various stages of the disease. A team of expert pathologists annotated the images for training and evaluating the developed model. As a result, the effectiveness of the transformer-based method for FHB-disease detection, through extensive experiments across typical tasks for plant image segmentation, is demonstrated.

{{</citation>}}


### (72/133) Segmentation Framework for Heat Loss Identification in Thermal Images: Empowering Scottish Retrofitting and Thermographic Survey Companies (Md Junayed Hasan et al., 2023)

{{<citation>}}

Md Junayed Hasan, Eyad Elyan, Yijun Yan, Jinchang Ren, Md Mostafa Kamal Sarker. (2023)  
**Segmentation Framework for Heat Loss Identification in Thermal Images: Empowering Scottish Retrofitting and Thermographic Survey Companies**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03631v1)  

---


**ABSTRACT**  
Retrofitting and thermographic survey (TS) companies in Scotland collaborate with social housing providers to tackle fuel poverty. They employ ground-level infrared (IR) camera-based-TSs (GIRTSs) for collecting thermal images to identi-fy the heat loss sources resulting from poor insulation. However, this identifica-tion process is labor-intensive and time-consuming, necessitating extensive data processing. To automate this, an AI-driven approach is necessary. Therefore, this study proposes a deep learning (DL)-based segmentation framework using the Mask Region Proposal Convolutional Neural Network (Mask RCNN) to validate its applicability to these thermal images. The objective of the framework is to au-tomatically identify, and crop heat loss sources caused by weak insulation, while also eliminating obstructive objects present in those images. By doing so, it min-imizes labor-intensive tasks and provides an automated, consistent, and reliable solution. To validate the proposed framework, approximately 2500 thermal imag-es were collected in collaboration with industrial TS partner. Then, 1800 repre-sentative images were carefully selected with the assistance of experts and anno-tated to highlight the target objects (TO) to form the final dataset. Subsequently, a transfer learning strategy was employed to train the dataset, progressively aug-menting the training data volume and fine-tuning the pre-trained baseline Mask RCNN. As a result, the final fine-tuned model achieved a mean average precision (mAP) score of 77.2% for segmenting the TO, demonstrating the significant po-tential of proposed framework in accurately quantifying energy loss in Scottish homes.

{{</citation>}}


### (73/133) Recurrent Self-Supervised Video Denoising with Denser Receptive Field (Zichun Wang et al., 2023)

{{<citation>}}

Zichun Wang, Yulun Zhang, Debing Zhang, Ying Fu. (2023)  
**Recurrent Self-Supervised Video Denoising with Denser Receptive Field**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.03608v1)  

---


**ABSTRACT**  
Self-supervised video denoising has seen decent progress through the use of blind spot networks. However, under their blind spot constraints, previous self-supervised video denoising methods suffer from significant information loss and texture destruction in either the whole reference frame or neighbor frames, due to their inadequate consideration of the receptive field. Moreover, the limited number of available neighbor frames in previous methods leads to the discarding of distant temporal information. Nonetheless, simply adopting existing recurrent frameworks does not work, since they easily break the constraints on the receptive field imposed by self-supervision. In this paper, we propose RDRF for self-supervised video denoising, which not only fully exploits both the reference and neighbor frames with a denser receptive field, but also better leverages the temporal information from both local and distant neighbor features. First, towards a comprehensive utilization of information from both reference and neighbor frames, RDRF realizes a denser receptive field by taking more neighbor pixels along the spatial and temporal dimensions. Second, it features a self-supervised recurrent video denoising framework, which concurrently integrates distant and near-neighbor temporal features. This enables long-term bidirectional information aggregation, while mitigating error accumulation in the plain recurrent framework. Our method exhibits superior performance on both synthetic and real video denoising datasets. Codes will be available at https://github.com/Wang-XIaoDingdd/RDRF.

{{</citation>}}


### (74/133) FeatEnHancer: Enhancing Hierarchical Features for Object Detection and Beyond Under Low-Light Vision (Khurram Azeem Hashmi et al., 2023)

{{<citation>}}

Khurram Azeem Hashmi, Goutham Kallempudi, Didier Stricker, Muhammamd Zeshan Afzal. (2023)  
**FeatEnHancer: Enhancing Hierarchical Features for Object Detection and Beyond Under Low-Light Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.03594v1)  

---


**ABSTRACT**  
Extracting useful visual cues for the downstream tasks is especially challenging under low-light vision. Prior works create enhanced representations by either correlating visual quality with machine perception or designing illumination-degrading transformation methods that require pre-training on synthetic datasets. We argue that optimizing enhanced image representation pertaining to the loss of the downstream task can result in more expressive representations. Therefore, in this work, we propose a novel module, FeatEnHancer, that hierarchically combines multiscale features using multiheaded attention guided by task-related loss function to create suitable representations. Furthermore, our intra-scale enhancement improves the quality of features extracted at each scale or level, as well as combines features from different scales in a way that reflects their relative importance for the task at hand. FeatEnHancer is a general-purpose plug-and-play module and can be incorporated into any low-light vision pipeline. We show with extensive experimentation that the enhanced representation produced with FeatEnHancer significantly and consistently improves results in several low-light vision tasks, including dark object detection (+5.7 mAP on ExDark), face detection (+1.5 mAPon DARK FACE), nighttime semantic segmentation (+5.1 mIoU on ACDC ), and video object detection (+1.8 mAP on DarkVision), highlighting the effectiveness of enhancing hierarchical features under low-light vision.

{{</citation>}}


### (75/133) SoilNet: An Attention-based Spatio-temporal Deep Learning Framework for Soil Organic Carbon Prediction with Digital Soil Mapping in Europe (Nafiseh Kakhani et al., 2023)

{{<citation>}}

Nafiseh Kakhani, Moien Rangzan, Ali Jamali, Sara Attarchi, Seyed Kazem Alavipanah, Thomas Scholten. (2023)  
**SoilNet: An Attention-based Spatio-temporal Deep Learning Framework for Soil Organic Carbon Prediction with Digital Soil Mapping in Europe**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention, LSTM  
[Paper Link](http://arxiv.org/abs/2308.03586v1)  

---


**ABSTRACT**  
Digital soil mapping (DSM) is an advanced approach that integrates statistical modeling and cutting-edge technologies, including machine learning (ML) methods, to accurately depict soil properties and their spatial distribution. Soil organic carbon (SOC) is a crucial soil attribute providing valuable insights into soil health, nutrient cycling, greenhouse gas emissions, and overall ecosystem productivity. This study highlights the significance of spatial-temporal deep learning (DL) techniques within the DSM framework. A novel architecture is proposed, incorporating spatial information using a base convolutional neural network (CNN) model and spatial attention mechanism, along with climate temporal information using a long short-term memory (LSTM) network, for SOC prediction across Europe. The model utilizes a comprehensive set of environmental features, including Landsat-8 images, topography, remote sensing indices, and climate time series, as input features. Results demonstrate that the proposed framework outperforms conventional ML approaches like random forest commonly used in DSM, yielding lower root mean square error (RMSE). This model is a robust tool for predicting SOC and could be applied to other soil properties, thereby contributing to the advancement of DSM techniques and facilitating land management and decision-making processes based on accurate information.

{{</citation>}}


### (76/133) Exploring the Physical World Adversarial Robustness of Vehicle Detection (Wei Jiang et al., 2023)

{{<citation>}}

Wei Jiang, Tianyuan Zhang, Shuangcheng Liu, Weiyu Ji, Zichao Zhang, Gang Xiao. (2023)  
**Exploring the Physical World Adversarial Robustness of Vehicle Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Yolo  
[Paper Link](http://arxiv.org/abs/2308.03476v1)  

---


**ABSTRACT**  
Adversarial attacks can compromise the robustness of real-world detection models. However, evaluating these models under real-world conditions poses challenges due to resource-intensive experiments. Virtual simulations offer an alternative, but the absence of standardized benchmarks hampers progress. Addressing this, we propose an innovative instant-level data generation pipeline using the CARLA simulator. Through this pipeline, we establish the Discrete and Continuous Instant-level (DCI) dataset, enabling comprehensive experiments involving three detection models and three physical adversarial attacks. Our findings highlight diverse model performances under adversarial conditions. Yolo v6 demonstrates remarkable resilience, experiencing just a marginal 6.59% average drop in average precision (AP). In contrast, the ASA attack yields a substantial 14.51% average AP reduction, twice the effect of other algorithms. We also note that static scenes yield higher recognition AP values, and outcomes remain relatively consistent across varying weather conditions. Intriguingly, our study suggests that advancements in adversarial attack algorithms may be approaching its ``limitation''.In summary, our work underscores the significance of adversarial attacks in real-world contexts and introduces the DCI dataset as a versatile benchmark. Our findings provide valuable insights for enhancing the robustness of detection models and offer guidance for future research endeavors in the realm of adversarial attacks.

{{</citation>}}


### (77/133) GaFET: Learning Geometry-aware Facial Expression Translation from In-The-Wild Images (Tianxiang Ma et al., 2023)

{{<citation>}}

Tianxiang Ma, Bingchuan Li, Qian He, Jing Dong, Tieniu Tan. (2023)  
**GaFET: Learning Geometry-aware Facial Expression Translation from In-The-Wild Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.03413v1)  

---


**ABSTRACT**  
While current face animation methods can manipulate expressions individually, they suffer from several limitations. The expressions manipulated by some motion-based facial reenactment models are crude. Other ideas modeled with facial action units cannot generalize to arbitrary expressions not covered by annotations. In this paper, we introduce a novel Geometry-aware Facial Expression Translation (GaFET) framework, which is based on parametric 3D facial representations and can stably decoupled expression. Among them, a Multi-level Feature Aligned Transformer is proposed to complement non-geometric facial detail features while addressing the alignment challenge of spatial features. Further, we design a De-expression model based on StyleGAN, in order to reduce the learning difficulty of GaFET in unpaired "in-the-wild" images. Extensive qualitative and quantitative experiments demonstrate that we achieve higher-quality and more accurate facial expression transfer results compared to state-of-the-art methods, and demonstrate applicability of various poses and complex textures. Besides, videos or annotated training data are omitted, making our method easier to use and generalize.

{{</citation>}}


### (78/133) A Horse with no Labels: Self-Supervised Horse Pose Estimation from Unlabelled Images and Synthetic Prior (Jose Sosa et al., 2023)

{{<citation>}}

Jose Sosa, David Hogg. (2023)  
**A Horse with no Labels: Self-Supervised Horse Pose Estimation from Unlabelled Images and Synthetic Prior**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.03411v1)  

---


**ABSTRACT**  
Obtaining labelled data to train deep learning methods for estimating animal pose is challenging. Recently, synthetic data has been widely used for pose estimation tasks, but most methods still rely on supervised learning paradigms utilising synthetic images and labels. Can training be fully unsupervised? Is a tiny synthetic dataset sufficient? What are the minimum assumptions that we could make for estimating animal pose? Our proposal addresses these questions through a simple yet effective self-supervised method that only assumes the availability of unlabelled images and a small set of synthetic 2D poses. We completely remove the need for any 3D or 2D pose annotations (or complex 3D animal models), and surprisingly our approach can still learn accurate 3D and 2D poses simultaneously. We train our method with unlabelled images of horses mainly collected for YouTube videos and a prior consisting of 2D synthetic poses. The latter is three times smaller than the number of images needed for training. We test our method on a challenging set of horse images and evaluate the predicted 3D and 2D poses. We demonstrate that it is possible to learn accurate animal poses even with as few assumptions as unlabelled images and a small set of 2D poses generated from synthetic data. Given the minimum requirements and the abundance of unlabelled data, our method could be easily deployed to different animals.

{{</citation>}}


### (79/133) DiT: Efficient Vision Transformers with Dynamic Token Routing (Yuchen Ma et al., 2023)

{{<citation>}}

Yuchen Ma, Zhengcong Fei, Junshi Huang. (2023)  
**DiT: Efficient Vision Transformers with Dynamic Token Routing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.03409v1)  

---


**ABSTRACT**  
Recently, the tokens of images share the same static data flow in many dense networks. However, challenges arise from the variance among the objects in images, such as large variations in the spatial scale and difficulties of recognition for visual entities. In this paper, we propose a data-dependent token routing strategy to elaborate the routing paths of image tokens for Dynamic Vision Transformer, dubbed DiT. The proposed framework generates a data-dependent path per token, adapting to the object scales and visual discrimination of tokens. In feed-forward, the differentiable routing gates are designed to select the scaling paths and feature transformation paths for image tokens, leading to multi-path feature propagation. In this way, the impact of object scales and visual discrimination of image representation can be carefully tuned. Moreover, the computational cost can be further reduced by giving budget constraints to the routing gate and early-stopping of feature extraction. In experiments, our DiT achieves superior performance and favorable complexity/accuracy trade-offs than many SoTA methods on ImageNet classification, object detection, instance segmentation, and semantic segmentation. Particularly, the DiT-B5 obtains 84.8\% top-1 Acc on ImageNet with 10.3 GFLOPs, which is 1.0\% higher than that of the SoTA method with similar computational complexity. These extensive results demonstrate that DiT can serve as versatile backbones for various vision tasks.

{{</citation>}}


### (80/133) Dual Aggregation Transformer for Image Super-Resolution (Zheng Chen et al., 2023)

{{<citation>}}

Zheng Chen, Yulun Zhang, Jinjin Gu, Linghe Kong, Xiaokang Yang, Fisher Yu. (2023)  
**Dual Aggregation Transformer for Image Super-Resolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03364v1)  

---


**ABSTRACT**  
Transformer has recently gained considerable popularity in low-level vision tasks, including image super-resolution (SR). These networks utilize self-attention along different dimensions, spatial or channel, and achieve impressive performance. This inspires us to combine the two dimensions in Transformer for a more powerful representation capability. Based on the above idea, we propose a novel Transformer model, Dual Aggregation Transformer (DAT), for image SR. Our DAT aggregates features across spatial and channel dimensions, in the inter-block and intra-block dual manner. Specifically, we alternately apply spatial and channel self-attention in consecutive Transformer blocks. The alternate strategy enables DAT to capture the global context and realize inter-block feature aggregation. Furthermore, we propose the adaptive interaction module (AIM) and the spatial-gate feed-forward network (SGFN) to achieve intra-block feature aggregation. AIM complements two self-attention mechanisms from corresponding dimensions. Meanwhile, SGFN introduces additional non-linear spatial information in the feed-forward network. Extensive experiments show that our DAT surpasses current methods. Code and models are obtainable at https://github.com/zhengchen1999/DAT.

{{</citation>}}


### (81/133) Distortion-aware Transformer in 360° Salient Object Detection (Yinjie Zhao et al., 2023)

{{<citation>}}

Yinjie Zhao, Lichen Zhao, Qian Yu, Jing Zhang, Lu Sheng, Dong Xu. (2023)  
**Distortion-aware Transformer in 360° Salient Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03359v1)  

---


**ABSTRACT**  
With the emergence of VR and AR, 360{\deg} data attracts increasing attention from the computer vision and multimedia communities. Typically, 360{\deg} data is projected into 2D ERP (equirectangular projection) images for feature extraction. However, existing methods cannot handle the distortions that result from the projection, hindering the development of 360-data-based tasks. Therefore, in this paper, we propose a Transformer-based model called DATFormer to address the distortion problem. We tackle this issue from two perspectives. Firstly, we introduce two distortion-adaptive modules. The first is a Distortion Mapping Module, which guides the model to pre-adapt to distorted features globally. The second module is a Distortion-Adaptive Attention Block that reduces local distortions on multi-scale features. Secondly, to exploit the unique characteristics of 360{\deg} data, we present a learnable relation matrix and use it as part of the positional embedding to further improve performance. Extensive experiments are conducted on three public datasets, and the results show that our model outperforms existing 2D SOD (salient object detection) and 360 SOD methods.

{{</citation>}}


### (82/133) A Hybrid CNN-Transformer Architecture with Frequency Domain Contrastive Learning for Image Deraining (Cheng Wang et al., 2023)

{{<citation>}}

Cheng Wang, Wei Li. (2023)  
**A Hybrid CNN-Transformer Architecture with Frequency Domain Contrastive Learning for Image Deraining**  

---
Primary Category: cs.CV  
Categories: I-4-4, cs-CV, cs.CV, eess-IV  
Keywords: Contrastive Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03340v1)  

---


**ABSTRACT**  
Image deraining is a challenging task that involves restoring degraded images affected by rain streaks.

{{</citation>}}


### (83/133) Part-Aware Transformer for Generalizable Person Re-identification (Hao Ni et al., 2023)

{{<citation>}}

Hao Ni, Yuke Li, Heng Tao Shen, Jingkuan Song. (2023)  
**Part-Aware Transformer for Generalizable Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.03322v1)  

---


**ABSTRACT**  
Domain generalization person re-identification (DG-ReID) aims to train a model on source domains and generalize well on unseen domains. Vision Transformer usually yields better generalization ability than common CNN networks under distribution shifts. However, Transformer-based ReID models inevitably over-fit to domain-specific biases due to the supervised learning strategy on the source domain. We observe that while the global images of different IDs should have different features, their similar local parts (e.g., black backpack) are not bounded by this constraint. Motivated by this, we propose a pure Transformer model (termed Part-aware Transformer) for DG-ReID by designing a proxy task, named Cross-ID Similarity Learning (CSL), to mine local visual information shared by different IDs. This proxy task allows the model to learn generic features because it only cares about the visual similarity of the parts regardless of the ID labels, thus alleviating the side effect of domain-specific biases. Based on the local similarity obtained in CSL, a Part-guided Self-Distillation (PSD) is proposed to further improve the generalization of global features. Our method achieves state-of-the-art performance under most DG ReID settings. Under the Market$\to$Duke setting, our method exceeds state-of-the-art by 10.9% and 12.8% in Rank1 and mAP, respectively. The code is available at https://github.com/liyuke65535/Part-Aware-Transformer.

{{</citation>}}


### (84/133) FLIQS: One-Shot Mixed-Precision Floating-Point and Integer Quantization Search (Jordan Dotzel et al., 2023)

{{<citation>}}

Jordan Dotzel, Gang Wu, Andrew Li, Muhammad Umar, Yun Ni, Mohamed S. Abdelfattah, Zhiru Zhang, Liqun Cheng, Martin G. Dixon, Norman P. Jouppi, Quoc V. Le, Sheng Li. (2023)  
**FLIQS: One-Shot Mixed-Precision Floating-Point and Integer Quantization Search**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Quantization  
[Paper Link](http://arxiv.org/abs/2308.03290v1)  

---


**ABSTRACT**  
Quantization has become a mainstream compression technique for reducing model size, computational requirements, and energy consumption for modern deep neural networks (DNNs). With the improved numerical support in recent hardware, including multiple variants of integer and floating point, mixed-precision quantization has become necessary to achieve high-quality results with low model cost. Prior mixed-precision quantization methods have performed a post-training quantization search, which compromises on accuracy, or a differentiable quantization search, which leads to high memory usage from branching. Therefore, we propose the first one-shot mixed-precision quantization search that eliminates the need for retraining in both integer and low-precision floating point models. We evaluate our floating-point and integer quantization search (FLIQS) on multiple convolutional networks and vision transformer models to discover Pareto-optimal models. Our approach discovers models that improve upon uniform precision, manual mixed-precision, and recent integer quantization search methods. With the proposed integer quantization search, we increase the accuracy of ResNet-18 on ImageNet by 1.31% points and ResNet-50 by 0.90% points with equivalent model cost over previous methods. Additionally, for the first time, we explore a novel mixed-precision floating-point search and improve MobileNetV2 by up to 0.98% points compared to prior state-of-the-art FP8 models. Finally, we extend FLIQS to simultaneously search a joint quantization and neural architecture space and improve the ImageNet accuracy by 2.69% points with similar model cost on a MobileNetV2 search space.

{{</citation>}}


### (85/133) Multi-Label Self-Supervised Learning with Scene Images (Ke Zhu et al., 2023)

{{<citation>}}

Ke Zhu, Minghao Fu, Jianxin Wu. (2023)  
**Multi-Label Self-Supervised Learning with Scene Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.03286v2)  

---


**ABSTRACT**  
Self-supervised learning (SSL) methods targeting scene images have seen a rapid growth recently, and they mostly rely on either a dedicated dense matching mechanism or a costly unsupervised object discovery module. This paper shows that instead of hinging on these strenuous operations, quality image representations can be learned by treating scene/multi-label image SSL simply as a multi-label classification problem, which greatly simplifies the learning framework. Specifically, multiple binary pseudo-labels are assigned for each input image by comparing its embeddings with those in two dictionaries, and the network is optimized using the binary cross entropy loss. The proposed method is named Multi-Label Self-supervised learning (MLS). Visualizations qualitatively show that clearly the pseudo-labels by MLS can automatically find semantically similar pseudo-positive pairs across different images to facilitate contrastive learning. MLS learns high quality representations on MS-COCO and achieves state-of-the-art results on classification, detection and segmentation benchmarks. At the same time, MLS is much simpler than existing methods, making it easier to deploy and for further exploration.

{{</citation>}}


### (86/133) Environment-Invariant Curriculum Relation Learning for Fine-Grained Scene Graph Generation (Yukuan Min et al., 2023)

{{<citation>}}

Yukuan Min, Aming Wu, Cheng Deng. (2023)  
**Environment-Invariant Curriculum Relation Learning for Fine-Grained Scene Graph Generation**  

---
Primary Category: cs.CV  
Categories: 68Txx, I-4, cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.03282v1)  

---


**ABSTRACT**  
The scene graph generation (SGG) task is designed to identify the predicates based on the subject-object pairs.However,existing datasets generally include two imbalance cases: one is the class imbalance from the predicted predicates and another is the context imbalance from the given subject-object pairs, which presents significant challenges for SGG. Most existing methods focus on the imbalance of the predicted predicate while ignoring the imbalance of the subject-object pairs, which could not achieve satisfactory results. To address the two imbalance cases, we propose a novel Environment Invariant Curriculum Relation learning (EICR) method, which can be applied in a plug-and-play fashion to existing SGG methods. Concretely, to remove the imbalance of the subject-object pairs, we first construct different distribution environments for the subject-object pairs and learn a model invariant to the environment changes. Then, we construct a class-balanced curriculum learning strategy to balance the different environments to remove the predicate imbalance. Comprehensive experiments conducted on VG and GQA datasets demonstrate that our EICR framework can be taken as a general strategy for various SGG models, and achieve significant improvements.

{{</citation>}}


### (87/133) Feature-Suppressed Contrast for Self-Supervised Food Pre-training (Xinda Liu et al., 2023)

{{<citation>}}

Xinda Liu, Yaohui Zhu, Linhu Liu, Jiang Tian, Lili Wang. (2023)  
**Feature-Suppressed Contrast for Self-Supervised Food Pre-training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.03272v1)  

---


**ABSTRACT**  
Most previous approaches for analyzing food images have relied on extensively annotated datasets, resulting in significant human labeling expenses due to the varied and intricate nature of such images. Inspired by the effectiveness of contrastive self-supervised methods in utilizing unlabelled data, weiqing explore leveraging these techniques on unlabelled food images. In contrastive self-supervised methods, two views are randomly generated from an image by data augmentations. However, regarding food images, the two views tend to contain similar informative contents, causing large mutual information, which impedes the efficacy of contrastive self-supervised learning. To address this problem, we propose Feature Suppressed Contrast (FeaSC) to reduce mutual information between views. As the similar contents of the two views are salient or highly responsive in the feature map, the proposed FeaSC uses a response-aware scheme to localize salient features in an unsupervised manner. By suppressing some salient features in one view while leaving another contrast view unchanged, the mutual information between the two views is reduced, thereby enhancing the effectiveness of contrast learning for self-supervised food pre-training. As a plug-and-play module, the proposed method consistently improves BYOL and SimSiam by 1.70\% $\sim$ 6.69\% classification accuracy on four publicly available food recognition datasets. Superior results have also been achieved on downstream segmentation tasks, demonstrating the effectiveness of the proposed method.

{{</citation>}}


### (88/133) Redundancy-aware Transformer for Video Question Answering (Yicong Li et al., 2023)

{{<citation>}}

Yicong Li, Xun Yang, An Zhang, Chun Feng, Xiang Wang, Tat-Seng Chua. (2023)  
**Redundancy-aware Transformer for Video Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Question Answering, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03267v1)  

---


**ABSTRACT**  
This paper identifies two kinds of redundancy in the current VideoQA paradigm. Specifically, the current video encoders tend to holistically embed all video clues at different granularities in a hierarchical manner, which inevitably introduces \textit{neighboring-frame redundancy} that can overwhelm detailed visual clues at the object level. Subsequently, prevailing vision-language fusion designs introduce the \textit{cross-modal redundancy} by exhaustively fusing all visual elements with question tokens without explicitly differentiating their pairwise vision-language interactions, thus making a pernicious impact on the answering.   To this end, we propose a novel transformer-based architecture, that aims to model VideoQA in a redundancy-aware manner. To address the neighboring-frame redundancy, we introduce a video encoder structure that emphasizes the object-level change in neighboring frames, while adopting an out-of-neighboring message-passing scheme that imposes attention only on distant frames. As for the cross-modal redundancy, we equip our fusion module with a novel adaptive sampling, which explicitly differentiates the vision-language interactions by identifying a small subset of visual elements that exclusively support the answer. Upon these advancements, we find this \underline{R}edundancy-\underline{a}ware trans\underline{former} (RaFormer) can achieve state-of-the-art results on multiple VideoQA benchmarks.

{{</citation>}}


### (89/133) Learning a Graph Neural Network with Cross Modality Interaction for Image Fusion (Jiawei Li et al., 2023)

{{<citation>}}

Jiawei Li, Jiansheng Chen, Jinyuan Liu, Huimin Ma. (2023)  
**Learning a Graph Neural Network with Cross Modality Interaction for Image Fusion**  

---
Primary Category: cs.CV  
Categories: I-4; I-2, cs-CV, cs.CV  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.03256v1)  

---


**ABSTRACT**  
Infrared and visible image fusion has gradually proved to be a vital fork in the field of multi-modality imaging technologies. In recent developments, researchers not only focus on the quality of fused images but also evaluate their performance in downstream tasks. Nevertheless, the majority of methods seldom put their eyes on the mutual learning from different modalities, resulting in fused images lacking significant details and textures. To overcome this issue, we propose an interactive graph neural network (GNN)-based architecture between cross modality for fusion, called IGNet. Specifically, we first apply a multi-scale extractor to achieve shallow features, which are employed as the necessary input to build graph structures. Then, the graph interaction module can construct the extracted intermediate features of the infrared/visible branch into graph structures. Meanwhile, the graph structures of two branches interact for cross-modality and semantic learning, so that fused images can maintain the important feature expressions and enhance the performance of downstream tasks. Besides, the proposed leader nodes can improve information propagation in the same modality. Finally, we merge all graph features to get the fusion result. Extensive experiments on different datasets (TNO, MFNet and M3FD) demonstrate that our IGNet can generate visually appealing fused images while scoring averagely 2.59% mAP@.5 and 7.77% mIoU higher in detection and segmentation than the compared state-of-the-art methods. The source code of the proposed IGNet can be available at https://github.com/lok-18/IGNet.

{{</citation>}}


## cs.AI (13)



### (90/133) Establishing Trust in ChatGPT BioMedical Generated Text: An Ontology-Based Knowledge Graph to Validate Disease-Symptom Links (Ahmed Abdeen Hamed et al., 2023)

{{<citation>}}

Ahmed Abdeen Hamed, Alessandro Crimi, Magdalena M. Misiak, Byung Suk Lee. (2023)  
**Establishing Trust in ChatGPT BioMedical Generated Text: An Ontology-Based Knowledge Graph to Validate Disease-Symptom Links**  

---
Primary Category: cs.AI  
Categories: I-2, cs-AI, cs-CL, cs.AI  
Keywords: AI, ChatGPT, GPT, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.03929v1)  

---


**ABSTRACT**  
Methods: Through an innovative approach, we construct ontology-based knowledge graphs from authentic medical literature and AI-generated content. Our goal is to distinguish factual information from unverified data. We compiled two datasets: one from biomedical literature using a "human disease and symptoms" query, and another generated by ChatGPT, simulating articles. With these datasets (PubMed and ChatGPT), we curated 10 sets of 250 abstracts each, selected randomly with a specific seed. Our method focuses on utilizing disease ontology (DOID) and symptom ontology (SYMP) to build knowledge graphs, robust mathematical models that facilitate unbiased comparisons. By employing our fact-checking algorithms and network centrality metrics, we conducted GPT disease-symptoms link analysis to quantify the accuracy of factual knowledge amid noise, hypotheses, and significant findings.   Results: The findings obtained from the comparison of diverse ChatGPT knowledge graphs with their PubMed counterparts revealed some interesting observations. While PubMed knowledge graphs exhibit a wealth of disease-symptom terms, it is surprising to observe that some ChatGPT graphs surpass them in the number of connections. Furthermore, some GPT graphs are demonstrating supremacy of the centrality scores, especially for the overlapping nodes. This striking contrast indicates the untapped potential of knowledge that can be derived from AI-generated content, awaiting verification. Out of all the graphs, the factual link ratio between any two graphs reached its peak at 60%.   Conclusions: An intriguing insight from our findings was the striking number of links among terms in the knowledge graph generated from ChatGPT datasets, surpassing some of those in its PubMed counterpart. This early discovery has prompted further investigation using universal network metrics to unveil the new knowledge the links may hold.

{{</citation>}}


### (91/133) AgentBench: Evaluating LLMs as Agents (Xiao Liu et al., 2023)

{{<citation>}}

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, Jie Tang. (2023)  
**AgentBench: Evaluating LLMs as Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.03688v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are becoming increasingly smart and autonomous, targeting real-world pragmatic missions beyond traditional NLP tasks. As a result, there has been an urgent need to evaluate LLMs as agents on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional evolving benchmark that currently consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities in a multi-turn open-ended generation setting. Our extensive test over 25 LLMs (including APIs and open-sourced models) shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and open-sourced competitors. It also serves as a component of an ongoing project with wider coverage and deeper consideration towards systematic LLM evaluation. Datasets, environments, and an integrated evaluation package for AgentBench are released at https://github.com/THUDM/AgentBench

{{</citation>}}


### (92/133) QDax: A Library for Quality-Diversity and Population-based Algorithms with Hardware Acceleration (Felix Chalumeau et al., 2023)

{{<citation>}}

Felix Chalumeau, Bryan Lim, Raphael Boige, Maxime Allard, Luca Grillotti, Manon Flageat, Valentin Macé, Arthur Flajolet, Thomas Pierrot, Antoine Cully. (2023)  
**QDax: A Library for Quality-Diversity and Population-based Algorithms with Hardware Acceleration**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-NE, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.03665v1)  

---


**ABSTRACT**  
QDax is an open-source library with a streamlined and modular API for Quality-Diversity (QD) optimization algorithms in Jax. The library serves as a versatile tool for optimization purposes, ranging from black-box optimization to continuous control. QDax offers implementations of popular QD, Neuroevolution, and Reinforcement Learning (RL) algorithms, supported by various examples. All the implementations can be just-in-time compiled with Jax, facilitating efficient execution across multiple accelerators, including GPUs and TPUs. These implementations effectively demonstrate the framework's flexibility and user-friendliness, easing experimentation for research purposes. Furthermore, the library is thoroughly documented and tested with 95\% coverage.

{{</citation>}}


### (93/133) Stock Market Price Prediction: A Hybrid LSTM and Sequential Self-Attention based Approach (Karan Pardeshi et al., 2023)

{{<citation>}}

Karan Pardeshi, Sukhpal Singh Gill, Ahmed M. Abdelmoniem. (2023)  
**Stock Market Price Prediction: A Hybrid LSTM and Sequential Self-Attention based Approach**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Attention, LSTM, Self-Attention  
[Paper Link](http://arxiv.org/abs/2308.04419v1)  

---


**ABSTRACT**  
One of the most enticing research areas is the stock market, and projecting stock prices may help investors profit by making the best decisions at the correct time. Deep learning strategies have emerged as a critical technique in the field of the financial market. The stock market is impacted due to two aspects, one is the geo-political, social and global events on the bases of which the price trends could be affected. Meanwhile, the second aspect purely focuses on historical price trends and seasonality, allowing us to forecast stock prices. In this paper, our aim is to focus on the second aspect and build a model that predicts future prices with minimal errors. In order to provide better prediction results of stock price, we propose a new model named Long Short-Term Memory (LSTM) with Sequential Self-Attention Mechanism (LSTM-SSAM). Finally, we conduct extensive experiments on the three stock datasets: SBIN, HDFCBANK, and BANKBARODA. The experimental results prove the effectiveness and feasibility of the proposed model compared to existing models. The experimental findings demonstrate that the root-mean-squared error (RMSE), and R-square (R2) evaluation indicators are giving the best results.

{{</citation>}}


### (94/133) Why We Don't Have AGI Yet (Peter Voss et al., 2023)

{{<citation>}}

Peter Voss, Mladjan Jovanovic. (2023)  
**Why We Don't Have AGI Yet**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03598v1)  

---


**ABSTRACT**  
The original vision of AI was re-articulated in 2002 via the term 'Artificial General Intelligence' or AGI. This vision is to build 'Thinking Machines' - computer systems that can learn, reason, and solve problems similar to the way humans do. This is in stark contrast to the 'Narrow AI' approach practiced by almost everyone in the field over the many decades. While several large-scale efforts have nominally been working on AGI (most notably DeepMind), the field of pure focused AGI development has not been well funded or promoted. This is surprising given the fantastic value that true AGI can bestow on humanity. In addition to the dearth of effort in this field, there are also several theoretical and methodical missteps that are hampering progress. We highlight why purely statistical approaches are unlikely to lead to AGI, and identify several crucial cognitive abilities required to achieve human-like adaptability and autonomous learning. We conclude with a survey of socio-technical factors that have undoubtedly slowed progress towards AGI.

{{</citation>}}


### (95/133) Feature Importance versus Feature Influence and What It Signifies for Explainable AI (Kary Främling, 2023)

{{<citation>}}

Kary Främling. (2023)  
**Feature Importance versus Feature Influence and What It Signifies for Explainable AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03589v1)  

---


**ABSTRACT**  
When used in the context of decision theory, feature importance expresses how much changing the value of a feature can change the model outcome (or the utility of the outcome), compared to other features. Feature importance should not be confused with the feature influence used by most state-of-the-art post-hoc Explainable AI methods. Contrary to feature importance, feature influence is measured against a reference level or baseline. The Contextual Importance and Utility (CIU) method provides a unified definition of global and local feature importance that is applicable also for post-hoc explanations, where the value utility concept provides instance-level assessment of how favorable or not a feature value is for the outcome. The paper shows how CIU can be applied to both global and local explainability, assesses the fidelity and stability of different methods, and shows how explanations that use contextual importance and contextual utility can provide more expressive and flexible explanations than when using influence only.

{{</citation>}}


### (96/133) Exploring ChatGPT's Empathic Abilities (Kristina Schaaff et al., 2023)

{{<citation>}}

Kristina Schaaff, Caroline Reinig, Tim Schlippe. (2023)  
**Exploring ChatGPT's Empathic Abilities**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: ChatGPT, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2308.03527v1)  

---


**ABSTRACT**  
Empathy is often understood as the ability to share and understand another individual's state of mind or emotion. With the increasing use of chatbots in various domains, e.g., children seeking help with homework, individuals looking for medical advice, and people using the chatbot as a daily source of everyday companionship, the importance of empathy in human-computer interaction has become more apparent. Therefore, our study investigates the extent to which ChatGPT based on GPT-3.5 can exhibit empathetic responses and emotional expressions. We analyzed the following three aspects: (1) understanding and expressing emotions, (2) parallel emotional response, and (3) empathic personality. Thus, we not only evaluate ChatGPT on various empathy aspects and compare it with human behavior but also show a possible way to analyze the empathy of chatbots in general. Our results show, that in 91.7% of the cases, ChatGPT was able to correctly identify emotions and produces appropriate answers. In conversations, ChatGPT reacted with a parallel emotion in 70.7% of cases. The empathic capabilities of ChatGPT were evaluated using a set of five questionnaires covering different aspects of empathy. Even though the results indicate that the empathic abilities of ChatGPT are still below the average of healthy humans, the scores are better than those of people who have been diagnosed with Asperger syndrome / high-functioning autism.

{{</citation>}}


### (97/133) Intelligence-Endogenous Management Platform for Computing and Network Convergence (Zicong Hong et al., 2023)

{{<citation>}}

Zicong Hong, Xiaoyu Qiu, Jian Lin, Wuhui Chen, Yue Yu, Hui Wang, Song Guo, Wen Gao. (2023)  
**Intelligence-Endogenous Management Platform for Computing and Network Convergence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Azure, Microsoft  
[Paper Link](http://arxiv.org/abs/2308.03450v1)  

---


**ABSTRACT**  
Massive emerging applications are driving demand for the ubiquitous deployment of computing power today. This trend not only spurs the recent popularity of the \emph{Computing and Network Convergence} (CNC), but also introduces an urgent need for the intelligentization of a management platform to coordinate changing resources and tasks in the CNC. Therefore, in this article, we present the concept of an intelligence-endogenous management platform for CNCs called \emph{CNC brain} based on artificial intelligence technologies. It aims at efficiently and automatically matching the supply and demand with high heterogeneity in a CNC via four key building blocks, i.e., perception, scheduling, adaptation, and governance, throughout the CNC's life cycle. Their functionalities, goals, and challenges are presented. To examine the effectiveness of the proposed concept and framework, we also implement a prototype for the CNC brain based on a deep reinforcement learning technology. Also, it is evaluated on a CNC testbed that integrates two open-source and popular frameworks (OpenFaas and Kubernetes) and a real-world business dataset provided by Microsoft Azure. The evaluation results prove the proposed method's effectiveness in terms of resource utilization and performance. Finally, we highlight the future research directions of the CNC brain.

{{</citation>}}


### (98/133) Biomedical Knowledge Graph Embeddings with Negative Statements (Rita T. Sousa et al., 2023)

{{<citation>}}

Rita T. Sousa, Sara Silva, Heiko Paulheim, Catia Pesquita. (2023)  
**Biomedical Knowledge Graph Embeddings with Negative Statements**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.03447v1)  

---


**ABSTRACT**  
A knowledge graph is a powerful representation of real-world entities and their relations. The vast majority of these relations are defined as positive statements, but the importance of negative statements is increasingly recognized, especially under an Open World Assumption. Explicitly considering negative statements has been shown to improve performance on tasks such as entity summarization and question answering or domain-specific tasks such as protein function prediction. However, no attention has been given to the exploration of negative statements by knowledge graph embedding approaches despite the potential of negative statements to produce more accurate representations of entities in a knowledge graph.   We propose a novel approach, TrueWalks, to incorporate negative statements into the knowledge graph representation learning process. In particular, we present a novel walk-generation method that is able to not only differentiate between positive and negative statements but also take into account the semantic implications of negation in ontology-rich knowledge graphs. This is of particular importance for applications in the biomedical domain, where the inadequacy of embedding approaches regarding negative statements at the ontology level has been identified as a crucial limitation.   We evaluate TrueWalks in ontology-rich biomedical knowledge graphs in two different predictive tasks based on KG embeddings: protein-protein interaction prediction and gene-disease association prediction. We conduct an extensive analysis over established benchmarks and demonstrate that our method is able to improve the performance of knowledge graph embeddings on all tasks.

{{</citation>}}


### (99/133) TPTU: Task Planning and Tool Usage of Large Language Model-based AI Agents (Jingqing Ruan et al., 2023)

{{<citation>}}

Jingqing Ruan, Yihong Chen, Bin Zhang, Zhiwei Xu, Tianpeng Bao, Guoqing Du, Shiwei Shi, Hangyu Mao, Xingyu Zeng, Rui Zhao. (2023)  
**TPTU: Task Planning and Tool Usage of Large Language Model-based AI Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03427v1)  

---


**ABSTRACT**  
With recent advancements in natural language processing, Large Language Models (LLMs) have emerged as powerful tools for various real-world applications. Despite their prowess, the intrinsic generative abilities of LLMs may prove insufficient for handling complex tasks which necessitate a combination of task planning and the usage of external tools. In this paper, we first propose a structured framework tailored for LLM-based AI Agents and discuss the crucial capabilities necessary for tackling intricate problems. Within this framework, we design two distinct types of agents (i.e., one-step agent and sequential agent) to execute the inference process. Subsequently, we instantiate the framework using various LLMs and evaluate their Task Planning and Tool Usage (TPTU) abilities on typical tasks. By highlighting key findings and challenges, our goal is to provide a helpful resource for researchers and practitioners to leverage the power of LLMs in their AI applications. Our study emphasizes the substantial potential of these models, while also identifying areas that need more investigation and improvement.

{{</citation>}}


### (100/133) Minimizing Return Gaps with Discrete Communications in Decentralized POMDP (Jingdi Chen et al., 2023)

{{<citation>}}

Jingdi Chen, Tian Lan. (2023)  
**Minimizing Return Gaps with Discrete Communications in Decentralized POMDP**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.03358v1)  

---


**ABSTRACT**  
Communication is crucial for solving cooperative Multi-Agent Reinforcement Learning tasks in Partially-Observable Markov Decision Processes. Existing works often rely on black-box methods to encode local information/features into messages shared with other agents. However, such black-box approaches are unable to provide any quantitative guarantees on the expected return and often lead to the generation of continuous messages with high communication overhead and poor interpretability. In this paper, we establish an upper bound on the return gap between an ideal policy with full observability and an optimal partially-observable policy with discrete communication. This result enables us to recast multi-agent communication into a novel online clustering problem over the local observations at each agent, with messages as cluster labels and the upper bound on the return gap as clustering loss. By minimizing the upper bound, we propose a surprisingly simple design of message generation functions in multi-agent communication and integrate it with reinforcement learning using a Regularized Information Maximization loss function. Evaluations show that the proposed discrete communication significantly outperforms state-of-the-art multi-agent communication baselines and can achieve nearly-optimal returns with few-bit messages that are naturally interpretable.

{{</citation>}}


### (101/133) Generative AI trial for nonviolent communication mediation (Takeshi Kato, 2023)

{{<citation>}}

Takeshi Kato. (2023)  
**Generative AI trial for nonviolent communication mediation**  

---
Primary Category: cs.AI  
Categories: 68T50, 91C99, I-2-1; J-4, cs-AI, cs-CY, cs.AI  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.03326v1)  

---


**ABSTRACT**  
Aiming for a mixbiotic society that combines freedom and solidarity among people with diverse values, I focused on nonviolent communication (NVC) that enables compassionate giving in various situations of social division and conflict, and tried a generative AI for it. Specifically, ChatGPT was used in place of the traditional certified trainer to test the possibility of mediating (modifying) input sentences in four processes: observation, feelings, needs, and requests. The results indicate that there is potential for the application of generative AI, although not yet at a practical level. Suggested improvement guidelines included adding model responses, relearning revised responses, specifying appropriate terminology for each process, and re-asking for required information. The use of generative AI will be useful initially to assist certified trainers, to prepare for and review events and workshops, and in the future to support consensus building and cooperative behavior in digital democracy, platform cooperatives, and cyber-human social co-operating systems. It is hoped that the widespread use of NVC mediation using generative AI will lead to the early realization of a mixbiotic society.

{{</citation>}}


### (102/133) What has ChatGPT read? The origins of archaeological citations used by a generative artificial intelligence application (Dirk HR Spennemann, 2023)

{{<citation>}}

Dirk HR Spennemann. (2023)  
**What has ChatGPT read? The origins of archaeological citations used by a generative artificial intelligence application**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IT, cs.AI, math-IT  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.03301v1)  

---


**ABSTRACT**  
The public release of ChatGPT has resulted in considerable publicity and has led to wide-spread discussion of the usefulness and capabilities of generative AI language models. Its ability to extract and summarise data from textual sources and present them as human-like contextual responses makes it an eminently suitable tool to answer questions users might ask. This paper tested what archaeological literature appears to have been included in ChatGPT's training phase. While ChatGPT offered seemingly pertinent references, a large percentage proved to be fictitious. Using cloze analysis to make inferences on the sources 'memorised' by a generative AI model, this paper was unable to prove that ChatGPT had access to the full texts of the genuine references. It can be shown that all references provided by ChatGPT that were found to be genuine have also been cited on Wikipedia pages. This strongly indicates that the source base for at least some of the data is found in those pages. The implications of this in relation to data quality are discussed.

{{</citation>}}


## cs.HC (7)



### (103/133) Advancements In Crowd-Monitoring System: A Comprehensive Analysis of Systematic Approaches and Automation Algorithms: State-of-The-Art (Mohammed Ameen et al., 2023)

{{<citation>}}

Mohammed Ameen, Richard Stone. (2023)  
**Advancements In Crowd-Monitoring System: A Comprehensive Analysis of Systematic Approaches and Automation Algorithms: State-of-The-Art**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs-LG, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03907v1)  

---


**ABSTRACT**  
Growing apprehensions surrounding public safety have captured the attention of numerous governments and security agencies across the globe. These entities are increasingly acknowledging the imperative need for reliable and secure crowd-monitoring systems to address these concerns. Effectively managing human gatherings necessitates proactive measures to prevent unforeseen events or complications, ensuring a safe and well-coordinated environment. The scarcity of research focusing on crowd monitoring systems and their security implications has given rise to a burgeoning area of investigation, exploring potential approaches to safeguard human congregations effectively. Crowd monitoring systems depend on a bifurcated approach, encompassing vision-based and non-vision-based technologies. An in-depth analysis of these two methodologies will be conducted in this research. The efficacy of these approaches is contingent upon the specific environment and temporal context in which they are deployed, as they each offer distinct advantages. This paper endeavors to present an in-depth analysis of the recent incorporation of artificial intelligence (AI) algorithms and models into automated systems, emphasizing their contemporary applications and effectiveness in various contexts.

{{</citation>}}


### (104/133) Average Estimates in Line Graphs Are Biased Toward Areas of Higher Variability (Dominik Moritz et al., 2023)

{{<citation>}}

Dominik Moritz, Lace M. Padilla, Francis Nguyen, Steven L. Franconeri. (2023)  
**Average Estimates in Line Graphs Are Biased Toward Areas of Higher Variability**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.03903v1)  

---


**ABSTRACT**  
We investigate variability overweighting, a previously undocumented bias in line graphs, where estimates of average value are biased toward areas of higher variability in that line. We found this effect across two preregistered experiments with 140 and 420 participants. These experiments also show that the bias is reduced when using a dot encoding of the same series. We can model the bias with the average of the data series and the average of the points drawn along the line. This bias might arise because higher variability leads to stronger weighting in the average calculation, either due to the longer line segments (even though those segments contain the same number of data values) or line segments with higher variability being otherwise more visually salient. Understanding and predicting this bias is important for visualization design guidelines, recommendation systems, and tool builders, as the bias can adversely affect estimates of averages and trends.

{{</citation>}}


### (105/133) Storyfier: Exploring Vocabulary Learning Support with Text Generation Models (Zhenhui Peng et al., 2023)

{{<citation>}}

Zhenhui Peng, Xingbo Wang, Qiushi Han, Junkai Zhu, Xiaojuan Ma, Huamin Qu. (2023)  
**Storyfier: Exploring Vocabulary Learning Support with Text Generation Models**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs.HC  
Keywords: AI, Text Generation  
[Paper Link](http://arxiv.org/abs/2308.03864v1)  

---


**ABSTRACT**  
Vocabulary learning support tools have widely exploited existing materials, e.g., stories or video clips, as contexts to help users memorize each target word. However, these tools could not provide a coherent context for any target words of learners' interests, and they seldom help practice word usage. In this paper, we work with teachers and students to iteratively develop Storyfier, which leverages text generation models to enable learners to read a generated story that covers any target words, conduct a story cloze test, and use these words to write a new story with adaptive AI assistance. Our within-subjects study (N=28) shows that learners generally favor the generated stories for connecting target words and writing assistance for easing their learning workload. However, in the read-cloze-write learning sessions, participants using Storyfier perform worse in recalling and using target words than learning with a baseline tool without our AI features. We discuss insights into supporting learning tasks with generative models.

{{</citation>}}


### (106/133) Screen-based 3D Subjective Experiment Software (Songlin Fan et al., 2023)

{{<citation>}}

Songlin Fan, Wei Gao. (2023)  
**Screen-based 3D Subjective Experiment Software**  

---
Primary Category: cs.HC  
Categories: cs-CV, cs-HC, cs.HC  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.03698v2)  

---


**ABSTRACT**  
Recently, widespread 3D graphics (e.g., point clouds and meshes) have drawn considerable efforts from academia and industry to assess their perceptual quality by conducting subjective experiments. However, lacking a handy software for 3D subjective experiments complicates the construction of 3D graphics quality assessment datasets, thus hindering the prosperity of relevant fields. In this paper, we develop a powerful platform with which users can flexibly design their 3D subjective methodologies and build high-quality datasets, easing a broad spectrum of 3D graphics subjective quality study. To accurately illustrate the perceptual quality differences of 3D stimuli, our software can simultaneously render the source stimulus and impaired stimulus and allows both stimuli to respond synchronously to viewer interactions. Compared with amateur 3D visualization tool-based or image/video rendering-based schemes, our approach embodies typical 3D applications while minimizing cognitive overload during subjective experiments. We organized a subjective experiment involving 40 participants to verify the validity of the proposed software. Experimental analyses demonstrate that subjective tests on our software can produce reasonable subjective quality scores of 3D models. All resources in this paper can be found at https://openi.pcl.ac.cn/OpenDatasets/3DQA.

{{</citation>}}


### (107/133) XAI in Automated Fact-Checking? The Benefits Are Modest And There's No One-Explanation-Fits-All (Gionnieve Lim et al., 2023)

{{<citation>}}

Gionnieve Lim, Simon T. Perrault. (2023)  
**XAI in Automated Fact-Checking? The Benefits Are Modest And There's No One-Explanation-Fits-All**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Fact-Checking  
[Paper Link](http://arxiv.org/abs/2308.03372v1)  

---


**ABSTRACT**  
Fact-checking is a popular countermeasure against misinformation but the massive volume of information online has spurred active research in the automation of the task. Like expert fact-checking, it is not enough for an automated fact-checker to just be accurate, but also be able to inform and convince the user of the validity of its prediction. This becomes viable with explainable artificial intelligence (XAI). In this work, we conduct a study of XAI fact-checkers involving 180 participants to determine how users' actions towards news and their attitudes towards explanations are affected by the XAI. Our results suggest that XAI has limited effects on users' agreement with the veracity prediction of the automated fact-checker and on their intents to share news. However, XAI does nudge them towards forming uniform judgments of news veracity, thereby signaling a reliance on the explanations. We also found polarizing preferences towards XAI, raising several design considerations on these.

{{</citation>}}


### (108/133) My Model is Unfair, Do People Even Care? Visual Design Affects Trust and Perceived Bias in Machine Learning (Aimen Gaba et al., 2023)

{{<citation>}}

Aimen Gaba, Zhanna Kaufman, Jason Chueng, Marie Shvakel, Kyle Wm. Hall, Yuriy Brun, Cindy Xiong Bearfield. (2023)  
**My Model is Unfair, Do People Even Care? Visual Design Affects Trust and Perceived Bias in Machine Learning**  

---
Primary Category: cs.HC  
Categories: H-5-0, cs-HC, cs.HC  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.03299v1)  

---


**ABSTRACT**  
Machine learning technology has become ubiquitous, but, unfortunately, often exhibits bias. As a consequence, disparate stakeholders need to interact with and make informed decisions about using machine learning models in everyday systems. Visualization technology can support stakeholders in understanding and evaluating trade-offs between, for example, accuracy and fairness of models. This paper aims to empirically answer "Can visualization design choices affect a stakeholder's perception of model bias, trust in a model, and willingness to adopt a model?" Through a series of controlled, crowd-sourced experiments with more than 1,500 participants, we identify a set of strategies people follow in deciding which models to trust. Our results show that men and women prioritize fairness and performance differently and that visual design choices significantly affect that prioritization. For example, women trust fairer models more often than men do, participants value fairness more when it is explained using text than as a bar chart, and being explicitly told a model is biased has a bigger impact than showing past biased performance. We test the generalizability of our results by comparing the effect of multiple textual and visual design choices and offer potential explanations of the cognitive mechanisms behind the difference in fairness perception and trust. Our research guides design considerations to support future work developing visualization systems for machine learning.

{{</citation>}}


### (109/133) Notably Inaccessible -- Data Driven Understanding of Data Science Notebook (In)Accessibility (Venkatesh Potluri et al., 2023)

{{<citation>}}

Venkatesh Potluri, Sudheesh Singanamalla, Nussara Tieanklin, Jennifer Mankoff. (2023)  
**Notably Inaccessible -- Data Driven Understanding of Data Science Notebook (In)Accessibility**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs-SE, cs.HC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.03241v1)  

---


**ABSTRACT**  
Computational notebooks, tools that facilitate storytelling through exploration, data analysis, and information visualization, have become the widely accepted standard in the data science community. These notebooks have been widely adopted through notebook software such as Jupyter, Datalore and Google Colab, both in academia and industry. While there is extensive research to learn how data scientists use computational notebooks, identify their pain points, and enable collaborative data science practices, very little is known about the various accessibility barriers experienced by blind and visually impaired (BVI) users using these notebooks. BVI users are unable to use computational notebook interfaces due to (1) inaccessibility of the interface, (2) common ways in which data is represented in these interfaces, and (3) inability for popular libraries to provide accessible outputs. We perform a large scale systematic analysis of 100000 Jupyter notebooks to identify various accessibility challenges in published notebooks affecting the creation and consumption of these notebooks. Through our findings, we make recommendations to improve accessibility of the artifacts of a notebook, suggest authoring practices, and propose changes to infrastructure to make notebooks accessible. An accessible PDF can be obtained at https://blvi.dev/noteably-inaccessible-paper

{{</citation>}}


## cs.DB (3)



### (110/133) Generative Benchmark Creation for Table Union Search (Koyena Pal et al., 2023)

{{<citation>}}

Koyena Pal, Aamod Khatiwada, Roee Shraga, Renée J. Miller. (2023)  
**Generative Benchmark Creation for Table Union Search**  

---
Primary Category: cs.DB  
Categories: cs-CL, cs-DB, cs-LG, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03883v1)  

---


**ABSTRACT**  
Data management has traditionally relied on synthetic data generators to generate structured benchmarks, like the TPC suite, where we can control important parameters like data size and its distribution precisely. These benchmarks were central to the success and adoption of database management systems. But more and more, data management problems are of a semantic nature. An important example is finding tables that can be unioned. While any two tables with the same cardinality can be unioned, table union search is the problem of finding tables whose union is semantically coherent. Semantic problems cannot be benchmarked using synthetic data. Our current methods for creating benchmarks involve the manual curation and labeling of real data. These methods are not robust or scalable and perhaps more importantly, it is not clear how robust the created benchmarks are. We propose to use generative AI models to create structured data benchmarks for table union search. We present a novel method for using generative models to create tables with specified properties. Using this method, we create a new benchmark containing pairs of tables that are both unionable and non-unionable but related. We thoroughly evaluate recent existing table union search methods over existing benchmarks and our new benchmark. We also present and evaluate a new table search methods based on recent large language models over all benchmarks. We show that the new benchmark is more challenging for all methods than hand-curated benchmarks, specifically, the top-performing method achieves a Mean Average Precision of around 60%, over 30% less than its performance on existing manually created benchmarks. We examine why this is the case and show that the new benchmark permits more detailed analysis of methods, including a study of both false positives and false negatives that were not possible with existing benchmarks.

{{</citation>}}


### (111/133) A Polystore Architecture Using Knowledge Graphs to Support Queries on Heterogeneous Data Stores (Leonardo Guerreiro Azevedo et al., 2023)

{{<citation>}}

Leonardo Guerreiro Azevedo, Renan Francisco Santos Souza, Elton F. de S. Soares, Raphael M. Thiago, Julio Cesar Cardoso Tesolin, Ann C. Oliveira, Marcio Ferreira Moreno. (2023)  
**A Polystore Architecture Using Knowledge Graphs to Support Queries on Heterogeneous Data Stores**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.03584v1)  

---


**ABSTRACT**  
Modern applications commonly need to manage dataset types composed of heterogeneous data and schemas, making it difficult to access them in an integrated way. A single data store to manage heterogeneous data using a common data model is not effective in such a scenario, which results in the domain data being fragmented in the data stores that best fit their storage and access requirements (e.g., NoSQL, relational DBMS, or HDFS). Besides, organization workflows independently consume these fragments, and usually, there is no explicit link among the fragments that would be useful to support an integrated view. The research challenge tackled by this work is to provide the means to query heterogeneous data residing on distinct data repositories that are not explicitly connected. We propose a federated database architecture by providing a single abstract global conceptual schema to users, allowing them to write their queries, encapsulating data heterogeneity, location, and linkage by employing: (i) meta-models to represent the global conceptual schema, the remote data local conceptual schemas, and mappings among them; (ii) provenance to create explicit links among the consumed and generated data residing in separate datasets. We evaluated the architecture through its implementation as a polystore service, following a microservice architecture approach, in a scenario that simulates a real case in Oil \& Gas industry. Also, we compared the proposed architecture to a relational multidatabase system based on foreign data wrappers, measuring the user's cognitive load to write a query (or query complexity) and the query processing time. The results demonstrated that the proposed architecture allows query writing two times less complex than the one written for the relational multidatabase system, adding an excess of no more than 30% in query processing time.

{{</citation>}}


### (112/133) CAESURA: Language Models as Multi-Modal Query Planners (Matthias Urban et al., 2023)

{{<citation>}}

Matthias Urban, Carsten Binnig. (2023)  
**CAESURA: Language Models as Multi-Modal Query Planners**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03424v1)  

---


**ABSTRACT**  
Traditional query planners translate SQL queries into query plans to be executed over relational data. However, it is impossible to query other data modalities, such as images, text, or video stored in modern data systems such as data lakes using these query planners. In this paper, we propose Language-Model-Driven Query Planning, a new paradigm of query planning that uses Language Models to translate natural language queries into executable query plans. Different from relational query planners, the resulting query plans can contain complex operators that are able to process arbitrary modalities. As part of this paper, we present a first GPT-4 based prototype called CEASURA and show the general feasibility of this idea on two datasets. Finally, we discuss several ideas to improve the query planning capabilities of today's Language Models.

{{</citation>}}


## cs.SE (1)



### (113/133) Evaluating and Explaining Large Language Models for Code Using Syntactic Structures (David N Palacio et al., 2023)

{{<citation>}}

David N Palacio, Alejandro Velasco, Daniel Rodriguez-Cardenas, Kevin Moran, Denys Poshyvanyk. (2023)  
**Evaluating and Explaining Large Language Models for Code Using Syntactic Structures**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03873v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) for code are a family of high-parameter, transformer-based neural networks pre-trained on massive datasets of both natural and programming languages. These models are rapidly being employed in commercial AI-based developer tools, such as GitHub CoPilot. However, measuring and explaining their effectiveness on programming tasks is a challenging proposition, given their size and complexity. The methods for evaluating and explaining LLMs for code are inextricably linked. That is, in order to explain a model's predictions, they must be reliably mapped to fine-grained, understandable concepts. Once this mapping is achieved, new methods for detailed model evaluations are possible. However, most current explainability techniques and evaluation benchmarks focus on model robustness or individual task performance, as opposed to interpreting model predictions.   To this end, this paper introduces ASTxplainer, an explainability method specific to LLMs for code that enables both new methods for LLM evaluation and visualizations of LLM predictions that aid end-users in understanding model predictions. At its core, ASTxplainer provides an automated method for aligning token predictions with AST nodes, by extracting and aggregating normalized model logits within AST structures. To demonstrate the practical benefit of ASTxplainer, we illustrate the insights that our framework can provide by performing an empirical evaluation on 12 popular LLMs for code using a curated dataset of the most popular GitHub projects. Additionally, we perform a user study examining the usefulness of an ASTxplainer-derived visualization of model predictions aimed at enabling model users to explain predictions. The results of these studies illustrate the potential for ASTxplainer to provide insights into LLM effectiveness, and aid end-users in understanding predictions.

{{</citation>}}


## cs.CY (1)



### (114/133) A Cost Analysis of Generative Language Models and Influence Operations (Micah Musser, 2023)

{{<citation>}}

Micah Musser. (2023)  
**A Cost Analysis of Generative Language Models and Influence Operations**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.03740v1)  

---


**ABSTRACT**  
Despite speculation that recent large language models (LLMs) are likely to be used maliciously to improve the quality or scale of influence operations, uncertainty persists regarding the economic value that LLMs offer propagandists. This research constructs a model of costs facing propagandists for content generation at scale and analyzes (1) the potential savings that LLMs could offer propagandists, (2) the potential deterrent effect of monitoring controls on API-accessible LLMs, and (3) the optimal strategy for propagandists choosing between multiple private and/or open source LLMs when conducting influence operations. Primary results suggest that LLMs need only produce usable outputs with relatively low reliability (roughly 25%) to offer cost savings to propagandists, that the potential reduction in content generation costs can be quite high (up to 70% for a highly reliable model), and that monitoring capabilities have sharply limited cost imposition effects when alternative open source models are available. In addition, these results suggest that nation-states -- even those conducting many large-scale influence operations per year -- are unlikely to benefit economically from training custom LLMs specifically for use in influence operations.

{{</citation>}}


## cs.RO (3)



### (115/133) SEM-GAT: Explainable Semantic Pose Estimation using Learned Graph Attention (Efimia Panagiotaki et al., 2023)

{{<citation>}}

Efimia Panagiotaki, Daniele De Martini, Georgi Pramatarov, Matthew Gadd, Lars Kunze. (2023)  
**SEM-GAT: Explainable Semantic Pose Estimation using Learned Graph Attention**  

---
Primary Category: cs.RO  
Categories: I-2-9; I-2-10; I-2-4; I-4-8; I-5-1; I-5-2, cs-AI, cs-CV, cs-RO, cs.RO  
Keywords: Attention, GNN  
[Paper Link](http://arxiv.org/abs/2308.03718v1)  

---


**ABSTRACT**  
This paper proposes a GNN-based method for exploiting semantics and local geometry to guide the identification of reliable pointcloud registration candidates. Semantic and morphological features of the environment serve as key reference points for registration, enabling accurate lidar-based pose estimation. Our novel lightweight static graph structure informs our attention-based keypoint node aggregation GNN network by identifying semantic instance-based relationships, acting as inductive bias to significantly reduce the computational burden of pointcloud registration. By connecting candidate nodes and exploiting cross-graph attention, we identify confidence scores for all potential registration correspondences, estimating the displacement between pointcloud scans. Our pipeline enables introspective analysis of the model's performance by correlating it with the individual contributions of local structures in the environment, providing valuable insights into the system's behaviour. We test our method on the KITTI odometry dataset, achieving competitive accuracy compared to benchmark methods and a higher track smoothness while relying on significantly fewer network parameters.

{{</citation>}}


### (116/133) Robots as AI Double Agents: Privacy in Motion Planning (Rahul Shome et al., 2023)

{{<citation>}}

Rahul Shome, Zachary Kingston, Lydia E. Kavraki. (2023)  
**Robots as AI Double Agents: Privacy in Motion Planning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03385v1)  

---


**ABSTRACT**  
Robotics and automation are poised to change the landscape of home and work in the near future. Robots are adept at deliberately moving, sensing, and interacting with their environments. The pervasive use of this technology promises societal and economic payoffs due to its capabilities - conversely, the capabilities of robots to move within and sense the world around them is susceptible to abuse. Robots, unlike typical sensors, are inherently autonomous, active, and deliberate. Such automated agents can become AI double agents liable to violate the privacy of coworkers, privileged spaces, and other stakeholders. In this work we highlight the understudied and inevitable threats to privacy that can be posed by the autonomous, deliberate motions and sensing of robots. We frame the problem within broader sociotechnological questions alongside a comprehensive review. The privacy-aware motion planning problem is formulated in terms of cost functions that can be modified to induce privacy-aware behavior - preserving, agnostic, or violating. Simulated case studies in manipulation and navigation, with altered cost functions, are used to demonstrate how privacy-violating threats can be easily injected, sometimes with only small changes in performance (solution path lengths). Such functionality is already widely available. This preliminary work is meant to lay the foundations for near-future, holistic, interdisciplinary investigations that can address questions surrounding privacy in intelligent robotic behaviors determined by planning algorithms.

{{</citation>}}


### (117/133) TempFuser: Learning Tactical and Agile Flight Maneuvers in Aerial Dogfights using a Long Short-Term Temporal Fusion Transformer (Hyunki Seong et al., 2023)

{{<citation>}}

Hyunki Seong, David Hyunchul Shim. (2023)  
**TempFuser: Learning Tactical and Agile Flight Maneuvers in Aerial Dogfights using a Long Short-Term Temporal Fusion Transformer**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03257v1)  

---


**ABSTRACT**  
Aerial dogfights necessitate understanding the tactically changing maneuvers from a long-term perspective, along with the rapidly changing aerodynamics from a short-term view. In this paper, we propose a novel long short-term temporal fusion transformer (TempFuser) for a policy network in aerial dogfights. Our method uses two LSTM-based input embeddings to encode long-term, sparse state trajectories, as well as short-term, dense state trajectories. By integrating the two embeddings through a transformer encoder, the method subsequently derives end-to-end flight commands for agile and tactical maneuvers. We formulate a deep reinforcement learning framework to train our TempFuser-based policy model. We then extensively validate our model, demonstrating that it outperforms other baseline models against a diverse range of opponent aircraft in a high-fidelity environment. Our model successfully learns basic fighter maneuvers, human pilot-like tactical maneuvers, and robust supersonic pursuit in low altitudes without explicitly coded prior knowledge. Videos are available at \url{https://sites.google.com/view/tempfuser}

{{</citation>}}


## cs.LO (1)



### (118/133) Combining Proofs for Description Logic and Concrete Domain Reasoning (Technical Report) (Christian Alrabbaa et al., 2023)

{{<citation>}}

Christian Alrabbaa, Franz Baader, Stefan Borgwardt, Patrick Koopmann, Alisa Kovtunova. (2023)  
**Combining Proofs for Description Logic and Concrete Domain Reasoning (Technical Report)**  

---
Primary Category: cs.LO  
Categories: 68-06, F-4-1; I-2-4, cs-LO, cs.LO  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.03705v1)  

---


**ABSTRACT**  
Logic-based approaches to AI have the advantage that their behavior can in principle be explained with the help of proofs of the computed consequences. For ontologies based on Description Logic (DL), we have put this advantage into practice by showing how proofs for consequences derived by DL reasoners can be computed and displayed in a user-friendly way. However, these methods are insufficient in applications where also numerical reasoning is relevant. The present paper considers proofs for DLs extended with concrete domains (CDs) based on the rational numbers, which leave reasoning tractable if integrated into the lightweight DL $\mathcal{E}\hspace{-0.1em}\mathcal{L}_\bot$. Since no implemented DL reasoner supports these CDs, we first develop reasoning procedures for them, and show how they can be combined with reasoning approaches for pure DLs, both for $\mathcal{E}\hspace{-0.1em}\mathcal{L}_\bot$ and the more expressive DL $\mathcal{ALC}$. These procedures are designed such that it is easy to extract proofs from them. We show how the extracted CD proofs can be combined with proofs on the DL side into integrated proofs that explain both the DL and the CD reasoning.

{{</citation>}}


## cs.PF (1)



### (119/133) Evaluation of ARM CPUs for IceCube available through Google Kubernetes Engine (Igor Sfiligoi et al., 2023)

{{<citation>}}

Igor Sfiligoi, David Schultz, Benedikt Riedel, Frank Würthwein. (2023)  
**Evaluation of ARM CPUs for IceCube available through Google Kubernetes Engine**  

---
Primary Category: cs.PF  
Categories: astro-ph-IM, cs-PF, cs.PF  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.03678v1)  

---


**ABSTRACT**  
The IceCube experiment has substantial simulation needs and is in continuous search for the most cost-effective ways to satisfy them. The most CPU-intensive part relies on CORSIKA, a cosmic ray air shower simulation. Historically, IceCube relied exclusively on x86-based CPUs, like Intel Xeon and AMD EPYC, but recently server-class ARM-based CPUs are also becoming available, both on-prem and in the cloud. In this paper we present our experience in running a sample CORSIKA simulation on both ARM and x86 CPUs available through Google Kubernetes Engine (GKE). We used the production binaries for the x86 instances, but had to build the binaries for ARM instances from source code, which turned out to be mostly painless. Our benchmarks show that ARM-based CPUs in GKE were not only the most cost-effective but were also the fastest in absolute terms in all the tested configurations. While the advantage is not drastic, about 20% in cost-effectiveness and less than 10% in absolute terms, it is still large enough to warrant an investment in ARM support for IceCube.

{{</citation>}}


## cs.NE (1)



### (120/133) Implementing Immune Repertoire Models Using Weighted Finite State Machines (Gijs Schröder et al., 2023)

{{<citation>}}

Gijs Schröder, Inge MN Wortel, Johannes Textor. (2023)  
**Implementing Immune Repertoire Models Using Weighted Finite State Machines**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03637v1)  

---


**ABSTRACT**  
The adaptive immune system's T and B cells can be viewed as large populations of simple, diverse classifiers. Artificial immune systems (AIS) $\unicode{x2013}$ algorithmic models of T or B cell repertoires $\unicode{x2013}$ are used in both computational biology and natural computing to investigate how the immune system adapts to its changing environments. However, researchers have struggled to build such systems at scale. For string-based AISs, finite state machines (FSMs) can store cell repertoires in compressed representations that are orders of magnitude smaller than explicitly stored receptor sets. This strategy allows AISs with billions of receptors to be generated in a matter of seconds. However, to date, these FSM-based AISs have been unable to deal with multiplicity in input data. Here, we show how weighted FSMs can be used to represent cell repertoires and model immunological processes like negative and positive selection, while also taking into account the multiplicity of input data. We use our method to build simple immune-inspired classifier systems that solve various toy problems in anomaly detection, showing how weights can be crucial for both performance and robustness to parameters. Our approach can potentially be extended to increase the scale of other population-based machine learning algorithms such as learning classifier systems.

{{</citation>}}


## eess.IV (3)



### (121/133) Adaptive Semi-Supervised Segmentation of Brain Vessels with Ambiguous Labels (Fengming Lin et al., 2023)

{{<citation>}}

Fengming Lin, Yan Xia, Nishant Ravikumar, Qiongyao Liu, Michael MacRaild, Alejandro F Frangi. (2023)  
**Adaptive Semi-Supervised Segmentation of Brain Vessels with Ambiguous Labels**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.03613v1)  

---


**ABSTRACT**  
Accurate segmentation of brain vessels is crucial for cerebrovascular disease diagnosis and treatment. However, existing methods face challenges in capturing small vessels and handling datasets that are partially or ambiguously annotated. In this paper, we propose an adaptive semi-supervised approach to address these challenges. Our approach incorporates innovative techniques including progressive semi-supervised learning, adaptative training strategy, and boundary enhancement. Experimental results on 3DRA datasets demonstrate the superiority of our method in terms of mesh-based segmentation metrics. By leveraging the partially and ambiguously labeled data, which only annotates the main vessels, our method achieves impressive segmentation performance on mislabeled fine vessels, showcasing its potential for clinical applications.

{{</citation>}}


### (122/133) High-Resolution Cranial Defect Reconstruction by Iterative, Low-Resolution, Point Cloud Completion Transformers (Marek Wodzinski et al., 2023)

{{<citation>}}

Marek Wodzinski, Mateusz Daniol, Daria Hemmerling, Miroslaw Socha. (2023)  
**High-Resolution Cranial Defect Reconstruction by Iterative, Low-Resolution, Point Cloud Completion Transformers**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, cs-LG, cs-NA, eess-IV, eess.IV, math-NA  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.03813v1)  

---


**ABSTRACT**  
Each year thousands of people suffer from various types of cranial injuries and require personalized implants whose manual design is expensive and time-consuming. Therefore, an automatic, dedicated system to increase the availability of personalized cranial reconstruction is highly desirable. The problem of the automatic cranial defect reconstruction can be formulated as the shape completion task and solved using dedicated deep networks. Currently, the most common approach is to use the volumetric representation and apply deep networks dedicated to image segmentation. However, this approach has several limitations and does not scale well into high-resolution volumes, nor takes into account the data sparsity. In our work, we reformulate the problem into a point cloud completion task. We propose an iterative, transformer-based method to reconstruct the cranial defect at any resolution while also being fast and resource-efficient during training and inference. We compare the proposed methods to the state-of-the-art volumetric approaches and show superior performance in terms of GPU memory consumption while maintaining high-quality of the reconstructed defects.

{{</citation>}}


### (123/133) Enhancing Nucleus Segmentation with HARU-Net: A Hybrid Attention Based Residual U-Blocks Network (Junzhou Chen et al., 2023)

{{<citation>}}

Junzhou Chen, Qian Huang, Yulin Chen, Linyi Qian, Chengyuan Yu. (2023)  
**Enhancing Nucleus Segmentation with HARU-Net: A Hybrid Attention Based Residual U-Blocks Network**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.03382v2)  

---


**ABSTRACT**  
Nucleus image segmentation is a crucial step in the analysis, pathological diagnosis, and classification, which heavily relies on the quality of nucleus segmentation. However, the complexity of issues such as variations in nucleus size, blurred nucleus contours, uneven staining, cell clustering, and overlapping cells poses significant challenges. Current methods for nucleus segmentation primarily rely on nuclear morphology or contour-based approaches. Nuclear morphology-based methods exhibit limited generalization ability and struggle to effectively predict irregular-shaped nuclei, while contour-based extraction methods face challenges in accurately segmenting overlapping nuclei. To address the aforementioned issues, we propose a dual-branch network using hybrid attention based residual U-blocks for nucleus instance segmentation. The network simultaneously predicts target information and target contours. Additionally, we introduce a post-processing method that combines the target information and target contours to distinguish overlapping nuclei and generate an instance segmentation image. Within the network, we propose a context fusion block (CF-block) that effectively extracts and merges contextual information from the network. Extensive quantitative evaluations are conducted to assess the performance of our method. Experimental results demonstrate the superior performance of the proposed method compared to state-of-the-art approaches on the BNS, MoNuSeg, CoNSeg, and CPM-17 datasets.

{{</citation>}}


## cs.IR (4)



### (124/133) Multi-View Graph Convolutional Network for Multimedia Recommendation (Penghang Yu et al., 2023)

{{<citation>}}

Penghang Yu, Zhiyi Tan, Guanming Lu, Bing-Kun Bao. (2023)  
**Multi-View Graph Convolutional Network for Multimedia Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2308.03588v1)  

---


**ABSTRACT**  
Multimedia recommendation has received much attention in recent years. It models user preferences based on both behavior information and item multimodal information. Though current GCN-based methods achieve notable success, they suffer from two limitations: (1) Modality noise contamination to the item representations. Existing methods often mix modality features and behavior features in a single view (e.g., user-item view) for propagation, the noise in the modality features may be amplified and coupled with behavior features. In the end, it leads to poor feature discriminability; (2) Incomplete user preference modeling caused by equal treatment of modality features. Users often exhibit distinct modality preferences when purchasing different items. Equally fusing each modality feature ignores the relative importance among different modalities, leading to the suboptimal user preference modeling. To tackle the above issues, we propose a novel Multi-View Graph Convolutional Network for the multimedia recommendation. Specifically, to avoid modality noise contamination, the modality features are first purified with the aid of item behavior information. Then, the purified modality features of items and behavior features are enriched in separate views, including the user-item view and the item-item view. In this way, the distinguishability of features is enhanced. Meanwhile, a behavior-aware fuser is designed to comprehensively model user preferences by adaptively learning the relative importance of different modality features. Furthermore, we equip the fuser with a self-supervised auxiliary task. This task is expected to maximize the mutual information between the fused multimodal features and behavior features, so as to capture complementary and supplementary preference information simultaneously. Extensive experiments on three public datasets demonstrate the effectiveness of our methods.

{{</citation>}}


### (125/133) Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation (Taichi Liu et al., 2023)

{{<citation>}}

Taichi Liu, Chen Gao, Zhenyu Wang, Dong Li, Jianye Hao, Depeng Jin, Yong Li. (2023)  
**Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.03470v1)  

---


**ABSTRACT**  
Graph Neural Network (GNN)-based models have become the mainstream approach for recommender systems. Despite the effectiveness, they are still suffering from the cold-start problem, i.e., recommend for few-interaction items. Existing GNN-based recommendation models to address the cold-start problem mainly focus on utilizing auxiliary features of users and items, leaving the user-item interactions under-utilized. However, embeddings distributions of cold and warm items are still largely different, since cold items' embeddings are learned from lower-popularity interactions, while warm items' embeddings are from higher-popularity interactions. Thus, there is a seesaw phenomenon, where the recommendation performance for the cold and warm items cannot be improved simultaneously. To this end, we proposed a Uncertainty-aware Consistency learning framework for Cold-start item recommendation (shorten as UCC) solely based on user-item interactions. Under this framework, we train the teacher model (generator) and student model (recommender) with consistency learning, to ensure the cold items with additionally generated low-uncertainty interactions can have similar distribution with the warm items. Therefore, the proposed framework improves the recommendation of cold and warm items at the same time, without hurting any one of them. Extensive experiments on benchmark datasets demonstrate that our proposed method significantly outperforms state-of-the-art methods on both warm and cold items, with an average performance improvement of 27.6%.

{{</citation>}}


### (126/133) Hierarchical Contrastive Learning with Multiple Augmentation for Sequential Recommendation (Dongjun Lee et al., 2023)

{{<citation>}}

Dongjun Lee, Donggeun Ko, Jaekwang Kim. (2023)  
**Hierarchical Contrastive Learning with Multiple Augmentation for Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Augmentation, Contrastive Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.03400v1)  

---


**ABSTRACT**  
Sequential recommendation addresses the issue of preference drift by predicting the next item based on the user's previous behaviors. Recently, a promising approach using contrastive learning has emerged, demonstrating its effectiveness in recommending items under sparse user-item interactions. Significantly, the effectiveness of combinations of various augmentation methods has been demonstrated in different domains, particularly in computer vision. However, when it comes to augmentation within a contrastive learning framework in sequential recommendation, previous research has only focused on limited conditions and simple structures. Thus, it is still possible to extend existing approaches to boost the effects of augmentation methods by using progressed structures with the combinations of multiple augmentation methods. In this work, we propose a novel framework called Hierarchical Contrastive Learning with Multiple Augmentation for Sequential Recommendation(HCLRec) to overcome the aforementioned limitation. Our framework leverages existing augmentation methods hierarchically to improve performance. By combining augmentation methods continuously, we generate low-level and high-level view pairs. We employ a Transformers-based model to encode the input sequence effectively. Furthermore, we introduce additional blocks consisting of Transformers and position-wise feed-forward network(PFFN) layers to learn the invariance of the original sequences from hierarchically augmented views. We pass the input sequence to subsequent layers based on the number of increment levels applied to the views to handle various augmentation levels. Within each layer, we compute contrastive loss between pairs of views at the same level. Extensive experiments demonstrate that our proposed method outperforms state-of-the-art approaches and that HCLRec is robust even when faced with the problem of sparse interaction.

{{</citation>}}


### (127/133) Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM (Bin Yin et al., 2023)

{{<citation>}}

Bin Yin, Junjie Xie, Yu Qin, Zixiang Ding, Zhichao Feng, Xiang Li, Wei Lin. (2023)  
**Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.03333v1)  

---


**ABSTRACT**  
The analysis and mining of user heterogeneous behavior are of paramount importance in recommendation systems. However, the conventional approach of incorporating various types of heterogeneous behavior into recommendation models leads to feature sparsity and knowledge fragmentation issues. To address this challenge, we propose a novel approach for personalized recommendation via Large Language Model (LLM), by extracting and fusing heterogeneous knowledge from user heterogeneous behavior information. In addition, by combining heterogeneous knowledge and recommendation tasks, instruction tuning is performed on LLM for personalized recommendations. The experimental results demonstrate that our method can effectively integrate user heterogeneous behavior and significantly improve recommendation performance.

{{</citation>}}


## cs.MM (2)



### (128/133) COPA: Efficient Vision-Language Pre-training Through Collaborative Object- and Patch-Text Alignment (Chaoya Jiang et al., 2023)

{{<citation>}}

Chaoya Jiang, Haiyang Xu, Wei Ye, Qinghao Ye, Chenliang Li, Ming Yan, Bin Bi, Shikun Zhang, Ji Zhang, Fei Huang. (2023)  
**COPA: Efficient Vision-Language Pre-training Through Collaborative Object- and Patch-Text Alignment**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.03475v1)  

---


**ABSTRACT**  
Vision-Language Pre-training (VLP) methods based on object detection enjoy the rich knowledge of fine-grained object-text alignment but at the cost of computationally expensive inference. Recent Visual-Transformer (ViT)-based approaches circumvent this issue while struggling with long visual sequences without detailed cross-modal alignment information. This paper introduces a ViT-based VLP technique that efficiently incorporates object information through a novel patch-text alignment mechanism. Specifically, we convert object-level signals into patch-level ones and devise a Patch-Text Alignment pre-training task (PTA) to learn a text-aware patch detector. By using off-the-shelf delicate object annotations in 5\% training images, we jointly train PTA with other conventional VLP objectives in an end-to-end manner, bypassing the high computational cost of object detection and yielding an effective patch detector that accurately detects text-relevant patches, thus considerably reducing patch sequences and accelerating computation within the ViT backbone. Our experiments on a variety of widely-used benchmarks reveal that our method achieves a speedup of nearly 88\% compared to prior VLP models while maintaining competitive or superior performance on downstream tasks with similar model size and data scale.

{{</citation>}}


### (129/133) Cuing Without Sharing: A Federated Cued Speech Recognition Framework via Mutual Knowledge Distillation (Yuxuan Zhang et al., 2023)

{{<citation>}}

Yuxuan Zhang, Lei Liu, Li Liu. (2023)  
**Cuing Without Sharing: A Federated Cued Speech Recognition Framework via Mutual Knowledge Distillation**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Knowledge Distillation, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2308.03432v1)  

---


**ABSTRACT**  
Cued Speech (CS) is a visual coding tool to encode spoken languages at the phonetic level, which combines lip-reading and hand gestures to effectively assist communication among people with hearing impairments. The Automatic CS Recognition (ACSR) task aims to recognize CS videos into linguistic texts, which involves both lips and hands as two distinct modalities conveying complementary information. However, the traditional centralized training approach poses potential privacy risks due to the use of facial and gesture videos in CS data. To address this issue, we propose a new Federated Cued Speech Recognition (FedCSR) framework to train an ACSR model over the decentralized CS data without sharing private information. In particular, a mutual knowledge distillation method is proposed to maintain cross-modal semantic consistency of the Non-IID CS data, which ensures learning a unified feature space for both linguistic and visual information. On the server side, a globally shared linguistic model is trained to capture the long-term dependencies in the text sentences, which is aligned with the visual information from the local clients via visual-to-linguistic distillation. On the client side, the visual model of each client is trained with its own local data, assisted by linguistic-to-visual distillation treating the linguistic model as the teacher. To the best of our knowledge, this is the first approach to consider the federated ACSR task for privacy protection. Experimental results on the Chinese CS dataset with multiple cuers demonstrate that our approach outperforms both mainstream federated learning baselines and existing centralized state-of-the-art ACSR methods, achieving 9.7% performance improvement for character error rate (CER) and 15.0% for word error rate (WER).

{{</citation>}}


## math.NA (2)



### (130/133) Positional Embeddings for Solving PDEs with Evolutional Deep Neural Networks (Mariella Kast et al., 2023)

{{<citation>}}

Mariella Kast, Jan S Hesthaven. (2023)  
**Positional Embeddings for Solving PDEs with Evolutional Deep Neural Networks**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.03461v1)  

---


**ABSTRACT**  
This work extends the paradigm of evolutional deep neural networks (EDNNs) to solving parametric time-dependent partial differential equations (PDEs) on domains with geometric structure. By introducing positional embeddings based on eigenfunctions of the Laplace-Beltrami operator, geometric properties are encoded intrinsically and Dirichlet, Neumann and periodic boundary conditions of the PDE solution are enforced directly through the neural network architecture. The proposed embeddings lead to improved error convergence for static PDEs and extend EDNNs towards computational domains of realistic complexity. Several steps are taken to improve performance of EDNNs: Solving the EDNN update equation with a Krylov solver avoids the explicit assembly of Jacobians and enables scaling to larger neural networks. Computational efficiency is further improved by an ad-hoc active sampling scheme that uses the PDE dynamics to effectively sample collocation points. A modified linearly implicit Rosenbrock method is proposed to alleviate the time step requirements of stiff PDEs. Lastly, a completely training-free approach, which automatically enforces initial conditions and only requires time integration, is compared against EDNNs that are trained on the initial conditions. We report results for the Korteweg-de Vries equation, a nonlinear heat equation and (nonlinear) advection-diffusion problems on domains with and without holes and various boundary conditions, to demonstrate the effectiveness of the method. The numerical results highlight EDNNs as a promising surrogate model for parametrized PDEs with slow decaying Kolmogorov n-width.

{{</citation>}}


### (131/133) Friedrichs' systems discretized with the Discontinuous Galerkin method: domain decomposable model order reduction and Graph Neural Networks approximating vanishing viscosity solutions (Francesco Romor et al., 2023)

{{<citation>}}

Francesco Romor, Davide Torlo, Gianluigi Rozza. (2023)  
**Friedrichs' systems discretized with the Discontinuous Galerkin method: domain decomposable model order reduction and Graph Neural Networks approximating vanishing viscosity solutions**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.03378v1)  

---


**ABSTRACT**  
Friedrichs' systems (FS) are symmetric positive linear systems of first-order partial differential equations (PDEs), which provide a unified framework for describing various elliptic, parabolic and hyperbolic semi-linear PDEs such as the linearized Euler equations of gas dynamics, the equations of compressible linear elasticity and the Dirac-Klein-Gordon system. FS were studied to approximate PDEs of mixed elliptic and hyperbolic type in the same domain. For this and other reasons, the versatility of the discontinuous Galerkin method (DGM) represents the best approximation space for FS. We implement a distributed memory solver for stationary FS in deal.II. Our focus is model order reduction. Since FS model hyperbolic PDEs, they often suffer from a slow Kolmogorov n-width decay. We develop two approaches to tackle this problem. The first is domain decomposable reduced-order models (DD-ROMs). We will show that the DGM offers a natural formulation of DD-ROMs, in particular regarding interface penalties, compared to the continuous finite element method. We also develop new repartitioning strategies to obtain more efficient local approximations of the solution manifold. The second approach involves graph neural networks used to infer the limit of a succession of projection-based linear ROMs corresponding to lower viscosity constants: the heuristic behind is to develop a multi-fidelity super-resolution paradigm to mimic the mathematical convergence to vanishing viscosity solutions while exploiting to the most interpretable and certified projection-based ROMs.

{{</citation>}}


## cs.SD (1)



### (132/133) Improving Deep Attractor Network by BGRU and GMM for Speech Separation (Rawad Melhem et al., 2023)

{{<citation>}}

Rawad Melhem, Assef Jafar, Riad Hamadeh. (2023)  
**Improving Deep Attractor Network by BGRU and GMM for Speech Separation**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2308.03332v1)  

---


**ABSTRACT**  
Deep Attractor Network (DANet) is the state-of-the-art technique in speech separation field, which uses Bidirectional Long Short-Term Memory (BLSTM), but the complexity of the DANet model is very high. In this paper, a simplified and powerful DANet model is proposed using Bidirectional Gated neural network (BGRU) instead of BLSTM. The Gaussian Mixture Model (GMM) other than the k-means was applied in DANet as a clustering algorithm to reduce the complexity and increase the learning speed and accuracy. The metrics used in this paper are Signal to Distortion Ratio (SDR), Signal to Interference Ratio (SIR), Signal to Artifact Ratio (SAR), and Perceptual Evaluation Speech Quality (PESQ) score. Two speaker mixture datasets from TIMIT corpus were prepared to evaluate the proposed model, and the system achieved 12.3 dB and 2.94 for SDR and PESQ scores respectively, which were better than the original DANet model. Other improvements were 20.7% and 17.9% in the number of parameters and time training, respectively. The model was applied on mixed Arabic speech signals and the results were better than that in English.

{{</citation>}}


## cs.SI (1)



### (133/133) Quantifying the Impact of Large Language Models on Collective Opinion Dynamics (Chao Li et al., 2023)

{{<citation>}}

Chao Li, Xing Su, Chao Fan, Haoying Han, Cong Xue, Chunmo Zheng. (2023)  
**Quantifying the Impact of Large Language Models on Collective Opinion Dynamics**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.03313v1)  

---


**ABSTRACT**  
The process of opinion expression and exchange is a critical component of democratic societies. As people interact with large language models (LLMs) in the opinion shaping process different from traditional media, the impacts of LLMs are increasingly recognized and being concerned. However, the knowledge about how LLMs affect the process of opinion expression and exchange of social opinion networks is very limited. Here, we create an opinion network dynamics model to encode the opinions of LLMs, cognitive acceptability and usage strategies of individuals, and simulate the impact of LLMs on opinion dynamics in a variety of scenarios. The outcomes of the simulations inform about effective demand-oriented opinion network interventions. The results from this study suggested that the output opinion of LLMs has a unique and positive effect on the collective opinion difference. The marginal effect of cognitive acceptability on collective opinion formation is nonlinear and shows a decreasing trend. When people partially rely on LLMs, the exchange process of opinion becomes more intense and the diversity of opinion becomes more favorable. In fact, there is 38.6% more opinion diversity when people all partially rely on LLMs, compared to prohibiting the use of LLMs entirely. The optimal diversity of opinion was found when the fractions of people who do not use, partially rely on, and fully rely on LLMs reached roughly 4:12:1. Our experiments also find that introducing extra agents with opposite/neutral/random opinions, we can effectively mitigate the impact of biased/toxic output from LLMs. Our findings provide valuable insights into opinion dynamics in the age of LLMs, highlighting the need for customized interventions tailored to specific scenarios to address the drawbacks of improper output and use of LLMs.

{{</citation>}}
