---
draft: false
title: "arXiv @ 2023.07.02"
date: 2023-07-02
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.02"
    identifier: arxiv_20230702
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (26)](#cscl-26)
- [eess.IV (2)](#eessiv-2)
- [cs.IR (3)](#csir-3)
- [cs.CV (22)](#cscv-22)
- [cs.HC (1)](#cshc-1)
- [cs.LG (9)](#cslg-9)
- [cs.RO (6)](#csro-6)
- [cs.CY (1)](#cscy-1)
- [cs.SI (2)](#cssi-2)
- [cs.AI (6)](#csai-6)
- [stat.ML (2)](#statml-2)
- [cs.DS (1)](#csds-1)
- [cs.MA (1)](#csma-1)
- [cs.NE (2)](#csne-2)
- [cs.NI (1)](#csni-1)
- [cs.CR (2)](#cscr-2)
- [cs.SD (2)](#cssd-2)
- [cs.IT (1)](#csit-1)
- [q-bio.OT (1)](#q-bioot-1)

## cs.CL (26)



### (1/91) Still No Lie Detector for Language Models: Probing Empirical and Conceptual Roadblocks (B. A. Levinstein et al., 2023)

{{<citation>}}

B. A. Levinstein, Daniel A. Herrmann. (2023)  
**Still No Lie Detector for Language Models: Probing Empirical and Conceptual Roadblocks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.00175v1)  

---


**ABSTRACT**  
We consider the questions of whether or not large language models (LLMs) have beliefs, and, if they do, how we might measure them. First, we evaluate two existing approaches, one due to Azaria and Mitchell (2023) and the other to Burns et al. (2022). We provide empirical results that show that these methods fail to generalize in very basic ways. We then argue that, even if LLMs have beliefs, these methods are unlikely to be successful for conceptual reasons. Thus, there is still no lie-detector for LLMs. After describing our empirical results we take a step back and consider whether or not we should expect LLMs to have something like beliefs in the first place. We consider some recent arguments aiming to show that LLMs cannot have beliefs. We show that these arguments are misguided. We provide a more productive framing of questions surrounding the status of beliefs in LLMs, and highlight the empirical nature of the problem. We conclude by suggesting some concrete paths for future work.

{{</citation>}}


### (2/91) What do self-supervised speech models know about words? (Ankita Pasad et al., 2023)

{{<citation>}}

Ankita Pasad, Chung-Ming Chien, Shane Settle, Karen Livescu. (2023)  
**What do self-supervised speech models know about words?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.00162v1)  

---


**ABSTRACT**  
Many self-supervised speech models (S3Ms) have been introduced over the last few years, producing performance and data efficiency improvements for a variety of speech tasks. Evidence is emerging that different S3Ms encode linguistic information in different layers, and also that some S3Ms appear to learn phone-like sub-word units. However, the extent to which these models capture larger linguistic units, such as words, and where word-related information is encoded, remains unclear. In this study, we conduct several analyses of word segment representations extracted from different layers of three S3Ms: wav2vec2, HuBERT, and WavLM. We employ canonical correlation analysis (CCA), a lightweight analysis tool, to measure the similarity between these representations and word-level linguistic properties. We find that the maximal word-level linguistic content tends to be found in intermediate model layers, while some lower-level information like pronunciation is also retained in higher layers of HuBERT and WavLM. Syntactic and semantic word attributes have similar layer-wise behavior. We also find that, for all of the models tested, word identity information is concentrated near the center of each word segment. We then test the layer-wise performance of the same models, when used directly with no additional learned parameters, on several tasks: acoustic word discrimination, word segmentation, and semantic sentence similarity. We find similar layer-wise trends in performance, and furthermore, find that when using the best-performing layer of HuBERT or WavLM, it is possible to achieve performance on word segmentation and sentence similarity that rivals more complex existing approaches.

{{</citation>}}


### (3/91) SMILE: Evaluation and Domain Adaptation for Social Media Language Understanding (Vasilisa Bashlovkina et al., 2023)

{{<citation>}}

Vasilisa Bashlovkina, Riley Matthews, Zhaobin Kuang, Simon Baumgartner, Michael Bendersky. (2023)  
**SMILE: Evaluation and Domain Adaptation for Social Media Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2307.00135v1)  

---


**ABSTRACT**  
We study the ability of transformer-based language models (LMs) to understand social media language. Social media (SM) language is distinct from standard written language, yet existing benchmarks fall short of capturing LM performance in this socially, economically, and politically important domain. We quantify the degree to which social media language differs from conventional language and conclude that the difference is significant both in terms of token distribution and rate of linguistic shift. Next, we introduce a new benchmark for Social MedIa Language Evaluation (SMILE) that covers four SM platforms and eleven tasks. Finally, we show that learning a tokenizer and pretraining on a mix of social media and conventional language yields an LM that outperforms the best similar-sized alternative by 4.2 points on the overall SMILE score.

{{</citation>}}


### (4/91) iMETRE: Incorporating Markers of Entity Types for Relation Extraction (N Harsha Vardhan et al., 2023)

{{<citation>}}

N Harsha Vardhan, Manav Chaudhary. (2023)  
**iMETRE: Incorporating Markers of Entity Types for Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2307.00132v1)  

---


**ABSTRACT**  
Sentence-level relation extraction (RE) aims to identify the relationship between 2 entities given a contextual sentence. While there have been many attempts to solve this problem, the current solutions have a lot of room to improve. In this paper, we approach the task of relationship extraction in the financial dataset REFinD. Our approach incorporates typed entity markers representations and various models finetuned on the dataset, which has allowed us to achieve an F1 score of 69.65% on the validation set. Through this paper, we discuss various approaches and possible limitations.

{{</citation>}}


### (5/91) Information Extraction in Domain and Generic Documents: Findings from Heuristic-based and Data-driven Approaches (Shiyu Yuan et al., 2023)

{{<citation>}}

Shiyu Yuan, Carlo Lipizzi. (2023)  
**Information Extraction in Domain and Generic Documents: Findings from Heuristic-based and Data-driven Approaches**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, NER, NLP  
[Paper Link](http://arxiv.org/abs/2307.00130v1)  

---


**ABSTRACT**  
Information extraction (IE) plays very important role in natural language processing (NLP) and is fundamental to many NLP applications that used to extract structured information from unstructured text data. Heuristic-based searching and data-driven learning are two main stream implementation approaches. However, no much attention has been paid to document genre and length influence on IE tasks. To fill the gap, in this study, we investigated the accuracy and generalization abilities of heuristic-based searching and data-driven to perform two IE tasks: named entity recognition (NER) and semantic role labeling (SRL) on domain-specific and generic documents with different length. We posited two hypotheses: first, short documents may yield better accuracy results compared to long documents; second, generic documents may exhibit superior extraction outcomes relative to domain-dependent documents due to training document genre limitations. Our findings reveals that no single method demonstrated overwhelming performance in both tasks. For named entity extraction, data-driven approaches outperformed symbolic methods in terms of accuracy, particularly in short texts. In the case of semantic roles extraction, we observed that heuristic-based searching method and data-driven based model with syntax representation surpassed the performance of pure data-driven approach which only consider semantic information. Additionally, we discovered that different semantic roles exhibited varying accuracy levels with the same method. This study offers valuable insights for downstream text mining tasks, such as NER and SRL, when addressing various document features and genres.

{{</citation>}}


### (6/91) Meta-training with Demonstration Retrieval for Efficient Few-shot Learning (Aaron Mueller et al., 2023)

{{<citation>}}

Aaron Mueller, Kanika Narang, Lambert Mathias, Qifan Wang, Hamed Firooz. (2023)  
**Meta-training with Demonstration Retrieval for Efficient Few-shot Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLI, NLP, QA  
[Paper Link](http://arxiv.org/abs/2307.00119v1)  

---


**ABSTRACT**  
Large language models show impressive results on few-shot NLP tasks. However, these models are memory and computation-intensive. Meta-training allows one to leverage smaller models for few-shot generalization in a domain-general and task-agnostic manner; however, these methods alone results in models that may not have sufficient parameterization or knowledge to adapt quickly to a large variety of tasks. To overcome this issue, we propose meta-training with demonstration retrieval, where we use a dense passage retriever to retrieve semantically similar labeled demonstrations to each example for more varied supervision. By separating external knowledge from model parameters, we can use meta-training to train parameter-efficient models that generalize well on a larger variety of tasks. We construct a meta-training set from UnifiedQA and CrossFit, and propose a demonstration bank based on UnifiedQA tasks. To our knowledge, our work is the first to combine retrieval with meta-training, to use DPR models to retrieve demonstrations, and to leverage demonstrations from many tasks simultaneously, rather than randomly sampling demonstrations from the training set of the target task. Our approach outperforms a variety of targeted parameter-efficient and retrieval-augmented few-shot methods on QA, NLI, and text classification tasks (including SQuAD, QNLI, and TREC). Our approach can be meta-trained and fine-tuned quickly on a single GPU.

{{</citation>}}


### (7/91) Ticket-BERT: Labeling Incident Management Tickets with Language Models (Zhexiong Liu et al., 2023)

{{<citation>}}

Zhexiong Liu, Cris Benge, Siduo Jiang. (2023)  
**Ticket-BERT: Labeling Incident Management Tickets with Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Azure, BERT, Language Model, Microsoft  
[Paper Link](http://arxiv.org/abs/2307.00108v1)  

---


**ABSTRACT**  
An essential aspect of prioritizing incident tickets for resolution is efficiently labeling tickets with fine-grained categories. However, ticket data is often complex and poses several unique challenges for modern machine learning methods: (1) tickets are created and updated either by machines with pre-defined algorithms or by engineers with domain expertise that share different protocols, (2) tickets receive frequent revisions that update ticket status by modifying all or parts of ticket descriptions, and (3) ticket labeling is time-sensitive and requires knowledge updates and new labels per the rapid software and hardware improvement lifecycle. To handle these issues, we introduce Ticket- BERT which trains a simple yet robust language model for labeling tickets using our proposed ticket datasets. Experiments demonstrate the superiority of Ticket-BERT over baselines and state-of-the-art text classifiers on Azure Cognitive Services. We further encapsulate Ticket-BERT with an active learning cycle and deploy it on the Microsoft IcM system, which enables the model to quickly finetune on newly-collected tickets with a few annotations.

{{</citation>}}


### (8/91) Queer People are People First: Deconstructing Sexual Identity Stereotypes in Large Language Models (Harnoor Dhingra et al., 2023)

{{<citation>}}

Harnoor Dhingra, Preetiha Jayashanker, Sayali Moghe, Emma Strubell. (2023)  
**Queer People are People First: Deconstructing Sexual Identity Stereotypes in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.00101v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are trained primarily on minimally processed web text, which exhibits the same wide range of social biases held by the humans who created that content. Consequently, text generated by LLMs can inadvertently perpetuate stereotypes towards marginalized groups, like the LGBTQIA+ community. In this paper, we perform a comparative study of how LLMs generate text describing people with different sexual identities. Analyzing bias in the text generated by an LLM using regard score shows measurable bias against queer people. We then show that a post-hoc method based on chain-of-thought prompting using SHAP analysis can increase the regard of the sentence, representing a promising approach towards debiasing the output of LLMs in this setting.

{{</citation>}}


### (9/91) Meta-Reasoning: Semantics-Symbol Deconstruction For Large Language Models (Yiming Wang et al., 2023)

{{<citation>}}

Yiming Wang, Zhuosheng Zhang, Rui Wang. (2023)  
**Meta-Reasoning: Semantics-Symbol Deconstruction For Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2306.17820v1)  

---


**ABSTRACT**  
Symbolization methods in large language models (LLMs) have been shown effective to improve LLMs' reasoning ability. However, most of these approaches hinge on mapping natural languages to formal languages (e.g., Python, SQL) that are more syntactically complete and free of ambiguity. Although effective, they depart from the natural language itself and deviate from the habits of human thinking, and instead cater more to the execution mindset of computers. In contrast, we hope to simplify natural language by starting from the concept of symbols in linguistics itself, so that LLMs can learn the common formulation and general solution of reasoning problems wrapped in different natural semantics. From this consideration, we propose \textbf{Meta-Reasoning}, which allows LLMs to automatically accomplish semantic-symbol deconstruction, i.e., semantic resolution, to maximally reduce different questions of certain reasoning tasks to similar natural language representation, thus gaining the ability to learn by analogy and facilitating data-efficient in-context learning. Our experiments show that the Meta-Reasoning paradigm saliently enhances LLMs' reasoning performance with fewer demonstrations. They can learn not only reasoning chains but also general solutions to certain types of tasks. In particular, for symbolic reasoning tasks, such as 7-step Tracking Shuffled Objects, GPT-3 (text-davinci-002) achieves over 99% accuracy with only one Meta-Reasoning demonstration, outperforming all current LLMs with the standard chain-of-thought prompting.

{{</citation>}}


### (10/91) A Massive Scale Semantic Similarity Dataset of Historical English (Emily Silcock et al., 2023)

{{<citation>}}

Emily Silcock, Melissa Dell. (2023)  
**A Massive Scale Semantic Similarity Dataset of Historical English**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, econ-GN, q-fin-EC  
Keywords: Semantic Similarity  
[Paper Link](http://arxiv.org/abs/2306.17810v1)  

---


**ABSTRACT**  
A diversity of tasks use language models trained on semantic similarity data. While there are a variety of datasets that capture semantic similarity, they are either constructed from modern web data or are relatively small datasets created in the past decade by human annotators. This study utilizes a novel source, newly digitized articles from off-copyright, local U.S. newspapers, to assemble a massive-scale semantic similarity dataset spanning 70 years from 1920 to 1989 and containing nearly 400M positive semantic similarity pairs. Historically, around half of articles in U.S. local newspapers came from newswires like the Associated Press. While local papers reproduced articles from the newswire, they wrote their own headlines, which form abstractive summaries of the associated articles. We associate articles and their headlines by exploiting document layouts and language understanding. We then use deep neural methods to detect which articles are from the same underlying source, in the presence of substantial noise and abridgement. The headlines of reproduced articles form positive semantic similarity pairs. The resulting publicly available HEADLINES dataset is significantly larger than most existing semantic similarity datasets and covers a much longer span of time. It will facilitate the application of contrastively trained semantic similarity models to a variety of tasks, including the study of semantic change across space and time.

{{</citation>}}


### (11/91) Stay on topic with Classifier-Free Guidance (Guillaume Sanchez et al., 2023)

{{<citation>}}

Guillaume Sanchez, Honglu Fan, Alexander Spangher, Elad Levi, Pawan Sasanka Ammanamanchi, Stella Biderman. (2023)  
**Stay on topic with Classifier-Free Guidance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: GPT, LLaMA, PaLM  
[Paper Link](http://arxiv.org/abs/2306.17806v1)  

---


**ABSTRACT**  
Classifier-Free Guidance (CFG) has recently emerged in text-to-image generation as a lightweight technique to encourage prompt-adherence in generations. In this work, we demonstrate that CFG can be used broadly as an inference-time technique in pure language modeling. We show that CFG (1) improves the performance of Pythia, GPT-2 and LLaMA-family models across an array of tasks: Q\&A, reasoning, code generation, and machine translation, achieving SOTA on LAMBADA with LLaMA-7B over PaLM-540B; (2) brings improvements equivalent to a model with twice the parameter-count; (3) can stack alongside other inference-time methods like Chain-of-Thought and Self-Consistency, yielding further improvements in difficult tasks; (4) can be used to increase the faithfulness and coherence of assistants in challenging form-driven and content-driven prompts: in a human evaluation we show a 75\% preference for GPT4All using CFG over baseline.

{{</citation>}}


### (12/91) Towards Improving the Performance of Pre-Trained Speech Models for Low-Resource Languages Through Lateral Inhibition (Andrei-Marius Avram et al., 2023)

{{<citation>}}

Andrei-Marius Avram, Răzvan-Alexandru Smădu, Vasile Păiş, Dumitru-Clementin Cercel, Radu Ion, Dan Tufiş. (2023)  
**Towards Improving the Performance of Pre-Trained Speech Models for Low-Resource Languages Through Lateral Inhibition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Low-Resource, Transformer  
[Paper Link](http://arxiv.org/abs/2306.17792v1)  

---


**ABSTRACT**  
With the rise of bidirectional encoder representations from Transformer models in natural language processing, the speech community has adopted some of their development methodologies. Therefore, the Wav2Vec models were introduced to reduce the data required to obtain state-of-the-art results. This work leverages this knowledge and improves the performance of the pre-trained speech models by simply replacing the fine-tuning dense layer with a lateral inhibition layer inspired by the biological process. Our experiments on Romanian, a low-resource language, show an average improvement of 12.5% word error rate (WER) using the lateral inhibition layer. In addition, we obtain state-of-the-art results on both the Romanian Speech Corpus and the Robin Technical Acquisition Corpus with 1.78% WER and 29.64% WER, respectively.

{{</citation>}}


### (13/91) Token-Event-Role Structure-based Multi-Channel Document-Level Event Extraction (Qizhi Wan et al., 2023)

{{<citation>}}

Qizhi Wan, Changxuan Wan, Keli Xiao, Hui Xiong, Dexi Liu, Xiping Liu. (2023)  
**Token-Event-Role Structure-based Multi-Channel Document-Level Event Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Event Extraction  
[Paper Link](http://arxiv.org/abs/2306.17733v1)  

---


**ABSTRACT**  
Document-level event extraction is a long-standing challenging information retrieval problem involving a sequence of sub-tasks: entity extraction, event type judgment, and event type-specific multi-event extraction. However, addressing the problem as multiple learning tasks leads to increased model complexity. Also, existing methods insufficiently utilize the correlation of entities crossing different events, resulting in limited event extraction performance. This paper introduces a novel framework for document-level event extraction, incorporating a new data structure called token-event-role and a multi-channel argument role prediction module. The proposed data structure enables our model to uncover the primary role of tokens in multiple events, facilitating a more comprehensive understanding of event relationships. By leveraging the multi-channel prediction module, we transform entity and multi-event extraction into a single task of predicting token-event pairs, thereby reducing the overall parameter size and enhancing model efficiency. The results demonstrate that our approach outperforms the state-of-the-art method by 9.5 percentage points in terms of the F1 score, highlighting its superior performance in event extraction. Furthermore, an ablation study confirms the significant value of the proposed data structure in improving event extraction tasks, further validating its importance in enhancing the overall performance of the framework.

{{</citation>}}


### (14/91) A New Task and Dataset on Detecting Attacks on Human Rights Defenders (Shihao Ran et al., 2023)

{{<citation>}}

Shihao Ran, Di Lu, Joel Tetreault, Aoife Cahill, Alejandro Jaimes. (2023)  
**A New Task and Dataset on Detecting Attacks on Human Rights Defenders**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2306.17695v1)  

---


**ABSTRACT**  
The ability to conduct retrospective analyses of attacks on human rights defenders over time and by location is important for humanitarian organizations to better understand historical or ongoing human rights violations and thus better manage the global impact of such events. We hypothesize that NLP can support such efforts by quickly processing large collections of news articles to detect and summarize the characteristics of attacks on human rights defenders. To that end, we propose a new dataset for detecting Attacks on Human Rights Defenders (HRDsAttack) consisting of crowdsourced annotations on 500 online news articles. The annotations include fine-grained information about the type and location of the attacks, as well as information about the victim(s). We demonstrate the usefulness of the dataset by using it to train and evaluate baseline models on several sub-tasks to predict the annotated characteristics.

{{</citation>}}


### (15/91) X-RiSAWOZ: High-Quality End-to-End Multilingual Dialogue Datasets and Few-shot Agents (Mehrad Moradshahi et al., 2023)

{{<citation>}}

Mehrad Moradshahi, Tianhao Shen, Kalika Bali, Monojit Choudhury, Gaël de Chalendar, Anmol Goel, Sungkyun Kim, Prashant Kodali, Ponnurangam Kumaraguru, Nasredine Semmar, Sina J. Semnani, Jiwon Seo, Vivek Seshadri, Manish Shrivastava, Michael Sun, Aditya Yadavalli, Chaobin You, Deyi Xiong, Monica S. Lam. (2023)  
**X-RiSAWOZ: High-Quality End-to-End Multilingual Dialogue Datasets and Few-shot Agents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Multilingual  
[Paper Link](http://arxiv.org/abs/2306.17674v1)  

---


**ABSTRACT**  
Task-oriented dialogue research has mainly focused on a few popular languages like English and Chinese, due to the high dataset creation cost for a new language. To reduce the cost, we apply manual editing to automatically translated data. We create a new multilingual benchmark, X-RiSAWOZ, by translating the Chinese RiSAWOZ to 4 languages: English, French, Hindi, Korean; and a code-mixed English-Hindi language. X-RiSAWOZ has more than 18,000 human-verified dialogue utterances for each language, and unlike most multilingual prior work, is an end-to-end dataset for building fully-functioning agents.   The many difficulties we encountered in creating X-RiSAWOZ led us to develop a toolset to accelerate the post-editing of a new language dataset after translation. This toolset improves machine translation with a hybrid entity alignment technique that combines neural with dictionary-based methods, along with many automated and semi-automated validation checks.   We establish strong baselines for X-RiSAWOZ by training dialogue agents in the zero- and few-shot settings where limited gold data is available in the target language. Our results suggest that our translation and post-editing methodology and toolset can be used to create new high-quality multilingual dialogue agents cost-effectively. Our dataset, code, and toolkit are released open-source.

{{</citation>}}


### (16/91) Biomedical Language Models are Robust to Sub-optimal Tokenization (Bernal Jiménez Gutiérrez et al., 2023)

{{<citation>}}

Bernal Jiménez Gutiérrez, Huan Sun, Yu Su. (2023)  
**Biomedical Language Models are Robust to Sub-optimal Tokenization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NER, NLP  
[Paper Link](http://arxiv.org/abs/2306.17649v3)  

---


**ABSTRACT**  
As opposed to general English, many concepts in biomedical terminology have been designed in recent history by biomedical professionals with the goal of being precise and concise. This is often achieved by concatenating meaningful biomedical morphemes to create new semantic units. Nevertheless, most modern biomedical language models (LMs) are pre-trained using standard domain-specific tokenizers derived from large scale biomedical corpus statistics without explicitly leveraging the agglutinating nature of biomedical language. In this work, we first find that standard open-domain and biomedical tokenizers are largely unable to segment biomedical terms into meaningful components. Therefore, we hypothesize that using a tokenizer which segments biomedical terminology more accurately would enable biomedical LMs to improve their performance on downstream biomedical NLP tasks, especially ones which involve biomedical terms directly such as named entity recognition (NER) and entity linking. Surprisingly, we find that pre-training a biomedical LM using a more accurate biomedical tokenizer does not improve the entity representation quality of a language model as measured by several intrinsic and extrinsic measures such as masked language modeling prediction (MLM) accuracy as well as NER and entity linking performance. These quantitative findings, along with a case study which explores entity representation quality more directly, suggest that the biomedical pre-training process is quite robust to instances of sub-optimal tokenization.

{{</citation>}}


### (17/91) Feature Representation Learning for NL2SQL Generation Based on Coupling and Decoupling (Chenduo Hao et al., 2023)

{{<citation>}}

Chenduo Hao, Xu Zhang, Chuanbao Gao, Deyu Zhou. (2023)  
**Feature Representation Learning for NL2SQL Generation Based on Coupling and Decoupling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2306.17646v1)  

---


**ABSTRACT**  
The NL2SQL task involves parsing natural language statements into SQL queries. While most state-of-the-art methods treat NL2SQL as a slot-filling task and use feature representation learning techniques, they overlook explicit correlation features between the SELECT and WHERE clauses and implicit correlation features between sub-tasks within a single clause. To address this issue, we propose the Clause Feature Correlation Decoupling and Coupling (CFCDC) model, which uses a feature representation decoupling method to separate the SELECT and WHERE clauses at the parameter level. Next, we introduce a multi-task learning architecture to decouple implicit correlation feature representation between different SQL tasks in a specific clause. Moreover, we present an improved feature representation coupling module to integrate the decoupled tasks in the SELECT and WHERE clauses and predict the final SQL query. Our proposed CFCDC model demonstrates excellent performance on the WikiSQL dataset, with significant improvements in logic precision and execution accuracy. The source code for the model will be publicly available on GitHub

{{</citation>}}


### (18/91) Augmenting Holistic Review in University Admission using Natural Language Processing for Essays and Recommendation Letters (Jinsook Lee et al., 2023)

{{<citation>}}

Jinsook Lee, Bradon Thymes, Joyce Zhou, Thorsten Joachims, Rene F. Kizilcec. (2023)  
**Augmenting Holistic Review in University Admission using Natural Language Processing for Essays and Recommendation Letters**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2306.17575v1)  

---


**ABSTRACT**  
University admission at many highly selective institutions uses a holistic review process, where all aspects of the application, including protected attributes (e.g., race, gender), grades, essays, and recommendation letters are considered, to compose an excellent and diverse class. In this study, we empirically evaluate how influential protected attributes are for predicting admission decisions using a machine learning (ML) model, and in how far textual information (e.g., personal essay, teacher recommendation) may substitute for the loss of protected attributes in the model. Using data from 14,915 applicants to an undergraduate admission office at a selective U.S. institution in the 2022-2023 cycle, we find that the exclusion of protected attributes from the ML model leads to substantially reduced admission-prediction performance. The inclusion of textual information via both a TF-IDF representation and a Latent Dirichlet allocation (LDA) model partially restores model performance, but does not appear to provide a full substitute for admitting a similarly diverse class. In particular, while the text helps with gender diversity, the proportion of URM applicants is severely impacted by the exclusion of protected attributes, and the inclusion of new attributes generated from the textual information does not recover this performance loss.

{{</citation>}}


### (19/91) A Cost-aware Study of Depression Language on Social Media using Topic and Affect Contextualization (Andrea Laguna et al., 2023)

{{<citation>}}

Andrea Laguna, Oscar Araque. (2023)  
**A Cost-aware Study of Depression Language on Social Media using Topic and Affect Contextualization**  

---
Primary Category: cs.CL  
Categories: 68T50, cs-CL, cs.CL  
Keywords: Social Media, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2306.17564v1)  

---


**ABSTRACT**  
Depression is a growing issue in society's mental health that affects all areas of life and can even lead to suicide. Fortunately, prevention programs can be effective in its treatment. In this context, this work proposes an automatic system for detecting depression on social media based on machine learning and natural language processing methods. This paper presents the following contributions: (i) an ensemble learning system that combines several types of text representations for depression detection, including recent advances in the field; (ii) a contextualization schema through topic and affective information; (iii) an analysis of models' energy consumption, establishing a trade-off between classification performance and overall computational costs. To assess the proposed models' effectiveness, a thorough evaluation is performed in two datasets that model depressive text. Experiments indicate that the proposed contextualization strategies can improve the classification and that approaches that use Transformers can improve the overall F-score by 2% while augmenting the energy cost a hundred times. Finally, this work paves the way for future energy-wise systems by considering both the performance classification and the energy consumption.

{{</citation>}}


### (20/91) GPT-FinRE: In-context Learning for Financial Relation Extraction using Large Language Models (Pawan Kumar Rajpoot et al., 2023)

{{<citation>}}

Pawan Kumar Rajpoot, Ankur Parikh. (2023)  
**GPT-FinRE: In-context Learning for Financial Relation Extraction using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Financial, GPT, Language Model, NLP, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2306.17519v1)  

---


**ABSTRACT**  
Relation extraction (RE) is a crucial task in natural language processing (NLP) that aims to identify and classify relationships between entities mentioned in text. In the financial domain, relation extraction plays a vital role in extracting valuable information from financial documents, such as news articles, earnings reports, and company filings. This paper describes our solution to relation extraction on one such dataset REFinD. The dataset was released along with shared task as a part of the Fourth Workshop on Knowledge Discovery from Unstructured Data in Financial Services, co-located with SIGIR 2023. In this paper, we employed OpenAI models under the framework of in-context learning (ICL). We utilized two retrieval strategies to find top K relevant in-context learning demonstrations / examples from training data for a given test example. The first retrieval mechanism, we employed, is a learning-free dense retriever and the other system is a learning-based retriever. We were able to achieve 4th rank on the leaderboard. Our best F1-score is 0.718.

{{</citation>}}


### (21/91) Preference Ranking Optimization for Human Alignment (Feifan Song et al., 2023)

{{<citation>}}

Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, Houfeng Wang. (2023)  
**Preference Ranking Optimization for Human Alignment**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2306.17492v1)  

---


**ABSTRACT**  
Large language models (LLMs) often contain misleading content, emphasizing the need to align them with human values to ensure secur AI systems. Reinforcement learning from human feedback (RLHF) has been employed to achieve this alignment by combining a reward model, typically based on Bradley-Terry paired comparison, with an RL algorithm such as Proximal Policy Optimization (PPO) to optimize LLM responses. However, RLHF exhibits complexity, instability, and sensitivity to hyperparameters. In this paper, we propose Preference Ranking Optimization (PRO) as an alternative to PPO for directly aligning LLMs with the Bradley-Terry comparison. PRO extends the pairwise Bradley-Terry comparison to accommodate preference rankings of any length. By iteratively contrasting the likelihood of generating responses, PRO instructs the LLM to prioritize the best response while progressively ranking the remaining responses. In this manner, PRO effectively transforms human alignment into aligning the probability ranking of $n$ responses generated by LLM with the preference ranking of humans towards these responses. Experiments have shown that PRO outperforms existing alignment algorithms, achieving comparable results to ChatGPT and human responses through automatic-based, reward-based, GPT-4, and human evaluations. Furthermore, we demonstrate that longer, more diverse, and higher-quality preference ranking sequences can consistently enhance the performance of human alignment.

{{</citation>}}


### (22/91) Progressive Multi-task Learning Framework for Chinese Text Error Correction (Shirong Ma et al., 2023)

{{<citation>}}

Shirong Ma, Yinghui Li, Haojing Huang, Shulin Huang, Yangning Li, Hai-Tao Zheng, Ying Shen. (2023)  
**Progressive Multi-task Learning Framework for Chinese Text Error Correction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2306.17447v2)  

---


**ABSTRACT**  
Chinese Text Error Correction (CTEC) aims to detect and correct errors in the input text, which benefits human's daily life and various downstream tasks. Recent approaches mainly employ Pre-trained Language Models (PLMs) to resolve CTEC task and achieve tremendous success. However, previous approaches suffer from issues of over-correction and under-correction, and the former is especially conspicuous in the precision-critical CTEC task. To mitigate the issue of overcorrection, we propose a novel model-agnostic progressive multitask learning framework for CTEC, named ProTEC, which guides a CTEC model to learn the task from easy to difficult. We divide CTEC task into three sub-tasks from easy to difficult: Error Detection, Error Type Identification, and Correction Result Generation. During the training process, ProTEC guides the model to learn text error correction progressively by incorporating these sub-tasks into a multi-task training objective. During the inference process, the model completes these sub-tasks in turn to generate the correction results. Extensive experiments and detailed analyses fully demonstrate the effectiveness and efficiency of our proposed framework.

{{</citation>}}


### (23/91) Provable Robust Watermarking for AI-Generated Text (Xuandong Zhao et al., 2023)

{{<citation>}}

Xuandong Zhao, Prabhanjan Ananth, Lei Li, Yu-Xiang Wang. (2023)  
**Provable Robust Watermarking for AI-Generated Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2306.17439v1)  

---


**ABSTRACT**  
As AI-generated text increasingly resembles human-written content, the ability to detect machine-generated text becomes crucial. To address this challenge, we present GPTWatermark, a robust and high-quality solution designed to ascertain whether a piece of text originates from a specific model. Our approach extends existing watermarking strategies and employs a fixed group design to enhance robustness against editing and paraphrasing attacks. We show that our watermarked language model enjoys strong provable guarantees on generation quality, correctness in detection, and security against evasion attacks. Experimental results on various large language models (LLMs) and diverse datasets demonstrate that our method achieves superior detection accuracy and comparable generation quality in perplexity, thus promoting the responsible use of LLMs.

{{</citation>}}


### (24/91) Japanese Lexical Complexity for Non-Native Readers: A New Dataset (Yusuke Ide et al., 2023)

{{<citation>}}

Yusuke Ide, Masato Mita, Adam Nohejl, Hiroki Ouchi, Taro Watanabe. (2023)  
**Japanese Lexical Complexity for Non-Native Readers: A New Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2306.17399v1)  

---


**ABSTRACT**  
Lexical complexity prediction (LCP) is the task of predicting the complexity of words in a text on a continuous scale. It plays a vital role in simplifying or annotating complex words to assist readers. To study lexical complexity in Japanese, we construct the first Japanese LCP dataset. Our dataset provides separate complexity scores for Chinese/Korean annotators and others to address the readers' L1-specific needs. In the baseline experiment, we demonstrate the effectiveness of a BERT-based system for Japanese LCP.

{{</citation>}}


### (25/91) SummQA at MEDIQA-Chat 2023:In-Context Learning with GPT-4 for Medical Summarization (Yash Mathur et al., 2023)

{{<citation>}}

Yash Mathur, Sanketh Rangreji, Raghav Kapoor, Medha Palavalli, Amanda Bertsch, Matthew R. Gormley. (2023)  
**SummQA at MEDIQA-Chat 2023:In-Context Learning with GPT-4 for Medical Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT, GPT-4, QA, Summarization  
[Paper Link](http://arxiv.org/abs/2306.17384v1)  

---


**ABSTRACT**  
Medical dialogue summarization is challenging due to the unstructured nature of medical conversations, the use of medical terminology in gold summaries, and the need to identify key information across multiple symptom sets. We present a novel system for the Dialogue2Note Medical Summarization tasks in the MEDIQA 2023 Shared Task. Our approach for section-wise summarization (Task A) is a two-stage process of selecting semantically similar dialogues and using the top-k similar dialogues as in-context examples for GPT-4. For full-note summarization (Task B), we use a similar solution with k=1. We achieved 3rd place in Task A (2nd among all teams), 4th place in Task B Division Wise Summarization (2nd among all teams), 15th place in Task A Section Header Classification (9th among all teams), and 8th place among all teams in Task B. Our results highlight the effectiveness of few-shot prompting for this task, though we also identify several weaknesses of prompting-based approaches. We compare GPT-4 performance with several finetuned baselines. We find that GPT-4 summaries are more abstractive and shorter. We make our code publicly available.

{{</citation>}}


### (26/91) Multi-Dialectal Representation Learning of Sinitic Phonology (Zhibai Jia, 2023)

{{<citation>}}

Zhibai Jia. (2023)  
**Multi-Dialectal Representation Learning of Sinitic Phonology**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SI, cs.CL  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.01209v1)  

---


**ABSTRACT**  
Machine learning techniques have shown their competence for representing and reasoning in symbolic systems such as language and phonology. In Sinitic Historical Phonology, notable tasks that could benefit from machine learning include the comparison of dialects and reconstruction of proto-languages systems. Motivated by this, this paper provides an approach for obtaining multi-dialectal representations of Sinitic syllables, by constructing a knowledge graph from structured phonological data, then applying the BoxE technique from knowledge base learning. We applied unsupervised clustering techniques to the obtained representations to observe that the representations capture phonemic contrast from the input dialects. Furthermore, we trained classifiers to perform inference of unobserved Middle Chinese labels, showing the representations' potential for indicating archaic, proto-language features. The representations can be used for performing completion of fragmented Sinitic phonological knowledge bases, estimating divergences between different characters, or aiding the exploration and reconstruction of archaic features.

{{</citation>}}


## eess.IV (2)



### (27/91) Multiscale Progressive Text Prompt Network for Medical Image Segmentation (Xianjun Han et al., 2023)

{{<citation>}}

Xianjun Han, Qianqian Chen, Zhaoyang Xie, Xuejun Li, Hongyu Yang. (2023)  
**Multiscale Progressive Text Prompt Network for Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.00174v1)  

---


**ABSTRACT**  
The accurate segmentation of medical images is a crucial step in obtaining reliable morphological statistics. However, training a deep neural network for this task requires a large amount of labeled data to ensure high-accuracy results. To address this issue, we propose using progressive text prompts as prior knowledge to guide the segmentation process. Our model consists of two stages. In the first stage, we perform contrastive learning on natural images to pretrain a powerful prior prompt encoder (PPE). This PPE leverages text prior prompts to generate multimodality features. In the second stage, medical image and text prior prompts are sent into the PPE inherited from the first stage to achieve the downstream medical image segmentation task. A multiscale feature fusion block (MSFF) combines the features from the PPE to produce multiscale multimodality features. These two progressive features not only bridge the semantic gap but also improve prediction accuracy. Finally, an UpAttention block refines the predicted results by merging the image and text features. This design provides a simple and accurate way to leverage multiscale progressive text prior prompts for medical image segmentation. Compared with using only images, our model achieves high-quality results with low data annotation costs. Moreover, our model not only has excellent reliability and validity on medical images but also performs well on natural images. The experimental results on different image datasets demonstrate that our model is effective and robust for image segmentation.

{{</citation>}}


### (28/91) MedAugment: Universal Automatic Data Augmentation Plug-in for Medical Image Analysis (Zhaoshan Liu et al., 2023)

{{<citation>}}

Zhaoshan Liu, Qiujie Lv, Yifan Li, Ziduo Yang, Lei Shen. (2023)  
**MedAugment: Universal Automatic Data Augmentation Plug-in for Medical Image Analysis**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2306.17466v1)  

---


**ABSTRACT**  
Data Augmentation (DA) technique has been widely implemented in the computer vision field to relieve the data shortage, while the DA in Medical Image Analysis (MIA) is still mostly experience-driven. Here, we develop a plug-and-use DA method, named MedAugment, to introduce the automatic DA argumentation to the MIA field. To settle the difference between natural images and medical images, we divide the augmentation space into pixel augmentation space and spatial augmentation space. A novel operation sampling strategy is also proposed when sampling DA operations from the spaces. To demonstrate the performance and universality of MedAugment, we implement extensive experiments on four classification datasets and three segmentation datasets. The results show that our MedAugment outperforms most state-of-the-art DA methods. This work shows that the plug-and-use MedAugment may benefit the MIA community. Code is available at https://github.com/NUS-Tim/MedAugment_Pytorch.

{{</citation>}}


## cs.IR (3)



### (29/91) Counterfactual Collaborative Reasoning (Jianchao Ji et al., 2023)

{{<citation>}}

Jianchao Ji, Zelong Li, Shuyuan Xu, Max Xiong, Juntao Tan, Yingqiang Ge, Hao Wang, Yongfeng Zhang. (2023)  
**Counterfactual Collaborative Reasoning**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2307.00165v1)  

---


**ABSTRACT**  
Causal reasoning and logical reasoning are two important types of reasoning abilities for human intelligence. However, their relationship has not been extensively explored under machine intelligence context. In this paper, we explore how the two reasoning abilities can be jointly modeled to enhance both accuracy and explainability of machine learning models. More specifically, by integrating two important types of reasoning ability -- counterfactual reasoning and (neural) logical reasoning -- we propose Counterfactual Collaborative Reasoning (CCR), which conducts counterfactual logic reasoning to improve the performance. In particular, we use recommender system as an example to show how CCR alleviate data scarcity, improve accuracy and enhance transparency. Technically, we leverage counterfactual reasoning to generate "difficult" counterfactual training examples for data augmentation, which -- together with the original training examples -- can enhance the model performance. Since the augmented data is model irrelevant, they can be used to enhance any model, enabling the wide applicability of the technique. Besides, most of the existing data augmentation methods focus on "implicit data augmentation" over users' implicit feedback, while our framework conducts "explicit data augmentation" over users explicit feedback based on counterfactual logic reasoning. Experiments on three real-world datasets show that CCR achieves better performance than non-augmented models and implicitly augmented models, and also improves model transparency by generating counterfactual explanations.

{{</citation>}}


### (30/91) Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting (Zhen Qin et al., 2023)

{{<citation>}}

Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang, Junru Wu, Jiaming Shen, Tianqi Liu, Jialu Liu, Donald Metzler, Xuanhui Wang, Michael Bendersky. (2023)  
**Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2306.17563v1)  

---


**ABSTRACT**  
Ranking documents using Large Language Models (LLMs) by directly feeding the query and candidate documents into the prompt is an interesting and practical problem. However, there has been limited success so far, as researchers have found it difficult to outperform fine-tuned baseline rankers on benchmark datasets. We analyze pointwise and listwise ranking prompts used by existing methods and argue that off-the-shelf LLMs do not fully understand these ranking formulations, possibly due to the nature of how LLMs are trained. In this paper, we propose to significantly reduce the burden on LLMs by using a new technique called Pairwise Ranking Prompting (PRP). Our results are the first in the literature to achieve state-of-the-art ranking performance on standard benchmarks using moderate-sized open-sourced LLMs. On TREC-DL2020, PRP based on the Flan-UL2 model with 20B parameters outperforms the previous best approach in the literature, which is based on the blackbox commercial GPT-4 that has 50x (estimated) model size, by over 5% at NDCG@1. On TREC-DL2019, PRP is only inferior to the GPT-4 solution on the NDCG@5 and NDCG@10 metrics, while outperforming other existing solutions, such as InstructGPT which has 175B parameters, by over 10% for nearly all ranking metrics. Furthermore, we propose several variants of PRP to improve efficiency and show that it is possible to achieve competitive results even with linear complexity. We also discuss other benefits of PRP, such as supporting both generation and scoring LLM APIs, as well as being insensitive to input ordering.

{{</citation>}}


### (31/91) DeepTagger: Knowledge Enhanced Named Entity Recognition for Web-Based Ads Queries (Simiao Zuo et al., 2023)

{{<citation>}}

Simiao Zuo, Pengfei Tang, Xinyu Hu, Qiang Lou, Jian Jiao, Denis Charles. (2023)  
**DeepTagger: Knowledge Enhanced Named Entity Recognition for Web-Based Ads Queries**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: ChatGPT, GPT, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2306.17413v1)  

---


**ABSTRACT**  
Named entity recognition (NER) is a crucial task for online advertisement. State-of-the-art solutions leverage pre-trained language models for this task. However, three major challenges remain unresolved: web queries differ from natural language, on which pre-trained models are trained; web queries are short and lack contextual information; and labeled data for NER is scarce. We propose DeepTagger, a knowledge-enhanced NER model for web-based ads queries. The proposed knowledge enhancement framework leverages both model-free and model-based approaches. For model-free enhancement, we collect unlabeled web queries to augment domain knowledge; and we collect web search results to enrich the information of ads queries. We further leverage effective prompting methods to automatically generate labels using large language models such as ChatGPT. Additionally, we adopt a model-based knowledge enhancement method based on adversarial data augmentation. We employ a three-stage training framework to train DeepTagger models. Empirical results in various NER tasks demonstrate the effectiveness of the proposed framework.

{{</citation>}}


## cs.CV (22)



### (32/91) Stitched ViTs are Flexible Vision Backbones (Zizheng Pan et al., 2023)

{{<citation>}}

Zizheng Pan, Jing Liu, Haoyu He, Jianfei Cai, Bohan Zhuang. (2023)  
**Stitched ViTs are Flexible Vision Backbones**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.00154v1)  

---


**ABSTRACT**  
Large pretrained plain vision Transformers (ViTs) have been the workhorse for many downstream tasks. However, existing works utilizing off-the-shelf ViTs are inefficient in terms of training and deployment, because adopting ViTs with individual sizes requires separate training and is restricted by fixed performance-efficiency trade-offs. In this paper, we are inspired by stitchable neural networks, which is a new framework that cheaply produces a single model that covers rich subnetworks by stitching pretrained model families, supporting diverse performance-efficiency trade-offs at runtime. Building upon this foundation, we introduce SN-Netv2, a systematically improved model stitching framework to facilitate downstream task adaptation. Specifically, we first propose a Two-way stitching scheme to enlarge the stitching space. We then design a resource-constrained sampling strategy that takes into account the underlying FLOPs distributions in the space for improved sampling. Finally, we observe that learning stitching layers is a low-rank update, which plays an essential role on downstream tasks to stabilize training and ensure a good Pareto frontier. With extensive experiments on ImageNet-1K, ADE20K, COCO-Stuff-10K, NYUv2 and COCO-2017, SN-Netv2 demonstrates strong ability to serve as a flexible vision backbone, achieving great advantages in both training efficiency and adaptation. Code will be released at https://github.com/ziplab/SN-Netv2.

{{</citation>}}


### (33/91) Prompting classes: Exploring the Power of Prompt Class Learning in Weakly Supervised Semantic Segmentation (Balamurali Murugesan et al., 2023)

{{<citation>}}

Balamurali Murugesan, Rukhshanda Hussain, Rajarshi Bhattacharya, Ismail Ben Ayed, Jose Dolz. (2023)  
**Prompting classes: Exploring the Power of Prompt Class Learning in Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.00097v2)  

---


**ABSTRACT**  
Recently, CLIP-based approaches have exhibited remarkable performance on generalization and few-shot learning tasks, fueled by the power of contrastive language-vision pre-training. In particular, prompt tuning has emerged as an effective strategy to adapt the pre-trained language-vision models to downstream tasks by employing task-related textual tokens. Motivated by this progress, in this work we question whether other fundamental problems, such as weakly supervised semantic segmentation (WSSS), can benefit from prompt tuning. Our findings reveal two interesting observations that shed light on the impact of prompt tuning on WSSS. First, modifying only the class token of the text prompt results in a greater impact on the Class Activation Map (CAM), compared to arguably more complex strategies that optimize the context. And second, the class token associated with the image ground truth does not necessarily correspond to the category that yields the best CAM. Motivated by these observations, we introduce a novel approach based on a PrOmpt cLass lEarning (POLE) strategy. Through extensive experiments we demonstrate that our simple, yet efficient approach achieves SOTA performance in a well-known WSSS benchmark. These results highlight not only the benefits of language-vision models in WSSS but also the potential of prompt learning for this problem. The code is available at https://github.com/rB080/WSS_POLE.

{{</citation>}}


### (34/91) Situated Cameras, Situated Knowledges: Towards an Egocentric Epistemology for Computer Vision (Samuel Goree et al., 2023)

{{<citation>}}

Samuel Goree, David Crandall. (2023)  
**Situated Cameras, Situated Knowledges: Towards an Egocentric Epistemology for Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2307.00064v1)  

---


**ABSTRACT**  
In her influential 1988 paper, Situated Knowledges, Donna Haraway uses vision and perspective as a metaphor to discuss scientific knowledge. Today, egocentric computer vision discusses many of the same issues, except in a literal vision context. In this short position paper, we collapse that metaphor, and explore the interactions between feminist epistemology and egocentric CV as "Egocentric Epistemology." Using this framework, we argue for the use of qualitative, human-centric methods as a complement to performance benchmarks, to center both the literal and metaphorical perspective of human crowd workers in CV.

{{</citation>}}


### (35/91) SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs (Lijun Yu et al., 2023)

{{<citation>}}

Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David A. Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, Kevin Murphy, Alexander G. Hauptmann, Lu Jiang. (2023)  
**SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: GPT, PaLM  
[Paper Link](http://arxiv.org/abs/2306.17842v2)  

---


**ABSTRACT**  
In this work, we introduce Semantic Pyramid AutoEncoder (SPAE) for enabling frozen LLMs to perform both understanding and generation tasks involving non-linguistic modalities such as images or videos. SPAE converts between raw pixels and interpretable lexical tokens (or words) extracted from the LLM's vocabulary. The resulting tokens capture both the semantic meaning and the fine-grained details needed for visual reconstruction, effectively translating the visual content into a language comprehensible to the LLM, and empowering it to perform a wide array of multimodal tasks. Our approach is validated through in-context learning experiments with frozen PaLM 2 and GPT 3.5 on a diverse set of image understanding and generation tasks. Our method marks the first successful attempt to enable a frozen LLM to generate image content while surpassing state-of-the-art performance in image understanding tasks, under the same setting, by over 25%.

{{</citation>}}


### (36/91) Federated Ensemble YOLOv5 - A Better Generalized Object Detection Algorithm (Vinit Hegiste et al., 2023)

{{<citation>}}

Vinit Hegiste, Tatjana Legler, Martin Ruskowski. (2023)  
**Federated Ensemble YOLOv5 - A Better Generalized Object Detection Algorithm**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2306.17829v1)  

---


**ABSTRACT**  
Federated learning (FL) has gained significant traction as a privacy-preserving algorithm, but the underlying resembles of federated learning algorithm like Federated averaging (FED Avg) or Federated SGD (FED SGD) to ensemble learning algorithms has not been fully explored. The purpose of this paper is to examine the application of FL to object detection as a method to enhance generalizability, and to compare its performance against a centralized training approach for an object detection algorithm. Specifically, we investigate the performance of a YOLOv5 model trained using FL across multiple clients and employ a random sampling strategy without replacement, so each client holds a portion of the same dataset used for centralized training. Our experimental results showcase the superior efficiency of the FL object detector's global model in generating accurate bounding boxes for unseen objects, with the test set being a mixture of objects from two distinct clients not represented in the training dataset. These findings suggest that FL can be viewed from an ensemble algorithm perspective, akin to a synergistic blend of Bagging and Boosting techniques. As a result, FL can be seen not only as a method to enhance privacy, but also as a method to enhance the performance of a machine learning model.

{{</citation>}}


### (37/91) DisCo: Disentangled Control for Referring Human Dance Generation in Real World (Tan Wang et al., 2023)

{{<citation>}}

Tan Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, Lijuan Wang. (2023)  
**DisCo: Disentangled Control for Referring Human Dance Generation in Real World**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2307.00040v1)  

---


**ABSTRACT**  
Generative AI has made significant strides in computer vision, particularly in image/video synthesis conditioned on text descriptions. Despite the advancements, it remains challenging especially in the generation of human-centric content such as dance synthesis. Existing dance synthesis methods struggle with the gap between synthesized content and real-world dance scenarios. In this paper, we define a new problem setting: Referring Human Dance Generation, which focuses on real-world dance scenarios with three important properties: (i) Faithfulness: the synthesis should retain the appearance of both human subject foreground and background from the reference image, and precisely follow the target pose; (ii) Generalizability: the model should generalize to unseen human subjects, backgrounds, and poses; (iii) Compositionality: it should allow for composition of seen/unseen subjects, backgrounds, and poses from different sources. To address these challenges, we introduce a novel approach, DISCO, which includes a novel model architecture with disentangled control to improve the faithfulness and compositionality of dance synthesis, and an effective human attribute pre-training for better generalizability to unseen humans. Extensive qualitative and quantitative results demonstrate that DISCO can generate high-quality human dance images and videos with diverse appearances and flexible motions. Code, demo, video and visualization are available at: https://disco-dance.github.io/.

{{</citation>}}


### (38/91) Look, Remember and Reason: Visual Reasoning with Grounded Rationales (Apratim Bhattacharyya et al., 2023)

{{<citation>}}

Apratim Bhattacharyya, Sunny Panchal, Mingu Lee, Reza Pourreza, Pulkit Madan, Roland Memisevic. (2023)  
**Look, Remember and Reason: Visual Reasoning with Grounded Rationales**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2306.17778v1)  

---


**ABSTRACT**  
Large language models have recently shown human level performance on a variety of reasoning tasks. However, the ability of these models to perform complex visual reasoning has not been studied in detail yet. A key challenge in many visual reasoning tasks is that the visual information needs to be tightly integrated in the reasoning process. We propose to address this challenge by drawing inspiration from human visual problem solving which depends on a variety of low-level visual capabilities. It can often be cast as the three step-process of ``Look, Remember, Reason'': visual information is incrementally extracted using low-level visual routines in a step-by-step fashion until a final answer is reached. We follow the same paradigm to enable existing large language models, with minimal changes to the architecture, to solve visual reasoning problems. To this end, we introduce rationales over the visual input that allow us to integrate low-level visual capabilities, such as object recognition and tracking, as surrogate tasks. We show competitive performance on diverse visual reasoning tasks from the CLEVR, CATER, and ACRE datasets over state-of-the-art models designed specifically for these tasks.

{{</citation>}}


### (39/91) Exploration and Exploitation of Unlabeled Data for Open-Set Semi-Supervised Learning (Ganlong Zhao et al., 2023)

{{<citation>}}

Ganlong Zhao, Guanbin Li, Yipeng Qin, Jinjin Zhang, Zhenhua Chai, Xiaolin Wei, Liang Lin, Yizhou Yu. (2023)  
**Exploration and Exploitation of Unlabeled Data for Open-Set Semi-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2306.17699v1)  

---


**ABSTRACT**  
In this paper, we address a complex but practical scenario in semi-supervised learning (SSL) named open-set SSL, where unlabeled data contain both in-distribution (ID) and out-of-distribution (OOD) samples. Unlike previous methods that only consider ID samples to be useful and aim to filter out OOD ones completely during training, we argue that the exploration and exploitation of both ID and OOD samples can benefit SSL. To support our claim, i) we propose a prototype-based clustering and identification algorithm that explores the inherent similarity and difference among samples at feature level and effectively cluster them around several predefined ID and OOD prototypes, thereby enhancing feature learning and facilitating ID/OOD identification; ii) we propose an importance-based sampling method that exploits the difference in importance of each ID and OOD sample to SSL, thereby reducing the sampling bias and improving the training. Our proposed method achieves state-of-the-art in several challenging benchmarks, and improves upon existing SSL methods even when ID samples are totally absent in unlabeled data.

{{</citation>}}


### (40/91) Multimodal Prompt Retrieval for Generative Visual Question Answering (Timothy Ossowski et al., 2023)

{{<citation>}}

Timothy Ossowski, Junjie Hu. (2023)  
**Multimodal Prompt Retrieval for Generative Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2306.17675v1)  

---


**ABSTRACT**  
Recent years have witnessed impressive results of pre-trained vision-language models on knowledge-intensive tasks such as visual question answering (VQA). Despite the recent advances in VQA, existing methods mainly adopt a discriminative formulation that predicts answers within a pre-defined label set, leading to easy overfitting on low-resource domains with limited labeled data (e.g., medicine) and poor generalization under domain shift to another dataset. To tackle this limitation, we propose a novel generative model enhanced by multimodal prompt retrieval (MPR) that integrates retrieved prompts and multimodal features to generate answers in free text. Our generative model enables rapid zero-shot dataset adaptation to unseen data distributions and open-set answer labels across datasets. Our experiments on medical VQA tasks show that MPR outperforms its non-retrieval counterpart by up to 30% accuracy points in a few-shot domain adaptation setting.

{{</citation>}}


### (41/91) Sphere2Vec: A General-Purpose Location Representation Learning over a Spherical Surface for Large-Scale Geospatial Predictions (Gengchen Mai et al., 2023)

{{<citation>}}

Gengchen Mai, Yao Xuan, Wenyun Zuo, Yutong He, Jiaming Song, Stefano Ermon, Krzysztof Janowicz, Ni Lao. (2023)  
**Sphere2Vec: A General-Purpose Location Representation Learning over a Spherical Surface for Large-Scale Geospatial Predictions**  

---
Primary Category: cs.CV  
Categories: 68T07, 68T45, I-2-0; I-2-6; I-2-10; I-5-1; J-2, cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2306.17624v2)  

---


**ABSTRACT**  
Generating learning-friendly representations for points in space is a fundamental and long-standing problem in ML. Recently, multi-scale encoding schemes (such as Space2Vec and NeRF) were proposed to directly encode any point in 2D/3D Euclidean space as a high-dimensional vector, and has been successfully applied to various geospatial prediction and generative tasks. However, all current 2D and 3D location encoders are designed to model point distances in Euclidean space. So when applied to large-scale real-world GPS coordinate datasets, which require distance metric learning on the spherical surface, both types of models can fail due to the map projection distortion problem (2D) and the spherical-to-Euclidean distance approximation error (3D). To solve these problems, we propose a multi-scale location encoder called Sphere2Vec which can preserve spherical distances when encoding point coordinates on a spherical surface. We developed a unified view of distance-reserving encoding on spheres based on the DFS. We also provide theoretical proof that the Sphere2Vec preserves the spherical surface distance between any two points, while existing encoding schemes do not. Experiments on 20 synthetic datasets show that Sphere2Vec can outperform all baseline models on all these datasets with up to 30.8% error rate reduction. We then apply Sphere2Vec to three geo-aware image classification tasks - fine-grained species recognition, Flickr image recognition, and remote sensing image classification. Results on 7 real-world datasets show the superiority of Sphere2Vec over multiple location encoders on all three tasks. Further analysis shows that Sphere2Vec outperforms other location encoder models, especially in the polar regions and data-sparse areas because of its nature for spherical surface distance preservation. Code and data are available at https://gengchenmai.github.io/sphere2vec-website/.

{{</citation>}}


### (42/91) Razor SNN: Efficient Spiking Neural Network with Temporal Embeddings (Yuan Zhang et al., 2023)

{{<citation>}}

Yuan Zhang, Jian Cao, Ling Zhang, Jue Chen, Wenyu Sun, Yuan Wang. (2023)  
**Razor SNN: Efficient Spiking Neural Network with Temporal Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2306.17597v1)  

---


**ABSTRACT**  
The event streams generated by dynamic vision sensors (DVS) are sparse and non-uniform in the spatial domain, while still dense and redundant in the temporal domain. Although spiking neural network (SNN), the event-driven neuromorphic model, has the potential to extract spatio-temporal features from the event streams, it is not effective and efficient. Based on the above, we propose an events sparsification spiking framework dubbed as Razor SNN, pruning pointless event frames progressively. Concretely, we extend the dynamic mechanism based on the global temporal embeddings, reconstruct the features, and emphasize the events effect adaptively at the training stage. During the inference stage, eliminate fruitless frames hierarchically according to a binary mask generated by the trained temporal embeddings. Comprehensive experiments demonstrate that our Razor SNN achieves competitive performance consistently on four events-based benchmarks: DVS 128 Gesture, N-Caltech 101, CIFAR10-DVS and SHD.

{{</citation>}}


### (43/91) Miniaturized Graph Convolutional Networks with Topologically Consistent Pruning (Hichem Sahbi, 2023)

{{<citation>}}

Hichem Sahbi. (2023)  
**Miniaturized Graph Convolutional Networks with Topologically Consistent Pruning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network, Pruning  
[Paper Link](http://arxiv.org/abs/2306.17590v1)  

---


**ABSTRACT**  
Magnitude pruning is one of the mainstream methods in lightweight architecture design whose goal is to extract subnetworks with the largest weight connections. This method is known to be successful, but under very high pruning regimes, it suffers from topological inconsistency which renders the extracted subnetworks disconnected, and this hinders their generalization ability. In this paper, we devise a novel magnitude pruning method that allows extracting subnetworks while guarantying their topological consistency. The latter ensures that only accessible and co-accessible -- impactful -- connections are kept in the resulting lightweight networks. Our solution is based on a novel reparametrization and two supervisory bi-directional networks which implement accessibility/co-accessibility and guarantee that only connected subnetworks will be selected during training. This solution allows enhancing generalization significantly, under very high pruning regimes, as corroborated through extensive experiments, involving graph convolutional networks, on the challenging task of skeleton-based action recognition.

{{</citation>}}


### (44/91) SpATr: MoCap 3D Human Action Recognition based on Spiral Auto-encoder and Transformer Network (Hamza Bouzid et al., 2023)

{{<citation>}}

Hamza Bouzid, Lahoucine Ballihi. (2023)  
**SpATr: MoCap 3D Human Action Recognition based on Spiral Auto-encoder and Transformer Network**  

---
Primary Category: cs.CV  
Categories: I-5-0; I-5-1; I-5-2; I-5-4, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2306.17574v1)  

---


**ABSTRACT**  
Recent advancements in technology have expanded the possibilities of human action recognition by leveraging 3D data, which offers a richer representation of actions through the inclusion of depth information, enabling more accurate analysis of spatial and temporal characteristics. However, 3D human action recognition is a challenging task due to the irregularity and Disarrangement of the data points in action sequences. In this context, we present our novel model for human action recognition from fixed topology mesh sequences based on Spiral Auto-encoder and Transformer Network, namely SpATr. The proposed method first disentangles space and time in the mesh sequences. Then, an auto-encoder is utilized to extract spatial geometrical features, and tiny transformer is used to capture the temporal evolution of the sequence. Previous methods either use 2D depth images, sample skeletons points or they require a huge amount of memory leading to the ability to process short sequences only. In this work, we show competitive recognition rate and high memory efficiency by building our auto-encoder based on spiral convolutions, which are light weight convolution directly applied to mesh data with fixed topologies, and by modeling temporal evolution using a attention, that can handle large sequences. The proposed method is evaluated on on two 3D human action datasets: MoVi and BMLrub from the Archive of Motion Capture As Surface Shapes (AMASS). The results analysis shows the effectiveness of our method in 3D human action recognition while maintaining high memory efficiency. The code will soon be made publicly available.

{{</citation>}}


### (45/91) Why does my medical AI look at pictures of birds? Exploring the efficacy of transfer learning across domain boundaries (Frederic Jonske et al., 2023)

{{<citation>}}

Frederic Jonske, Moon Kim, Enrico Nasca, Janis Evers, Johannes Haubold, René Hosch, Felix Nensa, Michael Kamp, Constantin Seibold, Jan Egger, Jens Kleesiek. (2023)  
**Why does my medical AI look at pictures of birds? Exploring the efficacy of transfer learning across domain boundaries**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2306.17555v1)  

---


**ABSTRACT**  
It is an open secret that ImageNet is treated as the panacea of pretraining. Particularly in medical machine learning, models not trained from scratch are often finetuned based on ImageNet-pretrained models. We posit that pretraining on data from the domain of the downstream task should almost always be preferred instead. We leverage RadNet-12M, a dataset containing more than 12 million computed tomography (CT) image slices, to explore the efficacy of self-supervised pretraining on medical and natural images. Our experiments cover intra- and cross-domain transfer scenarios, varying data scales, finetuning vs. linear evaluation, and feature space analysis. We observe that intra-domain transfer compares favorably to cross-domain transfer, achieving comparable or improved performance (0.44% - 2.07% performance increase using RadNet pretraining, depending on the experiment) and demonstrate the existence of a domain boundary-related generalization gap and domain-specific learned features.

{{</citation>}}


### (46/91) Manga109Dialog A Large-scale Dialogue Dataset for Comics Speaker Detection (Yingxuan Li et al., 2023)

{{<citation>}}

Yingxuan Li, Kiyoharu Aizawa, Yusuke Matsui. (2023)  
**Manga109Dialog A Large-scale Dialogue Dataset for Comics Speaker Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2306.17469v1)  

---


**ABSTRACT**  
The expanding market for e-comics has spurred interest in the development of automated methods to analyze comics. For further understanding of comics, an automated approach is needed to link text in comics to characters speaking the words. Comics speaker detection research has practical applications, such as automatic character assignment for audiobooks, automatic translation according to characters' personalities, and inference of character relationships and stories.   To deal with the problem of insufficient speaker-to-text annotations, we created a new annotation dataset Manga109Dialog based on Manga109. Manga109Dialog is the world's largest comics speaker annotation dataset, containing 132,692 speaker-to-text pairs. We further divided our dataset into different levels by prediction difficulties to evaluate speaker detection methods more appropriately. Unlike existing methods mainly based on distances, we propose a deep learning-based method using scene graph generation models. Due to the unique features of comics, we enhance the performance of our proposed model by considering the frame reading order. We conducted experiments using Manga109Dialog and other datasets. Experimental results demonstrate that our scene-graph-based approach outperforms existing methods, achieving a prediction accuracy of over 75%.

{{</citation>}}


### (47/91) CausalVLR: A Toolbox and Benchmark for Visual-Linguistic Causal Reasoning (Yang Liu et al., 2023)

{{<citation>}}

Yang Liu, Weixing Chen, Guanbin Li, Liang Lin. (2023)  
**CausalVLR: A Toolbox and Benchmark for Visual-Linguistic Causal Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2306.17462v1)  

---


**ABSTRACT**  
We present CausalVLR (Causal Visual-Linguistic Reasoning), an open-source toolbox containing a rich set of state-of-the-art causal relation discovery and causal inference methods for various visual-linguistic reasoning tasks, such as VQA, image/video captioning, medical report generation, model generalization and robustness, etc. These methods have been included in the toolbox with PyTorch implementations under NVIDIA computing system. It not only includes training and inference codes, but also provides model weights. We believe this toolbox is by far the most complete visual-linguitic causal reasoning toolbox. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to re-implement existing methods and develop their own new causal reasoning methods. Code and models are available at https://github.com/HCPLab-SYSU/Causal-VLReasoning. The project is under active development by HCP-Lab's contributors and we will keep this document updated.

{{</citation>}}


### (48/91) Designing strong baselines for ternary neural network quantization through support and mass equalization (Edouard Yvinec et al., 2023)

{{<citation>}}

Edouard Yvinec, Arnaud Dapogny, Kevin Bailly. (2023)  
**Designing strong baselines for ternary neural network quantization through support and mass equalization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2306.17442v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) offer the highest performance in a wide range of applications in computer vision. These results rely on over-parameterized backbones, which are expensive to run. This computational burden can be dramatically reduced by quantizing (in either data-free (DFQ), post-training (PTQ) or quantization-aware training (QAT) scenarios) floating point values to ternary values (2 bits, with each weight taking value in {-1,0,1}). In this context, we observe that rounding to nearest minimizes the expected error given a uniform distribution and thus does not account for the skewness and kurtosis of the weight distribution, which strongly affects ternary quantization performance. This raises the following question: shall one minimize the highest or average quantization error? To answer this, we design two operators: TQuant and MQuant that correspond to these respective minimization tasks. We show experimentally that our approach allows to significantly improve the performance of ternary quantization through a variety of scenarios in DFQ, PTQ and QAT and give strong insights to pave the way for future research in deep neural network quantization.

{{</citation>}}


### (49/91) Efficient Backdoor Removal Through Natural Gradient Fine-tuning (Nazmul Karim et al., 2023)

{{<citation>}}

Nazmul Karim, Abdullah Al Arafat, Umar Khalid, Zhishan Guo, Naznin Rahnavard. (2023)  
**Efficient Backdoor Removal Through Natural Gradient Fine-tuning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2306.17441v1)  

---


**ABSTRACT**  
The success of a deep neural network (DNN) heavily relies on the details of the training scheme; e.g., training data, architectures, hyper-parameters, etc. Recent backdoor attacks suggest that an adversary can take advantage of such training details and compromise the integrity of a DNN. Our studies show that a backdoor model is usually optimized to a bad local minima, i.e. sharper minima as compared to a benign model. Intuitively, a backdoor model can be purified by reoptimizing the model to a smoother minima through fine-tuning with a few clean validation data. However, fine-tuning all DNN parameters often requires huge computational costs and often results in sub-par clean test performance. To address this concern, we propose a novel backdoor purification technique, Natural Gradient Fine-tuning (NGF), which focuses on removing the backdoor by fine-tuning only one layer. Specifically, NGF utilizes a loss surface geometry-aware optimizer that can successfully overcome the challenge of reaching a smooth minima under a one-layer optimization scenario. To enhance the generalization performance of our proposed method, we introduce a clean data distribution-aware regularizer based on the knowledge of loss surface curvature matrix, i.e., Fisher Information Matrix. Extensive experiments show that the proposed method achieves state-of-the-art performance on a wide range of backdoor defense benchmarks: four different datasets- CIFAR10, GTSRB, Tiny-ImageNet, and ImageNet; 13 recent backdoor attacks, e.g. Blend, Dynamic, WaNet, ISSBA, etc.

{{</citation>}}


### (50/91) Defense against Adversarial Cloud Attack on Remote Sensing Salient Object Detection (Huiming Sun et al., 2023)

{{<citation>}}

Huiming Sun, Lan Fu, Jinlong Li, Qing Guo, Zibo Meng, Tianyun Zhang, Yuewei Lin, Hongkai Yu. (2023)  
**Defense against Adversarial Cloud Attack on Remote Sensing Salient Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2306.17431v2)  

---


**ABSTRACT**  
Detecting the salient objects in a remote sensing image has wide applications for the interdisciplinary research. Many existing deep learning methods have been proposed for Salient Object Detection (SOD) in remote sensing images and get remarkable results. However, the recent adversarial attack examples, generated by changing a few pixel values on the original remote sensing image, could result in a collapse for the well-trained deep learning based SOD model. Different with existing methods adding perturbation to original images, we propose to jointly tune adversarial exposure and additive perturbation for attack and constrain image close to cloudy image as Adversarial Cloud. Cloud is natural and common in remote sensing images, however, camouflaging cloud based adversarial attack and defense for remote sensing images are not well studied before. Furthermore, we design DefenseNet as a learn-able pre-processing to the adversarial cloudy images so as to preserve the performance of the deep learning based remote sensing SOD model, without tuning the already deployed deep SOD model. By considering both regular and generalized adversarial examples, the proposed DefenseNet can defend the proposed Adversarial Cloud in white-box setting and other attack methods in black-box setting. Experimental results on a synthesized benchmark from the public remote sensing SOD dataset (EORSSD) show the promising defense against adversarial cloud attacks.

{{</citation>}}


### (51/91) Topological Data Analysis Guided Segment Anything Model Prompt Optimization for Zero-Shot Segmentation in Biological Imaging (Ruben Glatt et al., 2023)

{{<citation>}}

Ruben Glatt, Shusen Liu. (2023)  
**Topological Data Analysis Guided Segment Anything Model Prompt Optimization for Zero-Shot Segmentation in Biological Imaging**  

---
Primary Category: cs.CV  
Categories: 68T45, I-4-6, cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2306.17400v1)  

---


**ABSTRACT**  
Emerging foundation models in machine learning are models trained on vast amounts of data that have been shown to generalize well to new tasks. Often these models can be prompted with multi-modal inputs that range from natural language descriptions over images to point clouds. In this paper, we propose topological data analysis (TDA) guided prompt optimization for the Segment Anything Model (SAM) and show preliminary results in the biological image segmentation domain. Our approach replaces the standard grid search approach that is used in the original implementation and finds point locations based on their topological significance. Our results show that the TDA optimized point cloud is much better suited for finding small objects and massively reduces computational complexity despite the extra step in scenarios which require many segmentations.

{{</citation>}}


### (52/91) EyeBAG: Accurate Control of Eye Blink and Gaze Based on Data Augmentation Leveraging Style Mixing (Bryan S. Kim et al., 2023)

{{<citation>}}

Bryan S. Kim, Jeong Young Jeong, Wonjong Ryu. (2023)  
**EyeBAG: Accurate Control of Eye Blink and Gaze Based on Data Augmentation Leveraging Style Mixing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2306.17391v1)  

---


**ABSTRACT**  
Recent developments in generative models have enabled the generation of photo-realistic human face images, and downstream tasks utilizing face generation technology have advanced accordingly. However, models for downstream tasks are yet substandard at eye control (e.g. eye blink, gaze redirection). To overcome such eye control problems, we introduce a novel framework consisting of two distinct modules: a blink control module and a gaze redirection module. We also propose a novel data augmentation method to train each module, leveraging style mixing to obtain images with desired features. We show that our framework produces eye-controlled images of high quality, and demonstrate how it can be used to improve the performance of downstream tasks.

{{</citation>}}


### (53/91) HVTSurv: Hierarchical Vision Transformer for Patient-Level Survival Prediction from Whole Slide Image (Zhuchen Shao et al., 2023)

{{<citation>}}

Zhuchen Shao, Yang Chen, Hao Bian, Jian Zhang, Guojun Liu, Yongbing Zhang. (2023)  
**HVTSurv: Hierarchical Vision Transformer for Patient-Level Survival Prediction from Whole Slide Image**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2306.17373v1)  

---


**ABSTRACT**  
Survival prediction based on whole slide images (WSIs) is a challenging task for patient-level multiple instance learning (MIL). Due to the vast amount of data for a patient (one or multiple gigapixels WSIs) and the irregularly shaped property of WSI, it is difficult to fully explore spatial, contextual, and hierarchical interaction in the patient-level bag. Many studies adopt random sampling pre-processing strategy and WSI-level aggregation models, which inevitably lose critical prognostic information in the patient-level bag. In this work, we propose a hierarchical vision Transformer framework named HVTSurv, which can encode the local-level relative spatial information, strengthen WSI-level context-aware communication, and establish patient-level hierarchical interaction. Firstly, we design a feature pre-processing strategy, including feature rearrangement and random window masking. Then, we devise three layers to progressively obtain patient-level representation, including a local-level interaction layer adopting Manhattan distance, a WSI-level interaction layer employing spatial shuffle, and a patient-level interaction layer using attention pooling. Moreover, the design of hierarchical network helps the model become more computationally efficient. Finally, we validate HVTSurv with 3,104 patients and 3,752 WSIs across 6 cancer types from The Cancer Genome Atlas (TCGA). The average C-Index is 2.50-11.30% higher than all the prior weakly supervised methods over 6 TCGA datasets. Ablation study and attention visualization further verify the superiority of the proposed HVTSurv. Implementation is available at: https://github.com/szc19990412/HVTSurv.

{{</citation>}}


## cs.HC (1)



### (54/91) Large Language Models (GPT) for automating feedback on programming assignments (Maciej Pankiewicz et al., 2023)

{{<citation>}}

Maciej Pankiewicz, Ryan S. Baker. (2023)  
**Large Language Models (GPT) for automating feedback on programming assignments**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2307.00150v1)  

---


**ABSTRACT**  
Addressing the challenge of generating personalized feedback for programming assignments is demanding due to several factors, like the complexity of code syntax or different ways to correctly solve a task. In this experimental study, we automated the process of feedback generation by employing OpenAI's GPT-3.5 model to generate personalized hints for students solving programming assignments on an automated assessment platform. Students rated the usefulness of GPT-generated hints positively. The experimental group (with GPT hints enabled) relied less on the platform's regular feedback but performed better in terms of percentage of successful submissions across consecutive attempts for tasks, where GPT hints were enabled. For tasks where the GPT feedback was made unavailable, the experimental group needed significantly less time to solve assignments. Furthermore, when GPT hints were unavailable, students in the experimental condition were initially less likely to solve the assignment correctly. This suggests potential over-reliance on GPT-generated feedback. However, students in the experimental condition were able to correct reasonably rapidly, reaching the same percentage correct after seven submission attempts. The availability of GPT hints did not significantly impact students' affective state.

{{</citation>}}


## cs.LG (9)



### (55/91) Generalization Limits of Graph Neural Networks in Identity Effects Learning (Giuseppe Alessio D'Inverno et al., 2023)

{{<citation>}}

Giuseppe Alessio D'Inverno, Simone Brugiapaglia, Mirco Ravanelli. (2023)  
**Generalization Limits of Graph Neural Networks in Identity Effects Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.00134v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have emerged as a powerful tool for data-driven learning on various graph domains. They are usually based on a message-passing mechanism and have gained increasing popularity for their intuitive formulation, which is closely linked to the Weisfeiler-Lehman (WL) test for graph isomorphism to which they have been proven equivalent in terms of expressive power. In this work, we establish new generalization properties and fundamental limits of GNNs in the context of learning so-called identity effects, i.e., the task of determining whether an object is composed of two identical components or not. Our study is motivated by the need to understand the capabilities of GNNs when performing simple cognitive tasks, with potential applications in computational linguistics and chemistry. We analyze two case studies: (i) two-letters words, for which we show that GNNs trained via stochastic gradient descent are unable to generalize to unseen letters when utilizing orthogonal encodings like one-hot representations; (ii) dicyclic graphs, i.e., graphs composed of two cycles, for which we present positive existence results leveraging the connection between GNNs and the WL test. Our theoretical analysis is supported by an extensive numerical study.

{{</citation>}}


### (56/91) Redeeming Data Science by Decision Modelling (John Mark Agosta et al., 2023)

{{<citation>}}

John Mark Agosta, Robert Horton. (2023)  
**Redeeming Data Science by Decision Modelling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.00088v1)  

---


**ABSTRACT**  
With the explosion of applications of Data Science, the field is has come loose from its foundations. This article argues for a new program of applied research in areas familiar to researchers in Bayesian methods in AI that are needed to ground the practice of Data Science by borrowing from AI techniques for model formulation that we term ``Decision Modelling.'' This article briefly reviews the formulation process as building a causal graphical model, then discusses the process in terms of six principles that comprise \emph{Decision Quality}, a framework from the popular business literature. We claim that any successful applied ML modelling effort must include these six principles.   We explain how Decision Modelling combines a conventional machine learning model with an explicit value model. To give a specific example we show how this is done by integrating a model's ROC curve with a utility model.

{{</citation>}}


### (57/91) Improving the Transferability of Time Series Forecasting with Decomposition Adaptation (Yan Gao et al., 2023)

{{<citation>}}

Yan Gao, Yan Wang, Qiang Wang. (2023)  
**Improving the Transferability of Time Series Forecasting with Decomposition Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.00066v1)  

---


**ABSTRACT**  
Due to effective pattern mining and feature representation, neural forecasting models based on deep learning have achieved great progress. The premise of effective learning is to collect sufficient data. However, in time series forecasting, it is difficult to obtain enough data, which limits the performance of neural forecasting models. To alleviate the data scarcity limitation, we design Sequence Decomposition Adaptation Network (SeDAN) which is a novel transfer architecture to improve forecasting performance on the target domain by aligning transferable knowledge from cross-domain datasets. Rethinking the transferability of features in time series data, we propose Implicit Contrastive Decomposition to decompose the original features into components including seasonal and trend features, which are easier to transfer. Then we design the corresponding adaptation methods for decomposed features in different domains. Specifically, for seasonal features, we perform joint distribution adaptation and for trend features, we design an Optimal Local Adaptation. We conduct extensive experiments on five benchmark datasets for multivariate time series forecasting. The results demonstrate the effectiveness of our SeDAN. It can provide more efficient and stable knowledge transfer.

{{</citation>}}


### (58/91) Vision Through the Veil: Differential Privacy in Federated Learning for Medical Image Classification (Kishore Babu Nampalle et al., 2023)

{{<citation>}}

Kishore Babu Nampalle, Pradeep Singh, Uppala Vivek Narayan, Balasubramanian Raman. (2023)  
**Vision Through the Veil: Differential Privacy in Federated Learning for Medical Image Classification**  

---
Primary Category: cs.LG  
Categories: 68U10, I-2-1, cs-CR, cs-LG, cs.LG  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2306.17794v1)  

---


**ABSTRACT**  
The proliferation of deep learning applications in healthcare calls for data aggregation across various institutions, a practice often associated with significant privacy concerns. This concern intensifies in medical image analysis, where privacy-preserving mechanisms are paramount due to the data being sensitive in nature. Federated learning, which enables cooperative model training without direct data exchange, presents a promising solution. Nevertheless, the inherent vulnerabilities of federated learning necessitate further privacy safeguards. This study addresses this need by integrating differential privacy, a leading privacy-preserving technique, into a federated learning framework for medical image classification. We introduce a novel differentially private federated learning model and meticulously examine its impacts on privacy preservation and model performance. Our research confirms the existence of a trade-off between model accuracy and privacy settings. However, we demonstrate that strategic calibration of the privacy budget in differential privacy can uphold robust image classification performance while providing substantial privacy protection.

{{</citation>}}


### (59/91) Federated Object Detection for Quality Inspection in Shared Production (Vinit Hegiste et al., 2023)

{{<citation>}}

Vinit Hegiste, Tatjana Legler, Martin Ruskowski. (2023)  
**Federated Object Detection for Quality Inspection in Shared Production**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2306.17645v1)  

---


**ABSTRACT**  
Federated learning (FL) has emerged as a promising approach for training machine learning models on decentralized data without compromising data privacy. In this paper, we propose a FL algorithm for object detection in quality inspection tasks using YOLOv5 as the object detection algorithm and Federated Averaging (FedAvg) as the FL algorithm. We apply this approach to a manufacturing use-case where multiple factories/clients contribute data for training a global object detection model while preserving data privacy on a non-IID dataset. Our experiments demonstrate that our FL approach achieves better generalization performance on the overall clients' test dataset and generates improved bounding boxes around the objects compared to models trained using local clients' datasets. This work showcases the potential of FL for quality inspection tasks in the manufacturing industry and provides valuable insights into the performance and feasibility of utilizing YOLOv5 and FedAvg for federated object detection.

{{</citation>}}


### (60/91) Design of Induction Machines using Reinforcement Learning (Yasmin SarcheshmehPour et al., 2023)

{{<citation>}}

Yasmin SarcheshmehPour, Tommi Ryyppo, Victor Mukherjee, Alex Jung. (2023)  
**Design of Induction Machines using Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.17626v1)  

---


**ABSTRACT**  
The design of induction machine is a challenging task due to different electromagnetic and thermal constraints. Quick estimation of machine's dimensions is important in the sales tool to provide quick quotations to customers based on specific requirements. The key part of this process is to select different design parameters like length, diameter, tooth tip height and winding turns to achieve certain torque, current and temperature of the machine. Electrical machine designers, with their experience know how to alter different machine design parameters to achieve a customer specific operation requirements. We propose a reinforcement learning algorithm to design a customised induction motor. The neural network model is trained off-line by simulating different instances of of electrical machine design game with a reward or penalty function when a good or bad design choice is made. The results demonstrate that the suggested method automates electrical machine design without applying any human engineering knowledge.

{{</citation>}}


### (61/91) Class-Incremental Learning using Diffusion Model for Distillation and Replay (Quentin Jodelet et al., 2023)

{{<citation>}}

Quentin Jodelet, Xin Liu, Yin Jun Phua, Tsuyoshi Murata. (2023)  
**Class-Incremental Learning using Diffusion Model for Distillation and Replay**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2306.17560v1)  

---


**ABSTRACT**  
Class-incremental learning aims to learn new classes in an incremental fashion without forgetting the previously learned ones. Several research works have shown how additional data can be used by incremental models to help mitigate catastrophic forgetting. In this work, following the recent breakthrough in text-to-image generative models and their wide distribution, we propose the use of a pretrained Stable Diffusion model as a source of additional data for class-incremental learning. Compared to competitive methods that rely on external, often unlabeled, datasets of real images, our approach can generate synthetic samples belonging to the same classes as the previously encountered images. This allows us to use those additional data samples not only in the distillation loss but also for replay in the classification loss. Experiments on the competitive benchmarks CIFAR100, ImageNet-Subset, and ImageNet demonstrate how this new approach can be used to further improve the performance of state-of-the-art methods for class-incremental learning on large scale datasets.

{{</citation>}}


### (62/91) The Implicit Bias of Minima Stability in Multivariate Shallow ReLU Networks (Mor Shpigel Nacson et al., 2023)

{{<citation>}}

Mor Shpigel Nacson, Rotem Mulayoff, Greg Ongie, Tomer Michaeli, Daniel Soudry. (2023)  
**The Implicit Bias of Minima Stability in Multivariate Shallow ReLU Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2306.17499v1)  

---


**ABSTRACT**  
We study the type of solutions to which stochastic gradient descent converges when used to train a single hidden-layer multivariate ReLU network with the quadratic loss. Our results are based on a dynamical stability analysis. In the univariate case, it was shown that linearly stable minima correspond to network functions (predictors), whose second derivative has a bounded weighted $L^1$ norm. Notably, the bound gets smaller as the step size increases, implying that training with a large step size leads to `smoother' predictors. Here we generalize this result to the multivariate case, showing that a similar result applies to the Laplacian of the predictor. We demonstrate the tightness of our bound on the MNIST dataset, and show that it accurately captures the behavior of the solutions as a function of the step size. Additionally, we prove a depth separation result on the approximation power of ReLU networks corresponding to stable minima of the loss. Specifically, although shallow ReLU networks are universal approximators, we prove that stable shallow networks are not. Namely, there is a function that cannot be well-approximated by stable single hidden-layer ReLU networks trained with a non-vanishing step size. This is while the same function can be realized as a stable two hidden-layer ReLU network. Finally, we prove that if a function is sufficiently smooth (in a Sobolev sense) then it can be approximated arbitrarily well using shallow ReLU networks that correspond to stable solutions of gradient descent.

{{</citation>}}


### (63/91) Graphtester: Exploring Theoretical Boundaries of GNNs on Graph Datasets (Eren Akbiyik et al., 2023)

{{<citation>}}

Eren Akbiyik, Florian Grötschla, Beni Egressy, Roger Wattenhofer. (2023)  
**Graphtester: Exploring Theoretical Boundaries of GNNs on Graph Datasets**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2306.17482v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have emerged as a powerful tool for learning from graph-structured data. However, even state-of-the-art architectures have limitations on what structures they can distinguish, imposing theoretical limits on what the networks can achieve on different datasets. In this paper, we provide a new tool called Graphtester for a comprehensive analysis of the theoretical capabilities of GNNs for various datasets, tasks, and scores. We use Graphtester to analyze over 40 different graph datasets, determining upper bounds on the performance of various GNNs based on the number of layers. Further, we show that the tool can also be used for Graph Transformers using positional node encodings, thereby expanding its scope. Finally, we demonstrate that features generated by Graphtester can be used for practical applications such as Graph Transformers, and provide a synthetic dataset to benchmark node and edge features, such as positional encodings. The package is freely available at the following URL: https://github.com/meakbiyik/graphtester.

{{</citation>}}


## cs.RO (6)



### (64/91) Goal Representations for Instruction Following: A Semi-Supervised Language Interface to Control (Vivek Myers et al., 2023)

{{<citation>}}

Vivek Myers, Andre He, Kuan Fang, Homer Walke, Philippe Hansen-Estruch, Ching-An Cheng, Mihai Jalobeanu, Andrey Kolobov, Anca Dragan, Sergey Levine. (2023)  
**Goal Representations for Instruction Following: A Semi-Supervised Language Interface to Control**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.00117v1)  

---


**ABSTRACT**  
Our goal is for robots to follow natural language instructions like "put the towel next to the microwave." But getting large amounts of labeled data, i.e. data that contains demonstrations of tasks labeled with the language instruction, is prohibitive. In contrast, obtaining policies that respond to image goals is much easier, because any autonomous trial or demonstration can be labeled in hindsight with its final state as the goal. In this work, we contribute a method that taps into joint image- and goal- conditioned policies with language using only a small amount of language data. Prior work has made progress on this using vision-language models or by jointly training language-goal-conditioned policies, but so far neither method has scaled effectively to real-world robot tasks without significant human annotation. Our method achieves robust performance in the real world by learning an embedding from the labeled data that aligns language not to the goal image, but rather to the desired change between the start and goal images that the instruction corresponds to. We then train a policy on this embedding: the policy benefits from all the unlabeled data, but the aligned embedding provides an interface for language to steer the policy. We show instruction following across a variety of manipulation tasks in different scenes, with generalization to language instructions outside of the labeled data. Videos and code for our approach can be found on our website: http://tiny.cc/grif .

{{</citation>}}


### (65/91) Statler: State-Maintaining Language Models for Embodied Reasoning (Takuma Yoneda et al., 2023)

{{<citation>}}

Takuma Yoneda, Jiading Fang, Peng Li, Huanyu Zhang, Tianchong Jiang, Shengjie Lin, Ben Picker, David Yunis, Hongyuan Mei, Matthew R. Walter. (2023)  
**Statler: State-Maintaining Language Models for Embodied Reasoning**  

---
Primary Category: cs.RO  
Categories: cs-CL, cs-RO, cs.RO  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2306.17840v2)  

---


**ABSTRACT**  
Large language models (LLMs) provide a promising tool that enable robots to perform complex robot reasoning tasks. However, the limited context window of contemporary LLMs makes reasoning over long time horizons difficult. Embodied tasks such as those that one might expect a household robot to perform typically require that the planner consider information acquired a long time ago (e.g., properties of the many objects that the robot previously encountered in the environment). Attempts to capture the world state using an LLM's implicit internal representation is complicated by the paucity of task- and environment-relevant information available in a robot's action history, while methods that rely on the ability to convey information via the prompt to the LLM are subject to its limited context window. In this paper, we propose Statler, a framework that endows LLMs with an explicit representation of the world state as a form of ``memory'' that is maintained over time. Integral to Statler is its use of two instances of general LLMs -- a world-model reader and a world-model writer -- that interface with and maintain the world state. By providing access to this world state ``memory'', Statler improves the ability of existing LLMs to reason over longer time horizons without the constraint of context length. We evaluate the effectiveness of our approach on three simulated table-top manipulation domains and a real robot domain, and show that it improves the state-of-the-art in LLM-based robot reasoning. Project website: https://statler-lm.github.io/

{{</citation>}}


### (66/91) Act3D: Infinite Resolution Action Detection Transformer for Robotic Manipulation (Theophile Gervet et al., 2023)

{{<citation>}}

Theophile Gervet, Zhou Xian, Nikolaos Gkanatsios, Katerina Fragkiadaki. (2023)  
**Act3D: Infinite Resolution Action Detection Transformer for Robotic Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2306.17817v1)  

---


**ABSTRACT**  
3D perceptual representations are well suited for robot manipulation as they easily encode occlusions and simplify spatial reasoning. Many manipulation tasks require high spatial precision in end-effector pose prediction, typically demanding high-resolution 3D perceptual grids that are computationally expensive to process. As a result, most manipulation policies operate directly in 2D, foregoing 3D inductive biases. In this paper, we propose Act3D, a manipulation policy Transformer that casts 6-DoF keypose prediction as 3D detection with adaptive spatial computation. It takes as input 3D feature clouds unprojected from one or more camera views, iteratively samples 3D point grids in free space in a coarse-to-fine manner, featurizes them using relative spatial attention to the physical feature cloud, and selects the best feature point for end-effector pose prediction. Act3D sets a new state-of-the-art in RLbench, an established manipulation benchmark. Our model achieves 10% absolute improvement over the previous SOTA 2D multi-view policy on 74 RLbench tasks and 22% absolute improvement with 3x less compute over the previous SOTA 3D policy. In thorough ablations, we show the importance of relative spatial attention, large-scale vision-language pre-trained 2D backbones, and weight tying across coarse-to-fine attentions. Code and videos are available at our project site: https://act3d.github.io/.

{{</citation>}}


### (67/91) Navigation of micro-robot swarms for targeted delivery using reinforcement learning (Akshatha Jagadish et al., 2023)

{{<citation>}}

Akshatha Jagadish, Manoj Varma. (2023)  
**Navigation of micro-robot swarms for targeted delivery using reinforcement learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.17598v1)  

---


**ABSTRACT**  
Micro robotics is quickly emerging to be a promising technological solution to many medical treatments with focus on targeted drug delivery. They are effective when working in swarms whose individual control is mostly infeasible owing to their minute size. Controlling a number of robots with a single controller is thus important and artificial intelligence can help us perform this task successfully. In this work, we use the Reinforcement Learning (RL) algorithms Proximal Policy Optimization (PPO) and Robust Policy Optimization (RPO) to navigate a swarm of 4, 9 and 16 microswimmers under hydrodynamic effects, controlled by their orientation, towards a circular absorbing target. We look at both PPO and RPO performances with limited state information scenarios and also test their robustness for random target location and size. We use curriculum learning to improve upon the performance and demonstrate the same in learning to navigate a swarm of 25 swimmers and steering the swarm to exemplify the manoeuvring capabilities of the RL model.

{{</citation>}}


### (68/91) What Could a Social Mediator Robot Do? Lessons from Real-World Mediation Scenarios (Thomas H. Weisswange et al., 2023)

{{<citation>}}

Thomas H. Weisswange, Hifza Javed, Manuel Dietrich, Tuan Vu Pham, Maria Teresa Parreira, Michael Sack, Nawid Jamali. (2023)  
**What Could a Social Mediator Robot Do? Lessons from Real-World Mediation Scenarios**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-RO, cs.RO  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2306.17379v1)  

---


**ABSTRACT**  
The use of social robots as instruments for social mediation has been gaining traction in the field of Human-Robot Interaction (HRI). So far, the design of such robots and their behaviors is often driven by technological platforms and experimental setups in controlled laboratory environments. To address complex social relationships in the real world, it is crucial to consider the actual needs and consequences of the situations found therein. This includes understanding when a mediator is necessary, what specific role such a robot could play, and how it moderates human social dynamics. In this paper, we discuss six relevant roles for robotic mediators that we identified by investigating a collection of videos showing realistic group situations. We further discuss mediation behaviors and target measures to evaluate the success of such interventions. We hope that our findings can inspire future research on robot-assisted social mediation by highlighting a wider set of mediation applications than those found in prior studies. Specifically, we aim to inform the categorization and selection of interaction scenarios that reflect real situations, where a mediation robot can have a positive and meaningful impact on group dynamics.

{{</citation>}}


### (69/91) Group Dynamics: Survey of Existing Multimodal Models and Considerations for Social Mediation (Hifza Javed et al., 2023)

{{<citation>}}

Hifza Javed, Nawid Jamali. (2023)  
**Group Dynamics: Survey of Existing Multimodal Models and Considerations for Social Mediation**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-RO, cs.RO  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2306.17374v1)  

---


**ABSTRACT**  
Social mediator robots facilitate human-human interactions by producing behavior strategies that positively influence how humans interact with each other in social settings. As robots for social mediation gain traction in the field of human-human-robot interaction, their ability to "understand" the humans in their environments becomes crucial. This objective requires models of human understanding that consider multiple humans in an interaction as a collective entity and represent the group dynamics that exist among its members. Group dynamics are defined as the influential actions, processes, and changes that occur within and between group interactants. Since an individual's behavior may be deeply influenced by their interactions with other group members, the social dynamics existing within a group can influence the behaviors, attitudes, and opinions of each individual and the group as a whole. Therefore, models of group dynamics are critical for a social mediator robot to be effective in its role. In this paper, we survey existing models of group dynamics and categorize them into models of social dominance, affect, social cohesion, conflict resolution, and engagement. We highlight the multimodal features these models utilize, and emphasize the importance of capturing the interpersonal aspects of a social interaction. Finally, we make a case for models of relational affect as an approach that may be able to capture a representation of human-human interactions that can be useful for social mediation.

{{</citation>}}


## cs.CY (1)



### (70/91) Performance of ChatGPT on USMLE: Unlocking the Potential of Large Language Models for AI-Assisted Medical Education (Prabin Sharma et al., 2023)

{{<citation>}}

Prabin Sharma, Kisan Thapa, Prastab Dhakal, Mala Deep Upadhaya, Santosh Adhikari, Salik Ram Khanal. (2023)  
**Performance of ChatGPT on USMLE: Unlocking the Potential of Large Language Models for AI-Assisted Medical Education**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2307.00112v1)  

---


**ABSTRACT**  
Artificial intelligence is gaining traction in more ways than ever before. The popularity of language models and AI-based businesses has soared since ChatGPT was made available to the general public via OpenAI. It is becoming increasingly common for people to use ChatGPT both professionally and personally. Considering the widespread use of ChatGPT and the reliance people place on it, this study determined how reliable ChatGPT can be for answering complex medical and clinical questions. Harvard University gross anatomy along with the United States Medical Licensing Examination (USMLE) questionnaire were used to accomplish the objective. The paper evaluated the obtained results using a 2-way ANOVA and posthoc analysis. Both showed systematic covariation between format and prompt. Furthermore, the physician adjudicators independently rated the outcome's accuracy, concordance, and insight. As a result of the analysis, ChatGPT-generated answers were found to be more context-oriented and represented a better model for deductive reasoning than regular Google search results. Furthermore, ChatGPT obtained 58.8% on logical questions and 60% on ethical questions. This means that the ChatGPT is approaching the passing range for logical questions and has crossed the threshold for ethical questions. The paper believes ChatGPT and other language learning models can be invaluable tools for e-learners; however, the study suggests that there is still room to improve their accuracy. In order to improve ChatGPT's performance in the future, further research is needed to better understand how it can answer different types of questions.

{{</citation>}}


## cs.SI (2)



### (71/91) DECOR: Degree-Corrected Social Graph Refinement for Fake News Detection (Jiaying Wu et al., 2023)

{{<citation>}}

Jiaying Wu, Bryan Hooi. (2023)  
**DECOR: Degree-Corrected Social Graph Refinement for Fake News Detection**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Fake News, GNN  
[Paper Link](http://arxiv.org/abs/2307.00077v1)  

---


**ABSTRACT**  
Recent efforts in fake news detection have witnessed a surge of interest in using graph neural networks (GNNs) to exploit rich social context. Existing studies generally leverage fixed graph structures, assuming that the graphs accurately represent the related social engagements. However, edge noise remains a critical challenge in real-world graphs, as training on suboptimal structures can severely limit the expressiveness of GNNs. Despite initial efforts in graph structure learning (GSL), prior works often leverage node features to update edge weights, resulting in heavy computational costs that hinder the methods' applicability to large-scale social graphs. In this work, we approach the fake news detection problem with a novel aspect of social graph refinement. We find that the degrees of news article nodes exhibit distinctive patterns, which are indicative of news veracity. Guided by this, we propose DECOR, a novel application of Degree-Corrected Stochastic Blockmodels to the fake news detection problem. Specifically, we encapsulate our empirical observations into a lightweight social graph refinement component that iteratively updates the edge weights via a learnable degree correction mask, which allows for joint optimization with a GNN-based detector. Extensive experiments on two real-world benchmarks validate the effectiveness and efficiency of DECOR.

{{</citation>}}


### (72/91) Beyond Active Engagement: The Significance of Lurkers in a Polarized Twitter Debate (Anees Baqir et al., 2023)

{{<citation>}}

Anees Baqir, Yijing Chen, Fernando Diaz-Diaz, Sercan Kiyak, Thomas Louf, Virginia Morini, Valentina Pansanella, Maddalena Torricelli, Alessandro Galeazzi. (2023)  
**Beyond Active Engagement: The Significance of Lurkers in a Polarized Twitter Debate**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2306.17538v1)  

---


**ABSTRACT**  
The emergence of new public forums in the shape of online social media has introduced unprecedented challenges to public discourse, including polarization, misinformation, and the emergence of echo chambers. While existing research has extensively studied the behavior of active users within echo chambers, little attention has been given to the hidden audience, also known as lurkers, who passively consume content without actively engaging. This study aims to estimate the share of the hidden audience and investigate their interplay with the echo chamber effect. Using Twitter as a case study, we analyze a polarized political debate to understand the engagement patterns and factors influencing the hidden audience's presence. Our findings reveal a relevant fraction of users that consume content without active interaction, which underscores the importance of considering their presence in online debates. Notably, our results indicate that the engagement of the hidden audience is primarily influenced by factors such as the reliability of media sources mentioned in tweets rather than the ideological stance of the user that produced the content. These findings highlight the need for a comprehensive understanding of the hidden audience's role in online debates and how they may influence public opinion.

{{</citation>}}


## cs.AI (6)



### (73/91) Transformers in Healthcare: A Survey (Subhash Nerella et al., 2023)

{{<citation>}}

Subhash Nerella, Sabyasachi Bandyopadhyay, Jiaqing Zhang, Miguel Contreras, Scott Siegel, Aysegul Bumin, Brandon Silva, Jessica Sena, Benjamin Shickel, Azra Bihorac, Kia Khezeli, Parisa Rashidi. (2023)  
**Transformers in Healthcare: A Survey**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-LG, cs.AI  
Keywords: AI, NLP, Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.00067v1)  

---


**ABSTRACT**  
With Artificial Intelligence (AI) increasingly permeating various aspects of society, including healthcare, the adoption of the Transformers neural network architecture is rapidly changing many applications. Transformer is a type of deep learning architecture initially developed to solve general-purpose Natural Language Processing (NLP) tasks and has subsequently been adapted in many fields, including healthcare. In this survey paper, we provide an overview of how this architecture has been adopted to analyze various forms of data, including medical imaging, structured and unstructured Electronic Health Records (EHR), social media, physiological signals, and biomolecular sequences. Those models could help in clinical diagnosis, report generation, data reconstruction, and drug/protein synthesis. We identified relevant studies using the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines. We also discuss the benefits and limitations of using transformers in healthcare and examine issues such as computational cost, model interpretability, fairness, alignment with human values, ethical implications, and environmental impact.

{{</citation>}}


### (74/91) Comparing Reinforcement Learning and Human Learning using the Game of Hidden Rules (Eric Pulick et al., 2023)

{{<citation>}}

Eric Pulick, Vladimir Menkov, Yonatan Mintz, Paul Kantor, Vicki Bier. (2023)  
**Comparing Reinforcement Learning and Human Learning using the Game of Hidden Rules**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.17766v1)  

---


**ABSTRACT**  
Reliable real-world deployment of reinforcement learning (RL) methods requires a nuanced understanding of their strengths and weaknesses and how they compare to those of humans. Human-machine systems are becoming more prevalent and the design of these systems relies on a task-oriented understanding of both human learning (HL) and RL. Thus, an important line of research is characterizing how the structure of a learning task affects learning performance. While increasingly complex benchmark environments have led to improved RL capabilities, such environments are difficult to use for the dedicated study of task structure. To address this challenge we present a learning environment built to support rigorous study of the impact of task structure on HL and RL. We demonstrate the environment's utility for such study through example experiments in task structure that show performance differences between humans and RL algorithms.

{{</citation>}}


### (75/91) Systematic Investigation of Sparse Perturbed Sharpness-Aware Minimization Optimizer (Peng Mi et al., 2023)

{{<citation>}}

Peng Mi, Li Shen, Tianhe Ren, Yiyi Zhou, Tianshuo Xu, Xiaoshuai Sun, Tongliang Liu, Rongrong Ji, Dacheng Tao. (2023)  
**Systematic Investigation of Sparse Perturbed Sharpness-Aware Minimization Optimizer**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2306.17504v1)  

---


**ABSTRACT**  
Deep neural networks often suffer from poor generalization due to complex and non-convex loss landscapes. Sharpness-Aware Minimization (SAM) is a popular solution that smooths the loss landscape by minimizing the maximized change of training loss when adding a perturbation to the weight. However, indiscriminate perturbation of SAM on all parameters is suboptimal and results in excessive computation, double the overhead of common optimizers like Stochastic Gradient Descent (SGD). In this paper, we propose Sparse SAM (SSAM), an efficient and effective training scheme that achieves sparse perturbation by a binary mask. To obtain the sparse mask, we provide two solutions based on Fisher information and dynamic sparse training, respectively. We investigate the impact of different masks, including unstructured, structured, and $N$:$M$ structured patterns, as well as explicit and implicit forms of implementing sparse perturbation. We theoretically prove that SSAM can converge at the same rate as SAM, i.e., $O(\log T/\sqrt{T})$. Sparse SAM has the potential to accelerate training and smooth the loss landscape effectively. Extensive experimental results on CIFAR and ImageNet-1K confirm that our method is superior to SAM in terms of efficiency, and the performance is preserved or even improved with a perturbation of merely 50\% sparsity. Code is available at https://github.com/Mi-Peng/Systematic-Investigation-of-Sparse-Perturbed-Sharpness-Aware-Minimization-Optimizer.

{{</citation>}}


### (76/91) An automated method for the ontological representation of security directives (Giampaolo Bella et al., 2023)

{{<citation>}}

Giampaolo Bella, Gianpietro Castiglione, Daniele Francesco Santamaria. (2023)  
**An automated method for the ontological representation of security directives**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.01211v1)  

---


**ABSTRACT**  
Large documents written in juridical language are difficult to interpret, with long sentences leading to intricate and intertwined relations between the nouns. The present paper frames this problem in the context of recent European security directives. The complexity of their language is here thwarted by automating the extraction of the relevant information, namely of the parts of speech from each clause, through a specific tailoring of Natural Language Processing (NLP) techniques. These contribute, in combination with ontology development principles, to the design of our automated method for the representation of security directives as ontologies. The method is showcased on a practical problem, namely to derive an ontology representing the NIS 2 directive, which is the peak of cybersecurity prescripts at the European level. Although the NLP techniques adopted showed some limitations and had to be complemented by manual analysis, the overall results provide valid support for directive compliance in general and for ontology development in particular.

{{</citation>}}


### (77/91) Harnessing LLMs in Curricular Design: Using GPT-4 to Support Authoring of Learning Objectives (Pragnya Sridhar et al., 2023)

{{<citation>}}

Pragnya Sridhar, Aidan Doyle, Arav Agarwal, Christopher Bogart, Jaromir Savelka, Majd Sakr. (2023)  
**Harnessing LLMs in Curricular Design: Using GPT-4 to Support Authoring of Learning Objectives**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2306.17459v1)  

---


**ABSTRACT**  
We evaluated the capability of a generative pre-trained transformer (GPT-4) to automatically generate high-quality learning objectives (LOs) in the context of a practically oriented university course on Artificial Intelligence. Discussions of opportunities (e.g., content generation, explanation) and risks (e.g., cheating) of this emerging technology in education have intensified, but to date there has not been a study of the models' capabilities in supporting the course design and authoring of LOs. LOs articulate the knowledge and skills learners are intended to acquire by engaging with a course. To be effective, LOs must focus on what students are intended to achieve, focus on specific cognitive processes, and be measurable. Thus, authoring high-quality LOs is a challenging and time consuming (i.e., expensive) effort. We evaluated 127 LOs that were automatically generated based on a carefully crafted prompt (detailed guidelines on high-quality LOs authoring) submitted to GPT-4 for conceptual modules and projects of an AI Practitioner course. We analyzed the generated LOs if they follow certain best practices such as beginning with action verbs from Bloom's taxonomy in regards to the level of sophistication intended. Our analysis showed that the generated LOs are sensible, properly expressed (e.g., starting with an action verb), and that they largely operate at the appropriate level of Bloom's taxonomy, respecting the different nature of the conceptual modules (lower levels) and projects (higher levels). Our results can be leveraged by instructors and curricular designers wishing to take advantage of the state-of-the-art generative models to support their curricular and course design efforts.

{{</citation>}}


### (78/91) LMBot: Distilling Graph Knowledge into Language Model for Graph-less Deployment in Twitter Bot Detection (Zijian Cai et al., 2023)

{{<citation>}}

Zijian Cai, Zhaoxuan Tan, Zhenyu Lei, Zifeng Zhu, Hongrui Wang, Qinghua Zheng, Minnan Luo. (2023)  
**LMBot: Distilling Graph Knowledge into Language Model for Graph-less Deployment in Twitter Bot Detection**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-SI, cs.AI  
Keywords: GNN, Language Model, Twitter  
[Paper Link](http://arxiv.org/abs/2306.17408v2)  

---


**ABSTRACT**  
As malicious actors employ increasingly advanced and widespread bots to disseminate misinformation and manipulate public opinion, the detection of Twitter bots has become a crucial task. Though graph-based Twitter bot detection methods achieve state-of-the-art performance, we find that their inference depends on the neighbor users multi-hop away from the targets, and fetching neighbors is time-consuming and may introduce bias. At the same time, we find that after finetuning on Twitter bot detection, pretrained language models achieve competitive performance and do not require a graph structure during deployment. Inspired by this finding, we propose a novel bot detection framework LMBot that distills the knowledge of graph neural networks (GNNs) into language models (LMs) for graph-less deployment in Twitter bot detection to combat the challenge of data dependency. Moreover, LMBot is compatible with graph-based and graph-less datasets. Specifically, we first represent each user as a textual sequence and feed them into the LM for domain adaptation. For graph-based datasets, the output of LMs provides input features for the GNN, enabling it to optimize for bot detection and distill knowledge back to the LM in an iterative, mutually enhancing process. Armed with the LM, we can perform graph-less inference, which resolves the graph data dependency and sampling bias issues. For datasets without graph structure, we simply replace the GNN with an MLP, which has also shown strong performance. Our experiments demonstrate that LMBot achieves state-of-the-art performance on four Twitter bot detection benchmarks. Extensive studies also show that LMBot is more robust, versatile, and efficient compared to graph-based Twitter bot detection methods.

{{</citation>}}


## stat.ML (2)



### (79/91) The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit (Lorenzo Noci et al., 2023)

{{<citation>}}

Lorenzo Noci, Chuning Li, Mufan Bill Li, Bobby He, Thomas Hofmann, Chris Maddison, Daniel M. Roy. (2023)  
**The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2306.17759v1)  

---


**ABSTRACT**  
In deep learning theory, the covariance matrix of the representations serves as a proxy to examine the network's trainability. Motivated by the success of Transformers, we study the covariance matrix of a modified Softmax-based attention model with skip connections in the proportional limit of infinite-depth-and-width. We show that at initialization the limiting distribution can be described by a stochastic differential equation (SDE) indexed by the depth-to-width ratio. To achieve a well-defined stochastic limit, the Transformer's attention mechanism is modified by centering the Softmax output at identity, and scaling the Softmax logits by a width-dependent temperature parameter. We examine the stability of the network through the corresponding SDE, showing how the scale of both the drift and diffusion can be elegantly controlled with the aid of residual connections. The existence of a stable SDE implies that the covariance structure is well-behaved, even for very large depth and width, thus preventing the notorious issues of rank degeneracy in deep attention models. Finally, we show, through simulations, that the SDE provides a surprisingly good description of the corresponding finite-size model. We coin the name shaped Transformer for these architectural modifications.

{{</citation>}}


### (80/91) Generalized Time Warping Invariant Dictionary Learning for Time Series Classification and Clustering (Ruiyu Xu et al., 2023)

{{<citation>}}

Ruiyu Xu, Chao Wang, Yongxiang Li, Jianguo Wu. (2023)  
**Generalized Time Warping Invariant Dictionary Learning for Time Series Classification and Clustering**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2306.17690v1)  

---


**ABSTRACT**  
Dictionary learning is an effective tool for pattern recognition and classification of time series data. Among various dictionary learning techniques, the dynamic time warping (DTW) is commonly used for dealing with temporal delays, scaling, transformation, and many other kinds of temporal misalignments issues. However, the DTW suffers overfitting or information loss due to its discrete nature in aligning time series data. To address this issue, we propose a generalized time warping invariant dictionary learning algorithm in this paper. Our approach features a generalized time warping operator, which consists of linear combinations of continuous basis functions for facilitating continuous temporal warping. The integration of the proposed operator and the dictionary learning is formulated as an optimization problem, where the block coordinate descent method is employed to jointly optimize warping paths, dictionaries, and sparseness coefficients. The optimized results are then used as hyperspace distance measures to feed classification and clustering algorithms. The superiority of the proposed method in terms of dictionary learning, classification, and clustering is validated through ten sets of public datasets in comparing with various benchmark methods.

{{</citation>}}


## cs.DS (1)



### (81/91) An Improved Deterministic Algorithm for the Online Min-Sum Set Cover Problem (Mateusz Basiak et al., 2023)

{{<citation>}}

Mateusz Basiak, Marcin Bienkowski, Agnieszka Tatarczuk. (2023)  
**An Improved Deterministic Algorithm for the Online Min-Sum Set Cover Problem**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs.DS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.17755v1)  

---


**ABSTRACT**  
We study the online variant of the Min-Sum Set Cover (MSSC) problem, a generalization of the well-known list update problem. In the MSSC problem, an algorithm has to maintain the time-varying permutation of the list of $n$ elements, and serve a sequence of requests $R_1, R_2, \dots, R_t, \dots$. Each $R_t$ is a subset of elements of cardinality at most $r$. For a requested set $R_t$, an online algorithm has to pay the cost equal to the position of the first element from $R_t$ on its list. Then, it may arbitrarily permute its list, paying the number of swapped adjacent element pairs.   We present the first constructive deterministic algorithm for this problem, whose competitive ratio does not depend on $n$. Our algorithm is $O(r^2)$-competitive, which beats both the existential upper bound of $O(r^4)$ by Bienkowski and Mucha [AAAI '23] and the previous constructive bound of $O(r^{3/2} \cdot \sqrt{n})$ by Fotakis et al. [ICALP '20]. Furthermore, we show that our algorithm attains an asymptotically optimal competitive ratio of $O(r)$ when compared to the best fixed permutation of elements.

{{</citation>}}


## cs.MA (1)



### (82/91) Discriminatory or Samaritan -- which AI is needed for humanity? An Evolutionary Game Theory Analysis of Hybrid Human-AI populations (Tim Booker et al., 2023)

{{<citation>}}

Tim Booker, Manuel Miranda, Jesús A. Moreno López, José María Ramos Fernández, Max Reddel, Valeria Widler, Filippo Zimmaro, Alberto Antonioni, The Anh Han. (2023)  
**Discriminatory or Samaritan -- which AI is needed for humanity? An Evolutionary Game Theory Analysis of Hybrid Human-AI populations**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-MA, cs.MA, math-DS, math-OC, nlin-AO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.17747v2)  

---


**ABSTRACT**  
As artificial intelligence (AI) systems are increasingly embedded in our lives, their presence leads to interactions that shape our behaviour, decision-making, and social interactions. Existing theoretical research has primarily focused on human-to-human interactions, overlooking the unique dynamics triggered by the presence of AI. In this paper, resorting to methods from evolutionary game theory, we study how different forms of AI influence the evolution of cooperation in a human population playing the one-shot Prisoner's Dilemma game in both well-mixed and structured populations. We found that Samaritan AI agents that help everyone unconditionally, including defectors, can promote higher levels of cooperation in humans than Discriminatory AI that only help those considered worthy/cooperative, especially in slow-moving societies where change is viewed with caution or resistance (small intensities of selection). Intuitively, in fast-moving societies (high intensities of selection), Discriminatory AIs promote higher levels of cooperation than Samaritan AIs.

{{</citation>}}


## cs.NE (2)



### (83/91) Towards Brain Inspired Design for Addressing the Shortcomings of ANNs (Fahad Sarfraz et al., 2023)

{{<citation>}}

Fahad Sarfraz, Elahe Arani, Bahram Zonooz. (2023)  
**Towards Brain Inspired Design for Addressing the Shortcomings of ANNs**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-CV, cs-LG, cs-NE, cs.NE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.00039v1)  

---


**ABSTRACT**  
As our understanding of the mechanisms of brain function is enhanced, the value of insights gained from neuroscience to the development of AI algorithms deserves further consideration. Here, we draw parallels with an existing tree-based ANN architecture and a recent neuroscience study[27] arguing that the error-based organization of neurons in the cerebellum that share a preference for a personalized view of the entire error space, may account for several desirable features of behavior and learning. We then analyze the learning behavior and characteristics of the model under varying scenarios to gauge the potential benefits of a similar mechanism in ANN. Our empirical results suggest that having separate populations of neurons with personalized error views can enable efficient learning under class imbalance and limited data, and reduce the susceptibility to unintended shortcut strategies, leading to improved generalization. This work highlights the potential of translating the learning machinery of the brain into the design of a new generation of ANNs and provides further credence to the argument that biologically inspired AI may hold the key to overcoming the shortcomings of ANNs.

{{</citation>}}


### (84/91) Comparing Algorithm Selection Approaches on Black-Box Optimization Problems (Ana Kostovska et al., 2023)

{{<citation>}}

Ana Kostovska, Anja Jankovic, Diederick Vermetten, Sašo Džeroski, Tome Eftimov, Carola Doerr. (2023)  
**Comparing Algorithm Selection Approaches on Black-Box Optimization Problems**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2306.17585v1)  

---


**ABSTRACT**  
Performance complementarity of solvers available to tackle black-box optimization problems gives rise to the important task of algorithm selection (AS). Automated AS approaches can help replace tedious and labor-intensive manual selection, and have already shown promising performance in various optimization domains. Automated AS relies on machine learning (ML) techniques to recommend the best algorithm given the information about the problem instance. Unfortunately, there are no clear guidelines for choosing the most appropriate one from a variety of ML techniques. Tree-based models such as Random Forest or XGBoost have consistently demonstrated outstanding performance for automated AS. Transformers and other tabular deep learning models have also been increasingly applied in this context.   We investigate in this work the impact of the choice of the ML technique on AS performance. We compare four ML models on the task of predicting the best solver for the BBOB problems for 7 different runtime budgets in 2 dimensions. While our results confirm that a per-instance AS has indeed impressive potential, we also show that the particular choice of the ML technique is of much minor importance.

{{</citation>}}


## cs.NI (1)



### (85/91) Federated Multi-Agent Deep Reinforcement Learning for Dynamic and Flexible 3D Operation of 5G Multi-MAP Networks (Esteban Catté et al., 2023)

{{<citation>}}

Esteban Catté, Mohamed Sana, Mickael Maman. (2023)  
**Federated Multi-Agent Deep Reinforcement Learning for Dynamic and Flexible 3D Operation of 5G Multi-MAP Networks**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-MA, cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06842v1)  

---


**ABSTRACT**  
This paper addresses the efficient management of Mobile Access Points (MAPs), which are Unmanned Aerial Vehicles (UAV), in 5G networks. We propose a two-level hierarchical architecture, which dynamically reconfigures the network while considering Integrated Access-Backhaul (IAB) constraints. The high-layer decision process determines the number of MAPs through consensus, and we develop a joint optimization process to account for co-dependence in network self-management. In the low-layer, MAPs manage their placement using a double-attention based Deep Reinforcement Learning (DRL) model that encourages cooperation without retraining. To improve generalization and reduce complexity, we propose a federated mechanism for training and sharing one placement model for every MAP in the low-layer. Additionally, we jointly optimize the placement and backhaul connectivity of MAPs using a multi-objective reward function, considering the impact of varying MAP placement on wireless backhaul connectivity.

{{</citation>}}


## cs.CR (2)



### (86/91) A Quic(k) Security Overview: A Literature Research on Implemented Security Recommendations (Stefan Tatschner et al., 2023)

{{<citation>}}

Stefan Tatschner, Sebastian N. Peters, David Emeis, John Morris, Thomas Newe. (2023)  
**A Quic(k) Security Overview: A Literature Research on Implemented Security Recommendations**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2306.17568v1)  

---


**ABSTRACT**  
Built on top of UDP, the relatively new QUIC protocol serves as the baseline for modern web protocol stacks. Equipped with a rich feature set, the protocol is defined by a 151 pages strong IETF standard complemented by several additional documents. Enabling fast updates and feature iteration, most QUIC implementations are implemented as user space libraries leading to a large and fragmented ecosystem. This work addresses the research question, "if a complex standard with a large number of different implementations leads to an insecure ecosystem?". The relevant RFC documents were studied and "Security Consideration" items describing conceptional problems were extracted. During the research, 13 popular production ready QUIC implementations were compared by evaluating 10 security considerations from RFC9000. While related studies mostly focused on the functional part of QUIC, this study confirms that available QUIC implementations are not yet mature enough from a security point of view.

{{</citation>}}


### (87/91) Research on Virus Cyberattack-Defense Based on Electromagnetic Radiation (Ruochen Wu, 2023)

{{<citation>}}

Ruochen Wu. (2023)  
**Research on Virus Cyberattack-Defense Based on Electromagnetic Radiation**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.17508v1)  

---


**ABSTRACT**  
Information technology and telecommunications have rapidly permeated various domains, resulting in a significant influx of data traversing the networks between computers. Consequently, research of cyberattacks in computer systems has become crucial for many organizations. Accordingly, recent cybersecurity incidents have underscored the rapidly evolving nature of future threats and attack methods, particularly those involving computer viruses wireless injection. This paper aims to study and demonstrate the feasibility of remote computer virus radiation injection. To achieve this objective, digital signal processing (DSP) plays a vital role. By studying the principles and models of radiation attacks and computer virus propagation, the modulation of the binary data stream of the simulated virus into a terahertz radar carrier signal by Phase-Shift Keying (PSK) is simulated, enabling the implementation of an attack through the "field to line" coupling of electromagnetic signals. Finally, the defense and countermeasures based on signal recognition are discussed for such attacks. Additionally, an idea of establishing a virus library for cyberattack signals and employing artificial intelligence (AI) algorithms for automated intrusion detection is proposed as a means to achieve cybersecurity situation awareness.

{{</citation>}}


## cs.SD (2)



### (88/91) Empirical Interpretation of the Relationship Between Speech Acoustic Context and Emotion Recognition (Anna Ollerenshaw et al., 2023)

{{<citation>}}

Anna Ollerenshaw, Md Asif Jalal, Rosanna Milner, Thomas Hain. (2023)  
**Empirical Interpretation of the Relationship Between Speech Acoustic Context and Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2306.17500v1)  

---


**ABSTRACT**  
Speech emotion recognition (SER) is vital for obtaining emotional intelligence and understanding the contextual meaning of speech. Variations of consonant-vowel (CV) phonemic boundaries can enrich acoustic context with linguistic cues, which impacts SER. In practice, speech emotions are treated as single labels over an acoustic segment for a given time duration. However, phone boundaries within speech are not discrete events, therefore the perceived emotion state should also be distributed over potentially continuous time-windows.   This research explores the implication of acoustic context and phone boundaries on local markers for SER using an attention-based approach. The benefits of using a distributed approach to speech emotion understanding are supported by the results of cross-corpora analysis experiments. Experiments where phones and words are mapped to the attention vectors along with the fundamental frequency to observe the overlapping distributions and thereby the relationship between acoustic context and emotion. This work aims to bridge psycholinguistic theory research with computational modelling for SER.

{{</citation>}}


### (89/91) Audio Embeddings as Teachers for Music Classification (Yiwei Ding et al., 2023)

{{<citation>}}

Yiwei Ding, Alexander Lerch. (2023)  
**Audio Embeddings as Teachers for Music Classification**  

---
Primary Category: cs.SD  
Categories: cs-IR, cs-SD, cs.SD, eess-AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2306.17424v1)  

---


**ABSTRACT**  
Music classification has been one of the most popular tasks in the field of music information retrieval. With the development of deep learning models, the last decade has seen impressive improvements in a wide range of classification tasks. However, the increasing model complexity makes both training and inference computationally expensive. In this paper, we integrate the ideas of transfer learning and feature-based knowledge distillation and systematically investigate using pre-trained audio embeddings as teachers to guide the training of low-complexity student networks. By regularizing the feature space of the student networks with the pre-trained embeddings, the knowledge in the teacher embeddings can be transferred to the students. We use various pre-trained audio embeddings and test the effectiveness of the method on the tasks of musical instrument classification and music auto-tagging. Results show that our method significantly improves the results in comparison to the identical model trained without the teacher's knowledge. This technique can also be combined with classical knowledge distillation approaches to further improve the model's performance.

{{</citation>}}


## cs.IT (1)



### (90/91) TransDetector: A Transformer-Based Detector for Underwater Acoustic Differential OFDM Communications (Yuzhou Li et al., 2023)

{{<citation>}}

Yuzhou Li, Sixiang Wang, Di Liu, Chuang Zhou. (2023)  
**TransDetector: A Transformer-Based Detector for Underwater Acoustic Differential OFDM Communications**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2306.17392v1)  

---


**ABSTRACT**  
Inter-carrier interference (ICI) and noise mitigation is crucial for precise signal detection in underwater acoustic (UWA) differential orthogonal frequency division multiplexing (DOFDM) communication systems. In this paper, we adopt the Transformer to design a detector, referred to as the TransDetector, which can dramatically mitigate ICI implicitly and noise explicitly, even without requiring any pilot. Compared with the standard Transformer, we come up with three creative designs. Firstly, we break the inner-encoder computation paradigm of the multi-head attention (MHA) in the standard Transformer, and design a brand new inter-encoder attention mechanism, referred to as the interactive MHA, which can significantly improve the performance, as well as accelerate the convergence rate. Secondly, to reduce the noise component attached to the received signal, we design an auto-perception denoising structure, which allows the network to learn the noise distribution in received signals. Thirdly, to better match the characteristics of DOFDM signals and selectively focus on the data at specified locations, we propose a trapezoidal positional encoding (PE), instead of adopting the original sine-cosine PE in the Transformer. Experimental results on both the realistic underwater channel and the simulation channel show that the TransDetector outperforms the classical $\mathcal{X}$-FFT algorithms and the DNNDetector in terms of the BER and the MSE. For example, the BER achieved by the TransDetector is reduced by $27.21\%$ and $12.50\%$ when the signal-to-noise ratio $(\text{SNR})=0$~dB and by $47.44\%$ and $33.49\%$ when $\text{SNR}=20$~dB against the PS-FFT and the DNNDetector based on the realistic channel, respectively.

{{</citation>}}


## q-bio.OT (1)



### (91/91) AI and Non AI Assessments for Dementia (Mahboobeh Parsapoor et al., 2023)

{{<citation>}}

Mahboobeh Parsapoor, Hamed Ghodrati, Vincenzo Dentamaro, Christopher R. Madan, Ioulietta Lazarou, Spiros Nikolopoulos, Ioannis Kompatsiaris. (2023)  
**AI and Non AI Assessments for Dementia**  

---
Primary Category: q-bio.OT  
Categories: cs-AI, cs-CY, q-bio-OT, q-bio.OT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.01210v1)  

---


**ABSTRACT**  
Current progress in the artificial intelligence domain has led to the development of various types of AI-powered dementia assessments, which can be employed to identify patients at the early stage of dementia. It can revolutionize the dementia care settings. It is essential that the medical community be aware of various AI assessments and choose them considering their degrees of validity, efficiency, practicality, reliability, and accuracy concerning the early identification of patients with dementia (PwD). On the other hand, AI developers should be informed about various non-AI assessments as well as recently developed AI assessments. Thus, this paper, which can be readable by both clinicians and AI engineers, fills the gap in the literature in explaining the existing solutions for the recognition of dementia to clinicians, as well as the techniques used and the most widespread dementia datasets to AI engineers. It follows a review of papers on AI and non-AI assessments for dementia to provide valuable information about various dementia assessments for both the AI and medical communities. The discussion and conclusion highlight the most prominent research directions and the maturity of existing solutions.

{{</citation>}}
