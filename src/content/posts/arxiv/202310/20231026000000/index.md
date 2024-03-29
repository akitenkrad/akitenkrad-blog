---
draft: false
title: "arXiv @ 2023.10.26"
date: 2023-10-26
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.26"
    identifier: arxiv_20231026
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (74)](#cscl-74)
- [cs.LG (31)](#cslg-31)
- [cs.CV (29)](#cscv-29)
- [eess.IV (4)](#eessiv-4)
- [cs.HC (5)](#cshc-5)
- [cs.CR (4)](#cscr-4)
- [cs.IR (5)](#csir-5)
- [stat.AP (1)](#statap-1)
- [cs.SD (3)](#cssd-3)
- [cs.AI (4)](#csai-4)
- [cs.SE (6)](#csse-6)
- [cs.CE (1)](#csce-1)
- [cs.PL (1)](#cspl-1)
- [cs.CY (1)](#cscy-1)
- [math.DS (1)](#mathds-1)
- [cs.SI (1)](#cssi-1)
- [cs.DC (1)](#csdc-1)
- [cs.IT (1)](#csit-1)
- [stat.ML (1)](#statml-1)
- [cs.RO (3)](#csro-3)
- [cs.MM (1)](#csmm-1)

## cs.CL (74)



### (1/178) GlotLID: Language Identification for Low-Resource Languages (Amir Hossein Kargaran et al., 2023)

{{<citation>}}

Amir Hossein Kargaran, Ayyoob Imani, François Yvon, Hinrich Schütze. (2023)  
**GlotLID: Language Identification for Low-Resource Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Identification, Low-Resource, NLP  
[Paper Link](http://arxiv.org/abs/2310.16248v1)  

---


**ABSTRACT**  
Several recent papers have published good solutions for language identification (LID) for about 300 high-resource and medium-resource languages. However, there is no LID available that (i) covers a wide range of low-resource languages, (ii) is rigorously evaluated and reliable and (iii) efficient and easy to use. Here, we publish GlotLID-M, an LID model that satisfies the desiderata of wide coverage, reliability and efficiency. It identifies 1665 languages, a large increase in coverage compared to prior work. In our experiments, GlotLID-M outperforms four baselines (CLD3, FT176, OpenLID and NLLB) when balancing F1 and false positive rate (FPR). We analyze the unique challenges that low-resource LID poses: incorrect corpus metadata, leakage from high-resource languages, difficulty separating closely related languages, handling of macrolanguage vs varieties and in general noisy data. We hope that integrating GlotLID-M into dataset creation pipelines will improve quality and enhance accessibility of NLP technology for low-resource languages and cultures. GlotLID-M model, code, and list of data sources are available: https://github.com/cisnlp/GlotLID.

{{</citation>}}


### (2/178) Mixture-of-Linguistic-Experts Adapters for Improving and Interpreting Pre-trained Language Models (Raymond Li et al., 2023)

{{<citation>}}

Raymond Li, Gabriel Murray, Giuseppe Carenini. (2023)  
**Mixture-of-Linguistic-Experts Adapters for Improving and Interpreting Pre-trained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.16240v1)  

---


**ABSTRACT**  
In this work, we propose a method that combines two popular research areas by injecting linguistic structures into pre-trained language models in the parameter-efficient fine-tuning (PEFT) setting. In our approach, parallel adapter modules encoding different linguistic structures are combined using a novel Mixture-of-Linguistic-Experts architecture, where Gumbel-Softmax gates are used to determine the importance of these modules at each layer of the model. To reduce the number of parameters, we first train the model for a fixed small number of steps before pruning the experts based on their importance scores. Our experiment results with three different pre-trained models show that our approach can outperform state-of-the-art PEFT methods with a comparable number of parameters. In addition, we provide additional analysis to examine the experts selected by each model at each layer to provide insights for future studies.

{{</citation>}}


### (3/178) CleanCoNLL: A Nearly Noise-Free Named Entity Recognition Dataset (Susanna Rücker et al., 2023)

{{<citation>}}

Susanna Rücker, Alan Akbik. (2023)  
**CleanCoNLL: A Nearly Noise-Free Named Entity Recognition Dataset**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2310.16225v1)  

---


**ABSTRACT**  
The CoNLL-03 corpus is arguably the most well-known and utilized benchmark dataset for named entity recognition (NER). However, prior works found significant numbers of annotation errors, incompleteness, and inconsistencies in the data. This poses challenges to objectively comparing NER approaches and analyzing their errors, as current state-of-the-art models achieve F1-scores that are comparable to or even exceed the estimated noise level in CoNLL-03. To address this issue, we present a comprehensive relabeling effort assisted by automatic consistency checking that corrects 7.0% of all labels in the English CoNLL-03. Our effort adds a layer of entity linking annotation both for better explainability of NER labels and as additional safeguard of annotation quality. Our experimental evaluation finds not only that state-of-the-art approaches reach significantly higher F1-scores (97.1%) on our data, but crucially that the share of correct predictions falsely counted as errors due to annotation noise drops from 47% to 6%. This indicates that our resource is well suited to analyze the remaining errors made by state-of-the-art models, and that the theoretical upper bound even on high resource, coarse-grained NER is not yet reached. To facilitate such analysis, we make CleanCoNLL publicly available to the research community.

{{</citation>}}


### (4/178) Knowledge Editing for Large Language Models: A Survey (Song Wang et al., 2023)

{{<citation>}}

Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, Jundong Li. (2023)  
**Knowledge Editing for Large Language Models: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.16218v2)  

---


**ABSTRACT**  
Large language models (LLMs) have recently transformed both the academic and industrial landscapes due to their remarkable capacity to understand, analyze, and generate texts based on their vast knowledge and reasoning ability. Nevertheless, one major drawback of LLMs is their substantial computational cost for pre-training due to their unprecedented amounts of parameters. The disadvantage is exacerbated when new knowledge frequently needs to be introduced into the pre-trained model. Therefore, it is imperative to develop effective and efficient techniques to update pre-trained LLMs. Traditional methods encode new knowledge in pre-trained LLMs through direct fine-tuning. However, naively re-training LLMs can be computationally intensive and risks degenerating valuable pre-trained knowledge irrelevant to the update in the model. Recently, Knowledge-based Model Editing (KME) has attracted increasing attention, which aims to precisely modify the LLMs to incorporate specific knowledge, without negatively influencing other irrelevant knowledge. In this survey, we aim to provide a comprehensive and in-depth overview of recent advances in the field of KME. We first introduce a general formulation of KME to encompass different KME strategies. Afterward, we provide an innovative taxonomy of KME techniques based on how the new knowledge is introduced into pre-trained LLMs, and investigate existing KME strategies while analyzing key insights, advantages, and limitations of methods from each category. Moreover, representative metrics, datasets, and applications of KME are introduced accordingly. Finally, we provide an in-depth analysis regarding the practicality and remaining challenges of KME and suggest promising research directions for further advancement in this field.

{{</citation>}}


### (5/178) Background Summarization of Event Timelines (Adithya Pratapa et al., 2023)

{{<citation>}}

Adithya Pratapa, Kevin Small, Markus Dreyer. (2023)  
**Background Summarization of Event Timelines**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Summarization, T5  
[Paper Link](http://arxiv.org/abs/2310.16197v1)  

---


**ABSTRACT**  
Generating concise summaries of news events is a challenging natural language processing task. While journalists often curate timelines to highlight key sub-events, newcomers to a news event face challenges in catching up on its historical context. In this paper, we address this need by introducing the task of background news summarization, which complements each timeline update with a background summary of relevant preceding events. We construct a dataset by merging existing timeline datasets and asking human annotators to write a background summary for each timestep of each news event. We establish strong baseline performance using state-of-the-art summarization systems and propose a query-focused variant to generate background summaries. To evaluate background summary quality, we present a question-answering-based evaluation metric, Background Utility Score (BUS), which measures the percentage of questions about a current event timestep that a background summary answers. Our experiments show the effectiveness of instruction fine-tuned systems such as Flan-T5, in addition to strong zero-shot performance using GPT-3.5.

{{</citation>}}


### (6/178) BLP 2023 Task 2: Sentiment Analysis (Md. Arid Hasan et al., 2023)

{{<citation>}}

Md. Arid Hasan, Firoj Alam, Anika Anjum, Shudipta Das, Afiyat Anjum. (2023)  
**BLP 2023 Task 2: Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2310.16183v1)  

---


**ABSTRACT**  
We present an overview of the BLP Sentiment Shared Task, organized as part of the inaugural BLP 2023 workshop, co-located with EMNLP 2023. The task is defined as the detection of sentiment in a given piece of social media text. This task attracted interest from 71 participants, among whom 29 and 30 teams submitted systems during the development and evaluation phases, respectively. In total, participants submitted 597 runs. However, a total of 15 teams submitted system description papers. The range of approaches in the submitted systems spans from classical machine learning models, fine-tuning pre-trained models, to leveraging Large Language Model (LLMs) in zero- and few-shot settings. In this paper, we provide a detailed account of the task setup, including dataset development and evaluation setup. Additionally, we provide a brief overview of the systems submitted by the participants. All datasets and evaluation scripts from the shared task have been made publicly available for the research community, to foster further research in this domain

{{</citation>}}


### (7/178) Correction with Backtracking Reduces Hallucination in Summarization (Zhenzhen Liu et al., 2023)

{{<citation>}}

Zhenzhen Liu, Chao Wan, Varsha Kishore, Jin Peng Zhou, Minmin Chen, Kilian Q. Weinberger. (2023)  
**Correction with Backtracking Reduces Hallucination in Summarization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2310.16176v1)  

---


**ABSTRACT**  
Abstractive summarization aims at generating natural language summaries of a source document that are succinct while preserving the important elements. Despite recent advances, neural text summarization models are known to be susceptible to hallucinating (or more correctly confabulating), that is to produce summaries with details that are not grounded in the source document. In this paper, we introduce a simple yet efficient technique, CoBa, to reduce hallucination in abstractive summarization. The approach is based on two steps: hallucination detection and mitigation. We show that the former can be achieved through measuring simple statistics about conditional word probabilities and distance to context words. Further, we demonstrate that straight-forward backtracking is surprisingly effective at mitigation. We thoroughly evaluate the proposed method with prior art on three benchmark datasets for text summarization. The results show that CoBa is effective and efficient in reducing hallucination, and offers great adaptability and flexibility.

{{</citation>}}


### (8/178) WojoodNER 2023: The First Arabic Named Entity Recognition Shared Task (Mustafa Jarrar et al., 2023)

{{<citation>}}

Mustafa Jarrar, Muhammad Abdul-Mageed, Mohammed Khalilia, Bashar Talafha, AbdelRahim Elmadany, Nagham Hamad, Alaa' Omar. (2023)  
**WojoodNER 2023: The First Arabic Named Entity Recognition Shared Task**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2310.16153v1)  

---


**ABSTRACT**  
We present WojoodNER-2023, the first Arabic Named Entity Recognition (NER) Shared Task. The primary focus of WojoodNER-2023 is on Arabic NER, offering novel NER datasets (i.e., Wojood) and the definition of subtasks designed to facilitate meaningful comparisons between different NER approaches. WojoodNER-2023 encompassed two Subtasks: FlatNER and NestedNER. A total of 45 unique teams registered for this shared task, with 11 of them actively participating in the test phase. Specifically, 11 teams participated in FlatNER, while $8$ teams tackled NestedNER. The winning teams achieved F1 scores of 91.96 and 93.73 in FlatNER and NestedNER, respectively.

{{</citation>}}


### (9/178) PreWoMe: Exploiting Presuppositions as Working Memory for Long Form Question Answering (Wookje Han et al., 2023)

{{<citation>}}

Wookje Han, Jinsol Park, Kyungjae Lee. (2023)  
**PreWoMe: Exploiting Presuppositions as Working Memory for Long Form Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.16147v1)  

---


**ABSTRACT**  
Information-seeking questions in long-form question answering (LFQA) often prove misleading due to ambiguity or false presupposition in the question. While many existing approaches handle misleading questions, they are tailored to limited questions, which are insufficient in a real-world setting with unpredictable input characteristics. In this work, we propose PreWoMe, a unified approach capable of handling any type of information-seeking question. The key idea of PreWoMe involves extracting presuppositions in the question and exploiting them as working memory to generate feedback and action about the question. Our experiment shows that PreWoMe is effective not only in tackling misleading questions but also in handling normal ones, thereby demonstrating the effectiveness of leveraging presuppositions, feedback, and action for real-world QA settings.

{{</citation>}}


### (10/178) A Language Model with Limited Memory Capacity Captures Interference in Human Sentence Processing (William Timkey et al., 2023)

{{<citation>}}

William Timkey, Tal Linzen. (2023)  
**A Language Model with Limited Memory Capacity Captures Interference in Human Sentence Processing**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.16142v1)  

---


**ABSTRACT**  
Two of the central factors believed to underpin human sentence processing difficulty are expectations and retrieval from working memory. A recent attempt to create a unified cognitive model integrating these two factors relied on the parallels between the self-attention mechanism of transformer language models and cue-based retrieval theories of working memory in human sentence processing (Ryu and Lewis 2021). While Ryu and Lewis show that attention patterns in specialized attention heads of GPT-2 are consistent with similarity-based interference, a key prediction of cue-based retrieval models, their method requires identifying syntactically specialized attention heads, and makes the cognitively implausible assumption that hundreds of memory retrieval operations take place in parallel. In the present work, we develop a recurrent neural language model with a single self-attention head, which more closely parallels the memory system assumed by cognitive theories. We show that our model's single attention head captures semantic and syntactic interference effects observed in human experiments.

{{</citation>}}


### (11/178) Can You Follow Me? Testing Situational Understanding in ChatGPT (Chenghao Yang et al., 2023)

{{<citation>}}

Chenghao Yang, Allyson Ettinger. (2023)  
**Can You Follow Me? Testing Situational Understanding in ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.16135v1)  

---


**ABSTRACT**  
Understanding sentence meanings and updating information states appropriately across time -- what we call "situational understanding" (SU) -- is a critical ability for human-like AI agents. SU is essential in particular for chat models, such as ChatGPT, to enable consistent, coherent, and effective dialogue between humans and AI. Previous works have identified certain SU limitations in non-chatbot Large Language models (LLMs), but the extent and causes of these limitations are not well understood, and capabilities of current chat-based models in this domain have not been explored. In this work we tackle these questions, proposing a novel synthetic environment for SU testing which allows us to do controlled and systematic testing of SU in chat-oriented models, through assessment of models' ability to track and enumerate environment states. Our environment also allows for close analysis of dynamics of model performance, to better understand underlying causes for performance patterns. We apply our test to ChatGPT, the state-of-the-art chatbot, and find that despite the fundamental simplicity of the task, the model's performance reflects an inability to retain correct environment states across time. Our follow-up analyses suggest that performance degradation is largely because ChatGPT has non-persistent in-context memory (although it can access the full dialogue history) and it is susceptible to hallucinated updates -- including updates that artificially inflate accuracies. Our findings suggest overall that ChatGPT is not currently equipped for robust tracking of situation states, and that trust in the impressive dialogue performance of ChatGPT comes with risks. We release the codebase for reproducing our test environment, as well as all prompts and API responses from ChatGPT, at https://github.com/yangalan123/SituationalTesting.

{{</citation>}}


### (12/178) GenKIE: Robust Generative Multimodal Document Key Information Extraction (Panfeng Cao et al., 2023)

{{<citation>}}

Panfeng Cao, Ye Wang, Qiang Zhang, Zaiqiao Meng. (2023)  
**GenKIE: Robust Generative Multimodal Document Key Information Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, OCR  
[Paper Link](http://arxiv.org/abs/2310.16131v1)  

---


**ABSTRACT**  
Key information extraction (KIE) from scanned documents has gained increasing attention because of its applications in various domains. Although promising results have been achieved by some recent KIE approaches, they are usually built based on discriminative models, which lack the ability to handle optical character recognition (OCR) errors and require laborious token-level labelling. In this paper, we propose a novel generative end-to-end model, named GenKIE, to address the KIE task. GenKIE is a sequence-to-sequence multimodal generative model that utilizes multimodal encoders to embed visual, layout and textual features and a decoder to generate the desired output. Well-designed prompts are leveraged to incorporate the label semantics as the weakly supervised signals and entice the generation of the key information. One notable advantage of the generative model is that it enables automatic correction of OCR errors. Besides, token-level granular annotation is not required. Extensive experiments on multiple public real-world datasets show that GenKIE effectively generalizes over different types of documents and achieves state-of-the-art results. Our experiments also validate the model's robustness against OCR errors, making GenKIE highly applicable in real-world scenarios.

{{</citation>}}


### (13/178) Octopus: A Multitask Model and Toolkit for Arabic Natural Language Generation (AbdelRahim Elmadany et al., 2023)

{{<citation>}}

AbdelRahim Elmadany, El Moatez Billah Nagoudi, Muhammad Abdul-Mageed. (2023)  
**Octopus: A Multitask Model and Toolkit for Arabic Natural Language Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Natural Language Generation, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2310.16127v1)  

---


**ABSTRACT**  
Understanding Arabic text and generating human-like responses is a challenging endeavor. While many researchers have proposed models and solutions for individual problems, there is an acute shortage of a comprehensive Arabic natural language generation toolkit that is capable of handling a wide range of tasks. In this work, we present a novel Arabic text-to-text Transformer model, namely AraT5v2. Our new model is methodically trained on extensive and diverse data, utilizing an extended sequence length of 2,048 tokens. We explore various pretraining strategies including unsupervised, supervised, and joint pertaining, under both single and multitask settings. Our models outperform competitive baselines with large margins. We take our work one step further by developing and publicly releasing Octopus, a Python-based package and command-line toolkit tailored for eight Arabic generation tasks all exploiting a single model. We release the models and the toolkit on our public repository.

{{</citation>}}


### (14/178) NADI 2023: The Fourth Nuanced Arabic Dialect Identification Shared Task (Muhammad Abdul-Mageed et al., 2023)

{{<citation>}}

Muhammad Abdul-Mageed, AbdelRahim Elmadany, Chiyu Zhang, El Moatez Billah Nagoudi, Houda Bouamor, Nizar Habash. (2023)  
**NADI 2023: The Fourth Nuanced Arabic Dialect Identification Shared Task**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.16117v1)  

---


**ABSTRACT**  
We describe the findings of the fourth Nuanced Arabic Dialect Identification Shared Task (NADI 2023). The objective of NADI is to help advance state-of-the-art Arabic NLP by creating opportunities for teams of researchers to collaboratively compete under standardized conditions. It does so with a focus on Arabic dialects, offering novel datasets and defining subtasks that allow for meaningful comparisons between different approaches. NADI 2023 targeted both dialect identification (Subtask 1) and dialect-to-MSA machine translation (Subtask 2 and Subtask 3). A total of 58 unique teams registered for the shared task, of whom 18 teams have participated (with 76 valid submissions during test phase). Among these, 16 teams participated in Subtask 1, 5 participated in Subtask 2, and 3 participated in Subtask 3. The winning teams achieved 87.27   F1 on Subtask 1, 14.76 Bleu in Subtask 2, and 21.10 Bleu in Subtask 3, respectively. Results show that all three subtasks remain challenging, thereby motivating future work in this area. We describe the methods employed by the participating teams and briefly offer an outlook for NADI.

{{</citation>}}


### (15/178) Locally Differentially Private Document Generation Using Zero Shot Prompting (Saiteja Utpala et al., 2023)

{{<citation>}}

Saiteja Utpala, Sara Hooker, Pin Yu Chen. (2023)  
**Locally Differentially Private Document Generation Using Zero Shot Prompting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.16111v1)  

---


**ABSTRACT**  
Numerous studies have highlighted the privacy risks associated with pretrained large language models. In contrast, our research offers a unique perspective by demonstrating that pretrained large language models can effectively contribute to privacy preservation. We propose a locally differentially private mechanism called DP-Prompt, which leverages the power of pretrained large language models and zero-shot prompting to counter author de-anonymization attacks while minimizing the impact on downstream utility. When DP-Prompt is used with a powerful language model like ChatGPT (gpt-3.5), we observe a notable reduction in the success rate of de-anonymization attacks, showing that it surpasses existing approaches by a considerable margin despite its simpler design. For instance, in the case of the IMDB dataset, DP-Prompt (with ChatGPT) perfectly recovers the clean sentiment F1 score while achieving a 46\% reduction in author identification F1 score against static attackers and a 26\% reduction against adaptive attackers. We conduct extensive experiments across six open-source large language models, ranging up to 7 billion parameters, to analyze various effects of the privacy-utility tradeoff.

{{</citation>}}


### (16/178) CR-COPEC: Causal Rationale of Corporate Performance Changes to Learn from Financial Reports (Ye Eun Chun et al., 2023)

{{<citation>}}

Ye Eun Chun, Sunjae Kwon, Kyunghwan Sohn, Nakwon Sung, Junyoup Lee, Byungki Seo, Kevin Compher, Seung-won Hwang, Jaesik Choi. (2023)  
**CR-COPEC: Causal Rationale of Corporate Performance Changes to Learn from Financial Reports**  

---
Primary Category: cs.CL  
Categories: cs-CE, cs-CL, cs.CL  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2310.16095v1)  

---


**ABSTRACT**  
In this paper, we introduce CR-COPEC called Causal Rationale of Corporate Performance Changes from financial reports. This is a comprehensive large-scale domain-adaptation causal sentence dataset to detect financial performance changes of corporate. CR-COPEC contributes to two major achievements. First, it detects causal rationale from 10-K annual reports of the U.S. companies, which contain experts' causal analysis following accounting standards in a formal manner. This dataset can be widely used by both individual investors and analysts as material information resources for investing and decision making without tremendous effort to read through all the documents. Second, it carefully considers different characteristics which affect the financial performance of companies in twelve industries. As a result, CR-COPEC can distinguish causal sentences in various industries by taking unique narratives in each industry into consideration. We also provide an extensive analysis of how well CR-COPEC dataset is constructed and suited for classifying target sentences as causal ones with respect to industry characteristics. Our dataset and experimental codes are publicly available.

{{</citation>}}


### (17/178) MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning (Zayne Sprague et al., 2023)

{{<citation>}}

Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, Greg Durrett. (2023)  
**MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.16049v1)  

---


**ABSTRACT**  
While large language models (LLMs) equipped with techniques like chain-of-thought prompting have demonstrated impressive capabilities, they still fall short in their ability to reason robustly in complex settings. However, evaluating LLM reasoning is challenging because system capabilities continue to grow while benchmark datasets for tasks like logical deduction have remained static. We introduce MuSR, a dataset for evaluating language models on multistep soft reasoning tasks specified in a natural language narrative. This dataset has two crucial features. First, it is created through a novel neurosymbolic synthetic-to-natural generation algorithm, enabling the construction of complex reasoning instances that challenge GPT-4 (e.g., murder mysteries roughly 1000 words in length) and which can be scaled further as more capable LLMs are released. Second, our dataset instances are free text narratives corresponding to real-world domains of reasoning; this makes it simultaneously much more challenging than other synthetically-crafted benchmarks while remaining realistic and tractable for human annotators to solve with high accuracy. We evaluate a range of LLMs and prompting techniques on this dataset and characterize the gaps that remain for techniques like chain-of-thought to perform robust reasoning.

{{</citation>}}


### (18/178) WebWISE: Web Interface Control and Sequential Exploration with Large Language Models (Heyi Tao et al., 2023)

{{<citation>}}

Heyi Tao, Sethuraman T V, Michal Shlapentokh-Rothman, Derek Hoiem. (2023)  
**WebWISE: Web Interface Control and Sequential Exploration with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.16042v2)  

---


**ABSTRACT**  
The paper investigates using a Large Language Model (LLM) to automatically perform web software tasks using click, scroll, and text input operations. Previous approaches, such as reinforcement learning (RL) or imitation learning, are inefficient to train and task-specific. Our method uses filtered Document Object Model (DOM) elements as observations and performs tasks step-by-step, sequentially generating small programs based on the current observations. We use in-context learning, either benefiting from a single manually provided example, or an automatically generated example based on a successful zero-shot trial. We evaluate the proposed method on the MiniWob++ benchmark. With only one in-context example, our WebWISE method achieves similar or better performance than other methods that require many demonstrations or trials.

{{</citation>}}


### (19/178) Instruct and Extract: Instruction Tuning for On-Demand Information Extraction (Yizhu Jiao et al., 2023)

{{<citation>}}

Yizhu Jiao, Ming Zhong, Sha Li, Ruining Zhao, Siru Ouyang, Heng Ji, Jiawei Han. (2023)  
**Instruct and Extract: Instruction Tuning for On-Demand Information Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2310.16040v1)  

---


**ABSTRACT**  
Large language models with instruction-following capabilities open the door to a wider group of users. However, when it comes to information extraction - a classic task in natural language processing - most task-specific systems cannot align well with long-tail ad hoc extraction use cases for non-expert users. To address this, we propose a novel paradigm, termed On-Demand Information Extraction, to fulfill the personalized demands of real-world users. Our task aims to follow the instructions to extract the desired content from the associated text and present it in a structured tabular format. The table headers can either be user-specified or inferred contextually by the model. To facilitate research in this emerging area, we present a benchmark named InstructIE, inclusive of both automatically generated training data, as well as the human-annotated test set. Building on InstructIE, we further develop an On-Demand Information Extractor, ODIE. Comprehensive evaluations on our benchmark reveal that ODIE substantially outperforms the existing open-source models of similar size. Our code and dataset are released on https://github.com/yzjiao/On-Demand-IE.

{{</citation>}}


### (20/178) Dissecting In-Context Learning of Translations in GPTs (Vikas Raunak et al., 2023)

{{<citation>}}

Vikas Raunak, Hany Hassan Awadalla, Arul Menezes. (2023)  
**Dissecting In-Context Learning of Translations in GPTs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model, Machine Translation, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.15987v1)  

---


**ABSTRACT**  
Most of the recent work in leveraging Large Language Models (LLMs) such as GPT-3 for Machine Translation (MT) has focused on selecting the few-shot samples for prompting. In this work, we try to better understand the role of demonstration attributes for the in-context learning of translations through perturbations of high-quality, in-domain demonstrations. We find that asymmetric perturbation of the source-target mappings yield vastly different results. We show that the perturbation of the source side has surprisingly little impact, while target perturbation can drastically reduce translation quality, suggesting that it is the output text distribution that provides the most important learning signal during in-context learning of translations. We propose a method named Zero-Shot-Context to add this signal automatically in Zero-Shot prompting. We demonstrate that it improves upon the zero-shot translation performance of GPT-3, even making it competitive with few-shot prompted translations.

{{</citation>}}


### (21/178) Accented Speech Recognition With Accent-specific Codebooks (Darshan Prabhu et al., 2023)

{{<citation>}}

Darshan Prabhu, Preethi Jyothi, Sriram Ganapathy, Vinit Unni. (2023)  
**Accented Speech Recognition With Accent-specific Codebooks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.15970v3)  

---


**ABSTRACT**  
Speech accents pose a significant challenge to state-of-the-art automatic speech recognition (ASR) systems. Degradation in performance across underrepresented accents is a severe deterrent to the inclusive adoption of ASR. In this work, we propose a novel accent adaptation approach for end-to-end ASR systems using cross-attention with a trainable set of codebooks. These learnable codebooks capture accent-specific information and are integrated within the ASR encoder layers. The model is trained on accented English speech, while the test data also contained accents which were not seen during training. On the Mozilla Common Voice multi-accented dataset, we show that our proposed approach yields significant performance gains not only on the seen English accents (up to $37\%$ relative improvement in word error rate) but also on the unseen accents (up to $5\%$ relative improvement in WER). Further, we illustrate benefits for a zero-shot transfer setup on the L2Artic dataset. We also compare the performance with other approaches based on accent adversarial training.

{{</citation>}}


### (22/178) Mixture of Tokens: Efficient LLMs through Cross-Example Aggregation (Szymon Antoniak et al., 2023)

{{<citation>}}

Szymon Antoniak, Sebastian Jaszczur, Michał Krutul, Maciej Pióro, Jakub Krajewski, Jan Ludziejewski, Tomasz Odrzygóźdź, Marek Cygan. (2023)  
**Mixture of Tokens: Efficient LLMs through Cross-Example Aggregation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.15961v1)  

---


**ABSTRACT**  
Despite the promise of Mixture of Experts (MoE) models in increasing parameter counts of Transformer models while maintaining training and inference costs, their application carries notable drawbacks. The key strategy of these models is to, for each processed token, activate at most a few experts - subsets of an extensive feed-forward layer. But this approach is not without its challenges. The operation of matching experts and tokens is discrete, which makes MoE models prone to issues like training instability and uneven expert utilization. Existing techniques designed to address these concerns, such as auxiliary losses or balance-aware matching, result either in lower model performance or are more difficult to train. In response to these issues, we propose Mixture of Tokens, a fully-differentiable model that retains the benefits of MoE architectures while avoiding the aforementioned difficulties. Rather than routing tokens to experts, this approach mixes tokens from different examples prior to feeding them to experts, enabling the model to learn from all token-expert combinations. Importantly, this mixing can be disabled to avoid mixing of different sequences during inference. Crucially, this method is fully compatible with both masked and causal Large Language Model training and inference.

{{</citation>}}


### (23/178) NoteChat: A Dataset of Synthetic Doctor-Patient Conversations Conditioned on Clinical Notes (Junda Wang et al., 2023)

{{<citation>}}

Junda Wang, Zonghai Yao, Zhichao Yang, Huixue Zhou, Rumeng Li, Xun Wang, Yucheng Xu, Hong Yu. (2023)  
**NoteChat: A Dataset of Synthetic Doctor-Patient Conversations Conditioned on Clinical Notes**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, Clinical, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.15959v1)  

---


**ABSTRACT**  
The detailed clinical records drafted by doctors after each patient's visit are crucial for medical practitioners and researchers. Automating the creation of these notes with language models can reduce the workload of doctors. However, training such models can be difficult due to the limited public availability of conversations between patients and doctors. In this paper, we introduce NoteChat, a cooperative multi-agent framework leveraging Large Language Models (LLMs) for generating synthetic doctor-patient conversations conditioned on clinical notes. NoteChat consists of Planning, Roleplay, and Polish modules. We provide a comprehensive automatic and human evaluation of NoteChat, comparing it with state-of-the-art models, including OpenAI's ChatGPT and GPT-4. Results demonstrate that NoteChat facilitates high-quality synthetic doctor-patient conversations, underscoring the untapped potential of LLMs in healthcare. This work represents the first instance of multiple LLMs cooperating to complete a doctor-patient conversation conditioned on clinical notes, offering promising avenues for the intersection of AI and healthcare

{{</citation>}}


### (24/178) This is not a Dataset: A Large Negation Benchmark to Challenge Large Language Models (Iker García-Ferrero et al., 2023)

{{<citation>}}

Iker García-Ferrero, Begoña Altuna, Javier Álvez, Itziar Gonzalez-Dios, German Rigau. (2023)  
**This is not a Dataset: A Large Negation Benchmark to Challenge Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.15941v1)  

---


**ABSTRACT**  
Although large language models (LLMs) have apparently acquired a certain level of grammatical knowledge and the ability to make generalizations, they fail to interpret negation, a crucial step in Natural Language Processing. We try to clarify the reasons for the sub-optimal performance of LLMs understanding negation. We introduce a large semi-automatically generated dataset of circa 400,000 descriptive sentences about commonsense knowledge that can be true or false in which negation is present in about 2/3 of the corpus in different forms. We have used our dataset with the largest available open LLMs in a zero-shot approach to grasp their generalization and inference capability and we have also fine-tuned some of the models to assess whether the understanding of negation can be trained. Our findings show that, while LLMs are proficient at classifying affirmative sentences, they struggle with negative sentences and lack a deep understanding of negation, often relying on superficial cues. Although fine-tuning the models on negative sentences improves their performance, the lack of generalization in handling negation is persistent, highlighting the ongoing challenges of LLMs regarding negation understanding and generalization. The dataset and code are publicly available.

{{</citation>}}


### (25/178) Contrastive Learning-based Sentence Encoders Implicitly Weight Informative Words (Hiroto Kurita et al., 2023)

{{<citation>}}

Hiroto Kurita, Goro Kobayashi, Sho Yokoi, Kentaro Inui. (2023)  
**Contrastive Learning-based Sentence Encoders Implicitly Weight Informative Words**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.15921v1)  

---


**ABSTRACT**  
The performance of sentence encoders can be significantly improved through the simple practice of fine-tuning using contrastive loss. A natural question arises: what characteristics do models acquire during contrastive learning? This paper theoretically and experimentally shows that contrastive-based sentence encoders implicitly weight words based on information-theoretic quantities; that is, more informative words receive greater weight, while others receive less. The theory states that, in the lower bound of the optimal value of the contrastive learning objective, the norm of word embedding reflects the information gain associated with the distribution of surrounding words. We also conduct comprehensive experiments using various models, multiple datasets, two methods to measure the implicit weighting of models (Integrated Gradients and SHAP), and two information-theoretic quantities (information gain and self-information). The results provide empirical evidence that contrastive fine-tuning emphasizes informative words.

{{</citation>}}


### (26/178) In-Context Learning Creates Task Vectors (Roee Hendel et al., 2023)

{{<citation>}}

Roee Hendel, Mor Geva, Amir Globerson. (2023)  
**In-Context Learning Creates Task Vectors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15916v1)  

---


**ABSTRACT**  
In-context learning (ICL) in Large Language Models (LLMs) has emerged as a powerful new learning paradigm. However, its underlying mechanism is still not well understood. In particular, it is challenging to map it to the "standard" machine learning framework, where one uses a training set $S$ to find a best-fitting function $f(x)$ in some hypothesis class. Here we make progress on this problem by showing that the functions learned by ICL often have a very simple structure: they correspond to the transformer LLM whose only inputs are the query $x$ and a single "task vector" calculated from the training set. Thus, ICL can be seen as compressing $S$ into a single task vector $\boldsymbol{\theta}(S)$ and then using this task vector to modulate the transformer to produce the output. We support the above claim via comprehensive experiments across a range of models and tasks.

{{</citation>}}


### (27/178) Characterizing Mechanisms for Factual Recall in Language Models (Qinan Yu et al., 2023)

{{<citation>}}

Qinan Yu, Jack Merullo, Ellie Pavlick. (2023)  
**Characterizing Mechanisms for Factual Recall in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.15910v1)  

---


**ABSTRACT**  
Language Models (LMs) often must integrate facts they memorized in pretraining with new information that appears in a given context. These two sources can disagree, causing competition within the model, and it is unclear how an LM will resolve the conflict. On a dataset that queries for knowledge of world capitals, we investigate both distributional and mechanistic determinants of LM behavior in such situations. Specifically, we measure the proportion of the time an LM will use a counterfactual prefix (e.g., "The capital of Poland is London") to overwrite what it learned in pretraining ("Warsaw"). On Pythia and GPT2, the training frequency of both the query country ("Poland") and the in-context city ("London") highly affect the models' likelihood of using the counterfactual. We then use head attribution to identify individual attention heads that either promote the memorized answer or the in-context answer in the logits. By scaling up or down the value vector of these heads, we can control the likelihood of using the in-context answer on new data. This method can increase the rate of generating the in-context answer to 88\% of the time simply by scaling a single head at runtime. Our work contributes to a body of evidence showing that we can often localize model behaviors to specific components and provides a proof of concept for how future methods might control model behavior dynamically at runtime.

{{</citation>}}


### (28/178) Is Probing All You Need? Indicator Tasks as an Alternative to Probing Embedding Spaces (Tal Levy et al., 2023)

{{<citation>}}

Tal Levy, Omer Goldman, Reut Tsarfaty. (2023)  
**Is Probing All You Need? Indicator Tasks as an Alternative to Probing Embedding Spaces**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.15905v1)  

---


**ABSTRACT**  
The ability to identify and control different kinds of linguistic information encoded in vector representations of words has many use cases, especially for explainability and bias removal. This is usually done via a set of simple classification tasks, termed probes, to evaluate the information encoded in the embedding space. However, the involvement of a trainable classifier leads to entanglement between the probe's results and the classifier's nature. As a result, contemporary works on probing include tasks that do not involve training of auxiliary models. In this work we introduce the term indicator tasks for non-trainable tasks which are used to query embedding spaces for the existence of certain properties, and claim that this kind of tasks may point to a direction opposite to probes, and that this contradiction complicates the decision on whether a property exists in an embedding space. We demonstrate our claims with two test cases, one dealing with gender debiasing and another with the erasure of morphological information from embedding spaces. We show that the application of a suitable indicator provides a more accurate picture of the information captured and removed compared to probes. We thus conclude that indicator tasks should be implemented and taken into consideration when eliciting information from embedded representations.

{{</citation>}}


### (29/178) Do Stochastic Parrots have Feelings Too? Improving Neural Detection of Synthetic Text via Emotion Recognition (Alan Cowap et al., 2023)

{{<citation>}}

Alan Cowap, Yvette Graham, Jennifer Foster. (2023)  
**Do Stochastic Parrots have Feelings Too? Improving Neural Detection of Synthetic Text via Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, ChatGPT, Emotion Recognition, GPT  
[Paper Link](http://arxiv.org/abs/2310.15904v1)  

---


**ABSTRACT**  
Recent developments in generative AI have shone a spotlight on high-performance synthetic text generation technologies. The now wide availability and ease of use of such models highlights the urgent need to provide equally powerful technologies capable of identifying synthetic text. With this in mind, we draw inspiration from psychological studies which suggest that people can be driven by emotion and encode emotion in the text they compose. We hypothesize that pretrained language models (PLMs) have an affective deficit because they lack such an emotional driver when generating text and consequently may generate synthetic text which has affective incoherence i.e. lacking the kind of emotional coherence present in human-authored text. We subsequently develop an emotionally aware detector by fine-tuning a PLM on emotion. Experiment results indicate that our emotionally-aware detector achieves improvements across a range of synthetic text generators, various sized models, datasets, and domains. Finally, we compare our emotionally-aware synthetic text detector to ChatGPT in the task of identification of its own output and show substantial gains, reinforcing the potential of emotion as a signal to identify synthetic text. Code, models, and datasets are available at https: //github.com/alanagiasi/emoPLMsynth

{{</citation>}}


### (30/178) BianQue: Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT (Yirong Chen et al., 2023)

{{<citation>}}

Yirong Chen, Zhenyu Wang, Xiaofen Xing, huimin zheng, Zhipei Xu, Kai Fang, Junhong Wang, Sihang Li, Jieling Wu, Qi Liu, Xiangmin Xu. (2023)  
**BianQue: Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: ChatGPT, GLM, GPT  
[Paper Link](http://arxiv.org/abs/2310.15896v1)  

---


**ABSTRACT**  
Large language models (LLMs) have performed well in providing general and extensive health suggestions in single-turn conversations, exemplified by systems such as ChatGPT, ChatGLM, ChatDoctor, DoctorGLM, and etc. However, the limited information provided by users during single turn results in inadequate personalization and targeting of the generated suggestions, which requires users to independently select the useful part. It is mainly caused by the missing ability to engage in multi-turn questioning. In real-world medical consultations, doctors usually employ a series of iterative inquiries to comprehend the patient's condition thoroughly, enabling them to provide effective and personalized suggestions subsequently, which can be defined as chain of questioning (CoQ) for LLMs. To improve the CoQ of LLMs, we propose BianQue, a ChatGLM-based LLM finetuned with the self-constructed health conversation dataset BianQueCorpus that is consist of multiple turns of questioning and health suggestions polished by ChatGPT. Experimental results demonstrate that the proposed BianQue can simultaneously balance the capabilities of both questioning and health suggestions, which will help promote the research and application of LLMs in the field of proactive health.

{{</citation>}}


### (31/178) Using Artificial French Data to Understand the Emergence of Gender Bias in Transformer Language Models (Lina Conti et al., 2023)

{{<citation>}}

Lina Conti, Guillaume Wisniewski. (2023)  
**Using Artificial French Data to Understand the Emergence of Gender Bias in Transformer Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.15852v1)  

---


**ABSTRACT**  
Numerous studies have demonstrated the ability of neural language models to learn various linguistic properties without direct supervision. This work takes an initial step towards exploring the less researched topic of how neural models discover linguistic properties of words, such as gender, as well as the rules governing their usage. We propose to use an artificial corpus generated by a PCFG based on French to precisely control the gender distribution in the training data and determine under which conditions a model correctly captures gender information or, on the contrary, appears gender-biased.

{{</citation>}}


### (32/178) Self-Guard: Empower the LLM to Safeguard Itself (Zezhong Wang et al., 2023)

{{<citation>}}

Zezhong Wang, Fangkai Yang, Lu Wang, Pu Zhao, Hongru Wang, Liang Chen, Qingwei Lin, Kam-Fai Wong. (2023)  
**Self-Guard: Empower the LLM to Safeguard Itself**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15851v1)  

---


**ABSTRACT**  
The jailbreak attack can bypass the safety measures of a Large Language Model (LLM), generating harmful content. This misuse of LLM has led to negative societal consequences. Currently, there are two main approaches to address jailbreak attacks: safety training and safeguards. Safety training focuses on further training LLM to enhance its safety. On the other hand, safeguards involve implementing external models or filters to prevent harmful outputs. However, safety training has constraints in its ability to adapt to new attack types and often leads to a drop in model performance. Safeguards have proven to be of limited help. To tackle these issues, we propose a novel approach called Self-Guard, which combines the strengths of both safety methods. Self-Guard includes two stages. In the first stage, we enhance the model's ability to assess harmful content, and in the second stage, we instruct the model to consistently perform harmful content detection on its own responses. The experiment has demonstrated that Self-Guard is robust against jailbreak attacks. In the bad case analysis, we find that LLM occasionally provides harmless responses to harmful queries. Additionally, we evaluated the general capabilities of the LLM before and after safety training, providing evidence that Self-Guard does not result in the LLM's performance degradation. In sensitivity tests, Self-Guard not only avoids inducing over-sensitivity in LLM but also can even mitigate this issue.

{{</citation>}}


### (33/178) Rosetta Stone at KSAA-RD Shared Task: A Hop From Language Modeling To Word--Definition Alignment (Ahmed ElBakry et al., 2023)

{{<citation>}}

Ahmed ElBakry, Mohamed Gabr, Muhammad ElNokrashy, Badr AlKhamissi. (2023)  
**Rosetta Stone at KSAA-RD Shared Task: A Hop From Language Modeling To Word--Definition Alignment**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.15823v1)  

---


**ABSTRACT**  
A Reverse Dictionary is a tool enabling users to discover a word based on its provided definition, meaning, or description. Such a technique proves valuable in various scenarios, aiding language learners who possess a description of a word without its identity, and benefiting writers seeking precise terminology. These scenarios often encapsulate what is referred to as the "Tip-of-the-Tongue" (TOT) phenomena. In this work, we present our winning solution for the Arabic Reverse Dictionary shared task. This task focuses on deriving a vector representation of an Arabic word from its accompanying description. The shared task encompasses two distinct subtasks: the first involves an Arabic definition as input, while the second employs an English definition. For the first subtask, our approach relies on an ensemble of finetuned Arabic BERT-based models, predicting the word embedding for a given definition. The final representation is obtained through averaging the output embeddings from each model within the ensemble. In contrast, the most effective solution for the second subtask involves translating the English test definitions into Arabic and applying them to the finetuned models originally trained for the first subtask. This straightforward method achieves the highest score across both subtasks.

{{</citation>}}


### (34/178) Generative Language Models Exhibit Social Identity Biases (Tiancheng Hu et al., 2023)

{{<citation>}}

Tiancheng Hu, Yara Kyrychenko, Steve Rathje, Nigel Collier, Sander van der Linden, Jon Roozenbeek. (2023)  
**Generative Language Models Exhibit Social Identity Biases**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2310.15819v1)  

---


**ABSTRACT**  
The surge in popularity of large language models has given rise to concerns about biases that these models could learn from humans. In this study, we investigate whether ingroup solidarity and outgroup hostility, fundamental social biases known from social science, are present in 51 large language models. We find that almost all foundational language models and some instruction fine-tuned models exhibit clear ingroup-positive and outgroup-negative biases when prompted to complete sentences (e.g., "We are..."). A comparison of LLM-generated sentences with human-written sentences on the internet reveals that these models exhibit similar level, if not greater, levels of bias than human text. To investigate where these biases stem from, we experimentally varied the amount of ingroup-positive or outgroup-negative sentences the model was exposed to during fine-tuning in the context of the United States Democrat-Republican divide. Doing so resulted in the models exhibiting a marked increase in ingroup solidarity and an even greater increase in outgroup hostility. Furthermore, removing either ingroup-positive or outgroup-negative sentences (or both) from the fine-tuning data leads to a significant reduction in both ingroup solidarity and outgroup hostility, suggesting that biases can be reduced by removing biased training data. Our findings suggest that modern language models exhibit fundamental social identity biases and that such biases can be mitigated by curating training data. Our results have practical implications for creating less biased large-language models and further underscore the need for more research into user interactions with LLMs to prevent potential bias reinforcement in humans.

{{</citation>}}


### (35/178) DALE: Generative Data Augmentation for Low-Resource Legal NLP (Sreyan Ghosh et al., 2023)

{{<citation>}}

Sreyan Ghosh, Chandra Kiran Evuru, Sonal Kumar, S Ramaneswaran, S Sakshi, Utkarsh Tyagi, Dinesh Manocha. (2023)  
**DALE: Generative Data Augmentation for Low-Resource Legal NLP**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation, Language Model, Legal, Low-Resource, NLP  
[Paper Link](http://arxiv.org/abs/2310.15799v1)  

---


**ABSTRACT**  
We present DALE, a novel and effective generative Data Augmentation framework for low-resource LEgal NLP. DALE addresses the challenges existing frameworks pose in generating effective data augmentations of legal documents - legal language, with its specialized vocabulary and complex semantics, morphology, and syntax, does not benefit from data augmentations that merely rephrase the source sentence. To address this, DALE, built on an Encoder-Decoder Language Model, is pre-trained on a novel unsupervised text denoising objective based on selective masking - our masking strategy exploits the domain-specific language characteristics of templatized legal documents to mask collocated spans of text. Denoising these spans helps DALE acquire knowledge about legal concepts, principles, and language usage. Consequently, it develops the ability to generate coherent and diverse augmentations with novel contexts. Finally, DALE performs conditional generation to generate synthetic augmentations for low-resource Legal NLP tasks. We demonstrate the effectiveness of DALE on 13 datasets spanning 6 tasks and 4 low-resource settings. DALE outperforms all our baselines, including LLMs, qualitatively and quantitatively, with improvements of 1%-50%.

{{</citation>}}


### (36/178) MindLLM: Pre-training Lightweight Large Language Model from Scratch, Evaluations and Domain Applications (Yizhe Yang et al., 2023)

{{<citation>}}

Yizhe Yang, Huashan Sun, Jiawei Li, Runheng Liu, Yinghao Li, Yuhang Liu, Heyan Huang, Yang Gao. (2023)  
**MindLLM: Pre-training Lightweight Large Language Model from Scratch, Evaluations and Domain Applications**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15777v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable performance across various natural language tasks, marking significant strides towards general artificial intelligence. While general artificial intelligence is leveraged by developing increasingly large-scale models, there could be another branch to develop lightweight custom models that better serve certain domains, taking into account the high cost of training and deploying LLMs and the scarcity of resources. In this paper, we present MindLLM, a novel series of bilingual lightweight large language models, trained from scratch, alleviating such burdens by offering models with 1.3 billion and 3 billion parameters. A thorough account of experiences accrued during large model development is given, covering every step of the process, including data construction, model architecture, evaluation, and applications. Such insights are hopefully valuable for fellow academics and developers. MindLLM consistently matches or surpasses the performance of other open-source larger models on some public benchmarks. We also introduce an innovative instruction tuning framework tailored for smaller models to enhance their capabilities efficiently. Moreover, we explore the application of MindLLM in specific vertical domains such as law and finance, underscoring the agility and adaptability of our lightweight models.

{{</citation>}}


### (37/178) BLESS: Benchmarking Large Language Models on Sentence Simplification (Tannon Kew et al., 2023)

{{<citation>}}

Tannon Kew, Alison Chi, Laura Vásquez-Rodríguez, Sweta Agrawal, Dennis Aumiller, Fernando Alva-Manchego, Matthew Shardlow. (2023)  
**BLESS: Benchmarking Large Language Models on Sentence Simplification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15773v1)  

---


**ABSTRACT**  
We present BLESS, a comprehensive performance benchmark of the most recent state-of-the-art large language models (LLMs) on the task of text simplification (TS). We examine how well off-the-shelf LLMs can solve this challenging task, assessing a total of 44 models, differing in size, architecture, pre-training methods, and accessibility, on three test sets from different domains (Wikipedia, news, and medical) under a few-shot setting. Our analysis considers a suite of automatic metrics as well as a large-scale quantitative investigation into the types of common edit operations performed by the different models. Furthermore, we perform a manual qualitative analysis on a subset of model outputs to better gauge the quality of the generated simplifications. Our evaluation indicates that the best LLMs, despite not being trained on TS, perform comparably with state-of-the-art TS baselines. Additionally, we find that certain LLMs demonstrate a greater range and diversity of edit operations. Our performance benchmark will be available as a resource for the development of future TS methods and evaluation metrics.

{{</citation>}}


### (38/178) Learning From Free-Text Human Feedback -- Collect New Datasets Or Extend Existing Ones? (Dominic Petrak et al., 2023)

{{<citation>}}

Dominic Petrak, Nafise Sadat Moosavi, Ye Tian, Nikolai Rozanov, Iryna Gurevych. (2023)  
**Learning From Free-Text Human Feedback -- Collect New Datasets Or Extend Existing Ones?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, T5  
[Paper Link](http://arxiv.org/abs/2310.15758v1)  

---


**ABSTRACT**  
Learning from free-text human feedback is essential for dialog systems, but annotated data is scarce and usually covers only a small fraction of error types known in conversational AI. Instead of collecting and annotating new datasets from scratch, recent advances in synthetic dialog generation could be used to augment existing dialog datasets with the necessary annotations. However, to assess the feasibility of such an effort, it is important to know the types and frequency of free-text human feedback included in these datasets. In this work, we investigate this question for a variety of commonly used dialog datasets, including MultiWoZ, SGD, BABI, PersonaChat, Wizards-of-Wikipedia, and the human-bot split of the Self-Feeding Chatbot. Using our observations, we derive new taxonomies for the annotation of free-text human feedback in dialogs and investigate the impact of including such data in response generation for three SOTA language generation models, including GPT-2, LLAMA, and Flan-T5. Our findings provide new insights into the composition of the datasets examined, including error types, user response types, and the relations between them.

{{</citation>}}


### (39/178) Integrating Language Models into Direct Speech Translation: An Inference-Time Solution to Control Gender Inflection (Dennis Fucci et al., 2023)

{{<citation>}}

Dennis Fucci, Marco Gaido, Sara Papi, Mauro Cettolo, Matteo Negri, Luisa Bentivogli. (2023)  
**Integrating Language Models into Direct Speech Translation: An Inference-Time Solution to Control Gender Inflection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15752v1)  

---


**ABSTRACT**  
When translating words referring to the speaker, speech translation (ST) systems should not resort to default masculine generics nor rely on potentially misleading vocal traits. Rather, they should assign gender according to the speakers' preference. The existing solutions to do so, though effective, are hardly feasible in practice as they involve dedicated model re-training on gender-labeled ST data. To overcome these limitations, we propose the first inference-time solution to control speaker-related gender inflections in ST. Our approach partially replaces the (biased) internal language model (LM) implicitly learned by the ST decoder with gender-specific external LMs. Experiments on en->es/fr/it show that our solution outperforms the base models and the best training-time mitigation strategy by up to 31.0 and 1.6 points in gender accuracy, respectively, for feminine forms. The gains are even larger (up to 32.0 and 3.4) in the challenging condition where speakers' vocal traits conflict with their gender.

{{</citation>}}


### (40/178) Failures Pave the Way: Enhancing Large Language Models through Tuning-free Rule Accumulation (Zeyuan Yang et al., 2023)

{{<citation>}}

Zeyuan Yang, Peng Li, Yang Liu. (2023)  
**Failures Pave the Way: Enhancing Large Language Models through Tuning-free Rule Accumulation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15746v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have showcased impressive performance. However, due to their inability to capture relationships among samples, these frozen LLMs inevitably keep repeating similar mistakes. In this work, we propose our Tuning-free Rule Accumulation (TRAN) framework, which guides LLMs in improving their performance by learning from previous mistakes. Considering data arrives sequentially, LLMs gradually accumulate rules from incorrect cases, forming a rule collection. These rules are then utilized by the LLMs to avoid making similar mistakes when processing subsequent inputs. Moreover, the rules remain independent of the primary prompts, seamlessly complementing prompt design strategies. Experimentally, we show that TRAN improves over recent baselines by a large margin.

{{</citation>}}


### (41/178) RAPL: A Relation-Aware Prototype Learning Approach for Few-Shot Document-Level Relation Extraction (Shiao Meng et al., 2023)

{{<citation>}}

Shiao Meng, Xuming Hu, Aiwei Liu, Shu'ang Li, Fukun Ma, Yawen Yang, Lijie Wen. (2023)  
**RAPL: A Relation-Aware Prototype Learning Approach for Few-Shot Document-Level Relation Extraction**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-CL, cs.CL  
Keywords: Few-Shot, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2310.15743v1)  

---


**ABSTRACT**  
How to identify semantic relations among entities in a document when only a few labeled documents are available? Few-shot document-level relation extraction (FSDLRE) is crucial for addressing the pervasive data scarcity problem in real-world scenarios. Metric-based meta-learning is an effective framework widely adopted for FSDLRE, which constructs class prototypes for classification. However, existing works often struggle to obtain class prototypes with accurate relational semantics: 1) To build prototype for a target relation type, they aggregate the representations of all entity pairs holding that relation, while these entity pairs may also hold other relations, thus disturbing the prototype. 2) They use a set of generic NOTA (none-of-the-above) prototypes across all tasks, neglecting that the NOTA semantics differs in tasks with different target relation types. In this paper, we propose a relation-aware prototype learning method for FSDLRE to strengthen the relational semantics of prototype representations. By judiciously leveraging the relation descriptions and realistic NOTA instances as guidance, our method effectively refines the relation prototypes and generates task-specific NOTA prototypes. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches by average 2.61% $F_1$ across various settings of two FSDLRE benchmarks.

{{</citation>}}


### (42/178) Variator: Accelerating Pre-trained Models with Plug-and-Play Compression Modules (Chaojun Xiao et al., 2023)

{{<citation>}}

Chaojun Xiao, Yuqi Luo, Wenbin Zhang, Pengle Zhang, Xu Han, Yankai Lin, Zhengyan Zhang, Ruobing Xie, Zhiyuan Liu, Maosong Sun, Jie Zhou. (2023)  
**Variator: Accelerating Pre-trained Models with Plug-and-Play Compression Modules**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.15724v1)  

---


**ABSTRACT**  
Pre-trained language models (PLMs) have achieved remarkable results on NLP tasks but at the expense of huge parameter sizes and the consequent computational costs. In this paper, we propose Variator, a parameter-efficient acceleration method that enhances computational efficiency through plug-and-play compression plugins. Compression plugins are designed to reduce the sequence length via compressing multiple hidden vectors into one and trained with original PLMs frozen. Different from traditional model acceleration methods, which compress PLMs to smaller sizes, Variator offers two distinct advantages: (1) In real-world applications, the plug-and-play nature of our compression plugins enables dynamic selection of different compression plugins with varying acceleration ratios based on the current workload. (2) The compression plugin comprises a few compact neural network layers with minimal parameters, significantly saving storage and memory overhead, particularly in scenarios with a growing number of tasks. We validate the effectiveness of Variator on seven datasets. Experimental results show that Variator can save 53% computational costs using only 0.9% additional parameters with a performance drop of less than 2%. Moreover, when the model scales to billions of parameters, Variator matches the strong performance of uncompressed PLMs.

{{</citation>}}


### (43/178) Re-Temp: Relation-Aware Temporal Representation Learning for Temporal Knowledge Graph Completion (Kunze Wang et al., 2023)

{{<citation>}}

Kunze Wang, Soyeon Caren Han, Josiah Poon. (2023)  
**Re-Temp: Relation-Aware Temporal Representation Learning for Temporal Knowledge Graph Completion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.15722v1)  

---


**ABSTRACT**  
Temporal Knowledge Graph Completion (TKGC) under the extrapolation setting aims to predict the missing entity from a fact in the future, posing a challenge that aligns more closely with real-world prediction problems. Existing research mostly encodes entities and relations using sequential graph neural networks applied to recent snapshots. However, these approaches tend to overlook the ability to skip irrelevant snapshots according to entity-related relations in the query and disregard the importance of explicit temporal information. To address this, we propose our model, Re-Temp (Relation-Aware Temporal Representation Learning), which leverages explicit temporal embedding as input and incorporates skip information flow after each timestamp to skip unnecessary information for prediction. Additionally, we introduce a two-phase forward propagation method to prevent information leakage. Through the evaluation on six TKGC (extrapolation) datasets, we demonstrate that our model outperforms all eight recent state-of-the-art models by a significant margin.

{{</citation>}}


### (44/178) Ensemble of Task-Specific Language Models for Brain Encoding (Sanjai Kumaran et al., 2023)

{{<citation>}}

Sanjai Kumaran, Arvindh Arun, Jerrin John. (2023)  
**Ensemble of Task-Specific Language Models for Brain Encoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-NE, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15720v1)  

---


**ABSTRACT**  
Language models have been shown to be rich enough to encode fMRI activations of certain Regions of Interest in our Brains. Previous works have explored transfer learning from representations learned for popular natural language processing tasks for predicting brain responses. In our work, we improve the performance of such encoders by creating an ensemble model out of 10 popular Language Models (2 syntactic and 8 semantic). We beat the current baselines by 10% on average across all ROIs through our ensembling methods.

{{</citation>}}


### (45/178) Enhancing Biomedical Lay Summarisation with External Knowledge Graphs (Tomas Goldsack et al., 2023)

{{<citation>}}

Tomas Goldsack, Zhihao Zhang, Chen Tang, Carolina Scarton, Chenghua Lin. (2023)  
**Enhancing Biomedical Lay Summarisation with External Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.15702v1)  

---


**ABSTRACT**  
Previous approaches for automatic lay summarisation are exclusively reliant on the source article that, given it is written for a technical audience (e.g., researchers), is unlikely to explicitly define all technical concepts or state all of the background information that is relevant for a lay audience. We address this issue by augmenting eLife, an existing biomedical lay summarisation dataset, with article-specific knowledge graphs, each containing detailed information on relevant biomedical concepts. Using both automatic and human evaluations, we systematically investigate the effectiveness of three different approaches for incorporating knowledge graphs within lay summarisation models, with each method targeting a distinct area of the encoder-decoder model architecture. Our results confirm that integrating graph-based domain knowledge can significantly benefit lay summarisation by substantially increasing the readability of generated text and improving the explanation of technical concepts.

{{</citation>}}


### (46/178) Towards Automated Recipe Genre Classification using Semi-Supervised Learning (Nazmus Sakib et al., 2023)

{{<citation>}}

Nazmus Sakib, G. M. Shahariar, Md. Mohsinul Kabir, Md. Kamrul Hasan, Hasan Mahmud. (2023)  
**Towards Automated Recipe Genre Classification using Semi-Supervised Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.15693v1)  

---


**ABSTRACT**  
Sharing cooking recipes is a great way to exchange culinary ideas and provide instructions for food preparation. However, categorizing raw recipes found online into appropriate food genres can be challenging due to a lack of adequate labeled data. In this study, we present a dataset named the ``Assorted, Archetypal, and Annotated Two Million Extended (3A2M+) Cooking Recipe Dataset" that contains two million culinary recipes labeled in respective categories with extended named entities extracted from recipe descriptions. This collection of data includes various features such as title, NER, directions, and extended NER, as well as nine different labels representing genres including bakery, drinks, non-veg, vegetables, fast food, cereals, meals, sides, and fusions. The proposed pipeline named 3A2M+ extends the size of the Named Entity Recognition (NER) list to address missing named entities like heat, time or process from the recipe directions using two NER extraction tools. 3A2M+ dataset provides a comprehensive solution to the various challenging recipe-related tasks, including classification, named entity recognition, and recipe generation. Furthermore, we have demonstrated traditional machine learning, deep learning and pre-trained language models to classify the recipes into their corresponding genre and achieved an overall accuracy of 98.6\%. Our investigation indicates that the title feature played a more significant role in classifying the genre.

{{</citation>}}


### (47/178) How Much Context Does My Attention-Based ASR System Need? (Robert Flynn et al., 2023)

{{<citation>}}

Robert Flynn, Anton Ragni. (2023)  
**How Much Context Does My Attention-Based ASR System Need?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.15672v1)  

---


**ABSTRACT**  
For the task of speech recognition, the use of more than 30 seconds of acoustic context during training is uncommon, and under-investigated in literature. In this work, we examine the effect of scaling the sequence length used to train/evaluate (dense-attention based) acoustic and language models on speech recognition performance. For these experiments a dataset of roughly 100,000 pseudo-labelled Spotify podcasts is used, with context lengths of 5 seconds to 1 hour being explored. Zero-shot evaluations on long-format datasets Earnings-22 and Tedlium demonstrate a benefit from training with around 80 seconds of acoustic context, showing up to a 14.9% relative improvement from a limited context baseline. Furthermore, we perform a system combination with long-context transformer language models via beam search for a fully long-context ASR system, with results that are competitive with the current state-of-the-art.

{{</citation>}}


### (48/178) A Survey on Detection of LLMs-Generated Content (Xianjun Yang et al., 2023)

{{<citation>}}

Xianjun Yang, Liangming Pan, Xuandong Zhao, Haifeng Chen, Linda Petzold, William Yang Wang, Wei Cheng. (2023)  
**A Survey on Detection of LLMs-Generated Content**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-HC, cs-LG, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.15654v1)  

---


**ABSTRACT**  
The burgeoning capabilities of advanced large language models (LLMs) such as ChatGPT have led to an increase in synthetic content generation with implications across a variety of sectors, including media, cybersecurity, public discourse, and education. As such, the ability to detect LLMs-generated content has become of paramount importance. We aim to provide a detailed overview of existing detection strategies and benchmarks, scrutinizing their differences and identifying key challenges and prospects in the field, advocating for more adaptable and robust models to enhance detection accuracy. We also posit the necessity for a multi-faceted approach to defend against various attacks to counter the rapidly advancing capabilities of LLMs. To the best of our knowledge, this work is the first comprehensive survey on the detection in the era of LLMs. We hope it will provide a broad understanding of the current landscape of LLMs-generated content detection, offering a guiding reference for researchers and practitioners striving to uphold the integrity of digital information in an era increasingly dominated by synthetic content. The relevant papers are summarized and will be consistently updated at https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git.

{{</citation>}}


### (49/178) CoAnnotating: Uncertainty-Guided Work Allocation between Human and Large Language Models for Data Annotation (Minzhi Li et al., 2023)

{{<citation>}}

Minzhi Li, Taiwei Shi, Caleb Ziems, Min-Yen Kan, Nancy F. Chen, Zhengyuan Liu, Diyi Yang. (2023)  
**CoAnnotating: Uncertainty-Guided Work Allocation between Human and Large Language Models for Data Annotation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.15638v1)  

---


**ABSTRACT**  
Annotated data plays a critical role in Natural Language Processing (NLP) in training models and evaluating their performance. Given recent developments in Large Language Models (LLMs), models such as ChatGPT demonstrate zero-shot capability on many text-annotation tasks, comparable with or even exceeding human annotators. Such LLMs can serve as alternatives for manual annotation, due to lower costs and higher scalability. However, limited work has leveraged LLMs as complementary annotators, nor explored how annotation work is best allocated among humans and LLMs to achieve both quality and cost objectives. We propose CoAnnotating, a novel paradigm for Human-LLM co-annotation of unstructured texts at scale. Under this framework, we utilize uncertainty to estimate LLMs' annotation capability. Our empirical study shows CoAnnotating to be an effective means to allocate work from results on different datasets, with up to 21% performance improvement over random baseline. For code implementation, see https://github.com/SALT-NLP/CoAnnotating.

{{</citation>}}


### (50/178) Career Path Prediction using Resume Representation Learning and Skill-based Matching (Jens-Joris Decorte et al., 2023)

{{<citation>}}

Jens-Joris Decorte, Jeroen Van Hautte, Johannes Deleu, Chris Develder, Thomas Demeester. (2023)  
**Career Path Prediction using Resume Representation Learning and Skill-based Matching**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.15636v1)  

---


**ABSTRACT**  
The impact of person-job fit on job satisfaction and performance is widely acknowledged, which highlights the importance of providing workers with next steps at the right time in their career. This task of predicting the next step in a career is known as career path prediction, and has diverse applications such as turnover prevention and internal job mobility. Existing methods to career path prediction rely on large amounts of private career history data to model the interactions between job titles and companies. We propose leveraging the unexplored textual descriptions that are part of work experience sections in resumes. We introduce a structured dataset of 2,164 anonymized career histories, annotated with ESCO occupation labels. Based on this dataset, we present a novel representation learning approach, CareerBERT, specifically designed for work history data. We develop a skill-based model and a text-based model for career path prediction, which achieve 35.24% and 39.61% recall@10 respectively on our dataset. Finally, we show that both approaches are complementary as a hybrid approach achieves the strongest result with 43.01% recall@10.

{{</citation>}}


### (51/178) Machine Translation for Nko: Tools, Corpora and Baseline Results (Moussa Koulako Bala Doumbouya et al., 2023)

{{<citation>}}

Moussa Koulako Bala Doumbouya, Baba Mamadi Diané, Solo Farabado Cissé, Djibrila Diané, Abdoulaye Sow, Séré Moussa Doumbouya, Daouda Bangoura, Fodé Moriba Bayo, Ibrahima Sory 2. Condé, Kalo Mory Diané, Chris Piech, Christopher Manning. (2023)  
**Machine Translation for Nko: Tools, Corpora and Baseline Results**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-HC, cs-LG, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.15612v1)  

---


**ABSTRACT**  
Currently, there is no usable machine translation system for Nko, a language spoken by tens of millions of people across multiple West African countries, which holds significant cultural and educational value. To address this issue, we present a set of tools, resources, and baseline results aimed towards the development of usable machine translation systems for Nko and other languages that do not currently have sufficiently large parallel text corpora available. (1) Friallel: A novel collaborative parallel text curation software that incorporates quality control through copyedit-based workflows. (2) Expansion of the FLoRes-200 and NLLB-Seed corpora with 2,009 and 6,193 high-quality Nko translations in parallel with 204 and 40 other languages. (3) nicolingua-0005: A collection of trilingual and bilingual corpora with 130,850 parallel segments and monolingual corpora containing over 3 million Nko words. (4) Baseline bilingual and multilingual neural machine translation results with the best model scoring 30.83 English-Nko chrF++ on FLoRes-devtest.

{{</citation>}}


### (52/178) MUSER: A Multi-View Similar Case Retrieval Dataset (Qingquan Li et al., 2023)

{{<citation>}}

Qingquan Li, Yiran Hu, Feng Yao, Chaojun Xiao, Zhiyuan Liu, Maosong Sun, Weixing Shen. (2023)  
**MUSER: A Multi-View Similar Case Retrieval Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15602v1)  

---


**ABSTRACT**  
Similar case retrieval (SCR) is a representative legal AI application that plays a pivotal role in promoting judicial fairness. However, existing SCR datasets only focus on the fact description section when judging the similarity between cases, ignoring other valuable sections (e.g., the court's opinion) that can provide insightful reasoning process behind. Furthermore, the case similarities are typically measured solely by the textual semantics of the fact descriptions, which may fail to capture the full complexity of legal cases from the perspective of legal knowledge. In this work, we present MUSER, a similar case retrieval dataset based on multi-view similarity measurement and comprehensive legal element with sentence-level legal element annotations. Specifically, we select three perspectives (legal fact, dispute focus, and law statutory) and build a comprehensive and structured label schema of legal elements for each of them, to enable accurate and knowledgeable evaluation of case similarities. The constructed dataset originates from Chinese civil cases and contains 100 query cases and 4,024 candidate cases. We implement several text classification algorithms for legal element prediction and various retrieval methods for retrieving similar cases on MUSER. The experimental results indicate that incorporating legal elements can benefit the performance of SCR models, but further efforts are still required to address the remaining challenges posed by MUSER. The source code and dataset are released at https://github.com/THUlawtech/MUSER.

{{</citation>}}


### (53/178) Retrieval-based Knowledge Transfer: An Effective Approach for Extreme Large Language Model Compression (Jiduan Liu et al., 2023)

{{<citation>}}

Jiduan Liu, Jiahao Liu, Qifan Wang, Jingang Wang, Xunliang Cai, Dongyan Zhao, Ran Lucien Wang, Rui Yan. (2023)  
**Retrieval-based Knowledge Transfer: An Effective Approach for Extreme Large Language Model Compression**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLUE, Language Model, NLP, SuperGLUE  
[Paper Link](http://arxiv.org/abs/2310.15594v1)  

---


**ABSTRACT**  
Large-scale pre-trained language models (LLMs) have demonstrated exceptional performance in various natural language processing (NLP) tasks. However, the massive size of these models poses huge challenges for their deployment in real-world applications. While numerous model compression techniques have been proposed, most of them are not well-suited for achieving extreme model compression when there is a significant gap in model scale. In this paper, we introduce a novel compression paradigm called Retrieval-based Knowledge Transfer (RetriKT), which effectively transfers the knowledge of LLMs to extremely small-scale models (e.g., 1%). In particular, our approach extracts knowledge from LLMs to construct a knowledge store, from which the small-scale model can retrieve relevant information and leverage it for effective inference. To improve the quality of the model, soft prompt tuning and Proximal Policy Optimization (PPO) reinforcement learning techniques are employed. Extensive experiments are conducted on low-resource tasks from SuperGLUE and GLUE benchmarks. The results demonstrate that the proposed approach significantly enhances the performance of small-scale models by leveraging the knowledge from LLMs.

{{</citation>}}


### (54/178) Multimodal Representations for Teacher-Guided Compositional Visual Reasoning (Wafa Aissa et al., 2023)

{{<citation>}}

Wafa Aissa, Marin Ferecatu, Michel Crucianu. (2023)  
**Multimodal Representations for Teacher-Guided Compositional Visual Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.15585v1)  

---


**ABSTRACT**  
Neural Module Networks (NMN) are a compelling method for visual question answering, enabling the translation of a question into a program consisting of a series of reasoning sub-tasks that are sequentially executed on the image to produce an answer. NMNs provide enhanced explainability compared to integrated models, allowing for a better understanding of the underlying reasoning process. To improve the effectiveness of NMNs we propose to exploit features obtained by a large-scale cross-modal encoder. Also, the current training approach of NMNs relies on the propagation of module outputs to subsequent modules, leading to the accumulation of prediction errors and the generation of false answers. To mitigate this, we introduce an NMN learning strategy involving scheduled teacher guidance. Initially, the model is fully guided by the ground-truth intermediate outputs, but gradually transitions to an autonomous behavior as training progresses. This reduces error accumulation, thus improving training efficiency and final performance.We demonstrate that by incorporating cross-modal features and employing more effective training techniques for NMN, we achieve a favorable balance between performance and transparency in the reasoning process.

{{</citation>}}


### (55/178) POE: Process of Elimination for Multiple Choice Reasoning (Chenkai Ma et al., 2023)

{{<citation>}}

Chenkai Ma, Xinya Du. (2023)  
**POE: Process of Elimination for Multiple Choice Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.15575v1)  

---


**ABSTRACT**  
Language models (LMs) are capable of conducting in-context learning for multiple choice reasoning tasks, but the options in these tasks are treated equally. As humans often first eliminate wrong options before picking the final correct answer, we argue a similar two-step strategy can make LMs better at these tasks. To this end, we present the Process of Elimination (POE), a two-step scoring method. In the first step, POE scores each option, and eliminates seemingly wrong options. In the second step, POE masks these wrong options, and makes the final prediction from the remaining options. Zero-shot experiments on 8 reasoning tasks illustrate the effectiveness of POE, and a following analysis finds our method to be especially performant on logical reasoning tasks. We further analyze the effect of masks, and show that POE applies to few-shot settings and large language models (LLMs) like ChatGPT.

{{</citation>}}


### (56/178) Natural Language Processing for Drug Discovery Knowledge Graphs: promises and pitfalls (J. Charles G. Jeynes et al., 2023)

{{<citation>}}

J. Charles G. Jeynes, Tim James, Matthew Corney. (2023)  
**Natural Language Processing for Drug Discovery Knowledge Graphs: promises and pitfalls**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.15572v1)  

---


**ABSTRACT**  
Building and analysing knowledge graphs (KGs) to aid drug discovery is a topical area of research. A salient feature of KGs is their ability to combine many heterogeneous data sources in a format that facilitates discovering connections. The utility of KGs has been exemplified in areas such as drug repurposing, with insights made through manual exploration and modelling of the data. In this article, we discuss promises and pitfalls of using natural language processing (NLP) to mine unstructured text typically from scientific literature as a data source for KGs. This draws on our experience of initially parsing structured data sources such as ChEMBL as the basis for data within a KG, and then enriching or expanding upon them using NLP. The fundamental promise of NLP for KGs is the automated extraction of data from millions of documents a task practically impossible to do via human curation alone. However, there are many potential pitfalls in NLP-KG pipelines such as incorrect named entity recognition and ontology linking all of which could ultimately lead to erroneous inferences and conclusions.

{{</citation>}}


### (57/178) MuLMS: A Multi-Layer Annotated Text Corpus for Information Extraction in the Materials Science Domain (Timo Pierre Schrader et al., 2023)

{{<citation>}}

Timo Pierre Schrader, Matteo Finco, Stefan Grünewald, Felix Hildebrand, Annemarie Friedrich. (2023)  
**MuLMS: A Multi-Layer Annotated Text Corpus for Information Extraction in the Materials Science Domain**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2310.15569v1)  

---


**ABSTRACT**  
Keeping track of all relevant recent publications and experimental results for a research area is a challenging task. Prior work has demonstrated the efficacy of information extraction models in various scientific areas. Recently, several datasets have been released for the yet understudied materials science domain. However, these datasets focus on sub-problems such as parsing synthesis procedures or on sub-domains, e.g., solid oxide fuel cells. In this resource paper, we present MuLMS, a new dataset of 50 open-access articles, spanning seven sub-domains of materials science. The corpus has been annotated by domain experts with several layers ranging from named entities over relations to frame structures. We present competitive neural models for all tasks and demonstrate that multi-task training with existing related resources leads to benefits.

{{</citation>}}


### (58/178) TCRA-LLM: Token Compression Retrieval Augmented Large Language Model for Inference Cost Reduction (Junyi Liu et al., 2023)

{{<citation>}}

Junyi Liu, Liangzhi Li, Tong Xiang, Bowen Wang, Yiming Qian. (2023)  
**TCRA-LLM: Token Compression Retrieval Augmented Large Language Model for Inference Cost Reduction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: ChatGPT, GPT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2310.15556v2)  

---


**ABSTRACT**  
Since ChatGPT released its API for public use, the number of applications built on top of commercial large language models (LLMs) increase exponentially. One popular usage of such models is leveraging its in-context learning ability and generating responses given user queries leveraging knowledge obtained by retrieval augmentation. One problem of deploying commercial retrieval-augmented LLMs is the cost due to the additionally retrieved context that largely increases the input token size of the LLMs. To mitigate this, we propose a token compression scheme that includes two methods: summarization compression and semantic compression. The first method applies a T5-based model that is fine-tuned by datasets generated using self-instruct containing samples with varying lengths and reduce token size by doing summarization. The second method further compresses the token size by removing words with lower impact on the semantic. In order to adequately evaluate the effectiveness of the proposed methods, we propose and utilize a dataset called Food-Recommendation DB (FRDB) focusing on food recommendation for women around pregnancy period or infants. Our summarization compression can reduce 65% of the retrieval token size with further 0.3% improvement on the accuracy; semantic compression provides a more flexible way to trade-off the token size with performance, for which we can reduce the token size by 20% with only 1.6% of accuracy drop.

{{</citation>}}


### (59/178) Unveiling Multilinguality in Transformer Models: Exploring Language Specificity in Feed-Forward Networks (Sunit Bhattacharya et al., 2023)

{{<citation>}}

Sunit Bhattacharya, Ondrej Bojar. (2023)  
**Unveiling Multilinguality in Transformer Models: Exploring Language Specificity in Feed-Forward Networks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.15552v1)  

---


**ABSTRACT**  
Recent research suggests that the feed-forward module within Transformers can be viewed as a collection of key-value memories, where the keys learn to capture specific patterns from the input based on the training examples. The values then combine the output from the 'memories' of the keys to generate predictions about the next token. This leads to an incremental process of prediction that gradually converges towards the final token choice near the output layers. This interesting perspective raises questions about how multilingual models might leverage this mechanism. Specifically, for autoregressive models trained on two or more languages, do all neurons (across layers) respond equally to all languages? No! Our hypothesis centers around the notion that during pretraining, certain model parameters learn strong language-specific features, while others learn more language-agnostic (shared across languages) features. To validate this, we conduct experiments utilizing parallel corpora of two languages that the model was initially pretrained on. Our findings reveal that the layers closest to the network's input or output tend to exhibit more language-specific behaviour compared to the layers in the middle.

{{</citation>}}


### (60/178) Improving Language Models Meaning Understanding and Consistency by Learning Conceptual Roles from Dictionary (Myeongjun Erik Jang et al., 2023)

{{<citation>}}

Myeongjun Erik Jang, Thomas Lukasiewicz. (2023)  
**Improving Language Models Meaning Understanding and Consistency by Learning Conceptual Roles from Dictionary**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15541v1)  

---


**ABSTRACT**  
The non-humanlike behaviour of contemporary pre-trained language models (PLMs) is a leading cause undermining their trustworthiness. A striking phenomenon of such faulty behaviours is the generation of inconsistent predictions, which produces logically contradictory results, such as generating different predictions for texts delivering the same meaning or violating logical properties. Previous studies exploited data augmentation or implemented specialised loss functions to alleviate the issue. However, their usage is limited, because they consume expensive training resources for large-sized PLMs and can only handle a certain consistency type. To this end, we propose a practical approach that alleviates the inconsistent behaviour issue by fundamentally improving PLMs' meaning awareness. Based on the conceptual role theory, our method allows PLMs to capture accurate meaning by learning precise interrelationships between concepts from word-definition pairs in a dictionary. Next, we propose an efficient parameter integration technique that updates only a few additional parameters to combine the learned interrelationship with PLMs' pre-trained knowledge. Our experimental results reveal that the approach can concurrently improve multiple types of consistency, enables efficient knowledge integration, and easily applies to other languages.

{{</citation>}}


### (61/178) SteloCoder: a Decoder-Only LLM for Multi-Language to Python Code Translation (Jialing Pan et al., 2023)

{{<citation>}}

Jialing Pan, Adrien Sadé, Jin Kim, Eric Soriano, Guillem Sole, Sylvain Flamant. (2023)  
**SteloCoder: a Decoder-Only LLM for Multi-Language to Python Code Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU, Language Model  
[Paper Link](http://arxiv.org/abs/2310.15539v1)  

---


**ABSTRACT**  
With the recent focus on Large Language Models (LLMs), both StarCoder (Li et al., 2023) and Code Llama (Rozi\`ere et al., 2023) have demonstrated remarkable performance in code generation. However, there is still a need for improvement in code translation functionality with efficient training techniques. In response to this, we introduce SteloCoder, a decoder-only StarCoder-based LLM designed specifically for multi-programming language-to-Python code translation. In particular, SteloCoder achieves C++, C#, JavaScript, Java, or PHP-to-Python code translation without specifying the input programming language. We modified StarCoder model architecture by incorporating a Mixture-of-Experts (MoE) technique featuring five experts and a gating network for multi-task handling. Experts are obtained by StarCoder fine-tuning. Specifically, we use a Low-Rank Adaptive Method (LoRA) technique, limiting each expert size as only 0.06% of number of StarCoder's parameters. At the same time, to enhance training efficiency in terms of time, we adopt curriculum learning strategy and use self-instruct data for efficient fine-tuning. As a result, each expert takes only 6 hours to train on one single 80Gb A100 HBM. With experiments on XLCoST datasets, SteloCoder achieves an average of 73.76 CodeBLEU score in multi-programming language-to-Python translation, surpassing the top performance from the leaderboard by at least 3.5. This accomplishment is attributed to only 45M extra parameters with StarCoder as the backbone and 32 hours of valid training on one 80GB A100 HBM. The source code is release here: https://github.com/sade-adrien/SteloCoder.

{{</citation>}}


### (62/178) MarkQA: A large scale KBQA dataset with numerical reasoning (Xiang Huang et al., 2023)

{{<citation>}}

Xiang Huang, Sitao Cheng, Yuheng Bao, Shanshan Huang, Yuzhong Qu. (2023)  
**MarkQA: A large scale KBQA dataset with numerical reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.15517v1)  

---


**ABSTRACT**  
While question answering over knowledge bases (KBQA) has shown progress in addressing factoid questions, KBQA with numerical reasoning remains relatively unexplored. In this paper, we focus on the complex numerical reasoning in KBQA and propose a new task, NR-KBQA, which necessitates the ability to perform both multi-hop reasoning and numerical reasoning. We design a logic form in Python format called PyQL to represent the reasoning process of numerical reasoning questions. To facilitate the development of NR-KBQA, we present a large dataset called MarkQA, which is automatically constructed from a small set of seeds. Each question in MarkQA is equipped with its corresponding SPARQL query, alongside the step-by-step reasoning process in the QDMR format and PyQL program. Experimental results of some state-of-the-art QA methods on the MarkQA show that complex numerical reasoning in KBQA faces great challenges.

{{</citation>}}


### (63/178) Fighting Fire with Fire: The Dual Role of LLMs in Crafting and Detecting Elusive Disinformation (Jason Lucas et al., 2023)

{{<citation>}}

Jason Lucas, Adaku Uchendu, Michiharu Yamashita, Jooyoung Lee, Shaurya Rohatgi, Dongwon Lee. (2023)  
**Fighting Fire with Fire: The Dual Role of LLMs in Crafting and Detecting Elusive Disinformation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2310.15515v1)  

---


**ABSTRACT**  
Recent ubiquity and disruptive impacts of large language models (LLMs) have raised concerns about their potential to be misused (.i.e, generating large-scale harmful and misleading content). To combat this emerging risk of LLMs, we propose a novel "Fighting Fire with Fire" (F3) strategy that harnesses modern LLMs' generative and emergent reasoning capabilities to counter human-written and LLM-generated disinformation. First, we leverage GPT-3.5-turbo to synthesize authentic and deceptive LLM-generated content through paraphrase-based and perturbation-based prefix-style prompts, respectively. Second, we apply zero-shot in-context semantic reasoning techniques with cloze-style prompts to discern genuine from deceptive posts and news articles. In our extensive experiments, we observe GPT-3.5-turbo's zero-shot superiority for both in-distribution and out-of-distribution datasets, where GPT-3.5-turbo consistently achieved accuracy at 68-72%, unlike the decline observed in previous customized and fine-tuned disinformation detectors. Our codebase and dataset are available at https://github.com/mickeymst/F3.

{{</citation>}}


### (64/178) A Joint Matrix Factorization Analysis of Multilingual Representations (Zheng Zhao et al., 2023)

{{<citation>}}

Zheng Zhao, Yftah Ziser, Bonnie Webber, Shay B. Cohen. (2023)  
**A Joint Matrix Factorization Analysis of Multilingual Representations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2310.15513v1)  

---


**ABSTRACT**  
We present an analysis tool based on joint matrix factorization for comparing latent representations of multilingual and monolingual models. An alternative to probing, this tool allows us to analyze multiple sets of representations in a joint manner. Using this tool, we study to what extent and how morphosyntactic features are reflected in the representations learned by multilingual pre-trained models. We conduct a large-scale empirical study of over 33 languages and 17 morphosyntactic categories. Our findings demonstrate variations in the encoding of morphosyntactic information across upper and lower layers, with category-specific differences influenced by language properties. Hierarchical clustering of the factorization outputs yields a tree structure that is related to phylogenetic trees manually crafted by linguists. Moreover, we find the factorization outputs exhibit strong associations with performance observed across different cross-lingual tasks. We release our code to facilitate future research.

{{</citation>}}


### (65/178) TRAMS: Training-free Memory Selection for Long-range Language Modeling (Haofei Yu et al., 2023)

{{<citation>}}

Haofei Yu, Cunxiang wang, Yue Zhang, Wei Bi. (2023)  
**TRAMS: Training-free Memory Selection for Long-range Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.15494v1)  

---


**ABSTRACT**  
The Transformer architecture is crucial for numerous AI models, but it still faces challenges in long-range language modeling. Though several specific transformer architectures have been designed to tackle issues of long-range dependencies, existing methods like Transformer-XL are plagued by a high percentage of ineffective memories. In this study, we present a plug-and-play strategy, known as TRAining-free Memory Selection (TRAMS), that selects tokens participating in attention calculation based on one simple metric. This strategy allows us to keep tokens that are likely to have a high attention score with the current queries and ignore the other ones. We have tested our approach on the word-level benchmark (WikiText-103) and the character-level benchmark (enwik8), and the results indicate an improvement without having additional training or adding additional parameters.

{{</citation>}}


### (66/178) NuTrea: Neural Tree Search for Context-guided Multi-hop KGQA (Hyeong Kyu Choi et al., 2023)

{{<citation>}}

Hyeong Kyu Choi, Seunghun Lee, Jaewon Chu, Hyunwoo J. Kim. (2023)  
**NuTrea: Neural Tree Search for Context-guided Multi-hop KGQA**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GNN, Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.15484v1)  

---


**ABSTRACT**  
Multi-hop Knowledge Graph Question Answering (KGQA) is a task that involves retrieving nodes from a knowledge graph (KG) to answer natural language questions. Recent GNN-based approaches formulate this task as a KG path searching problem, where messages are sequentially propagated from the seed node towards the answer nodes. However, these messages are past-oriented, and they do not consider the full KG context. To make matters worse, KG nodes often represent proper noun entities and are sometimes encrypted, being uninformative in selecting between paths. To address these problems, we propose Neural Tree Search (NuTrea), a tree search-based GNN model that incorporates the broader KG context. Our model adopts a message-passing scheme that probes the unreached subtree regions to boost the past-oriented embeddings. In addition, we introduce the Relation Frequency-Inverse Entity Frequency (RF-IEF) node embedding that considers the global KG context to better characterize ambiguous KG nodes. The general effectiveness of our approach is demonstrated through experiments on three major multi-hop KGQA benchmark datasets, and our extensive analyses further validate its expressiveness and robustness. Overall, NuTrea provides a powerful means to query the KG with complex natural language questions. Code is available at https://github.com/mlvlab/NuTrea.

{{</citation>}}


### (67/178) CRaSh: Clustering, Removing, and Sharing Enhance Fine-tuning without Full Large Language Model (Kaiyan Zhang et al., 2023)

{{<citation>}}

Kaiyan Zhang, Ning Ding, Biqing Qi, Xuekai Zhu, Xinwei Long, Bowen Zhou. (2023)  
**CRaSh: Clustering, Removing, and Sharing Enhance Fine-tuning without Full Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15477v1)  

---


**ABSTRACT**  
Instruction tuning has recently been recognized as an effective way of aligning Large Language Models (LLMs) to enhance their generalization ability across various tasks. However, when tuning publicly accessible, centralized LLMs with private instruction data, privacy concerns are inevitable. While direct transfer of parameterized modules between models is a plausible approach to address this, its implications and effectiveness need further exploration. This paper focuses on Offsite-Tuning (OFT), a representative technique that transfers transformer blocks between centralized LLMs and downstream emulators. Given the limited understanding of the underlying mechanism of OFT, we perform an empirical analysis on LLMs from the perspectives of representation and functional similarity. Interestingly, our findings reveal a unique modular structure within the layers of LLMs that appears to emerge as the model size expands. Simultaneously, we note subtle but potentially significant changes in representation and intermediate predictions across the layers. Inspired by these observations, we propose CRaSh, involving Clustering, Removing, and Sharing, a training-free strategy to derive improved emulators from LLMs. CRaSh significantly boosts performance of OFT with billions of parameters. Furthermore, we investigate the optimal solutions yielded by fine-tuning with and without full model through the lens of loss landscape. Our findings demonstrate a linear connectivity among these optima falling over the same basin, thereby highlighting the effectiveness of CRaSh and OFT. The source code is publicly available at https://github.com/TsinghuaC3I/CRaSh.

{{</citation>}}


### (68/178) Continual Event Extraction with Semantic Confusion Rectification (Zitao Wang et al., 2023)

{{<citation>}}

Zitao Wang, Xinyi Wang, Wei Hu. (2023)  
**Continual Event Extraction with Semantic Confusion Rectification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Event Extraction  
[Paper Link](http://arxiv.org/abs/2310.15470v1)  

---


**ABSTRACT**  
We study continual event extraction, which aims to extract incessantly emerging event information while avoiding forgetting. We observe that the semantic confusion on event types stems from the annotations of the same text being updated over time. The imbalance between event types even aggravates this issue. This paper proposes a novel continual event extraction model with semantic confusion rectification. We mark pseudo labels for each sentence to alleviate semantic confusion. We transfer pivotal knowledge between current and previous models to enhance the understanding of event types. Moreover, we encourage the model to focus on the semantics of long-tailed event types by leveraging other associated types. Experimental results show that our model outperforms state-of-the-art baselines and is proficient in imbalanced datasets.

{{</citation>}}


### (69/178) Interpreting Answers to Yes-No Questions in User-Generated Content (Shivam Mathur et al., 2023)

{{<citation>}}

Shivam Mathur, Keun Hee Park, Dhivya Chinnappa, Saketh Kotamraju, Eduardo Blanco. (2023)  
**Interpreting Answers to Yes-No Questions in User-Generated Content**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.15464v1)  

---


**ABSTRACT**  
Interpreting answers to yes-no questions in social media is difficult. Yes and no keywords are uncommon, and the few answers that include them are rarely to be interpreted what the keywords suggest. In this paper, we present a new corpus of 4,442 yes-no question-answer pairs from Twitter. We discuss linguistic characteristics of answers whose interpretation is yes or no, as well as answers whose interpretation is unknown. We show that large language models are far from solving this problem, even after fine-tuning and blending other corpora for the same problem but outside social media.

{{</citation>}}


### (70/178) K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings (Chaewon Park et al., 2023)

{{<citation>}}

Chaewon Park, Soohwan Kim, Kyubyong Park, Kunwoo Park. (2023)  
**K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Hate Speech Detection, NLP  
[Paper Link](http://arxiv.org/abs/2310.15439v1)  

---


**ABSTRACT**  
Numerous datasets have been proposed to combat the spread of online hate. Despite these efforts, a majority of these resources are English-centric, primarily focusing on overt forms of hate. This research gap calls for developing high-quality corpora in diverse languages that also encapsulate more subtle hate expressions. This study introduces K-HATERS, a new corpus for hate speech detection in Korean, comprising approximately 192K news comments with target-specific offensiveness ratings. This resource is the largest offensive language corpus in Korean and is the first to offer target-specific ratings on a three-point Likert scale, enabling the detection of hate expressions in Korean across varying degrees of offensiveness. We conduct experiments showing the effectiveness of the proposed corpus, including a comparison with existing datasets. Additionally, to address potential noise and bias in human annotations, we explore a novel idea of adopting the Cognitive Reflection Test, which is widely used in social science for assessing an individual's cognitive ability, as a proxy of labeling quality. Findings indicate that annotations from individuals with the lowest test scores tend to yield detection models that make biased predictions toward specific target groups and are less accurate. This study contributes to the NLP research on hate speech detection and resource construction. The code and dataset can be accessed at https://github.com/ssu-humane/K-HATERS.

{{</citation>}}


### (71/178) What Makes it Ok to Set a Fire? Iterative Self-distillation of Contexts and Rationales for Disambiguating Defeasible Social and Moral Situations (Kavel Rao et al., 2023)

{{<citation>}}

Kavel Rao, Liwei Jiang, Valentina Pyatkin, Yuling Gu, Niket Tandon, Nouha Dziri, Faeze Brahman, Yejin Choi. (2023)  
**What Makes it Ok to Set a Fire? Iterative Self-distillation of Contexts and Rationales for Disambiguating Defeasible Social and Moral Situations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, NLI  
[Paper Link](http://arxiv.org/abs/2310.15431v1)  

---


**ABSTRACT**  
Moral or ethical judgments rely heavily on the specific contexts in which they occur. Understanding varying shades of defeasible contextualizations (i.e., additional information that strengthens or attenuates the moral acceptability of an action) is critical to accurately represent the subtlety and intricacy of grounded human moral judgment in real-life scenarios.   We introduce defeasible moral reasoning: a task to provide grounded contexts that make an action more or less morally acceptable, along with commonsense rationales that justify the reasoning. To elicit high-quality task data, we take an iterative self-distillation approach that starts from a small amount of unstructured seed knowledge from GPT-3 and then alternates between (1) self-distillation from student models; (2) targeted filtering with a critic model trained by human judgment (to boost validity) and NLI (to boost diversity); (3) self-imitation learning (to amplify the desired data quality). This process yields a student model that produces defeasible contexts with improved validity, diversity, and defeasibility. From this model we distill a high-quality dataset, \delta-Rules-of-Thumb, of 1.2M entries of contextualizations and rationales for 115K defeasible moral actions rated highly by human annotators 85.9% to 99.8% of the time. Using \delta-RoT we obtain a final student model that wins over all intermediate student models by a notable margin.

{{</citation>}}


### (72/178) Beyond Sentiment: Leveraging Topic Metrics for Political Stance Classification (Weihong Qi, 2023)

{{<citation>}}

Weihong Qi. (2023)  
**Beyond Sentiment: Leveraging Topic Metrics for Political Stance Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.15429v1)  

---


**ABSTRACT**  
Sentiment analysis, widely critiqued for capturing merely the overall tone of a corpus, falls short in accurately reflecting the latent structures and political stances within texts. This study introduces topic metrics, dummy variables converted from extracted topics, as both an alternative and complement to sentiment metrics in stance classification. By employing three datasets identified by Bestvater and Monroe (2023), this study demonstrates BERTopic's proficiency in extracting coherent topics and the effectiveness of topic metrics in stance classification. The experiment results show that BERTopic improves coherence scores by 17.07% to 54.20% when compared to traditional approaches such as Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF), prevalent in earlier political science research. Additionally, our results indicate topic metrics outperform sentiment metrics in stance classification, increasing performance by as much as 18.95%. Our findings suggest topic metrics are especially effective for context-rich texts and corpus where stance and sentiment correlations are weak. The combination of sentiment and topic metrics achieve an optimal performance in most of the scenarios and can further address the limitations of relying solely on sentiment as well as the low coherence score of topic metrics.

{{</citation>}}


### (73/178) Let the Pretrained Language Models 'Imagine' for Short Texts Topic Modeling (Pritom Saha Akash et al., 2023)

{{<citation>}}

Pritom Saha Akash, Jie Huang, Kevin Chen-Chuan Chang. (2023)  
**Let the Pretrained Language Models 'Imagine' for Short Texts Topic Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Pretrained Language Models, Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2310.15420v1)  

---


**ABSTRACT**  
Topic models are one of the compelling methods for discovering latent semantics in a document collection. However, it assumes that a document has sufficient co-occurrence information to be effective. However, in short texts, co-occurrence information is minimal, which results in feature sparsity in document representation. Therefore, existing topic models (probabilistic or neural) mostly fail to mine patterns from them to generate coherent topics. In this paper, we take a new approach to short-text topic modeling to address the data-sparsity issue by extending short text into longer sequences using existing pre-trained language models (PLMs). Besides, we provide a simple solution extending a neural topic model to reduce the effect of noisy out-of-topics text generation from PLMs. We observe that our model can substantially improve the performance of short-text topic modeling. Extensive experiments on multiple real-world datasets under extreme data sparsity scenarios show that our models can generate high-quality topics outperforming state-of-the-art models.

{{</citation>}}


### (74/178) Mind the Gap Between Conversations for Improved Long-Term Dialogue Generation (Qiang Zhang et al., 2023)

{{<citation>}}

Qiang Zhang, Jason Naradowsky, Yusuke Miyao. (2023)  
**Mind the Gap Between Conversations for Improved Long-Term Dialogue Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2310.15415v1)  

---


**ABSTRACT**  
Knowing how to end and resume conversations over time is a natural part of communication, allowing for discussions to span weeks, months, or years. The duration of gaps between conversations dictates which topics are relevant and which questions to ask, and dialogue systems which do not explicitly model time may generate responses that are unnatural. In this work we explore the idea of making dialogue models aware of time, and present GapChat, a multi-session dialogue dataset in which the time between each session varies. While the dataset is constructed in real-time, progress on events in speakers' lives is simulated in order to create realistic dialogues occurring across a long timespan. We expose time information to the model and compare different representations of time and event progress. In human evaluation we show that time-aware models perform better in metrics that judge the relevance of the chosen topics and the information gained from the conversation.

{{</citation>}}


## cs.LG (31)



### (75/178) ZzzGPT: An Interactive GPT Approach to Enhance Sleep Quality (Yonchanok Khaokaew et al., 2023)

{{<citation>}}

Yonchanok Khaokaew, Thuc Hanh Nguyen, Kaixin Ji, Hiruni Kegalle, Marwah Alaofi. (2023)  
**ZzzGPT: An Interactive GPT Approach to Enhance Sleep Quality**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.16242v1)  

---


**ABSTRACT**  
In today's world, sleep quality is pivotal for overall well-being. While wearable sensors offer real-time monitoring, they often lack actionable insights, leading to user abandonment. This paper delves into the role of technology in understanding sleep patterns. We introduce a two-stage framework, utilizing Large Language Models (LLMs), aiming to provide accurate sleep predictions with actionable feedback. Leveraging the GLOBEM dataset and synthetic data from LLMs, we highlight enhanced results with models like XGBoost. Our approach merges advanced machine learning with user-centric design, blending scientific accuracy with practicality.

{{</citation>}}


### (76/178) Attention-Based Ensemble Pooling for Time Series Forecasting (Dhruvit Patel et al., 2023)

{{<citation>}}

Dhruvit Patel, Alexander Wikner. (2023)  
**Attention-Based Ensemble Pooling for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, nlin-CD  
Keywords: Attention, Time Series  
[Paper Link](http://arxiv.org/abs/2310.16231v1)  

---


**ABSTRACT**  
A common technique to reduce model bias in time-series forecasting is to use an ensemble of predictive models and pool their output into an ensemble forecast. In cases where each predictive model has different biases, however, it is not always clear exactly how each model forecast should be weighed during this pooling. We propose a method for pooling that performs a weighted average over candidate model forecasts, where the weights are learned by an attention-based ensemble pooling model. We test this method on two time-series forecasting problems: multi-step forecasting of the dynamics of the non-stationary Lorenz `63 equation, and one-step forecasting of the weekly incident deaths due to COVID-19. We find that while our model achieves excellent valid times when forecasting the non-stationary Lorenz `63 equation, it does not consistently perform better than the existing ensemble pooling when forecasting COVID-19 weekly incident deaths.

{{</citation>}}


### (77/178) Context-aware feature attribution through argumentation (Jinfeng Zhong et al., 2023)

{{<citation>}}

Jinfeng Zhong, Elsa Negre. (2023)  
**Context-aware feature attribution through argumentation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IR, cs-LG, cs.LG, stat-AP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.16157v1)  

---


**ABSTRACT**  
Feature attribution is a fundamental task in both machine learning and data analysis, which involves determining the contribution of individual features or variables to a model's output. This process helps identify the most important features for predicting an outcome. The history of feature attribution methods can be traced back to General Additive Models (GAMs), which extend linear regression models by incorporating non-linear relationships between dependent and independent variables. In recent years, gradient-based methods and surrogate models have been applied to unravel complex Artificial Intelligence (AI) systems, but these methods have limitations. GAMs tend to achieve lower accuracy, gradient-based methods can be difficult to interpret, and surrogate models often suffer from stability and fidelity issues. Furthermore, most existing methods do not consider users' contexts, which can significantly influence their preferences. To address these limitations and advance the current state-of-the-art, we define a novel feature attribution framework called Context-Aware Feature Attribution Through Argumentation (CA-FATA). Our framework harnesses the power of argumentation by treating each feature as an argument that can either support, attack or neutralize a prediction. Additionally, CA-FATA formulates feature attribution as an argumentation procedure, and each computation has explicit semantics, which makes it inherently interpretable. CA-FATA also easily integrates side information, such as users' contexts, resulting in more accurate predictions.

{{</citation>}}


### (78/178) Alquist 5.0: Dialogue Trees Meet Generative Models. A Novel Approach for Enhancing SocialBot Conversations (Ondřej Kobza et al., 2023)

{{<citation>}}

Ondřej Kobza, Jan Čuhel, Tommaso Gargiani, David Herel, Petr Marek. (2023)  
**Alquist 5.0: Dialogue Trees Meet Generative Models. A Novel Approach for Enhancing SocialBot Conversations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2310.16119v1)  

---


**ABSTRACT**  
We present our SocialBot -- Alquist~5.0 -- developed for the Alexa Prize SocialBot Grand Challenge~5. Building upon previous versions of our system, we introduce the NRG Barista and outline several innovative approaches for integrating Barista into our SocialBot, improving the overall conversational experience. Additionally, we extend our SocialBot to support multimodal devices. This paper offers insights into the development of Alquist~5.0, which meets evolving user expectations while maintaining empathetic and knowledgeable conversational abilities across diverse topics.

{{</citation>}}


### (79/178) Finetuning Offline World Models in the Real World (Yunhai Feng et al., 2023)

{{<citation>}}

Yunhai Feng, Nicklas Hansen, Ziyan Xiong, Chandramouli Rajagopalan, Xiaolong Wang. (2023)  
**Finetuning Offline World Models in the Real World**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.16029v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) is notoriously data-inefficient, which makes training on a real robot difficult. While model-based RL algorithms (world models) improve data-efficiency to some extent, they still require hours or days of interaction to learn skills. Recently, offline RL has been proposed as a framework for training RL policies on pre-existing datasets without any online interaction. However, constraining an algorithm to a fixed dataset induces a state-action distribution shift between training and inference, and limits its applicability to new tasks. In this work, we seek to get the best of both worlds: we consider the problem of pretraining a world model with offline data collected on a real robot, and then finetuning the model on online data collected by planning with the learned model. To mitigate extrapolation errors during online interaction, we propose to regularize the planner at test-time by balancing estimated returns and (epistemic) model uncertainty. We evaluate our method on a variety of visuo-motor control tasks in simulation and on a real robot, and find that our method enables few-shot finetuning to seen and unseen tasks even when offline data is limited. Videos, code, and data are available at https://yunhaifeng.com/FOWM .

{{</citation>}}


### (80/178) What Algorithms can Transformers Learn? A Study in Length Generalization (Hattie Zhou et al., 2023)

{{<citation>}}

Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio, Preetum Nakkiran. (2023)  
**What Algorithms can Transformers Learn? A Study in Length Generalization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.16028v1)  

---


**ABSTRACT**  
Large language models exhibit surprising emergent generalization properties, yet also struggle on many simple reasoning tasks such as arithmetic and parity. This raises the question of if and when Transformer models can learn the true algorithm for solving a task. We study the scope of Transformers' abilities in the specific setting of length generalization on algorithmic tasks. Here, we propose a unifying framework to understand when and how Transformers can exhibit strong length generalization on a given task. Specifically, we leverage RASP (Weiss et al., 2021) -- a programming language designed for the computational model of a Transformer -- and introduce the RASP-Generalization Conjecture: Transformers tend to length generalize on a task if the task can be solved by a short RASP program which works for all input lengths. This simple conjecture remarkably captures most known instances of length generalization on algorithmic tasks. Moreover, we leverage our insights to drastically improve generalization performance on traditionally hard tasks (such as parity and addition). On the theoretical side, we give a simple example where the "min-degree-interpolator" model of learning from Abbe et al. (2023) does not correctly predict Transformers' out-of-distribution behavior, but our conjecture does. Overall, our work provides a novel perspective on the mechanisms of compositional generalization and the algorithmic capabilities of Transformers.

{{</citation>}}


### (81/178) TimewarpVAE: Simultaneous Time-Warping and Representation Learning of Trajectories (Travers Rhodes et al., 2023)

{{<citation>}}

Travers Rhodes, Daniel D. Lee. (2023)  
**TimewarpVAE: Simultaneous Time-Warping and Representation Learning of Trajectories**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.16027v1)  

---


**ABSTRACT**  
Human demonstrations of trajectories are an important source of training data for many machine learning problems. However, the difficulty of collecting human demonstration data for complex tasks makes learning efficient representations of those trajectories challenging. For many problems, such as for handwriting or for quasistatic dexterous manipulation, the exact timings of the trajectories should be factored from their spatial path characteristics. In this work, we propose TimewarpVAE, a fully differentiable manifold-learning algorithm that incorporates Dynamic Time Warping (DTW) to simultaneously learn both timing variations and latent factors of spatial variation. We show how the TimewarpVAE algorithm learns appropriate time alignments and meaningful representations of spatial variations in small handwriting and fork manipulation datasets. Our results have lower spatial reconstruction test error than baseline approaches and the learned low-dimensional representations can be used to efficiently generate semantically meaningful novel trajectories.

{{</citation>}}


### (82/178) Practical Computational Power of Linear Transformers and Their Recurrent and Self-Referential Extensions (Kazuki Irie et al., 2023)

{{<citation>}}

Kazuki Irie, Róbert Csordás, Jürgen Schmidhuber. (2023)  
**Practical Computational Power of Linear Transformers and Their Recurrent and Self-Referential Extensions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.16076v1)  

---


**ABSTRACT**  
Recent studies of the computational power of recurrent neural networks (RNNs) reveal a hierarchy of RNN architectures, given real-time and finite-precision assumptions. Here we study auto-regressive Transformers with linearised attention, a.k.a. linear Transformers (LTs) or Fast Weight Programmers (FWPs). LTs are special in the sense that they are equivalent to RNN-like sequence processors with a fixed-size state, while they can also be expressed as the now-popular self-attention networks. We show that many well-known results for the standard Transformer directly transfer to LTs/FWPs. Our formal language recognition experiments demonstrate how recently proposed FWP extensions such as recurrent FWPs and self-referential weight matrices successfully overcome certain limitations of the LT, e.g., allowing for generalisation on the parity problem. Our code is public.

{{</citation>}}


### (83/178) Graph Deep Learning for Time Series Forecasting (Andrea Cini et al., 2023)

{{<citation>}}

Andrea Cini, Ivan Marisca, Daniele Zambon, Cesare Alippi. (2023)  
**Graph Deep Learning for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.15978v1)  

---


**ABSTRACT**  
Graph-based deep learning methods have become popular tools to process collections of correlated time series. Differently from traditional multivariate forecasting methods, neural graph-based predictors take advantage of pairwise relationships by conditioning forecasts on a (possibly dynamic) graph spanning the time series collection. The conditioning can take the form of an architectural inductive bias on the neural forecasting architecture, resulting in a family of deep learning models called spatiotemporal graph neural networks. Such relational inductive biases enable the training of global forecasting models on large time-series collections, while at the same time localizing predictions w.r.t. each element in the set (i.e., graph nodes) by accounting for local correlations among them (i.e., graph edges). Indeed, recent theoretical and practical advances in graph neural networks and deep learning for time series forecasting make the adoption of such processing frameworks appealing and timely. However, most of the studies in the literature focus on proposing variations of existing neural architectures by taking advantage of modern deep learning practices, while foundational and methodological aspects have not been subject to systematic investigation. To fill the gap, this paper aims to introduce a comprehensive methodological framework that formalizes the forecasting problem and provides design principles for graph-based predictive models and methods to assess their performance. At the same time, together with an overview of the field, we provide design guidelines, recommendations, and best practices, as well as an in-depth discussion of open challenges and future research directions.

{{</citation>}}


### (84/178) Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles (Xing Shen et al., 2023)

{{<citation>}}

Xing Shen, Hengguan Huang, Brennan Nichyporuk, Tal Arbel. (2023)  
**Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2310.15952v2)  

---


**ABSTRACT**  
While deep learning models have achieved remarkable success across a range of medical image analysis tasks, deployment of these models in real clinical contexts requires that they be robust to variability in the acquired images. While many methods apply predefined transformations to augment the training data to enhance test-time robustness, these transformations may not ensure the model's robustness to the diverse variability seen in patient images. In this paper, we introduce a novel three-stage approach based on transformers coupled with conditional diffusion models, with the goal of improving model robustness to the kinds of imaging variability commonly encountered in practice without the need for pre-determined data augmentation strategies. To this end, multiple image encoders first learn hierarchical feature representations to build discriminative latent spaces. Next, a reverse diffusion process, guided by the latent code, acts on an informative prior and proposes prediction candidates in a generative manner. Finally, several prediction candidates are aggregated in a bi-level aggregation protocol to produce the final output. Through extensive experiments on medical imaging benchmark datasets, we show that our method improves upon state-of-the-art methods in terms of robustness and confidence calibration. Additionally, we introduce a strategy to quantify the prediction uncertainty at the instance level, increasing their trustworthiness to clinicians using them in clinical practice.

{{</citation>}}


### (85/178) ABKD: Graph Neural Network Compression with Attention-Based Knowledge Distillation (Anshul Ahluwalia et al., 2023)

{{<citation>}}

Anshul Ahluwalia, Rohit Das, Payman Behnam, Alind Khare, Pan Li, Alexey Tumanov. (2023)  
**ABKD: Graph Neural Network Compression with Attention-Based Knowledge Distillation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, GNN, Graph Neural Network, Graph Neural Networks, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.15938v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have proven to be quite versatile for a variety of applications, including recommendation systems, fake news detection, drug discovery, and even computer vision. Due to the expanding size of graph-structured data, GNN models have also increased in complexity, leading to substantial latency issues. This is primarily attributed to the irregular structure of graph data and its access pattern into memory. The natural solution to reduce latency is to compress large GNNs into small GNNs. One way to do this is via knowledge distillation (KD). However, most KD approaches for GNNs only consider the outputs of the last layers and do not consider the outputs of the intermediate layers of the GNNs; these layers may contain important inductive biases indicated by the graph structure. To address this shortcoming, we propose a novel KD approach to GNN compression that we call Attention-Based Knowledge Distillation (ABKD). ABKD is a KD approach that uses attention to identify important intermediate teacher-student layer pairs and focuses on aligning their outputs. ABKD enables higher compression of GNNs with a smaller accuracy dropoff compared to existing KD approaches. On average, we achieve a 1.79% increase in accuracy with a 32.3x compression ratio on OGBN-Mag, a large graph dataset, compared to state-of-the-art approaches.

{{</citation>}}


### (86/178) E-Sparse: Boosting the Large Language Model Inference through Entropy-based N:M Sparsity (Yun Li et al., 2023)

{{<citation>}}

Yun Li, Lin Niu, Xipeng Zhang, Kai Liu, Jianchen Zhu, Zhanhui Kang. (2023)  
**E-Sparse: Boosting the Large Language Model Inference through Entropy-based N:M Sparsity**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: AI, Generative AI, LLaMA, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.15929v1)  

---


**ABSTRACT**  
Traditional pruning methods are known to be challenging to work in Large Language Models (LLMs) for Generative AI because of their unaffordable training process and large computational demands. For the first time, we introduce the information entropy of hidden state features into a pruning metric design, namely E-Sparse, to improve the accuracy of N:M sparsity on LLM. E-Sparse employs the information richness to leverage the channel importance, and further incorporates several novel techniques to put it into effect: (1) it introduces information entropy to enhance the significance of parameter weights and input feature norms as a novel pruning metric, and performs N:M sparsity without modifying the remaining weights. (2) it designs global naive shuffle and local block shuffle to quickly optimize the information distribution and adequately cope with the impact of N:M sparsity on LLMs' accuracy. E-Sparse is implemented as a Sparse-GEMM on FasterTransformer and runs on NVIDIA Ampere GPUs. Extensive experiments on the LLaMA family and OPT models show that E-Sparse can significantly speed up the model inference over the dense model (up to 1.53X) and obtain significant memory saving (up to 43.52%), with acceptable accuracy loss.

{{</citation>}}


### (87/178) Cross-feature Contrastive Loss for Decentralized Deep Learning on Heterogeneous Data (Sai Aparna Aketi et al., 2023)

{{<citation>}}

Sai Aparna Aketi, Kaushik Roy. (2023)  
**Cross-feature Contrastive Loss for Decentralized Deep Learning on Heterogeneous Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Computer Vision, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.15890v2)  

---


**ABSTRACT**  
The current state-of-the-art decentralized learning algorithms mostly assume the data distribution to be Independent and Identically Distributed (IID). However, in practical scenarios, the distributed datasets can have significantly heterogeneous data distributions across the agents. In this work, we present a novel approach for decentralized learning on heterogeneous data, where data-free knowledge distillation through contrastive loss on cross-features is utilized to improve performance. Cross-features for a pair of neighboring agents are the features (i.e., last hidden layer activations) obtained from the data of an agent with respect to the model parameters of the other agent. We demonstrate the effectiveness of the proposed technique through an exhaustive set of experiments on various Computer Vision datasets (CIFAR-10, CIFAR-100, Fashion MNIST, Imagenette, and ImageNet), model architectures, and network topologies. Our experiments show that the proposed method achieves superior performance (0.2-4% improvement in test accuracy) compared to other existing techniques for decentralized learning on heterogeneous data.

{{</citation>}}


### (88/178) State Sequences Prediction via Fourier Transform for Representation Learning (Mingxuan Ye et al., 2023)

{{<citation>}}

Mingxuan Ye, Yufei Kuang, Jie Wang, Rui Yang, Wengang Zhou, Houqiang Li, Feng Wu. (2023)  
**State Sequences Prediction via Fourier Transform for Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.15888v1)  

---


**ABSTRACT**  
While deep reinforcement learning (RL) has been demonstrated effective in solving complex control tasks, sample efficiency remains a key challenge due to the large amounts of data required for remarkable performance. Existing research explores the application of representation learning for data-efficient RL, e.g., learning predictive representations by predicting long-term future states. However, many existing methods do not fully exploit the structural information inherent in sequential state signals, which can potentially improve the quality of long-term decision-making but is difficult to discern in the time domain. To tackle this problem, we propose State Sequences Prediction via Fourier Transform (SPF), a novel method that exploits the frequency domain of state sequences to extract the underlying patterns in time series data for learning expressive representations efficiently. Specifically, we theoretically analyze the existence of structural information in state sequences, which is closely related to policy performance and signal regularity, and then propose to predict the Fourier transform of infinite-step future state sequences to extract such information. One of the appealing features of SPF is that it is simple to implement while not requiring storage of infinite-step future states as prediction targets. Experiments demonstrate that the proposed method outperforms several state-of-the-art algorithms in terms of both sample efficiency and performance.

{{</citation>}}


### (89/178) Using Causality-Aware Graph Neural Networks to Predict Temporal Centralities in Dynamic Graphs (Franziska Heeg et al., 2023)

{{<citation>}}

Franziska Heeg, Ingo Scholtes. (2023)  
**Using Causality-Aware Graph Neural Networks to Predict Temporal Centralities in Dynamic Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.15865v1)  

---


**ABSTRACT**  
Node centralities play a pivotal role in network science, social network analysis, and recommender systems. In temporal data, static path-based centralities like closeness or betweenness can give misleading results about the true importance of nodes in a temporal graph. To address this issue, temporal generalizations of betweenness and closeness have been defined that are based on the shortest time-respecting paths between pairs of nodes. However, a major issue of those generalizations is that the calculation of such paths is computationally expensive. Addressing this issue, we study the application of De Bruijn Graph Neural Networks (DBGNN), a causality-aware graph neural network architecture, to predict temporal path-based centralities in time series data. We experimentally evaluate our approach in 13 temporal graphs from biological and social systems and show that it considerably improves the prediction of both betweenness and closeness centrality compared to a static Graph Convolutional Neural Network.

{{</citation>}}


### (90/178) On Responsible Machine Learning Datasets with Fairness, Privacy, and Regulatory Norms (Surbhi Mittal et al., 2023)

{{<citation>}}

Surbhi Mittal, Kartik Thakral, Richa Singh, Mayank Vatsa, Tamar Glaser, Cristian Canton Ferrer, Tal Hassner. (2023)  
**On Responsible Machine Learning Datasets with Fairness, Privacy, and Regulatory Norms**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15848v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) has made its way into various scientific fields, providing astonishing improvements over existing algorithms for a wide variety of tasks. In recent years, there have been severe concerns over the trustworthiness of AI technologies. The scientific community has focused on the development of trustworthy AI algorithms. However, machine and deep learning algorithms, popular in the AI community today, depend heavily on the data used during their development. These learning algorithms identify patterns in the data, learning the behavioral objective. Any flaws in the data have the potential to translate directly into algorithms. In this study, we discuss the importance of Responsible Machine Learning Datasets and propose a framework to evaluate the datasets through a responsible rubric. While existing work focuses on the post-hoc evaluation of algorithms for their trustworthiness, we provide a framework that considers the data component separately to understand its role in the algorithm. We discuss responsible datasets through the lens of fairness, privacy, and regulatory compliance and provide recommendations for constructing future datasets. After surveying over 100 datasets, we use 60 datasets for analysis and demonstrate that none of these datasets is immune to issues of fairness, privacy preservation, and regulatory compliance. We provide modifications to the ``datasheets for datasets" with important additions for improved dataset documentation. With governments around the world regularizing data protection laws, the method for the creation of datasets in the scientific community requires revision. We believe this study is timely and relevant in today's era of AI.

{{</citation>}}


### (91/178) Grid Frequency Forecasting in University Campuses using Convolutional LSTM (Aneesh Sathe et al., 2023)

{{<citation>}}

Aneesh Sathe, Wen Ren Yang. (2023)  
**Grid Frequency Forecasting in University Campuses using Convolutional LSTM**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.16071v1)  

---


**ABSTRACT**  
The modern power grid is facing increasing complexities, primarily stemming from the integration of renewable energy sources and evolving consumption patterns. This paper introduces an innovative methodology that harnesses Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to establish robust time series forecasting models for grid frequency. These models effectively capture the spatiotemporal intricacies inherent in grid frequency data, significantly enhancing prediction accuracy and bolstering power grid reliability. The research explores the potential and development of individualized Convolutional LSTM (ConvLSTM) models for buildings within a university campus, enabling them to be independently trained and evaluated for each building. Individual ConvLSTM models are trained on power consumption data for each campus building and forecast the grid frequency based on historical trends. The results convincingly demonstrate the superiority of the proposed models over traditional forecasting techniques, as evidenced by performance metrics such as Mean Square Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). Additionally, an Ensemble Model is formulated to aggregate insights from the building-specific models, delivering comprehensive forecasts for the entire campus. This approach ensures the privacy and security of power consumption data specific to each building.

{{</citation>}}


### (92/178) Improving generalization in large language models by learning prefix subspaces (Louis Falissard et al., 2023)

{{<citation>}}

Louis Falissard, Vincent Guigue, Laure Soulier. (2023)  
**Improving generalization in large language models by learning prefix subspaces**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GLUE  
[Paper Link](http://arxiv.org/abs/2310.15793v1)  

---


**ABSTRACT**  
This article focuses on large language models (LLMs) fine-tuning in the scarce data regime (also known as the "few-shot" learning setting). We propose a method to increase the generalization capabilities of LLMs based on neural network subspaces. This optimization method, recently introduced in computer vision, aims to improve model generalization by identifying wider local optima through the joint optimization of an entire simplex of models in parameter space. Its adaptation to massive, pretrained transformers, however, poses some challenges. First, their considerable number of parameters makes it difficult to train several models jointly, and second, their deterministic parameter initialization schemes make them unfit for the subspace method as originally proposed. We show in this paper that "Parameter Efficient Fine-Tuning" (PEFT) methods, however, are perfectly compatible with this original approach, and propose to learn entire simplex of continuous prefixes. We test our method on a variant of the GLUE benchmark adapted to the few-shot learning setting, and show that both our contributions jointly lead to a gain in average performances compared to sota methods. The implementation can be found at the following link: https://github.com/Liloulou/prefix_subspace

{{</citation>}}


### (93/178) Recurrent Linear Transformers (Subhojeet Pramanik et al., 2023)

{{<citation>}}

Subhojeet Pramanik, Esraa Elelimy, Marlos C. Machado, Adam White. (2023)  
**Recurrent Linear Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.15719v1)  

---


**ABSTRACT**  
The self-attention mechanism in the transformer architecture is capable of capturing long-range dependencies and it is the main reason behind its effectiveness in processing sequential data. Nevertheless, despite their success, transformers have two significant drawbacks that still limit their broader applicability: (1) In order to remember past information, the self-attention mechanism requires access to the whole history to be provided as context. (2) The inference cost in transformers is expensive. In this paper we introduce recurrent alternatives to the transformer self-attention mechanism that offer a context-independent inference cost, leverage long-range dependencies effectively, and perform well in practice. We evaluate our approaches in reinforcement learning problems where the aforementioned computational limitations make the application of transformers nearly infeasible. We quantify the impact of the different components of our architecture in a diagnostic environment and assess performance gains in 2D and 3D pixel-based partially-observable environments. When compared to a state-of-the-art architecture, GTrXL, inference in our approach is at least 40% cheaper while reducing memory use in more than 50%. Our approach either performs similarly or better than GTrXL, improving more than 37% upon GTrXL performance on harder tasks.

{{</citation>}}


### (94/178) COPF: Continual Learning Human Preference through Optimal Policy Fitting (Han Zhang et al., 2023)

{{<citation>}}

Han Zhang, Lin Gui, Yuanzhao Zhai, Hui Wang, Yu Lei, Ruifeng Xu. (2023)  
**COPF: Continual Learning Human Preference through Optimal Policy Fitting**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.15694v3)  

---


**ABSTRACT**  
The technique of Reinforcement Learning from Human Feedback (RLHF) is a commonly employed method to improve pre-trained Language Models (LM), enhancing their ability to conform to human preferences. Nevertheless, the current RLHF-based LMs necessitate full retraining each time novel queries or feedback are introduced, which becomes a challenging task because human preferences can vary between different domains or tasks. Retraining LMs poses practical difficulties in many real-world situations due to the significant time and computational resources required, along with concerns related to data privacy. To address this limitation, we propose a new method called Continual Optimal Policy Fitting (COPF), in which we estimate a series of optimal policies using the Monte Carlo method, and then continually fit the policy sequence with the function regularization. COPF involves a single learning phase and doesn't necessitate complex reinforcement learning. Importantly, it shares the capability with RLHF to learn from unlabeled data, making it flexible for continual preference learning. Our experimental results show that COPF outperforms strong Continuous learning (CL) baselines when it comes to consistently aligning with human preferences on different tasks and domains.

{{</citation>}}


### (95/178) Interactive Generalized Additive Model and Its Applications in Electric Load Forecasting (Linxiao Yang et al., 2023)

{{<citation>}}

Linxiao Yang, Rui Ren, Xinyue Gu, Liang Sun. (2023)  
**Interactive Generalized Additive Model and Its Applications in Electric Load Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15662v1)  

---


**ABSTRACT**  
Electric load forecasting is an indispensable component of electric power system planning and management. Inaccurate load forecasting may lead to the threat of outages or a waste of energy. Accurate electric load forecasting is challenging when there is limited data or even no data, such as load forecasting in holiday, or under extreme weather conditions. As high-stakes decision-making usually follows after load forecasting, model interpretability is crucial for the adoption of forecasting models. In this paper, we propose an interactive GAM which is not only interpretable but also can incorporate specific domain knowledge in electric power industry for improved performance. This boosting-based GAM leverages piecewise linear functions and can be learned through our efficient algorithm. In both public benchmark and electricity datasets, our interactive GAM outperforms current state-of-the-art methods and demonstrates good generalization ability in the cases of extreme weather events. We launched a user-friendly web-based tool based on interactive GAM and already incorporated it into our eForecaster product, a unified AI platform for electricity forecasting.

{{</citation>}}


### (96/178) Confounder Balancing in Adversarial Domain Adaptation for Pre-Trained Large Models Fine-Tuning (Shuoran Jiang et al., 2023)

{{<citation>}}

Shuoran Jiang, Qingcai Chen, Yang Xiang, Youcheng Pan, Xiangping Wu. (2023)  
**Confounder Balancing in Adversarial Domain Adaptation for Pre-Trained Large Models Fine-Tuning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: GPT, GPT-4, LLaMA  
[Paper Link](http://arxiv.org/abs/2310.16062v1)  

---


**ABSTRACT**  
The excellent generalization, contextual learning, and emergence abilities in the pre-trained large models (PLMs) handle specific tasks without direct training data, making them the better foundation models in the adversarial domain adaptation (ADA) methods to transfer knowledge learned from the source domain to target domains. However, existing ADA methods fail to account for the confounder properly, which is the root cause of the source data distribution that differs from the target domains. This study proposes an adversarial domain adaptation with confounder balancing for PLMs fine-tuning (ADA-CBF). The ADA-CBF includes a PLM as the foundation model for a feature extractor, a domain classifier and a confounder classifier, and they are jointly trained with an adversarial loss. This loss is designed to improve the domain-invariant representation learning by diluting the discrimination in the domain classifier. At the same time, the adversarial loss also balances the confounder distribution among source and unmeasured domains in training. Compared to existing ADA methods, ADA-CBF can correctly identify confounders in domain-invariant features, thereby eliminating the confounder biases in the extracted features from PLMs. The confounder classifier in ADA-CBF is designed as a plug-and-play and can be applied in the confounder measurable, unmeasurable, or partially measurable environments. Empirical results on natural language processing and computer vision downstream tasks show that ADA-CBF outperforms the newest GPT-4, LLaMA2, ViT and ADA methods.

{{</citation>}}


### (97/178) Momentum Gradient-based Untargeted Attack on Hypergraph Neural Networks (Yang Chen et al., 2023)

{{<citation>}}

Yang Chen, Stjepan Picek, Zhonglin Ye, Zhaoyang Wang, Haixing Zhao. (2023)  
**Momentum Gradient-based Untargeted Attack on Hypergraph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.15656v1)  

---


**ABSTRACT**  
Hypergraph Neural Networks (HGNNs) have been successfully applied in various hypergraph-related tasks due to their excellent higher-order representation capabilities. Recent works have shown that deep learning models are vulnerable to adversarial attacks. Most studies on graph adversarial attacks have focused on Graph Neural Networks (GNNs), and the study of adversarial attacks on HGNNs remains largely unexplored. In this paper, we try to reduce this gap. We design a new HGNNs attack model for the untargeted attack, namely MGHGA, which focuses on modifying node features. We consider the process of HGNNs training and use a surrogate model to implement the attack before hypergraph modeling. Specifically, MGHGA consists of two parts: feature selection and feature modification. We use a momentum gradient mechanism to choose the attack node features in the feature selection module. In the feature modification module, we use two feature generation approaches (direct modification and sign gradient) to enable MGHGA to be employed on discrete and continuous datasets. We conduct extensive experiments on five benchmark datasets to validate the attack performance of MGHGA in the node and the visual object classification tasks. The results show that MGHGA improves performance by an average of 2% compared to the than the baselines.

{{</citation>}}


### (98/178) Detecting Intentional AIS Shutdown in Open Sea Maritime Surveillance Using Self-Supervised Deep Learning (Pierre Bernabé et al., 2023)

{{<citation>}}

Pierre Bernabé, Arnaud Gotlieb, Bruno Legeard, Dusica Marijan, Frank Olaf Sem-Jacobsen, Helge Spieker. (2023)  
**Detecting Intentional AIS Shutdown in Open Sea Maritime Surveillance Using Self-Supervised Deep Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.15586v1)  

---


**ABSTRACT**  
In maritime traffic surveillance, detecting illegal activities, such as illegal fishing or transshipment of illicit products is a crucial task of the coastal administration. In the open sea, one has to rely on Automatic Identification System (AIS) message transmitted by on-board transponders, which are captured by surveillance satellites. However, insincere vessels often intentionally shut down their AIS transponders to hide illegal activities. In the open sea, it is very challenging to differentiate intentional AIS shutdowns from missing reception due to protocol limitations, bad weather conditions or restricting satellite positions. This paper presents a novel approach for the detection of abnormal AIS missing reception based on self-supervised deep learning techniques and transformer models. Using historical data, the trained model predicts if a message should be received in the upcoming minute or not. Afterwards, the model reports on detected anomalies by comparing the prediction with what actually happens. Our method can process AIS messages in real-time, in particular, more than 500 Millions AIS messages per month, corresponding to the trajectories of more than 60 000 ships. The method is evaluated on 1-year of real-world data coming from four Norwegian surveillance satellites. Using related research results, we validated our method by rediscovering already detected intentional AIS shutdowns.

{{</citation>}}


### (99/178) Accelerating Split Federated Learning over Wireless Communication Networks (Ce Xu et al., 2023)

{{<citation>}}

Ce Xu, Jinxuan Li, Yuan Liu, Yushi Ling, Miaowen Wen. (2023)  
**Accelerating Split Federated Learning over Wireless Communication Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15584v1)  

---


**ABSTRACT**  
The development of artificial intelligence (AI) provides opportunities for the promotion of deep neural network (DNN)-based applications. However, the large amount of parameters and computational complexity of DNN makes it difficult to deploy it on edge devices which are resource-constrained. An efficient method to address this challenge is model partition/splitting, in which DNN is divided into two parts which are deployed on device and server respectively for co-training or co-inference. In this paper, we consider a split federated learning (SFL) framework that combines the parallel model training mechanism of federated learning (FL) and the model splitting structure of split learning (SL). We consider a practical scenario of heterogeneous devices with individual split points of DNN. We formulate a joint problem of split point selection and bandwidth allocation to minimize the system latency. By using alternating optimization, we decompose the problem into two sub-problems and solve them optimally. Experiment results demonstrate the superiority of our work in latency reduction and accuracy improvement.

{{</citation>}}


### (100/178) Symmetry-preserving graph attention network to solve routing problems at multiple resolutions (Cong Dao Tran et al., 2023)

{{<citation>}}

Cong Dao Tran, Thong Bach, Truong Son Hy. (2023)  
**Symmetry-preserving graph attention network to solve routing problems at multiple resolutions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.15543v1)  

---


**ABSTRACT**  
Travelling Salesperson Problems (TSPs) and Vehicle Routing Problems (VRPs) have achieved reasonable improvement in accuracy and computation time with the adaptation of Machine Learning (ML) methods. However, none of the previous works completely respects the symmetries arising from TSPs and VRPs including rotation, translation, permutation, and scaling. In this work, we introduce the first-ever completely equivariant model and training to solve combinatorial problems. Furthermore, it is essential to capture the multiscale structure (i.e. from local to global information) of the input graph, especially for the cases of large and long-range graphs, while previous methods are limited to extracting only local information that can lead to a local or sub-optimal solution. To tackle the above limitation, we propose a Multiresolution scheme in combination with Equivariant Graph Attention network (mEGAT) architecture, which can learn the optimal route based on low-level and high-level graph resolutions in an efficient way. In particular, our approach constructs a hierarchy of coarse-graining graphs from the input graph, in which we try to solve the routing problems on simple low-level graphs first, then utilize that knowledge for the more complex high-level graphs. Experimentally, we have shown that our model outperforms existing baselines and proved that symmetry preservation and multiresolution are important recipes for solving combinatorial problems in a data-driven manner. Our source code is publicly available at https://github.com/HySonLab/Multires-NP-hard

{{</citation>}}


### (101/178) Generative and Contrastive Paradigms Are Complementary for Graph Self-Supervised Learning (Yuxiang Wang et al., 2023)

{{<citation>}}

Yuxiang Wang, Xiao Yan, Chuang Hu, Fangcheng Fu, Wentao Zhang, Hao Wang, Shuo Shang, Jiawei Jiang. (2023)  
**Generative and Contrastive Paradigms Are Complementary for Graph Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.15523v1)  

---


**ABSTRACT**  
For graph self-supervised learning (GSSL), masked autoencoder (MAE) follows the generative paradigm and learns to reconstruct masked graph edges or node features. Contrastive Learning (CL) maximizes the similarity between augmented views of the same graph and is widely used for GSSL. However, MAE and CL are considered separately in existing works for GSSL. We observe that the MAE and CL paradigms are complementary and propose the graph contrastive masked autoencoder (GCMAE) framework to unify them. Specifically, by focusing on local edges or node features, MAE cannot capture global information of the graph and is sensitive to particular edges and features. On the contrary, CL excels in extracting global information because it considers the relation between graphs. As such, we equip GCMAE with an MAE branch and a CL branch, and the two branches share a common encoder, which allows the MAE branch to exploit the global information extracted by the CL branch. To force GCMAE to capture global graph structures, we train it to reconstruct the entire adjacency matrix instead of only the masked edges as in existing works. Moreover, a discrimination loss is proposed for feature reconstruction, which improves the disparity between node embeddings rather than reducing the reconstruction error to tackle the feature smoothing problem of MAE. We evaluate GCMAE on four popular graph tasks (i.e., node classification, node clustering, link prediction, and graph classification) and compare with 14 state-of-the-art baselines. The results show that GCMAE consistently provides good accuracy across these tasks, and the maximum accuracy improvement is up to 3.2% compared with the best-performing baseline.

{{</citation>}}


### (102/178) Graph Attention-based Deep Reinforcement Learning for solving the Chinese Postman Problem with Load-dependent costs (Cong Dao Tran et al., 2023)

{{<citation>}}

Cong Dao Tran, Truong Son Hy. (2023)  
**Graph Attention-based Deep Reinforcement Learning for solving the Chinese Postman Problem with Load-dependent costs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.15516v1)  

---


**ABSTRACT**  
Recently, Deep reinforcement learning (DRL) models have shown promising results in solving routing problems. However, most DRL solvers are commonly proposed to solve node routing problems, such as the Traveling Salesman Problem (TSP). Meanwhile, there has been limited research on applying neural methods to arc routing problems, such as the Chinese Postman Problem (CPP), since they often feature irregular and complex solution spaces compared to TSP. To fill these gaps, this paper proposes a novel DRL framework to address the CPP with load-dependent costs (CPP-LC) (Corberan et al., 2018), which is a complex arc routing problem with load constraints. The novelty of our method is two-fold. First, we formulate the CPP-LC as a Markov Decision Process (MDP) sequential model. Subsequently, we introduce an autoregressive model based on DRL, namely Arc-DRL, consisting of an encoder and decoder to address the CPP-LC challenge effectively. Such a framework allows the DRL model to work efficiently and scalably to arc routing problems. Furthermore, we propose a new bio-inspired meta-heuristic solution based on Evolutionary Algorithm (EA) for CPP-LC. Extensive experiments show that Arc-DRL outperforms existing meta-heuristic methods such as Iterative Local Search (ILS) and Variable Neighborhood Search (VNS) proposed by (Corberan et al., 2018) on large benchmark datasets for CPP-LC regarding both solution quality and running time; while the EA gives the best solution quality with much more running time. We release our C++ implementations for metaheuristics such as EA, ILS and VNS along with the code for data generation and our generated data at https://github.com/HySonLab/Chinese_Postman_Problem

{{</citation>}}


### (103/178) KITAB: Evaluating LLMs on Constraint Satisfaction for Information Retrieval (Marah I Abdin et al., 2023)

{{<citation>}}

Marah I Abdin, Suriya Gunasekar, Varun Chandrasekaran, Jerry Li, Mert Yuksekgonul, Rahee Ghosh Peshawaria, Ranjita Naik, Besmira Nushi. (2023)  
**KITAB: Evaluating LLMs on Constraint Satisfaction for Information Retrieval**  

---
Primary Category: cs.LG  
Categories: I-2-7, cs-AI, cs-CL, cs-IR, cs-LG, cs.LG  
Keywords: GPT, Information Retrieval  
[Paper Link](http://arxiv.org/abs/2310.15511v1)  

---


**ABSTRACT**  
We study the ability of state-of-the art models to answer constraint satisfaction queries for information retrieval (e.g., 'a list of ice cream shops in San Diego'). In the past, such queries were considered to be tasks that could only be solved via web-search or knowledge bases. More recently, large language models (LLMs) have demonstrated initial emergent abilities in this task. However, many current retrieval benchmarks are either saturated or do not measure constraint satisfaction. Motivated by rising concerns around factual incorrectness and hallucinations of LLMs, we present KITAB, a new dataset for measuring constraint satisfaction abilities of language models. KITAB consists of book-related data across more than 600 authors and 13,000 queries, and also offers an associated dynamic data collection and constraint verification approach for acquiring similar test data for other authors. Our extended experiments on GPT4 and GPT3.5 characterize and decouple common failure modes across dimensions such as information popularity, constraint types, and context availability. Results show that in the absence of context, models exhibit severe limitations as measured by irrelevant information, factual errors, and incompleteness, many of which exacerbate as information popularity decreases. While context availability mitigates irrelevant information, it is not helpful for satisfying constraints, identifying fundamental barriers to constraint satisfaction. We open source our contributions to foster further research on improving constraint satisfaction abilities of future models.

{{</citation>}}


### (104/178) General Identifiability and Achievability for Causal Representation Learning (Burak Varıcı et al., 2023)

{{<citation>}}

Burak Varıcı, Emre Acartürk, Karthikeyan Shanmugam, Ali Tajer. (2023)  
**General Identifiability and Achievability for Causal Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.15450v1)  

---


**ABSTRACT**  
This paper focuses on causal representation learning (CRL) under a general nonparametric causal latent model and a general transformation model that maps the latent data to the observational data. It establishes \textbf{identifiability} and \textbf{achievability} results using two hard \textbf{uncoupled} interventions per node in the latent causal graph. Notably, one does not know which pair of intervention environments have the same node intervened (hence, uncoupled environments). For identifiability, the paper establishes that perfect recovery of the latent causal model and variables is guaranteed under uncoupled interventions. For achievability, an algorithm is designed that uses observational and interventional data and recovers the latent causal model and variables with provable guarantees for the algorithm. This algorithm leverages score variations across different environments to estimate the inverse of the transformer and, subsequently, the latent variables. The analysis, additionally, recovers the existing identifiability result for two hard \textbf{coupled} interventions, that is when metadata about the pair of environments that have the same node intervened is known. It is noteworthy that the existing results on non-parametric identifiability require assumptions on interventions and additional faithfulness assumptions. This paper shows that when observational data is available, additional faithfulness assumptions are unnecessary.

{{</citation>}}


### (105/178) Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction (Chih-Yu Lai et al., 2023)

{{<citation>}}

Chih-Yu Lai, Fan-Keng Sun, Zhengqi Gao, Jeffrey H. Lang, Duane S. Boning. (2023)  
**Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2310.15416v1)  

---


**ABSTRACT**  
Time series anomaly detection is challenging due to the complexity and variety of patterns that can occur. One major difficulty arises from modeling time-dependent relationships to find contextual anomalies while maintaining detection accuracy for point anomalies. In this paper, we propose a framework for unsupervised time series anomaly detection that utilizes point-based and sequence-based reconstruction models. The point-based model attempts to quantify point anomalies, and the sequence-based model attempts to quantify both point and contextual anomalies. Under the formulation that the observed time point is a two-stage deviated value from a nominal time point, we introduce a nominality score calculated from the ratio of a combined value of the reconstruction errors. We derive an induced anomaly score by further integrating the nominality score and anomaly score, then theoretically prove the superiority of the induced anomaly score over the original anomaly score under certain conditions. Extensive studies conducted on several public datasets show that the proposed framework outperforms most state-of-the-art baselines for time series anomaly detection.

{{</citation>}}


## cs.CV (29)



### (106/178) TiC-CLIP: Continual Training of CLIP Models (Saurabh Garg et al., 2023)

{{<citation>}}

Saurabh Garg, Mehrdad Farajtabar, Hadi Pouransari, Raviteja Vemulapalli, Sachin Mehta, Oncel Tuzel, Vaishaal Shankar, Fartash Faghri. (2023)  
**TiC-CLIP: Continual Training of CLIP Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.16226v1)  

---


**ABSTRACT**  
Keeping large foundation models up to date on latest data is inherently expensive. To avoid the prohibitive costs of constantly retraining, it is imperative to continually train these models. This problem is exacerbated by the lack of any large scale continual learning benchmarks or baselines. We introduce the first set of web-scale Time-Continual (TiC) benchmarks for training vision-language models: TiC-DataCompt, TiC-YFCC, and TiC-RedCaps with over 12.7B timestamped image-text pairs spanning 9 years (2014--2022). We first use our benchmarks to curate various dynamic evaluations to measure temporal robustness of existing models. We show OpenAI's CLIP (trained on data up to 2020) loses $\approx 8\%$ zero-shot accuracy on our curated retrieval task from 2021--2022 compared with more recently trained models in OpenCLIP repository. We then study how to efficiently train models on time-continuous data. We demonstrate that a simple rehearsal-based approach that continues training from the last checkpoint and replays old data reduces compute by $2.5\times$ when compared to the standard practice of retraining from scratch.

{{</citation>}}


### (107/178) ShadowSense: Unsupervised Domain Adaptation and Feature Fusion for Shadow-Agnostic Tree Crown Detection from RGB-Thermal Drone Imagery (Rudraksh Kapil et al., 2023)

{{<citation>}}

Rudraksh Kapil, Seyed Mojtaba Marvasti-Zadeh, Nadir Erbilgin, Nilanjan Ray. (2023)  
**ShadowSense: Unsupervised Domain Adaptation and Feature Fusion for Shadow-Agnostic Tree Crown Detection from RGB-Thermal Drone Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.16212v1)  

---


**ABSTRACT**  
Accurate detection of individual tree crowns from remote sensing data poses a significant challenge due to the dense nature of forest canopy and the presence of diverse environmental variations, e.g., overlapping canopies, occlusions, and varying lighting conditions. Additionally, the lack of data for training robust models adds another limitation in effectively studying complex forest conditions. This paper presents a novel method for detecting shadowed tree crowns and provides a challenging dataset comprising roughly 50k paired RGB-thermal images to facilitate future research for illumination-invariant detection. The proposed method (ShadowSense) is entirely self-supervised, leveraging domain adversarial training without source domain annotations for feature extraction and foreground feature alignment for feature pyramid networks to adapt domain-invariant representations by focusing on visible foreground regions, respectively. It then fuses complementary information of both modalities to effectively improve upon the predictions of an RGB-trained detector and boost the overall accuracy. Extensive experiments demonstrate the superiority of the proposed method over both the baseline RGB-trained detector and state-of-the-art techniques that rely on unsupervised domain adaptation or early image fusion. Our code and data are available: https://github.com/rudrakshkapil/ShadowSense

{{</citation>}}


### (108/178) Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning (Jon Alvarez Justo et al., 2023)

{{<citation>}}

Jon Alvarez Justo, Joseph Landon Garrett, Mariana-Iuliana Georgescu, Jesus Gonzalez-Llorente, Radu Tudor Ionescu, Tor Arne Johansen. (2023)  
**Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.16210v1)  

---


**ABSTRACT**  
Satellites are increasingly adopting on-board Artificial Intelligence (AI) techniques to enhance platforms' autonomy through edge inference. In this context, the utilization of deep learning (DL) techniques for segmentation in HS satellite imagery offers advantages for remote sensing applications, and therefore, we train 16 different models, whose codes are made available through our study, which we consider to be relevant for on-board multi-class segmentation of HS imagery, focusing on classifying oceanic (sea), terrestrial (land), and cloud formations. We employ the HYPSO-1 mission as an illustrative case for sea-land-cloud segmentation, and to demonstrate the utility of the segments, we introduce a novel sea-land-cloud ranking application scenario. Our system prioritizes HS image downlink based on sea, land, and cloud coverage levels from the segmented images. We comparatively evaluate the models for in-orbit deployment, considering performance, parameter count, and inference time. The models include both shallow and deep models, and after we propose four new DL models, we demonstrate that segmenting single spectral signatures (1D) outperforms 3D data processing comprising both spectral (1D) and spatial (2D) contexts. We conclude that our lightweight DL model, called 1D-Justo-LiuNet, consistently surpasses state-of-the-art models for sea-land-cloud segmentation, such as U-Net and its variations, in terms of performance (0.93 accuracy) and parameter count (4,563). However, the 1D models present longer inference time (15s) in the tested processing architecture, which is clearly suboptimal. Finally, after demonstrating that in-orbit image segmentation should occur post L1b radiance calibration rather than on raw data, we additionally show that reducing spectral channels down to 3 lowers models' parameters and inference time, at the cost of weaker segmentation performance.

{{</citation>}}


### (109/178) iNVS: Repurposing Diffusion Inpainters for Novel View Synthesis (Yash Kant et al., 2023)

{{<citation>}}

Yash Kant, Aliaksandr Siarohin, Michael Vasilkovsky, Riza Alp Guler, Jian Ren, Sergey Tulyakov, Igor Gilitschenski. (2023)  
**iNVS: Repurposing Diffusion Inpainters for Novel View Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.16167v1)  

---


**ABSTRACT**  
We present a method for generating consistent novel views from a single source image. Our approach focuses on maximizing the reuse of visible pixels from the source image. To achieve this, we use a monocular depth estimator that transfers visible pixels from the source view to the target view. Starting from a pre-trained 2D inpainting diffusion model, we train our method on the large-scale Objaverse dataset to learn 3D object priors. While training we use a novel masking mechanism based on epipolar lines to further improve the quality of our approach. This allows our framework to perform zero-shot novel view synthesis on a variety of objects. We evaluate the zero-shot abilities of our framework on three challenging datasets: Google Scanned Objects, Ray Traced Multiview, and Common Objects in 3D. See our webpage for more details: https://yashkant.github.io/invs/

{{</citation>}}


### (110/178) MyriadAL: Active Few Shot Learning for Histopathology (Nico Schiavone et al., 2023)

{{<citation>}}

Nico Schiavone, Jingyi Wang, Shuangzhi Li, Roger Zemp, Xingyu Li. (2023)  
**MyriadAL: Active Few Shot Learning for Histopathology**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2310.16161v1)  

---


**ABSTRACT**  
Active Learning (AL) and Few Shot Learning (FSL) are two label-efficient methods which have achieved excellent results recently. However, most prior arts in both learning paradigms fail to explore the wealth of the vast unlabelled data. In this study, we address this issue in the scenario where the annotation budget is very limited, yet a large amount of unlabelled data for the target task is available. We frame this work in the context of histopathology where labelling is prohibitively expensive. To this end, we introduce an active few shot learning framework, Myriad Active Learning (MAL), including a contrastive-learning encoder, pseudo-label generation, and novel query sample selection in the loop. Specifically, we propose to massage unlabelled data in a self-supervised manner, where the obtained data representations and clustering knowledge form the basis to activate the AL loop. With feedback from the oracle in each AL cycle, the pseudo-labels of the unlabelled data are refined by optimizing a shallow task-specific net on top of the encoder. These updated pseudo-labels serve to inform and improve the active learning query selection process. Furthermore, we introduce a novel recipe to combine existing uncertainty measures and utilize the entire uncertainty list to reduce sample redundancy in AL. Extensive experiments on two public histopathology datasets show that MAL has superior test accuracy, macro F1-score, and label efficiency compared to prior works, and can achieve a comparable test accuracy to a fully supervised algorithm while labelling only 5% of the dataset.

{{</citation>}}


### (111/178) Yin Yang Convolutional Nets: Image Manifold Extraction by the Analysis of Opposites (Augusto Seben da Rosa et al., 2023)

{{<citation>}}

Augusto Seben da Rosa, Frederico Santos de Oliveira, Anderson da Silva Soares, Arnaldo Candido Junior. (2023)  
**Yin Yang Convolutional Nets: Image Manifold Extraction by the Analysis of Opposites**  

---
Primary Category: cs.CV  
Categories: I-2-10, cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.16148v1)  

---


**ABSTRACT**  
Computer vision in general presented several advances such as training optimizations, new architectures (pure attention, efficient block, vision language models, generative models, among others). This have improved performance in several tasks such as classification, and others. However, the majority of these models focus on modifications that are taking distance from realistic neuroscientific approaches related to the brain. In this work, we adopt a more bio-inspired approach and present the Yin Yang Convolutional Network, an architecture that extracts visual manifold, its blocks are intended to separate analysis of colors and forms at its initial layers, simulating occipital lobe's operations. Our results shows that our architecture provides State-of-the-Art efficiency among low parameter architectures in the dataset CIFAR-10. Our first model reached 93.32\% test accuracy, 0.8\% more than the older SOTA in this category, while having 150k less parameters (726k in total). Our second model uses 52k parameters, losing only 3.86\% test accuracy. We also performed an analysis on ImageNet, where we reached 66.49\% validation accuracy with 1.6M parameters. We make the code publicly available at: https://github.com/NoSavedDATA/YinYang_CNN.

{{</citation>}}


### (112/178) Wakening Past Concepts without Past Data: Class-Incremental Learning from Online Placebos (Yaoyao Liu et al., 2023)

{{<citation>}}

Yaoyao Liu, Yingying Li, Bernt Schiele, Qianru Sun. (2023)  
**Wakening Past Concepts without Past Data: Class-Incremental Learning from Online Placebos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.16115v1)  

---


**ABSTRACT**  
Not forgetting old class knowledge is a key challenge for class-incremental learning (CIL) when the model continuously adapts to new classes. A common technique to address this is knowledge distillation (KD), which penalizes prediction inconsistencies between old and new models. Such prediction is made with almost new class data, as old class data is extremely scarce due to the strict memory limitation in CIL. In this paper, we take a deep dive into KD losses and find that "using new class data for KD" not only hinders the model adaption (for learning new classes) but also results in low efficiency for preserving old class knowledge. We address this by "using the placebos of old classes for KD", where the placebos are chosen from a free image stream, such as Google Images, in an automatical and economical fashion. To this end, we train an online placebo selection policy to quickly evaluate the quality of streaming images (good or bad placebos) and use only good ones for one-time feed-forward computation of KD. We formulate the policy training process as an online Markov Decision Process (MDP), and introduce an online learning algorithm to solve this MDP problem without causing much computation costs. In experiments, we show that our method 1) is surprisingly effective even when there is no class overlap between placebos and original old class data, 2) does not require any additional supervision or memory budget, and 3) significantly outperforms a number of top-performing CIL methods, in particular when using lower memory budgets for old class exemplars, e.g., five exemplars per class.

{{</citation>}}


### (113/178) LaksNet: an end-to-end deep learning model for self-driving cars in Udacity simulator (Lakshmikar R. Polamreddy et al., 2023)

{{<citation>}}

Lakshmikar R. Polamreddy, Youshan Zhang. (2023)  
**LaksNet: an end-to-end deep learning model for self-driving cars in Udacity simulator**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.16103v1)  

---


**ABSTRACT**  
The majority of road accidents occur because of human errors, including distraction, recklessness, and drunken driving. One of the effective ways to overcome this dangerous situation is by implementing self-driving technologies in vehicles. In this paper, we focus on building an efficient deep-learning model for self-driving cars. We propose a new and effective convolutional neural network model called `LaksNet' consisting of four convolutional layers and two fully connected layers. We conduct extensive experiments using our LaksNet model with the training data generated from the Udacity simulator. Our model outperforms many existing pre-trained ImageNet and NVIDIA models in terms of the duration of the car for which it drives without going off the track on the simulator.

{{</citation>}}


### (114/178) Synthetic Data as Validation (Qixin Hu et al., 2023)

{{<citation>}}

Qixin Hu, Alan Yuille, Zongwei Zhou. (2023)  
**Synthetic Data as Validation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.16052v1)  

---


**ABSTRACT**  
This study leverages synthetic data as a validation set to reduce overfitting and ease the selection of the best model in AI development. While synthetic data have been used for augmenting the training set, we find that synthetic data can also significantly diversify the validation set, offering marked advantages in domains like healthcare, where data are typically limited, sensitive, and from out-domain sources (i.e., hospitals). In this study, we illustrate the effectiveness of synthetic data for early cancer detection in computed tomography (CT) volumes, where synthetic tumors are generated and superimposed onto healthy organs, thereby creating an extensive dataset for rigorous validation. Using synthetic data as validation can improve AI robustness in both in-domain and out-domain test sets. Furthermore, we establish a new continual learning framework that continuously trains AI models on a stream of out-domain data with synthetic tumors. The AI model trained and validated in dynamically expanding synthetic data can consistently outperform models trained and validated exclusively on real-world data. Specifically, the DSC score for liver tumor segmentation improves from 26.7% (95% CI: 22.6%-30.9%) to 34.5% (30.8%-38.2%) when evaluated on an in-domain dataset and from 31.1% (26.0%-36.2%) to 35.4% (32.1%-38.7%) on an out-domain dataset. Importantly, the performance gain is particularly significant in identifying very tiny liver tumors (radius < 5mm) in CT volumes, with Sensitivity improving from 33.1% to 55.4% on an in-domain dataset and 33.9% to 52.3% on an out-domain dataset, justifying the efficacy in early detection of cancer. The application of synthetic data, from both training and validation perspectives, underlines a promising avenue to enhance AI robustness when dealing with data from varying domains.

{{</citation>}}


### (115/178) Woodpecker: Hallucination Correction for Multimodal Large Language Models (Shukang Yin et al., 2023)

{{<citation>}}

Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, Enhong Chen. (2023)  
**Woodpecker: Hallucination Correction for Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.16045v1)  

---


**ABSTRACT**  
Hallucination is a big shadow hanging over the rapidly evolving Multimodal Large Language Models (MLLMs), referring to the phenomenon that the generated text is inconsistent with the image content. In order to mitigate hallucinations, existing studies mainly resort to an instruction-tuning manner that requires retraining the models with specific data. In this paper, we pave a different way, introducing a training-free method named Woodpecker. Like a woodpecker heals trees, it picks out and corrects hallucinations from the generated text. Concretely, Woodpecker consists of five stages: key concept extraction, question formulation, visual knowledge validation, visual claim generation, and hallucination correction. Implemented in a post-remedy manner, Woodpecker can easily serve different MLLMs, while being interpretable by accessing intermediate outputs of the five stages. We evaluate Woodpecker both quantitatively and qualitatively and show the huge potential of this new paradigm. On the POPE benchmark, our method obtains a 30.66%/24.33% improvement in accuracy over the baseline MiniGPT-4/mPLUG-Owl. The source code is released at https://github.com/BradyFU/Woodpecker.

{{</citation>}}


### (116/178) What's Left? Concept Grounding with Logic-Enhanced Foundation Models (Joy Hsu et al., 2023)

{{<citation>}}

Joy Hsu, Jiayuan Mao, Joshua B. Tenenbaum, Jiajun Wu. (2023)  
**What's Left? Concept Grounding with Logic-Enhanced Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV, stat-ML  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.16035v1)  

---


**ABSTRACT**  
Recent works such as VisProg and ViperGPT have smartly composed foundation models for visual reasoning-using large language models (LLMs) to produce programs that can be executed by pre-trained vision-language models. However, they operate in limited domains, such as 2D images, not fully exploiting the generalization of language: abstract concepts like "left" can also be grounded in 3D, temporal, and action data, as in moving to your left. This limited generalization stems from these inference-only methods' inability to learn or adapt pre-trained models to a new domain. We propose the Logic-Enhanced Foundation Model (LEFT), a unified framework that learns to ground and reason with concepts across domains with a differentiable, domain-independent, first-order logic-based program executor. LEFT has an LLM interpreter that outputs a program represented in a general, logic-based reasoning language, which is shared across all domains and tasks. LEFT's executor then executes the program with trainable domain-specific grounding modules. We show that LEFT flexibly learns concepts in four domains: 2D images, 3D scenes, human motions, and robotic manipulation. It exhibits strong reasoning ability in a wide variety of tasks, including those that are complex and not seen during training, and can be easily applied to new domains.

{{</citation>}}


### (117/178) Visual Cropping Improves Zero-Shot Question Answering of Multimodal Large Language Models (Jiarui Zhang et al., 2023)

{{<citation>}}

Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, Filip Ilievski. (2023)  
**Visual Cropping Improves Zero-Shot Question Answering of Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model, QA, Question Answering, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.16033v1)  

---


**ABSTRACT**  
Multimodal Large Language Models (LLMs) have recently achieved promising zero-shot accuracy on visual question answering (VQA) -- a fundamental task affecting various downstream applications and domains. Given the great potential for the broad use of these models, it is important to investigate their limitations in dealing with different image and question properties. In this work, we investigate whether multimodal LLMs can perceive small details as well as large details in images. In particular, we show that their zero-shot accuracy in answering visual questions is very sensitive to the size of the visual subject of the question, declining up to $46\%$ with size. Furthermore, we show that this effect is causal by observing that human visual cropping can significantly mitigate their sensitivity to size. Inspired by the usefulness of human cropping, we then propose three automatic visual cropping methods as inference time mechanisms to improve the zero-shot performance of multimodal LLMs. We study their effectiveness on four popular VQA datasets, and a subset of the VQAv2 dataset tailored towards fine visual details. Our findings suggest that multimodal LLMs should be used with caution in detail-sensitive VQA applications, and that visual cropping is a promising direction to improve their zero-shot performance. Our code and data are publicly available.

{{</citation>}}


### (118/178) CVPR 2023 Text Guided Video Editing Competition (Jay Zhangjie Wu et al., 2023)

{{<citation>}}

Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang, Youzeng Li, Zuwei Huang, Yuanxi Sun, Rui He, Feng Hu, Junhua Hu, Hai Huang, Hanyu Zhu, Xu Cheng, Jie Tang, Mike Zheng Shou, Kurt Keutzer, Forrest Iandola. (2023)  
**CVPR 2023 Text Guided Video Editing Competition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.16003v1)  

---


**ABSTRACT**  
Humans watch more than a billion hours of video per day. Most of this video was edited manually, which is a tedious process. However, AI-enabled video-generation and video-editing is on the rise. Building on text-to-image models like Stable Diffusion and Imagen, generative AI has improved dramatically on video tasks. But it's hard to evaluate progress in these video tasks because there is no standard benchmark. So, we propose a new dataset for text-guided video editing (TGVE), and we run a competition at CVPR to evaluate models on our TGVE dataset. In this paper we present a retrospective on the competition and describe the winning method. The competition dataset is available at https://sites.google.com/view/loveucvpr23/track4.

{{</citation>}}


### (119/178) Geometry-Aware Video Quality Assessment for Dynamic Digital Human (Zicheng Zhang et al., 2023)

{{<citation>}}

Zicheng Zhang, Yingjie Zhou, Wei Sun, Xiongkuo Min, Guangtao Zhai. (2023)  
**Geometry-Aware Video Quality Assessment for Dynamic Digital Human**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.15984v1)  

---


**ABSTRACT**  
Dynamic Digital Humans (DDHs) are 3D digital models that are animated using predefined motions and are inevitably bothered by noise/shift during the generation process and compression distortion during the transmission process, which needs to be perceptually evaluated. Usually, DDHs are displayed as 2D rendered animation videos and it is natural to adapt video quality assessment (VQA) methods to DDH quality assessment (DDH-QA) tasks. However, the VQA methods are highly dependent on viewpoints and less sensitive to geometry-based distortions. Therefore, in this paper, we propose a novel no-reference (NR) geometry-aware video quality assessment method for DDH-QA challenge. Geometry characteristics are described by the statistical parameters estimated from the DDHs' geometry attribute distributions. Spatial and temporal features are acquired from the rendered videos. Finally, all kinds of features are integrated and regressed into quality values. Experimental results show that the proposed method achieves state-of-the-art performance on the DDH-QA database.

{{</citation>}}


### (120/178) Decoupled DETR: Spatially Disentangling Localization and Classification for Improved End-to-End Object Detection (Manyuan Zhang et al., 2023)

{{<citation>}}

Manyuan Zhang, Guanglu Song, Yu Liu, Hongsheng Li. (2023)  
**Decoupled DETR: Spatially Disentangling Localization and Classification for Improved End-to-End Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.15955v1)  

---


**ABSTRACT**  
The introduction of DETR represents a new paradigm for object detection. However, its decoder conducts classification and box localization using shared queries and cross-attention layers, leading to suboptimal results. We observe that different regions of interest in the visual feature map are suitable for performing query classification and box localization tasks, even for the same object. Salient regions provide vital information for classification, while the boundaries around them are more favorable for box regression. Unfortunately, such spatial misalignment between these two tasks greatly hinders DETR's training. Therefore, in this work, we focus on decoupling localization and classification tasks in DETR. To achieve this, we introduce a new design scheme called spatially decoupled DETR (SD-DETR), which includes a task-aware query generation module and a disentangled feature learning process. We elaborately design the task-aware query initialization process and divide the cross-attention block in the decoder to allow the task-aware queries to match different visual regions. Meanwhile, we also observe that the prediction misalignment problem for high classification confidence and precise localization exists, so we propose an alignment loss to further guide the spatially decoupled DETR training. Through extensive experiments, we demonstrate that our approach achieves a significant improvement in MSCOCO datasets compared to previous work. For instance, we improve the performance of Conditional DETR by 4.5 AP. By spatially disentangling the two tasks, our method overcomes the misalignment problem and greatly improves the performance of DETR for object detection.

{{</citation>}}


### (121/178) CPSeg: Finer-grained Image Semantic Segmentation via Chain-of-Thought Language Prompting (Lei Li, 2023)

{{<citation>}}

Lei Li. (2023)  
**CPSeg: Finer-grained Image Semantic Segmentation via Chain-of-Thought Language Prompting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.16069v2)  

---


**ABSTRACT**  
Natural scene analysis and remote sensing imagery offer immense potential for advancements in large-scale language-guided context-aware data utilization. This potential is particularly significant for enhancing performance in downstream tasks such as object detection and segmentation with designed language prompting. In light of this, we introduce the CPSeg, Chain-of-Thought Language Prompting for Finer-grained Semantic Segmentation), an innovative framework designed to augment image segmentation performance by integrating a novel "Chain-of-Thought" process that harnesses textual information associated with images. This groundbreaking approach has been applied to a flood disaster scenario. CPSeg encodes prompt texts derived from various sentences to formulate a coherent chain-of-thought. We propose a new vision-language dataset, FloodPrompt, which includes images, semantic masks, and corresponding text information. This not only strengthens the semantic understanding of the scenario but also aids in the key task of semantic segmentation through an interplay of pixel and text matching maps. Our qualitative and quantitative analyses validate the effectiveness of CPSeg.

{{</citation>}}


### (122/178) Automatic Aorta Segmentation with Heavily Augmented, High-Resolution 3-D ResUNet: Contribution to the SEG.A Challenge (Marek Wodzinski et al., 2023)

{{<citation>}}

Marek Wodzinski, Henning Müller. (2023)  
**Automatic Aorta Segmentation with Heavily Augmented, High-Resolution 3-D ResUNet: Contribution to the SEG.A Challenge**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15827v1)  

---


**ABSTRACT**  
Automatic aorta segmentation from 3-D medical volumes is an important yet difficult task. Several factors make the problem challenging, e.g. the possibility of aortic dissection or the difficulty with segmenting and annotating the small branches. This work presents a contribution by the MedGIFT team to the SEG.A challenge organized during the MICCAI 2023 conference. We propose a fully automated algorithm based on deep encoder-decoder architecture. The main assumption behind our work is that data preprocessing and augmentation are much more important than the deep architecture, especially in low data regimes. Therefore, the solution is based on a variant of traditional convolutional U-Net. The proposed solution achieved a Dice score above 0.9 for all testing cases with the highest stability among all participants. The method scored 1st, 4th, and 3rd in terms of the clinical evaluation, quantitative results, and volumetric meshing quality, respectively. We freely release the source code, pretrained model, and provide access to the algorithm on the Grand-Challenge platform.

{{</citation>}}


### (123/178) SequenceMatch: Revisiting the design of weak-strong augmentations for Semi-supervised learning (Khanh-Binh Nguyen, 2023)

{{<citation>}}

Khanh-Binh Nguyen. (2023)  
**SequenceMatch: Revisiting the design of weak-strong augmentations for Semi-supervised learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.15787v1)  

---


**ABSTRACT**  
Semi-supervised learning (SSL) has become popular in recent years because it allows the training of a model using a large amount of unlabeled data. However, one issue that many SSL methods face is the confirmation bias, which occurs when the model is overfitted to the small labeled training dataset and produces overconfident, incorrect predictions. To address this issue, we propose SequenceMatch, an efficient SSL method that utilizes multiple data augmentations. The key element of SequenceMatch is the inclusion of a medium augmentation for unlabeled data. By taking advantage of different augmentations and the consistency constraints between each pair of augmented examples, SequenceMatch helps reduce the divergence between the prediction distribution of the model for weakly and strongly augmented examples. In addition, SequenceMatch defines two different consistency constraints for high and low-confidence predictions. As a result, SequenceMatch is more data-efficient than ReMixMatch, and more time-efficient than both ReMixMatch ($\times4$) and CoMatch ($\times2$) while having higher accuracy. Despite its simplicity, SequenceMatch consistently outperforms prior methods on standard benchmarks, such as CIFAR-10/100, SVHN, and STL-10. It also surpasses prior state-of-the-art methods by a large margin on large-scale datasets such as ImageNet, with a 38.46\% error rate. Code is available at https://github.com/beandkay/SequenceMatch.

{{</citation>}}


### (124/178) Debiasing, calibrating, and improving Semi-supervised Learning performance via simple Ensemble Projector (Khanh-Binh Nguyen, 2023)

{{<citation>}}

Khanh-Binh Nguyen. (2023)  
**Debiasing, calibrating, and improving Semi-supervised Learning performance via simple Ensemble Projector**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.15764v1)  

---


**ABSTRACT**  
Recent studies on semi-supervised learning (SSL) have achieved great success. Despite their promising performance, current state-of-the-art methods tend toward increasingly complex designs at the cost of introducing more network components and additional training procedures. In this paper, we propose a simple method named Ensemble Projectors Aided for Semi-supervised Learning (EPASS), which focuses mainly on improving the learned embeddings to boost the performance of the existing contrastive joint-training semi-supervised learning frameworks. Unlike standard methods, where the learned embeddings from one projector are stored in memory banks to be used with contrastive learning, EPASS stores the ensemble embeddings from multiple projectors in memory banks. As a result, EPASS improves generalization, strengthens feature representation, and boosts performance. For instance, EPASS improves strong baselines for semi-supervised learning by 39.47\%/31.39\%/24.70\% top-1 error rate, while using only 100k/1\%/10\% of labeled data for SimMatch, and achieves 40.24\%/32.64\%/25.90\% top-1 error rate for CoMatch on the ImageNet dataset. These improvements are consistent across methods, network architectures, and datasets, proving the general effectiveness of the proposed methods. Code is available at https://github.com/beandkay/EPASS.

{{</citation>}}


### (125/178) Large Language Models are Temporal and Causal Reasoners for Video Question Answering (Dohwan Ko et al., 2023)

{{<citation>}}

Dohwan Ko, Ji Soo Lee, Wooyoung Kang, Byungseok Roh, Hyunwoo J. Kim. (2023)  
**Large Language Models are Temporal and Causal Reasoners for Video Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, LLaMA, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.15747v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown remarkable performances on a wide range of natural language understanding and generation tasks. We observe that the LLMs provide effective priors in exploiting $\textit{linguistic shortcuts}$ for temporal and causal reasoning in Video Question Answering (VideoQA). However, such priors often cause suboptimal results on VideoQA by leading the model to over-rely on questions, $\textit{i.e.}$, $\textit{linguistic bias}$, while ignoring visual content. This is also known as `ungrounded guesses' or `hallucinations'. To address this problem while leveraging LLMs' prior on VideoQA, we propose a novel framework, Flipped-VQA, encouraging the model to predict all the combinations of $\langle$V, Q, A$\rangle$ triplet by flipping the source pair and the target label to understand their complex relationships, $\textit{i.e.}$, predict A, Q, and V given a VQ, VA, and QA pairs, respectively. In this paper, we develop LLaMA-VQA by applying Flipped-VQA to LLaMA, and it outperforms both LLMs-based and non-LLMs-based models on five challenging VideoQA benchmarks. Furthermore, our Flipped-VQA is a general framework that is applicable to various LLMs (OPT and GPT-J) and consistently improves their performances. We empirically demonstrate that Flipped-VQA not only enhances the exploitation of linguistic shortcuts but also mitigates the linguistic bias, which causes incorrect answers over-relying on the question. Code is available at https://github.com/mlvlab/Flipped-VQA.

{{</citation>}}


### (126/178) Interpretable Medical Image Classification using Prototype Learning and Privileged Information (Luisa Gallee et al., 2023)

{{<citation>}}

Luisa Gallee, Meinrad Beer, Michael Goetz. (2023)  
**Interpretable Medical Image Classification using Prototype Learning and Privileged Information**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2310.15741v1)  

---


**ABSTRACT**  
Interpretability is often an essential requirement in medical imaging. Advanced deep learning methods are required to address this need for explainability and high performance. In this work, we investigate whether additional information available during the training process can be used to create an understandable and powerful model. We propose an innovative solution called Proto-Caps that leverages the benefits of capsule networks, prototype learning and the use of privileged information. Evaluating the proposed solution on the LIDC-IDRI dataset shows that it combines increased interpretability with above state-of-the-art prediction performance. Compared to the explainable baseline model, our method achieves more than 6 % higher accuracy in predicting both malignancy (93.0 %) and mean characteristic features of lung nodules. Simultaneously, the model provides case-based reasoning with prototype representations that allow visual validation of radiologist-defined attributes.

{{</citation>}}


### (127/178) Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection (Linyan Huang et al., 2023)

{{<citation>}}

Linyan Huang, Zhiqi Li, Chonghao Sima, Wenhai Wang, Jingdong Wang, Yu Qiao, Hongyang Li. (2023)  
**Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.15670v1)  

---


**ABSTRACT**  
Current research is primarily dedicated to advancing the accuracy of camera-only 3D object detectors (apprentice) through the knowledge transferred from LiDAR- or multi-modal-based counterparts (expert). However, the presence of the domain gap between LiDAR and camera features, coupled with the inherent incompatibility in temporal fusion, significantly hinders the effectiveness of distillation-based enhancements for apprentices. Motivated by the success of uni-modal distillation, an apprentice-friendly expert model would predominantly rely on camera features, while still achieving comparable performance to multi-modal models. To this end, we introduce VCD, a framework to improve the camera-only apprentice model, including an apprentice-friendly multi-modal expert and temporal-fusion-friendly distillation supervision. The multi-modal expert VCD-E adopts an identical structure as that of the camera-only apprentice in order to alleviate the feature disparity, and leverages LiDAR input as a depth prior to reconstruct the 3D scene, achieving the performance on par with other heterogeneous multi-modal experts. Additionally, a fine-grained trajectory-based distillation module is introduced with the purpose of individually rectifying the motion misalignment for each object in the scene. With those improvements, our camera-only apprentice VCD-A sets new state-of-the-art on nuScenes with a score of 63.1% NDS.

{{</citation>}}


### (128/178) Region-controlled Style Transfer (Junjie Kang et al., 2023)

{{<citation>}}

Junjie Kang, Jinsong Wu, Shiqi Jiang. (2023)  
**Region-controlled Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2310.15658v1)  

---


**ABSTRACT**  
Image style transfer is a challenging task in computational vision. Existing algorithms transfer the color and texture of style images by controlling the neural network's feature layers. However, they fail to control the strength of textures in different regions of the content image. To address this issue, we propose a training method that uses a loss function to constrain the style intensity in different regions. This method guides the transfer strength of style features in different regions based on the gradient relationship between style and content images. Additionally, we introduce a novel feature fusion method that linearly transforms content features to resemble style features while preserving their semantic relationships. Extensive experiments have demonstrated the effectiveness of our proposed approach.

{{</citation>}}


### (129/178) Mean Teacher DETR with Masked Feature Alignment: A Robust Domain Adaptive Detection Transformer Framework (Weixi Weng et al., 2023)

{{<citation>}}

Weixi Weng, Chun Yuan. (2023)  
**Mean Teacher DETR with Masked Feature Alignment: A Robust Domain Adaptive Detection Transformer Framework**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.15646v1)  

---


**ABSTRACT**  
Unsupervised domain adaptation object detection(UDAOD) research on Detection Transformer(DETR) mainly focuses on feature alignment and existing methods can be divided into two kinds, each of which has its unresolved issues. One-stage feature alignment methods can easily lead to performance fluctuation and training stagnation. Two-stage feature alignment method based on mean teacher comprises a pretraining stage followed by a self-training stage, each facing problems in obtaining reliable pretrained model and achieving consistent performance gains. Methods mentioned above have not yet explore how to utilize the third related domain such as target-like domain to assist adaptation. To address these issues, we propose a two-stage framework named MTM, i.e. Mean Teacher-DETR with Masked Feature Alignment. In the pretraining stage, we utilize labeled target-like images produced by image style transfer to avoid performance fluctuation. In the self-training stage, we leverage unlabeled target images by pseudo labels based on mean teacher and propose a module called Object Queries Knowledge Transfer(OQKT) to ensure consistent performance gains of the student model. Most importantly, we propose masked feature alignment methods including Masked Domain Query-based Feature Alignment(MDQFA) and Masked Token-wise Feature Alignment(MTWFA) to alleviate domain shift in a more robust way, which not only prevent training stagnation and lead to a robust pretrained model in the pretraining stage, but also enhance the model's target performance in the self-training stage. Experiments on three challenging scenarios and a theoretical analysis verify the effectiveness of MTM.

{{</citation>}}


### (130/178) GUPNet++: Geometry Uncertainty Propagation Network for Monocular 3D Object Detection (Yan Lu et al., 2023)

{{<citation>}}

Yan Lu, Xinzhu Ma, Lei Yang, Tianzhu Zhang, Yating Liu, Qi Chu, Tong He, Yonghui Li, Wanli Ouyang. (2023)  
**GUPNet++: Geometry Uncertainty Propagation Network for Monocular 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.15624v1)  

---


**ABSTRACT**  
Geometry plays a significant role in monocular 3D object detection. It can be used to estimate object depth by using the perspective projection between object's physical size and 2D projection in the image plane, which can introduce mathematical priors into deep models. However, this projection process also introduces error amplification, where the error of the estimated height is amplified and reflected into the projected depth. It leads to unreliable depth inferences and also impairs training stability. To tackle this problem, we propose a novel Geometry Uncertainty Propagation Network (GUPNet++) by modeling geometry projection in a probabilistic manner. This ensures depth predictions are well-bounded and associated with a reasonable uncertainty. The significance of introducing such geometric uncertainty is two-fold: (1). It models the uncertainty propagation relationship of the geometry projection during training, improving the stability and efficiency of the end-to-end model learning. (2). It can be derived to a highly reliable confidence to indicate the quality of the 3D detection result, enabling more reliable detection inference. Experiments show that the proposed approach not only obtains (state-of-the-art) SOTA performance in image-based monocular 3D detection but also demonstrates superiority in efficacy with a simplified framework.

{{</citation>}}


### (131/178) I$^2$MD: 3D Action Representation Learning with Inter- and Intra-modal Mutual Distillation (Yunyao Mao et al., 2023)

{{<citation>}}

Yunyao Mao, Jiajun Deng, Wengang Zhou, Zhenbo Lu, Wanli Ouyang, Houqiang Li. (2023)  
**I$^2$MD: 3D Action Representation Learning with Inter- and Intra-modal Mutual Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.15568v1)  

---


**ABSTRACT**  
Recent progresses on self-supervised 3D human action representation learning are largely attributed to contrastive learning. However, in conventional contrastive frameworks, the rich complementarity between different skeleton modalities remains under-explored. Moreover, optimized with distinguishing self-augmented samples, models struggle with numerous similar positive instances in the case of limited action categories. In this work, we tackle the aforementioned problems by introducing a general Inter- and Intra-modal Mutual Distillation (I$^2$MD) framework. In I$^2$MD, we first re-formulate the cross-modal interaction as a Cross-modal Mutual Distillation (CMD) process. Different from existing distillation solutions that transfer the knowledge of a pre-trained and fixed teacher to the student, in CMD, the knowledge is continuously updated and bidirectionally distilled between modalities during pre-training. To alleviate the interference of similar samples and exploit their underlying contexts, we further design the Intra-modal Mutual Distillation (IMD) strategy, In IMD, the Dynamic Neighbors Aggregation (DNA) mechanism is first introduced, where an additional cluster-level discrimination branch is instantiated in each modality. It adaptively aggregates highly-correlated neighboring features, forming local cluster-level contrasting. Mutual distillation is then performed between the two branches for cross-level knowledge exchange. Extensive experiments on three datasets show that our approach sets a series of new records.

{{</citation>}}


### (132/178) Learning with Noisy Labels Using Collaborative Sample Selection and Contrastive Semi-Supervised Learning (Qing Miao et al., 2023)

{{<citation>}}

Qing Miao, Xiaohe Wu, Chao Xu, Yanli Ji, Wangmeng Zuo, Yiwen Guo, Zhaopeng Meng. (2023)  
**Learning with Noisy Labels Using Collaborative Sample Selection and Contrastive Semi-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.15533v1)  

---


**ABSTRACT**  
Learning with noisy labels (LNL) has been extensively studied, with existing approaches typically following a framework that alternates between clean sample selection and semi-supervised learning (SSL). However, this approach has a limitation: the clean set selected by the Deep Neural Network (DNN) classifier, trained through self-training, inevitably contains noisy samples. This mixture of clean and noisy samples leads to misguidance in DNN training during SSL, resulting in impaired generalization performance due to confirmation bias caused by error accumulation in sample selection. To address this issue, we propose a method called Collaborative Sample Selection (CSS), which leverages the large-scale pre-trained model CLIP. CSS aims to remove the mixed noisy samples from the identified clean set. We achieve this by training a 2-Dimensional Gaussian Mixture Model (2D-GMM) that combines the probabilities from CLIP with the predictions from the DNN classifier. To further enhance the adaptation of CLIP to LNL, we introduce a co-training mechanism with a contrastive loss in semi-supervised learning. This allows us to jointly train the prompt of CLIP and the DNN classifier, resulting in improved feature representation, boosted classification performance of DNNs, and reciprocal benefits to our Collaborative Sample Selection. By incorporating auxiliary information from CLIP and utilizing prompt fine-tuning, we effectively eliminate noisy samples from the clean set and mitigate confirmation bias during training. Experimental results on multiple benchmark datasets demonstrate the effectiveness of our proposed method in comparison with the state-of-the-art approaches.

{{</citation>}}


### (133/178) Salient Object Detection in RGB-D Videos (Ao Mou et al., 2023)

{{<citation>}}

Ao Mou, Yukang Lu, Jiahao He, Dingyao Min, Keren Fu, Qijun Zhao. (2023)  
**Salient Object Detection in RGB-D Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.15482v1)  

---


**ABSTRACT**  
Given the widespread adoption of depth-sensing acquisition devices, RGB-D videos and related data/media have gained considerable traction in various aspects of daily life. Consequently, conducting salient object detection (SOD) in RGB-D videos presents a highly promising and evolving avenue. Despite the potential of this area, SOD in RGB-D videos remains somewhat under-explored, with RGB-D SOD and video SOD (VSOD) traditionally studied in isolation. To explore this emerging field, this paper makes two primary contributions: the dataset and the model. On one front, we construct the RDVS dataset, a new RGB-D VSOD dataset with realistic depth and characterized by its diversity of scenes and rigorous frame-by-frame annotations. We validate the dataset through comprehensive attribute and object-oriented analyses, and provide training and testing splits. Moreover, we introduce DCTNet+, a three-stream network tailored for RGB-D VSOD, with an emphasis on RGB modality and treats depth and optical flow as auxiliary modalities. In pursuit of effective feature enhancement, refinement, and fusion for precise final prediction, we propose two modules: the multi-modal attention module (MAM) and the refinement fusion module (RFM). To enhance interaction and fusion within RFM, we design a universal interaction module (UIM) and then integrate holistic multi-modal attentive paths (HMAPs) for refining multi-modal low-level features before reaching RFMs. Comprehensive experiments, conducted on pseudo RGB-D video datasets alongside our RDVS, highlight the superiority of DCTNet+ over 17 VSOD models and 14 RGB-D SOD models. Ablation experiments were performed on both pseudo and realistic RGB-D video datasets to demonstrate the advantages of individual modules as well as the necessity of introducing realistic depth. Our code together with RDVS dataset will be available at https://github.com/kerenfu/RDVS/.

{{</citation>}}


### (134/178) Fast Propagation is Better: Accelerating Single-Step Adversarial Training via Sampling Subnetworks (Xiaojun Jia et al., 2023)

{{<citation>}}

Xiaojun Jia, Jianshu Li, Jindong Gu, Yang Bai, Xiaochun Cao. (2023)  
**Fast Propagation is Better: Accelerating Single-Step Adversarial Training via Sampling Subnetworks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Training, QA  
[Paper Link](http://arxiv.org/abs/2310.15444v1)  

---


**ABSTRACT**  
Adversarial training has shown promise in building robust models against adversarial examples. A major drawback of adversarial training is the computational overhead introduced by the generation of adversarial examples. To overcome this limitation, adversarial training based on single-step attacks has been explored. Previous work improves the single-step adversarial training from different perspectives, e.g., sample initialization, loss regularization, and training strategy. Almost all of them treat the underlying model as a black box. In this work, we propose to exploit the interior building blocks of the model to improve efficiency. Specifically, we propose to dynamically sample lightweight subnetworks as a surrogate model during training. By doing this, both the forward and backward passes can be accelerated for efficient adversarial training. Besides, we provide theoretical analysis to show the model robustness can be improved by the single-step adversarial training with sampled subnetworks. Furthermore, we propose a novel sampling strategy where the sampling varies from layer to layer and from iteration to iteration. Compared with previous methods, our method not only reduces the training cost but also achieves better model robustness. Evaluations on a series of popular datasets demonstrate the effectiveness of the proposed FB-Better. Our code has been released at https://github.com/jiaxiaojunQAQ/FP-Better.

{{</citation>}}


## eess.IV (4)



### (135/178) G-CASCADE: Efficient Cascaded Graph Convolutional Decoding for 2D Medical Image Segmentation (Md Mostafijur Rahman et al., 2023)

{{<citation>}}

Md Mostafijur Rahman, Radu Marculescu. (2023)  
**G-CASCADE: Efficient Cascaded Graph Convolutional Decoding for 2D Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: I-4; J-3, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.16175v1)  

---


**ABSTRACT**  
In recent years, medical image segmentation has become an important application in the field of computer-aided diagnosis. In this paper, we are the first to propose a new graph convolution-based decoder namely, Cascaded Graph Convolutional Attention Decoder (G-CASCADE), for 2D medical image segmentation. G-CASCADE progressively refines multi-stage feature maps generated by hierarchical transformer encoders with an efficient graph convolution block. The encoder utilizes the self-attention mechanism to capture long-range dependencies, while the decoder refines the feature maps preserving long-range information due to the global receptive fields of the graph convolution block. Rigorous evaluations of our decoder with multiple transformer encoders on five medical image segmentation tasks (i.e., Abdomen organs, Cardiac organs, Polyp lesions, Skin lesions, and Retinal vessels) show that our model outperforms other state-of-the-art (SOTA) methods. We also demonstrate that our decoder achieves better DICE scores than the SOTA CASCADE decoder with 80.8% fewer parameters and 82.3% fewer FLOPs. Our decoder can easily be used with other hierarchical encoders for general-purpose semantic and medical image segmentation tasks.

{{</citation>}}


### (136/178) YOLO-Angio: An Algorithm for Coronary Anatomy Segmentation (Tom Liu et al., 2023)

{{<citation>}}

Tom Liu, Hui Lin, Aggelos K. Katsaggelos, Adrienne Kline. (2023)  
**YOLO-Angio: An Algorithm for Coronary Anatomy Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15898v1)  

---


**ABSTRACT**  
Coronary angiography remains the gold standard for diagnosis of coronary artery disease, the most common cause of death worldwide. While this procedure is performed more than 2 million times annually, there remain few methods for fast and accurate automated measurement of disease and localization of coronary anatomy. Here, we present our solution to the Automatic Region-based Coronary Artery Disease diagnostics using X-ray angiography images (ARCADE) challenge held at MICCAI 2023. For the artery segmentation task, our three-stage approach combines preprocessing and feature selection by classical computer vision to enhance vessel contrast, followed by an ensemble model based on YOLOv8 to propose possible vessel candidates by generating a vessel map. A final segmentation is based on a logic-based approach to reconstruct the coronary tree in a graph-based sorting method. Our entry to the ARCADE challenge placed 3rd overall. Using the official metric for evaluation, we achieved an F1 score of 0.422 and 0.4289 on the validation and hold-out sets respectively.

{{</citation>}}


### (137/178) Unpaired MRI Super Resolution with Self-Supervised Contrastive Learning (Hao Li et al., 2023)

{{<citation>}}

Hao Li, Quanwei Liu, Jianan Liu, Xiling Liu, Yanni Dong, Tao Huang, Zhihan Lv. (2023)  
**Unpaired MRI Super Resolution with Self-Supervised Contrastive Learning**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Contrastive Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.15767v1)  

---


**ABSTRACT**  
High-resolution (HR) magnetic resonance imaging (MRI) is crucial for enhancing diagnostic accuracy in clinical settings. Nonetheless, the inherent limitation of MRI resolution restricts its widespread applicability. Deep learning-based image super-resolution (SR) methods exhibit promise in improving MRI resolution without additional cost. However, these methods frequently require a substantial number of HR MRI images for training, which can be challenging to acquire. In this paper, we propose an unpaired MRI SR approach that employs self-supervised contrastive learning to enhance SR performance with limited training data. Our approach leverages both authentic HR images and synthetically generated SR images to construct positive and negative sample pairs, thus facilitating the learning of discriminative features. Empirical results presented in this study underscore significant enhancements in the peak signal-to-noise ratio and structural similarity index, even when a paucity of HR images is available. These findings accentuate the potential of our approach in addressing the challenge of limited training data, thereby contributing to the advancement of high-resolution MRI in clinical applications.

{{</citation>}}


### (138/178) Deep Learning Models for Classification of COVID-19 Cases by Medical Images (Amir Ali, 2023)

{{<citation>}}

Amir Ali. (2023)  
**Deep Learning Models for Classification of COVID-19 Cases by Medical Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.16851v1)  

---


**ABSTRACT**  
In recent times, the use of chest Computed Tomography (CT) images for detecting coronavirus infections has gained significant attention, owing to their ability to reveal bilateral changes in affected individuals. However, classifying patients from medical images presents a formidable challenge, particularly in identifying such bilateral changes. To tackle this challenge, our study harnesses the power of deep learning models for the precise classification of infected patients. Our research involves a comparative analysis of deep transfer learning-based classification models, including DenseNet201, GoogleNet, and AlexNet, against carefully chosen supervised learning models. Additionally, our work encompasses Covid-19 classification, which involves the identification and differentiation of medical images, such as X-rays and electrocardiograms, that exhibit telltale signs of Covid-19 infection. This comprehensive approach ensures that our models can handle a wide range of medical image types and effectively identify characteristic patterns indicative of Covid-19. By conducting meticulous research and employing advanced deep learning techniques, we have made significant strides in enhancing the accuracy and speed of Covid-19 diagnosis. Our results demonstrate the effectiveness of these models and their potential to make substantial contributions to the global effort to combat COVID-19.

{{</citation>}}


## cs.HC (5)



### (139/178) Conversational Challenges in AI-Powered Data Science: Obstacles, Needs, and Design Opportunities (Bhavya Chopra et al., 2023)

{{<citation>}}

Bhavya Chopra, Ananya Singha, Anna Fariha, Sumit Gulwani, Chris Parnin, Ashish Tiwari, Austin Z. Henley. (2023)  
**Conversational Challenges in AI-Powered Data Science: Obstacles, Needs, and Design Opportunities**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.16164v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are being increasingly employed in data science for tasks like data preprocessing and analytics. However, data scientists encounter substantial obstacles when conversing with LLM-powered chatbots and acting on their suggestions and answers. We conducted a mixed-methods study, including contextual observations, semi-structured interviews (n=14), and a survey (n=114), to identify these challenges. Our findings highlight key issues faced by data scientists, including contextual data retrieval, formulating prompts for complex tasks, adapting generated code to local environments, and refining prompts iteratively. Based on these insights, we propose actionable design recommendations, such as data brushing to support context selection, and inquisitive feedback loops to improve communications with AI-based assistants in data-science tools.

{{</citation>}}


### (140/178) Facilitating Self-Guided Mental Health Interventions Through Human-Language Model Interaction: A Case Study of Cognitive Restructuring (Ashish Sharma et al., 2023)

{{<citation>}}

Ashish Sharma, Kevin Rushton, Inna Wanyin Lin, Theresa Nguyen, Tim Althoff. (2023)  
**Facilitating Self-Guided Mental Health Interventions Through Human-Language Model Interaction: A Case Study of Cognitive Restructuring**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15461v1)  

---


**ABSTRACT**  
Self-guided mental health interventions, such as "do-it-yourself" tools to learn and practice coping strategies, show great promise to improve access to mental health care. However, these interventions are often cognitively demanding and emotionally triggering, creating accessibility barriers that limit their wide-scale implementation and adoption. In this paper, we study how human-language model interaction can support self-guided mental health interventions. We take cognitive restructuring, an evidence-based therapeutic technique to overcome negative thinking, as a case study. In an IRB-approved randomized field study on a large mental health website with 15,531 participants, we design and evaluate a system that uses language models to support people through various steps of cognitive restructuring. Our findings reveal that our system positively impacts emotional intensity for 67% of participants and helps 65% overcome negative thoughts. Although adolescents report relatively worse outcomes, we find that tailored interventions that simplify language model generations improve overall effectiveness and equity.

{{</citation>}}


### (141/178) UI Layout Generation with LLMs Guided by UI Grammar (Yuwen Lu et al., 2023)

{{<citation>}}

Yuwen Lu, Ziang Tong, Qinyi Zhao, Chengzhi Zhang, Toby Jia-Jun Li. (2023)  
**UI Layout Generation with LLMs Guided by UI Grammar**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.15455v1)  

---


**ABSTRACT**  
The recent advances in Large Language Models (LLMs) have stimulated interest among researchers and industry professionals, particularly in their application to tasks concerning mobile user interfaces (UIs). This position paper investigates the use of LLMs for UI layout generation. Central to our exploration is the introduction of UI grammar -- a novel approach we proposed to represent the hierarchical structure inherent in UI screens. The aim of this approach is to guide the generative capacities of LLMs more effectively and improve the explainability and controllability of the process. Initial experiments conducted with GPT-4 showed the promising capability of LLMs to produce high-quality user interfaces via in-context learning. Furthermore, our preliminary comparative study suggested the potential of the grammar-based approach in improving the quality of generative results in specific aspects.

{{</citation>}}


### (142/178) PromptInfuser: How Tightly Coupling AI and UI Design Impacts Designers' Workflows (Savvas Petridis et al., 2023)

{{<citation>}}

Savvas Petridis, Michael Terry, Carrie J. Cai. (2023)  
**PromptInfuser: How Tightly Coupling AI and UI Design Impacts Designers' Workflows**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15435v1)  

---


**ABSTRACT**  
Prototyping AI applications is notoriously difficult. While large language model (LLM) prompting has dramatically lowered the barriers to AI prototyping, designers are still prototyping AI functionality and UI separately. We investigate how coupling prompt and UI design affects designers' workflows. Grounding this research, we developed PromptInfuser, a Figma plugin that enables users to create semi-functional mockups, by connecting UI elements to the inputs and outputs of prompts. In a study with 14 designers, we compare PromptInfuser to designers' current AI-prototyping workflow. PromptInfuser was perceived to be significantly more useful for communicating product ideas, more capable of producing prototypes that realistically represent the envisioned artifact, more efficient for prototyping, and more helpful for anticipating UI issues and technical constraints. PromptInfuser encouraged iteration over prompt and UI together, which helped designers identify UI and prompt incompatibilities and reflect upon their total solution. Together, these findings inform future systems for prototyping AI applications.

{{</citation>}}


### (143/178) ConstitutionMaker: Interactively Critiquing Large Language Models by Converting Feedback into Principles (Savvas Petridis et al., 2023)

{{<citation>}}

Savvas Petridis, Ben Wedin, James Wexler, Aaron Donsbach, Mahima Pushkarna, Nitesh Goyal, Carrie J. Cai, Michael Terry. (2023)  
**ConstitutionMaker: Interactively Critiquing Large Language Models by Converting Feedback into Principles**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15428v1)  

---


**ABSTRACT**  
Large language model (LLM) prompting is a promising new approach for users to create and customize their own chatbots. However, current methods for steering a chatbot's outputs, such as prompt engineering and fine-tuning, do not support users in converting their natural feedback on the model's outputs to changes in the prompt or model. In this work, we explore how to enable users to interactively refine model outputs through their feedback, by helping them convert their feedback into a set of principles (i.e. a constitution) that dictate the model's behavior. From a formative study, we (1) found that users needed support converting their feedback into principles for the chatbot and (2) classified the different principle types desired by users. Inspired by these findings, we developed ConstitutionMaker, an interactive tool for converting user feedback into principles, to steer LLM-based chatbots. With ConstitutionMaker, users can provide either positive or negative feedback in natural language, select auto-generated feedback, or rewrite the chatbot's response; each mode of feedback automatically generates a principle that is inserted into the chatbot's prompt. In a user study with 14 participants, we compare ConstitutionMaker to an ablated version, where users write their own principles. With ConstitutionMaker, participants felt that their principles could better guide the chatbot, that they could more easily convert their feedback into principles, and that they could write principles more efficiently, with less mental demand. ConstitutionMaker helped users identify ways to improve the chatbot, formulate their intuitive responses to the model into feedback, and convert this feedback into specific and clear principles. Together, these findings inform future tools that support the interactive critiquing of LLM outputs.

{{</citation>}}


## cs.CR (4)



### (144/178) FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering (Md Rafi Ur Rashid et al., 2023)

{{<citation>}}

Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz. (2023)  
**FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.16152v1)  

---


**ABSTRACT**  
Federated learning (FL) is becoming a key component in many technology-based applications including language modeling -- where individual FL participants often have privacy-sensitive text data in their local datasets. However, realizing the extent of privacy leakage in federated language models is not straightforward and the existing attacks only intend to extract data regardless of how sensitive or naive it is. To fill this gap, in this paper, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other user in FL even without any cooperation from the server. Our best-performing method improves the membership inference recall by 29% and achieves up to 70% private data reconstruction, evidently outperforming existing attacks with stronger assumptions of adversary capabilities.

{{</citation>}}


### (145/178) Facial Data Minimization: Shallow Model as Your Privacy Filter (Yuwen Pu et al., 2023)

{{<citation>}}

Yuwen Pu, Jiahao Chen, Jiayu Pan, Hao li, Diqun Yan, Xuhong Zhang, Shouling Ji. (2023)  
**Facial Data Minimization: Shallow Model as Your Privacy Filter**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15590v1)  

---


**ABSTRACT**  
Face recognition service has been used in many fields and brings much convenience to people. However, once the user's facial data is transmitted to a service provider, the user will lose control of his/her private data. In recent years, there exist various security and privacy issues due to the leakage of facial data. Although many privacy-preserving methods have been proposed, they usually fail when they are not accessible to adversaries' strategies or auxiliary data. Hence, in this paper, by fully considering two cases of uploading facial images and facial features, which are very typical in face recognition service systems, we proposed a data privacy minimization transformation (PMT) method. This method can process the original facial data based on the shallow model of authorized services to obtain the obfuscated data. The obfuscated data can not only maintain satisfactory performance on authorized models and restrict the performance on other unauthorized models but also prevent original privacy data from leaking by AI methods and human visual theft. Additionally, since a service provider may execute preprocessing operations on the received data, we also propose an enhanced perturbation method to improve the robustness of PMT. Besides, to authorize one facial image to multiple service models simultaneously, a multiple restriction mechanism is proposed to improve the scalability of PMT. Finally, we conduct extensive experiments and evaluate the effectiveness of the proposed PMT in defending against face reconstruction, data abuse, and face attribute estimation attacks. These experimental results demonstrate that PMT performs well in preventing facial data abuse and privacy leakage while maintaining face recognition accuracy.

{{</citation>}}


### (146/178) Non-Fungible Token Security (Ryleigh McKinney et al., 2023)

{{<citation>}}

Ryleigh McKinney, Sundar Krishnan. (2023)  
**Non-Fungible Token Security**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.15518v1)  

---


**ABSTRACT**  
Non-fungible tokens (NFTs) are unique digital assets stored on the blockchain and is used to certify ownership and authenticity of the digital asset. NFTs were first created in 2014 while their popularity peaked between 2021 and 2022. In this paper, the authors dive into the world of Non-Fungible Tokens (NFTs), their history, the Future of NFTs, as well as the security concerns.

{{</citation>}}


### (147/178) The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks (Xiaoyi Chen et al., 2023)

{{<citation>}}

Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, XiaoFeng Wang, Haixu Tang. (2023)  
**The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2310.15469v1)  

---


**ABSTRACT**  
The era post-2018 marked the advent of Large Language Models (LLMs), with innovations such as OpenAI's ChatGPT showcasing prodigious linguistic prowess. As the industry galloped toward augmenting model parameters and capitalizing on vast swaths of human language data, security and privacy challenges also emerged. Foremost among these is the potential inadvertent accrual of Personal Identifiable Information (PII) during web-based data acquisition, posing risks of unintended PII disclosure. While strategies like RLHF during training and Catastrophic Forgetting have been marshaled to control the risk of privacy infringements, recent advancements in LLMs, epitomized by OpenAI's fine-tuning interface for GPT-3.5, have reignited concerns. One may ask: can the fine-tuning of LLMs precipitate the leakage of personal information embedded within training datasets? This paper reports the first endeavor to seek the answer to the question, particularly our discovery of a new LLM exploitation avenue, called the Janus attack. In the attack, one can construct a PII association task, whereby an LLM is fine-tuned using a minuscule PII dataset, to potentially reinstate and reveal concealed PIIs. Our findings indicate that, with a trivial fine-tuning outlay, LLMs such as GPT-3.5 can transition from being impermeable to PII extraction to a state where they divulge a substantial proportion of concealed PII. This research, through its deep dive into the Janus attack vector, underscores the imperative of navigating the intricate interplay between LLM utility and privacy preservation.

{{</citation>}}


## cs.IR (5)



### (148/178) Clinfo.ai: An Open-Source Retrieval-Augmented Large Language Model System for Answering Medical Questions using Scientific Literature (Alejandro Lozano et al., 2023)

{{<citation>}}

Alejandro Lozano, Scott L Fleming, Chia-Chun Chiang, Nigam Shah. (2023)  
**Clinfo.ai: An Open-Source Retrieval-Augmented Large Language Model System for Answering Medical Questions using Scientific Literature**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs.IR  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.16146v1)  

---


**ABSTRACT**  
The quickly-expanding nature of published medical literature makes it challenging for clinicians and researchers to keep up with and summarize recent, relevant findings in a timely manner. While several closed-source summarization tools based on large language models (LLMs) now exist, rigorous and systematic evaluations of their outputs are lacking. Furthermore, there is a paucity of high-quality datasets and appropriate benchmark tasks with which to evaluate these tools. We address these issues with four contributions: we release Clinfo.ai, an open-source WebApp that answers clinical questions based on dynamically retrieved scientific literature; we specify an information retrieval and abstractive summarization task to evaluate the performance of such retrieval-augmented LLM systems; we release a dataset of 200 questions and corresponding answers derived from published systematic reviews, which we name PubMed Retrieval and Synthesis (PubMedRS-200); and report benchmark results for Clinfo.ai and other publicly available OpenQA systems on PubMedRS-200.

{{</citation>}}


### (149/178) Context-aware explainable recommendations over knowledge graphs (Jinfeng Zhong et al., 2023)

{{<citation>}}

Jinfeng Zhong, Elsa Negre. (2023)  
**Context-aware explainable recommendations over knowledge graphs**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Graph Convolutional Network, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.16141v1)  

---


**ABSTRACT**  
Knowledge graphs contain rich semantic relationships related to items and incorporating such semantic relationships into recommender systems helps to explore the latent connections of items, thus improving the accuracy of prediction and enhancing the explainability of recommendations. However, such explainability is not adapted to users' contexts, which can significantly influence their preferences. In this work, we propose CA-KGCN (Context-Aware Knowledge Graph Convolutional Network), an end-to-end framework that can model users' preferences adapted to their contexts and can incorporate rich semantic relationships in the knowledge graph related to items. This framework captures users' attention to different factors: contexts and features of items. More specifically, the framework can model users' preferences adapted to their contexts and provide explanations adapted to the given context. Experiments on three real-world datasets show the effectiveness of our framework: modeling users' preferences adapted to their contexts and explaining the recommendations generated.

{{</citation>}}


### (150/178) Representation Learning with Large Language Models for Recommendation (Xubin Ren et al., 2023)

{{<citation>}}

Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang. (2023)  
**Representation Learning with Large Language Models for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.15950v1)  

---


**ABSTRACT**  
Recommender systems have seen significant advancements with the influence of deep learning and graph neural networks, particularly in capturing complex user-item relationships. However, these graph-based recommenders heavily depend on ID-based data, potentially disregarding valuable textual information associated with users and items, resulting in less informative learned representations. Moreover, the utilization of implicit feedback data introduces potential noise and bias, posing challenges for the effectiveness of user preference learning. While the integration of large language models (LLMs) into traditional ID-based recommenders has gained attention, challenges such as scalability issues, limitations in text-only reliance, and prompt input constraints need to be addressed for effective implementation in practical recommender systems. To address these challenges, we propose a model-agnostic framework RLMRec that aims to enhance existing recommenders with LLM-empowered representation learning. It proposes a recommendation paradigm that integrates representation learning with LLMs to capture intricate semantic aspects of user behaviors and preferences. RLMRec incorporates auxiliary textual signals, develops a user/item profiling paradigm empowered by LLMs, and aligns the semantic space of LLMs with the representation space of collaborative relational signals through a cross-view alignment framework. This work further establish a theoretical foundation demonstrating that incorporating textual signals through mutual information maximization enhances the quality of representations. In our evaluation, we integrate RLMRec with state-of-the-art recommender models, while also analyzing its efficiency and robustness to noise data. Our implementation codes are available at https://github.com/HKUDS/RLMRec.

{{</citation>}}


### (151/178) Topology-aware Debiased Self-supervised Graph Learning for Recommendation (Lei Han et al., 2023)

{{<citation>}}

Lei Han, Hui Yan, Zhicheng Qiao. (2023)  
**Topology-aware Debiased Self-supervised Graph Learning for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.15858v1)  

---


**ABSTRACT**  
In recommendation, graph-based Collaborative Filtering (CF) methods mitigate the data sparsity by introducing Graph Contrastive Learning (GCL). However, the random negative sampling strategy in these GCL-based CF models neglects the semantic structure of users (items), which not only introduces false negatives (negatives that are similar to anchor user (item)) but also ignores the potential positive samples. To tackle the above issues, we propose Topology-aware Debiased Self-supervised Graph Learning (TDSGL) for recommendation, which constructs contrastive pairs according to the semantic similarity between users (items). Specifically, since the original user-item interaction data commendably reflects the purchasing intent of users and certain characteristics of items, we calculate the semantic similarity between users (items) on interaction data. Then, given a user (item), we construct its negative pairs by selecting users (items) which embed different semantic structures to ensure the semantic difference between the given user (item) and its negatives. Moreover, for a user (item), we design a feature extraction module that converts other semantically similar users (items) into an auxiliary positive sample to acquire a more informative representation. Experimental results show that the proposed model outperforms the state-of-the-art models significantly on three public datasets. Our model implementation codes are available at https://github.com/malajikuai/TDSGL.

{{</citation>}}


### (152/178) Robust Representation Learning for Unified Online Top-K Recommendation (Minfang Lu et al., 2023)

{{<citation>}}

Minfang Lu, Yuchen Jiang, Huihui Dong, Qi Li, Ziru Xu, Yuanlin Liu, Lixia Wu, Haoyuan Hu, Han Zhu, Yuning Jiang, Jian Xu, Bo Zheng. (2023)  
**Robust Representation Learning for Unified Online Top-K Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.15492v1)  

---


**ABSTRACT**  
In large-scale industrial e-commerce, the efficiency of an online recommendation system is crucial in delivering highly relevant item/content advertising that caters to diverse business scenarios. However, most existing studies focus solely on item advertising, neglecting the significance of content advertising. This oversight results in inconsistencies within the multi-entity structure and unfair retrieval. Furthermore, the challenge of retrieving top-k advertisements from multi-entity advertisements across different domains adds to the complexity. Recent research proves that user-entity behaviors within different domains exhibit characteristics of differentiation and homogeneity. Therefore, the multi-domain matching models typically rely on the hybrid-experts framework with domain-invariant and domain-specific representations. Unfortunately, most approaches primarily focus on optimizing the combination mode of different experts, failing to address the inherent difficulty in optimizing the expert modules themselves. The existence of redundant information across different domains introduces interference and competition among experts, while the distinct learning objectives of each domain lead to varying optimization challenges among experts. To tackle these issues, we propose robust representation learning for the unified online top-k recommendation. Our approach constructs unified modeling in entity space to ensure data fairness. The robust representation learning employs domain adversarial learning and multi-view wasserstein distribution learning to learn robust representations. Moreover, the proposed method balances conflicting objectives through the homoscedastic uncertainty weights and orthogonality constraints. Various experiments validate the effectiveness and rationality of our proposed method, which has been successfully deployed online to serve real business scenarios.

{{</citation>}}


## stat.AP (1)



### (153/178) Analyzing Disparity and Temporal Progression of Internet Quality through Crowdsourced Measurements with Bias-Correction (Hyeongseong Lee et al., 2023)

{{<citation>}}

Hyeongseong Lee, Udit Paul, Arpit Gupta, Elizabeth Belding, Mengyang Gu. (2023)  
**Analyzing Disparity and Temporal Progression of Internet Quality through Crowdsourced Measurements with Bias-Correction**  

---
Primary Category: stat.AP  
Categories: cs-NI, stat-AP, stat.AP  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.16136v1)  

---


**ABSTRACT**  
Crowdsourced speedtest measurements are an important tool for studying internet performance from the end user perspective. Nevertheless, despite the accuracy of individual measurements, simplistic aggregation of these data points is problematic due to their intrinsic sampling bias. In this work, we utilize a dataset of nearly 1 million individual Ookla Speedtest measurements, correlate each datapoint with 2019 Census demographic data, and develop new methods to present a novel analysis to quantify regional sampling bias and the relationship of internet performance to demographic profile. We find that the crowdsourced Ookla Speedtest data points contain significant sampling bias across different census block groups based on a statistical test of homogeneity. We introduce two methods to correct the regional bias by the population of each census block group. Whereas the sampling bias leads to a small discrepancy in the overall cumulative distribution function of internet speed in a city between estimation from original samples and bias-corrected estimation, the discrepancy is much smaller compared to the size of the sampling heterogeneity across regions. Further, we show that the sampling bias is strongly associated with a few demographic variables, such as income, education level, age, and ethnic distribution. Through regression analysis, we find that regions with higher income, younger populations, and lower representation of Hispanic residents tend to measure faster internet speeds along with substantial collinearity amongst socioeconomic attributes and ethnic composition. Finally, we find that average internet speed increases over time based on both linear and nonlinear analysis from state space models, though the regional sampling bias may result in a small overestimation of the temporal increase of internet speed.

{{</citation>}}


## cs.SD (3)



### (154/178) Complex Image Generation SwinTransformer Network for Audio Denoising (Youshan Zhang et al., 2023)

{{<citation>}}

Youshan Zhang, Jialu Li. (2023)  
**Complex Image Generation SwinTransformer Network for Audio Denoising**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.16109v1)  

---


**ABSTRACT**  
Achieving high-performance audio denoising is still a challenging task in real-world applications. Existing time-frequency methods often ignore the quality of generated frequency domain images. This paper converts the audio denoising problem into an image generation task. We first develop a complex image generation SwinTransformer network to capture more information from the complex Fourier domain. We then impose structure similarity and detailed loss functions to generate high-quality images and develop an SDR loss to minimize the difference between denoised and clean audios. Extensive experiments on two benchmark datasets demonstrate that our proposed model is better than state-of-the-art methods.

{{</citation>}}


### (155/178) CDSD: Chinese Dysarthria Speech Database (Mengyi Sun et al., 2023)

{{<citation>}}

Mengyi Sun, Ming Gao, Xinchen Kang, Shiru Wang, Jun Du, Dengfeng Yao, Su-Jing Wang. (2023)  
**CDSD: Chinese Dysarthria Speech Database**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15930v1)  

---


**ABSTRACT**  
We present the Chinese Dysarthria Speech Database (CDSD) as a valuable resource for dysarthria research. This database comprises speech data from 24 participants with dysarthria. Among these participants, one recorded an additional 10 hours of speech data, while each recorded one hour, resulting in 34 hours of speech material. To accommodate participants with varying cognitive levels, our text pool primarily consists of content from the AISHELL-1 dataset and speeches by primary and secondary school students. When participants read these texts, they must use a mobile device or the ZOOM F8n multi-track field recorder to record their speeches. In this paper, we elucidate the data collection and annotation processes and present an approach for establishing a baseline for dysarthric speech recognition. Furthermore, we conducted a speaker-dependent dysarthric speech recognition experiment using an additional 10 hours of speech data from one of our participants. Our research findings indicate that, through extensive data-driven model training, fine-tuning limited quantities of specific individual data yields commendable results in speaker-dependent dysarthric speech recognition. However, we observe significant variations in recognition results among different dysarthric speakers. These insights provide valuable reference points for speaker-dependent dysarthric speech recognition.

{{</citation>}}


### (156/178) Dynamic Convolutional Neural Networks as Efficient Pre-trained Audio Models (Florian Schmid et al., 2023)

{{<citation>}}

Florian Schmid, Khaled Koutini, Gerhard Widmer. (2023)  
**Dynamic Convolutional Neural Networks as Efficient Pre-trained Audio Models**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Knowledge Distillation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.15648v1)  

---


**ABSTRACT**  
The introduction of large-scale audio datasets, such as AudioSet, paved the way for Transformers to conquer the audio domain and replace CNNs as the state-of-the-art neural network architecture for many tasks. Audio Spectrogram Transformers are excellent at exploiting large datasets, creating powerful pre-trained models that surpass CNNs when fine-tuned on downstream tasks. However, current popular Audio Spectrogram Transformers are demanding in terms of computational complexity compared to CNNs. Recently, we have shown that, by employing Transformer-to-CNN Knowledge Distillation, efficient CNNs can catch up with and even outperform Transformers on large datasets. In this work, we extend this line of research and increase the capacity of efficient CNNs by introducing dynamic CNN blocks, constructed of dynamic non-linearities, dynamic convolutions and attention mechanisms. We show that these dynamic CNNs outperform traditional efficient CNNs, in terms of the performance-complexity trade-off and parameter efficiency, at the task of audio tagging on the large-scale AudioSet. Our experiments further indicate that the introduced dynamic CNNs achieve better performance on downstream tasks and scale up well, attaining Transformer performance and even outperforming them on AudioSet and several downstream tasks.

{{</citation>}}


## cs.AI (4)



### (157/178) AI Alignment and Social Choice: Fundamental Limitations and Policy Implications (Abhilash Mishra, 2023)

{{<citation>}}

Abhilash Mishra. (2023)  
**AI Alignment and Social Choice: Fundamental Limitations and Policy Implications**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CY, cs-HC, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.16048v1)  

---


**ABSTRACT**  
Aligning AI agents to human intentions and values is a key bottleneck in building safe and deployable AI applications. But whose values should AI agents be aligned with? Reinforcement learning with human feedback (RLHF) has emerged as the key framework for AI alignment. RLHF uses feedback from human reinforcers to fine-tune outputs; all widely deployed large language models (LLMs) use RLHF to align their outputs to human values. It is critical to understand the limitations of RLHF and consider policy challenges arising from these limitations. In this paper, we investigate a specific challenge in building RLHF systems that respect democratic norms. Building on impossibility results in social choice theory, we show that, under fairly broad assumptions, there is no unique voting protocol to universally align AI systems using RLHF through democratic processes. Further, we show that aligning AI agents with the values of all individuals will always violate certain private ethical preferences of an individual user i.e., universal AI alignment using RLHF is impossible. We discuss policy implications for the governance of AI systems built using RLHF: first, the need for mandating transparent voting rules to hold model builders accountable. Second, the need for model builders to focus on developing AI agents that are narrowly aligned to specific user groups.

{{</citation>}}


### (158/178) Random Entity Quantization for Parameter-Efficient Compositional Knowledge Graph Representation (Jiaang Li et al., 2023)

{{<citation>}}

Jiaang Li, Quan Wang, Yi Liu, Licheng Zhang, Zhendong Mao. (2023)  
**Random Entity Quantization for Parameter-Efficient Compositional Knowledge Graph Representation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Embedding, Knowledge Graph, Quantization, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.15797v1)  

---


**ABSTRACT**  
Representation Learning on Knowledge Graphs (KGs) is essential for downstream tasks. The dominant approach, KG Embedding (KGE), represents entities with independent vectors and faces the scalability challenge. Recent studies propose an alternative way for parameter efficiency, which represents entities by composing entity-corresponding codewords matched from predefined small-scale codebooks. We refer to the process of obtaining corresponding codewords of each entity as entity quantization, for which previous works have designed complicated strategies. Surprisingly, this paper shows that simple random entity quantization can achieve similar results to current strategies. We analyze this phenomenon and reveal that entity codes, the quantization outcomes for expressing entities, have higher entropy at the code level and Jaccard distance at the codeword level under random entity quantization. Therefore, different entities become more easily distinguished, facilitating effective KG representation. The above results show that current quantization strategies are not critical for KG representation, and there is still room for improvement in entity distinguishability beyond current strategies. The code to reproduce our results is available at https://github.com/JiaangL/RandomQuantization.

{{</citation>}}


### (159/178) Emergent Communication in Interactive Sketch Question Answering (Zixing Lei et al., 2023)

{{<citation>}}

Zixing Lei, Yiming Zhang, Yuxin Xiong, Siheng Chen. (2023)  
**Emergent Communication in Interactive Sketch Question Answering**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: QA, Question Answering, Sketch  
[Paper Link](http://arxiv.org/abs/2310.15597v1)  

---


**ABSTRACT**  
Vision-based emergent communication (EC) aims to learn to communicate through sketches and demystify the evolution of human communication. Ironically, previous works neglect multi-round interaction, which is indispensable in human communication. To fill this gap, we first introduce a novel Interactive Sketch Question Answering (ISQA) task, where two collaborative players are interacting through sketches to answer a question about an image in a multi-round manner. To accomplish this task, we design a new and efficient interactive EC system, which can achieve an effective balance among three evaluation factors, including the question answering accuracy, drawing complexity and human interpretability. Our experimental results including human evaluation demonstrate that multi-round interactive mechanism facilitates targeted and efficient communication between intelligent agents with decent human interpretability.

{{</citation>}}


### (160/178) Diverse Conventions for Human-AI Collaboration (Bidipta Sarkar et al., 2023)

{{<citation>}}

Bidipta Sarkar, Andy Shih, Dorsa Sadigh. (2023)  
**Diverse Conventions for Human-AI Collaboration**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15414v1)  

---


**ABSTRACT**  
Conventions are crucial for strong performance in cooperative multi-agent games, because they allow players to coordinate on a shared strategy without explicit communication. Unfortunately, standard multi-agent reinforcement learning techniques, such as self-play, converge to conventions that are arbitrary and non-diverse, leading to poor generalization when interacting with new partners. In this work, we present a technique for generating diverse conventions by (1) maximizing their rewards during self-play, while (2) minimizing their rewards when playing with previously discovered conventions (cross-play), stimulating conventions to be semantically different. To ensure that learned policies act in good faith despite the adversarial optimization of cross-play, we introduce \emph{mixed-play}, where an initial state is randomly generated by sampling self-play and cross-play transitions and the player learns to maximize the self-play reward from this initial state. We analyze the benefits of our technique on various multi-agent collaborative games, including Overcooked, and find that our technique can adapt to the conventions of humans, surpassing human-level performance when paired with real users.

{{</citation>}}


## cs.SE (6)



### (161/178) White-box Compiler Fuzzing Empowered by Large Language Models (Chenyuan Yang et al., 2023)

{{<citation>}}

Chenyuan Yang, Yinlin Deng, Runyu Lu, Jiayi Yao, Jiawei Liu, Reyhaneh Jabbarvand, Lingming Zhang. (2023)  
**White-box Compiler Fuzzing Empowered by Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-PL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.15991v1)  

---


**ABSTRACT**  
Compiler correctness is crucial, as miscompilation falsifying the program behaviors can lead to serious consequences. In the literature, fuzzing has been extensively studied to uncover compiler defects. However, compiler fuzzing remains challenging: Existing arts focus on black- and grey-box fuzzing, which generates tests without sufficient understanding of internal compiler behaviors. As such, they often fail to construct programs to exercise conditions of intricate optimizations. Meanwhile, traditional white-box techniques are computationally inapplicable to the giant codebase of compilers. Recent advances demonstrate that Large Language Models (LLMs) excel in code generation/understanding tasks and have achieved state-of-the-art performance in black-box fuzzing. Nonetheless, prompting LLMs with compiler source-code information remains a missing piece of research in compiler testing.   To this end, we propose WhiteFox, the first white-box compiler fuzzer using LLMs with source-code information to test compiler optimization. WhiteFox adopts a dual-model framework: (i) an analysis LLM examines the low-level optimization source code and produces requirements on the high-level test programs that can trigger the optimization; (ii) a generation LLM produces test programs based on the summarized requirements. Additionally, optimization-triggering tests are used as feedback to further enhance the test generation on the fly. Our evaluation on four popular compilers shows that WhiteFox can generate high-quality tests to exercise deep optimizations requiring intricate conditions, practicing up to 80 more optimizations than state-of-the-art fuzzers. To date, WhiteFox has found in total 96 bugs, with 80 confirmed as previously unknown and 51 already fixed. Beyond compiler testing, WhiteFox can also be adapted for white-box fuzzing of other complex, real-world software systems in general.

{{</citation>}}


### (162/178) Characterizing Issue Management in Runtime Systems (Salma Begum Tamanna et al., 2023)

{{<citation>}}

Salma Begum Tamanna, Gias Uddin, Lan Xia, Longyu Zhang. (2023)  
**Characterizing Issue Management in Runtime Systems**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2310.15971v1)  

---


**ABSTRACT**  
Modern programming languages like Java require runtime systems to support the implementation and deployment of software applications in diverse computing platforms and operating systems. These runtime systems are normally developed in GitHub-hosted repositories based on close collaboration between large software companies (e.g., IBM, Microsoft) and OSS developers. However, despite their popularity and broad usage; to the best of our knowledge, these repositories have never been studied. We report an empirical study of around 118K issues from 34 runtime system repos in GitHub. We found that issues regarding enhancement, test failure and bug are mostly posted on runtime system repositories and solution related discussion are mostly present on issue discussion. 82.69% issues in the runtime system repositories have been resolved and 0.69% issues are ignored; median of issue close rate, ignore rate and addressing time in these repositories are 76.1%, 2.2% and 58 days respectively. 82.65% issues are tagged with labels while only 28.30% issues have designated assignees and 90.65% issues contain at least one comment; also presence of these features in an issue report can affect issue closure. Based on the findings, we offer six recommendat

{{</citation>}}


### (163/178) Make LLM a Testing Expert: Bringing Human-like Interaction to Mobile GUI Testing via Functionality-aware Decisions (Zhe Liu et al., 2023)

{{<citation>}}

Zhe Liu, Chunyang Chen, Junjie Wang, Mengzhuo Chen, Boyu Wu, Xing Che, Dandan Wang, Qing Wang. (2023)  
**Make LLM a Testing Expert: Bringing Human-like Interaction to Mobile GUI Testing via Functionality-aware Decisions**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2310.15780v1)  

---


**ABSTRACT**  
Automated Graphical User Interface (GUI) testing plays a crucial role in ensuring app quality, especially as mobile applications have become an integral part of our daily lives. Despite the growing popularity of learning-based techniques in automated GUI testing due to their ability to generate human-like interactions, they still suffer from several limitations, such as low testing coverage, inadequate generalization capabilities, and heavy reliance on training data. Inspired by the success of Large Language Models (LLMs) like ChatGPT in natural language understanding and question answering, we formulate the mobile GUI testing problem as a Q&A task. We propose GPTDroid, asking LLM to chat with the mobile apps by passing the GUI page information to LLM to elicit testing scripts, and executing them to keep passing the app feedback to LLM, iterating the whole process. Within this framework, we have also introduced a functionality-aware memory prompting mechanism that equips the LLM with the ability to retain testing knowledge of the whole process and conduct long-term, functionality-based reasoning to guide exploration. We evaluate it on 93 apps from Google Play and demonstrate that it outperforms the best baseline by 32% in activity coverage, and detects 31% more bugs at a faster rate. Moreover, GPTDroid identify 53 new bugs on Google Play, of which 35 have been confirmed and fixed.

{{</citation>}}


### (164/178) Testing the Limits: Unusual Text Inputs Generation for Mobile App Crash Detection with Large Language Model (Zhe Liu et al., 2023)

{{<citation>}}

Zhe Liu, Chunyang Chen, Junjie Wang, Mengzhuo Chen, Boyu Wu, Xing Che, Dandan Wang, Qing Wang. (2023)  
**Testing the Limits: Unusual Text Inputs Generation for Mobile App Crash Detection with Large Language Model**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Google, Language Model  
[Paper Link](http://arxiv.org/abs/2310.15657v1)  

---


**ABSTRACT**  
Mobile applications have become a ubiquitous part of our daily life, providing users with access to various services and utilities. Text input, as an important interaction channel between users and applications, plays an important role in core functionality such as search queries, authentication, messaging, etc. However, certain special text (e.g., -18 for Font Size) can cause the app to crash, and generating diversified unusual inputs for fully testing the app is highly demanded. Nevertheless, this is also challenging due to the combination of explosion dilemma, high context sensitivity, and complex constraint relations. This paper proposes InputBlaster which leverages the LLM to automatically generate unusual text inputs for mobile app crash detection. It formulates the unusual inputs generation problem as a task of producing a set of test generators, each of which can yield a batch of unusual text inputs under the same mutation rule. In detail, InputBlaster leverages LLM to produce the test generators together with the mutation rules serving as the reasoning chain, and utilizes the in-context learning schema to demonstrate the LLM with examples for boosting the performance. InputBlaster is evaluated on 36 text input widgets with cash bugs involving 31 popular Android apps, and results show that it achieves 78% bug detection rate, with 136% higher than the best baseline. Besides, we integrate it with the automated GUI testing tool and detect 37 unseen crashes in real-world apps from Google Play.

{{</citation>}}


### (165/178) Navigating ICT In-House Procurement in Finland: Evaluating Legal Frameworks and Practical Challenges (Reetta Ghezzi et al., 2023)

{{<citation>}}

Reetta Ghezzi, Minnamaria Korhonen, Hannu Vilpponen, Tommi Mikkonen. (2023)  
**Navigating ICT In-House Procurement in Finland: Evaluating Legal Frameworks and Practical Challenges**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2310.15643v1)  

---


**ABSTRACT**  
In-house procurement is a controversial issue in the field of public procurement. Simply put, such procurement allows overlooking certain aspects of fair and equal treatment of vendors. This paper presents qualitative research on in-house ICT procurement within Finnish municipalities. Semi-structured interviews were conducted to gather insights from municipal stakeholders. Using grounded theory approach, data analysis shows intricate dynamics between Finnish municipalities and in-house entities associated with them. Still, it is clear that the legal framework governing in-house procurement remains intricate and debated.

{{</citation>}}


### (166/178) VGX: Large-Scale Sample Generation for Boosting Learning-Based Software Vulnerability Analyses (Yu Nong et al., 2023)

{{<citation>}}

Yu Nong, Richard Fang, Guangbei Yi, Kunsong Zhao, Xiapu Luo, Feng Chen, Haipeng Cai. (2023)  
**VGX: Large-Scale Sample Generation for Boosting Learning-Based Software Vulnerability Analyses**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GNN, Transformer  
[Paper Link](http://arxiv.org/abs/2310.15436v1)  

---


**ABSTRACT**  
Accompanying the successes of learning-based defensive software vulnerability analyses is the lack of large and quality sets of labeled vulnerable program samples, which impedes further advancement of those defenses. Existing automated sample generation approaches have shown potentials yet still fall short of practical expectations due to the high noise in the generated samples. This paper proposes VGX, a new technique aimed for large-scale generation of high-quality vulnerability datasets. Given a normal program, VGX identifies the code contexts in which vulnerabilities can be injected, using a customized Transformer featured with a new value-flowbased position encoding and pre-trained against new objectives particularly for learning code structure and context. Then, VGX materializes vulnerability-injection code editing in the identified contexts using patterns of such edits obtained from both historical fixes and human knowledge about real-world vulnerabilities. Compared to four state-of-the-art (SOTA) baselines (pattern-, Transformer-, GNN-, and pattern+Transformer-based), VGX achieved 99.09-890.06% higher F1 and 22.45%-328.47% higher label accuracy. For in-the-wild sample production, VGX generated 150,392 vulnerable samples, from which we randomly chose 10% to assess how much these samples help vulnerability detection, localization, and repair. Our results show SOTA techniques for these three application tasks achieved 19.15-330.80% higher F1, 12.86-19.31% higher top-10 accuracy, and 85.02-99.30% higher top-50 accuracy, respectively, by adding those samples to their original training data. These samples also helped a SOTA vulnerability detector discover 13 more real-world vulnerabilities (CVEs) in critical systems (e.g., Linux kernel) that would be missed by the original model.

{{</citation>}}


## cs.CE (1)



### (167/178) A Roadmap of Emerging Trends Discovery in Hydrology: A Topic Modeling Approach (Sila Ovgu Korkut et al., 2023)

{{<citation>}}

Sila Ovgu Korkut, Oznur Oztunc Kaymak, Aytug Onan, Erman Ulker, Femin Yalcin. (2023)  
**A Roadmap of Emerging Trends Discovery in Hydrology: A Topic Modeling Approach**  

---
Primary Category: cs.CE  
Categories: E-0; I-7; J-2, cs-CE, cs.CE  
Keywords: Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2310.15943v1)  

---


**ABSTRACT**  
In the new global era, determining trends can play an important role in guiding researchers, scientists, and agencies. The main faced challenge is to track the emerging topics among the stacked publications. Therefore, any study done to propose the trend topics in a field to foresee upcoming subjects is crucial. In the current study, the trend topics in the field of "Hydrology" have been attempted to evaluate. To do so, the model is composed of three key components: a gathering of data, preprocessing of the article's significant features, and determining trend topics. Various topic models including Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF), and Latent Semantic Analysis (LSA) have been implemented. Comparing the obtained results with respect to the $C_V$ coherence score, in 2022, the topics of "Climate change", "River basin", "Water management", "Natural hazards/erosion", and "Hydrologic cycle" have been obtained. According to a further analysis, it is shown that these topics keep their impact on the field in 2023, as well.

{{</citation>}}


## cs.PL (1)



### (168/178) CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code (Tong Ye et al., 2023)

{{<citation>}}

Tong Ye, Lingfei Wu, Tengfei Ma, Xuhong Zhang, Yangkai Du, Peiyu Liu, Shouling Ji, Wenhai Wang. (2023)  
**CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code**  

---
Primary Category: cs.PL  
Categories: cs-AI, cs-PL, cs.PL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2310.16853v1)  

---


**ABSTRACT**  
Automatically generating function summaries for binaries is an extremely valuable but challenging task, since it involves translating the execution behavior and semantics of the low-level language (assembly code) into human-readable natural language. However, most current works on understanding assembly code are oriented towards generating function names, which involve numerous abbreviations that make them still confusing. To bridge this gap, we focus on generating complete summaries for binary functions, especially for stripped binary (no symbol table and debug information in reality). To fully exploit the semantics of assembly code, we present a control flow graph and pseudo code guided binary code summarization framework called CP-BCS. CP-BCS utilizes a bidirectional instruction-level control flow graph and pseudo code that incorporates expert knowledge to learn the comprehensive binary function execution behavior and logic semantics. We evaluate CP-BCS on 3 different binary optimization levels (O1, O2, and O3) for 3 different computer architectures (X86, X64, and ARM). The evaluation results demonstrate CP-BCS is superior and significantly improves the efficiency of reverse engineering.

{{</citation>}}


## cs.CY (1)



### (169/178) A Novel Method for Analysing Racial Bias: Collection of Person Level References (Muhammed Yusuf Kocyigit et al., 2023)

{{<citation>}}

Muhammed Yusuf Kocyigit, Anietie Andy, Derry Wijaya. (2023)  
**A Novel Method for Analysing Racial Bias: Collection of Person Level References**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Bias, Google  
[Paper Link](http://arxiv.org/abs/2310.15847v1)  

---


**ABSTRACT**  
Long term exposure to biased content in literature or media can significantly influence people's perceptions of reality, leading to the development of implicit biases that are difficult to detect and address (Gerbner 1998). In this study, we propose a novel method to analyze the differences in representation between two groups and use it examine the representation of African Americans and White Americans in books between 1850 to 2000 with the Google Books dataset (Goldberg and Orwant 2013). By developing better tools to understand differences in representation, we aim to contribute to the ongoing efforts to recognize and mitigate biases. To improve upon the more common phrase based (men, women, white, black, etc) methods to differentiate context (Tripodi et al. 2019, Lucy; Tadimeti, and Bamman 2022), we propose collecting a comprehensive list of historically significant figures and using their names to select relevant context. This novel approach offers a more accurate and nuanced method for detecting implicit biases through reducing the risk of selection bias. We create group representations for each decade and analyze them in an aligned semantic space (Hamilton, Leskovec, and Jurafsky 2016). We further support our results by assessing the time adjusted toxicity (Bassignana, Basile, and Patti 2018) in the context for each group and identifying the semantic axes (Lucy, Tadimeti, and Bamman 2022) that exhibit the most significant differences between the groups across decades. We support our method by showing that our proposed method can capture known socio political changes accurately and our findings indicate that while the relative number of African American names mentioned in books have increased over time, the context surrounding them remains more toxic than white Americans.

{{</citation>}}


## math.DS (1)



### (170/178) Nonlinear dimensionality reduction then and now: AIMs for dissipative PDEs in the ML era (Eleni D. Koronaki et al., 2023)

{{<citation>}}

Eleni D. Koronaki, Nikolaos Evangelou, Cristina P. Martin-Linares, Edriss S. Titi, Ioannis G. Kevrekidis. (2023)  
**Nonlinear dimensionality reduction then and now: AIMs for dissipative PDEs in the ML era**  

---
Primary Category: math.DS  
Categories: cs-LG, math-DS, math.DS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.15816v1)  

---


**ABSTRACT**  
This study presents a collection of purely data-driven workflows for constructing reduced-order models (ROMs) for distributed dynamical systems. The ROMs we focus on, are data-assisted models inspired by, and templated upon, the theory of Approximate Inertial Manifolds (AIMs); the particular motivation is the so-called post-processing Galerkin method of Garcia-Archilla, Novo and Titi. Its applicability can be extended: the need for accurate truncated Galerkin projections and for deriving closed-formed corrections can be circumvented using machine learning tools. When the right latent variables are not a priori known, we illustrate how autoencoders as well as Diffusion Maps (a manifold learning scheme) can be used to discover good sets of latent variables and test their explainability. The proposed methodology can express the ROMs in terms of (a) theoretical (Fourier coefficients), (b) linear data-driven (POD modes) and/or (c) nonlinear data-driven (Diffusion Maps) coordinates. Both Black-Box and (theoretically-informed and data-corrected) Gray-Box models are described; the necessity for the latter arises when truncated Galerkin projections are so inaccurate as to not be amenable to post-processing. We use the Chafee-Infante reaction-diffusion and the Kuramoto-Sivashinsky dissipative partial differential equations to illustrate and successfully test the overall framework.

{{</citation>}}


## cs.SI (1)



### (171/178) Causal Understanding of Why Users Share Hate Speech on Social Media (Dominique Geissler et al., 2023)

{{<citation>}}

Dominique Geissler, Abdurahman Maarouf, Stefan Feuerriegel. (2023)  
**Causal Understanding of Why Users Share Hate Speech on Social Media**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-CY, cs-SI, cs.SI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2310.15772v1)  

---


**ABSTRACT**  
Hate speech on social media threatens the mental and physical well-being of individuals and is further responsible for real-world violence. An important driver behind the spread of hate speech and thus why hateful posts can go viral are reshares, yet little is known about why users reshare hate speech. In this paper, we present a comprehensive, causal analysis of the user attributes that make users reshare hate speech. However, causal inference from observational social media data is challenging, because such data likely suffer from selection bias, and there is further confounding due to differences in the vulnerability of users to hate speech. We develop a novel, three-step causal framework: (1) We debias the observational social media data by applying inverse propensity scoring. (2) We use the debiased propensity scores to model the latent vulnerability of users to hate speech as a latent embedding. (3) We model the causal effects of user attributes on users' probability of sharing hate speech, while controlling for the latent vulnerability of users to hate speech. Compared to existing baselines, a particular strength of our framework is that it models causal effects that are non-linear, yet still explainable. We find that users with fewer followers, fewer friends, and fewer posts share more hate speech. Younger accounts, in return, share less hate speech. Overall, understanding the factors that drive users to share hate speech is crucial for detecting individuals at risk of engaging in harmful behavior and for designing effective mitigation strategies.

{{</citation>}}


## cs.DC (1)



### (172/178) SharkGraph: A Time Series Distributed Graph System (Derong Tang, 2023)

{{<citation>}}

Derong Tang. (2023)  
**SharkGraph: A Time Series Distributed Graph System**  

---
Primary Category: cs.DC  
Categories: cs-DB, cs-DC, cs.DC  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.15762v1)  

---


**ABSTRACT**  
Current graph systems can easily process billions of data, however when increased to exceed hundred billions, the performance decreases dramatically, time series data always be very huge, consequently computation on time series graphs still remains challenging nowadays. In current piece of work, we introduces SharkGraph, a (distributed file system) DFS-based time series graph system, used a novel storage structure (Time Series Graph Data File) TGF, By reading file stream to iterate graph computation, SharkGraph is able to execute batch graph query, simulation, data mining, or clustering algorithm on exceed hundred billions edge size industry graph. Through well defined experiments that shows SharkGraph performs well on large-scale graph processing, also can support time traversal for graphs, and recover state at any position in the timeline. By repeating experiments reported for existing distributed systems like GraphX, we demonstrate that SharkGraph can easily handle hundreds billions of data, rather than GraphX which met many problems such as memory issues and skewed distribution on graph traversal. Compared with other graph systems SharkGraph uses less memory and more efficiently to process the same graph.

{{</citation>}}


## cs.IT (1)



### (173/178) Semantic-preserving image coding based on Conditional Diffusion models (Francesco Pezone et al., 2023)

{{<citation>}}

Francesco Pezone, Osman Musa, Giuseppe Caire, Sergio Barbarossa. (2023)  
**Semantic-preserving image coding based on Conditional Diffusion models**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.15737v1)  

---


**ABSTRACT**  
Semantic communication, rather than on a bit-by-bit recovery of the transmitted messages, focuses on the meaning and the goal of the communication itself. In this paper, we propose a novel semantic image coding scheme that preserves the semantic content of an image, while ensuring a good trade-off between coding rate and image quality. The proposed Semantic-Preserving Image Coding based on Conditional Diffusion Models (SPIC) transmitter encodes a Semantic Segmentation Map (SSM) and a low-resolution version of the image to be transmitted. The receiver then reconstructs a high-resolution image using a Denoising Diffusion Probabilistic Models (DDPM) doubly conditioned to the SSM and the low-resolution image. As shown by the numerical examples, compared to state-of-the-art (SOTA) approaches, the proposed SPIC exhibits a better balance between the conventional rate-distortion trade-off and the preservation of semantically-relevant features.

{{</citation>}}


## stat.ML (1)



### (174/178) Causal Representation Learning Made Identifiable by Grouping of Observational Variables (Hiroshi Morioka et al., 2023)

{{<citation>}}

Hiroshi Morioka, Aapo Hyvärinen. (2023)  
**Causal Representation Learning Made Identifiable by Grouping of Observational Variables**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.15709v1)  

---


**ABSTRACT**  
A topic of great current interest is Causal Representation Learning (CRL), whose goal is to learn a causal model for hidden features in a data-driven manner. Unfortunately, CRL is severely ill-posed since it is a combination of the two notoriously ill-posed problems of representation learning and causal discovery. Yet, finding practical identifiability conditions that guarantee a unique solution is crucial for its practical applicability. Most approaches so far have been based on assumptions on the latent causal mechanisms, such as temporal causality, or existence of supervision or interventions; these can be too restrictive in actual applications. Here, we show identifiability based on novel, weak constraints, which requires no temporal structure, intervention, nor weak supervision. The approach is based assuming the observational mixing exhibits a suitable grouping of the observational variables. We also propose a novel self-supervised estimation framework consistent with the model, prove its statistical consistency, and experimentally show its superior CRL performances compared to the state-of-the-art baselines. We further demonstrate its robustness against latent confounders and causal cycles.

{{</citation>}}


## cs.RO (3)



### (175/178) DACOOP-A: Decentralized Adaptive Cooperative Pursuit via Attention (Zheng Zhang et al., 2023)

{{<citation>}}

Zheng Zhang, Dengyu Zhan, Qingrui Zhang, Wei Pan, Tianjiang Hu. (2023)  
**DACOOP-A: Decentralized Adaptive Cooperative Pursuit via Attention**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.15699v1)  

---


**ABSTRACT**  
Integrating rule-based policies into reinforcement learning promises to improve data efficiency and generalization in cooperative pursuit problems. However, most implementations do not properly distinguish the influence of neighboring robots in observation embedding or inter-robot interaction rules, leading to information loss and inefficient cooperation. This paper proposes a cooperative pursuit algorithm named Decentralized Adaptive COOperative Pursuit via Attention (DACOOP-A) by empowering reinforcement learning with artificial potential field and attention mechanisms. An attention-based framework is developed to emphasize important neighbors by concurrently integrating the learned attention scores into observation embedding and inter-robot interaction rules. A KL divergence regularization is introduced to alleviate the resultant learning stability issue. Improvements in data efficiency and generalization are demonstrated through numerical simulations. Extensive quantitative analysis and ablation studies are performed to illustrate the advantages of the proposed modules. Real-world experiments are performed to justify the feasibility of deploying DACOOP-A in physical systems.

{{</citation>}}


### (176/178) tagE: Enabling an Embodied Agent to Understand Human Instructions (Chayan Sarkar et al., 2023)

{{<citation>}}

Chayan Sarkar, Avik Mitra, Pradip Pramanick, Tapas Nayak. (2023)  
**tagE: Enabling an Embodied Agent to Understand Human Instructions**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: NLU  
[Paper Link](http://arxiv.org/abs/2310.15605v1)  

---


**ABSTRACT**  
Natural language serves as the primary mode of communication when an intelligent agent with a physical presence engages with human beings. While a plethora of research focuses on natural language understanding (NLU), encompassing endeavors such as sentiment analysis, intent prediction, question answering, and summarization, the scope of NLU directed at situations necessitating tangible actions by an embodied agent remains limited. The inherent ambiguity and incompleteness inherent in natural language present challenges for intelligent agents striving to decipher human intention. To tackle this predicament head-on, we introduce a novel system known as task and argument grounding for Embodied agents (tagE). At its core, our system employs an inventive neural network model designed to extract a series of tasks from complex task instructions expressed in natural language. Our proposed model adopts an encoder-decoder framework enriched with nested decoding to effectively extract tasks and their corresponding arguments from these intricate instructions. These extracted tasks are then mapped (or grounded) to the robot's established collection of skills, while the arguments find grounding in objects present within the environment. To facilitate the training and evaluation of our system, we have curated a dataset featuring complex instructions. The results of our experiments underscore the prowess of our approach, as it outperforms robust baseline models.

{{</citation>}}


### (177/178) Learning Agility and Adaptive Legged Locomotion via Curricular Hindsight Reinforcement Learning (Sicen Li et al., 2023)

{{<citation>}}

Sicen Li, Yiming Pang, Panju Bai, Zhaojin Liu, Jiawei Li, Shihao Hu, Liquan Wang, Gang Wang. (2023)  
**Learning Agility and Adaptive Legged Locomotion via Curricular Hindsight Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.15583v1)  

---


**ABSTRACT**  
Agile and adaptive maneuvers such as fall recovery, high-speed turning, and sprinting in the wild are challenging for legged systems. We propose a Curricular Hindsight Reinforcement Learning (CHRL) that learns an end-to-end tracking controller that achieves powerful agility and adaptation for the legged robot. The two key components are (I) a novel automatic curriculum strategy on task difficulty and (ii) a Hindsight Experience Replay strategy adapted to legged locomotion tasks. We demonstrated successful agile and adaptive locomotion on a real quadruped robot that performed fall recovery autonomously, coherent trotting, sustained outdoor speeds up to 3.45 m/s, and tuning speeds up to 3.2 rad/s. This system produces adaptive behaviours responding to changing situations and unexpected disturbances on natural terrains like grass and dirt.

{{</citation>}}


## cs.MM (1)



### (178/178) RecipeMeta: Metapath-enhanced Recipe Recommendation on Heterogeneous Recipe Network (Jialiang Shi et al., 2023)

{{<citation>}}

Jialiang Shi, Takahiro Komamizu, Keisuke Doman, Haruya Kyutoku, Ichiro Ide. (2023)  
**RecipeMeta: Metapath-enhanced Recipe Recommendation on Heterogeneous Recipe Network**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.15593v1)  

---


**ABSTRACT**  
Recipe is a set of instructions that describes how to make food. It can help people from the preparation of ingredients, food cooking process, etc. to prepare the food, and increasingly in demand on the Web. To help users find the vast amount of recipes on the Web, we address the task of recipe recommendation. Due to multiple data types and relationships in a recipe, we can treat it as a heterogeneous network to describe its information more accurately. To effectively utilize the heterogeneous network, metapath was proposed to describe the higher-level semantic information between two entities by defining a compound path from peer entities. Therefore, we propose a metapath-enhanced recipe recommendation framework, RecipeMeta, that combines GNN (Graph Neural Network)-based representation learning and specific metapath-based information in a recipe to predict User-Recipe pairs for recommendation. Through extensive experiments, we demonstrate that the proposed model, RecipeMeta, outperforms state-of-the-art methods for recipe recommendation.

{{</citation>}}
