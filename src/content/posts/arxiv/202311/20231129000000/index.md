---
draft: false
title: "arXiv @ 2023.11.29"
date: 2023-11-29
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.29"
    identifier: arxiv_20231129
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (27)](#cscl-27)
- [cs.CV (59)](#cscv-59)
- [cs.IR (4)](#csir-4)
- [cs.LG (18)](#cslg-18)
- [cs.CY (4)](#cscy-4)
- [eess.SY (3)](#eesssy-3)
- [quant-ph (2)](#quant-ph-2)
- [eess.IV (2)](#eessiv-2)
- [cs.RO (5)](#csro-5)
- [cs.DM (1)](#csdm-1)
- [cs.HC (2)](#cshc-2)
- [math.NA (1)](#mathna-1)
- [cs.CR (6)](#cscr-6)
- [q-bio.BM (1)](#q-biobm-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.AI (2)](#csai-2)
- [cs.NI (2)](#csni-2)
- [cs.GT (1)](#csgt-1)
- [cs.SD (1)](#cssd-1)
- [eess.AS (1)](#eessas-1)
- [cs.DL (1)](#csdl-1)
- [stat.ML (1)](#statml-1)
- [cs.SI (1)](#cssi-1)
- [cs.DC (1)](#csdc-1)

## cs.CL (27)



### (1/147) Reducing Gender Bias in Machine Translation through Counterfactual Data Generation (Ranjita Naik et al., 2023)

{{<citation>}}

Ranjita Naik, Spencer Rarrick, Vishal Chowdhary. (2023)  
**Reducing Gender Bias in Machine Translation through Counterfactual Data Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.16362v1)  

---


**ABSTRACT**  
Recent advances in neural methods have led to substantial improvement in the quality of Neural Machine Translation (NMT) systems. However, these systems frequently produce translations with inaccurate gender (Stanovsky et al., 2019), which can be traced to bias in training data. Saunders and Byrne (2020) tackle this problem with a handcrafted dataset containing balanced gendered profession words. By using this data to fine-tune an existing NMT model, they show that gender bias can be significantly mitigated, albeit at the expense of translation quality due to catastrophic forgetting. They recover some of the lost quality with modified training objectives or additional models at inference. We find, however, that simply supplementing the handcrafted dataset with a random sample from the base model training corpus is enough to significantly reduce the catastrophic forgetting. We also propose a novel domain-adaptation technique that leverages in-domain data created with the counterfactual data generation techniques proposed by Zmigrod et al. (2019) to further improve accuracy on the WinoMT challenge test set without significant loss in translation quality. We show its effectiveness in NMT systems from English into three morphologically rich languages French, Spanish, and Italian. The relevant dataset and code will be available at Github.

{{</citation>}}


### (2/147) Releasing the CRaQAn (Coreference Resolution in Question-Answering): An open-source dataset and dataset creation methodology using instruction-following models (Rob Grzywinski et al., 2023)

{{<citation>}}

Rob Grzywinski, Joshua D'Arcy, Rob Naidoff, Ashish Shukla, Alex Browne, Ren Gibbons, Brinnae Bent. (2023)  
**Releasing the CRaQAn (Coreference Resolution in Question-Answering): An open-source dataset and dataset creation methodology using instruction-following models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2311.16338v1)  

---


**ABSTRACT**  
Instruction-following language models demand robust methodologies for information retrieval to augment instructions for question-answering applications. A primary challenge is the resolution of coreferences in the context of chunking strategies for long documents. The critical barrier to experimentation of handling coreferences is a lack of open source datasets, specifically in question-answering tasks that require coreference resolution. In this work we present our Coreference Resolution in Question-Answering (CRaQAn) dataset, an open-source dataset that caters to the nuanced information retrieval requirements of coreference resolution in question-answering tasks by providing over 250 question-answer pairs containing coreferences. To develop this dataset, we developed a novel approach for creating high-quality datasets using an instruction-following model (GPT-4) and a Recursive Criticism and Improvement Loop.

{{</citation>}}


### (3/147) Applications of Large Language Models in Data Processing: Innovative Approaches to Segmenting and Renewing Information (Yu-Chen Lin et al., 2023)

{{<citation>}}

Yu-Chen Lin, Akhilesh Kumar, Wen-Liang Zhang, Norman Chang, Muhammad Zakir, Rucha Apte, Chao Wang, Jyh-Shing Roger Jang. (2023)  
**Applications of Large Language Models in Data Processing: Innovative Approaches to Segmenting and Renewing Information**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SE, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16267v1)  

---


**ABSTRACT**  
Our paper investigates effective methods for code generation in "specific-domain" applications, including the use of Large Language Models (LLMs) for data segmentation and renewal, as well as stimulating deeper thinking in LLMs through prompt adjustments. Using a real company product as an example, we provide user manuals, API documentation, and other data. The ideas discussed in this paper help segment and then convert this data into semantic vectors to better reflect their true positioning. Subsequently, user requirements are transformed into vectors to retrieve the most relevant content, achieving about 70% accuracy in simple to medium-complexity tasks through various prompt techniques. This paper is the first to enhance specific-domain code generation effectiveness from this perspective. Additionally, we experiment with generating more scripts from a limited number using llama2-based fine-tuning to test its effectiveness in professional domain code generation. This is a challenging and promising field, and once achieved, it will not only lead to breakthroughs in LLM development across multiple industries but also enable LLMs to understand and learn any new knowledge effectively.

{{</citation>}}


### (4/147) BERT Goes Off-Topic: Investigating the Domain Transfer Challenge using Genre Classification (Dmitri Roussinov et al., 2023)

{{<citation>}}

Dmitri Roussinov, Serge Sharoff. (2023)  
**BERT Goes Off-Topic: Investigating the Domain Transfer Challenge using Genre Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16083v1)  

---


**ABSTRACT**  
While performance of many text classification tasks has been recently improved due to Pre-trained Language Models (PLMs), in this paper we show that they still suffer from a performance gap when the underlying distribution of topics changes. For example, a genre classifier trained on \textit{political} topics often fails when tested on documents about \textit{sport} or \textit{medicine}. In this work, we quantify this phenomenon empirically with a large corpus and a large set of topics. Consequently, we verify that domain transfer remains challenging both for classic PLMs, such as BERT, and for modern large models, such as GPT-3. We also suggest and successfully test a possible remedy: after augmenting the training dataset with topically-controlled synthetic texts, the F1 score improves by up to 50\% for some topics, nearing on-topic training results, while others show little to no improvement. While our empirical results focus on genre classification, our methodology is applicable to other classification tasks such as gender, authorship, or sentiment classification. The code and data to replicate the experiments are available at https://github.com/dminus1/genre

{{</citation>}}


### (5/147) MEDITRON-70B: Scaling Medical Pretraining for Large Language Models (Zeming Chen et al., 2023)

{{<citation>}}

Zeming Chen, Alejandro Hernández Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas Köpf, Amirkeivan Mohtashami, Alexandre Sallinen, Alireza Sakhaeirad, Vinitra Swamy, Igor Krawczuk, Deniz Bayazit, Axel Marmet, Syrielle Montariol, Mary-Anne Hartley, Martin Jaggi, Antoine Bosselut. (2023)  
**MEDITRON-70B: Scaling Medical Pretraining for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2311.16079v1)  

---


**ABSTRACT**  
Large language models (LLMs) can potentially democratize access to medical knowledge. While many efforts have been made to harness and improve LLMs' medical knowledge and reasoning capacities, the resulting models are either closed-source (e.g., PaLM, GPT-4) or limited in scale (<= 13B parameters), which restricts their abilities. In this work, we improve access to large-scale medical LLMs by releasing MEDITRON: a suite of open-source LLMs with 7B and 70B parameters adapted to the medical domain. MEDITRON builds on Llama-2 (through our adaptation of Nvidia's Megatron-LM distributed trainer), and extends pretraining on a comprehensively curated medical corpus, including selected PubMed articles, abstracts, and internationally-recognized medical guidelines. Evaluations using four major medical benchmarks show significant performance gains over several state-of-the-art baselines before and after task-specific finetuning. Overall, MEDITRON achieves a 6% absolute performance gain over the best public baseline in its parameter class and 3% over the strongest baseline we finetuned from Llama-2. Compared to closed-source LLMs, MEDITRON-70B outperforms GPT-3.5 and Med-PaLM and is within 5% of GPT-4 and 10% of Med-PaLM-2. We release our code for curating the medical pretraining corpus and the MEDITRON model weights to drive open-source development of more capable medical LLMs.

{{</citation>}}


### (6/147) BioLORD-2023: Semantic Textual Representations Fusing LLM and Clinical Knowledge Graph Insights (François Remy et al., 2023)

{{<citation>}}

François Remy, Kris Demuynck, Thomas Demeester. (2023)  
**BioLORD-2023: Semantic Textual Representations Fusing LLM and Clinical Knowledge Graph Insights**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: Clinical, Knowledge Graph, Language Model, NLI  
[Paper Link](http://arxiv.org/abs/2311.16075v1)  

---


**ABSTRACT**  
In this study, we investigate the potential of Large Language Models to complement biomedical knowledge graphs in the training of semantic models for the biomedical and clinical domains. Drawing on the wealth of the UMLS knowledge graph and harnessing cutting-edge Large Language Models, we propose a new state-of-the-art approach for obtaining high-fidelity representations of biomedical concepts and sentences, consisting of three steps: an improved contrastive learning phase, a novel self-distillation phase, and a weight averaging phase. Through rigorous evaluations via the extensive BioLORD testing suite and diverse downstream tasks, we demonstrate consistent and substantial performance improvements over the previous state of the art (e.g. +2pts on MedSTS, +2.5pts on MedNLI-S, +6.1pts on EHR-Rel-B). Besides our new state-of-the-art biomedical model for English, we also distill and release a multilingual model compatible with 50+ languages and finetuned on 7 European languages. Many clinical pipelines can benefit from our latest models. Our new multilingual model enables a range of languages to benefit from our advancements in biomedical semantic representation learning, opening a new avenue for bioinformatics researchers around the world. As a result, we hope to see BioLORD-2023 becoming a precious tool for future biomedical applications.

{{</citation>}}


### (7/147) MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI (Xiang Yue et al., 2023)

{{<citation>}}

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen. (2023)  
**MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.16502v1)  

---


**ABSTRACT**  
We introduce MMMU: a new benchmark designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes 11.5K meticulously collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions span 30 subjects and 183 subfields, comprising 30 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures. Unlike existing benchmarks, MMMU focuses on advanced perception and reasoning with domain-specific knowledge, challenging models to perform tasks akin to those faced by experts. Our evaluation of 14 open-source LMMs and the proprietary GPT-4V(ision) highlights the substantial challenges posed by MMMU. Even the advanced GPT-4V only achieves a 56% accuracy, indicating significant room for improvement. We believe MMMU will stimulate the community to build next-generation multimodal foundation models towards expert artificial general intelligence.

{{</citation>}}


### (8/147) A Quantitative Approach to Understand Self-Supervised Models as Cross-lingual Feature Extractors (Shuyue Stella Li et al., 2023)

{{<citation>}}

Shuyue Stella Li, Beining Xu, Xiangyu Zhang, Hexin Liu, Wenhan Chao, Leibny Paola Garcia. (2023)  
**A Quantitative Approach to Understand Self-Supervised Models as Cross-lingual Feature Extractors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.15954v1)  

---


**ABSTRACT**  
In this work, we study the features extracted by English self-supervised learning (SSL) models in cross-lingual contexts and propose a new metric to predict the quality of feature representations. Using automatic speech recognition (ASR) as a downstream task, we analyze the effect of model size, training objectives, and model architecture on the models' performance as a feature extractor for a set of topologically diverse corpora. We develop a novel metric, the Phonetic-Syntax Ratio (PSR), to measure the phonetic and synthetic information in the extracted representations using deep generalized canonical correlation analysis. Results show the contrastive loss in the wav2vec2.0 objective facilitates more effective cross-lingual feature extraction. There is a positive correlation between PSR scores and ASR performance, suggesting that phonetic information extracted by monolingual SSL models can be used for downstream tasks in cross-lingual settings. The proposed metric is an effective indicator of the quality of the representations and can be useful for model selection.

{{</citation>}}


### (9/147) Leveraging deep active learning to identify low-resource mobility functioning information in public clinical notes (Tuan-Dung Le et al., 2023)

{{<citation>}}

Tuan-Dung Le, Zhuqi Miao, Samuel Alvarado, Brittany Smith, William Paiva, Thanh Thieu. (2023)  
**Leveraging deep active learning to identify low-resource mobility functioning information in public clinical notes**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Clinical, NER, NLP  
[Paper Link](http://arxiv.org/abs/2311.15946v1)  

---


**ABSTRACT**  
Function is increasingly recognized as an important indicator of whole-person health, although it receives little attention in clinical natural language processing research. We introduce the first public annotated dataset specifically on the Mobility domain of the International Classification of Functioning, Disability and Health (ICF), aiming to facilitate automatic extraction and analysis of functioning information from free-text clinical notes. We utilize the National NLP Clinical Challenges (n2c2) research dataset to construct a pool of candidate sentences using keyword expansion. Our active learning approach, using query-by-committee sampling weighted by density representativeness, selects informative sentences for human annotation. We train BERT and CRF models, and use predictions from these models to guide the selection of new sentences for subsequent annotation iterations. Our final dataset consists of 4,265 sentences with a total of 11,784 entities, including 5,511 Action entities, 5,328 Mobility entities, 306 Assistance entities, and 639 Quantification entities. The inter-annotator agreement (IAA), averaged over all entity types, is 0.72 for exact matching and 0.91 for partial matching. We also train and evaluate common BERT models and state-of-the-art Nested NER models. The best F1 scores are 0.84 for Action, 0.7 for Mobility, 0.62 for Assistance, and 0.71 for Quantification. Empirical results demonstrate promising potential of NER models to accurately extract mobility functioning information from clinical text. The public availability of our annotated dataset will facilitate further research to comprehensively capture functioning information in electronic health records (EHRs).

{{</citation>}}


### (10/147) Tell2Design: A Dataset for Language-Guided Floor Plan Generation (Sicong Leng et al., 2023)

{{<citation>}}

Sicong Leng, Yang Zhou, Mohammed Haroon Dupty, Wee Sun Lee, Sam Conrad Joyce, Wei Lu. (2023)  
**Tell2Design: A Dataset for Language-Guided Floor Plan Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Sequence-to-Sequence  
[Paper Link](http://arxiv.org/abs/2311.15941v1)  

---


**ABSTRACT**  
We consider the task of generating designs directly from natural language descriptions, and consider floor plan generation as the initial research area. Language conditional generative models have recently been very successful in generating high-quality artistic images. However, designs must satisfy different constraints that are not present in generating artistic images, particularly spatial and relational constraints. We make multiple contributions to initiate research on this task. First, we introduce a novel dataset, \textit{Tell2Design} (T2D), which contains more than $80k$ floor plan designs associated with natural language instructions. Second, we propose a Sequence-to-Sequence model that can serve as a strong baseline for future research. Third, we benchmark this task with several text-conditional image generation models. We conclude by conducting human evaluations on the generated samples and providing an analysis of human performance. We hope our contributions will propel the research on language-guided design generation forward.

{{</citation>}}


### (11/147) WorldSense: A Synthetic Benchmark for Grounded Reasoning in Large Language Models (Youssef Benchekroun et al., 2023)

{{<citation>}}

Youssef Benchekroun, Megi Dervishi, Mark Ibrahim, Jean-Baptiste Gaya, Xavier Martinet, Grégoire Mialon, Thomas Scialom, Emmanuel Dupoux, Dieuwke Hupkes, Pascal Vincent. (2023)  
**WorldSense: A Synthetic Benchmark for Grounded Reasoning in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.15930v1)  

---


**ABSTRACT**  
We propose WorldSense, a benchmark designed to assess the extent to which LLMs are consistently able to sustain tacit world models, by testing how they draw simple inferences from descriptions of simple arrangements of entities. Worldsense is a synthetic benchmark with three problem types, each with their own trivial control, which explicitly avoids bias by decorrelating the abstract structure of problems from the vocabulary and expressions, and by decorrelating all problem subparts with the correct response. We run our benchmark on three state-of-the-art chat-LLMs (GPT3.5, GPT4 and Llama2-chat) and show that these models make errors even with as few as three objects. Furthermore, they have quite heavy response biases, preferring certain responses irrespective of the question. Errors persist even with chain-of-thought prompting and in-context learning. Lastly, we show that while finetuning on similar problems does result in substantial improvements -- within- and out-of-distribution -- the finetuned models do not generalise beyond a constraint problem space.

{{</citation>}}


### (12/147) YUAN 2.0: A Large Language Model with Localized Filtering-based Attention (Shaohua Wu et al., 2023)

{{<citation>}}

Shaohua Wu, Xudong Zhao, Shenling Wang, Jiangang Luo, Lingjun Li, Xi Chen, Bing Zhao, Wei Wang, Tong Yu, Rongguo Zhang, Jiahua Zhang, Chao Wang. (2023)  
**YUAN 2.0: A Large Language Model with Localized Filtering-based Attention**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2311.15786v1)  

---


**ABSTRACT**  
In this work, the Localized Filtering-based Attention (LFA) is introduced to incorporate prior knowledge of local dependencies of natural language into Attention. Based on LFA, we develop and release Yuan 2.0, a large language model with parameters ranging from 2.1 billion to 102.6 billion. A data filtering and generation method is presented to build pretraining and fine-tuning dataset in high quality. A distributed training method with non-uniform pipeline parallel, data parallel, and optimizer parallel is proposed, which greatly reduces the bandwidth requirements of intra-node communication, and achieves good performance in large-scale distributed training. Yuan 2.0 models display impressive ability in code generation, math problem-solving, and chat compared with existing models. The latest version of YUAN 2.0, including model weights and source code, is accessible at Github.

{{</citation>}}


### (13/147) Towards Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs (Yunxin Li et al., 2023)

{{<citation>}}

Yunxin Li, Baotian Hu, Wei Wang, Xiaochun Cao, Min Zhang. (2023)  
**Towards Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.15759v1)  

---


**ABSTRACT**  
Recent advancements in multimodal large language models (MLLMs) have achieved significant multimodal generation capabilities, akin to GPT-4. These models predominantly map visual information into language representation space, leveraging the vast knowledge and powerful text generation abilities of LLMs to produce multimodal instruction-following responses. We could term this method as LLMs for Vision because of its employing LLMs for visual-language understanding, yet observe that these MLLMs neglect the potential of harnessing visual knowledge to enhance overall capabilities of LLMs, which could be regraded as Vision Enhancing LLMs. In this paper, we propose an approach called MKS2, aimed at enhancing LLMs through empowering Multimodal Knowledge Storage and Sharing in LLMs. Specifically, we introduce the Modular Visual Memory, a component integrated into the internal blocks of LLMs, designed to store open-world visual information efficiently. Additionally, we present a soft Mixtures-of-Multimodal Experts architecture in LLMs to invoke multimodal knowledge collaboration during generation. Our comprehensive experiments demonstrate that MKS2 substantially augments the reasoning capabilities of LLMs in contexts necessitating physical or commonsense knowledge. It also delivers competitive results on multimodal benchmarks.

{{</citation>}}


### (14/147) Italian Crossword Generator: Enhancing Education through Interactive Word Puzzles (Kamyar Zeinalipour et al., 2023)

{{<citation>}}

Kamyar Zeinalipour, Tommaso laquinta, Asya Zanollo, Giovanni Angelini, Leonardo Rigutini, Marco Maggini, Marco Gori. (2023)  
**Italian Crossword Generator: Enhancing Education through Interactive Word Puzzles**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, GPT  
[Paper Link](http://arxiv.org/abs/2311.15723v1)  

---


**ABSTRACT**  
Educational crosswords offer numerous benefits for students, including increased engagement, improved understanding, critical thinking, and memory retention. Creating high-quality educational crosswords can be challenging, but recent advances in natural language processing and machine learning have made it possible to use language models to generate nice wordplays. The exploitation of cutting-edge language models like GPT3-DaVinci, GPT3-Curie, GPT3-Babbage, GPT3-Ada, and BERT-uncased has led to the development of a comprehensive system for generating and verifying crossword clues. A large dataset of clue-answer pairs was compiled to fine-tune the models in a supervised manner to generate original and challenging clues from a given keyword. On the other hand, for generating crossword clues from a given text, Zero/Few-shot learning techniques were used to extract clues from the input text, adding variety and creativity to the puzzles. We employed the fine-tuned model to generate data and labeled the acceptability of clue-answer parts with human supervision. To ensure quality, we developed a classifier by fine-tuning existing language models on the labeled dataset. Conversely, to assess the quality of clues generated from the given text using zero/few-shot learning, we employed a zero-shot learning approach to check the quality of generated clues. The results of the evaluation have been very promising, demonstrating the effectiveness of the approach in creating high-standard educational crosswords that offer students engaging and rewarding learning experiences.

{{</citation>}}


### (15/147) Justifiable Artificial Intelligence: Engineering Large Language Models for Legal Applications (Sabine Wehnert, 2023)

{{<citation>}}

Sabine Wehnert. (2023)  
**Justifiable Artificial Intelligence: Engineering Large Language Models for Legal Applications**  

---
Primary Category: cs.CL  
Categories: H-4-2; H-3-3; H-5-2, cs-CL, cs-HC, cs-IR, cs.CL  
Keywords: Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2311.15716v1)  

---


**ABSTRACT**  
In this work, I discuss how Large Language Models can be applied in the legal domain, circumventing their current drawbacks. Despite their large success and acceptance, their lack of explainability hinders legal experts to trust in their output, and this happens rightfully so. However, in this paper, I argue in favor of a new view, Justifiable Artificial Intelligence, instead of focusing on Explainable Artificial Intelligence. I discuss in this paper how gaining evidence for and against a Large Language Model's output may make their generated texts more trustworthy - or hold them accountable for misinformation.

{{</citation>}}


### (16/147) Cerbero-7B: A Leap Forward in Language-Specific LLMs Through Enhanced Chat Corpus Generation and Evaluation (Federico A. Galatolo et al., 2023)

{{<citation>}}

Federico A. Galatolo, Mario G. C. A. Cimino. (2023)  
**Cerbero-7B: A Leap Forward in Language-Specific LLMs Through Enhanced Chat Corpus Generation and Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.15698v1)  

---


**ABSTRACT**  
This study introduces a novel approach for generating high-quality, language-specific chat corpora using a self-chat mechanism. We combine a generator LLM for creating new samples and an embedder LLM to ensure diversity. A new Masked Language Modelling (MLM) model-based quality assessment metric is proposed for evaluating and filtering the corpora. Utilizing the llama2-70b as the generator and a multilingual sentence transformer as embedder, we generate an Italian chat corpus and refine the Fauno corpus, which is based on translated English ChatGPT self-chat data. The refinement uses structural assertions and Natural Language Processing techniques. Both corpora undergo a comprehensive quality evaluation using the proposed MLM model-based quality metric. The Italian LLM fine-tuned with these corpora demonstrates significantly enhanced language comprehension and question-answering skills. The resultant model, cerbero-7b, establishes a new state-of-the-art for Italian LLMs. This approach marks a substantial advancement in the development of language-specific LLMs, with a special emphasis on augmenting corpora for underrepresented languages like Italian.

{{</citation>}}


### (17/147) Injecting linguistic knowledge into BERT for Dialogue State Tracking (Xiaohan Feng et al., 2023)

{{<citation>}}

Xiaohan Feng, Xixin Wu, Helen Meng. (2023)  
**Injecting linguistic knowledge into BERT for Dialogue State Tracking**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2311.15623v1)  

---


**ABSTRACT**  
Dialogue State Tracking (DST) models often employ intricate neural network architectures, necessitating substantial training data, and their inference processes lack transparency. This paper proposes a method that extracts linguistic knowledge via an unsupervised framework and subsequently utilizes this knowledge to augment BERT's performance and interpretability in DST tasks. The knowledge extraction procedure is computationally economical and does not necessitate annotations or additional training data. The injection of the extracted knowledge necessitates the addition of only simple neural modules. We employ the Convex Polytopic Model (CPM) as a feature extraction tool for DST tasks and illustrate that the acquired features correlate with the syntactic and semantic patterns in the dialogues. This correlation facilitates a comprehensive understanding of the linguistic features influencing the DST model's decision-making process. We benchmark this framework on various DST tasks and observe a notable improvement in accuracy.

{{</citation>}}


### (18/147) FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models (Ruixuan Xiao et al., 2023)

{{<citation>}}

Ruixuan Xiao, Yiwen Dong, Junbo Zhao, Runze Wu, Minmin Lin, Gang Chen, Haobo Wang. (2023)  
**FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Active Learning, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.15614v1)  

---


**ABSTRACT**  
Collecting high-quality labeled data for model training is notoriously time-consuming and labor-intensive for various NLP tasks. While copious solutions, such as active learning for small language models (SLMs) and prevalent in-context learning in the era of large language models (LLMs), have been proposed and alleviate the labeling burden to some extent, their performances are still subject to human intervention. It is still underexplored how to reduce the annotation cost in the LLMs era. To bridge this, we revolutionize traditional active learning and propose an innovative collaborative learning framework FreeAL to interactively distill and filter the task-specific knowledge from LLMs. During collaborative training, an LLM serves as an active annotator inculcating its coarse-grained knowledge, while a downstream SLM is incurred as a student to filter out high-quality in-context samples to feedback LLM for the subsequent label refinery. Extensive experiments on eight benchmark datasets demonstrate that FreeAL largely enhances the zero-shot performances for both SLM and LLM without any human supervision. The code is available at https://github.com/Justherozen/FreeAL .

{{</citation>}}


### (19/147) Evaluating the Efficacy of Hybrid Deep Learning Models in Distinguishing AI-Generated Text (Finbarrs Oketunji, 2023)

{{<citation>}}

Finbarrs Oketunji. (2023)  
**Evaluating the Efficacy of Hybrid Deep Learning Models in Distinguishing AI-Generated Text**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15565v1)  

---


**ABSTRACT**  
My research investigates the use of cutting-edge hybrid deep learning models to accurately differentiate between AI-generated text and human writing. I applied a robust methodology, utilising a carefully selected dataset comprising AI and human texts from various sources, each tagged with instructions. Advanced natural language processing techniques facilitated the analysis of textual features. Combining sophisticated neural networks, the custom model enabled it to detect nuanced differences between AI and human content.

{{</citation>}}


### (20/147) Boot and Switch: Alternating Distillation for Zero-Shot Dense Retrieval (Fan Jiang et al., 2023)

{{<citation>}}

Fan Jiang, Qiongkai Xu, Tom Drummond, Trevor Cohn. (2023)  
**Boot and Switch: Alternating Distillation for Zero-Shot Dense Retrieval**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.15564v1)  

---


**ABSTRACT**  
Neural 'dense' retrieval models are state of the art for many datasets, however these models often exhibit limited domain transfer ability. Existing approaches to adaptation are unwieldy, such as requiring explicit supervision, complex model architectures, or massive external models. We present $\texttt{ABEL}$, a simple but effective unsupervised method to enhance passage retrieval in zero-shot settings. Our technique follows a straightforward loop: a dense retriever learns from supervision signals provided by a reranker, and subsequently, the reranker is updated based on feedback from the improved retriever. By iterating this loop, the two components mutually enhance one another's performance. Experimental results demonstrate that our unsupervised $\texttt{ABEL}$ model outperforms both leading supervised and unsupervised retrievers on the BEIR benchmark. Meanwhile, it exhibits strong adaptation abilities to tasks and domains that were unseen during training. By either fine-tuning $\texttt{ABEL}$ on labelled data or integrating it with existing supervised dense retrievers, we achieve state-of-the-art results.\footnote{Source code is available at \url{https://github.com/Fantabulous-J/BootSwitch}.}

{{</citation>}}


### (21/147) Deficiency of Large Language Models in Finance: An Empirical Examination of Hallucination (Haoqiang Kang et al., 2023)

{{<citation>}}

Haoqiang Kang, Xiao-Yang Liu. (2023)  
**Deficiency of Large Language Models in Finance: An Empirical Examination of Hallucination**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, q-fin-ST  
Keywords: Augmentation, Language Model  
[Paper Link](http://arxiv.org/abs/2311.15548v1)  

---


**ABSTRACT**  
The hallucination issue is recognized as a fundamental deficiency of large language models (LLMs), especially when applied to fields such as finance, education, and law. Despite the growing concerns, there has been a lack of empirical investigation. In this paper, we provide an empirical examination of LLMs' hallucination behaviors in financial tasks. First, we empirically investigate LLM model's ability of explaining financial concepts and terminologies. Second, we assess LLM models' capacity of querying historical stock prices. Third, to alleviate the hallucination issue, we evaluate the efficacy of four practical methods, including few-shot learning, Decoding by Contrasting Layers (DoLa), the Retrieval Augmentation Generation (RAG) method and the prompt-based tool learning method for a function to generate a query command. Finally, our major finding is that off-the-shelf LLMs experience serious hallucination behaviors in financial tasks. Therefore, there is an urgent need to call for research efforts in mitigating LLMs' hallucination.

{{</citation>}}


### (22/147) The effect of source disclosure on evaluation of AI-generated messages: A two-part study (Sue Lim et al., 2023)

{{<citation>}}

Sue Lim, Ralf Schmälzle. (2023)  
**The effect of source disclosure on evaluation of AI-generated messages: A two-part study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.15544v2)  

---


**ABSTRACT**  
Advancements in artificial intelligence (AI) over the last decade demonstrate that machines can exhibit communicative behavior and influence how humans think, feel, and behave. In fact, the recent development of ChatGPT has shown that large language models (LLMs) can be leveraged to generate high-quality communication content at scale and across domains, suggesting that they will be increasingly used in practice. However, many questions remain about how knowing the source of the messages influences recipients' evaluation of and preference for AI-generated messages compared to human-generated messages. This paper investigated this topic in the context of vaping prevention messaging. In Study 1, which was pre-registered, we examined the influence of source disclosure on people's evaluation of AI-generated health prevention messages compared to human-generated messages. We found that source disclosure (i.e., labeling the source of a message as AI vs. human) significantly impacted the evaluation of the messages but did not significantly alter message rankings. In a follow-up study (Study 2), we examined how the influence of source disclosure may vary by the participants' negative attitudes towards AI. We found a significant moderating effect of negative attitudes towards AI on message evaluation, but not for message selection. However, for those with moderate levels of negative attitudes towards AI, source disclosure decreased the preference for AI-generated messages. Overall, the results of this series of studies showed a slight bias against AI-generated messages once the source was disclosed, adding to the emerging area of study that lies at the intersection of AI and communication.

{{</citation>}}


### (23/147) Overview of the VLSP 2022 -- Abmusu Shared Task: A Data Challenge for Vietnamese Abstractive Multi-document Summarization (Mai-Vu Tran et al., 2023)

{{<citation>}}

Mai-Vu Tran, Hoang-Quynh Le, Duy-Cat Can, Quoc-An Nguyen. (2023)  
**Overview of the VLSP 2022 -- Abmusu Shared Task: A Data Challenge for Vietnamese Abstractive Multi-document Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.15525v1)  

---


**ABSTRACT**  
This paper reports the overview of the VLSP 2022 - Vietnamese abstractive multi-document summarization (Abmusu) shared task for Vietnamese News. This task is hosted at the 9$^{th}$ annual workshop on Vietnamese Language and Speech Processing (VLSP 2022). The goal of Abmusu shared task is to develop summarization systems that could create abstractive summaries automatically for a set of documents on a topic. The model input is multiple news documents on the same topic, and the corresponding output is a related abstractive summary. In the scope of Abmusu shared task, we only focus on Vietnamese news summarization and build a human-annotated dataset of 1,839 documents in 600 clusters, collected from Vietnamese news in 8 categories. Participated models are evaluated and ranked in terms of \texttt{ROUGE2-F1} score, the most typical evaluation metric for document summarization problem.

{{</citation>}}


### (24/147) A Comparative and Experimental Study on Automatic Question Answering Systems and its Robustness against Word Jumbling (Shashidhar Reddy Javaji et al., 2023)

{{<citation>}}

Shashidhar Reddy Javaji, Haoran Hu, Sai Sameer Vennam, Vijaya Gajanan Buddhavarapu. (2023)  
**A Comparative and Experimental Study on Automatic Question Answering Systems and its Robustness against Word Jumbling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Natural Language Processing, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.15513v1)  

---


**ABSTRACT**  
Question answer generation using Natural Language Processing models is ubiquitous in the world around us. It is used in many use cases such as the building of chat bots, suggestive prompts in google search and also as a way of navigating information in banking mobile applications etc. It is highly relevant because a frequently asked questions (FAQ) list can only have a finite amount of questions but a model which can perform question answer generation could be able to answer completely new questions that are within the scope of the data. This helps us to be able to answer new questions accurately as long as it is a relevant question. In commercial applications, it can be used to increase customer satisfaction and ease of usage. However a lot of data is generated by humans so it is susceptible to human error and this can adversely affect the model's performance and we are investigating this through our work

{{</citation>}}


### (25/147) A Corpus for Named Entity Recognition in Chinese Novels with Multi-genres (Hanjie Zhao et al., 2023)

{{<citation>}}

Hanjie Zhao, Jinge Xie, Yuchen Yan, Yuxiang Jia, Yawen Ye, Hongying Zan. (2023)  
**A Corpus for Named Entity Recognition in Chinese Novels with Multi-genres**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2311.15509v1)  

---


**ABSTRACT**  
Entities like person, location, organization are important for literary text analysis. The lack of annotated data hinders the progress of named entity recognition (NER) in literary domain. To promote the research of literary NER, we build the largest multi-genre literary NER corpus containing 263,135 entities in 105,851 sentences from 260 online Chinese novels spanning 13 different genres. Based on the corpus, we investigate characteristics of entities from different genres. We propose several baseline NER models and conduct cross-genre and cross-domain experiments. Experimental results show that genre difference significantly impact NER performance though not as much as domain difference like literary domain and news domain. Compared with NER in news domain, literary NER still needs much improvement and the Out-of-Vocabulary (OOV) problem is more challenging due to the high variety of entities in literary works.

{{</citation>}}


### (26/147) Improving Word Sense Disambiguation in Neural Machine Translation with Salient Document Context (Elijah Rippeth et al., 2023)

{{<citation>}}

Elijah Rippeth, Marine Carpuat, Kevin Duh, Matt Post. (2023)  
**Improving Word Sense Disambiguation in Neural Machine Translation with Salient Document Context**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Machine Translation, Word Sense Disambiguation  
[Paper Link](http://arxiv.org/abs/2311.15507v1)  

---


**ABSTRACT**  
Lexical ambiguity is a challenging and pervasive problem in machine translation (\mt). We introduce a simple and scalable approach to resolve translation ambiguity by incorporating a small amount of extra-sentential context in neural \mt. Our approach requires no sense annotation and no change to standard model architectures. Since actual document context is not available for the vast majority of \mt training data, we collect related sentences for each input to construct pseudo-documents. Salient words from pseudo-documents are then encoded as a prefix to each source sentence to condition the generation of the translation. To evaluate, we release \docmucow, a challenge set for translation disambiguation based on the English-German \mucow \cite{raganato-etal-2020-evaluation} augmented with document IDs. Extensive experiments show that our method translates ambiguous source words better than strong sentence-level baselines and comparable document-level baselines while reducing training costs.

{{</citation>}}


### (27/147) Optimizing and Fine-tuning Large Language Model for Urban Renewal (Xi Wang et al., 2023)

{{<citation>}}

Xi Wang, Xianyao Ling, Tom Zhang, Xuecao Li, Shaolan Wang, Zhixing Li, Liang Zhang, Peng Gong. (2023)  
**Optimizing and Fine-tuning Large Language Model for Urban Renewal**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLM, Language Model, QA, Rouge  
[Paper Link](http://arxiv.org/abs/2311.15490v1)  

---


**ABSTRACT**  
This study aims to innovatively explore adaptive applications of large language models (LLM) in urban renewal. It also aims to improve its performance and text generation quality for knowledge question-answering (QA) tasks. Based on the ChatGLM, we automatically generate QA datasets using urban renewal scientific literature corpora in a self-instruct manner and then conduct joint fine-tuning training on the model using the Prefix and LoRA fine-tuning methods to create an LLM for urban renewal. By guiding the LLM to automatically generate QA data based on prompt words and given text, it is possible to quickly obtain datasets in the urban renewal field and provide data support for the fine-tuning training of LLMs. The experimental results show that the joint fine-tuning training method proposed in this study can significantly improve the performance of LLM on the QA tasks. Compared with LoRA fine-tuning, the method improves the Bleu and Rouge metrics on the test by about 5%; compared with the model before fine-tuning, the method improves the Bleu and Rouge metrics by about 15%-20%. This study demonstrates the effectiveness and superiority of the joint fine-tuning method using Prefix and LoRA for ChatGLM in the urban renewal knowledge QA tasks. It provides a new approach for fine-tuning LLMs on urban renewal-related tasks.

{{</citation>}}


## cs.CV (59)



### (28/147) Compositional Chain-of-Thought Prompting for Large Multimodal Models (Chancharik Mitra et al., 2023)

{{<citation>}}

Chancharik Mitra, Brandon Huang, Trevor Darrell, Roei Herzig. (2023)  
**Compositional Chain-of-Thought Prompting for Large Multimodal Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17076v1)  

---


**ABSTRACT**  
The combination of strong visual backbones and Large Language Model (LLM) reasoning has led to Large Multimodal Models (LMMs) becoming the current standard for a wide range of vision and language (VL) tasks. However, recent research has shown that even the most advanced LMMs still struggle to capture aspects of compositional visual reasoning, such as attributes and relationships between objects. One solution is to utilize scene graphs (SGs)--a formalization of objects and their relations and attributes that has been extensively used as a bridge between the visual and textual domains. Yet, scene graph data requires scene graph annotations, which are expensive to collect and thus not easily scalable. Moreover, finetuning an LMM based on SG data can lead to catastrophic forgetting of the pretraining objective. To overcome this, inspired by chain-of-thought methods, we propose Compositional Chain-of-Thought (CCoT), a novel zero-shot Chain-of-Thought prompting method that utilizes SG representations in order to extract compositional knowledge from an LMM. Specifically, we first generate an SG using the LMM, and then use that SG in the prompt to produce a response. Through extensive experiments, we find that the proposed CCoT approach not only improves LMM performance on several vision and language VL compositional benchmarks but also improves the performance of several popular LMMs on general multimodal benchmarks, without the need for fine-tuning or annotated ground-truth SGs.

{{</citation>}}


### (29/147) Characterizing Video Question Answering with Sparsified Inputs (Shiyuan Huang et al., 2023)

{{<citation>}}

Shiyuan Huang, Robinson Piramuthu, Vicente Ordonez, Shih-Fu Chang, Gunnar A. Sigurdsson. (2023)  
**Characterizing Video Question Answering with Sparsified Inputs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.16311v1)  

---


**ABSTRACT**  
In Video Question Answering, videos are often processed as a full-length sequence of frames to ensure minimal loss of information. Recent works have demonstrated evidence that sparse video inputs are sufficient to maintain high performance. However, they usually discuss the case of single frame selection. In our work, we extend the setting to multiple number of inputs and other modalities. We characterize the task with different input sparsity and provide a tool for doing that. Specifically, we use a Gumbel-based learnable selection module to adaptively select the best inputs for the final task. In this way, we experiment over public VideoQA benchmarks and provide analysis on how sparsified inputs affect the performance. From our experiments, we have observed only 5.2%-5.8% loss of performance with only 10% of video lengths, which corresponds to 2-4 frames selected from each video. Meanwhile, we also observed the complimentary behaviour between visual and textual inputs, even under highly sparsified settings, suggesting the potential of improving data efficiency for video-and-language tasks.

{{</citation>}}


### (30/147) Aligning Non-Causal Factors for Transformer-Based Source-Free Domain Adaptation (Sunandini Sanyal et al., 2023)

{{<citation>}}

Sunandini Sanyal, Ashish Ramayee Asokan, Suvaansh Bhambri, Pradyumna YM, Akshay Kulkarni, Jogendra Nath Kundu, R Venkatesh Babu. (2023)  
**Aligning Non-Causal Factors for Transformer-Based Source-Free Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.16294v1)  

---


**ABSTRACT**  
Conventional domain adaptation algorithms aim to achieve better generalization by aligning only the task-discriminative causal factors between a source and target domain. However, we find that retaining the spurious correlation between causal and non-causal factors plays a vital role in bridging the domain gap and improving target adaptation. Therefore, we propose to build a framework that disentangles and supports causal factor alignment by aligning the non-causal factors first. We also investigate and find that the strong shape bias of vision transformers, coupled with its multi-head attention, make it a suitable architecture for realizing our proposed disentanglement. Hence, we propose to build a Causality-enforcing Source-Free Transformer framework (C-SFTrans) to achieve disentanglement via a novel two-stage alignment approach: a) non-causal factor alignment: non-causal factors are aligned using a style classification task which leads to an overall global alignment, b) task-discriminative causal factor alignment: causal factors are aligned via target adaptation. We are the first to investigate the role of vision transformers (ViTs) in a privacy-preserving source-free setting. Our approach achieves state-of-the-art results in several DA benchmarks.

{{</citation>}}


### (31/147) Self-Supervised Learning of Whole and Component-Based Semantic Representations for Person Re-Identification (Siyuan Huang et al., 2023)

{{<citation>}}

Siyuan Huang, Yifan Zhou, Ram Prabhakar Kathirvel, Rama Chellappa, Chun Pong Lau. (2023)  
**Self-Supervised Learning of Whole and Component-Based Semantic Representations for Person Re-Identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.17074v1)  

---


**ABSTRACT**  
Interactive Segmentation Models (ISMs) like the Segment Anything Model have significantly improved various computer vision tasks, yet their application to Person Re-identification (ReID) remains limited. On the other hand, existing semantic pre-training models for ReID often have limitations like predefined parsing ranges or coarse semantics. Additionally, ReID and Clothes-Changing ReID (CC-ReID) are usually treated separately due to their different domains. This paper investigates whether utilizing precise human-centric semantic representation can boost the ReID performance and improve the generalization among various ReID tasks. We propose SemReID, a self-supervised ReID model that leverages ISMs for adaptive part-based semantic extraction, contributing to the improvement of ReID performance. SemReID additionally refines its semantic representation through techniques such as image masking and KoLeo regularization. Evaluation across three types of ReID datasets -- standard ReID, CC-ReID, and unconstrained ReID -- demonstrates superior performance compared to state-of-the-art methods. In addition, recognizing the scarcity of large person datasets with fine-grained semantics, we introduce the novel LUPerson-Part dataset to assist ReID methods in acquiring the fine-grained part semantics for robust performance.

{{</citation>}}


### (32/147) Removing NSFW Concepts from Vision-and-Language Models for Text-to-Image Retrieval and Generation (Samuele Poppi et al., 2023)

{{<citation>}}

Samuele Poppi, Tobia Poppi, Federico Cocchi, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara. (2023)  
**Removing NSFW Concepts from Vision-and-Language Models for Text-to-Image Retrieval and Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16254v1)  

---


**ABSTRACT**  
Vision-and-Language models such as CLIP have demonstrated remarkable effectiveness across a wide range of tasks. However, these models are typically trained on web-scale data, which can introduce inappropriate content and lead to the development of unsafe and biased behavior. This, in turn, hampers their applicability in sensitive and trustworthy contexts and could raise significant concern in their adoption. To overcome these limitations, we introduce a methodology to make Vision-and-Language models safer by removing their sensitivity to not-safe-for-work concepts. We show how this can be done by distilling from a large language model which converts between safe and unsafe sentences and which is fine-tuned starting from just 100 manually-curated pairs. We conduct extensive experiments on the resulting embedding space for both retrieval and text-to-image generation, where we show that our model can also be properly employed with pre-trained image generators. Our source code and trained models are available at: https://github.com/aimagelab/safe-clip.

{{</citation>}}


### (33/147) SemiVL: Semi-Supervised Semantic Segmentation with Vision-Language Guidance (Lukas Hoyer et al., 2023)

{{<citation>}}

Lukas Hoyer, David Joseph Tan, Muhammad Ferjad Naeem, Luc Van Gool, Federico Tombari. (2023)  
**SemiVL: Semi-Supervised Semantic Segmentation with Vision-Language Guidance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.16241v1)  

---


**ABSTRACT**  
In semi-supervised semantic segmentation, a model is trained with a limited number of labeled images along with a large corpus of unlabeled images to reduce the high annotation effort. While previous methods are able to learn good segmentation boundaries, they are prone to confuse classes with similar visual appearance due to the limited supervision. On the other hand, vision-language models (VLMs) are able to learn diverse semantic knowledge from image-caption datasets but produce noisy segmentation due to the image-level training. In SemiVL, we propose to integrate rich priors from VLM pre-training into semi-supervised semantic segmentation to learn better semantic decision boundaries. To adapt the VLM from global to local reasoning, we introduce a spatial fine-tuning strategy for label-efficient learning. Further, we design a language-guided decoder to jointly reason over vision and language. Finally, we propose to handle inherent ambiguities in class labels by providing the model with language guidance in the form of class definitions. We evaluate SemiVL on 4 semantic segmentation datasets, where it significantly outperforms previous semi-supervised methods. For instance, SemiVL improves the state-of-the-art by +13.5 mIoU on COCO with 232 annotated images and by +6.1 mIoU on Pascal VOC with 92 labels. Project page: https://github.com/google-research/semivl

{{</citation>}}


### (34/147) IG Captioner: Information Gain Captioners are Strong Zero-shot Classifiers (Chenglin Yang et al., 2023)

{{<citation>}}

Chenglin Yang, Siyuan Qiao, Yuan Cao, Yu Zhang, Tao Zhu, Alan Yuille, Jiahui Yu. (2023)  
**IG Captioner: Information Gain Captioners are Strong Zero-shot Classifiers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.17072v1)  

---


**ABSTRACT**  
Generative training has been demonstrated to be powerful for building visual-language models. However, on zero-shot discriminative benchmarks, there is still a performance gap between models trained with generative and discriminative objectives. In this paper, we aim to narrow this gap by improving the efficacy of generative training on classification tasks, without any finetuning processes or additional modules.   Specifically, we focus on narrowing the gap between the generative captioner and the CLIP classifier. We begin by analysing the predictions made by the captioner and classifier and observe that the caption generation inherits the distribution bias from the language model trained with pure text modality, making it less grounded on the visual signal. To tackle this problem, we redesign the scoring objective for the captioner to alleviate the distributional bias and focus on measuring the gain of information brought by the visual inputs. We further design a generative training objective to match the evaluation objective. We name our model trained and evaluated from the novel procedures as Information Gain (IG) captioner. We pretrain the models on the public Laion-5B dataset and perform a series of discriminative evaluations. For the zero-shot classification on ImageNet, IG captioner achieves $> 18\%$ improvements over the standard captioner, achieving comparable performances with the CLIP classifier. IG captioner also demonstrated strong performance on zero-shot image-text retrieval tasks on MSCOCO and Flickr30K. We hope this paper inspires further research towards unifying generative and discriminative training procedures for visual-language models.

{{</citation>}}


### (35/147) Video-Bench: A Comprehensive Benchmark and Toolkit for Evaluating Video-based Large Language Models (Munan Ning et al., 2023)

{{<citation>}}

Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, Li Yuan. (2023)  
**Video-Bench: A Comprehensive Benchmark and Toolkit for Evaluating Video-based Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16103v2)  

---


**ABSTRACT**  
Video-based large language models (Video-LLMs) have been recently introduced, targeting both fundamental improvements in perception and comprehension, and a diverse range of user inquiries. In pursuit of the ultimate goal of achieving artificial general intelligence, a truly intelligent Video-LLM model should not only see and understand the surroundings, but also possess human-level commonsense, and make well-informed decisions for the users. To guide the development of such a model, the establishment of a robust and comprehensive evaluation system becomes crucial. To this end, this paper proposes \textit{Video-Bench}, a new comprehensive benchmark along with a toolkit specifically designed for evaluating Video-LLMs. The benchmark comprises 10 meticulously crafted tasks, evaluating the capabilities of Video-LLMs across three distinct levels: Video-exclusive Understanding, Prior Knowledge-based Question-Answering, and Comprehension and Decision-making. In addition, we introduce an automatic toolkit tailored to process model outputs for various tasks, facilitating the calculation of metrics and generating convenient final scores. We evaluate 8 representative Video-LLMs using \textit{Video-Bench}. The findings reveal that current Video-LLMs still fall considerably short of achieving human-like comprehension and analysis of real-world videos, offering valuable insights for future research directions. The benchmark and toolkit are available at: \url{https://github.com/PKU-YuanGroup/Video-Bench}.

{{</citation>}}


### (36/147) Diffusion-TTA: Test-time Adaptation of Discriminative Models via Generative Feedback (Mihir Prabhudesai et al., 2023)

{{<citation>}}

Mihir Prabhudesai, Tsung-Wei Ke, Alexander C. Li, Deepak Pathak, Katerina Fragkiadaki. (2023)  
**Diffusion-TTA: Test-time Adaptation of Discriminative Models via Generative Feedback**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.16102v2)  

---


**ABSTRACT**  
The advancements in generative modeling, particularly the advent of diffusion models, have sparked a fundamental question: how can these models be effectively used for discriminative tasks? In this work, we find that generative models can be great test-time adapters for discriminative models. Our method, Diffusion-TTA, adapts pre-trained discriminative models such as image classifiers, segmenters and depth predictors, to each unlabelled example in the test set using generative feedback from a diffusion model. We achieve this by modulating the conditioning of the diffusion model using the output of the discriminative model. We then maximize the image likelihood objective by backpropagating the gradients to discriminative model's parameters. We show Diffusion-TTA significantly enhances the accuracy of various large-scale pre-trained discriminative models, such as, ImageNet classifiers, CLIP models, image pixel labellers and image depth predictors. Diffusion-TTA outperforms existing test-time adaptation methods, including TTT-MAE and TENT, and particularly shines in online adaptation setups, where the discriminative model is continually adapted to each example in the test set. We provide access to code, results, and visualizations on our website: https://diffusion-tta.github.io/.

{{</citation>}}


### (37/147) How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs (Haoqin Tu et al., 2023)

{{<citation>}}

Haoqin Tu, Chenhang Cui, Zijun Wang, Yiyang Zhou, Bingchen Zhao, Junlin Han, Wangchunshu Zhou, Huaxiu Yao, Cihang Xie. (2023)  
**How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2311.16101v1)  

---


**ABSTRACT**  
This work focuses on the potential of Vision LLMs (VLLMs) in visual reasoning. Different from prior studies, we shift our focus from evaluating standard performance to introducing a comprehensive safety evaluation suite, covering both out-of-distribution (OOD) generalization and adversarial robustness. For the OOD evaluation, we present two novel VQA datasets, each with one variant, designed to test model performance under challenging conditions. In exploring adversarial robustness, we propose a straightforward attack strategy for misleading VLLMs to produce visual-unrelated responses. Moreover, we assess the efficacy of two jailbreaking strategies, targeting either the vision or language component of VLLMs. Our evaluation of 21 diverse models, ranging from open-source VLLMs to GPT-4V, yields interesting observations: 1) Current VLLMs struggle with OOD texts but not images, unless the visual information is limited; and 2) These VLLMs can be easily misled by deceiving vision encoders only, and their vision-language training often compromise safety protocols. We release this safety evaluation suite at https://github.com/UCSC-VLAA/vllm-safety-benchmark.

{{</citation>}}


### (38/147) ViT-Lens-2: Gateway to Omni-modal Intelligence (Weixian Lei et al., 2023)

{{<citation>}}

Weixian Lei, Yixiao Ge, Kun Yi, Jianfeng Zhang, Difei Gao, Dylan Sun, Yuying Ge, Ying Shan, Mike Zheng Shou. (2023)  
**ViT-Lens-2: Gateway to Omni-modal Intelligence**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16081v1)  

---


**ABSTRACT**  
Aiming to advance AI agents, large foundation models significantly improve reasoning and instruction execution, yet the current focus on vision and language neglects the potential of perceiving diverse modalities in open-world environments. However, the success of data-driven vision and language models is costly or even infeasible to be reproduced for rare modalities. In this paper, we present ViT-Lens-2 that facilitates efficient omni-modal representation learning by perceiving novel modalities with a pretrained ViT and aligning them to a pre-defined space. Specifically, the modality-specific lens is tuned to project any-modal signals to an intermediate embedding space, which are then processed by a strong ViT with pre-trained visual knowledge. The encoded representations are optimized toward aligning with the modal-independent space, pre-defined by off-the-shelf foundation models. ViT-Lens-2 provides a unified solution for representation learning of increasing modalities with two appealing advantages: (i) Unlocking the great potential of pretrained ViTs to novel modalities effectively with efficient data regime; (ii) Enabling emergent downstream capabilities through modality alignment and shared ViT parameters. We tailor ViT-Lens-2 to learn representations for 3D point cloud, depth, audio, tactile and EEG, and set new state-of-the-art results across various understanding tasks, such as zero-shot classification. By seamlessly integrating ViT-Lens-2 into Multimodal Foundation Models, we enable Any-modality to Text and Image Generation in a zero-shot manner. Code and models are available at https://github.com/TencentARC/ViT-Lens.

{{</citation>}}


### (39/147) OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving (Wenzhao Zheng et al., 2023)

{{<citation>}}

Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, Jiwen Lu. (2023)  
**OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.16038v1)  

---


**ABSTRACT**  
Understanding how the 3D scene evolves is vital for making decisions in autonomous driving. Most existing methods achieve this by predicting the movements of object boxes, which cannot capture more fine-grained scene information. In this paper, we explore a new framework of learning a world model, OccWorld, in the 3D Occupancy space to simultaneously predict the movement of the ego car and the evolution of the surrounding scenes. We propose to learn a world model based on 3D occupancy rather than 3D bounding boxes and segmentation maps for three reasons: 1) expressiveness. 3D occupancy can describe the more fine-grained 3D structure of the scene; 2) efficiency. 3D occupancy is more economical to obtain (e.g., from sparse LiDAR points). 3) versatility. 3D occupancy can adapt to both vision and LiDAR. To facilitate the modeling of the world evolution, we learn a reconstruction-based scene tokenizer on the 3D occupancy to obtain discrete scene tokens to describe the surrounding scenes. We then adopt a GPT-like spatial-temporal generative transformer to generate subsequent scene and ego tokens to decode the future occupancy and ego trajectory. Extensive experiments on the widely used nuScenes benchmark demonstrate the ability of OccWorld to effectively model the evolution of the driving scenes. OccWorld also produces competitive planning results without using instance and map supervision. Code: https://github.com/wzzheng/OccWorld.

{{</citation>}}


### (40/147) VLPrompt: Vision-Language Prompting for Panoptic Scene Graph Generation (Zijian Zhou et al., 2023)

{{<citation>}}

Zijian Zhou, Miaojing Shi, Holger Caesar. (2023)  
**VLPrompt: Vision-Language Prompting for Panoptic Scene Graph Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16492v1)  

---


**ABSTRACT**  
Panoptic Scene Graph Generation (PSG) aims at achieving a comprehensive image understanding by simultaneously segmenting objects and predicting relations among objects. However, the long-tail problem among relations leads to unsatisfactory results in real-world applications. Prior methods predominantly rely on vision information or utilize limited language information, such as object or relation names, thereby overlooking the utility of language information. Leveraging the recent progress in Large Language Models (LLMs), we propose to use language information to assist relation prediction, particularly for rare relations. To this end, we propose the Vision-Language Prompting (VLPrompt) model, which acquires vision information from images and language information from LLMs. Then, through a prompter network based on attention mechanism, it achieves precise relation prediction. Our extensive experiments show that VLPrompt significantly outperforms previous state-of-the-art methods on the PSG dataset, proving the effectiveness of incorporating language information and alleviating the long-tail problem of relations.

{{</citation>}}


### (41/147) Unified Batch Normalization: Identifying and Alleviating the Feature Condensation in Batch Normalization and a Unified Framework (Shaobo Wang et al., 2023)

{{<citation>}}

Shaobo Wang, Xiangdong Zhang, Junchi Yan. (2023)  
**Unified Batch Normalization: Identifying and Alleviating the Feature Condensation in Batch Normalization and a Unified Framework**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.15993v1)  

---


**ABSTRACT**  
Batch Normalization (BN) has become an essential technique in contemporary neural network design, enhancing training stability. Specifically, BN employs centering and scaling operations to standardize features along the batch dimension and uses an affine transformation to recover features. Although standard BN has shown its capability to improve deep neural network training and convergence, it still exhibits inherent limitations in certain cases. Most existing techniques that enhance BN consider a single or a few aspects of BN. In this paper, we first identify problems with BN from a feature perspective and explore that feature condensation exists in the learning when employing BN, which negatively affects testing performance. To tackle this problem, we propose a two-stage unified framework called Unified Batch Normalization (UBN). In the first stage, we utilize a simple feature condensation threshold to alleviate the feature condensation, which hinders inappropriate statistic updates in normalization. In the second stage, we unify various normalization variants to boost each component of BN. Our experimental results reveal that UBN significantly enhances performance across different visual backbones and notably expedites network training convergence, particularly in early training stages. Notably, our method improved about 3% in top-1 accuracy on ImageNet classification with large batch sizes, showing the effectiveness of our approach in real-world scenarios.

{{</citation>}}


### (42/147) CoSeR: Bridging Image and Language for Cognitive Super-Resolution (Haoze Sun et al., 2023)

{{<citation>}}

Haoze Sun, Wenbo Li, Jianzhuang Liu, Haoyu Chen, Renjing Pei, Xueyi Zou, Youliang Yan, Yujiu Yang. (2023)  
**CoSeR: Bridging Image and Language for Cognitive Super-Resolution**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.16512v2)  

---


**ABSTRACT**  
Existing super-resolution (SR) models primarily focus on restoring local texture details, often neglecting the global semantic information within the scene. This oversight can lead to the omission of crucial semantic details or the introduction of inaccurate textures during the recovery process. In our work, we introduce the Cognitive Super-Resolution (CoSeR) framework, empowering SR models with the capacity to comprehend low-resolution images. We achieve this by marrying image appearance and language understanding to generate a cognitive embedding, which not only activates prior information from large text-to-image diffusion models but also facilitates the generation of high-quality reference images to optimize the SR process. To further improve image fidelity, we propose a novel condition injection scheme called "All-in-Attention", consolidating all conditional information into a single module. Consequently, our method successfully restores semantically correct and photorealistic details, demonstrating state-of-the-art performance across multiple benchmarks. Code: https://github.com/VINHYU/CoSeR

{{</citation>}}


### (43/147) Direct2.5: Diverse Text-to-3D Generation via Multi-view 2.5D Diffusion (Yuanxun Lu et al., 2023)

{{<citation>}}

Yuanxun Lu, Jingyang Zhang, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan, Xun Cao, Yao Yao. (2023)  
**Direct2.5: Diverse Text-to-3D Generation via Multi-view 2.5D Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15980v1)  

---


**ABSTRACT**  
Recent advances in generative AI have unveiled significant potential for the creation of 3D content. However, current methods either apply a pre-trained 2D diffusion model with the time-consuming score distillation sampling (SDS), or a direct 3D diffusion model trained on limited 3D data losing generation diversity. In this work, we approach the problem by employing a multi-view 2.5D diffusion fine-tuned from a pre-trained 2D diffusion model. The multi-view 2.5D diffusion directly models the structural distribution of 3D data, while still maintaining the strong generalization ability of the original 2D diffusion model, filling the gap between 2D diffusion-based and direct 3D diffusion-based methods for 3D content generation. During inference, multi-view normal maps are generated using the 2.5D diffusion, and a novel differentiable rasterization scheme is introduced to fuse the almost consistent multi-view normal maps into a consistent 3D model. We further design a normal-conditioned multi-view image generation module for fast appearance generation given the 3D geometry. Our method is a one-pass diffusion process and does not require any SDS optimization as post-processing. We demonstrate through extensive experiments that, our direct 2.5D generation with the specially-designed fusion scheme can achieve diverse, mode-seeking-free, and high-fidelity 3D content generation in only 10 seconds. Project page: https://nju-3dv.github.io/projects/direct25.

{{</citation>}}


### (44/147) FALCON: Fairness Learning via Contrastive Attention Approach to Continual Semantic Scene Understanding in Open World (Thanh-Dat Truong et al., 2023)

{{<citation>}}

Thanh-Dat Truong, Utsav Prabhu, Bhiksha Raj, Jackson Cothren, Khoa Luu. (2023)  
**FALCON: Fairness Learning via Contrastive Attention Approach to Continual Semantic Scene Understanding in Open World**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.15965v1)  

---


**ABSTRACT**  
Continual Learning in semantic scene segmentation aims to continually learn new unseen classes in dynamic environments while maintaining previously learned knowledge. Prior studies focused on modeling the catastrophic forgetting and background shift challenges in continual learning. However, fairness, another major challenge that causes unfair predictions leading to low performance among major and minor classes, still needs to be well addressed. In addition, prior methods have yet to model the unknown classes well, thus resulting in producing non-discriminative features among unknown classes. This paper presents a novel Fairness Learning via Contrastive Attention Approach to continual learning in semantic scene understanding. In particular, we first introduce a new Fairness Contrastive Clustering loss to address the problems of catastrophic forgetting and fairness. Then, we propose an attention-based visual grammar approach to effectively model the background shift problem and unknown classes, producing better feature representations for different unknown classes. Through our experiments, our proposed approach achieves State-of-the-Art (SOTA) performance on different continual learning settings of three standard benchmarks, i.e., ADE20K, Cityscapes, and Pascal VOC. It promotes the fairness of the continual semantic segmentation model.

{{</citation>}}


### (45/147) Efficient Pre-training for Localized Instruction Generation of Videos (Anil Batra et al., 2023)

{{<citation>}}

Anil Batra, Davide Moltisanti, Laura Sevilla-Lara, Marcus Rohrbach, Frank Keller. (2023)  
**Efficient Pre-training for Localized Instruction Generation of Videos**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.15964v1)  

---


**ABSTRACT**  
Procedural videos show step-by-step demonstrations of tasks like recipe preparation. Understanding such videos is challenging, involving the precise localization of steps and the generation of textual instructions. Manually annotating steps and writing instructions is costly, which limits the size of current datasets and hinders effective learning. Leveraging large but noisy video-transcript datasets for pre-training can boost performance, but demands significant computational resources. Furthermore, transcripts contain irrelevant content and exhibit style variation compared to instructions written by human annotators. To mitigate both issues, we propose a technique, Sieve-&-Swap, to automatically curate a smaller dataset: (i) Sieve filters irrelevant transcripts and (ii) Swap enhances the quality of the text instruction by automatically replacing the transcripts with human-written instructions from a text-only recipe dataset. The curated dataset, three orders of magnitude smaller than current web-scale datasets, enables efficient training of large-scale models with competitive performance. We complement our Sieve-\&-Swap approach with a Procedure Transformer (ProcX) for end-to-end step localization and instruction generation for procedural videos. When this model is pre-trained on our curated dataset, it achieves state-of-the-art performance in zero-shot and finetuning settings on YouCook2 and Tasty, while using a fraction of the computational resources.

{{</citation>}}


### (46/147) From Pixels to Titles: Video Game Identification by Screenshots using Convolutional Neural Networks (Fabricio Breve, 2023)

{{<citation>}}

Fabricio Breve. (2023)  
**From Pixels to Titles: Video Game Identification by Screenshots using Convolutional Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-NE, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.15963v1)  

---


**ABSTRACT**  
This paper investigates video game identification through single screenshots, utilizing five convolutional neural network (CNN) architectures (MobileNet, DenseNet, EfficientNetB0, EfficientNetB2, and EfficientNetB3) across 22 home console systems, spanning from Atari 2600 to PlayStation 5. Confirming the hypothesis, CNNs autonomously extract image features, enabling the identification of game titles from screenshots without additional features. Using ImageNet pre-trained weights, EfficientNetB3 achieves the highest average accuracy (74.51%), while DenseNet169 excels in 14 of the 22 systems. Employing alternative initial weights from another screenshots dataset boosts accuracy for EfficientNetB2 and EfficientNetB3, with the latter reaching a peak accuracy of 76.36% and demonstrating reduced convergence epochs from 23.7 to 20.5 on average. Overall, the combination of optimal architecture and weights attains 77.67% accuracy, primarily led by EfficientNetB3 in 19 systems. These findings underscore the efficacy of CNNs in video game identification through screenshots.

{{</citation>}}


### (47/147) Computer Vision for Carriers: PATRIOT (Ari Goodman et al., 2023)

{{<citation>}}

Ari Goodman, Gurpreet Singh, James Hing, Ryan O'Shea. (2023)  
**Computer Vision for Carriers: PATRIOT**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2311.15914v1)  

---


**ABSTRACT**  
Deck tracking performed on carriers currently involves a team of sailors manually identifying aircraft and updating a digital user interface called the Ouija Board. Improvements to the deck tracking process would result in increased Sortie Generation Rates, and therefore applying automation is seen as a critical method to improve deck tracking. However, the requirements on a carrier ship do not allow for the installation of hardware-based location sensing technologies like Global Positioning System (GPS) sensors. PATRIOT (Panoramic Asset Tracking of Real-Time Information for the Ouija Tabletop) is a research effort and proposed solution to performing deck tracking with passive sensing and without the need for GPS sensors. PATRIOT is a prototype system which takes existing camera feeds, calculates aircraft poses, and updates a virtual Ouija board interface with the current status of the assets. PATRIOT would allow for faster, more accurate, and less laborious asset tracking for aircraft, people, and support equipment. PATRIOT is anticipated to benefit the warfighter by reducing cognitive workload, reducing manning requirements, collecting data to improve logistics, and enabling an automation gateway for future efforts to improve efficiency and safety. The authors have developed and tested algorithms to perform pose estimations of assets in real-time including OpenPifPaf, High-Resolution Network (HRNet), HigherHRNet (HHRNet), Faster R-CNN, and in-house developed encoder-decoder network. The software was tested with synthetic and real-world data and was able to accurately extract the pose of assets. Fusion, tracking, and real-world generality are planned to be improved to ensure a successful transition to the fleet.

{{</citation>}}


### (48/147) ChartLlama: A Multimodal LLM for Chart Understanding and Generation (Yucheng Han et al., 2023)

{{<citation>}}

Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, Hanwang Zhang. (2023)  
**ChartLlama: A Multimodal LLM for Chart Understanding and Generation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2311.16483v1)  

---


**ABSTRACT**  
Multi-modal large language models have demonstrated impressive performances on most vision-language tasks. However, the model generally lacks the understanding capabilities for specific domain data, particularly when it comes to interpreting chart figures. This is mainly due to the lack of relevant multi-modal instruction tuning datasets. In this article, we create a high-quality instruction-tuning dataset leveraging GPT-4. We develop a multi-step data generation process in which different steps are responsible for generating tabular data, creating chart figures, and designing instruction tuning data separately. Our method's flexibility enables us to generate diverse, high-quality instruction-tuning data consistently and efficiently while maintaining a low resource expenditure. Additionally, it allows us to incorporate a wider variety of chart and task types not yet featured in existing datasets. Next, we introduce ChartLlama, a multi-modal large language model that we've trained using our created dataset. ChartLlama outperforms all prior methods in ChartQA, Chart-to-text, and Chart-extraction evaluation benchmarks. Additionally, ChartLlama significantly improves upon the baseline in our specially compiled chart dataset, which includes new chart and task types. The results of ChartLlama confirm the value and huge potential of our proposed data generation method in enhancing chart comprehension.

{{</citation>}}


### (49/147) Data Generation for Post-OCR correction of Cyrillic handwriting (Evgenii Davydkin et al., 2023)

{{<citation>}}

Evgenii Davydkin, Aleksandr Markelov, Egor Iuldashev, Anton Dudkin, Ivan Krivorotov. (2023)  
**Data Generation for Post-OCR correction of Cyrillic handwriting**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: OCR, T5  
[Paper Link](http://arxiv.org/abs/2311.15896v1)  

---


**ABSTRACT**  
This paper introduces a novel approach to post-Optical Character Recognition Correction (POC) for handwritten Cyrillic text, addressing a significant gap in current research methodologies. This gap is due to the lack of large text corporas that provide OCR errors for further training of language-based POC models, which are demanding in terms of corpora size. Our study primarily focuses on the development and application of a synthetic handwriting generation engine based on B\'ezier curves. Such an engine generates highly realistic handwritten text in any amounts, which we utilize to create a substantial dataset by transforming Russian text corpora sourced from the internet. We apply a Handwritten Text Recognition (HTR) model to this dataset to identify OCR errors, forming the basis for our POC model training. The correction model is trained on a 90-symbol input context, utilizing a pre-trained T5 architecture with a seq2seq correction task. We evaluate our approach on HWR200 and School_notebooks_RU datasets as they provide significant challenges in the HTR domain. Furthermore, POC can be used to highlight errors for teachers, evaluating student performance. This can be done simply by comparing sentences before and after correction, displaying differences in text. Our primary contribution lies in the innovative use of B\'ezier curves for Cyrillic text generation and subsequent error correction using a specialized POC model. We validate our approach by presenting Word Accuracy Rate (WAR) and Character Accuracy Rate (CAR) results, both with and without post-OCR correction, using real open corporas of handwritten Cyrillic text. These results, coupled with our methodology, are designed to be reproducible, paving the way for further advancements in the field of OCR and handwritten text analysis. Paper contributions can be found in https://github.com/dbrainio/CyrillicHandwritingPOC

{{</citation>}}


### (50/147) EVCap: Retrieval-Augmented Image Captioning with External Visual-Name Memory for Open-World Comprehension (Jiaxuan Li et al., 2023)

{{<citation>}}

Jiaxuan Li, Duc Minh Vo, Akihiro Sugimoto, Hideki Nakayama. (2023)  
**EVCap: Retrieval-Augmented Image Captioning with External Visual-Name Memory for Open-World Comprehension**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2311.15879v1)  

---


**ABSTRACT**  
Large language models (LLMs)-based image captioning has the capability of describing objects not explicitly observed in training data; yet novel objects occur frequently, necessitating the requirement of sustaining up-to-date object knowledge for open-world comprehension. Instead of relying on large amounts of data and scaling up network parameters, we introduce a highly effective retrieval-augmented image captioning method that prompts LLMs with object names retrieved from External Visual--name memory (EVCap). We build ever-changing object knowledge memory using objects' visuals and names, enabling us to (i) update the memory at a minimal cost and (ii) effortlessly augment LLMs with retrieved object names utilizing a lightweight and fast-to-train model. Our model, which was trained only on the COCO dataset, can be adapted to out-domain data without additional fine-tuning or retraining. Our comprehensive experiments conducted on various benchmarks and synthetic commonsense-violating data demonstrate that EVCap, comprising solely 3.97M trainable parameters, exhibits superior performance compared to other methods of equivalent model size scale. Notably, it achieves competitive performance against specialist SOTAs with an enormous number of parameters. Our code is available at https://jiaxuan-li.github.io/EVCap.

{{</citation>}}


### (51/147) RO-LLaMA: Generalist LLM for Radiation Oncology via Noise Augmentation and Consistency Regularization (Kwanyoung Kim et al., 2023)

{{<citation>}}

Kwanyoung Kim, Yujin Oh, Sangjoon Park, Hwa Kyung Byun, Jin Sung Kim, Yong Bae Kim, Jong Chul Ye. (2023)  
**RO-LLaMA: Generalist LLM for Radiation Oncology via Noise Augmentation and Consistency Regularization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Augmentation, Embedding, LLaMA  
[Paper Link](http://arxiv.org/abs/2311.15876v1)  

---


**ABSTRACT**  
Recent advancements in Artificial Intelligence (AI) have profoundly influenced medical fields, by providing tools to reduce clinical workloads. However, most AI models are constrained to execute uni-modal tasks, in stark contrast to the comprehensive approaches utilized by medical professionals. To address this, here we present RO-LLaMA, a versatile generalist large language model (LLM) tailored for the field of radiation oncology. This model seamlessly covers a wide range of the workflow of radiation oncologists, adept at various tasks such as clinical report summarization, radiation therapy plan suggestion, and plan-guided therapy target volume segmentation. In particular, to maximize the end-to-end performance, we further present a novel Consistency Embedding Fine-Tuning (CEFTune) technique, which boosts LLM's robustness to additional errors at the intermediates while preserving the capability of handling clean inputs, and creatively transform this concept into LLM-driven segmentation framework as Consistency Embedding Segmentation (CESEG). Experimental results on multi-centre cohort sets demonstrate our proposed RO-LLaMA's promising performance for diverse tasks with generalization capabilities.

{{</citation>}}


### (52/147) InterControl: Generate Human Motion Interactions by Controlling Every Joint (Zhenzhi Wang et al., 2023)

{{<citation>}}

Zhenzhi Wang, Jingbo Wang, Dahua Lin, Bo Dai. (2023)  
**InterControl: Generate Human Motion Interactions by Controlling Every Joint**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.15864v1)  

---


**ABSTRACT**  
Text-conditioned human motion generation model has achieved great progress by introducing diffusion models and corresponding control signals. However, the interaction between humans are still under explored. To model interactions of arbitrary number of humans, we define interactions as human joint pairs that are either in contact or separated, and leverage {\em Large Language Model (LLM) Planner} to translate interaction descriptions into contact plans. Based on the contact plans, interaction generation could be achieved by spatially controllable motion generation methods by taking joint contacts as spatial conditions. We present a novel approach named InterControl for flexible spatial control of every joint in every person at any time by leveraging motion diffusion model only trained on single-person data. We incorporate a motion controlnet to generate coherent and realistic motions given sparse spatial control signals and a loss guidance module to precisely align any joint to the desired position in a classifier guidance manner via Inverse Kinematics (IK). Extensive experiments on HumanML3D and KIT-ML dataset demonstrate its effectiveness in versatile joint control. We also collect data of joint contact pairs by LLMs to show InterControl's ability in human interaction generation.

{{</citation>}}


### (53/147) Learning with Noisy Low-Cost MOS for Image Quality Assessment via Dual-Bias Calibration (Lei Wang et al., 2023)

{{<citation>}}

Lei Wang, Qingbo Wu, Desen Yuan, King Ngi Ngan, Hongliang Li, Fanman Meng, Linfeng Xu. (2023)  
**Learning with Noisy Low-Cost MOS for Image Quality Assessment via Dual-Bias Calibration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Bias, QA  
[Paper Link](http://arxiv.org/abs/2311.15846v1)  

---


**ABSTRACT**  
Learning based image quality assessment (IQA) models have obtained impressive performance with the help of reliable subjective quality labels, where mean opinion score (MOS) is the most popular choice. However, in view of the subjective bias of individual annotators, the labor-abundant MOS (LA-MOS) typically requires a large collection of opinion scores from multiple annotators for each image, which significantly increases the learning cost. In this paper, we aim to learn robust IQA models from low-cost MOS (LC-MOS), which only requires very few opinion scores or even a single opinion score for each image. More specifically, we consider the LC-MOS as the noisy observation of LA-MOS and enforce the IQA model learned from LC-MOS to approach the unbiased estimation of LA-MOS. In this way, we represent the subjective bias between LC-MOS and LA-MOS, and the model bias between IQA predictions learned from LC-MOS and LA-MOS (i.e., dual-bias) as two latent variables with unknown parameters. By means of the expectation-maximization based alternating optimization, we can jointly estimate the parameters of the dual-bias, which suppresses the misleading of LC-MOS via a gated dual-bias calibration (GDBC) module. To the best of our knowledge, this is the first exploration of robust IQA model learning from noisy low-cost labels. Theoretical analysis and extensive experiments on four popular IQA datasets show that the proposed method is robust toward different bias rates and annotation numbers and significantly outperforms the other learning based IQA models when only LC-MOS is available. Furthermore, we also achieve comparable performance with respect to the other models learned with LA-MOS.

{{</citation>}}


### (54/147) FlowZero: Zero-Shot Text-to-Video Synthesis with LLM-Driven Dynamic Scene Syntax (Yu Lu et al., 2023)

{{<citation>}}

Yu Lu, Linchao Zhu, Hehe Fan, Yi Yang. (2023)  
**FlowZero: Zero-Shot Text-to-Video Synthesis with LLM-Driven Dynamic Scene Syntax**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.15813v1)  

---


**ABSTRACT**  
Text-to-video (T2V) generation is a rapidly growing research area that aims to translate the scenes, objects, and actions within complex video text into a sequence of coherent visual frames. We present FlowZero, a novel framework that combines Large Language Models (LLMs) with image diffusion models to generate temporally-coherent videos. FlowZero uses LLMs to understand complex spatio-temporal dynamics from text, where LLMs can generate a comprehensive dynamic scene syntax (DSS) containing scene descriptions, object layouts, and background motion patterns. These elements in DSS are then used to guide the image diffusion model for video generation with smooth object motions and frame-to-frame coherence. Moreover, FlowZero incorporates an iterative self-refinement process, enhancing the alignment between the spatio-temporal layouts and the textual prompts for the videos. To enhance global coherence, we propose enriching the initial noise of each frame with motion dynamics to control the background movement and camera motion adaptively. By using spatio-temporal syntaxes to guide the diffusion process, FlowZero achieves improvement in zero-shot video synthesis, generating coherent videos with vivid motion.

{{</citation>}}


### (55/147) LLMGA: Multimodal Large Language Model based Generation Assistant (Bin Xia et al., 2023)

{{<citation>}}

Bin Xia, Shiyin Wang, Yingfan Tao, Yitong Wang, Jiaya Jia. (2023)  
**LLMGA: Multimodal Large Language Model based Generation Assistant**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16500v1)  

---


**ABSTRACT**  
In this paper, we introduce a Multimodal Large Language Model-based Generation Assistant (LLMGA), leveraging the vast reservoir of knowledge and proficiency in reasoning, comprehension, and response inherent in Large Language Models (LLMs) to assist users in image generation and editing. Diverging from existing approaches where Multimodal Large Language Models (MLLMs) generate fixed-size embeddings to control Stable Diffusion (SD), our LLMGA provides a detailed language generation prompt for precise control over SD. This not only augments LLM context understanding but also reduces noise in generation prompts, yields images with more intricate and precise content, and elevates the interpretability of the network. To this end, we curate a comprehensive dataset comprising prompt refinement, similar image generation, inpainting $\&$ outpainting, and visual question answering. Moreover, we propose a two-stage training scheme. In the first stage, we train the MLLM to grasp the properties of image generation and editing, enabling it to generate detailed prompts. In the second stage, we optimize SD to align with the MLLM's generation prompts. Additionally, we propose a reference-based restoration network to alleviate texture, brightness, and contrast disparities between generated and preserved regions during image editing. Extensive results show that LLMGA has promising generative capabilities and can enable wider applications in an interactive manner.

{{</citation>}}


### (56/147) C-SAW: Self-Supervised Prompt Learning for Image Generalization in Remote Sensing (Avigyan Bhattacharya et al., 2023)

{{<citation>}}

Avigyan Bhattacharya, Mainak Singha, Ankit Jha, Biplab Banerjee. (2023)  
**C-SAW: Self-Supervised Prompt Learning for Image Generalization in Remote Sensing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.15812v1)  

---


**ABSTRACT**  
We focus on domain and class generalization problems in analyzing optical remote sensing images, using the large-scale pre-trained vision-language model (VLM), CLIP. While contrastively trained VLMs show impressive zero-shot generalization performance, their effectiveness is limited when dealing with diverse domains during training and testing. Existing prompt learning techniques overlook the importance of incorporating domain and content information into the prompts, which results in a drop in performance while dealing with such multi-domain data. To address these challenges, we propose a solution that ensures domain-invariant prompt learning while enhancing the expressiveness of visual features. We observe that CLIP's vision encoder struggles to identify contextual image information, particularly when image patches are jumbled up. This issue is especially severe in optical remote sensing images, where land-cover classes exhibit well-defined contextual appearances. To this end, we introduce C-SAW, a method that complements CLIP with a self-supervised loss in the visual space and a novel prompt learning technique that emphasizes both visual domain and content-specific features. We keep the CLIP backbone frozen and introduce a small set of projectors for both the CLIP encoders to train C-SAW contrastively. Experimental results demonstrate the superiority of C-SAW across multiple remote sensing benchmarks and different generalization tasks.

{{</citation>}}


### (57/147) PIPE : Parallelized Inference Through Post-Training Quantization Ensembling of Residual Expansions (Edouard Yvinec et al., 2023)

{{<citation>}}

Edouard Yvinec, Arnaud Dapogny, Kevin Bailly. (2023)  
**PIPE : Parallelized Inference Through Post-Training Quantization Ensembling of Residual Expansions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP, Quantization  
[Paper Link](http://arxiv.org/abs/2311.15806v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) are ubiquitous in computer vision and natural language processing, but suffer from high inference cost. This problem can be addressed by quantization, which consists in converting floating point perations into a lower bit-width format. With the growing concerns on privacy rights, we focus our efforts on data-free methods. However, such techniques suffer from their lack of adaptability to the target devices, as a hardware typically only support specific bit widths. Thus, to adapt to a variety of devices, a quantization method shall be flexible enough to find good accuracy v.s. speed trade-offs for every bit width and target device. To achieve this, we propose PIPE, a quantization method that leverages residual error expansion, along with group sparsity and an ensemble approximation for better parallelization. PIPE is backed off by strong theoretical guarantees and achieves superior performance on every benchmarked application (from vision to NLP tasks), architecture (ConvNets, transformers) and bit-width (from int8 to ternary quantization).

{{</citation>}}


### (58/147) Video Anomaly Detection via Spatio-Temporal Pseudo-Anomaly Generation : A Unified Approach (Ayush K. Rai et al., 2023)

{{<citation>}}

Ayush K. Rai, Tarun Krishna, Feiyan Hu, Alexandru Drimbarean, Kevin McGuinness, Alan F. Smeaton, Noel E. O'Connor. (2023)  
**Video Anomaly Detection via Spatio-Temporal Pseudo-Anomaly Generation : A Unified Approach**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.16514v1)  

---


**ABSTRACT**  
Video Anomaly Detection (VAD) is an open-set recognition task, which is usually formulated as a one-class classification (OCC) problem, where training data is comprised of videos with normal instances while test data contains both normal and anomalous instances. Recent works have investigated the creation of pseudo-anomalies (PAs) using only the normal data and making strong assumptions about real-world anomalies with regards to abnormality of objects and speed of motion to inject prior information about anomalies in an autoencoder (AE) based reconstruction model during training. This work proposes a novel method for generating generic spatio-temporal PAs by inpainting a masked out region of an image using a pre-trained Latent Diffusion Model and further perturbing the optical flow using mixup to emulate spatio-temporal distortions in the data. In addition, we present a simple unified framework to detect real-world anomalies under the OCC setting by learning three types of anomaly indicators, namely reconstruction quality, temporal irregularity and semantic inconsistency. Extensive experiments on four VAD benchmark datasets namely Ped2, Avenue, ShanghaiTech and UBnormal demonstrate that our method performs on par with other existing state-of-the-art PAs generation and reconstruction based methods under the OCC setting. Our analysis also examines the transferability and generalisation of PAs across these datasets, offering valuable insights by identifying real-world anomalies through PAs.

{{</citation>}}


### (59/147) TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models (Yushi Huang et al., 2023)

{{<citation>}}

Yushi Huang, Ruihao Gong, Jing Liu, Tianlong Chen, Xianglong Liu. (2023)  
**TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.16503v1)  

---


**ABSTRACT**  
The Diffusion model, a prevalent framework for image generation, encounters significant challenges in terms of broad applicability due to its extended inference times and substantial memory requirements. Efficient Post-training Quantization (PTQ) is pivotal for addressing these issues in traditional models. Different from traditional models, diffusion models heavily depend on the time-step $t$ to achieve satisfactory multi-round denoising. Usually, $t$ from the finite set $\{1, \ldots, T\}$ is encoded to a temporal feature by a few modules totally irrespective of the sampling data. However, existing PTQ methods do not optimize these modules separately. They adopt inappropriate reconstruction targets and complex calibration methods, resulting in a severe disturbance of the temporal feature and denoising trajectory, as well as a low compression efficiency. To solve these, we propose a Temporal Feature Maintenance Quantization (TFMQ) framework building upon a Temporal Information Block which is just related to the time-step $t$ and unrelated to the sampling data. Powered by the pioneering block design, we devise temporal information aware reconstruction (TIAR) and finite set calibration (FSC) to align the full-precision temporal features in a limited time. Equipped with the framework, we can maintain the most temporal information and ensure the end-to-end generation quality. Extensive experiments on various datasets and diffusion models prove our state-of-the-art results. Remarkably, our quantization approach, for the first time, achieves model performance nearly on par with the full-precision model under 4-bit weight quantization. Additionally, our method incurs almost no extra computational cost and accelerates quantization time by $2.0 \times$ on LSUN-Bedrooms $256 \times 256$ compared to previous works.

{{</citation>}}


### (60/147) Machine Learning-Based Jamun Leaf Disease Detection: A Comprehensive Review (Auvick Chandra Bhowmik et al., 2023)

{{<citation>}}

Auvick Chandra Bhowmik, Dr. Md. Taimur Ahad, Yousuf Rayhan Emon. (2023)  
**Machine Learning-Based Jamun Leaf Disease Detection: A Comprehensive Review**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.15741v1)  

---


**ABSTRACT**  
Jamun leaf diseases pose a significant threat to agricultural productivity, negatively impacting both yield and quality in the jamun industry. The advent of machine learning has opened up new avenues for tackling these diseases effectively. Early detection and diagnosis are essential for successful crop management. While no automated systems have yet been developed specifically for jamun leaf disease detection, various automated systems have been implemented for similar types of disease detection using image processing techniques. This paper presents a comprehensive review of machine learning methodologies employed for diagnosing plant leaf diseases through image classification, which can be adapted for jamun leaf disease detection. It meticulously assesses the strengths and limitations of various Vision Transformer models, including Transfer learning model and vision transformer (TLMViT), SLViT, SE-ViT, IterationViT, Tiny-LeViT, IEM-ViT, GreenViT, and PMViT. Additionally, the paper reviews models such as Dense Convolutional Network (DenseNet), Residual Neural Network (ResNet)-50V2, EfficientNet, Ensemble model, Convolutional Neural Network (CNN), and Locally Reversible Transformer. These machine-learning models have been evaluated on various datasets, demonstrating their real-world applicability. This review not only sheds light on current advancements in the field but also provides valuable insights for future research directions in machine learning-based jamun leaf disease detection and classification.

{{</citation>}}


### (61/147) Optimization of Image Processing Algorithms for Character Recognition in Cultural Typewritten Documents (Mariana Dias et al., 2023)

{{<citation>}}

Mariana Dias, Carla Teixeira Lopes. (2023)  
**Optimization of Image Processing Algorithms for Character Recognition in Cultural Typewritten Documents**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-DL, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2311.15740v1)  

---


**ABSTRACT**  
Linked Data is used in various fields as a new way of structuring and connecting data. Cultural heritage institutions have been using linked data to improve archival descriptions and facilitate the discovery of information. Most archival records have digital representations of physical artifacts in the form of scanned images that are non-machine-readable. Optical Character Recognition (OCR) recognizes text in images and translates it into machine-encoded text. This paper evaluates the impact of image processing methods and parameter tuning in OCR applied to typewritten cultural heritage documents. The approach uses a multi-objective problem formulation to minimize Levenshtein edit distance and maximize the number of words correctly identified with a non-dominated sorting genetic algorithm (NSGA-II) to tune the methods' parameters. Evaluation results show that parameterization by digital representation typology benefits the performance of image pre-processing algorithms in OCR. Furthermore, our findings suggest that employing image pre-processing algorithms in OCR might be more suitable for typologies where the text recognition task without pre-processing does not produce good results. In particular, Adaptive Thresholding, Bilateral Filter, and Opening are the best-performing algorithms for the theatre plays' covers, letters, and overall dataset, respectively, and should be applied before OCR to improve its performance.

{{</citation>}}


### (62/147) GPT4Vis: What Can GPT-4 Do for Zero-shot Visual Recognition? (Wenhao Wu et al., 2023)

{{<citation>}}

Wenhao Wu, Huanjin Yao, Mengxi Zhang, Yuxin Song, Wanli Ouyang, Jingdong Wang. (2023)  
**GPT4Vis: What Can GPT-4 Do for Zero-shot Visual Recognition?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.15732v1)  

---


**ABSTRACT**  
This paper does not present a novel method. Instead, it delves into an essential, yet must-know baseline in light of the latest advancements in Generative Artificial Intelligence (GenAI): the utilization of GPT-4 for visual understanding. Our study centers on the evaluation of GPT-4's linguistic and visual capabilities in zero-shot visual recognition tasks. Specifically, we explore the potential of its generated rich textual descriptions across various categories to enhance recognition performance without any training. Additionally, we evaluate its visual proficiency in directly recognizing diverse visual content. To achieve this, we conduct an extensive series of experiments, systematically quantifying the performance of GPT-4 across three modalities: images, videos, and point clouds. This comprehensive evaluation encompasses a total of 16 widely recognized benchmark datasets, providing top-1 and top-5 accuracy metrics. Our study reveals that leveraging GPT-4's advanced linguistic knowledge to generate rich descriptions markedly improves zero-shot recognition. In terms of visual proficiency, GPT-4V's average performance across 16 datasets sits roughly between the capabilities of OpenAI-CLIP's ViT-L and EVA-CLIP's ViT-E. We hope that this research will contribute valuable data points and experience for future studies. We release our code at https://github.com/whwu95/GPT4Vis.

{{</citation>}}


### (63/147) Adinkra Symbol Recognition using Classical Machine Learning and Deep Learning (Michael Adjeisah et al., 2023)

{{<citation>}}

Michael Adjeisah, Kwame Omono Asamoah, Martha Asamoah Yeboah, Raji Rafiu King, Godwin Ferguson Achaab, Kingsley Adjei. (2023)  
**Adinkra Symbol Recognition using Classical Machine Learning and Deep Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15728v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) has emerged as a transformative influence, engendering paradigm shifts in global societies, spanning academia and industry. However, in light of these rapid advances, addressing the underrepresentation of black communities and African countries in AI is crucial. Boosting enthusiasm for AI can be effectively accomplished by showcasing straightforward applications around tasks like identifying and categorizing traditional symbols, such as Adinkra symbols, or familiar objects within the community. In this research endeavor, we dived into classical machine learning and harnessed the power of deep learning models to tackle the intricate task of classifying and recognizing Adinkra symbols. The idea led to a newly constructed ADINKRA dataset comprising 174,338 images meticulously organized into 62 distinct classes, each representing a singular and emblematic symbol. We constructed a CNN model for classification and recognition using six convolutional layers, three fully connected (FC) layers, and optional dropout regularization. The model is a simpler and smaller version of VGG, with fewer layers, smaller channel sizes, and a fixed kernel size. Additionally, we tap into the transfer learning capabilities provided by pre-trained models like VGG and ResNet. These models assist us in both classifying images and extracting features that can be used with classical machine learning models. We assess the model's performance by measuring its accuracy and convergence rate and visualizing the areas that significantly influence its predictions. These evaluations serve as a foundational benchmark for future assessments of the ADINKRA dataset. We hope this application exemplar inspires ideas on the various uses of AI in organizing our traditional and modern lives.

{{</citation>}}


### (64/147) MARIS: Referring Image Segmentation via Mutual-Aware Attention Features (Mengxi Zhang et al., 2023)

{{<citation>}}

Mengxi Zhang, Yiming Liu, Xiangjun Yin, Huanjing Yue, Jingyu Yang. (2023)  
**MARIS: Referring Image Segmentation via Mutual-Aware Attention Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.15727v1)  

---


**ABSTRACT**  
Referring image segmentation (RIS) aims to segment a particular region based on a language expression prompt. Existing methods incorporate linguistic features into visual features and obtain multi-modal features for mask decoding. However, these methods may segment the visually salient entity instead of the correct referring region, as the multi-modal features are dominated by the abundant visual context. In this paper, we propose MARIS, a referring image segmentation method that leverages the Segment Anything Model (SAM) and introduces a mutual-aware attention mechanism to enhance the cross-modal fusion via two parallel branches. Specifically, our mutual-aware attention mechanism consists of Vision-Guided Attention and Language-Guided Attention, which bidirectionally model the relationship between visual and linguistic features. Correspondingly, we design a Mask Decoder to enable explicit linguistic guidance for more consistent segmentation with the language expression. To this end, a multi-modal query token is proposed to integrate linguistic information and interact with visual information simultaneously. Extensive experiments on three benchmark datasets show that our method outperforms the state-of-the-art RIS methods. Our code will be publicly available.

{{</citation>}}


### (65/147) Variational Autoencoders for Feature Exploration and Malignancy Prediction of Lung Lesions (Benjamin Keel et al., 2023)

{{<citation>}}

Benjamin Keel, Aaron Quyn, David Jayne, Samuel D. Relton. (2023)  
**Variational Autoencoders for Feature Exploration and Malignancy Prediction of Lung Lesions**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15719v1)  

---


**ABSTRACT**  
Lung cancer is responsible for 21% of cancer deaths in the UK and five-year survival rates are heavily influenced by the stage the cancer was identified at. Recent studies have demonstrated the capability of AI methods for accurate and early diagnosis of lung cancer from routine scans. However, this evidence has not translated into clinical practice with one barrier being a lack of interpretable models. This study investigates the application Variational Autoencoders (VAEs), a type of generative AI model, to lung cancer lesions. Proposed models were trained on lesions extracted from 3D CT scans in the LIDC-IDRI public dataset. Latent vector representations of 2D slices produced by the VAEs were explored through clustering to justify their quality and used in an MLP classifier model for lung cancer diagnosis, the best model achieved state-of-the-art metrics of AUC 0.98 and 93.1% accuracy. Cluster analysis shows the VAE latent space separates the dataset of malignant and benign lesions based on meaningful feature components including tumour size, shape, patient and malignancy class. We also include a comparative analysis of the standard Gaussian VAE (GVAE) and the more recent Dirichlet VAE (DirVAE), which replaces the prior with a Dirichlet distribution to encourage a more explainable latent space with disentangled feature representation. Finally, we demonstrate the potential for latent space traversals corresponding to clinically meaningful feature changes.

{{</citation>}}


### (66/147) SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation (Jiehong Lin et al., 2023)

{{<citation>}}

Jiehong Lin, Lihua Liu, Dekun Lu, Kui Jia. (2023)  
**SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.15707v1)  

---


**ABSTRACT**  
Zero-shot 6D object pose estimation involves the detection of novel objects with their 6D poses in cluttered scenes, presenting significant challenges for model generalizability. Fortunately, the recent Segment Anything Model (SAM) has showcased remarkable zero-shot transfer performance, which provides a promising solution to tackle this task. Motivated by this, we introduce SAM-6D, a novel framework designed to realize the task through two steps, including instance segmentation and pose estimation. Given the target objects, SAM-6D employs two dedicated sub-networks, namely Instance Segmentation Model (ISM) and Pose Estimation Model (PEM), to perform these steps on cluttered RGB-D images. ISM takes SAM as an advanced starting point to generate all possible object proposals and selectively preserves valid ones through meticulously crafted object matching scores in terms of semantics, appearance and geometry. By treating pose estimation as a partial-to-partial point matching problem, PEM performs a two-stage point matching process featuring a novel design of background tokens to construct dense 3D-3D correspondence, ultimately yielding the pose estimates. Without bells and whistles, SAM-6D outperforms the existing methods on the seven core datasets of the BOP Benchmark for both instance segmentation and pose estimation of novel objects.

{{</citation>}}


### (67/147) ArGue: Attribute-Guided Prompt Tuning for Vision-Language Models (Xinyu Tian et al., 2023)

{{<citation>}}

Xinyu Tian, Shu Zou, Zhaoyuan Yang, Jing Zhang. (2023)  
**ArGue: Attribute-Guided Prompt Tuning for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16494v1)  

---


**ABSTRACT**  
Although soft prompt tuning is effective in efficiently adapting Vision-Language (V&L) models for downstream tasks, it shows limitations in dealing with distribution shifts. We address this issue with Attribute-Guided Prompt Tuning (ArGue), making three key contributions. 1) In contrast to the conventional approach of directly appending soft prompts preceding class names, we align the model with primitive visual attributes generated by Large Language Models (LLMs). We posit that a model's ability to express high confidence in these attributes signifies its capacity to discern the correct class rationales. 2) We introduce attribute sampling to eliminate disadvantageous attributes, thus only semantically meaningful attributes are preserved. 3) We propose negative prompting, explicitly enumerating class-agnostic attributes to activate spurious correlations and encourage the model to generate highly orthogonal probability distributions in relation to these negative features. In experiments, our method significantly outperforms current state-of-the-art prompt tuning methods on both novel class prediction and out-of-distribution generalization tasks.

{{</citation>}}


### (68/147) HAVE-FUN: Human Avatar Reconstruction from Few-Shot Unconstrained Images (Xihe Yang et al., 2023)

{{<citation>}}

Xihe Yang, Xingyu Chen, Shaohui Wang, Daiheng Gao, Xiaoguang Han, Baoyuan Wang. (2023)  
**HAVE-FUN: Human Avatar Reconstruction from Few-Shot Unconstrained Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.15672v1)  

---


**ABSTRACT**  
As for human avatar reconstruction, contemporary techniques commonly necessitate the acquisition of costly data and struggle to achieve satisfactory results from a small number of casual images. In this paper, we investigate this task from a few-shot unconstrained photo album. The reconstruction of human avatars from such data sources is challenging because of limited data amount and dynamic articulated poses. For handling dynamic data, we integrate a skinning mechanism with deep marching tetrahedra (DMTet) to form a drivable tetrahedral representation, which drives arbitrary mesh topologies generated by the DMTet for the adaptation of unconstrained images. To effectively mine instructive information from few-shot data, we devise a two-phase optimization method with few-shot reference and few-shot guidance. The former focuses on aligning avatar identity with reference images, while the latter aims to generate plausible appearances for unseen regions. Overall, our framework, called HaveFun, can undertake avatar reconstruction, rendering, and animation. Extensive experiments on our developed benchmarks demonstrate that HaveFun exhibits substantially superior performance in reconstructing the human body and hand. Project website: https://seanchenxy.github.io/HaveFunWeb/.

{{</citation>}}


### (69/147) Enhancing Diffusion Models with Text-Encoder Reinforcement Learning (Chaofeng Chen et al., 2023)

{{<citation>}}

Chaofeng Chen, Annan Wang, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin. (2023)  
**Enhancing Diffusion Models with Text-Encoder Reinforcement Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15657v1)  

---


**ABSTRACT**  
Text-to-image diffusion models are typically trained to optimize the log-likelihood objective, which presents challenges in meeting specific requirements for downstream tasks, such as image aesthetics and image-text alignment. Recent research addresses this issue by refining the diffusion U-Net using human rewards through reinforcement learning or direct backpropagation. However, many of them overlook the importance of the text encoder, which is typically pretrained and fixed during training. In this paper, we demonstrate that by finetuning the text encoder through reinforcement learning, we can enhance the text-image alignment of the results, thereby improving the visual quality. Our primary motivation comes from the observation that the current text encoder is suboptimal, often requiring careful prompt adjustment. While fine-tuning the U-Net can partially improve performance, it remains suffering from the suboptimal text encoder. Therefore, we propose to use reinforcement learning with low-rank adaptation to finetune the text encoder based on task-specific rewards, referred as \textbf{TexForce}. We first show that finetuning the text encoder can improve the performance of diffusion models. Then, we illustrate that TexForce can be simply combined with existing U-Net finetuned models to get much better results without additional training. Finally, we showcase the adaptability of our method in diverse applications, including the generation of high-quality face and hand images.

{{</citation>}}


### (70/147) Mitigating Hallucination in Visual Language Models with Visual Supervision (Zhiyang Chen et al., 2023)

{{<citation>}}

Zhiyang Chen, Yousong Zhu, Yufei Zhan, Zhaowen Li, Chaoyang Zhao, Jinqiao Wang, Ming Tang. (2023)  
**Mitigating Hallucination in Visual Language Models with Visual Supervision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16479v1)  

---


**ABSTRACT**  
Large vision-language models (LVLMs) suffer from hallucination a lot, generating responses that apparently contradict to the image content occasionally. The key problem lies in its weak ability to comprehend detailed content in a multi-modal context, which can be mainly attributed to two factors in training data and loss function. The vision instruction dataset primarily focuses on global description, and the auto-regressive loss function favors text modeling rather than image understanding. In this paper, we bring more detailed vision annotations and more discriminative vision models to facilitate the training of LVLMs, so that they can generate more precise responses without encounter hallucination. On one hand, we generate image-text pairs with detailed relationship annotations in panoptic scene graph dataset (PSG). These conversations pay more attention on detailed facts in the image, encouraging the model to answer questions based on multi-modal contexts. On the other hand, we integrate SAM and mask prediction loss as auxiliary supervision, forcing the LVLMs to have the capacity to identify context-related objects, so that they can generate more accurate responses, mitigating hallucination. Moreover, to provide a deeper evaluation on the hallucination in LVLMs, we propose a new benchmark, RAH-Bench. It divides vision hallucination into three different types that contradicts the image with wrong categories, attributes or relations, and introduces False Positive Rate as detailed sub-metric for each type. In this benchmark, our approach demonstrates an +8.4% enhancement compared to original LLaVA and achieves widespread performance improvements across other models.

{{</citation>}}


### (71/147) Reinforcement Learning from Diffusion Feedback: Q* for Image Search (Aboli Marathe, 2023)

{{<citation>}}

Aboli Marathe. (2023)  
**Reinforcement Learning from Diffusion Feedback: Q* for Image Search**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15648v1)  

---


**ABSTRACT**  
Large vision-language models are steadily gaining personalization capabilities at the cost of fine-tuning or data augmentation. We present two models for image generation using model-agnostic learning that align semantic priors with generative capabilities. RLDF, or Reinforcement Learning from Diffusion Feedback, is a singular approach for visual imitation through prior-preserving reward function guidance. This employs Q-learning (with standard Q*) for generation and follows a semantic-rewarded trajectory for image search through finite encoding-tailored actions. The second proposed method, noisy diffusion gradient, is optimization driven. At the root of both methods is a special CFG encoding that we propose for continual semantic guidance. Using only a single input image and no text input, RLDF generates high-quality images over varied domains including retail, sports and agriculture showcasing class-consistency and strong visual diversity. Project website is available at https://infernolia.github.io/RLDF.

{{</citation>}}


### (72/147) Only Positive Cases: 5-fold High-order Attention Interaction Model for Skin Segmentation Derived Classification (Renkai Wu et al., 2023)

{{<citation>}}

Renkai Wu, Yinghao Liu, Pengchen Liang, Qing Chang. (2023)  
**Only Positive Cases: 5-fold High-order Attention Interaction Model for Skin Segmentation Derived Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.15625v1)  

---


**ABSTRACT**  
Computer-aided diagnosis of skin diseases is an important tool. However, the interpretability of computer-aided diagnosis is currently poor. Dermatologists and patients cannot intuitively understand the learning and prediction process of neural networks, which will lead to a decrease in the credibility of computer-aided diagnosis. In addition, traditional methods need to be trained using negative samples in order to predict the presence or absence of a lesion, but medical data is often in short supply. In this paper, we propose a multiple high-order attention interaction model (MHA-UNet) for use in a highly explainable skin lesion segmentation task. MHA-UNet is able to obtain the presence or absence of a lesion by explainable reasoning without the need for training on negative samples. Specifically, we propose a high-order attention interaction mechanism that introduces squeeze attention to a higher level for feature attention. In addition, a multiple high-order attention interaction (MHAblock) module is proposed by combining the different features of different orders. For classifying the presence or absence of lesions, we conducted classification experiments on several publicly available datasets in the absence of negative samples, based on explainable reasoning about the interaction of 5 attention orders of MHAblock. The highest positive detection rate obtained from the experiments was 81.0% and the highest negative detection rate was 83.5%. For segmentation experiments, comparison experiments of the proposed method with 13 medical segmentation models and external validation experiments with 8 state-of-the-art models in three public datasets and our clinical dataset demonstrate the state-of-the-art performance of our model. The code is available from https://github.com/wurenkai/MHA-UNet.

{{</citation>}}


### (73/147) RetouchUAA: Unconstrained Adversarial Attack via Image Retouching (Mengda Xie et al., 2023)

{{<citation>}}

Mengda Xie, Yiling He, Meie Fang. (2023)  
**RetouchUAA: Unconstrained Adversarial Attack via Image Retouching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.16478v1)  

---


**ABSTRACT**  
Deep Neural Networks (DNNs) are susceptible to adversarial examples. Conventional attacks generate controlled noise-like perturbations that fail to reflect real-world scenarios and hard to interpretable. In contrast, recent unconstrained attacks mimic natural image transformations occurring in the real world for perceptible but inconspicuous attacks, yet compromise realism due to neglect of image post-processing and uncontrolled attack direction. In this paper, we propose RetouchUAA, an unconstrained attack that exploits a real-life perturbation: image retouching styles, highlighting its potential threat to DNNs. Compared to existing attacks, RetouchUAA offers several notable advantages. Firstly, RetouchUAA excels in generating interpretable and realistic perturbations through two key designs: the image retouching attack framework and the retouching style guidance module. The former custom-designed human-interpretability retouching framework for adversarial attack by linearizing images while modelling the local processing and retouching decision-making in human retouching behaviour, provides an explicit and reasonable pipeline for understanding the robustness of DNNs against retouching. The latter guides the adversarial image towards standard retouching styles, thereby ensuring its realism. Secondly, attributed to the design of the retouching decision regularization and the persistent attack strategy, RetouchUAA also exhibits outstanding attack capability and defense robustness, posing a heavy threat to DNNs. Experiments on ImageNet and Place365 reveal that RetouchUAA achieves nearly 100\% white-box attack success against three DNNs, while achieving a better trade-off between image naturalness, transferability and defense robustness than baseline attacks.

{{</citation>}}


### (74/147) 2D Feature Distillation for Weakly- and Semi-Supervised 3D Semantic Segmentation (Ozan Unal et al., 2023)

{{<citation>}}

Ozan Unal, Dengxin Dai, Lukas Hoyer, Yigit Baran Can, Luc Van Gool. (2023)  
**2D Feature Distillation for Weakly- and Semi-Supervised 3D Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.15605v1)  

---


**ABSTRACT**  
As 3D perception problems grow in popularity and the need for large-scale labeled datasets for LiDAR semantic segmentation increase, new methods arise that aim to reduce the necessity for dense annotations by employing weakly-supervised training. However these methods continue to show weak boundary estimation and high false negative rates for small objects and distant sparse regions. We argue that such weaknesses can be compensated by using RGB images which provide a denser representation of the scene. We propose an image-guidance network (IGNet) which builds upon the idea of distilling high level feature information from a domain adapted synthetically trained 2D semantic segmentation network. We further utilize a one-way contrastive learning scheme alongside a novel mixing strategy called FOVMix, to combat the horizontal field-of-view mismatch between the two sensors and enhance the effects of image guidance. IGNet achieves state-of-the-art results for weakly-supervised LiDAR semantic segmentation on ScribbleKITTI, boasting up to 98% relative performance to fully supervised training with only 8% labeled points, while introducing no additional annotation burden or computational/memory cost during inference. Furthermore, we show that our contributions also prove effective for semi-supervised training, where IGNet claims state-of-the-art results on both ScribbleKITTI and SemanticKITTI.

{{</citation>}}


### (75/147) UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition (Xiaohan Ding et al., 2023)

{{<citation>}}

Xiaohan Ding, Yiyuan Zhang, Yixiao Ge, Sijie Zhao, Lin Song, Xiangyu Yue, Ying Shan. (2023)  
**UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.15599v1)  

---


**ABSTRACT**  
Large-kernel convolutional neural networks (ConvNets) have recently received extensive research attention, but there are two unresolved and critical issues that demand further investigation. 1) The architectures of existing large-kernel ConvNets largely follow the design principles of conventional ConvNets or transformers, while the architectural design for large-kernel ConvNets remains under-addressed. 2) As transformers have dominated multiple modalities, it remains to be investigated whether ConvNets also have a strong universal perception ability in domains beyond vision. In this paper, we contribute from two aspects. 1) We propose four architectural guidelines for designing large-kernel ConvNets, the core of which is to exploit the essential characteristics of large kernels that distinguish them from small kernels - they can see wide without going deep. Following such guidelines, our proposed large-kernel ConvNet shows leading performance in image recognition. For example, our models achieve an ImageNet accuracy of 88.0%, ADE20K mIoU of 55.6%, and COCO box AP of 56.4%, demonstrating better performance and higher speed than a number of recently proposed powerful competitors. 2) We discover that large kernels are the key to unlocking the exceptional performance of ConvNets in domains where they were originally not proficient. With certain modality-related preprocessing approaches, the proposed model achieves state-of-the-art performance on time-series forecasting and audio recognition tasks even without modality-specific customization to the architecture. Code and all the models at https://github.com/AILab-CVC/UniRepLKNet.

{{</citation>}}


### (76/147) Can Vision-Language Models Think from a First-Person Perspective? (Sijie Cheng et al., 2023)

{{<citation>}}

Sijie Cheng, Zhicheng Guo, Jingwen Wu, Kechen Fang, Peng Li, Huaping Liu, Yang Liu. (2023)  
**Can Vision-Language Models Think from a First-Person Perspective?**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.15596v1)  

---


**ABSTRACT**  
Vision-language models (VLMs) have recently shown promising results in traditional downstream tasks. Evaluation studies have emerged to assess their abilities, with the majority focusing on the third-person perspective, and only a few addressing specific tasks from the first-person perspective. However, the capability of VLMs to "think" from a first-person perspective, a crucial attribute for advancing autonomous agents and robotics, remains largely unexplored. To bridge this research gap, we introduce EgoThink, a novel visual question-answering benchmark that encompasses six core capabilities with twelve detailed dimensions. The benchmark is constructed using selected clips from egocentric videos, with manually annotated question-answer pairs containing first-person information. To comprehensively assess VLMs, we evaluate eighteen popular VLMs on EgoThink. Moreover, given the open-ended format of the answers, we use GPT-4 as the automatic judge to compute single-answer grading. Experimental results indicate that although GPT-4V leads in numerous dimensions, all evaluated VLMs still possess considerable potential for improvement in first-person perspective tasks. Meanwhile, enlarging the number of trainable parameters has the most significant impact on model performance on EgoThink. In conclusion, EgoThink serves as a valuable addition to existing evaluation benchmarks for VLMs, providing an indispensable resource for future research in the realm of embodied artificial intelligence and robotics.

{{</citation>}}


### (77/147) Progressive Target-Styled Feature Augmentation for Unsupervised Domain Adaptation on Point Clouds (Zicheng Wang et al., 2023)

{{<citation>}}

Zicheng Wang, Zhen Zhao, Yiming Wu, Luping Zhou, Dong Xu. (2023)  
**Progressive Target-Styled Feature Augmentation for Unsupervised Domain Adaptation on Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.16474v1)  

---


**ABSTRACT**  
Unsupervised domain adaptation is a critical challenge in the field of point cloud analysis, as models trained on one set of data often struggle to perform well in new scenarios due to domain shifts. Previous works tackle the problem by using adversarial training or self-supervised learning for feature extractor adaptation, but ensuring that features extracted from the target domain can be distinguished by the source-supervised classifier remains challenging. In this work, we propose a novel approach called progressive target-styled feature augmentation (PTSFA). Unlike previous works that focus on feature extractor adaptation, our PTSFA approach focuses on classifier adaptation. It aims to empower the classifier to recognize target-styled source features and progressively adapt to the target domain. To enhance the reliability of predictions within the PTSFA framework and encourage discriminative feature extraction, we further introduce a new intermediate domain approaching (IDA) strategy. We validate our method on the benchmark datasets, where our method achieves new state-of-the-art performance. Our code is available at https://github.com/xiaoyao3302/PTSFA.

{{</citation>}}


### (78/147) Pre-trained Language Models Do Not Help Auto-regressive Text-to-Image Generation (Yuhui Zhang et al., 2023)

{{<citation>}}

Yuhui Zhang, Brandon McKinzie, Zhe Gan, Vaishaal Shankar, Alexander Toshev. (2023)  
**Pre-trained Language Models Do Not Help Auto-regressive Text-to-Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16201v1)  

---


**ABSTRACT**  
Recent advances in image tokenizers, such as VQ-VAE, have enabled text-to-image generation using auto-regressive methods, similar to language modeling. However, these methods have yet to leverage pre-trained language models, despite their adaptability to various downstream tasks. In this work, we explore this gap by adapting a pre-trained language model for auto-regressive text-to-image generation, and find that pre-trained language models offer limited help. We provide a two-fold explanation by analyzing tokens from each modality. First, we demonstrate that image tokens possess significantly different semantics compared to text tokens, rendering pre-trained language models no more effective in modeling them than randomly initialized ones. Second, the text tokens in the image-text datasets are too simple compared to normal language model pre-training data, which causes the catastrophic degradation of language models' capability.

{{</citation>}}


### (79/147) Improving Adaptability and Generalizability of Efficient Transfer Learning for Vision-Language Models (Yongjin Yang et al., 2023)

{{<citation>}}

Yongjin Yang, Jongwoo Ko, Se-Young Yun. (2023)  
**Improving Adaptability and Generalizability of Efficient Transfer Learning for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.15569v1)  

---


**ABSTRACT**  
Vision-Language Models (VLMs) like CLIP have demonstrated remarkable applicability across a variety of downstream tasks, including zero-shot image classification. Recently, the use of prompts or adapters for efficient transfer learning has gained significant attention for effectively adapting to downstream tasks. However, the roles of vision and text prompts, as well as adapters in terms of generalization and transfer difficulty, have been overlooked, limiting performance on unseen tasks. In this paper, we empirically analyze how VLMs behave when using vision and text prompts, adapters, and a combination of these components, marking a novel exploration by our study. Our observations find that utilizing vision prompts for class separability and text adapters for task adaptation is crucial for adaptability and generalizability. Moreover, to improve generalization across every domain, we propose an adaptive ensemble method that effectively combines the general knowledge of VLMs with task-specific knowledge according to transfer difficulty. Upon experimenting with extensive benchmarks, our method consistently outperforms all baselines, particularly on unseen tasks, demonstrating the effectiveness of our proposed approach.

{{</citation>}}


### (80/147) Fully Authentic Visual Question Answering Dataset from Online Communities (Chongyan Chen et al., 2023)

{{<citation>}}

Chongyan Chen, Mengchen Liu, Noel Codella, Yunsheng Li, Lu Yuan, Danna Gurari. (2023)  
**Fully Authentic Visual Question Answering Dataset from Online Communities**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.15562v1)  

---


**ABSTRACT**  
Visual Question Answering (VQA) entails answering questions about images. We introduce the first VQA dataset in which all contents originate from an authentic use case. Sourced from online question answering community forums, we call it VQAonline. We then characterize our dataset and how it relates to eight other VQA datasets. Observing that answers in our dataset tend to be much longer (e.g., with a mean of 173 words) and thus incompatible with standard VQA evaluation metrics, we next analyze which of the six popular metrics for longer text evaluation align best with human judgments. We then use the best-suited metrics to evaluate six state-of-the-art vision and language foundation models on VQAonline and reveal where they struggle most. We will release the dataset soon to facilitate future extensions.

{{</citation>}}


### (81/147) PKU-I2IQA: An Image-to-Image Quality Assessment Database for AI Generated Images (Jiquan Yuan et al., 2023)

{{<citation>}}

Jiquan Yuan, Xinyan Cao, Changjin Li, Fanyi Yang, Jinlong Lin, Xixin Cao. (2023)  
**PKU-I2IQA: An Image-to-Image Quality Assessment Database for AI Generated Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2311.15556v2)  

---


**ABSTRACT**  
As image generation technology advances, AI-based image generation has been applied in various fields and Artificial Intelligence Generated Content (AIGC) has garnered widespread attention. However, the development of AI-based image generative models also brings new problems and challenges. A significant challenge is that AI-generated images (AIGI) may exhibit unique distortions compared to natural images, and not all generated images meet the requirements of the real world. Therefore, it is of great significance to evaluate AIGIs more comprehensively. Although previous work has established several human perception-based AIGC image quality assessment (AIGCIQA) databases for text-generated images, the AI image generation technology includes scenarios like text-to-image and image-to-image, and assessing only the images generated by text-to-image models is insufficient. To address this issue, we establish a human perception-based image-to-image AIGCIQA database, named PKU-I2IQA. We conduct a well-organized subjective experiment to collect quality labels for AIGIs and then conduct a comprehensive analysis of the PKU-I2IQA database. Furthermore, we have proposed two benchmark models: NR-AIGCIQA based on the no-reference image quality assessment method and FR-AIGCIQA based on the full-reference image quality assessment method. Finally, leveraging this database, we conduct benchmark experiments and compare the performance of the proposed benchmark models. The PKU-I2IQA database and benchmarks will be released to facilitate future research on \url{https://github.com/jiquan123/I2IQA}.

{{</citation>}}


### (82/147) Instruct2Attack: Language-Guided Semantic Adversarial Attacks (Jiang Liu et al., 2023)

{{<citation>}}

Jiang Liu, Chen Wei, Yuxiang Guo, Heng Yu, Alan Yuille, Soheil Feizi, Chun Pong Lau, Rama Chellappa. (2023)  
**Instruct2Attack: Language-Guided Semantic Adversarial Attacks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CR, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Adversarial Attack, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.15551v1)  

---


**ABSTRACT**  
We propose Instruct2Attack (I2A), a language-guided semantic attack that generates semantically meaningful perturbations according to free-form language instructions. We make use of state-of-the-art latent diffusion models, where we adversarially guide the reverse diffusion process to search for an adversarial latent code conditioned on the input image and text instruction. Compared to existing noise-based and semantic attacks, I2A generates more natural and diverse adversarial examples while providing better controllability and interpretability. We further automate the attack process with GPT-4 to generate diverse image-specific text instructions. We show that I2A can successfully break state-of-the-art deep neural networks even under strong adversarial defenses, and demonstrate great transferability among a variety of network architectures.

{{</citation>}}


### (83/147) Beyond Pixels: Exploring Human-Readable SVG Generation for Simple Images with Vision Language Models (Tong Zhang et al., 2023)

{{<citation>}}

Tong Zhang, Haoyang Liu, Peiyan Zhang, Yuxuan Cheng, Haohan Wang. (2023)  
**Beyond Pixels: Exploring Human-Readable SVG Generation for Simple Images with Vision Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.15543v1)  

---


**ABSTRACT**  
In the field of computer graphics, the use of vector graphics, particularly Scalable Vector Graphics (SVG), represents a notable development from traditional pixel-based imagery. SVGs, with their XML-based format, are distinct in their ability to directly and explicitly represent visual elements such as shape, color, and path. This direct representation facilitates a more accurate and logical depiction of graphical elements, enhancing reasoning and interpretability. Recognizing the potential of SVGs, the machine learning community has introduced multiple methods for image vectorization. However, transforming images into SVG format while retaining the relational properties and context of the original scene remains a key challenge. Most vectorization methods often yield SVGs that are overly complex and not easily interpretable. In response to this challenge, we introduce our method, Simple-SVG-Generation (S\textsuperscript{2}VG\textsuperscript{2}). Our method focuses on producing SVGs that are both accurate and simple, aligning with human readability and understanding. With simple images, we evaluate our method with reasoning tasks together with advanced language models, the results show a clear improvement over previous SVG generation methods. We also conducted surveys for human evaluation on the readability of our generated SVGs, the results also favor our methods.

{{</citation>}}


### (84/147) EAFP-Med: An Efficient Adaptive Feature Processing Module Based on Prompts for Medical Image Detection (Xiang Li et al., 2023)

{{<citation>}}

Xiang Li, Long Lan, Husam Lahza, Shaowu Yang, Shuihua Wang, Wenjing Yang, Hengzhu Liu, Yudong Zhang. (2023)  
**EAFP-Med: An Efficient Adaptive Feature Processing Module Based on Prompts for Medical Image Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.15540v1)  

---


**ABSTRACT**  
In the face of rapid advances in medical imaging, cross-domain adaptive medical image detection is challenging due to the differences in lesion representations across various medical imaging technologies. To address this issue, we draw inspiration from large language models to propose EAFP-Med, an efficient adaptive feature processing module based on prompts for medical image detection. EAFP-Med can efficiently extract lesion features of different scales from a diverse range of medical images based on prompts while being flexible and not limited by specific imaging techniques. Furthermore, it serves as a feature preprocessing module that can be connected to any model front-end to enhance the lesion features in input images. Moreover, we propose a novel adaptive disease detection model named EAFP-Med ST, which utilizes the Swin Transformer V2 - Tiny (SwinV2-T) as its backbone and connects it to EAFP-Med. We have compared our method to nine state-of-the-art methods. Experimental results demonstrate that EAFP-Med ST achieves the best performance on all three datasets (chest X-ray images, cranial magnetic resonance imaging images, and skin images). EAFP-Med can efficiently extract lesion features from various medical images based on prompts, enhancing the model's performance. This holds significant potential for improving medical image analysis and diagnosis.

{{</citation>}}


### (85/147) SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation (Bin Xie et al., 2023)

{{<citation>}}

Bin Xie, Jiale Cao, Jin Xie, Fahad Shahbaz Khan, Yanwei Pang. (2023)  
**SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.15537v1)  

---


**ABSTRACT**  
Open-vocabulary semantic segmentation strives to distinguish pixels into different semantic groups from an open set of categories. Most existing methods explore utilizing pre-trained vision-language models, in which the key is to adopt the image-level model for pixel-level segmentation task. In this paper, we propose a simple encoder-decoder, named SED, for open-vocabulary semantic segmentation, which comprises a hierarchical encoder-based cost map generation and a gradual fusion decoder with category early rejection. The hierarchical encoder-based cost map generation employs hierarchical backbone, instead of plain transformer, to predict pixel-level image-text cost map. Compared to plain transformer, hierarchical backbone better captures local spatial information and has linear computational complexity with respect to input size. Our gradual fusion decoder employs a top-down structure to combine cost map and the feature maps of different backbone levels for segmentation. To accelerate inference speed, we introduce a category early rejection scheme in the decoder that rejects many no-existing categories at the early layer of decoder, resulting in at most 4.7 times acceleration without accuracy degradation. Experiments are performed on multiple open-vocabulary semantic segmentation datasets, which demonstrates the efficacy of our SED method. When using ConvNeXt-B, our SED method achieves mIoU score of 31.6\% on ADE20K with 150 categories at 82 millisecond ($ms$) per image on a single A6000. We will release it at \url{https://github.com/xb534/SED.git}.

{{</citation>}}


### (86/147) MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers (Yawar Siddiqui et al., 2023)

{{<citation>}}

Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, Matthias Nießner. (2023)  
**MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.15475v1)  

---


**ABSTRACT**  
We introduce MeshGPT, a new approach for generating triangle meshes that reflects the compactness typical of artist-created meshes, in contrast to dense triangle meshes extracted by iso-surfacing methods from neural fields. Inspired by recent advances in powerful large language models, we adopt a sequence-based approach to autoregressively generate triangle meshes as sequences of triangles. We first learn a vocabulary of latent quantized embeddings, using graph convolutions, which inform these embeddings of the local mesh geometry and topology. These embeddings are sequenced and decoded into triangles by a decoder, ensuring that they can effectively reconstruct the mesh. A transformer is then trained on this learned vocabulary to predict the index of the next embedding given previous embeddings. Once trained, our model can be autoregressively sampled to generate new triangle meshes, directly generating compact meshes with sharp edges, more closely imitating the efficient triangulation patterns of human-crafted meshes. MeshGPT demonstrates a notable improvement over state of the art mesh generation methods, with a 9% increase in shape coverage and a 30-point enhancement in FID scores across various categories.

{{</citation>}}


## cs.IR (4)



### (87/147) Robust Basket Recommendation via Noise-tolerated Graph Contrastive Learning (Xinrui He et al., 2023)

{{<citation>}}

Xinrui He, Tianxin Wei, Jingrui He. (2023)  
**Robust Basket Recommendation via Noise-tolerated Graph Contrastive Learning**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Amazon, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.16334v1)  

---


**ABSTRACT**  
The growth of e-commerce has seen a surge in popularity of platforms like Amazon, eBay, and Taobao. This has given rise to a unique shopping behavior involving baskets - sets of items purchased together. As a less studied interaction mode in the community, the question of how should shopping basket complement personalized recommendation systems remains under-explored. While previous attempts focused on jointly modeling user purchases and baskets, the distinct semantic nature of these elements can introduce noise when directly integrated. This noise negatively impacts the model's performance, further exacerbated by significant noise within both user and basket behaviors.   In order to cope with the above difficulties, we propose a novel Basket recommendation framework via Noise-tolerated Contrastive Learning, named BNCL, to handle the noise existing in the cross-behavior integration and within-behavior modeling. First, we represent the basket-item interactions as the hypergraph to model the complex basket behavior, where all items appearing in the same basket are treated as a single hyperedge. Second, cross-behavior contrastive learning is designed to suppress the noise during the fusion of diverse behaviors. Next, to further inhibit the within-behavior noise of the user and basket interactions, we propose to exploit invariant properties of the recommenders w.r.t augmentations through within-behavior contrastive learning. A novel consistency-aware augmentation approach is further designed to better identify noisy interactions with the consideration of the above two types of interactions. Our framework BNCL offers a generic training paradigm that is applicable to different backbones. Extensive experiments on three shopping transaction datasets verify the effectiveness of our proposed method. Our code is available.

{{</citation>}}


### (88/147) SEINE: SEgment-based Indexing for NEural information retrieval (Sibo Dong et al., 2023)

{{<citation>}}

Sibo Dong, Justin Goldstein, Grace Hui Yang. (2023)  
**SEINE: SEgment-based Indexing for NEural information retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2311.15923v1)  

---


**ABSTRACT**  
Many early neural Information Retrieval (NeurIR) methods are re-rankers that rely on a traditional first-stage retriever due to expensive query time computations. Recently, representation-based retrievers have gained much attention, which learns query representation and document representation separately, making it possible to pre-compute document representations offline and reduce the workload at query time. Both dense and sparse representation-based retrievers have been explored. However, these methods focus on finding the representation that best represents a text (aka metric learning) and the actual retrieval function that is responsible for similarity matching between query and document is kept at a minimum by using dot product. One drawback is that unlike traditional term-level inverted index, the index formed by these embeddings cannot be easily re-used by another retrieval method. Another drawback is that keeping the interaction at minimum hurts retrieval effectiveness. On the contrary, interaction-based retrievers are known for their better retrieval effectiveness. In this paper, we propose a novel SEgment-based Neural Indexing method, SEINE, which provides a general indexing framework that can flexibly support a variety of interaction-based neural retrieval methods. We emphasize on a careful decomposition of common components in existing neural retrieval methods and propose to use segment-level inverted index to store the atomic query-document interaction values. Experiments on LETOR MQ2007 and MQ2008 datasets show that our indexing method can accelerate multiple neural retrieval methods up to 28-times faster without sacrificing much effectiveness.

{{</citation>}}


### (89/147) A Social-aware Gaussian Pre-trained Model for Effective Cold-start Recommendation (Siwei Liu et al., 2023)

{{<citation>}}

Siwei Liu, Xi Wang, Craig Macdonald, Iadh Ounis. (2023)  
**A Social-aware Gaussian Pre-trained Model for Effective Cold-start Recommendation**  

---
Primary Category: cs.IR  
Categories: 68P20, H-3-3, cs-AI, cs-IR, cs.IR  
Keywords: BERT, GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.15790v1)  

---


**ABSTRACT**  
The use of pre-training is an emerging technique to enhance a neural model's performance, which has been shown to be effective for many neural language models such as BERT. This technique has also been used to enhance the performance of recommender systems. In such recommender systems, pre-training models are used to learn a better initialisation for both users and items. However, recent existing pre-trained recommender systems tend to only incorporate the user interaction data at the pre-training stage, making it difficult to deliver good recommendations, especially when the interaction data is sparse. To alleviate this common data sparsity issue, we propose to pre-train the recommendation model not only with the interaction data but also with other available information such as the social relations among users, thereby providing the recommender system with a better initialisation compared with solely relying on the user interaction data. We propose a novel recommendation model, the Social-aware Gaussian Pre-trained model (SGP), which encodes the user social relations and interaction data at the pre-training stage in a Graph Neural Network (GNN). Afterwards, in the subsequent fine-tuning stage, our SGP model adopts a Gaussian Mixture Model (GMM) to factorise these pre-trained embeddings for further training, thereby benefiting the cold-start users from these pre-built social relations. Our extensive experiments on three public datasets show that, in comparison to 16 competitive baselines, our SGP model significantly outperforms the best baseline by upto 7.7% in terms of NDCG@10. In addition, we show that SGP permits to effectively alleviate the cold-start problem, especially when users newly register to the system through their friends' suggestions.

{{</citation>}}


### (90/147) UFIN: Universal Feature Interaction Network for Multi-Domain Click-Through Rate Prediction (Zhen Tian et al., 2023)

{{<citation>}}

Zhen Tian, Changwang Zhang, Wayne Xin Zhao, Xin Zhao, Ji-Rong Wen, Zhao Cao. (2023)  
**UFIN: Universal Feature Interaction Network for Multi-Domain Click-Through Rate Prediction**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.15493v1)  

---


**ABSTRACT**  
Click-Through Rate (CTR) prediction, which aims to estimate the probability of a user clicking on an item, is a key task in online advertising. Numerous existing CTR models concentrate on modeling the feature interactions within a solitary domain, thereby rendering them inadequate for fulfilling the requisites of multi-domain recommendations in real industrial scenarios. Some recent approaches propose intricate architectures to enhance knowledge sharing and augment model training across multiple domains. However, these approaches encounter difficulties when being transferred to new recommendation domains, owing to their reliance on the modeling of ID features (e.g., item id). To address the above issue, we propose the Universal Feature Interaction Network (UFIN) approach for CTR prediction. UFIN exploits textual data to learn universal feature interactions that can be effectively transferred across diverse domains. For learning universal feature representations, we regard the text and feature as two different modalities and propose an encoder-decoder network founded on a Large Language Model (LLM) to enforce the transfer of data from the text modality to the feature modality. Building upon the above foundation, we further develop a mixtureof-experts (MoE) enhanced adaptive feature interaction model to learn transferable collaborative patterns across multiple domains. Furthermore, we propose a multi-domain knowledge distillation framework to enhance feature interaction learning. Based on the above methods, UFIN can effectively bridge the semantic gap to learn common knowledge across various domains, surpassing the constraints of ID-based models. Extensive experiments conducted on eight datasets show the effectiveness of UFIN, in both multidomain and cross-platform settings. Our code is available at https://github.com/RUCAIBox/UFIN.

{{</citation>}}


## cs.LG (18)



### (91/147) Target-Free Compound Activity Prediction via Few-Shot Learning (Peter Eckmann et al., 2023)

{{<citation>}}

Peter Eckmann, Jake Anderson, Michael K. Gilson, Rose Yu. (2023)  
**Target-Free Compound Activity Prediction via Few-Shot Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.16328v1)  

---


**ABSTRACT**  
Predicting the activities of compounds against protein-based or phenotypic assays using only a few known compounds and their activities is a common task in target-free drug discovery. Existing few-shot learning approaches are limited to predicting binary labels (active/inactive). However, in real-world drug discovery, degrees of compound activity are highly relevant. We study Few-Shot Compound Activity Prediction (FS-CAP) and design a novel neural architecture to meta-learn continuous compound activities across large bioactivity datasets. Our model aggregates encodings generated from the known compounds and their activities to capture assay information. We also introduce a separate encoder for the unknown compound. We show that FS-CAP surpasses traditional similarity-based techniques as well as other state of the art few-shot learning methods on a variety of target-free drug discovery settings and datasets.

{{</citation>}}


### (92/147) Influence Scores at Scale for Efficient Language Data Sampling (Nikhil Anand et al., 2023)

{{<citation>}}

Nikhil Anand, Joshua Tan, Maria Minakova. (2023)  
**Influence Scores at Scale for Efficient Language Data Sampling**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: NLI, NLU  
[Paper Link](http://arxiv.org/abs/2311.16298v1)  

---


**ABSTRACT**  
Modern ML systems ingest data aggregated from diverse sources, such as synthetic, human-annotated, and live customer traffic. Understanding \textit{which} examples are important to the performance of a learning algorithm is crucial for efficient model training. Recently, a growing body of literature has given rise to various "influence scores," which use training artifacts such as model confidence or checkpointed gradients to identify important subsets of data. However, these methods have primarily been developed in computer vision settings, and it remains unclear how well they generalize to language-based tasks using pretrained models.   In this paper, we explore the applicability of influence scores in language classification tasks. We evaluate a diverse subset of these scores on the SNLI dataset by quantifying accuracy changes in response to pruning training data through random and influence-score-based sampling. We then stress-test one of the scores -- "variance of gradients" (VoG) from Agarwal et al. (2022) -- in an NLU model stack that was exposed to dynamic user speech patterns in a voice assistant type of setting. Our experiments demonstrate that in many cases, encoder-based language models can be finetuned on roughly 50% of the original data without degradation in performance metrics. Along the way, we summarize lessons learned from applying out-of-the-box implementations of influence scores, quantify the effects of noisy and class-imbalanced data, and offer recommendations on score-based sampling for better accuracy and training efficiency.

{{</citation>}}


### (93/147) A Graph Neural Network-Based QUBO-Formulated Hamiltonian-Inspired Loss Function for Combinatorial Optimization using Reinforcement Learning (Redwan Ahmed Rizvee et al., 2023)

{{<citation>}}

Redwan Ahmed Rizvee, Raheeb Hasan, Md. Mosaddek Khan. (2023)  
**A Graph Neural Network-Based QUBO-Formulated Hamiltonian-Inspired Loss Function for Combinatorial Optimization using Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.16277v1)  

---


**ABSTRACT**  
Quadratic Unconstrained Binary Optimization (QUBO) is a generic technique to model various NP-hard Combinatorial Optimization problems (CO) in the form of binary variables. Ising Hamiltonian is used to model the energy function of a system. QUBO to Ising Hamiltonian is regarded as a technique to solve various canonical optimization problems through quantum optimization algorithms. Recently, PI-GNN, a generic framework, has been proposed to address CO problems over graphs based on Graph Neural Network (GNN) architecture. They introduced a generic QUBO-formulated Hamiltonian-inspired loss function that was directly optimized using GNN. PI-GNN is highly scalable but there lies a noticeable decrease in the number of satisfied constraints when compared to problem-specific algorithms and becomes more pronounced with increased graph densities. Here, We identify a behavioral pattern related to it and devise strategies to improve its performance. Another group of literature uses Reinforcement learning (RL) to solve the aforementioned NP-hard problems using problem-specific reward functions. In this work, we also focus on creating a bridge between the RL-based solutions and the QUBO-formulated Hamiltonian. We formulate and empirically evaluate the compatibility of the QUBO-formulated Hamiltonian as the generic reward function in the RL-based paradigm in the form of rewards. Furthermore, we also introduce a novel Monty Carlo Tree Search-based strategy with GNN where we apply a guided search through manual perturbation of node labels during training. We empirically evaluated our methods and observed up to 44% improvement in the number of constraint violations compared to the PI-GNN.

{{</citation>}}


### (94/147) Metric Space Magnitude for Evaluating Unsupervised Representation Learning (Katharina Limbeck et al., 2023)

{{<citation>}}

Katharina Limbeck, Rayna Andreeva, Rik Sarkar, Bastian Rieck. (2023)  
**Metric Space Magnitude for Evaluating Unsupervised Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-GT, stat-ML  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.16054v1)  

---


**ABSTRACT**  
The magnitude of a metric space was recently established as a novel invariant, providing a measure of the `effective size' of a space across multiple scales. By capturing both geometrical and topological properties of data, magnitude is poised to address challenges in unsupervised representation learning tasks. We formalise a novel notion of dissimilarity between magnitude functions of finite metric spaces and use them to derive a quality measure for dimensionality reduction tasks. Our measure is provably stable under perturbations of the data, can be efficiently calculated, and enables a rigorous multi-scale comparison of embeddings. We show the utility of our measure in an experimental suite that comprises different domains and tasks, including the comparison of data visualisations.

{{</citation>}}


### (95/147) Forecasting Auxiliary Energy Consumption for Electric Heavy-Duty Vehicles (Yuantao Fan et al., 2023)

{{<citation>}}

Yuantao Fan, Zhenkan Wang, Sepideh Pashami, Slawomir Nowaczyk, Henrik Ydreskog. (2023)  
**Forecasting Auxiliary Energy Consumption for Electric Heavy-Duty Vehicles**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16003v1)  

---


**ABSTRACT**  
Accurate energy consumption prediction is crucial for optimizing the operation of electric commercial heavy-duty vehicles, e.g., route planning for charging. Moreover, understanding why certain predictions are cast is paramount for such a predictive model to gain user trust and be deployed in practice. Since commercial vehicles operate differently as transportation tasks, ambient, and drivers vary, a heterogeneous population is expected when building an AI system for forecasting energy consumption. The dependencies between the input features and the target values are expected to also differ across sub-populations. One well-known example of such a statistical phenomenon is the Simpson paradox. In this paper, we illustrate that such a setting poses a challenge for existing XAI methods that produce global feature statistics, e.g. LIME or SHAP, causing them to yield misleading results. We demonstrate a potential solution by training multiple regression models on subsets of data. It not only leads to superior regression performance but also more relevant and consistent LIME explanations. Given that the employed groupings correspond to relevant sub-populations, the associations between the input features and the target values are consistent within each cluster but different across clusters. Experiments on both synthetic and real-world datasets show that such splitting of a complex problem into simpler ones yields better regression performance and interpretability.

{{</citation>}}


### (96/147) Sparsify-then-Classify: From Internal Neurons of Large Language Models To Efficient Text Classifiers (Yilun Liu et al., 2023)

{{<citation>}}

Yilun Liu, Difan Jiao, Ashton Anderson. (2023)  
**Sparsify-then-Classify: From Internal Neurons of Large Language Models To Efficient Text Classifiers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.15983v1)  

---


**ABSTRACT**  
Among the many tasks that Large Language Models (LLMs) have revolutionized is text classification. However, existing approaches for applying pretrained LLMs to text classification predominantly rely on using single token outputs from only the last layer of hidden states. As a result, they suffer from limitations in efficiency, task-specificity, and interpretability. In our work, we contribute an approach that uses all internal representations by employing multiple pooling strategies on all activation and hidden states. Our novel lightweight strategy, Sparsify-then-Classify (STC) first sparsifies task-specific features layer-by-layer, then aggregates across layers for text classification. STC can be applied as a seamless plug-and-play module on top of existing LLMs. Our experiments on a comprehensive set of models and datasets demonstrate that STC not only consistently improves the classification performance of pretrained and fine-tuned models, but is also more efficient for both training and inference, and is more intrinsically interpretable.

{{</citation>}}


### (97/147) Soil Organic Carbon Estimation from Climate-related Features with Graph Neural Network (Weiying Zhao et al., 2023)

{{<citation>}}

Weiying Zhao, Natalia Efremova. (2023)  
**Soil Organic Carbon Estimation from Climate-related Features with Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2311.15979v1)  

---


**ABSTRACT**  
Soil organic carbon (SOC) plays a pivotal role in the global carbon cycle, impacting climate dynamics and necessitating accurate estimation for sustainable land and agricultural management. While traditional methods of SOC estimation face resolution and accuracy challenges, recent technological solutions harness remote sensing, machine learning, and high-resolution satellite mapping. Graph Neural Networks (GNNs), especially when integrated with positional encoders, can capture complex relationships between soil and climate. Using the LUCAS database, this study compared four GNN operators in the positional encoder framework. Results revealed that the PESAGE and PETransformer models outperformed others in SOC estimation, indicating their potential in capturing the complex relationship between SOC and climate features. Our findings confirm the feasibility of applications of GNN architectures in SOC prediction, establishing a framework for future explorations of this topic with more advanced GNN models.

{{</citation>}}


### (98/147) Over-Squashing in Riemannian Graph Neural Networks (Julia Balla, 2023)

{{<citation>}}

Julia Balla. (2023)  
**Over-Squashing in Riemannian Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.15945v1)  

---


**ABSTRACT**  
Most graph neural networks (GNNs) are prone to the phenomenon of over-squashing in which node features become insensitive to information from distant nodes in the graph. Recent works have shown that the topology of the graph has the greatest impact on over-squashing, suggesting graph rewiring approaches as a suitable solution. In this work, we explore whether over-squashing can be mitigated through the embedding space of the GNN. In particular, we consider the generalization of Hyperbolic GNNs (HGNNs) to Riemannian manifolds of variable curvature in which the geometry of the embedding space is faithful to the graph's topology. We derive bounds on the sensitivity of the node features in these Riemannian GNNs as the number of layers increases, which yield promising theoretical and empirical results for alleviating over-squashing in graphs with negative curvature.

{{</citation>}}


### (99/147) Reinforcement Learning for Wildfire Mitigation in Simulated Disaster Environments (Alexander Tapley et al., 2023)

{{<citation>}}

Alexander Tapley, Marissa Dotter, Michael Doyle, Aidan Fennelly, Dhanuj Gandikota, Savanna Smith, Michael Threet, Tim Welsh. (2023)  
**Reinforcement Learning for Wildfire Mitigation in Simulated Disaster Environments**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs-SE, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15925v1)  

---


**ABSTRACT**  
Climate change has resulted in a year over year increase in adverse weather and weather conditions which contribute to increasingly severe fire seasons. Without effective mitigation, these fires pose a threat to life, property, ecology, cultural heritage, and critical infrastructure. To better prepare for and react to the increasing threat of wildfires, more accurate fire modelers and mitigation responses are necessary. In this paper, we introduce SimFire, a versatile wildland fire projection simulator designed to generate realistic wildfire scenarios, and SimHarness, a modular agent-based machine learning wrapper capable of automatically generating land management strategies within SimFire to reduce the overall damage to the area. Together, this publicly available system allows researchers and practitioners the ability to emulate and assess the effectiveness of firefighter interventions and formulate strategic plans that prioritize value preservation and resource allocation optimization. The repositories are available for download at https://github.com/mitrefireline.

{{</citation>}}


### (100/147) Diagnosis driven Anomaly Detection for CPS (Henrik S. Steude et al., 2023)

{{<citation>}}

Henrik S. Steude, Lukas Moddemann, Alexander Diedrich, Jonas Ehrhardt, Oliver Niggemann. (2023)  
**Diagnosis driven Anomaly Detection for CPS**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.15924v1)  

---


**ABSTRACT**  
In Cyber-Physical Systems (CPS) research, anomaly detection (detecting abnormal behavior) and diagnosis (identifying the underlying root cause) are often treated as distinct, isolated tasks. However, diagnosis algorithms require symptoms, i.e. temporally and spatially isolated anomalies, as input. Thus, anomaly detection and diagnosis must be developed together to provide a holistic solution for diagnosis in CPS. We therefore propose a method for utilizing deep learning-based anomaly detection to generate inputs for Consistency-Based Diagnosis (CBD). We evaluate our approach on a simulated and a real-world CPS dataset, where our model demonstrates strong performance relative to other state-of-the-art models.

{{</citation>}}


### (101/147) Utilizing Explainability Techniques for Reinforcement Learning Model Assurance (Alexander Tapley et al., 2023)

{{<citation>}}

Alexander Tapley, Kyle Gatesman, Luis Robaina, Brett Bissey, Joseph Weissman. (2023)  
**Utilizing Explainability Techniques for Reinforcement Learning Model Assurance**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15838v1)  

---


**ABSTRACT**  
Explainable Reinforcement Learning (XRL) can provide transparency into the decision-making process of a Deep Reinforcement Learning (DRL) model and increase user trust and adoption in real-world use cases. By utilizing XRL techniques, researchers can identify potential vulnerabilities within a trained DRL model prior to deployment, therefore limiting the potential for mission failure or mistakes by the system. This paper introduces the ARLIN (Assured RL Model Interrogation) Toolkit, an open-source Python library that identifies potential vulnerabilities and critical points within trained DRL models through detailed, human-interpretable explainability outputs. To illustrate ARLIN's effectiveness, we provide explainability visualizations and vulnerability analysis for a publicly available DRL model. The open-source code repository is available for download at https://github.com/mitre/arlin.

{{</citation>}}


### (102/147) Exploring Artificial Intelligence Methods for Energy Prediction in Healthcare Facilities: An In-Depth Extended Systematic Review (Marjan FatehiJananloo et al., 2023)

{{<citation>}}

Marjan FatehiJananloo, Helen Stopps, J. J. McArthur. (2023)  
**Exploring Artificial Intelligence Methods for Energy Prediction in Healthcare Facilities: An In-Depth Extended Systematic Review**  

---
Primary Category: cs.LG  
Categories: A-1; I-2; J-2, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15807v1)  

---


**ABSTRACT**  
Hospitals, due to their complexity and unique requirements, play a pivotal role in global energy consumption patterns. This study conducted a comprehensive literature review, utilizing the PRISMA framework, of articles that employed machine learning and artificial intelligence techniques for predicting energy consumption in hospital buildings. Of the 1884 publications identified, 17 were found to address this specific domain and have been thoroughly reviewed to establish the state-of-the-art and identify gaps where future research is needed. This review revealed a diverse range of data inputs influencing energy prediction, with occupancy and meteorological data emerging as significant predictors. However, many studies failed to delve deep into the implications of their data choices, and gaps were evident regarding the understanding of time dynamics, operational status, and preprocessing methods. Machine learning, especially deep learning models like ANNs, have shown potential in this domain, yet they come with challenges, including interpretability and computational demands. The findings underscore the immense potential of AI in optimizing hospital energy consumption but also highlight the need for more comprehensive and granular research. Key areas for future research include the optimization of ANN approaches, new optimization and data integration techniques, the integration of real-time data into Intelligent Energy Management Systems, and increasing focus on long-term energy forecasting.

{{</citation>}}


### (103/147) Attend Who is Weak: Enhancing Graph Condensation via Cross-Free Adversarial Training (Xinglin Li et al., 2023)

{{<citation>}}

Xinglin Li, Kun Wang, Hanhui Deng, Yuxuan Liang, Di Wu. (2023)  
**Attend Who is Weak: Enhancing Graph Condensation via Cross-Free Adversarial Training**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Training, GNN  
[Paper Link](http://arxiv.org/abs/2311.15772v1)  

---


**ABSTRACT**  
In this paper, we study the \textit{graph condensation} problem by compressing the large, complex graph into a concise, synthetic representation that preserves the most essential and discriminative information of structure and features. We seminally propose the concept of Shock Absorber (a type of perturbation) that enhances the robustness and stability of the original graphs against changes in an adversarial training fashion. Concretely, (I) we forcibly match the gradients between pre-selected graph neural networks (GNNs) trained on a synthetic, simplified graph and the original training graph at regularly spaced intervals. (II) Before each update synthetic graph point, a Shock Absorber serves as a gradient attacker to maximize the distance between the synthetic dataset and the original graph by selectively perturbing the parts that are underrepresented or insufficiently informative. We iteratively repeat the above two processes (I and II) in an adversarial training fashion to maintain the highly-informative context without losing correlation with the original dataset. More importantly, our shock absorber and the synthesized graph parallelly share the backward process in a free training manner. Compared to the original adversarial training, it introduces almost no additional time overhead.   We validate our framework across 8 datasets (3 graph and 5 node classification datasets) and achieve prominent results: for example, on Cora, Citeseer and Ogbn-Arxiv, we can gain nearly 1.13% to 5.03% improvements compare with SOTA models. Moreover, our algorithm adds only about 0.2% to 2.2% additional time overhead over Flicker, Citeseer and Ogbn-Arxiv. Compared to the general adversarial training, our approach improves time efficiency by nearly 4-fold.

{{</citation>}}


### (104/147) ChatTraffic: Text-to-Traffic Generation via Diffusion Model (Chengyang Zhang et al., 2023)

{{<citation>}}

Chengyang Zhang, Yong Zhang, Qitan Shao, Bo Li, Yisheng Lv, Xinglin Piao, Baocai Yin. (2023)  
**ChatTraffic: Text-to-Traffic Generation via Diffusion Model**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2311.16203v2)  

---


**ABSTRACT**  
Traffic prediction is one of the most significant foundations in Intelligent Transportation Systems (ITS). Traditional traffic prediction methods rely only on historical traffic data to predict traffic trends and face two main challenges. 1) insensitivity to unusual events. 2) poor performance in long-term prediction. In this work, we explore how generative models combined with text describing the traffic system can be applied for traffic generation and name the task Text-to-Traffic Generation (TTG). The key challenge of the TTG task is how to associate text with the spatial structure of the road network and traffic data for generating traffic situations. To this end, we propose ChatTraffic, the first diffusion model for text-to-traffic generation. To guarantee the consistency between synthetic and real data, we augment a diffusion model with the Graph Convolutional Network (GCN) to extract spatial correlations of traffic data. In addition, we construct a large dataset containing text-traffic pairs for the TTG task. We benchmarked our model qualitatively and quantitatively on the released dataset. The experimental results indicate that ChatTraffic can generate realistic traffic situations from the text. Our code and dataset are available at https://github.com/ChyaZhang/ChatTraffic.

{{</citation>}}


### (105/147) Leveraging Out-of-Domain Data for Domain-Specific Prompt Tuning in Multi-Modal Fake News Detection (Debarshi Brahma et al., 2023)

{{<citation>}}

Debarshi Brahma, Amartya Bhattacharya, Suraj Nagaje Mahadev, Anmol Asati, Vikas Verma, Soma Biswas. (2023)  
**Leveraging Out-of-Domain Data for Domain-Specific Prompt Tuning in Multi-Modal Fake News Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Fake News, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16496v1)  

---


**ABSTRACT**  
The spread of fake news using out-of-context images has become widespread and is a challenging task in this era of information overload. Since annotating huge amounts of such data requires significant time of domain experts, it is imperative to develop methods which can work in limited annotated data scenarios. In this work, we explore whether out-of-domain data can help to improve out-of-context misinformation detection (termed here as multi-modal fake news detection) of a desired domain, eg. politics, healthcare, etc. Towards this goal, we propose a novel framework termed DPOD (Domain-specific Prompt-tuning using Out-of-Domain data). First, to compute generalizable features, we modify the Vision-Language Model, CLIP to extract features that helps to align the representations of the images and corresponding text captions of both the in-domain and out-of-domain data in a label-aware manner. Further, we propose a domain-specific prompt learning technique which leverages the training samples of all the available domains based on the the extent they can be useful to the desired domain. Extensive experiments on a large-scale benchmark dataset, namely NewsClippings demonstrate that the proposed framework achieves state of-the-art performance, significantly surpassing the existing approaches for this challenging task.

{{</citation>}}


### (106/147) Out-of-Distribution Generalized Dynamic Graph Neural Network for Human Albumin Prediction (Zeyang Zhang et al., 2023)

{{<citation>}}

Zeyang Zhang, Xingwang Li, Fei Teng, Ning Lin, Xueling Zhu, Xin Wang, Wenwu Zhu. (2023)  
**Out-of-Distribution Generalized Dynamic Graph Neural Network for Human Albumin Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-LG, cs.LG  
Keywords: Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.15545v1)  

---


**ABSTRACT**  
Human albumin is essential for indicating the body's overall health. Accurately predicting plasma albumin levels and determining appropriate doses are urgent clinical challenges, particularly in critically ill patients, to maintain optimal blood levels. However, human albumin prediction is non-trivial that has to leverage the dynamics of biochemical markers as well as the experience of treating patients. Moreover, the problem of distribution shift is often encountered in real clinical data, which may lead to a decline in the model prediction performance and reduce the reliability of the model's application. In this paper, we propose a framework named Out-of-Distribution Generalized Dynamic Graph Neural Network for Human Albumin Prediction (DyG-HAP), which is able to provide accurate albumin predictions for Intensity Care Unit (ICU) patients during hospitalization. We first model human albumin prediction as a dynamic graph regression problem to model the dynamics and patient relationship. Then, we propose a disentangled dynamic graph attention mechanism to capture and disentangle the patterns whose relationship to labels under distribution shifts is invariant and variant respectively. Last, we propose an invariant dynamic graph regression method to encourage the model to rely on invariant patterns to make predictions. Moreover, we propose a dataset named Albumin level testing and nutritional dosing data for Intensive Care (ANIC) for evaluation. Extensive experiments demonstrate the superiority of our method compared to several baseline methods in human albumin prediction.

{{</citation>}}


### (107/147) SSIN: Self-Supervised Learning for Rainfall Spatial Interpolation (Jia Li et al., 2023)

{{<citation>}}

Jia Li, Yanyan Shen, Lei Chen, Charles Wang Wai NG. (2023)  
**SSIN: Self-Supervised Learning for Rainfall Spatial Interpolation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, physics-ao-ph  
Keywords: BERT, Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2311.15530v1)  

---


**ABSTRACT**  
The acquisition of accurate rainfall distribution in space is an important task in hydrological analysis and natural disaster pre-warning. However, it is impossible to install rain gauges on every corner. Spatial interpolation is a common way to infer rainfall distribution based on available raingauge data. However, the existing works rely on some unrealistic pre-settings to capture spatial correlations, which limits their performance in real scenarios. To tackle this issue, we propose the SSIN, which is a novel data-driven self-supervised learning framework for rainfall spatial interpolation by mining latent spatial patterns from historical observation data. Inspired by the Cloze task and BERT, we fully consider the characteristics of spatial interpolation and design the SpaFormer model based on the Transformer architecture as the core of SSIN. Our main idea is: by constructing rich self-supervision signals via random masking, SpaFormer can learn informative embeddings for raw data and then adaptively model spatial correlations based on rainfall spatial context. Extensive experiments on two real-world raingauge datasets show that our method outperforms the state-of-the-art solutions. In addition, we take traffic spatial interpolation as another use case to further explore the performance of our method, and SpaFormer achieves the best performance on one large real-world traffic dataset, which further confirms the effectiveness and generality of our method.

{{</citation>}}


### (108/147) Automatic Time Signature Determination for New Scores Using Lyrics for Latent Rhythmic Structure (Callie C. Liao et al., 2023)

{{<citation>}}

Callie C. Liao, Duoduo Liao, Jesse Guessford. (2023)  
**Automatic Time Signature Determination for New Scores Using Lyrics for Latent Rhythmic Structure**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs-MM, cs-SD, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15480v1)  

---


**ABSTRACT**  
There has recently been a sharp increase in interest in Artificial Intelligence-Generated Content (AIGC). Despite this, musical components such as time signatures have not been studied sufficiently to form an algorithmic determination approach for new compositions, especially lyrical songs. This is likely because of the neglect of musical details, which is critical for constructing a robust framework. Specifically, time signatures establish the fundamental rhythmic structure for almost all aspects of a song, including the phrases and notes. In this paper, we propose a novel approach that only uses lyrics as input to automatically generate a fitting time signature for lyrical songs and uncover the latent rhythmic structure utilizing explainable machine learning models. In particular, we devise multiple methods that are associated with discovering lyrical patterns and creating new features that simultaneously contain lyrical, rhythmic, and statistical information. In this approach, the best of our experimental results reveal a 97.6% F1 score and a 0.996 Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) score. In conclusion, our research directly generates time signatures from lyrics automatically for new scores utilizing machine learning, which is an innovative idea that approaches an understudied component of musicology and therefore contributes significantly to the future of Artificial Intelligence (AI) music generation.

{{</citation>}}


## cs.CY (4)



### (109/147) Student Mastery or AI Deception? Analyzing ChatGPT's Assessment Proficiency and Evaluating Detection Strategies (Kevin Wang et al., 2023)

{{<citation>}}

Kevin Wang, Seth Akins, Abdallah Mohammed, Ramon Lawrence. (2023)  
**Student Mastery or AI Deception? Analyzing ChatGPT's Assessment Proficiency and Evaluating Detection Strategies**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.16292v1)  

---


**ABSTRACT**  
Generative AI systems such as ChatGPT have a disruptive effect on learning and assessment. Computer science requires practice to develop skills in problem solving and programming that are traditionally developed using assignments. Generative AI has the capability of completing these assignments for students with high accuracy, which dramatically increases the potential for academic integrity issues and students not achieving desired learning outcomes. This work investigates the performance of ChatGPT by evaluating it across three courses (CS1,CS2,databases). ChatGPT completes almost all introductory assessments perfectly. Existing detection methods, such as MOSS and JPlag (based on similarity metrics) and GPTzero (AI detection), have mixed success in identifying AI solutions. Evaluating instructors and teaching assistants using heuristics to distinguish between student and AI code shows that their detection is not sufficiently accurate. These observations emphasize the need for adapting assessments and improved detection methods.

{{</citation>}}


### (110/147) Generative AI and US Intellectual Property Law (Cherie M Poland, 2023)

{{<citation>}}

Cherie M Poland. (2023)  
**Generative AI and US Intellectual Property Law**  

---
Primary Category: cs.CY  
Categories: K-4; K-5, cs-AI, cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.16023v1)  

---


**ABSTRACT**  
The rapidity with which generative AI has been adopted and advanced has raised legal and ethical questions related to the impact on artists rights, content production, data collection, privacy, accuracy of information, and intellectual property rights. Recent administrative and case law challenges have shown that generative AI software systems do not have independent intellectual property rights in the content that they generate. It remains to be seen whether human content creators can retain their intellectual property rights against generative AI software, its developers, operators, and owners for the misappropriation of the work of human creatives, given the metes and bounds of existing law. Early signs from various courts are mixed as to whether and to what degree the results generated by AI models meet the legal standards of infringement under existing law.

{{</citation>}}


### (111/147) Towards Responsible Governance of Biological Design Tools (Richard Moulange et al., 2023)

{{<citation>}}

Richard Moulange, Max Langenkamp, Tessa Alexanian, Samuel Curtis, Morgan Livingston. (2023)  
**Towards Responsible Governance of Biological Design Tools**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-LG, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15936v3)  

---


**ABSTRACT**  
Recent advancements in generative machine learning have enabled rapid progress in biological design tools (BDTs) such as protein structure and sequence prediction models. The unprecedented predictive accuracy and novel design capabilities of BDTs present new and significant dual-use risks. For example, their predictive accuracy allows biological agents, whether vaccines or pathogens, to be developed more quickly, while the design capabilities could be used to discover drugs or evade DNA screening techniques. Similar to other dual-use AI systems, BDTs present a wicked problem: how can regulators uphold public safety without stifling innovation? We highlight how current regulatory proposals that are primarily tailored toward large language models may be less effective for BDTs, which require fewer computational resources to train and are often developed in an open-source manner. We propose a range of measures to mitigate the risk that BDTs are misused, across the areas of responsible development, risk assessment, transparency, access management, cybersecurity, and investing in resilience. Implementing such measures will require close coordination between developers and governments.

{{</citation>}}


### (112/147) Public sentiment analysis and topic modeling regarding ChatGPT in mental health on Reddit: Negative sentiments increase over time (Yunna Cai et al., 2023)

{{<citation>}}

Yunna Cai, Fan Wang, Haowei Wang, Qianwen Qian. (2023)  
**Public sentiment analysis and topic modeling regarding ChatGPT in mental health on Reddit: Negative sentiments increase over time**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.15800v1)  

---


**ABSTRACT**  
In order to uncover users' attitudes towards ChatGPT in mental health, this study examines public opinions about ChatGPT in mental health discussions on Reddit. Researchers used the bert-base-multilingual-uncased-sentiment techniques for sentiment analysis and the BERTopic model for topic modeling. It was found that overall, negative sentiments prevail, followed by positive ones, with neutral sentiments being the least common. The prevalence of negative emotions has increased over time. Negative emotions encompass discussions on ChatGPT providing bad mental health advice, debates on machine vs. human value, the fear of AI, and concerns about Universal Basic Income (UBI). In contrast, positive emotions highlight ChatGPT's effectiveness in counseling, with mentions of keywords like "time" and "wallet." Neutral discussions center around private data concerns. These findings shed light on public attitudes toward ChatGPT in mental health, potentially contributing to the development of trustworthy AI in mental health from the public perspective.

{{</citation>}}


## eess.SY (3)



### (113/147) Optimal Observer Design Using Reinforcement Learning and Quadratic Neural Networks (Soroush Asri et al., 2023)

{{<citation>}}

Soroush Asri, Luis Rodrigues. (2023)  
**Optimal Observer Design Using Reinforcement Learning and Quadratic Neural Networks**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.16272v1)  

---


**ABSTRACT**  
This paper introduces an innovative approach based on policy iteration (PI), a reinforcement learning (RL) algorithm, to obtain an optimal observer with a quadratic cost function. This observer is designed for systems with a given linearized model and a stabilizing Luenberger observer gain. We utilize two-layer quadratic neural networks (QNN) for policy evaluation and derive a linear correction term using the input and output data. This correction term effectively rectifies inaccuracies introduced by the linearized model employed within the observer design. A unique feature of the proposed methodology is that the QNN is trained through convex optimization. The main advantage is that the QNN's input-output mapping has an analytical expression as a quadratic form, which can then be used to obtain a linear correction term policy. This is in stark contrast to the available techniques in the literature that must train a second neural network to obtain policy improvement. It is proven that the obtained linear correction term is optimal for linear systems, as both the value function and the QNN's input-output mapping are quadratic. The proposed method is applied to a simple pendulum, demonstrating an enhanced correction term policy compared to relying solely on the linearized model. This shows its promise for addressing nonlinear systems.

{{</citation>}}


### (114/147) Networked Multiagent Safe Reinforcement Learning for Low-carbon Demand Management in Distribution Network (Jichen Zhang et al., 2023)

{{<citation>}}

Jichen Zhang, Linwei Sang, Yinliang Xu, Hongbin Sun. (2023)  
**Networked Multiagent Safe Reinforcement Learning for Low-carbon Demand Management in Distribution Network**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15594v1)  

---


**ABSTRACT**  
This paper proposes a multiagent based bi-level operation framework for the low-carbon demand management in distribution networks considering the carbon emission allowance on the demand side. In the upper level, the aggregate load agents optimize the control signals for various types of loads to maximize the profits; in the lower level, the distribution network operator makes optimal dispatching decisions to minimize the operational costs and calculates the distribution locational marginal price and carbon intensity. The distributed flexible load agent has only incomplete information of the distribution network and cooperates with other agents using networked communication. Finally, the problem is formulated into a networked multi-agent constrained Markov decision process, which is solved using a safe reinforcement learning algorithm called consensus multi-agent constrained policy optimization considering the carbon emission allowance for each agent. Case studies with the IEEE 33-bus and 123-bus distribution network systems demonstrate the effectiveness of the proposed approach, in terms of satisfying the carbon emission constraint on demand side, ensuring the safe operation of the distribution network and preserving privacy of both sides.

{{</citation>}}


### (115/147) Active Foundational Models for Fault Diagnosis of Electrical Motors (Sriram Anbalagan et al., 2023)

{{<citation>}}

Sriram Anbalagan, Sai Shashank GP, Deepesh Agarwal, Balasubramaniam Natarajan, Babji Srinivasan. (2023)  
**Active Foundational Models for Fault Diagnosis of Electrical Motors**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Active Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.15516v1)  

---


**ABSTRACT**  
Fault detection and diagnosis of electrical motors are of utmost importance in ensuring the safe and reliable operation of several industrial systems. Detection and diagnosis of faults at the incipient stage allows corrective actions to be taken in order to reduce the severity of faults. The existing data-driven deep learning approaches for machine fault diagnosis rely extensively on huge amounts of labeled samples, where annotations are expensive and time-consuming. However, a major portion of unlabeled condition monitoring data is not exploited in the training process. To overcome this limitation, we propose a foundational model-based Active Learning framework that utilizes less amount of labeled samples, which are most informative and harnesses a large amount of available unlabeled data by effectively combining Active Learning and Contrastive Self-Supervised Learning techniques. It consists of a transformer network-based backbone model trained using an advanced nearest-neighbor contrastive self-supervised learning method. This approach empowers the backbone to learn improved representations of samples derived from raw, unlabeled vibration data. Subsequently, the backbone can undergo fine-tuning to address a range of downstream tasks, both within the same machines and across different machines. The effectiveness of the proposed methodology has been assessed through the fine-tuning of the backbone for multiple target tasks using three distinct machine-bearing fault datasets. The experimental evaluation demonstrates a superior performance as compared to existing state-of-the-art fault diagnosis methods with less amount of labeled data.

{{</citation>}}


## quant-ph (2)



### (116/147) Transformer-QEC: Quantum Error Correction Code Decoding with Transferable Transformers (Hanrui Wang et al., 2023)

{{<citation>}}

Hanrui Wang, Pengyu Liu, Kevin Shao, Dantong Li, Jiaqi Gu, David Z. Pan, Yongshan Ding, Song Han. (2023)  
**Transformer-QEC: Quantum Error Correction Code Decoding with Transferable Transformers**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-AR, cs-ET, cs-LG, quant-ph, quant-ph  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.16082v1)  

---


**ABSTRACT**  
Quantum computing has the potential to solve problems that are intractable for classical systems, yet the high error rates in contemporary quantum devices often exceed tolerable limits for useful algorithm execution. Quantum Error Correction (QEC) mitigates this by employing redundancy, distributing quantum information across multiple data qubits and utilizing syndrome qubits to monitor their states for errors. The syndromes are subsequently interpreted by a decoding algorithm to identify and correct errors in the data qubits. This task is complex due to the multiplicity of error sources affecting both data and syndrome qubits as well as syndrome extraction operations. Additionally, identical syndromes can emanate from different error sources, necessitating a decoding algorithm that evaluates syndromes collectively. Although machine learning (ML) decoders such as multi-layer perceptrons (MLPs) and convolutional neural networks (CNNs) have been proposed, they often focus on local syndrome regions and require retraining when adjusting for different code distances. We introduce a transformer-based QEC decoder which employs self-attention to achieve a global receptive field across all input syndromes. It incorporates a mixed loss training approach, combining both local physical error and global parity label losses. Moreover, the transformer architecture's inherent adaptability to variable-length inputs allows for efficient transfer learning, enabling the decoder to adapt to varying code distances without retraining.   Evaluation on six code distances and ten different error configurations demonstrates that our model consistently outperforms non-ML decoders, such as Union Find (UF) and Minimum Weight Perfect Matching (MWPM), and other ML decoders, thereby achieving best logical error rates. Moreover, the transfer learning can save over 10x of training cost.

{{</citation>}}


### (117/147) Towards Transfer Learning for Large-Scale Image Classification Using Annealing-based Quantum Boltzmann Machines (Daniëlle Schuman et al., 2023)

{{<citation>}}

Daniëlle Schuman, Leo Sünkel, Philipp Altmann, Jonas Stein, Christoph Roch, Thomas Gabor, Claudia Linnhoff-Popien. (2023)  
**Towards Transfer Learning for Large-Scale Image Classification Using Annealing-based Quantum Boltzmann Machines**  

---
Primary Category: quant-ph  
Categories: cs-ET, cs-LG, eess-IV, quant-ph, quant-ph  
Keywords: Image Classification, QA  
[Paper Link](http://arxiv.org/abs/2311.15966v1)  

---


**ABSTRACT**  
Quantum Transfer Learning (QTL) recently gained popularity as a hybrid quantum-classical approach for image classification tasks by efficiently combining the feature extraction capabilities of large Convolutional Neural Networks with the potential benefits of Quantum Machine Learning (QML). Existing approaches, however, only utilize gate-based Variational Quantum Circuits for the quantum part of these procedures. In this work we present an approach to employ Quantum Annealing (QA) in QTL-based image classification. Specifically, we propose using annealing-based Quantum Boltzmann Machines as part of a hybrid quantum-classical pipeline to learn the classification of real-world, large-scale data such as medical images through supervised training. We demonstrate our approach by applying it to the three-class COVID-CT-MD dataset, a collection of lung Computed Tomography (CT) scan slices. Using Simulated Annealing as a stand-in for actual QA, we compare our method to classical transfer learning, using a neural network of the same order of magnitude, to display its improved classification performance. We find that our approach consistently outperforms its classical baseline in terms of test accuracy and AUC-ROC-Score and needs less training epochs to do this.

{{</citation>}}


## eess.IV (2)



### (118/147) Seeing Beyond Cancer: Multi-Institutional Validation of Object Localization and 3D Semantic Segmentation using Deep Learning for Breast MRI (Arda Pekis et al., 2023)

{{<citation>}}

Arda Pekis, Vignesh Kannan, Evandros Kaklamanos, Anu Antony, Snehal Patel, Tyler Earnest. (2023)  
**Seeing Beyond Cancer: Multi-Institutional Validation of Object Localization and 3D Semantic Segmentation using Deep Learning for Breast MRI**  

---
Primary Category: eess.IV  
Categories: I-4-6; J-3, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: ImageNet, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.16213v1)  

---


**ABSTRACT**  
The clinical management of breast cancer depends on an accurate understanding of the tumor and its anatomical context to adjacent tissues and landmark structures. This context may be provided by semantic segmentation methods; however, previous works have been largely limited to a singular focus on the tumor alone and rarely other tissue types. In contrast, we present a method that exploits tissue-tissue interactions to accurately segment every major tissue type in the breast including: chest wall, skin, adipose tissue, fibroglandular tissue, vasculature and tumor via standard-of-care Dynamic Contrast Enhanced MRI. Comparing our method to prior state-of-the-art, we achieved a superior Dice score on tumor segmentation while maintaining competitive performance on other studied tissues across multiple institutions. Briefly, our method proceeds by localizing the tumor using 2D object detectors, then segmenting the tumor and surrounding tissues independently using two 3D U-nets, and finally integrating these results while mitigating false positives by checking for anatomically plausible tissue-tissue contacts. The object detection models were pre-trained on ImageNet and COCO, and operated on MIP (maximum intensity projection) images in the axial and sagittal planes, establishing a 3D tumor bounding box. By integrating multiple relevant peri-tumoral tissues, our work enables clinical applications in breast cancer staging, prognosis and surgical planning.

{{</citation>}}


### (119/147) Spatially Covariant Image Registration with Text Prompts (Hang Zhang et al., 2023)

{{<citation>}}

Hang Zhang, Xiang Chen, Rongguang Wang, Renjiu Hu, Dongdong Liu, Gaolei Li. (2023)  
**Spatially Covariant Image Registration with Text Prompts**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15607v1)  

---


**ABSTRACT**  
Medical images are often characterized by their structured anatomical representations and spatially inhomogeneous contrasts. Leveraging anatomical priors in neural networks can greatly enhance their utility in resource-constrained clinical settings. Prior research has harnessed such information for image segmentation, yet progress in deformable image registration has been modest. Our work introduces textSCF, a novel method that integrates spatially covariant filters and textual anatomical prompts encoded by visual-language models, to fill this gap. This approach optimizes an implicit function that correlates text embeddings of anatomical regions to filter weights, relaxing the typical translation-invariance constraint of convolutional operations. TextSCF not only boosts computational efficiency but can also retain or improve registration accuracy. By capturing the contextual interplay between anatomical regions, it offers impressive inter-regional transferability and the ability to preserve structural discontinuities during registration. TextSCF's performance has been rigorously tested on inter-subject brain MRI and abdominal CT registration tasks, outperforming existing state-of-the-art models in the MICCAI Learn2Reg 2021 challenge and leading the leaderboard. In abdominal registrations, textSCF's larger model variant improved the Dice score by 11.3% over the second-best model, while its smaller variant maintained similar accuracy but with an 89.13% reduction in network parameters and a 98.34\% decrease in computational operations.

{{</citation>}}


## cs.RO (5)



### (120/147) Evaluating the Impact of Personalized Value Alignment in Human-Robot Interaction: Insights into Trust and Team Performance Outcomes (Shreyas Bhat et al., 2023)

{{<citation>}}

Shreyas Bhat, Joseph B. Lyons, Cong Shi, X. Jessie Yang. (2023)  
**Evaluating the Impact of Personalized Value Alignment in Human-Robot Interaction: Insights into Trust and Team Performance Outcomes**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.16051v1)  

---


**ABSTRACT**  
This paper examines the effect of real-time, personalized alignment of a robot's reward function to the human's values on trust and team performance. We present and compare three distinct robot interaction strategies: a non-learner strategy where the robot presumes the human's reward function mirrors its own, a non-adaptive-learner strategy in which the robot learns the human's reward function for trust estimation and human behavior modeling, but still optimizes its own reward function, and an adaptive-learner strategy in which the robot learns the human's reward function and adopts it as its own. Two human-subject experiments with a total number of 54 participants were conducted. In both experiments, the human-robot team searches for potential threats in a town. The team sequentially goes through search sites to look for threats. We model the interaction between the human and the robot as a trust-aware Markov Decision Process (trust-aware MDP) and use Bayesian Inverse Reinforcement Learning (IRL) to estimate the reward weights of the human as they interact with the robot. In Experiment 1, we start our learning algorithm with an informed prior of the human's values/goals. In Experiment 2, we start the learning algorithm with an uninformed prior. Results indicate that when starting with a good informed prior, personalized value alignment does not seem to benefit trust or team performance. On the other hand, when an informed prior is unavailable, alignment to the human's values leads to high trust and higher perceived performance while maintaining the same objective team performance.

{{</citation>}}


### (121/147) Modular Customizable ROS-Based Framework for Rapid Development of Social Robots (Mahta Akhyani et al., 2023)

{{<citation>}}

Mahta Akhyani, Hadi Moradi. (2023)  
**Modular Customizable ROS-Based Framework for Rapid Development of Social Robots**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SE, cs-SY, cs.RO, eess-IV, eess-SY  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.15780v1)  

---


**ABSTRACT**  
Developing socially competent robots requires tight integration of robotics, computer vision, speech processing, and web technologies. We present the Socially-interactive Robot Software platform (SROS), an open-source framework addressing this need through a modular layered architecture. SROS bridges the Robot Operating System (ROS) layer for mobility with web and Android interface layers using standard messaging and APIs. Specialized perceptual and interactive skills are implemented as ROS services for reusable deployment on any robot. This facilitates rapid prototyping of collaborative behaviors that synchronize perception with physical actuation. We experimentally validated core SROS technologies including computer vision, speech processing, and GPT2 autocomplete speech implemented as plug-and-play ROS services. Modularity is demonstrated through the successful integration of an additional ROS package, without changes to hardware or software platforms. The capabilities enabled confirm SROS's effectiveness in developing socially interactive robots through synchronized cross-domain interaction. Through demonstrations showing synchronized multimodal behaviors on an example platform, we illustrate how the SROS architectural approach addresses shortcomings of previous work by lowering barriers for researchers to advance the state-of-the-art in adaptive, collaborative customizable human-robot systems through novel applications integrating perceptual and social abilities.

{{</citation>}}


### (122/147) SceneDM: Scene-level Multi-agent Trajectory Generation with Consistent Diffusion Models (Zhiming Guo et al., 2023)

{{<citation>}}

Zhiming Guo, Xing Gao, Jianlan Zhou, Xinyu Cai, Botian Shi. (2023)  
**SceneDM: Scene-level Multi-agent Trajectory Generation with Consistent Diffusion Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.15736v1)  

---


**ABSTRACT**  
Realistic scene-level multi-agent motion simulations are crucial for developing and evaluating self-driving algorithms. However, most existing works focus on generating trajectories for a certain single agent type, and typically ignore the consistency of generated trajectories. In this paper, we propose a novel framework based on diffusion models, called SceneDM, to generate joint and consistent future motions of all the agents, including vehicles, bicycles, pedestrians, etc., in a scene. To enhance the consistency of the generated trajectories, we resort to a new Transformer-based network to effectively handle agent-agent interactions in the inverse process of motion diffusion. In consideration of the smoothness of agent trajectories, we further design a simple yet effective consistent diffusion approach, to improve the model in exploiting short-term temporal dependencies. Furthermore, a scene-level scoring function is attached to evaluate the safety and road-adherence of the generated agent's motions and help filter out unrealistic simulations. Finally, SceneDM achieves state-of-the-art results on the Waymo Sim Agents Benchmark. Project webpage is available at https://alperen-hub.github.io/SceneDM.

{{</citation>}}


### (123/147) MOT-DETR: 3D Single Shot Detection and Tracking with Transformers to build 3D representations for Agro-Food Robots (David Rapado-Rincon et al., 2023)

{{<citation>}}

David Rapado-Rincon, Henk Nap, Katarina Smolenova, Eldert J. van Henten, Gert Kootstra. (2023)  
**MOT-DETR: 3D Single Shot Detection and Tracking with Transformers to build 3D representations for Agro-Food Robots**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.15674v1)  

---


**ABSTRACT**  
In the current demand for automation in the agro-food industry, accurately detecting and localizing relevant objects in 3D is essential for successful robotic operations. However, this is a challenge due the presence of occlusions. Multi-view perception approaches allow robots to overcome occlusions, but a tracking component is needed to associate the objects detected by the robot over multiple viewpoints. Most multi-object tracking (MOT) algorithms are designed for high frame rate sequences and struggle with the occlusions generated by robots' motions and 3D environments. In this paper, we introduce MOT-DETR, a novel approach to detect and track objects in 3D over time using a combination of convolutional networks and transformers. Our method processes 2D and 3D data, and employs a transformer architecture to perform data fusion. We show that MOT-DETR outperforms state-of-the-art multi-object tracking methods. Furthermore, we prove that MOT-DETR can leverage 3D data to deal with long-term occlusions and large frame-to-frame distances better than state-of-the-art methods. Finally, we show how our method is resilient to camera pose noise that can affect the accuracy of point clouds. The implementation of MOT-DETR can be found here: https://github.com/drapado/mot-detr

{{</citation>}}


### (124/147) RoboGPT: an intelligent agent of making embodied long-term decisions for daily instruction tasks (Yaran Chen et al., 2023)

{{<citation>}}

Yaran Chen, Wenbo Cui, Yuanwen Chen, Mining Tan, Xinyao Zhang, Dongbin Zhao, He Wang. (2023)  
**RoboGPT: an intelligent agent of making embodied long-term decisions for daily instruction tasks**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.15649v1)  

---


**ABSTRACT**  
Robotic agents must master common sense and long-term sequential decisions to solve daily tasks through natural language instruction. The developments in Large Language Models (LLMs) in natural language processing have inspired efforts to use LLMs in complex robot planning. Despite LLMs' great generalization and comprehension of instruction tasks, LLMs-generated task plans sometimes lack feasibility and correctness. To address the problem, we propose a RoboGPT agent\footnote{our code and dataset will be released soon} for making embodied long-term decisions for daily tasks, with two modules: 1) LLMs-based planning with re-plan to break the task into multiple sub-goals; 2) RoboSkill individually designed for sub-goals to learn better navigation and manipulation skills. The LLMs-based planning is enhanced with a new robotic dataset and re-plan, called RoboGPT. The new robotic dataset of 67k daily instruction tasks is gathered for fine-tuning the Llama model and obtaining RoboGPT. RoboGPT planner with strong generalization can plan hundreds of daily instruction tasks. Additionally, a low-computational Re-Plan module is designed to allow plans to flexibly adapt to the environment, thereby addressing the nomenclature diversity challenge. The proposed RoboGPT agent outperforms SOTA methods on the ALFRED daily tasks. Moreover, RoboGPT planner exceeds SOTA LLM-based planners like ChatGPT in task-planning rationality for hundreds of unseen daily tasks, and even other domain tasks, while keeping the large model's original broad application and generality.

{{</citation>}}


## cs.DM (1)



### (125/147) Application of Diagnostic Test Methods To The Classification Of Time Series With Discrete Values (Artyom Gevorgyan et al., 2023)

{{<citation>}}

Artyom Gevorgyan, Albert Gevorgyan. (2023)  
**Application of Diagnostic Test Methods To The Classification Of Time Series With Discrete Values**  

---
Primary Category: cs.DM  
Categories: cs-DM, cs.DM  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.16034v1)  

---


**ABSTRACT**  
Discrete-value time series are sequences of measurements where each measurement is a discrete (categorical or integer) value. These time series are widely used in various fields, and their classification and clustering are essential for data analysis. This article presents the possibility of applying diagnostic test methods to such time series and estimates the probability of finding ``matching tests''.

{{</citation>}}


## cs.HC (2)



### (126/147) An HCAI Methodological Framework: Putting It Into Action to Enable Human-Centered AI (Wei Xu et al., 2023)

{{<citation>}}

Wei Xu, Zaifeng Gao, Marvin Dainoff. (2023)  
**An HCAI Methodological Framework: Putting It Into Action to Enable Human-Centered AI**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16027v2)  

---


**ABSTRACT**  
Human-centered AI (HCAI), as a design philosophy, advocates prioritizing humans in designing, developing, and deploying intelligent systems, aiming to maximize the benefits of AI technology to humans and avoid its potential adverse effects. While HCAI has gained momentum, the lack of guidance on methodology in its implementation makes its adoption challenging. After assessing the needs for a methodological framework for HCAI, this paper first proposes a comprehensive and interdisciplinary HCAI methodological framework integrated with seven components, including design goals, design principles, implementation approaches, design paradigms, interdisciplinary teams, methods, and processes. THe implications of the framework are also discussed. This paper also presents a "three-layer" approach to facilitate the implementation of the framework. We believe the proposed framework is systematic and executable, which can overcome the weaknesses in current frameworks and the challenges currently faced in implementing HCAI. Thus, the framework can help put it into action to develop, transfer, and implement HCAI in practice, eventually enabling the design, development, and deployment of HCAI-based intelligent systems.

{{</citation>}}


### (127/147) Decoding Logic Errors: A Comparative Study on Bug Detection by Students and Large Language Models (Stephen MacNeil et al., 2023)

{{<citation>}}

Stephen MacNeil, Paul Denny, Andrew Tran, Juho Leinonen, Seth Bernstein, Arto Hellas, Sami Sarsa, Joanne Kim. (2023)  
**Decoding Logic Errors: A Comparative Study on Bug Detection by Students and Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16017v1)  

---


**ABSTRACT**  
Identifying and resolving logic errors can be one of the most frustrating challenges for novices programmers. Unlike syntax errors, for which a compiler or interpreter can issue a message, logic errors can be subtle. In certain conditions, buggy code may even exhibit correct behavior -- in other cases, the issue might be about how a problem statement has been interpreted. Such errors can be hard to spot when reading the code, and they can also at times be missed by automated tests. There is great educational potential in automatically detecting logic errors, especially when paired with suitable feedback for novices. Large language models (LLMs) have recently demonstrated surprising performance for a range of computing tasks, including generating and explaining code. These capabilities are closely linked to code syntax, which aligns with the next token prediction behavior of LLMs. On the other hand, logic errors relate to the runtime performance of code and thus may not be as well suited to analysis by LLMs. To explore this, we investigate the performance of two popular LLMs, GPT-3 and GPT-4, for detecting and providing a novice-friendly explanation of logic errors. We compare LLM performance with a large cohort of introductory computing students $(n=964)$ solving the same error detection task. Through a mixed-methods analysis of student and model responses, we observe significant improvement in logic error identification between the previous and current generation of LLMs, and find that both LLM generations significantly outperform students. We outline how such models could be integrated into computing education tools, and discuss their potential for supporting students when learning programming.

{{</citation>}}


## math.NA (1)



### (128/147) Sketched and Truncated Polynomial Krylov Subspace Methods: Matrix Equations (Davide Palitta et al., 2023)

{{<citation>}}

Davide Palitta, Marcel Schweitzer, Valeria Simoncini. (2023)  
**Sketched and Truncated Polynomial Krylov Subspace Methods: Matrix Equations**  

---
Primary Category: math.NA  
Categories: 65F45, 68W20, 65F25, 65F50, cs-NA, math-NA, math.NA  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2311.16019v1)  

---


**ABSTRACT**  
Thanks to its great potential in reducing both computational cost and memory requirements, combining sketching and Krylov subspace techniques has attracted a lot of attention in the recent literature on projection methods for linear systems, matrix function approximations, and eigenvalue problems. Applying this appealing strategy in the context of linear matrix equations turns out to be far more involved than a straightforward generalization. These difficulties include establishing well-posedness of the projected problem and deriving possible error estimates depending on the sketching properties. Further computational complications include the lack of a natural residual norm estimate and of an explicit basis for the generated subspace. In this paper we propose a new sketched-and-truncated polynomial Krylov subspace method for Sylvester equations that aims to address all these issues. The potential of our novel approach, in terms of both computational time and storage demand, is illustrated with numerical experiments. Comparisons with a state-of-the-art projection scheme based on rational Krylov subspaces are also included.

{{</citation>}}


## cs.CR (6)



### (129/147) RIDE: Real-time Intrusion Detection via Explainable Machine Learning Implemented in a Memristor Hardware Architecture (Jingdi Chen et al., 2023)

{{<citation>}}

Jingdi Chen, Lei Zhang, Joseph Riem, Gina Adam, Nathaniel D. Bastian, Tian Lan. (2023)  
**RIDE: Real-time Intrusion Detection via Explainable Machine Learning Implemented in a Memristor Hardware Architecture**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2311.16018v1)  

---


**ABSTRACT**  
Deep Learning (DL) based methods have shown great promise in network intrusion detection by identifying malicious network traffic behavior patterns with high accuracy, but their applications to real-time, packet-level detections in high-speed communication networks are challenging due to the high computation time and resource requirements of Deep Neural Networks (DNNs), as well as lack of explainability. To this end, we propose a packet-level network intrusion detection solution that makes novel use of Recurrent Autoencoders to integrate an arbitrary-length sequence of packets into a more compact joint feature embedding, which is fed into a DNN-based classifier. To enable explainability and support real-time detections at micro-second speed, we further develop a Software-Hardware Co-Design approach to efficiently realize the proposed solution by converting the learned detection policies into decision trees and implementing them using an emerging architecture based on memristor devices. By jointly optimizing associated software and hardware constraints, we show that our approach leads to an extremely efficient, real-time solution with high detection accuracy at the packet level. Evaluation results on real-world datasets (e.g., UNSW and CIC-IDS datasets) demonstrate nearly three-nines detection accuracy with a substantial speedup of nearly four orders of magnitude.

{{</citation>}}


### (130/147) Microarchitectural Security of AWS Firecracker VMM for Serverless Cloud Platforms (Zane Weissman et al., 2023)

{{<citation>}}

Zane Weissman, Thore Tiemann, Thomas Eisenbarth, Berk Sunar. (2023)  
**Microarchitectural Security of AWS Firecracker VMM for Serverless Cloud Platforms**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AWS, Amazon, Security  
[Paper Link](http://arxiv.org/abs/2311.15999v1)  

---


**ABSTRACT**  
Firecracker is a virtual machine manager (VMM) built by Amazon Web Services (AWS) for serverless cloud platforms, services that run code for end users on a per-task basis, automatically managing server infrastructure. Firecracker provides fast and lightweight VMs and promises a combination of the speed of containers, typically used to isolate small tasks, and the security of VMs, which tend to provide greater isolation at the cost of performance. This combination of security and efficiency, AWS claims, makes it not only possible but safe to run thousands of user tasks from different users on the same hardware, with the host system frequently switching between active tasks. Though AWS states that microarchitectural attacks are included in their threat model, this class of attacks directly relies on shared hardware, just as the scalability of serverless computing relies on sharing hardware between unprecedented numbers of users. In this work, we investigate how secure Firecracker is against microarchitectural attacks. First, we review Firecracker's stated isolation model and recommended best practices for deployment, identify potential threat models for serverless platforms, and analyze potential weak points. Then, we use microarchitectural attack proof-of-concepts to test the isolation provided by Firecracker and find that it offers little protection against Spectre or MDS attacks. We discover two particularly concerning cases: 1) a Medusa variant that threatens Firecracker VMs but not processes running outside them, and is not mitigated by defenses recommended by AWS, and 2) a Spectre-PHT variant that remains exploitable even if recommended countermeasures are in place and SMT is disabled in the system. In summary, we show that AWS overstates the security inherent to the Firecracker VMM and provides incomplete guidance for properly securing cloud systems that use Firecracker.

{{</citation>}}


### (131/147) When Graph Convolution Meets Double Attention: Online Privacy Disclosure Detection with Multi-Label Text Classification (Zhanbo Liang et al., 2023)

{{<citation>}}

Zhanbo Liang, Jie Guo, Weidong Qiu, Zheng Huang, Shujun Li. (2023)  
**When Graph Convolution Meets Double Attention: Online Privacy Disclosure Detection with Multi-Label Text Classification**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Attention, Text Classification, Twitter  
[Paper Link](http://arxiv.org/abs/2311.15917v1)  

---


**ABSTRACT**  
With the rise of Web 2.0 platforms such as online social media, people's private information, such as their location, occupation and even family information, is often inadvertently disclosed through online discussions. Therefore, it is important to detect such unwanted privacy disclosures to help alert people affected and the online platform. In this paper, privacy disclosure detection is modeled as a multi-label text classification (MLTC) problem, and a new privacy disclosure detection model is proposed to construct an MLTC classifier for detecting online privacy disclosures. This classifier takes an online post as the input and outputs multiple labels, each reflecting a possible privacy disclosure. The proposed presentation method combines three different sources of information, the input text itself, the label-to-text correlation and the label-to-label correlation. A double-attention mechanism is used to combine the first two sources of information, and a graph convolutional network (GCN) is employed to extract the third source of information that is then used to help fuse features extracted from the first two sources of information. Our extensive experimental results, obtained on a public dataset of privacy-disclosing posts on Twitter, demonstrated that our proposed privacy disclosure detection method significantly and consistently outperformed other state-of-the-art methods in terms of all key performance indicators.

{{</citation>}}


### (132/147) Towards Adaptive RF Fingerprint-based Authentication of IIoT devices (Emmanuel Lomba et al., 2023)

{{<citation>}}

Emmanuel Lomba, Ricardo Severino, Ana Fernández Vilas. (2023)  
**Towards Adaptive RF Fingerprint-based Authentication of IIoT devices**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15888v1)  

---


**ABSTRACT**  
As IoT technologies mature, they are increasingly finding their way into more sensitive domains, such as Medical and Industrial IoT, in which safety and cyber-security are of great importance. While the number of deployed IoT devices continues to increase exponentially, they still present severe cyber-security vulnerabilities. Effective authentication is paramount to support trustworthy IIoT communications, however, current solutions focus on upper-layer identity verification or key-based cryptography which are often inadequate to the heterogeneous IIoT environment. In this work, we present a first step towards achieving powerful and flexible IIoT device authentication, by leveraging AI adaptive Radio Frequency Fingerprinting technique selection and tuning, at the PHY layer for highly accurate device authentication over challenging RF environments.

{{</citation>}}


### (133/147) Ontologising Trustworthy in the Telecommunications Domain (Ian Oliver et al., 2023)

{{<citation>}}

Ian Oliver, Pekka Kuure, Wiktor Sedkowski, Thore Sommer. (2023)  
**Ontologising Trustworthy in the Telecommunications Domain**  

---
Primary Category: cs.CR  
Categories: D-2-1, cs-CR, cs-SE, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15839v1)  

---


**ABSTRACT**  
Based upon trusted and confidential computing platforms, telecommunications systems must provide guaranteed security for the processes and data running atop them. This in turn requires us to provide trustworthy systems. The term trustworthy is poorly defined with corresponding misunderstanding and misapplication. We present a definition of this term, as well as others, demonstrate its application against certain telecommunications use cases and address how the learnings from ontologising these structures contribute to standardisation and the necessity for FAIR ontologies across telecommunications standards and hosting organisations.

{{</citation>}}


### (134/147) Privacy-Preserving Data Sharing in Agriculture: Enforcing Policy Rules for Secure and Confidential Data Synthesis (Anantaa Kotal et al., 2023)

{{<citation>}}

Anantaa Kotal, Lavanya Elluri, Deepti Gupta, Varun Mandalapu, Anupam Joshi. (2023)  
**Privacy-Preserving Data Sharing in Agriculture: Enforcing Policy Rules for Secure and Confidential Data Synthesis**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15460v1)  

---


**ABSTRACT**  
Big Data empowers the farming community with the information needed to optimize resource usage, increase productivity, and enhance the sustainability of agricultural practices. The use of Big Data in farming requires the collection and analysis of data from various sources such as sensors, satellites, and farmer surveys. While Big Data can provide the farming community with valuable insights and improve efficiency, there is significant concern regarding the security of this data as well as the privacy of the participants. Privacy regulations, such as the EU GDPR, the EU Code of Conduct on agricultural data sharing by contractual agreement, and the proposed EU AI law, have been created to address the issue of data privacy and provide specific guidelines on when and how data can be shared between organizations. To make confidential agricultural data widely available for Big Data analysis without violating the privacy of the data subjects, we consider privacy-preserving methods of data sharing in agriculture. Deep learning-based synthetic data generation has been proposed for privacy-preserving data sharing. However, there is a lack of compliance with documented data privacy policies in such privacy-preserving efforts. In this study, we propose a novel framework for enforcing privacy policy rules in privacy-preserving data generation algorithms. We explore several available agricultural codes of conduct, extract knowledge related to the privacy constraints in data, and use the extracted knowledge to define privacy bounds in a privacy-preserving generative model. We use our framework to generate synthetic agricultural data and present experimental results that demonstrate the utility of the synthetic dataset in downstream tasks. We also show that our framework can evade potential threats and secure data based on applicable regulatory policy rules.

{{</citation>}}


## q-bio.BM (1)



### (135/147) InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery (He Cao et al., 2023)

{{<citation>}}

He Cao, Zijing Liu, Xingyu Lu, Yuan Yao, Yu Li. (2023)  
**InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery**  

---
Primary Category: q-bio.BM  
Categories: cs-AI, cs-LG, q-bio-BM, q-bio.BM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16208v1)  

---


**ABSTRACT**  
The rapid evolution of artificial intelligence in drug discovery encounters challenges with generalization and extensive training, yet Large Language Models (LLMs) offer promise in reshaping interactions with complex molecular data. Our novel contribution, InstructMol, a multi-modal LLM, effectively aligns molecular structures with natural language via an instruction-tuning approach, utilizing a two-stage training strategy that adeptly combines limited domain-specific data with molecular and textual information. InstructMol showcases substantial performance improvements in drug discovery-related molecular tasks, surpassing leading LLMs and significantly reducing the gap with specialized models, thereby establishing a robust foundation for a versatile and dependable drug discovery assistant.

{{</citation>}}


## q-bio.QM (1)



### (136/147) The Graph Convolutional Network with Multi-representation Alignment for Drug Synergy Prediction (Xinxing Yang et al., 2023)

{{<citation>}}

Xinxing Yang, Genke Yang, Jian Chu. (2023)  
**The Graph Convolutional Network with Multi-representation Alignment for Drug Synergy Prediction**  

---
Primary Category: q-bio.QM  
Categories: cs-IR, cs-LG, q-bio-QM, q-bio.QM  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2311.16207v1)  

---


**ABSTRACT**  
Drug combination refers to the use of two or more drugs to treat a specific disease at the same time. It is currently the mainstream way to treat complex diseases. Compared with single drugs, drug combinations have better efficacy and can better inhibit toxicity and drug resistance. The computational model based on deep learning concatenates the representation of multiple drugs and the corresponding cell line feature as input, and the output is whether the drug combination can have an inhibitory effect on the cell line. However, this strategy of concatenating multiple representations has the following defects: the alignment of drug representation and cell line representation is ignored, resulting in the synergistic relationship not being reflected positionally in the embedding space. Moreover, the alignment measurement function in deep learning cannot be suitable for drug synergy prediction tasks due to differences in input types. Therefore, in this work, we propose a graph convolutional network with multi-representation alignment (GCNMRA) for predicting drug synergy. In the GCNMRA model, we designed a multi-representation alignment function suitable for the drug synergy prediction task so that the positional relationship between drug representations and cell line representation is reflected in the embedding space. In addition, the vector modulus of drug representations and cell line representation is considered to improve the accuracy of calculation results and accelerate model convergence. Finally, many relevant experiments were run on multiple drug synergy datasets to verify the effectiveness of the above innovative elements and the excellence of the GCNMRA model.

{{</citation>}}


## cs.AI (2)



### (137/147) A Fully Data-Driven Approach for Realistic Traffic Signal Control Using Offline Reinforcement Learning (Jianxiong Li et al., 2023)

{{<citation>}}

Jianxiong Li, Shichao Lin, Tianyu Shi, Chujie Tian, Yu Mei, Jian Song, Xianyuan Zhan, Ruimin Li. (2023)  
**A Fully Data-Driven Approach for Realistic Traffic Signal Control Using Offline Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15920v1)  

---


**ABSTRACT**  
The optimization of traffic signal control (TSC) is critical for an efficient transportation system. In recent years, reinforcement learning (RL) techniques have emerged as a popular approach for TSC and show promising results for highly adaptive control. However, existing RL-based methods suffer from notably poor real-world applicability and hardly have any successful deployments. The reasons for such failures are mostly due to the reliance on over-idealized traffic simulators for policy optimization, as well as using unrealistic fine-grained state observations and reward signals that are not directly obtainable from real-world sensors. In this paper, we propose a fully Data-Driven and simulator-free framework for realistic Traffic Signal Control (D2TSC). Specifically, we combine well-established traffic flow theory with machine learning to construct a reward inference model to infer the reward signals from coarse-grained traffic data. With the inferred rewards, we further propose a sample-efficient offline RL method to enable direct signal control policy learning from historical offline datasets of real-world intersections. To evaluate our approach, we collect historical traffic data from a real-world intersection, and develop a highly customized simulation environment that strictly follows real data characteristics. We demonstrate through extensive experiments that our approach achieves superior performance over conventional and offline RL baselines, and also enjoys much better real-world applicability.

{{</citation>}}


### (138/147) Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs (Simone Conia et al., 2023)

{{<citation>}}

Simone Conia, Min Li, Daniel Lee, Umar Farooq Minhas, Ihab Ilyas, Yunyao Li. (2023)  
**Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Computer Vision, Knowledge Graph, Language Model, Machine Translation, Multilingual, Natural Language Processing, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.15781v1)  

---


**ABSTRACT**  
Recent work in Natural Language Processing and Computer Vision has been using textual information -- e.g., entity names and descriptions -- available in knowledge graphs to ground neural models to high-quality structured data. However, when it comes to non-English languages, the quantity and quality of textual information are comparatively scarce. To address this issue, we introduce the novel task of automatic Knowledge Graph Enhancement (KGE) and perform a thorough investigation on bridging the gap in both the quantity and quality of textual information between English and non-English languages. More specifically, we: i) bring to light the problem of increasing multilingual coverage and precision of entity names and descriptions in Wikidata; ii) demonstrate that state-of-the-art methods, namely, Machine Translation (MT), Web Search (WS), and Large Language Models (LLMs), struggle with this task; iii) present M-NTA, a novel unsupervised approach that combines MT, WS, and LLMs to generate high-quality textual information; and, iv) study the impact of increasing multilingual coverage and precision of non-English textual information in Entity Linking, Knowledge Graph Completion, and Question Answering. As part of our effort towards better multilingual knowledge graphs, we also introduce WikiKGE-10, the first human-curated benchmark to evaluate KGE approaches in 10 languages across 7 language families.

{{</citation>}}


## cs.NI (2)



### (139/147) Distributed Attacks over Federated Reinforcement Learning-enabled Cell Sleep Control (Han Zhang et al., 2023)

{{<citation>}}

Han Zhang, Hao Zhou, Medhat Elsayed, Majid Bavand, Raimundas Gaigalas, Yigit Ozcan, Melike Erol-Kantarci. (2023)  
**Distributed Attacks over Federated Reinforcement Learning-enabled Cell Sleep Control**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15894v1)  

---


**ABSTRACT**  
Federated learning (FL) is particularly useful in wireless networks due to its distributed implementation and privacy-preserving features. However, as a distributed learning system, FL can be vulnerable to malicious attacks from both internal and external sources. Our work aims to investigate the attack models in a FL-enabled wireless networks. Specifically, we consider a cell sleep control scenario, and apply federated reinforcement learning to improve energy-efficiency. We design three attacks, namely free rider attacks, Byzantine data poisoning attacks and backdoor attacks. The simulation results show that the designed attacks can degrade the network performance and lead to lower energy-efficiency. Moreover, we also explore possible ways to mitigate the above attacks. We design a defense model called refined-Krum to defend against attacks by enabling a secure aggregation on the global server. The proposed refined- Krum scheme outperforms the existing Krum scheme and can effectively prevent wireless networks from malicious attacks, improving the system energy-efficiency performance.

{{</citation>}}


### (140/147) Multi-Agent Reinforcement Learning for Power Control in Wireless Networks via Adaptive Graphs (Lorenzo Mario Amorosa et al., 2023)

{{<citation>}}

Lorenzo Mario Amorosa, Marco Skocaj, Roberto Verdone, Deniz Gündüz. (2023)  
**Multi-Agent Reinforcement Learning for Power Control in Wireless Networks via Adaptive Graphs**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-MA, cs-NI, cs.NI  
Keywords: GNN, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15858v1)  

---


**ABSTRACT**  
The ever-increasing demand for high-quality and heterogeneous wireless communication services has driven extensive research on dynamic optimization strategies in wireless networks. Among several possible approaches, multi-agent deep reinforcement learning (MADRL) has emerged as a promising method to address a wide range of complex optimization problems like power control. However, the seamless application of MADRL to a variety of network optimization problems faces several challenges related to convergence. In this paper, we present the use of graphs as communication-inducing structures among distributed agents as an effective means to mitigate these challenges. Specifically, we harness graph neural networks (GNNs) as neural architectures for policy parameterization to introduce a relational inductive bias in the collective decision-making process. Most importantly, we focus on modeling the dynamic interactions among sets of neighboring agents through the introduction of innovative methods for defining a graph-induced framework for integrated communication and learning. Finally, the superior generalization capabilities of the proposed methodology to larger networks and to networks with different user categories is verified through simulations.

{{</citation>}}


## cs.GT (1)



### (141/147) Characterising and Verifying the Core in Concurrent Multi-Player Mean-Payoff Games (Full Version) (Julian Gutierrez et al., 2023)

{{<citation>}}

Julian Gutierrez, Anthony W. Lin, Muhammad Najib, Thomas Steeples, Michael Wooldridge. (2023)  
**Characterising and Verifying the Core in Concurrent Multi-Player Mean-Payoff Games (Full Version)**  

---
Primary Category: cs.GT  
Categories: cs-FL, cs-GT, cs-LO, cs-MA, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15883v1)  

---


**ABSTRACT**  
Concurrent multi-player mean-payoff games are important models for systems of agents with individual, non-dichotomous preferences. Whilst these games have been extensively studied in terms of their equilibria in non-cooperative settings, this paper explores an alternative solution concept: the core from cooperative game theory. This concept is particularly relevant for cooperative AI systems, as it enables the modelling of cooperation among agents, even when their goals are not fully aligned. Our contribution is twofold. First, we provide a characterisation of the core using discrete geometry techniques and establish a necessary and sufficient condition for its non-emptiness. We then use the characterisation to prove the existence of polynomial witnesses in the core. Second, we use the existence of such witnesses to solve key decision problems in rational verification and provide tight complexity bounds for the problem of checking whether some/every equilibrium in a game satisfies a given LTL or GR(1) specification. Our approach is general and can be adapted to handle other specifications expressed in various fragments of LTL without incurring additional computational costs.

{{</citation>}}


## cs.SD (1)



### (142/147) A-JEPA: Joint-Embedding Predictive Architecture Can Listen (Zhengcong Fei et al., 2023)

{{<citation>}}

Zhengcong Fei, Mingyuan Fan, Junshi Huang. (2023)  
**A-JEPA: Joint-Embedding Predictive Architecture Can Listen**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-SD, cs.SD, eess-AS  
Keywords: Embedding, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.15830v2)  

---


**ABSTRACT**  
This paper presents that the masked-modeling principle driving the success of large foundational vision models can be effectively applied to audio by making predictions in a latent space. We introduce Audio-based Joint-Embedding Predictive Architecture (A-JEPA), a simple extension method for self-supervised learning from the audio spectrum. Following the design of I-JEPA, our A-JEPA encodes visible audio spectrogram patches with a curriculum masking strategy via context encoder, and predicts the representations of regions sampled at well-designed locations. The target representations of those regions are extracted by the exponential moving average of context encoder, \emph{i.e.}, target encoder, on the whole spectrogram. We find it beneficial to transfer random block masking into time-frequency aware masking in a curriculum manner, considering the complexity of highly correlated in local time and frequency in audio spectrograms. To enhance contextual semantic understanding and robustness, we fine-tune the encoder with a regularized masking on target datasets, instead of input dropping or zero. Empirically, when built with Vision Transformers structure, we find A-JEPA to be highly scalable and sets new state-of-the-art performance on multiple audio and speech classification tasks, outperforming other recent models that use externally supervised pre-training.

{{</citation>}}


## eess.AS (1)



### (143/147) Voice Anonymization for All -- Bias Evaluation of the Voice Privacy Challenge Baseline System (Anna Leschanowsky et al., 2023)

{{<citation>}}

Anna Leschanowsky, Ünal Ege Gaznepoglu, Nils Peters. (2023)  
**Voice Anonymization for All -- Bias Evaluation of the Voice Privacy Challenge Baseline System**  

---
Primary Category: eess.AS  
Categories: cs-CY, eess-AS, eess.AS  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.15804v1)  

---


**ABSTRACT**  
In an age of voice-enabled technology, voice anonymization offers a solution to protect people's privacy, provided these systems work equally well across subgroups. This study investigates bias in voice anonymization systems within the context of the Voice Privacy Challenge. We curate a novel benchmark dataset to assess performance disparities among speaker subgroups based on sex and dialect. We analyze the impact of three anonymization systems and attack models on speaker subgroup bias and reveal significant performance variations. Notably, subgroup bias intensifies with advanced attacker capabilities, emphasizing the challenge of achieving equal performance across all subgroups. Our study highlights the need for inclusive benchmark datasets and comprehensive evaluation strategies that address subgroup bias in voice anonymization.

{{</citation>}}


## cs.DL (1)



### (144/147) A Knowledge Graph Approach for Exploratory Search in Research Institutions (Tim Schopf et al., 2023)

{{<citation>}}

Tim Schopf, Nektrios Machner, Florian Matthes. (2023)  
**A Knowledge Graph Approach for Exploratory Search in Research Institutions**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.15688v1)  

---


**ABSTRACT**  
Over the past decades, research institutions have grown increasingly and consequently also their research output. This poses a significant challenge for researchers seeking to understand the research landscape of an institution. The process of exploring the research landscape of institutions has a vague information need, no precise goal, and is open-ended. Current applications are not designed to fulfill the requirements for exploratory search in research institutions. In this paper, we analyze exploratory search in research institutions and propose a knowledge graph-based approach to enhance this process.

{{</citation>}}


## stat.ML (1)



### (145/147) Universal Event Detection in Time Series (Menouar Azib et al., 2023)

{{<citation>}}

Menouar Azib, Benjamin Renard, Philippe Garnier, Vincent Génot, Nicolas André. (2023)  
**Universal Event Detection in Time Series**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Event Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2311.15654v1)  

---


**ABSTRACT**  
In our previously published work, we introduced a supervised deep learning method for event detection in multivariate time series data, employing regression instead of binary classification. This simplification avoids the need for point-wise labels throughout the entire dataset, relying solely on ground truth events defined as time points or intervals. In this paper, we establish mathematically that our method is universal, and capable of detecting any type of event with arbitrary precision under mild continuity assumptions on the time series. These events may encompass change points, frauds, anomalies, physical occurrences, and more. We substantiate our theoretical results using the universal approximation theorem for feed-forward neural networks (FFN). Additionally, we provide empirical validations that confirm our claims, demonstrating that our method, with a limited number of parameters, outperforms other deep learning approaches, particularly for rare events and imbalanced datasets from different domains.

{{</citation>}}


## cs.SI (1)



### (146/147) InfoPattern: Unveiling Information Propagation Patterns in Social Media (Chi Han et al., 2023)

{{<citation>}}

Chi Han, Jialiang Xu, Manling Li, Hanning Zhang, Tarek Abdelzaher, Heng Ji. (2023)  
**InfoPattern: Unveiling Information Propagation Patterns in Social Media**  

---
Primary Category: cs.SI  
Categories: cs-CL, cs-SI, cs.SI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2311.15642v1)  

---


**ABSTRACT**  
Social media play a significant role in shaping public opinion and influencing ideological communities through information propagation. Our demo InfoPattern centers on the interplay between language and human ideology. The demo (Code: https://github.com/blender-nlp/InfoPattern ) is capable of: (1) red teaming to simulate adversary responses from opposite ideology communities; (2) stance detection to identify the underlying political sentiments in each message; (3) information propagation graph discovery to reveal the evolution of claims across various communities over time. (Live Demo: https://incas.csl.illinois.edu/blender/About )

{{</citation>}}


## cs.DC (1)



### (147/147) SpotServe: Serving Generative Large Language Models on Preemptible Instances (Xupeng Miao et al., 2023)

{{<citation>}}

Xupeng Miao, Chunan Shi, Jiangfei Duan, Xiaoli Xi, Dahua Lin, Bin Cui, Zhihao Jia. (2023)  
**SpotServe: Serving Generative Large Language Models on Preemptible Instances**  

---
Primary Category: cs.DC  
Categories: cs-CL, cs-DC, cs-LG, cs.DC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.15566v1)  

---


**ABSTRACT**  
The high computational and memory requirements of generative large language models (LLMs) make it challenging to serve them cheaply. This paper aims to reduce the monetary cost for serving LLMs by leveraging preemptible GPU instances on modern clouds, which offer accesses to spare GPUs at a much cheaper price than regular instances but may be preempted by the cloud at any time. Serving LLMs on preemptible instances requires addressing challenges induced by frequent instance preemptions and the necessity of migrating instances to handle these preemptions.   This paper presents SpotServe, the first distributed LLM serving system on preemptible instances. Several key techniques in SpotServe realize fast and reliable serving of generative LLMs on cheap preemptible instances. First, SpotServe dynamically adapts the LLM parallelization configuration for dynamic instance availability and fluctuating workload, while balancing the trade-off among the overall throughput, inference latency and monetary costs. Second, to minimize the cost of migrating instances for dynamic reparallelization, the task of migrating instances is formulated as a bipartite graph matching problem, which uses the Kuhn-Munkres algorithm to identify an optimal migration plan that minimizes communications. Finally, to take advantage of the grace period offered by modern clouds, we introduce stateful inference recovery, a new inference mechanism that commits inference progress at a much finer granularity and allows SpotServe to cheaply resume inference upon preemption. We evaluate on real spot instance preemption traces and various popular LLMs and show that SpotServe can reduce the P99 tail latency by 2.4 - 9.1x compared with the best existing LLM serving systems. We also show that SpotServe can leverage the price advantage of preemptive instances, saving 54% monetary cost compared with only using on-demand instances.

{{</citation>}}
