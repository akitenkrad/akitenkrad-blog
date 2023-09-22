---
draft: false
title: "arXiv @ 2023.09.21"
date: 2023-09-21
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.21"
    identifier: arxiv_20230921
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (37)](#cscl-37)
- [math.OC (1)](#mathoc-1)
- [cs.HC (9)](#cshc-9)
- [cs.CV (23)](#cscv-23)
- [cs.RO (4)](#csro-4)
- [eess.AS (4)](#eessas-4)
- [cs.LG (13)](#cslg-13)
- [cs.MA (1)](#csma-1)
- [cs.AI (15)](#csai-15)
- [eess.IV (4)](#eessiv-4)
- [eess.SY (1)](#eesssy-1)
- [cs.CY (2)](#cscy-2)
- [cs.SD (3)](#cssd-3)
- [cs.ET (1)](#cset-1)
- [cs.IT (1)](#csit-1)
- [cs.DC (2)](#csdc-2)
- [cs.IR (2)](#csir-2)
- [stat.ML (1)](#statml-1)
- [q-fin.CP (1)](#q-fincp-1)
- [cs.CR (2)](#cscr-2)
- [cs.SI (1)](#cssi-1)
- [quant-ph (1)](#quant-ph-1)
- [physics.flu-dyn (1)](#physicsflu-dyn-1)
- [cs.SE (1)](#csse-1)

## cs.CL (37)



### (1/131) MBR and QE Finetuning: Training-time Distillation of the Best and Most Expensive Decoding Methods (Mara Finkelstein et al., 2023)

{{<citation>}}

Mara Finkelstein, Markus Freitag. (2023)  
**MBR and QE Finetuning: Training-time Distillation of the Best and Most Expensive Decoding Methods**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2309.10966v1)  

---


**ABSTRACT**  
Recent research in decoding methods for Natural Language Generation (NLG) tasks has shown that the traditional beam search and greedy decoding algorithms are not optimal, because model probabilities do not always align with human preferences. Stronger decoding methods, including Quality Estimation (QE) reranking and Minimum Bayes' Risk (MBR) decoding, have since been proposed to mitigate the model-perplexity-vs-quality mismatch. While these decoding methods achieve state-of-the-art performance, they are prohibitively expensive to compute. In this work, we propose MBR finetuning and QE finetuning which distill the quality gains from these decoding methods at training time, while using an efficient decoding algorithm at inference time. Using the canonical NLG task of Neural Machine Translation (NMT), we show that even with self-training, these finetuning methods significantly outperform the base model. Moreover, when using an external LLM as a teacher model, these finetuning methods outperform finetuning on human-generated references. These findings suggest new ways to leverage monolingual data to achieve improvements in model quality that are on par with, or even exceed, improvements from human-curated data, while maintaining maximum efficiency during decoding.

{{</citation>}}


### (2/131) In-Context Learning for Text Classification with Many Labels (Aristides Milios et al., 2023)

{{<citation>}}

Aristides Milios, Siva Reddy, Dzmitry Bahdanau. (2023)  
**In-Context Learning for Text Classification with Many Labels**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: LLaMA, Text Classification  
[Paper Link](http://arxiv.org/abs/2309.10954v1)  

---


**ABSTRACT**  
In-context learning (ICL) using large language models for tasks with many labels is challenging due to the limited context window, which makes it difficult to fit a sufficient number of examples in the prompt. In this paper, we use a pre-trained dense retrieval model to bypass this limitation, giving the model only a partial view of the full label space for each inference call. Testing with recent open-source LLMs (OPT, LLaMA), we set new state of the art performance in few-shot settings for three common intent classification datasets, with no finetuning. We also surpass fine-tuned performance on fine-grained sentiment classification in certain cases. We analyze the performance across number of in-context examples and different model scales, showing that larger models are necessary to effectively and consistently make use of larger context lengths for ICL. By running several ablations, we analyze the model's use of: a) the similarity of the in-context examples to the current input, b) the semantic content of the class names, and c) the correct correspondence between examples and labels. We demonstrate that all three are needed to varying degrees depending on the domain, contrary to certain recent works.

{{</citation>}}


### (3/131) LMDX: Language Model-based Document Information Extraction and Localization (Vincent Perot et al., 2023)

{{<citation>}}

Vincent Perot, Kai Kang, Florian Luisier, Guolong Su, Xiaoyu Sun, Ramya Sree Boppana, Zilong Wang, Jiaqi Mu, Hao Zhang, Nan Hua. (2023)  
**LMDX: Language Model-based Document Information Extraction and Localization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Information Extraction, Language Model, NLP, Natural Language Processing, PaLM  
[Paper Link](http://arxiv.org/abs/2309.10952v1)  

---


**ABSTRACT**  
Large Language Models (LLM) have revolutionized Natural Language Processing (NLP), improving state-of-the-art on many existing tasks and exhibiting emergent capabilities. However, LLMs have not yet been successfully applied on semi-structured document information extraction, which is at the core of many document processing workflows and consists of extracting key entities from a visually rich document (VRD) given a predefined target schema. The main obstacles to LLM adoption in that task have been the absence of layout encoding within LLMs, critical for a high quality extraction, and the lack of a grounding mechanism ensuring the answer is not hallucinated. In this paper, we introduce Language Model-based Document Information Extraction and Localization (LMDX), a methodology to adapt arbitrary LLMs for document information extraction. LMDX can do extraction of singular, repeated, and hierarchical entities, both with and without training data, while providing grounding guarantees and localizing the entities within the document. In particular, we apply LMDX to the PaLM 2-S LLM and evaluate it on VRDU and CORD benchmarks, setting a new state-of-the-art and showing how LMDX enables the creation of high quality, data-efficient parsers.

{{</citation>}}


### (4/131) A Family of Pretrained Transformer Language Models for Russian (Dmitry Zmitrovich et al., 2023)

{{<citation>}}

Dmitry Zmitrovich, Alexander Abramov, Andrey Kalmykov, Maria Tikhonova, Ekaterina Taktasheva, Danil Astafurov, Mark Baushenko, Artem Snegirev, Tatiana Shavrina, Sergey Markov, Vladislav Mikhailov, Alena Fenogenova. (2023)  
**A Family of Pretrained Transformer Language Models for Russian**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, Language Model, NLP, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2309.10931v1)  

---


**ABSTRACT**  
Nowadays, Transformer language models (LMs) represent a fundamental component of the NLP research methodologies and applications. However, the development of such models specifically for the Russian language has received little attention. This paper presents a collection of 13 Russian Transformer LMs based on the encoder (ruBERT, ruRoBERTa, ruELECTRA), decoder (ruGPT-3), and encoder-decoder (ruT5, FRED-T5) models in multiple sizes. Access to these models is readily available via the HuggingFace platform. We provide a report of the model architecture design and pretraining, and the results of evaluating their generalization abilities on Russian natural language understanding and generation datasets and benchmarks. By pretraining and releasing these specialized Transformer LMs, we hope to broaden the scope of the NLP research directions and enable the development of industrial solutions for the Russian language.

{{</citation>}}


### (5/131) Specializing Small Language Models towards Complex Style Transfer via Latent Attribute Pre-Training (Ruiqi Xu et al., 2023)

{{<citation>}}

Ruiqi Xu, Yongfeng Huang, Xin Chen, Lin Zhang. (2023)  
**Specializing Small Language Models towards Complex Style Transfer via Latent Attribute Pre-Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Style Transfer, T5  
[Paper Link](http://arxiv.org/abs/2309.10929v1)  

---


**ABSTRACT**  
In this work, we introduce the concept of complex text style transfer tasks, and constructed complex text datasets based on two widely applicable scenarios. Our dataset is the first large-scale data set of its kind, with 700 rephrased sentences and 1,000 sentences from the game Genshin Impact. While large language models (LLM) have shown promise in complex text style transfer, they have drawbacks such as data privacy concerns, network instability, and high deployment costs. To address these issues, we explore the effectiveness of small models (less than T5-3B) with implicit style pre-training through contrastive learning. We also propose a method for automated evaluation of text generation quality based on alignment with human evaluations using ChatGPT. Finally, we compare our approach with existing methods and show that our model achieves state-of-art performances of few-shot text style transfer models.

{{</citation>}}


### (6/131) Semi-Autoregressive Streaming ASR With Label Context (Siddhant Arora et al., 2023)

{{<citation>}}

Siddhant Arora, George Saon, Shinji Watanabe, Brian Kingsbury. (2023)  
**Semi-Autoregressive Streaming ASR With Label Context**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.10926v1)  

---


**ABSTRACT**  
Non-autoregressive (NAR) modeling has gained significant interest in speech processing since these models achieve dramatically lower inference time than autoregressive (AR) models while also achieving good transcription accuracy. Since NAR automatic speech recognition (ASR) models must wait for the completion of the entire utterance before processing, some works explore streaming NAR models based on blockwise attention for low-latency applications. However, streaming NAR models significantly lag in accuracy compared to streaming AR and non-streaming NAR models. To address this, we propose a streaming "semi-autoregressive" ASR model that incorporates the labels emitted in previous blocks as additional context using a Language Model (LM) subnetwork. We also introduce a novel greedy decoding algorithm that addresses insertion and deletion errors near block boundaries while not significantly increasing the inference time. Experiments show that our method outperforms the existing streaming NAR model by 19% relative on Tedlium2, 16%/8% on Librispeech-100 clean/other test sets, and 19%/8% on the Switchboard(SWB) / Callhome(CH) test sets. It also reduced the accuracy gap with streaming AR and non-streaming NAR models while achieving 2.5x lower latency. We also demonstrate that our approach can effectively utilize external text data to pre-train the LM subnetwork to further improve streaming ASR accuracy.

{{</citation>}}


### (7/131) RedPenNet for Grammatical Error Correction: Outputs to Tokens, Attentions to Spans (Bohdan Didenko et al., 2023)

{{<citation>}}

Bohdan Didenko, Andrii Sameliuk. (2023)  
**RedPenNet for Grammatical Error Correction: Outputs to Tokens, Attentions to Spans**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Machine Translation, NER, NLP  
[Paper Link](http://arxiv.org/abs/2309.10898v1)  

---


**ABSTRACT**  
The text editing tasks, including sentence fusion, sentence splitting and rephrasing, text simplification, and Grammatical Error Correction (GEC), share a common trait of dealing with highly similar input and output sequences. This area of research lies at the intersection of two well-established fields: (i) fully autoregressive sequence-to-sequence approaches commonly used in tasks like Neural Machine Translation (NMT) and (ii) sequence tagging techniques commonly used to address tasks such as Part-of-speech tagging, Named-entity recognition (NER), and similar. In the pursuit of a balanced architecture, researchers have come up with numerous imaginative and unconventional solutions, which we're discussing in the Related Works section. Our approach to addressing text editing tasks is called RedPenNet and is aimed at reducing architectural and parametric redundancies presented in specific Sequence-To-Edits models, preserving their semi-autoregressive advantages. Our models achieve $F_{0.5}$ scores of 77.60 on the BEA-2019 (test), which can be considered as state-of-the-art the only exception for system combination and 67.71 on the UAGEC+Fluency (test) benchmarks.   This research is being conducted in the context of the UNLP 2023 workshop, where it was presented as a paper as a paper for the Shared Task in Grammatical Error Correction (GEC) for Ukrainian. This study aims to apply the RedPenNet approach to address the GEC problem in the Ukrainian language.

{{</citation>}}


### (8/131) Self-Augmentation Improves Zero-Shot Cross-Lingual Transfer (Fei Wang et al., 2023)

{{<citation>}}

Fei Wang, Kuan-Hao Huang, Kai-Wei Chang, Muhao Chen. (2023)  
**Self-Augmentation Improves Zero-Shot Cross-Lingual Transfer**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AWS, Augmentation, NLI, NLP, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.10891v1)  

---


**ABSTRACT**  
Zero-shot cross-lingual transfer is a central task in multilingual NLP, allowing models trained in languages with more sufficient training resources to generalize to other low-resource languages. Earlier efforts on this task use parallel corpora, bilingual dictionaries, or other annotated alignment data to improve cross-lingual transferability, which are typically expensive to obtain. In this paper, we propose a simple yet effective method, SALT, to improve the zero-shot cross-lingual transfer of the multilingual pretrained language models without the help of such external data. By incorporating code-switching and embedding mixup with self-augmentation, SALT effectively distills cross-lingual knowledge from the multilingual PLM and enhances its transferability on downstream tasks. Experimental results on XNLI and PAWS-X show that our method is able to improve zero-shot cross-lingual transferability without external data. Our code is available at https://github.com/luka-group/SALT.

{{</citation>}}


### (9/131) Classifying Organizations for Food System Ontologies using Natural Language Processing (Tianyu Jiang et al., 2023)

{{<citation>}}

Tianyu Jiang, Sonia Vinogradova, Nathan Stringham, E. Louise Earl, Allan D. Hollander, Patrick R. Huber, Ellen Riloff, R. Sandra Schillo, Giorgio A. Ubbiali, Matthew Lange. (2023)  
**Classifying Organizations for Food System Ontologies using Natural Language Processing**  

---
Primary Category: cs.CL  
Categories: H-3-1; I-2-7; J-3; J-4; K-4-3, cs-AI, cs-CL, cs-CY, cs-IR, cs.CL  
Keywords: Google, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.10880v1)  

---


**ABSTRACT**  
Our research explores the use of natural language processing (NLP) methods to automatically classify entities for the purpose of knowledge graph population and integration with food system ontologies. We have created NLP models that can automatically classify organizations with respect to categories associated with environmental issues as well as Standard Industrial Classification (SIC) codes, which are used by the U.S. government to characterize business activities. As input, the NLP models are provided with text snippets retrieved by the Google search engine for each organization, which serves as a textual description of the organization that is used for learning. Our experimental results show that NLP models can achieve reasonably good performance for these two classification tasks, and they rely on a general framework that could be applied to many other classification problems as well. We believe that NLP models represent a promising approach for automatically harvesting information to populate knowledge graphs and aligning the information with existing ontologies through shared categories and concepts.

{{</citation>}}


### (10/131) SlimPajama-DC: Understanding Data Combinations for LLM Training (Zhiqiang Shen et al., 2023)

{{<citation>}}

Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Joel Hestness, Natalia Vassilieva, Daria Soboleva, Eric Xing. (2023)  
**SlimPajama-DC: Understanding Data Combinations for LLM Training**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2309.10818v1)  

---


**ABSTRACT**  
This paper aims to understand the impacts of various data combinations (e.g., web text, wikipedia, github, books) on the training of large language models using SlimPajama. SlimPajama is a rigorously deduplicated, multi-source dataset, which has been refined and further deduplicated to 627B tokens from the extensive 1.2T tokens RedPajama dataset contributed by Together. We've termed our research as SlimPajama-DC, an empirical analysis designed to uncover fundamental characteristics and best practices associated with employing SlimPajama in the training of large language models. During our research with SlimPajama, two pivotal observations emerged: (1) Global deduplication vs. local deduplication. We analyze and discuss how global (across different sources of datasets) and local (within the single source of dataset) deduplications affect the performance of trained models. (2) Proportions of high-quality/highly-deduplicated multi-source datasets in the combination. To study this, we construct six configurations of SlimPajama dataset and train individual ones using 1.3B Cerebras-GPT model with Alibi and SwiGLU. Our best configuration outperforms the 1.3B model trained on RedPajama using the same number of training tokens by a significant margin. All our 1.3B models are trained on Cerebras 16$\times$ CS-2 cluster with a total of 80 PFLOP/s in bf16 mixed precision. We further extend our discoveries (such as increasing data diversity is crucial after global deduplication) on a 7B model with large batch-size training. Our models and the separate SlimPajama-DC datasets are available at: https://huggingface.co/MBZUAI-LLM and https://huggingface.co/datasets/cerebras/SlimPajama-627B.

{{</citation>}}


### (11/131) Natural Language Embedded Programs for Hybrid Language Symbolic Reasoning (Tianhua Zhang et al., 2023)

{{<citation>}}

Tianhua Zhang, Jiaxin Ge, Hongyin Luo, Yung-Sung Chuang, Mingye Gao, Yuan Gong, Xixin Wu, Yoon Kim, Helen Meng, James Glass. (2023)  
**Natural Language Embedded Programs for Hybrid Language Symbolic Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.10814v1)  

---


**ABSTRACT**  
How can we perform computations over natural language representations to solve tasks that require symbolic and numeric reasoning? We propose natural language embedded programs (NLEP) as a unifying framework for addressing math/symbolic reasoning, natural language understanding, and instruction following tasks. Our approach prompts a language model to generate full Python programs that define functions over data structures which contain natural language representations of structured knowledge. A Python interpreter then executes the generated code and prints the output. Despite using a task-general prompt, we find that this approach can improve upon strong baselines across a range of different tasks including math and symbolic reasoning, text classification, question answering, and instruction following. We further find the generated programs are often interpretable and enable post-hoc verification of the intermediate reasoning steps.

{{</citation>}}


### (12/131) FRASIMED: a Clinical French Annotated Resource Produced through Crosslingual BERT-Based Annotation Projection (Jamil Zaghir et al., 2023)

{{<citation>}}

Jamil Zaghir, Mina Bjelogrlic, Jean-Philippe Goldman, Soukaïna Aananou, Christophe Gaudet-Blavignac, Christian Lovis. (2023)  
**FRASIMED: a Clinical French Annotated Resource Produced through Crosslingual BERT-Based Annotation Projection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Clinical, NER, NLP  
[Paper Link](http://arxiv.org/abs/2309.10770v1)  

---


**ABSTRACT**  
Natural language processing (NLP) applications such as named entity recognition (NER) for low-resource corpora do not benefit from recent advances in the development of large language models (LLMs) where there is still a need for larger annotated datasets. This research article introduces a methodology for generating translated versions of annotated datasets through crosslingual annotation projection. Leveraging a language agnostic BERT-based approach, it is an efficient solution to increase low-resource corpora with few human efforts and by only using already available open data resources. Quantitative and qualitative evaluations are often lacking when it comes to evaluating the quality and effectiveness of semi-automatic data generation strategies. The evaluation of our crosslingual annotation projection approach showed both effectiveness and high accuracy in the resulting dataset. As a practical application of this methodology, we present the creation of French Annotated Resource with Semantic Information for Medical Entities Detection (FRASIMED), an annotated corpus comprising 2'051 synthetic clinical cases in French. The corpus is now available for researchers and practitioners to develop and refine French natural language processing (NLP) applications in the clinical field (https://zenodo.org/record/8355629), making it the largest open annotated corpus with linked medical concepts in French.

{{</citation>}}


### (13/131) OpenBA: An Open-sourced 15B Bilingual Asymmetric seq2seq Model Pre-trained from Scratch (Juntao Li et al., 2023)

{{<citation>}}

Juntao Li, Zecheng Tang, Yuyang Ding, Pinzheng Wang, Pei Guo, Wangjie You, Dan Qiao, Wenliang Chen, Guohong Fu, Qiaoming Zhu, Guodong Zhou, Min Zhang. (2023)  
**OpenBA: An Open-sourced 15B Bilingual Asymmetric seq2seq Model Pre-trained from Scratch**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM, GLM, LLaMA, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.10706v1)  

---


**ABSTRACT**  
Large language models (LLMs) with billions of parameters have demonstrated outstanding performance on various natural language processing tasks. This report presents OpenBA, an open-sourced 15B bilingual asymmetric seq2seq model, to contribute an LLM variant to the Chinese-oriented open-source model community. We enhance OpenBA with effective and efficient techniques as well as adopt a three-stage training strategy to train the model from scratch. Our solution can also achieve very competitive performance with only 380B tokens, which is better than LLaMA-70B on the BELEBELE benchmark, BLOOM-176B on the MMLU benchmark, GLM-130B on the C-Eval (hard) benchmark. This report provides the main details to pre-train an analogous model, including pre-training data processing, Bilingual Flan data collection, the empirical observations that inspire our model architecture design, training objectives of different stages, and other enhancement techniques. We have refactored our code to follow the design principles of the Huggingface Transformers Library, making it more convenient for developers to use, and released checkpoints of different training stages at https://huggingface.co/openBA. More details of our project are available at https://github.com/OpenNLG/openBA.git.

{{</citation>}}


### (14/131) MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback (Xingyao Wang et al., 2023)

{{<citation>}}

Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, Heng Ji. (2023)  
**MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.10691v1)  

---


**ABSTRACT**  
To solve complex tasks, large language models (LLMs) often require multiple rounds of interactions with the user, sometimes assisted by external tools. However, current evaluation paradigms often focus solely on benchmark performance with single-turn exchanges, neglecting the intricate interactions among the user, LLMs, and external tools, creating a discrepancy between benchmark evaluation and real-world use cases. We introduce MINT benchmark to evaluate LLMs' ability to solve tasks with multi-turn interactions by (1) using tools and (2) leveraging natural language feedback. To ensure reproducibility, we provide an evaluation framework where LLMs can access tools by executing Python code and receive natural language feedback from the user simulated with GPT-4. We repurpose a diverse set of established datasets and tasks focusing on reasoning, coding, and decision-making and carefully curate them into a compact subset of instances for efficient evaluation. Our analysis of 20 open- and closed-source LLMs offers intriguing findings. (1) LLMs generally benefit from tool interactions and language feedback, with performance gains (absolute, same below) of 1--8% per additional turn with tool use and 2--17% with natural language feedback. (2) Better single-turn performance does not guarantee better multi-turn performance. (3) Surprisingly, on LLMs we evaluated, we found supervised instruction-finetuning (SIFT) and reinforcement learning from human feedback (RLHF) generally hurt multi-turn capabilities. We hope MINT can help measure progress and incentivize research in improving LLMs' capabilities in multi-turn interactions, especially for open-source communities where multi-turn human evaluation has been less accessible compared to commercial LLMs with a larger user base.

{{</citation>}}


### (15/131) Estimating Contamination via Perplexity: Quantifying Memorisation in Language Model Evaluation (Yucheng Li, 2023)

{{<citation>}}

Yucheng Li. (2023)  
**Estimating Contamination via Perplexity: Quantifying Memorisation in Language Model Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Perplexity  
[Paper Link](http://arxiv.org/abs/2309.10677v1)  

---


**ABSTRACT**  
Data contamination in model evaluation is getting increasingly prevalent as the massive training corpora of large language models often unintentionally include benchmark samples. Therefore, contamination analysis has became an inevitable part of reliable model evaluation. However, existing method of contamination analysis requires the access of the entire training data which is often confidential for recent models. This prevent the community to rigorously audit these models and conduct accurate assessment of their capability. In this paper, we propose a novel method to quantify contamination without the access of the full training set, that measure the extent of contamination with perplexity. Our analysis provides evidence of significant memorisation of recent foundation models in popular reading comprehension, summarisation benchmarks, while multiple choice appears less contaminated.

{{</citation>}}


### (16/131) NusaWrites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages (Samuel Cahyawijaya et al., 2023)

{{<citation>}}

Samuel Cahyawijaya, Holy Lovenia, Fajri Koto, Dea Adhista, Emmanuel Dave, Sarah Oktavianti, Salsabil Maulana Akbar, Jhonson Lee, Nuur Shadieq, Tjeng Wawan Cenggoro, Hanung Wahyuning Linuwih, Bryan Wilie, Galih Pradipta Muridan, Genta Indra Winata, David Moeljadi, Alham Fikri Aji, Ayu Purwarianti, Pascale Fung. (2023)  
**NusaWrites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Low-Resource, NLP  
[Paper Link](http://arxiv.org/abs/2309.10661v2)  

---


**ABSTRACT**  
Democratizing access to natural language processing (NLP) technology is crucial, especially for underrepresented and extremely low-resource languages. Previous research has focused on developing labeled and unlabeled corpora for these languages through online scraping and document translation. While these methods have proven effective and cost-efficient, we have identified limitations in the resulting corpora, including a lack of lexical diversity and cultural relevance to local communities. To address this gap, we conduct a case study on Indonesian local languages. We compare the effectiveness of online scraping, human translation, and paragraph writing by native speakers in constructing datasets. Our findings demonstrate that datasets generated through paragraph writing by native speakers exhibit superior quality in terms of lexical diversity and cultural content. In addition, we present the \datasetname{} benchmark, encompassing 12 underrepresented and extremely low-resource languages spoken by millions of individuals in Indonesia. Our empirical experiment results using existing multilingual large language models conclude the need to extend these models to more underrepresented languages. We release the NusaWrites dataset at https://github.com/IndoNLP/nusa-writes.

{{</citation>}}


### (17/131) CFGPT: Chinese Financial Assistant with Large Language Model (Jiangtong Li et al., 2023)

{{<citation>}}

Jiangtong Li, Yuxuan Bian, Guoxuan Wang, Yang Lei, Dawei Cheng, Zhijun Ding, Changjun Jiang. (2023)  
**CFGPT: Chinese Financial Assistant with Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CE, cs-CL, cs.CL  
Keywords: Financial, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.10654v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated great potential in natural language processing tasks within the financial domain. In this work, we present a Chinese Financial Generative Pre-trained Transformer framework, named CFGPT, which includes a dataset~(CFData) for pre-training and supervised fine-tuning, a financial LLM~(CFLLM) to adeptly manage financial texts, and a deployment framework~(CFAPP) designed to navigate real-world financial applications. The CFData comprising both a pre-training dataset and a supervised fine-tuning dataset, where the pre-training dataset collates Chinese financial data and analytics, alongside a smaller subset of general-purpose text with 584M documents and 141B tokens in total, and the supervised fine-tuning dataset is tailored for six distinct financial tasks, embodying various facets of financial analysis and decision-making with 1.5M instruction pairs and 1.5B tokens in total. The CFLLM, which is based on InternLM-7B to balance the model capability and size, is trained on CFData in two stage, continued pre-training and supervised fine-tuning. The CFAPP is centered on large language models (LLMs) and augmented with additional modules to ensure multifaceted functionality in real-world application. Our codes are released at https://github.com/TongjiFinLab/CFGPT.

{{</citation>}}


### (18/131) Improving Medical Dialogue Generation with Abstract Meaning Representations (Bohao Yang et al., 2023)

{{<citation>}}

Bohao Yang, Chen Tang, Chenghua Lin. (2023)  
**Improving Medical Dialogue Generation with Abstract Meaning Representations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Abstract Meaning Representation, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2309.10608v1)  

---


**ABSTRACT**  
Medical Dialogue Generation serves a critical role in telemedicine by facilitating the dissemination of medical expertise to patients. Existing studies focus on incorporating textual representations, which have limited their ability to represent the semantics of text, such as ignoring important medical entities. To enhance the model's understanding of the textual semantics and the medical knowledge including entities and relations, we introduce the use of Abstract Meaning Representations (AMR) to construct graphical representations that delineate the roles of language constituents and medical entities within the dialogues. In this paper, We propose a novel framework that models dialogues between patients and healthcare professionals using AMR graphs, where the neural networks incorporate textual and graphical knowledge with a dual attention mechanism. Experimental results show that our framework outperforms strong baseline models in medical dialogue generation, demonstrating the effectiveness of AMR graphs in enhancing the representations of medical knowledge and logical relationships. Furthermore, to support future research in this domain, we provide the corresponding source code at https://github.com/Bernard-Yang/MedDiaAMR.

{{</citation>}}


### (19/131) FRACAS: A FRench Annotated Corpus of Attribution relations in newS (Ange Richard et al., 2023)

{{<citation>}}

Ange Richard, Laura Alonzo-Canul, François Portet. (2023)  
**FRACAS: A FRench Annotated Corpus of Attribution relations in newS**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.10604v1)  

---


**ABSTRACT**  
Quotation extraction is a widely useful task both from a sociological and from a Natural Language Processing perspective. However, very little data is available to study this task in languages other than English. In this paper, we present a manually annotated corpus of 1676 newswire texts in French for quotation extraction and source attribution. We first describe the composition of our corpus and the choices that were made in selecting the data. We then detail the annotation guidelines and annotation process, as well as a few statistics about the final corpus and the obtained balance between quote types (direct, indirect and mixed, which are particularly challenging). We end by detailing our inter-annotator agreement between the 8 annotators who worked on manual labelling, which is substantially high for such a difficult linguistic phenomenon.

{{</citation>}}


### (20/131) Unsupervised Deep Cross-Language Entity Alignment (Chuanyu Jiang et al., 2023)

{{<citation>}}

Chuanyu Jiang, Yiming Qian, Lijun Chen, Yang Gu, Xia Xie. (2023)  
**Unsupervised Deep Cross-Language Entity Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Entity Alignment  
[Paper Link](http://arxiv.org/abs/2309.10598v1)  

---


**ABSTRACT**  
Cross-lingual entity alignment is the task of finding the same semantic entities from different language knowledge graphs. In this paper, we propose a simple and novel unsupervised method for cross-language entity alignment. We utilize the deep learning multi-language encoder combined with a machine translator to encode knowledge graph text, which reduces the reliance on label data. Unlike traditional methods that only emphasize global or local alignment, our method simultaneously considers both alignment strategies. We first view the alignment task as a bipartite matching problem and then adopt the re-exchanging idea to accomplish alignment. Compared with the traditional bipartite matching algorithm that only gives one optimal solution, our algorithm generates ranked matching results which enabled many potentials downstream tasks. Additionally, our method can adapt two different types of optimization (minimal and maximal) in the bipartite matching process, which provides more flexibility. Our evaluation shows, we each scored 0.966, 0.990, and 0.996 Hits@1 rates on the DBP15K dataset in Chinese, Japanese, and French to English alignment tasks. We outperformed the state-of-the-art method in unsupervised and semi-supervised categories. Compared with the state-of-the-art supervised method, our method outperforms 2.6% and 0.4% in Ja-En and Fr-En alignment tasks while marginally lower by 0.2% in the Zh-En alignment task.

{{</citation>}}


### (21/131) Multimodal Modeling For Spoken Language Identification (Shikhar Bharadwaj et al., 2023)

{{<citation>}}

Shikhar Bharadwaj, Min Ma, Shikhar Vashishth, Ankur Bapna, Sriram Ganapathy, Vera Axelrod, Siddharth Dalmia, Wei Han, Yu Zhang, Daan van Esch, Sandy Ritchie, Partha Talukdar, Jason Riesa. (2023)  
**Multimodal Modeling For Spoken Language Identification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Language Identification  
[Paper Link](http://arxiv.org/abs/2309.10567v1)  

---


**ABSTRACT**  
Spoken language identification refers to the task of automatically predicting the spoken language in a given utterance. Conventionally, it is modeled as a speech-based language identification task. Prior techniques have been constrained to a single modality; however in the case of video data there is a wealth of other metadata that may be beneficial for this task. In this work, we propose MuSeLI, a Multimodal Spoken Language Identification method, which delves into the use of various metadata sources to enhance language identification. Our study reveals that metadata such as video title, description and geographic location provide substantial information to identify the spoken language of the multimedia recording. We conduct experiments using two diverse public datasets of YouTube videos, and obtain state-of-the-art results on the language identification task. We additionally conduct an ablation study that describes the distinct contribution of each modality for language recognition.

{{</citation>}}


### (22/131) OpenMSD: Towards Multilingual Scientific Documents Similarity Measurement (Yang Gao et al., 2023)

{{<citation>}}

Yang Gao, Ji Ma, Ivan Korotkov, Keith Hall, Dana Alon, Don Metzler. (2023)  
**OpenMSD: Towards Multilingual Scientific Documents Similarity Measurement**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2309.10539v1)  

---


**ABSTRACT**  
We develop and evaluate multilingual scientific documents similarity measurement models in this work. Such models can be used to find related works in different languages, which can help multilingual researchers find and explore papers more efficiently. We propose the first multilingual scientific documents dataset, Open-access Multilingual Scientific Documents (OpenMSD), which has 74M papers in 103 languages and 778M citation pairs. With OpenMSD, we pretrain science-specialized language models, and explore different strategies to derive "related" paper pairs to fine-tune the models, including using a mixture of citation, co-citation, and bibliographic-coupling pairs. To further improve the models' performance for non-English papers, we explore the use of generative language models to enrich the non-English papers with English summaries. This allows us to leverage the models' English capabilities to create better representations for non-English papers. Our best model significantly outperforms strong baselines by 7-16% (in mean average precision).

{{</citation>}}


### (23/131) NSOAMT -- New Search Only Approach to Machine Translation (João Luís et al., 2023)

{{<citation>}}

João Luís, Diogo Cardoso, José Marques, Luís Campos. (2023)  
**NSOAMT -- New Search Only Approach to Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.10526v1)  

---


**ABSTRACT**  
Translation automation mechanisms and tools have been developed for several years to bring people who speak different languages together. A "new search only approach to machine translation" was adopted to tackle some of the slowness and inaccuracy of the other technologies. The idea is to develop a solution that, by indexing an incremental set of words that combine a certain semantic meaning, makes it possible to create a process of correspondence between their native language record and the language of translation. This research principle assumes that the vocabulary used in a given type of publication/document is relatively limited in terms of language style and word diversity, which enhances the greater effect of instantaneously and rigor in the translation process through the indexing process. A volume of electronic text documents where processed and loaded into a database, and analyzed and measured in order confirm the previous premise. Although the observed and projected metric values did not give encouraging results, it was possible to develop and make available a translation tool using this approach.

{{</citation>}}


### (24/131) Enhancing Open-Domain Table Question Answering via Syntax- and Structure-aware Dense Retrieval (Nengzheng Jin et al., 2023)

{{<citation>}}

Nengzheng Jin, Dongfang Li, Junying Chen, Joanna Siebert, Qingcai Chen. (2023)  
**Enhancing Open-Domain Table Question Answering via Syntax- and Structure-aware Dense Retrieval**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.10506v1)  

---


**ABSTRACT**  
Open-domain table question answering aims to provide answers to a question by retrieving and extracting information from a large collection of tables. Existing studies of open-domain table QA either directly adopt text retrieval methods or consider the table structure only in the encoding layer for table retrieval, which may cause syntactical and structural information loss during table scoring. To address this issue, we propose a syntax- and structure-aware retrieval method for the open-domain table QA task. It provides syntactical representations for the question and uses the structural header and value representations for the tables to avoid the loss of fine-grained syntactical and structural information. Then, a syntactical-to-structural aggregator is used to obtain the matching score between the question and a candidate table by mimicking the human retrieval process. Experimental results show that our method achieves the state-of-the-art on the NQ-tables dataset and overwhelms strong baselines on a newly curated open-domain Text-to-SQL dataset.

{{</citation>}}


### (25/131) An Evaluation of GPT-4 on the ETHICS Dataset (Sergey Rodionov et al., 2023)

{{<citation>}}

Sergey Rodionov, Zarathustra Amadeus Goertzel, Ben Goertzel. (2023)  
**An Evaluation of GPT-4 on the ETHICS Dataset**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.10492v1)  

---


**ABSTRACT**  
This report summarizes a short study of the performance of GPT-4 on the ETHICS dataset. The ETHICS dataset consists of five sub-datasets covering different fields of ethics: Justice, Deontology, Virtue Ethics, Utilitarianism, and Commonsense Ethics. The moral judgments were curated so as to have a high degree of agreement with the aim of representing shared human values rather than moral dilemmas. GPT-4's performance is much better than that of previous models and suggests that learning to work with common human values is not the hard problem for AI ethics.

{{</citation>}}


### (26/131) Toward Unified Controllable Text Generation via Regular Expression Instruction (Xin Zheng et al., 2023)

{{<citation>}}

Xin Zheng, Hongyu Lin, Xianpei Han, Le Sun. (2023)  
**Toward Unified Controllable Text Generation via Regular Expression Instruction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2309.10447v2)  

---


**ABSTRACT**  
Controllable text generation is a fundamental aspect of natural language generation, with numerous methods proposed for different constraint types. However, these approaches often require significant architectural or decoding modifications, making them challenging to apply to additional constraints or resolve different constraint combinations. To address this, our paper introduces Regular Expression Instruction (REI), which utilizes an instruction-based mechanism to fully exploit regular expressions' advantages to uniformly model diverse constraints. Specifically, our REI supports all popular fine-grained controllable generation constraints, i.e., lexical, positional, and length, as well as their complex combinations, via regular expression-style instructions. Our method only requires fine-tuning on medium-scale language models or few-shot, in-context learning on large language models, and requires no further adjustment when applied to various constraint combinations. Experiments demonstrate that our straightforward approach yields high success rates and adaptability to various constraints while maintaining competitiveness in automatic metrics and outperforming most previous baselines.

{{</citation>}}


### (27/131) PICK: Polished & Informed Candidate Scoring for Knowledge-Grounded Dialogue Systems (Bryan Wilie et al., 2023)

{{<citation>}}

Bryan Wilie, Yan Xu, Willy Chung, Samuel Cahyawijaya, Holy Lovenia, Pascale Fung. (2023)  
**PICK: Polished & Informed Candidate Scoring for Knowledge-Grounded Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2309.10413v1)  

---


**ABSTRACT**  
Grounding dialogue response generation on external knowledge is proposed to produce informative and engaging responses. However, current knowledge-grounded dialogue (KGD) systems often fail to align the generated responses with human-preferred qualities due to several issues like hallucination and the lack of coherence. Upon analyzing multiple language model generations, we observe the presence of alternative generated responses within a single decoding process. These alternative responses are more faithful and exhibit a comparable or higher level of relevance to prior conversational turns compared to the optimal responses prioritized by the decoding processes. To address these challenges and driven by these observations, we propose Polished \& Informed Candidate Scoring (PICK), a generation re-scoring framework that empowers models to generate faithful and relevant responses without requiring additional labeled data or model tuning. Through comprehensive automatic and human evaluations, we demonstrate the effectiveness of PICK in generating responses that are more faithful while keeping them relevant to the dialogue history. Furthermore, PICK consistently improves the system's performance with both oracle and retrieved knowledge in all decoding strategies. We provide the detailed implementation in https://github.com/bryanwilie/pick .

{{</citation>}}


### (28/131) PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training (Dawei Zhu et al., 2023)

{{<citation>}}

Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, Sujian Li. (2023)  
**PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2309.10400v1)  

---


**ABSTRACT**  
In this paper, we introduce Positional Skip-wisE (PoSE) training for efficient adaptation of large language models~(LLMs) to extremely long context windows. PoSE decouples train length from target context window size by simulating long inputs using a fixed context window with manipulated position indices during training. Concretely, we select several short chunks from a long input sequence, and introduce distinct skipping bias terms to modify the position indices of each chunk. These bias terms, along with the length of each chunk, are altered for each training example, allowing the model to adapt to all positions within the target context window without training on full length inputs. Experiments show that, compared with fine-tuning on the full length, PoSE greatly reduces memory and time overhead with minimal impact on performance. Leveraging this advantage, we have successfully extended the LLaMA model to 128k tokens. Furthermore, we empirically confirm that PoSE is compatible with all RoPE-based LLMs and various position interpolation strategies. Notably, by decoupling fine-tuning length from target context window, PoSE can theoretically extend the context window infinitely, constrained only by memory usage for inference. With ongoing advancements for efficient inference, we believe PoSE holds great promise for scaling the context window even further.

{{</citation>}}


### (29/131) KoBigBird-large: Transformation of Transformer for Korean Language Understanding (Kisu Yang et al., 2023)

{{<citation>}}

Kisu Yang, Yoonna Jang, Taewoo Lee, Jinwoo Seong, Hyungjin Lee, Hwanseok Jang, Heuiseok Lim. (2023)  
**KoBigBird-large: Transformation of Transformer for Korean Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.10339v1)  

---


**ABSTRACT**  
This work presents KoBigBird-large, a large size of Korean BigBird that achieves state-of-the-art performance and allows long sequence processing for Korean language understanding. Without further pretraining, we only transform the architecture and extend the positional encoding with our proposed Tapered Absolute Positional Encoding Representations (TAPER). In experiments, KoBigBird-large shows state-of-the-art overall performance on Korean language understanding benchmarks and the best performance on document classification and question answering tasks for longer sequences against the competitive baseline models. We publicly release our model here.

{{</citation>}}


### (30/131) QASnowball: An Iterative Bootstrapping Framework for High-Quality Question-Answering Data Generation (Kunlun Zhu et al., 2023)

{{<citation>}}

Kunlun Zhu, Shihao Liang, Xu Han, Zhi Zheng, Guoyang Zeng, Zhiyuan Liu, Maosong Sun. (2023)  
**QASnowball: An Iterative Bootstrapping Framework for High-Quality Question-Answering Data Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, QA  
[Paper Link](http://arxiv.org/abs/2309.10326v2)  

---


**ABSTRACT**  
Recent years have witnessed the success of question answering (QA), especially its potential to be a foundation paradigm for tackling diverse NLP tasks. However, obtaining sufficient data to build an effective and stable QA system still remains an open problem. For this problem, we introduce an iterative bootstrapping framework for QA data augmentation (named QASnowball), which can iteratively generate large-scale high-quality QA data based on a seed set of supervised examples. Specifically, QASnowball consists of three modules, an answer extractor to extract core phrases in unlabeled documents as candidate answers, a question generator to generate questions based on documents and candidate answers, and a QA data filter to filter out high-quality QA data. Moreover, QASnowball can be self-enhanced by reseeding the seed set to fine-tune itself in different iterations, leading to continual improvements in the generation quality. We conduct experiments in the high-resource English scenario and the medium-resource Chinese scenario, and the experimental results show that the data generated by QASnowball can facilitate QA models: (1) training models on the generated data achieves comparable results to using supervised data, and (2) pre-training on the generated data and fine-tuning on supervised data can achieve better performance. Our code and generated data will be released to advance further work.

{{</citation>}}


### (31/131) Investigating the Catastrophic Forgetting in Multimodal Large Language Models (Yuexiang Zhai et al., 2023)

{{<citation>}}

Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, Yi Ma. (2023)  
**Investigating the Catastrophic Forgetting in Multimodal Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10313v1)  

---


**ABSTRACT**  
Following the success of GPT4, there has been a surge in interest in multimodal large language model (MLLM) research. This line of research focuses on developing general-purpose LLMs through fine-tuning pre-trained LLMs and vision models. However, catastrophic forgetting, a notorious phenomenon where the fine-tuned model fails to retain similar performance compared to the pre-trained model, still remains an inherent problem in multimodal LLMs (MLLM). In this paper, we introduce EMT: Evaluating MulTimodality for evaluating the catastrophic forgetting in MLLMs, by treating each MLLM as an image classifier. We first apply EMT to evaluate several open-source fine-tuned MLLMs and we discover that almost all evaluated MLLMs fail to retain the same performance levels as their vision encoders on standard image classification tasks. Moreover, we continue fine-tuning LLaVA, an MLLM and utilize EMT to assess performance throughout the fine-tuning. Interestingly, our results suggest that early-stage fine-tuning on an image dataset improves performance across other image datasets, by enhancing the alignment of text and visual features. However, as fine-tuning proceeds, the MLLMs begin to hallucinate, resulting in a significant loss of generalizability, even when the image encoder remains frozen. Our results suggest that MLLMs have yet to demonstrate performance on par with their vision models on standard image classification tasks and the current MLLM fine-tuning procedure still has room for improvement.

{{</citation>}}


### (32/131) Rigorously Assessing Natural Language Explanations of Neurons (Jing Huang et al., 2023)

{{<citation>}}

Jing Huang, Atticus Geiger, Karel D'Oosterlinck, Zhengxuan Wu, Christopher Potts. (2023)  
**Rigorously Assessing Natural Language Explanations of Neurons**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.10312v1)  

---


**ABSTRACT**  
Natural language is an appealing medium for explaining how large language models process and store information, but evaluating the faithfulness of such explanations is challenging. To help address this, we develop two modes of evaluation for natural language explanations that claim individual neurons represent a concept in a text input. In the observational mode, we evaluate claims that a neuron $a$ activates on all and only input strings that refer to a concept picked out by the proposed explanation $E$. In the intervention mode, we construe $E$ as a claim that the neuron $a$ is a causal mediator of the concept denoted by $E$. We apply our framework to the GPT-4-generated explanations of GPT-2 XL neurons of Bills et al. (2023) and show that even the most confident explanations have high error rates and little to no causal efficacy. We close the paper by critically assessing whether natural language is a good choice for explanations and whether neurons are the best level of analysis.

{{</citation>}}


### (33/131) Baichuan 2: Open Large-scale Language Models (Aiyuan Yang et al., 2023)

{{<citation>}}

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, JunTao Dai, Kun Fang, Lei Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie, Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, Zhiying Wu. (2023)  
**Baichuan 2: Open Large-scale Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.10305v2)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated remarkable performance on a variety of natural language tasks based on just a few examples of natural language instructions, reducing the need for extensive feature engineering. However, most powerful LLMs are closed-source or limited in their capability for languages other than English. In this technical report, we present Baichuan 2, a series of large-scale multilingual language models containing 7 billion and 13 billion parameters, trained from scratch, on 2.6 trillion tokens. Baichuan 2 matches or outperforms other open-source models of similar size on public benchmarks like MMLU, CMMLU, GSM8K, and HumanEval. Furthermore, Baichuan 2 excels in vertical domains such as medicine and law. We will release all pre-training model checkpoints to benefit the research community in better understanding the training dynamics of Baichuan 2.

{{</citation>}}


### (34/131) Leveraging Speech PTM, Text LLM, and Emotional TTS for Speech Emotion Recognition (Ziyang Ma et al., 2023)

{{<citation>}}

Ziyang Ma, Wen Wu, Zhisheng Zheng, Yiwei Guo, Qian Chen, Shiliang Zhang, Xie Chen. (2023)  
**Leveraging Speech PTM, Text LLM, and Emotional TTS for Speech Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Azure, Emotion Recognition, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.10294v1)  

---


**ABSTRACT**  
In this paper, we explored how to boost speech emotion recognition (SER) with the state-of-the-art speech pre-trained model (PTM), data2vec, text generation technique, GPT-4, and speech synthesis technique, Azure TTS. First, we investigated the representation ability of different speech self-supervised pre-trained models, and we found that data2vec has a good representation ability on the SER task. Second, we employed a powerful large language model (LLM), GPT-4, and emotional text-to-speech (TTS) model, Azure TTS, to generate emotionally congruent text and speech. We carefully designed the text prompt and dataset construction, to obtain the synthetic emotional speech data with high quality. Third, we studied different ways of data augmentation to promote the SER task with synthetic speech, including random mixing, adversarial training, transfer learning, and curriculum learning. Experiments and ablation studies on the IEMOCAP dataset demonstrate the effectiveness of our method, compared with other data augmentation methods, and data augmentation with other synthetic data.

{{</citation>}}


### (35/131) Mixed-Distil-BERT: Code-mixed Language Modeling for Bangla, English, and Hindi (Md Nishat Raihan et al., 2023)

{{<citation>}}

Md Nishat Raihan, Dhiman Goswami, Antara Mahmud. (2023)  
**Mixed-Distil-BERT: Code-mixed Language Modeling for Bangla, English, and Hindi**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.10272v1)  

---


**ABSTRACT**  
One of the most popular downstream tasks in the field of Natural Language Processing is text classification. Text classification tasks have become more daunting when the texts are code-mixed. Though they are not exposed to such text during pre-training, different BERT models have demonstrated success in tackling Code-Mixed NLP challenges. Again, in order to enhance their performance, Code-Mixed NLP models have depended on combining synthetic data with real-world data. It is crucial to understand how the BERT models' performance is impacted when they are pretrained using corresponding code-mixed languages. In this paper, we introduce Tri-Distil-BERT, a multilingual model pre-trained on Bangla, English, and Hindi, and Mixed-Distil-BERT, a model fine-tuned on code-mixed data. Both models are evaluated across multiple NLP tasks and demonstrate competitive performance against larger models like mBERT and XLM-R. Our two-tiered pre-training approach offers efficient alternatives for multilingual and code-mixed language understanding, contributing to advancements in the field.

{{</citation>}}


### (36/131) What is the Best Automated Metric for Text to Motion Generation? (Jordan Voas et al., 2023)

{{<citation>}}

Jordan Voas, Yili Wang, Qixing Huang, Raymond Mooney. (2023)  
**What is the Best Automated Metric for Text to Motion Generation?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-GR, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.10248v1)  

---


**ABSTRACT**  
There is growing interest in generating skeleton-based human motions from natural language descriptions. While most efforts have focused on developing better neural architectures for this task, there has been no significant work on determining the proper evaluation metric. Human evaluation is the ultimate accuracy measure for this task, and automated metrics should correlate well with human quality judgments. Since descriptions are compatible with many motions, determining the right metric is critical for evaluating and designing effective generative models. This paper systematically studies which metrics best align with human evaluations and proposes new metrics that align even better. Our findings indicate that none of the metrics currently used for this task show even a moderate correlation with human judgments on a sample level. However, for assessing average model performance, commonly used metrics such as R-Precision and less-used coordinate errors show strong correlations. Additionally, several recently developed metrics are not recommended due to their low correlation compared to alternatives. We also introduce a novel metric based on a multimodal BERT-like model, MoBERT, which offers strongly human-correlated sample-level evaluations while maintaining near-perfect model-level correlation. Our results demonstrate that this new metric exhibits extensive benefits over all current alternatives.

{{</citation>}}


### (37/131) PolicyGPT: Automated Analysis of Privacy Policies with Large Language Models (Chenhao Tang et al., 2023)

{{<citation>}}

Chenhao Tang, Zhengliang Liu, Chong Ma, Zihao Wu, Yiwei Li, Wei Liu, Dajiang Zhu, Quanzheng Li, Xiang Li, Tianming Liu, Lei Fan. (2023)  
**PolicyGPT: Automated Analysis of Privacy Policies with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10238v1)  

---


**ABSTRACT**  
Privacy policies serve as the primary conduit through which online service providers inform users about their data collection and usage procedures. However, in a bid to be comprehensive and mitigate legal risks, these policy documents are often quite verbose. In practical use, users tend to click the Agree button directly rather than reading them carefully. This practice exposes users to risks of privacy leakage and legal issues. Recently, the advent of Large Language Models (LLM) such as ChatGPT and GPT-4 has opened new possibilities for text analysis, especially for lengthy documents like privacy policies. In this study, we investigate a privacy policy text analysis framework PolicyGPT based on the LLM. This framework was tested using two datasets. The first dataset comprises of privacy policies from 115 websites, which were meticulously annotated by legal experts, categorizing each segment into one of 10 classes. The second dataset consists of privacy policies from 304 popular mobile applications, with each sentence manually annotated and classified into one of another 10 categories. Under zero-shot learning conditions, PolicyGPT demonstrated robust performance. For the first dataset, it achieved an accuracy rate of 97%, while for the second dataset, it attained an 87% accuracy rate, surpassing that of the baseline machine learning and neural network models.

{{</citation>}}


## math.OC (1)



### (38/131) Deep Reinforcement Learning for Infinite Horizon Mean Field Problems in Continuous Spaces (Andrea Angiuli et al., 2023)

{{<citation>}}

Andrea Angiuli, Jean-Pierre Fouque, Ruimeng Hu, Alan Raydan. (2023)  
**Deep Reinforcement Learning for Infinite Horizon Mean Field Problems in Continuous Spaces**  

---
Primary Category: math.OC  
Categories: cs-LG, math-OC, math.OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10953v1)  

---


**ABSTRACT**  
We present the development and analysis of a reinforcement learning (RL) algorithm designed to solve continuous-space mean field game (MFG) and mean field control (MFC) problems in a unified manner. The proposed approach pairs the actor-critic (AC) paradigm with a representation of the mean field distribution via a parameterized score function, which can be efficiently updated in an online fashion, and uses Langevin dynamics to obtain samples from the resulting distribution. The AC agent and the score function are updated iteratively to converge, either to the MFG equilibrium or the MFC optimum for a given mean field problem, depending on the choice of learning rates. A straightforward modification of the algorithm allows us to solve mixed mean field control games (MFCGs). The performance of our algorithm is evaluated using linear-quadratic benchmarks in the asymptotic infinite horizon framework.

{{</citation>}}


## cs.HC (9)



### (39/131) How Do Analysts Understand and Verify AI-Assisted Data Analyses? (Ken Gu et al., 2023)

{{<citation>}}

Ken Gu, Ruoxi Shang, Tim Althoff, Chenglong Wang, Steven M. Drucker. (2023)  
**How Do Analysts Understand and Verify AI-Assisted Data Analyses?**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.10947v1)  

---


**ABSTRACT**  
Data analysis is challenging as it requires synthesizing domain knowledge, statistical expertise, and programming skills. Assistants powered by large language models (LLMs), such as ChatGPT, can assist analysts by translating natural language instructions into code. However, AI-assistant responses and analysis code can be misaligned with the analyst's intent or be seemingly correct but lead to incorrect conclusions Therefore, validating AI assistance is crucial and challenging. Here, we explore how analysts across a range of backgrounds and expertise understand and verify the correctness of AI-generated analyses. We develop a design probe that allows analysts to pursue diverse verification workflows using natural language explanations, code, visualizations, inspecting data tables, and performing common data operations. Through a qualitative user study (n=22) using this probe, we uncover common patterns of verification workflows influenced by analysts' programming, analysis, and AI backgrounds. Additionally, we highlight open challenges and opportunities for improving future AI analysis assistant experiences.

{{</citation>}}


### (40/131) Field evaluation of a mobile app for assisting blind and visually impaired travelers to find bus stops (Shrinivas Pundlik et al., 2023)

{{<citation>}}

Shrinivas Pundlik, Prerana Shivshanker, Tim Traut-Savino, Gang Luo. (2023)  
**Field evaluation of a mobile app for assisting blind and visually impaired travelers to find bus stops**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.10940v1)  

---


**ABSTRACT**  
Purpose: It is reported that there can be considerable gaps due to GPS inaccuracy and mapping errors if blind and visually impaired (BVI) travelers rely on digital maps to go to their desired bus stops. We evaluated the ability of a mobile app, All_Aboard, to guide BVI travelers precisely to the bus-stops. Methods: The All_Aboard app detected bus-stop signs in real-time via smartphone camera using a neural network model, and provided distance coded audio feedback to help localize the detected sign. BVI individuals used the All_Aboard and Google Maps app to localize 10 bus-stop locations in Boston downtown and another 10 in a sub-urban area. For each bus stop, the subjects used the apps to navigate as close as possible to the physical bus-stop sign, starting from 30 to 50 meters away. The outcome measures were success rate and gap distance between the app-indicated location and the actual physical location of the bus stop. Results: The study was conducted with 24 legally blind participants (mean age [SD]: 51[14] years; 11 (46%) Female). The success rate of the All_Aboard app (91%) was significantly higher than the Google Maps (52%, p<0.001). The gap distance when using the All_Aboard app was significantly lower (mean [95%CI]: 1.8 [1.2-2.3] meters) compared to the Google Maps (7 [6.5-7.5] meters; p<0.001). Conclusion: The All_Aboard app localizes bus stops more accurately and reliably than GPS-based smartphone navigation options in real-world environments.

{{</citation>}}


### (41/131) Large Language Models as Agents in the Clinic (Nikita Mehandru et al., 2023)

{{<citation>}}

Nikita Mehandru, Brenda Y. Miao, Eduardo Rodriguez Almaraz, Madhumita Sushil, Atul J. Butte, Ahmed Alaa. (2023)  
**Large Language Models as Agents in the Clinic**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-MA, cs.HC  
Keywords: AI, Clinical, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10895v1)  

---


**ABSTRACT**  
Recent developments in large language models (LLMs) have unlocked new opportunities for healthcare, from information synthesis to clinical decision support. These new LLMs are not just capable of modeling language, but can also act as intelligent "agents" that interact with stakeholders in open-ended conversations and even influence clinical decision-making. Rather than relying on benchmarks that measure a model's ability to process clinical data or answer standardized test questions, LLM agents should be assessed for their performance on real-world clinical tasks. These new evaluation frameworks, which we call "Artificial-intelligence Structured Clinical Examinations" ("AI-SCI"), can draw from comparable technologies where machines operate with varying degrees of self-governance, such as self-driving cars. High-fidelity simulations may also be used to evaluate interactions between users and LLMs within a clinical workflow, or to model the dynamic interactions of multiple LLMs. Developing these robust, real-world clinical evaluations will be crucial towards deploying LLM agents into healthcare.

{{</citation>}}


### (42/131) Redefining Qualitative Analysis in the AI Era: Utilizing ChatGPT for Efficient Thematic Analysis (He Zhang et al., 2023)

{{<citation>}}

He Zhang, Chuhao Wu, Jingyi Xie, Yao Lyu, Jie Cai, John M. Carroll. (2023)  
**Redefining Qualitative Analysis in the AI Era: Utilizing ChatGPT for Efficient Thematic Analysis**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.10771v1)  

---


**ABSTRACT**  
Thematic analysis is a cornerstone of qualitative research, yet it is often marked by labor-intensive procedures. Recent advances in artificial intelligence (AI), especially with large-scale language models (LLMs) such as ChatGPT, present potential avenues to enhance qualitative data analysis. This research delves into the effectiveness of ChatGPT in refining the thematic analysis process. We conducted semi-structured interviews with 17 participants, inclusive of a 4-participant pilot study, to identify the challenges and reservations concerning the incorporation of ChatGPT in qualitative analysis. In partnership with 13 qualitative analysts, we crafted cueing frameworks to bolster ChatGPT's contribution to thematic analysis. The results indicate that these frameworks not only amplify the quality of thematic analysis but also bridge a significant connection between AI and qualitative research. These insights carry pivotal implications for academics and professionals keen on harnessing AI for qualitative data exploration.

{{</citation>}}


### (43/131) From 'Let's Google' to 'Let's ChatGPT': Student and Instructor Perspectives on the influence of LLMs on Undergraduate Engineering Education (Ishika Joshi et al., 2023)

{{<citation>}}

Ishika Joshi, Ritvik Budhiraja, Pranav Deepak Tanna, Lovenya Jain, Mihika Deshpande, Arjun Srivastava, Srinivas Rallapalli, Harshal D Akolekar, Jagat Sesh Challa, Dhruv Kumar. (2023)  
**From 'Let's Google' to 'Let's ChatGPT': Student and Instructor Perspectives on the influence of LLMs on Undergraduate Engineering Education**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-ET, cs-HC, cs.HC  
Keywords: ChatGPT, GPT, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10694v1)  

---


**ABSTRACT**  
The rise in popularity of Large Language Models (LLMs) has prompted discussions in academic circles, with students exploring LLM-based tools for coursework inquiries and instructors exploring them for teaching and research. Even though a lot of work is underway to create LLM-based tools tailored for students and instructors, there is a lack of comprehensive user studies that capture the perspectives of students and instructors regarding LLMs. This paper addresses this gap by conducting surveys and interviews within undergraduate engineering universities in India. Using 1306 survey responses among students, 112 student interviews, and 27 instructor interviews around the academic usage of ChatGPT (a popular LLM), this paper offers insights into the current usage patterns, perceived benefits, threats, and challenges, as well as recommendations for enhancing the adoption of LLMs among students and instructors. These insights are further utilized to discuss the practical implications of LLMs in undergraduate engineering education and beyond.

{{</citation>}}


### (44/131) Writer-Defined AI Personas for On-Demand Feedback Generation (Karim Benharrak et al., 2023)

{{<citation>}}

Karim Benharrak, Tim Zindulka, Florian Lehmann, Hendrik Heuer, Daniel Buschek. (2023)  
**Writer-Defined AI Personas for On-Demand Feedback Generation**  

---
Primary Category: cs.HC  
Categories: H-5-2; I-2-7, cs-CL, cs-HC, cs.HC  
Keywords: AI, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2309.10433v1)  

---


**ABSTRACT**  
Compelling writing is tailored to its audience. This is challenging, as writers may struggle to empathize with readers, get feedback in time, or gain access to the target group. We propose a concept that generates on-demand feedback, based on writer-defined AI personas of any target audience. We explore this concept with a prototype (using GPT-3.5) in two user studies (N=5 and N=11): Writers appreciated the concept and strategically used personas for getting different perspectives. The feedback was seen as helpful and inspired revisions of text and personas, although it was often verbose and unspecific. We discuss the impact of on-demand feedback, the limited representativity of contemporary AI systems, and further ideas for defining AI personas. This work contributes to the vision of supporting writers with AI by expanding the socio-technical perspective in AI tool design: To empower creators, we also need to keep in mind their relationship to an audience.

{{</citation>}}


### (45/131) Learning from Teaching Assistants to Program with Subgoals: Exploring the Potential for AI Teaching Assistants (Changyoon Lee et al., 2023)

{{<citation>}}

Changyoon Lee, Junho Myung, Jieun Han, Jiho Jin, Alice Oh. (2023)  
**Learning from Teaching Assistants to Program with Subgoals: Exploring the Potential for AI Teaching Assistants**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.10419v1)  

---


**ABSTRACT**  
With recent advances in generative AI, conversational models like ChatGPT have become feasible candidates for TAs. We investigate the practicality of using generative AI as TAs in introductory programming education by examining novice learners' interaction with TAs in a subgoal learning environment. To compare the learners' interaction and perception of the AI and human TAs, we conducted a between-subject study with 20 novice programming learners. Learners solve programming tasks by producing subgoals and subsolutions with the guidance of a TA. Our study shows that learners can solve tasks faster with comparable scores with AI TAs. Learners' perception of the AI TA is on par with that of human TAs in terms of speed and comprehensiveness of the replies and helpfulness, difficulty, and satisfaction of the conversation. Finally, we suggest guidelines to better design and utilize generative AI as TAs in programming education from the result of our chat log analysis.

{{</citation>}}


### (46/131) Natural Language Dataset Generation Framework for Visualizations Powered by Large Language Models (Hyung-Kwon Ko et al., 2023)

{{<citation>}}

Hyung-Kwon Ko, Hyeon Jeon, Gwanmo Park, Dae Hyun Kim, Nam Wook Kim, Juho Kim, Jinwook Seo. (2023)  
**Natural Language Dataset Generation Framework for Visualizations Powered by Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model, NLI  
[Paper Link](http://arxiv.org/abs/2309.10245v1)  

---


**ABSTRACT**  
We introduce a Large Language Model (LLM) framework that generates rich and diverse NL datasets using only Vega-Lite specifications as input, thereby streamlining the development of Natural Language Interfaces (NLIs) for data visualization. We propose two techniques to synthesize relevant chart semantics accurately and enhance syntactic diversity in each NL dataset, respectively: 1) a guided discovery incorporated into prompting so that LLMs can steer themselves to create varying NL datasets in a self-directed manner; 2) a score-based paraphrasing to augment NL syntax along with four well-defined language axes. We also present a new chart collection of 1,981 real-world Vega-Lite specifications that have increased diversity and complexity compared to benchmarks, to demonstrate the generalizability of our framework. The experimental results show that our framework accurately extracts chart semantics and generates L1/L2 captions with 89.4% and 76.0% accuracy, respectively, while generating and paraphrasing utterances and questions with greater diversity than benchmarks. The codes and chart collection are available at https://github.com/hyungkwonko/chart-llm.

{{</citation>}}


### (47/131) Drive as You Speak: Enabling Human-Like Interaction with Large Language Models in Autonomous Vehicles (Can Cui et al., 2023)

{{<citation>}}

Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Ziran Wang. (2023)  
**Drive as You Speak: Enabling Human-Like Interaction with Large Language Models in Autonomous Vehicles**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10228v1)  

---


**ABSTRACT**  
The future of autonomous vehicles lies in the convergence of human-centric design and advanced AI capabilities. Autonomous vehicles of the future will not only transport passengers but also interact and adapt to their desires, making the journey comfortable, efficient, and pleasant. In this paper, we present a novel framework that leverages Large Language Models (LLMs) to enhance autonomous vehicles' decision-making processes. By integrating LLMs' natural language capabilities and contextual understanding, specialized tools usage, synergizing reasoning, and acting with various modules on autonomous vehicles, this framework aims to seamlessly integrate the advanced language and reasoning capabilities of LLMs into autonomous vehicles. The proposed framework holds the potential to revolutionize the way autonomous vehicles operate, offering personalized assistance, continuous learning, and transparent decision-making, ultimately contributing to safer and more efficient autonomous driving technologies.

{{</citation>}}


## cs.CV (23)



### (48/131) A Geometric Flow Approach for Segmentation of Images with Inhomongeneous Intensity and Missing Boundaries (Paramjyoti Mohapatra et al., 2023)

{{<citation>}}

Paramjyoti Mohapatra, Richard Lartey, Weihong Guo, Michael Judkovich, Xiaojuan Li. (2023)  
**A Geometric Flow Approach for Segmentation of Images with Inhomongeneous Intensity and Missing Boundaries**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.10935v1)  

---


**ABSTRACT**  
Image segmentation is a complex mathematical problem, especially for images that contain intensity inhomogeneity and tightly packed objects with missing boundaries in between. For instance, Magnetic Resonance (MR) muscle images often contain both of these issues, making muscle segmentation especially difficult. In this paper we propose a novel intensity correction and a semi-automatic active contour based segmentation approach. The approach uses a geometric flow that incorporates a reproducing kernel Hilbert space (RKHS) edge detector and a geodesic distance penalty term from a set of markers and anti-markers. We test the proposed scheme on MR muscle segmentation and compare with some state of the art methods. To help deal with the intensity inhomogeneity in this particular kind of image, a new approach to estimate the bias field using a fat fraction image, called Prior Bias-Corrected Fuzzy C-means (PBCFCM), is introduced. Numerical experiments show that the proposed scheme leads to significantly better results than compared ones. The average dice values of the proposed method are 92.5%, 85.3%, 85.3% for quadriceps, hamstrings and other muscle groups while other approaches are at least 10% worse.

{{</citation>}}


### (49/131) Language as the Medium: Multimodal Video Classification through text only (Laura Hanu et al., 2023)

{{<citation>}}

Laura Hanu, Anita L. Verő, James Thewlis. (2023)  
**Language as the Medium: Multimodal Video Classification through text only**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2309.10783v1)  

---


**ABSTRACT**  
Despite an exciting new wave of multimodal machine learning models, current approaches still struggle to interpret the complex contextual relationships between the different modalities present in videos. Going beyond existing methods that emphasize simple activities or objects, we propose a new model-agnostic approach for generating detailed textual descriptions that captures multimodal video information. Our method leverages the extensive knowledge learnt by large language models, such as GPT-3.5 or Llama2, to reason about textual descriptions of the visual and aural modalities, obtained from BLIP-2, Whisper and ImageBind. Without needing additional finetuning of video-text models or datasets, we demonstrate that available LLMs have the ability to use these multimodal textual descriptions as proxies for ``sight'' or ``hearing'' and perform zero-shot multimodal classification of videos in-context. Our evaluations on popular action recognition benchmarks, such as UCF-101 or Kinetics, show these context-rich descriptions can be successfully used in video understanding tasks. This method points towards a promising new research direction in multimodal classification, demonstrating how an interplay between textual, visual and auditory machine learning models can enable more holistic video understanding.

{{</citation>}}


### (50/131) MAGIC-TBR: Multiview Attention Fusion for Transformer-based Bodily Behavior Recognition in Group Settings (Surbhi Madan et al., 2023)

{{<citation>}}

Surbhi Madan, Rishabh Jain, Gulshan Sharma, Ramanathan Subramanian, Abhinav Dhall. (2023)  
**MAGIC-TBR: Multiview Attention Fusion for Transformer-based Bodily Behavior Recognition in Group Settings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs-MM, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.10765v1)  

---


**ABSTRACT**  
Bodily behavioral language is an important social cue, and its automated analysis helps in enhancing the understanding of artificial intelligence systems. Furthermore, behavioral language cues are essential for active engagement in social agent-based user interactions. Despite the progress made in computer vision for tasks like head and body pose estimation, there is still a need to explore the detection of finer behaviors such as gesturing, grooming, or fumbling. This paper proposes a multiview attention fusion method named MAGIC-TBR that combines features extracted from videos and their corresponding Discrete Cosine Transform coefficients via a transformer-based approach. The experiments are conducted on the BBSI dataset and the results demonstrate the effectiveness of the proposed feature fusion with multiview attention. The code is available at: https://github.com/surbhimadan92/MAGIC-TBR

{{</citation>}}


### (51/131) Few-Shot Panoptic Segmentation With Foundation Models (Markus Käppeler et al., 2023)

{{<citation>}}

Markus Käppeler, Kürsat Petek, Niclas Vödisch, Wolfram Burgard, Abhinav Valada. (2023)  
**Few-Shot Panoptic Segmentation With Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.10726v1)  

---


**ABSTRACT**  
Current state-of-the-art methods for panoptic segmentation require an immense amount of annotated training data that is both arduous and expensive to obtain posing a significant challenge for their widespread adoption. Concurrently, recent breakthroughs in visual representation learning have sparked a paradigm shift leading to the advent of large foundation models that can be trained with completely unlabeled images. In this work, we propose to leverage such task-agnostic image features to enable few-shot panoptic segmentation by presenting Segmenting Panoptic Information with Nearly 0 labels (SPINO). In detail, our method combines a DINOv2 backbone with lightweight network heads for semantic segmentation and boundary estimation. We show that our approach, albeit being trained with only ten annotated images, predicts high-quality pseudo-labels that can be used with any existing panoptic segmentation method. Notably, we demonstrate that SPINO achieves competitive results compared to fully supervised baselines while using less than 0.3% of the ground truth labels, paving the way for learning complex visual recognition tasks leveraging foundation models. To illustrate its general applicability, we further deploy SPINO on real-world robotic vision systems for both outdoor and indoor environments. To foster future research, we make the code and trained models publicly available at http://spino.cs.uni-freiburg.de.

{{</citation>}}


### (52/131) Interpret Vision Transformers as ConvNets with Dynamic Convolutions (Chong Zhou et al., 2023)

{{<citation>}}

Chong Zhou, Chen Change Loy, Bo Dai. (2023)  
**Interpret Vision Transformers as ConvNets with Dynamic Convolutions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.10713v1)  

---


**ABSTRACT**  
There has been a debate about the superiority between vision Transformers and ConvNets, serving as the backbone of computer vision models. Although they are usually considered as two completely different architectures, in this paper, we interpret vision Transformers as ConvNets with dynamic convolutions, which enables us to characterize existing Transformers and dynamic ConvNets in a unified framework and compare their design choices side by side. In addition, our interpretation can also guide the network design as researchers now can consider vision Transformers from the design space of ConvNets and vice versa. We demonstrate such potential through two specific studies. First, we inspect the role of softmax in vision Transformers as the activation function and find it can be replaced by commonly used ConvNets modules, such as ReLU and Layer Normalization, which results in a faster convergence rate and better performance. Second, following the design of depth-wise convolution, we create a corresponding depth-wise vision Transformer that is more efficient with comparable performance. The potential of the proposed unified interpretation is not limited to the given examples and we hope it can inspire the community and give rise to more advanced network architectures.

{{</citation>}}


### (53/131) Latent Space Energy-based Model for Fine-grained Open Set Recognition (Wentao Bao et al., 2023)

{{<citation>}}

Wentao Bao, Qi Yu, Yu Kong. (2023)  
**Latent Space Energy-based Model for Fine-grained Open Set Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10711v1)  

---


**ABSTRACT**  
Fine-grained open-set recognition (FineOSR) aims to recognize images belonging to classes with subtle appearance differences while rejecting images of unknown classes. A recent trend in OSR shows the benefit of generative models to discriminative unknown detection. As a type of generative model, energy-based models (EBM) are the potential for hybrid modeling of generative and discriminative tasks. However, most existing EBMs suffer from density estimation in high-dimensional space, which is critical to recognizing images from fine-grained classes. In this paper, we explore the low-dimensional latent space with energy-based prior distribution for OSR in a fine-grained visual world. Specifically, based on the latent space EBM, we propose an attribute-aware information bottleneck (AIB), a residual attribute feature aggregation (RAFA) module, and an uncertainty-based virtual outlier synthesis (UVOS) module to improve the expressivity, granularity, and density of the samples in fine-grained classes, respectively. Our method is flexible to take advantage of recent vision transformers for powerful visual classification and generation. The method is validated on both fine-grained and general visual classification datasets while preserving the capability of generating photo-realistic fake images with high resolution.

{{</citation>}}


### (54/131) Learning Tri-modal Embeddings for Zero-Shot Soundscape Mapping (Subash Khanal et al., 2023)

{{<citation>}}

Subash Khanal, Srikumar Sastry, Aayush Dhakal, Nathan Jacobs. (2023)  
**Learning Tri-modal Embeddings for Zero-Shot Soundscape Mapping**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: Embedding, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.10667v1)  

---


**ABSTRACT**  
We focus on the task of soundscape mapping, which involves predicting the most probable sounds that could be perceived at a particular geographic location. We utilise recent state-of-the-art models to encode geotagged audio, a textual description of the audio, and an overhead image of its capture location using contrastive pre-training. The end result is a shared embedding space for the three modalities, which enables the construction of soundscape maps for any geographic region from textual or audio queries. Using the SoundingEarth dataset, we find that our approach significantly outperforms the existing SOTA, with an improvement of image-to-audio Recall@100 from 0.256 to 0.450. Our code is available at https://github.com/mvrl/geoclap.

{{</citation>}}


### (55/131) Multi-Stain Self-Attention Graph Multiple Instance Learning Pipeline for Histopathology Whole Slide Images (Amaya Gallagher-Syed et al., 2023)

{{<citation>}}

Amaya Gallagher-Syed, Luca Rossi, Felice Rivellese, Costantino Pitzalis, Myles Lewis, Michael Barnes, Gregory Slabaugh. (2023)  
**Multi-Stain Self-Attention Graph Multiple Instance Learning Pipeline for Histopathology Whole Slide Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, q-bio-QM  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2309.10650v1)  

---


**ABSTRACT**  
Whole Slide Images (WSIs) present a challenging computer vision task due to their gigapixel size and presence of numerous artefacts. Yet they are a valuable resource for patient diagnosis and stratification, often representing the gold standard for diagnostic tasks. Real-world clinical datasets tend to come as sets of heterogeneous WSIs with labels present at the patient-level, with poor to no annotations. Weakly supervised attention-based multiple instance learning approaches have been developed in recent years to address these challenges, but can fail to resolve both long and short-range dependencies. Here we propose an end-to-end multi-stain self-attention graph (MUSTANG) multiple instance learning pipeline, which is designed to solve a weakly-supervised gigapixel multi-image classification task, where the label is assigned at the patient-level, but no slide-level labels or region annotations are available. The pipeline uses a self-attention based approach by restricting the operations to a highly sparse k-Nearest Neighbour Graph of embedded WSI patches based on the Euclidean distance. We show this approach achieves a state-of-the-art F1-score/AUC of 0.89/0.92, outperforming the widely used CLAM model. Our approach is highly modular and can easily be modified to suit different clinical datasets, as it only requires a patient-level label without annotations and accepts WSI sets of different sizes, as the graphs can be of varying sizes and structures. The source code can be found at https://github.com/AmayaGS/MUSTANG.

{{</citation>}}


### (56/131) Few-shot Object Detection in Remote Sensing: Lifting the Curse of Incompletely Annotated Novel Objects (Fahong Zhang et al., 2023)

{{<citation>}}

Fahong Zhang, Yilei Shi, Zhitong Xiong, Xiao Xiang Zhu. (2023)  
**Few-shot Object Detection in Remote Sensing: Lifting the Curse of Incompletely Annotated Novel Objects**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.10588v1)  

---


**ABSTRACT**  
Object detection is an essential and fundamental task in computer vision and satellite image processing. Existing deep learning methods have achieved impressive performance thanks to the availability of large-scale annotated datasets. Yet, in real-world applications the availability of labels is limited. In this context, few-shot object detection (FSOD) has emerged as a promising direction, which aims at enabling the model to detect novel objects with only few of them annotated. However, many existing FSOD algorithms overlook a critical issue: when an input image contains multiple novel objects and only a subset of them are annotated, the unlabeled objects will be considered as background during training. This can cause confusions and severely impact the model's ability to recall novel objects. To address this issue, we propose a self-training-based FSOD (ST-FSOD) approach, which incorporates the self-training mechanism into the few-shot fine-tuning process. ST-FSOD aims to enable the discovery of novel objects that are not annotated, and take them into account during training. On the one hand, we devise a two-branch region proposal networks (RPN) to separate the proposal extraction of base and novel objects, On another hand, we incorporate the student-teacher mechanism into RPN and the region of interest (RoI) head to include those highly confident yet unlabeled targets as pseudo labels. Experimental results demonstrate that our proposed method outperforms the state-of-the-art in various FSOD settings by a large margin. The codes will be publicly available at https://github.com/zhu-xlab/ST-FSOD.

{{</citation>}}


### (57/131) Adversarial Attacks Against Uncertainty Quantification (Emanuele Ledda et al., 2023)

{{<citation>}}

Emanuele Ledda, Daniele Angioni, Giorgio Piras, Giorgio Fumera, Battista Biggio, Fabio Roli. (2023)  
**Adversarial Attacks Against Uncertainty Quantification**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2309.10586v1)  

---


**ABSTRACT**  
Machine-learning models can be fooled by adversarial examples, i.e., carefully-crafted input perturbations that force models to output wrong predictions. While uncertainty quantification has been recently proposed to detect adversarial inputs, under the assumption that such attacks exhibit a higher prediction uncertainty than pristine data, it has been shown that adaptive attacks specifically aimed at reducing also the uncertainty estimate can easily bypass this defense mechanism. In this work, we focus on a different adversarial scenario in which the attacker is still interested in manipulating the uncertainty estimate, but regardless of the correctness of the prediction; in particular, the goal is to undermine the use of machine-learning models when their outputs are consumed by a downstream module or by a human operator. Following such direction, we: \textit{(i)} design a threat model for attacks targeting uncertainty quantification; \textit{(ii)} devise different attack strategies on conceptually different UQ techniques spanning for both classification and semantic segmentation problems; \textit{(iii)} conduct a first complete and extensive analysis to compare the differences between some of the most employed UQ approaches under attack. Our extensive experimental analysis shows that our attacks are more effective in manipulating uncertainty quantification measures than attacks aimed to also induce misclassifications.

{{</citation>}}


### (58/131) An overview of some mathematical techniques and problems linking 3D vision to 3D printing (Emiliano Cristiani et al., 2023)

{{<citation>}}

Emiliano Cristiani, Maurizio Falcone, Silvia Tozza. (2023)  
**An overview of some mathematical techniques and problems linking 3D vision to 3D printing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-NA, cs.CV, math-NA  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2309.10549v1)  

---


**ABSTRACT**  
Computer Vision and 3D printing have rapidly evolved in the last 10 years but interactions among them have been very limited so far, despite the fact that they share several mathematical techniques. We try to fill the gap presenting an overview of some techniques for Shape-from-Shading problems as well as for 3D printing with an emphasis on the approaches based on nonlinear partial differential equations and optimization. We also sketch possible couplings to complete the process of object manufacturing starting from one or more images of the object and ending with its final 3D print. We will give some practical examples of this procedure.

{{</citation>}}


### (59/131) Retinex-guided Channel-grouping based Patch Swap for Arbitrary Style Transfer (Chang Liu et al., 2023)

{{<citation>}}

Chang Liu, Yi Niu, Mingming Ma, Fu Li, Guangming Shi. (2023)  
**Retinex-guided Channel-grouping based Patch Swap for Arbitrary Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.10528v1)  

---


**ABSTRACT**  
The basic principle of the patch-matching based style transfer is to substitute the patches of the content image feature maps by the closest patches from the style image feature maps. Since the finite features harvested from one single aesthetic style image are inadequate to represent the rich textures of the content natural image, existing techniques treat the full-channel style feature patches as simple signal tensors and create new style feature patches via signal-level fusion, which ignore the implicit diversities existed in style features and thus fail for generating better stylised results. In this paper, we propose a Retinex theory guided, channel-grouping based patch swap technique to solve the above challenges. Channel-grouping strategy groups the style feature maps into surface and texture channels, which prevents the winner-takes-all problem. Retinex theory based decomposition controls a more stable channel code rate generation. In addition, we provide complementary fusion and multi-scale generation strategy to prevent unexpected black area and over-stylised results respectively. Experimental results demonstrate that the proposed method outperforms the existing techniques in providing more style-consistent textures while keeping the content fidelity.

{{</citation>}}


### (60/131) Spatial-Assistant Encoder-Decoder Network for Real Time Semantic Segmentation (Yalun Wang et al., 2023)

{{<citation>}}

Yalun Wang, Shidong Chen, Huicong Bian, Weixiao Li, Qin Lu. (2023)  
**Spatial-Assistant Encoder-Decoder Network for Real Time Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.10519v1)  

---


**ABSTRACT**  
Semantic segmentation is an essential technology for self-driving cars to comprehend their surroundings. Currently, real-time semantic segmentation networks commonly employ either encoder-decoder architecture or two-pathway architecture. Generally speaking, encoder-decoder models tend to be quicker,whereas two-pathway models exhibit higher accuracy. To leverage both strengths, we present the Spatial-Assistant Encoder-Decoder Network (SANet) to fuse the two architectures. In the overall architecture, we uphold the encoder-decoder design while maintaining the feature maps in the middle section of the encoder and utilizing atrous convolution branches for same-resolution feature extraction. Toward the end of the encoder, we integrate the asymmetric pooling pyramid pooling module (APPPM) to optimize the semantic extraction of the feature maps. This module incorporates asymmetric pooling layers that extract features at multiple resolutions. In the decoder, we present a hybrid attention module, SAD, that integrates horizontal and vertical attention to facilitate the combination of various branches. To ascertain the effectiveness of our approach, our SANet model achieved competitive results on the real-time CamVid and cityscape datasets. By employing a single 2080Ti GPU, SANet achieved a 78.4 % mIOU at 65.1 FPS on the Cityscape test dataset and 78.8 % mIOU at 147 FPS on the CamVid test dataset. The training code and model for SANet are available at https://github.com/CuZaoo/SANet-main

{{</citation>}}


### (61/131) RECALL+: Adversarial Web-based Replay for Continual Learning in Semantic Segmentation (Chang Liu et al., 2023)

{{<citation>}}

Chang Liu, Giulia Rizzoli, Francesco Barbato, Umberto Michieli, Yi Niu, Pietro Zanuttigh. (2023)  
**RECALL+: Adversarial Web-based Replay for Continual Learning in Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.10479v1)  

---


**ABSTRACT**  
Catastrophic forgetting of previous knowledge is a critical issue in continual learning typically handled through various regularization strategies. However, existing methods struggle especially when several incremental steps are performed. In this paper, we extend our previous approach (RECALL) and tackle forgetting by exploiting unsupervised web-crawled data to retrieve examples of old classes from online databases. Differently from the original approach that did not perform any evaluation of the web data, here we introduce two novel approaches based on adversarial learning and adaptive thresholding to select from web data only samples strongly resembling the statistics of the no longer available training ones. Furthermore, we improved the pseudo-labeling scheme to achieve a more accurate labeling of web data that also consider classes being learned in the current step. Experimental results show that this enhanced approach achieves remarkable results, especially when multiple incremental learning steps are performed.

{{</citation>}}


### (62/131) AutoDiffusion: Training-Free Optimization of Time Steps and Architectures for Automated Diffusion Model Acceleration (Lijiang Li et al., 2023)

{{<citation>}}

Lijiang Li, Huixia Li, Xiawu Zheng, Jie Wu, Xuefeng Xiao, Rui Wang, Min Zheng, Xin Pan, Fei Chao, Rongrong Ji. (2023)  
**AutoDiffusion: Training-Free Optimization of Time Steps and Architectures for Automated Diffusion Model Acceleration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.10438v1)  

---


**ABSTRACT**  
Diffusion models are emerging expressive generative models, in which a large number of time steps (inference steps) are required for a single image generation. To accelerate such tedious process, reducing steps uniformly is considered as an undisputed principle of diffusion models. We consider that such a uniform assumption is not the optimal solution in practice; i.e., we can find different optimal time steps for different models. Therefore, we propose to search the optimal time steps sequence and compressed model architecture in a unified framework to achieve effective image generation for diffusion models without any further training. Specifically, we first design a unified search space that consists of all possible time steps and various architectures. Then, a two stage evolutionary algorithm is introduced to find the optimal solution in the designed search space. To further accelerate the search process, we employ FID score between generated and real samples to estimate the performance of the sampled examples. As a result, the proposed method is (i).training-free, obtaining the optimal time steps and model architecture without any training process; (ii). orthogonal to most advanced diffusion samplers and can be integrated to gain better sample quality. (iii). generalized, where the searched time steps and architectures can be directly applied on different diffusion models with the same guidance scale. Experimental results show that our method achieves excellent performance by using only a few time steps, e.g. 17.86 FID score on ImageNet 64 $\times$ 64 with only four steps, compared to 138.66 with DDIM.

{{</citation>}}


### (63/131) Sample-adaptive Augmentation for Point Cloud Recognition Against Real-world Corruptions (Jie Wang et al., 2023)

{{<citation>}}

Jie Wang, Lihe Ding, Tingfa Xu, Shaocong Dong, Xinli Xu, Long Bai, Jianan Li. (2023)  
**Sample-adaptive Augmentation for Point Cloud Recognition Against Real-world Corruptions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.10431v1)  

---


**ABSTRACT**  
Robust 3D perception under corruption has become an essential task for the realm of 3D vision. While current data augmentation techniques usually perform random transformations on all point cloud objects in an offline way and ignore the structure of the samples, resulting in over-or-under enhancement. In this work, we propose an alternative to make sample-adaptive transformations based on the structure of the sample to cope with potential corruption via an auto-augmentation framework, named as AdaptPoint. Specially, we leverage a imitator, consisting of a Deformation Controller and a Mask Controller, respectively in charge of predicting deformation parameters and producing a per-point mask, based on the intrinsic structural information of the input point cloud, and then conduct corruption simulations on top. Then a discriminator is utilized to prevent the generation of excessive corruption that deviates from the original data distribution. In addition, a perception-guidance feedback mechanism is incorporated to guide the generation of samples with appropriate difficulty level. Furthermore, to address the paucity of real-world corrupted point cloud, we also introduce a new dataset ScanObjectNN-C, that exhibits greater similarity to actual data in real-world environments, especially when contrasted with preceding CAD datasets. Experiments show that our method achieves state-of-the-art results on multiple corruption benchmarks, including ModelNet-C, our ScanObjectNN-C, and ShapeNet-C.

{{</citation>}}


### (64/131) Pointing out Human Answer Mistakes in a Goal-Oriented Visual Dialogue (Ryosuke Oshima et al., 2023)

{{<citation>}}

Ryosuke Oshima, Seitaro Shinagawa, Hideki Tsunashima, Qi Feng, Shigeo Morishima. (2023)  
**Pointing out Human Answer Mistakes in a Goal-Oriented Visual Dialogue**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Dialog, Dialogue, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2309.10375v1)  

---


**ABSTRACT**  
Effective communication between humans and intelligent agents has promising applications for solving complex problems. One such approach is visual dialogue, which leverages multimodal context to assist humans. However, real-world scenarios occasionally involve human mistakes, which can cause intelligent agents to fail. While most prior research assumes perfect answers from human interlocutors, we focus on a setting where the agent points out unintentional mistakes for the interlocutor to review, better reflecting real-world situations. In this paper, we show that human answer mistakes depend on question type and QA turn in the visual dialogue by analyzing a previously unused data collection of human mistakes. We demonstrate the effectiveness of those factors for the model's accuracy in a pointing-human-mistake task through experiments using a simple MLP model and a Visual Language Model.

{{</citation>}}


### (65/131) Improving CLIP Robustness with Knowledge Distillation and Self-Training (Clement Laroudie et al., 2023)

{{<citation>}}

Clement Laroudie, Andrei Bursuc, Mai Lan Ha, Gianni Franchi. (2023)  
**Improving CLIP Robustness with Knowledge Distillation and Self-Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.10361v1)  

---


**ABSTRACT**  
This paper examines the robustness of a multi-modal computer vision model, CLIP (Contrastive Language-Image Pretraining), in the context of unsupervised learning. The main objective is twofold: first, to evaluate the robustness of CLIP, and second, to explore strategies for augmenting its robustness. To achieve this, we introduce a novel approach named LP-CLIP. This technique involves the distillation of CLIP features through the incorporation of a linear probing layer positioned atop its encoding structure. This newly added layer is trained utilizing pseudo-labels produced by CLIP, coupled with a self-training strategy. The LP-CLIP technique offers a promising approach to enhance the robustness of CLIP without the need for annotations. By leveraging a simple linear probing layer, we aim to improve the model's ability to withstand various uncertainties and challenges commonly encountered in real-world scenarios. Importantly, our approach does not rely on annotated data, which makes it particularly valuable in situations where labeled data might be scarce or costly to obtain. Our proposed approach increases the robustness of CLIP with SOTA results compared to supervised technique on various datasets.

{{</citation>}}


### (66/131) RoadFormer: Duplex Transformer for RGB-Normal Semantic Road Scene Parsing (Jiahang Li et al., 2023)

{{<citation>}}

Jiahang Li, Yikang Zhang, Peng Yun, Guangliang Zhou, Qijun Chen, Rui Fan. (2023)  
**RoadFormer: Duplex Transformer for RGB-Normal Semantic Road Scene Parsing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.10356v1)  

---


**ABSTRACT**  
The recent advancements in deep convolutional neural networks have shown significant promise in the domain of road scene parsing. Nevertheless, the existing works focus primarily on freespace detection, with little attention given to hazardous road defects that could compromise both driving safety and comfort. In this paper, we introduce RoadFormer, a novel Transformer-based data-fusion network developed for road scene parsing. RoadFormer utilizes a duplex encoder architecture to extract heterogeneous features from both RGB images and surface normal information. The encoded features are subsequently fed into a novel heterogeneous feature synergy block for effective feature fusion and recalibration. The pixel decoder then learns multi-scale long-range dependencies from the fused and recalibrated heterogeneous features, which are subsequently processed by a Transformer decoder to produce the final semantic prediction. Additionally, we release SYN-UDTIRI, the first large-scale road scene parsing dataset that contains over 10,407 RGB images, dense depth images, and the corresponding pixel-level annotations for both freespace and road defects of different shapes and sizes. Extensive experimental evaluations conducted on our SYN-UDTIRI dataset, as well as on three public datasets, including KITTI road, CityScapes, and ORFD, demonstrate that RoadFormer outperforms all other state-of-the-art networks for road scene parsing. Specifically, RoadFormer ranks first on the KITTI road benchmark. Our source code, created dataset, and demo video are publicly available at mias.group/RoadFormer.

{{</citation>}}


### (67/131) Transferable Adversarial Attack on Image Tampering Localization (Yuqi Wang et al., 2023)

{{<citation>}}

Yuqi Wang, Gang Cao, Zijie Lou, Haochen Zhu. (2023)  
**Transferable Adversarial Attack on Image Tampering Localization**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2309.10243v1)  

---


**ABSTRACT**  
It is significant to evaluate the security of existing digital image tampering localization algorithms in real-world applications. In this paper, we propose an adversarial attack scheme to reveal the reliability of such tampering localizers, which would be fooled and fail to predict altered regions correctly. Specifically, the adversarial examples based on optimization and gradient are implemented for white/black-box attacks. Correspondingly, the adversarial example is optimized via reverse gradient propagation, and the perturbation is added adaptively in the direction of gradient rising. The black-box attack is achieved by relying on the transferability of such adversarial examples to different localizers. Extensive evaluations verify that the proposed attack sharply reduces the localization accuracy while preserving high visual quality of the attacked images.

{{</citation>}}


### (68/131) Learning Point-wise Abstaining Penalty for Point Cloud Anomaly Detection (Shaocong Xu et al., 2023)

{{<citation>}}

Shaocong Xu, Pengfei Li, Xinyu Liu, Qianpu Sun, Yang Li, Shihui Guo, Zhen Wang, Bo Jiang, Rui Wang, Kehua Sheng, Bo Zhang, Hao Zhao. (2023)  
**Learning Point-wise Abstaining Penalty for Point Cloud Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2309.10230v2)  

---


**ABSTRACT**  
LiDAR-based semantic scene understanding is an important module in the modern autonomous driving perception stack. However, identifying Out-Of-Distribution (OOD) points in a LiDAR point cloud is challenging as point clouds lack semantically rich features when compared with RGB images. We revisit this problem from the perspective of selective classification, which introduces a selective function into the standard closed-set classification setup. Our solution is built upon the basic idea of abstaining from choosing any known categories but learns a point-wise abstaining penalty with a marginbased loss. Synthesizing outliers to approximate unlimited OOD samples is also critical to this idea, so we propose a strong synthesis pipeline that generates outliers originated from various factors: unrealistic object categories, sampling patterns and sizes. We demonstrate that learning different abstaining penalties, apart from point-wise penalty, for different types of (synthesized) outliers can further improve the performance. We benchmark our method on SemanticKITTI and nuScenes and achieve state-of-the-art results. Risk-coverage analysis further reveals intrinsic properties of different methods. Codes and models will be publicly available.

{{</citation>}}


### (69/131) Multi-level feature fusion network combining attention mechanisms for polyp segmentation (Junzhuo Liu et al., 2023)

{{<citation>}}

Junzhuo Liu, Qiaosong Chen, Ye Zhang, Zhixiang Wang, Deng Xin, Jin Wang. (2023)  
**Multi-level feature fusion network combining attention mechanisms for polyp segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Clinical  
[Paper Link](http://arxiv.org/abs/2309.10219v1)  

---


**ABSTRACT**  
Clinically, automated polyp segmentation techniques have the potential to significantly improve the efficiency and accuracy of medical diagnosis, thereby reducing the risk of colorectal cancer in patients. Unfortunately, existing methods suffer from two significant weaknesses that can impact the accuracy of segmentation. Firstly, features extracted by encoders are not adequately filtered and utilized. Secondly, semantic conflicts and information redundancy caused by feature fusion are not attended to. To overcome these limitations, we propose a novel approach for polyp segmentation, named MLFF-Net, which leverages multi-level feature fusion and attention mechanisms. Specifically, MLFF-Net comprises three modules: Multi-scale Attention Module (MAM), High-level Feature Enhancement Module (HFEM), and Global Attention Module (GAM). Among these, MAM is used to extract multi-scale information and polyp details from the shallow output of the encoder. In HFEM, the deep features of the encoders complement each other by aggregation. Meanwhile, the attention mechanism redistributes the weight of the aggregated features, weakening the conflicting redundant parts and highlighting the information useful to the task. GAM combines features from the encoder and decoder features, as well as computes global dependencies to prevent receptive field locality. Experimental results on five public datasets show that the proposed method not only can segment multiple types of polyps but also has advantages over current state-of-the-art methods in both accuracy and generalization ability.

{{</citation>}}


### (70/131) An Empirical Study of Attention Networks for Semantic Segmentation (Hao Guo et al., 2023)

{{<citation>}}

Hao Guo, Hongbiao Si, Guilin Jiang, Wei Zhang, Zhiyan Liu, Xuanyi Zhu, Xulong Zhang, Yang Liu. (2023)  
**An Empirical Study of Attention Networks for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.10217v1)  

---


**ABSTRACT**  
Semantic segmentation is a vital problem in computer vision. Recently, a common solution to semantic segmentation is the end-to-end convolution neural network, which is much more accurate than traditional methods.Recently, the decoders based on attention achieve state-of-the-art (SOTA) performance on various datasets. But these networks always are compared with the mIoU of previous SOTA networks to prove their superiority and ignore their characteristics without considering the computation complexity and precision in various categories, which is essential for engineering applications. Besides, the methods to analyze the FLOPs and memory are not consistent between different networks, which makes the comparison hard to be utilized. What's more, various methods utilize attention in semantic segmentation, but the conclusion of these methods is lacking. This paper first conducts experiments to analyze their computation complexity and compare their performance. Then it summarizes suitable scenes for these networks and concludes key points that should be concerned when constructing an attention network. Last it points out some future directions of the attention network.

{{</citation>}}


## cs.RO (4)



### (71/131) Open-Vocabulary Affordance Detection using Knowledge Distillation and Text-Point Correlation (Tuan Van Vo et al., 2023)

{{<citation>}}

Tuan Van Vo, Minh Nhat Vu, Baoru Huang, Toan Nguyen, Ngan Le, Thieu Vo, Anh Nguyen. (2023)  
**Open-Vocabulary Affordance Detection using Knowledge Distillation and Text-Point Correlation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.10932v1)  

---


**ABSTRACT**  
Affordance detection presents intricate challenges and has a wide range of robotic applications. Previous works have faced limitations such as the complexities of 3D object shapes, the wide range of potential affordances on real-world objects, and the lack of open-vocabulary support for affordance understanding. In this paper, we introduce a new open-vocabulary affordance detection method in 3D point clouds, leveraging knowledge distillation and text-point correlation. Our approach employs pre-trained 3D models through knowledge distillation to enhance feature extraction and semantic understanding in 3D point clouds. We further introduce a new text-point correlation method to learn the semantic links between point cloud features and open-vocabulary labels. The intensive experiments show that our approach outperforms previous works and adapts to new affordance labels and unseen objects. Notably, our method achieves the improvement of 7.96% mIOU score compared to the baselines. Furthermore, it offers real-time inference which is well-suitable for robotic manipulation applications.

{{</citation>}}


### (72/131) Augmenting Tactile Simulators with Real-like and Zero-Shot Capabilities (Osher Azulay et al., 2023)

{{<citation>}}

Osher Azulay, Alon Mizrahi, Nimrod Curtis, Avishai Sintov. (2023)  
**Augmenting Tactile Simulators with Real-like and Zero-Shot Capabilities**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.10409v1)  

---


**ABSTRACT**  
Simulating tactile perception could potentially leverage the learning capabilities of robotic systems in manipulation tasks. However, the reality gap of simulators for high-resolution tactile sensors remains large. Models trained on simulated data often fail in zero-shot inference and require fine-tuning with real data. In addition, work on high-resolution sensors commonly focus on ones with flat surfaces while 3D round sensors are essential for dexterous manipulation. In this paper, we propose a bi-directional Generative Adversarial Network (GAN) termed SightGAN. SightGAN relies on the early CycleGAN while including two additional loss components aimed to accurately reconstruct background and contact patterns including small contact traces. The proposed SightGAN learns real-to-sim and sim-to-real processes over difference images. It is shown to generate real-like synthetic images while maintaining accurate contact positioning. The generated images can be used to train zero-shot models for newly fabricated sensors. Consequently, the resulted sim-to-real generator could be built on top of the tactile simulator to provide a real-world framework. Potentially, the framework can be used to train, for instance, reinforcement learning policies of manipulation tasks. The proposed model is verified in extensive experiments with test data collected from real sensors and also shown to maintain embedded force information within the tactile images.

{{</citation>}}


### (73/131) Crowd-Aware Multi-Agent Pathfinding With Boosted Curriculum Reinforcement Learning (Phu Pham et al., 2023)

{{<citation>}}

Phu Pham, Aniket Bera. (2023)  
**Crowd-Aware Multi-Agent Pathfinding With Boosted Curriculum Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-MA, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10275v1)  

---


**ABSTRACT**  
Multi-Agent Path Finding (MAPF) in crowded environments presents a challenging problem in motion planning, aiming to find collision-free paths for all agents in the system. MAPF finds a wide range of applications in various domains, including aerial swarms, autonomous warehouse robotics, and self-driving vehicles. The current approaches for MAPF can be broadly categorized into two main categories: centralized and decentralized planning. Centralized planning suffers from the curse of dimensionality and thus does not scale well in large and complex environments. On the other hand, decentralized planning enables agents to engage in real-time path planning within a partially observable environment, demonstrating implicit coordination. However, they suffer from slow convergence and performance degradation in dense environments. In this paper, we introduce CRAMP, a crowd-aware decentralized approach to address this problem by leveraging reinforcement learning guided by a boosted curriculum-based training strategy. We test CRAMP on simulated environments and demonstrate that our method outperforms the state-of-the-art decentralized methods for MAPF on various metrics. CRAMP improves the solution quality up to 58% measured in makespan and collision count, and up to 5% in success rate in comparison to previous methods.

{{</citation>}}


### (74/131) Memory-based Controllers for Efficient Data-driven Control of Soft Robots (Yuzhe Wu et al., 2023)

{{<citation>}}

Yuzhe Wu, Ehsan Nekouei. (2023)  
**Memory-based Controllers for Efficient Data-driven Control of Soft Robots**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.10273v1)  

---


**ABSTRACT**  
Controller design for soft robots is challenging due to nonlinear deformation and high degrees of freedom of flexible material. The data-driven approach is a promising solution to the controller design problem for soft robots. However, the existing data-driven controller design methods for soft robots suffer from two drawbacks: (i) they require excessively long training time, and (ii) they may result in potentially inefficient controllers. This paper addresses these issues by developing two memory-based controllers for soft robots that can be trained in a data-driven fashion: the finite memory controller (FMC) approach and the long short-term memory (LSTM) based approach. An FMC stores the tracking errors at different time instances and computes the actuation signal according to a weighted sum of the stored tracking errors. We develop three reinforcement learning algorithms for computing the optimal weights of an FMC using the Q-learning, soft actor-critic, and deterministic policy gradient (DDPG) methods. An LSTM-based controller is composed of an LSTM network where the inputs of the network are the robot's desired configuration and current configuration. The LSTM network computes the required actuation signal for the soft robot to follow the desired configuration. We study the performance of the proposed approaches in controlling a soft finger where, as benchmarks, we use the existing reinforcement learning (RL) based controllers and proportional-integral-derivative (PID) controllers. Our numerical results show that the training time of the proposed memory-based controllers is significantly shorter than that of the classical RL-based controllers. Moreover, the proposed controllers achieve a smaller tracking error compared with the classical RL algorithms and the PID controller.

{{</citation>}}


## eess.AS (4)



### (75/131) Discrete Audio Representation as an Alternative to Mel-Spectrograms for Speaker and Speech Recognition (Krishna C. Puvvada et al., 2023)

{{<citation>}}

Krishna C. Puvvada, Nithin Rao Koluguri, Kunal Dhawan, Jagadeesh Balam, Boris Ginsburg. (2023)  
**Discrete Audio Representation as an Alternative to Mel-Spectrograms for Speaker and Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Quantization, Speaker Verification, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.10922v1)  

---


**ABSTRACT**  
Discrete audio representation, aka audio tokenization, has seen renewed interest driven by its potential to facilitate the application of text language modeling approaches in audio domain. To this end, various compression and representation-learning based tokenization schemes have been proposed. However, there is limited investigation into the performance of compression-based audio tokens compared to well-established mel-spectrogram features across various speaker and speech related tasks. In this paper, we evaluate compression based audio tokens on three tasks: Speaker Verification, Diarization and (Multi-lingual) Speech Recognition. Our findings indicate that (i) the models trained on audio tokens perform competitively, on average within $1\%$ of mel-spectrogram features for all the tasks considered, and do not surpass them yet. (ii) these models exhibit robustness for out-of-domain narrowband data, particularly in speaker tasks. (iii) audio tokens allow for compression to 20x compared to mel-spectrogram features with minimal loss of performance in speech and speaker related tasks, which is crucial for low bit-rate applications, and (iv) the examined Residual Vector Quantization (RVQ) based audio tokenizer exhibits a low-pass frequency response characteristic, offering a plausible explanation for the observed results, and providing insight for future tokenizer designs.

{{</citation>}}


### (76/131) End-to-End Speech Recognition Contextualization with Large Language Models (Egor Lakomkin et al., 2023)

{{<citation>}}

Egor Lakomkin, Chunyang Wu, Yassir Fathullah, Ozlem Kalinli, Michael L. Seltzer, Christian Fuegen. (2023)  
**End-to-End Speech Recognition Contextualization with Large Language Models**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CL, cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Language Model, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.10917v1)  

---


**ABSTRACT**  
In recent years, Large Language Models (LLMs) have garnered significant attention from the research community due to their exceptional performance and generalization capabilities. In this paper, we introduce a novel method for contextualizing speech recognition models incorporating LLMs. Our approach casts speech recognition as a mixed-modal language modeling task based on a pretrained LLM. We provide audio features, along with optional text tokens for context, to train the system to complete transcriptions in a decoder-only fashion. As a result, the system is implicitly incentivized to learn how to leverage unstructured contextual information during training. Our empirical results demonstrate a significant improvement in performance, with a 6% WER reduction when additional textual context is provided. Moreover, we find that our method performs competitively and improve by 7.5% WER overall and 17% WER on rare words against a baseline contextualized RNN-T system that has been trained on more than twenty five times larger speech dataset. Overall, we demonstrate that by only adding a handful number of trainable parameters via adapters, we can unlock contextualized speech recognition capability for the pretrained LLM while keeping the same text-only input functionality.

{{</citation>}}


### (77/131) FoleyGen: Visually-Guided Audio Generation (Xinhao Mei et al., 2023)

{{<citation>}}

Xinhao Mei, Varun Nagaraja, Gael Le Lan, Zhaoheng Ni, Ernie Chang, Yangyang Shi, Vikas Chandra. (2023)  
**FoleyGen: Visually-Guided Audio Generation**  

---
Primary Category: eess.AS  
Categories: cs-MM, cs-SD, eess-AS, eess.AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.10537v1)  

---


**ABSTRACT**  
Recent advancements in audio generation have been spurred by the evolution of large-scale deep learning models and expansive datasets. However, the task of video-to-audio (V2A) generation continues to be a challenge, principally because of the intricate relationship between the high-dimensional visual and auditory data, and the challenges associated with temporal synchronization. In this study, we introduce FoleyGen, an open-domain V2A generation system built on a language modeling paradigm. FoleyGen leverages an off-the-shelf neural audio codec for bidirectional conversion between waveforms and discrete tokens. The generation of audio tokens is facilitated by a single Transformer model, which is conditioned on visual features extracted from a visual encoder. A prevalent problem in V2A generation is the misalignment of generated audio with the visible actions in the video. To address this, we explore three novel visual attention mechanisms. We further undertake an exhaustive evaluation of multiple visual encoders, each pretrained on either single-modal or multi-modal tasks. The experimental results on VGGSound dataset show that our proposed FoleyGen outperforms previous systems across all objective metrics and human evaluations.

{{</citation>}}


### (78/131) Harnessing the Zero-Shot Power of Instruction-Tuned Large Language Model in End-to-End Speech Recognition (Yosuke Higuchi et al., 2023)

{{<citation>}}

Yosuke Higuchi, Tetsuji Ogawa, Tetsunori Kobayashi. (2023)  
**Harnessing the Zero-Shot Power of Instruction-Tuned Large Language Model in End-to-End Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Language Model, Speech Recognition, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.10524v1)  

---


**ABSTRACT**  
We present a novel integration of an instruction-tuned large language model (LLM) and end-to-end automatic speech recognition (ASR). Modern LLMs can perform a wide range of linguistic tasks within zero-shot learning when provided with a precise instruction or a prompt to guide the text generation process towards the desired task. We explore using this zero-shot capability of LLMs to extract linguistic information that can contribute to improving ASR performance. Specifically, we direct an LLM to correct grammatical errors in an ASR hypothesis and harness the embedded linguistic knowledge to conduct end-to-end ASR. The proposed model is built on the hybrid connectionist temporal classification (CTC) and attention architecture, where an instruction-tuned LLM (i.e., Llama2) is employed as a front-end of the decoder. An ASR hypothesis, subject to correction, is obtained from the encoder via CTC decoding, which is then fed into the LLM along with an instruction. The decoder subsequently takes as input the LLM embeddings to perform sequence generation, incorporating acoustic information from the encoder output. Experimental results and analyses demonstrate that the proposed integration yields promising performance improvements, and our approach largely benefits from LLM-based rescoring.

{{</citation>}}


## cs.LG (13)



### (79/131) What Learned Representations and Influence Functions Can Tell Us About Adversarial Examples (Shakila Mahjabin Tonni et al., 2023)

{{<citation>}}

Shakila Mahjabin Tonni, Mark Dras. (2023)  
**What Learned Representations and Influence Functions Can Tell Us About Adversarial Examples**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.10916v2)  

---


**ABSTRACT**  
Adversarial examples, deliberately crafted using small perturbations to fool deep neural networks, were first studied in image processing and more recently in NLP. While approaches to detecting adversarial examples in NLP have largely relied on search over input perturbations, image processing has seen a range of techniques that aim to characterise adversarial subspaces over the learned representations.   In this paper, we adapt two such approaches to NLP, one based on nearest neighbors and influence functions and one on Mahalanobis distances. The former in particular produces a state-of-the-art detector when compared against several strong baselines; moreover, the novel use of influence functions provides insight into how the nature of adversarial example subspaces in NLP relate to those in image processing, and also how they differ depending on the kind of NLP task.

{{</citation>}}


### (80/131) DeepliteRT: Computer Vision at the Edge (Saad Ashfaq et al., 2023)

{{<citation>}}

Saad Ashfaq, Alexander Hoffman, Saptarshi Mitra, Sudhakar Sah, MohammadHossein AskariHemmat, Ehsan Saboori. (2023)  
**DeepliteRT: Computer Vision at the Edge**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2309.10878v1)  

---


**ABSTRACT**  
The proliferation of edge devices has unlocked unprecedented opportunities for deep learning model deployment in computer vision applications. However, these complex models require considerable power, memory and compute resources that are typically not available on edge platforms. Ultra low-bit quantization presents an attractive solution to this problem by scaling down the model weights and activations from 32-bit to less than 8-bit. We implement highly optimized ultra low-bit convolution operators for ARM-based targets that outperform existing methods by up to 4.34x. Our operator is implemented within Deeplite Runtime (DeepliteRT), an end-to-end solution for the compilation, tuning, and inference of ultra low-bit models on ARM devices. Compiler passes in DeepliteRT automatically convert a fake-quantized model in full precision to a compact ultra low-bit representation, easing the process of quantized model deployment on commodity hardware. We analyze the performance of DeepliteRT on classification and detection models against optimized 32-bit floating-point, 8-bit integer, and 2-bit baselines, achieving significant speedups of up to 2.20x, 2.33x and 2.17x, respectively.

{{</citation>}}


### (81/131) AI Foundation Models for Weather and Climate: Applications, Design, and Implementation (S. Karthik Mukkavilli et al., 2023)

{{<citation>}}

S. Karthik Mukkavilli, Daniel Salles Civitarese, Johannes Schmude, Johannes Jakubik, Anne Jones, Nam Nguyen, Christopher Phillips, Sujit Roy, Shraddha Singh, Campbell Watson, Raghu Ganti, Hendrik Hamann, Udaysankar Nair, Rahul Ramachandran, Kommy Weldemariam. (2023)  
**AI Foundation Models for Weather and Climate: Applications, Design, and Implementation**  

---
Primary Category: cs.LG  
Categories: 68T07 (Primary), 68T01, 86A08, I-2-0; I-4-0; J-2-5, cs-AI, cs-LG, cs.LG, physics-ao-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10808v2)  

---


**ABSTRACT**  
Machine learning and deep learning methods have been widely explored in understanding the chaotic behavior of the atmosphere and furthering weather forecasting. There has been increasing interest from technology companies, government institutions, and meteorological agencies in building digital twins of the Earth. Recent approaches using transformers, physics-informed machine learning, and graph neural networks have demonstrated state-of-the-art performance on relatively narrow spatiotemporal scales and specific tasks. With the recent success of generative artificial intelligence (AI) using pre-trained transformers for language modeling and vision with prompt engineering and fine-tuning, we are now moving towards generalizable AI. In particular, we are witnessing the rise of AI foundation models that can perform competitively on multiple domain-specific downstream tasks. Despite this progress, we are still in the nascent stages of a generalizable AI model for global Earth system models, regional climate models, and mesoscale weather models. Here, we review current state-of-the-art AI approaches, primarily from transformer and operator learning literature in the context of meteorology. We provide our perspective on criteria for success towards a family of foundation models for nowcasting and forecasting weather and climate predictions. We also discuss how such models can perform competitively on downstream tasks such as downscaling (super-resolution), identifying conditions conducive to the occurrence of wildfires, and predicting consequential meteorological phenomena across various spatiotemporal scales such as hurricanes and atmospheric rivers. In particular, we examine current AI methodologies and contend they have matured enough to design and implement a weather foundation model.

{{</citation>}}


### (82/131) GPT4AIGChip: Towards Next-Generation AI Accelerator Design Automation via Large Language Models (Yonggan Fu et al., 2023)

{{<citation>}}

Yonggan Fu, Yongan Zhang, Zhongzhi Yu, Sixu Li, Zhifan Ye, Chaojian Li, Cheng Wan, Yingyan Lin. (2023)  
**GPT4AIGChip: Towards Next-Generation AI Accelerator Design Automation via Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs.LG  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10730v1)  

---


**ABSTRACT**  
The remarkable capabilities and intricate nature of Artificial Intelligence (AI) have dramatically escalated the imperative for specialized AI accelerators. Nonetheless, designing these accelerators for various AI workloads remains both labor- and time-intensive. While existing design exploration and automation tools can partially alleviate the need for extensive human involvement, they still demand substantial hardware expertise, posing a barrier to non-experts and stifling AI accelerator development. Motivated by the astonishing potential of large language models (LLMs) for generating high-quality content in response to human language instructions, we embark on this work to examine the possibility of harnessing LLMs to automate AI accelerator design. Through this endeavor, we develop GPT4AIGChip, a framework intended to democratize AI accelerator design by leveraging human natural languages instead of domain-specific languages. Specifically, we first perform an in-depth investigation into LLMs' limitations and capabilities for AI accelerator design, thus aiding our understanding of our current position and garnering insights into LLM-powered automated AI accelerator design. Furthermore, drawing inspiration from the above insights, we develop a framework called GPT4AIGChip, which features an automated demo-augmented prompt-generation pipeline utilizing in-context learning to guide LLMs towards creating high-quality AI accelerator design. To our knowledge, this work is the first to demonstrate an effective pipeline for LLM-powered automated AI accelerator generation. Accordingly, we anticipate that our insights and framework can serve as a catalyst for innovations in next-generation LLM-powered design automation tools.

{{</citation>}}


### (83/131) Language Modeling Is Compression (Grégoire Delétang et al., 2023)

{{<citation>}}

Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, Marcus Hutter, Joel Veness. (2023)  
**Language Modeling Is Compression**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-IT, cs-LG, cs.LG, math-IT  
Keywords: ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10668v1)  

---


**ABSTRACT**  
It has long been established that predictive models can be transformed into lossless compressors and vice versa. Incidentally, in recent years, the machine learning community has focused on training increasingly large and powerful self-supervised (language) models. Since these large language models exhibit impressive predictive capabilities, they are well-positioned to be strong compressors. In this work, we advocate for viewing the prediction problem through the lens of compression and evaluate the compression capabilities of large (foundation) models. We show that large language models are powerful general-purpose predictors and that the compression viewpoint provides novel insights into scaling laws, tokenization, and in-context learning. For example, Chinchilla 70B, while trained primarily on text, compresses ImageNet patches to 43.4% and LibriSpeech samples to 16.4% of their raw size, beating domain-specific compressors like PNG (58.5%) or FLAC (30.3%), respectively. Finally, we show that the prediction-compression equivalence allows us to use any compressor (like gzip) to build a conditional generative model.

{{</citation>}}


### (84/131) PDRL: Multi-Agent based Reinforcement Learning for Predictive Monitoring (Thanveer Shaik et al., 2023)

{{<citation>}}

Thanveer Shaik, Xiaohui Tao, Lin Li, Haoran Xie, U R Acharya, Raj Gururajan, Xujuan Zhou. (2023)  
**PDRL: Multi-Agent based Reinforcement Learning for Predictive Monitoring**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10576v2)  

---


**ABSTRACT**  
Reinforcement learning has been increasingly applied in monitoring applications because of its ability to learn from previous experiences and can make adaptive decisions. However, existing machine learning-based health monitoring applications are mostly supervised learning algorithms, trained on labels and they cannot make adaptive decisions in an uncertain complex environment. This study proposes a novel and generic system, predictive deep reinforcement learning (PDRL) with multiple RL agents in a time series forecasting environment. The proposed generic framework accommodates virtual Deep Q Network (DQN) agents to monitor predicted future states of a complex environment with a well-defined reward policy so that the agent learns existing knowledge while maximizing their rewards. In the evaluation process of the proposed framework, three DRL agents were deployed to monitor a subject's future heart rate, respiration, and temperature predicted using a BiLSTM model. With each iteration, the three agents were able to learn the associated patterns and their cumulative rewards gradually increased. It outperformed the baseline models for all three monitoring agents. The proposed PDRL framework is able to achieve state-of-the-art performance in the time series forecasting process. The proposed DRL agents and deep learning model in the PDRL framework are customized to implement the transfer learning in other forecasting applications like traffic and weather and monitor their states. The PDRL framework is able to learn the future states of the traffic and weather forecasting and the cumulative rewards are gradually increasing over each episode.

{{</citation>}}


### (85/131) A Neighbourhood-Aware Differential Privacy Mechanism for Static Word Embeddings (Danushka Bollegala et al., 2023)

{{<citation>}}

Danushka Bollegala, Shuichi Otake, Tomoya Machide, Ken-ichi Kawarabayashi. (2023)  
**A Neighbourhood-Aware Differential Privacy Mechanism for Static Word Embeddings**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Embedding, Word Embedding  
[Paper Link](http://arxiv.org/abs/2309.10551v1)  

---


**ABSTRACT**  
We propose a Neighbourhood-Aware Differential Privacy (NADP) mechanism considering the neighbourhood of a word in a pretrained static word embedding space to determine the minimal amount of noise required to guarantee a specified privacy level. We first construct a nearest neighbour graph over the words using their embeddings, and factorise it into a set of connected components (i.e. neighbourhoods). We then separately apply different levels of Gaussian noise to the words in each neighbourhood, determined by the set of words in that neighbourhood. Experiments show that our proposed NADP mechanism consistently outperforms multiple previously proposed DP mechanisms such as Laplacian, Gaussian, and Mahalanobis in multiple downstream tasks, while guaranteeing higher levels of privacy.

{{</citation>}}


### (86/131) Model Leeching: An Extraction Attack Targeting LLMs (Lewis Birch et al., 2023)

{{<citation>}}

Lewis Birch, William Hackett, Stefan Trawicki, Neeraj Suri, Peter Garraghan. (2023)  
**Model Leeching: An Extraction Attack Targeting LLMs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: ChatGPT, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10544v1)  

---


**ABSTRACT**  
Model Leeching is a novel extraction attack targeting Large Language Models (LLMs), capable of distilling task-specific knowledge from a target LLM into a reduced parameter model. We demonstrate the effectiveness of our attack by extracting task capability from ChatGPT-3.5-Turbo, achieving 73% Exact Match (EM) similarity, and SQuAD EM and F1 accuracy scores of 75% and 87%, respectively for only $50 in API cost. We further demonstrate the feasibility of adversarial attack transferability from an extracted model extracted via Model Leeching to perform ML attack staging against a target LLM, resulting in an 11% increase to attack success rate when applied to ChatGPT-3.5-Turbo.

{{</citation>}}


### (87/131) Graph Neural Networks for Dynamic Modeling of Roller Bearing (Vinay Sharma et al., 2023)

{{<citation>}}

Vinay Sharma, Jens Ravesloot, Cees Taal, Olga Fink. (2023)  
**Graph Neural Networks for Dynamic Modeling of Roller Bearing**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs-NA, cs.LG, math-NA  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.10418v1)  

---


**ABSTRACT**  
In the presented work, we propose to apply the framework of graph neural networks (GNNs) to predict the dynamics of a rolling element bearing. This approach offers generalizability and interpretability, having the potential for scalable use in real-time operational digital twin systems for monitoring the health state of rotating machines. By representing the bearing's components as nodes in a graph, the GNN can effectively model the complex relationships and interactions among them. We utilize a dynamic spring-mass-damper model of a bearing to generate the training data for the GNN. In this model, discrete masses represent bearing components such as rolling elements, inner raceways, and outer raceways, while a Hertzian contact model is employed to calculate the forces between these components.   We evaluate the learning and generalization capabilities of the proposed GNN framework by testing different bearing configurations that deviate from the training configurations. Through this approach, we demonstrate the effectiveness of the GNN-based method in accurately predicting the dynamics of rolling element bearings, highlighting its potential for real-time health monitoring of rotating machinery.

{{</citation>}}


### (88/131) Unsupervised Learning via Network-Aware Embeddings (Anne Sophie Riis Damstrup et al., 2023)

{{<citation>}}

Anne Sophie Riis Damstrup, Sofie Tosti Madsen, Michele Coscia. (2023)  
**Unsupervised Learning via Network-Aware Embeddings**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG, physics-data-an  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.10408v1)  

---


**ABSTRACT**  
Data clustering, the task of grouping observations according to their similarity, is a key component of unsupervised learning -- with real world applications in diverse fields such as biology, medicine, and social science. Often in these fields the data comes with complex interdependencies between the dimensions of analysis, for instance the various characteristics and opinions people can have live on a complex social network. Current clustering methods are ill-suited to tackle this complexity: deep learning can approximate these dependencies, but not take their explicit map as the input of the analysis. In this paper, we aim at fixing this blind spot in the unsupervised learning literature. We can create network-aware embeddings by estimating the network distance between numeric node attributes via the generalized Euclidean distance. Differently from all methods in the literature that we know of, we do not cluster the nodes of the network, but rather its node attributes. In our experiments we show that having these network embeddings is always beneficial for the learning task; that our method scales to large networks; and that we can actually provide actionable insights in applications in a variety of fields such as marketing, economics, and political science. Our method is fully open source and data and code are available to reproduce all results in the paper.

{{</citation>}}


### (89/131) Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks (Hao Liu et al., 2023)

{{<citation>}}

Hao Liu, Jiarui Feng, Lecheng Kong, Dacheng Tao, Yixin Chen, Muhan Zhang. (2023)  
**Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning, Few-Shot, GNN, Graph Neural Network, Graph Neural Networks, Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.10376v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have become popular in Graph Representation Learning (GRL). One fundamental application is few-shot node classification. Most existing methods follow the meta learning paradigm, showing the ability of fast generalization to few-shot tasks. However, recent works indicate that graph contrastive learning combined with fine-tuning can significantly outperform meta learning methods. Despite the empirical success, there is limited understanding of the reasons behind it. In our study, we first identify two crucial advantages of contrastive learning compared to meta learning, including (1) the comprehensive utilization of graph nodes and (2) the power of graph augmentations. To integrate the strength of both contrastive learning and meta learning on the few-shot node classification tasks, we introduce a new paradigm: Contrastive Few-Shot Node Classification (COLA). Specifically, COLA employs graph augmentations to identify semantically similar nodes, which enables the construction of meta-tasks without the need for label information. Therefore, COLA can utilize all nodes to construct meta-tasks, further reducing the risk of overfitting. Through extensive experiments, we validate the essentiality of each component in our design and demonstrate that COLA achieves new state-of-the-art on all tasks.

{{</citation>}}


### (90/131) Explaining Agent Behavior with Large Language Models (Xijia Zhang et al., 2023)

{{<citation>}}

Xijia Zhang, Yue Guo, Simon Stepputtis, Katia Sycara, Joseph Campbell. (2023)  
**Explaining Agent Behavior with Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.10346v1)  

---


**ABSTRACT**  
Intelligent agents such as robots are increasingly deployed in real-world, safety-critical settings. It is vital that these agents are able to explain the reasoning behind their decisions to human counterparts, however, their behavior is often produced by uninterpretable models such as deep neural networks. We propose an approach to generate natural language explanations for an agent's behavior based only on observations of states and actions, agnostic to the underlying model representation. We show how a compact representation of the agent's behavior can be learned and used to produce plausible explanations with minimal hallucination while affording user interaction with a pre-trained large language model. Through user studies and empirical experiments, we show that our approach generates explanations as helpful as those generated by a human domain expert while enabling beneficial interactions such as clarification and counterfactual queries.

{{</citation>}}


### (91/131) FRAMU: Attention-based Machine Unlearning using Federated Reinforcement Learning (Thanveer Shaik et al., 2023)

{{<citation>}}

Thanveer Shaik, Xiaohui Tao, Lin Li, Haoran Xie, Taotao Cai, Xiaofeng Zhu, Qing Li. (2023)  
**FRAMU: Attention-based Machine Unlearning using Federated Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: Attention, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10283v1)  

---


**ABSTRACT**  
Machine Unlearning is an emerging field that addresses data privacy issues by enabling the removal of private or irrelevant data from the Machine Learning process. Challenges related to privacy and model efficiency arise from the use of outdated, private, and irrelevant data. These issues compromise both the accuracy and the computational efficiency of models in both Machine Learning and Unlearning. To mitigate these challenges, we introduce a novel framework, Attention-based Machine Unlearning using Federated Reinforcement Learning (FRAMU). This framework incorporates adaptive learning mechanisms, privacy preservation techniques, and optimization strategies, making it a well-rounded solution for handling various data sources, either single-modality or multi-modality, while maintaining accuracy and privacy. FRAMU's strength lies in its adaptability to fluctuating data landscapes, its ability to unlearn outdated, private, or irrelevant data, and its support for continual model evolution without compromising privacy. Our experiments, conducted on both single-modality and multi-modality datasets, revealed that FRAMU significantly outperformed baseline models. Additional assessments of convergence behavior and optimization strategies further validate the framework's utility in federated learning applications. Overall, FRAMU advances Machine Unlearning by offering a robust, privacy-preserving solution that optimizes model performance while also addressing key challenges in dynamic data environments.

{{</citation>}}


## cs.MA (1)



### (92/131) Multicopy Reinforcement Learning Agents (Alicia P. Wolfe et al., 2023)

{{<citation>}}

Alicia P. Wolfe, Oliver Diamond, Remi Feuerman, Magdalena Kisielinska, Brigitte Goeler-Slough, Victoria Manfredi. (2023)  
**Multicopy Reinforcement Learning Agents**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10908v1)  

---


**ABSTRACT**  
This paper examines a novel type of multi-agent problem, in which an agent makes multiple identical copies of itself in order to achieve a single agent task better or more efficiently. This strategy improves performance if the environment is noisy and the task is sometimes unachievable by a single agent copy. We propose a learning algorithm for this multicopy problem which takes advantage of the structure of the value function to efficiently learn how to balance the advantages and costs of adding additional copies.

{{</citation>}}


## cs.AI (15)



### (93/131) Artificial Intelligence-Enabled Intelligent Assistant for Personalized and Adaptive Learning in Higher Education (Ramteja Sajja et al., 2023)

{{<citation>}}

Ramteja Sajja, Yusuf Sermet, Muhammed Cikmaz, David Cwiertny, Ibrahim Demir. (2023)  
**Artificial Intelligence-Enabled Intelligent Assistant for Personalized and Adaptive Learning in Higher Education**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-IR, cs.AI  
Keywords: AI, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.10892v1)  

---


**ABSTRACT**  
This paper presents a novel framework, Artificial Intelligence-Enabled Intelligent Assistant (AIIA), for personalized and adaptive learning in higher education. The AIIA system leverages advanced AI and Natural Language Processing (NLP) techniques to create an interactive and engaging learning platform. This platform is engineered to reduce cognitive load on learners by providing easy access to information, facilitating knowledge assessment, and delivering personalized learning support tailored to individual needs and learning styles. The AIIA's capabilities include understanding and responding to student inquiries, generating quizzes and flashcards, and offering personalized learning pathways. The research findings have the potential to significantly impact the design, implementation, and evaluation of AI-enabled Virtual Teaching Assistants (VTAs) in higher education, informing the development of innovative educational tools that can enhance student learning outcomes, engagement, and satisfaction. The paper presents the methodology, system architecture, intelligent services, and integration with Learning Management Systems (LMSs) while discussing the challenges, limitations, and future directions for the development of AI-enabled intelligent assistants in education.

{{</citation>}}


### (94/131) Using AI Uncertainty Quantification to Improve Human Decision-Making (Laura R. Marusich et al., 2023)

{{<citation>}}

Laura R. Marusich, Jonathan Z. Bakdash, Yan Zhou, Murat Kantarcioglu. (2023)  
**Using AI Uncertainty Quantification to Improve Human Decision-Making**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10852v1)  

---


**ABSTRACT**  
AI Uncertainty Quantification (UQ) has the potential to improve human decision-making beyond AI predictions alone by providing additional useful probabilistic information to users. The majority of past research on AI and human decision-making has concentrated on model explainability and interpretability. We implemented instance-based UQ for three real datasets. To achieve this, we trained different AI models for classification for each dataset, and used random samples generated around the neighborhood of the given instance to create confidence intervals for UQ. The computed UQ was calibrated using a strictly proper scoring rule as a form of quality assurance for UQ. We then conducted two preregistered online behavioral experiments that compared objective human decision-making performance under different AI information conditions, including UQ. In Experiment 1, we compared decision-making for no AI (control), AI prediction alone, and AI prediction with a visualization of UQ. We found UQ significantly improved decision-making beyond the other two conditions. In Experiment 2, we focused on comparing different representations of UQ information: Point vs. distribution of uncertainty and visualization type (needle vs. dotplot). We did not find meaningful differences in decision-making performance among these different representations of UQ. Overall, our results indicate that human decision-making can be improved by providing UQ information along with AI predictions, and that this benefit generalizes across a variety of representations of UQ.

{{</citation>}}


### (95/131) Exploring the Influence of Information Entropy Change in Learning Systems (Xiaowei Yu et al., 2023)

{{<citation>}}

Xiaowei Yu, Yao Xue, Lu Zhang, Li Wang, Tianming Liu, Dajiang Zhu. (2023)  
**Exploring the Influence of Information Entropy Change in Learning Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.10625v1)  

---


**ABSTRACT**  
In this work, we explore the influence of entropy change in deep learning systems by adding noise to the inputs/latent features. The applications in this paper focus on deep learning tasks within computer vision, but the proposed theory can be further applied to other fields. Noise is conventionally viewed as a harmful perturbation in various deep learning architectures, such as convolutional neural networks (CNNs) and vision transformers (ViTs), as well as different learning tasks like image classification and transfer learning. However, this paper aims to rethink whether the conventional proposition always holds. We demonstrate that specific noise can boost the performance of various deep architectures under certain conditions. We theoretically prove the enhancement gained from positive noise by reducing the task complexity defined by information entropy and experimentally show the significant performance gain in large image datasets, such as the ImageNet. Herein, we use the information entropy to define the complexity of the task. We categorize the noise into two types, positive noise (PN) and harmful noise (HN), based on whether the noise can help reduce the complexity of the task. Extensive experiments of CNNs and ViTs have shown performance improvements by proactively injecting positive noise, where we achieved an unprecedented top 1 accuracy of over 95% on ImageNet. Both theoretical analysis and empirical evidence have confirmed that the presence of positive noise can benefit the learning process, while the traditionally perceived harmful noise indeed impairs deep learning models. The different roles of noise offer new explanations for deep models on specific tasks and provide a new paradigm for improving model performance. Moreover, it reminds us that we can influence the performance of learning systems via information entropy change.

{{</citation>}}


### (96/131) A Dynamic Linear Bias Incorporation Scheme for Nonnegative Latent Factor Analysis (Yurong Zhong et al., 2023)

{{<citation>}}

Yurong Zhong, Zhe Xie, Weiling Li, Xin Luo. (2023)  
**A Dynamic Linear Bias Incorporation Scheme for Nonnegative Latent Factor Analysis**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.10618v1)  

---


**ABSTRACT**  
High-Dimensional and Incomplete (HDI) data is commonly encountered in big data-related applications like social network services systems, which are concerning the limited interactions among numerous nodes. Knowledge acquisition from HDI data is a vital issue in the domain of data science due to their embedded rich patterns like node behaviors, where the fundamental task is to perform HDI data representation learning. Nonnegative Latent Factor Analysis (NLFA) models have proven to possess the superiority to address this issue, where a linear bias incorporation (LBI) scheme is important in present the training overshooting and fluctuation, as well as preventing the model from premature convergence. However, existing LBI schemes are all statistic ones where the linear biases are fixed, which significantly restricts the scalability of the resultant NLFA model and results in loss of representation learning ability to HDI data. Motivated by the above discoveries, this paper innovatively presents the dynamic linear bias incorporation (DLBI) scheme. It firstly extends the linear bias vectors into matrices, and then builds a binary weight matrix to switch the active/inactive states of the linear biases. The weight matrix's each entry switches between the binary states dynamically corresponding to the linear bias value variation, thereby establishing the dynamic linear biases for an NLFA model. Empirical studies on three HDI datasets from real applications demonstrate that the proposed DLBI-based NLFA model obtains higher representation accuracy several than state-of-the-art models do, as well as highly-competitive computational efficiency.

{{</citation>}}


### (97/131) Towards Generative Modeling of Urban Flow through Knowledge-enhanced Denoising Diffusion (Zhilun Zhou et al., 2023)

{{<citation>}}

Zhilun Zhou, Jingtao Ding, Yu Liu, Depeng Jin, Yong Li. (2023)  
**Towards Generative Modeling of Urban Flow through Knowledge-enhanced Denoising Diffusion**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10547v1)  

---


**ABSTRACT**  
Although generative AI has been successful in many areas, its ability to model geospatial data is still underexplored. Urban flow, a typical kind of geospatial data, is critical for a wide range of urban applications. Existing studies mostly focus on predictive modeling of urban flow that predicts the future flow based on historical flow data, which may be unavailable in data-sparse areas or newly planned regions. Some other studies aim to predict OD flow among regions but they fail to model dynamic changes of urban flow over time. In this work, we study a new problem of urban flow generation that generates dynamic urban flow for regions without historical flow data. To capture the effect of multiple factors on urban flow, such as region features and urban environment, we employ diffusion model to generate urban flow for regions under different conditions. We first construct an urban knowledge graph (UKG) to model the urban environment and relationships between regions, based on which we design a knowledge-enhanced spatio-temporal diffusion model (KSTDiff) to generate urban flow for each region. Specifically, to accurately generate urban flow for regions with different flow volumes, we design a novel diffusion process guided by a volume estimator, which is learnable and customized for each region. Moreover, we propose a knowledge-enhanced denoising network to capture the spatio-temporal dependencies of urban flow as well as the impact of urban environment in the denoising process. Extensive experiments on four real-world datasets validate the superiority of our model over state-of-the-art baselines in urban flow generation. Further in-depth studies demonstrate the utility of generated urban flow data and the ability of our model for long-term flow generation and urban flow prediction. Our code is released at: https://github.com/tsinghua-fib-lab/KSTDiff-Urban-flow-generation.

{{</citation>}}


### (98/131) A Cognitively-Inspired Neural Architecture for Visual Abstract Reasoning Using Contrastive Perceptual and Conceptual Processing (Yuan Yang et al., 2023)

{{<citation>}}

Yuan Yang, Deepayan Sanyal, James Ainooson, Joel Michelson, Effat Farhana, Maithilee Kunda. (2023)  
**A Cognitively-Inspired Neural Architecture for Visual Abstract Reasoning Using Contrastive Perceptual and Conceptual Processing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.10532v2)  

---


**ABSTRACT**  
We introduce a new neural architecture for solving visual abstract reasoning tasks inspired by human cognition, specifically by observations that human abstract reasoning often interleaves perceptual and conceptual processing as part of a flexible, iterative, and dynamic cognitive process. Inspired by this principle, our architecture models visual abstract reasoning as an iterative, self-contrasting learning process that pursues consistency between perceptual and conceptual processing of visual stimuli. We explain how this new Contrastive Perceptual-Conceptual Network (CPCNet) works using matrix reasoning problems in the style of the well-known Raven's Progressive Matrices intelligence test. Experiments on the machine learning dataset RAVEN show that CPCNet achieves higher accuracy than all previously published models while also using the weakest inductive bias. We also point out a substantial and previously unremarked class imbalance in the original RAVEN dataset, and we propose a new variant of RAVEN -- AB-RAVEN -- that is more balanced in terms of abstract concepts.

{{</citation>}}


### (99/131) Human-AI Interactions and Societal Pitfalls (Francisco Castro et al., 2023)

{{<citation>}}

Francisco Castro, Jian Gao, Sébastien Martin. (2023)  
**Human-AI Interactions and Societal Pitfalls**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI, econ-GN, q-fin-EC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10448v1)  

---


**ABSTRACT**  
When working with generative artificial intelligence (AI), users may see productivity gains, but the AI-generated content may not match their preferences exactly. To study this effect, we introduce a Bayesian framework in which heterogeneous users choose how much information to share with the AI, facing a trade-off between output fidelity and communication cost. We show that the interplay between these individual-level decisions and AI training may lead to societal challenges. Outputs may become more homogenized, especially when the AI is trained on AI-generated content. And any AI bias may become societal bias. A solution to the homogenization and bias issues is to improve human-AI interactions, enabling personalized outputs without sacrificing productivity.

{{</citation>}}


### (100/131) Exploring Self-Reinforcement for Improving Learnersourced Multiple-Choice Question Explanations with Large Language Models (Qiming Bao et al., 2023)

{{<citation>}}

Qiming Bao, Juho Leinonen, Alex Yuxuan Peng, Wanjun Zhong, Tim Pistotti, Alice Huang, Paul Denny, Michael Witbrock, Jiamou Liu. (2023)  
**Exploring Self-Reinforcement for Improving Learnersourced Multiple-Choice Question Explanations with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10444v1)  

---


**ABSTRACT**  
Learnersourcing involves students generating and sharing learning resources with their peers. When learnersourcing multiple-choice questions, creating explanations for the generated questions is a crucial step as it facilitates a deeper understanding of the related concepts. However, it is often difficult for students to craft effective explanations due to limited subject understanding and a tendency to merely restate the question stem, distractors, and correct answer. To help scaffold this task, in this work we propose a self-reinforcement large-language-model framework, with the goal of generating and evaluating explanations automatically. Comprising three modules, the framework generates student-aligned explanations, evaluates these explanations to ensure their quality and iteratively enhances the explanations. If an explanation's evaluation score falls below a defined threshold, the framework iteratively refines and reassesses the explanation. Importantly, our framework emulates the manner in which students compose explanations at the relevant grade level. For evaluation, we had a human subject-matter expert compare the explanations generated by students with the explanations created by the open-source large language model Vicuna-13B, a version of Vicuna-13B that had been fine-tuned using our method, and by GPT-4. We observed that, when compared to other large language models, GPT-4 exhibited a higher level of creativity in generating explanations. We also found that explanations generated by GPT-4 were ranked higher by the human expert than both those created by the other models and the original student-created explanations. Our findings represent a significant advancement in enriching the learnersourcing experience for students and enhancing the capabilities of large language models in educational applications.

{{</citation>}}


### (101/131) Functional requirements to mitigate the Risk of Harm to Patients from Artificial Intelligence in Healthcare (Juan M. García-Gómez et al., 2023)

{{<citation>}}

Juan M. García-Gómez, Vicent Blanes-Selva, José Carlos de Bartolomé Cenzano, Jaime Cebolla-Cornejo, Ascensión Doñate-Martínez. (2023)  
**Functional requirements to mitigate the Risk of Harm to Patients from Artificial Intelligence in Healthcare**  

---
Primary Category: cs.AI  
Categories: 68, cs-AI, cs.AI  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2309.10424v1)  

---


**ABSTRACT**  
The Directorate General for Parliamentary Research Services of the European Parliament has prepared a report to the Members of the European Parliament where they enumerate seven main risks of Artificial Intelligence (AI) in medicine and healthcare: patient harm due to AI errors, misuse of medical AI tools, bias in AI and the perpetuation of existing inequities, lack of transparency, privacy and security issues, gaps in accountability, and obstacles in implementation.   In this study, we propose fourteen functional requirements that AI systems may implement to reduce the risks associated with their medical purpose: AI passport, User management, Regulation check, Academic use only disclaimer, data quality assessment, Clinicians double check, Continuous performance evaluation, Audit trail, Continuous usability test, Review of retrospective/simulated cases, Bias check, eXplainable AI, Encryption and use of field-tested libraries, and Semantic interoperability.   Our intention here is to provide specific high-level specifications of technical solutions to ensure continuous good performance and use of AI systems to benefit patients in compliance with the future EU regulatory framework.

{{</citation>}}


### (102/131) Adaptive questionnaires for facilitating patient data entry in clinical decision support systems: Methods and application to STOPP/START v2 (Jean-Baptiste Lamy et al., 2023)

{{<citation>}}

Jean-Baptiste Lamy, Abdelmalek Mouazer, Karima Sedki, Sophie Dubois, Hector Falcoff. (2023)  
**Adaptive questionnaires for facilitating patient data entry in clinical decision support systems: Methods and application to STOPP/START v2**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2309.10398v1)  

---


**ABSTRACT**  
Clinical decision support systems are software tools that help clinicians to make medical decisions. However, their acceptance by clinicians is usually rather low. A known problem is that they often require clinicians to manually enter lots of patient data, which is long and tedious. Existing solutions, such as the automatic data extraction from electronic health record, are not fully satisfying, because of low data quality and availability. In practice, many systems still include long questionnaire for data entry.   In this paper, we propose an original solution to simplify patient data entry, using an adaptive questionnaire, i.e. a questionnaire that evolves during user interaction, showing or hiding questions dynamically. Considering a rule-based decision support systems, we designed methods for translating the system's clinical rules into display rules that determine the items to show in the questionnaire, and methods for determining the optimal order of priority among the items in the questionnaire. We applied this approach to a decision support system implementing STOPP/START v2, a guideline for managing polypharmacy. We show that it permits reducing by about two thirds the number of clinical conditions displayed in the questionnaire. Presented to clinicians during focus group sessions, the adaptive questionnaire was found "pretty easy to use". In the future, this approach could be applied to other guidelines, and adapted for data entry by patients.

{{</citation>}}


### (103/131) Generative AI vs. AGI: The Cognitive Strengths and Weaknesses of Modern LLMs (Ben Goertzel, 2023)

{{<citation>}}

Ben Goertzel. (2023)  
**Generative AI vs. AGI: The Cognitive Strengths and Weaknesses of Modern LLMs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, ChatGPT, GPT, GPT-4, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.10371v1)  

---


**ABSTRACT**  
A moderately detailed consideration of interactive LLMs as cognitive systems is given, focusing on LLMs circa mid-2023 such as ChatGPT, GPT-4, Bard, Llama, etc.. Cognitive strengths of these systems are reviewed, and then careful attention is paid to the substantial differences between the sort of cognitive system these LLMs are, and the sort of cognitive systems human beings are. It is found that many of the practical weaknesses of these AI systems can be tied specifically to lacks in the basic cognitive architectures according to which these systems are built. It is argued that incremental improvement of such LLMs is not a viable approach to working toward human-level AGI, in practical terms given realizable amounts of compute resources. This does not imply there is nothing to learn about human-level AGI from studying and experimenting with LLMs, nor that LLMs cannot form significant parts of human-level AGI architectures that also incorporate other ideas. Social and ethical matters regarding LLMs are very briefly touched from this perspective, which implies that while care should be taken regarding misinformation and other issues, and economic upheavals will need their own social remedies based on their unpredictable course as with any powerfully impactful technology, overall the sort of policy needed as regards modern LLMs is quite different than would be the case if a more credible approximation to human-level AGI were at hand.

{{</citation>}}


### (104/131) Metastatic Breast Cancer Prognostication Through Multimodal Integration of Dimensionality Reduction Algorithms and Classification Algorithms (Bliss Singhal et al., 2023)

{{<citation>}}

Bliss Singhal, Fnu Pooja. (2023)  
**Metastatic Breast Cancer Prognostication Through Multimodal Integration of Dimensionality Reduction Algorithms and Classification Algorithms**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10324v1)  

---


**ABSTRACT**  
Machine learning (ML) is a branch of Artificial Intelligence (AI) where computers analyze data and find patterns in the data. The study focuses on the detection of metastatic cancer using ML. Metastatic cancer is the point where the cancer has spread to other parts of the body and is the cause of approximately 90% of cancer related deaths. Normally, pathologists spend hours each day to manually classify whether tumors are benign or malignant. This tedious task contributes to mislabeling metastasis being over 60% of time and emphasizes the importance to be aware of human error, and other inefficiencies. ML is a good candidate to improve the correct identification of metastatic cancer saving thousands of lives and can also improve the speed and efficiency of the process thereby taking less resources and time. So far, deep learning methodology of AI has been used in the research to detect cancer. This study is a novel approach to determine the potential of using preprocessing algorithms combined with classification algorithms in detecting metastatic cancer. The study used two preprocessing algorithms: principal component analysis (PCA) and the genetic algorithm to reduce the dimensionality of the dataset, and then used three classification algorithms: logistic regression, decision tree classifier, and k-nearest neighbors to detect metastatic cancer in the pathology scans. The highest accuracy of 71.14% was produced by the ML pipeline comprising of PCA, the genetic algorithm, and the k-nearest neighbors algorithm, suggesting that preprocessing and classification algorithms have great potential for detecting metastatic cancer.

{{</citation>}}


### (105/131) Who to Trust, How and Why: Untangling AI Ethics Principles, Trustworthiness and Trust (Andreas Duenser et al., 2023)

{{<citation>}}

Andreas Duenser, David M. Douglas. (2023)  
**Who to Trust, How and Why: Untangling AI Ethics Principles, Trustworthiness and Trust**  

---
Primary Category: cs.AI  
Categories: I-2-0; J-4; K-4-1, cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10318v1)  

---


**ABSTRACT**  
We present an overview of the literature on trust in AI and AI trustworthiness and argue for the need to distinguish these concepts more clearly and to gather more empirically evidence on what contributes to people s trusting behaviours. We discuss that trust in AI involves not only reliance on the system itself, but also trust in the developers of the AI system. AI ethics principles such as explainability and transparency are often assumed to promote user trust, but empirical evidence of how such features actually affect how users perceive the system s trustworthiness is not as abundance or not that clear. AI systems should be recognised as socio-technical systems, where the people involved in designing, developing, deploying, and using the system are as important as the system for determining whether it is trustworthy. Without recognising these nuances, trust in AI and trustworthy AI risk becoming nebulous terms for any desirable feature for AI systems.

{{</citation>}}


### (106/131) QXAI: Explainable AI Framework for Quantitative Analysis in Patient Monitoring Systems (Thanveer Shaik et al., 2023)

{{<citation>}}

Thanveer Shaik, Xiaohui Tao, Haoran Xie, Lin Li, Juan D. Velasquez, Niall Higgins. (2023)  
**QXAI: Explainable AI Framework for Quantitative Analysis in Patient Monitoring Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2309.10293v2)  

---


**ABSTRACT**  
Artificial Intelligence techniques can be used to classify a patient's physical activities and predict vital signs for remote patient monitoring. Regression analysis based on non-linear models like deep learning models has limited explainability due to its black-box nature. This can require decision-makers to make blind leaps of faith based on non-linear model results, especially in healthcare applications. In non-invasive monitoring, patient data from tracking sensors and their predisposing clinical attributes act as input features for predicting future vital signs. Explaining the contributions of various features to the overall output of the monitoring application is critical for a clinician's decision-making. In this study, an Explainable AI for Quantitative analysis (QXAI) framework is proposed with post-hoc model explainability and intrinsic explainability for regression and classification tasks in a supervised learning approach. This was achieved by utilizing the Shapley values concept and incorporating attention mechanisms in deep learning models. We adopted the artificial neural networks (ANN) and attention-based Bidirectional LSTM (BiLSTM) models for the prediction of heart rate and classification of physical activities based on sensor data. The deep learning models achieved state-of-the-art results in both prediction and classification tasks. Global explanation and local explanation were conducted on input data to understand the feature contribution of various patient data. The proposed QXAI framework was evaluated using PPG-DaLiA data to predict heart rate and mobile health (MHEALTH) data to classify physical activities based on sensor data. Monte Carlo approximation was applied to the framework to overcome the time complexity and high computation power requirements required for Shapley value calculations.

{{</citation>}}


### (107/131) GPTFUZZER : Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts (Jiahao Yu et al., 2023)

{{<citation>}}

Jiahao Yu, Xingwei Lin, Xinyu Xing. (2023)  
**GPTFUZZER : Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10253v1)  

---


**ABSTRACT**  
Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial "jailbreak" attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging. In this paper, we introduce \fuzzer, a novel black-box jailbreak fuzzing framework inspired by AFL fuzzing framework. Instead of manual engineering, \fuzzer automates the generation of jailbreak templates for red-teaming LLMs. At its core, \fuzzer starts with human-written templates as seeds, then mutates them using mutate operators to produce new templates. We detail three key components of \fuzzer: a seed selection strategy for balancing efficiency and variability, metamorphic relations for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack. We tested \fuzzer on various commercial and open-source LLMs, such as ChatGPT, LLaMa-2, and Claude2, under diverse attack scenarios. Our results indicate that \fuzzer consistently produces jailbreak templates with a high success rate, even in settings where all human-crafted templates fail. Notably, even starting with suboptimal seed templates, \fuzzer maintains over 90\% attack success rate against ChatGPT and Llama-2 models. We believe \fuzzer will aid researchers and practitioners in assessing LLM robustness and will spur further research into LLM safety.

{{</citation>}}


## eess.IV (4)



### (108/131) Multi-Context Dual Hyper-Prior Neural Image Compression (Atefeh Khoshkhahtinat et al., 2023)

{{<citation>}}

Atefeh Khoshkhahtinat, Ali Zafari, Piyush M. Mehta, Mohammad Akyash, Hossein Kashiani, Nasser M. Nasrabadi. (2023)  
**Multi-Context Dual Hyper-Prior Neural Image Compression**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.10799v1)  

---


**ABSTRACT**  
Transform and entropy models are the two core components in deep image compression neural networks. Most existing learning-based image compression methods utilize convolutional-based transform, which lacks the ability to model long-range dependencies, primarily due to the limited receptive field of the convolution operation. To address this limitation, we propose a Transformer-based nonlinear transform. This transform has the remarkable ability to efficiently capture both local and global information from the input image, leading to a more decorrelated latent representation. In addition, we introduce a novel entropy model that incorporates two different hyperpriors to model cross-channel and spatial dependencies of the latent representation. To further improve the entropy model, we add a global context that leverages distant relationships to predict the current latent more accurately. This global context employs a causal attention mechanism to extract long-range information in a content-dependent manner. Our experiments show that our proposed framework performs better than the state-of-the-art methods in terms of rate-distortion performance.

{{</citation>}}


### (109/131) Context-Aware Neural Video Compression on Solar Dynamics Observatory (Atefeh Khoshkhahtinat et al., 2023)

{{<citation>}}

Atefeh Khoshkhahtinat, Ali Zafari, Piyush M. Mehta, Nasser M. Nasrabadi, Barbara J. Thompson, Michael S. F. Kirk, Daniel da Silva. (2023)  
**Context-Aware Neural Video Compression on Solar Dynamics Observatory**  

---
Primary Category: eess.IV  
Categories: astro-ph-SR, cs-CV, cs-IT, cs-LG, eess-IV, eess.IV, math-IT  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.10784v1)  

---


**ABSTRACT**  
NASA's Solar Dynamics Observatory (SDO) mission collects large data volumes of the Sun's daily activity. Data compression is crucial for space missions to reduce data storage and video bandwidth requirements by eliminating redundancies in the data. In this paper, we present a novel neural Transformer-based video compression approach specifically designed for the SDO images. Our primary objective is to efficiently exploit the temporal and spatial redundancies inherent in solar images to obtain a high compression ratio. Our proposed architecture benefits from a novel Transformer block called Fused Local-aware Window (FLaWin), which incorporates window-based self-attention modules and an efficient fused local-aware feed-forward (FLaFF) network. This architectural design allows us to simultaneously capture short-range and long-range information while facilitating the extraction of rich and diverse contextual representations. Moreover, this design choice results in reduced computational complexity. Experimental results demonstrate the significant contribution of the FLaWin Transformer block to the compression performance, outperforming conventional hand-engineered video codecs such as H.264 and H.265 in terms of rate-distortion trade-off.

{{</citation>}}


### (110/131) Self-Supervised Super-Resolution Approach for Isotropic Reconstruction of 3D Electron Microscopy Images from Anisotropic Acquisition (Mohammad Khateri et al., 2023)

{{<citation>}}

Mohammad Khateri, Morteza Ghahremani, Alejandra Sierra, Jussi Tohka. (2023)  
**Self-Supervised Super-Resolution Approach for Isotropic Reconstruction of 3D Electron Microscopy Images from Anisotropic Acquisition**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.10646v1)  

---


**ABSTRACT**  
Three-dimensional electron microscopy (3DEM) is an essential technique to investigate volumetric tissue ultra-structure. Due to technical limitations and high imaging costs, samples are often imaged anisotropically, where resolution in the axial direction ($z$) is lower than in the lateral directions $(x,y)$. This anisotropy 3DEM can hamper subsequent analysis and visualization tasks. To overcome this limitation, we propose a novel deep-learning (DL)-based self-supervised super-resolution approach that computationally reconstructs isotropic 3DEM from the anisotropic acquisition. The proposed DL-based framework is built upon the U-shape architecture incorporating vision-transformer (ViT) blocks, enabling high-capability learning of local and global multi-scale image dependencies. To train the tailored network, we employ a self-supervised approach. Specifically, we generate pairs of anisotropic and isotropic training datasets from the given anisotropic 3DEM data. By feeding the given anisotropic 3DEM dataset in the trained network through our proposed framework, the isotropic 3DEM is obtained. Importantly, this isotropic reconstruction approach relies solely on the given anisotropic 3DEM dataset and does not require pairs of co-registered anisotropic and isotropic 3DEM training datasets. To evaluate the effectiveness of the proposed method, we conducted experiments using three 3DEM datasets acquired from brain. The experimental results demonstrated that our proposed framework could successfully reconstruct isotropic 3DEM from the anisotropic acquisition.

{{</citation>}}


### (111/131) Learning Dynamic MRI Reconstruction with Convolutional Network Assisted Reconstruction Swin Transformer (Di Xu et al., 2023)

{{<citation>}}

Di Xu, Hengjie Liu, Dan Ruan, Ke Sheng. (2023)  
**Learning Dynamic MRI Reconstruction with Convolutional Network Assisted Reconstruction Swin Transformer**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.10227v1)  

---


**ABSTRACT**  
Dynamic magnetic resonance imaging (DMRI) is an effective imaging tool for diagnosis tasks that require motion tracking of a certain anatomy. To speed up DMRI acquisition, k-space measurements are commonly undersampled along spatial or spatial-temporal domains. The difficulty of recovering useful information increases with increasing undersampling ratios. Compress sensing was invented for this purpose and has become the most popular method until deep learning (DL) based DMRI reconstruction methods emerged in the past decade. Nevertheless, existing DL networks are still limited in long-range sequential dependency understanding and computational efficiency and are not fully automated. Considering the success of Transformers positional embedding and "swin window" self-attention mechanism in the vision community, especially natural video understanding, we hereby propose a novel architecture named Reconstruction Swin Transformer (RST) for 4D MRI. RST inherits the backbone design of the Video Swin Transformer with a novel reconstruction head introduced to restore pixel-wise intensity. A convolution network called SADXNet is used for rapid initialization of 2D MR frames before RST learning to effectively reduce the model complexity, GPU hardware demand, and training time. Experimental results in the cardiac 4D MR dataset further substantiate the superiority of RST, achieving the lowest RMSE of 0.0286 +/- 0.0199 and 1 - SSIM of 0.0872 +/- 0.0783 on 9 times accelerated validation sequences.

{{</citation>}}


## eess.SY (1)



### (112/131) Physics-Informed Machine Learning for Data Anomaly Detection, Classification, Localization, and Mitigation: A Review, Challenges, and Path Forward (Mehdi Jabbari Zideh et al., 2023)

{{<citation>}}

Mehdi Jabbari Zideh, Paroma Chatterjee, Anurag K. Srivastava. (2023)  
**Physics-Informed Machine Learning for Data Anomaly Detection, Classification, Localization, and Mitigation: A Review, Challenges, and Path Forward**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2309.10788v1)  

---


**ABSTRACT**  
Advancements in digital automation for smart grids have led to the installation of measurement devices like phasor measurement units (PMUs), micro-PMUs ($\mu$-PMUs), and smart meters. However, a large amount of data collected by these devices brings several challenges as control room operators need to use this data with models to make confident decisions for reliable and resilient operation of the cyber-power systems. Machine-learning (ML) based tools can provide a reliable interpretation of the deluge of data obtained from the field. For the decision-makers to ensure reliable network operation under all operating conditions, these tools need to identify solutions that are feasible and satisfy the system constraints, while being efficient, trustworthy, and interpretable. This resulted in the increasing popularity of physics-informed machine learning (PIML) approaches, as these methods overcome challenges that model-based or data-driven ML methods face in silos. This work aims at the following: a) review existing strategies and techniques for incorporating underlying physical principles of the power grid into different types of ML approaches (supervised/semi-supervised learning, unsupervised learning, and reinforcement learning (RL)); b) explore the existing works on PIML methods for anomaly detection, classification, localization, and mitigation in power transmission and distribution systems, c) discuss improvements in existing methods through consideration of potential challenges while also addressing the limitations to make them suitable for real-world applications.

{{</citation>}}


## cs.CY (2)



### (113/131) EU law and emotion data (Andreas Hauselmann et al., 2023)

{{<citation>}}

Andreas Hauselmann, Alan M. Sears, Lex Zard, Eduard Fosch-Villaronga. (2023)  
**EU law and emotion data**  

---
Primary Category: cs.CY  
Categories: K-4; K-5, cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10776v1)  

---


**ABSTRACT**  
This article sheds light on legal implications and challenges surrounding emotion data processing within the EU's legal framework. Despite the sensitive nature of emotion data, the GDPR does not categorize it as special data, resulting in a lack of comprehensive protection. The article also discusses the nuances of different approaches to affective computing and their relevance to the processing of special data under the GDPR. Moreover, it points to potential tensions with data protection principles, such as fairness and accuracy. Our article also highlights some of the consequences, including harm, that processing of emotion data may have for individuals concerned. Additionally, we discuss how the AI Act proposal intends to regulate affective computing. Finally, the article outlines the new obligations and transparency requirements introduced by the DSA for online platforms utilizing emotion data. Our article aims at raising awareness among the affective computing community about the applicable legal requirements when developing AC systems intended for the EU market, or when working with study participants located in the EU. We also stress the importance of protecting the fundamental rights of individuals even when the law struggles to keep up with technological developments that capture sensitive emotion data.

{{</citation>}}


### (114/131) Dialogues with algorithms (Joost J. Joosten, 2023)

{{<citation>}}

Joost J. Joosten. (2023)  
**Dialogues with algorithms**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2309.10678v1)  

---


**ABSTRACT**  
In this short paper we focus on human in the loop for rule-based software used for law enforcement. For example, one can think of software that computes fines like tachograph software, software that prepares evidence like DNA sequencing software or social profiling software to patrol in high-risk zones, among others. An important difference between a legal human agent and a software application lies in possible dialogues. A human agent can be interrogated to motivate her decisions. Often such dialogues with software are at the best extremely hard but mostly impossible. We observe that the absence of a dialogue can sincerely violate civil rights and legal principles like, for example, Transparency or Contestability. Thus, possible dialogues with legal algorithms are at the least highly desirable. Futuristic as this may sound, we observe that in various realms of formal methods, such dialogues are easily obtainable. However, this triggers the usual tension between the expressibility of the dialogue language and the feasibility of the corresponding computations.

{{</citation>}}


## cs.SD (3)



### (115/131) MelodyGLM: Multi-task Pre-training for Symbolic Melody Generation (Xinda Wu et al., 2023)

{{<citation>}}

Xinda Wu, Zhijie Huang, Kejun Zhang, Jiaxing Yu, Xu Tan, Tieyao Zhang, Zihao Wang, Lingyun Sun. (2023)  
**MelodyGLM: Multi-task Pre-training for Symbolic Melody Generation**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CL, cs-IR, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2309.10738v2)  

---


**ABSTRACT**  
Pre-trained language models have achieved impressive results in various music understanding and generation tasks. However, existing pre-training methods for symbolic melody generation struggle to capture multi-scale, multi-dimensional structural information in note sequences, due to the domain knowledge discrepancy between text and music. Moreover, the lack of available large-scale symbolic melody datasets limits the pre-training improvement. In this paper, we propose MelodyGLM, a multi-task pre-training framework for generating melodies with long-term structure. We design the melodic n-gram and long span sampling strategies to create local and global blank infilling tasks for modeling the local and global structures in melodies. Specifically, we incorporate pitch n-grams, rhythm n-grams, and their combined n-grams into the melodic n-gram blank infilling tasks for modeling the multi-dimensional structures in melodies. To this end, we have constructed a large-scale symbolic melody dataset, MelodyNet, containing more than 0.4 million melody pieces. MelodyNet is utilized for large-scale pre-training and domain-specific n-gram lexicon construction. Both subjective and objective evaluations demonstrate that MelodyGLM surpasses the standard and previous pre-training methods. In particular, subjective evaluations show that, on the melody continuation task, MelodyGLM gains average improvements of 0.82, 0.87, 0.78, and 0.94 in consistency, rhythmicity, structure, and overall quality, respectively. Notably, MelodyGLM nearly matches the quality of human-composed melodies on the melody inpainting task.

{{</citation>}}


### (116/131) Motif-Centric Representation Learning for Symbolic Music (Yuxuan Wu et al., 2023)

{{<citation>}}

Yuxuan Wu, Roger B. Dannenberg, Gus Xia. (2023)  
**Motif-Centric Representation Learning for Symbolic Music**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.10597v1)  

---


**ABSTRACT**  
Music motif, as a conceptual building block of composition, is crucial for music structure analysis and automatic composition. While human listeners can identify motifs easily, existing computational models fall short in representing motifs and their developments. The reason is that the nature of motifs is implicit, and the diversity of motif variations extends beyond simple repetitions and modulations. In this study, we aim to learn the implicit relationship between motifs and their variations via representation learning, using the Siamese network architecture and a pretraining and fine-tuning pipeline. A regularization-based method, VICReg, is adopted for pretraining, while contrastive learning is used for fine-tuning. Experimental results on a retrieval-based task show that these two methods complement each other, yielding an improvement of 12.6% in the area under the precision-recall curve. Lastly, we visualize the acquired motif representations, offering an intuitive comprehension of the overall structure of a music piece. As far as we know, this work marks a noteworthy step forward in computational modeling of music motifs. We believe that this work lays the foundations for future applications of motifs in automatic music composition and music information retrieval.

{{</citation>}}


### (117/131) Bridging the Spoof Gap: A Unified Parallel Aggregation Network for Voice Presentation Attacks (Awais Khan et al., 2023)

{{<citation>}}

Awais Khan, Khalid Mahmood Malik. (2023)  
**Bridging the Spoof Gap: A Unified Parallel Aggregation Network for Voice Presentation Attacks**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2309.10560v1)  

---


**ABSTRACT**  
Automatic Speaker Verification (ASV) systems are increasingly used in voice bio-metrics for user authentication but are susceptible to logical and physical spoofing attacks, posing security risks. Existing research mainly tackles logical or physical attacks separately, leading to a gap in unified spoofing detection. Moreover, when existing systems attempt to handle both types of attacks, they often exhibit significant disparities in the Equal Error Rate (EER). To bridge this gap, we present a Parallel Stacked Aggregation Network that processes raw audio. Our approach employs a split-transform-aggregation technique, dividing utterances into convolved representations, applying transformations, and aggregating the results to identify logical (LA) and physical (PA) spoofing attacks. Evaluation of the ASVspoof-2019 and VSDC datasets shows the effectiveness of the proposed system. It outperforms state-of-the-art solutions, displaying reduced EER disparities and superior performance in detecting spoofing attacks. This highlights the proposed method's generalizability and superiority. In a world increasingly reliant on voice-based security, our unified spoofing detection system provides a robust defense against a spectrum of voice spoofing attacks, safeguarding ASVs and user data effectively.

{{</citation>}}


## cs.ET (1)



### (118/131) QuBEC: Boosting Equivalence Checking for Quantum Circuits with QEC Embedding (Chao Lu et al., 2023)

{{<citation>}}

Chao Lu, Navnil Choudhury, Utsav Banerjee, Abdullah Ash Saki, Kanad Basu. (2023)  
**QuBEC: Boosting Equivalence Checking for Quantum Circuits with QEC Embedding**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs.ET, quant-ph  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.10728v1)  

---


**ABSTRACT**  
Quantum computing has proven to be capable of accelerating many algorithms by performing tasks that classical computers cannot. Currently, Noisy Intermediate Scale Quantum (NISQ) machines struggle from scalability and noise issues to render a commercial quantum computer. However, the physical and software improvements of a quantum computer can efficiently control quantum gate noise. As the complexity of quantum algorithms and implementation increases, software control of quantum circuits may lead to a more intricate design. Consequently, the verification of quantum circuits becomes crucial in ensuring the correctness of the compilation, along with other processes, including quantum error correction and assertions, that can increase the fidelity of quantum circuits. In this paper, we propose a Decision Diagram-based quantum equivalence checking approach, QuBEC, that requires less latency compared to existing techniques, while accounting for circuits with quantum error correction redundancy. Our proposed methodology reduces verification time on certain benchmark circuits by up to $271.49 \times$, while the number of Decision Diagram nodes required is reduced by up to $798.31 \times$, compared to state-of-the-art strategies. The proposed QuBEC framework can contribute to the advancement of quantum computing by enabling faster and more efficient verification of quantum circuits, paving the way for the development of larger and more complex quantum algorithms.

{{</citation>}}


## cs.IT (1)



### (119/131) AI/ML for Beam Management in 5G-Advanced (Qing Xue et al., 2023)

{{<citation>}}

Qing Xue, Jiajia Guo, Binggui Zhou, Yongjun Xu, Zhidu Li, Shaodan Ma. (2023)  
**AI/ML for Beam Management in 5G-Advanced**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-SY, cs.IT, eess-SY, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10575v1)  

---


**ABSTRACT**  
In beamformed wireless cellular systems such as 5G New Radio (NR) networks, beam management (BM) is a crucial operation. In the second phase of 5G NR standardization, known as 5G-Advanced, which is being vigorously promoted, the key component is the use of artificial intelligence (AI) based on machine learning (ML) techniques. AI/ML for BM is selected as a representative use case. This article provides an overview of the AI/ML for BM in 5G-Advanced. The legacy non-AI and prime AI-enabled BM frameworks are first introduced and compared. Then, the main scope of AI/ML for BM is presented, including improving accuracy, reducing overhead and latency. Finally, the key challenges and open issues in the standardization of AI/ML for BM are discussed, especially the design of new protocols for AI-enabled BM. This article provides a guideline for the study of AI/ML-based BM standardization.

{{</citation>}}


## cs.DC (2)



### (120/131) Task Graph offloading via Deep Reinforcement Learning in Mobile Edge Computing (Jiagang Liu et al., 2023)

{{<citation>}}

Jiagang Liu, Yun Mi, Xinyu Zhang. (2023)  
**Task Graph offloading via Deep Reinforcement Learning in Mobile Edge Computing**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10569v1)  

---


**ABSTRACT**  
Various mobile applications that comprise dependent tasks are gaining widespread popularity and are increasingly complex. These applications often have low-latency requirements, resulting in a significant surge in demand for computing resources. With the emergence of mobile edge computing (MEC), it becomes the most significant issue to offload the application tasks onto small-scale devices deployed at the edge of the mobile network for obtaining a high-quality user experience. However, since the environment of MEC is dynamic, most existing works focusing on task graph offloading, which rely heavily on expert knowledge or accurate analytical models, fail to fully adapt to such environmental changes, resulting in the reduction of user experience. This paper investigates the task graph offloading in MEC, considering the time-varying computation capabilities of edge computing devices. To adapt to environmental changes, we model the task graph scheduling for computation offloading as a Markov Decision Process (MDP). Then, we design a deep reinforcement learning algorithm (SATA-DRL) to learn the task scheduling strategy from the interaction with the environment, to improve user experience. Extensive simulations validate that SATA-DRL is superior to existing strategies in terms of reducing average makespan and deadline violation.

{{</citation>}}


### (121/131) Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity (Haojun Xia et al., 2023)

{{<citation>}}

Haojun Xia, Zhen Zheng, Yuchao Li, Donglin Zhuang, Zhongzhu Zhou, Xiafei Qiu, Yong Li, Wei Lin, Shuaiwen Leon Song. (2023)  
**Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity**  

---
Primary Category: cs.DC  
Categories: cs-AR, cs-DC, cs-LG, cs.DC  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.10285v1)  

---


**ABSTRACT**  
With the fast growth of parameter size, it becomes increasingly challenging to deploy large generative models as they typically require large GPU memory consumption and massive computation. Unstructured model pruning has been a common approach to reduce both GPU memory footprint and the overall computation while retaining good model accuracy. However, the existing solutions do not provide a highly-efficient support for handling unstructured sparsity on modern GPUs, especially on the highly-structured Tensor Core hardware. Therefore, we propose Flash-LLM for enabling low-cost and highly-efficient large generative model inference with the sophisticated support of unstructured sparsity on high-performance but highly restrictive Tensor Cores. Based on our key observation that the main bottleneck of generative model inference is the several skinny matrix multiplications for which Tensor Cores would be significantly under-utilized due to low computational intensity, we propose a general Load-as-Sparse and Compute-as-Dense methodology for unstructured sparse matrix multiplication. The basic insight is to address the significant memory bandwidth bottleneck while tolerating redundant computations that are not critical for end-to-end performance on Tensor Cores. Based on this, we design an effective software framework for Tensor Core based unstructured SpMM, leveraging on-chip resources for efficient sparse data extraction and computation/memory-access overlapping. At SpMM kernel level, Flash-LLM significantly outperforms the state-of-the-art library, i.e., Sputnik and SparTA by an average of 2.9x and 1.5x, respectively. At end-to-end framework level on OPT-30B/66B/175B models, for tokens per GPU-second, Flash-LLM achieves up to 3.8x and 3.6x improvement over DeepSpeed and FasterTransformer, respectively, with significantly lower inference cost.

{{</citation>}}


## cs.IR (2)



### (122/131) A Hierarchical Neural Framework for Classification and its Explanation in Large Unstructured Legal Documents (Nishchal Prasad et al., 2023)

{{<citation>}}

Nishchal Prasad, Mohand Boughanem, Taoufik Dkaki. (2023)  
**A Hierarchical Neural Framework for Classification and its Explanation in Large Unstructured Legal Documents**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: GPT, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2309.10563v1)  

---


**ABSTRACT**  
Automatic legal judgment prediction and its explanation suffer from the problem of long case documents exceeding tens of thousands of words, in general, and having a non-uniform structure. Predicting judgments from such documents and extracting their explanation becomes a challenging task, more so on documents with no structural annotation. We define this problem as "scarce annotated legal documents" and explore their lack of structural information and their long lengths with a deep learning-based classification framework which we call MESc; "Multi-stage Encoder-based Supervised with-clustering"; for judgment prediction. Specifically, we divide a document into parts to extract their embeddings from the last four layers of a custom fine-tuned Large Language Model, and try to approximate their structure through unsupervised clustering. Which we use in another set of transformer encoder layers to learn the inter-chunk representations. We explore the adaptability of LLMs with multi-billion parameters (GPT-Neo, and GPT-J) to legal texts and their intra-domain(legal) transfer learning capacity. Alongside this, we compare their performance with MESc and the impact of combining embeddings from their last layers. For such hierarchical models, we also propose an explanation extraction algorithm named ORSE; Occlusion sensitivity-based Relevant Sentence Extractor;

{{</citation>}}


### (123/131) Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling (Junzhe Jiang et al., 2023)

{{<citation>}}

Junzhe Jiang, Shang Qu, Mingyue Cheng, Qi Liu. (2023)  
**Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.10435v1)  

---


**ABSTRACT**  
Recommender systems are essential for online applications, and sequential recommendation has enjoyed significant prevalence due to its expressive ability to capture dynamic user interests. However, previous sequential modeling methods still have limitations in capturing contextual information. The primary reason for this issue is that language models often lack an understanding of domain-specific knowledge and item-related textual content. To address this issue, we adopt a new sequential recommendation paradigm and propose LANCER, which leverages the semantic understanding capabilities of pre-trained language models to generate personalized recommendations. Our approach bridges the gap between language models and recommender systems, resulting in more human-like recommendations. We demonstrate the effectiveness of our approach through experiments on several benchmark datasets, showing promising results and providing valuable insights into the influence of our model on sequential recommendation tasks. Furthermore, our experimental codes are publicly available.

{{</citation>}}


## stat.ML (1)



### (124/131) Hybrid State Space-based Learning for Sequential Data Prediction with Joint Optimization (Mustafa E. Aydın et al., 2023)

{{<citation>}}

Mustafa E. Aydın, Arda Fazla, Suleyman S. Kozat. (2023)  
**Hybrid State Space-based Learning for Sequential Data Prediction with Joint Optimization**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.10553v1)  

---


**ABSTRACT**  
We investigate nonlinear prediction/regression in an online setting and introduce a hybrid model that effectively mitigates, via a joint mechanism through a state space formulation, the need for domain-specific feature engineering issues of conventional nonlinear prediction models and achieves an efficient mix of nonlinear and linear components. In particular, we use recursive structures to extract features from raw sequential sequences and a traditional linear time series model to deal with the intricacies of the sequential data, e.g., seasonality, trends. The state-of-the-art ensemble or hybrid models typically train the base models in a disjoint manner, which is not only time consuming but also sub-optimal due to the separation of modeling or independent training. In contrast, as the first time in the literature, we jointly optimize an enhanced recurrent neural network (LSTM) for automatic feature extraction from raw data and an ARMA-family time series model (SARIMAX) for effectively addressing peculiarities associated with time series data. We achieve this by introducing novel state space representations for the base models, which are then combined to provide a full state space representation of the hybrid or the ensemble. Hence, we are able to jointly optimize both models in a single pass via particle filtering, for which we also provide the update equations. The introduced architecture is generic so that one can use other recurrent architectures, e.g., GRUs, traditional time series-specific models, e.g., ETS or other optimization methods, e.g., EKF, UKF. Due to such novel combination and joint optimization, we demonstrate significant improvements in widely publicized real life competition datasets. We also openly share our code for further research and replicability of our results.

{{</citation>}}


## q-fin.CP (1)



### (125/131) Mean Absolute Directional Loss as a New Loss Function for Machine Learning Problems in Algorithmic Investment Strategies (Jakub Michańków et al., 2023)

{{<citation>}}

Jakub Michańków, Paweł Sakowski, Robert Ślepaczuk. (2023)  
**Mean Absolute Directional Loss as a New Loss Function for Machine Learning Problems in Algorithmic Investment Strategies**  

---
Primary Category: q-fin.CP  
Categories: cs-AI, cs-LG, q-fin-CP, q-fin-GN, q-fin-PM, q-fin.CP  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2309.10546v1)  

---


**ABSTRACT**  
This paper investigates the issue of an adequate loss function in the optimization of machine learning models used in the forecasting of financial time series for the purpose of algorithmic investment strategies (AIS) construction. We propose the Mean Absolute Directional Loss (MADL) function, solving important problems of classical forecast error functions in extracting information from forecasts to create efficient buy/sell signals in algorithmic investment strategies. Finally, based on the data from two different asset classes (cryptocurrencies: Bitcoin and commodities: Crude Oil), we show that the new loss function enables us to select better hyperparameters for the LSTM model and obtain more efficient investment strategies, with regard to risk-adjusted return metrics on the out-of-sample data.

{{</citation>}}


## cs.CR (2)



### (126/131) Exploring the Dark Side of AI: Advanced Phishing Attack Design and Deployment Using ChatGPT (Nils Begou et al., 2023)

{{<citation>}}

Nils Begou, Jeremy Vinoy, Andrzej Duda, Maciej Korczynski. (2023)  
**Exploring the Dark Side of AI: Advanced Phishing Attack Design and Deployment Using ChatGPT**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.10463v1)  

---


**ABSTRACT**  
This paper explores the possibility of using ChatGPT to develop advanced phishing attacks and automate their large-scale deployment. We make ChatGPT generate the following parts of a phishing attack: i) cloning a targeted website, ii) integrating code for stealing credentials, iii) obfuscating code, iv) automating website deployment on a hosting provider, v) registering a phishing domain name, and vi) integrating the website with a reverse proxy. The initial assessment of the automatically generated phishing kits highlights their rapid generation and deployment process as well as the close resemblance of the resulting pages to the target website. More broadly, we demonstrate that recent advances in AI underscore the potential risks of its misuse in phishing attacks, which can lead to their increased prevalence and severity. This highlights the necessity for enhanced countermeasures within AI systems.

{{</citation>}}


### (127/131) LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins (Umar Iqbal et al., 2023)

{{<citation>}}

Umar Iqbal, Tadayoshi Kohno, Franziska Roesner. (2023)  
**LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs-CY, cs-LG, cs.CR  
Keywords: AI, ChatGPT, GPT, Security  
[Paper Link](http://arxiv.org/abs/2309.10254v1)  

---


**ABSTRACT**  
Large language model (LLM) platforms, such as ChatGPT, have recently begun offering a plugin ecosystem to interface with third-party services on the internet. While these plugins extend the capabilities of LLM platforms, they are developed by arbitrary third parties and thus cannot be implicitly trusted. Plugins also interface with LLM platforms and users using natural language, which can have imprecise interpretations. In this paper, we propose a framework that lays a foundation for LLM platform designers to analyze and improve the security, privacy, and safety of current and future plugin-integrated LLM platforms. Our framework is a formulation of an attack taxonomy that is developed by iteratively exploring how LLM platform stakeholders could leverage their capabilities and responsibilities to mount attacks against each other. As part of our iterative process, we apply our framework in the context of OpenAI's plugin ecosystem. We uncover plugins that concretely demonstrate the potential for the types of issues that we outline in our attack taxonomy. We conclude by discussing novel challenges and by providing recommendations to improve the security, privacy, and safety of present and future LLM-based computing platforms.

{{</citation>}}


## cs.SI (1)



### (128/131) Gaining a better understanding of online polarization by approaching it as a dynamic process (Celina Treuillier et al., 2023)

{{<citation>}}

Celina Treuillier, Sylvain Castagnos, Christèle Lagier, Armelle Brun. (2023)  
**Gaining a better understanding of online polarization by approaching it as a dynamic process**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.10423v1)  

---


**ABSTRACT**  
Polarization is often a clich{\'e}, its conceptualization remains approximate and no consensus has been reached so far. Often simply seen as an inevitable result of the use of social networks, polarization nevertheless remains a complex social phenomenon that must be placed in a wider context. To contribute to a better understanding of polarization, we approach it as an evolving process, drawing on a dual expertise in political and data sciences. We compare the polarization process between one mature debate (COVID-19 vaccine) and one emerging debate (Ukraine conflict) at the time of data collection. Both debates are studied on Twitter users, a highly politicized population, and on the French population to provide key elements beyond the traditional US context. This unprecedented analysis confirms that polarization varies over time, through a succession of specific periods, whose existence and duration depend on the maturity of the debate. Importantly, we highlight that polarization is paced by context-related events. Bearing this in mind, we pave the way for a new generation of personalized depolarization strategies, adapted to the context and maturity of debates.

{{</citation>}}


## quant-ph (1)



### (129/131) Differentiable Quantum Architecture Search for Quantum Reinforcement Learning (Yize Sun et al., 2023)

{{<citation>}}

Yize Sun, Yunpu Ma, Volker Tresp. (2023)  
**Differentiable Quantum Architecture Search for Quantum Reinforcement Learning**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: QA, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10392v2)  

---


**ABSTRACT**  
Differentiable quantum architecture search (DQAS) is a gradient-based framework to design quantum circuits automatically in the NISQ era. It was motivated by such as low fidelity of quantum hardware, low flexibility of circuit architecture, high circuit design cost, barren plateau (BP) problem, and periodicity of weights. People used it to address error mitigation, unitary decomposition, and quantum approximation optimization problems based on fixed datasets. Quantum reinforcement learning (QRL) is a part of quantum machine learning and often has various data. QRL usually uses a manually designed circuit. However, the pre-defined circuit needs more flexibility for different tasks, and the circuit design based on various datasets could become intractable in the case of a large circuit. The problem of whether DQAS can be applied to quantum deep Q-learning with various datasets is still open. The main target of this work is to discover the capability of DQAS to solve quantum deep Q-learning problems. We apply a gradient-based framework DQAS on reinforcement learning tasks and evaluate it in two different environments - cart pole and frozen lake. It contains input- and output weights, progressive search, and other new features. The experiments conclude that DQAS can design quantum circuits automatically and efficiently. The evaluation results show significant outperformance compared to the manually designed circuit. Furthermore, the performance of the automatically created circuit depends on whether the super-circuit learned well during the training process. This work is the first to show that gradient-based quantum architecture search is applicable to QRL tasks.

{{</citation>}}


## physics.flu-dyn (1)



### (130/131) Correlation between morphological evolution of splashing drop and exerted impact force revealed by interpretation of explainable artificial intelligence (Jingzu Yee et al., 2023)

{{<citation>}}

Jingzu Yee, Daichi Igarashi, Pradipto, Akinori Yamanaka, Yoshiyuki Tagawa. (2023)  
**Correlation between morphological evolution of splashing drop and exerted impact force revealed by interpretation of explainable artificial intelligence**  

---
Primary Category: physics.flu-dyn  
Categories: cs-AI, cs-CV, physics-flu-dyn, physics.flu-dyn  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10266v1)  

---


**ABSTRACT**  
This study reveals a possible correlation between splashing morphology and the normalized impact force exerted by an impacting drop on a solid surface. This finding is obtained from a newly proposed feature extraction method and a subsequent interpretation of the classification of splashing and non-splashing drops performed by an explainable artificial intelligence (XAI) video classifier. Notably, the values of the weight matrix elements of the XAI that correspond to the extracted features are found to change with the temporal evolution of the drop morphology. We compute the rate of change of the contributions of each frame with respect to the classification value of a video as an important index to quantify the contributions of the extracted splashing and non-splashing features at different impact times to the classification of the XAI model. Remarkably, the rate computed for the extracted splashing features is found to closely match the profile of the normalized impact force, where the splashing features are most pronounced immediately after the normalized impact force reaches its peak value. This study has provided an example that clarifies the relationship between the complex morphological evolution of a splashing drop and physical parameters by interpreting the classification of an XAI video classifier.

{{</citation>}}


## cs.SE (1)



### (131/131) Revisiting and Improving Retrieval-Augmented Deep Assertion Generation (Weifeng Sun et al., 2023)

{{<citation>}}

Weifeng Sun, Hongyan Li, Meng Yan, Yan Lei, Hongyu Zhang. (2023)  
**Revisiting and Improving Retrieval-Augmented Deep Assertion Generation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2309.10264v1)  

---


**ABSTRACT**  
Unit testing validates the correctness of the unit under test and has become an essential activity in software development process. A unit test consists of a test prefix that drives the unit under test into a particular state, and a test oracle (e.g., assertion), which specifies the behavior in that state. To reduce manual efforts in conducting unit testing, Yu et al. proposed an integrated approach (integration for short), combining information retrieval (IR) with a deep learning-based approach, to generate assertions for a unit test. Despite promising, there is still a knowledge gap as to why or where integration works or does not work. In this paper, we describe an in-depth analysis of the effectiveness of integration. Our analysis shows that: 1) The overall performance of integration is mainly due to its success in retrieving assertions. 2) integration struggles to understand the semantic differences between the retrieved focal-test (focal-test includes a test prefix and a unit under test) and the input focal-test; 3) integration is limited to specific types of edit operations and cannot handle token addition or deletion. To improve the effectiveness of assertion generation, this paper proposes a novel retrieve-and-edit approach named EditAS. Specifically, EditAS first retrieves a similar focal-test from a pre-defined corpus and treats its assertion as a prototype. Then, EditAS reuses the information in the prototype and edits the prototype automatically. EditAS is more generalizable than integration. We conduct experiments on two large-scale datasets and experimental results demonstrate that EditAS outperforms the state-of-the-art approaches, with an average improvement of 10.00%-87.48% and 3.30%-42.65% in accuracy and BLEU score, respectively.

{{</citation>}}
