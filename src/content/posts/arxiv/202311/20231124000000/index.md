---
draft: false
title: "arXiv @ 2023.11.24"
date: 2023-11-24
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.24"
    identifier: arxiv_20231124
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (22)](#cscl-22)
- [cs.HC (3)](#cshc-3)
- [cs.SE (3)](#csse-3)
- [cs.AI (14)](#csai-14)
- [cs.CV (27)](#cscv-27)
- [eess.IV (3)](#eessiv-3)
- [cs.LG (28)](#cslg-28)
- [cs.IT (1)](#csit-1)
- [cs.DC (3)](#csdc-3)
- [cs.CR (1)](#cscr-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.CY (2)](#cscy-2)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.DM (1)](#csdm-1)
- [q-bio.BM (1)](#q-biobm-1)
- [cs.AR (2)](#csar-2)
- [cs.SI (2)](#cssi-2)
- [cs.NI (1)](#csni-1)
- [cs.RO (1)](#csro-1)
- [eess.SY (1)](#eesssy-1)

## cs.CL (22)



### (1/118) Surpassing GPT-4 Medical Coding with a Two-Stage Approach (Zhichao Yang et al., 2023)

{{<citation>}}

Zhichao Yang, Sanjit Singh Batra, Joel Stremmel, Eran Halperin. (2023)  
**Surpassing GPT-4 Medical Coding with a Two-Stage Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, LSTM  
[Paper Link](http://arxiv.org/abs/2311.13735v1)  

---


**ABSTRACT**  
Recent advances in large language models (LLMs) show potential for clinical applications, such as clinical decision support and trial recommendations. However, the GPT-4 LLM predicts an excessive number of ICD codes for medical coding tasks, leading to high recall but low precision. To tackle this challenge, we introduce LLM-codex, a two-stage approach to predict ICD codes that first generates evidence proposals using an LLM and then employs an LSTM-based verification stage. The LSTM learns from both the LLM's high recall and human expert's high precision, using a custom loss function. Our model is the only approach that simultaneously achieves state-of-the-art results in medical coding accuracy, accuracy on rare codes, and sentence-level evidence identification to support coding decisions without training on human-annotated evidence according to experiments on the MIMIC dataset.

{{</citation>}}


### (2/118) Comparison of pipeline, sequence-to-sequence, and GPT models for end-to-end relation extraction: experiments with the rare disease use-case (Shashank Gupta et al., 2023)

{{<citation>}}

Shashank Gupta, Xuguang Ai, Ramakanth Kavuluru. (2023)  
**Comparison of pipeline, sequence-to-sequence, and GPT models for end-to-end relation extraction: experiments with the rare disease use-case**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, NER, NLP  
[Paper Link](http://arxiv.org/abs/2311.13729v1)  

---


**ABSTRACT**  
End-to-end relation extraction (E2ERE) is an important and realistic application of natural language processing (NLP) in biomedicine. In this paper, we aim to compare three prevailing paradigms for E2ERE using a complex dataset focused on rare diseases involving discontinuous and nested entities. We use the RareDis information extraction dataset to evaluate three competing approaches (for E2ERE): NER $\rightarrow$ RE pipelines, joint sequence to sequence models, and generative pre-trained transformer (GPT) models. We use comparable state-of-the-art models and best practices for each of these approaches and conduct error analyses to assess their failure modes. Our findings reveal that pipeline models are still the best, while sequence-to-sequence models are not far behind; GPT models with eight times as many parameters are worse than even sequence-to-sequence models and lose to pipeline models by over 10 F1 points. Partial matches and discontinuous entities caused many NER errors contributing to lower overall E2E performances. We also verify these findings on a second E2ERE dataset for chemical-protein interactions. Although generative LM-based methods are more suitable for zero-shot settings, when training data is available, our results show that it is better to work with more conventional models trained and tailored for E2ERE. More innovative methods are needed to marry the best of the both worlds from smaller encoder-decoder pipeline models and the larger GPT models to improve E2ERE. As of now, we see that well designed pipeline models offer substantial performance gains at a lower cost and carbon footprint for E2ERE. Our contribution is also the first to conduct E2ERE for the RareDis dataset.

{{</citation>}}


### (3/118) Dynamic Analysis Method for Hidden Dangers in Substation Based on Knowledge Graph (Weiwei Li et al., 2023)

{{<citation>}}

Weiwei Li, Xing Liu, Wei Wang, Lu Chen, Sizhe Li, Hui Fan. (2023)  
**Dynamic Analysis Method for Hidden Dangers in Substation Based on Knowledge Graph**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.13708v1)  

---


**ABSTRACT**  
To address the challenge of identifying and understanding hidden dangers in substations from unstructured text data, a novel dynamic analysis method is proposed. This approach begins by analyzing and extracting data from the unstructured text related to hidden dangers. It then leverages a flexible, distributed data search engine built on Elastic-Search to handle this information. Following this, the hidden Markov model is employed to train the data within the engine. The Viterbi algorithm is integrated to decipher the hidden state sequences, facilitating the segmentation and labeling of entities related to hidden dangers. The final step involves using the Neo4j graph database to dynamically create a knowledge map that visualizes hidden dangers in the substation. This method's effectiveness is demonstrated through an example analysis using data from a specific substation's hidden dangers.

{{</citation>}}


### (4/118) MAIRA-1: A specialised large multimodal model for radiology report generation (Stephanie L. Hyland et al., 2023)

{{<citation>}}

Stephanie L. Hyland, Shruthi Bannur, Kenza Bouzid, Daniel C. Castro, Mercy Ranjit, Anton Schwaighofer, Fernando Pérez-García, Valentina Salvatelli, Shaury Srivastav, Anja Thieme, Noel Codella, Matthew P. Lungren, Maria Teodora Wetscherek, Ozan Oktay, Javier Alvarez-Valle. (2023)  
**MAIRA-1: A specialised large multimodal model for radiology report generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13668v1)  

---


**ABSTRACT**  
We present a radiology-specific multimodal model for the task for generating radiological reports from chest X-rays (CXRs). Our work builds on the idea that large language model(s) can be equipped with multimodal capabilities through alignment with pre-trained vision encoders. On natural images, this has been shown to allow multimodal models to gain image understanding and description capabilities. Our proposed model (MAIRA-1) leverages a CXR-specific image encoder in conjunction with a fine-tuned large language model based on Vicuna-7B, and text-based data augmentation, to produce reports with state-of-the-art quality. In particular, MAIRA-1 significantly improves on the radiologist-aligned RadCliQ metric and across all lexical metrics considered. Manual review of model outputs demonstrates promising fluency and accuracy of generated reports while uncovering failure modes not captured by existing evaluation practices. More information and resources can be found on the project website: https://aka.ms/maira.

{{</citation>}}


### (5/118) Efficient Transformer Knowledge Distillation: A Performance Review (Nathan Brown et al., 2023)

{{<citation>}}

Nathan Brown, Ashton Williamson, Tahj Anderson, Logan Lawrence. (2023)  
**Efficient Transformer Knowledge Distillation: A Performance Review**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GLUE, Knowledge Distillation, NER, Named Entity Recognition, Natural Language Processing, QA, Transformer  
[Paper Link](http://arxiv.org/abs/2311.13657v1)  

---


**ABSTRACT**  
As pretrained transformer language models continue to achieve state-of-the-art performance, the Natural Language Processing community has pushed for advances in model compression and efficient attention mechanisms to address high computational requirements and limited input sequence length. Despite these separate efforts, no investigation has been done into the intersection of these two fields. In this work, we provide an evaluation of model compression via knowledge distillation on efficient attention transformers. We provide cost-performance trade-offs for the compression of state-of-the-art efficient attention architectures and the gains made in performance in comparison to their full attention counterparts. Furthermore, we introduce a new long-context Named Entity Recognition dataset, GONERD, to train and test the performance of NER models on long sequences. We find that distilled efficient attention transformers can preserve a significant amount of original model performance, preserving up to 98.6% across short-context tasks (GLUE, SQUAD, CoNLL-2003), up to 94.6% across long-context Question-and-Answering tasks (HotpotQA, TriviaQA), and up to 98.8% on long-context Named Entity Recognition (GONERD), while decreasing inference times by up to 57.8%. We find that, for most models on most tasks, performing knowledge distillation is an effective method to yield high-performing efficient attention models with low costs.

{{</citation>}}


### (6/118) Language Model Inversion (John X. Morris et al., 2023)

{{<citation>}}

John X. Morris, Wenting Zhao, Justin T. Chiu, Vitaly Shmatikov, Alexander M. Rush. (2023)  
**Language Model Inversion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13647v1)  

---


**ABSTRACT**  
Language models produce a distribution over the next token; can we use this information to recover the prompt tokens? We consider the problem of language model inversion and show that next-token probabilities contain a surprising amount of information about the preceding text. Often we can recover the text in cases where it is hidden from the user, motivating a method for recovering unknown prompts given only the model's current distribution output. We consider a variety of model access scenarios, and show how even without predictions for every token in the vocabulary we can recover the probability vector through search. On Llama-2 7b, our inversion method reconstructs prompts with a BLEU of $59$ and token-level F1 of $78$ and recovers $27\%$ of prompts exactly. Code for reproducing all experiments is available at http://github.com/jxmorris12/vec2text.

{{</citation>}}


### (7/118) Drilling Down into the Discourse Structure with LLMs for Long Document Question Answering (Inderjeet Nair et al., 2023)

{{<citation>}}

Inderjeet Nair, Shwetha Somasundaram, Apoorv Saxena, Koustava Goswami. (2023)  
**Drilling Down into the Discourse Structure with LLMs for Long Document Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: AI, GPT, NLP, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.13565v1)  

---


**ABSTRACT**  
We address the task of evidence retrieval for long document question answering, which involves locating relevant paragraphs within a document to answer a question. We aim to assess the applicability of large language models (LLMs) in the task of zero-shot long document evidence retrieval, owing to their unprecedented performance across various NLP tasks. However, currently the LLMs can consume limited context lengths as input, thus providing document chunks as inputs might overlook the global context while missing out on capturing the inter-segment dependencies. Moreover, directly feeding the large input sets can incur significant computational costs, particularly when processing the entire document (and potentially incurring monetary expenses with enterprise APIs like OpenAI's GPT variants). To address these challenges, we propose a suite of techniques that exploit the discourse structure commonly found in documents. By utilizing this structure, we create a condensed representation of the document, enabling a more comprehensive understanding and analysis of relationships between different parts. We retain $99.6\%$ of the best zero-shot approach's performance, while processing only $26\%$ of the total tokens used by the best approach in the information seeking evidence retrieval setup. We also show how our approach can be combined with \textit{self-ask} reasoning agent to achieve best zero-shot performance in complex multi-hop question answering, just $\approx 4\%$ short of zero-shot performance using gold evidence.

{{</citation>}}


### (8/118) LM-Cocktail: Resilient Tuning of Language Models via Model Merging (Shitao Xiao et al., 2023)

{{<citation>}}

Shitao Xiao, Zheng Liu, Peitian Zhang, Xingrun Xing. (2023)  
**LM-Cocktail: Resilient Tuning of Language Models via Model Merging**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13534v2)  

---


**ABSTRACT**  
The pre-trained language models are continually fine-tuned to better support downstream applications. However, this operation may result in significant performance degeneration on general tasks beyond the targeted domain. To overcome this problem, we propose a novel method which enables the fine-tuned model to stay resilient in general perspectives. Our method is conducted in the form of model merging (namely LM-Cocktail), where the fine-tuned language model is merged with the pre-trained base model or the peer models from other domains through weighted average. Despite simplicity, LM-Cocktail is surprisingly effective: the resulted model is able to achieve a strong empirical performance in the whole scope of general tasks while preserving a superior capacity in its targeted domain. We conduct comprehensive experiments with LLama and BGE model on popular benchmarks, including FLAN, MMLU, MTEB, whose results validate the efficacy of our proposed method. The code and checkpoints are available at https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail.

{{</citation>}}


### (9/118) Machine Translation to Control Formality Features in the Target Language (Harshita Tyagi et al., 2023)

{{<citation>}}

Harshita Tyagi, Prashasta Jung, Hyowon Lee. (2023)  
**Machine Translation to Control Formality Features in the Target Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs-LG, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.13475v1)  

---


**ABSTRACT**  
Formality plays a significant role in language communication, especially in low-resource languages such as Hindi, Japanese and Korean. These languages utilise formal and informal expressions to convey messages based on social contexts and relationships. When a language translation technique is used to translate from a source language that does not pertain the formality (e.g. English) to a target language that does, there is a missing information on formality that could be a challenge in producing an accurate outcome. This research explores how this issue should be resolved when machine learning methods are used to translate from English to languages with formality, using Hindi as the example data. This was done by training a bilingual model in a formality-controlled setting and comparing its performance with a pre-trained multilingual model in a similar setting. Since there are not a lot of training data with ground truth, automated annotation techniques were employed to increase the data size. The primary modeling approach involved leveraging transformer models, which have demonstrated effectiveness in various natural language processing tasks. We evaluate the official formality accuracy(ACC) by comparing the predicted masked tokens with the ground truth. This metric provides a quantitative measure of how well the translations align with the desired outputs. Our study showcases a versatile translation strategy that considers the nuances of formality in the target language, catering to diverse language communication needs and scenarios.

{{</citation>}}


### (10/118) Complexity-Guided Curriculum Learning for Text Graphs (Nidhi Vakil et al., 2023)

{{<citation>}}

Nidhi Vakil, Hadi Amiri. (2023)  
**Complexity-Guided Curriculum Learning for Text Graphs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.13472v1)  

---


**ABSTRACT**  
Curriculum learning provides a systematic approach to training. It refines training progressively, tailors training to task requirements, and improves generalization through exposure to diverse examples. We present a curriculum learning approach that builds on existing knowledge about text and graph complexity formalisms for training with text graph data. The core part of our approach is a novel data scheduler, which employs "spaced repetition" and complexity formalisms to guide the training process. We demonstrate the effectiveness of the proposed approach on several text graph tasks and graph neural network architectures. The proposed model gains more and uses less data; consistently prefers text over graph complexity indices throughout training, while the best curricula derived from text and graph complexity indices are equally effective; and it learns transferable curricula across GNN models and datasets. In addition, we find that both node-level (local) and graph-level (global) graph complexity indices, as well as shallow and traditional text complexity indices play a crucial role in effective curriculum learning.

{{</citation>}}


### (11/118) Fact-based Court Judgment Prediction (Shubham Kumar Nigam et al., 2023)

{{<citation>}}

Shubham Kumar Nigam, Aniket Deroy. (2023)  
**Fact-based Court Judgment Prediction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2311.13350v1)  

---


**ABSTRACT**  
This extended abstract extends the research presented in "ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation" \cite{malik-etal-2021-ildc}, focusing on fact-based judgment prediction within the context of Indian legal documents. We introduce two distinct problem variations: one based solely on facts, and another combining facts with rulings from lower courts (RLC). Our research aims to enhance early-phase case outcome prediction, offering significant benefits to legal professionals and the general public. The results, however, indicated a performance decline compared to the original ILDC for CJPE study, even after implementing various weightage schemes in our DELSumm algorithm. Additionally, using only facts for legal judgment prediction with different transformer models yielded results inferior to the state-of-the-art outcomes reported in the "ILDC for CJPE" study.

{{</citation>}}


### (12/118) Mitigating Large Language Model Hallucinations via Autonomous Knowledge Graph-based Retrofitting (Xinyan Guan et al., 2023)

{{<citation>}}

Xinyan Guan, Yanjiang Liu, Hongyu Lin, Yaojie Lu, Ben He, Xianpei Han, Le Sun. (2023)  
**Mitigating Large Language Model Hallucinations via Autonomous Knowledge Graph-based Retrofitting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.13314v1)  

---


**ABSTRACT**  
Incorporating factual knowledge in knowledge graph is regarded as a promising approach for mitigating the hallucination of large language models (LLMs). Existing methods usually only use the user's input to query the knowledge graph, thus failing to address the factual hallucination generated by LLMs during its reasoning process. To address this problem, this paper proposes Knowledge Graph-based Retrofitting (KGR), a new framework that incorporates LLMs with KGs to mitigate factual hallucination during the reasoning process by retrofitting the initial draft responses of LLMs based on the factual knowledge stored in KGs. Specifically, KGR leverages LLMs to extract, select, validate, and retrofit factual statements within the model-generated responses, which enables an autonomous knowledge verifying and refining procedure without any additional manual efforts. Experiments show that KGR can significantly improve the performance of LLMs on factual QA benchmarks especially when involving complex reasoning processes, which demonstrates the necessity and effectiveness of KGR in mitigating hallucination and enhancing the reliability of LLMs.

{{</citation>}}


### (13/118) Enhancing Summarization Performance through Transformer-Based Prompt Engineering in Automated Medical Reporting (Daphne van Zandvoort et al., 2023)

{{<citation>}}

Daphne van Zandvoort, Laura Wiersema, Tom Huibers, Sandra van Dulmen, Sjaak Brinkkemper. (2023)  
**Enhancing Summarization Performance through Transformer-Based Prompt Engineering in Automated Medical Reporting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Summarization, Transformer  
[Paper Link](http://arxiv.org/abs/2311.13274v1)  

---


**ABSTRACT**  
Customized medical prompts enable Large Language Models (LLM) to effectively address medical dialogue summarization. The process of medical reporting is often time-consuming for healthcare professionals. Implementing medical dialogue summarization techniques presents a viable solution to alleviate this time constraint by generating automated medical reports. The effectiveness of LLMs in this process is significantly influenced by the formulation of the prompt, which plays a crucial role in determining the quality and relevance of the generated reports. In this research, we used a combination of two distinct prompting strategies, known as shot prompting and pattern prompting to enhance the performance of automated medical reporting. The evaluation of the automated medical reports is carried out using the ROUGE score and a human evaluation with the help of an expert panel. The two-shot prompting approach in combination with scope and domain context outperforms other methods and achieves the highest score when compared to the human reference set by a general practitioner. However, the automated reports are approximately twice as long as the human references, due to the addition of both redundant and relevant statements that are added to the report.

{{</citation>}}


### (14/118) Comparative Experimentation of Accuracy Metrics in Automated Medical Reporting: The Case of Otitis Consultations (Wouter Faber et al., 2023)

{{<citation>}}

Wouter Faber, Renske Eline Bootsma, Tom Huibers, Sandra van Dulmen, Sjaak Brinkkemper. (2023)  
**Comparative Experimentation of Accuracy Metrics in Automated Medical Reporting: The Case of Otitis Consultations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13273v1)  

---


**ABSTRACT**  
Generative Artificial Intelligence (AI) can be used to automatically generate medical reports based on transcripts of medical consultations. The aim is to reduce the administrative burden that healthcare professionals face. The accuracy of the generated reports needs to be established to ensure their correctness and usefulness. There are several metrics for measuring the accuracy of AI generated reports, but little work has been done towards the application of these metrics in medical reporting. A comparative experimentation of 10 accuracy metrics has been performed on AI generated medical reports against their corresponding General Practitioner's (GP) medical reports concerning Otitis consultations. The number of missing, incorrect, and additional statements of the generated reports have been correlated with the metric scores. In addition, we introduce and define a Composite Accuracy Score which produces a single score for comparing the metrics within the field of automated medical reporting. Findings show that based on the correlation study and the Composite Accuracy Score, the ROUGE-L and Word Mover's Distance metrics are the preferred metrics, which is not in line with previous work. These findings help determine the accuracy of an AI generated medical report, which aids the development of systems that generate medical reports for GPs to reduce the administrative burden.

{{</citation>}}


### (15/118) @ve: A Chatbot for Latin (Oliver Bendel et al., 2023)

{{<citation>}}

Oliver Bendel, Karim N'diaye. (2023)  
**@ve: A Chatbot for Latin**  

---
Primary Category: cs.CL  
Categories: I-2; K-3, cs-AI, cs-CL, cs-RO, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.14741v1)  

---


**ABSTRACT**  
Dead, extinct, and endangered languages have been preserved primarily through audio conservation and the collection and digitization of scripts and have been promoted through targeted language acquisition efforts. Another possibility would be to build conversational agents that can master these languages. This would provide an artificial, active conversational partner which has knowledge of the vocabulary and grammar, and one learns with it in a different way. The chatbot @ve, with which one can communicate in Latin, was developed in 2022/2023 based on GPT-3.0. It was additionally equipped with a manually created knowledge base. After conceptual groundwork, this paper presents the preparation and implementation of the project. In addition, it summarizes the test that a Latin expert conducted with the chatbot. A critical discussion elaborates advantages and disadvantages. @ve could be a new tool for teaching Latin in a memorable and entertaining way through dialogue. However, the present implementation is still too prone to glitches for stand-alone use - i.e., without the accompaniment of a teacher. The use of GPT-4 could be a solution as well as the extension of the knowledge base. In conclusion, it can be argued that conversational agents are an innovative approach to promoting and preserving languages.

{{</citation>}}


### (16/118) AutoKG: Efficient Automated Knowledge Graph Generation for Language Models (Bohan Chen et al., 2023)

{{<citation>}}

Bohan Chen, Andrea L. Bertozzi. (2023)  
**AutoKG: Efficient Automated Knowledge Graph Generation for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2311.14740v1)  

---


**ABSTRACT**  
Traditional methods of linking large language models (LLMs) to knowledge bases via the semantic similarity search often fall short of capturing complex relational dynamics. To address these limitations, we introduce AutoKG, a lightweight and efficient approach for automated knowledge graph (KG) construction. For a given knowledge base consisting of text blocks, AutoKG first extracts keywords using a LLM and then evaluates the relationship weight between each pair of keywords using graph Laplace learning. We employ a hybrid search scheme combining vector similarity and graph-based associations to enrich LLM responses. Preliminary experiments demonstrate that AutoKG offers a more comprehensive and interconnected knowledge retrieval mechanism compared to the semantic similarity search, thereby enhancing the capabilities of LLMs in generating more insightful and relevant outputs.

{{</citation>}}


### (17/118) On the Calibration of Large Language Models and Alignment (Chiwei Zhu et al., 2023)

{{<citation>}}

Chiwei Zhu, Benfeng Xu, Quan Wang, Yongdong Zhang, Zhendong Mao. (2023)  
**On the Calibration of Large Language Models and Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.13240v1)  

---


**ABSTRACT**  
As large language models attract increasing attention and find widespread application, concurrent challenges of reliability also arise at the same time. Confidence calibration, an effective analysis method for gauging the reliability of deep models, serves as a crucial tool for assessing and improving their reliability. However, such investigation has been comparatively underexplored. In this work, we conduct a systematic examination of the calibration of aligned language models throughout the entire construction process, including pretraining and alignment training. At each stage, we investigate how different training settings, such as parameter scales and training data, affect model calibration. To thoroughly assess model calibration, we evaluate models on three most concerned aspects: generation, factuality and understanding. Our work sheds light on whether popular LLMs are well-calibrated and how the training process influences model calibration.

{{</citation>}}


### (18/118) Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus (Tianhang Zhang et al., 2023)

{{<citation>}}

Tianhang Zhang, Lin Qiu, Qipeng Guo, Cheng Deng, Yue Zhang, Zheng Zhang, Chenghu Zhou, Xinbing Wang, Luoyi Fu. (2023)  
**Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.13230v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have gained significant popularity for their impressive performance across diverse fields. However, LLMs are prone to hallucinate untruthful or nonsensical outputs that fail to meet user expectations in many real-world applications. Existing works for detecting hallucinations in LLMs either rely on external knowledge for reference retrieval or require sampling multiple responses from the LLM for consistency verification, making these methods costly and inefficient. In this paper, we propose a novel reference-free, uncertainty-based method for detecting hallucinations in LLMs. Our approach imitates human focus in factuality checking from three aspects: 1) focus on the most informative and important keywords in the given text; 2) focus on the unreliable tokens in historical context which may lead to a cascade of hallucinations; and 3) focus on the token properties such as token type and token frequency. Experimental results on relevant datasets demonstrate the effectiveness of our proposed method, which achieves state-of-the-art performance across all the evaluation metrics and eliminates the need for additional information.

{{</citation>}}


### (19/118) Towards Better Parameter-Efficient Fine-Tuning for Large Language Models: A Position Paper (Chengyu Wang et al., 2023)

{{<citation>}}

Chengyu Wang, Junbing Yan, Wei Zhang, Jun Huang. (2023)  
**Towards Better Parameter-Efficient Fine-Tuning for Large Language Models: A Position Paper**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.13126v1)  

---


**ABSTRACT**  
This paper delves into the pressing need in Parameter-Efficient Fine-Tuning (PEFT) for Large Language Models (LLMs). While LLMs possess remarkable capabilities, their extensive parameter requirements and associated computational demands hinder their practicality and scalability for real-world applications. Our position paper highlights current states and the necessity of further studying into the topic, and recognizes significant challenges and open issues that must be addressed to fully harness the powerful abilities of LLMs. These challenges encompass novel efficient PEFT architectures, PEFT for different learning settings, PEFT combined with model compression techniques, and the exploration of PEFT for multi-modal LLMs. By presenting this position paper, we aim to stimulate further research and foster discussions surrounding more efficient and accessible PEFT for LLMs.

{{</citation>}}


### (20/118) Detecting out-of-distribution text using topological features of transformer-based language models (Andres Pollano et al., 2023)

{{<citation>}}

Andres Pollano, Anupam Chaudhuri, Anj Simmons. (2023)  
**Detecting out-of-distribution text using topological features of transformer-based language models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL, math-AT  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2311.13102v1)  

---


**ABSTRACT**  
We attempt to detect out-of-distribution (OOD) text samples though applying Topological Data Analysis (TDA) to attention maps in transformer-based language models. We evaluate our proposed TDA-based approach for out-of-distribution detection on BERT, a transformer-based language model, and compare the to a more traditional OOD approach based on BERT CLS embeddings. We found that our TDA approach outperforms the CLS embedding approach at distinguishing in-distribution data (politics and entertainment news articles from HuffPost) from far out-of-domain samples (IMDB reviews), but its effectiveness deteriorates with near out-of-domain (CNN/Dailymail) or same-domain (business news articles from HuffPost) datasets.

{{</citation>}}


### (21/118) Enhancing Logical Reasoning in Large Language Models to Facilitate Legal Applications (Ha-Thanh Nguyen et al., 2023)

{{<citation>}}

Ha-Thanh Nguyen, Wachara Fungwacharakorn, Ken Satoh. (2023)  
**Enhancing Logical Reasoning in Large Language Models to Facilitate Legal Applications**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Legal, Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.13095v1)  

---


**ABSTRACT**  
Language serves as a vehicle for conveying thought, enabling communication among individuals. The ability to distinguish between diverse concepts, identify fairness and injustice, and comprehend a range of legal notions fundamentally relies on logical reasoning. Large Language Models (LLMs) attempt to emulate human language understanding and generation, but their competency in logical reasoning remains limited. This paper seeks to address the philosophical question: How can we effectively teach logical reasoning to LLMs while maintaining a deep understanding of the intricate relationship between language and logic? By focusing on bolstering LLMs' capabilities in logical reasoning, we aim to expand their applicability in law and other logic-intensive disciplines. To this end, we propose a Reinforcement Learning from Logical Feedback (RLLF) approach, which serves as a potential framework for refining LLMs' reasoning capacities. Through RLLF and a revised evaluation methodology, we explore new avenues for research in this domain and contribute to the development of LLMs capable of handling complex legal reasoning tasks while acknowledging the fundamental connection between language and logic.

{{</citation>}}


### (22/118) Positional Description Matters for Transformers Arithmetic (Ruoqi Shen et al., 2023)

{{<citation>}}

Ruoqi Shen, Sébastien Bubeck, Ronen Eldan, Yin Tat Lee, Yuanzhi Li, Yi Zhang. (2023)  
**Positional Description Matters for Transformers Arithmetic**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.14737v1)  

---


**ABSTRACT**  
Transformers, central to the successes in modern Natural Language Processing, often falter on arithmetic tasks despite their vast capabilities --which paradoxically include remarkable coding abilities. We observe that a crucial challenge is their naive reliance on positional information to solve arithmetic problems with a small number of digits, leading to poor performance on larger numbers. Herein, we delve deeper into the role of positional encoding, and propose several ways to fix the issue, either by modifying the positional encoding directly, or by modifying the representation of the arithmetic task to leverage standard positional encoding differently. We investigate the value of these modifications for three tasks: (i) classical multiplication, (ii) length extrapolation in addition, and (iii) addition in natural language context. For (i) we train a small model on a small dataset (100M parameters and 300k samples) with remarkable aptitude in (direct, no scratchpad) 15 digits multiplication and essentially perfect up to 12 digits, while usual training in this context would give a model failing at 4 digits multiplication. In the experiments on addition, we use a mere 120k samples to demonstrate: for (ii) extrapolation from 10 digits to testing on 12 digits numbers while usual training would have no extrapolation, and for (iii) almost perfect accuracy up to 5 digits while usual training would be correct only up to 3 digits (which is essentially memorization with a training set of 120k samples).

{{</citation>}}


## cs.HC (3)



### (23/118) Studying Artist Sentiments around AI-generated Artwork (Safinah Ali et al., 2023)

{{<citation>}}

Safinah Ali, Cynthia Breazeal. (2023)  
**Studying Artist Sentiments around AI-generated Artwork**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-SI, cs.HC  
Keywords: AI, Twitter  
[Paper Link](http://arxiv.org/abs/2311.13725v1)  

---


**ABSTRACT**  
Art created using generated Artificial Intelligence has taken the world by storm and generated excitement for many digital creators and technologists. However, the reception and reaction from artists have been mixed. Concerns about plagiarizing their artworks and styles for datasets and uncertainty around the future of digital art sparked movements in artist communities shunning the use of AI for generating art and protecting artists' rights. Collaborating with these tools for novel creative use cases also sparked hope from some creators. Artists are an integral stakeholder in the rapidly evolving digital creativity industry and understanding their concerns and hopes inform responsible development and use of creativity support tools. In this work, we study artists' sentiments about AI-generated art. We interviewed 7 artists and analyzed public posts from artists on social media platforms Reddit, Twitter and Artstation. We report artists' main concerns and hopes around AI-generated artwork, informing a way forward for inclusive development of these tools.

{{</citation>}}


### (24/118) Panda or not Panda? Understanding Adversarial Attacks with Interactive Visualization (Yuzhe You et al., 2023)

{{<citation>}}

Yuzhe You, Jarvis Tse, Jian Zhao. (2023)  
**Panda or not Panda? Understanding Adversarial Attacks with Interactive Visualization**  

---
Primary Category: cs.HC  
Categories: cs-CV, cs-HC, cs.HC  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2311.13656v1)  

---


**ABSTRACT**  
Adversarial machine learning (AML) studies attacks that can fool machine learning algorithms into generating incorrect outcomes as well as the defenses against worst-case attacks to strengthen model robustness. Specifically for image classification, it is challenging to understand adversarial attacks due to their use of subtle perturbations that are not human-interpretable, as well as the variability of attack impacts influenced by diverse methodologies, instance differences, and model architectures. Through a design study with AML learners and teachers, we introduce AdvEx, a multi-level interactive visualization system that comprehensively presents the properties and impacts of evasion attacks on different image classifiers for novice AML learners. We quantitatively and qualitatively assessed AdvEx in a two-part evaluation including user studies and expert interviews. Our results show that AdvEx is not only highly effective as a visualization tool for understanding AML mechanisms, but also provides an engaging and enjoyable learning experience, thus demonstrating its overall benefits for AML learners.

{{</citation>}}


### (25/118) Design Recommendations Based on Speech Analysis for Disability-Friendly Interfaces for the Control of a Home Automation Environment (Nadine Vigouroux et al., 2023)

{{<citation>}}

Nadine Vigouroux, Frédéric Vella, Gaëlle Lepage, Éric Campo. (2023)  
**Design Recommendations Based on Speech Analysis for Disability-Friendly Interfaces for the Control of a Home Automation Environment**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2311.13223v1)  

---


**ABSTRACT**  
The objective of this paper is to describe the study on speech interaction mode for home automation control of equipment by impaired people for an inclusive housing. The study is related to the HIP HOPE project concerning a building of 19 inclusive housing units. 7 participants with different types of disabilities were invited to carry out use cases using voice and touch control. Only the results obtained on the voice interaction mode through the Amazon voice assistant are reported here. The results show, according to the type of handicap, the success rates in the speech recognition of the command emitted on the equipment and highlight the errors related to the formulation, the noisy environment, the intelligible speech, the speech segmentation and the bad synchronization of the audio channel opening.

{{</citation>}}


## cs.SE (3)



### (26/118) Nova$^+$: Generative Language Models for Binaries (Nan Jiang et al., 2023)

{{<citation>}}

Nan Jiang, Chengxiao Wang, Kevin Liu, Xiangzhe Xu, Lin Tan, Xiangyu Zhang. (2023)  
**Nova$^+$: Generative Language Models for Binaries**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13721v2)  

---


**ABSTRACT**  
Generative large language models (LLMs) pre-trained on code have shown impressive effectiveness in code generation, program repair, and document analysis. However, existing generative LLMs focus on source code and are not specialized for binaries. There are three main challenges for LLMs to model and learn binary code: hex-decimal values, complex global dependencies, and compiler optimization levels. To bring the benefit of LLMs to the binary domain, we develop Nova and Nova$^+$, which are LLMs pre-trained on binary corpora. Nova is pre-trained with the standard language modeling task, showing significantly better capability on five benchmarks for three downstream tasks: binary code similarity detection (BCSD), binary code translation (BCT), and binary code recovery (BCR), over GPT-3.5 and other existing techniques. We build Nova$^+$ to further boost Nova using two new pre-training tasks, i.e., optimization generation and optimization level prediction, which are designed to learn binary optimization and align equivalent binaries. Nova$^+$ shows overall the best performance for all three downstream tasks on five benchmarks, demonstrating the contributions of the new pre-training tasks.

{{</citation>}}


### (27/118) Naturalness of Attention: Revisiting Attention in Code Language Models (Mootez Saad et al., 2023)

{{<citation>}}

Mootez Saad, Tushar Sharma. (2023)  
**Naturalness of Attention: Revisiting Attention in Code Language Models**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Attention, BERT, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.13508v1)  

---


**ABSTRACT**  
Language models for code such as CodeBERT offer the capability to learn advanced source code representation, but their opacity poses barriers to understanding of captured properties. Recent attention analysis studies provide initial interpretability insights by focusing solely on attention weights rather than considering the wider context modeling of Transformers. This study aims to shed some light on the previously ignored factors of the attention mechanism beyond the attention weights. We conduct an initial empirical study analyzing both attention distributions and transformed representations in CodeBERT. Across two programming languages, Java and Python, we find that the scaled transformation norms of the input better capture syntactic structure compared to attention weights alone. Our analysis reveals characterization of how CodeBERT embeds syntactic code properties. The findings demonstrate the importance of incorporating factors beyond just attention weights for rigorously understanding neural code models. This lays the groundwork for developing more interpretable models and effective uses of attention mechanisms in program analysis.

{{</citation>}}


### (28/118) From Principles to Practice: An Accountability Metrics Catalogue for Managing AI Risks (Boming Xia et al., 2023)

{{<citation>}}

Boming Xia, Qinghua Lu, Liming Zhu, Sung Une Lee, Yue Liu, Zhenchang Xing. (2023)  
**From Principles to Practice: An Accountability Metrics Catalogue for Managing AI Risks**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13158v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI), particularly through the advent of large-scale generative AI (GenAI) models such as Large Language Models (LLMs), has become a transformative element in contemporary technology. While these models have unlocked new possibilities, they simultaneously present significant challenges, such as concerns over data privacy and the propensity to generate misleading or fabricated content. Current frameworks for Responsible AI (RAI) often fall short in providing the granular guidance necessary for tangible application, especially for Accountability-a principle that is pivotal for ensuring transparent and auditable decision-making, bolstering public trust, and meeting increasing regulatory expectations. This study bridges the accountability gap by introducing a comprehensive metrics catalogue, formulated through a systematic multivocal literature review (MLR) that integrates findings from both academic and grey literature. Our catalogue delineates process metrics that underpin procedural integrity, resource metrics that provide necessary tools and frameworks, and product metrics that reflect the outputs of AI systems. This tripartite framework is designed to operationalize Accountability in AI, with a special emphasis on addressing the intricacies of GenAI. The proposed metrics catalogue provides a robust framework for instilling Accountability in AI systems. It offers practical, actionable guidance for organizations, thereby shaping responsible practices in the field.

{{</citation>}}


## cs.AI (14)



### (29/118) Towards More Likely Models for AI Planning (Turgay Caglar et al., 2023)

{{<citation>}}

Turgay Caglar, Sirine Belhaj, Tathagata Chakraborti, Michael Katz, Sarath Sreedharan. (2023)  
**Towards More Likely Models for AI Planning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13720v1)  

---


**ABSTRACT**  
This is the first work to look at the application of large language models (LLMs) for the purpose of model space edits in automated planning tasks. To set the stage for this sangam, we explore two different flavors of model space problems that have been studied in the AI planning literature and explore the effect of an LLM on those tasks. We empirically demonstrate how the performance of an LLM contrasts with combinatorial search (CS) - an approach that has been traditionally used to solve model space tasks in planning, both with the LLM in the role of a standalone model space reasoner as well as in the role of a statistical signal in concert with the CS approach as part of a two-stage process. Our experiments show promising results suggesting further forays of LLMs into the exciting world of model space reasoning for planning tasks in the future.

{{</citation>}}


### (30/118) Data Acquisition: A New Frontier in Data-centric AI (Lingjiao Chen et al., 2023)

{{<citation>}}

Lingjiao Chen, Bilge Acun, Newsha Ardalani, Yifan Sun, Feiyang Kang, Hanrui Lyu, Yongchan Kwon, Ruoxi Jia, Carole-Jean Wu, Matei Zaharia, James Zou. (2023)  
**Data Acquisition: A New Frontier in Data-centric AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13712v1)  

---


**ABSTRACT**  
As Machine Learning (ML) systems continue to grow, the demand for relevant and comprehensive datasets becomes imperative. There is limited study on the challenges of data acquisition due to ad-hoc processes and lack of consistent methodologies. We first present an investigation of current data marketplaces, revealing lack of platforms offering detailed information about datasets, transparent pricing, standardized data formats. With the objective of inciting participation from the data-centric AI community, we then introduce the DAM challenge, a benchmark to model the interaction between the data providers and acquirers. The benchmark was released as a part of DataPerf. Our evaluation of the submitted strategies underlines the need for effective data acquisition strategies in ML.

{{</citation>}}


### (31/118) Physical Reasoning and Object Planning for Household Embodied Agents (Ayush Agrawal et al., 2023)

{{<citation>}}

Ayush Agrawal, Raghav Prabhakar, Anirudh Goyal, Dianbo Liu. (2023)  
**Physical Reasoning and Object Planning for Household Embodied Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.13577v1)  

---


**ABSTRACT**  
In this study, we explore the sophisticated domain of task planning for robust household embodied agents, with a particular emphasis on the intricate task of selecting substitute objects. We introduce the CommonSense Object Affordance Task (COAT), a novel framework designed to analyze reasoning capabilities in commonsense scenarios. This approach is centered on understanding how these agents can effectively identify and utilize alternative objects when executing household tasks, thereby offering insights into the complexities of practical decision-making in real-world environments.Drawing inspiration from human decision-making, we explore how large language models tackle this challenge through three meticulously crafted commonsense question-and-answer datasets, featuring refined rules and human annotations. Our evaluation of state-of-the-art language models on these datasets sheds light on three pivotal considerations: 1) aligning an object's inherent utility with the task at hand, 2) navigating contextual dependencies (societal norms, safety, appropriateness, and efficiency), and 3) accounting for the current physical state of the object. To maintain accessibility, we introduce five abstract variables reflecting an object's physical condition, modulated by human insights to simulate diverse household scenarios. Our contributions include insightful Object-Utility mappings addressing the first consideration and two extensive QA datasets (15k and 130k questions) probing the intricacies of contextual dependencies and object states. The datasets, along with our findings, are accessible at: \url{https://github.com/com-phy-affordance/COAT}. This research not only advances our understanding of physical commonsense reasoning in language models but also paves the way for future improvements in household agent intelligence.

{{</citation>}}


### (32/118) Speak Like a Native: Prompting Large Language Models in a Native Style (Zhicheng Yang et al., 2023)

{{<citation>}}

Zhicheng Yang, Yiwei Wang, Yinya Huang, Jing Xiong, Xiaodan Liang, Jing Tang. (2023)  
**Speak Like a Native: Prompting Large Language Models in a Native Style**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13538v1)  

---


**ABSTRACT**  
Existing work has found that the prompt engineering heavily influences the performance of large language models (LLMs). Chain-of-thought (CoT), as a popular prompt engineering technique, prompted LLMs using in-context examples with reasoning steps. In current studies, the few-shot examples of CoT are generally handcrafted by humans. However, how the text style of in-context examples influence the outputs of LLMs still remains under-explored. This paper presents a novel and effective approach, named \textbf{AlignCoT}, to improve the reasoning capability of LLMs by aligning the in-context examples with the native style of LLMs. ``Native'' refers to the inherent characteristic style of LLMs which can be probed by original zero-shot scenarios. AlignCoT is orthogonal to other prompt engineering methods, making it easy to combine with state-of-the-art techniques to further improve the LLMs' performance. We conduct extensive and comprehensive experiments on several benchmarks. The empirical results demonstrate that our AlignCoTsignificantly improves performance over the carefully handcrafted in-context examples. For instance, with GPT-3.5-turbo, we observed a +2.5\% improvement on GSM8K. Furthermore, our AlignCoT consistently improve the performance when combined with other state-of-the-art prompt engineering methods. The source code and dataset will be available at \href{https://github.com/yangzhch6/AlignCoT}{https://github.com/yangzhch6/AlignCoT}.

{{</citation>}}


### (33/118) Generation of Explanations for Logic Reasoning (Yanyi Pu, 2023)

{{<citation>}}

Yanyi Pu. (2023)  
**Generation of Explanations for Logic Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, GPT, GPT-3.5, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.13455v1)  

---


**ABSTRACT**  
This thesis delves into a fortiori arguments in deductive reasoning, underscoring their relevance in various domains such as law, philosophy, and artificial intelligence. The research is centred on employing GPT-3.5-turbo to automate the analysis of these arguments, with a focus on understanding intricate reasoning processes, generating clear and coherent explanations, and creating novel arguments. The methodology encompasses a series of tasks including detailed reasoning, interpretation, and the augmentation of a fortiori arguments. It involves meticulously identifying these arguments in diverse contexts, differentiating comparative elements, and categorizing them based on their logical structure.   Extensive experiments reveals the challenges encountered by GPT-3.5-turbo in accurately detecting and classifying a fortiori arguments. Nevertheless, the model demonstrates a performance that rivals specialized models, particularly in extracting key components and interpreting underlying properties. The integration of external information into the model's processing significantly elevates the quality of the generated explanations. Additionally, the model exhibits a noteworthy capability in augmenting arguments, thus contributing to the enrichment of the data set.   Despite facing certain limitations, this thesis makes significant contributions to the fields of artificial intelligence and logical reasoning. It introduces novel methodologies, establishes a rigorous evaluation framework, and provides deep insights that set the stage for future advancements in automated logical reasoning. The findings and methodologies presented herein not only underscore the potential of AI in complex reasoning tasks but also highlight areas for future research and development.

{{</citation>}}


### (34/118) Deriving Comprehensible Theories from Probabilistic Circuits (Sieben Bocklandt et al., 2023)

{{<citation>}}

Sieben Bocklandt, Wannes Meert, Koen Vanderstraeten, Wouter Pijpops, Kurt Jaspers. (2023)  
**Deriving Comprehensible Theories from Probabilistic Circuits**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Pruning  
[Paper Link](http://arxiv.org/abs/2311.13379v1)  

---


**ABSTRACT**  
The field of Explainable AI (XAI) is seeking to shed light on the inner workings of complex AI models and uncover the rationale behind their decisions. One of the models gaining attention are probabilistic circuits (PCs), which are a general and unified framework for tractable probabilistic models that support efficient computation of various probabilistic queries. Probabilistic circuits guarantee inference that is polynomial in the size of the circuit. In this paper, we improve the explainability of probabilistic circuits by computing a comprehensible, readable logical theory that covers the high-density regions generated by a PC. To achieve this, pruning approaches based on generative significance are used in a new method called PUTPUT (Probabilistic circuit Understanding Through Pruning Underlying logical Theories). The method is applied to a real world use case where music playlists are automatically generated and expressed as readable (database) queries. Evaluation shows that this approach can effectively produce a comprehensible logical theory that describes the high-density regions of a PC and outperforms state of the art methods when exploring the performance-comprehensibility trade-off.

{{</citation>}}


### (35/118) Large Language Model is a Good Policy Teacher for Training Reinforcement Learning Agents (Zihao Zhou et al., 2023)

{{<citation>}}

Zihao Zhou, Bin Hu, Pu Zhang, Chenyang Zhao, Bin Liu. (2023)  
**Large Language Model is a Good Policy Teacher for Training Reinforcement Learning Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.13373v2)  

---


**ABSTRACT**  
Recent studies have shown that Large Language Models (LLMs) can be utilized for solving complex sequential decision-making tasks by providing high-level instructions. However, LLM-based agents face limitations in real-time dynamic environments due to their lack of specialization in solving specific target problems. Moreover, the deployment of such LLM-based agents is both costly and time-consuming in practical scenarios. In this paper, we introduce a novel framework that addresses these challenges by training a smaller scale specialized student agent using instructions from an LLM-based teacher agent. By leveraging guided actions provided by the teachers, the prior knowledge of the LLM is distilled into the local student model. Consequently, the student agent can be trained with significantly less data. Furthermore, subsequent training with environment feedback empowers the student agents to surpass the capabilities of their teachers. We conducted experiments on three challenging MiniGrid environments to evaluate the effectiveness of our framework. The results demonstrate that our approach enhances sample efficiency and achieves superior performance compared to baseline methods.

{{</citation>}}


### (36/118) Applying Large Language Models to Power Systems: Potential Security Threats (Jiaqi Ruan et al., 2023)

{{<citation>}}

Jiaqi Ruan, Gaoqi Liang, Huan Zhao, Guolong Liu, Jing Qiu, Junhua Zhao, Zhao Xu, Fushuan Wen, Zhao Yang Dong. (2023)  
**Applying Large Language Models to Power Systems: Potential Security Threats**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-SY, cs.AI, eess-SY  
Keywords: Language Model, Security  
[Paper Link](http://arxiv.org/abs/2311.13361v1)  

---


**ABSTRACT**  
Applying large language models (LLMs) to power systems presents a promising avenue for enhancing decision-making and operational efficiency. However, this action may also incur potential security threats, which have not been fully recognized so far. To this end, this letter analyzes potential threats incurred by applying LLMs to power systems, emphasizing the need for urgent research and development of countermeasures.

{{</citation>}}


### (37/118) Quantum learning and essential cognition under the traction of meta-characteristics in an open world (Jin Wang et al., 2023)

{{<citation>}}

Jin Wang, Changlin Song. (2023)  
**Quantum learning and essential cognition under the traction of meta-characteristics in an open world**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13335v1)  

---


**ABSTRACT**  
Artificial intelligence has made significant progress in the Close World problem, being able to accurately recognize old knowledge through training and classification. However, AI faces significant challenges in the Open World problem, as it involves a new and unknown exploration journey. AI is not inherently proactive in exploration, and its challenge lies in not knowing how to approach and adapt to the unknown world. How do humans acquire knowledge of the unknown world. Humans identify new knowledge through intrinsic cognition. In the process of recognizing new colors, the cognitive cues are different from known color features and involve hue, saturation, brightness, and other characteristics. When AI encounters objects with different features in the new world, it faces another challenge: where are the distinguishing features between influential features of new and old objects? AI often mistakes a new world's brown bear for a known dog because it has not learned the differences in feature distributions between knowledge systems. This is because things in the new and old worlds have different units and dimensions for their features. This paper proposes an open-world model and elemental feature system that focuses on fundamentally recognizing the distribution differences in objective features between the new and old worlds. The quantum tunneling effect of learning ability in the new and old worlds is realized through the tractive force of meta-characteristic. The outstanding performance of the model system in learning new knowledge (using pedestrian re-identification datasets as an example) demonstrates that AI has acquired the ability to recognize the new world with an accuracy of $96.71\%$ at most and has gained the capability to explore new knowledge, similar to humans.

{{</citation>}}


### (38/118) The Rise of Creative Machines: Exploring the Impact of Generative AI (Saad Shaikh et al., 2023)

{{<citation>}}

Saad Shaikh, Rajat bendre, Sakshi Mhaske. (2023)  
**The Rise of Creative Machines: Exploring the Impact of Generative AI**  

---
Primary Category: cs.AI  
Categories: I-2-7, cs-AI, cs.AI  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.13262v1)  

---


**ABSTRACT**  
This study looks at how generative artificial intelligence (AI) can revolutionize marketing, product development, and research. It discusses the latest developments in the field, easy-to-use resources, and moral and social hazards. In addition to addressing mitigating techniques for issues like prejudice and disinformation, the debate emphasizes the significance of responsible development through continual stakeholder communication and ethical principles.

{{</citation>}}


### (39/118) Artificial Intelligence in the Service of Entrepreneurial Finance: Knowledge Structure and the Foundational Algorithmic Paradigm (Robert Kudelić et al., 2023)

{{<citation>}}

Robert Kudelić, Tamara Šmaguc, Sherry Robinson. (2023)  
**Artificial Intelligence in the Service of Entrepreneurial Finance: Knowledge Structure and the Foundational Algorithmic Paradigm**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2311.13213v1)  

---


**ABSTRACT**  
While the application of Artificial Intelligence in Finance has a long tradition, its potential in Entrepreneurship has been intensively explored only recently. In this context, Entrepreneurial Finance is a particularly fertile ground for future Artificial Intelligence proliferation. To support the latter, the study provides a bibliometric review of Artificial Intelligence applications in (1) entrepreneurial finance literature, and (2) corporate finance literature with implications for Entrepreneurship. Rigorous search and screening procedures of the scientific database Web of Science Core Collection resulted in the identification of 1890 relevant journal articles subjected to analysis. The bibliometric analysis gives a rich insight into the knowledge field's conceptual, intellectual, and social structure, indicating nascent and underdeveloped research directions. As far as we were able to identify, this is the first study to map and bibliometrically analyze the academic field concerning the relationship between Artificial Intelligence, Entrepreneurship, and Finance, and the first review that deals with Artificial Intelligence methods in Entrepreneurship. According to the results, Artificial Neural Network, Deep Neural Network and Support Vector Machine are highly represented in almost all identified topic niches. At the same time, applying Topic Modeling, Fuzzy Neural Network and Growing Hierarchical Self-organizing Map is quite rare. As an element of the research, and before final remarks, the article deals as well with a discussion of certain gaps in the relationship between Computer Science and Economics. These gaps do represent problems in the application of Artificial Intelligence in Economic Science. As a way to at least in part remedy this situation, the foundational paradigm and the bespoke demonstration of the Monte Carlo randomized algorithm are presented.

{{</citation>}}


### (40/118) Multimodal Large Language Models: A Survey (Jiayang Wu et al., 2023)

{{<citation>}}

Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, Philip S. Yu. (2023)  
**Multimodal Large Language Models: A Survey**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.13165v1)  

---


**ABSTRACT**  
The exploration of multimodal language models integrates multiple data types, such as images, text, language, audio, and other heterogeneity. While the latest large language models excel in text-based tasks, they often struggle to understand and process other data types. Multimodal models address this limitation by combining various modalities, enabling a more comprehensive understanding of diverse data. This paper begins by defining the concept of multimodal and examining the historical development of multimodal algorithms. Furthermore, we introduce a range of multimodal products, focusing on the efforts of major technology companies. A practical guide is provided, offering insights into the technical aspects of multimodal models. Moreover, we present a compilation of the latest algorithms and commonly used datasets, providing researchers with valuable resources for experimentation and evaluation. Lastly, we explore the applications of multimodal models and discuss the challenges associated with their development. By addressing these aspects, this paper aims to facilitate a deeper understanding of multimodal models and their potential in various domains.

{{</citation>}}


### (41/118) Large Language Models in Education: Vision and Opportunities (Wensheng Gan et al., 2023)

{{<citation>}}

Wensheng Gan, Zhenlian Qi, Jiayang Wu, Jerry Chun-Wei Lin. (2023)  
**Large Language Models in Education: Vision and Opportunities**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.13160v1)  

---


**ABSTRACT**  
With the rapid development of artificial intelligence technology, large language models (LLMs) have become a hot research topic. Education plays an important role in human social development and progress. Traditional education faces challenges such as individual student differences, insufficient allocation of teaching resources, and assessment of teaching effectiveness. Therefore, the applications of LLMs in the field of digital/smart education have broad prospects. The research on educational large models (EduLLMs) is constantly evolving, providing new methods and approaches to achieve personalized learning, intelligent tutoring, and educational assessment goals, thereby improving the quality of education and the learning experience. This article aims to investigate and summarize the application of LLMs in smart education. It first introduces the research background and motivation of LLMs and explains the essence of LLMs. It then discusses the relationship between digital education and EduLLMs and summarizes the current research status of educational large models. The main contributions are the systematic summary and vision of the research background, motivation, and application of large models for education (LLM4Edu). By reviewing existing research, this article provides guidance and insights for educators, researchers, and policy-makers to gain a deep understanding of the potential and challenges of LLM4Edu. It further provides guidance for further advancing the development and application of LLM4Edu, while still facing technical, ethical, and practical challenges requiring further research and exploration.

{{</citation>}}


### (42/118) Building the Future of Responsible AI: A Pattern-Oriented Reference Architecture for Designing Large Language Model based Agents (Qinghua Lu et al., 2023)

{{<citation>}}

Qinghua Lu, Liming Zhu, Xiwei Xu, Zhenchang Xing, Stefan Harrer, Jon Whittle. (2023)  
**Building the Future of Responsible AI: A Pattern-Oriented Reference Architecture for Designing Large Language Model based Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SE, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13148v1)  

---


**ABSTRACT**  
Large language models (LLMs) have been widely recognized as transformative technology due to their capabilities to understand and generate natural language text, including plans with some limited reasoning capabilities. LLM-based agents derive their autonomy from the capabilities of LLMs, which enable them to autonomously break down the given goal into a set of manageable tasks and orchestrate the task execution to fulfill the goal. Despite the huge efforts put into building LLM-based autonomous agents, the architecture design of the agents has not yet been systematically explored. Also, while there are significant benefits of using autonomous agents for planning and execution, there are serious considerations regarding responsible AI related software quality attributes, such as security and accountability. Therefore, this paper presents a pattern-oriented reference architecture that serves as architecture design guidelines and enables responsible-AI-by-design when designing LLM-based autonomous agents. We evaluate the completeness and utility of the proposed reference architecture by mapping it to the architecture of two real-world agents.

{{</citation>}}


## cs.CV (27)



### (43/118) Importance of Feature Extraction in the Calculation of Fréchet Distance for Medical Imaging (McKell Woodland et al., 2023)

{{<citation>}}

McKell Woodland, Mais Al Taie, Jessica Albuquerque Marques Silva, Mohamed Eltaher, Frank Mohn, Alexander Shieh, Austin Castelo, Suprateek Kundu, Joshua P. Yung, Ankit B. Patel, Kristy K. Brock. (2023)  
**Importance of Feature Extraction in the Calculation of Fréchet Distance for Medical Imaging**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2311.13717v1)  

---


**ABSTRACT**  
Fr\'echet Inception Distance is a widely used metric for evaluating synthetic image quality that utilizes an ImageNet-trained InceptionV3 network as a feature extractor. However, its application in medical imaging lacks a standard feature extractor, leading to biased and inconsistent comparisons. This study aimed to compare state-of-the-art feature extractors for computing Fr\'echet Distances (FDs) in medical imaging. A StyleGAN2 network was trained with data augmentation techniques tailored for limited data domains on datasets comprising three medical imaging modalities and four anatomical locations. Human evaluation of generative quality (via a visual Turing test) was compared to FDs calculated using ImageNet-trained InceptionV3, ResNet50, SwAV, DINO, and Swin Transformer architectures, in addition to an InceptionV3 network trained on a large medical dataset, RadImageNet. All ImageNet-based extractors were consistent with each other, but only SwAV was significantly correlated with medical expert judgment. The RadImageNet-based FD showed volatility and lacked correlation with human judgment. Caution is advised when using medical image-trained extraction networks in the FD calculation. These networks should be rigorously evaluated on the imaging modality under consideration and publicly released. ImageNet-based extractors, while imperfect, are consistent and widely understood. Training extraction networks with SwAV is a promising approach for synthetic medical image evaluation.

{{</citation>}}


### (44/118) DiverseNet: Decision Diversified Semi-supervised Semantic Segmentation Networks for Remote Sensing Imagery (Wanli Ma et al., 2023)

{{<citation>}}

Wanli Ma, Oktay Karakus, Paul L. Rosin. (2023)  
**DiverseNet: Decision Diversified Semi-supervised Semantic Segmentation Networks for Remote Sensing Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.13716v1)  

---


**ABSTRACT**  
Semi-supervised learning is designed to help reduce the cost of the manual labelling process by exploiting the use of useful features from a large quantity of unlabelled data during training. Since pixel-level manual labelling in large-scale remote sensing imagery is expensive, semi-supervised learning becomes an appropriate solution to this. However, most of the existing semi-supervised learning methods still lack efficient perturbation methods to promote diversity of features and the precision of pseudo labels during training. In order to fill this gap, we propose DiverseNet architectures which explore multi-head and multi-model semi-supervised learning algorithms by simultaneously promoting precision and diversity during training. The two proposed methods of DiverseNet, namely the DiverseHead and DiverseModel, achieve the highest semantic segmentation performance in four widely utilised remote sensing imagery data sets compared to state-of-the-art semi-supervised learning methods. Meanwhile, the proposed DiverseHead architecture is relatively lightweight in terms of parameter space compared to the state-of-the-art methods whilst reaching high-performance results for all the tested data sets.

{{</citation>}}


### (45/118) BenthIQ: a Transformer-Based Benthic Classification Model for Coral Restoration (Rupa Kurinchi-Vendhan et al., 2023)

{{<citation>}}

Rupa Kurinchi-Vendhan, Drew Gray, Elijah Cole. (2023)  
**BenthIQ: a Transformer-Based Benthic Classification Model for Coral Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.13661v1)  

---


**ABSTRACT**  
Coral reefs are vital for marine biodiversity, coastal protection, and supporting human livelihoods globally. However, they are increasingly threatened by mass bleaching events, pollution, and unsustainable practices with the advent of climate change. Monitoring the health of these ecosystems is crucial for effective restoration and management. Current methods for creating benthic composition maps often compromise between spatial coverage and resolution. In this paper, we introduce BenthIQ, a multi-label semantic segmentation network designed for high-precision classification of underwater substrates, including live coral, algae, rock, and sand. Although commonly deployed CNNs are limited in learning long-range semantic information, transformer-based models have recently achieved state-of-the-art performance in vision tasks such as object detection and image classification. We integrate the hierarchical Swin Transformer as the backbone of a U-shaped encoder-decoder architecture for local-global semantic feature learning. Using a real-world case study in French Polynesia, we demonstrate that our approach outperforms traditional CNN and attention-based models on pixel-wise classification of shallow reef imagery.

{{</citation>}}


### (46/118) Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation (Daichi Horita et al., 2023)

{{<citation>}}

Daichi Horita, Naoto Inoue, Kotaro Kikuchi, Kota Yamaguchi, Kiyoharu Aizawa. (2023)  
**Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.13602v1)  

---


**ABSTRACT**  
Content-aware graphic layout generation aims to automatically arrange visual elements along with a given content, such as an e-commerce product image. In this paper, we argue that the current layout generation approaches suffer from the limited training data for the high-dimensional layout structure. We show that a simple retrieval augmentation can significantly improve the generation quality. Our model, which is named Retrieval-Augmented Layout Transformer (RALF), retrieves nearest neighbor layout examples based on an input image and feeds these results into an autoregressive generator. Our model can apply retrieval augmentation to various controllable generation tasks and yield high-quality layouts within a unified architecture. Our extensive experiments show that RALF successfully generates content-aware layouts in both constrained and unconstrained settings and significantly outperforms the baselines.

{{</citation>}}


### (47/118) Soulstyler: Using Large Language Model to Guide Image Style Transfer for Target Object (Junhao Chen et al., 2023)

{{<citation>}}

Junhao Chen, Peng Rong, Jingbo Sun, Chao Li, Xiang Li, Hongwu Lv. (2023)  
**Soulstyler: Using Large Language Model to Guide Image Style Transfer for Target Object**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, Style Transfer  
[Paper Link](http://arxiv.org/abs/2311.13562v1)  

---


**ABSTRACT**  
Image style transfer occupies an important place in both computer graphics and computer vision. However, most current methods require reference to stylized images and cannot individually stylize specific objects. To overcome this limitation, we propose the "Soulstyler" framework, which allows users to guide the stylization of specific objects in an image through simple textual descriptions. We introduce a large language model to parse the text and identify stylization goals and specific styles. Combined with a CLIP-based semantic visual embedding encoder, the model understands and matches text and image content. We also introduce a novel localized text-image block matching loss that ensures that style transfer is performed only on specified target objects, while non-target regions remain in their original style. Experimental results demonstrate that our model is able to accurately perform style transfer on target objects according to textual descriptions without affecting the style of background regions. Our code will be available at https://github.com/yisuanwang/Soulstyler.

{{</citation>}}


### (48/118) Vamos: Versatile Action Models for Video Understanding (Shijie Wang et al., 2023)

{{<citation>}}

Shijie Wang, Qi Zhao, Minh Quan Do, Nakul Agarwal, Kwonjoon Lee, Chen Sun. (2023)  
**Vamos: Versatile Action Models for Video Understanding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.13627v1)  

---


**ABSTRACT**  
What makes good video representations for video understanding, such as anticipating future activities, or answering video-conditioned questions? While earlier approaches focus on end-to-end learning directly from video pixels, we propose to revisit text-based representations, such as discrete action labels, or free-form video captions, which are interpretable and can be directly consumed by large language models (LLMs). Intuitively, different video understanding tasks may require representations that are complementary and at different granularities. To this end, we propose versatile action models (Vamos), a learning framework powered by a large language model as the "reasoner", and can flexibly leverage visual embeddings, action labels, and free-form descriptions extracted from videos as its input. We evaluate Vamos on four complementary video understanding benchmarks, Ego4D, Next-QA, IntentQA, and EgoSchema, on its capability to model temporal dynamics, encode visual history, and perform reasoning. Surprisingly, we observe that text-based representations consistently achieve competitive performance on all benchmarks, and that visual embeddings provide marginal or no performance improvement, demonstrating the effectiveness of text-based video representation in the LLM era. We perform extensive ablation study and qualitative analysis to support our observations, and achieve state-of-the-art performance on three benchmarks.

{{</citation>}}


### (49/118) Medical Image Retrieval Using Pretrained Embeddings (Farnaz Khun Jush et al., 2023)

{{<citation>}}

Farnaz Khun Jush, Tuan Truong, Steffen Vogler, Matthias Lenga. (2023)  
**Medical Image Retrieval Using Pretrained Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.13547v1)  

---


**ABSTRACT**  
A wide range of imaging techniques and data formats available for medical images make accurate retrieval from image databases challenging.   Efficient retrieval systems are crucial in advancing medical research, enabling large-scale studies and innovative diagnostic tools. Thus, addressing the challenges of medical image retrieval is essential for the continued enhancement of healthcare and research.   In this study, we evaluated the feasibility of employing four state-of-the-art pretrained models for medical image retrieval at modality, body region, and organ levels and compared the results of two similarity indexing approaches. Since the employed networks take 2D images, we analyzed the impacts of weighting and sampling strategies to incorporate 3D information during retrieval of 3D volumes. We showed that medical image retrieval is feasible using pretrained networks without any additional training or fine-tuning steps. Using pretrained embeddings, we achieved a recall of 1 for various tasks at modality, body region, and organ level.

{{</citation>}}


### (50/118) Leveraging CNNs and Ensemble Learning for Automated Disaster Image Classification (Archit Rathod et al., 2023)

{{<citation>}}

Archit Rathod, Veer Pariawala, Mokshit Surana, Kumkum Saxena. (2023)  
**Leveraging CNNs and Ensemble Learning for Automated Disaster Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2311.13531v1)  

---


**ABSTRACT**  
Natural disasters act as a serious threat globally, requiring effective and efficient disaster management and recovery. This paper focuses on classifying natural disaster images using Convolutional Neural Networks (CNNs). Multiple CNN architectures were built and trained on a dataset containing images of earthquakes, floods, wildfires, and volcanoes. A stacked CNN ensemble approach proved to be the most effective, achieving 95% accuracy and an F1 score going up to 0.96 for individual classes. Tuning hyperparameters of individual models for optimization was critical to maximize the models' performance. The stacking of CNNs with XGBoost acting as the meta-model utilizes the strengths of the CNN and ResNet models to improve the overall accuracy of the classification. Results obtained from the models illustrated the potency of CNN-based models for automated disaster image classification. This lays the foundation for expanding these techniques to build robust systems for disaster response, damage assessment, and recovery management.

{{</citation>}}


### (51/118) PG-Video-LLaVA: Pixel Grounding Large Video-Language Models (Shehan Munasinghe et al., 2023)

{{<citation>}}

Shehan Munasinghe, Rusiru Thushara, Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, Mubarak Shah, Fahad Khan. (2023)  
**PG-Video-LLaVA: Pixel Grounding Large Video-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ChatGPT, GPT, GPT-3.5, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13435v1)  

---


**ABSTRACT**  
Extending image-based Large Multimodal Models (LMM) to videos is challenging due to the inherent complexity of video data. The recent approaches extending image-based LMM to videos either lack the grounding capabilities (e.g., VideoChat, Video-ChatGPT, Video-LLaMA) or do not utilize the audio-signals for better video understanding (e.g., Video-ChatGPT). Addressing these gaps, we propose Video-LLaVA, the first LMM with pixel-level grounding capability, integrating audio cues by transcribing them into text to enrich video-context understanding. Our framework uses an off-the-shelf tracker and a novel grounding module, enabling it to spatially and temporally localize objects in videos following user instructions. We evaluate Video-LLaVA using video-based generative and question-answering benchmarks and introduce new benchmarks specifically designed to measure prompt-based object grounding performance in videos. Further, we propose the use of Vicuna over GPT-3.5, as utilized in Video-ChatGPT, for video-based conversation benchmarking, ensuring reproducibility of results which is a concern with the proprietary nature of GPT-3.5. Our framework builds on SoTA image-based LLaVA model and extends its advantages to the video domain, delivering promising gains on video-based conversation and grounding tasks. Project Page: https://github.com/mbzuai-oryx/Video-LLaVA

{{</citation>}}


### (52/118) Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images (Jaeyoung Chung et al., 2023)

{{<citation>}}

Jaeyoung Chung, Jeongtaek Oh, Kyoung Mu Lee. (2023)  
**Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.13398v1)  

---


**ABSTRACT**  
In this paper, we present a method to optimize Gaussian splatting with a limited number of images while avoiding overfitting. Representing a 3D scene by combining numerous Gaussian splats has yielded outstanding visual quality. However, it tends to overfit the training views when only a small number of images are available. To address this issue, we introduce a dense depth map as a geometry guide to mitigate overfitting. We obtained the depth map using a pre-trained monocular depth estimation model and aligning the scale and offset using sparse COLMAP feature points. The adjusted depth aids in the color-based optimization of 3D Gaussian splatting, mitigating floating artifacts, and ensuring adherence to geometric constraints. We verify the proposed method on the NeRF-LLFF dataset with varying numbers of few images. Our approach demonstrates robust geometry compared to the original method that relies solely on images.

{{</citation>}}


### (53/118) SegVol: Universal and Interactive Volumetric Medical Image Segmentation (Yuxin Du et al., 2023)

{{<citation>}}

Yuxin Du, Fan Bai, Tiejun Huang, Bo Zhao. (2023)  
**SegVol: Universal and Interactive Volumetric Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13385v1)  

---


**ABSTRACT**  
Precise image segmentation provides clinical study with meaningful and well-structured information. Despite the remarkable progress achieved in medical image segmentation, there is still an absence of foundation segmentation model that can segment a wide range of anatomical categories with easy user interaction. In this paper, we propose a universal and interactive volumetric medical image segmentation model, named SegVol. By training on 90k unlabeled Computed Tomography (CT) volumes and 6k labeled CTs, this foundation model supports the segmentation of over 200 anatomical categories using semantic and spatial prompts. Extensive experiments verify that SegVol outperforms the state of the art by a large margin on multiple segmentation benchmarks. Notably, on three challenging lesion datasets, our method achieves around 20% higher Dice score than nnU-Net. The model and data are publicly available at: https://github.com/BAAI-DCAI/SegVol.

{{</citation>}}


### (54/118) Rethinking Radiology Report Generation via Causal Reasoning and Counterfactual Augmentation (Xiao Song et al., 2023)

{{<citation>}}

Xiao Song, Jiafan Liu, Yun Li, Wenbin Lei, Ruxin Wang. (2023)  
**Rethinking Radiology Report Generation via Causal Reasoning and Counterfactual Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: Augmentation, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.13307v1)  

---


**ABSTRACT**  
Radiology Report Generation (RRG) draws attention as an interaction between vision and language fields. Previous works inherited the ideology of vision-to-language generation tasks,aiming to generate paragraphs with high consistency as reports. However, one unique characteristic of RRG, the independence between diseases, was neglected, leading to the injection of the spurious confounder, i.e., the disease co-occurrence. Unfortunately, this confounder confuses the process of report generation worse because of the biased RRG data distribution. In this paper, to rethink this issue thoroughly, we reason about its causes and effects from a novel perspective of statistics and causality, where the Joint Vision Coupling and the Conditional Sentence Coherence Coupling are two aspects prone to implicitly decrease the accuracy of reports. Then, a counterfactual augmentation strategy that contains the Counterfactual Sample Synthesis and the Counterfactual Report Reconstruction sub-methods is proposed to break these two aspects of spurious effects. Experimental results and further analyses on two widely used datasets justify our reasoning and proposed methods.

{{</citation>}}


### (55/118) CMFDFormer: Transformer-based Copy-Move Forgery Detection with Continual Learning (Yaqi Liu et al., 2023)

{{<citation>}}

Yaqi Liu, Chao Xia, Song Xiao, Qingxiao Guan, Wenqian Dong, Yifan Zhang, Nenghai Yu. (2023)  
**CMFDFormer: Transformer-based Copy-Move Forgery Detection with Continual Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.13263v1)  

---


**ABSTRACT**  
Copy-move forgery detection aims at detecting duplicated regions in a suspected forged image, and deep learning based copy-move forgery detection methods are in the ascendant. These deep learning based methods heavily rely on synthetic training data, and the performance will degrade when facing new tasks. In this paper, we propose a Transformer-style copy-move forgery detection network named as CMFDFormer, and provide a novel PCSD (Pooled Cube and Strip Distillation) continual learning framework to help CMFDFormer handle new tasks. CMFDFormer consists of a MiT (Mix Transformer) backbone network and a PHD (Pluggable Hybrid Decoder) mask prediction network. The MiT backbone network is a Transformer-style network which is adopted on the basis of comprehensive analyses with CNN-style and MLP-style backbones. The PHD network is constructed based on self-correlation computation, hierarchical feature integration, a multi-scale cycle fully-connected block and a mask reconstruction block. The PHD network is applicable to feature extractors of different styles for hierarchical multi-scale information extraction, achieving comparable performance. Last but not least, we propose a PCSD continual learning framework to improve the forgery detectability and avoid catastrophic forgetting when handling new tasks. Our continual learning framework restricts intermediate features from the PHD network, and takes advantage of both cube pooling and strip pooling. Extensive experiments on publicly available datasets demonstrate the good performance of CMFDFormer and the effectiveness of the PCSD continual learning framework.

{{</citation>}}


### (56/118) DA-STC: Domain Adaptive Video Semantic Segmentation via Spatio-Temporal Consistency (Zhe Zhang et al., 2023)

{{<citation>}}

Zhe Zhang, Gaochang Wu, Jing Zhang, Chunhua Shen, Dacheng Tao, Tianyou Chai. (2023)  
**DA-STC: Domain Adaptive Video Semantic Segmentation via Spatio-Temporal Consistency**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV, eess-IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.13254v1)  

---


**ABSTRACT**  
Video semantic segmentation is a pivotal aspect of video representation learning. However, significant domain shifts present a challenge in effectively learning invariant spatio-temporal features across the labeled source domain and unlabeled target domain for video semantic segmentation. To solve the challenge, we propose a novel DA-STC method for domain adaptive video semantic segmentation, which incorporates a bidirectional multi-level spatio-temporal fusion module and a category-aware spatio-temporal feature alignment module to facilitate consistent learning for domain-invariant features. Firstly, we perform bidirectional spatio-temporal fusion at the image sequence level and shallow feature level, leading to the construction of two fused intermediate video domains. This prompts the video semantic segmentation model to consistently learn spatio-temporal features of shared patch sequences which are influenced by domain-specific contexts, thereby mitigating the feature gap between the source and target domain. Secondly, we propose a category-aware feature alignment module to promote the consistency of spatio-temporal features, facilitating adaptation to the target domain. Specifically, we adaptively aggregate the domain-specific deep features of each category along spatio-temporal dimensions, which are further constrained to achieve cross-domain intra-class feature alignment and inter-class feature separation. Extensive experiments demonstrate the effectiveness of our method, which achieves state-of-the-art mIOUs on multiple challenging benchmarks. Furthermore, we extend the proposed DA-STC to the image domain, where it also exhibits superior performance for domain adaptive semantic segmentation. The source code and models will be made available at \url{https://github.com/ZHE-SAPI/DA-STC}.

{{</citation>}}


### (57/118) Towards Hetero-Client Federated Multi-Task Learning (Yuxiang Lu et al., 2023)

{{<citation>}}

Yuxiang Lu, Suizhi Huang, Yuwen Yang, Shalayiding Sirejiding, Yue Ding, Hongtao Lu. (2023)  
**Towards Hetero-Client Federated Multi-Task Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.13250v1)  

---


**ABSTRACT**  
Federated Learning (FL) enables joint training across distributed clients using their local data privately. Federated Multi-Task Learning (FMTL) builds on FL to handle multiple tasks, assuming model congruity that identical model architecture is deployed in each client. To relax this assumption and thus extend real-world applicability, we introduce a novel problem setting, Hetero-Client Federated Multi-Task Learning (HC-FMTL), to accommodate diverse task setups. The main challenge of HC-FMTL is the model incongruity issue that invalidates conventional aggregation methods. It also escalates the difficulties in accurate model aggregation to deal with data and task heterogeneity inherent in FMTL. To address these challenges, we propose the FedHCA$^2$ framework, which allows for federated training of personalized models by modeling relationships among heterogeneous clients. Drawing on our theoretical insights into the difference between multi-task and federated optimization, we propose the Hyper Conflict-Averse Aggregation scheme to mitigate conflicts during encoder updates. Additionally, inspired by task interaction in MTL, the Hyper Cross Attention Aggregation scheme uses layer-wise cross attention to enhance decoder interactions while alleviating model incongruity. Moreover, we employ learnable Hyper Aggregation Weights for each client to customize personalized parameter updates. Extensive experiments demonstrate the superior performance of FedHCA$^2$ in various HC-FMTL scenarios compared to representative methods. Our code will be made publicly available.

{{</citation>}}


### (58/118) TSegFormer: 3D Tooth Segmentation in Intraoral Scans with Geometry Guided Transformer (Huimin Xiong et al., 2023)

{{<citation>}}

Huimin Xiong, Kunle Li, Kaiyuan Tan, Yang Feng, Joey Tianyi Zhou, Jin Hao, Haochao Ying, Jian Wu, Zuozhu Liu. (2023)  
**TSegFormer: 3D Tooth Segmentation in Intraoral Scans with Geometry Guided Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.13234v1)  

---


**ABSTRACT**  
Optical Intraoral Scanners (IOS) are widely used in digital dentistry to provide detailed 3D information of dental crowns and the gingiva. Accurate 3D tooth segmentation in IOSs is critical for various dental applications, while previous methods are error-prone at complicated boundaries and exhibit unsatisfactory results across patients. In this paper, we propose TSegFormer which captures both local and global dependencies among different teeth and the gingiva in the IOS point clouds with a multi-task 3D transformer architecture. Moreover, we design a geometry-guided loss based on a novel point curvature to refine boundaries in an end-to-end manner, avoiding time-consuming post-processing to reach clinically applicable segmentation. In addition, we create a dataset with 16,000 IOSs, the largest ever IOS dataset to the best of our knowledge. The experimental results demonstrate that our TSegFormer consistently surpasses existing state-of-the-art baselines. The superiority of TSegFormer is corroborated by extensive analysis, visualizations and real-world clinical applicability tests. Our code is available at https://github.com/huiminxiong/TSegFormer.

{{</citation>}}


### (59/118) Knowledge From the Dark Side: Entropy-Reweighted Knowledge Distillation for Balanced Knowledge Transfer (Chi-Ping Su et al., 2023)

{{<citation>}}

Chi-Ping Su, Ching-Hsun Tseng, Shin-Jye Lee. (2023)  
**Knowledge From the Dark Side: Entropy-Reweighted Knowledge Distillation for Balanced Knowledge Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.13621v1)  

---


**ABSTRACT**  
Knowledge Distillation (KD) transfers knowledge from a larger "teacher" model to a compact "student" model, guiding the student with the "dark knowledge" $\unicode{x2014}$ the implicit insights present in the teacher's soft predictions. Although existing KDs have shown the potential of transferring knowledge, the gap between the two parties still exists. With a series of investigations, we argue the gap is the result of the student's overconfidence in prediction, signaling an imbalanced focus on pronounced features while overlooking the subtle yet crucial dark knowledge. To overcome this, we introduce the Entropy-Reweighted Knowledge Distillation (ER-KD), a novel approach that leverages the entropy in the teacher's predictions to reweight the KD loss on a sample-wise basis. ER-KD precisely refocuses the student on challenging instances rich in the teacher's nuanced insights while reducing the emphasis on simpler cases, enabling a more balanced knowledge transfer. Consequently, ER-KD not only demonstrates compatibility with various state-of-the-art KD methods but also further enhances their performance at negligible cost. This approach offers a streamlined and effective strategy to refine the knowledge transfer process in KD, setting a new paradigm in the meticulous handling of dark knowledge. Our code is available at https://github.com/cpsu00/ER-KD.

{{</citation>}}


### (60/118) Self-guided Few-shot Semantic Segmentation for Remote Sensing Imagery Based on Large Vision Models (Xiyu Qi et al., 2023)

{{<citation>}}

Xiyu Qi, Yifan Wu, Yongqiang Mao, Wenhui Zhang, Yidan Zhang. (2023)  
**Self-guided Few-shot Semantic Segmentation for Remote Sensing Imagery Based on Large Vision Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.13200v1)  

---


**ABSTRACT**  
The Segment Anything Model (SAM) exhibits remarkable versatility and zero-shot learning abilities, owing largely to its extensive training data (SA-1B). Recognizing SAM's dependency on manual guidance given its category-agnostic nature, we identified unexplored potential within few-shot semantic segmentation tasks for remote sensing imagery. This research introduces a structured framework designed for the automation of few-shot semantic segmentation. It utilizes the SAM model and facilitates a more efficient generation of semantically discernible segmentation outcomes. Central to our methodology is a novel automatic prompt learning approach, leveraging prior guided masks to produce coarse pixel-wise prompts for SAM. Extensive experiments on the DLRSD datasets underline the superiority of our approach, outperforming other available few-shot methodologies.

{{</citation>}}


### (61/118) Towards Improving Document Understanding: An Exploration on Text-Grounding via MLLMs (Yonghui Wang et al., 2023)

{{<citation>}}

Yonghui Wang, Wengang Zhou, Hao Feng, Keyi Zhou, Houqiang Li. (2023)  
**Towards Improving Document Understanding: An Exploration on Text-Grounding via MLLMs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13194v1)  

---


**ABSTRACT**  
In the field of document understanding, significant advances have been made in the fine-tuning of Multimodal Large Language Models (MLLMs) with instruction-following data. Nevertheless, the potential of text-grounding capability within text-rich scenarios remains underexplored. In this paper, we present a text-grounding document understanding model, termed TGDoc, which addresses this deficiency by enhancing MLLMs with the ability to discern the spatial positioning of text within images. Empirical evidence suggests that text-grounding improves the model's interpretation of textual content, thereby elevating its proficiency in comprehending text-rich images. Specifically, we compile a dataset containing 99K PowerPoint presentations sourced from the internet. We formulate instruction tuning tasks including text detection, recognition, and spotting to facilitate the cohesive alignment between the visual encoder and large language model. Moreover, we curate a collection of text-rich images and prompt the text-only GPT-4 to generate 12K high-quality conversations, featuring textual locations within text-rich scenarios. By integrating text location data into the instructions, TGDoc is adept at discerning text locations during the visual question process. Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple text-rich benchmarks, validating the effectiveness of our method.

{{</citation>}}


### (62/118) HEViTPose: High-Efficiency Vision Transformer for Human Pose Estimation (Chengpeng Wu et al., 2023)

{{<citation>}}

Chengpeng Wu, Guangxing Tan, Chunyu Li. (2023)  
**HEViTPose: High-Efficiency Vision Transformer for Human Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.13615v1)  

---


**ABSTRACT**  
Human pose estimation in complicated situations has always been a challenging task. Many Transformer-based pose networks have been proposed recently, achieving encouraging progress in improving performance. However, the remarkable performance of pose networks is always accompanied by heavy computation costs and large network scale. In order to deal with this problem, this paper proposes a High-Efficiency Vision Transformer for Human Pose Estimation (HEViTPose). In HEViTPose, a Cascaded Group Spatial Reduction Multi-Head Attention Module (CGSR-MHA) is proposed, which reduces the computational cost through feature grouping and spatial degradation mechanisms, while preserving feature diversity through multiple low-dimensional attention heads. Moreover, a concept of Patch Embedded Overlap Width (PEOW) is defined to help understand the relationship between the amount of overlap and local continuity. By optimising PEOW, our model gains improvements in performance, parameters and GFLOPs.   Comprehensive experiments on two benchmark datasets (MPII and COCO) demonstrate that the small and large HEViTPose models are on par with state-of-the-art models while being more lightweight. Specifically, HEViTPose-B achieves 90.7 PCK@0.5 on the MPII test set and 72.6 AP on the COCO test-dev2017 set. Compared with HRNet-W32 and Swin-S, our HEViTPose-B significantly reducing Params ($\downarrow$62.1%,$\downarrow$80.4%,) and GFLOPs ($\downarrow$43.4%,$\downarrow$63.8%,). Code and models are available at \url{here}.

{{</citation>}}


### (63/118) Learning to Complement with Multiple Humans (LECOMH): Integrating Multi-rater and Noisy-Label Learning into Human-AI Collaboration (Zheng Zhang et al., 2023)

{{<citation>}}

Zheng Zhang, Kevin Wells, Gustavo Carneiro. (2023)  
**Learning to Complement with Multiple Humans (LECOMH): Integrating Multi-rater and Noisy-Label Learning into Human-AI Collaboration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13172v1)  

---


**ABSTRACT**  
The advent of learning with noisy labels (LNL), multi-rater learning, and human-AI collaboration has revolutionised the development of robust classifiers, enabling them to address the challenges posed by different types of data imperfections and complex decision processes commonly encountered in real-world applications. While each of these methodologies has individually made significant strides in addressing their unique challenges, the development of techniques that can simultaneously tackle these three problems remains underexplored. This paper addresses this research gap by integrating noisy-label learning, multi-rater learning, and human-AI collaboration with new benchmarks and the innovative Learning to Complement with Multiple Humans (LECOMH) approach. LECOMH optimises the level of human collaboration during testing, aiming to optimise classification accuracy while minimising collaboration costs that vary from 0 to M, where M is the maximum number of human collaborators. We quantitatively compare LECOMH with leading human-AI collaboration methods using our proposed benchmarks. LECOMH consistently outperforms the competition, with accuracy improving as collaboration costs increase. Notably, LECOMH is the only method enhancing human labeller performance across all benchmarks.

{{</citation>}}


### (64/118) 3D Face Style Transfer with a Hybrid Solution of NeRF and Mesh Rasterization (Jianwei Feng et al., 2023)

{{<citation>}}

Jianwei Feng, Prateek Singhal. (2023)  
**3D Face Style Transfer with a Hybrid Solution of NeRF and Mesh Rasterization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2311.13168v1)  

---


**ABSTRACT**  
Style transfer for human face has been widely researched in recent years. Majority of the existing approaches work in 2D image domain and have 3D inconsistency issue when applied on different viewpoints of the same face. In this paper, we tackle the problem of 3D face style transfer which aims at generating stylized novel views of a 3D human face with multi-view consistency. We propose to use a neural radiance field (NeRF) to represent 3D human face and combine it with 2D style transfer to stylize the 3D face. We find that directly training a NeRF on stylized images from 2D style transfer brings in 3D inconsistency issue and causes blurriness. On the other hand, training a NeRF jointly with 2D style transfer objectives shows poor convergence due to the identity and head pose gap between style image and content image. It also poses challenge in training time and memory due to the need of volume rendering for full image to apply style transfer loss functions. We therefore propose a hybrid framework of NeRF and mesh rasterization to combine the benefits of high fidelity geometry reconstruction of NeRF and fast rendering speed of mesh. Our framework consists of three stages: 1. Training a NeRF model on input face images to learn the 3D geometry; 2. Extracting a mesh from the trained NeRF model and optimizing it with style transfer objectives via differentiable rasterization; 3. Training a new color network in NeRF conditioned on a style embedding to enable arbitrary style transfer to the 3D face. Experiment results show that our approach generates high quality face style transfer with great 3D consistency, while also enabling a flexible style control.

{{</citation>}}


### (65/118) HalluciDoctor: Mitigating Hallucinatory Toxicity in Visual Instruction Data (Qifan Yu et al., 2023)

{{<citation>}}

Qifan Yu, Juncheng Li, Longhui Wei, Liang Pang, Wentao Ye, Bosheng Qin, Siliang Tang, Qi Tian, Yueting Zhuang. (2023)  
**HalluciDoctor: Mitigating Hallucinatory Toxicity in Visual Instruction Data**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.13614v1)  

---


**ABSTRACT**  
Multi-modal Large Language Models (MLLMs) tuned on machine-generated instruction-following data have demonstrated remarkable performance in various multi-modal understanding and generation tasks. However, the hallucinations inherent in machine-generated data, which could lead to hallucinatory outputs in MLLMs, remain under-explored. This work aims to investigate various hallucinations (i.e., object, relation, attribute hallucinations) and mitigate those hallucinatory toxicities in large-scale machine-generated visual instruction datasets. Drawing on the human ability to identify factual errors, we present a novel hallucination detection and elimination framework, HalluciDoctor, based on the cross-checking paradigm. We use our framework to identify and eliminate hallucinations in the training data automatically. Interestingly, HalluciDoctor also indicates that spurious correlations arising from long-tail object co-occurrences contribute to hallucinations. Based on that, we execute counterfactual visual instruction expansion to balance data distribution, thereby enhancing MLLMs' resistance to hallucinations. Comprehensive experiments on hallucination evaluation benchmarks show that our method successfully mitigates 44.6% hallucinations relatively and maintains competitive performance compared to LLaVA.The source code will be released at \url{https://github.com/Yuqifan1117/HalluciDoctor}.

{{</citation>}}


### (66/118) Test-Time Augmentation for 3D Point Cloud Classification and Segmentation (Tuan-Anh Vu et al., 2023)

{{<citation>}}

Tuan-Anh Vu, Srinjay Sarkar, Zhiyuan Zhang, Binh-Son Hua, Sai-Kit Yeung. (2023)  
**Test-Time Augmentation for 3D Point Cloud Classification and Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.13152v1)  

---


**ABSTRACT**  
Data augmentation is a powerful technique to enhance the performance of a deep learning task but has received less attention in 3D deep learning. It is well known that when 3D shapes are sparsely represented with low point density, the performance of the downstream tasks drops significantly. This work explores test-time augmentation (TTA) for 3D point clouds. We are inspired by the recent revolution of learning implicit representation and point cloud upsampling, which can produce high-quality 3D surface reconstruction and proximity-to-surface, respectively. Our idea is to leverage the implicit field reconstruction or point cloud upsampling techniques as a systematic way to augment point cloud data. Mainly, we test both strategies by sampling points from the reconstructed results and using the sampled point cloud as test-time augmented data. We show that both strategies are effective in improving accuracy. We observed that point cloud upsampling for test-time augmentation can lead to more significant performance improvement on downstream tasks such as object classification and segmentation on the ModelNet40, ShapeNet, ScanObjectNN, and SemanticKITTI datasets, especially for sparse point clouds.

{{</citation>}}


### (67/118) Spanning Training Progress: Temporal Dual-Depth Scoring (TDDS) for Enhanced Dataset Pruning (Xin Zhang et al., 2023)

{{<citation>}}

Xin Zhang, Jiawei Du, Yunsong Li, Weiying Xie, Joey Tianyi Zhou. (2023)  
**Spanning Training Progress: Temporal Dual-Depth Scoring (TDDS) for Enhanced Dataset Pruning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2311.13613v1)  

---


**ABSTRACT**  
Dataset pruning aims to construct a coreset capable of achieving performance comparable to the original, full dataset. Most existing dataset pruning methods rely on snapshot-based criteria to identify representative samples, often resulting in poor generalization across various pruning and cross-architecture scenarios. Recent studies have addressed this issue by expanding the scope of training dynamics considered, including factors such as forgetting event and probability change, typically using an averaging approach. However, these works struggle to integrate a broader range of training dynamics without overlooking well-generalized samples, which may not be sufficiently highlighted in an averaging manner. In this study, we propose a novel dataset pruning method termed as Temporal Dual-Depth Scoring (TDDS), to tackle this problem. TDDS utilizes a dual-depth strategy to achieve a balance between incorporating extensive training dynamics and identifying representative samples for dataset pruning. In the first depth, we estimate the series of each sample's individual contributions spanning the training progress, ensuring comprehensive integration of training dynamics. In the second depth, we focus on the variability of the sample-wise contributions identified in the first depth to highlight well-generalized samples. Extensive experiments conducted on CIFAR and ImageNet datasets verify the superiority of TDDS over previous SOTA methods. Specifically on CIFAR-100, our method achieves 54.51% accuracy with only 10% training data, surpassing random selection by 7.83% and other comparison methods by at least 12.69%.

{{</citation>}}


### (68/118) P2RBox: A Single Point is All You Need for Oriented Object Detection (Guangming Cao et al., 2023)

{{<citation>}}

Guangming Cao, Xuehui Yu, Wenwen Yu, Xumeng Han, Xue Yang, Guorong Li, Jianbin Jiao, Zhenjun Han. (2023)  
**P2RBox: A Single Point is All You Need for Oriented Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.13128v1)  

---


**ABSTRACT**  
Oriented object detection, a specialized subfield in computer vision, finds applications across diverse scenarios, excelling particularly when dealing with objects of arbitrary orientations. Conversely, point annotation, which treats objects as single points, offers a cost-effective alternative to rotated and horizontal bounding boxes but sacrifices performance due to the loss of size and orientation information. In this study, we introduce the P2RBox network, which leverages point annotations and a mask generator to create mask proposals, followed by filtration through our Inspector Module and Constrainer Module. This process selects high-quality masks, which are subsequently converted into rotated box annotations for training a fully supervised detector. Specifically, we've thoughtfully crafted an Inspector Module rooted in multi-instance learning principles to evaluate the semantic score of masks. We've also proposed a more robust mask quality assessment in conjunction with the Constrainer Module. Furthermore, we've introduced a Symmetry Axis Estimation (SAE) Module inspired by the spectral theorem for symmetric matrices to transform the top-performing mask proposal into rotated bounding boxes. P2RBox performs well with three fully supervised rotated object detectors: RetinaNet, Rotated FCOS, and Oriented R-CNN. By combining with Oriented R-CNN, P2RBox achieves 62.26% on DOTA-v1.0 test dataset. As far as we know, this is the first attempt at training an oriented object detector with point supervision.

{{</citation>}}


### (69/118) FuseNet: Self-Supervised Dual-Path Network for Medical Image Segmentation (Amirhossein Kazerouni et al., 2023)

{{<citation>}}

Amirhossein Kazerouni, Sanaz Karimijafarbigloo, Reza Azad, Yury Velichko, Ulas Bagci, Dorit Merhof. (2023)  
**FuseNet: Self-Supervised Dual-Path Network for Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.13069v1)  

---


**ABSTRACT**  
Semantic segmentation, a crucial task in computer vision, often relies on labor-intensive and costly annotated datasets for training. In response to this challenge, we introduce FuseNet, a dual-stream framework for self-supervised semantic segmentation that eliminates the need for manual annotation. FuseNet leverages the shared semantic dependencies between the original and augmented images to create a clustering space, effectively assigning pixels to semantically related clusters, and ultimately generating the segmentation map. Additionally, FuseNet incorporates a cross-modal fusion technique that extends the principles of CLIP by replacing textual data with augmented images. This approach enables the model to learn complex visual representations, enhancing robustness against variations similar to CLIP's text invariance. To further improve edge alignment and spatial consistency between neighboring pixels, we introduce an edge refinement loss. This loss function considers edge information to enhance spatial coherence, facilitating the grouping of nearby pixels with similar visual features. Extensive experiments on skin lesion and lung segmentation datasets demonstrate the effectiveness of our method. \href{https://github.com/xmindflow/FuseNet}{Codebase.}

{{</citation>}}


## eess.IV (3)



### (70/118) Multi-view Hybrid Graph Convolutional Network for Volume-to-mesh Reconstruction in Cardiovascular MRI (Nicolás Gaggion et al., 2023)

{{<citation>}}

Nicolás Gaggion, Benjamin A. Matheson, Yan Xia, Rodrigo Bonazzola, Nishant Ravikumar, Zeike A. Taylor, Diego H. Milone, Alejandro F. Frangi, Enzo Ferrante. (2023)  
**Multi-view Hybrid Graph Convolutional Network for Volume-to-mesh Reconstruction in Cardiovascular MRI**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2311.13706v1)  

---


**ABSTRACT**  
Cardiovascular magnetic resonance imaging is emerging as a crucial tool to examine cardiac morphology and function. Essential to this endeavour are anatomical 3D surface and volumetric meshes derived from CMR images, which facilitate computational anatomy studies, biomarker discovery, and in-silico simulations. However, conventional surface mesh generation methods, such as active shape models and multi-atlas segmentation, are highly time-consuming and require complex processing pipelines to generate simulation-ready 3D meshes. In response, we introduce HybridVNet, a novel architecture for direct image-to-mesh extraction seamlessly integrating standard convolutional neural networks with graph convolutions, which we prove can efficiently handle surface and volumetric meshes by encoding them as graph structures. To further enhance accuracy, we propose a multiview HybridVNet architecture which processes both long axis and short axis CMR, showing that it can increase the performance of cardiac MR mesh generation. Our model combines traditional convolutional networks with variational graph generative models, deep supervision and mesh-specific regularisation. Experiments on a comprehensive dataset from the UK Biobank confirm the potential of HybridVNet to significantly advance cardiac imaging and computational cardiology by efficiently generating high-fidelity and simulation ready meshes from CMR images.

{{</citation>}}


### (71/118) Immunohistochemistry guided segmentation of benign epithelial cells, in situ lesions, and invasive epithelial cells in breast cancer slides (Maren Høibø et al., 2023)

{{<citation>}}

Maren Høibø, André Pedersen, Vibeke Grotnes Dale, Sissel Marie Berget, Borgny Ytterhus, Cecilia Lindskog, Elisabeth Wik, Lars A. Akslen, Ingerid Reinertsen, Erik Smistad, Marit Valla. (2023)  
**Immunohistochemistry guided segmentation of benign epithelial cells, in situ lesions, and invasive epithelial cells in breast cancer slides**  

---
Primary Category: eess.IV  
Categories: I-4-6, I-4-6; I-4-9; I-5-4; J-3, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13261v1)  

---


**ABSTRACT**  
Digital pathology enables automatic analysis of histopathological sections using artificial intelligence (AI). Automatic evaluation could improve diagnostic efficiency and help find associations between morphological features and clinical outcome. For development of such prediction models, identifying invasive epithelial cells, and separating these from benign epithelial cells and in situ lesions would be the first step. In this study, we aimed to develop an AI model for segmentation of epithelial cells in sections from breast cancer. We generated epithelial ground truth masks by restaining hematoxylin and eosin (HE) sections with cytokeratin (CK) AE1/AE3, and by pathologists' annotations. HE/CK image pairs were used to train a convolutional neural network, and data augmentation was used to make the model more robust. Tissue microarrays (TMAs) from 839 patients, and whole slide images from two patients were used for training and evaluation of the models. The sections were derived from four cohorts of breast cancer patients. TMAs from 21 patients from a fifth cohort was used as a second test set. In quantitative evaluation, a mean Dice score of 0.70, 0.79, and 0.75 for invasive epithelial cells, benign epithelial cells, and in situ lesions, respectively, were achieved. In qualitative scoring (0-5) by pathologists, results were best for all epithelium and invasive epithelium, with scores of 4.7 and 4.4. Scores for benign epithelium and in situ lesions were 3.7 and 2.0. The proposed model segmented epithelial cells in HE stained breast cancer slides well, but further work is needed for accurate division between the classes. Immunohistochemistry, together with pathologists' annotations, enabled the creation of accurate ground truths. The model is made freely available in FastPathology and the code is available at https://github.com/AICAN-Research/breast-epithelium-segmentation

{{</citation>}}


### (72/118) Single Image Compressed Sensing MRI via a Self-Supervised Deep Denoising Approach (Marlon Bran Lorenzana et al., 2023)

{{<citation>}}

Marlon Bran Lorenzana, Feng Liu, Shekhar S. Chandra. (2023)  
**Single Image Compressed Sensing MRI via a Self-Supervised Deep Denoising Approach**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.13144v1)  

---


**ABSTRACT**  
Popular methods in compressed sensing (CS) are dependent on deep learning (DL), where large amounts of data are used to train non-linear reconstruction models. However, ensuring generalisability over and access to multiple datasets is challenging to realise for real-world applications. To address these concerns, this paper proposes a single image, self-supervised (SS) CS-MRI framework that enables a joint deep and sparse regularisation of CS artefacts. The approach effectively dampens structured CS artefacts, which can be difficult to remove assuming sparse reconstruction, or relying solely on the inductive biases of CNN to produce noise-free images. Image quality is thereby improved compared to either approach alone. Metrics are evaluated using Cartesian 1D masks on a brain and knee dataset, with PSNR improving by 2-4dB on average.

{{</citation>}}


## cs.LG (28)



### (73/118) Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts (Jayeon Yi et al., 2023)

{{<citation>}}

Jayeon Yi, Sungho Lee, Kyogu Lee. (2023)  
**Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MM, cs-SD, cs.LG, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.13687v1)  

---


**ABSTRACT**  
In the heart of "rhythm games" - games where players must perform actions in sync with a piece of music - are "charts", the directives to be given to players. We newly formulate chart generation as a sequence generation task and train a Transformer using a large dataset. We also introduce tempo-informed preprocessing and training procedures, some of which are suggested to be integral for a successful training. Our model is found to outperform the baselines on a large dataset, and is also found to benefit from pretraining and finetuning.

{{</citation>}}


### (74/118) Evaluating Pretrained models for Deployable Lifelong Learning (Kiran Lekkala et al., 2023)

{{<citation>}}

Kiran Lekkala, Eshan Bhargava, Laurent Itti. (2023)  
**Evaluating Pretrained models for Deployable Lifelong Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.13648v1)  

---


**ABSTRACT**  
We create a novel benchmark for evaluating a Deployable Lifelong Learning system for Visual Reinforcement Learning (RL) that is pretrained on a curated dataset, and propose a novel Scalable Lifelong Learning system capable of retaining knowledge from the previously learnt RL tasks. Our benchmark measures the efficacy of a deployable Lifelong Learning system that is evaluated on scalability, performance and resource utilization. Our proposed system, once pretrained on the dataset, can be deployed to perform continual learning on unseen tasks. Our proposed method consists of a Few Shot Class Incremental Learning (FSCIL) based task-mapper and an encoder/backbone trained entirely using the pretrain dataset. The policy parameters corresponding to the recognized task are then loaded to perform the task. We show that this system can be scaled to incorporate a large number of tasks due to the small memory footprint and fewer computational resources. We perform experiments on our DeLL (Deployment for Lifelong Learning) benchmark on the Atari games to determine the efficacy of the system.

{{</citation>}}


### (75/118) Prompt Risk Control: A Rigorous Framework for Responsible Deployment of Large Language Models (Thomas P. Zollo et al., 2023)

{{<citation>}}

Thomas P. Zollo, Todd Morrill, Zhun Deng, Jake C. Snell, Toniann Pitassi, Richard Zemel. (2023)  
**Prompt Risk Control: A Rigorous Framework for Responsible Deployment of Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.13628v1)  

---


**ABSTRACT**  
The recent explosion in the capabilities of large language models has led to a wave of interest in how best to prompt a model to perform a given task. While it may be tempting to simply choose a prompt based on average performance on a validation set, this can lead to a deployment where unexpectedly poor responses are generated, especially for the worst-off users. To mitigate this prospect, we propose Prompt Risk Control, a lightweight framework for selecting a prompt based on rigorous upper bounds on families of informative risk measures. We offer methods for producing bounds on a diverse set of metrics, including quantities that measure worst-case responses and disparities in generation quality across the population of users. In addition, we extend the underlying statistical bounding techniques to accommodate the possibility of distribution shifts in deployment. Experiments on applications such as open-ended chat, medical question summarization, and code generation highlight how such a framework can foster responsible deployment by reducing the risk of the worst outcomes.

{{</citation>}}


### (76/118) Risk-sensitive Markov Decision Process and Learning under General Utility Functions (Zhengqi Wu et al., 2023)

{{<citation>}}

Zhengqi Wu, Renyuan Xu. (2023)  
**Risk-sensitive Markov Decision Process and Learning under General Utility Functions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.13589v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) has gained substantial attention across diverse application domains and theoretical investigations. Existing literature on RL theory largely focuses on risk-neutral settings where the decision-maker learns to maximize the expected cumulative reward. However, in practical scenarios such as portfolio management and e-commerce recommendations, decision-makers often persist in heterogeneous risk preferences subject to outcome uncertainties, which can not be well-captured by the risk-neural framework. Incorporating these preferences can be approached through utility theory, yet the development of risk-sensitive RL under general utility functions remains an open question for theoretical exploration.   In this paper, we consider a scenario where the decision-maker seeks to optimize a general utility function of the cumulative reward in the framework of a Markov decision process (MDP). To facilitate the Dynamic Programming Principle and Bellman equation, we enlarge the state space with an additional dimension that accounts for the cumulative reward. We propose a discretized approximation scheme to the MDP under enlarged state space, which is tractable and key for algorithmic design. We then propose a modified value iteration algorithm that employs an epsilon-covering over the space of cumulative reward. When a simulator is accessible, our algorithm efficiently learns a near-optimal policy with guaranteed sample complexity. In the absence of a simulator, our algorithm, designed with an upper-confidence-bound exploration approach, identifies a near-optimal policy while ensuring a guaranteed regret bound. For both algorithms, we match the theoretical lower bounds for the risk-neutral setting.

{{</citation>}}


### (77/118) Linear Log-Normal Attention with Unbiased Concentration (Yury Nahshan et al., 2023)

{{<citation>}}

Yury Nahshan, Joseph Kampeas, Emir Haleva. (2023)  
**Linear Log-Normal Attention with Unbiased Concentration**  

---
Primary Category: cs.LG  
Categories: I-7-0; G-3, cs-AI, cs-LG, cs.LG  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.13541v1)  

---


**ABSTRACT**  
Transformer models have achieved remarkable results in a wide range of applications. However, their scalability is hampered by the quadratic time and memory complexity of the self-attention mechanism concerning the sequence length. This limitation poses a substantial obstacle when dealing with long documents or high-resolution images. In this work, we study the self-attention mechanism by analyzing the distribution of the attention matrix and its concentration ability. Furthermore, we propose instruments to measure these quantities and introduce a novel self-attention mechanism, Linear Log-Normal Attention, designed to emulate the distribution and concentration behavior of the original self-attention. Our experimental results on popular natural language benchmarks reveal that our proposed Linear Log-Normal Attention outperforms other linearized attention alternatives, offering a promising avenue for enhancing the scalability of transformer models. Our code is available in supplementary materials.

{{</citation>}}


### (78/118) Applying Dimensionality Reduction as Precursor to LSTM-CNN Models for Classifying Imagery and Motor Signals in ECoG-Based BCIs (Soham Bafana, 2023)

{{<citation>}}

Soham Bafana. (2023)  
**Applying Dimensionality Reduction as Precursor to LSTM-CNN Models for Classifying Imagery and Motor Signals in ECoG-Based BCIs**  

---
Primary Category: cs.LG  
Categories: cs-HC, cs-LG, cs.LG, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.13507v1)  

---


**ABSTRACT**  
Motor impairments, frequently caused by neurological incidents like strokes or traumatic brain injuries, present substantial obstacles in rehabilitation therapy. This research aims to elevate the field by optimizing motor imagery classification algorithms within Brain-Computer Interfaces (BCIs). By improving the efficiency of BCIs, we offer a novel approach that holds significant promise for enhancing motor rehabilitation outcomes. Utilizing unsupervised techniques for dimensionality reduction, namely Uniform Manifold Approximation and Projection (UMAP) coupled with K-Nearest Neighbors (KNN), we evaluate the necessity of employing supervised methods such as Long Short-Term Memory (LSTM) and Convolutional Neural Networks (CNNs) for classification tasks. Importantly, participants who exhibited high KNN scores following UMAP dimensionality reduction also achieved high accuracy in supervised deep learning (DL) models. Due to individualized model requirements and massive neural training data, dimensionality reduction becomes an effective preprocessing step that minimizes the need for extensive data labeling and supervised deep learning techniques. This approach has significant implications not only for targeted therapies in motor dysfunction but also for addressing regulatory, safety, and reliability concerns in the rapidly evolving BCI field.

{{</citation>}}


### (79/118) Bitformer: An efficient Transformer with bitwise operation-based attention for Big Data Analytics at low-cost low-precision devices (Gaoxiang Duan et al., 2023)

{{<citation>}}

Gaoxiang Duan, Junkai Zhang, Xiaoying Zheng, Yongxin Zhu. (2023)  
**Bitformer: An efficient Transformer with bitwise operation-based attention for Big Data Analytics at low-cost low-precision devices**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.13502v1)  

---


**ABSTRACT**  
In the current landscape of large models, the Transformer stands as a cornerstone, playing a pivotal role in shaping the trajectory of modern models. However, its application encounters challenges attributed to the substantial computational intricacies intrinsic to its attention mechanism. Moreover, its reliance on high-precision floating-point operations presents specific hurdles, particularly evident in computation-intensive scenarios such as edge computing environments. These environments, characterized by resource-constrained devices and a preference for lower precision, necessitate innovative solutions.   To tackle the exacting data processing demands posed by edge devices, we introduce the Bitformer model, an inventive extension of the Transformer paradigm. Central to this innovation is a novel attention mechanism that adeptly replaces conventional floating-point matrix multiplication with bitwise operations. This strategic substitution yields dual advantages. Not only does it maintain the attention mechanism's prowess in capturing intricate long-range information dependencies, but it also orchestrates a profound reduction in the computational complexity inherent in the attention operation. The transition from an $O(n^2d)$ complexity, typical of floating-point operations, to an $O(n^2T)$ complexity characterizing bitwise operations, substantiates this advantage. Notably, in this context, the parameter $T$ remains markedly smaller than the conventional dimensionality parameter $d$.   The Bitformer model in essence endeavors to reconcile the indomitable requirements of modern computing landscapes with the constraints posed by edge computing scenarios. By forging this innovative path, we bridge the gap between high-performing models and resource-scarce environments, thus unveiling a promising trajectory for further advancements in the field.

{{</citation>}}


### (80/118) The Tempered Hilbert Simplex Distance and Its Application To Non-linear Embeddings of TEMs (Ehsan Amid et al., 2023)

{{<citation>}}

Ehsan Amid, Frank Nielsen, Richard Nock, Manfred K. Warmuth. (2023)  
**The Tempered Hilbert Simplex Distance and Its Application To Non-linear Embeddings of TEMs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.13459v1)  

---


**ABSTRACT**  
Tempered Exponential Measures (TEMs) are a parametric generalization of the exponential family of distributions maximizing the tempered entropy function among positive measures subject to a probability normalization of their power densities. Calculus on TEMs relies on a deformed algebra of arithmetic operators induced by the deformed logarithms used to define the tempered entropy. In this work, we introduce three different parameterizations of finite discrete TEMs via Legendre functions of the negative tempered entropy function. In particular, we establish an isometry between such parameterizations in terms of a generalization of the Hilbert log cross-ratio simplex distance to a tempered Hilbert co-simplex distance. Similar to the Hilbert geometry, the tempered Hilbert distance is characterized as a $t$-symmetrization of the oriented tempered Funk distance. We motivate our construction by introducing the notion of $t$-lengths of smooth curves in a tautological Finsler manifold. We then demonstrate the properties of our generalized structure in different settings and numerically examine the quality of its differentiable approximations for optimization in machine learning settings.

{{</citation>}}


### (81/118) Explaining high-dimensional text classifiers (Odelia Melamed et al., 2023)

{{<citation>}}

Odelia Melamed, Rich Caruana. (2023)  
**Explaining high-dimensional text classifiers**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-NE, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13454v1)  

---


**ABSTRACT**  
Explainability has become a valuable tool in the last few years, helping humans better understand AI-guided decisions. However, the classic explainability tools are sometimes quite limited when considering high-dimensional inputs and neural network classifiers. We present a new explainability method using theoretically proven high-dimensional properties in neural network classifiers. We present two usages of it: 1) On the classical sentiment analysis task for the IMDB reviews dataset, and 2) our Malware-Detection task for our PowerShell scripts dataset.

{{</citation>}}


### (82/118) Transfer Attacks and Defenses for Large Language Models on Coding Tasks (Chi Zhang et al., 2023)

{{<citation>}}

Chi Zhang, Zifan Wang, Ravi Mangal, Matt Fredrikson, Limin Jia, Corina Pasareanu. (2023)  
**Transfer Attacks and Defenses for Large Language Models on Coding Tasks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.13445v1)  

---


**ABSTRACT**  
Modern large language models (LLMs), such as ChatGPT, have demonstrated impressive capabilities for coding tasks including writing and reasoning about code. They improve upon previous neural network models of code, such as code2seq or seq2seq, that already demonstrated competitive results when performing tasks such as code summarization and identifying code vulnerabilities. However, these previous code models were shown vulnerable to adversarial examples, i.e. small syntactic perturbations that do not change the program's semantics, such as the inclusion of "dead code" through false conditions or the addition of inconsequential print statements, designed to "fool" the models. LLMs can also be vulnerable to the same adversarial perturbations but a detailed study on this concern has been lacking so far. In this paper we aim to investigate the effect of adversarial perturbations on coding tasks with LLMs. In particular, we study the transferability of adversarial examples, generated through white-box attacks on smaller code models, to LLMs. Furthermore, to make the LLMs more robust against such adversaries without incurring the cost of retraining, we propose prompt-based defenses that involve modifying the prompt to include additional information such as examples of adversarially perturbed code and explicit instructions for reversing adversarial perturbations. Our experiments show that adversarial examples obtained with a smaller code model are indeed transferable, weakening the LLMs' performance. The proposed defenses show promise in improving the model's resilience, paving the way to more robust defensive solutions for LLMs in code-related applications.

{{</citation>}}


### (83/118) From Images to Connections: Can DQN with GNNs learn the Strategic Game of Hex? (Yannik Keller et al., 2023)

{{<citation>}}

Yannik Keller, Jannis Blüml, Gopika Sudhakaran, Kristian Kersting. (2023)  
**From Images to Connections: Can DQN with GNNs learn the Strategic Game of Hex?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-GT, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.13414v1)  

---


**ABSTRACT**  
The gameplay of strategic board games such as chess, Go and Hex is often characterized by combinatorial, relational structures -- capturing distinct interactions and non-local patterns -- and not just images. Nonetheless, most common self-play reinforcement learning (RL) approaches simply approximate policy and value functions using convolutional neural networks (CNN). A key feature of CNNs is their relational inductive bias towards locality and translational invariance. In contrast, graph neural networks (GNN) can encode more complicated and distinct relational structures. Hence, we investigate the crucial question: Can GNNs, with their ability to encode complex connections, replace CNNs in self-play reinforcement learning? To this end, we do a comparison with Hex -- an abstract yet strategically rich board game -- serving as our experimental platform. Our findings reveal that GNNs excel at dealing with long range dependency situations in game states and are less prone to overfitting, but also showing a reduced proficiency in discerning local patterns. This suggests a potential paradigm shift, signaling the use of game-specific structures to reshape self-play reinforcement learning.

{{</citation>}}


### (84/118) Confidant: Customizing Transformer-based LLMs via Collaborative Edge Training (Yuhao Chen et al., 2023)

{{<citation>}}

Yuhao Chen, Yuxuan Yan, Qianqian Yang, Yuanchao Shu, Shibo He, Jiming Chen. (2023)  
**Confidant: Customizing Transformer-based LLMs via Collaborative Edge Training**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2311.13381v1)  

---


**ABSTRACT**  
Transformer-based large language models (LLMs) have demonstrated impressive capabilities in a variety of natural language processing (NLP) tasks. Nonetheless, it is challenging to deploy and fine-tune LLMs on mobile edge devices with limited computing, memory, and energy budgets. In this paper, we propose Confidant, a multi-backend collaborative training framework for customizing state-of-the-art LLMs on commodity mobile devices like smartphones. Confidant partitions an LLM into several sub-models so that each fits into a mobile device's memory. A pipeline parallel training mechanism is further developed to ensure fast and efficient distributed training. In addition, we propose a novel backend scheduler to allocate different attention heads to heterogeneous compute hardware, including mobile CPU and GPUs, to maximize the compute resource utilization on each edge device. Our preliminary experimental results show that Confidant achieves at most 45.3% memory reduction and 8.03x inference speedup in practical settings.

{{</citation>}}


### (85/118) REDS: Resource-Efficient Deep Subnetworks for Dynamic Resource Constraints (Francesco Corti et al., 2023)

{{<citation>}}

Francesco Corti, Balz Maag, Joachim Schauer, Ulrich Pferschy, Olga Saukh. (2023)  
**REDS: Resource-Efficient Deep Subnetworks for Dynamic Resource Constraints**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.13349v1)  

---


**ABSTRACT**  
Deep models deployed on edge devices frequently encounter resource variability, which arises from fluctuating energy levels, timing constraints, or prioritization of other critical tasks within the system. State-of-the-art machine learning pipelines generate resource-agnostic models, not capable to adapt at runtime. In this work we introduce Resource-Efficient Deep Subnetworks (REDS) to tackle model adaptation to variable resources. In contrast to the state-of-the-art, REDS use structured sparsity constructively by exploiting permutation invariance of neurons, which allows for hardware-specific optimizations. Specifically, REDS achieve computational efficiency by (1) skipping sequential computational blocks identified by a novel iterative knapsack optimizer, and (2) leveraging simple math to re-arrange the order of operations in REDS computational graph to take advantage of the data cache. REDS support conventional deep networks frequently deployed on the edge and provide computational benefits even for small and simple networks. We evaluate REDS on six benchmark architectures trained on the Google Speech Commands, FMNIST and CIFAR10 datasets, and test on four off-the-shelf mobile and embedded hardware platforms. We provide a theoretical result and empirical evidence for REDS outstanding performance in terms of submodels' test set accuracy, and demonstrate an adaptation time in response to dynamic resource constraints of under 40$\mu$s, utilizing a 2-layer fully-connected network on Arduino Nano 33 BLE Sense.

{{</citation>}}


### (86/118) MergeSFL: Split Federated Learning with Feature Merging and Batch Size Regulation (Yunming Liao et al., 2023)

{{<citation>}}

Yunming Liao, Yang Xu, Hongli Xu, Lun Wang, Zhiwei Yao, Chunming Qiao. (2023)  
**MergeSFL: Split Federated Learning with Feature Merging and Batch Size Regulation**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs-NI, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13348v1)  

---


**ABSTRACT**  
Recently, federated learning (FL) has emerged as a popular technique for edge AI to mine valuable knowledge in edge computing (EC) systems. To mitigate the computing/communication burden on resource-constrained workers and protect model privacy, split federated learning (SFL) has been released by integrating both data and model parallelism. Despite resource limitations, SFL still faces two other critical challenges in EC, i.e., statistical heterogeneity and system heterogeneity. To address these challenges, we propose a novel SFL framework, termed MergeSFL, by incorporating feature merging and batch size regulation in SFL. Concretely, feature merging aims to merge the features from workers into a mixed feature sequence, which is approximately equivalent to the features derived from IID data and is employed to promote model accuracy. While batch size regulation aims to assign diverse and suitable batch sizes for heterogeneous workers to improve training efficiency. Moreover, MergeSFL explores to jointly optimize these two strategies upon their coupled relationship to better enhance the performance of SFL. Extensive experiments are conducted on a physical platform with 80 NVIDIA Jetson edge devices, and the experimental results show that MergeSFL can improve the final model accuracy by 5.82% to 26.22%, with a speedup by about 1.74x to 4.14x, compared to the baselines.

{{</citation>}}


### (87/118) Curriculum Learning and Imitation Learning for Model-free Control on Financial Time-series (Woosung Koh et al., 2023)

{{<citation>}}

Woosung Koh, Insu Choi, Yuntae Jang, Gimin Kang, Woo Chang Kim. (2023)  
**Curriculum Learning and Imitation Learning for Model-free Control on Financial Time-series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-fin-PM  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2311.13326v1)  

---


**ABSTRACT**  
Curriculum learning and imitation learning have been leveraged extensively in the robotics domain. However, minimal research has been done on leveraging these ideas on control tasks over highly stochastic time-series data. Here, we theoretically and empirically explore these approaches in a representative control task over complex time-series data. We implement the fundamental ideas of curriculum learning via data augmentation, while imitation learning is implemented via policy distillation from an oracle. Our findings reveal that curriculum learning should be considered a novel direction in improving control-task performance over complex time-series. Our ample random-seed out-sample empirics and ablation studies are highly encouraging for curriculum learning for time-series control. These findings are especially encouraging as we tune all overlapping hyperparameters on the baseline -- giving an advantage to the baseline. On the other hand, we find that imitation learning should be used with caution.

{{</citation>}}


### (88/118) Revisiting Supervision for Continual Representation Learning (Daniel Marczak et al., 2023)

{{<citation>}}

Daniel Marczak, Sebastian Cygert, Tomasz Trzciński, Bartłomiej Twardowski. (2023)  
**Revisiting Supervision for Continual Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.13321v1)  

---


**ABSTRACT**  
In the field of continual learning, models are designed to learn tasks one after the other. While most research has centered on supervised continual learning, recent studies have highlighted the strengths of self-supervised continual representation learning. The improved transferability of representations built with self-supervised methods is often associated with the role played by the multi-layer perceptron projector. In this work, we depart from this observation and reexamine the role of supervision in continual representation learning. We reckon that additional information, such as human annotations, should not deteriorate the quality of representations. Our findings show that supervised models when enhanced with a multi-layer perceptron head, can outperform self-supervised models in continual representation learning.

{{</citation>}}


### (89/118) Probabilistic Inference in Reinforcement Learning Done Right (Jean Tarbouriech et al., 2023)

{{<citation>}}

Jean Tarbouriech, Tor Lattimore, Brendan O'Donoghue. (2023)  
**Probabilistic Inference in Reinforcement Learning Done Right**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.13294v1)  

---


**ABSTRACT**  
A popular perspective in Reinforcement learning (RL) casts the problem as probabilistic inference on a graphical model of the Markov decision process (MDP). The core object of study is the probability of each state-action pair being visited under the optimal policy. Previous approaches to approximate this quantity can be arbitrarily poor, leading to algorithms that do not implement genuine statistical inference and consequently do not perform well in challenging problems. In this work, we undertake a rigorous Bayesian treatment of the posterior probability of state-action optimality and clarify how it flows through the MDP. We first reveal that this quantity can indeed be used to generate a policy that explores efficiently, as measured by regret. Unfortunately, computing it is intractable, so we derive a new variational Bayesian approximation yielding a tractable convex optimization problem and establish that the resulting policy also explores efficiently. We call our approach VAPOR and show that it has strong connections to Thompson sampling, K-learning, and maximum entropy exploration. We conclude with some experiments demonstrating the performance advantage of a deep RL version of VAPOR.

{{</citation>}}


### (90/118) A Theoretical Insight into Attack and Defense of Gradient Leakage in Transformer (Chenyang Li et al., 2023)

{{<citation>}}

Chenyang Li, Zhao Song, Weixin Wang, Chiwun Yang. (2023)  
**A Theoretical Insight into Attack and Defense of Gradient Leakage in Transformer**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.13624v1)  

---


**ABSTRACT**  
The Deep Leakage from Gradient (DLG) attack has emerged as a prevalent and highly effective method for extracting sensitive training data by inspecting exchanged gradients. This approach poses a substantial threat to the privacy of individuals and organizations alike. This research presents a comprehensive analysis of the gradient leakage method when applied specifically to transformer-based models. Through meticulous examination, we showcase the capability to accurately recover data solely from gradients and rigorously investigate the conditions under which gradient attacks can be executed, providing compelling evidence. Furthermore, we reevaluate the approach of introducing additional noise on gradients as a protective measure against gradient attacks. To address this, we outline a theoretical proof that analyzes the associated privacy costs within the framework of differential privacy. Additionally, we affirm the convergence of the Stochastic Gradient Descent (SGD) algorithm under perturbed gradients. The primary objective of this study is to augment the understanding of gradient leakage attack and defense strategies while actively contributing to the development of privacy-preserving techniques specifically tailored for transformer-based models. By shedding light on the vulnerabilities and countermeasures associated with gradient leakage, this research aims to foster advancements in safeguarding sensitive data and upholding privacy in the context of transformer-based models.

{{</citation>}}


### (91/118) Comprehensive Evaluation of GNN Training Systems: A Data Management Perspective (Hao Yuan et al., 2023)

{{<citation>}}

Hao Yuan, Yajiong Liu, Yanfeng Zhang, Xin Ai, Qiange Wang, Chaoyi Chen, Yu Gu, Ge Yu. (2023)  
**Comprehensive Evaluation of GNN Training Systems: A Data Management Perspective**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.13279v1)  

---


**ABSTRACT**  
Many Graph Neural Network (GNN) training systems have emerged recently to support efficient GNN training. Since GNNs embody complex data dependencies between training samples, the training of GNNs should address distinct challenges different from DNN training in data management, such as data partitioning, batch preparation for mini-batch training, and data transferring between CPUs and GPUs. These factors, which take up a large proportion of training time, make data management in GNN training more significant. This paper reviews GNN training from a data management perspective and provides a comprehensive analysis and evaluation of the representative approaches. We conduct extensive experiments on various benchmark datasets and show many interesting and valuable results. We also provide some practical tips learned from these experiments, which are helpful for designing GNN training systems in the future.

{{</citation>}}


### (92/118) Hard Label Black Box Node Injection Attack on Graph Neural Networks (Yu Zhou et al., 2023)

{{<citation>}}

Yu Zhou, Zihao Dong, Guofeng Zhang, Jingchen Tang. (2023)  
**Hard Label Black Box Node Injection Attack on Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.13244v1)  

---


**ABSTRACT**  
While graph neural networks have achieved state-of-the-art performances in many real-world tasks including graph classification and node classification, recent works have demonstrated they are also extremely vulnerable to adversarial attacks. Most previous works have focused on attacking node classification networks under impractical white-box scenarios. In this work, we will propose a non-targeted Hard Label Black Box Node Injection Attack on Graph Neural Networks, which to the best of our knowledge, is the first of its kind. Under this setting, more real world tasks can be studied because our attack assumes no prior knowledge about (1): the model architecture of the GNN we are attacking; (2): the model's gradients; (3): the output logits of the target GNN model. Our attack is based on an existing edge perturbation attack, from which we restrict the optimization process to formulate a node injection attack. In the work, we will evaluate the performance of the attack using three datasets, COIL-DEL, IMDB-BINARY, and NCI1.

{{</citation>}}


### (93/118) AS-LLM: When Algorithm Selection Meets Large Language Model (Xingyu Wu et al., 2023)

{{<citation>}}

Xingyu Wu, Yan Zhong, Jibin Wu, Kay Chen Tan. (2023)  
**AS-LLM: When Algorithm Selection Meets Large Language Model**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.13184v1)  

---


**ABSTRACT**  
Algorithm selection aims to identify the most suitable algorithm for solving a specific problem before execution, which has become a critical process of the AutoML. Current mainstream algorithm selection techniques rely heavily on feature representations of various problems and employ the performance of each algorithm as supervised information. However, there is a significant research gap concerning the consideration of algorithm features. This gap is primarily attributed to the inherent complexity of algorithms, making it particularly challenging to find a universally effective feature extraction method that is applicable across a diverse range of algorithms. Unfortunately, neglecting this aspect undoubtedly impacts the accuracy of algorithm selection and indirectly necessitates an increased volume of problem data for training purposes. This paper takes a significant stride towards addressing this gap by proposing an approach that integrates algorithm representation into the algorithm selection process. Specifically, our proposed model employs distinct modules to extract representations of both problems and algorithms, where the algorithm representation leverages the capabilities of pre-trained LLMs in the realm of code comprehension. Following the extraction of embedding vectors for both algorithms and problems, the most suitable algorithm is determined through calculations of matching degrees. Our experiments not only validate the effectiveness of the proposed model but also showcase the performance of different embedded pre-trained LLMs, which suggests that the proposed algorithm selection framework holds the potential to serve as a baseline task for evaluating the code representation capabilities of LLMs.

{{</citation>}}


### (94/118) ComPEFT: Compression for Communicating Parameter Efficient Updates via Sparsification and Quantization (Prateek Yadav et al., 2023)

{{<citation>}}

Prateek Yadav, Leshem Choshen, Colin Raffel, Mohit Bansal. (2023)  
**ComPEFT: Compression for Communicating Parameter Efficient Updates via Sparsification and Quantization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: LLaMA, Quantization, T5  
[Paper Link](http://arxiv.org/abs/2311.13171v1)  

---


**ABSTRACT**  
Parameter-efficient fine-tuning (PEFT) techniques make it possible to efficiently adapt a language model to create "expert" models that specialize to new tasks or domains. Recent techniques in model merging and compositional generalization leverage these expert models by dynamically composing modules to improve zero/few-shot generalization. Despite the efficiency of PEFT methods, the size of expert models can make it onerous to retrieve expert models per query over high-latency networks like the Internet or serve multiple experts on a single GPU. To address these issues, we present ComPEFT, a novel method for compressing fine-tuning residuals (task vectors) of PEFT based models. ComPEFT employs sparsification and ternary quantization to reduce the size of the PEFT module without performing any additional retraining while preserving or enhancing model performance. In extensive evaluation across T5, T0, and LLaMA-based models with 200M - 65B parameters, ComPEFT achieves compression ratios of 8x - 50x. In particular, we show that ComPEFT improves with scale - stronger models exhibit higher compressibility and better performance. For example, we show that ComPEFT applied to LLaMA outperforms QLoRA by 4.16% on MMLU with a storage size reduction of up to 26x. In addition, we show that the compressed experts produced by ComPEFT maintain few-shot compositional generalization capabilities, facilitate efficient communication and computation, and exhibit enhanced performance when merged. Lastly, we provide an analysis of different method components, compare it with other PEFT methods, and test ComPEFT's efficacy for compressing the residual of full-finetuning. Our code is available at https://github.com/prateeky2806/compeft.

{{</citation>}}


### (95/118) AdaptiveFL: Adaptive Heterogeneous Federated Learning for Resource-Constrained AIoT Systems (Chentao Jia et al., 2023)

{{<citation>}}

Chentao Jia, Ming Hu, Zekai Chen, Yanxin Yang, Xiaofei Xie, Yang Liu, Mingsong Chen. (2023)  
**AdaptiveFL: Adaptive Heterogeneous Federated Learning for Resource-Constrained AIoT Systems**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13166v1)  

---


**ABSTRACT**  
Although Federated Learning (FL) is promising to enable collaborative learning among Artificial Intelligence of Things (AIoT) devices, it suffers from the problem of low classification performance due to various heterogeneity factors (e.g., computing capacity, memory size) of devices and uncertain operating environments. To address these issues, this paper introduces an effective FL approach named AdaptiveFL based on a novel fine-grained width-wise model pruning strategy, which can generate various heterogeneous local models for heterogeneous AIoT devices. By using our proposed reinforcement learning-based device selection mechanism, AdaptiveFL can adaptively dispatch suitable heterogeneous models to corresponding AIoT devices on the fly based on their available resources for local training. Experimental results show that, compared to state-of-the-art methods, AdaptiveFL can achieve up to 16.83% inference improvements for both IID and non-IID scenarios.

{{</citation>}}


### (96/118) Have Your Cake and Eat It Too: Toward Efficient and Accurate Split Federated Learning (Dengke Yan et al., 2023)

{{<citation>}}

Dengke Yan, Ming Hu, Zeke Xia, Yanxin Yang, Jun Xia, Xiaofei Xie, Mingsong Chen. (2023)  
**Have Your Cake and Eat It Too: Toward Efficient and Accurate Split Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13163v1)  

---


**ABSTRACT**  
Due to its advantages in resource constraint scenarios, Split Federated Learning (SFL) is promising in AIoT systems. However, due to data heterogeneity and stragglers, SFL suffers from the challenges of low inference accuracy and low efficiency. To address these issues, this paper presents a novel SFL approach, named Sliding Split Federated Learning (S$^2$FL), which adopts an adaptive sliding model split strategy and a data balance-based training mechanism. By dynamically dispatching different model portions to AIoT devices according to their computing capability, S$^2$FL can alleviate the low training efficiency caused by stragglers. By combining features uploaded by devices with different data distributions to generate multiple larger batches with a uniform distribution for back-propagation, S$^2$FL can alleviate the performance degradation caused by data heterogeneity. Experimental results demonstrate that, compared to conventional SFL, S$^2$FL can achieve up to 16.5\% inference accuracy improvement and 3.54X training acceleration.

{{</citation>}}


### (97/118) LIMIT: Less Is More for Instruction Tuning Across Evaluation Paradigms (Aditi Jha et al., 2023)

{{<citation>}}

Aditi Jha, Sam Havens, Jeremey Dohmann, Alex Trott, Jacob Portes. (2023)  
**LIMIT: Less Is More for Instruction Tuning Across Evaluation Paradigms**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model, NLP, QA  
[Paper Link](http://arxiv.org/abs/2311.13133v1)  

---


**ABSTRACT**  
Large Language Models are traditionally finetuned on large instruction datasets. However recent studies suggest that small, high-quality datasets can suffice for general purpose instruction following. This lack of consensus surrounding finetuning best practices is in part due to rapidly diverging approaches to LLM evaluation. In this study, we ask whether a small amount of diverse finetuning samples can improve performance on both traditional perplexity-based NLP benchmarks, and on open-ended, model-based evaluation. We finetune open-source MPT-7B and MPT-30B models on instruction finetuning datasets of various sizes ranging from 1k to 60k samples. We find that subsets of 1k-6k instruction finetuning samples are sufficient to achieve good performance on both (1) traditional NLP benchmarks and (2) model-based evaluation. Finally, we show that mixing textbook-style and open-ended QA finetuning datasets optimizes performance on both evaluation paradigms.

{{</citation>}}


### (98/118) Combatting Human Trafficking in the Cyberspace: A Natural Language Processing-Based Methodology to Analyze the Language in Online Advertisements (Alejandro Rodriguez Perez et al., 2023)

{{<citation>}}

Alejandro Rodriguez Perez, Pablo Rivas. (2023)  
**Combatting Human Trafficking in the Cyberspace: A Natural Language Processing-Based Methodology to Analyze the Language in Online Advertisements**  

---
Primary Category: cs.LG  
Categories: 68T50, 62H30, 91C99, 68T068T50, 62H30, 91C99, 68T01, I-2-7; I-5-4; K-4-1; K-4-2, cs-AI, cs-CL, cs-CY, cs-LG, cs-SI, cs.LG  
Keywords: NLP, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2311.13118v1)  

---


**ABSTRACT**  
This project tackles the pressing issue of human trafficking in online C2C marketplaces through advanced Natural Language Processing (NLP) techniques. We introduce a novel methodology for generating pseudo-labeled datasets with minimal supervision, serving as a rich resource for training state-of-the-art NLP models. Focusing on tasks like Human Trafficking Risk Prediction (HTRP) and Organized Activity Detection (OAD), we employ cutting-edge Transformer models for analysis. A key contribution is the implementation of an interpretability framework using Integrated Gradients, providing explainable insights crucial for law enforcement. This work not only fills a critical gap in the literature but also offers a scalable, machine learning-driven approach to combat human exploitation online. It serves as a foundation for future research and practical applications, emphasizing the role of machine learning in addressing complex social issues.

{{</citation>}}


### (99/118) White-Box Transformers via Sparse Rate Reduction: Compression Is All There Is? (Yaodong Yu et al., 2023)

{{<citation>}}

Yaodong Yu, Sam Buchanan, Druv Pai, Tianzhe Chu, Ziyang Wu, Shengbang Tong, Hao Bai, Yuexiang Zhai, Benjamin D. Haeffele, Yi Ma. (2023)  
**White-Box Transformers via Sparse Rate Reduction: Compression Is All There Is?**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: BERT, GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.13110v2)  

---


**ABSTRACT**  
In this paper, we contend that a natural objective of representation learning is to compress and transform the distribution of the data, say sets of tokens, towards a low-dimensional Gaussian mixture supported on incoherent subspaces. The goodness of such a representation can be evaluated by a principled measure, called sparse rate reduction, that simultaneously maximizes the intrinsic information gain and extrinsic sparsity of the learned representation. From this perspective, popular deep network architectures, including transformers, can be viewed as realizing iterative schemes to optimize this measure. Particularly, we derive a transformer block from alternating optimization on parts of this objective: the multi-head self-attention operator compresses the representation by implementing an approximate gradient descent step on the coding rate of the features, and the subsequent multi-layer perceptron sparsifies the features. This leads to a family of white-box transformer-like deep network architectures, named CRATE, which are mathematically fully interpretable. We show, by way of a novel connection between denoising and compression, that the inverse to the aforementioned compressive encoding can be realized by the same class of CRATE architectures. Thus, the so-derived white-box architectures are universal to both encoders and decoders. Experiments show that these networks, despite their simplicity, indeed learn to compress and sparsify representations of large-scale real-world image and text datasets, and achieve performance very close to highly engineered transformer-based models: ViT, MAE, DINO, BERT, and GPT2. We believe the proposed computational framework demonstrates great potential in bridging the gap between theory and practice of deep learning, from a unified perspective of data compression. Code is available at: https://ma-lab-berkeley.github.io/CRATE .

{{</citation>}}


### (100/118) Stable Unlearnable Example: Enhancing the Robustness of Unlearnable Examples via Stable Error-Minimizing Noise (Yixin Liu et al., 2023)

{{<citation>}}

Yixin Liu, Kaidi Xu, Xun Chen, Lichao Sun. (2023)  
**Stable Unlearnable Example: Enhancing the Robustness of Unlearnable Examples via Stable Error-Minimizing Noise**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.13091v1)  

---


**ABSTRACT**  
The open source of large amounts of image data promotes the development of deep learning techniques. Along with this comes the privacy risk of these open-source image datasets being exploited by unauthorized third parties to train deep learning models for commercial or illegal purposes. To avoid the abuse of public data, a poisoning-based technique, the unlearnable example, is proposed to significantly degrade the generalization performance of models by adding a kind of imperceptible noise to the data. To further enhance its robustness against adversarial training, existing works leverage iterative adversarial training on both the defensive noise and the surrogate model. However, it still remains unknown whether the robustness of unlearnable examples primarily comes from the effect of enhancement in the surrogate model or the defensive noise. Observing that simply removing the adversarial noise on the training process of the defensive noise can improve the performance of robust unlearnable examples, we identify that solely the surrogate model's robustness contributes to the performance. Furthermore, we found a negative correlation exists between the robustness of defensive noise and the protection performance, indicating defensive noise's instability issue. Motivated by this, to further boost the robust unlearnable example, we introduce stable error-minimizing noise (SEM), which trains the defensive noise against random perturbation instead of the time-consuming adversarial perturbation to improve the stability of defensive noise. Through extensive experiments, we demonstrate that SEM achieves a new state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet Subset in terms of both effectiveness and efficiency. The code is available at https://github.com/liuyixin-louis/Stable-Unlearnable-Example.

{{</citation>}}


## cs.IT (1)



### (101/118) Private Inference in Quantized Models (Zirui Deng et al., 2023)

{{<citation>}}

Zirui Deng, Vinayak Ramkumar, Rawad Bitar, Netanel Raviv. (2023)  
**Private Inference in Quantized Models**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2311.13686v1)  

---


**ABSTRACT**  
A typical setup in many machine learning scenarios involves a server that holds a model and a user that possesses data, and the challenge is to perform inference while safeguarding the privacy of both parties. Private Inference has been extensively explored in recent years, mainly from a cryptographic standpoint via techniques like homomorphic encryption and multiparty computation. These approaches often come with high computational overhead and may degrade the accuracy of the model. In our work, we take a different approach inspired by the Private Information Retrieval literature. We view private inference as the task of retrieving inner products of parameter vectors with the data, a fundamental operation in many machine learning models. We introduce schemes that enable such retrieval of inner products for models with quantized (i.e., restricted to a finite set) weights; such models are extensively used in practice due to a wide range of benefits. In addition, our schemes uncover a fundamental tradeoff between user and server privacy. Our information-theoretic approach is applicable to a wide range of problems and robust in privacy guarantees for both the user and the server.

{{</citation>}}


## cs.DC (3)



### (102/118) A Survey of Serverless Machine Learning Model Inference (Kamil Kojs, 2023)

{{<citation>}}

Kamil Kojs. (2023)  
**A Survey of Serverless Machine Learning Model Inference**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs.DC  
Keywords: AI, Computer Vision, Generative AI, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.13587v1)  

---


**ABSTRACT**  
Recent developments in Generative AI, Computer Vision, and Natural Language Processing have led to an increased integration of AI models into various products. This widespread adoption of AI requires significant efforts in deploying these models in production environments. When hosting machine learning models for real-time predictions, it is important to meet defined Service Level Objectives (SLOs), ensuring reliability, minimal downtime, and optimizing operational costs of the underlying infrastructure. Large machine learning models often demand GPU resources for efficient inference to meet SLOs. In the context of these trends, there is growing interest in hosting AI models in a serverless architecture while still providing GPU access for inference tasks. This survey aims to summarize and categorize the emerging challenges and optimization opportunities for large-scale deep learning serving systems. By providing a novel taxonomy and summarizing recent trends, we hope that this survey could shed light on new optimization perspectives and motivate novel works in large-scale deep learning serving systems.

{{</citation>}}


### (103/118) Uncertainty Estimation in Multi-Agent Distributed Learning (Gleb Radchenko et al., 2023)

{{<citation>}}

Gleb Radchenko, Victoria Andrea Fill. (2023)  
**Uncertainty Estimation in Multi-Agent Distributed Learning**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13356v1)  

---


**ABSTRACT**  
Traditionally, IoT edge devices have been perceived primarily as low-power components with limited capabilities for autonomous operations. Yet, with emerging advancements in embedded AI hardware design, a foundational shift paves the way for future possibilities. Thus, the aim of the KDT NEUROKIT2E project is to establish a new open-source framework to further facilitate AI applications on edge devices by developing new methods in quantization, pruning-aware training, and sparsification. These innovations hold the potential to expand the functional range of such devices considerably, enabling them to manage complex Machine Learning (ML) tasks utilizing local resources and laying the groundwork for innovative learning approaches.   In the context of 6G's transformative potential, distributed learning among independent agents emerges as a pivotal application, attributed to 6G networks' support for ultra-reliable low-latency communication, enhanced data rates, and advanced edge computing capabilities.   Our research focuses on the mechanisms and methodologies that allow edge network-enabled agents to engage in collaborative learning in distributed environments. Particularly, one of the key issues within distributed collaborative learning is determining the degree of confidence in the learning results, considering the spatio-temporal locality of data sets perceived by independent agents.

{{</citation>}}


### (104/118) NeutronOrch: Rethinking Sample-based GNN Training under CPU-GPU Heterogeneous Environments (Xin Ai et al., 2023)

{{<citation>}}

Xin Ai, Qiange Wang, Chunyu Cao, Yanfeng Zhang, Chaoyi Chen, Hao Yuan, Yu Gu, Ge Yu. (2023)  
**NeutronOrch: Rethinking Sample-based GNN Training under CPU-GPU Heterogeneous Environments**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs.DC  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.13225v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have demonstrated outstanding performance in various applications. Existing frameworks utilize CPU-GPU heterogeneous environments to train GNN models and integrate mini-batch and sampling techniques to overcome the GPU memory limitation. In CPU-GPU heterogeneous environments, we can divide sample-based GNN training into three steps: sample, gather, and train. Existing GNN systems use different task orchestrating methods to employ each step on CPU or GPU. After extensive experiments and analysis, we find that existing task orchestrating methods fail to fully utilize the heterogeneous resources, limited by inefficient CPU processing or GPU resource contention. In this paper, we propose NeutronOrch, a system for sample-based GNN training that incorporates a layer-based task orchestrating method and ensures balanced utilization of the CPU and GPU. NeutronOrch decouples the training process by layer and pushes down the training task of the bottom layer to the CPU. This significantly reduces the computational load and memory footprint of GPU training. To avoid inefficient CPU processing, NeutronOrch only offloads the training of frequently accessed vertices to the CPU and lets GPU reuse their embeddings with bounded staleness. Furthermore, NeutronOrch provides a fine-grained pipeline design for the layer-based task orchestrating method, fully overlapping different tasks on heterogeneous resources while strictly guaranteeing bounded staleness. The experimental results show that compared with the state-of-the-art GNN systems, NeutronOrch can achieve up to 4.61x performance speedup.

{{</citation>}}


## cs.CR (1)



### (105/118) Summary Reports Optimization in the Privacy Sandbox Attribution Reporting API (Hidayet Aksu et al., 2023)

{{<citation>}}

Hidayet Aksu, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Adam Sealfon, Avinash V Varadarajan. (2023)  
**Summary Reports Optimization in the Privacy Sandbox Attribution Reporting API**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.13586v1)  

---


**ABSTRACT**  
The Privacy Sandbox Attribution Reporting API has been recently deployed by Google Chrome to support the basic advertising functionality of attribution reporting (aka conversion measurement) after deprecation of third-party cookies. The API implements a collection of privacy-enhancing guardrails including contribution bounding and noise injection. It also offers flexibility for the analyst to allocate the contribution budget.   In this work, we present methods for optimizing the allocation of the contribution budget for summary reports from the Attribution Reporting API. We evaluate them on real-world datasets as well as on a synthetic data model that we find to accurately capture real-world conversion data. Our results demonstrate that optimizing the parameters that can be set by the analyst can significantly improve the utility achieved by querying the API while satisfying the same privacy bounds.

{{</citation>}}


## quant-ph (1)



### (106/118) Enigma: Privacy-Preserving Execution of QAOA on Untrusted Quantum Computers (Ramin Ayanzadeh et al., 2023)

{{<citation>}}

Ramin Ayanzadeh, Ahmad Mousavi, Narges Alavisamani, Moinuddin Qureshi. (2023)  
**Enigma: Privacy-Preserving Execution of QAOA on Untrusted Quantum Computers**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-CR, cs-DM, cs-ET, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.13546v1)  

---


**ABSTRACT**  
Quantum computers can solve problems that are beyond the capabilities of conventional computers. As quantum computers are expensive and hard to maintain, the typical model for performing quantum computation is to send the circuit to a quantum cloud provider. This leads to privacy concerns for commercial entities as an untrusted server can learn protected information from the provided circuit. Current proposals for Secure Quantum Computing (SQC) either rely on emerging technologies (such as quantum networks) or incur prohibitive overheads (for Quantum Homomorphic Encryption). The goal of our paper is to enable low-cost privacy-preserving quantum computation that can be used with current systems.   We propose Enigma, a suite of privacy-preserving schemes specifically designed for the Quantum Approximate Optimization Algorithm (QAOA). Unlike previous SQC techniques that obfuscate quantum circuits, Enigma transforms the input problem of QAOA, such that the resulting circuit and the outcomes are unintelligible to the server. We introduce three variants of Enigma. Enigma-I protects the coefficients of QAOA using random phase flipping and fudging of values. Enigma-II protects the nodes of the graph by introducing decoy qubits, which are indistinguishable from primary ones. Enigma-III protects the edge information of the graph by modifying the graph such that each node has an identical number of connections. For all variants of Enigma, we demonstrate that we can still obtain the solution for the original problem. We evaluate Enigma using IBM quantum devices and show that the privacy improvements of Enigma come at only a small reduction in fidelity (1%-13%).

{{</citation>}}


## cs.CY (2)



### (107/118) Current Topological and Machine Learning Applications for Bias Detection in Text (Colleen Farrelly et al., 2023)

{{<citation>}}

Colleen Farrelly, Yashbir Singh, Quincy A. Hathaway, Gunnar Carlsson, Ashok Choudhary, Rahul Paul, Gianfranco Doretto, Yassine Himeur, Shadi Atalls, Wathiq Mansoor. (2023)  
**Current Topological and Machine Learning Applications for Bias Detection in Text**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs-LG, cs.CY  
Keywords: BERT, Bias  
[Paper Link](http://arxiv.org/abs/2311.13495v1)  

---


**ABSTRACT**  
Institutional bias can impact patient outcomes, educational attainment, and legal system navigation. Written records often reflect bias, and once bias is identified; it is possible to refer individuals for training to reduce bias. Many machine learning tools exist to explore text data and create predictive models that can search written records to identify real-time bias. However, few previous studies investigate large language model embeddings and geometric models of biased text data to understand geometry's impact on bias modeling accuracy. To overcome this issue, this study utilizes the RedditBias database to analyze textual biases. Four transformer models, including BERT and RoBERTa variants, were explored. Post-embedding, t-SNE allowed two-dimensional visualization of data. KNN classifiers differentiated bias types, with lower k-values proving more effective. Findings suggest BERT, particularly mini BERT, excels in bias classification, while multilingual models lag. The recommendation emphasizes refining monolingual models and exploring domain-specific biases.

{{</citation>}}


### (108/118) Intention and Context Elicitation with Large Language Models in the Legal Aid Intake Process (Nick Goodson et al., 2023)

{{<citation>}}

Nick Goodson, Rongfei Lu. (2023)  
**Intention and Context Elicitation with Large Language Models in the Legal Aid Intake Process**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY  
Keywords: Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2311.13281v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) and chatbots show significant promise in streamlining the legal intake process. This advancement can greatly reduce the workload and costs for legal aid organizations, improving availability while making legal assistance more accessible to a broader audience. However, a key challenge with current LLMs is their tendency to overconfidently deliver an immediate 'best guess' to a client's question based on the output distribution learned over the training data. This approach often overlooks the client's actual intentions or the specifics of their legal situation. As a result, clients may not realize the importance of providing essential additional context or expressing their underlying intentions, which are crucial for their legal cases. Traditionally, logic based decision trees have been used to automate intake for specific access to justice issues, such as immigration and eviction. But those solutions lack scalability. We demonstrate a proof-of-concept using LLMs to elicit and infer clients' underlying intentions and specific legal circumstances through free-form, language-based interactions. We also propose future research directions to use supervised fine-tuning or offline reinforcement learning to automatically incorporate intention and context elicitation in chatbots without explicit prompting.

{{</citation>}}


## q-bio.QM (1)



### (109/118) Benchmarking Toxic Molecule Classification using Graph Neural Networks and Few Shot Learning (Bhavya Mehta et al., 2023)

{{<citation>}}

Bhavya Mehta, Kush Kothari, Reshmika Nambiar, Seema Shrawne. (2023)  
**Benchmarking Toxic Molecule Classification using Graph Neural Networks and Few Shot Learning**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, q-bio-QM, q-bio.QM  
Keywords: Attention, Augmentation, Few-Shot, Graph Convolutional Network, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.13490v1)  

---


**ABSTRACT**  
Traditional methods like Graph Convolutional Networks (GCNs) face challenges with limited data and class imbalance, leading to suboptimal performance in graph classification tasks during toxicity prediction of molecules as a whole. To address these issues, we harness the power of Graph Isomorphic Networks, Multi Headed Attention and Free Large-scale Adversarial Augmentation separately on Graphs for precisely capturing the structural data of molecules and their toxicological properties. Additionally, we incorporate Few-Shot Learning to improve the model's generalization with limited annotated samples. Extensive experiments on a diverse toxicology dataset demonstrate that our method achieves an impressive state-of-art AUC-ROC value of 0.816, surpassing the baseline GCN model by 11.4%. This highlights the significance of our proposed methodology and Few Shot Learning in advancing Toxic Molecular Classification, with the potential to enhance drug discovery and environmental risk assessment processes.

{{</citation>}}


## cs.DM (1)



### (110/118) Solution discovery via reconfiguration for problems in P (Mario Grobler et al., 2023)

{{<citation>}}

Mario Grobler, Stephanie Maaz, Nicole Megow, Amer E. Mouawad, Vijayaragunathan Ramamoorthi, Daniel Schmand, Sebastian Siebertz. (2023)  
**Solution discovery via reconfiguration for problems in P**  

---
Primary Category: cs.DM  
Categories: cs-DM, cs-DS, cs.DM, math-CO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13478v1)  

---


**ABSTRACT**  
In the recently introduced framework of solution discovery via reconfiguration [Fellows et al., ECAI 2023], we are given an initial configuration of $k$ tokens on a graph and the question is whether we can transform this configuration into a feasible solution (for some problem) via a bounded number $b$ of small modification steps. In this work, we study solution discovery variants of polynomial-time solvable problems, namely Spanning Tree Discovery, Shortest Path Discovery, Matching Discovery, and Vertex/Edge Cut Discovery in the unrestricted token addition/removal model, the token jumping model, and the token sliding model. In the unrestricted token addition/removal model, we show that all four discovery variants remain in P. For the toking jumping model we also prove containment in P, except for Vertex/Edge Cut Discovery, for which we prove NP-completeness. Finally, in the token sliding model, almost all considered problems become NP-complete, the exception being Spanning Tree Discovery, which remains polynomial-time solvable. We then study the parameterized complexity of the NP-complete problems and provide a full classification of tractability with respect to the parameters solution size (number of tokens) $k$ and transformation budget (number of steps) $b$. Along the way, we observe strong connections between the solution discovery variants of our base problems and their (weighted) rainbow variants as well as their red-blue variants with cardinality constraints.

{{</citation>}}


## q-bio.BM (1)



### (111/118) Accelerating Inference in Molecular Diffusion Models with Latent Representations of Protein Structure (Ian Dunn et al., 2023)

{{<citation>}}

Ian Dunn, David Ryan Koes. (2023)  
**Accelerating Inference in Molecular Diffusion Models with Latent Representations of Protein Structure**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio.BM  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.13466v1)  

---


**ABSTRACT**  
Diffusion generative models have emerged as a powerful framework for addressing problems in structural biology and structure-based drug design. These models operate directly on 3D molecular structures. Due to the unfavorable scaling of graph neural networks (GNNs) with graph size as well as the relatively slow inference speeds inherent to diffusion models, many existing molecular diffusion models rely on coarse-grained representations of protein structure to make training and inference feasible. However, such coarse-grained representations discard essential information for modeling molecular interactions and impair the quality of generated structures. In this work, we present a novel GNN-based architecture for learning latent representations of molecular structure. When trained end-to-end with a diffusion model for de novo ligand design, our model achieves comparable performance to one with an all-atom protein representation while exhibiting a 3-fold reduction in inference time.

{{</citation>}}


## cs.AR (2)



### (112/118) SystemC Model of Power Side-Channel Attacks Against AI Accelerators: Superstition or not? (Andrija Nešković et al., 2023)

{{<citation>}}

Andrija Nešković, Saleh Mulhem, Alexander Treff, Rainer Buchty, Thomas Eisenbarth, Mladen Berekovic. (2023)  
**SystemC Model of Power Side-Channel Attacks Against AI Accelerators: Superstition or not?**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.13387v1)  

---


**ABSTRACT**  
As training artificial intelligence (AI) models is a lengthy and hence costly process, leakage of such a model's internal parameters is highly undesirable. In the case of AI accelerators, side-channel information leakage opens up the threat scenario of extracting the internal secrets of pre-trained models. Therefore, sufficiently elaborate methods for design verification as well as fault and security evaluation at the electronic system level are in demand. In this paper, we propose estimating information leakage from the early design steps of AI accelerators to aid in a more robust architectural design. We first introduce the threat scenario before diving into SystemC as a standard method for early design evaluation and how this can be applied to threat modeling. We present two successful side-channel attack methods executed via SystemC-based power modeling: correlation power analysis and template attack, both leading to total information leakage. The presented models are verified against an industry-standard netlist-level power estimation to prove general feasibility and determine accuracy. Consequently, we explore the impact of additive noise in our simulation to establish indicators for early threat evaluation. The presented approach is again validated via a model-vs-netlist comparison, showing high accuracy of the achieved results. This work hence is a solid step towards fast attack deployment and, subsequently, the design of attack-resilient AI accelerators.

{{</citation>}}


### (113/118) Softmax Acceleration with Adaptive Numeric Format for both Training and Inference (Tianhua Xia et al., 2023)

{{<citation>}}

Tianhua Xia, Sai Qian Zhang. (2023)  
**Softmax Acceleration with Adaptive Numeric Format for both Training and Inference**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.13290v1)  

---


**ABSTRACT**  
The attention mechanism is a pivotal element within the Transformer architecture, making a substantial contribution to its exceptional performance. Within this attention mechanism, Softmax is an imperative component that enables the model to assess the degree of correlation between various segments of the input. Yet, prior research has shown that Softmax operations can significantly increase processing latency and energy consumption in the Transformer network due to their internal nonlinear operations and data dependencies. In this work, we proposed~\textit{Hyft}, a hardware efficient floating point Softmax accelerator for both training and inference. Hyft aims to reduce the implementation cost of different nonlinear arithmetic operations by adaptively converting intermediate results into the most suitable numeric format for each specific operation, leading to reconfigurable accelerator with hybrid numeric format. The evaluation results highlight that Hyft achieves a remarkable $15\times$ reduction in hardware resource utilization and a $20 \times$ reduction in processing latency, all while maintaining a negligible impact on Transformer accuracy.

{{</citation>}}


## cs.SI (2)



### (114/118) An Analysis of Socialbots Activity and Influence in Modern Japanese Social Media (Shuhei Ippa et al., 2023)

{{<citation>}}

Shuhei Ippa, Masaki Hashimoto. (2023)  
**An Analysis of Socialbots Activity and Influence in Modern Japanese Social Media**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2311.13334v1)  

---


**ABSTRACT**  
In recent years, the proliferation of disinformation has become an issue against the backdrop of the spread of social media. In this study, we focus on socialbots, one of the causes of this problem, and analyze several domestic cases to clarify the actual activities and influence of socialbots. As a result of this analysis, we found that the influence of socialbots is greater in Japan than in the U.S. presidential election of 2016, which is a representative case of socialbot influence, and that socialbots retweeted by humans are not significantly different from human accounts. In addition, socialbot accounts retweeted by humans are not significantly different from human accounts. This paper also discusses specific methods and perspectives for further analysis and research on the influence of socialbots.

{{</citation>}}


### (115/118) Top-$L$ Most Influential Community Detection Over Social Networks (Technical Report) (Nan Zhang et al., 2023)

{{<citation>}}

Nan Zhang, Yutong Ye, Xiang Lian, Mingsong Chen. (2023)  
**Top-$L$ Most Influential Community Detection Over Social Networks (Technical Report)**  

---
Primary Category: cs.SI  
Categories: cs-DB, cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2311.13162v1)  

---


**ABSTRACT**  
In many real-world applications such as social network analysis and online marketing/advertising, the \textit{community detection} is a fundamental task to identify communities (subgraphs) in social networks with high structural cohesiveness. While previous works focus on detecting communities alone, they do not consider the collective influences of users in these communities on other user nodes in social networks. Inspired by this, in this paper, we investigate the influence propagation from some \textit{seed communities} and their influential effects that result in the \textit{influenced communities}. We propose a novel problem, named \textit{\underline{Top-$L$} most \underline{I}nfluential \underline{C}ommunity \underline{DE}tection} (Top$L$-ICDE) over social networks, which aims to retrieve top-$L$ seed communities with the highest influences, having high structural cohesiveness, and containing user-specified query keywords. In order to efficiently tackle the Top$L$-ICDE problem, we design effective pruning strategies to filter out false alarms of seed communities and propose an effective index mechanism to facilitate efficient Top-$L$ community retrieval. We develop an efficient Top$L$-ICDE answering algorithm by traversing the index and applying our proposed pruning strategies. We also formulate and tackle a variant of Top$L$-ICDE, named \textit{diversified top-$L$ most influential community detection} (DTop$L$-ICDE), which returns a set of $L$ diversified communities with the highest diversity score (i.e., collaborative influences by $L$ communities). We prove that DTop$L$-ICDE is NP-hard, and propose an efficient greedy algorithm with our designed diversity score pruning. Through extensive experiments, we verify the efficiency and effectiveness of our proposed Top$L$-ICDE and DTop$L$-ICDE approaches over real/synthetic social networks under various parameter settings.

{{</citation>}}


## cs.NI (1)



### (116/118) Ten issues of NetGPT (Wen Tong et al., 2023)

{{<citation>}}

Wen Tong, Chenghui Peng, Tingting Yang, Fei Wang, Juan Deng, Rongpeng Li, Lu Yang, Honggang Zhang, Dong Wang, Ming Ai, Li Yang, Guangyi Liu, Yang Yang, Yao Xiao, Liexiang Yue, Wanfei Sun, Zexu Li, Wenwen Sun. (2023)  
**Ten issues of NetGPT**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2311.13106v1)  

---


**ABSTRACT**  
With the rapid development and application of foundation models (FMs), it is foreseeable that FMs will play an important role in future wireless communications. As current Artificial Intelligence (AI) algorithms applied in wireless networks are dedicated models that aim for different neural network architectures and objectives, drawbacks in aspects of generality, performance gain, management, collaboration, etc. need to be conquered. In this paper, we define NetGPT (Network Generative Pre-trained Transformer) -- the foundation models for wireless communications, and summarize ten issues regarding design and application of NetGPT.

{{</citation>}}


## cs.RO (1)



### (117/118) Learning to Fly in Seconds (Jonas Eschmann et al., 2023)

{{<citation>}}

Jonas Eschmann, Dario Albani, Giuseppe Loianno. (2023)  
**Learning to Fly in Seconds**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.13081v1)  

---


**ABSTRACT**  
Learning-based methods, particularly Reinforcement Learning (RL), hold great promise for streamlining deployment, enhancing performance, and achieving generalization in the control of autonomous multirotor aerial vehicles. Deep RL has been able to control complex systems with impressive fidelity and agility in simulation but the simulation-to-reality transfer often brings a hard-to-bridge reality gap. Moreover, RL is commonly plagued by prohibitively long training times. In this work, we propose a novel asymmetric actor-critic-based architecture coupled with a highly reliable RL-based training paradigm for end-to-end quadrotor control. We show how curriculum learning and a highly optimized simulator enhance sample complexity and lead to fast training times. To precisely discuss the challenges related to low-level/end-to-end multirotor control, we also introduce a taxonomy that classifies the existing levels of control abstractions as well as non-linearities and domain parameters. Our framework enables Simulation-to-Reality (Sim2Real) transfer for direct RPM control after only 18 seconds of training on a consumer-grade laptop as well as its deployment on microcontrollers to control a multirotor under real-time guarantees. Finally, our solution exhibits competitive performance in trajectory tracking, as demonstrated through various experimental comparisons with existing state-of-the-art control solutions using a real Crazyflie nano quadrotor. We open source the code including a very fast multirotor dynamics simulator that can simulate about 5 months of flight per second on a laptop GPU. The fast training times and deployment to a cheap, off-the-shelf quadrotor lower the barriers to entry and help democratize the research and development of these systems.

{{</citation>}}


## eess.SY (1)



### (118/118) High-Speed Voltage Control in Active Distribution Systems with Smart Inverter Coordination and Deep Reinforcement Learning (Mohammad Golgol et al., 2023)

{{<citation>}}

Mohammad Golgol, Anamitra Pal. (2023)  
**High-Speed Voltage Control in Active Distribution Systems with Smart Inverter Coordination and Deep Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.13080v1)  

---


**ABSTRACT**  
The increasing penetration of renewable energy resources in distribution systems necessitates high-speed monitoring and control of voltage for ensuring reliable system operation. However, existing voltage control algorithms often make simplifying assumptions in their formulation, such as real-time availability of smart meter measurements (for monitoring), or real-time knowledge of every power injection information(for control).This paper leverages the recent advances made in highspeed state estimation for real-time unobservable distribution systems to formulate a deep reinforcement learning-based control algorithm that utilizes the state estimates alone to control the voltage of the entire system. The results obtained for a modified (renewable-rich) IEEE34-nodedistributionfeeder indicate that the proposed approach excels in monitoring and controlling voltage of active distribution systems.

{{</citation>}}
