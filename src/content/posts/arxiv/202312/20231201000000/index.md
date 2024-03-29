---
draft: false
title: "arXiv @ 2023.12.01"
date: 2023-12-01
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.01"
    identifier: arxiv_20231201
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (31)](#cscl-31)
- [cs.CV (67)](#cscv-67)
- [cs.GT (1)](#csgt-1)
- [cs.LG (25)](#cslg-25)
- [eess.IV (2)](#eessiv-2)
- [astro-ph.IM (1)](#astro-phim-1)
- [cs.CR (3)](#cscr-3)
- [physics.med-ph (1)](#physicsmed-ph-1)
- [cs.HC (1)](#cshc-1)
- [cs.CY (1)](#cscy-1)
- [physics.flu-dyn (1)](#physicsflu-dyn-1)
- [cs.DC (3)](#csdc-3)
- [cs.RO (2)](#csro-2)
- [cs.SD (1)](#cssd-1)
- [cs.IR (2)](#csir-2)
- [cs.IT (1)](#csit-1)
- [cs.SI (2)](#cssi-2)
- [eess.SY (1)](#eesssy-1)
- [cs.AI (4)](#csai-4)
- [cs.NI (2)](#csni-2)
- [cs.OS (1)](#csos-1)
- [cs.DB (1)](#csdb-1)
- [stat.ML (1)](#statml-1)

## cs.CL (31)



### (1/155) Uncertainty Guided Global Memory Improves Multi-Hop Question Answering (Alsu Sagirova et al., 2023)

{{<citation>}}

Alsu Sagirova, Mikhail Burtsev. (2023)  
**Uncertainty Guided Global Memory Improves Multi-Hop Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.18151v1)  

---


**ABSTRACT**  
Transformers have become the gold standard for many natural language processing tasks and, in particular, for multi-hop question answering (MHQA). This task includes processing a long document and reasoning over the multiple parts of it. The landscape of MHQA approaches can be classified into two primary categories. The first group focuses on extracting supporting evidence, thereby constraining the QA model's context to predicted facts. Conversely, the second group relies on the attention mechanism of the long input encoding model to facilitate multi-hop reasoning. However, attention-based token representations lack explicit global contextual information to connect reasoning steps. To address these issues, we propose GEMFormer, a two-stage method that first collects relevant information over the entire document to the memory and then combines it with local context to solve the task. Our experimental results show that fine-tuning a pre-trained model with memory-augmented input, including the most certain global elements, improves the model's performance on three MHQA datasets compared to the baseline. We also found that the global explicit memory contains information from supporting facts required for the correct answer.

{{</citation>}}


### (2/155) ROBBIE: Robust Bias Evaluation of Large Generative Language Models (David Esiobu et al., 2023)

{{<citation>}}

David Esiobu, Xiaoqing Tan, Saghar Hosseini, Megan Ung, Yuchen Zhang, Jude Fernandes, Jane Dwivedi-Yu, Eleonora Presani, Adina Williams, Eric Michael Smith. (2023)  
**ROBBIE: Robust Bias Evaluation of Large Generative Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18140v1)  

---


**ABSTRACT**  
As generative large language models (LLMs) grow more performant and prevalent, we must develop comprehensive enough tools to measure and improve their fairness. Different prompt-based datasets can be used to measure social bias across multiple text domains and demographic axes, meaning that testing LLMs on more datasets can potentially help us characterize their biases more fully, and better ensure equal and equitable treatment of marginalized demographic groups. In this work, our focus is two-fold:   (1) Benchmarking: a comparison of 6 different prompt-based bias and toxicity metrics across 12 demographic axes and 5 families of generative LLMs. Out of those 6 metrics, AdvPromptSet and HolisticBiasR are novel datasets proposed in the paper. The comparison of those benchmarks gives us insights about the bias and toxicity of the compared models. Therefore, we explore the frequency of demographic terms in common LLM pre-training corpora and how this may relate to model biases.   (2) Mitigation: we conduct a comprehensive study of how well 3 bias/toxicity mitigation techniques perform across our suite of measurements. ROBBIE aims to provide insights for practitioners while deploying a model, emphasizing the need to not only measure potential harms, but also understand how they arise by characterizing the data, mitigate harms once found, and balance any trade-offs. We open-source our analysis code in hopes of encouraging broader measurements of bias in future LLMs.

{{</citation>}}


### (3/155) TurkishBERTweet: Fast and Reliable Large Language Model for Social Media Analysis (Ali Najafi et al., 2023)

{{<citation>}}

Ali Najafi, Onur Varol. (2023)  
**TurkishBERTweet: Fast and Reliable Large Language Model for Social Media Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SI, cs.CL  
Keywords: AI, BERT, Hate Speech Detection, Language Model, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2311.18063v1)  

---


**ABSTRACT**  
Turkish is one of the most popular languages in the world. Wide us of this language on social media platforms such as Twitter, Instagram, or Tiktok and strategic position of the country in the world politics makes it appealing for the social network researchers and industry. To address this need, we introduce TurkishBERTweet, the first large scale pre-trained language model for Turkish social media built using almost 900 million tweets. The model shares the same architecture as base BERT model with smaller input length, making TurkishBERTweet lighter than BERTurk and can have significantly lower inference time. We trained our model using the same approach for RoBERTa model and evaluated on two text classification tasks: Sentiment Classification and Hate Speech Detection. We demonstrate that TurkishBERTweet outperforms the other available alternatives on generalizability and its lower inference time gives significant advantage to process large-scale datasets. We also compared our models with the commercial OpenAI solutions in terms of cost and performance to demonstrate TurkishBERTweet is scalable and cost-effective solution. As part of our research, we released TurkishBERTweet and fine-tuned LoRA adapters for the mentioned tasks under the MIT License to facilitate future research and applications on Turkish social media. Our TurkishBERTweet model is available at: https://github.com/ViralLab/TurkishBERTweet

{{</citation>}}


### (4/155) I Know You Did Not Write That! A Sampling Based Watermarking Method for Identifying Machine Generated Text (Kaan Efe Keleş et al., 2023)

{{<citation>}}

Kaan Efe Keleş, Ömer Kaan Gürbüz, Mucahid Kutlu. (2023)  
**I Know You Did Not Write That! A Sampling Based Watermarking Method for Identifying Machine Generated Text**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.18054v1)  

---


**ABSTRACT**  
Potential harms of Large Language Models such as mass misinformation and plagiarism can be partially mitigated if there exists a reliable way to detect machine generated text. In this paper, we propose a new watermarking method to detect machine-generated texts. Our method embeds a unique pattern within the generated text, ensuring that while the content remains coherent and natural to human readers, it carries distinct markers that can be identified algorithmically. Specifically, we intervene with the token sampling process in a way which enables us to trace back our token choices during the detection phase. We show how watermarking affects textual quality and compare our proposed method with a state-of-the-art watermarking method in terms of robustness and detectability. Through extensive experiments, we demonstrate the effectiveness of our watermarking scheme in distinguishing between watermarked and non-watermarked text, achieving high detection rates while maintaining textual quality.

{{</citation>}}


### (5/155) Zero-shot Conversational Summarization Evaluations with small Large Language Models (Ramesh Manuvinakurike et al., 2023)

{{<citation>}}

Ramesh Manuvinakurike, Saurav Sahay, Sangeeta Manepalli, Lama Nachman. (2023)  
**Zero-shot Conversational Summarization Evaluations with small Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2311.18041v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) exhibit powerful summarization abilities. However, their capabilities on conversational summarization remains under explored. In this work we evaluate LLMs (approx. 10 billion parameters) on conversational summarization and showcase their performance on various prompts. We show that the summaries generated by models depend on the instructions and the performance of LLMs vary with different instructions sometimes resulting steep drop in ROUGE scores if prompts are not selected carefully. We also evaluate the models with human evaluations and discuss the limitations of the models on conversational summarization

{{</citation>}}


### (6/155) Hyperpolyglot LLMs: Cross-Lingual Interpretability in Token Embeddings (Andrea W Wen-Yi et al., 2023)

{{<citation>}}

Andrea W Wen-Yi, David Mimno. (2023)  
**Hyperpolyglot LLMs: Cross-Lingual Interpretability in Token Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Embedding, T5  
[Paper Link](http://arxiv.org/abs/2311.18034v1)  

---


**ABSTRACT**  
Cross-lingual transfer learning is an important property of multilingual large language models (LLMs). But how do LLMs represent relationships between languages? Every language model has an input layer that maps tokens to vectors. This ubiquitous layer of language models is often overlooked. We find that similarities between these input embeddings are highly interpretable and that the geometry of these embeddings differs between model families. In one case (XLM-RoBERTa), embeddings encode language: tokens in different writing systems can be linearly separated with an average of 99.2% accuracy. Another family (mT5) represents cross-lingual semantic similarity: the 50 nearest neighbors for any token represent an average of 7.61 writing systems, and are frequently translations. This result is surprising given that there is no explicit parallel cross-lingual training corpora and no explicit incentive for translations in pre-training objectives. Our research opens the door for investigations in 1) The effect of pre-training and model architectures on representations of languages and 2) The applications of cross-lingual representations embedded in language models.

{{</citation>}}


### (7/155) Filtered Semi-Markov CRF (Urchade Zaratiana et al., 2023)

{{<citation>}}

Urchade Zaratiana, Nadi Tomeh, Niama El Khbir, Pierre Holat, Thierry Charnois. (2023)  
**Filtered Semi-Markov CRF**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2311.18028v1)  

---


**ABSTRACT**  
Semi-Markov CRF has been proposed as an alternative to the traditional Linear Chain CRF for text segmentation tasks such as Named Entity Recognition (NER). Unlike CRF, which treats text segmentation as token-level prediction, Semi-CRF considers segments as the basic unit, making it more expressive. However, Semi-CRF suffers from two major drawbacks: (1) quadratic complexity over sequence length, as it operates on every span of the input sequence, and (2) inferior performance compared to CRF for sequence labeling tasks like NER. In this paper, we introduce Filtered Semi-Markov CRF, a variant of Semi-CRF that addresses these issues by incorporating a filtering step to eliminate irrelevant segments, reducing complexity and search space. Our approach is evaluated on several NER benchmarks, where it outperforms both CRF and Semi-CRF while being significantly faster. The implementation of our method is available on \href{https://github.com/urchade/Filtered-Semi-Markov-CRF}{Github}.

{{</citation>}}


### (8/155) A Pipeline For Discourse Circuits From CCG (Jonathon Liu et al., 2023)

{{<citation>}}

Jonathon Liu, Razin A. Shaikh, Benjamin Rodatz, Richie Yeung, Bob Coecke. (2023)  
**A Pipeline For Discourse Circuits From CCG**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.17892v1)  

---


**ABSTRACT**  
There is a significant disconnect between linguistic theory and modern NLP practice, which relies heavily on inscrutable black-box architectures. DisCoCirc is a newly proposed model for meaning that aims to bridge this divide, by providing neuro-symbolic models that incorporate linguistic structure. DisCoCirc represents natural language text as a `circuit' that captures the core semantic information of the text. These circuits can then be interpreted as modular machine learning models. Additionally, DisCoCirc fulfils another major aim of providing an NLP model that can be implemented on near-term quantum computers.   In this paper we describe a software pipeline that converts English text to its DisCoCirc representation. The pipeline achieves coverage over a large fragment of the English language. It relies on Combinatory Categorial Grammar (CCG) parses of the input text as well as coreference resolution information. This semantic and syntactic information is used in several steps to convert the text into a simply-typed $\lambda$-calculus term, and then into a circuit diagram. This pipeline will enable the application of the DisCoCirc framework to NLP tasks, using both classical and quantum approaches.

{{</citation>}}


### (9/155) Supervising the Centroid Baseline for Extractive Multi-Document Summarization (Simão Gonçalves et al., 2023)

{{<citation>}}

Simão Gonçalves, Gonçalo Correia, Diogo Pernes, Afonso Mendes. (2023)  
**Supervising the Centroid Baseline for Extractive Multi-Document Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.17771v1)  

---


**ABSTRACT**  
The centroid method is a simple approach for extractive multi-document summarization and many improvements to its pipeline have been proposed. We further refine it by adding a beam search process to the sentence selection and also a centroid estimation attention model that leads to improved results. We demonstrate this in several multi-document summarization datasets, including in a multilingual scenario.

{{</citation>}}


### (10/155) Mukhyansh: A Headline Generation Dataset for Indic Languages (Lokesh Madasu et al., 2023)

{{<citation>}}

Lokesh Madasu, Gopichand Kanumolu, Nirmal Surange, Manish Shrivastava. (2023)  
**Mukhyansh: A Headline Generation Dataset for Indic Languages**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.17743v1)  

---


**ABSTRACT**  
The task of headline generation within the realm of Natural Language Processing (NLP) holds immense significance, as it strives to distill the true essence of textual content into concise and attention-grabbing summaries. While noteworthy progress has been made in headline generation for widely spoken languages like English, there persist numerous challenges when it comes to generating headlines in low-resource languages, such as the rich and diverse Indian languages. A prominent obstacle that specifically hinders headline generation in Indian languages is the scarcity of high-quality annotated data. To address this crucial gap, we proudly present Mukhyansh, an extensive multilingual dataset, tailored for Indian language headline generation. Comprising an impressive collection of over 3.39 million article-headline pairs, Mukhyansh spans across eight prominent Indian languages, namely Telugu, Tamil, Kannada, Malayalam, Hindi, Bengali, Marathi, and Gujarati. We present a comprehensive evaluation of several state-of-the-art baseline models. Additionally, through an empirical analysis of existing works, we demonstrate that Mukhyansh outperforms all other models, achieving an impressive average ROUGE-L score of 31.43 across all 8 languages.

{{</citation>}}


### (11/155) SenTest: Evaluating Robustness of Sentence Encoders (Tanmay Chavan et al., 2023)

{{<citation>}}

Tanmay Chavan, Shantanu Patankar, Aditya Kane, Omkar Gokhale, Geetanjali Kale, Raviraj Joshi. (2023)  
**SenTest: Evaluating Robustness of Sentence Encoders**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.17722v1)  

---


**ABSTRACT**  
Contrastive learning has proven to be an effective method for pre-training models using weakly labeled data in the vision domain. Sentence transformers are the NLP counterparts to this architecture, and have been growing in popularity due to their rich and effective sentence representations. Having effective sentence representations is paramount in multiple tasks, such as information retrieval, retrieval augmented generation (RAG), and sentence comparison. Keeping in mind the deployability factor of transformers, evaluating the robustness of sentence transformers is of utmost importance. This work focuses on evaluating the robustness of the sentence encoders. We employ several adversarial attacks to evaluate its robustness. This system uses character-level attacks in the form of random character substitution, word-level attacks in the form of synonym replacement, and sentence-level attacks in the form of intra-sentence word order shuffling. The results of the experiments strongly undermine the robustness of sentence encoders. The models produce significantly different predictions as well as embeddings on perturbed datasets. The accuracy of the models can fall up to 15 percent on perturbed datasets as compared to unperturbed datasets. Furthermore, the experiments demonstrate that these embeddings does capture the semantic and syntactic structure (sentence order) of sentences. However, existing supervised classification strategies fail to leverage this information, and merely function as n-gram detectors.

{{</citation>}}


### (12/155) How to Build an AI Tutor that Can Adapt to Any Course and Provide Accurate Answers Using Large Language Model and Retrieval-Augmented Generation (Chenxi Dong, 2023)

{{<citation>}}

Chenxi Dong. (2023)  
**How to Build an AI Tutor that Can Adapt to Any Course and Provide Accurate Answers Using Large Language Model and Retrieval-Augmented Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17696v2)  

---


**ABSTRACT**  
Artificial intelligence is transforming education through data-driven, personalized learning solutions. This paper introduces AI Tutor, an innovative web application that provides personalized tutoring in any subject using state-of-the-art Large Language Model (LLM). AI Tutor ingests course materials to construct an adaptive knowledge base tailored to the course. When students pose questions, it retrieves the most relevant information and generates detailed, conversational responses citing supporting evidence. The system is powered by advanced large language models and Retrieval-Augmented Generation (RAG) techniques for accurate, natural question answering. We present a fully-functional web interface and video demonstration that showcase AI Tutor's versatility across diverse subjects and its ability to produce pedagogically cogent responses. While an initial prototype, this work represents a pioneering step toward AI-enabled tutoring systems that can democratize access to high-quality, customized educational support.

{{</citation>}}


### (13/155) AviationGPT: A Large Language Model for the Aviation Domain (Liya Wang et al., 2023)

{{<citation>}}

Liya Wang, Jason Chou, Xin Zhou, Alex Tien, Diane M Baumgartner. (2023)  
**AviationGPT: A Large Language Model for the Aviation Domain**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, LLaMA, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.17686v1)  

---


**ABSTRACT**  
The advent of ChatGPT and GPT-4 has captivated the world with large language models (LLMs), demonstrating exceptional performance in question-answering, summarization, and content generation. The aviation industry is characterized by an abundance of complex, unstructured text data, replete with technical jargon and specialized terminology. Moreover, labeled data for model building are scarce in this domain, resulting in low usage of aviation text data. The emergence of LLMs presents an opportunity to transform this situation, but there is a lack of LLMs specifically designed for the aviation domain. To address this gap, we propose AviationGPT, which is built on open-source LLaMA-2 and Mistral architectures and continuously trained on a wealth of carefully curated aviation datasets. Experimental results reveal that AviationGPT offers users multiple advantages, including the versatility to tackle diverse natural language processing (NLP) problems (e.g., question-answering, summarization, document writing, information extraction, report querying, data cleaning, and interactive data exploration). It also provides accurate and contextually relevant responses within the aviation domain and significantly improves performance (e.g., over a 40% performance gain in tested cases). With AviationGPT, the aviation industry is better equipped to address more complex research problems and enhance the efficiency and safety of National Airspace System (NAS) operations.

{{</citation>}}


### (14/155) TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models (Zheng Chu et al., 2023)

{{<citation>}}

Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Haotian Wang, Ming Liu, Bing Qin. (2023)  
**TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, LLaMA, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.17667v1)  

---


**ABSTRACT**  
Understanding time is a pivotal aspect of human cognition, crucial in the broader framework of grasping the intricacies of the world. Previous studies typically focus on specific aspects of time, lacking a comprehensive temporal reasoning benchmark. To address this issue, we propose TimeBench, a comprehensive hierarchical temporal reasoning benchmark that covers a broad spectrum of temporal reasoning phenomena, which provides a thorough evaluation for investigating the temporal reasoning capabilities of large language models. We conduct extensive experiments on popular LLMs, such as GPT-4, LLaMA2, and Mistral, incorporating chain-of-thought prompting. Our experimental results indicate a significant performance gap between the state-of-the-art LLMs and humans, highlighting that there is still a considerable distance to cover in temporal reasoning. We aspire for TimeBench to serve as a comprehensive benchmark, fostering research in temporal reasoning for LLMs. Our resource is available at https://github.com/zchuz/TimeBench

{{</citation>}}


### (15/155) Introduction to Transformers: an NLP Perspective (Tong Xiao et al., 2023)

{{<citation>}}

Tong Xiao, Jingbo Zhu. (2023)  
**Introduction to Transformers: an NLP Perspective**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.17633v1)  

---


**ABSTRACT**  
Transformers have dominated empirical machine learning models of natural language processing. In this paper, we introduce basic concepts of Transformers and present key techniques that form the recent advances of these models. This includes a description of the standard Transformer architecture, a series of model refinements, and common applications. Given that Transformers and related deep learning techniques might be evolving in ways we have never seen, we cannot dive into all the model details or cover all the technical areas. Instead, we focus on just those concepts that are helpful for gaining a good understanding of Transformers and their variants. We also summarize the key ideas that impact this field, thereby yielding some insights into the strengths and limitations of these models.

{{</citation>}}


### (16/155) Reinforcement Replaces Supervision: Query focused Summarization using Deep Reinforcement Learning (Swaroop Nath et al., 2023)

{{<citation>}}

Swaroop Nath, Harshad Khadilkar, Pushpak Bhattacharyya. (2023)  
**Reinforcement Replaces Supervision: Query focused Summarization using Deep Reinforcement Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU, Embedding, Natural Language Generation, QA, Question Answering, Reinforcement Learning, Semantic Similarity, Summarization, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.17514v1)  

---


**ABSTRACT**  
Query-focused Summarization (QfS) deals with systems that generate summaries from document(s) based on a query. Motivated by the insight that Reinforcement Learning (RL) provides a generalization to Supervised Learning (SL) for Natural Language Generation, and thereby performs better (empirically) than SL, we use an RL-based approach for this task of QfS. Additionally, we also resolve the conflict of employing RL in Transformers with Teacher Forcing. We develop multiple Policy Gradient networks, trained on various reward signals: ROUGE, BLEU, and Semantic Similarity, which lead to a 10-point improvement over the State-of-the-Art approach on the ROUGE-L metric for a benchmark dataset (ELI5). We also show performance of our approach in zero-shot setting for another benchmark dataset (DebatePedia) -- our approach leads to results comparable to baselines, which were specifically trained on DebatePedia. To aid the RL training, we propose a better semantic similarity reward, enabled by a novel Passage Embedding scheme developed using Cluster Hypothesis. Lastly, we contribute a gold-standard test dataset to further research in QfS and Long-form Question Answering (LfQA).

{{</citation>}}


### (17/155) Enhancing Answer Selection in Community Question Answering with Pre-trained and Large Language Models (Xinghang Hu, 2023)

{{<citation>}}

Xinghang Hu. (2023)  
**Enhancing Answer Selection in Community Question Answering with Pre-trained and Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.17502v1)  

---


**ABSTRACT**  
Community Question Answering (CQA) becomes increasingly prevalent in recent years. However, there are a large number of answers, which is difficult for users to select the relevant answers. Therefore, answer selection is a very significant subtask of CQA. In this paper, we first propose the Question-Answer cross attention networks (QAN) with pre-trained models for answer selection and utilize large language model (LLM) to perform answer selection with knowledge augmentation. Specifically, we apply the BERT model as the encoder layer to do pre-training for question subjects, question bodies and answers, respectively, then the cross attention mechanism selects the most relevant answer for different questions. Experiments show that the QAN model achieves state-of-the-art performance on two datasets, SemEval2015 and SemEval2017. Moreover, we use the LLM to generate external knowledge from questions and correct answers to achieve knowledge augmentation for the answer selection task by LLM, while optimizing the prompt of LLM in different aspects. The results show that the introduction of external knowledge can improve the correct answer selection rate of LLM on datasets SemEval2015 and SemEval2017. Meanwhile, LLM can also select the correct answer on more questions by optimized prompt.

{{</citation>}}


### (18/155) Mergen: The First Manchu-Korean Machine Translation Model Trained on Augmented Data (Jean Seo et al., 2023)

{{<citation>}}

Jean Seo, Sungjoo Byun, Minha Kang, Sangah Lee. (2023)  
**Mergen: The First Manchu-Korean Machine Translation Model Trained on Augmented Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.17492v1)  

---


**ABSTRACT**  
The Manchu language, with its roots in the historical Manchurian region of Northeast China, is now facing a critical threat of extinction, as there are very few speakers left. In our efforts to safeguard the Manchu language, we introduce Mergen, the first-ever attempt at a Manchu-Korean Machine Translation (MT) model. To develop this model, we utilize valuable resources such as the Manwen Laodang(a historical book) and a Manchu-Korean dictionary. Due to the scarcity of a Manchu-Korean parallel dataset, we expand our data by employing word replacement guided by GloVe embeddings, trained on both monolingual and parallel texts. Our approach is built around an encoder-decoder neural machine translation model, incorporating a bi-directional Gated Recurrent Unit (GRU) layer. The experiments have yielded promising results, showcasing a significant enhancement in Manchu-Korean translation, with a remarkable 20-30 point increase in the BLEU score.

{{</citation>}}


### (19/155) Taiwan LLM: Bridging the Linguistic Divide with a Culturally Aligned Language Model (Yen-Ting Lin et al., 2023)

{{<citation>}}

Yen-Ting Lin, Yun-Nung Chen. (2023)  
**Taiwan LLM: Bridging the Linguistic Divide with a Culturally Aligned Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17487v1)  

---


**ABSTRACT**  
In the realm of language models, the nuanced linguistic and cultural intricacies of Traditional Chinese, as spoken in Taiwan, have been largely overlooked. This paper introduces Taiwan LLM, a pioneering Large Language Model that specifically caters to the Traditional Chinese language, with a focus on the variant used in Taiwan. Leveraging a comprehensive pretraining corpus and instruction-finetuning datasets, we have developed a model that not only understands the complexities of Traditional Chinese but also embodies the cultural context of Taiwan. Taiwan LLM represents the first of its kind, a model that is not only linguistically accurate but also culturally resonant with its user base. Our evaluations demonstrate that Taiwan LLM achieves superior performance in understanding and generating Traditional Chinese text, outperforming existing models that are predominantly trained on Simplified Chinese or English. The open-source release of Taiwan LLM invites collaboration and further innovation, ensuring that the linguistic diversity of Chinese speakers is embraced and well-served. The model, datasets, and further resources are made publicly available to foster ongoing research and development in this field.

{{</citation>}}


### (20/155) CLOMO: Counterfactual Logical Modification with Large Language Models (Yinya Huang et al., 2023)

{{<citation>}}

Yinya Huang, Ruixin Hong, Hongming Zhang, Wei Shao, Zhicheng Yang, Dong Yu, Changshui Zhang, Xiaodan Liang, Linqi Song. (2023)  
**CLOMO: Counterfactual Logical Modification with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17438v2)  

---


**ABSTRACT**  
In this study, we delve into the realm of counterfactual reasoning capabilities of large language models (LLMs). Our primary objective is to cultivate the counterfactual thought processes within LLMs and rigorously assess these processes for their validity. Specifically, we introduce a novel task, Counterfactual Logical Modification (CLOMO), and a high-quality human-annotated benchmark. In this task, LLMs must adeptly alter a given argumentative text to uphold a predetermined logical relationship. To effectively evaluate a generation model's counterfactual capabilities, we propose an innovative evaluation metric, the LogicAware Counterfactual Score to directly evaluate the natural language output of LLMs instead of modeling the task as a multiple-choice problem. Analysis shows that the proposed automatic metric aligns well with human preference. Our experimental results show that while LLMs demonstrate a notable capacity for logical counterfactual thinking, there remains a discernible gap between their current abilities and human performance.

{{</citation>}}


### (21/155) TARGET: Template-Transferable Backdoor Attack Against Prompt-based NLP Models via GPT4 (Zihao Tan et al., 2023)

{{<citation>}}

Zihao Tan, Qingliang Chen, Yongjian Huang, Chen Liang. (2023)  
**TARGET: Template-Transferable Backdoor Attack Against Prompt-based NLP Models via GPT4**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2311.17429v1)  

---


**ABSTRACT**  
Prompt-based learning has been widely applied in many low-resource NLP tasks such as few-shot scenarios. However, this paradigm has been shown to be vulnerable to backdoor attacks. Most of the existing attack methods focus on inserting manually predefined templates as triggers in the pre-training phase to train the victim model and utilize the same triggers in the downstream task to perform inference, which tends to ignore the transferability and stealthiness of the templates. In this work, we propose a novel approach of TARGET (Template-trAnsfeRable backdoor attack aGainst prompt-basEd NLP models via GPT4), which is a data-independent attack method. Specifically, we first utilize GPT4 to reformulate manual templates to generate tone-strong and normal templates, and the former are injected into the model as a backdoor trigger in the pre-training phase. Then, we not only directly employ the above templates in the downstream task, but also use GPT4 to generate templates with similar tone to the above templates to carry out transferable attacks. Finally we have conducted extensive experiments on five NLP datasets and three BERT series models, with experimental results justifying that our TARGET method has better attack performance and stealthiness compared to the two-external baseline methods on direct attacks, and in addition achieves satisfactory attack capability in the unseen tone-similar templates.

{{</citation>}}


### (22/155) Improving the Robustness of Transformer-based Large Language Models with Dynamic Attention (Lujia Shen et al., 2023)

{{<citation>}}

Lujia Shen, Yuwen Pu, Shouling Ji, Changjiang Li, Xuhong Zhang, Chunpeng Ge, Ting Wang. (2023)  
**Improving the Robustness of Transformer-based Large Language Models with Dynamic Attention**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: Attention, BERT, GPT, Language Model, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2311.17400v2)  

---


**ABSTRACT**  
Transformer-based models, such as BERT and GPT, have been widely adopted in natural language processing (NLP) due to their exceptional performance. However, recent studies show their vulnerability to textual adversarial attacks where the model's output can be misled by intentionally manipulating the text inputs. Despite various methods that have been proposed to enhance the model's robustness and mitigate this vulnerability, many require heavy consumption resources (e.g., adversarial training) or only provide limited protection (e.g., defensive dropout). In this paper, we propose a novel method called dynamic attention, tailored for the transformer architecture, to enhance the inherent robustness of the model itself against various adversarial attacks. Our method requires no downstream task knowledge and does not incur additional costs. The proposed dynamic attention consists of two modules: (I) attention rectification, which masks or weakens the attention value of the chosen tokens, and (ii) dynamic modeling, which dynamically builds the set of candidate tokens. Extensive experiments demonstrate that dynamic attention significantly mitigates the impact of adversarial attacks, improving up to 33\% better performance than previous methods against widely-used adversarial attacks. The model-level design of dynamic attention enables it to be easily combined with other defense methods (e.g., adversarial training) to further enhance the model's robustness. Furthermore, we demonstrate that dynamic attention preserves the state-of-the-art robustness space of the original model compared to other dynamic modeling methods.

{{</citation>}}


### (23/155) Unveiling the Implicit Toxicity in Large Language Models (Jiaxin Wen et al., 2023)

{{<citation>}}

Jiaxin Wen, Pei Ke, Hao Sun, Zhexin Zhang, Chengfei Li, Jinfeng Bai, Minlie Huang. (2023)  
**Unveiling the Implicit Toxicity in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17391v1)  

---


**ABSTRACT**  
The open-endedness of large language models (LLMs) combined with their impressive capabilities may lead to new safety issues when being exploited for malicious use. While recent studies primarily focus on probing toxic outputs that can be easily detected with existing toxicity classifiers, we show that LLMs can generate diverse implicit toxic outputs that are exceptionally difficult to detect via simply zero-shot prompting. Moreover, we propose a reinforcement learning (RL) based attacking method to further induce the implicit toxicity in LLMs. Specifically, we optimize the language model with a reward that prefers implicit toxic outputs to explicit toxic and non-toxic ones. Experiments on five widely-adopted toxicity classifiers demonstrate that the attack success rate can be significantly improved through RL fine-tuning. For instance, the RL-finetuned LLaMA-13B model achieves an attack success rate of 90.04% on BAD and 62.85% on Davinci003. Our findings suggest that LLMs pose a significant threat in generating undetectable implicit toxic outputs. We further show that fine-tuning toxicity classifiers on the annotated examples from our attacking method can effectively enhance their ability to detect LLM-generated implicit toxic language. The code is publicly available at https://github.com/thu-coai/Implicit-Toxicity.

{{</citation>}}


### (24/155) CESAR: Automatic Induction of Compositional Instructions for Multi-turn Dialogs (Taha Aksu et al., 2023)

{{<citation>}}

Taha Aksu, Devamanyu Hazarika, Shikib Mehri, Seokhwan Kim, Dilek Hakkani-Tür, Yang Liu, Mahdi Namazifar. (2023)  
**CESAR: Automatic Induction of Compositional Instructions for Multi-turn Dialogs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, GPT  
[Paper Link](http://arxiv.org/abs/2311.17376v1)  

---


**ABSTRACT**  
Instruction-based multitasking has played a critical role in the success of large language models (LLMs) in multi-turn dialog applications. While publicly available LLMs have shown promising performance, when exposed to complex instructions with multiple constraints, they lag against state-of-the-art models like ChatGPT. In this work, we hypothesize that the availability of large-scale complex demonstrations is crucial in bridging this gap. Focusing on dialog applications, we propose a novel framework, CESAR, that unifies a large number of dialog tasks in the same format and allows programmatic induction of complex instructions without any manual effort.   We apply CESAR on InstructDial, a benchmark for instruction-based dialog tasks. We further enhance InstructDial with new datasets and tasks and utilize CESAR to induce complex tasks with compositional instructions. This results in a new benchmark called InstructDial++, which includes 63 datasets with 86 basic tasks and 68 composite tasks. Through rigorous experiments, we demonstrate the scalability of CESAR in providing rich instructions. Models trained on InstructDial++ can follow compositional prompts, such as prompts that ask for multiple stylistic constraints.

{{</citation>}}


### (25/155) Are we going MAD? Benchmarking Multi-Agent Debate between Language Models for Medical Q&A (Andries Smit et al., 2023)

{{<citation>}}

Andries Smit, Paul Duckworth, Nathan Grinsztajn, Kale-ab Tessera, Thomas D. Barrett, Arnu Pretorius. (2023)  
**Are we going MAD? Benchmarking Multi-Agent Debate between Language Models for Medical Q&A**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17371v1)  

---


**ABSTRACT**  
Recent advancements in large language models (LLMs) underscore their potential for responding to medical inquiries. However, ensuring that generative agents provide accurate and reliable answers remains an ongoing challenge. In this context, multi-agent debate (MAD) has emerged as a prominent strategy for enhancing the truthfulness of LLMs. In this work, we provide a comprehensive benchmark of MAD strategies for medical Q&A, along with open-source implementations. This explores the effective utilization of various strategies including the trade-offs between cost, time, and accuracy. We build upon these insights to provide a novel debate-prompting strategy based on agent agreement that outperforms previously published strategies on medical Q&A tasks.

{{</citation>}}


### (26/155) Are Large Language Models Good Fact Checkers: A Preliminary Study (Han Cao et al., 2023)

{{<citation>}}

Han Cao, Lingwei Wei, Mengyang Chen, Wei Zhou, Songlin Hu. (2023)  
**Are Large Language Models Good Fact Checkers: A Preliminary Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17355v1)  

---


**ABSTRACT**  
Recently, Large Language Models (LLMs) have drawn significant attention due to their outstanding reasoning capabilities and extensive knowledge repository, positioning them as superior in handling various natural language processing tasks compared to other language models. In this paper, we present a preliminary investigation into the potential of LLMs in fact-checking. This study aims to comprehensively evaluate various LLMs in tackling specific fact-checking subtasks, systematically evaluating their capabilities, and conducting a comparative analysis of their performance against pre-trained and state-of-the-art low-parameter models. Experiments demonstrate that LLMs achieve competitive performance compared to other small models in most scenarios. However, they encounter challenges in effectively handling Chinese fact verification and the entirety of the fact-checking pipeline due to language inconsistencies and hallucinations. These findings underscore the need for further exploration and research to enhance the proficiency of LLMs as reliable fact-checkers, unveiling the potential capability of LLMs and the possible challenges in fact-checking tasks.

{{</citation>}}


### (27/155) Biomedical knowledge graph-enhanced prompt generation for large language models (Karthik Soman et al., 2023)

{{<citation>}}

Karthik Soman, Peter W Rose, John H Morris, Rabia E Akbas, Brett Smith, Braian Peetoom, Catalina Villouta-Reyes, Gabriel Cerono, Yongmei Shi, Angela Rizk-Jackson, Sharat Israni, Charlotte A Nelson, Sui Huang, Sergio E Baranzini. (2023)  
**Biomedical knowledge graph-enhanced prompt generation for large language models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-3.5, GPT-4, Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17330v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have been driving progress in AI at an unprecedented rate, yet still face challenges in knowledge-intensive domains like biomedicine. Solutions such as pre-training and domain-specific fine-tuning add substantial computational overhead, and the latter require domain-expertise. External knowledge infusion is task-specific and requires model training. Here, we introduce a task-agnostic Knowledge Graph-based Retrieval Augmented Generation (KG-RAG) framework by leveraging the massive biomedical KG SPOKE with LLMs such as Llama-2-13b, GPT-3.5-Turbo and GPT-4, to generate meaningful biomedical text rooted in established knowledge. KG-RAG consistently enhanced the performance of LLMs across various prompt types, including one-hop and two-hop prompts, drug repurposing queries, biomedical true/false questions, and multiple-choice questions (MCQ). Notably, KG-RAG provides a remarkable 71% boost in the performance of the Llama-2 model on the challenging MCQ dataset, demonstrating the framework's capacity to empower open-source models with fewer parameters for domain-specific questions. Furthermore, KG-RAG enhanced the performance of proprietary GPT models, such as GPT-3.5 which exhibited improvement over GPT-4 in context utilization on MCQ data. Our approach was also able to address drug repurposing questions, returning meaningful repurposing suggestions. In summary, the proposed framework combines explicit and implicit knowledge of KG and LLM, respectively, in an optimized fashion, thus enhancing the adaptability of general-purpose LLMs to tackle domain-specific questions in a unified framework.

{{</citation>}}


### (28/155) Universal Self-Consistency for Large Language Model Generation (Xinyun Chen et al., 2023)

{{<citation>}}

Xinyun Chen, Renat Aksitov, Uri Alon, Jie Ren, Kefan Xiao, Pengcheng Yin, Sushant Prakash, Charles Sutton, Xuezhi Wang, Denny Zhou. (2023)  
**Universal Self-Consistency for Large Language Model Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17311v1)  

---


**ABSTRACT**  
Self-consistency with chain-of-thought prompting (CoT) has demonstrated remarkable performance gains on various challenging tasks, by utilizing multiple reasoning paths sampled from large language models (LLMs). However, self-consistency relies on the answer extraction process to aggregate multiple solutions, which is not applicable to free-form answers. In this work, we propose Universal Self-Consistency (USC), which leverages LLMs themselves to select the most consistent answer among multiple candidates. We evaluate USC on a variety of benchmarks, including mathematical reasoning, code generation, long-context summarization, and open-ended question answering. On open-ended generation tasks where the original self-consistency method is not applicable, USC effectively utilizes multiple samples and improves the performance. For mathematical reasoning, USC matches the standard self-consistency performance without requiring the answer formats to be similar. Finally, without access to execution results, USC also matches the execution-based voting performance on code generation.

{{</citation>}}


### (29/155) RoKEPG: RoBERTa and Knowledge Enhancement for Prescription Generation of Traditional Chinese Medicine (Hua Pu et al., 2023)

{{<citation>}}

Hua Pu, Jiacong Mi, Shan Lu, Jieyue He. (2023)  
**RoKEPG: RoBERTa and Knowledge Enhancement for Prescription Generation of Traditional Chinese Medicine**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2311.17307v1)  

---


**ABSTRACT**  
Traditional Chinese medicine (TCM) prescription is the most critical form of TCM treatment, and uncovering the complex nonlinear relationship between symptoms and TCM is of great significance for clinical practice and assisting physicians in diagnosis and treatment. Although there have been some studies on TCM prescription generation, these studies consider a single factor and directly model the symptom-prescription generation problem mainly based on symptom descriptions, lacking guidance from TCM knowledge. To this end, we propose a RoBERTa and Knowledge Enhancement model for Prescription Generation of Traditional Chinese Medicine (RoKEPG). RoKEPG is firstly pre-trained by our constructed TCM corpus, followed by fine-tuning the pre-trained model, and the model is guided to generate TCM prescriptions by introducing four classes of knowledge of TCM through the attention mask matrix. Experimental results on the publicly available TCM prescription dataset show that RoKEPG improves the F1 metric by about 2% over the baseline model with the best results.

{{</citation>}}


### (30/155) Language Models: A Guide for the Perplexed (Sofia Serrano et al., 2023)

{{<citation>}}

Sofia Serrano, Zander Brumbaugh, Noah A. Smith. (2023)  
**Language Models: A Guide for the Perplexed**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17301v1)  

---


**ABSTRACT**  
Given the growing importance of AI literacy, we decided to write this tutorial to help narrow the gap between the discourse among those who study language models -- the core technology underlying ChatGPT and similar products -- and those who are intrigued and want to learn more about them. In short, we believe the perspective of researchers and educators can add some clarity to the public's understanding of the technologies beyond what's currently available, which tends to be either extremely technical or promotional material generated about products by their purveyors.   Our approach teases apart the concept of a language model from products built on them, from the behaviors attributed to or desired from those products, and from claims about similarity to human cognition. As a starting point, we (1) offer a scientific viewpoint that focuses on questions amenable to study through experimentation; (2) situate language models as they are today in the context of the research that led to their development; and (3) describe the boundaries of what is known about the models at this writing.

{{</citation>}}


### (31/155) Elo Uncovered: Robustness and Best Practices in Language Model Evaluation (Meriem Boubdir et al., 2023)

{{<citation>}}

Meriem Boubdir, Edward Kim, Beyza Ermis, Sara Hooker, Marzieh Fadaee. (2023)  
**Elo Uncovered: Robustness and Best Practices in Language Model Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.17295v1)  

---


**ABSTRACT**  
In Natural Language Processing (NLP), the Elo rating system, originally designed for ranking players in dynamic games such as chess, is increasingly being used to evaluate Large Language Models (LLMs) through "A vs B" paired comparisons. However, while popular, the system's suitability for assessing entities with constant skill levels, such as LLMs, remains relatively unexplored. We study two fundamental axioms that evaluation methods should adhere to: reliability and transitivity. We conduct extensive evaluation of Elo behaviour, illustrating that individual Elo computations exhibit volatility and delving into the impact of varying the Elo rating system's hyperparameters. We show that these axioms are not always satisfied raising questions about the reliability of current comparative evaluations of LLMs. If the current use of Elo scores is intended to substitute the costly head-to-head comparison of LLMs, it is crucial to ensure the ranking is as robust as possible. Guided by the axioms, our findings offer concrete guidelines for enhancing the reliability of LLM evaluation methods, suggesting a need for reassessment of existing comparative approaches.

{{</citation>}}


## cs.CV (67)



### (32/155) STF: Spatial Temporal Fusion for Trajectory Prediction (Pengqian Han et al., 2023)

{{<citation>}}

Pengqian Han, Partha Roop, Jiamou Liu, Tianzhe Bao, Yifei Wang. (2023)  
**STF: Spatial Temporal Fusion for Trajectory Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.18149v1)  

---


**ABSTRACT**  
Trajectory prediction is a challenging task that aims to predict the future trajectory of vehicles or pedestrians over a short time horizon based on their historical positions. The main reason is that the trajectory is a kind of complex data, including spatial and temporal information, which is crucial for accurate prediction. Intuitively, the more information the model can capture, the more precise the future trajectory can be predicted. However, previous works based on deep learning methods processed spatial and temporal information separately, leading to inadequate spatial information capture, which means they failed to capture the complete spatial information. Therefore, it is of significance to capture information more fully and effectively on vehicle interactions. In this study, we introduced an integrated 3D graph that incorporates both spatial and temporal edges. Based on this, we proposed the integrated 3D graph, which considers the cross-time interaction information. In specific, we design a Spatial-Temporal Fusion (STF) model including Multi-layer perceptions (MLP) and Graph Attention (GAT) to capture the spatial and temporal information historical trajectories simultaneously on the 3D graph. Our experiment on the ApolloScape Trajectory Datasets shows that the proposed STF outperforms several baseline methods, especially on the long-time-horizon trajectory prediction.

{{</citation>}}


### (33/155) CRAFT: Contextual Re-Activation of Filters for face recogntion Training (Aman Bhatta, 2023)

{{<citation>}}

Aman Bhatta. (2023)  
**CRAFT: Contextual Re-Activation of Filters for face recogntion Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.00072v1)  

---


**ABSTRACT**  
The first layer of a deep CNN backbone applies filters to an image to extract the basic features available to later layers. During training, some filters may go inactive, mean ing all weights in the filter approach zero. An inactive fil ter in the final model represents a missed opportunity to extract a useful feature. This phenomenon is especially prevalent in specialized CNNs such as for face recogni tion (as opposed to, e.g., ImageNet). For example, in one the most widely face recognition model (ArcFace), about half of the convolution filters in the first layer are inactive. We propose a novel approach designed and tested specif ically for face recognition networks, known as "CRAFT: Contextual Re-Activation of Filters for Face Recognition Training". CRAFT identifies inactive filters during training and reinitializes them based on the context of strong filters at that stage in training. We show that CRAFT reduces fraction of inactive filters from 44% to 32% on average and discovers filter patterns not found by standard training. Compared to standard training without reactivation, CRAFT demonstrates enhanced model accuracy on standard face-recognition benchmark datasets including AgeDB-30, CPLFW, LFW, CALFW, and CFP-FP, as well as on more challenging datasets like IJBB and IJBC.

{{</citation>}}


### (34/155) Back to 3D: Few-Shot 3D Keypoint Detection with Back-Projected 2D Features (Thomas Wimmer et al., 2023)

{{<citation>}}

Thomas Wimmer, Peter Wonka, Maks Ovsjanikov. (2023)  
**Back to 3D: Few-Shot 3D Keypoint Detection with Back-Projected 2D Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Few-Shot, NLP  
[Paper Link](http://arxiv.org/abs/2311.18113v1)  

---


**ABSTRACT**  
With the immense growth of dataset sizes and computing resources in recent years, so-called foundation models have become popular in NLP and vision tasks. In this work, we propose to explore foundation models for the task of keypoint detection on 3D shapes. A unique characteristic of keypoint detection is that it requires semantic and geometric awareness while demanding high localization accuracy. To address this problem, we propose, first, to back-project features from large pre-trained 2D vision models onto 3D shapes and employ them for this task. We show that we obtain robust 3D features that contain rich semantic information and analyze multiple candidate features stemming from different 2D foundation models. Second, we employ a keypoint candidate optimization module which aims to match the average observed distribution of keypoints on the shape and is guided by the back-projected features. The resulting approach achieves a new state of the art for few-shot keypoint detection on the KeyPointNet dataset, almost doubling the performance of the previous best methods.

{{</citation>}}


### (35/155) Meta Co-Training: Two Views are Better than One (Jay C. Rothenberger et al., 2023)

{{<citation>}}

Jay C. Rothenberger, Dimitrios I. Diochnos. (2023)  
**Meta Co-Training: Two Views are Better than One**  

---
Primary Category: cs.CV  
Categories: I-2-6; I-4-10, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.18083v1)  

---


**ABSTRACT**  
In many practical computer vision scenarios unlabeled data is plentiful, but labels are scarce and difficult to obtain. As a result, semi-supervised learning which leverages unlabeled data to boost the performance of supervised classifiers have received significant attention in recent literature. One major class of semi-supervised algorithms is co-training. In co-training two different models leverage different independent and sufficient "views" of the data to jointly make better predictions. During co-training each model creates pseudo labels on unlabeled points which are used to improve the other model. We show that in the common case when independent views are not available we can construct such views inexpensively using pre-trained models. Co-training on the constructed views yields a performance improvement over any of the individual views we construct and performance comparable with recent approaches in semi-supervised learning, but has some undesirable properties. To alleviate the issues present with co-training we present Meta Co-Training which is an extension of the successful Meta Pseudo Labels approach to multiple views. Our method achieves new state-of-the-art performance on ImageNet-10% with very few training resources, as well as outperforming prior semi-supervised work on several other fine-grained image classification datasets.

{{</citation>}}


### (36/155) Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing (Piper Wolters et al., 2023)

{{<citation>}}

Piper Wolters, Favyen Bastani, Aniruddha Kembhavi. (2023)  
**Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18082v1)  

---


**ABSTRACT**  
Super-Resolution for remote sensing has the potential for huge impact on planet monitoring by producing accurate and realistic high resolution imagery on a frequent basis and a global scale. Despite a lot of attention, several inconsistencies and challenges have prevented it from being deployed in practice. These include the lack of effective metrics, fragmented and relatively small-scale datasets for training, insufficient comparisons across a suite of methods, and unclear evidence for the use of super-resolution outputs for machine consumption. This work presents a new metric for super-resolution, CLIPScore, that corresponds far better with human judgments than previous metrics on an extensive study. We use CLIPScore to evaluate four standard methods on a new large-scale dataset, S2-NAIP, and three existing benchmark datasets, and find that generative adversarial networks easily outperform more traditional L2 loss-based models and are more semantically accurate than modern diffusion models. We also find that using CLIPScore as an auxiliary loss can speed up the training of GANs by 18x and lead to improved outputs, resulting in an effective model in diverse geographies across the world which we will release publicly. The dataset, pre-trained model weights, and code are available at https://github.com/allenai/satlas-super-resolution/.

{{</citation>}}


### (37/155) GELDA: A generative language annotation framework to reveal visual biases in datasets (Krish Kabra et al., 2023)

{{<citation>}}

Krish Kabra, Kathleen M. Lewis, Guha Balakrishnan. (2023)  
**GELDA: A generative language annotation framework to reveal visual biases in datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.18064v1)  

---


**ABSTRACT**  
Bias analysis is a crucial step in the process of creating fair datasets for training and evaluating computer vision models. The bottleneck in dataset analysis is annotation, which typically requires: (1) specifying a list of attributes relevant to the dataset domain, and (2) classifying each image-attribute pair. While the second step has made rapid progress in automation, the first has remained human-centered, requiring an experimenter to compile lists of in-domain attributes. However, an experimenter may have limited foresight leading to annotation "blind spots," which in turn can lead to flawed downstream dataset analyses. To combat this, we propose GELDA, a nearly automatic framework that leverages large generative language models (LLMs) to propose and label various attributes for a domain. GELDA takes a user-defined domain caption (e.g., "a photo of a bird," "a photo of a living room") and uses an LLM to hierarchically generate attributes. In addition, GELDA uses the LLM to decide which of a set of vision-language models (VLMs) to use to classify each attribute in images. Results on real datasets show that GELDA can generate accurate and diverse visual attribute suggestions, and uncover biases such as confounding between class labels and background features. Results on synthetic datasets demonstrate that GELDA can be used to evaluate the biases of text-to-image diffusion models and generative adversarial networks. Overall, we show that while GELDA is not accurate enough to replace human annotators, it can serve as a complementary tool to help humans analyze datasets in a cheap, low-effort, and flexible manner.

{{</citation>}}


### (38/155) MoMask: Generative Masked Modeling of 3D Human Motions (Chuan Guo et al., 2023)

{{<citation>}}

Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, Li Cheng. (2023)  
**MoMask: Generative Masked Modeling of 3D Human Motions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2312.00063v1)  

---


**ABSTRACT**  
We introduce MoMask, a novel masked modeling framework for text-driven 3D human motion generation. In MoMask, a hierarchical quantization scheme is employed to represent human motion as multi-layer discrete motion tokens with high-fidelity details. Starting at the base layer, with a sequence of motion tokens obtained by vector quantization, the residual tokens of increasing orders are derived and stored at the subsequent layers of the hierarchy. This is consequently followed by two distinct bidirectional transformers. For the base-layer motion tokens, a Masked Transformer is designated to predict randomly masked motion tokens conditioned on text input at training stage. During generation (i.e. inference) stage, starting from an empty sequence, our Masked Transformer iteratively fills up the missing tokens; Subsequently, a Residual Transformer learns to progressively predict the next-layer tokens based on the results from current layer. Extensive experiments demonstrate that MoMask outperforms the state-of-art methods on the text-to-motion generation task, with an FID of 0.045 (vs e.g. 0.141 of T2M-GPT) on the HumanML3D dataset, and 0.228 (vs 0.514) on KIT-ML, respectively. MoMask can also be seamlessly applied in related tasks without further model fine-tuning, such as text-guided temporal inpainting.

{{</citation>}}


### (39/155) OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation (Qidong Huang et al., 2023)

{{<citation>}}

Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, Nenghai Yu. (2023)  
**OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17911v1)  

---


**ABSTRACT**  
Hallucination, posed as a pervasive challenge of multi-modal large language models (MLLMs), has significantly impeded their real-world usage that demands precise judgment. Existing methods mitigate this issue with either training with specific designed data or inferencing with external knowledge from other sources, incurring inevitable additional costs. In this paper, we present OPERA, a novel MLLM decoding method grounded in an Over-trust Penalty and a Retrospection-Allocation strategy, serving as a nearly free lunch to alleviate the hallucination issue without additional data, knowledge, or training. Our approach begins with an interesting observation that, most hallucinations are closely tied to the knowledge aggregation patterns manifested in the self-attention matrix, i.e., MLLMs tend to generate new tokens by focusing on a few summary tokens, but not all the previous tokens. Such partial over-trust inclination results in the neglecting of image tokens and describes the image content with hallucination. Statistically, we observe an 80%$\sim$95% co-currency rate between hallucination contents and such knowledge aggregation patterns. Based on the observation, OPERA introduces a penalty term on the model logits during the beam-search decoding to mitigate the over-trust issue, along with a rollback strategy that retrospects the presence of summary tokens in the previously generated tokens, and re-allocate the token selection if necessary. With extensive experiments, OPERA shows significant hallucination-mitigating performance on different MLLMs and metrics, proving its effectiveness and generality. Our code is available at: https://github.com/shikiw/OPERA.

{{</citation>}}


### (40/155) Language-conditioned Detection Transformer (Jang Hyun Cho et al., 2023)

{{<citation>}}

Jang Hyun Cho, Philipp Krähenbühl. (2023)  
**Language-conditioned Detection Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.17902v1)  

---


**ABSTRACT**  
We present a new open-vocabulary detection framework. Our framework uses both image-level labels and detailed detection annotations when available. Our framework proceeds in three steps. We first train a language-conditioned object detector on fully-supervised detection data. This detector gets to see the presence or absence of ground truth classes during training, and conditions prediction on the set of present classes. We use this detector to pseudo-label images with image-level labels. Our detector provides much more accurate pseudo-labels than prior approaches with its conditioning mechanism. Finally, we train an unconditioned open-vocabulary detector on the pseudo-annotated images. The resulting detector, named DECOLA, shows strong zero-shot performance in open-vocabulary LVIS benchmark as well as direct zero-shot transfer benchmarks on LVIS, COCO, Object365, and OpenImages. DECOLA outperforms the prior arts by 17.1 AP-rare and 9.4 mAP on zero-shot LVIS benchmark. DECOLA achieves state-of-the-art results in various model sizes, architectures, and datasets by only training on open-sourced data and academic-scale computing. Code is available at https://github.com/janghyuncho/DECOLA.

{{</citation>}}


### (41/155) SODA: Bottleneck Diffusion Models for Representation Learning (Drew A. Hudson et al., 2023)

{{<citation>}}

Drew A. Hudson, Daniel Zoran, Mateusz Malinowski, Andrew K. Lampinen, Andrew Jaegle, James L. McClelland, Loic Matthey, Felix Hill, Alexander Lerchner. (2023)  
**SODA: Bottleneck Diffusion Models for Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.17901v1)  

---


**ABSTRACT**  
We introduce SODA, a self-supervised diffusion model, designed for representation learning. The model incorporates an image encoder, which distills a source view into a compact representation, that, in turn, guides the generation of related novel views. We show that by imposing a tight bottleneck between the encoder and a denoising decoder, and leveraging novel view synthesis as a self-supervised objective, we can turn diffusion models into strong representation learners, capable of capturing visual semantics in an unsupervised manner. To the best of our knowledge, SODA is the first diffusion model to succeed at ImageNet linear-probe classification, and, at the same time, it accomplishes reconstruction, editing and synthesis tasks across a wide range of datasets. Further investigation reveals the disentangled nature of its emergent latent space, that serves as an effective interface to control and manipulate the model's produced images. All in all, we aim to shed light on the exciting and promising potential of diffusion models, not only for image generation, but also for learning rich and robust representations.

{{</citation>}}


### (42/155) Knowledge Pursuit Prompting for Zero-Shot Multimodal Synthesis (Jinqi Luo et al., 2023)

{{<citation>}}

Jinqi Luo, Kwan Ho Ryan Chan, Dimitris Dimos, René Vidal. (2023)  
**Knowledge Pursuit Prompting for Zero-Shot Multimodal Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.17898v2)  

---


**ABSTRACT**  
Hallucinations and unfaithful synthesis due to inaccurate prompts with insufficient semantic details are widely observed in multimodal generative models. A prevalent strategy to align multiple modalities is to fine-tune the generator with a large number of annotated text-image pairs. However, such a procedure is labor-consuming and resource-draining. The key question we ask is: can we enhance the quality and faithfulness of text-driven generative models beyond extensive text-image pair annotations? To address this question, we propose Knowledge Pursuit Prompting (KPP), a zero-shot framework that iteratively incorporates external knowledge to help generators produce reliable visual content. Instead of training generators to handle generic prompts, KPP employs a recursive knowledge query process to gather informative external facts from the knowledge base, instructs a language model to compress the acquired knowledge for prompt refinement, and utilizes text-driven generators for visual synthesis. The entire process is zero-shot, without accessing the architectures and parameters of generative models. We evaluate the framework across multiple text-driven generative tasks (image, 3D rendering, and video) on datasets of different domains. We further demonstrate the extensibility and adaptability of KPP through varying foundation model bases and instructions. Our results show that KPP is capable of generating faithful and semantically rich content across diverse visual domains, offering a promising solution to improve multimodal generative models.

{{</citation>}}


### (43/155) Improving Faithfulness for Vision Transformers (Lijie Hu et al., 2023)

{{<citation>}}

Lijie Hu, Yixin Liu, Ninghao Liu, Mengdi Huai, Lichao Sun, Di Wang. (2023)  
**Improving Faithfulness for Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.17983v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have achieved state-of-the-art performance for various vision tasks. One reason behind the success lies in their ability to provide plausible innate explanations for the behavior of neural architectures. However, ViTs suffer from issues with explanation faithfulness, as their focal points are fragile to adversarial attacks and can be easily changed with even slight perturbations on the input image. In this paper, we propose a rigorous approach to mitigate these issues by introducing Faithful ViTs (FViTs). Briefly speaking, an FViT should have the following two properties: (1) The top-$k$ indices of its self-attention vector should remain mostly unchanged under input perturbation, indicating stable explanations; (2) The prediction distribution should be robust to perturbations. To achieve this, we propose a new method called Denoised Diffusion Smoothing (DDS), which adopts randomized smoothing and diffusion-based denoising. We theoretically prove that processing ViTs directly with DDS can turn them into FViTs. We also show that Gaussian noise is nearly optimal for both $\ell_2$ and $\ell_\infty$-norm cases. Finally, we demonstrate the effectiveness of our approach through comprehensive experiments and evaluations. Specifically, we compare our FViTs with other baselines through visual interpretation and robustness accuracy under adversarial attacks. Results show that FViTs are more robust against adversarial attacks while maintaining the explainability of attention, indicating higher faithfulness.

{{</citation>}}


### (44/155) Betrayed by Attention: A Simple yet Effective Approach for Self-supervised Video Object Segmentation (Shuangrui Ding et al., 2023)

{{<citation>}}

Shuangrui Ding, Rui Qian, Haohang Xu, Dahua Lin, Hongkai Xiong. (2023)  
**Betrayed by Attention: A Simple yet Effective Approach for Self-supervised Video Object Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.17893v1)  

---


**ABSTRACT**  
In this paper, we propose a simple yet effective approach for self-supervised video object segmentation (VOS). Our key insight is that the inherent structural dependencies present in DINO-pretrained Transformers can be leveraged to establish robust spatio-temporal correspondences in videos. Furthermore, simple clustering on this correspondence cue is sufficient to yield competitive segmentation results. Previous self-supervised VOS techniques majorly resort to auxiliary modalities or utilize iterative slot attention to assist in object discovery, which restricts their general applicability and imposes higher computational requirements. To deal with these challenges, we develop a simplified architecture that capitalizes on the emerging objectness from DINO-pretrained Transformers, bypassing the need for additional modalities or slot attention. Specifically, we first introduce a single spatio-temporal Transformer block to process the frame-wise DINO features and establish spatio-temporal dependencies in the form of self-attention. Subsequently, utilizing these attention maps, we implement hierarchical clustering to generate object segmentation masks. To train the spatio-temporal block in a fully self-supervised manner, we employ semantic and dynamic motion consistency coupled with entropy normalization. Our method demonstrates state-of-the-art performance across multiple unsupervised VOS benchmarks and particularly excels in complex real-world multi-object video segmentation tasks such as DAVIS-17-Unsupervised and YouTube-VIS-19. The code and model checkpoints will be released at https://github.com/shvdiwnkozbw/SSL-UVOS.

{{</citation>}}


### (45/155) Pose Anything: A Graph-Based Approach for Category-Agnostic Pose Estimation (Or Hirschorn et al., 2023)

{{<citation>}}

Or Hirschorn, Shai Avidan. (2023)  
**Pose Anything: A Graph-Based Approach for Category-Agnostic Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.17891v1)  

---


**ABSTRACT**  
Traditional 2D pose estimation models are limited by their category-specific design, making them suitable only for predefined object categories. This restriction becomes particularly challenging when dealing with novel objects due to the lack of relevant training data.   To address this limitation, category-agnostic pose estimation (CAPE) was introduced. CAPE aims to enable keypoint localization for arbitrary object categories using a single model, requiring minimal support images with annotated keypoints. This approach not only enables object pose generation based on arbitrary keypoint definitions but also significantly reduces the associated costs, paving the way for versatile and adaptable pose estimation applications.   We present a novel approach to CAPE that leverages the inherent geometrical relations between keypoints through a newly designed Graph Transformer Decoder. By capturing and incorporating this crucial structural information, our method enhances the accuracy of keypoint localization, marking a significant departure from conventional CAPE techniques that treat keypoints as isolated entities.   We validate our approach on the MP-100 benchmark, a comprehensive dataset comprising over 20,000 images spanning more than 100 categories. Our method outperforms the prior state-of-the-art by substantial margins, achieving remarkable improvements of 2.16% and 1.82% under 1-shot and 5-shot settings, respectively. Furthermore, our method's end-to-end training demonstrates both scalability and efficiency compared to previous CAPE approaches.

{{</citation>}}


### (46/155) Enhancing Post-Hoc Explanation Benchmark Reliability for Image Classification (Tristan Gomez et al., 2023)

{{<citation>}}

Tristan Gomez, Harold Mouchère. (2023)  
**Enhancing Post-Hoc Explanation Benchmark Reliability for Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2311.17876v1)  

---


**ABSTRACT**  
Deep neural networks, while powerful for image classification, often operate as "black boxes," complicating the understanding of their decision-making processes. Various explanation methods, particularly those generating saliency maps, aim to address this challenge. However, the inconsistency issues of faithfulness metrics hinder reliable benchmarking of explanation methods. This paper employs an approach inspired by psychometrics, utilizing Krippendorf's alpha to quantify the benchmark reliability of post-hoc methods in image classification. The study proposes model training modifications, including feeding perturbed samples and employing focal loss, to enhance robustness and calibration. Empirical evaluations demonstrate significant improvements in benchmark reliability across metrics, datasets, and post-hoc methods. This pioneering work establishes a foundation for more reliable evaluation practices in the realm of post-hoc explanation methods, emphasizing the importance of model robustness in the assessment process.

{{</citation>}}


### (47/155) Evaluating VLMs for Score-Based, Multi-Probe Annotation of 3D Objects (Rishabh Kabra et al., 2023)

{{<citation>}}

Rishabh Kabra, Loic Matthey, Alexander Lerchner, Niloy J. Mitra. (2023)  
**Evaluating VLMs for Score-Based, Multi-Probe Annotation of 3D Objects**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.17851v1)  

---


**ABSTRACT**  
Unlabeled 3D objects present an opportunity to leverage pretrained vision language models (VLMs) on a range of annotation tasks -- from describing object semantics to physical properties. An accurate response must take into account the full appearance of the object in 3D, various ways of phrasing the question/prompt, and changes in other factors that affect the response. We present a method to marginalize over any factors varied across VLM queries, utilizing the VLM's scores for sampled responses. We first show that this probabilistic aggregation can outperform a language model (e.g., GPT4) for summarization, for instance avoiding hallucinations when there are contrasting details between responses. Secondly, we show that aggregated annotations are useful for prompt-chaining; they help improve downstream VLM predictions (e.g., of object material when the object's type is specified as an auxiliary input in the prompt). Such auxiliary inputs allow ablating and measuring the contribution of visual reasoning over language-only reasoning. Using these evaluations, we show how VLMs can approach, without additional training or in-context learning, the quality of human-verified type and material annotations on the large-scale Objaverse dataset.

{{</citation>}}


### (48/155) SPiC-E : Structural Priors in 3D Diffusion Models using Cross-Entity Attention (Etai Sella et al., 2023)

{{<citation>}}

Etai Sella, Gal Fiebelman, Noam Atia, Hadar Averbuch-Elor. (2023)  
**SPiC-E : Structural Priors in 3D Diffusion Models using Cross-Entity Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.17834v2)  

---


**ABSTRACT**  
We are witnessing rapid progress in automatically generating and manipulating 3D assets due to the availability of pretrained text-image diffusion models. However, time-consuming optimization procedures are required for synthesizing each sample, hindering their potential for democratizing 3D content creation. Conversely, 3D diffusion models now train on million-scale 3D datasets, yielding high-quality text-conditional 3D samples within seconds. In this work, we present SPiC-E - a neural network that adds structural guidance to 3D diffusion models, extending their usage beyond text-conditional generation. At its core, our framework introduces a cross-entity attention mechanism that allows for multiple entities (in particular, paired input and guidance 3D shapes) to interact via their internal representations within the denoising network. We utilize this mechanism for learning task-specific structural priors in 3D diffusion models from auxiliary guidance shapes. We show that our approach supports a variety of applications, including 3D stylization, semantic shape editing and text-conditional abstraction-to-3D, which transforms primitive-based abstractions into highly-expressive shapes. Extensive experiments demonstrate that SPiC-E achieves SOTA performance over these tasks while often being considerably faster than alternative methods. Importantly, this is accomplished without tailoring our approach for any specific task.

{{</citation>}}


### (49/155) Analyzing and Explaining Image Classifiers via Diffusion Guidance (Maximilian Augustin et al., 2023)

{{<citation>}}

Maximilian Augustin, Yannic Neuhaus, Matthias Hein. (2023)  
**Analyzing and Explaining Image Classifiers via Diffusion Guidance**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.17833v1)  

---


**ABSTRACT**  
While deep learning has led to huge progress in complex image classification tasks like ImageNet, unexpected failure modes, e.g. via spurious features, call into question how reliably these classifiers work in the wild. Furthermore, for safety-critical tasks the black-box nature of their decisions is problematic, and explanations or at least methods which make decisions plausible are needed urgently. In this paper, we address these problems by generating images that optimize a classifier-derived objective using a framework for guided image generation. We analyze the behavior and decisions of image classifiers by visual counterfactual explanations (VCEs), detection of systematic mistakes by analyzing images where classifiers maximally disagree, and visualization of neurons to verify potential spurious features. In this way, we validate existing observations, e.g. the shape bias of adversarially robust models, as well as novel failure modes, e.g. systematic errors of zero-shot CLIP classifiers, or identify harmful spurious features. Moreover, our VCEs outperform previous work while being more versatile.

{{</citation>}}


### (50/155) AutArch: An AI-assisted workflow for object detection and automated recording in archaeological catalogues (Kevin Klein et al., 2023)

{{<citation>}}

Kevin Klein, Alyssa Wohde, Alexander V. Gorelik, Volker Heyd, Yoan Diekmann, Maxime Brami. (2023)  
**AutArch: An AI-assisted workflow for object detection and automated recording in archaeological catalogues**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17978v1)  

---


**ABSTRACT**  
Compiling large datasets from published resources, such as archaeological find catalogues presents fundamental challenges: identifying relevant content and manually recording it is a time-consuming, repetitive and error-prone task. For the data to be useful, it must be of comparable quality and adhere to the same recording standards, which is hardly ever the case in archaeology. Here, we present a new data collection method exploiting recent advances in Artificial Intelligence. Our software uses an object detection neural network combined with further classification networks to speed up, automate, and standardise data collection from legacy resources, such as archaeological drawings and photographs in large unsorted PDF files. The AI-assisted workflow detects common objects found in archaeological catalogues, such as graves, skeletons, ceramics, ornaments, stone tools and maps, and spatially relates and analyses these objects on the page to extract real-life attributes, such as the size and orientation of a grave based on the north arrow and the scale. A graphical interface allows for and assists with manual validation. We demonstrate the benefits of this approach by collecting a range of shapes and numerical attributes from richly-illustrated archaeological catalogues, and benchmark it in a real-world experiment with ten users. Moreover, we record geometric whole-outlines through contour detection, an alternative to landmark-based geometric morphometrics not achievable by hand.

{{</citation>}}


### (51/155) GeoDeformer: Geometric Deformable Transformer for Action Recognition (Jinhui Ye et al., 2023)

{{<citation>}}

Jinhui Ye, Jiaming Zhou, Hui Xiong, Junwei Liang. (2023)  
**GeoDeformer: Geometric Deformable Transformer for Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.17975v1)  

---


**ABSTRACT**  
Vision transformers have recently emerged as an effective alternative to convolutional networks for action recognition. However, vision transformers still struggle with geometric variations prevalent in video data. This paper proposes a novel approach, GeoDeformer, designed to capture the variations inherent in action video by integrating geometric comprehension directly into the ViT architecture. Specifically, at the core of GeoDeformer is the Geometric Deformation Predictor, a module designed to identify and quantify potential spatial and temporal geometric deformations within the given video. Spatial deformations adjust the geometry within individual frames, while temporal deformations capture the cross-frame geometric dynamics, reflecting motion and temporal progression. To demonstrate the effectiveness of our approach, we incorporate it into the established MViTv2 framework, replacing the standard self-attention blocks with GeoDeformer blocks. Our experiments at UCF101, HMDB51, and Mini-K200 achieve significant increases in both Top-1 and Top-5 accuracy, establishing new state-of-the-art results with only a marginal increase in computational cost. Additionally, visualizations affirm that GeoDeformer effectively manifests explicit geometric deformations and minimizes geometric variations. Codes and checkpoints will be released.

{{</citation>}}


### (52/155) PillarNeSt: Embracing Backbone Scaling and Pretraining for Pillar-based 3D Object Detection (Weixin Mao et al., 2023)

{{<citation>}}

Weixin Mao, Tiancai Wang, Diankun Zhang, Junjie Yan, Osamu Yoshie. (2023)  
**PillarNeSt: Embracing Backbone Scaling and Pretraining for Pillar-based 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: ImageNet, Object Detection  
[Paper Link](http://arxiv.org/abs/2311.17770v1)  

---


**ABSTRACT**  
This paper shows the effectiveness of 2D backbone scaling and pretraining for pillar-based 3D object detectors. Pillar-based methods mainly employ randomly initialized 2D convolution neural network (ConvNet) for feature extraction and fail to enjoy the benefits from the backbone scaling and pretraining in the image domain. To show the scaling-up capacity in point clouds, we introduce the dense ConvNet pretrained on large-scale image datasets (e.g., ImageNet) as the 2D backbone of pillar-based detectors. The ConvNets are adaptively designed based on the model size according to the specific features of point clouds, such as sparsity and irregularity. Equipped with the pretrained ConvNets, our proposed pillar-based detector, termed PillarNeSt, outperforms the existing 3D object detectors by a large margin on the nuScenes and Argoversev2 datasets. Our code shall be released upon acceptance.

{{</citation>}}


### (53/155) BAND-2k: Banding Artifact Noticeable Database for Banding Detection and Quality Assessment (Zijian Chen et al., 2023)

{{<citation>}}

Zijian Chen, Wei Sun, Jun Jia, Fangfang Lu, Zicheng Zhang, Jing Liu, Ru Huang, Xiongkuo Min, Guangtao Zhai. (2023)  
**BAND-2k: Banding Artifact Noticeable Database for Banding Detection and Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-DB, cs-MM, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.17752v1)  

---


**ABSTRACT**  
Banding, also known as staircase-like contours, frequently occurs in flat areas of images/videos processed by the compression or quantization algorithms. As undesirable artifacts, banding destroys the original image structure, thus degrading users' quality of experience (QoE). In this paper, we systematically investigate the banding image quality assessment (IQA) problem, aiming to detect the image banding artifacts and evaluate their perceptual visual quality. Considering that the existing image banding databases only contain limited content sources and banding generation methods, and lack perceptual quality labels (i.e. mean opinion scores), we first build the largest banding IQA database so far, named Banding Artifact Noticeable Database (BAND-2k), which consists of 2,000 banding images generated by 15 compression and quantization schemes. A total of 23 workers participated in the subjective IQA experiment, yielding over 214,000 patch-level banding class labels and 44,371 reliable image-level quality ratings. Subsequently, we develop an effective no-reference (NR) banding evaluator for banding detection and quality assessment by leveraging frequency characteristics of banding artifacts. A dual convolutional neural network is employed to concurrently learn the feature representation from the high-frequency and low-frequency maps, thereby enhancing the ability to discern banding artifacts. The quality score of a banding image is generated by pooling the banding detection maps masked by the spatial frequency filters. Experiments demonstrate that our banding evaluator achieves a remarkably high accuracy in banding detection and also exhibits high SRCC and PLCC results with the perceptual quality labels. These findings unveil the strong correlations between the intensity of banding artifacts and the perceptual visual quality, thus validating the necessity of banding quality assessment.

{{</citation>}}


### (54/155) GenZI: Zero-Shot 3D Human-Scene Interaction Generation (Lei Li et al., 2023)

{{<citation>}}

Lei Li, Angela Dai. (2023)  
**GenZI: Zero-Shot 3D Human-Scene Interaction Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.17737v1)  

---


**ABSTRACT**  
Can we synthesize 3D humans interacting with scenes without learning from any 3D human-scene interaction data? We propose GenZI, the first zero-shot approach to generating 3D human-scene interactions. Key to GenZI is our distillation of interaction priors from large vision-language models (VLMs), which have learned a rich semantic space of 2D human-scene compositions. Given a natural language description and a coarse point location of the desired interaction in a 3D scene, we first leverage VLMs to imagine plausible 2D human interactions inpainted into multiple rendered views of the scene. We then formulate a robust iterative optimization to synthesize the pose and shape of a 3D human model in the scene, guided by consistency with the 2D interaction hypotheses. In contrast to existing learning-based approaches, GenZI circumvents the conventional need for captured 3D interaction data, and allows for flexible control of the 3D interaction synthesis with easy-to-use text prompts. Extensive experiments show that our zero-shot approach has high flexibility and generality, making it applicable to diverse scene types, including both indoor and outdoor environments.

{{</citation>}}


### (55/155) SAMPro3D: Locating SAM Prompts in 3D for Zero-Shot Scene Segmentation (Mutian Xu et al., 2023)

{{<citation>}}

Mutian Xu, Xingyilang Yin, Lingteng Qiu, Yang Liu, Xin Tong, Xiaoguang Han. (2023)  
**SAMPro3D: Locating SAM Prompts in 3D for Zero-Shot Scene Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.17707v1)  

---


**ABSTRACT**  
We introduce SAMPro3D for zero-shot 3D indoor scene segmentation. Given the 3D point cloud and multiple posed 2D frames of 3D scenes, our approach segments 3D scenes by applying the pretrained Segment Anything Model (SAM) to 2D frames. Our key idea involves locating 3D points in scenes as natural 3D prompts to align their projected pixel prompts across frames, ensuring frame-consistency in both pixel prompts and their SAM-predicted masks. Moreover, we suggest filtering out low-quality 3D prompts based on feedback from all 2D frames, for enhancing segmentation quality. We also propose to consolidate different 3D prompts if they are segmenting the same object, bringing a more comprehensive segmentation. Notably, our method does not require any additional training on domain-specific data, enabling us to preserve the zero-shot power of SAM. Extensive qualitative and quantitative results show that our method consistently achieves higher quality and more diverse segmentation than previous zero-shot or fully supervised approaches, and in many cases even surpasses human-level annotations. The project page can be accessed at https://mutianxu.github.io/sampro3d/.

{{</citation>}}


### (56/155) Multiple Toddler Tracking in Indoor Videos (Somaieh Amraee et al., 2023)

{{<citation>}}

Somaieh Amraee, Bishoy Galoaa, Matthew Goodwin, Elaheh Hatamimajoumerd, Sarah Ostadabbas. (2023)  
**Multiple Toddler Tracking in Indoor Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17656v1)  

---


**ABSTRACT**  
Multiple toddler tracking (MTT) involves identifying and differentiating toddlers in video footage. While conventional multi-object tracking (MOT) algorithms are adept at tracking diverse objects, toddlers pose unique challenges due to their unpredictable movements, various poses, and similar appearance. Tracking toddlers in indoor environments introduces additional complexities such as occlusions and limited fields of view. In this paper, we address the challenges of MTT and propose MTTSort, a customized method built upon the DeepSort algorithm. MTTSort is designed to track multiple toddlers in indoor videos accurately. Our contributions include discussing the primary challenges in MTT, introducing a genetic algorithm to optimize hyperparameters, proposing an accurate tracking algorithm, and curating the MTTrack dataset using unbiased AI co-labeling techniques. We quantitatively compare MTTSort to state-of-the-art MOT methods on MTTrack, DanceTrack, and MOT15 datasets. In our evaluation, the proposed method outperformed other MOT methods, achieving 0.98, 0.68, and 0.98 in multiple object tracking accuracy (MOTA), higher order tracking accuracy (HOTA), and iterative and discriminative framework 1 (IDF1) metrics, respectively.

{{</citation>}}


### (57/155) VIM: Probing Multimodal Large Language Models for Visual Embedded Instruction Following (Yujie Lu et al., 2023)

{{<citation>}}

Yujie Lu, Xiujun Li, William Yang Wang, Yejin Choi. (2023)  
**VIM: Probing Multimodal Large Language Models for Visual Embedded Instruction Following**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.17647v1)  

---


**ABSTRACT**  
We introduce VISUAL EMBEDDED INSTRUCTION (VIM), a new framework designed to evaluate the visual instruction following capability of Multimodal Large Language Models (MLLMs). As illustrated in Figure 2, VIM challenges the MLLMs by embedding the instructions into the visual scenes, demanding strong visual interpretative skills for instruction following. We adapt VIM to various benchmarks, including VQAv2, MME, MM-Vet, and RefCOCO series, compose a VIM bench, and probe diverse MLLMs across three distinct in-context learning settings: Zero Shot, One Shot, and Pair Shot. We observe that there is a significant performance disparity between the open-source MLLMs and GPT-4V, implying that their proficiency in visual instruction comprehension is not up to par. Our results highlight a promising direction for the enhancement of MLLMs capabilities on instruction following. We aim VIM to serve as a useful norm for advancing the state of the art and driving further progress in the field.

{{</citation>}}


### (58/155) Efficient Decoder for End-to-End Oriented Object Detection in Remote Sensing Images (Jiaqi Zhao et al., 2023)

{{<citation>}}

Jiaqi Zhao, Zeyu Ding, Yong Zhou, Hancheng Zhu, Wenliang Du, Rui Yao, Abdulmotaleb El Saddik. (2023)  
**Efficient Decoder for End-to-End Oriented Object Detection in Remote Sensing Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.17629v2)  

---


**ABSTRACT**  
Object instances in remote sensing images often distribute with multi-orientations, varying scales, and dense distribution. These issues bring challenges to end-to-end oriented object detectors including multi-scale features alignment and a large number of queries. To address these limitations, we propose an end-to-end oriented detector equipped with an efficient decoder, which incorporates two technologies, Rotated RoI attention (RRoI attention) and Selective Distinct Queries (SDQ). Specifically, RRoI attention effectively focuses on oriented regions of interest through a cross-attention mechanism and aligns multi-scale features. SDQ collects queries from intermediate decoder layers and then filters similar queries to obtain distinct queries. The proposed SDQ can facilitate the optimization of one-to-one label assignment, without introducing redundant initial queries or extra auxiliary branches. Extensive experiments on five datasets demonstrate the effectiveness of our method. Notably, our method achieves state-of-the-art performance on DIOR-R (67.31% mAP), DOTA-v1.5 (67.43% mAP), and DOTA-v2.0 (53.28% mAP) with the ResNet50 backbone.

{{</citation>}}


### (59/155) Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation (Yuan Wang et al., 2023)

{{<citation>}}

Yuan Wang, Naisong Luo, Tianzhu Zhang. (2023)  
**Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Transformer  
[Paper Link](http://arxiv.org/abs/2311.17626v1)  

---


**ABSTRACT**  
Few-shot segmentation (FSS) aims to segment objects of new categories given only a handful of annotated samples. Previous works focus their efforts on exploring the support information while paying less attention to the mining of the critical query branch. In this paper, we rethink the importance of support information and propose a new query-centric FSS model Adversarial Mining Transformer (AMFormer), which achieves accurate query image segmentation with only rough support guidance or even weak support labels. The proposed AMFormer enjoys several merits. First, we design an object mining transformer (G) that can achieve the expansion of incomplete region activated by support clue, and a detail mining transformer (D) to discriminate the detailed local difference between the expanded mask and the ground truth. Second, we propose to train G and D via an adversarial process, where G is optimized to generate more accurate masks approaching ground truth to fool D. We conduct extensive experiments on commonly used Pascal-5i and COCO-20i benchmarks and achieve state-of-the-art results across all settings. In addition, the decent performance with weak support labels in our query-centric paradigm may inspire the development of more general FSS models. Code will be available at https://github.com/Wyxdm/AMNet.

{{</citation>}}


### (60/155) ShapeGPT: 3D Shape Generation with A Unified Multi-modal Language Model (Fukun Yin et al., 2023)

{{<citation>}}

Fukun Yin, Xin Chen, Chi Zhang, Biao Jiang, Zibo Zhao, Jiayuan Fan, Gang Yu, Taihao Li, Tao Chen. (2023)  
**ShapeGPT: 3D Shape Generation with A Unified Multi-modal Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17618v3)  

---


**ABSTRACT**  
The advent of large language models, enabling flexibility through instruction-driven approaches, has revolutionized many traditional generative tasks, but large models for 3D data, particularly in comprehensively handling 3D shapes with other modalities, are still under-explored. By achieving instruction-based shape generations, versatile multimodal generative shape models can significantly benefit various fields like 3D virtual construction and network-aided design. In this work, we present ShapeGPT, a shape-included multi-modal framework to leverage strong pre-trained language models to address multiple shape-relevant tasks. Specifically, ShapeGPT employs a word-sentence-paragraph framework to discretize continuous shapes into shape words, further assembles these words for shape sentences, as well as integrates shape with instructional text for multi-modal paragraphs. To learn this shape-language model, we use a three-stage training scheme, including shape representation, multimodal alignment, and instruction-based generation, to align shape-language codebooks and learn the intricate correlations among these modalities. Extensive experiments demonstrate that ShapeGPT achieves comparable performance across shape-relevant tasks, including text-to-shape, shape-to-text, shape completion, and shape editing.

{{</citation>}}


### (61/155) Adversarial Robust Memory-Based Continual Learner (Xiaoyue Mi et al., 2023)

{{<citation>}}

Xiaoyue Mi, Fan Tang, Zonghan Yang, Danding Wang, Juan Cao, Peng Li, Yang Liu. (2023)  
**Adversarial Robust Memory-Based Continual Learner**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.17608v1)  

---


**ABSTRACT**  
Despite the remarkable advances that have been made in continual learning, the adversarial vulnerability of such methods has not been fully discussed. We delve into the adversarial robustness of memory-based continual learning algorithms and observe limited robustness improvement by directly applying adversarial training techniques. Preliminary studies reveal the twin challenges for building adversarial robust continual learners: accelerated forgetting in continual learning and gradient obfuscation in adversarial robustness. In this study, we put forward a novel adversarial robust memory-based continual learner that adjusts data logits to mitigate the forgetting of pasts caused by adversarial samples. Furthermore, we devise a gradient-based data selection mechanism to overcome the gradient obfuscation caused by limited stored data. The proposed approach can widely integrate with existing memory-based continual learning as well as adversarial training algorithms in a plug-and-play way. Extensive experiments on Split-CIFAR10/100 and Split-Tiny-ImageNet demonstrate the effectiveness of our approach, achieving up to 8.13% higher accuracy for adversarial data.

{{</citation>}}


### (62/155) Topology-Preserving Adversarial Training (Xiaoyue Mi et al., 2023)

{{<citation>}}

Xiaoyue Mi, Fan Tang, Yepeng Weng, Danding Wang, Juan Cao, Sheng Tang, Peng Li, Yang Liu. (2023)  
**Topology-Preserving Adversarial Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Adversarial Training, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.17607v1)  

---


**ABSTRACT**  
Despite the effectiveness in improving the robustness of neural networks, adversarial training has suffered from the natural accuracy degradation problem, i.e., accuracy on natural samples has reduced significantly. In this study, we reveal that natural accuracy degradation is highly related to the disruption of the natural sample topology in the representation space by quantitative and qualitative experiments. Based on this observation, we propose Topology-pReserving Adversarial traINing (TRAIN) to alleviate the problem by preserving the topology structure of natural samples from a standard model trained only on natural samples during adversarial training. As an additional regularization, our method can easily be combined with various popular adversarial training algorithms in a plug-and-play manner, taking advantage of both sides. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny ImageNet show that our proposed method achieves consistent and significant improvements over various strong baselines in most cases. Specifically, without additional data, our proposed method achieves up to 8.78% improvement in natural accuracy and 4.50% improvement in robust accuracy.

{{</citation>}}


### (63/155) Query-Relevant Images Jailbreak Large Multi-Modal Models (Xin Liu et al., 2023)

{{<citation>}}

Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao. (2023)  
**Query-Relevant Images Jailbreak Large Multi-Modal Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17600v1)  

---


**ABSTRACT**  
Warning: This paper contains examples of harmful language and images, and reader discretion is recommended. The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Large Multi-Modal Models (LMMs) remains understudied. In our study, we present a novel visual prompt attack that exploits query-relevant images to jailbreak the open-source LMMs. Our method creates a composite image from one image generated by diffusion models and another that displays the text as typography, based on keywords extracted from a malicious query. We show LLMs can be easily attacked by our approach, even if the employed Large Language Models are safely aligned. To evaluate the extent of this vulnerability in open-source LMMs, we have compiled a substantial dataset encompassing 13 scenarios with a total of 5,040 text-image pairs, using our presented attack technique. Our evaluation of 12 cutting-edge LMMs using this dataset shows the vulnerability of existing multi-modal models on adversarial attacks. This finding underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source LMMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.

{{</citation>}}


### (64/155) Continual Self-supervised Learning: Towards Universal Multi-modal Medical Data Representation Learning (Yiwen Ye et al., 2023)

{{<citation>}}

Yiwen Ye, Yutong Xie, Jianpeng Zhang, Ziyang Chen, Qi Wu, Yong Xia. (2023)  
**Continual Self-supervised Learning: Towards Universal Multi-modal Medical Data Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.17597v2)  

---


**ABSTRACT**  
Self-supervised learning is an efficient pre-training method for medical image analysis. However, current research is mostly confined to specific-modality data pre-training, consuming considerable time and resources without achieving universality across different modalities. A straightforward solution is combining all modality data for joint self-supervised pre-training, which poses practical challenges. Firstly, our experiments reveal conflicts in representation learning as the number of modalities increases. Secondly, multi-modal data collected in advance cannot cover all real-world scenarios. In this paper, we reconsider versatile self-supervised learning from the perspective of continual learning and propose MedCoSS, a continuous self-supervised learning approach for multi-modal medical data. Unlike joint self-supervised learning, MedCoSS assigns different modality data to different training stages, forming a multi-stage pre-training process. To balance modal conflicts and prevent catastrophic forgetting, we propose a rehearsal-based continual learning method. We introduce the k-means sampling strategy to retain data from previous modalities and rehearse it when learning new modalities. Instead of executing the pretext task on buffer data, a feature distillation strategy and an intra-modal mixup strategy are applied to these data for knowledge retention. We conduct continuous self-supervised pre-training on a large-scale multi-modal unlabeled dataset, including clinical reports, X-rays, CT scans, MRI scans, and pathological images. Experimental results demonstrate MedCoSS's exceptional generalization ability across nine downstream datasets and its significant scalability in integrating new modality data. Code and pre-trained weight are available at https://github.com/yeerwen/MedCoSS.

{{</citation>}}


### (65/155) LGFCTR: Local and Global Feature Convolutional Transformer for Image Matching (Wenhao Zhong et al., 2023)

{{<citation>}}

Wenhao Zhong, Jie Jiang. (2023)  
**LGFCTR: Local and Global Feature Convolutional Transformer for Image Matching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.17571v1)  

---


**ABSTRACT**  
Image matching that finding robust and accurate correspondences across images is a challenging task under extreme conditions. Capturing local and global features simultaneously is an important way to mitigate such an issue but recent transformer-based decoders were still stuck in the issues that CNN-based encoders only extract local features and the transformers lack locality. Inspired by the locality and implicit positional encoding of convolutions, a novel convolutional transformer is proposed to capture both local contexts and global structures more sufficiently for detector-free matching. Firstly, a universal FPN-like framework captures global structures in self-encoder as well as cross-decoder by transformers and compensates local contexts as well as implicit positional encoding by convolutions. Secondly, a novel convolutional transformer module explores multi-scale long range dependencies by a novel multi-scale attention and further aggregates local information inside dependencies for enhancing locality. Finally, a novel regression-based sub-pixel refinement module exploits the whole fine-grained window features for fine-level positional deviation regression. The proposed method achieves superior performances on a wide range of benchmarks. The code will be available on https://github.com/zwh0527/LGFCTR.

{{</citation>}}


### (66/155) ChatIllusion: Efficient-Aligning Interleaved Generation ability with Visual Instruction Model (Xiaowei Chi et al., 2023)

{{<citation>}}

Xiaowei Chi, Yijiang Liu, Zhengkai Jiang, Rongyu Zhang, Ziyi Lin, Renrui Zhang, Peng Gao, Chaoyou Fu, Shanghang Zhang, Qifeng Liu, Yike Guo. (2023)  
**ChatIllusion: Efficient-Aligning Interleaved Generation ability with Visual Instruction Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17963v1)  

---


**ABSTRACT**  
As the capabilities of Large-Language Models (LLMs) become widely recognized, there is an increasing demand for human-machine chat applications. Human interaction with text often inherently invokes mental imagery, an aspect that existing LLM-based chatbots like GPT-4 do not currently emulate, as they are confined to generating text-only content. To bridge this gap, we introduce ChatIllusion, an advanced Generative multimodal large language model (MLLM) that combines the capabilities of LLM with not only visual comprehension but also creativity. Specifically, ChatIllusion integrates Stable Diffusion XL and Llama, which have been fine-tuned on modest image-caption data, to facilitate multiple rounds of illustrated chats. The central component of ChatIllusion is the "GenAdapter," an efficient approach that equips the multimodal language model with capabilities for visual representation, without necessitating modifications to the foundational model. Extensive experiments validate the efficacy of our approach, showcasing its ability to produce diverse and superior-quality image outputs Simultaneously, it preserves semantic consistency and control over the dialogue, significantly enhancing the overall user's quality of experience (QoE). The code is available at https://github.com/litwellchi/ChatIllusion.

{{</citation>}}


### (67/155) VINNA for Neonates -- Orientation Independence through Latent Augmentations (Leonie Henschel et al., 2023)

{{<citation>}}

Leonie Henschel, David Kügler, Lilla Zöllei, Martin Reuter. (2023)  
**VINNA for Neonates -- Orientation Independence through Latent Augmentations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.17546v1)  

---


**ABSTRACT**  
Fast and accurate segmentation of neonatal brain images is highly desired to better understand and detect changes during development and disease. Yet, the limited availability of ground truth datasets, lack of standardized acquisition protocols, and wide variations of head positioning pose challenges for method development. A few automated image analysis pipelines exist for newborn brain MRI segmentation, but they often rely on time-consuming procedures and require resampling to a common resolution, subject to loss of information due to interpolation and down-sampling. Without registration and image resampling, variations with respect to head positions and voxel resolutions have to be addressed differently. In deep-learning, external augmentations are traditionally used to artificially expand the representation of spatial variability, increasing the training dataset size and robustness. However, these transformations in the image space still require resampling, reducing accuracy specifically in the context of label interpolation. We recently introduced the concept of resolution-independence with the Voxel-size Independent Neural Network framework, VINN. Here, we extend this concept by additionally shifting all rigid-transforms into the network architecture with a four degree of freedom (4-DOF) transform module, enabling resolution-aware internal augmentations (VINNA). In this work we show that VINNA (i) significantly outperforms state-of-the-art external augmentation approaches, (ii) effectively addresses the head variations present specifically in newborn datasets, and (iii) retains high segmentation accuracy across a range of resolutions (0.5-1.0 mm). The 4-DOF transform module is a powerful, general approach to implement spatial augmentation without requiring image or label interpolation. The specific network application to newborns will be made publicly available as VINNA4neonates.

{{</citation>}}


### (68/155) Weakly-Supervised Emotion Transition Learning for Diverse 3D Co-speech Gesture Generation (Xingqun Qi et al., 2023)

{{<citation>}}

Xingqun Qi, Jiahao Pan, Peng Li, Ruibin Yuan, Xiaowei Chi, Mengfei Li, Wenhan Luo, Wei Xue, Shanghang Zhang, Qifeng Liu, Yike Guo. (2023)  
**Weakly-Supervised Emotion Transition Learning for Diverse 3D Co-speech Gesture Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.17532v1)  

---


**ABSTRACT**  
Generating vivid and emotional 3D co-speech gestures is crucial for virtual avatar animation in human-machine interaction applications. While the existing methods enable generating the gestures to follow a single emotion label, they overlook that long gesture sequence modeling with emotion transition is more practical in real scenes. In addition, the lack of large-scale available datasets with emotional transition speech and corresponding 3D human gestures also limits the addressing of this task. To fulfill this goal, we first incorporate the ChatGPT-4 and an audio inpainting approach to construct the high-fidelity emotion transition human speeches. Considering obtaining the realistic 3D pose annotations corresponding to the dynamically inpainted emotion transition audio is extremely difficult, we propose a novel weakly supervised training strategy to encourage authority gesture transitions. Specifically, to enhance the coordination of transition gestures w.r.t different emotional ones, we model the temporal association representation between two different emotional gesture sequences as style guidance and infuse it into the transition generation. We further devise an emotion mixture mechanism that provides weak supervision based on a learnable mixed emotion label for transition gestures. Last, we present a keyframe sampler to supply effective initial posture cues in long sequences, enabling us to generate diverse gestures. Extensive experiments demonstrate that our method outperforms the state-of-the-art models constructed by adapting single emotion-conditioned counterparts on our newly defined emotion transition task and datasets.

{{</citation>}}


### (69/155) HiDiffusion: Unlocking High-Resolution Creativity and Efficiency in Low-Resolution Trained Diffusion Models (Shen Zhang et al., 2023)

{{<citation>}}

Shen Zhang, Zhaowei Chen, Zhenyu Zhao, Zhenyuan Chen, Yao Tang, Yuhao Chen, Wengang Cao, Jiajun Liang. (2023)  
**HiDiffusion: Unlocking High-Resolution Creativity and Efficiency in Low-Resolution Trained Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2311.17528v1)  

---


**ABSTRACT**  
We introduce HiDiffusion, a tuning-free framework comprised of Resolution-Aware U-Net (RAU-Net) and Modified Shifted Window Multi-head Self-Attention (MSW-MSA) to enable pretrained large text-to-image diffusion models to efficiently generate high-resolution images (e.g. 1024$\times$1024) that surpass the training image resolution. Pretrained diffusion models encounter unreasonable object duplication in generating images beyond the training image resolution. We attribute it to the mismatch between the feature map size of high-resolution images and the receptive field of U-Net's convolution. To address this issue, we propose a simple yet scalable method named RAU-Net. RAU-Net dynamically adjusts the feature map size to match the convolution's receptive field in the deep block of U-Net. Another obstacle in high-resolution synthesis is the slow inference speed of U-Net. Our observations reveal that the global self-attention in the top block, which exhibits locality, however, consumes the majority of computational resources. To tackle this issue, we propose MSW-MSA. Unlike previous window attention mechanisms, our method uses a much larger window size and dynamically shifts windows to better accommodate diffusion models. Extensive experiments demonstrate that our HiDiffusion can scale diffusion models to generate 1024$\times$1024, 2048$\times$2048, or even 4096$\times$4096 resolution images, while simultaneously reducing inference time by 40\%-60\%, achieving state-of-the-art performance on high-resolution image synthesis. The most significant revelation of our work is that a pretrained diffusion model on low-resolution images is scalable for high-resolution generation without further tuning. We hope this revelation can provide insights for future research on the scalability of diffusion models.

{{</citation>}}


### (70/155) PViT-6D: Overclocking Vision Transformers for 6D Pose Estimation with Confidence-Level Prediction and Pose Tokens (Sebastian Stapf et al., 2023)

{{<citation>}}

Sebastian Stapf, Tobias Bauernfeind, Marco Riboldi. (2023)  
**PViT-6D: Overclocking Vision Transformers for 6D Pose Estimation with Confidence-Level Prediction and Pose Tokens**  

---
Primary Category: cs.CV  
Categories: I-2-10, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.17504v1)  

---


**ABSTRACT**  
In the current state of 6D pose estimation, top-performing techniques depend on complex intermediate correspondences, specialized architectures, and non-end-to-end algorithms. In contrast, our research reframes the problem as a straightforward regression task by exploring the capabilities of Vision Transformers for direct 6D pose estimation through a tailored use of classification tokens. We also introduce a simple method for determining pose confidence, which can be readily integrated into most 6D pose estimation frameworks. This involves modifying the transformer architecture by decreasing the number of query elements based on the network's assessment of the scene complexity. Our method that we call Pose Vision Transformer or PViT-6D provides the benefits of simple implementation and being end-to-end learnable while outperforming current state-of-the-art methods by +0.3% ADD(-S) on Linemod-Occlusion and +2.7% ADD(-S) on the YCB-V dataset. Moreover, our method enhances both the model's interpretability and the reliability of its performance during inference.

{{</citation>}}


### (71/155) Towards Higher Ranks via Adversarial Weight Pruning (Yuchuan Tian et al., 2023)

{{<citation>}}

Yuchuan Tian, Hanting Chen, Tianyu Guo, Chao Xu, Yunhe Wang. (2023)  
**Towards Higher Ranks via Adversarial Weight Pruning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2311.17493v1)  

---


**ABSTRACT**  
Convolutional Neural Networks (CNNs) are hard to deploy on edge devices due to its high computation and storage complexities. As a common practice for model compression, network pruning consists of two major categories: unstructured and structured pruning, where unstructured pruning constantly performs better. However, unstructured pruning presents a structured pattern at high pruning rates, which limits its performance. To this end, we propose a Rank-based PruninG (RPG) method to maintain the ranks of sparse weights in an adversarial manner. In each step, we minimize the low-rank approximation error for the weight matrices using singular value decomposition, and maximize their distance by pushing the weight matrices away from its low rank approximation. This rank-based optimization objective guides sparse weights towards a high-rank topology. The proposed method is conducted in a gradual pruning fashion to stabilize the change of rank during training. Experimental results on various datasets and different tasks demonstrate the effectiveness of our algorithm in high sparsity. The proposed RPG outperforms the state-of-the-art performance by 1.13% top-1 accuracy on ImageNet in ResNet-50 with 98% sparsity. The codes are available at https://github.com/huawei-noah/Efficient-Computing/tree/master/Pruning/RPG and https://gitee.com/mindspore/models/tree/master/research/cv/RPG.

{{</citation>}}


### (72/155) Spherical Frustum Sparse Convolution Network for LiDAR Point Cloud Semantic Segmentation (Yu Zheng et al., 2023)

{{<citation>}}

Yu Zheng, Guangming Wang, Jiuming Liu, Marc Pollefeys, Hesheng Wang. (2023)  
**Spherical Frustum Sparse Convolution Network for LiDAR Point Cloud Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.17491v1)  

---


**ABSTRACT**  
LiDAR point cloud semantic segmentation enables the robots to obtain fine-grained semantic information of the surrounding environment. Recently, many works project the point cloud onto the 2D image and adopt the 2D Convolutional Neural Networks (CNNs) or vision transformer for LiDAR point cloud semantic segmentation. However, since more than one point can be projected onto the same 2D position but only one point can be preserved, the previous 2D image-based segmentation methods suffer from inevitable quantized information loss. To avoid quantized information loss, in this paper, we propose a novel spherical frustum structure. The points projected onto the same 2D position are preserved in the spherical frustums. Moreover, we propose a memory-efficient hash-based representation of spherical frustums. Through the hash-based representation, we propose the Spherical Frustum sparse Convolution (SFC) and Frustum Fast Point Sampling (F2PS) to convolve and sample the points stored in spherical frustums respectively. Finally, we present the Spherical Frustum sparse Convolution Network (SFCNet) to adopt 2D CNNs for LiDAR point cloud semantic segmentation without quantized information loss. Extensive experiments on the SemanticKITTI and nuScenes datasets demonstrate that our SFCNet outperforms the 2D image-based semantic segmentation methods based on conventional spherical projection. The source code will be released later.

{{</citation>}}


### (73/155) CLiSA: A Hierarchical Hybrid Transformer Model using Orthogonal Cross Attention for Satellite Image Cloud Segmentation (Subhajit Paul et al., 2023)

{{<citation>}}

Subhajit Paul, Ashutosh Gupta. (2023)  
**CLiSA: A Hierarchical Hybrid Transformer Model using Orthogonal Cross Attention for Satellite Image Cloud Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.17475v2)  

---


**ABSTRACT**  
Clouds in optical satellite images are a major concern since their presence hinders the ability to carry accurate analysis as well as processing. Presence of clouds also affects the image tasking schedule and results in wastage of valuable storage space on ground as well as space-based systems. Due to these reasons, deriving accurate cloud masks from optical remote-sensing images is an important task. Traditional methods such as threshold-based, spatial filtering for cloud detection in satellite images suffer from lack of accuracy. In recent years, deep learning algorithms have emerged as a promising approach to solve image segmentation problems as it allows pixel-level classification and semantic-level segmentation. In this paper, we introduce a deep-learning model based on hybrid transformer architecture for effective cloud mask generation named CLiSA - Cloud segmentation via Lipschitz Stable Attention network. In this context, we propose an concept of orthogonal self-attention combined with hierarchical cross attention model, and we validate its Lipschitz stability theoretically and empirically. We design the whole setup under adversarial setting in presence of Lov\'asz-Softmax loss. We demonstrate both qualitative and quantitative outcomes for multiple satellite image datasets including Landsat-8, Sentinel-2, and Cartosat-2s. Performing comparative study we show that our model performs preferably against other state-of-the-art methods and also provides better generalization in precise cloud extraction from satellite multi-spectral (MX) images. We also showcase different ablation studies to endorse our choices corresponding to different architectural elements and objective functions.

{{</citation>}}


### (74/155) Continual Learning for Image Segmentation with Dynamic Query (Weijia Wu et al., 2023)

{{<citation>}}

Weijia Wu, Yuzhong Zhao, Zhuang Li, Lianlei Shan, Hong Zhou, Mike Zheng Shou. (2023)  
**Continual Learning for Image Segmentation with Dynamic Query**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.17450v1)  

---


**ABSTRACT**  
Image segmentation based on continual learning exhibits a critical drop of performance, mainly due to catastrophic forgetting and background shift, as they are required to incorporate new classes continually. In this paper, we propose a simple, yet effective Continual Image Segmentation method with incremental Dynamic Query (CISDQ), which decouples the representation learning of both old and new knowledge with lightweight query embedding. CISDQ mainly includes three contributions: 1) We define dynamic queries with adaptive background class to exploit past knowledge and learn future classes naturally. 2) CISDQ proposes a class/instance-aware Query Guided Knowledge Distillation strategy to overcome catastrophic forgetting by capturing the inter-class diversity and intra-class identity. 3) Apart from semantic segmentation, CISDQ introduce the continual learning for instance segmentation in which instance-wise labeling and supervision are considered. Extensive experiments on three datasets for two tasks (i.e., continual semantic and instance segmentation are conducted to demonstrate that CISDQ achieves the state-of-the-art performance, specifically, obtaining 4.4% and 2.9% mIoU improvements for the ADE 100-10 (6 steps) setting and ADE 100-5 (11 steps) setting.

{{</citation>}}


### (75/155) Weakly-semi-supervised object detection in remotely sensed imagery (Ji Hun Wang et al., 2023)

{{<citation>}}

Ji Hun Wang, Jeremy Irvin, Beri Kohen Behar, Ha Tran, Raghav Samavedam, Quentin Hsu, Andrew Y. Ng. (2023)  
**Weakly-semi-supervised object detection in remotely sensed imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17449v1)  

---


**ABSTRACT**  
Deep learning for detecting objects in remotely sensed imagery can enable new technologies for important applications including mitigating climate change. However, these models often require large datasets labeled with bounding box annotations which are expensive to curate, prohibiting the development of models for new tasks and geographies. To address this challenge, we develop weakly-semi-supervised object detection (WSSOD) models on remotely sensed imagery which can leverage a small amount of bounding boxes together with a large amount of point labels that are easy to acquire at scale in geospatial data. We train WSSOD models which use large amounts of point-labeled images with varying fractions of bounding box labeled images in FAIR1M and a wind turbine detection dataset, and demonstrate that they substantially outperform fully supervised models trained with the same amount of bounding box labeled images on both datasets. Furthermore, we find that the WSSOD models trained with 2-10x fewer bounding box labeled images can perform similarly to or outperform fully supervised models trained on the full set of bounding-box labeled images. We believe that the approach can be extended to other remote sensing tasks to reduce reliance on bounding box labels and increase development of models for impactful applications.

{{</citation>}}


### (76/155) MM-Narrator: Narrating Long-form Videos with Multimodal In-Context Learning (Chaoyi Zhang et al., 2023)

{{<citation>}}

Chaoyi Zhang, Kevin Lin, Zhengyuan Yang, Jianfeng Wang, Linjie Li, Chung-Ching Lin, Zicheng Liu, Lijuan Wang. (2023)  
**MM-Narrator: Narrating Long-form Videos with Multimodal In-Context Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.17435v1)  

---


**ABSTRACT**  
We present MM-Narrator, a novel system leveraging GPT-4 with multimodal in-context learning for the generation of audio descriptions (AD). Unlike previous methods that primarily focused on downstream fine-tuning with short video clips, MM-Narrator excels in generating precise audio descriptions for videos of extensive lengths, even beyond hours, in an autoregressive manner. This capability is made possible by the proposed memory-augmented generation process, which effectively utilizes both the short-term textual context and long-term visual memory through an efficient register-and-recall mechanism. These contextual memories compile pertinent past information, including storylines and character identities, ensuring an accurate tracking and depicting of story-coherent and character-centric audio descriptions. Maintaining the training-free design of MM-Narrator, we further propose a complexity-based demonstration selection strategy to largely enhance its multi-step reasoning capability via few-shot multimodal in-context learning (MM-ICL). Experimental results on MAD-eval dataset demonstrate that MM-Narrator consistently outperforms both the existing fine-tuning-based approaches and LLM-based approaches in most scenarios, as measured by standard evaluation metrics. Additionally, we introduce the first segment-based evaluator for recurrent text generation. Empowered by GPT-4, this evaluator comprehensively reasons and marks AD generation performance in various extendable dimensions.

{{</citation>}}


### (77/155) Group-wise Sparse and Explainable Adversarial Attacks (Shpresim Sadiku et al., 2023)

{{<citation>}}

Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta. (2023)  
**Group-wise Sparse and Explainable Adversarial Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV, math-OC  
Keywords: Adversarial Attack, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.17434v1)  

---


**ABSTRACT**  
Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, typically regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs than previously anticipated. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. In this paper, we tackle this challenge by presenting an algorithm that simultaneously generates group-wise sparse attacks within semantically meaningful areas of an image. In each iteration, the core operation of our algorithm involves the optimization of a quasinorm adversarial loss. This optimization is achieved by employing the $1/2$-quasinorm proximal operator for some iterations, a method tailored for nonconvex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2$-norm regularization applied to perturbation magnitudes. We rigorously evaluate the efficacy of our novel attack in both targeted and non-targeted attack scenarios, on CIFAR-10 and ImageNet datasets. When compared to state-of-the-art methods, our attack consistently results in a remarkable increase in group-wise sparsity, e.g., an increase of $48.12\%$ on CIFAR-10 and $40.78\%$ on ImageNet (average case, targeted attack), all while maintaining lower perturbation magnitudes. Notably, this performance is complemented by a significantly faster computation time and a $100\%$ attack success rate.

{{</citation>}}


### (78/155) PEAN: A Diffusion-based Prior-Enhanced Attention Network for Scene Text Image Super-Resolution (Zuoyan Zhao et al., 2023)

{{<citation>}}

Zuoyan Zhao, Shipeng Zhu, Pengfei Fang, Hui Xue. (2023)  
**PEAN: A Diffusion-based Prior-Enhanced Attention Network for Scene Text Image Super-Resolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.17955v1)  

---


**ABSTRACT**  
Scene text image super-resolution (STISR) aims at simultaneously increasing the resolution and readability of low-resolution scene text images, thus boosting the performance of the downstream recognition task. Two factors in scene text images, semantic information and visual structure, affect the recognition performance significantly. To mitigate the effects from these factors, this paper proposes a Prior-Enhanced Attention Network (PEAN). Specifically, a diffusion-based module is developed to enhance the text prior, hence offering better guidance for the SR network to generate SR images with higher semantic accuracy. Meanwhile, the proposed PEAN leverages an attention-based modulation module to understand scene text images by neatly perceiving the local and global dependence of images, despite the shape of the text. A multi-task learning paradigm is employed to optimize the network, enabling the model to generate legible SR images. As a result, PEAN establishes new SOTA results on the TextZoom benchmark. Experiments are also conducted to analyze the importance of the enhanced text prior as a means of improving the performance of the SR network. Code will be made available at https://github.com/jdfxzzy/PEAN.

{{</citation>}}


### (79/155) Transformer-empowered Multi-modal Item Embedding for Enhanced Image Search in E-Commerce (Chang Liu et al., 2023)

{{<citation>}}

Chang Liu, Peng Hou, Anxiang Zeng, Han Yu. (2023)  
**Transformer-empowered Multi-modal Item Embedding for Enhanced Image Search in E-Commerce**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2311.17954v1)  

---


**ABSTRACT**  
Over the past decade, significant advances have been made in the field of image search for e-commerce applications. Traditional image-to-image retrieval models, which focus solely on image details such as texture, tend to overlook useful semantic information contained within the images. As a result, the retrieved products might possess similar image details, but fail to fulfil the user's search goals. Moreover, the use of image-to-image retrieval models for products containing multiple images results in significant online product feature storage overhead and complex mapping implementations. In this paper, we report the design and deployment of the proposed Multi-modal Item Embedding Model (MIEM) to address these limitations. It is capable of utilizing both textual information and multiple images about a product to construct meaningful product features. By leveraging semantic information from images, MIEM effectively supplements the image search process, improving the overall accuracy of retrieval results. MIEM has become an integral part of the Shopee image search platform. Since its deployment in March 2023, it has achieved a remarkable 9.90% increase in terms of clicks per user and a 4.23% boost in terms of orders per user for the image search feature on the Shopee e-commerce platform.

{{</citation>}}


### (80/155) SigFormer: Sparse Signal-Guided Transformer for Multi-Modal Human Action Segmentation (Qi Liu et al., 2023)

{{<citation>}}

Qi Liu, Xinchen Liu, Kun Liu, Xiaoyan Gu, Wu Liu. (2023)  
**SigFormer: Sparse Signal-Guided Transformer for Multi-Modal Human Action Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.17428v1)  

---


**ABSTRACT**  
Multi-modal human action segmentation is a critical and challenging task with a wide range of applications. Nowadays, the majority of approaches concentrate on the fusion of dense signals (i.e., RGB, optical flow, and depth maps). However, the potential contributions of sparse IoT sensor signals, which can be crucial for achieving accurate recognition, have not been fully explored. To make up for this, we introduce a Sparse signalguided Transformer (SigFormer) to combine both dense and sparse signals. We employ mask attention to fuse localized features by constraining cross-attention within the regions where sparse signals are valid. However, since sparse signals are discrete, they lack sufficient information about the temporal action boundaries. Therefore, in SigFormer, we propose to emphasize the boundary information at two stages to alleviate this problem. In the first feature extraction stage, we introduce an intermediate bottleneck module to jointly learn both category and boundary features of each dense modality through the inner loss functions. After the fusion of dense modalities and sparse signals, we then devise a two-branch architecture that explicitly models the interrelationship between action category and temporal boundary. Experimental results demonstrate that SigFormer outperforms the state-of-the-art approaches on a multi-modal action segmentation dataset from real industrial environments, reaching an outstanding F1 score of 0.958. The codes and pre-trained models have been available at https://github.com/LIUQI-creat/SigFormer.

{{</citation>}}


### (81/155) Rethinking Image Editing Detection in the Era of Generative AI Revolution (Zhihao Sun et al., 2023)

{{<citation>}}

Zhihao Sun, Haipeng Fang, Xinying Zhao, Danding Wang, Juan Cao. (2023)  
**Rethinking Image Editing Detection in the Era of Generative AI Revolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.17953v1)  

---


**ABSTRACT**  
The accelerated advancement of generative AI significantly enhance the viability and effectiveness of generative regional editing methods. This evolution render the image manipulation more accessible, thereby intensifying the risk of altering the conveyed information within original images and even propagating misinformation. Consequently, there exists a critical demand for robust capable of detecting the edited images. However, the lack of comprehensive dataset containing images edited with abundant and advanced generative regional editing methods poses a substantial obstacle to the advancement of corresponding detection methods.   We endeavor to fill the vacancy by constructing the GRE dataset, a large-scale generative regional editing dataset with the following advantages: 1) Collection of real-world original images, focusing on two frequently edited scenarios. 2) Integration of a logical and simulated editing pipeline, leveraging multiple large models in various modalities. 3) Inclusion of various editing approaches with distinct architectures. 4) Provision of comprehensive analysis tasks. We perform comprehensive experiments with proposed three tasks: edited image classification, edited method attribution and edited region localization, providing analysis of distinct editing methods and evaluation of detection methods in related fields. We expect that the GRE dataset can promote further research and exploration in the field of generative region editing detection.

{{</citation>}}


### (82/155) Dynamic Dense Graph Convolutional Network for Skeleton-based Human Motion Prediction (Xinshun Wang et al., 2023)

{{<citation>}}

Xinshun Wang, Wanying Zhang, Can Wang, Yuan Gao, Mengyuan Liu. (2023)  
**Dynamic Dense Graph Convolutional Network for Skeleton-based Human Motion Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2311.17408v1)  

---


**ABSTRACT**  
Graph Convolutional Networks (GCN) which typically follows a neural message passing framework to model dependencies among skeletal joints has achieved high success in skeleton-based human motion prediction task. Nevertheless, how to construct a graph from a skeleton sequence and how to perform message passing on the graph are still open problems, which severely affect the performance of GCN. To solve both problems, this paper presents a Dynamic Dense Graph Convolutional Network (DD-GCN), which constructs a dense graph and implements an integrated dynamic message passing. More specifically, we construct a dense graph with 4D adjacency modeling as a comprehensive representation of motion sequence at different levels of abstraction. Based on the dense graph, we propose a dynamic message passing framework that learns dynamically from data to generate distinctive messages reflecting sample-specific relevance among nodes in the graph. Extensive experiments on benchmark Human 3.6M and CMU Mocap datasets verify the effectiveness of our DD-GCN which obviously outperforms state-of-the-art GCN-based methods, especially when using long-term and our proposed extremely long-term protocol.

{{</citation>}}


### (83/155) VITATECS: A Diagnostic Dataset for Temporal Concept Understanding of Video-Language Models (Shicheng Li et al., 2023)

{{<citation>}}

Shicheng Li, Lei Li, Shuhuai Ren, Yuanxin Liu, Yi Liu, Rundong Gao, Xu Sun, Lu Hou. (2023)  
**VITATECS: A Diagnostic Dataset for Temporal Concept Understanding of Video-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17404v1)  

---


**ABSTRACT**  
The ability to perceive how objects change over time is a crucial ingredient in human intelligence. However, current benchmarks cannot faithfully reflect the temporal understanding abilities of video-language models (VidLMs) due to the existence of static visual shortcuts. To remedy this issue, we present VITATECS, a diagnostic VIdeo-Text dAtaset for the evaluation of TEmporal Concept underStanding. Specifically, we first introduce a fine-grained taxonomy of temporal concepts in natural language in order to diagnose the capability of VidLMs to comprehend different temporal aspects. Furthermore, to disentangle the correlation between static and temporal information, we generate counterfactual video descriptions that differ from the original one only in the specified temporal aspect. We employ a semi-automatic data collection framework using large language models and human-in-the-loop annotation to obtain high-quality counterfactual descriptions efficiently. Evaluation of representative video-language understanding models confirms their deficiency in temporal understanding, revealing the need for greater emphasis on the temporal elements in video-language research.

{{</citation>}}


### (84/155) Generalized Large-Scale Data Condensation via Various Backbone and Statistical Matching (Shitong Shao et al., 2023)

{{<citation>}}

Shitong Shao, Zeyuan Yin, Muxin Zhou, Xindong Zhang, Zhiqiang Shen. (2023)  
**Generalized Large-Scale Data Condensation via Various Backbone and Statistical Matching**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.17950v1)  

---


**ABSTRACT**  
The lightweight "local-match-global" matching introduced by SRe2L successfully creates a distilled dataset with comprehensive information on the full 224x224 ImageNet-1k. However, this one-sided approach is limited to a particular backbone, layer, and statistics, which limits the improvement of the generalization of a distilled dataset. We suggest that sufficient and various "local-match-global" matching are more precise and effective than a single one and has the ability to create a distilled dataset with richer information and better generalization. We call this perspective "generalized matching" and propose Generalized Various Backbone and Statistical Matching (G-VBSM) in this work, which aims to create a synthetic dataset with densities, ensuring consistency with the complete dataset across various backbones, layers, and statistics. As experimentally demonstrated, G-VBSM is the first algorithm to obtain strong performance across both small-scale and large-scale datasets. Specifically, G-VBSM achieves a performance of 38.7% on CIFAR-100 with 128-width ConvNet, 47.6% on Tiny-ImageNet with ResNet18, and 31.4% on the full 224x224 ImageNet-1k with ResNet18, under images per class (IPC) 10, 50, and 10, respectively. These results surpass all SOTA methods by margins of 3.9%, 6.5%, and 10.1%, respectively.

{{</citation>}}


### (85/155) Zero-shot Retrieval: Augmenting Pre-trained Models with Search Engines (Hamed Damirchi et al., 2023)

{{<citation>}}

Hamed Damirchi, Cristian Rodríguez-Opazo, Ehsan Abbasnejad, Damien Teney, Javen Qinfeng Shi, Stephen Gould, Anton van den Hengel. (2023)  
**Zero-shot Retrieval: Augmenting Pre-trained Models with Search Engines**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.17949v1)  

---


**ABSTRACT**  
Large pre-trained models can dramatically reduce the amount of task-specific data required to solve a problem, but they often fail to capture domain-specific nuances out of the box. The Web likely contains the information necessary to excel on any specific application, but identifying the right data a priori is challenging. This paper shows how to leverage recent advances in NLP and multi-modal learning to augment a pre-trained model with search engine retrieval. We propose to retrieve useful data from the Web at test time based on test cases that the model is uncertain about. Different from existing retrieval-augmented approaches, we then update the model to address this underlying uncertainty. We demonstrate substantial improvements in zero-shot performance, e.g. a remarkable increase of 15 percentage points in accuracy on the Stanford Cars and Flowers datasets. We also present extensive experiments that explore the impact of noisy retrieval and different learning strategies.

{{</citation>}}


### (86/155) Generative Hierarchical Temporal Transformer for Hand Action Recognition and Motion Prediction (Yilin Wen et al., 2023)

{{<citation>}}

Yilin Wen, Hao Pan, Takehiko Ohkawa, Lei Yang, Jia Pan, Yoichi Sato, Taku Komura, Wenping Wang. (2023)  
**Generative Hierarchical Temporal Transformer for Hand Action Recognition and Motion Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.17366v1)  

---


**ABSTRACT**  
We present a novel framework that concurrently tackles hand action recognition and 3D future hand motion prediction. While previous works focus on either recognition or prediction, we propose a generative Transformer VAE architecture to jointly capture both aspects, facilitating realistic motion prediction by leveraging the short-term hand motion and long-term action consistency observed across timestamps.To ensure faithful representation of the semantic dependency and different temporal granularity of hand pose and action, our framework is decomposed into two cascaded VAE blocks. The lower pose block models short-span poses, while the upper action block models long-span action. These are connected by a mid-level feature that represents sub-second series of hand poses.Our framework is trained across multiple datasets, where pose and action blocks are trained separately to fully utilize pose-action annotations of different qualities. Evaluations show that on multiple datasets, the joint modeling of recognition and prediction improves over separate solutions, and the semantic and temporal hierarchy enables long-term pose and action modeling.

{{</citation>}}


### (87/155) Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning (Xiaoqian Wu et al., 2023)

{{<citation>}}

Xiaoqian Wu, Yong-Lu Li, Jianhua Sun, Cewu Lu. (2023)  
**Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.17365v1)  

---


**ABSTRACT**  
Human reasoning can be understood as a cooperation between the intuitive, associative "System-1" and the deliberative, logical "System-2". For existing System-1-like methods in visual activity understanding, it is crucial to integrate System-2 processing to improve explainability, generalization, and data efficiency. One possible path of activity reasoning is building a symbolic system composed of symbols and rules, where one rule connects multiple symbols, implying human knowledge and reasoning abilities. Previous methods have made progress, but are defective with limited symbols from handcraft and limited rules from visual-based annotations, failing to cover the complex patterns of activities and lacking compositional generalization. To overcome the defects, we propose a new symbolic system with two ideal important properties: broad-coverage symbols and rational rules. Collecting massive human knowledge via manual annotations is expensive to instantiate this symbolic system. Instead, we leverage the recent advancement of LLMs (Large Language Models) as an approximation of the two ideal properties, i.e., Symbols from Large Language Models (Symbol-LLM). Then, given an image, visual contents from the images are extracted and checked as symbols and activity semantics are reasoned out based on rules via fuzzy logic calculation. Our method shows superiority in extensive activity understanding tasks. Code and data are available at https://mvig-rhos.com/symbol_llm.

{{</citation>}}


### (88/155) How does spatial structure affect psychological restoration? A method based on Graph Neural Networks and Street View Imagery (Haoran Ma et al., 2023)

{{<citation>}}

Haoran Ma, Yan Zhang, Pengyuan Liu, Fan Zhang, Pengyu Zhu. (2023)  
**How does spatial structure affect psychological restoration? A method based on Graph Neural Networks and Street View Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.17361v2)  

---


**ABSTRACT**  
The Attention Restoration Theory (ART) presents a theoretical framework with four essential indicators (being away, extent, fascinating, and compatibility) for comprehending urban and natural restoration quality. However, previous studies relied on non-sequential data and non-spatial dependent methods, which overlooks the impact of spatial structure defined here as the positional relationships between scene entities on restoration quality. The past methods also make it challenging to measure restoration quality on an urban scale. In this work, a spatial-dependent graph neural networks (GNNs) approach is proposed to reveal the relation between spatial structure and restoration quality on an urban scale. Specifically, we constructed two different types of graphs at the street and city levels. The street-level graphs, using sequential street view images (SVIs) of road segments to capture position relationships between entities, were used to represent spatial structure. The city-level graph, modeling the topological relationships of roads as non-Euclidean data structures and embedding urban features (including Perception-features, Spatial-features, and Socioeconomic-features), was used to measure restoration quality. The results demonstrate that: 1) spatial-dependent GNNs model outperforms traditional methods (Acc = 0.735, F1 = 0.732); 2) spatial structure portrayed through sequential SVIs data significantly influences restoration quality; 3) spaces with the same restoration quality exhibited distinct spatial structures patterns. This study clarifies the association between spatial structure and restoration quality, providing a new perspective to improve urban well-being in the future.

{{</citation>}}


### (89/155) A natural language processing-based approach: mapping human perception by understanding deep semantic features in street view images (Haoran Ma et al., 2023)

{{<citation>}}

Haoran Ma, Dongdong Wu. (2023)  
**A natural language processing-based approach: mapping human perception by understanding deep semantic features in street view images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2311.17354v1)  

---


**ABSTRACT**  
In the past decade, using Street View images and machine learning to measure human perception has become a mainstream research approach in urban science. However, this approach using only image-shallow information makes it difficult to comprehensively understand the deep semantic features of human perception of a scene. In this study, we proposed a new framework based on a pre-train natural language model to understand the relationship between human perception and the sense of a scene. Firstly, Place Pulse 2.0 was used as our base dataset, which contains a variety of human-perceived labels, namely, beautiful, safe, wealthy, depressing, boring, and lively. An image captioning network was used to extract the description information of each street view image. Secondly, a pre-trained BERT model was finetuning and added a regression function for six human perceptual dimensions. Furthermore, we compared the performance of five traditional regression methods with our approach and conducted a migration experiment in Hong Kong. Our results show that human perception scoring by deep semantic features performed better than previous studies by machine learning methods with shallow features. The use of deep scene semantic features provides new ideas for subsequent human perception research, as well as better explanatory power in the face of spatial heterogeneity.

{{</citation>}}


### (90/155) LEAP: LLM-Generation of Egocentric Action Programs (Eadom Dessalene et al., 2023)

{{<citation>}}

Eadom Dessalene, Michael Maynord, Cornelia Fermüller, Yiannis Aloimonos. (2023)  
**LEAP: LLM-Generation of Egocentric Action Programs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00055v1)  

---


**ABSTRACT**  
We introduce LEAP (illustrated in Figure 1), a novel method for generating video-grounded action programs through use of a Large Language Model (LLM). These action programs represent the motoric, perceptual, and structural aspects of action, and consist of sub-actions, pre- and post-conditions, and control flows. LEAP's action programs are centered on egocentric video and employ recent developments in LLMs both as a source for program knowledge and as an aggregator and assessor of multimodal video information. We apply LEAP over a majority (87\%) of the training set of the EPIC Kitchens dataset, and release the resulting action programs as a publicly available dataset here (https://drive.google.com/drive/folders/1Cpkw_TI1IIxXdzor0pOXG3rWJWuKU5Ex?usp=drive_link). We employ LEAP as a secondary source of supervision, using its action programs in a loss term applied to action recognition and anticipation networks. We demonstrate sizable improvements in performance in both tasks due to training with the LEAP dataset. Our method achieves 1st place on the EPIC Kitchens Action Recognition leaderboard as of November 17 among the networks restricted to RGB-input (see Supplementary Materials).

{{</citation>}}


### (91/155) DreamSync: Aligning Text-to-Image Generation with Image Understanding Feedback (Jiao Sun et al., 2023)

{{<citation>}}

Jiao Sun, Deqing Fu, Yushi Hu, Su Wang, Royi Rassin, Da-Cheng Juan, Dana Alon, Charles Herrmann, Sjoerd van Steenkiste, Ranjay Krishna, Cyrus Rashtchian. (2023)  
**DreamSync: Aligning Text-to-Image Generation with Image Understanding Feedback**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2311.17946v1)  

---


**ABSTRACT**  
Despite their wide-spread success, Text-to-Image models (T2I) still struggle to produce images that are both aesthetically pleasing and faithful to the user's input text. We introduce DreamSync, a model-agnostic training algorithm by design that improves T2I models to be faithful to the text input. DreamSync builds off a recent insight from TIFA's evaluation framework -- that large vision-language models (VLMs) can effectively identify the fine-grained discrepancies between generated images and the text inputs. DreamSync uses this insight to train T2I models without any labeled data; it improves T2I models using its own generations. First, it prompts the model to generate several candidate images for a given input text. Then, it uses two VLMs to select the best generation: a Visual Question Answering model that measures the alignment of generated images to the text, and another that measures the generation's aesthetic quality. After selection, we use LoRA to iteratively finetune the T2I model to guide its generation towards the selected best generations. DreamSync does not need any additional human annotation. model architecture changes, or reinforcement learning. Despite its simplicity, DreamSync improves both the semantic alignment and aesthetic appeal of two diffusion-based T2I models, evidenced by multiple benchmarks (+1.7% on TIFA, +2.9% on DSG1K, +3.4% on VILA aesthetic) and human evaluation.

{{</citation>}}


### (92/155) VideoAssembler: Identity-Consistent Video Generation with Reference Entities using Diffusion Model (Haoyu Zhao et al., 2023)

{{<citation>}}

Haoyu Zhao, Tianyi Lu, Jiaxi Gu, Xing Zhang, Zuxuan Wu, Hang Xu, Yu-Gang Jiang. (2023)  
**VideoAssembler: Identity-Consistent Video Generation with Reference Entities using Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.17338v2)  

---


**ABSTRACT**  
Identity-consistent video generation seeks to synthesize videos that are guided by both textual prompts and reference images of entities. Current approaches typically utilize cross-attention layers to integrate the appearance of the entity, which predominantly captures semantic attributes, resulting in compromised fidelity of entities. Moreover, these methods necessitate iterative fine-tuning for each new entity encountered, thereby limiting their applicability. To address these challenges, we introduce VideoAssembler, a novel end-to-end framework for identity-consistent video generation that can conduct inference directly when encountering new entities. VideoAssembler is adept at producing videos that are not only flexible with respect to the input reference entities but also responsive to textual conditions. Additionally, by modulating the quantity of input images for the entity, VideoAssembler enables the execution of tasks ranging from image-to-video generation to sophisticated video editing. VideoAssembler comprises two principal components: the Reference Entity Pyramid (REP) encoder and the Entity-Prompt Attention Fusion (EPAF) module. The REP encoder is designed to infuse comprehensive appearance details into the denoising stages of the stable diffusion model. Concurrently, the EPAF module is utilized to integrate text-aligned features effectively. Furthermore, to mitigate the challenge of scarce data, we present a methodology for the preprocessing of training data. Our evaluation of the VideoAssembler framework on the UCF-101, MSR-VTT, and DAVIS datasets indicates that it achieves good performances in both quantitative and qualitative analyses (346.84 in FVD and 48.01 in IS on UCF-101). Our project page is at https://gulucaptain.github.io/videoassembler/.

{{</citation>}}


### (93/155) Contrastive Vision-Language Alignment Makes Efficient Instruction Learner (Lizhao Liu et al., 2023)

{{<citation>}}

Lizhao Liu, Xinyu Sun, Tianhang Xiang, Zhuangwei Zhuang, Liuren Yin, Mingkui Tan. (2023)  
**Contrastive Vision-Language Alignment Makes Efficient Instruction Learner**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.17945v1)  

---


**ABSTRACT**  
We study the task of extending the large language model (LLM) into a vision-language instruction-following model. This task is crucial but challenging since the LLM is trained on text modality only, making it hard to effectively digest the visual modality. To address this, existing methods typically train a visual adapter to align the representation between a pre-trained vision transformer (ViT) and the LLM by a generative image captioning loss. However, we find that the generative objective can only produce weak alignment for vision and language, making the aligned vision-language model very hungry for the instruction fine-tuning data. In this paper, we propose CG-VLM that applies both Contrastive and Generative alignment objectives to effectively align the representation of ViT and LLM. Different from image level and sentence level alignment in common contrastive learning settings, CG-VLM aligns the image-patch level features and text-token level embeddings, which, however, is very hard to achieve as no explicit grounding patch-token relation provided in standard image captioning datasets. To address this issue, we propose to maximize the averaged similarity between pooled image-patch features and text-token embeddings. Extensive experiments demonstrate that the proposed CG-VLM produces strong vision-language alignment and is an efficient instruction learner. For example, using only 10% instruction tuning data, we reach 95% performance of state-of-the-art method LLaVA [29] on the zero-shot ScienceQA-Image benchmark.

{{</citation>}}


### (94/155) eMotions: A Large-Scale Dataset for Emotion Recognition in Short Videos (Xuecheng Wu et al., 2023)

{{<citation>}}

Xuecheng Wu, Heli Sun, Junxiao Xue, Ruofan Zhai, Xiangyan Kong, Jiayu Nie, Liang He. (2023)  
**eMotions: A Large-Scale Dataset for Emotion Recognition in Short Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2311.17335v1)  

---


**ABSTRACT**  
Nowadays, short videos (SVs) are essential to information acquisition and sharing in our life. The prevailing use of SVs to spread emotions leads to the necessity of emotion recognition in SVs. Considering the lack of SVs emotion data, we introduce a large-scale dataset named eMotions, comprising 27,996 videos. Meanwhile, we alleviate the impact of subjectivities on labeling quality by emphasizing better personnel allocations and multi-stage annotations. In addition, we provide the category-balanced and test-oriented variants through targeted data sampling. Some commonly used videos (e.g., facial expressions and postures) have been well studied. However, it is still challenging to understand the emotions in SVs. Since the enhanced content diversity brings more distinct semantic gaps and difficulties in learning emotion-related features, and there exists information gaps caused by the emotion incompleteness under the prevalently audio-visual co-expressions. To tackle these problems, we present an end-to-end baseline method AV-CPNet that employs the video transformer to better learn semantically relevant representations. We further design the two-stage cross-modal fusion module to complementarily model the correlations of audio-visual features. The EP-CE Loss, incorporating three emotion polarities, is then applied to guide model optimization. Extensive experimental results on nine datasets verify the effectiveness of AV-CPNet. Datasets and code will be open on https://github.com/XuecWu/eMotions.

{{</citation>}}


### (95/155) Towards Top-Down Reasoning: An Explainable Multi-Agent Approach for Visual Question Answering (Zeqing Wang et al., 2023)

{{<citation>}}

Zeqing Wang, Wentao Wan, Runmeng Chen, Qiqing Lao, Minjie Lang, Keze Wang. (2023)  
**Towards Top-Down Reasoning: An Explainable Multi-Agent Approach for Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.17331v1)  

---


**ABSTRACT**  
Recently, Vision Language Models (VLMs) have gained significant attention, exhibiting notable advancements across various tasks by leveraging extensive image-text paired data. However, prevailing VLMs often treat Visual Question Answering (VQA) as perception tasks, employing black-box models that overlook explicit modeling of relationships between different questions within the same visual scene. Moreover, the existing VQA methods that rely on Knowledge Bases (KBs) might frequently encounter biases from limited data and face challenges in relevant information indexing. Attempt to overcome these limitations, this paper introduces an explainable multi-agent collaboration framework by tapping into knowledge embedded in Large Language Models (LLMs) trained on extensive corpora. Inspired by human cognition, our framework uncovers latent information within the given question by employing three agents, i.e., Seeker, Responder, and Integrator, to perform a top-down reasoning process. The Seeker agent generates relevant issues related to the original question. The Responder agent, based on VLM, handles simple VQA tasks and provides candidate answers. The Integrator agent combines information from the Seeker agent and the Responder agent to produce the final VQA answer. Through the above collaboration mechanism, our framework explicitly constructs a multi-view knowledge base for a specific image scene, reasoning answers in a top-down processing manner. We extensively evaluate our method on diverse VQA datasets and VLMs, demonstrating its broad applicability and interpretability with comprehensive experimental results.

{{</citation>}}


### (96/155) LALM: Long-Term Action Anticipation with Language Models (Sanghwan Kim et al., 2023)

{{<citation>}}

Sanghwan Kim, Daoji Huang, Yongqin Xian, Otmar Hilliges, Luc Van Gool, Xi Wang. (2023)  
**LALM: Long-Term Action Anticipation with Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17944v1)  

---


**ABSTRACT**  
Understanding human activity is a crucial yet intricate task in egocentric vision, a field that focuses on capturing visual perspectives from the camera wearer's viewpoint. While traditional methods heavily rely on representation learning trained on extensive video data, there exists a significant limitation: obtaining effective video representations proves challenging due to the inherent complexity and variability in human activities.Furthermore, exclusive dependence on video-based learning may constrain a model's capability to generalize across long-tail classes and out-of-distribution scenarios.   In this study, we introduce a novel approach for long-term action anticipation using language models (LALM), adept at addressing the complex challenges of long-term activity understanding without the need for extensive training. Our method incorporates an action recognition model to track previous action sequences and a vision-language model to articulate relevant environmental details. By leveraging the context provided by these past events, we devise a prompting strategy for action anticipation using large language models (LLMs). Moreover, we implement Maximal Marginal Relevance for example selection to facilitate in-context learning of the LLMs. Our experimental results demonstrate that LALM surpasses the state-of-the-art methods in the task of long-term action anticipation on the Ego4D benchmark. We further validate LALM on two additional benchmarks, affirming its capacity for generalization across intricate activities with different sets of taxonomies. These are achieved without specific fine-tuning.

{{</citation>}}


### (97/155) Explaining CLIP's performance disparities on data from blind/low vision users (Daniela Massiceti et al., 2023)

{{<citation>}}

Daniela Massiceti, Camilla Longden, Agnieszka Słowik, Samuel Wills, Martin Grayson, Cecily Morrison. (2023)  
**Explaining CLIP's performance disparities on data from blind/low vision users**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17315v2)  

---


**ABSTRACT**  
Large multi-modal models (LMMs) hold the potential to usher in a new era of automated visual assistance for people who are blind or low vision (BLV). Yet, these models have not been systematically evaluated on data captured by BLV users. We address this by empirically assessing CLIP, a widely-used LMM likely to underpin many assistive technologies. Testing 25 CLIP variants in a zero-shot classification task, we find that their accuracy is 15 percentage points lower on average for images captured by BLV users than web-crawled images. This disparity stems from CLIP's sensitivities to 1) image content (e.g. not recognizing disability objects as well as other objects); 2) image quality (e.g. not being robust to lighting variation); and 3) text content (e.g. not recognizing objects described by tactile adjectives as well as visual ones). We delve deeper with a textual analysis of three common pre-training datasets: LAION-400M, LAION-2B and DataComp-1B, showing that disability content is rarely mentioned. We then provide three examples that illustrate how the performance disparities extend to three downstream models underpinned by CLIP: OWL-ViT, CLIPSeg and DALL-E2. We find that few-shot learning with as few as 5 images can mitigate CLIP's quality-of-service disparities for BLV users in some scenarios, which we discuss alongside a set of other possible mitigations.

{{</citation>}}


### (98/155) LEOD: Label-Efficient Object Detection for Event Cameras (Ziyi Wu et al., 2023)

{{<citation>}}

Ziyi Wu, Mathias Gehrig, Qing Lyu, Xudong Liu, Igor Gilitschenski. (2023)  
**LEOD: Label-Efficient Object Detection for Event Cameras**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.17286v1)  

---


**ABSTRACT**  
Object detection with event cameras enjoys the property of low latency and high dynamic range, making it suitable for safety-critical scenarios such as self-driving. However, labeling event streams with high temporal resolutions for supervised training is costly. We address this issue with LEOD, the first framework for label-efficient event-based detection. Our method unifies weakly- and semi-supervised object detection with a self-training mechanism. We first utilize a detector pre-trained on limited labels to produce pseudo ground truth on unlabeled events, and then re-train the detector with both real and generated labels. Leveraging the temporal consistency of events, we run bi-directional inference and apply tracking-based post-processing to enhance the quality of pseudo labels. To stabilize training, we further design a soft anchor assignment strategy to mitigate the noise in labels. We introduce new experimental protocols to evaluate the task of label-efficient event-based detection on Gen1 and 1Mpx datasets. LEOD consistently outperforms supervised baselines across various labeling ratios. For example, on Gen1, it improves mAP by 8.6% and 7.8% for RVT-S trained with 1% and 2% labels. On 1Mpx, RVT-S with 10% labels even surpasses its fully-supervised counterpart using 100% labels. LEOD maintains its effectiveness even when all labeled data are available, reaching new state-of-the-art results. Finally, we show that our method readily scales to improve larger detectors as well.

{{</citation>}}


## cs.GT (1)



### (99/155) Algorithmic Persuasion Through Simulation: Information Design in the Age of Generative AI (Keegan Harris et al., 2023)

{{<citation>}}

Keegan Harris, Nicole Immorlica, Brendan Lucier, Aleksandrs Slivkins. (2023)  
**Algorithmic Persuasion Through Simulation: Information Design in the Age of Generative AI**  

---
Primary Category: cs.GT  
Categories: cs-AI, cs-GT, cs.GT, econ-TH  
Keywords: AI, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18138v1)  

---


**ABSTRACT**  
How can an informed sender persuade a receiver, having only limited information about the receiver's beliefs? Motivated by research showing generative AI can simulate economic agents, we initiate the study of information design with an oracle. We assume the sender can learn more about the receiver by querying this oracle, e.g., by simulating the receiver's behavior. Aside from AI motivations such as general-purpose Large Language Models (LLMs) and problem-specific machine learning models, alternate motivations include customer surveys and querying a small pool of live users.   Specifically, we study Bayesian Persuasion where the sender has a second-order prior over the receiver's beliefs. After a fixed number of queries to an oracle to refine this prior, the sender commits to an information structure. Upon receiving the message, the receiver takes a payoff-relevant action maximizing her expected utility given her posterior beliefs. We design polynomial-time querying algorithms that optimize the sender's expected utility in this Bayesian Persuasion game. As a technical contribution, we show that queries form partitions of the space of receiver beliefs that can be used to quantify the sender's knowledge.

{{</citation>}}


## cs.LG (25)



### (100/155) Mixed-Precision Quantization for Federated Learning on Resource-Constrained Heterogeneous Devices (Huancheng Chen et al., 2023)

{{<citation>}}

Huancheng Chen, Haris Vikalo. (2023)  
**Mixed-Precision Quantization for Federated Learning on Resource-Constrained Heterogeneous Devices**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.18129v1)  

---


**ABSTRACT**  
While federated learning (FL) systems often utilize quantization to battle communication and computational bottlenecks, they have heretofore been limited to deploying fixed-precision quantization schemes. Meanwhile, the concept of mixed-precision quantization (MPQ), where different layers of a deep learning model are assigned varying bit-width, remains unexplored in the FL settings. We present a novel FL algorithm, FedMPQ, which introduces mixed-precision quantization to resource-heterogeneous FL systems. Specifically, local models, quantized so as to satisfy bit-width constraint, are trained by optimizing an objective function that includes a regularization term which promotes reduction of precision in some of the layers without significant performance degradation. The server collects local model updates, de-quantizes them into full-precision models, and then aggregates them into a global model. To initialize the next round of local training, the server relies on the information learned in the previous training round to customize bit-width assignments of the models delivered to different clients. In extensive benchmarking experiments on several model architectures and different datasets in both iid and non-iid settings, FedMPQ outperformed the baseline FL schemes that utilize fixed-precision quantization while incurring only a minor computational overhead on the participating devices.

{{</citation>}}


### (101/155) The perpetual motion machine of AI-generated data and the distraction of ChatGPT-as-scientist (Jennifer Listgarten, 2023)

{{<citation>}}

Jennifer Listgarten. (2023)  
**The perpetual motion machine of AI-generated data and the distraction of ChatGPT-as-scientist**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.00818v1)  

---


**ABSTRACT**  
Since ChatGPT works so well, are we on the cusp of solving science with AI? Is not AlphaFold2 suggestive that the potential of LLMs in biology and the sciences more broadly is limitless? Can we use AI itself to bridge the lack of data in the sciences in order to then train an AI? Herein we present a discussion of these topics.

{{</citation>}}


### (102/155) The Forecastability of Underlying Building Electricity Demand from Time Series Data (Mohamad Khalil et al., 2023)

{{<citation>}}

Mohamad Khalil, A. Stephen McGough, Hussain Kazmi, Sara Walker. (2023)  
**The Forecastability of Underlying Building Electricity Demand from Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.18078v1)  

---


**ABSTRACT**  
Forecasting building energy consumption has become a promising solution in Building Energy Management Systems for energy saving and optimization. Furthermore, it can play an important role in the efficient management of the operation of a smart grid. Different data-driven approaches to forecast the future energy demand of buildings at different scale, and over various time horizons, can be found in the scientific literature, including extensive Machine Learning and Deep Learning approaches. However, the identification of the most accurate forecaster model which can be utilized to predict the energy demand of such a building is still challenging.In this paper, the design and implementation of a data-driven approach to predict how forecastable the future energy demand of a building is, without first utilizing a data-driven forecasting model, is presented. The investigation utilizes a historical electricity consumption time series data set with a half-hour interval that has been collected from a group of residential buildings located in the City of London, United Kingdom

{{</citation>}}


### (103/155) Self-Supervised Learning for Large-Scale Preventive Security Constrained DC Optimal Power Flow (Seonho Park et al., 2023)

{{<citation>}}

Seonho Park, Pascal Van Hentenryck. (2023)  
**Self-Supervised Learning for Large-Scale Preventive Security Constrained DC Optimal Power Flow**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC  
Keywords: Security, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.18072v1)  

---


**ABSTRACT**  
Security-Constrained Optimal Power Flow (SCOPF) plays a crucial role in power grid stability but becomes increasingly complex as systems grow. This paper introduces PDL-SCOPF, a self-supervised end-to-end primal-dual learning framework for producing near-optimal solutions to large-scale SCOPF problems in milliseconds. Indeed, PDL-SCOPF remedies the limitations of supervised counterparts that rely on training instances with their optimal solutions, which becomes impractical for large-scale SCOPF problems. PDL-SCOPF mimics an Augmented Lagrangian Method (ALM) for training primal and dual networks that learn the primal solutions and the Lagrangian multipliers, respectively, to the unconstrained optimizations. In addition, PDL-SCOPF incorporates a repair layer to ensure the feasibility of the power balance in the nominal case, and a binary search layer to compute, using the Automatic Primary Response (APR), the generator dispatches in the contingencies. The resulting differentiable program can then be trained end-to-end using the objective function of the SCOPF and the power balance constraints of the contingencies. Experimental results demonstrate that the PDL-SCOPF delivers accurate feasible solutions with minimal optimality gaps. The framework underlying PDL-SCOPF aims at bridging the gap between traditional optimization methods and machine learning, highlighting the potential of self-supervised end-to-end primal-dual learning for large-scale optimization tasks.

{{</citation>}}


### (104/155) Understanding Your Agent: Leveraging Large Language Models for Behavior Explanation (Xijia Zhang et al., 2023)

{{<citation>}}

Xijia Zhang, Yue Guo, Simon Stepputtis, Katia Sycara, Joseph Campbell. (2023)  
**Understanding Your Agent: Leveraging Large Language Models for Behavior Explanation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.18062v1)  

---


**ABSTRACT**  
Intelligent agents such as robots are increasingly deployed in real-world, safety-critical settings. It is vital that these agents are able to explain the reasoning behind their decisions to human counterparts; however, their behavior is often produced by uninterpretable models such as deep neural networks. We propose an approach to generate natural language explanations for an agent's behavior based only on observations of states and actions, thus making our method independent from the underlying model's representation. For such models, we first learn a behavior representation and subsequently use it to produce plausible explanations with minimal hallucination while affording user interaction with a pre-trained large language model. We evaluate our method in a multi-agent search-and-rescue environment and demonstrate the effectiveness of our explanations for agents executing various behaviors. Through user studies and empirical experiments, we show that our approach generates explanations as helpful as those produced by a human domain expert while enabling beneficial interactions such as clarification and counterfactual queries.

{{</citation>}}


### (105/155) TransNAS-TSAD: Harnessing Transformers for Multi-Objective Neural Architecture Search in Time Series Anomaly Detection (Ijaz Ul Haq et al., 2023)

{{<citation>}}

Ijaz Ul Haq, Byung Suk Lee. (2023)  
**TransNAS-TSAD: Harnessing Transformers for Multi-Objective Neural Architecture Search in Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: Anomaly Detection, Time Series, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.18061v2)  

---


**ABSTRACT**  
The surge in real-time data collection across various industries has underscored the need for advanced anomaly detection in both univariate and multivariate time series data. Traditional methods, while comprehensive, often struggle to capture the complex interdependencies in such data. This paper introduces TransNAS-TSAD, a novel framework that synergizes transformer architecture with neural architecture search (NAS), enhanced through NSGA-II algorithm optimization. This innovative approach effectively tackles the complexities of both univariate and multivariate time series, balancing computational efficiency with detection accuracy. Our evaluation reveals that TransNAS-TSAD surpasses conventional anomaly detection models, demonstrating marked improvements in diverse data scenarios. We also propose the Efficiency-Accuracy-Complexity Score (EACS) as a new metric for assessing model performance, emphasizing the crucial balance between accuracy and computational resources. TransNAS-TSAD sets a new benchmark in time series anomaly detection, offering a versatile, efficient solution for complex real-world applications. This research paves the way for future developments in the field, highlighting its potential in a wide range of industry applications.

{{</citation>}}


### (106/155) TransOpt: Transformer-based Representation Learning for Optimization Problem Classification (Gjorgjina Cenikj et al., 2023)

{{<citation>}}

Gjorgjina Cenikj, Gašper Petelin, Tome Eftimov. (2023)  
**TransOpt: Transformer-based Representation Learning for Optimization Problem Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC  
Keywords: Representation Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2311.18035v1)  

---


**ABSTRACT**  
We propose a representation of optimization problem instances using a transformer-based neural network architecture trained for the task of problem classification of the 24 problem classes from the Black-box Optimization Benchmarking (BBOB) benchmark. We show that transformer-based methods can be trained to recognize problem classes with accuracies in the range of 70\%-80\% for different problem dimensions, suggesting the possible application of transformer architectures in acquiring representations for black-box optimization problems.

{{</citation>}}


### (107/155) A Bag of Receptive Fields for Time Series Extrinsic Predictions (Francesco Spinnato et al., 2023)

{{<citation>}}

Francesco Spinnato, Riccardo Guidotti, Anna Monreale, Mirco Nanni. (2023)  
**A Bag of Receptive Fields for Time Series Extrinsic Predictions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.18029v1)  

---


**ABSTRACT**  
High-dimensional time series data poses challenges due to its dynamic nature, varying lengths, and presence of missing values. This kind of data requires extensive preprocessing, limiting the applicability of existing Time Series Classification and Time Series Extrinsic Regression techniques. For this reason, we propose BORF, a Bag-Of-Receptive-Fields model, which incorporates notions from time series convolution and 1D-SAX to handle univariate and multivariate time series with varying lengths and missing values. We evaluate BORF on Time Series Classification and Time Series Extrinsic Regression tasks using the full UEA and UCR repositories, demonstrating its competitive performance against state-of-the-art methods. Finally, we outline how this representation can naturally provide saliency and feature-based explanations.

{{</citation>}}


### (108/155) TimelyGPT: Recurrent Convolutional Transformer for Long Time-series Representation (Ziyang Song et al., 2023)

{{<citation>}}

Ziyang Song, Qincheng Lu, Hao Xu, Yue Li. (2023)  
**TimelyGPT: Recurrent Convolutional Transformer for Long Time-series Representation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Computer Vision, GPT, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2312.00817v1)  

---


**ABSTRACT**  
Pre-trained models (PTMs) have gained prominence in Natural Language Processing and Computer Vision domains. When it comes to time-series PTMs, their development has been limited. Previous research on time-series transformers has mainly been devoted to small-scale tasks, yet these models have not consistently outperformed traditional models. Additionally, the performance of these transformers on large-scale data remains unexplored. These findings raise doubts about Transformer's capabilities to scale up and capture temporal dependencies. In this study, we re-examine time-series transformers and identify the shortcomings of prior studies. Drawing from these insights, we then introduce a pioneering architecture called Timely Generative Pre-trained Transformer (\model). This architecture integrates recurrent attention and temporal convolution modules to effectively capture global-local temporal dependencies in long sequences. The relative position embedding with time decay can effectively deal with trend and periodic patterns from time-series. Our experiments show that \model~excels in modeling continuously monitored biosignal as well as irregularly-sampled time-series data commonly observed in longitudinal electronic health records. This breakthrough suggests a priority shift in time-series deep learning research, moving from small-scale modeling from scratch to large-scale pre-training.

{{</citation>}}


### (109/155) SAIBench: A Structural Interpretation of AI for Science Through Benchmarks (Yatao Li et al., 2023)

{{<citation>}}

Yatao Li, Jianfeng Zhan. (2023)  
**SAIBench: A Structural Interpretation of AI for Science Through Benchmarks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17869v1)  

---


**ABSTRACT**  
Artificial Intelligence for Science (AI4S) is an emerging research field that utilizes machine learning advancements to tackle complex scientific computational issues, aiming to enhance computational efficiency and accuracy. However, the data-driven nature of AI4S lacks the correctness or accuracy assurances of conventional scientific computing, posing challenges when deploying AI4S models in real-world applications. To mitigate these, more comprehensive benchmarking procedures are needed to better understand AI4S models. This paper introduces a novel benchmarking approach, known as structural interpretation, which addresses two key requirements: identifying the trusted operating range in the problem space and tracing errors back to their computational components. This method partitions both the problem and metric spaces, facilitating a structural exploration of these spaces. The practical utility and effectiveness of structural interpretation are illustrated through its application to three distinct AI4S workloads: machine-learning force fields (MLFF), jet tagging, and precipitation nowcasting. The benchmarks effectively model the trusted operating range, trace errors, and reveal novel perspectives for refining the model, training process, and data sampling strategy. This work is part of the SAIBench project, an AI4S benchmarking suite.

{{</citation>}}


### (110/155) Maximum Entropy Model Correction in Reinforcement Learning (Amin Rakhsha et al., 2023)

{{<citation>}}

Amin Rakhsha, Mete Kemertas, Mohammad Ghavamzadeh, Amir-massoud Farahmand. (2023)  
**Maximum Entropy Model Correction in Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY, math-OC, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.17855v1)  

---


**ABSTRACT**  
We propose and theoretically analyze an approach for planning with an approximate model in reinforcement learning that can reduce the adverse impact of model error. If the model is accurate enough, it accelerates the convergence to the true value function too. One of its key components is the MaxEnt Model Correction (MoCo) procedure that corrects the model's next-state distributions based on a Maximum Entropy density estimation formulation. Based on MoCo, we introduce the Model Correcting Value Iteration (MoCoVI) algorithm, and its sampled-based variant MoCoDyna. We show that MoCoVI and MoCoDyna's convergence can be much faster than the conventional model-free algorithms. Unlike traditional model-based algorithms, MoCoVI and MoCoDyna effectively utilize an approximate model and still converge to the correct value function.

{{</citation>}}


### (111/155) On the Adversarial Robustness of Graph Contrastive Learning Methods (Filippo Guerranti et al., 2023)

{{<citation>}}

Filippo Guerranti, Zinuo Yi, Anna Starovoit, Rafiq Kamel, Simon Geisler, Stephan Günnemann. (2023)  
**On the Adversarial Robustness of Graph Contrastive Learning Methods**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.17853v2)  

---


**ABSTRACT**  
Contrastive learning (CL) has emerged as a powerful framework for learning representations of images and text in a self-supervised manner while enhancing model robustness against adversarial attacks. More recently, researchers have extended the principles of contrastive learning to graph-structured data, giving birth to the field of graph contrastive learning (GCL). However, whether GCL methods can deliver the same advantages in adversarial robustness as their counterparts in the image and text domains remains an open question. In this paper, we introduce a comprehensive robustness evaluation protocol tailored to assess the robustness of GCL models. We subject these models to adaptive adversarial attacks targeting the graph structure, specifically in the evasion scenario. We evaluate node and graph classification tasks using diverse real-world datasets and attack strategies. With our work, we aim to offer insights into the robustness of GCL methods and hope to open avenues for potential future research directions.

{{</citation>}}


### (112/155) Propagate & Distill: Towards Effective Graph Learners Using Propagation-Embracing MLPs (Yong-Min Shin et al., 2023)

{{<citation>}}

Yong-Min Shin, Won-Yong Shin. (2023)  
**Propagate & Distill: Towards Effective Graph Learners Using Propagation-Embracing MLPs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IT, cs-LG, cs-NE, cs-SI, cs.LG, math-IT  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.17781v1)  

---


**ABSTRACT**  
Recent studies attempted to utilize multilayer perceptrons (MLPs) to solve semisupervised node classification on graphs, by training a student MLP by knowledge distillation from a teacher graph neural network (GNN). While previous studies have focused mostly on training the student MLP by matching the output probability distributions between the teacher and student models during distillation, it has not been systematically studied how to inject the structural information in an explicit and interpretable manner. Inspired by GNNs that separate feature transformation $T$ and propagation $\Pi$, we re-frame the distillation process as making the student MLP learn both $T$ and $\Pi$. Although this can be achieved by applying the inverse propagation $\Pi^{-1}$ before distillation from the teacher, it still comes with a high computational cost from large matrix multiplications during training. To solve this problem, we propose Propagate & Distill (P&D), which propagates the output of the teacher before distillation, which can be interpreted as an approximate process of the inverse propagation. We demonstrate that P&D can readily improve the performance of the student MLP.

{{</citation>}}


### (113/155) Improving embedding of graphs with missing data by soft manifolds (Andrea Marinoni et al., 2023)

{{<citation>}}

Andrea Marinoni, Pietro Lio', Alessandro Barp, Christian Jutten, Mark Girolami. (2023)  
**Improving embedding of graphs with missing data by soft manifolds**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CG, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.17598v1)  

---


**ABSTRACT**  
Embedding graphs in continous spaces is a key factor in designing and developing algorithms for automatic information extraction to be applied in diverse tasks (e.g., learning, inferring, predicting). The reliability of graph embeddings directly depends on how much the geometry of the continuous space matches the graph structure. Manifolds are mathematical structure that can enable to incorporate in their topological spaces the graph characteristics, and in particular nodes distances. State-of-the-art of manifold-based graph embedding algorithms take advantage of the assumption that the projection on a tangential space of each point in the manifold (corresponding to a node in the graph) would locally resemble a Euclidean space. Although this condition helps in achieving efficient analytical solutions to the embedding problem, it does not represent an adequate set-up to work with modern real life graphs, that are characterized by weighted connections across nodes often computed over sparse datasets with missing records. In this work, we introduce a new class of manifold, named soft manifold, that can solve this situation. In particular, soft manifolds are mathematical structures with spherical symmetry where the tangent spaces to each point are hypocycloids whose shape is defined according to the velocity of information propagation across the data points. Using soft manifolds for graph embedding, we can provide continuous spaces to pursue any task in data analysis over complex datasets. Experimental results on reconstruction tasks on synthetic and real datasets show how the proposed approach enable more accurate and reliable characterization of graphs in continuous spaces with respect to the state-of-the-art.

{{</citation>}}


### (114/155) LoCoMotif: Discovering time-warped motifs in time series (Daan Van Wesenbeeck et al., 2023)

{{<citation>}}

Daan Van Wesenbeeck, Aras Yurtman, Wannes Meert, Hendrik Blockeel. (2023)  
**LoCoMotif: Discovering time-warped motifs in time series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.17582v1)  

---


**ABSTRACT**  
Time Series Motif Discovery (TSMD) refers to the task of identifying patterns that occur multiple times (possibly with minor variations) in a time series. All existing methods for TSMD have one or more of the following limitations: they only look for the two most similar occurrences of a pattern; they only look for patterns of a pre-specified, fixed length; they cannot handle variability along the time axis; and they only handle univariate time series. In this paper, we present a new method, LoCoMotif, that has none of these limitations. The method is motivated by a concrete use case from physiotherapy. We demonstrate the value of the proposed method on this use case. We also introduce a new quantitative evaluation metric for motif discovery, and benchmark data for comparing TSMD methods. LoCoMotif substantially outperforms the existing methods, on top of being more broadly applicable.

{{</citation>}}


### (115/155) Bias Resilient Multi-Step Off-Policy Goal-Conditioned Reinforcement Learning (Lisheng Wu et al., 2023)

{{<citation>}}

Lisheng Wu, Ke Chen. (2023)  
**Bias Resilient Multi-Step Off-Policy Goal-Conditioned Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Bias, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.17565v1)  

---


**ABSTRACT**  
In goal-conditioned reinforcement learning (GCRL), sparse rewards present significant challenges, often obstructing efficient learning. Although multi-step GCRL can boost this efficiency, it can also lead to off-policy biases in target values. This paper dives deep into these biases, categorizing them into two distinct categories: "shooting" and "shifting". Recognizing that certain behavior policies can hasten policy refinement, we present solutions designed to capitalize on the positive aspects of these biases while minimizing their drawbacks, enabling the use of larger step sizes to speed up GCRL. An empirical study demonstrates that our approach ensures a resilient and robust improvement, even in ten-step learning scenarios, leading to superior learning efficiency and performance that generally surpass the baseline and several state-of-the-art multi-step GCRL benchmarks.

{{</citation>}}


### (116/155) Transformer Based Model for Predicting Rapid Impact Compaction Outcomes: A Case Study of Utapao International Airport (Sompote Youwai et al., 2023)

{{<citation>}}

Sompote Youwai, Sirasak Detcheewa. (2023)  
**Transformer Based Model for Predicting Rapid Impact Compaction Outcomes: A Case Study of Utapao International Airport**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.17959v1)  

---


**ABSTRACT**  
This paper introduces a novel deep learning approach to predict the engineering properties of the ground improved by Rapid Impact Compaction (RIC), which is a ground improvement technique that uses a drop hammer to compact the soil and fill layers. The proposed approach uses transformer-based neural networks to capture the complex nonlinear relationships between the input features, such as the hammer energy, drop height, and number of blows, and the output variables, such as the cone resistance. The approach is applied to a real-world dataset from a trial test section for the new apron construction of the Utapao International Airport in Thailand. The results show that the proposed approach outperforms the existing methods in terms of prediction accuracy and efficiency and provides interpretable attention maps that reveal the importance of different features for RIC prediction. The paper also discusses the limitations and future directions of applying deep learning methods to RIC prediction.

{{</citation>}}


### (117/155) CommunityAI: Towards Community-based Federated Learning (Ilir Murturi et al., 2023)

{{<citation>}}

Ilir Murturi, Praveen Kumar Donta, Schahram Dustdar. (2023)  
**CommunityAI: Towards Community-based Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17958v1)  

---


**ABSTRACT**  
Federated Learning (FL) has emerged as a promising paradigm to train machine learning models collaboratively while preserving data privacy. However, its widespread adoption faces several challenges, including scalability, heterogeneous data and devices, resource constraints, and security concerns. Despite its promise, FL has not been specifically adapted for community domains, primarily due to the wide-ranging differences in data types and context, devices and operational conditions, environmental factors, and stakeholders. In response to these challenges, we present a novel framework for Community-based Federated Learning called CommunityAI. CommunityAI enables participants to be organized into communities based on their shared interests, expertise, or data characteristics. Community participants collectively contribute to training and refining learning models while maintaining data and participant privacy within their respective groups. Within this paper, we discuss the conceptual architecture, system requirements, processes, and future challenges that must be solved. Finally, our goal within this paper is to present our vision regarding enabling a collaborative learning process within various communities.

{{</citation>}}


### (118/155) QuadraNet: Improving High-Order Neural Interaction Efficiency with Hardware-Aware Quadratic Neural Networks (Chenhui Xu et al., 2023)

{{<citation>}}

Chenhui Xu, Fuxun Yu, Zirui Xu, Chenchen Liu, Jinjun Xiong, Xiang Chen. (2023)  
**QuadraNet: Improving High-Order Neural Interaction Efficiency with Hardware-Aware Quadratic Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-NE, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.17956v1)  

---


**ABSTRACT**  
Recent progress in computer vision-oriented neural network designs is mostly driven by capturing high-order neural interactions among inputs and features. And there emerged a variety of approaches to accomplish this, such as Transformers and its variants. However, these interactions generate a large amount of intermediate state and/or strong data dependency, leading to considerable memory consumption and computing cost, and therefore compromising the overall runtime performance. To address this challenge, we rethink the high-order interactive neural network design with a quadratic computing approach. Specifically, we propose QuadraNet -- a comprehensive model design methodology from neuron reconstruction to structural block and eventually to the overall neural network implementation. Leveraging quadratic neurons' intrinsic high-order advantages and dedicated computation optimization schemes, QuadraNet could effectively achieve optimal cognition and computation performance. Incorporating state-of-the-art hardware-aware neural architecture search and system integration techniques, QuadraNet could also be well generalized in different hardware constraint settings and deployment scenarios. The experiment shows thatQuadraNet achieves up to 1.5$\times$ throughput, 30% less memory footprint, and similar cognition performance, compared with the state-of-the-art high-order approaches.

{{</citation>}}


### (119/155) Uncertainty in Additive Feature Attribution methods (Abhishek Madaan et al., 2023)

{{<citation>}}

Abhishek Madaan, Tanya Chowdhury, Neha Rana, James Allan, Tanmoy Chakraborty. (2023)  
**Uncertainty in Additive Feature Attribution methods**  

---
Primary Category: cs.LG  
Categories: I-2-6, cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17446v1)  

---


**ABSTRACT**  
In this work, we explore various topics that fall under the umbrella of Uncertainty in post-hoc Explainable AI (XAI) methods. We in particular focus on the class of additive feature attribution explanation methods. We first describe our specifications of uncertainty and compare various statistical and recent methods to quantify the same. Next, for a particular instance, we study the relationship between a feature's attribution and its uncertainty and observe little correlation. As a result, we propose a modification in the distribution from which perturbations are sampled in LIME-based algorithms such that the important features have minimal uncertainty without an increase in computational cost. Next, while studying how the uncertainty in explanations varies across the feature space of a classifier, we observe that a fraction of instances show near-zero uncertainty. We coin the term "stable instances" for such instances and diagnose factors that make an instance stable. Next, we study how an XAI algorithm's uncertainty varies with the size and complexity of the underlying model. We observe that the more complex the model, the more inherent uncertainty is exhibited by it. As a result, we propose a measure to quantify the relative complexity of a blackbox classifier. This could be incorporated, for example, in LIME-based algorithms' sampling densities, to help different explanation algorithms achieve tighter confidence levels. Together, the above measures would have a strong impact on making XAI models relatively trustworthy for the end-user as well as aiding scientific discovery.

{{</citation>}}


### (120/155) Grounding Foundation Models through Federated Transfer Learning: A General Framework (Yan Kang et al., 2023)

{{<citation>}}

Yan Kang, Tao Fan, Hanlin Gu, Lixin Fan, Qiang Yang. (2023)  
**Grounding Foundation Models through Federated Transfer Learning: A General Framework**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.17431v3)  

---


**ABSTRACT**  
Foundation Models (FMs) such as GPT-4 encoded with vast knowledge and powerful emergent abilities have achieved remarkable success in various natural language processing and computer vision tasks. Grounding FMs by adapting them to domain-specific tasks or augmenting them with domain-specific knowledge enables us to exploit the full potential of FMs. However, grounding FMs faces several challenges, stemming primarily from constrained computing resources, data privacy, model heterogeneity, and model ownership. Federated Transfer Learning (FTL), the combination of federated learning and transfer learning, provides promising solutions to address these challenges. In recent years, the need for grounding FMs leveraging FTL, coined FTL-FM, has arisen strongly in both academia and industry. Motivated by the strong growth in FTL-FM research and the potential impact of FTL-FM on industrial applications, we propose an FTL-FM framework that formulates problems of grounding FMs in the federated learning setting, construct a detailed taxonomy based on the FTL-FM framework to categorize state-of-the-art FTL-FM works, and comprehensively overview FTL-FM works based on the proposed taxonomy. We also establish correspondences between FTL-FM and conventional phases of adapting FM so that FM practitioners can align their research works with FTL-FM. In addition, we overview advanced efficiency-improving and privacy-preserving techniques because efficiency and privacy are critical concerns in FTL-FM. Last, we discuss opportunities and future research directions of FTL-FM.

{{</citation>}}


### (121/155) The Devil is in the Data: Learning Fair Graph Neural Networks via Partial Knowledge Distillation (Yuchang Zhu et al., 2023)

{{<citation>}}

Yuchang Zhu, Jintang Li, Liang Chen, Zibin Zheng. (2023)  
**The Devil is in the Data: Learning Fair Graph Neural Networks via Partial Knowledge Distillation**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.17373v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) are being increasingly used in many high-stakes tasks, and as a result, there is growing attention on their fairness recently. GNNs have been shown to be unfair as they tend to make discriminatory decisions toward certain demographic groups, divided by sensitive attributes such as gender and race. While recent works have been devoted to improving their fairness performance, they often require accessible demographic information. This greatly limits their applicability in real-world scenarios due to legal restrictions. To address this problem, we present a demographic-agnostic method to learn fair GNNs via knowledge distillation, namely FairGKD. Our work is motivated by the empirical observation that training GNNs on partial data (i.e., only node attributes or topology data) can improve their fairness, albeit at the cost of utility. To make a balanced trade-off between fairness and utility performance, we employ a set of fairness experts (i.e., GNNs trained on different partial data) to construct the synthetic teacher, which distills fairer and informative knowledge to guide the learning of the GNN student. Experiments on several benchmark datasets demonstrate that FairGKD, which does not require access to demographic information, significantly improves the fairness of GNNs by a large margin while maintaining their utility.

{{</citation>}}


### (122/155) Efficient Stitchable Task Adaptation (Haoyu He et al., 2023)

{{<citation>}}

Haoyu He, Zizheng Pan, Jing Liu, Jianfei Cai, Bohan Zhuang. (2023)  
**Efficient Stitchable Task Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2311.17352v1)  

---


**ABSTRACT**  
The paradigm of pre-training and fine-tuning has laid the foundation for deploying deep learning models. However, most fine-tuning methods are designed to meet a specific resource budget. Recently, considering diverse deployment scenarios with various resource budgets, stitchable neural network (SN-Net) is introduced to quickly obtain numerous new networks (stitches) from the pre-trained models (anchors) in a model family via model stitching. Although promising, SN-Net confronts new challenges when adapting it to new target domains, including huge memory and storage requirements and a long and sub-optimal multistage adaptation process. In this work, we present a novel framework, Efficient Stitchable Task Adaptation (ESTA), to efficiently produce a palette of fine-tuned models that adhere to diverse resource constraints. Specifically, we first tailor parameter-efficient fine-tuning to share low-rank updates among the stitches while maintaining independent bias terms. In this way, we largely reduce fine-tuning memory burdens and mitigate the interference among stitches that arises in task adaptation. Furthermore, we streamline a simple yet effective one-stage deployment pipeline, which estimates the important stitches to deploy with training-time gradient statistics. By assigning higher sampling probabilities to important stitches, we also get a boosted Pareto frontier. Extensive experiments on 25 downstream visual recognition tasks demonstrate that our ESTA is capable of generating stitches with smooth accuracy-efficiency trade-offs and surpasses the direct SN-Net adaptation by remarkable margins with significantly lower training time and fewer trainable parameters. Furthermore, we demonstrate the flexibility and scalability of our ESTA framework by stitching LLMs from LLaMA family, obtaining chatbot stitches of assorted sizes.

{{</citation>}}


### (123/155) Improving Self-supervised Molecular Representation Learning using Persistent Homology (Yuankai Luo et al., 2023)

{{<citation>}}

Yuankai Luo, Lei Shi, Veronika Thost. (2023)  
**Improving Self-supervised Molecular Representation Learning using Persistent Homology**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.17327v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) has great potential for molecular representation learning given the complexity of molecular graphs, the large amounts of unlabelled data available, the considerable cost of obtaining labels experimentally, and the hence often only small training datasets. The importance of the topic is reflected in the variety of paradigms and architectures that have been investigated recently. Yet the differences in performance seem often minor and are barely understood to date. In this paper, we study SSL based on persistent homology (PH), a mathematical tool for modeling topological features of data that persist across multiple scales. It has several unique features which particularly suit SSL, naturally offering: different views of the data, stability in terms of distance preservation, and the opportunity to flexibly incorporate domain knowledge. We (1) investigate an autoencoder, which shows the general representational power of PH, and (2) propose a contrastive loss that complements existing approaches. We rigorously evaluate our approach for molecular property prediction and demonstrate its particular features in improving the embedding space: after SSL, the representations are better and offer considerably more predictive power than the baselines over different probing tasks; our loss increases baseline performance, sometimes largely; and we often obtain substantial improvements over very small datasets, a common scenario in practice.

{{</citation>}}


### (124/155) LayerCollapse: Adaptive compression of neural networks (Soheil Zibakhsh Shabgahi et al., 2023)

{{<citation>}}

Soheil Zibakhsh Shabgahi, Mohammad Soheil Shariff, Farinaz Koushanfar. (2023)  
**LayerCollapse: Adaptive compression of neural networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.17943v1)  

---


**ABSTRACT**  
Handling the ever-increasing scale of contemporary deep learning and transformer-based models poses a significant challenge. Although great strides have been made in optimizing model compression techniques such as model architecture search and knowledge distillation, the availability of data and computational resources remains a considerable hurdle for these optimizations. This paper introduces LayerCollapse, a novel alternative adaptive model compression methodology. LayerCollapse works by eliminating non-linearities within the network and collapsing two consecutive fully connected layers into a single linear transformation. This approach simultaneously reduces both the number of layers and the parameter count, thereby enhancing model efficiency. We also introduce a compression aware regularizer, which compresses the model in alignment with the dataset quality and model expressiveness, consequently reducing overfitting across tasks. Our results demonstrate LayerCollapse's effective compression and regularization capabilities in multiple fine-grained classification benchmarks, achieving up to 74% post training compression with minimal accuracy loss. We compare this method with knowledge distillation on the same target network, showcasing a five-fold increase in computational efficiency and 8% improvement in overall accuracy on the ImageNet dataset.

{{</citation>}}


## eess.IV (2)



### (125/155) Corner-to-Center Long-range Context Model for Efficient Learned Image Compression (Yang Sui et al., 2023)

{{<citation>}}

Yang Sui, Ding Ding, Xiang Pan, Xiaozhong Xu, Shan Liu, Bo Yuan, Zhenzhong Chen. (2023)  
**Corner-to-Center Long-range Context Model for Efficient Learned Image Compression**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.18103v1)  

---


**ABSTRACT**  
In the framework of learned image compression, the context model plays a pivotal role in capturing the dependencies among latent representations. To reduce the decoding time resulting from the serial autoregressive context model, the parallel context model has been proposed as an alternative that necessitates only two passes during the decoding phase, thus facilitating efficient image compression in real-world scenarios. However, performance degradation occurs due to its incomplete casual context. To tackle this issue, we conduct an in-depth analysis of the performance degradation observed in existing parallel context models, focusing on two aspects: the Quantity and Quality of information utilized for context prediction and decoding. Based on such analysis, we propose the \textbf{Corner-to-Center transformer-based Context Model (C$^3$M)} designed to enhance context and latent predictions and improve rate-distortion performance. Specifically, we leverage the logarithmic-based prediction order to predict more context features from corner to center progressively. In addition, to enlarge the receptive field in the analysis and synthesis transformation, we use the Long-range Crossing Attention Module (LCAM) in the encoder/decoder to capture the long-range semantic information by assigning the different window shapes in different channels. Extensive experimental evaluations show that the proposed method is effective and outperforms the state-of-the-art parallel methods. Finally, according to the subjective analysis, we suggest that improving the detailed representation in transformer-based image compression is a promising direction to be explored.

{{</citation>}}


### (126/155) Cross-Scope Spatial-Spectral Information Aggregation for Hyperspectral Image Super-Resolution (Shi Chen et al., 2023)

{{<citation>}}

Shi Chen, Lefei Zhang, Liangpei Zhang. (2023)  
**Cross-Scope Spatial-Spectral Information Aggregation for Hyperspectral Image Super-Resolution**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.17340v1)  

---


**ABSTRACT**  
Hyperspectral image super-resolution has attained widespread prominence to enhance the spatial resolution of hyperspectral images. However, convolution-based methods have encountered challenges in harnessing the global spatial-spectral information. The prevailing transformer-based methods have not adequately captured the long-range dependencies in both spectral and spatial dimensions. To alleviate this issue, we propose a novel cross-scope spatial-spectral Transformer (CST) to efficiently investigate long-range spatial and spectral similarities for single hyperspectral image super-resolution. Specifically, we devise cross-attention mechanisms in spatial and spectral dimensions to comprehensively model the long-range spatial-spectral characteristics. By integrating global information into the rectangle-window self-attention, we first design a cross-scope spatial self-attention to facilitate long-range spatial interactions. Then, by leveraging appropriately characteristic spatial-spectral features, we construct a cross-scope spectral self-attention to effectively capture the intrinsic correlations among global spectral bands. Finally, we elaborate a concise feed-forward neural network to enhance the feature representation capacity in the Transformer structure. Extensive experiments over three hyperspectral datasets demonstrate that the proposed CST is superior to other state-of-the-art methods both quantitatively and visually. The code is available at \url{https://github.com/Tomchenshi/CST.git}.

{{</citation>}}


## astro-ph.IM (1)



### (127/155) Self-Driving Telescopes: Autonomous Scheduling of Astronomical Observation Campaigns with Offline Reinforcement Learning (Franco Terranova et al., 2023)

{{<citation>}}

Franco Terranova, M. Voetberg, Brian Nord, Amanda Pagul. (2023)  
**Self-Driving Telescopes: Autonomous Scheduling of Astronomical Observation Campaigns with Offline Reinforcement Learning**  

---
Primary Category: astro-ph.IM  
Categories: astro-ph-CO, astro-ph-IM, astro-ph.IM, cs-AI, cs-LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.18094v1)  

---


**ABSTRACT**  
Modern astronomical experiments are designed to achieve multiple scientific goals, from studies of galaxy evolution to cosmic acceleration. These goals require data of many different classes of night-sky objects, each of which has a particular set of observational needs. These observational needs are typically in strong competition with one another. This poses a challenging multi-objective optimization problem that remains unsolved. The effectiveness of Reinforcement Learning (RL) as a valuable paradigm for training autonomous systems has been well-demonstrated, and it may provide the basis for self-driving telescopes capable of optimizing the scheduling for astronomy campaigns. Simulated datasets containing examples of interactions between a telescope and a discrete set of sky locations on the celestial sphere can be used to train an RL model to sequentially gather data from these several locations to maximize a cumulative reward as a measure of the quality of the data gathered. We use simulated data to test and compare multiple implementations of a Deep Q-Network (DQN) for the task of optimizing the schedule of observations from the Stone Edge Observatory (SEO). We combine multiple improvements on the DQN and adjustments to the dataset, showing that DQNs can achieve an average reward of 87%+-6% of the maximum achievable reward in each state on the test set. This is the first comparison of offline RL algorithms for a particular astronomical challenge and the first open-source framework for performing such a comparison and assessment task.

{{</citation>}}


## cs.CR (3)



### (128/155) Leveraging a Randomized Key Matrix to Enhance the Security of Symmetric Substitution Ciphers (Shubham Gandhi et al., 2023)

{{<citation>}}

Shubham Gandhi, Om Khare, Mihika Dravid, Mihika Sanghvi, Sunil Mane, Aadesh Gajaralwar, Saloni Gandhi. (2023)  
**Leveraging a Randomized Key Matrix to Enhance the Security of Symmetric Substitution Ciphers**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.18085v1)  

---


**ABSTRACT**  
An innovative strategy to enhance the security of symmetric substitution ciphers is presented, through the implementation of a randomized key matrix suitable for various file formats, including but not limited to binary and text files. Despite their historical relevance, symmetric substitution ciphers have been limited by vulnerabilities to cryptanalytic methods like frequency analysis and known plaintext attacks. The aim of our research is to mitigate these vulnerabilities by employing a polyalphabetic substitution strategy that incorporates a distinct randomized key matrix. This matrix plays a pivotal role in generating a unique random key, comprising characters, encompassing both uppercase and lowercase letters, numeric, and special characters, to derive the corresponding ciphertext. The effectiveness of the proposed methodology in enhancing the security of conventional substitution methods for file encryption and decryption is supported by comprehensive testing and analysis, which encompass computational speed, frequency analysis, keyspace examination, Kasiski test, entropy analysis, and the utilization of a large language model.

{{</citation>}}


### (129/155) Learning-driven Zero Trust in Distributed Computing Continuum Systems (Ilir Murturi et al., 2023)

{{<citation>}}

Ilir Murturi, Praveen Kumar Donta, Victor Casamayor Pujol, Andrea Morichetta, Schahram Dustdar. (2023)  
**Learning-driven Zero Trust in Distributed Computing Continuum Systems**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-DC, cs.CR  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.17447v1)  

---


**ABSTRACT**  
Converging Zero Trust (ZT) with learning techniques can solve various operational and security challenges in Distributed Computing Continuum Systems (DCCS). Implementing centralized ZT architecture is seen as unsuitable for the computing continuum (e.g., computing entities with limited connectivity and visibility, etc.). At the same time, implementing decentralized ZT in the computing continuum requires understanding infrastructure limitations and novel approaches to enhance resource access management decisions. To overcome such challenges, we present a novel learning-driven ZT conceptual architecture designed for DCCS. We aim to enhance ZT architecture service quality by incorporating lightweight learning strategies such as Representation Learning (ReL) and distributing ZT components across the computing continuum. The ReL helps to improve the decision-making process by predicting threats or untrusted requests. Through an illustrative example, we show how the learning process detects and blocks the requests, enhances resource access control, and reduces network and computation overheads. Lastly, we discuss the conceptual architecture, processes, and provide a research agenda.

{{</citation>}}


### (130/155) Deepfakes, Misinformation, and Disinformation in the Era of Frontier AI, Generative AI, and Large AI Models (Mohamed R. Shoaib et al., 2023)

{{<citation>}}

Mohamed R. Shoaib, Zefan Wang, Milad Taleby Ahvanooey, Jun Zhao. (2023)  
**Deepfakes, Misinformation, and Disinformation in the Era of Frontier AI, Generative AI, and Large AI Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.17394v1)  

---


**ABSTRACT**  
With the advent of sophisticated artificial intelligence (AI) technologies, the proliferation of deepfakes and the spread of m/disinformation have emerged as formidable threats to the integrity of information ecosystems worldwide. This paper provides an overview of the current literature. Within the frontier AI's crucial application in developing defense mechanisms for detecting deepfakes, we highlight the mechanisms through which generative AI based on large models (LM-based GenAI) craft seemingly convincing yet fabricated contents. We explore the multifaceted implications of LM-based GenAI on society, politics, and individual privacy violations, underscoring the urgent need for robust defense strategies. To address these challenges, in this study, we introduce an integrated framework that combines advanced detection algorithms, cross-platform collaboration, and policy-driven initiatives to mitigate the risks associated with AI-Generated Content (AIGC). By leveraging multi-modal analysis, digital watermarking, and machine learning-based authentication techniques, we propose a defense mechanism adaptable to AI capabilities of ever-evolving nature. Furthermore, the paper advocates for a global consensus on the ethical usage of GenAI and implementing cyber-wellness educational programs to enhance public awareness and resilience against m/disinformation. Our findings suggest that a proactive and collaborative approach involving technological innovation and regulatory oversight is essential for safeguarding netizens while interacting with cyberspace against the insidious effects of deepfakes and GenAI-enabled m/disinformation campaigns.

{{</citation>}}


## physics.med-ph (1)



### (131/155) Predicting breast cancer with AI for individual risk-adjusted MRI screening and early detection (Lukas Hirsch et al., 2023)

{{<citation>}}

Lukas Hirsch, Yu Huang, Hernan A. Makse, Danny F. Martinez, Mary Hughes, Sarah Eskreis-Winkler, Katja Pinker, Elizabeth Morris, Lucas C. Parra, Elizabeth J. Sutton. (2023)  
**Predicting breast cancer with AI for individual risk-adjusted MRI screening and early detection**  

---
Primary Category: physics.med-ph  
Categories: cs-CV, cs-LG, eess-IV, physics-med-ph, physics.med-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00067v1)  

---


**ABSTRACT**  
Women with an increased life-time risk of breast cancer undergo supplemental annual screening MRI. We propose to predict the risk of developing breast cancer within one year based on the current MRI, with the objective of reducing screening burden and facilitating early detection. An AI algorithm was developed on 53,858 breasts from 12,694 patients who underwent screening or diagnostic MRI and accrued over 12 years, with 2,331 confirmed cancers. A first U-Net was trained to segment lesions and identify regions of concern. A second convolutional network was trained to detect malignant cancer using features extracted by the U-Net. This network was then fine-tuned to estimate the risk of developing cancer within a year in cases that radiologists considered normal or likely benign. Risk predictions from this AI were evaluated with a retrospective analysis of 9,183 breasts from a high-risk screening cohort, which were not used for training. Statistical analysis focused on the tradeoff between number of omitted exams versus negative predictive value, and number of potential early detections versus positive predictive value. The AI algorithm identified regions of concern that coincided with future tumors in 52% of screen-detected cancers. Upon directed review, a radiologist found that 71.3% of cancers had a visible correlate on the MRI prior to diagnosis, 65% of these correlates were identified by the AI model. Reevaluating these regions in 10% of all cases with higher AI-predicted risk could have resulted in up to 33% early detections by a radiologist. Additionally, screening burden could have been reduced in 16% of lower-risk cases by recommending a later follow-up without compromising current interval cancer rate. With increasing datasets and improving image quality we expect this new AI-aided, adaptive screening to meaningfully reduce screening burden and improve early detection.

{{</citation>}}


## cs.HC (1)



### (132/155) Making Data Work Count (Srravya Chandhiramowuli et al., 2023)

{{<citation>}}

Srravya Chandhiramowuli, Alex Taylor, Sara Heitlinger, Ding Wang. (2023)  
**Making Data Work Count**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18046v1)  

---


**ABSTRACT**  
In this paper, we examine the work of data annotation. Specifically, we focus on the role of counting or quantification in organising annotation work. Based on an ethnographic study of data annotation in two outsourcing centres in India, we observe that counting practices and its associated logics are an integral part of day-to-day annotation activities. In particular, we call attention to the presumption of total countability observed in annotation - the notion that everything, from tasks, datasets and deliverables, to workers, work time, quality and performance, can be managed by applying the logics of counting. To examine this, we draw on sociological and socio-technical scholarship on quantification and develop the lens of a 'regime of counting' that makes explicit the specific counts, practices, actors and structures that underpin the pervasive counting in annotation. We find that within the AI supply chain and data work, counting regimes aid the assertion of authority by the AI clients (also called requesters) over annotation processes, constituting them as reductive, standardised, and homogenous. We illustrate how this has implications for i) how annotation work and workers get valued, ii) the role human discretion plays in annotation, and iii) broader efforts to introduce accountable and more just practices in AI. Through these implications, we illustrate the limits of operating within the logic of total countability. Instead, we argue for a view of counting as partial - located in distinct geographies, shaped by specific interests and accountable in only limited ways. This, we propose, sets the stage for a fundamentally different orientation to counting and what counts in data annotation.

{{</citation>}}


## cs.CY (1)



### (133/155) Evaluating Trustworthiness of AI-Enabled Decision Support Systems: Validation of the Multisource AI Scorecard Table (MAST) (Pouria Salehi et al., 2023)

{{<citation>}}

Pouria Salehi, Yang Ba, Nayoung Kim, Ahmadreza Mosallanezhad, Anna Pan, Myke C. Cohen, Yixuan Wang, Jieqiong Zhao, Shawaiz Bhatti, James Sung, Erik Blasch, Michelle V. Mancenido, Erin K. Chiou. (2023)  
**Evaluating Trustworthiness of AI-Enabled Decision Support Systems: Validation of the Multisource AI Scorecard Table (MAST)**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18040v1)  

---


**ABSTRACT**  
The Multisource AI Scorecard Table (MAST) is a checklist tool based on analytic tradecraft standards to inform the design and evaluation of trustworthy AI systems. In this study, we evaluate whether MAST is associated with people's trust perceptions in AI-enabled decision support systems (AI-DSSs). Evaluating trust in AI-DSSs poses challenges to researchers and practitioners. These challenges include identifying the components, capabilities, and potential of these systems, many of which are based on the complex deep learning algorithms that drive DSS performance and preclude complete manual inspection. We developed two interactive, AI-DSS test environments using the MAST criteria. One emulated an identity verification task in security screening, and another emulated a text summarization system to aid in an investigative reporting task. Each test environment had one version designed to match low-MAST ratings, and another designed to match high-MAST ratings, with the hypothesis that MAST ratings would be positively related to the trust ratings of these systems. A total of 177 subject matter experts were recruited to interact with and evaluate these systems. Results generally show higher MAST ratings for the high-MAST conditions compared to the low-MAST groups, and that measures of trust perception are highly correlated with the MAST ratings. We conclude that MAST can be a useful tool for designing and evaluating systems that will engender high trust perceptions, including AI-DSS that may be used to support visual screening and text summarization tasks. However, higher MAST ratings may not translate to higher joint performance.

{{</citation>}}


## physics.flu-dyn (1)



### (134/155) Enhancing Data-Assimilation in CFD using Graph Neural Networks (Michele Quattromini et al., 2023)

{{<citation>}}

Michele Quattromini, Michele Alessandro Bucci, Stefania Cherubini, Onofrio Semeraro. (2023)  
**Enhancing Data-Assimilation in CFD using Graph Neural Networks**  

---
Primary Category: physics.flu-dyn  
Categories: cs-LG, physics-flu-dyn, physics.flu-dyn  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.18027v1)  

---


**ABSTRACT**  
We present a novel machine learning approach for data assimilation applied in fluid mechanics, based on adjoint-optimization augmented by Graph Neural Networks (GNNs) models. We consider as baseline the Reynolds-Averaged Navier-Stokes (RANS) equations, where the unknown is the meanflow and a closure model based on the Reynolds-stress tensor is required for correctly computing the solution. An end-to-end process is cast; first, we train a GNN model for the closure term. Second, the GNN model is introduced in the training process of data assimilation, where the RANS equations act as a physics constraint for a consistent prediction. We obtain our results using direct numerical simulations based on a Finite Element Method (FEM) solver; a two-fold interface between the GNN model and the solver allows the GNN's predictions to be incorporated into post-processing steps of the FEM analysis. The proposed scheme provides an excellent reconstruction of the meanflow without any features selection; preliminary results show promising generalization properties over unseen flow configurations.

{{</citation>}}


## cs.DC (3)



### (135/155) FastSample: Accelerating Distributed Graph Neural Network Training for Billion-Scale Graphs (Hesham Mostafa et al., 2023)

{{<citation>}}

Hesham Mostafa, Adam Grabowski, Md Asadullah Turja, Juan Cervino, Alejandro Ribeiro, Nageen Himayat. (2023)  
**FastSample: Accelerating Distributed Graph Neural Network Training for Billion-Scale Graphs**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.17847v1)  

---


**ABSTRACT**  
Training Graph Neural Networks(GNNs) on a large monolithic graph presents unique challenges as the graph cannot fit within a single machine and it cannot be decomposed into smaller disconnected components. Distributed sampling-based training distributes the graph across multiple machines and trains the GNN on small parts of the graph that are randomly sampled every training iteration. We show that in a distributed environment, the sampling overhead is a significant component of the training time for large-scale graphs. We propose FastSample which is composed of two synergistic techniques that greatly reduce the distributed sampling time: 1)a new graph partitioning method that eliminates most of the communication rounds in distributed sampling , 2)a novel highly optimized sampling kernel that reduces memory movement during sampling. We test FastSample on large-scale graph benchmarks and show that FastSample speeds up distributed sampling-based GNN training by up to 2x with no loss in accuracy.

{{</citation>}}


### (136/155) Performance Evaluation of Checkpoint/Restart Techniques (Basma Abdel Azeem et al., 2023)

{{<citation>}}

Basma Abdel Azeem, Manal Helal. (2023)  
**Performance Evaluation of Checkpoint/Restart Techniques**  

---
Primary Category: cs.DC  
Categories: D-1-3, cs-DC, cs.DC  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2311.17545v1)  

---


**ABSTRACT**  
Distributed applications running on a large cluster environment, such as the cloud instances will have shorter execution time. However, the application might suffer from sudden termination due to unpredicted computing node failures, thus loosing the whole computation. Checkpoint/restart is a fault tolerance technique used to solve this problem. In this work we evaluated the performance of two of the most commonly used checkpoint/restart techniques (Distributed Multithreaded Checkpointing (DMTCP) and Berkeley Lab Checkpoint/Restart library (BLCR) integrated into the OpenMPI framework). We aimed to test their validity and evaluate their performance in both local and Amazon Elastic Compute Cloud (EC2) environments. The experiments were conducted on Amazon EC2 as a well-known proprietary cloud computing service provider. Results obtained were reported and compared to evaluate checkpoint and restart time values, data scalability and compute processes scalability. The findings proved that DMTCP performs better than BLCR for checkpoint and restart speed, data scalability and compute processes scalability experiments.

{{</citation>}}


### (137/155) GNNFlow: A Distributed Framework for Continuous Temporal GNN Learning on Dynamic Graphs (Yuchen Zhong et al., 2023)

{{<citation>}}

Yuchen Zhong, Guangming Sheng, Tianzuo Qin, Minjie Wang, Quan Gan, Chuan Wu. (2023)  
**GNNFlow: A Distributed Framework for Continuous Temporal GNN Learning on Dynamic Graphs**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs.DC  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.17410v2)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) play a crucial role in various fields. However, most existing deep graph learning frameworks assume pre-stored static graphs and do not support training on graph streams. In contrast, many real-world graphs are dynamic and contain time domain information. We introduce GNNFlow, a distributed framework that enables efficient continuous temporal graph representation learning on dynamic graphs on multi-GPU machines. GNNFlow introduces an adaptive time-indexed block-based data structure that effectively balances memory usage with graph update and sampling operation efficiency. It features a hybrid GPU-CPU graph data placement for rapid GPU-based temporal neighborhood sampling and kernel optimizations for enhanced sampling processes. A dynamic GPU cache for node and edge features is developed to maximize cache hit rates through reuse and restoration strategies. GNNFlow supports distributed training across multiple machines with static scheduling to ensure load balance. We implement GNNFlow based on DGL and PyTorch. Our experimental results show that GNNFlow provides up to 21.1x faster continuous learning than existing systems.

{{</citation>}}


## cs.RO (2)



### (138/155) Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning (Yingdong Hu et al., 2023)

{{<citation>}}

Yingdong Hu, Fanqi Lin, Tong Zhang, Li Yi, Yang Gao. (2023)  
**Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.17842v1)  

---


**ABSTRACT**  
In this study, we are interested in imbuing robots with the capability of physically-grounded task planning. Recent advancements have shown that large language models (LLMs) possess extensive knowledge useful in robotic tasks, especially in reasoning and planning. However, LLMs are constrained by their lack of world grounding and dependence on external affordance models to perceive environmental information, which cannot jointly reason with LLMs. We argue that a task planner should be an inherently grounded, unified multimodal system. To this end, we introduce Robotic Vision-Language Planning (ViLa), a novel approach for long-horizon robotic planning that leverages vision-language models (VLMs) to generate a sequence of actionable steps. ViLa directly integrates perceptual data into its reasoning and planning process, enabling a profound understanding of commonsense knowledge in the visual world, including spatial layouts and object attributes. It also supports flexible multimodal goal specification and naturally incorporates visual feedback. Our extensive evaluation, conducted in both real-robot and simulated environments, demonstrates ViLa's superiority over existing LLM-based planners, highlighting its effectiveness in a wide array of open-world manipulation tasks.

{{</citation>}}


### (139/155) LLM-State: Expandable State Representation for Long-horizon Task Planning in the Open World (Siwei Chen et al., 2023)

{{<citation>}}

Siwei Chen, Anxing Xiao, David Hsu. (2023)  
**LLM-State: Expandable State Representation for Long-horizon Task Planning in the Open World**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17406v1)  

---


**ABSTRACT**  
This work addresses the problem of long-horizon task planning with the Large Language Model (LLM) in an open-world household environment. Existing works fail to explicitly track key objects and attributes, leading to erroneous decisions in long-horizon tasks, or rely on highly engineered state features and feedback, which is not generalizable. We propose a novel, expandable state representation that provides continuous expansion and updating of object attributes from the LLM's inherent capabilities for context understanding and historical action reasoning. Our proposed representation maintains a comprehensive record of an object's attributes and changes, enabling robust retrospective summary of the sequence of actions leading to the current state. This allows enhanced context understanding for decision-making in task planning. We validate our model through experiments across simulated and real-world task planning scenarios, demonstrating significant improvements over baseline methods in a variety of tasks requiring long-horizon state tracking and reasoning.

{{</citation>}}


## cs.SD (1)



### (140/155) FAT-HuBERT: Front-end Adaptive Training of Hidden-unit BERT for Distortion-Invariant Robust Speech Recognition (Dongning Yang et al., 2023)

{{<citation>}}

Dongning Yang, Wei Wang, Yanmin Qian. (2023)  
**FAT-HuBERT: Front-end Adaptive Training of Hidden-unit BERT for Distortion-Invariant Robust Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: BERT, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2311.17790v1)  

---


**ABSTRACT**  
Advancements in monaural speech enhancement (SE) techniques have greatly improved the perceptual quality of speech. However, integrating these techniques into automatic speech recognition (ASR) systems has not yielded the expected performance gains, primarily due to the introduction of distortions during the SE process. In this paper, we propose a novel approach called FAT-HuBERT, which leverages distortion-invariant self-supervised learning (SSL) to enhance the robustness of ASR. To address the distortions introduced by the SE frontends, we introduce layer-wise fusion modules that incorporate features extracted from both observed noisy signals and enhanced signals. During training, the SE frontend is randomly selected from a pool of models. We evaluate the performance of FAT-HuBERT on simulated noisy speech generated from LibriSpeech as well as real-world noisy speech from the CHiME-4 1-channel dataset. The experimental results demonstrate a significant relative reduction in word error rate (WER).

{{</citation>}}


## cs.IR (2)



### (141/155) $Q_{bias}$ -- A Dataset on Media Bias in Search Queries and Query Suggestions (Fabian Haak et al., 2023)

{{<citation>}}

Fabian Haak, Philipp Schaer. (2023)  
**$Q_{bias}$ -- A Dataset on Media Bias in Search Queries and Query Suggestions**  

---
Primary Category: cs.IR  
Categories: K-4, cs-IR, cs.IR  
Keywords: Bias, Google  
[Paper Link](http://arxiv.org/abs/2311.17780v1)  

---


**ABSTRACT**  
This publication describes the motivation and generation of $Q_{bias}$, a large dataset of Google and Bing search queries, a scraping tool and dataset for biased news articles, as well as language models for the investigation of bias in online search. Web search engines are a major factor and trusted source in information search, especially in the political domain. However, biased information can influence opinion formation and lead to biased opinions. To interact with search engines, users formulate search queries and interact with search query suggestions provided by the search engines. A lack of datasets on search queries inhibits research on the subject. We use $Q_{bias}$ to evaluate different approaches to fine-tuning transformer-based language models with the goal of producing models capable of biasing text with left and right political stance. Additionally to this work we provided datasets and language models for biasing texts that allow further research on bias in online information search.

{{</citation>}}


### (142/155) Attribute Simulation for Item Embedding Enhancement in Multi-interest Recommendation (Yaokun Liu et al., 2023)

{{<citation>}}

Yaokun Liu, Xiaowang Zhang, Minghui Zou, Zhiyong Feng. (2023)  
**Attribute Simulation for Item Embedding Enhancement in Multi-interest Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.17374v1)  

---


**ABSTRACT**  
Although multi-interest recommenders have achieved significant progress in the matching stage, our research reveals that existing models tend to exhibit an under-clustered item embedding space, which leads to a low discernibility between items and hampers item retrieval. This highlights the necessity for item embedding enhancement. However, item attributes, which serve as effective and straightforward side information for enhancement, are either unavailable or incomplete in many public datasets due to the labor-intensive nature of manual annotation tasks. This dilemma raises two meaningful questions: 1. Can we bypass manual annotation and directly simulate complete attribute information from the interaction data? And 2. If feasible, how to simulate attributes with high accuracy and low complexity in the matching stage?   In this paper, we first establish an inspiring theoretical feasibility that the item-attribute correlation matrix can be approximated through elementary transformations on the item co-occurrence matrix. Then based on formula derivation, we propose a simple yet effective module, SimEmb (Item Embedding Enhancement via Simulated Attribute), in the multi-interest recommendation of the matching stage to implement our findings. By simulating attributes with the co-occurrence matrix, SimEmb discards the item ID-based embedding and employs the attribute-weighted summation for item embedding enhancement. Comprehensive experiments on four benchmark datasets demonstrate that our approach notably enhances the clustering of item embedding and significantly outperforms SOTA models with an average improvement of 25.59% on Recall@20.

{{</citation>}}


## cs.IT (1)



### (143/155) Multi-dimensional Energy Limitation in Sphere Shaping for Nonlinear Interference Noise Mitigation (Jingtian Liu et al., 2023)

{{<citation>}}

Jingtian Liu, Élie Awwad, Yves Jaouën. (2023)  
**Multi-dimensional Energy Limitation in Sphere Shaping for Nonlinear Interference Noise Mitigation**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.17726v1)  

---


**ABSTRACT**  
We propose Four-Dimensional (4D) energy limit enumerative sphere shaping (ESS) of $M$-QAM signaling to minimize rate loss and improve the transmission performance over non-linear WDM optical-fiber systems. Simulation results show that the proposed scheme outperforms the conventional ESS by $0.19$~bit/4D-symbol in achievable information rate over a $205$-km single-span link and a WDM transmission of five polarization-division-multiplexed channels with $400$-Gbit/s net rate per channel. We also study the achieved performance over several shaping block lengths and show that the achieved gains do not scale well over multi-span systems.

{{</citation>}}


## cs.SI (2)



### (144/155) Invisible Women in Digital Diplomacy: A Multidimensional Framework for Online Gender Bias Against Women Ambassadors Worldwide (Yevgeniy Golovchenko et al., 2023)

{{<citation>}}

Yevgeniy Golovchenko, Karolina Stańczak, Rebecca Adler-Nissen, Patrice Wangen, Isabelle Augenstein. (2023)  
**Invisible Women in Digital Diplomacy: A Multidimensional Framework for Online Gender Bias Against Women Ambassadors Worldwide**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.17627v1)  

---


**ABSTRACT**  
Despite mounting evidence that women in foreign policy often bear the brunt of online hostility, the extent of online gender bias against diplomats remains unexplored. This paper offers the first global analysis of the treatment of women diplomats on social media. Introducing a multidimensional and multilingual methodology for studying online gender bias, it focuses on three critical elements: gendered language, negativity in tweets directed at diplomats, and the visibility of women diplomats. Our unique dataset encompasses ambassadors from 164 countries, their tweets, and the direct responses to these tweets in 65 different languages. Using automated content and sentiment analysis, our findings reveal a crucial gender bias. The language in responses to diplomatic tweets is only mildly gendered and largely pertains to international affairs and, generally, women ambassadors do not receive more negative reactions to their tweets than men, yet the pronounced discrepancy in online visibility stands out as a significant form of gender bias. Women receive a staggering 66.4% fewer retweets than men. By unraveling the invisibility that obscures women diplomats on social media, we hope to spark further research on online bias in international politics.

{{</citation>}}


### (145/155) CrimeGNN: Harnessing the Power of Graph Neural Networks for Community Detection in Criminal Networks (Chen Yang, 2023)

{{<citation>}}

Chen Yang. (2023)  
**CrimeGNN: Harnessing the Power of Graph Neural Networks for Community Detection in Criminal Networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.17479v1)  

---


**ABSTRACT**  
In this paper, we introduce CrimeGNN, a novel application of Graph Neural Networks (GNNs) specifically designed to uncover hidden communities within criminal networks. As criminal activities increasingly rely on complex network structures, traditional methods of network analysis often fall short in detecting the intricate and dynamic communities within these networks. Leveraging the power of GNNs, CrimeGNN provides an advanced and specialized solution to this problem. The model ingests a graph structure of a criminal network, where vertices represent individuals and edges represent relationships between them. CrimeGNN aims to identify a partition of the vertex set, such that each subset represents a distinct community within the network, maximizing the modularity function. Experimental results on several benchmark datasets demonstrate the effectiveness of CrimeGNN, outperforming existing methods in terms of both accuracy and computational efficiency. The proposed framework offers significant potential for aiding law enforcement agencies in proactive policing and crime prevention measures by providing a more in-depth understanding of the structure and operation of criminal networks.

{{</citation>}}


## eess.SY (1)



### (146/155) Deep Reinforcement Learning Graphs: Feedback Motion Planning via Neural Lyapunov Verification (Armin Ghanbarzadeh et al., 2023)

{{<citation>}}

Armin Ghanbarzadeh, Esmaeil Najafi. (2023)  
**Deep Reinforcement Learning Graphs: Feedback Motion Planning via Neural Lyapunov Verification**  

---
Primary Category: eess.SY  
Categories: cs-RO, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.17587v1)  

---


**ABSTRACT**  
Recent advancements in model-free deep reinforcement learning have enabled efficient agent training. However, challenges arise when determining the region of attraction for these controllers, especially if the region does not fully cover the desired area. This paper addresses this issue by introducing a feedback motion control algorithm that utilizes data-driven techniques and neural networks. The algorithm constructs a graph of connected reinforcement-learning based controllers, each with its own defined region of attraction. This incremental approach effectively covers a bounded region of interest, creating a trajectory of interconnected nodes that guide the system from an initial state to the goal. Two approaches are presented for connecting nodes within the algorithm. The first is a tree-structured method, facilitating "point-to-point" control by constructing a tree connecting the initial state to the goal state. The second is a graph-structured method, enabling "space-to-space" control by building a graph within a bounded region. This approach allows for control from arbitrary initial and goal states. The proposed method's performance is evaluated on a first-order dynamic system, considering scenarios both with and without obstacles. The results demonstrate the effectiveness of the proposed algorithm in achieving the desired control objectives.

{{</citation>}}


## cs.AI (4)



### (147/155) TaskWeaver: A Code-First Agent Framework (Bo Qiao et al., 2023)

{{<citation>}}

Bo Qiao, Liqun Li, Xu Zhang, Shilin He, Yu Kang, Chaoyun Zhang, Fangkai Yang, Hang Dong, Jue Zhang, Lu Wang, Minghua Ma, Pu Zhao, Si Qin, Xiaoting Qin, Chao Du, Yong Xu, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang. (2023)  
**TaskWeaver: A Code-First Agent Framework**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17541v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown impressive abilities in natural language understanding and generation, leading to their use in applications such as chatbots and virtual assistants. However, existing LLM frameworks face limitations in handling domain-specific data analytics tasks with rich data structures. Moreover, they struggle with flexibility to meet diverse user requirements. To address these issues, TaskWeaver is proposed as a code-first framework for building LLM-powered autonomous agents. It converts user requests into executable code and treats user-defined plugins as callable functions. TaskWeaver provides support for rich data structures, flexible plugin usage, and dynamic plugin selection, and leverages LLM coding capabilities for complex logic. It also incorporates domain-specific knowledge through examples and ensures the secure execution of generated code. TaskWeaver offers a powerful and flexible framework for creating intelligent conversational agents that can handle complex tasks and adapt to domain-specific scenarios. The code is open-sourced at https://github.com/microsoft/TaskWeaver/.

{{</citation>}}


### (148/155) Distributed AI in Zero-touch Provisioning for Edge Networks: Challenges and Research Directions (Abhishek Hazra et al., 2023)

{{<citation>}}

Abhishek Hazra, Andrea Morichetta, Ilir Murturi, Lauri Lovén, Chinmaya Kumar Dehury, Victor Casamayor Pujol, Praveen Kumar Donta, Schahram Dustdar. (2023)  
**Distributed AI in Zero-touch Provisioning for Edge Networks: Challenges and Research Directions**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-NI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17471v1)  

---


**ABSTRACT**  
Zero-touch network is anticipated to inaugurate the generation of intelligent and highly flexible resource provisioning strategies where multiple service providers collaboratively offer computation and storage resources. This transformation presents substantial challenges to network administration and service providers regarding sustainability and scalability. This article combines Distributed Artificial Intelligence (DAI) with Zero-touch Provisioning (ZTP) for edge networks. This combination helps to manage network devices seamlessly and intelligently by minimizing human intervention. In addition, several advantages are also highlighted that come with incorporating Distributed AI into ZTP in the context of edge networks. Further, we draw potential research directions to foster novel studies in this field and overcome the current limitations.

{{</citation>}}


### (149/155) Exploring Large Language Models for Human Mobility Prediction under Public Events (Yuebing Liang et al., 2023)

{{<citation>}}

Yuebing Liang, Yichao Liu, Xiaohan Wang, Zhan Zhao. (2023)  
**Exploring Large Language Models for Human Mobility Prediction under Public Events**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17351v1)  

---


**ABSTRACT**  
Public events, such as concerts and sports games, can be major attractors for large crowds, leading to irregular surges in travel demand. Accurate human mobility prediction for public events is thus crucial for event planning as well as traffic or crowd management. While rich textual descriptions about public events are commonly available from online sources, it is challenging to encode such information in statistical or machine learning models. Existing methods are generally limited in incorporating textual information, handling data sparsity, or providing rationales for their predictions. To address these challenges, we introduce a framework for human mobility prediction under public events (LLM-MPE) based on Large Language Models (LLMs), leveraging their unprecedented ability to process textual data, learn from minimal examples, and generate human-readable explanations. Specifically, LLM-MPE first transforms raw, unstructured event descriptions from online sources into a standardized format, and then segments historical mobility data into regular and event-related components. A prompting strategy is designed to direct LLMs in making and rationalizing demand predictions considering historical mobility and event features. A case study is conducted for Barclays Center in New York City, based on publicly available event information and taxi trip data. Results show that LLM-MPE surpasses traditional models, particularly on event days, with textual data significantly enhancing its accuracy. Furthermore, LLM-MPE offers interpretable insights into its predictions. Despite the great potential of LLMs, we also identify key challenges including misinformation and high costs that remain barriers to their broader adoption in large-scale human mobility analysis.

{{</citation>}}


### (150/155) Two-Step Reinforcement Learning for Multistage Strategy Card Game (Konrad Godlewski et al., 2023)

{{<citation>}}

Konrad Godlewski, Bartosz Sawicki. (2023)  
**Two-Step Reinforcement Learning for Multistage Strategy Card Game**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-GT, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.17305v1)  

---


**ABSTRACT**  
In the realm of artificial intelligence and card games, this study introduces a two-step reinforcement learning (RL) strategy tailored for "The Lord of the Rings: The Card Game (LOTRCG)," a complex multistage strategy card game. This research diverges from conventional RL methods by adopting a phased learning approach, beginning with a foundational learning stage in a simplified version of the game and subsequently progressing to the complete, intricate game environment. This methodology notably enhances the AI agent's adaptability and performance in the face of LOTRCG's unpredictable and challenging nature. The paper also explores a multi-agent system, where distinct RL agents are employed for various decision-making aspects of the game. This approach has demonstrated a remarkable improvement in game outcomes, with the RL agents achieving a winrate of 78.5% across a set of 10,000 random games.

{{</citation>}}


## cs.NI (2)



### (151/155) Large Language Models for Networking: Applications, Enabling Techniques, and Challenges (Yudong Huang et al., 2023)

{{<citation>}}

Yudong Huang, Hongyang Du, Xinyuan Zhang, Dusit Niyato, Jiawen Kang, Zehui Xiong, Shuo Wang, Tao Huang. (2023)  
**Large Language Models for Networking: Applications, Enabling Techniques, and Challenges**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17474v1)  

---


**ABSTRACT**  
The rapid evolution of network technologies and the growing complexity of network tasks necessitate a paradigm shift in how networks are designed, configured, and managed. With a wealth of knowledge and expertise, large language models (LLMs) are one of the most promising candidates. This paper aims to pave the way for constructing domain-adapted LLMs for networking. Firstly, we present potential LLM applications for vertical network fields and showcase the mapping from natural language to network language. Then, several enabling technologies are investigated, including parameter-efficient finetuning and prompt engineering. The insight is that language understanding and tool usage are both required for network LLMs. Driven by the idea of embodied intelligence, we propose the ChatNet, a domain-adapted network LLM framework with access to various external network tools. ChatNet can reduce the time required for burdensome network planning tasks significantly, leading to a substantial improvement in efficiency. Finally, key challenges and future research directions are highlighted.

{{</citation>}}


### (152/155) Wireless Network Digital Twin for 6G: Generative AI as A Key Enabler (Zhenyu Tao et al., 2023)

{{<citation>}}

Zhenyu Tao, Wei Xu, Yongming Huang, Xiaoyun Wang, Xiaohu You. (2023)  
**Wireless Network Digital Twin for 6G: Generative AI as A Key Enabler**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.17451v1)  

---


**ABSTRACT**  
Digital twin, which enables emulation, evaluation, and optimization of physical entities through synchronized digital replicas, has gained increasingly attention as a promising technology for intricate wireless networks. For 6G, numerous innovative wireless technologies and network architectures have posed new challenges in establishing wireless network digital twins. To tackle these challenges, artificial intelligence (AI), particularly the flourishing generative AI, emerges as a potential solution. In this article, we discuss emerging prerequisites for wireless network digital twins considering the complicated network architecture, tremendous network scale, extensive coverage, and diversified application scenarios in the 6G era. We further explore the applications of generative AI, such as transformer and diffusion model, to empower the 6G digital twin from multiple perspectives including implementation, physical-digital synchronization, and slicing capability. Subsequently, we propose a hierarchical generative AI-enabled wireless network digital twin at both the message-level and policy-level, and provide a typical use case with numerical results to validate the effectiveness and efficiency. Finally, open research issues for wireless network digital twins in the 6G era are discussed.

{{</citation>}}


## cs.OS (1)



### (153/155) Cascade: A Platform for Delay-Sensitive Edge Intelligence (Weijia Song et al., 2023)

{{<citation>}}

Weijia Song, Thiago Garrett, Yuting Yang, Mingzhao Liu, Edward Tremel, Lorenzo Rosa, Andrea Merlina, Roman Vitenberg, Ken Birman. (2023)  
**Cascade: A Platform for Delay-Sensitive Edge Intelligence**  

---
Primary Category: cs.OS  
Categories: cs-AI, cs-OS, cs.OS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17329v1)  

---


**ABSTRACT**  
Interactive intelligent computing applications are increasingly prevalent, creating a need for AI/ML platforms optimized to reduce per-event latency while maintaining high throughput and efficient resource management. Yet many intelligent applications run on AI/ML platforms that optimize for high throughput even at the cost of high tail-latency. Cascade is a new AI/ML hosting platform intended to untangle this puzzle. Innovations include a legacy-friendly storage layer that moves data with minimal copying and a "fast path" that collocates data and computation to maximize responsiveness. Our evaluation shows that Cascade reduces latency by orders of magnitude with no loss of throughput.

{{</citation>}}


## cs.DB (1)



### (154/155) Analyzing Query Optimizer Performance in the Presence and Absence of Cardinality Estimates (Asoke Datta et al., 2023)

{{<citation>}}

Asoke Datta, Brian Tsan, Yesdaulet Izenov, Florin Rusu. (2023)  
**Analyzing Query Optimizer Performance in the Presence and Absence of Cardinality Estimates**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17293v1)  

---


**ABSTRACT**  
Most query optimizers rely on cardinality estimates to determine optimal execution plans. While traditional databases such as PostgreSQL, Oracle, and Db2 utilize many types of synopses -- including histograms, samples, and sketches -- recent main-memory databases like DuckDB and Heavy.AI often operate with minimal or no estimates, yet their performance does not necessarily suffer. To the best of our knowledge, no analytical comparison has been conducted between optimizers with and without cardinality estimates to understand their performance characteristics in different settings, such as indexed, non-indexed, and multi-threaded. In this paper, we present a comparative analysis between optimizers that use cardinality estimates and those that do not. We use the Join Order Benchmark (JOB) for our evaluation and true cardinalities as the baseline. Our investigation reveals that cardinality estimates have marginal impact in non-indexed settings. Meanwhile, when indexes are available, inaccurate estimates may lead to sub-optimal physical operators -- even with an optimal join order. Furthermore, the impact of cardinality estimates is less significant in highly-parallel main-memory databases.

{{</citation>}}


## stat.ML (1)



### (155/155) Is Inverse Reinforcement Learning Harder than Standard Reinforcement Learning? (Lei Zhao et al., 2023)

{{<citation>}}

Lei Zhao, Mengdi Wang, Yu Bai. (2023)  
**Is Inverse Reinforcement Learning Harder than Standard Reinforcement Learning?**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.00054v1)  

---


**ABSTRACT**  
Inverse Reinforcement Learning (IRL) -- the problem of learning reward functions from demonstrations of an \emph{expert policy} -- plays a critical role in developing intelligent systems, such as those that understand and imitate human behavior. While widely used in applications, theoretical understandings of IRL admit unique challenges and remain less developed compared with standard RL theory. For example, it remains open how to do IRL efficiently in standard \emph{offline} settings with pre-collected data, where states are obtained from a \emph{behavior policy} (which could be the expert policy itself), and actions are sampled from the expert policy.   This paper provides the first line of results for efficient IRL in vanilla offline and online settings using polynomial samples and runtime. We first design a new IRL algorithm for the offline setting, Reward Learning with Pessimism (RLP), and show that it achieves polynomial sample complexity in terms of the size of the MDP, a concentrability coefficient between the behavior policy and the expert policy, and the desired accuracy. Building on RLP, we further design an algorithm Reward Learning with Exploration (RLE), which operates in a natural online setting where the learner can both actively explore the environment and query the expert policy, and obtain a stronger notion of IRL guarantee from polynomial samples. We establish sample complexity lower bounds for both settings showing that RLP and RLE are nearly optimal. Finally, as an application, we show that the learned reward functions can \emph{transfer} to another target MDP with suitable guarantees when the target MDP satisfies certain similarity assumptions with the original (source) MDP.

{{</citation>}}
