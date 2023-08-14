---
draft: false
title: "arXiv @ 2023.08.10"
date: 2023-08-10
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.10"
    identifier: arxiv_20230810
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (24)](#cscl-24)
- [cs.AI (12)](#csai-12)
- [cs.LG (11)](#cslg-11)
- [cs.CR (3)](#cscr-3)
- [cs.CV (35)](#cscv-35)
- [cs.IR (3)](#csir-3)
- [cs.NI (3)](#csni-3)
- [eess.IV (3)](#eessiv-3)
- [q-bio.GN (1)](#q-biogn-1)
- [cs.GT (1)](#csgt-1)
- [cs.HC (6)](#cshc-6)
- [cs.IT (2)](#csit-2)
- [cs.RO (2)](#csro-2)
- [cs.SE (1)](#csse-1)
- [cs.SI (1)](#cssi-1)
- [cs.SD (2)](#cssd-2)
- [cs.MM (1)](#csmm-1)
- [cs.MA (2)](#csma-2)
- [cs.NE (1)](#csne-1)
- [cond-mat.soft (1)](#cond-matsoft-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [math.OC (1)](#mathoc-1)
- [stat.ML (1)](#statml-1)
- [cs.DB (1)](#csdb-1)

## cs.CL (24)



### (1/119) A Comparative Study of Sentence Embedding Models for Assessing Semantic Variation (Deven M. Mistry et al., 2023)

{{<citation>}}

Deven M. Mistry, Ali A. Minai. (2023)  
**A Comparative Study of Sentence Embedding Models for Assessing Semantic Variation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2308.04625v1)  

---


**ABSTRACT**  
Analyzing the pattern of semantic variation in long real-world texts such as books or transcripts is interesting from the stylistic, cognitive, and linguistic perspectives. It is also useful for applications such as text segmentation, document summarization, and detection of semantic novelty. The recent emergence of several vector-space methods for sentence embedding has made such analysis feasible. However, this raises the issue of how consistent and meaningful the semantic representations produced by various methods are in themselves. In this paper, we compare several recent sentence embedding methods via time-series of semantic similarity between successive sentences and matrices of pairwise sentence similarity for multiple books of literature. In contrast to previous work using target tasks and curated datasets to compare sentence embedding methods, our approach provides an evaluation of the methods 'in the wild'. We find that most of the sentence embedding methods considered do infer highly correlated patterns of semantic similarity in a given document, but show interesting differences.

{{</citation>}}


### (2/119) Benchmarking LLM powered Chatbots: Methods and Metrics (Debarag Banerjee et al., 2023)

{{<citation>}}

Debarag Banerjee, Pooja Singh, Arjun Avadhanam, Saksham Srivastava. (2023)  
**Benchmarking LLM powered Chatbots: Methods and Metrics**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.04624v1)  

---


**ABSTRACT**  
Autonomous conversational agents, i.e. chatbots, are becoming an increasingly common mechanism for enterprises to provide support to customers and partners. In order to rate chatbots, especially ones powered by Generative AI tools like Large Language Models (LLMs) we need to be able to accurately assess their performance. This is where chatbot benchmarking becomes important. In this paper, we propose the use of a novel benchmark that we call the E2E (End to End) benchmark, and show how the E2E benchmark can be used to evaluate accuracy and usefulness of the answers provided by chatbots, especially ones powered by LLMs. We evaluate an example chatbot at different levels of sophistication based on both our E2E benchmark, as well as other available metrics commonly used in the state of art, and observe that the proposed benchmark show better results compared to others. In addition, while some metrics proved to be unpredictable, the metric associated with the E2E benchmark, which uses cosine similarity performed well in evaluating chatbots. The performance of our best models shows that there are several benefits of using the cosine similarity score as a metric in the E2E benchmark.

{{</citation>}}


### (3/119) Shepherd: A Critic for Language Model Generation (Tianlu Wang et al., 2023)

{{<citation>}}

Tianlu Wang, Ping Yu, Xiaoqing Ellen Tan, Sean O'Brien, Ramakanth Pasunuru, Jane Dwivedi-Yu, Olga Golovneva, Luke Zettlemoyer, Maryam Fazel-Zarandi, Asli Celikyilmaz. (2023)  
**Shepherd: A Critic for Language Model Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.04592v1)  

---


**ABSTRACT**  
As large language models improve, there is increasing interest in techniques that leverage these models' capabilities to refine their own outputs. In this work, we introduce Shepherd, a language model specifically tuned to critique responses and suggest refinements, extending beyond the capabilities of an untuned model to identify diverse errors and provide suggestions to remedy them. At the core of our approach is a high quality feedback dataset, which we curate from community feedback and human annotations. Even though Shepherd is small (7B parameters), its critiques are either equivalent or preferred to those from established models including ChatGPT. Using GPT-4 for evaluation, Shepherd reaches an average win-rate of 53-87% compared to competitive alternatives. In human evaluation, Shepherd strictly outperforms other models and on average closely ties with ChatGPT.

{{</citation>}}


### (4/119) Single-Sentence Reader: A Novel Approach for Addressing Answer Position Bias (Son Quoc Tran et al., 2023)

{{<citation>}}

Son Quoc Tran, Matt Kretchmar. (2023)  
**Single-Sentence Reader: A Novel Approach for Addressing Answer Position Bias**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Machine Reading Comprehension  
[Paper Link](http://arxiv.org/abs/2308.04566v1)  

---


**ABSTRACT**  
Machine Reading Comprehension (MRC) models tend to take advantage of spurious correlations (also known as dataset bias or annotation artifacts in the research community). Consequently, these models may perform the MRC task without fully comprehending the given context and question, which is undesirable since it may result in low robustness against distribution shift. This paper delves into the concept of answer-position bias, where a significant percentage of training questions have answers located solely in the first sentence of the context. We propose a Single-Sentence Reader as a new approach for addressing answer position bias in MRC. We implement this approach using six different models and thoroughly analyze their performance. Remarkably, our proposed Single-Sentence Readers achieve results that nearly match those of models trained on conventional training sets, proving their effectiveness. Our study also discusses several challenges our Single-Sentence Readers encounter and proposes a potential solution.

{{</citation>}}


### (5/119) Ahead of the Text: Leveraging Entity Preposition for Financial Relation Extraction (Stefan Pasch et al., 2023)

{{<citation>}}

Stefan Pasch, Dimitrios Petridis. (2023)  
**Ahead of the Text: Leveraging Entity Preposition for Financial Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Financial, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2308.04534v1)  

---


**ABSTRACT**  
In the context of the ACM KDF-SIGIR 2023 competition, we undertook an entity relation task on a dataset of financial entity relations called REFind. Our top-performing solution involved a multi-step approach. Initially, we inserted the provided entities at their corresponding locations within the text. Subsequently, we fine-tuned the transformer-based language model roberta-large for text classification by utilizing a labeled training set to predict the entity relations. Lastly, we implemented a post-processing phase to identify and handle improbable predictions generated by the model. As a result of our methodology, we achieved the 1st place ranking on the competition's public leaderboard.

{{</citation>}}


### (6/119) Revisiting Disentanglement and Fusion on Modality and Context in Conversational Multimodal Emotion Recognition (Bobo Li et al., 2023)

{{<citation>}}

Bobo Li, Hao Fei, Lizi Liao, Yu Zhao, Chong Teng, Tat-Seng Chua, Donghong Ji, Fei Li. (2023)  
**Revisiting Disentanglement and Fusion on Modality and Context in Conversational Multimodal Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.04502v1)  

---


**ABSTRACT**  
It has been a hot research topic to enable machines to understand human emotions in multimodal contexts under dialogue scenarios, which is tasked with multimodal emotion analysis in conversation (MM-ERC). MM-ERC has received consistent attention in recent years, where a diverse range of methods has been proposed for securing better task performance. Most existing works treat MM-ERC as a standard multimodal classification problem and perform multimodal feature disentanglement and fusion for maximizing feature utility. Yet after revisiting the characteristic of MM-ERC, we argue that both the feature multimodality and conversational contextualization should be properly modeled simultaneously during the feature disentanglement and fusion steps. In this work, we target further pushing the task performance by taking full consideration of the above insights. On the one hand, during feature disentanglement, based on the contrastive learning technique, we devise a Dual-level Disentanglement Mechanism (DDM) to decouple the features into both the modality space and utterance space. On the other hand, during the feature fusion stage, we propose a Contribution-aware Fusion Mechanism (CFM) and a Context Refusion Mechanism (CRM) for multimodal and context integration, respectively. They together schedule the proper integrations of multimodal and context features. Specifically, CFM explicitly manages the multimodal feature contributions dynamically, while CRM flexibly coordinates the introduction of dialogue contexts. On two public MM-ERC datasets, our system achieves new state-of-the-art performance consistently. Further analyses demonstrate that all our proposed mechanisms greatly facilitate the MM-ERC task by making full use of the multimodal and context features adaptively. Note that our proposed methods have the great potential to facilitate a broader range of other conversational multimodal tasks.

{{</citation>}}


### (7/119) DialogRE^C+: An Extension of DialogRE to Investigate How Much Coreference Helps Relation Extraction in Dialogs (Yiyun Xiong et al., 2023)

{{<citation>}}

Yiyun Xiong, Mengwei Dai, Fei Li, Hao Fei, Bobo Li, Shengqiong Wu, Donghong Ji, Chong Teng. (2023)  
**DialogRE^C+: An Extension of DialogRE to Investigate How Much Coreference Helps Relation Extraction in Dialogs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2308.04498v1)  

---


**ABSTRACT**  
Dialogue relation extraction (DRE) that identifies the relations between argument pairs in dialogue text, suffers much from the frequent occurrence of personal pronouns, or entity and speaker coreference. This work introduces a new benchmark dataset DialogRE^C+, introducing coreference resolution into the DRE scenario. With the aid of high-quality coreference knowledge, the reasoning of argument relations is expected to be enhanced. In DialogRE^C+ dataset, we manually annotate total 5,068 coreference chains over 36,369 argument mentions based on the existing DialogRE data, where four different coreference chain types namely speaker chain, person chain, location chain and organization chain are explicitly marked. We further develop 4 coreference-enhanced graph-based DRE models, which learn effective coreference representations for improving the DRE task. We also train a coreference resolution model based on our annotations and evaluate the effect of automatically extracted coreference chains demonstrating the practicality of our dataset and its potential to other domains and tasks.

{{</citation>}}


### (8/119) SILO Language Models: Isolating Legal Risk In a Nonparametric Datastore (Sewon Min et al., 2023)

{{<citation>}}

Sewon Min, Suchin Gururangan, Eric Wallace, Hannaneh Hajishirzi, Noah A. Smith, Luke Zettlemoyer. (2023)  
**SILO Language Models: Isolating Legal Risk In a Nonparametric Datastore**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2308.04430v1)  

---


**ABSTRACT**  
The legality of training language models (LMs) on copyrighted or otherwise restricted data is under intense debate. However, as we show, model performance significantly degrades if trained only on low-risk text (e.g., out-of-copyright books or government documents), due to its limited size and domain coverage. We present SILO, a new language model that manages this risk-performance tradeoff during inference. SILO is built by (1) training a parametric LM on Open License Corpus (OLC), a new corpus we curate with 228B tokens of public domain and permissively licensed text and (2) augmenting it with a more general and easily modifiable nonparametric datastore (e.g., containing copyrighted books or news) that is only queried during inference. The datastore allows use of high-risk data without training on it, supports sentence-level data attribution, and enables data producers to opt out from the model by removing content from the store. These capabilities can foster compliance with data-use regulations such as the fair use doctrine in the United States and the GDPR in the European Union. Our experiments show that the parametric LM struggles on domains not covered by OLC. However, access to the datastore greatly improves out of domain performance, closing 90% of the performance gap with an LM trained on the Pile, a more diverse corpus with mostly high-risk text. We also analyze which nonparametric approach works best, where the remaining errors lie, and how performance scales with datastore size. Our results suggest that it is possible to build high quality language models while mitigating their legal risk.

{{</citation>}}


### (9/119) A Bi-directional Multi-hop Inference Model for Joint Dialog Sentiment Classification and Act Recognition (Li Zheng et al., 2023)

{{<citation>}}

Li Zheng, Fei Li, Yuyang Chai, Chong Teng, Donghong Ji. (2023)  
**A Bi-directional Multi-hop Inference Model for Joint Dialog Sentiment Classification and Act Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog  
[Paper Link](http://arxiv.org/abs/2308.04424v1)  

---


**ABSTRACT**  
The joint task of Dialog Sentiment Classification (DSC) and Act Recognition (DAR) aims to predict the sentiment label and act label for each utterance in a dialog simultaneously. However, current methods encode the dialog context in only one direction, which limits their ability to thoroughly comprehend the context. Moreover, these methods overlook the explicit correlations between sentiment and act labels, which leads to an insufficient ability to capture rich sentiment and act clues and hinders effective and accurate reasoning. To address these issues, we propose a Bi-directional Multi-hop Inference Model (BMIM) that leverages a feature selection network and a bi-directional multi-hop inference network to iteratively extract and integrate rich sentiment and act clues in a bi-directional manner. We also employ contrastive learning and dual learning to explicitly model the correlations of sentiment and act labels. Our experiments on two widely-used datasets show that BMIM outperforms state-of-the-art baselines by at least 2.6% on F1 score in DAR and 1.4% on F1 score in DSC. Additionally, Our proposed model not only improves the performance but also enhances the interpretability of the joint sentiment and act prediction task.

{{</citation>}}


### (10/119) Character-level NMT and language similarity (Josef Jon et al., 2023)

{{<citation>}}

Josef Jon, Ondřej Bojar. (2023)  
**Character-level NMT and language similarity**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.04398v1)  

---


**ABSTRACT**  
We explore the effectiveness of character-level neural machine translation using Transformer architecture for various levels of language similarity and size of the training dataset on translation between Czech and Croatian, German, Hungarian, Slovak, and Spanish. We evaluate the models using automatic MT metrics and show that translation between similar languages benefits from character-level input segmentation, while for less related languages, character-level vanilla Transformer-base often lags behind subword-level segmentation. We confirm previous findings that it is possible to close the gap by finetuning the already trained subword-level models to character-level.

{{</citation>}}


### (11/119) Learning Evaluation Models from Large Language Models for Sequence Generation (Chenglong Wang et al., 2023)

{{<citation>}}

Chenglong Wang, Hang Zhou, Kaiyan Chang, Tongran Liu, Chunliang Zhang, Quan Du, Tong Xiao, Jingbo Zhu. (2023)  
**Learning Evaluation Models from Large Language Models for Sequence Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.04386v1)  

---


**ABSTRACT**  
Large language models achieve state-of-the-art performance on sequence generation evaluation, but typically have a large number of parameters. This is a computational challenge as presented by applying their evaluation capability at scale. To overcome the challenge, in this paper, we propose \textbf{ECT}, an \textbf{e}valuation \textbf{c}apability \textbf{t}ransfer method, to transfer the evaluation capability from LLMs to relatively lightweight language models. Based on the proposed ECT, we learn various evaluation models from ChatGPT, and employ them as reward models to improve sequence generation models via reinforcement learning and reranking approaches. Experimental results on machine translation, text style transfer, and summarization tasks demonstrate the effectiveness of our ECT. Notably, applying the learned evaluation models to sequence generation models results in better generated sequences as evaluated by commonly used metrics and ChatGPT.

{{</citation>}}


### (12/119) Unmasking Nationality Bias: A Study of Human Perception of Nationalities in AI-Generated Articles (Pranav Narayanan Venkit et al., 2023)

{{<citation>}}

Pranav Narayanan Venkit, Sanjana Gautam, Ruchi Panchanadikar, Ting-Hao `Kenneth' Huang, Shomir Wilson. (2023)  
**Unmasking Nationality Bias: A Study of Human Perception of Nationalities in AI-Generated Articles**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: AI, Bias, NLP  
[Paper Link](http://arxiv.org/abs/2308.04346v1)  

---


**ABSTRACT**  
We investigate the potential for nationality biases in natural language processing (NLP) models using human evaluation methods. Biased NLP models can perpetuate stereotypes and lead to algorithmic discrimination, posing a significant challenge to the fairness and justice of AI systems. Our study employs a two-step mixed-methods approach that includes both quantitative and qualitative analysis to identify and understand the impact of nationality bias in a text generation model. Through our human-centered quantitative analysis, we measure the extent of nationality bias in articles generated by AI sources. We then conduct open-ended interviews with participants, performing qualitative coding and thematic analysis to understand the implications of these biases on human readers. Our findings reveal that biased NLP models tend to replicate and amplify existing societal biases, which can translate to harm if used in a sociotechnical setting. The qualitative analysis from our interviews offers insights into the experience readers have when encountering such articles, highlighting the potential to shift a reader's perception of a country. These findings emphasize the critical role of public perception in shaping AI's impact on society and the need to correct biases in AI systems.

{{</citation>}}


### (13/119) In-Context Alignment: Chat with Vanilla Language Models Before Fine-Tuning (Xiaochuang Han, 2023)

{{<citation>}}

Xiaochuang Han. (2023)  
**In-Context Alignment: Chat with Vanilla Language Models Before Fine-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.04275v1)  

---


**ABSTRACT**  
In this note, we explore inference-time alignment through in-context learning. We consider a vanilla pretrained language model Llama-2 before any fine-tuning and retrieve an average of 9 demonstration alignment examples when the model is prompted to follow chat-style instructions. Compared to direct prompting, the in-context alignment without changing model weights leads to a 7x increase in win-rate w.r.t. the text-davinci-003 model from OpenAI, making the vanilla language model comparable to strong baselines with alignment fine-tuning.

{{</citation>}}


### (14/119) Gloss Alignment Using Word Embeddings (Harry Walsh et al., 2023)

{{<citation>}}

Harry Walsh, Ozge Mercanoglu Sincan, Ben Saunders, Richard Bowden. (2023)  
**Gloss Alignment Using Word Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU, Embedding, Word Embedding  
[Paper Link](http://arxiv.org/abs/2308.04248v1)  

---


**ABSTRACT**  
Capturing and annotating Sign language datasets is a time consuming and costly process. Current datasets are orders of magnitude too small to successfully train unconstrained \acf{slt} models. As a result, research has turned to TV broadcast content as a source of large-scale training data, consisting of both the sign language interpreter and the associated audio subtitle. However, lack of sign language annotation limits the usability of this data and has led to the development of automatic annotation techniques such as sign spotting. These spottings are aligned to the video rather than the subtitle, which often results in a misalignment between the subtitle and spotted signs. In this paper we propose a method for aligning spottings with their corresponding subtitles using large spoken language models. Using a single modality means our method is computationally inexpensive and can be utilized in conjunction with existing alignment techniques. We quantitatively demonstrate the effectiveness of our method on the \acf{mdgs} and \acf{bobsl} datasets, recovering up to a 33.22 BLEU-1 score in word alignment.

{{</citation>}}


### (15/119) Hybrid Retrieval-Augmented Generation for Real-time Composition Assistance (Xuchao Zhang et al., 2023)

{{<citation>}}

Xuchao Zhang, Menglin Xia, Camille Couturier, Guoqing Zheng, Saravan Rajmohan, Victor Ruhle. (2023)  
**Hybrid Retrieval-Augmented Generation for Real-time Composition Assistance**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DC, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.04215v1)  

---


**ABSTRACT**  
Retrieval augmented models show promise in enhancing traditional language models by improving their contextual understanding, integrating private data, and reducing hallucination. However, the processing time required for retrieval augmented large language models poses a challenge when applying them to tasks that require real-time responses, such as composition assistance.   To overcome this limitation, we propose the Hybrid Retrieval-Augmented Generation (HybridRAG) framework that leverages a hybrid setting that combines both client and cloud models. HybridRAG incorporates retrieval-augmented memory generated asynchronously by a Large Language Model (LLM) in the cloud. By integrating this retrieval augmented memory, the client model acquires the capability to generate highly effective responses, benefiting from the LLM's capabilities. Furthermore, through asynchronous memory integration, the client model is capable of delivering real-time responses to user requests without the need to wait for memory synchronization from the cloud. Our experiments on Wikitext and Pile subsets show that HybridRAG achieves lower latency than a cloud-based retrieval-augmented LLM, while outperforming client-only models in utility.

{{</citation>}}


### (16/119) On Monotonic Aggregation for Open-domain QA (Sang-eun Han et al., 2023)

{{<citation>}}

Sang-eun Han, Yeonseok Jeong, Seung-won Hwang, Kyungjae Lee. (2023)  
**On Monotonic Aggregation for Open-domain QA**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.04176v1)  

---


**ABSTRACT**  
Question answering (QA) is a critical task for speech-based retrieval from knowledge sources, by sifting only the answers without requiring to read supporting documents. Specifically, open-domain QA aims to answer user questions on unrestricted knowledge sources. Ideally, adding a source should not decrease the accuracy, but we find this property (denoted as "monotonicity") does not hold for current state-of-the-art methods. We identify the cause, and based on that we propose Judge-Specialist framework. Our framework consists of (1) specialist retrievers/readers to cover individual sources, and (2) judge, a dedicated language model to select the final answer. Our experiments show that our framework not only ensures monotonicity, but also outperforms state-of-the-art multi-source QA methods on Natural Questions. Additionally, we show that our models robustly preserve the monotonicity against noise from speech recognition. We publicly release our code and setting.

{{</citation>}}


### (17/119) Large Language Model Prompt Chaining for Long Legal Document Classification (Dietrich Trautmann, 2023)

{{<citation>}}

Dietrich Trautmann. (2023)  
**Large Language Model Prompt Chaining for Long Legal Document Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2308.04138v1)  

---


**ABSTRACT**  
Prompting is used to guide or steer a language model in generating an appropriate response that is consistent with the desired outcome. Chaining is a strategy used to decompose complex tasks into smaller, manageable components. In this study, we utilize prompt chaining for extensive legal document classification tasks, which present difficulties due to their intricate domain-specific language and considerable length. Our approach begins with the creation of a concise summary of the original document, followed by a semantic search for related exemplar texts and their corresponding annotations from a training corpus. Finally, we prompt for a label - based on the task - to assign, by leveraging the in-context learning from the few-shot prompt. We demonstrate that through prompt chaining, we can not only enhance the performance over zero-shot, but also surpass the micro-F1 score achieved by larger models, such as ChatGPT zero-shot, using smaller models.

{{</citation>}}


### (18/119) Social Media, Topic Modeling and Sentiment Analysis in Municipal Decision Support (Miloš Švaňa, 2023)

{{<citation>}}

Miloš Švaňa. (2023)  
**Social Media, Topic Modeling and Sentiment Analysis in Municipal Decision Support**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Sentiment Analysis, Social Media, Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2308.04124v1)  

---


**ABSTRACT**  
Many cities around the world are aspiring to become. However, smart initiatives often give little weight to the opinions of average citizens.   Social media are one of the most important sources of citizen opinions. This paper presents a prototype of a framework for processing social media posts with municipal decision-making in mind. The framework consists of a sequence of three steps: (1) determining the sentiment polarity of each social media post (2) identifying prevalent topics and mapping these topics to individual posts, and (3) aggregating these two pieces of information into a fuzzy number representing the overall sentiment expressed towards each topic. Optionally, the fuzzy number can be reduced into a tuple of two real numbers indicating the "amount" of positive and negative opinion expressed towards each topic.   The framework is demonstrated on tweets published from Ostrava, Czechia over a period of about two months. This application illustrates how fuzzy numbers represent sentiment in a richer way and capture the diversity of opinions expressed on social media.

{{</citation>}}


### (19/119) Collective Human Opinions in Semantic Textual Similarity (Yuxia Wang et al., 2023)

{{<citation>}}

Yuxia Wang, Shimin Tao, Ning Xie, Hao Yang, Timothy Baldwin, Karin Verspoor. (2023)  
**Collective Human Opinions in Semantic Textual Similarity**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Textual Similarity  
[Paper Link](http://arxiv.org/abs/2308.04114v1)  

---


**ABSTRACT**  
Despite the subjective nature of semantic textual similarity (STS) and pervasive disagreements in STS annotation, existing benchmarks have used averaged human ratings as the gold standard. Averaging masks the true distribution of human opinions on examples of low agreement, and prevents models from capturing the semantic vagueness that the individual ratings represent. In this work, we introduce USTS, the first Uncertainty-aware STS dataset with ~15,000 Chinese sentence pairs and 150,000 labels, to study collective human opinions in STS. Analysis reveals that neither a scalar nor a single Gaussian fits a set of observed judgements adequately. We further show that current STS models cannot capture the variance caused by human disagreement on individual instances, but rather reflect the predictive confidence over the aggregate dataset.

{{</citation>}}


### (20/119) I-WAS: a Data Augmentation Method with GPT-2 for Simile Detection (Yongzhu Chang et al., 2023)

{{<citation>}}

Yongzhu Chang, Rongsheng Zhang, Jiashu Pu. (2023)  
**I-WAS: a Data Augmentation Method with GPT-2 for Simile Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2308.04109v1)  

---


**ABSTRACT**  
Simile detection is a valuable task for many natural language processing (NLP)-based applications, particularly in the field of literature. However, existing research on simile detection often relies on corpora that are limited in size and do not adequately represent the full range of simile forms. To address this issue, we propose a simile data augmentation method based on \textbf{W}ord replacement And Sentence completion using the GPT-2 language model. Our iterative process called I-WAS, is designed to improve the quality of the augmented sentences. To better evaluate the performance of our method in real-world applications, we have compiled a corpus containing a more diverse set of simile forms for experimentation. Our experimental results demonstrate the effectiveness of our proposed data augmentation method for simile detection.

{{</citation>}}


### (21/119) A Comparative Study on TF-IDF feature Weighting Method and its Analysis using Unstructured Dataset (Mamata Das et al., 2023)

{{<citation>}}

Mamata Das, Selvakumar K., P. J. A. Alphonse. (2023)  
**A Comparative Study on TF-IDF feature Weighting Method and its Analysis using Unstructured Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Amazon, NLP, Natural Language Processing, Text Classification  
[Paper Link](http://arxiv.org/abs/2308.04037v1)  

---


**ABSTRACT**  
Text Classification is the process of categorizing text into the relevant categories and its algorithms are at the core of many Natural Language Processing (NLP). Term Frequency-Inverse Document Frequency (TF-IDF) and NLP are the most highly used information retrieval methods in text classification. We have investigated and analyzed the feature weighting method for text classification on unstructured data. The proposed model considered two features N-Grams and TF-IDF on the IMDB movie reviews and Amazon Alexa reviews dataset for sentiment analysis. Then we have used the state-of-the-art classifier to validate the method i.e., Support Vector Machine (SVM), Logistic Regression, Multinomial Naive Bayes (Multinomial NB), Random Forest, Decision Tree, and k-nearest neighbors (KNN). From those two feature extractions, a significant increase in feature extraction with TF-IDF features rather than based on N-Gram. TF-IDF got the maximum accuracy (93.81%), precision (94.20%), recall (93.81%), and F1-score (91.99%) value in Random Forest classifier.

{{</citation>}}


### (22/119) Top K Relevant Passage Retrieval for Biomedical Question Answering (Shashank Gupta, 2023)

{{<citation>}}

Shashank Gupta. (2023)  
**Top K Relevant Passage Retrieval for Biomedical Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.04028v1)  

---


**ABSTRACT**  
Question answering is a task that answers factoid questions using a large collection of documents. It aims to provide precise answers in response to the user's questions in natural language. Question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. On the web, there is no single article that could provide all the possible answers available on the internet to the question of the problem asked by the user. The existing Dense Passage Retrieval model has been trained on Wikipedia dump from Dec. 20, 2018, as the source documents for answering questions. Question answering (QA) has made big strides with several open-domain and machine comprehension systems built using large-scale annotated datasets. However, in the clinical domain, this problem remains relatively unexplored. According to multiple surveys, Biomedical Questions cannot be answered correctly from Wikipedia Articles. In this work, we work on the existing DPR framework for the biomedical domain and retrieve answers from the Pubmed articles which is a reliable source to answer medical questions. When evaluated on a BioASQ QA dataset, our fine-tuned dense retriever results in a 0.81 F1 score.

{{</citation>}}


### (23/119) Continual Pre-Training of Large Language Models: How to (re)warm your model? (Kshitij Gupta et al., 2023)

{{<citation>}}

Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats L. Richter, Quentin Anthony, Eugene Belilovsky, Irina Rish, Timothée Lesort. (2023)  
**Continual Pre-Training of Large Language Models: How to (re)warm your model?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.04014v1)  

---


**ABSTRACT**  
Large language models (LLMs) are routinely pre-trained on billions of tokens, only to restart the process over again once new data becomes available. A much cheaper and more efficient solution would be to enable the continual pre-training of these models, i.e. updating pre-trained models with new data instead of re-training them from scratch. However, the distribution shift induced by novel data typically results in degraded performance on past data. Taking a step towards efficient continual pre-training, in this work, we examine the effect of different warm-up strategies. Our hypothesis is that the learning rate must be re-increased to improve compute efficiency when training on a new dataset. We study the warmup phase of models pre-trained on the Pile (upstream data, 300B tokens) as we continue to pre-train on SlimPajama (downstream data, 297B tokens), following a linear warmup and cosine decay schedule. We conduct all experiments on the Pythia 410M language model architecture and evaluate performance through validation perplexity. We experiment with different pre-training checkpoints, various maximum learning rates, and various warmup lengths. Our results show that while rewarming models first increases the loss on upstream and downstream data, in the longer run it improves the downstream performance, outperforming models trained from scratch$\unicode{x2013}$even for a large downstream dataset.

{{</citation>}}


### (24/119) SimplyRetrieve: A Private and Lightweight Retrieval-Centric Generative AI Tool (Youyang Ng et al., 2023)

{{<citation>}}

Youyang Ng, Daisuke Miyashita, Yasuto Hoshi, Yasuhiro Morioka, Osamu Torii, Tomoya Kodama, Jun Deguchi. (2023)  
**SimplyRetrieve: A Private and Lightweight Retrieval-Centric Generative AI Tool**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03983v1)  

---


**ABSTRACT**  
Large Language Model (LLM) based Generative AI systems have seen significant progress in recent years. Integrating a knowledge retrieval architecture allows for seamless integration of private data into publicly available Generative AI systems using pre-trained LLM without requiring additional model fine-tuning. Moreover, Retrieval-Centric Generation (RCG) approach, a promising future research direction that explicitly separates roles of LLMs and retrievers in context interpretation and knowledge memorization, potentially leads to more efficient implementation. SimplyRetrieve is an open-source tool with the goal of providing a localized, lightweight, and user-friendly interface to these sophisticated advancements to the machine learning community. SimplyRetrieve features a GUI and API based RCG platform, assisted by a Private Knowledge Base Constructor and a Retrieval Tuning Module. By leveraging these capabilities, users can explore the potential of RCG for improving generative AI performance while maintaining privacy standards. The tool is available at https://github.com/RCGAI/SimplyRetrieve with an MIT license.

{{</citation>}}


## cs.AI (12)



### (25/119) Accelerating LLM Inference with Staged Speculative Decoding (Benjamin Spector et al., 2023)

{{<citation>}}

Benjamin Spector, Chris Re. (2023)  
**Accelerating LLM Inference with Staged Speculative Decoding**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2308.04623v1)  

---


**ABSTRACT**  
Recent advances with large language models (LLM) illustrate their diverse capabilities. We propose a novel algorithm, staged speculative decoding, to accelerate LLM inference in small-batch, on-device scenarios. We address the low arithmetic intensity of small-batch inference by improving upon previous work in speculative decoding. First, we restructure the speculative batch as a tree, which reduces generation costs and increases the expected tokens per batch. Second, we add a second stage of speculative decoding. Taken together, we reduce single-batch decoding latency by 3.16x with a 762M parameter GPT-2-L model while perfectly preserving output quality.

{{</citation>}}


### (26/119) Developmental Bootstrapping of AIs (Mark Stefik et al., 2023)

{{<citation>}}

Mark Stefik, Robert Price. (2023)  
**Developmental Bootstrapping of AIs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04586v2)  

---


**ABSTRACT**  
Although some current AIs surpass human abilities especially in closed artificial worlds such as board games, their abilities in the real world are limited. They make strange mistakes and do not notice them. They cannot be instructed easily, fail to use common sense, and lack curiosity. They do not make good collaborators. Mainstream approaches for creating AIs are built using the traditional manually-constructed symbolic AI approach and generative and deep learning AI approaches including large language models (LLMs). These systems are not well suited for creating robust and trustworthy AIs. Although it is outside of the mainstream, the developmental bootstrapping approach has more promise. In developmental bootstrapping, AIs develop competences like human children do. They start with innate competences. They interact with the environment and learn from their interactions. They incrementally extend their innate competences with self-developed competences. They interact and learn from people and establish perceptual, cognitive, and common grounding. They acquire the competences that they need through an incremental bootstrapping process. However, developmental robotics has not yet produced AIs with robust adult-level competences. Projects have typically stopped at the Toddler Barrier corresponding to human infant development at about two years of age, before their speech is fluent. They also do not bridge the Reading Barrier, to skillfully and skeptically tap into the vast socially developed recorded information resources that power LLMs. The next competences in human cognitive development involve intrinsic motivation, imitation learning, imagination, coordination, and communication. This position paper lays out the logic, prospects, gaps, and challenges for extending the practice of developmental bootstrapping to acquire further competences and create robust and resilient AIs.

{{</citation>}}


### (27/119) ChatGPT for Arabic Grammatical Error Correction (Sang Yun Kwon et al., 2023)

{{<citation>}}

Sang Yun Kwon, Gagan Bhatia, El Moatez Billah Nagoud, Muhammad Abdul-Mageed. (2023)  
**ChatGPT for Arabic Grammatical Error Correction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: ChatGPT, GPT, GPT-4, NLP, QA  
[Paper Link](http://arxiv.org/abs/2308.04492v1)  

---


**ABSTRACT**  
Recently, large language models (LLMs) fine-tuned to follow human instruction have exhibited significant capabilities in various English NLP tasks. However, their performance in grammatical error correction (GEC) tasks, particularly in non-English languages, remains significantly unexplored. In this paper, we delve into abilities of instruction fine-tuned LLMs in Arabic GEC, a task made complex due to Arabic's rich morphology. Our findings suggest that various prompting methods, coupled with (in-context) few-shot learning, demonstrate considerable effectiveness, with GPT-4 achieving up to $65.49$ F\textsubscript{1} score under expert prompting (approximately $5$ points higher than our established baseline). This highlights the potential of LLMs in low-resource settings, offering a viable approach for generating useful synthetic data for model training. Despite these positive results, we find that instruction fine-tuned models, regardless of their size, significantly underperform compared to fully fine-tuned models of significantly smaller sizes. This disparity highlights a substantial room for improvements for LLMs. Inspired by methods from low-resource machine translation, we also develop a method exploiting synthetic data that significantly outperforms previous models on two standard Arabic benchmarks. Our work sets new SoTA for Arabic GEC, with $72.19\%$ and $73.26$ F$_{1}$ on the 2014 and 2015 QALB datasets, respectively.

{{</citation>}}


### (28/119) Cumulative Reasoning with Large Language Models (Yifan Zhang et al., 2023)

{{<citation>}}

Yifan Zhang, Jingqin Yang, Yang Yuan, Andrew Chi-Chih Yao. (2023)  
**Cumulative Reasoning with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.04371v3)  

---


**ABSTRACT**  
While language models are powerful and versatile, they often fail to address highly complex problems. This is because solving complex problems requires deliberate thinking, which has been only minimally guided during training. In this paper, we propose a new method called Cumulative Reasoning (CR), which employs language models in a cumulative and iterative manner to emulate human thought processes. By decomposing tasks into smaller components, CR streamlines the problem-solving process, rendering it both more manageable and effective. For logical inference tasks, CR consistently outperforms existing methods with an improvement up to 9.3%, and achieves the astonishing accuracy of 98.04% on the curated FOLIO wiki dataset. In the context of the Game of 24, CR achieves an accuracy of 94%, which signifies a substantial enhancement of 20% over the previous state-of-the-art method (code is available at https://github.com/iiis-ai/cumulative-reasoning).

{{</citation>}}


### (29/119) AutoPCF: Efficient Product Carbon Footprint Accounting with Large Language Models (Zhu Deng et al., 2023)

{{<citation>}}

Zhu Deng, Jinjie Liu, Biao Luo, Can Yuan, Qingrun Yang, Lei Xiao, Wenwen Zhou, Zhu Liu. (2023)  
**AutoPCF: Efficient Product Carbon Footprint Accounting with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.04241v2)  

---


**ABSTRACT**  
The product carbon footprint (PCF) is crucial for decarbonizing the supply chain, as it measures the direct and indirect greenhouse gas emissions caused by all activities during the product's life cycle. However, PCF accounting often requires expert knowledge and significant time to construct life cycle models. In this study, we test and compare the emergent ability of five large language models (LLMs) in modeling the 'cradle-to-gate' life cycles of products and generating the inventory data of inputs and outputs, revealing their limitations as a generalized PCF knowledge database. By utilizing LLMs, we propose an automatic AI-driven PCF accounting framework, called AutoPCF, which also applies deep learning algorithms to automatically match calculation parameters, and ultimately calculate the PCF. The results of estimating the carbon footprint for three case products using the AutoPCF framework demonstrate its potential in achieving automatic modeling and estimation of PCF with a large reduction in modeling time from days to minutes.

{{</citation>}}


### (30/119) Adding Why to What? Analyses of an Everyday Explanation (Lutz Terfloth et al., 2023)

{{<citation>}}

Lutz Terfloth, Michael Schaffer, Heike M. Buhl, Carsten Schulte. (2023)  
**Adding Why to What? Analyses of an Everyday Explanation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.04187v1)  

---


**ABSTRACT**  
In XAI it is important to consider that, in contrast to explanations for professional audiences, one cannot assume common expertise when explaining for laypeople. But such explanations between humans vary greatly, making it difficult to research commonalities across explanations. We used the dual nature theory, a techno-philosophical approach, to cope with these challenges. According to it, one can explain, for example, an XAI's decision by addressing its dual nature: by focusing on the Architecture (e.g., the logic of its algorithms) or the Relevance (e.g., the severity of a decision, the implications of a recommendation). We investigated 20 game explanations using the theory as an analytical framework. We elaborate how we used the theory to quickly structure and compare explanations of technological artifacts. We supplemented results from analyzing the explanation contents with results from a video recall to explore how explainers justified their explanation. We found that explainers were focusing on the physical aspects of the game first (Architecture) and only later on aspects of the Relevance. Reasoning in the video recalls indicated that EX regarded the focus on the Architecture as important for structuring the explanation initially by explaining the basic components before focusing on more complex, intangible aspects. Shifting between addressing the two sides was justified by explanation goals, emerging misunderstandings, and the knowledge needs of the explainee. We discovered several commonalities that inspire future research questions which, if further generalizable, provide first ideas for the construction of synthetic explanations.

{{</citation>}}


### (31/119) Assistive Chatbots for healthcare: a succinct review (Basabdatta Sen Bhattacharya et al., 2023)

{{<citation>}}

Basabdatta Sen Bhattacharya, Vibhav Sinai Pissurlenkar. (2023)  
**Assistive Chatbots for healthcare: a succinct review**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-RO, cs.AI  
Keywords: AI, ChatGPT, GPT, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.04178v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) for supporting healthcare services has never been more necessitated than by the recent global pandemic. Here, we review the state-of-the-art in AI-enabled Chatbots in healthcare proposed during the last 10 years (2013-2023). The focus on AI-enabled technology is because of its potential for enhancing the quality of human-machine interaction via Chatbots, reducing dependence on human-human interaction and saving man-hours. Our review indicates that there are a handful of (commercial) Chatbots that are being used for patient support, while there are others (non-commercial) that are in the clinical trial phases. However, there is a lack of trust on this technology regarding patient safety and data protection, as well as a lack of wider awareness on its benefits among the healthcare workers and professionals. Also, patients have expressed dissatisfaction with Natural Language Processing (NLP) skills of the Chatbots in comparison to humans. Notwithstanding the recent introduction of ChatGPT that has raised the bar for the NLP technology, this Chatbot cannot be trusted with patient safety and medical ethics without thorough and rigorous checks to serve in the `narrow' domain of assistive healthcare. Our review suggests that to enable deployment and integration of AI-enabled Chatbots in public health services, the need of the hour is: to build technology that is simple and safe to use; to build confidence on the technology among: (a) the medical community by focussed training and development; (b) the patients and wider community through outreach.

{{</citation>}}


### (32/119) Predicting Drug-Drug Interactions Using Knowledge Graphs (Lizzy Farrugia et al., 2023)

{{<citation>}}

Lizzy Farrugia, Lilian M. Azzopardi, Jeremy Debattista, Charlie Abela. (2023)  
**Predicting Drug-Drug Interactions Using Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding, GNN, Graph Neural Network, Knowledge Graph, LSTM  
[Paper Link](http://arxiv.org/abs/2308.04172v2)  

---


**ABSTRACT**  
In the last decades, people have been consuming and combining more drugs than before, increasing the number of Drug-Drug Interactions (DDIs). To predict unknown DDIs, recently, studies started incorporating Knowledge Graphs (KGs) since they are able to capture the relationships among entities providing better drug representations than using a single drug property. In this paper, we propose the medicX end-to-end framework that integrates several drug features from public drug repositories into a KG and embeds the nodes in the graph using various translation, factorisation and Neural Network (NN) based KG Embedding (KGE) methods. Ultimately, we use a Machine Learning (ML) algorithm that predicts unknown DDIs. Among the different translation and factorisation-based KGE models, we found that the best performing combination was the ComplEx embedding method with a Long Short-Term Memory (LSTM) network, which obtained an F1-score of 95.19% on a dataset based on the DDIs found in DrugBank version 5.1.8. This score is 5.61% better than the state-of-the-art model DeepDDI. Additionally, we also developed a graph auto-encoder model that uses a Graph Neural Network (GNN), which achieved an F1-score of 91.94%. Consequently, GNNs have demonstrated a stronger ability to mine the underlying semantics of the KG than the ComplEx model, and thus using higher dimension embeddings within the GNN can lead to state-of-the-art performance.

{{</citation>}}


### (33/119) Current and Future Challenges in Knowledge Representation and Reasoning (James P. Delgrande et al., 2023)

{{<citation>}}

James P. Delgrande, Birte Glimm, Thomas Meyer, Miroslaw Truszczynski, Frank Wolter. (2023)  
**Current and Future Challenges in Knowledge Representation and Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.04161v1)  

---


**ABSTRACT**  
Knowledge Representation and Reasoning is a central, longstanding, and active area of Artificial Intelligence. Over the years it has evolved significantly; more recently it has been challenged and complemented by research in areas such as machine learning and reasoning under uncertainty. In July 2022 a Dagstuhl Perspectives workshop was held on Knowledge Representation and Reasoning. The goal of the workshop was to describe the state of the art in the field, including its relation with other areas, its shortcomings and strengths, together with recommendations for future progress. We developed this manifesto based on the presentations, panels, working groups, and discussions that took place at the Dagstuhl Workshop. It is a declaration of our views on Knowledge Representation: its origins, goals, milestones, and current foci; its relation to other disciplines, especially to Artificial Intelligence; and on its challenges, along with key priorities for the next decade.

{{</citation>}}


### (34/119) Gentopia: A Collaborative Platform for Tool-Augmented LLMs (Binfeng Xu et al., 2023)

{{<citation>}}

Binfeng Xu, Xukun Liu, Hua Shen, Zeyu Han, Yuhan Li, Murong Yue, Zhiyuan Peng, Yuchen Liu, Ziyu Yao, Dongkuan Xu. (2023)  
**Gentopia: A Collaborative Platform for Tool-Augmented LLMs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.04030v1)  

---


**ABSTRACT**  
Augmented Language Models (ALMs) empower large language models with the ability to use tools, transforming them into intelligent agents for real-world interactions. However, most existing frameworks for ALMs, to varying degrees, are deficient in the following critical features: flexible customization, collaborative democratization, and holistic evaluation. We present gentopia, an ALM framework enabling flexible customization of agents through simple configurations, seamlessly integrating various language models, task formats, prompting modules, and plugins into a unified paradigm. Furthermore, we establish gentpool, a public platform enabling the registration and sharing of user-customized agents. Agents registered in gentpool are composable such that they can be assembled together for agent collaboration, advancing the democratization of artificial intelligence. To ensure high-quality agents, gentbench, an integral component of gentpool, is designed to thoroughly evaluate user-customized agents across diverse aspects such as safety, robustness, efficiency, etc. We release gentopia on Github and will continuously move forward.

{{</citation>}}


### (35/119) AgentSims: An Open-Source Sandbox for Large Language Model Evaluation (Jiaju Lin et al., 2023)

{{<citation>}}

Jiaju Lin, Haoran Zhao, Aochi Zhang, Yiting Wu, Huqiuyue Ping, Qin Chen. (2023)  
**AgentSims: An Open-Source Sandbox for Large Language Model Evaluation**  

---
Primary Category: cs.AI  
Categories: 14J60 (Primary) 14F05, 14J26 (Secondary) MSC-class: 14J60 (Primary)
  14F05, 14J26 (Secondary) 68T42, cs-AI, cs.AI  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.04026v1)  

---


**ABSTRACT**  
With ChatGPT-like large language models (LLM) prevailing in the community, how to evaluate the ability of LLMs is an open question. Existing evaluation methods suffer from following shortcomings: (1) constrained evaluation abilities, (2) vulnerable benchmarks, (3) unobjective metrics. We suggest that task-based evaluation, where LLM agents complete tasks in a simulated environment, is a one-for-all solution to solve above problems. We present AgentSims, an easy-to-use infrastructure for researchers from all disciplines to test the specific capacities they are interested in. Researchers can build their evaluation tasks by adding agents and buildings on an interactive GUI or deploy and test new support mechanisms, i.e. memory, planning and tool-use systems, by a few lines of codes. Our demo is available at https://agentsims.com .

{{</citation>}}


### (36/119) AI Chatbots as Multi-Role Pedagogical Agents: Transforming Engagement in CS Education (Cassie Chen Cao et al., 2023)

{{<citation>}}

Cassie Chen Cao, Zijian Ding, Jionghao Lin, Frank Hopfgartner. (2023)  
**AI Chatbots as Multi-Role Pedagogical Agents: Transforming Engagement in CS Education**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.03992v1)  

---


**ABSTRACT**  
This study investigates the use of Artificial Intelligence (AI)-powered, multi-role chatbots as a means to enhance learning experiences and foster engagement in computer science education. Leveraging a design-based research approach, we develop, implement, and evaluate a novel learning environment enriched with four distinct chatbot roles: Instructor Bot, Peer Bot, Career Advising Bot, and Emotional Supporter Bot. These roles, designed around the tenets of Self-Determination Theory, cater to the three innate psychological needs of learners - competence, autonomy, and relatedness. Additionally, the system embraces an inquiry-based learning paradigm, encouraging students to ask questions, seek solutions, and explore their curiosities.   We test this system in a higher education context over a period of one month with 200 participating students, comparing outcomes with conditions involving a human tutor and a single chatbot. Our research utilizes a mixed-methods approach, encompassing quantitative measures such as chat log sequence analysis, and qualitative methods including surveys and focus group interviews. By integrating cutting-edge Natural Language Processing techniques such as topic modelling and sentiment analysis, we offer an in-depth understanding of the system's impact on learner engagement, motivation, and inquiry-based learning.   This study, through its rigorous design and innovative approach, provides significant insights into the potential of AI-empowered, multi-role chatbots in reshaping the landscape of computer science education and fostering an engaging, supportive, and motivating learning environment.

{{</citation>}}


## cs.LG (11)



### (37/119) Deep Learning Driven Detection of Tsunami Related Internal GravityWaves: a path towards open-ocean natural hazards detection (Valentino Constantinou et al., 2023)

{{<citation>}}

Valentino Constantinou, Michela Ravanelli, Hamlin Liu, Jacob Bortnik. (2023)  
**Deep Learning Driven Detection of Tsunami Related Internal GravityWaves: a path towards open-ocean natural hazards detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-ao-ph  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2308.04611v1)  

---


**ABSTRACT**  
Tsunamis can trigger internal gravity waves (IGWs) in the ionosphere, perturbing the Total Electron Content (TEC) - referred to as Traveling Ionospheric Disturbances (TIDs) that are detectable through the Global Navigation Satellite System (GNSS). The GNSS are constellations of satellites providing signals from Earth orbit - Europe's Galileo, the United States' Global Positioning System (GPS), Russia's Global'naya Navigatsionnaya Sputnikovaya Sistema (GLONASS) and China's BeiDou. The real-time detection of TIDs provides an approach for tsunami detection, enhancing early warning systems by providing open-ocean coverage in geographic areas not serviceable by buoy-based warning systems. Large volumes of the GNSS data is leveraged by deep learning, which effectively handles complex non-linear relationships across thousands of data streams. We describe a framework leveraging slant total electron content (sTEC) from the VARION (Variometric Approach for Real-Time Ionosphere Observation) algorithm by Gramian Angular Difference Fields (from Computer Vision) and Convolutional Neural Networks (CNNs) to detect TIDs in near-real-time. Historical data from the 2010 Maule, 2011 Tohoku and the 2012 Haida-Gwaii earthquakes and tsunamis are used in model training, and the later-occurring 2015 Illapel earthquake and tsunami in Chile for out-of-sample model validation. Using the experimental framework described in the paper, we achieved a 91.7% F1 score. Source code is available at: https://github.com/vc1492a/tidd. Our work represents a new frontier in detecting tsunami-driven IGWs in open-ocean, dramatically improving the potential for natural hazards detection for coastal communities.

{{</citation>}}


### (38/119) Quantization Aware Factorization for Deep Neural Network Compression (Daria Cherniuk et al., 2023)

{{<citation>}}

Daria Cherniuk, Stanislav Abukhovich, Anh-Huy Phan, Ivan Oseledets, Andrzej Cichocki, Julia Gusak. (2023)  
**Quantization Aware Factorization for Deep Neural Network Compression**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2308.04595v1)  

---


**ABSTRACT**  
Tensor decomposition of convolutional and fully-connected layers is an effective way to reduce parameters and FLOP in neural networks. Due to memory and power consumption limitations of mobile or embedded devices, the quantization step is usually necessary when pre-trained models are deployed. A conventional post-training quantization approach applied to networks with decomposed weights yields a drop in accuracy. This motivated us to develop an algorithm that finds tensor approximation directly with quantized factors and thus benefit from both compression techniques while keeping the prediction quality of the model. Namely, we propose to use Alternating Direction Method of Multipliers (ADMM) for Canonical Polyadic (CP) decomposition with factors whose elements lie on a specified quantization grid. We compress neural network weights with a devised algorithm and evaluate it's prediction quality and performance. We compare our approach to state-of-the-art post-training quantization methods and demonstrate competitive results and high flexibility in achiving a desirable quality-performance tradeoff.

{{</citation>}}


### (39/119) Pelta: Shielding Transformers to Mitigate Evasion Attacks in Federated Learning (Simon Queyrut et al., 2023)

{{<citation>}}

Simon Queyrut, Yérom-David Bromberg, Valerio Schiavoni. (2023)  
**Pelta: Shielding Transformers to Mitigate Evasion Attacks in Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.04373v1)  

---


**ABSTRACT**  
The main premise of federated learning is that machine learning model updates are computed locally, in particular to preserve user data privacy, as those never leave the perimeter of their device. This mechanism supposes the general model, once aggregated, to be broadcast to collaborating and non malicious nodes. However, without proper defenses, compromised clients can easily probe the model inside their local memory in search of adversarial examples. For instance, considering image-based applications, adversarial examples consist of imperceptibly perturbed images (to the human eye) misclassified by the local model, which can be later presented to a victim node's counterpart model to replicate the attack. To mitigate such malicious probing, we introduce Pelta, a novel shielding mechanism leveraging trusted hardware. By harnessing the capabilities of Trusted Execution Environments (TEEs), Pelta masks part of the back-propagation chain rule, otherwise typically exploited by attackers for the design of malicious samples. We evaluate Pelta on a state of the art ensemble model and demonstrate its effectiveness against the Self Attention Gradient adversarial Attack.

{{</citation>}}


### (40/119) Teacher-Student Architecture for Knowledge Distillation: A Survey (Chengming Hu et al., 2023)

{{<citation>}}

Chengming Hu, Xuan Li, Dan Liu, Haolun Wu, Xi Chen, Ju Wang, Xue Liu. (2023)  
**Teacher-Student Architecture for Knowledge Distillation: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.04268v1)  

---


**ABSTRACT**  
Although Deep neural networks (DNNs) have shown a strong capacity to solve large-scale problems in many areas, such DNNs are hard to be deployed in real-world systems due to their voluminous parameters. To tackle this issue, Teacher-Student architectures were proposed, where simple student networks with a few parameters can achieve comparable performance to deep teacher networks with many parameters. Recently, Teacher-Student architectures have been effectively and widely embraced on various knowledge distillation (KD) objectives, including knowledge compression, knowledge expansion, knowledge adaptation, and knowledge enhancement. With the help of Teacher-Student architectures, current studies are able to achieve multiple distillation objectives through lightweight and generalized student networks. Different from existing KD surveys that primarily focus on knowledge compression, this survey first explores Teacher-Student architectures across multiple distillation objectives. This survey presents an introduction to various knowledge representations and their corresponding optimization objectives. Additionally, we provide a systematic overview of Teacher-Student architectures with representative learning algorithms and effective distillation schemes. This survey also summarizes recent applications of Teacher-Student architectures across multiple purposes, including classification, recognition, generation, ranking, and regression. Lastly, potential research directions in KD are investigated, focusing on architecture design, knowledge quality, and theoretical studies of regression-based learning, respectively. Through this comprehensive survey, industry practitioners and the academic community can gain valuable insights and guidelines for effectively designing, learning, and applying Teacher-Student architectures on various distillation objectives.

{{</citation>}}


### (41/119) BarlowRL: Barlow Twins for Data-Efficient Reinforcement Learning (Omer Veysel Cagatan, 2023)

{{<citation>}}

Omer Veysel Cagatan. (2023)  
**BarlowRL: Barlow Twins for Data-Efficient Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.04263v1)  

---


**ABSTRACT**  
This paper introduces BarlowRL, a data-efficient reinforcement learning agent that combines the Barlow Twins self-supervised learning framework with DER (Data-Efficient Rainbow) algorithm. BarlowRL outperforms both DER and its contrastive counterpart CURL on the Atari 100k benchmark. BarlowRL avoids dimensional collapse by enforcing information spread to the whole space. This helps RL algorithms to utilize uniformly spread state representation that eventually results in a remarkable performance. The integration of Barlow Twins with DER enhances data efficiency and achieves superior performance in the RL tasks. BarlowRL demonstrates the potential of incorporating self-supervised learning techniques to improve RL algorithms.

{{</citation>}}


### (42/119) Semantic Interpretation and Validation of Graph Attention-based Explanations for GNN Models (Efimia Panagiotaki et al., 2023)

{{<citation>}}

Efimia Panagiotaki, Daniele De Martini, Lars Kunze. (2023)  
**Semantic Interpretation and Validation of Graph Attention-based Explanations for GNN Models**  

---
Primary Category: cs.LG  
Categories: G-3; I-2-10; I-2-9; I-4-8; I-5-2; I-5-1, cs-AI, cs-CY, cs-LG, cs-RO, cs.LG  
Keywords: AI, Attention, GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.04220v1)  

---


**ABSTRACT**  
In this work, we propose a methodology for investigating the application of semantic attention to enhance the explainability of Graph Neural Network (GNN)-based models, introducing semantically-informed perturbations and establishing a correlation between predicted feature-importance weights and model accuracy. Graph Deep Learning (GDL) has emerged as a promising field for tasks like scene interpretation, leveraging flexible graph structures to concisely describe complex features and relationships. As traditional explainability methods used in eXplainable AI (XAI) cannot be directly applied to such structures, graph-specific approaches are introduced. Attention mechanisms have demonstrated their efficacy in estimating the importance of input features in deep learning models and thus have been previously employed to provide feature-based explanations for GNN predictions. Building upon these insights, we extend existing attention-based graph-explainability methods investigating the use of attention weights as importance indicators of semantically sorted feature sets. Through analysing the behaviour of predicted attention-weights distribution in correlation with model accuracy, we gain valuable insights into feature importance with respect to the behaviour of the GNN model. We apply our methodology to a lidar pointcloud estimation model successfully identifying key semantic classes that contribute to enhanced performance effectively generating reliable post-hoc semantic explanations.

{{</citation>}}


### (43/119) Enhancing Adversarial Robustness in Low-Label Regime via Adaptively Weighted Regularization and Knowledge Distillation (Dongyoon Yang et al., 2023)

{{<citation>}}

Dongyoon Yang, Insung Kong, Yongdai Kim. (2023)  
**Enhancing Adversarial Robustness in Low-Label Regime via Adaptively Weighted Regularization and Knowledge Distillation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.04061v1)  

---


**ABSTRACT**  
Adversarial robustness is a research area that has recently received a lot of attention in the quest for trustworthy artificial intelligence. However, recent works on adversarial robustness have focused on supervised learning where it is assumed that labeled data is plentiful. In this paper, we investigate semi-supervised adversarial training where labeled data is scarce. We derive two upper bounds for the robust risk and propose a regularization term for unlabeled data motivated by these two upper bounds. Then, we develop a semi-supervised adversarial training algorithm that combines the proposed regularization term with knowledge distillation using a semi-supervised teacher (i.e., a teacher model trained using a semi-supervised learning algorithm). Our experiments show that our proposed algorithm achieves state-of-the-art performance with significant margins compared to existing algorithms. In particular, compared to supervised learning algorithms, performance of our proposed algorithm is not much worse even when the amount of labeled data is very small. For example, our algorithm with only 8\% labeled data is comparable to supervised adversarial training algorithms that use all labeled data, both in terms of standard and robust accuracies on CIFAR-10.

{{</citation>}}


### (44/119) The Five-Dollar Model: Generating Game Maps and Sprites from Sentence Embeddings (Timothy Merino et al., 2023)

{{<citation>}}

Timothy Merino, Roman Negri, Dipika Rajesh, M Charity, Julian Togelius. (2023)  
**The Five-Dollar Model: Generating Game Maps and Sprites from Sentence Embeddings**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: Embedding, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2308.04052v1)  

---


**ABSTRACT**  
The five-dollar model is a lightweight text-to-image generative architecture that generates low dimensional images from an encoded text prompt. This model can successfully generate accurate and aesthetically pleasing content in low dimensional domains, with limited amounts of training data. Despite the small size of both the model and datasets, the generated images are still able to maintain the encoded semantic meaning of the textual prompt. We apply this model to three small datasets: pixel art video game maps, video game sprite images, and down-scaled emoji images and apply novel augmentation strategies to improve the performance of our model on these limited datasets. We evaluate our models performance using cosine similarity score between text-image pairs generated by the CLIP VIT-B/32 model.

{{</citation>}}


### (45/119) Improving Performance of Semi-Supervised Learning by Adversarial Attacks (Dongyoon Yang et al., 2023)

{{<citation>}}

Dongyoon Yang, Kunwoong Kim, Yongdai Kim. (2023)  
**Improving Performance of Semi-Supervised Learning by Adversarial Attacks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Adversarial Attack, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.04018v1)  

---


**ABSTRACT**  
Semi-supervised learning (SSL) algorithm is a setup built upon a realistic assumption that access to a large amount of labeled data is tough. In this study, we present a generalized framework, named SCAR, standing for Selecting Clean samples with Adversarial Robustness, for improving the performance of recent SSL algorithms. By adversarially attacking pre-trained models with semi-supervision, our framework shows substantial advances in classifying images. We introduce how adversarial attacks successfully select high-confident unlabeled data to be labeled with current predictions. On CIFAR10, three recent SSL algorithms with SCAR result in significantly improved image classification.

{{</citation>}}


### (46/119) Understanding CNN Hidden Neuron Activations Using Structured Background Knowledge and Deductive Reasoning (Abhilekha Dalal et al., 2023)

{{<citation>}}

Abhilekha Dalal, Md Kamruzzaman Sarker, Adrita Barua, Eugene Vasserman, Pascal Hitzler. (2023)  
**Understanding CNN Hidden Neuron Activations Using Structured Background Knowledge and Deductive Reasoning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.03999v2)  

---


**ABSTRACT**  
A major challenge in Explainable AI is in correctly interpreting activations of hidden neurons: accurate interpretations would provide insights into the question of what a deep learning system has internally detected as relevant on the input, demystifying the otherwise black-box character of deep learning systems. The state of the art indicates that hidden node activations can, in some cases, be interpretable in a way that makes sense to humans, but systematic automated methods that would be able to hypothesize and verify interpretations of hidden neuron activations are underexplored. In this paper, we provide such a method and demonstrate that it provides meaningful interpretations. Our approach is based on using large-scale background knowledge approximately 2 million classes curated from the Wikipedia concept hierarchy together with a symbolic reasoning approach called Concept Induction based on description logics, originally developed for applications in the Semantic Web field. Our results show that we can automatically attach meaningful labels from the background knowledge to individual neurons in the dense layer of a Convolutional Neural Network through a hypothesis and verification process.

{{</citation>}}


### (47/119) Characterization of Human Balance through a Reinforcement Learning-based Muscle Controller (Kübra Akbaş et al., 2023)

{{<citation>}}

Kübra Akbaş, Carlotta Mummolo, Xianlian Zhou. (2023)  
**Characterization of Human Balance through a Reinforcement Learning-based Muscle Controller**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.04462v1)  

---


**ABSTRACT**  
Balance assessment during physical rehabilitation often relies on rubric-oriented battery tests to score a patient's physical capabilities, leading to subjectivity. While some objective balance assessments exist, they are often limited to tracking the center of pressure (COP), which does not fully capture the whole-body postural stability. This study explores the use of the center of mass (COM) state space and presents a promising avenue for monitoring the balance capabilities in humans. We employ a musculoskeletal model integrated with a balance controller, trained through reinforcement learning (RL), to investigate balancing capabilities. The RL framework consists of two interconnected neural networks governing balance recovery and muscle coordination respectively, trained using Proximal Policy Optimization (PPO) with reference state initialization, early termination, and multiple training strategies. By exploring recovery from random initial COM states (position and velocity) space for a trained controller, we obtain the final BR enclosing successful balance recovery trajectories. Comparing the BRs with analytical postural stability limits from a linear inverted pendulum model, we observe a similar trend in successful COM states but more limited ranges in the recoverable areas. We further investigate the effect of muscle weakness and neural excitation delay on the BRs, revealing reduced balancing capability in different regions. Overall, our approach of learning muscular balance controllers presents a promising new method for establishing balance recovery limits and objectively assessing balance capability in bipedal systems, particularly in humans.

{{</citation>}}


## cs.CR (3)



### (48/119) Different Mechanisms of Machine Learning and Optimization Algorithms Utilized in Intrusion Detection Systems (Mohammad Aziz et al., 2023)

{{<citation>}}

Mohammad Aziz, Ali Saeed Alfoudi. (2023)  
**Different Mechanisms of Machine Learning and Optimization Algorithms Utilized in Intrusion Detection Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2308.04607v1)  

---


**ABSTRACT**  
Malicious software is an integral part of cybercrime defense. Due to the growing number of malicious attacks and their target sources, detecting and preventing the attack becomes more challenging due to the assault's changing behavior. The bulk of classic malware detection systems is based on statistics, analytic techniques, or machine learning. Virus signature methods are widely used to identify malware. The bulk of anti-malware systems categorizes malware using regular expressions and patterns. While antivirus software is less likely to update its databases to identify and block malware, file features must be updated to detect and prevent newly generated malware. Creating attack signatures requires practically all of a human being's work. The purpose of this study is to undertake a review of the current research on intrusion detection models and the datasets that support them. In this article, we discuss the state-of-the-art, focusing on the strategy that was devised and executed, the dataset that was utilized, the findings, and the assessment that was undertaken. Additionally, the surveyed articles undergo critical analysis and statements in order to give a thorough comparative review. Machine learning and deep learning methods, as well as new classification and feature selection methodologies, are studied and researched. Thus far, each technique has proved the capability of constructing very accurate intrusion detection models. The survey findings reveal that Clearly, the MultiTree and adaptive voting algorithms surpassed all other models in terms of persistency and performance, averaging 99.98 percent accuracy on average.

{{</citation>}}


### (49/119) XGBD: Explanation-Guided Graph Backdoor Detection (Zihan Guan et al., 2023)

{{<citation>}}

Zihan Guan, Mengnan Du, Ninghao Liu. (2023)  
**XGBD: Explanation-Guided Graph Backdoor Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs-SI, cs.CR  
Keywords: GNN, NLP  
[Paper Link](http://arxiv.org/abs/2308.04406v1)  

---


**ABSTRACT**  
Backdoor attacks pose a significant security risk to graph learning models. Backdoors can be embedded into the target model by inserting backdoor triggers into the training dataset, causing the model to make incorrect predictions when the trigger is present. To counter backdoor attacks, backdoor detection has been proposed. An emerging detection strategy in the vision and NLP domains is based on an intriguing phenomenon: when training models on a mixture of backdoor and clean samples, the loss on backdoor samples drops significantly faster than on clean samples, allowing backdoor samples to be easily detected by selecting samples with the lowest loss values. However, the ignorance of topological feature information on graph data limits its detection effectiveness when applied directly to the graph domain. To this end, we propose an explanation-guided backdoor detection method to take advantage of the topological information. Specifically, we train a helper model on the graph dataset, feed graph samples into the model, and then adopt explanation methods to attribute model prediction to an important subgraph. We observe that backdoor samples have distinct attribution distribution than clean samples, so the explanatory subgraph could serve as more discriminative features for detecting backdoor samples. Comprehensive experiments on multiple popular datasets and attack methods demonstrate the effectiveness and explainability of our method. Our code is available: https://github.com/GuanZihan/GNN_backdoor_detection.

{{</citation>}}


### (50/119) Evil Operation: Breaking Speaker Recognition with PaddingBack (Zhe Ye et al., 2023)

{{<citation>}}

Zhe Ye, Diqun Yan, Li Dong, Kailai Shen. (2023)  
**Evil Operation: Breaking Speaker Recognition with PaddingBack**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SD, cs.CR, eess-AS, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04179v1)  

---


**ABSTRACT**  
Machine Learning as a Service (MLaaS) has gained popularity due to advancements in machine learning. However, untrusted third-party platforms have raised concerns about AI security, particularly in backdoor attacks. Recent research has shown that speech backdoors can utilize transformations as triggers, similar to image backdoors. However, human ears easily detect these transformations, leading to suspicion. In this paper, we introduce PaddingBack, an inaudible backdoor attack that utilizes malicious operations to make poisoned samples indistinguishable from clean ones. Instead of using external perturbations as triggers, we exploit the widely used speech signal operation, padding, to break speaker recognition systems. Our experimental results demonstrate the effectiveness of the proposed approach, achieving a significantly high attack success rate while maintaining a high rate of benign accuracy. Furthermore, PaddingBack demonstrates the ability to resist defense methods while maintaining its stealthiness against human perception. The results of the stealthiness experiment have been made available at https://nbufabio25.github.io/paddingback/.

{{</citation>}}


## cs.CV (35)



### (51/119) Temporal DINO: A Self-supervised Video Strategy to Enhance Action Prediction (Izzeddin Teeti et al., 2023)

{{<citation>}}

Izzeddin Teeti, Rongali Sai Bhargav, Vivek Singh, Andrew Bradley, Biplab Banerjee, Fabio Cuzzolin. (2023)  
**Temporal DINO: A Self-supervised Video Strategy to Enhance Action Prediction**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2308.04589v1)  

---


**ABSTRACT**  
The emerging field of action prediction plays a vital role in various computer vision applications such as autonomous driving, activity analysis and human-computer interaction. Despite significant advancements, accurately predicting future actions remains a challenging problem due to high dimensionality, complex dynamics and uncertainties inherent in video data. Traditional supervised approaches require large amounts of labelled data, which is expensive and time-consuming to obtain. This paper introduces a novel self-supervised video strategy for enhancing action prediction inspired by DINO (self-distillation with no labels). The Temporal-DINO approach employs two models; a 'student' processing past frames; and a 'teacher' processing both past and future frames, enabling a broader temporal context. During training, the teacher guides the student to learn future context by only observing past frames. The strategy is evaluated on ROAD dataset for the action prediction downstream task using 3D-ResNet, Transformer, and LSTM architectures. The experimental results showcase significant improvements in prediction performance across these architectures, with our method achieving an average enhancement of 9.9% Precision Points (PP), highlighting its effectiveness in enhancing the backbones' capabilities of capturing long-term dependencies. Furthermore, our approach demonstrates efficiency regarding the pretraining dataset size and the number of epochs required. This method overcomes limitations present in other approaches, including considering various backbone architectures, addressing multiple prediction horizons, reducing reliance on hand-crafted augmentations, and streamlining the pretraining process into a single stage. These findings highlight the potential of our approach in diverse video-based tasks such as activity recognition, motion planning, and scene understanding.

{{</citation>}}


### (52/119) LATR: 3D Lane Detection from Monocular Images with Transformer (Yueru Luo et al., 2023)

{{<citation>}}

Yueru Luo, Chaoda Zheng, Xu Yan, Tang Kun, Chao Zheng, Shuguang Cui, Zhen Li. (2023)  
**LATR: 3D Lane Detection from Monocular Images with Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.04583v1)  

---


**ABSTRACT**  
3D lane detection from monocular images is a fundamental yet challenging task in autonomous driving. Recent advances primarily rely on structural 3D surrogates (e.g., bird's eye view) that are built from front-view image features and camera parameters. However, the depth ambiguity in monocular images inevitably causes misalignment between the constructed surrogate feature map and the original image, posing a great challenge for accurate lane detection. To address the above issue, we present a novel LATR model, an end-to-end 3D lane detector that uses 3D-aware front-view features without transformed view representation. Specifically, LATR detects 3D lanes via cross-attention based on query and key-value pairs, constructed using our lane-aware query generator and dynamic 3D ground positional embedding. On the one hand, each query is generated based on 2D lane-aware features and adopts a hybrid embedding to enhance the lane information. On the other hand, 3D space information is injected as positional embedding from an iteratively-updated 3D ground plane. LATR outperforms previous state-of-the-art methods on both synthetic Apollo and realistic OpenLane by large margins (e.g., 11.4 gains in terms of F1 score on OpenLane). Code will be released at https://github.com/JMoonr/LATR.

{{</citation>}}


### (53/119) FocalFormer3D : Focusing on Hard Instance for 3D Object Detection (Yilun Chen et al., 2023)

{{<citation>}}

Yilun Chen, Zhiding Yu, Yukang Chen, Shiyi Lan, Animashree Anandkumar, Jiaya Jia, Jose Alvarez. (2023)  
**FocalFormer3D : Focusing on Hard Instance for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.04556v1)  

---


**ABSTRACT**  
False negatives (FN) in 3D object detection, {\em e.g.}, missing predictions of pedestrians, vehicles, or other obstacles, can lead to potentially dangerous situations in autonomous driving. While being fatal, this issue is understudied in many current 3D detection methods. In this work, we propose Hard Instance Probing (HIP), a general pipeline that identifies \textit{FN} in a multi-stage manner and guides the models to focus on excavating difficult instances. For 3D object detection, we instantiate this method as FocalFormer3D, a simple yet effective detector that excels at excavating difficult objects and improving prediction recall. FocalFormer3D features a multi-stage query generation to discover hard objects and a box-level transformer decoder to efficiently distinguish objects from massive object candidates. Experimental results on the nuScenes and Waymo datasets validate the superior performance of FocalFormer3D. The advantage leads to strong performance on both detection and tracking, in both LiDAR and multi-modal settings. Notably, FocalFormer3D achieves a 70.5 mAP and 73.9 NDS on nuScenes detection benchmark, while the nuScenes tracking benchmark shows 72.1 AMOTA, both ranking 1st place on the nuScenes LiDAR leaderboard. Our code is available at \url{https://github.com/NVlabs/FocalFormer3D}.

{{</citation>}}


### (54/119) Prune Spatio-temporal Tokens by Semantic-aware Temporal Accumulation (Shuangrui Ding et al., 2023)

{{<citation>}}

Shuangrui Ding, Peisen Zhao, Xiaopeng Zhang, Rui Qian, Hongkai Xiong, Qi Tian. (2023)  
**Prune Spatio-temporal Tokens by Semantic-aware Temporal Accumulation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.04549v1)  

---


**ABSTRACT**  
Transformers have become the primary backbone of the computer vision community due to their impressive performance. However, the unfriendly computation cost impedes their potential in the video recognition domain. To optimize the speed-accuracy trade-off, we propose Semantic-aware Temporal Accumulation score (STA) to prune spatio-temporal tokens integrally. STA score considers two critical factors: temporal redundancy and semantic importance. The former depicts a specific region based on whether it is a new occurrence or a seen entity by aggregating token-to-token similarity in consecutive frames while the latter evaluates each token based on its contribution to the overall prediction. As a result, tokens with higher scores of STA carry more temporal redundancy as well as lower semantics thus being pruned. Based on the STA score, we are able to progressively prune the tokens without introducing any additional parameters or requiring further re-training. We directly apply the STA module to off-the-shelf ViT and VideoSwin backbones, and the empirical results on Kinetics-400 and Something-Something V2 achieve over 30% computation reduction with a negligible ~0.2% accuracy drop. The code is released at https://github.com/Mark12Ding/STA.

{{</citation>}}


### (55/119) YUDO: YOLO for Uniform Directed Object Detection (Đorđe Nedeljković, 2023)

{{<citation>}}

Đorđe Nedeljković. (2023)  
**YUDO: YOLO for Uniform Directed Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Yolo  
[Paper Link](http://arxiv.org/abs/2308.04542v1)  

---


**ABSTRACT**  
This paper presents an efficient way of detecting directed objects by predicting their center coordinates and direction angle. Since the objects are of uniform size, the proposed model works without predicting the object's width and height. The dataset used for this problem is presented in Honeybee Segmentation and Tracking Datasets project. One of the contributions of this work is an examination of the ability of the standard real-time object detection architecture like YoloV7 to be customized for position and direction detection. A very efficient, tiny version of the architecture is used in this approach. Moreover, only one of three detection heads without anchors is sufficient for this task. We also introduce the extended Skew Intersection over Union (SkewIoU) calculation for rotated boxes - directed IoU (DirIoU), which includes an absolute angle difference. DirIoU is used both in the matching procedure of target and predicted bounding boxes for mAP calculation, and in the NMS filtering procedure. The code and models are available at https://github.com/djordjened92/yudo.

{{</citation>}}


### (56/119) Estimation of Human Condition at Disaster Site Using Aerial Drone Images (Tomoki Arai et al., 2023)

{{<citation>}}

Tomoki Arai, Kenji Iwata, Kensho Hara, Yutaka Satoh. (2023)  
**Estimation of Human Condition at Disaster Site Using Aerial Drone Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2308.04535v1)  

---


**ABSTRACT**  
Drones are being used to assess the situation in various disasters. In this study, we investigate a method to automatically estimate the damage status of people based on their actions in aerial drone images in order to understand disaster sites faster and save labor. We constructed a new dataset of aerial images of human actions in a hypothetical disaster that occurred in an urban area, and classified the human damage status using 3D ResNet. The results showed that the status with characteristic human actions could be classified with a recall rate of more than 80%, while other statuses with similar human actions could only be classified with a recall rate of about 50%. In addition, a cloud-based VR presentation application suggested the effectiveness of using drones to understand the disaster site and estimate the human condition.

{{</citation>}}


### (57/119) Unsupervised Camouflaged Object Segmentation as Domain Adaptation (Yi Zhang et al., 2023)

{{<citation>}}

Yi Zhang, Chengyi Wu. (2023)  
**Unsupervised Camouflaged Object Segmentation as Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.04528v1)  

---


**ABSTRACT**  
Deep learning for unsupervised image segmentation remains challenging due to the absence of human labels. The common idea is to train a segmentation head, with the supervision of pixel-wise pseudo-labels generated based on the representation of self-supervised backbones. By doing so, the model performance depends much on the distance between the distributions of target datasets and the pre-training dataset (e.g., ImageNet). In this work, we investigate a new task, namely unsupervised camouflaged object segmentation (UCOS), where the target objects own a common rarely-seen attribute, i.e., camouflage. Unsurprisingly, we find that the state-of-the-art unsupervised models struggle in adapting UCOS, due to the domain gap between the properties of generic and camouflaged objects. To this end, we formulate the UCOS as a source-free unsupervised domain adaptation task (UCOS-DA), where both source labels and target labels are absent during the whole model training process. Specifically, we define a source model consisting of self-supervised vision transformers pre-trained on ImageNet. On the other hand, the target domain includes a simple linear layer (i.e., our target model) and unlabeled camouflaged objects. We then design a pipeline for foreground-background-contrastive self-adversarial domain adaptation, to achieve robust UCOS. As a result, our baseline model achieves superior segmentation performance when compared with competing unsupervised models on the UCOS benchmark, with the training set which's scale is only one tenth of the supervised COS counterpart.

{{</citation>}}


### (58/119) Toward unlabeled multi-view 3D pedestrian detection by generalizable AI: techniques and performance analysis (João Paulo Lima et al., 2023)

{{<citation>}}

João Paulo Lima, Diego Thomas, Hideaki Uchiyama, Veronica Teichrieb. (2023)  
**Toward unlabeled multi-view 3D pedestrian detection by generalizable AI: techniques and performance analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04515v1)  

---


**ABSTRACT**  
We unveil how generalizable AI can be used to improve multi-view 3D pedestrian detection in unlabeled target scenes. One way to increase generalization to new scenes is to automatically label target data, which can then be used for training a detector model. In this context, we investigate two approaches for automatically labeling target data: pseudo-labeling using a supervised detector and automatic labeling using an untrained detector (that can be applied out of the box without any training). We adopt a training framework for optimizing detector models using automatic labeling procedures. This framework encompasses different training sets/modes and multi-round automatic labeling strategies. We conduct our analyses on the publicly-available WILDTRACK and MultiviewX datasets. We show that, by using the automatic labeling approach based on an untrained detector, we can obtain superior results than directly using the untrained detector or a detector trained with an existing labeled source dataset. It achieved a MODA about 4% and 1% better than the best existing unlabeled method when using WILDTRACK and MultiviewX as target datasets, respectively.

{{</citation>}}


### (59/119) A Deep-Learning Method Using Auto-encoder and Generative Adversarial Network for Anomaly Detection on Ancient Stone Stele Surfaces (Yikun Liu et al., 2023)

{{<citation>}}

Yikun Liu, Yuning Wang, Cheng Liu. (2023)  
**A Deep-Learning Method Using Auto-encoder and Generative Adversarial Network for Anomaly Detection on Ancient Stone Stele Surfaces**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.04426v1)  

---


**ABSTRACT**  
Accurate detection of natural deterioration and man-made damage on the surfaces of ancient stele in the first instance is essential for their preventive conservation. Existing methods for cultural heritage preservation are not able to achieve this goal perfectly due to the difficulty of balancing accuracy, efficiency, timeliness, and cost. This paper presents a deep-learning method to automatically detect above mentioned emergencies on ancient stone stele in real time, employing autoencoder (AE) and generative adversarial network (GAN). The proposed method overcomes the limitations of existing methods by requiring no extensive anomaly samples while enabling comprehensive detection of unpredictable anomalies. the method includes stages of monitoring, data acquisition, pre-processing, model structuring, and post-processing. Taking the Longmen Grottoes' stone steles as a case study, an unsupervised learning model based on AE and GAN architectures is proposed and validated with a reconstruction accuracy of 99.74\%. The method's evaluation revealed the proficient detection of seven artificially designed anomalies and demonstrated precision and reliability without false alarms. This research provides novel ideas and possibilities for the application of deep learning in the field of cultural heritage.

{{</citation>}}


### (60/119) V-DETR: DETR with Vertex Relative Position Encoding for 3D Object Detection (Yichao Shen et al., 2023)

{{<citation>}}

Yichao Shen, Zigang Geng, Yuhui Yuan, Yutong Lin, Ze Liu, Chunyu Wang, Han Hu, Nanning Zheng, Baining Guo. (2023)  
**V-DETR: DETR with Vertex Relative Position Encoding for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.04409v1)  

---


**ABSTRACT**  
We introduce a highly performant 3D object detector for point clouds using the DETR framework. The prior attempts all end up with suboptimal results because they fail to learn accurate inductive biases from the limited scale of training data. In particular, the queries often attend to points that are far away from the target objects, violating the locality principle in object detection. To address the limitation, we introduce a novel 3D Vertex Relative Position Encoding (3DV-RPE) method which computes position encoding for each point based on its relative position to the 3D boxes predicted by the queries in each decoder layer, thus providing clear information to guide the model to focus on points near the objects, in accordance with the principle of locality. In addition, we systematically improve the pipeline from various aspects such as data normalization based on our understanding of the task. We show exceptional results on the challenging ScanNetV2 benchmark, achieving significant improvements over the previous 3DETR in $\rm{AP}_{25}$/$\rm{AP}_{50}$ from 65.0\%/47.0\% to 77.8\%/66.0\%, respectively. In addition, our method sets a new record on ScanNetV2 and SUN RGB-D datasets.Code will be released at http://github.com/yichaoshen-MS/V-DETR.

{{</citation>}}


### (61/119) LEFormer: A Hybrid CNN-Transformer Architecture for Accurate Lake Extraction from Remote Sensing Imagery (Ben Chen et al., 2023)

{{<citation>}}

Ben Chen, Xuechao Zou, Yu Zhang, Jiayu Li, Kai Li, Pin Tao. (2023)  
**LEFormer: A Hybrid CNN-Transformer Architecture for Accurate Lake Extraction from Remote Sensing Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.04397v1)  

---


**ABSTRACT**  
Lake extraction from remote sensing imagery is challenging due to the complex shapes of lakes and the presence of noise. Existing methods suffer from blurred segmentation boundaries and poor foreground modeling. In this paper, we propose a hybrid CNN-Transformer architecture, called LEFormer, for accurate lake extraction. LEFormer contains four main modules: CNN encoder, Transformer encoder, cross-encoder fusion, and lightweight decoder. The CNN encoder recovers local spatial information and improves fine-scale details. Simultaneously, the Transformer encoder captures long-range dependencies between sequences of any length, allowing them to obtain global features and context information better. Finally, a lightweight decoder is employed for mask prediction. We evaluate the performance and efficiency of LEFormer on two datasets, the Surface Water (SW) and the Qinghai-Tibet Plateau Lake (QTPL). Experimental results show that LEFormer consistently achieves state-of-the-art (SOTA) performance and efficiency on these two datasets, outperforming existing methods. Specifically, LEFormer achieves 90.86% and 97.42% mIoU on the SW and QTPL datasets with a parameter count of 3.61M, respectively, while being 20x minor than the previous SOTA method.

{{</citation>}}


### (62/119) When Super-Resolution Meets Camouflaged Object Detection: A Comparison Study (Juan Wen et al., 2023)

{{<citation>}}

Juan Wen, Shupeng Cheng, Peng Xu, Bowen Zhou, Radu Timofte, Weiyan Hou, Luc Van Gool. (2023)  
**When Super-Resolution Meets Camouflaged Object Detection: A Comparison Study**  

---
Primary Category: cs.CV  
Categories: 68T45, I-4-3, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.04370v1)  

---


**ABSTRACT**  
Super Resolution (SR) and Camouflaged Object Detection (COD) are two hot topics in computer vision with various joint applications. For instance, low-resolution surveillance images can be successively processed by super-resolution techniques and camouflaged object detection. However, in previous work, these two areas are always studied in isolation. In this paper, we, for the first time, conduct an integrated comparative evaluation for both. Specifically, we benchmark different super-resolution methods on commonly used COD datasets, and meanwhile, we evaluate the robustness of different COD models by using COD data processed by SR methods. Our goal is to bridge these two domains, discover novel experimental phenomena, summarize new experim.

{{</citation>}}


### (63/119) SSTFormer: Bridging Spiking Neural Network and Memory Support Transformer for Frame-Event based Recognition (Xiao Wang et al., 2023)

{{<citation>}}

Xiao Wang, Zongzhen Wu, Yao Rong, Lin Zhu, Bo Jiang, Jin Tang, Yonghong Tian. (2023)  
**SSTFormer: Bridging Spiking Neural Network and Memory Support Transformer for Frame-Event based Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs-NE, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.04369v1)  

---


**ABSTRACT**  
Event camera-based pattern recognition is a newly arising research topic in recent years. Current researchers usually transform the event streams into images, graphs, or voxels, and adopt deep neural networks for event-based classification. Although good performance can be achieved on simple event recognition datasets, however, their results may be still limited due to the following two issues. Firstly, they adopt spatial sparse event streams for recognition only, which may fail to capture the color and detailed texture information well. Secondly, they adopt either Spiking Neural Networks (SNN) for energy-efficient recognition with suboptimal results, or Artificial Neural Networks (ANN) for energy-intensive, high-performance recognition. However, seldom of them consider achieving a balance between these two aspects. In this paper, we formally propose to recognize patterns by fusing RGB frames and event streams simultaneously and propose a new RGB frame-event recognition framework to address the aforementioned issues. The proposed method contains four main modules, i.e., memory support Transformer network for RGB frame encoding, spiking neural network for raw event stream encoding, multi-modal bottleneck fusion module for RGB-Event feature aggregation, and prediction head. Due to the scarce of RGB-Event based classification dataset, we also propose a large-scale PokerEvent dataset which contains 114 classes, and 27102 frame-event pairs recorded using a DVS346 event camera. Extensive experiments on two RGB-Event based classification datasets fully validated the effectiveness of our proposed framework. We hope this work will boost the development of pattern recognition by fusing RGB frames and event streams. Both our dataset and source code of this work will be released at https://github.com/Event-AHU/SSTFormer.

{{</citation>}}


### (64/119) 3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment (Ziyu Zhu et al., 2023)

{{<citation>}}

Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan Huang, Qing Li. (2023)  
**3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2308.04352v1)  

---


**ABSTRACT**  
3D vision-language grounding (3D-VL) is an emerging field that aims to connect the 3D physical world with natural language, which is crucial for achieving embodied intelligence. Current 3D-VL models rely heavily on sophisticated modules, auxiliary losses, and optimization tricks, which calls for a simple and unified model. In this paper, we propose 3D-VisTA, a pre-trained Transformer for 3D Vision and Text Alignment that can be easily adapted to various downstream tasks. 3D-VisTA simply utilizes self-attention layers for both single-modal modeling and multi-modal fusion without any sophisticated task-specific design. To further enhance its performance on 3D-VL tasks, we construct ScanScribe, the first large-scale 3D scene-text pairs dataset for 3D-VL pre-training. ScanScribe contains 2,995 RGB-D scans for 1,185 unique indoor scenes originating from ScanNet and 3R-Scan datasets, along with paired 278K scene descriptions generated from existing 3D-VL tasks, templates, and GPT-3. 3D-VisTA is pre-trained on ScanScribe via masked language/object modeling and scene-text matching. It achieves state-of-the-art results on various 3D-VL tasks, ranging from visual grounding and dense captioning to question answering and situated reasoning. Moreover, 3D-VisTA demonstrates superior data efficiency, obtaining strong performance even with limited annotations during downstream task fine-tuning.

{{</citation>}}


### (65/119) Unifying Two-Stream Encoders with Transformers for Cross-Modal Retrieval (Yi Bin et al., 2023)

{{<citation>}}

Yi Bin, Haoxuan Li, Yahui Xu, Xing Xu, Yang Yang, Heng Tao Shen. (2023)  
**Unifying Two-Stream Encoders with Transformers for Cross-Modal Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-IR, cs-MM, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.04343v1)  

---


**ABSTRACT**  
Most existing cross-modal retrieval methods employ two-stream encoders with different architectures for images and texts, \textit{e.g.}, CNN for images and RNN/Transformer for texts. Such discrepancy in architectures may induce different semantic distribution spaces and limit the interactions between images and texts, and further result in inferior alignment between images and texts. To fill this research gap, inspired by recent advances of Transformers in vision tasks, we propose to unify the encoder architectures with Transformers for both modalities. Specifically, we design a cross-modal retrieval framework purely based on two-stream Transformers, dubbed \textbf{Hierarchical Alignment Transformers (HAT)}, which consists of an image Transformer, a text Transformer, and a hierarchical alignment module. With such identical architectures, the encoders could produce representations with more similar characteristics for images and texts, and make the interactions and alignments between them much easier. Besides, to leverage the rich semantics, we devise a hierarchical alignment scheme to explore multi-level correspondences of different layers between images and texts. To evaluate the effectiveness of the proposed HAT, we conduct extensive experiments on two benchmark datasets, MSCOCO and Flickr30K. Experimental results demonstrate that HAT outperforms SOTA baselines by a large margin. Specifically, on two key tasks, \textit{i.e.}, image-to-text and text-to-image retrieval, HAT achieves 7.6\% and 16.7\% relative score improvement of Recall@1 on MSCOCO, and 4.4\% and 11.6\% on Flickr30k respectively. The code is available at \url{https://github.com/LuminosityX/HAT}.

{{</citation>}}


### (66/119) Domain Adaptive Person Search via GAN-based Scene Synthesis for Cross-scene Videos (Huibing Wang et al., 2023)

{{<citation>}}

Huibing Wang, Tianxiang Cui, Mingze Yao, Huijuan Pang, Yushan Du. (2023)  
**Domain Adaptive Person Search via GAN-based Scene Synthesis for Cross-scene Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04322v1)  

---


**ABSTRACT**  
Person search has recently been a challenging task in the computer vision domain, which aims to search specific pedestrians from real cameras.Nevertheless, most surveillance videos comprise only a handful of images of each pedestrian, which often feature identical backgrounds and clothing. Hence, it is difficult to learn more discriminative features for person search in real scenes. To tackle this challenge, we draw on Generative Adversarial Networks (GAN) to synthesize data from surveillance videos. GAN has thrived in computer vision problems because it produces high-quality images efficiently. We merely alter the popular Fast R-CNN model, which is capable of processing videos and yielding accurate detection outcomes. In order to appropriately relieve the pressure brought by the two-stage model, we design an Assisted-Identity Query Module (AIDQ) to provide positive images for the behind part. Besides, the proposed novel GAN-based Scene Synthesis model that can synthesize high-quality cross-id person images for person search tasks. In order to facilitate the feature learning of the GAN-based Scene Synthesis model, we adopt an online learning strategy that collaboratively learns the synthesized images and original images. Extensive experiments on two widely used person search benchmarks, CUHK-SYSU and PRW, have shown that our method has achieved great performance, and the extensive ablation study further justifies our GAN-synthetic data can effectively increase the variability of the datasets and be more realistic.

{{</citation>}}


### (67/119) All-pairs Consistency Learning for Weakly Supervised Semantic Segmentation (Weixuan Sun et al., 2023)

{{<citation>}}

Weixuan Sun, Yanhao Zhang, Zhen Qin, Zheyuan Liu, Lin Cheng, Fanyi Wang, Yiran Zhong, Nick Barnes. (2023)  
**All-pairs Consistency Learning for Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.04321v1)  

---


**ABSTRACT**  
In this work, we propose a new transformer-based regularization to better localize objects for Weakly supervised semantic segmentation (WSSS). In image-level WSSS, Class Activation Map (CAM) is adopted to generate object localization as pseudo segmentation labels. To address the partial activation issue of the CAMs, consistency regularization is employed to maintain activation intensity invariance across various image augmentations. However, such methods ignore pair-wise relations among regions within each CAM, which capture context and should also be invariant across image views. To this end, we propose a new all-pairs consistency regularization (ACR). Given a pair of augmented views, our approach regularizes the activation intensities between a pair of augmented views, while also ensuring that the affinity across regions within each view remains consistent. We adopt vision transformers as the self-attention mechanism naturally embeds pair-wise affinity. This enables us to simply regularize the distance between the attention matrices of augmented image pairs. Additionally, we introduce a novel class-wise localization method that leverages the gradients of the class token. Our method can be seamlessly integrated into existing WSSS methods using transformers without modifying the architectures. We evaluate our method on PASCAL VOC and MS COCO datasets. Our method produces noticeably better class localization maps (67.3% mIoU on PASCAL VOC train), resulting in superior WSSS performances.

{{</citation>}}


### (68/119) AICSD: Adaptive Inter-Class Similarity Distillation for Semantic Segmentation (Amir M. Mansourian et al., 2023)

{{<citation>}}

Amir M. Mansourian, Rozhan Ahmadi, Shohreh Kasaei. (2023)  
**AICSD: Adaptive Inter-Class Similarity Distillation for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.04243v1)  

---


**ABSTRACT**  
In recent years, deep neural networks have achieved remarkable accuracy in computer vision tasks. With inference time being a crucial factor, particularly in dense prediction tasks such as semantic segmentation, knowledge distillation has emerged as a successful technique for improving the accuracy of lightweight student networks. The existing methods often neglect the information in channels and among different classes. To overcome these limitations, this paper proposes a novel method called Inter-Class Similarity Distillation (ICSD) for the purpose of knowledge distillation. The proposed method transfers high-order relations from the teacher network to the student network by independently computing intra-class distributions for each class from network outputs. This is followed by calculating inter-class similarity matrices for distillation using KL divergence between distributions of each pair of classes. To further improve the effectiveness of the proposed method, an Adaptive Loss Weighting (ALW) training strategy is proposed. Unlike existing methods, the ALW strategy gradually reduces the influence of the teacher network towards the end of training process to account for errors in teacher's predictions. Extensive experiments conducted on two well-known datasets for semantic segmentation, Cityscapes and Pascal VOC 2012, validate the effectiveness of the proposed method in terms of mIoU and pixel accuracy. The proposed method outperforms most of existing knowledge distillation methods as demonstrated by both quantitative and qualitative evaluations. Code is available at: https://github.com/AmirMansurian/AICSD

{{</citation>}}


### (69/119) Exploring Transformers for Open-world Instance Segmentation (Jiannan Wu et al., 2023)

{{<citation>}}

Jiannan Wu, Yi Jiang, Bin Yan, Huchuan Lu, Zehuan Yuan, Ping Luo. (2023)  
**Exploring Transformers for Open-world Instance Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.04206v1)  

---


**ABSTRACT**  
Open-world instance segmentation is a rising task, which aims to segment all objects in the image by learning from a limited number of base-category objects. This task is challenging, as the number of unseen categories could be hundreds of times larger than that of seen categories. Recently, the DETR-like models have been extensively studied in the closed world while stay unexplored in the open world. In this paper, we utilize the Transformer for open-world instance segmentation and present SWORD. Firstly, we introduce to attach the stop-gradient operation before classification head and further add IoU heads for discovering novel objects. We demonstrate that a simple stop-gradient operation not only prevents the novel objects from being suppressed as background, but also allows the network to enjoy the merit of heuristic label assignment. Secondly, we propose a novel contrastive learning framework to enlarge the representations between objects and background. Specifically, we maintain a universal object queue to obtain the object center, and dynamically select positive and negative samples from the object queries for contrastive learning. While the previous works only focus on pursuing average recall and neglect average precision, we show the prominence of SWORD by giving consideration to both criteria. Our models achieve state-of-the-art performance in various open-world cross-category and cross-dataset generalizations. Particularly, in VOC to non-VOC setup, our method sets new state-of-the-art results of 40.0% on ARb100 and 34.9% on ARm100. For COCO to UVO generalization, SWORD significantly outperforms the previous best open-world model by 5.9% on APm and 8.1% on ARm100.

{{</citation>}}


### (70/119) D3G: Exploring Gaussian Prior for Temporal Sentence Grounding with Glance Annotation (Hanjun Li et al., 2023)

{{<citation>}}

Hanjun Li, Xiujun Shu, Sunan He, Ruizhi Qiao, Wei Wen, Taian Guo, Bei Gan, Xing Sun. (2023)  
**D3G: Exploring Gaussian Prior for Temporal Sentence Grounding with Glance Annotation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.04197v1)  

---


**ABSTRACT**  
Temporal sentence grounding (TSG) aims to locate a specific moment from an untrimmed video with a given natural language query. Recently, weakly supervised methods still have a large performance gap compared to fully supervised ones, while the latter requires laborious timestamp annotations. In this study, we aim to reduce the annotation cost yet keep competitive performance for TSG task compared to fully supervised ones. To achieve this goal, we investigate a recently proposed glance-supervised temporal sentence grounding task, which requires only single frame annotation (referred to as glance annotation) for each query. Under this setup, we propose a Dynamic Gaussian prior based Grounding framework with Glance annotation (D3G), which consists of a Semantic Alignment Group Contrastive Learning module (SA-GCL) and a Dynamic Gaussian prior Adjustment module (DGA). Specifically, SA-GCL samples reliable positive moments from a 2D temporal map via jointly leveraging Gaussian prior and semantic consistency, which contributes to aligning the positive sentence-moment pairs in the joint embedding space. Moreover, to alleviate the annotation bias resulting from glance annotation and model complex queries consisting of multiple events, we propose the DGA module, which adjusts the distribution dynamically to approximate the ground truth of target moments. Extensive experiments on three challenging benchmarks verify the effectiveness of the proposed D3G. It outperforms the state-of-the-art weakly supervised methods by a large margin and narrows the performance gap compared to fully supervised methods. Code is available at https://github.com/solicucu/D3G.

{{</citation>}}


### (71/119) EPCFormer: Expression Prompt Collaboration Transformer for Universal Referring Video Object Segmentation (Jiajun Chen et al., 2023)

{{<citation>}}

Jiajun Chen, Jiacheng Lin, Zhiqiang Xiao, Haolong Fu, Ke Nai, Kailun Yang, Zhiyong Li. (2023)  
**EPCFormer: Expression Prompt Collaboration Transformer for Universal Referring Video Object Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-AS, eess-IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.04162v1)  

---


**ABSTRACT**  
Audio-guided Video Object Segmentation (A-VOS) and Referring Video Object Segmentation (R-VOS) are two highly-related tasks, which both aim to segment specific objects from video sequences according to user-provided expression prompts. However, due to the challenges in modeling representations for different modalities, contemporary methods struggle to strike a balance between interaction flexibility and high-precision localization and segmentation. In this paper, we address this problem from two perspectives: the alignment representation of audio and text and the deep interaction among audio, text, and visual features. First, we propose a universal architecture, the Expression Prompt Collaboration Transformer, herein EPCFormer. Next, we propose an Expression Alignment (EA) mechanism for audio and text expressions. By introducing contrastive learning for audio and text expressions, the proposed EPCFormer realizes comprehension of the semantic equivalence between audio and text expressions denoting the same objects. Then, to facilitate deep interactions among audio, text, and video features, we introduce an Expression-Visual Attention (EVA) mechanism. The knowledge of video object segmentation in terms of the expression prompts can seamlessly transfer between the two tasks by deeply exploring complementary cues between text and audio. Experiments on well-recognized benchmarks demonstrate that our universal EPCFormer attains state-of-the-art results on both tasks. The source code of EPCFormer will be made publicly available at https://github.com/lab206/EPCFormer.

{{</citation>}}


### (72/119) Towards Top-Down Stereoscopic Image Quality Assessment via Stereo Attention (Huilin Zhang et al., 2023)

{{<citation>}}

Huilin Zhang, Sumei Li, Yongli Chang. (2023)  
**Towards Top-Down Stereoscopic Image Quality Assessment via Stereo Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV, eess-IV  
Keywords: Attention, QA  
[Paper Link](http://arxiv.org/abs/2308.04156v1)  

---


**ABSTRACT**  
Stereoscopic image quality assessment (SIQA) plays a crucial role in evaluating and improving the visual experience of 3D content. Existing binocular properties and attention-based methods for SIQA have achieved promising performance. However, these bottom-up approaches are inadequate in exploiting the inherent characteristics of the human visual system (HVS). This paper presents a novel network for SIQA via stereo attention, employing a top-down perspective to guide the quality assessment process. Our proposed method realizes the guidance from high-level binocular signals down to low-level monocular signals, while the binocular and monocular information can be calibrated progressively throughout the processing pipeline. We design a generalized Stereo AttenTion (SAT) block to implement the top-down philosophy in stereo perception. This block utilizes the fusion-generated attention map as a high-level binocular modulator, influencing the representation of two low-level monocular features. Additionally, we introduce an Energy Coefficient (EC) to account for recent findings indicating that binocular responses in the primate primary visual cortex are less than the sum of monocular responses. The adaptive EC can tune the magnitude of binocular response flexibly, thus enhancing the formation of robust binocular features within our framework. To extract the most discriminative quality information from the summation and subtraction of the two branches of monocular features, we utilize a dual-pooling strategy that applies min-pooling and max-pooling operations to the respective branches. Experimental results highlight the superiority of our top-down method in simulating the property of visual perception and advancing the state-of-the-art in the SIQA field. The code of this work is available at https://github.com/Fanning-Zhang/SATNet.

{{</citation>}}


### (73/119) Empowering Vision-Language Models to Follow Interleaved Vision-Language Instructions (Juncheng Li et al., 2023)

{{<citation>}}

Juncheng Li, Kaihang Pan, Zhiqi Ge, Minghe Gao, Hanwang Zhang, Wei Ji, Wenqiao Zhang, Tat-Seng Chua, Siliang Tang, Yueting Zhuang. (2023)  
**Empowering Vision-Language Models to Follow Interleaved Vision-Language Instructions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2308.04152v2)  

---


**ABSTRACT**  
Multimodal Large Language Models (MLLMs) have recently sparked significant interest, which demonstrates emergent capabilities to serve as a general-purpose model for various vision-language tasks. However, existing methods mainly focus on limited types of instructions with a single image as visual context, which hinders the widespread availability of MLLMs. In this paper, we introduce the I4 benchmark to comprehensively evaluate the instruction following ability on complicated interleaved vision-language instructions, which involve intricate image-text sequential context, covering a diverse range of scenarios (e.g., visually-rich webpages/textbooks, lecture slides, embodied dialogue). Systematic evaluation on our I4 benchmark reveals a common defect of existing methods: the Visual Prompt Generator (VPG) trained on image-captioning alignment objective tends to attend to common foreground information for captioning but struggles to extract specific information required by particular tasks. To address this issue, we propose a generic and lightweight controllable knowledge re-injection module, which utilizes the sophisticated reasoning ability of LLMs to control the VPG to conditionally extract instruction-specific visual information and re-inject it into the LLM. Further, we introduce an annotation-free cross-attention guided counterfactual image training strategy to methodically learn the proposed module by collaborating a cascade of foundation models. Enhanced by the proposed module and training strategy, we present Cheetor, a Transformer-based MLLM that can effectively handle a wide variety of interleaved vision-language instructions and achieves state-of-the-art zero-shot performance across all tasks of I4, without high-quality multimodal instruction tuning data. Cheetor also exhibits competitive performance compared with state-of-the-art instruction tuned models on MME benchmark.

{{</citation>}}


### (74/119) Class-level Structural Relation Modelling and Smoothing for Visual Representation Learning (Zitan Chen et al., 2023)

{{<citation>}}

Zitan Chen, Zhuang Qi, Xiao Cao, Xiangxian Li, Xiangxu Meng, Lei Meng. (2023)  
**Class-level Structural Relation Modelling and Smoothing for Visual Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.04142v1)  

---


**ABSTRACT**  
Representation learning for images has been advanced by recent progress in more complex neural models such as the Vision Transformers and new learning theories such as the structural causal models. However, these models mainly rely on the classification loss to implicitly regularize the class-level data distributions, and they may face difficulties when handling classes with diverse visual patterns. We argue that the incorporation of the structural information between data samples may improve this situation. To achieve this goal, this paper presents a framework termed \textbf{C}lass-level Structural Relation Modeling and Smoothing for Visual Representation Learning (CSRMS), which includes the Class-level Relation Modelling, Class-aware Graph Sampling, and Relational Graph-Guided Representation Learning modules to model a relational graph of the entire dataset and perform class-aware smoothing and regularization operations to alleviate the issue of intra-class visual diversity and inter-class similarity. Specifically, the Class-level Relation Modelling module uses a clustering algorithm to learn the data distributions in the feature space and identify three types of class-level sample relations for the training set; Class-aware Graph Sampling module extends typical training batch construction process with three strategies to sample dataset-level sub-graphs; and Relational Graph-Guided Representation Learning module employs a graph convolution network with knowledge-guided smoothing operations to ease the projection from different visual patterns to the same class. Experiments demonstrate the effectiveness of structured knowledge modelling for enhanced representation learning and show that CSRMS can be incorporated with any state-of-the-art visual representation learning models for performance gains. The source codes and demos have been released at https://github.com/czt117/CSRMS.

{{</citation>}}


### (75/119) OmniDataComposer: A Unified Data Structure for Multimodal Data Fusion and Infinite Data Generation (Dongyang Yu et al., 2023)

{{<citation>}}

Dongyang Yu, Shihao Wang, Yuan Fang, Wangpeng An. (2023)  
**OmniDataComposer: A Unified Data Structure for Multimodal Data Fusion and Infinite Data Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: AI, ChatGPT, GPT, OCR, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2308.04126v1)  

---


**ABSTRACT**  
This paper presents OmniDataComposer, an innovative approach for multimodal data fusion and unlimited data generation with an intent to refine and uncomplicate interplay among diverse data modalities. Coming to the core breakthrough, it introduces a cohesive data structure proficient in processing and merging multimodal data inputs, which include video, audio, and text. Our crafted algorithm leverages advancements across multiple operations such as video/image caption extraction, dense caption extraction, Automatic Speech Recognition (ASR), Optical Character Recognition (OCR), Recognize Anything Model(RAM), and object tracking. OmniDataComposer is capable of identifying over 6400 categories of objects, substantially broadening the spectrum of visual information. It amalgamates these diverse modalities, promoting reciprocal enhancement among modalities and facilitating cross-modal data correction. \textbf{The final output metamorphoses each video input into an elaborate sequential document}, virtually transmuting videos into thorough narratives, making them easier to be processed by large language models. Future prospects include optimizing datasets for each modality to encourage unlimited data generation. This robust base will offer priceless insights to models like ChatGPT, enabling them to create higher quality datasets for video captioning and easing question-answering tasks based on video content. OmniDataComposer inaugurates a new stage in multimodal learning, imparting enormous potential for augmenting AI's understanding and generation of complex, real-world data.

{{</citation>}}


### (76/119) An Empirical Analysis of Range for 3D Object Detection (Neehar Peri et al., 2023)

{{<citation>}}

Neehar Peri, Mengtian Li, Benjamin Wilson, Yu-Xiong Wang, James Hays, Deva Ramanan. (2023)  
**An Empirical Analysis of Range for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.04054v1)  

---


**ABSTRACT**  
LiDAR-based 3D detection plays a vital role in autonomous navigation. Surprisingly, although autonomous vehicles (AVs) must detect both near-field objects (for collision avoidance) and far-field objects (for longer-term planning), contemporary benchmarks focus only on near-field 3D detection. However, AVs must detect far-field objects for safe navigation. In this paper, we present an empirical analysis of far-field 3D detection using the long-range detection dataset Argoverse 2.0 to better understand the problem, and share the following insight: near-field LiDAR measurements are dense and optimally encoded by small voxels, while far-field measurements are sparse and are better encoded with large voxels. We exploit this observation to build a collection of range experts tuned for near-vs-far field detection, and propose simple techniques to efficiently ensemble models for long-range detection that improve efficiency by 33% and boost accuracy by 3.2% CDS.

{{</citation>}}


### (77/119) SODFormer: Streaming Object Detection with Transformer Using Events and Frames (Dianze Li et al., 2023)

{{<citation>}}

Dianze Li, Jianing Li, Yonghong Tian. (2023)  
**SODFormer: Streaming Object Detection with Transformer Using Events and Frames**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2308.04047v1)  

---


**ABSTRACT**  
DAVIS camera, streaming two complementary sensing modalities of asynchronous events and frames, has gradually been used to address major object detection challenges (e.g., fast motion blur and low-light). However, how to effectively leverage rich temporal cues and fuse two heterogeneous visual streams remains a challenging endeavor. To address this challenge, we propose a novel streaming object detector with Transformer, namely SODFormer, which first integrates events and frames to continuously detect objects in an asynchronous manner. Technically, we first build a large-scale multimodal neuromorphic object detection dataset (i.e., PKU-DAVIS-SOD) over 1080.1k manual labels. Then, we design a spatiotemporal Transformer architecture to detect objects via an end-to-end sequence prediction problem, where the novel temporal Transformer module leverages rich temporal cues from two visual streams to improve the detection performance. Finally, an asynchronous attention-based fusion module is proposed to integrate two heterogeneous sensing modalities and take complementary advantages from each end, which can be queried at any time to locate objects and break through the limited output frequency from synchronized frame-based fusion strategies. The results show that the proposed SODFormer outperforms four state-of-the-art methods and our eight baselines by a significant margin. We also show that our unifying framework works well even in cases where the conventional frame-based camera fails, e.g., high-speed motion and low-light conditions. Our dataset and code can be available at https://github.com/dianzl/SODFormer.

{{</citation>}}


### (78/119) Synthetic Augmentation with Large-scale Unconditional Pre-training (Jiarong Ye et al., 2023)

{{<citation>}}

Jiarong Ye, Haomiao Ni, Peng Jin, Sharon X. Huang, Yuan Xue. (2023)  
**Synthetic Augmentation with Large-scale Unconditional Pre-training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.04020v1)  

---


**ABSTRACT**  
Deep learning based medical image recognition systems often require a substantial amount of training data with expert annotations, which can be expensive and time-consuming to obtain. Recently, synthetic augmentation techniques have been proposed to mitigate the issue by generating realistic images conditioned on class labels. However, the effectiveness of these methods heavily depends on the representation capability of the trained generative model, which cannot be guaranteed without sufficient labeled training data. To further reduce the dependency on annotated data, we propose a synthetic augmentation method called HistoDiffusion, which can be pre-trained on large-scale unlabeled datasets and later applied to a small-scale labeled dataset for augmented training. In particular, we train a latent diffusion model (LDM) on diverse unlabeled datasets to learn common features and generate realistic images without conditional inputs. Then, we fine-tune the model with classifier guidance in latent space on an unseen labeled dataset so that the model can synthesize images of specific categories. Additionally, we adopt a selective mechanism to only add synthetic samples with high confidence of matching to target labels. We evaluate our proposed method by pre-training on three histopathology datasets and testing on a histopathology dataset of colorectal cancer (CRC) excluded from the pre-training datasets. With HistoDiffusion augmentation, the classification accuracy of a backbone classifier is remarkably improved by 6.4% using a small set of the original labels. Our code is available at https://github.com/karenyyy/HistoDiffAug.

{{</citation>}}


### (79/119) Hierarchical Visual Primitive Experts for Compositional Zero-Shot Learning (Hanjae Kim et al., 2023)

{{<citation>}}

Hanjae Kim, Jiyoung Lee, Seongheon Park, Kwanghoon Sohn. (2023)  
**Hierarchical Visual Primitive Experts for Compositional Zero-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Transformer, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.04016v1)  

---


**ABSTRACT**  
Compositional zero-shot learning (CZSL) aims to recognize unseen compositions with prior knowledge of known primitives (attribute and object). Previous works for CZSL often suffer from grasping the contextuality between attribute and object, as well as the discriminability of visual features, and the long-tailed distribution of real-world compositional data. We propose a simple and scalable framework called Composition Transformer (CoT) to address these issues. CoT employs object and attribute experts in distinctive manners to generate representative embeddings, using the visual network hierarchically. The object expert extracts representative object embeddings from the final layer in a bottom-up manner, while the attribute expert makes attribute embeddings in a top-down manner with a proposed object-guided attention module that models contextuality explicitly. To remedy biased prediction caused by imbalanced data distribution, we develop a simple minority attribute augmentation (MAA) that synthesizes virtual samples by mixing two images and oversampling minority attribute classes. Our method achieves SoTA performance on several benchmarks, including MIT-States, C-GQA, and VAW-CZSL. We also demonstrate the effectiveness of CoT in improving visual discrimination and addressing the model bias from the imbalanced data distribution. The code is available at https://github.com/HanjaeKim98/CoT.

{{</citation>}}


### (80/119) Few-shot medical image classification with simple shape and texture text descriptors using vision-language models (Michal Byra et al., 2023)

{{<citation>}}

Michal Byra, Muhammad Febrian Rachmadi, Henrik Skibbe. (2023)  
**Few-shot medical image classification with simple shape and texture text descriptors using vision-language models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.04005v1)  

---


**ABSTRACT**  
In this work, we investigate the usefulness of vision-language models (VLMs) and large language models for binary few-shot classification of medical images. We utilize the GPT-4 model to generate text descriptors that encapsulate the shape and texture characteristics of objects in medical images. Subsequently, these GPT-4 generated descriptors, alongside VLMs pre-trained on natural images, are employed to classify chest X-rays and breast ultrasound images. Our results indicate that few-shot classification of medical images using VLMs and GPT-4 generated descriptors is a viable approach. However, accurate classification requires to exclude certain descriptors from the calculations of the classification scores. Moreover, we assess the ability of VLMs to evaluate shape features in breast mass ultrasound images. We further investigate the degree of variability among the sets of text descriptors produced by GPT-4. Our work provides several important insights about the application of VLMs for medical image analysis.

{{</citation>}}


### (81/119) PARTNER: Level up the Polar Representation for LiDAR 3D Object Detection (Ming Nie et al., 2023)

{{<citation>}}

Ming Nie, Yujing Xue, Chunwei Wang, Chaoqiang Ye, Hang Xu, Xinge Zhu, Qingqiu Huang, Michael Bi Mi, Xinchao Wang, Li Zhang. (2023)  
**PARTNER: Level up the Polar Representation for LiDAR 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NER, Object Detection  
[Paper Link](http://arxiv.org/abs/2308.03982v1)  

---


**ABSTRACT**  
Recently, polar-based representation has shown promising properties in perceptual tasks. In addition to Cartesian-based approaches, which separate point clouds unevenly, representing point clouds as polar grids has been recognized as an alternative due to (1) its advantage in robust performance under different resolutions and (2) its superiority in streaming-based approaches. However, state-of-the-art polar-based detection methods inevitably suffer from the feature distortion problem because of the non-uniform division of polar representation, resulting in a non-negligible performance gap compared to Cartesian-based approaches. To tackle this issue, we present PARTNER, a novel 3D object detector in the polar coordinate. PARTNER alleviates the dilemma of feature distortion with global representation re-alignment and facilitates the regression by introducing instance-level geometric information into the detection head. Extensive experiments show overwhelming advantages in streaming-based detection and different resolutions. Furthermore, our method outperforms the previous polar-based works with remarkable margins of 3.68% and 9.15% on Waymo and ONCE validation set, thus achieving competitive results over the state-of-the-art methods.

{{</citation>}}


### (82/119) PAIF: Perception-Aware Infrared-Visible Image Fusion for Attack-Tolerant Semantic Segmentation (Zhu Liu et al., 2023)

{{<citation>}}

Zhu Liu, Jinyuan Liu, Benzhuang Zhang, Long Ma, Xin Fan, Risheng Liu. (2023)  
**PAIF: Perception-Aware Infrared-Visible Image Fusion for Attack-Tolerant Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.03979v1)  

---


**ABSTRACT**  
Infrared and visible image fusion is a powerful technique that combines complementary information from different modalities for downstream semantic perception tasks. Existing learning-based methods show remarkable performance, but are suffering from the inherent vulnerability of adversarial attacks, causing a significant decrease in accuracy. In this work, a perception-aware fusion framework is proposed to promote segmentation robustness in adversarial scenes. We first conduct systematic analyses about the components of image fusion, investigating the correlation with segmentation robustness under adversarial perturbations. Based on these analyses, we propose a harmonized architecture search with a decomposition-based structure to balance standard accuracy and robustness. We also propose an adaptive learning strategy to improve the parameter robustness of image fusion, which can learn effective feature extraction under diverse adversarial perturbations. Thus, the goals of image fusion (\textit{i.e.,} extracting complementary features from source modalities and defending attack) can be realized from the perspectives of architectural and learning strategies. Extensive experimental results demonstrate that our scheme substantially enhances the robustness, with gains of 15.3% mIOU of segmentation in the adversarial scene, compared with advanced competitors. The source codes are available at https://github.com/LiuZhu-CV/PAIF.

{{</citation>}}


### (83/119) PUG: Photorealistic and Semantically Controllable Synthetic Data for Representation Learning (Florian Bordes et al., 2023)

{{<citation>}}

Florian Bordes, Shashank Shekhar, Mark Ibrahim, Diane Bouchacourt, Pascal Vincent, Ari S. Morcos. (2023)  
**PUG: Photorealistic and Semantically Controllable Synthetic Data for Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.03977v1)  

---


**ABSTRACT**  
Synthetic image datasets offer unmatched advantages for designing and evaluating deep neural networks: they make it possible to (i) render as many data samples as needed, (ii) precisely control each scene and yield granular ground truth labels (and captions), (iii) precisely control distribution shifts between training and testing to isolate variables of interest for sound experimentation. Despite such promise, the use of synthetic image data is still limited -- and often played down -- mainly due to their lack of realism. Most works therefore rely on datasets of real images, which have often been scraped from public images on the internet, and may have issues with regards to privacy, bias, and copyright, while offering little control over how objects precisely appear. In this work, we present a path to democratize the use of photorealistic synthetic data: we develop a new generation of interactive environments for representation learning research, that offer both controllability and realism. We use the Unreal Engine, a powerful game engine well known in the entertainment industry, to produce PUG (Photorealistic Unreal Graphics) environments and datasets for representation learning. In this paper, we demonstrate the potential of PUG to enable more rigorous evaluations of vision models.

{{</citation>}}


### (84/119) Prompted Contrast with Masked Motion Modeling: Towards Versatile 3D Action Representation Learning (Jiahang Zhang et al., 2023)

{{<citation>}}

Jiahang Zhang, Lilang Lin, Jiaying Liu. (2023)  
**Prompted Contrast with Masked Motion Modeling: Towards Versatile 3D Action Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.03975v1)  

---


**ABSTRACT**  
Self-supervised learning has proved effective for skeleton-based human action understanding, which is an important yet challenging topic. Previous works mainly rely on contrastive learning or masked motion modeling paradigm to model the skeleton relations. However, the sequence-level and joint-level representation learning cannot be effectively and simultaneously handled by these methods. As a result, the learned representations fail to generalize to different downstream tasks. Moreover, combining these two paradigms in a naive manner leaves the synergy between them untapped and can lead to interference in training. To address these problems, we propose Prompted Contrast with Masked Motion Modeling, PCM$^{\rm 3}$, for versatile 3D action representation learning. Our method integrates the contrastive learning and masked prediction tasks in a mutually beneficial manner, which substantially boosts the generalization capacity for various downstream tasks. Specifically, masked prediction provides novel training views for contrastive learning, which in turn guides the masked prediction training with high-level semantic information. Moreover, we propose a dual-prompted multi-task pretraining strategy, which further improves model representations by reducing the interference caused by learning the two different pretext tasks. Extensive experiments on five downstream tasks under three large-scale datasets are conducted, demonstrating the superior generalization capacity of PCM$^{\rm 3}$ compared to the state-of-the-art works. Our project is publicly available at: https://jhang2020.github.io/Projects/PCM3/PCM3.html .

{{</citation>}}


### (85/119) CheXFusion: Effective Fusion of Multi-View Features using Transformers for Long-Tailed Chest X-Ray Classification (Dongkyun Kim, 2023)

{{<citation>}}

Dongkyun Kim. (2023)  
**CheXFusion: Effective Fusion of Multi-View Features using Transformers for Long-Tailed Chest X-Ray Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.03968v1)  

---


**ABSTRACT**  
Medical image classification poses unique challenges due to the long-tailed distribution of diseases, the co-occurrence of diagnostic findings, and the multiple views available for each study or patient. This paper introduces our solution to the ICCV CVAMD 2023 Shared Task on CXR-LT: Multi-Label Long-Tailed Classification on Chest X-Rays. Our approach introduces CheXFusion, a transformer-based fusion module incorporating multi-view images. The fusion module, guided by self-attention and cross-attention mechanisms, efficiently aggregates multi-view features while considering label co-occurrence. Furthermore, we explore data balancing and self-training methods to optimize the model's performance. Our solution achieves state-of-the-art results with 0.372 mAP in the MIMIC-CXR test set, securing 1st place in the competition. Our success in the task underscores the significance of considering multi-view settings, class imbalance, and label co-occurrence in medical image classification. Public code is available at https://github.com/dongkyuk/CXR-LT-public-solution

{{</citation>}}


## cs.IR (3)



### (86/119) RECipe: Does a Multi-Modal Recipe Knowledge Graph Fit a Multi-Purpose Recommendation System? (Ali Pesaranghader et al., 2023)

{{<citation>}}

Ali Pesaranghader, Touqir Sajed. (2023)  
**RECipe: Does a Multi-Modal Recipe Knowledge Graph Fit a Multi-Purpose Recommendation System?**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Knowledge Graph, Microsoft, NLP  
[Paper Link](http://arxiv.org/abs/2308.04579v1)  

---


**ABSTRACT**  
Over the past two decades, recommendation systems (RSs) have used machine learning (ML) solutions to recommend items, e.g., movies, books, and restaurants, to clients of a business or an online platform. Recipe recommendation, however, has not yet received much attention compared to those applications. We introduce RECipe as a multi-purpose recipe recommendation framework with a multi-modal knowledge graph (MMKG) backbone. The motivation behind RECipe is to go beyond (deep) neural collaborative filtering (NCF) by recommending recipes to users when they query in natural language or by providing an image. RECipe consists of 3 subsystems: (1) behavior-based recommender, (2) review-based recommender, and (3) image-based recommender. Each subsystem relies on the embedding representations of entities and relations in the graph. We first obtain (pre-trained) embedding representations of textual entities, such as reviews or ingredients, from a fine-tuned model of Microsoft's MPNet. We initialize the weights of the entities with these embeddings to train our knowledge graph embedding (KGE) model. For the visual component, i.e., recipe images, we develop a KGE-Guided variational autoencoder (KG-VAE) to learn the distribution of images and their latent representations. Once KGE and KG-VAE models are fully trained, we use them as a multi-purpose recommendation framework. For benchmarking, we created two knowledge graphs (KGs) from public datasets on Kaggle for recipe recommendation. Our experiments show that the KGE models have comparable performance to the neural solutions. We also present pre-trained NLP embeddings to address important applications such as zero-shot inference for new users (or the cold start problem) and conditional recommendation with respect to recipe categories. We eventually demonstrate the application of RECipe in a multi-purpose recommendation setting.

{{</citation>}}


### (87/119) Online Distillation-enhanced Multi-modal Transformer for Sequential Recommendation (Wei Ji et al., 2023)

{{<citation>}}

Wei Ji, Xiangyan Liu, An Zhang, Yinwei Wei, Yongxin Ni, Xiang Wang. (2023)  
**Online Distillation-enhanced Multi-modal Transformer for Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.04067v1)  

---


**ABSTRACT**  
Multi-modal recommendation systems, which integrate diverse types of information, have gained widespread attention in recent years. However, compared to traditional collaborative filtering-based multi-modal recommendation systems, research on multi-modal sequential recommendation is still in its nascent stages. Unlike traditional sequential recommendation models that solely rely on item identifier (ID) information and focus on network structure design, multi-modal recommendation models need to emphasize item representation learning and the fusion of heterogeneous data sources. This paper investigates the impact of item representation learning on downstream recommendation tasks and examines the disparities in information fusion at different stages. Empirical experiments are conducted to demonstrate the need to design a framework suitable for collaborative learning and fusion of diverse information. Based on this, we propose a new model-agnostic framework for multi-modal sequential recommendation tasks, called Online Distillation-enhanced Multi-modal Transformer (ODMT), to enhance feature interaction and mutual learning among multi-source input (ID, text, and image), while avoiding conflicts among different features during training, thereby improving recommendation accuracy. To be specific, we first introduce an ID-aware Multi-modal Transformer module in the item representation learning stage to facilitate information interaction among different features. Secondly, we employ an online distillation training strategy in the prediction optimization stage to make multi-source data learn from each other and improve prediction robustness. Experimental results on a video content recommendation dataset and three e-commerce recommendation datasets demonstrate the effectiveness of the proposed two modules, which is approximately 10% improvement in performance compared to baseline models.

{{</citation>}}


### (88/119) Multi-Granularity Attention Model for Group Recommendation (Jianye Ji et al., 2023)

{{<citation>}}

Jianye Ji, Jiayan Pei, Shaochuan Lin, Taotao Zhou, Hengxu He, Jia Jia, Ning Hu. (2023)  
**Multi-Granularity Attention Model for Group Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.04017v1)  

---


**ABSTRACT**  
Group recommendation provides personalized recommendations to a group of users based on their shared interests, preferences, and characteristics. Current studies have explored different methods for integrating individual preferences and making collective decisions that benefit the group as a whole. However, most of them heavily rely on users with rich behavior and ignore latent preferences of users with relatively sparse behavior, leading to insufficient learning of individual interests. To address this challenge, we present the Multi-Granularity Attention Model (MGAM), a novel approach that utilizes multiple levels of granularity (i.e., subsets, groups, and supersets) to uncover group members' latent preferences and mitigate recommendation noise. Specially, we propose a Subset Preference Extraction module that enhances the representation of users' latent subset-level preferences by incorporating their previous interactions with items and utilizing a hierarchical mechanism. Additionally, our method introduces a Group Preference Extraction module and a Superset Preference Extraction module, which explore users' latent preferences on two levels: the group-level, which maintains users' original preferences, and the superset-level, which includes group-group exterior information. By incorporating the subset-level embedding, group-level embedding, and superset-level embedding, our proposed method effectively reduces group recommendation noise across multiple granularities and comprehensively learns individual interests. Extensive offline and online experiments have demonstrated the superiority of our method in terms of performance.

{{</citation>}}


## cs.NI (3)



### (89/119) Resource Cooperation in MEC and SDN based Vehicular Networks (Beiran Chen et al., 2023)

{{<citation>}}

Beiran Chen, Marco Ruffini. (2023)  
**Resource Cooperation in MEC and SDN based Vehicular Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04564v1)  

---


**ABSTRACT**  
Internet of Things (IoT) systems require highly scalable infrastructure to adaptively provide services to meet various performance requirements. Combining Software-Defined Networking (SDN) with Mobile Edge Cloud (MEC) technology brings more flexibility for IoT systems. We present a four-tier task processing architecture for MEC and vehicular networks, which includes processing tasks locally within a vehicle, on neighboring vehicles, on an edge cloud, and on a remote cloud. The flexible network connection is controlled by SDN. We propose a CPU resource allocation algorithm, called Partial Idle Resource Strategy (PIRS) with Vehicle to Vehicle (V2V) communications, based on Asymmetric Nash Bargaining Solution (ANBS) in Game Theory. PIRS encourages vehicles in the same location to cooperate by sharing part of their spare CPU resources. In our simulations, we adopt four applications running on the vehicles to generate workload. We compare the proposed algorithm with Non-Cooperation Strategy (NCS) and All Idle Resource Strategy (AIRS). In NCS, the vehicles execute tasks generated by the applications in their own On-Board Units (OBU), while in AIRS vehicles provide all their CPU resources to help other vehicles offloading requests. Our simulation results show that our PIRS strategy can execute more tasks on the V2V layer and lead to fewer number of task (and their length) to be offloaded to the cloud, reaching up to 28% improvement compared to NCS and up to 10% improvement compared to AIRS.

{{</citation>}}


### (90/119) Heterogeneous 360 Degree Videos in Metaverse: Differentiated Reinforcement Learning Approaches (Wenhan Yu et al., 2023)

{{<citation>}}

Wenhan Yu, Jun Zhao. (2023)  
**Heterogeneous 360 Degree Videos in Metaverse: Differentiated Reinforcement Learning Approaches**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.04083v1)  

---


**ABSTRACT**  
Advanced video technologies are driving the development of the futuristic Metaverse, which aims to connect users from anywhere and anytime. As such, the use cases for users will be much more diverse, leading to a mix of 360-degree videos with two types: non-VR and VR 360-degree videos. This paper presents a novel Quality of Service model for heterogeneous 360-degree videos with different requirements for frame rates and cybersickness. We propose a frame-slotted structure and conduct frame-wise optimization using self-designed differentiated deep reinforcement learning algorithms. Specifically, we design two structures, Separate Input Differentiated Output (SIDO) and Merged Input Differentiated Output (MIDO), for this heterogeneous scenario. We also conduct comprehensive experiments to demonstrate their effectiveness.

{{</citation>}}


### (91/119) Adapting Foundation Models for Information Synthesis of Wireless Communication Specifications (Manikanta Kotaru, 2023)

{{<citation>}}

Manikanta Kotaru. (2023)  
**Adapting Foundation Models for Information Synthesis of Wireless Communication Specifications**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-IR, cs-NI, cs.NI  
Keywords: BERT, BLEU, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.04033v2)  

---


**ABSTRACT**  
Existing approaches to understanding, developing and researching modern wireless communication technologies involves time-intensive and arduous process of sifting through numerous webpages and technical specification documents, gathering the required information and synthesizing it. This paper presents NextGen Communications Copilot, a conversational artificial intelligence tool for information synthesis of wireless communication specifications. The system builds on top of recent advancements in foundation models and consists of three key additional components: a domain-specific database, a context extractor, and a feedback mechanism. The system appends user queries with concise and query-dependent contextual information extracted from a database of wireless technical specifications and incorporates tools for expert feedback and data contributions. On evaluation using a benchmark dataset of queries and reference responses created by subject matter experts, the system demonstrated more relevant and accurate answers with an average BLEU score and BERTScore F1-measure of 0.37 and 0.79 respectively compared to the corresponding values of 0.07 and 0.59 achieved by state-of-the-art tools like ChatGPT.

{{</citation>}}


## eess.IV (3)



### (92/119) Improving Medical Image Classification in Noisy Labels Using Only Self-supervised Pretraining (Bidur Khanal et al., 2023)

{{<citation>}}

Bidur Khanal, Binod Bhattarai, Bishesh Khanal, Cristian A. Linte. (2023)  
**Improving Medical Image Classification in Noisy Labels Using Only Self-supervised Pretraining**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2308.04551v1)  

---


**ABSTRACT**  
Noisy labels hurt deep learning-based supervised image classification performance as the models may overfit the noise and learn corrupted feature extractors. For natural image classification training with noisy labeled data, model initialization with contrastive self-supervised pretrained weights has shown to reduce feature corruption and improve classification performance. However, no works have explored: i) how other self-supervised approaches, such as pretext task-based pretraining, impact the learning with noisy label, and ii) any self-supervised pretraining methods alone for medical images in noisy label settings. Medical images often feature smaller datasets and subtle inter class variations, requiring human expertise to ensure correct classification. Thus, it is not clear if the methods improving learning with noisy labels in natural image datasets such as CIFAR would also help with medical images. In this work, we explore contrastive and pretext task-based self-supervised pretraining to initialize the weights of a deep learning classification model for two medical datasets with self-induced noisy labels -- NCT-CRC-HE-100K tissue histological images and COVID-QU-Ex chest X-ray images. Our results show that models initialized with pretrained weights obtained from self-supervised learning can effectively learn better features and improve robustness against noisy labels.

{{</citation>}}


### (93/119) Data Augmentation-Based Unsupervised Domain Adaptation In Medical Imaging (Sebastian Nørgaard Llambias et al., 2023)

{{<citation>}}

Sebastian Nørgaard Llambias, Mads Nielsen, Mostafa Mehdipour Ghazi. (2023)  
**Data Augmentation-Based Unsupervised Domain Adaptation In Medical Imaging**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.04395v1)  

---


**ABSTRACT**  
Deep learning-based models in medical imaging often struggle to generalize effectively to new scans due to data heterogeneity arising from differences in hardware, acquisition parameters, population, and artifacts. This limitation presents a significant challenge in adopting machine learning models for clinical practice. We propose an unsupervised method for robust domain adaptation in brain MRI segmentation by leveraging MRI-specific augmentation techniques. To evaluate the effectiveness of our method, we conduct extensive experiments across diverse datasets, modalities, and segmentation tasks, comparing against the state-of-the-art methods. The results show that our proposed approach achieves high accuracy, exhibits broad applicability, and showcases remarkable robustness against domain shift in various tasks, surpassing the state-of-the-art performance in the majority of cases.

{{</citation>}}


### (94/119) SDLFormer: A Sparse and Dense Locality-enhanced Transformer for Accelerated MR Image Reconstruction (Rahul G. S. et al., 2023)

{{<citation>}}

Rahul G. S., Sriprabha Ramnarayanan, Mohammad Al Fahim, Keerthi Ram, Preejith S. P, Mohanasankar Sivaprakasam. (2023)  
**SDLFormer: A Sparse and Dense Locality-enhanced Transformer for Accelerated MR Image Reconstruction**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.04262v1)  

---


**ABSTRACT**  
Transformers have emerged as viable alternatives to convolutional neural networks owing to their ability to learn non-local region relationships in the spatial domain. The self-attention mechanism of the transformer enables transformers to capture long-range dependencies in the images, which might be desirable for accelerated MRI image reconstruction as the effect of undersampling is non-local in the image domain. Despite its computational efficiency, the window-based transformers suffer from restricted receptive fields as the dependencies are limited to within the scope of the image windows. We propose a window-based transformer network that integrates dilated attention mechanism and convolution for accelerated MRI image reconstruction. The proposed network consists of dilated and dense neighborhood attention transformers to enhance the distant neighborhood pixel relationship and introduce depth-wise convolutions within the transformer module to learn low-level translation invariant features for accelerated MRI image reconstruction. The proposed model is trained in a self-supervised manner. We perform extensive experiments for multi-coil MRI acceleration for coronal PD, coronal PDFS and axial T2 contrasts with 4x and 5x under-sampling in self-supervised learning based on k-space splitting. We compare our method against other reconstruction architectures and the parallel domain self-supervised learning baseline. Results show that the proposed model exhibits improvement margins of (i) around 1.40 dB in PSNR and around 0.028 in SSIM on average over other architectures (ii) around 1.44 dB in PSNR and around 0.029 in SSIM over parallel domain self-supervised learning. The code is available at https://github.com/rahul-gs-16/sdlformer.git

{{</citation>}}


## q-bio.GN (1)



### (95/119) Vector Embeddings by Sequence Similarity and Context for Improved Compression, Similarity Search, Clustering, Organization, and Manipulation of cDNA Libraries (Daniel H. Um et al., 2023)

{{<citation>}}

Daniel H. Um, David A. Knowles, Gail E. Kaiser. (2023)  
**Vector Embeddings by Sequence Similarity and Context for Improved Compression, Similarity Search, Clustering, Organization, and Manipulation of cDNA Libraries**  

---
Primary Category: q-bio.GN  
Categories: cs-LG, q-bio-GN, q-bio.GN  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.05118v1)  

---


**ABSTRACT**  
This paper demonstrates the utility of organized numerical representations of genes in research involving flat string gene formats (i.e., FASTA/FASTQ5). FASTA/FASTQ files have several current limitations, such as their large file sizes, slow processing speeds for mapping and alignment, and contextual dependencies. These challenges significantly hinder investigations and tasks that involve finding similar sequences. The solution lies in transforming sequences into an alternative representation that facilitates easier clustering into similar groups compared to the raw sequences themselves. By assigning a unique vector embedding to each short sequence, it is possible to more efficiently cluster and improve upon compression performance for the string representations of cDNA libraries. Furthermore, through learning alternative coordinate vector embeddings based on the contexts of codon triplets, we can demonstrate clustering based on amino acid properties. Finally, using this sequence embedding method to encode barcodes and cDNA sequences, we can improve the time complexity of the similarity search by coupling vector embeddings with an algorithm that determines the proximity of vectors in Euclidean space; this allows us to perform sequence similarity searches in a quicker and more modular fashion.

{{</citation>}}


## cs.GT (1)



### (96/119) Fine-Tuning Games: Bargaining and Adaptation for General-Purpose Models (Benjamin Laufer et al., 2023)

{{<citation>}}

Benjamin Laufer, Jon Kleinberg, Hoda Heidari. (2023)  
**Fine-Tuning Games: Bargaining and Adaptation for General-Purpose Models**  

---
Primary Category: cs.GT  
Categories: cs-AI, cs-GT, cs.GT, econ-TH  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04399v1)  

---


**ABSTRACT**  
Major advances in Machine Learning (ML) and Artificial Intelligence (AI) increasingly take the form of developing and releasing general-purpose models. These models are designed to be adapted by other businesses and agencies to perform a particular, domain-specific function. This process has become known as adaptation or fine-tuning. This paper offers a model of the fine-tuning process where a Generalist brings the technological product (here an ML model) to a certain level of performance, and one or more Domain-specialist(s) adapts it for use in a particular domain. Both entities are profit-seeking and incur costs when they invest in the technology, and they must reach a bargaining agreement on how to share the revenue for the technology to reach the market. For a relatively general class of cost and revenue functions, we characterize the conditions under which the fine-tuning game yields a profit-sharing solution. We observe that any potential domain-specialization will either contribute, free-ride, or abstain in their uptake of the technology, and we provide conditions yielding these different strategies. We show how methods based on bargaining solutions and sub-game perfect equilibria provide insights into the strategic behavior of firms in these types of interactions, and we find that profit-sharing can still arise even when one firm has significantly higher costs than another. We also provide methods for identifying Pareto-optimal bargaining arrangements for a general set of utility functions.

{{</citation>}}


## cs.HC (6)



### (97/119) Understanding the Effect of Counterfactual Explanations on Trust and Reliance on AI for Human-AI Collaborative Clinical Decision Making (Min Hun Lee et al., 2023)

{{<citation>}}

Min Hun Lee, Chong Jun Chew. (2023)  
**Understanding the Effect of Counterfactual Explanations on Trust and Reliance on AI for Human-AI Collaborative Clinical Decision Making**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-LG, cs.HC  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2308.04375v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) is increasingly being considered to assist human decision-making in high-stake domains (e.g. health). However, researchers have discussed an issue that humans can over-rely on wrong suggestions of the AI model instead of achieving human AI complementary performance. In this work, we utilized salient feature explanations along with what-if, counterfactual explanations to make humans review AI suggestions more analytically to reduce overreliance on AI and explored the effect of these explanations on trust and reliance on AI during clinical decision-making. We conducted an experiment with seven therapists and ten laypersons on the task of assessing post-stroke survivors' quality of motion, and analyzed their performance, agreement level on the task, and reliance on AI without and with two types of AI explanations. Our results showed that the AI model with both salient features and counterfactual explanations assisted therapists and laypersons to improve their performance and agreement level on the task when `right' AI outputs are presented. While both therapists and laypersons over-relied on `wrong' AI outputs, counterfactual explanations assisted both therapists and laypersons to reduce their over-reliance on `wrong' AI outputs by 21\% compared to salient feature explanations. Specifically, laypersons had higher performance degrades by 18.0 f1-score with salient feature explanations and 14.0 f1-score with counterfactual explanations than therapists with performance degrades of 8.6 and 2.8 f1-scores respectively. Our work discusses the potential of counterfactual explanations to better estimate the accuracy of an AI model and reduce over-reliance on `wrong' AI outputs and implications for improving human-AI collaborative decision-making.

{{</citation>}}


### (98/119) Towards an AI to Win Ghana's National Science and Maths Quiz (George Boateng et al., 2023)

{{<citation>}}

George Boateng, Jonathan Abrefah Mensah, Kevin Takyi Yeboah, William Edor, Andrew Kojo Mensah-Onumah, Naafi Dasana Ibrahim, Nana Sam Yeboah. (2023)  
**Towards an AI to Win Ghana's National Science and Maths Quiz**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-CY, cs-HC, cs-SD, cs.HC, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04333v1)  

---


**ABSTRACT**  
Can an AI win Ghana's National Science and Maths Quiz (NSMQ)? That is the question we seek to answer in the NSMQ AI project, an open-source project that is building AI to compete live in the NSMQ and win. The NSMQ is an annual live science and mathematics competition for senior secondary school students in Ghana in which 3 teams of 2 students compete by answering questions across biology, chemistry, physics, and math in 5 rounds over 5 progressive stages until a winning team is crowned for that year. The NSMQ is an exciting live quiz competition with interesting technical challenges across speech-to-text, text-to-speech, question-answering, and human-computer interaction. In this ongoing work that began in January 2023, we give an overview of the project, describe each of the teams, progress made thus far, and the next steps toward our planned launch and debut of the AI in October for NSMQ 2023. An AI that conquers this grand challenge can have real-world impact on education such as enabling millions of students across Africa to have one-on-one learning support from this AI.

{{</citation>}}


### (99/119) Generative AI in Computing Education: Perspectives of Students and Instructors (Cynthia Zastudil et al., 2023)

{{<citation>}}

Cynthia Zastudil, Magdalena Rogalska, Christine Kapp, Jennifer Vaughn, Stephen MacNeil. (2023)  
**Generative AI in Computing Education: Perspectives of Students and Instructors**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.04309v1)  

---


**ABSTRACT**  
Generative models are now capable of producing natural language text that is, in some cases, comparable in quality to the text produced by people. In the computing education context, these models are being used to generate code, code explanations, and programming exercises. The rapid adoption of these models has prompted multiple position papers and workshops which discuss the implications of these models for computing education, both positive and negative. This paper presents results from a series of semi-structured interviews with 12 students and 6 instructors about their awareness, experiences, and preferences regarding the use of tools powered by generative AI in computing classrooms. The results suggest that Generative AI (GAI) tools will play an increasingly significant role in computing education. However, students and instructors also raised numerous concerns about how these models should be integrated to best support the needs and learning goals of students. We also identified interesting tensions and alignments that emerged between how instructors and students prefer to engage with these models. We discuss these results and provide recommendations related to curriculum development, assessment methods, and pedagogical practice. As GAI tools become increasingly prevalent, it's important to understand educational stakeholders' preferences and values to ensure that these tools can be used for good and that potential harms can be mitigated.

{{</citation>}}


### (100/119) OpinionConv: Conversational Product Search with Grounded Opinions (Vahid Sadiri Javadi et al., 2023)

{{<citation>}}

Vahid Sadiri Javadi, Martin Potthast, Lucie Flek. (2023)  
**OpinionConv: Conversational Product Search with Grounded Opinions**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs-IR, cs-LG, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04226v1)  

---


**ABSTRACT**  
When searching for products, the opinions of others play an important role in making informed decisions. Subjective experiences about a product can be a valuable source of information. This is also true in sales conversations, where a customer and a sales assistant exchange facts and opinions about products. However, training an AI for such conversations is complicated by the fact that language models do not possess authentic opinions for their lack of real-world experience. We address this problem by leveraging product reviews as a rich source of product opinions to ground conversational AI in true subjective narratives. With OpinionConv, we develop the first conversational AI for simulating sales conversations. To validate the generated conversations, we conduct several user studies showing that the generated opinions are perceived as realistic. Our assessors also confirm the importance of opinions as an informative basis for decision-making.

{{</citation>}}


### (101/119) DataTales: Investigating the use of Large Language Models for Authoring Data-Driven Articles (Nicole Sultanum et al., 2023)

{{<citation>}}

Nicole Sultanum, Arjun Srinivasan. (2023)  
**DataTales: Investigating the use of Large Language Models for Authoring Data-Driven Articles**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.04076v1)  

---


**ABSTRACT**  
Authoring data-driven articles is a complex process requiring authors to not only analyze data for insights but also craft a cohesive narrative that effectively communicates the insights. Text generation capabilities of contemporary large language models (LLMs) present an opportunity to assist the authoring of data-driven articles and expedite the writing process. In this work, we investigate the feasibility and perceived value of leveraging LLMs to support authors of data-driven articles. We designed a prototype system, DataTales, that leverages a LLM to generate textual narratives accompanying a given chart. Using DataTales as a design probe, we conducted a qualitative study with 11 professionals to evaluate the concept, from which we distilled affordances and opportunities to further integrate LLMs as valuable data-driven article authoring assistants.

{{</citation>}}


### (102/119) Portrayal: Leveraging NLP and Visualization for Analyzing Fictional Characters (Md Naimul Hoque et al., 2023)

{{<citation>}}

Md Naimul Hoque, Bhavya Ghai, Kari Kraus, Niklas Elmqvist. (2023)  
**Portrayal: Leveraging NLP and Visualization for Analyzing Fictional Characters**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.04056v1)  

---


**ABSTRACT**  
Many creative writing tasks (e.g., fiction writing) require authors to write complex narrative components (e.g., characterization, events, dialogue) over the course of a long story. Similarly, literary scholars need to manually annotate and interpret texts to understand such abstract components. In this paper, we explore how Natural Language Processing (NLP) and interactive visualization can help writers and scholars in such scenarios. To this end, we present Portrayal, an interactive visualization system for analyzing characters in a story. Portrayal extracts natural language indicators from a text to capture the characterization process and then visualizes the indicators in an interactive interface. We evaluated the system with 12 creative writers and scholars in a one-week-long qualitative study. Our findings suggest Portrayal helped writers revise their drafts and create dynamic characters and scenes. It helped scholars analyze characters without the need for any manual annotation, and design literary arguments with concrete evidence.

{{</citation>}}


## cs.IT (2)



### (103/119) Preserving Sparsity and Privacy in Straggler-Resilient Distributed Matrix Computations (Anindya Bijoy Das et al., 2023)

{{<citation>}}

Anindya Bijoy Das, Aditya Ramamoorthy, David J. Love, Christopher G. Brinton. (2023)  
**Preserving Sparsity and Privacy in Straggler-Resilient Distributed Matrix Computations**  

---
Primary Category: cs.IT  
Categories: cs-CR, cs-IT, cs.IT, math-IT  
Keywords: AWS, Amazon  
[Paper Link](http://arxiv.org/abs/2308.04331v1)  

---


**ABSTRACT**  
Existing approaches to distributed matrix computations involve allocating coded combinations of submatrices to worker nodes, to build resilience to stragglers and/or enhance privacy. In this study, we consider the challenge of preserving input sparsity in such approaches to retain the associated computational efficiency enhancements. First, we find a lower bound on the weight of coding, i.e., the number of submatrices to be combined to obtain coded submatrices to provide the resilience to the maximum possible number of stragglers (for given number of nodes and their storage constraints). Next we propose a distributed matrix computation scheme which meets this exact lower bound on the weight of the coding. Further, we develop controllable trade-off between worker computation time and the privacy constraint for sparse input matrices in settings where the worker nodes are honest but curious. Numerical experiments conducted in Amazon Web Services (AWS) validate our assertions regarding straggler mitigation and computation speed for sparse matrices.

{{</citation>}}


### (104/119) Iterative Sketching for Secure Coded Regression (Neophytos Charalambides et al., 2023)

{{<citation>}}

Neophytos Charalambides, Hessam Mahdavifar, Mert Pilanci, Alfred O. Hero III. (2023)  
**Iterative Sketching for Secure Coded Regression**  

---
Primary Category: cs.IT  
Categories: 65B99, 68P20, 68P25, 68P27, 68P30, 94-10, 94A11, 94A16, 94B60, E-3; E-4; F-2-1; G-1-3, cs-CR, cs-DC, cs-IT, cs-LG, cs-NA, cs.IT, math-IT, math-NA  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2308.04185v1)  

---


**ABSTRACT**  
In this work, we propose methods for speeding up linear regression distributively, while ensuring security. We leverage randomized sketching techniques, and improve straggler resilience in asynchronous systems. Specifically, we apply a random orthonormal matrix and then subsample \textit{blocks}, to simultaneously secure the information and reduce the dimension of the regression problem. In our setup, the transformation corresponds to an encoded encryption in an \textit{approximate gradient coding scheme}, and the subsampling corresponds to the responses of the non-straggling workers; in a centralized coded computing network. This results in a distributive \textit{iterative sketching} approach for an $\ell_2$-subspace embedding, \textit{i.e.} a new sketch is considered at each iteration. We also focus on the special case of the \textit{Subsampled Randomized Hadamard Transform}, which we generalize to block sampling; and discuss how it can be modified in order to secure the data.

{{</citation>}}


## cs.RO (2)



### (105/119) Embracing Safe Contacts with Contact-aware Planning and Control (Zhaoting Li et al., 2023)

{{<citation>}}

Zhaoting Li, Miguel Zamora, Hehui Zheng, Stelian Coros. (2023)  
**Embracing Safe Contacts with Contact-aware Planning and Control**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2308.04323v1)  

---


**ABSTRACT**  
Unlike human beings that can employ the entire surface of their limbs as a means to establish contact with their environment, robots are typically programmed to interact with their environments via their end-effectors, in a collision-free fashion, to avoid damaging their environment. In a departure from such a traditional approach, this work presents a contact-aware controller for reference tracking that maintains interaction forces on the surface of the robot below a safety threshold in the presence of both rigid and soft contacts. Furthermore, we leveraged the proposed controller to extend the BiTRRT sample-based planning method to be contact-aware, using a simplified contact model. The effectiveness of our framework is demonstrated in hardware experiments using a Franka robot in a setup inspired by the Amazon stowing task. A demo video of our results can be seen here: https://youtu.be/2WeYytauhNg

{{</citation>}}


### (106/119) ChatSim: Underwater Simulation with Natural Language Prompting (Aadi Palnitkar et al., 2023)

{{<citation>}}

Aadi Palnitkar, Rashmi Kapu, Xiaomin Lin, Cheng Liu, Nare Karapetyan, Yiannis Aloimonos. (2023)  
**ChatSim: Underwater Simulation with Natural Language Prompting**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.04029v2)  

---


**ABSTRACT**  
Robots are becoming an essential part of many operations including marine exploration or environmental monitoring. However, the underwater environment presents many challenges, including high pressure, limited visibility, and harsh conditions that can damage equipment. Real-world experimentation can be expensive and difficult to execute. Therefore, it is essential to simulate the performance of underwater robots in comparable environments to ensure their optimal functionality within practical real-world contexts.OysterSim generates photo-realistic images and segmentation masks of objects in marine environments, providing valuable training data for underwater computer vision applications. By integrating ChatGPT into underwater simulations, users can convey their thoughts effortlessly and intuitively create desired underwater environments without intricate coding. \invis{Moreover, researchers can realize substantial time and cost savings by evaluating their algorithms across diverse underwater conditions in the simulation.} The objective of ChatSim is to integrate Large Language Models (LLM) with a simulation environment~(OysterSim), enabling direct control of the simulated environment via natural language input. This advancement can greatly enhance the capabilities of underwater simulation, with far-reaching benefits for marine exploration and broader scientific research endeavors.

{{</citation>}}


## cs.SE (1)



### (107/119) A Comparative Study of Code Generation using ChatGPT 3.5 across 10 Programming Languages (Alessio Buscemi, 2023)

{{<citation>}}

Alessio Buscemi. (2023)  
**A Comparative Study of Code Generation using ChatGPT 3.5 across 10 Programming Languages**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.04477v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are advanced Artificial Intelligence (AI) systems that have undergone extensive training using large datasets in order to understand and produce language that closely resembles that of humans. These models have reached a level of proficiency where they are capable of successfully completing university exams across several disciplines and generating functional code to handle novel problems. This research investigates the coding proficiency of ChatGPT 3.5, a LLM released by OpenAI in November 2022, which has gained significant recognition for its impressive text generating and code creation capabilities. The skill of the model in creating code snippets is evaluated across 10 various programming languages and 4 different software domains. Based on the findings derived from this research, major unexpected behaviors and limitations of the model have been identified. This study aims to identify potential areas for development and examine the ramifications of automated code generation on the evolution of programming languages and on the tech industry.

{{</citation>}}


## cs.SI (1)



### (108/119) MCDAN: a Multi-scale Context-enhanced Dynamic Attention Network for Diffusion Prediction (Xiaowen Wang et al., 2023)

{{<citation>}}

Xiaowen Wang, Lanjun Wang, Yuting Su, Yongdong Zhang, An-An Liu. (2023)  
**MCDAN: a Multi-scale Context-enhanced Dynamic Attention Network for Diffusion Prediction**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.04266v1)  

---


**ABSTRACT**  
Information diffusion prediction aims at predicting the target users in the information diffusion path on social networks. Prior works mainly focus on the observed structure or sequence of cascades, trying to predict to whom this cascade will be infected passively. In this study, we argue that user intent understanding is also a key part of information diffusion prediction. We thereby propose a novel Multi-scale Context-enhanced Dynamic Attention Network (MCDAN) to predict which user will most likely join the observed current cascades. Specifically, to consider the global interactive relationship among users, we take full advantage of user friendships and global cascading relationships, which are extracted from the social network and historical cascades, respectively. To refine the model's ability to understand the user's preference for the current cascade, we propose a multi-scale sequential hypergraph attention module to capture the dynamic preference of users at different time scales. Moreover, we design a contextual attention enhancement module to strengthen the interaction of user representations within the current cascade. Finally, to engage the user's own susceptibility, we construct a susceptibility label for each user based on user susceptibility analysis and use the rank of this label for auxiliary prediction. We conduct experiments over four widely used datasets and show that MCDAN significantly overperforms the state-of-the-art models. The average improvements are up to 10.61% in terms of Hits@100 and 9.71% in terms of MAP@100, respectively.

{{</citation>}}


## cs.SD (2)



### (109/119) Auditory Attention Decoding with Task-Related Multi-View Contrastive Learning (Xiaoyu Chen et al., 2023)

{{<citation>}}

Xiaoyu Chen, Changde Du, Qiongyi Zhou, Huiguang He. (2023)  
**Auditory Attention Decoding with Task-Related Multi-View Contrastive Learning**  

---
Primary Category: cs.SD  
Categories: cs-HC, cs-SD, cs.SD, eess-AS, q-bio-NC, q-bio-QM  
Keywords: Attention, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.04244v1)  

---


**ABSTRACT**  
The human brain can easily focus on one speaker and suppress others in scenarios such as a cocktail party. Recently, researchers found that auditory attention can be decoded from the electroencephalogram (EEG) data. However, most existing deep learning methods are difficult to use prior knowledge of different views (that is attended speech and EEG are task-related views) and extract an unsatisfactory representation. Inspired by Broadbent's filter model, we decode auditory attention in a multi-view paradigm and extract the most relevant and important information utilizing the missing view. Specifically, we propose an auditory attention decoding (AAD) method based on multi-view VAE with task-related multi-view contrastive (TMC) learning. Employing TMC learning in multi-view VAE can utilize the missing view to accumulate prior knowledge of different views into the fusion of representation, and extract the approximate task-related representation. We examine our method on two popular AAD datasets, and demonstrate the superiority of our method by comparing it to the state-of-the-art method.

{{</citation>}}


### (110/119) MSAC: Multiple Speech Attribute Control Method for Speech Emotion Recognition (Yu Pan, 2023)

{{<citation>}}

Yu Pan. (2023)  
**MSAC: Multiple Speech Attribute Control Method for Speech Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.04025v1)  

---


**ABSTRACT**  
Despite significant progress, speech emotion recognition (SER) remains challenging due to inherent complexity and ambiguity of the emotion attribute, particularly in wild world. Whereas current studies primarily focus on recognition and generalization capabilities, this work pioneers an exploration into the reliability of SER methods and investigates how to model the speech emotion from the aspect of data distribution across various speech attributes. Specifically, we first build a novel CNN-based SER model which adopts additive margin softmax loss to expand the distance between features of different classes, thereby enhancing their discrimination. Second, a novel multiple speech attribute control method MSAC is proposed to explicitly control speech attributes, enabling the model to be less affected by emotion-agnostic attributes and capture more fine-grained emotion-related features. Third, we make a first attempt to test and analyze the reliability of the proposed SER workflow using the out-of-distribution detection method. Extensive experiments on both single and cross-corpus SER scenarios show that our proposed unified SER workflow consistently outperforms the baseline in terms of recognition, generalization, and reliability performance. Besides, in single-corpus SER, the proposed SER workflow achieves superior recognition results with a WAR of 72.97\% and a UAR of 71.76\% on the IEMOCAP corpus.

{{</citation>}}


## cs.MM (1)



### (111/119) Collaborative Edge Caching: a Meta Reinforcement Learning Approach with Edge Sampling (Bowei He et al., 2023)

{{<citation>}}

Bowei He, Yinan Mao, Shiji Zhou, Chen Ma, Zhi Wang. (2023)  
**Collaborative Edge Caching: a Meta Reinforcement Learning Approach with Edge Sampling**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.04205v1)  

---


**ABSTRACT**  
Current learning-based edge caching schemes usually suffer from dynamic content popularity, e.g., in the emerging short video platforms, users' request patterns shift significantly over time and across different edges. An intuitive solution for a specific local edge cache is to collect more request histories from other edge caches. However, uniformly merging these request histories may not perform satisfactorily due to heterogeneous content distributions on different edges. To solve this problem, we propose a collaborative edge caching framework. First, we design a meta-learning-based collaborative strategy to guarantee that the local model can timely meet the continually changing content popularity. Then, we design an edge sampling method to select more "valuable" neighbor edges to participate in the local training. To evaluate the proposed framework, we conduct trace-driven experiments to demonstrate the effectiveness of our design: it improves the average cache hit rate by up to $10.12\%$ (normalized) compared with other baselines.

{{</citation>}}


## cs.MA (2)



### (112/119) Communication-Efficient Cooperative Multi-Agent PPO via Regulated Segment Mixture in Internet of Vehicles (Xiaoxue Yu et al., 2023)

{{<citation>}}

Xiaoxue Yu, Rongpeng Li, Fei Wang, Chenghui Peng, Chengchao Liang, Zhifeng Zhao, Honggang Zhang. (2023)  
**Communication-Efficient Cooperative Multi-Agent PPO via Regulated Segment Mixture in Internet of Vehicles**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.04198v1)  

---


**ABSTRACT**  
Multi-Agent Reinforcement Learning (MARL) has become a classic paradigm to solve diverse, intelligent control tasks like autonomous driving in Internet of Vehicles (IoV). However, the widely assumed existence of a central node to implement centralized federated learning-assisted MARL might be impractical in highly dynamic scenarios, and the excessive communication overheads possibly overwhelm the IoV system. Therefore, in this paper, we design a communication efficient cooperative MARL algorithm, named RSM-MAPPO, to reduce the communication overheads in a fully distributed architecture. In particular, RSM-MAPPO enhances the multi-agent Proximal Policy Optimization (PPO) by incorporating the idea of segment mixture and augmenting multiple model replicas from received neighboring policy segments. Afterwards, RSM-MAPPO adopts a theory-guided metric to regulate the selection of contributive replicas to guarantee the policy improvement. Finally, extensive simulations in a mixed-autonomy traffic control scenario verify the effectiveness of the RSM-MAPPO algorithm.

{{</citation>}}


### (113/119) Cooperative Multi-Type Multi-Agent Deep Reinforcement Learning for Resource Management in Space-Air-Ground Integrated Networks (Hengxi Zhang et al., 2023)

{{<citation>}}

Hengxi Zhang, Huaze Tang, Wenbo Ding, Xiao-Ping Zhang. (2023)  
**Cooperative Multi-Type Multi-Agent Deep Reinforcement Learning for Resource Management in Space-Air-Ground Integrated Networks**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-LG, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.03995v1)  

---


**ABSTRACT**  
The Space-Air-Ground Integrated Network (SAGIN), integrating heterogeneous devices including low earth orbit (LEO) satellites, unmanned aerial vehicles (UAVs), and ground users (GUs), holds significant promise for advancing smart city applications. However, resource management of the SAGIN is a challenge requiring urgent study in that inappropriate resource management will cause poor data transmission, and hence affect the services in smart cities. In this paper, we develop a comprehensive SAGIN system that encompasses five distinct communication links and propose an efficient cooperative multi-type multi-agent deep reinforcement learning (CMT-MARL) method to address the resource management issue. The experimental results highlight the efficacy of the proposed CMT-MARL, as evidenced by key performance indicators such as the overall transmission rate and transmission success rate. These results underscore the potential value and feasibility of future implementation of the SAGIN.

{{</citation>}}


## cs.NE (1)



### (114/119) D-Score: A Synapse-Inspired Approach for Filter Pruning (Doyoung Park et al., 2023)

{{<citation>}}

Doyoung Park, Jinsoo Kim, Jina Nam, Jooyoung Chang, Sang Min Park. (2023)  
**D-Score: A Synapse-Inspired Approach for Filter Pruning**  

---
Primary Category: cs.NE  
Categories: cs-LG, cs-NE, cs.NE  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2308.04470v1)  

---


**ABSTRACT**  
This paper introduces a new aspect for determining the rank of the unimportant filters for filter pruning on convolutional neural networks (CNNs). In the human synaptic system, there are two important channels known as excitatory and inhibitory neurotransmitters that transmit a signal from a neuron to a cell. Adopting the neuroscientific perspective, we propose a synapse-inspired filter pruning method, namely Dynamic Score (D-Score). D-Score analyzes the independent importance of positive and negative weights in the filters and ranks the independent importance by assigning scores. Filters having low overall scores, and thus low impact on the accuracy of neural networks are pruned. The experimental results on CIFAR-10 and ImageNet datasets demonstrate the effectiveness of our proposed method by reducing notable amounts of FLOPs and Params without significant Acc. Drop.

{{</citation>}}


## cond-mat.soft (1)



### (115/119) Constructing Custom Thermodynamics Using Deep Learning (Xiaoli Chen et al., 2023)

{{<citation>}}

Xiaoli Chen, Beatrice W. Soh, Zi-En Ooi, Eleonore Vissol-Gaudin, Haijun Yu, Kostya S. Novoselov, Kedar Hippalgaonkar, Qianxiao Li. (2023)  
**Constructing Custom Thermodynamics Using Deep Learning**  

---
Primary Category: cond-mat.soft  
Categories: cond-mat-soft, cond-mat-stat-mech, cond-mat.soft, cs-LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04119v1)  

---


**ABSTRACT**  
One of the most exciting applications of AI is automated scientific discovery based on previously amassed data, coupled with restrictions provided by the known physical principles, including symmetries and conservation laws. Such automated hypothesis creation and verification can assist scientists in studying complex phenomena, where traditional physical intuition may fail. Of particular importance are complex dynamic systems where their time evolution is strongly influenced by varying external parameters. In this paper we develop a platform based on a generalised Onsager principle to learn macroscopic dynamical descriptions of arbitrary stochastic dissipative systems directly from observations of their microscopic trajectories. We focus on systems whose complexity and sheer sizes render complete microscopic description impractical, and constructing theoretical macroscopic models requires extensive domain knowledge or trial-and-error. Our machine learning approach addresses this by simultaneously constructing reduced thermodynamic coordinates and interpreting the dynamics on these coordinates. We demonstrate our method by studying theoretically and validating experimentally, the stretching of long polymer chains in an externally applied field. Specifically, we learn three interpretable thermodynamic coordinates and build a dynamical landscape of polymer stretching, including (1) the identification of stable and transition states and (2) the control of the stretching rate. We further demonstrate the universality of our approach by applying it to an unrelated problem in a different domain: constructing macroscopic dynamics for spatial epidemics, showing that our method addresses wide scientific and technological applications.

{{</citation>}}


## q-bio.QM (1)



### (116/119) PTransIPs: Identification of phosphorylation sites based on protein pretrained language model and Transformer (Ziyang Xu et al., 2023)

{{<citation>}}

Ziyang Xu, Haitian Zhong. (2023)  
**PTransIPs: Identification of phosphorylation sites based on protein pretrained language model and Transformer**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, q-bio-QM, q-bio.QM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.05115v1)  

---


**ABSTRACT**  
Phosphorylation is central to numerous fundamental cellular processes, influencing the onset and progression of a variety of diseases. Identification of phosphorylation sites is thus an important step for understanding the molecular mechanisms of cells and virus infection, which potentially leads to new therapeutic targets. In this study, we present PTransIPs, a novel deep learning model for the identification of phosphorylation sites. PTransIPs treats amino acids in protein sequences as words in natural language, extracting unique encodings based on the types along with position of amino acids in the sequence. It also incorporates embeddings from large pre-trained protein models as additional data inputs. PTransIPS is further trained on a combination model of convolutional neural network with residual connections and Transformer model equipped with multi-head attention mechanisms. At last, the model outputs classification results through a fully connected layer. The results of independent testing reveal that PTransIPs outperforms existing state-of-the-art methodologies, achieving AUROCs of 0.9232 and 0.9660 for identifying phosphorylated S/T and Y sites respectively. In addition, ablation studies prove that pretrained model embeddings contribute to the performance of PTransIPs. Furthermore, PTransIPs has interpretable amino acid preference, visible training process and shows generalizability on other bioactivity classification tasks. To facilitate usage, our code and data are publicly accessible at \url{https://github.com/StatXzy7/PTransIPs}.

{{</citation>}}


## math.OC (1)



### (117/119) Online identification and control of PDEs via Reinforcement Learning methods (Alessandro Alla et al., 2023)

{{<citation>}}

Alessandro Alla, Agnese Pacifico, Michele Palladino, Andrea Pesare. (2023)  
**Online identification and control of PDEs via Reinforcement Learning methods**  

---
Primary Category: math.OC  
Categories: cs-NA, math-NA, math-OC, math.OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.04068v1)  

---


**ABSTRACT**  
We focus on the control of unknown Partial Differential Equations (PDEs). The system dynamics is unknown, but we assume we are able to observe its evolution for a given control input, as typical in a Reinforcement Learning framework. We propose an algorithm based on the idea to control and identify on the fly the unknown system configuration. In this work, the control is based on the State-Dependent Riccati approach, whereas the identification of the model on Bayesian linear regression. At each iteration, based on the observed data, we obtain an estimate of the a-priori unknown parameter configuration of the PDE and then we compute the control of the correspondent model. We show by numerical evidence the convergence of the method for infinite horizon control problems.

{{</citation>}}


## stat.ML (1)



### (118/119) Generative Models for Anomaly Detection and Design-Space Dimensionality Reduction in Shape Optimization (Danny D'Agostino, 2023)

{{<citation>}}

Danny D'Agostino. (2023)  
**Generative Models for Anomaly Detection and Design-Space Dimensionality Reduction in Shape Optimization**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-OC, physics-data-an, physics-flu-dyn, stat-ML, stat.ML  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.04051v1)  

---


**ABSTRACT**  
Our work presents a novel approach to shape optimization, that has the twofold objective to improve the efficiency of global optimization algorithms while promoting the generation of high-quality designs during the optimization process free of geometrical anomalies. This is accomplished by reducing the number of the original design variables defining a new reduced subspace where the geometrical variance is maximized and modeling the underlying generative process of the data via probabilistic linear latent variable models such as Factor Analysis and Probabilistic Principal Component Analysis. We show that the data follows approximately a Gaussian distribution when the shape modification method is linear and the design variables are sampled uniformly at random, due to the direct application of the central limit theorem. The model uncertainty is measured in terms of Mahalanobis distance, and the paper demonstrates that anomalous designs tend to exhibit a high value of this metric. This enables the definition of a new optimization model where anomalous geometries are penalized and consequently avoided during the optimization loop. The procedure is demonstrated for hull shape optimization of the DTMB 5415 model, extensively used as an international benchmark for shape optimization problems. The global optimization routine is carried out using Bayesian Optimization and the DIRECT algorithm. From the numerical results, the new framework improves the convergence of global optimization algorithms, while only designs with high-quality geometrical features are generated through the optimization routine thereby avoiding the wastage of precious computationally expensive simulations.

{{</citation>}}


## cs.DB (1)



### (119/119) A Benchmarking Study of Matching Algorithms for Knowledge Graph Entity Alignment (Nhat-Minh Dao et al., 2023)

{{<citation>}}

Nhat-Minh Dao, Thai V. Hoang, Zonghua Zhang. (2023)  
**A Benchmarking Study of Matching Algorithms for Knowledge Graph Entity Alignment**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: Entity Alignment, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.03961v1)  

---


**ABSTRACT**  
How to identify those equivalent entities between knowledge graphs (KGs), which is called Entity Alignment (EA), is a long-standing challenge. So far, many methods have been proposed, with recent focus on leveraging Deep Learning to solve this problem. However, we observe that most of the efforts has been paid to having better representation of entities, rather than improving entity matching from the learned representations. In fact, how to efficiently infer the entity pairs from this similarity matrix, which is essentially a matching problem, has been largely ignored by the community. Motivated by this observation, we conduct an in-depth analysis on existing algorithms that are particularly designed for solving this matching problem, and propose a novel matching method, named Bidirectional Matching (BMat). Our extensive experimental results on public datasets indicate that there is currently no single silver bullet solution for EA. In other words, different classes of entity similarity estimation may require different matching algorithms to reach the best EA results for each class. We finally conclude that using PARIS, the state-of-the-art EA approach, with BMat gives the best combination in terms of EA performance and the algorithm's time and space complexity.

{{</citation>}}
