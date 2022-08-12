---
draft: false
title: "S-Net: From Answer Extraction to Answer Generation for Machine Reading Comprehension"
date: 2022-06-02
author: "akitenkrad"
description: ""
tags: ["At:Round-2", "Published:2018", "DS:MS-MARCO", "Question Answering", "Generative MRC"]
menu:
  sidebar:
    name: 2022.06.02
    identifier: 20220602
    parent: 202206
    weight: 10
math: true
---

- [x] Round-1: Overview
- [x] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation

{{< citation >}}
Chuanqi Tan, Furu Wei, Nan Yang, Bowen Du,Weifeng Lv, and Ming Zhou. 2018.  
S-Net: Fromanswer extraction to answer synthesis for machinereading comprehension.  
InAssociation for the Ad-vancement of  Artificial  Intelligence (AAAI), pages5940–5947.
{{< /citation>}}

## Abstract

> In this paper, we present a novel approach to machine reading comprehension forthe MS-MARCO dataset. Unlike the SQuAD dataset that aims to answer a ques-tion with exact text spans in a passage, the MS-MARCO dataset defines the taskas answering a question from multiple passages and the words in the answer arenot necessary in the passages.  We therefore develop an extraction-then-synthesisframework to synthesize answers from extraction results. Specifically, the answerextraction model is first employed to predict the most important sub-spans fromthe passage as evidence, and the answer synthesis model takes the evidence as ad-ditional features along with the question and passage to further elaborate the finalanswers.  We build the answer extraction model with state-of-the-art neural net-works for single passage reading comprehension, and propose an additional taskof passage ranking to help answer extraction in multiple passages.  The answersynthesis model is based on the sequence-to-sequence neural networks with ex-tracted evidences as features. Experiments show that our extraction-then-synthesismethod outperforms state-of-the-art methods.

## Background & Wat's New

- 機械読解タスクにおいて，コンテキストから回答と関連の深いスパンを抽出した後，回答を生成するExtraction-then-Synthesisフレームワークを提案した
  - これにより，回答に含まれる単語は必ずしも質問やコンテキストに含まれていなくとも生成可能になる
- 回答を生成するにあたって，複数のコンテキストをランク付けした情報を利用した
- 回答を生成するデコーダ部分にはSequence-to-Sequenceモデルを採用し，これによりMS-MARCOのQAタスクにおいて単純な抽出型のモデルよりも高い精度を達成した

## Dataset

{{< ci-details summary="MSMARCO (Bajaj et al., 2018)" >}}
Daniel Fernando Campos, Tri Nguyen, M. Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, L. Deng, Bhaskar Mitra. (2016)  
**MS MARCO: A Human Generated MAchine Reading COmprehension Dataset**  
CoCo@NIPS  
[Paper Link](https://www.semanticscholar.org/paper/a69cf45d44a9d806d2487a1ffb9eca71ee73c2ee)  
Influential Citation Count (190), SS-ID (a69cf45d44a9d806d2487a1ffb9eca71ee73c2ee)  

**ABSTRACT**  
This paper presents our recent work on the design and development of a new, large scale dataset, which we name MS MARCO, for MAchine Reading COmprehension. This new dataset is aimed to overcome a number of well-known weaknesses of previous publicly available datasets for the same task of reading comprehension and question answering. In MS MARCO, all questions are sampled from real anonymized user queries. The context passages, from which answers in the dataset are derived, are extracted from real web documents using the most advanced version of the Bing search engine. The answers to the queries are human generated. Finally, a subset of these queries has multiple answers. We aim to release one million queries and the corresponding answers in the dataset, which, to the best of our knowledge, is the most comprehensive real-world dataset of its kind in both quantity and quality. We are currently releasing 100,000 queries with their corresponding answers to inspire work in reading comprehension and question answering along with gathering feedback from the research community.
{{< /ci-details >}}

## Model Description

### Evidence Extraction Layer

#### Problem Formulation

$$
\begin{align*}
  Q &= \left\lbrace w\_t^Q \right\rbrace \_{t=1}^m \\\\
  P\_k &= \left\lbrace w\_t^{P\_k} \right\rbrace \_{t=1}^n \\\\
  & \text{where} \\\\
  & m \mapsto \text{length of Question} \\\\
  & n \mapsto \text{length of a Passage} \\\\
  & k=\lbrace 1, \ldots, K \rbrace
\end{align*}
$$

#### Embedding

$$
\begin{align*}
  e\_{\text{word}}^Q &= \text{Embedding}(Q) &\in \mathbb{R}^{d \times m} \\\\
  e\_{\text{word}}^{P\_k} &= \text{Embedding}(P\_k) &\in \mathbb{R}^{d \times n} \\\\
  e\_{\text{char}}^Q &= h\_L^Q &\in \mathbb{R} \\\\
  e\_{\text{char}}^{P\_k} &= h\_L^{P\_k}
\end{align*}
$$

### Training Settings

## Results

## References


{{< ci-details summary="SQuAD: 100,000+ Questions for Machine Comprehension of Text (Pranav Rajpurkar et al., 2016)">}}

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang. (2016)  
**SQuAD: 100,000+ Questions for Machine Comprehension of Text**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/05dd7254b632376973f3a1b4d39485da17814df5)  
Influential Citation Count (1139), SS-ID (05dd7254b632376973f3a1b4d39485da17814df5)  

**ABSTRACT**  
We present the Stanford Question Answering Dataset (SQuAD), a new reading comprehension dataset consisting of 100,000+ questions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding reading passage. We analyze the dataset to understand the types of reasoning required to answer the questions, leaning heavily on dependency and constituency trees. We build a strong logistic regression model, which achieves an F1 score of 51.0%, a significant improvement over a simple baseline (20%). However, human performance (86.8%) is much higher, indicating that the dataset presents a good challenge problem for future research.  The dataset is freely available at this https URL

{{< /ci-details >}}

{{< ci-details summary="End-to-End Answer Chunk Extraction and Ranking for Reading Comprehension (Yang Yu et al., 2016)">}}

Yang Yu, Wei Zhang, K. Hasan, Mo Yu, Bing Xiang, Bowen Zhou. (2016)  
**End-to-End Answer Chunk Extraction and Ranking for Reading Comprehension**  
  
[Paper Link](https://www.semanticscholar.org/paper/0680f04750b1e257ffdd161e85382031dc73ea7f)  
Influential Citation Count (10), SS-ID (0680f04750b1e257ffdd161e85382031dc73ea7f)  

**ABSTRACT**  
This paper proposes dynamic chunk reader (DCR), an end-to-end neural reading comprehension (RC) model that is able to extract and rank a set of answer candidates from a given document to answer questions. DCR is able to predict answers of variable lengths, whereas previous neural RC models primarily focused on predicting single tokens or entities. DCR encodes a document and an input question with recurrent neural networks, and then applies a word-by-word attention mechanism to acquire question-aware representations for the document, followed by the generation of chunk representations and a ranking module to propose the top-ranked chunk as the answer. Experimental results show that DCR could achieve a 66.3% Exact match and 74.7% F1 score on the Stanford Question Answering Dataset.

{{< /ci-details >}}

{{< ci-details summary="Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation (Kyunghyun Cho et al., 2014)">}}

Kyunghyun Cho, Bart van Merrienboer, Çaglar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio. (2014)  
**Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/0b544dfe355a5070b60986319a3f51fb45d1348e)  
Influential Citation Count (2731), SS-ID (0b544dfe355a5070b60986319a3f51fb45d1348e)  

**ABSTRACT**  
In this paper, we propose a novel neural network model called RNN Encoder‐ Decoder that consists of two recurrent neural networks (RNN). One RNN encodes a sequence of symbols into a fixedlength vector representation, and the other decodes the representation into another sequence of symbols. The encoder and decoder of the proposed model are jointly trained to maximize the conditional probability of a target sequence given a source sequence. The performance of a statistical machine translation system is empirically found to improve by using the conditional probabilities of phrase pairs computed by the RNN Encoder‐Decoder as an additional feature in the existing log-linear model. Qualitatively, we show that the proposed model learns a semantically and syntactically meaningful representation of linguistic phrases.

{{< /ci-details >}}

{{< ci-details summary="Reasoning about Entailment with Neural Attention (Tim Rocktäschel et al., 2015)">}}

Tim Rocktäschel, Edward Grefenstette, K. Hermann, Tomás Kociský, P. Blunsom. (2015)  
**Reasoning about Entailment with Neural Attention**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/2846e83d405cbe3bf2f0f3b5f635dd8b3c680c45)  
Influential Citation Count (89), SS-ID (2846e83d405cbe3bf2f0f3b5f635dd8b3c680c45)  

**ABSTRACT**  
While most approaches to automatically recognizing entailment relations have used classifiers employing hand engineered features derived from complex natural language processing pipelines, in practice their performance has been only slightly better than bag-of-word pair classifiers using only lexical similarity. The only attempt so far to build an end-to-end differentiable neural network for entailment failed to outperform such a simple similarity classifier. In this paper, we propose a neural model that reads two sentences to determine entailment using long short-term memory units. We extend this model with a word-by-word neural attention mechanism that encourages reasoning over entailments of pairs of words and phrases. Furthermore, we present a qualitative analysis of attention weights produced by this model, demonstrating such reasoning capabilities. On a large entailment dataset this model outperforms the previous best neural model and a classifier with engineered features by a substantial margin. It is the first generic end-to-end differentiable system that achieves state-of-the-art accuracy on a textual entailment dataset.

{{< /ci-details >}}

{{< ci-details summary="Dropout: a simple way to prevent neural networks from overfitting (Nitish Srivastava et al., 2014)">}}

Nitish Srivastava, Geoffrey E. Hinton, A. Krizhevsky, Ilya Sutskever, R. Salakhutdinov. (2014)  
**Dropout: a simple way to prevent neural networks from overfitting**  
J. Mach. Learn. Res.  
[Paper Link](https://www.semanticscholar.org/paper/34f25a8704614163c4095b3ee2fc969b60de4698)  
Influential Citation Count (2315), SS-ID (34f25a8704614163c4095b3ee2fc969b60de4698)  

**ABSTRACT**  
Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem. The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much. During training, dropout samples from an exponential number of different "thinned" networks. At test time, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights. This significantly reduces overfitting and gives major improvements over other regularization methods. We show that dropout improves the performance of neural networks on supervised learning tasks in vision, speech recognition, document classification and computational biology, obtaining state-of-the-art results on many benchmark data sets.

{{< /ci-details >}}

{{< ci-details summary="The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations (Felix Hill et al., 2015)">}}

Felix Hill, Antoine Bordes, S. Chopra, J. Weston. (2015)  
**The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/35b91b365ceb016fb3e022577cec96fb9b445dc5)  
Influential Citation Count (107), SS-ID (35b91b365ceb016fb3e022577cec96fb9b445dc5)  

**ABSTRACT**  
We introduce a new test of how well language models capture meaning in children's books. Unlike standard language modelling benchmarks, it distinguishes the task of predicting syntactic function words from that of predicting lower-frequency words, which carry greater semantic content. We compare a range of state-of-the-art models, each with a different way of encoding what has been previously read. We show that models which store explicit representations of long-term contexts outperform state-of-the-art neural language models at predicting semantic content words, although this advantage is not observed for syntactic function words. Interestingly, we find that the amount of text encoded in a single memory representation is highly influential to the performance: there is a sweet-spot, not too big and not too small, between single words and full sentences that allows the most meaningful information in a text to be effectively retained and recalled. Further, the attention over such window-based memories can be trained effectively through self-supervision. We then assess the generality of this principle by applying it to the CNN QA benchmark, which involves identifying named entities in paraphrased summaries of news articles, and achieve state-of-the-art performance.

{{< /ci-details >}}

{{< ci-details summary="Bidirectional Attention Flow for Machine Comprehension (Minjoon Seo et al., 2016)">}}

Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi. (2016)  
**Bidirectional Attention Flow for Machine Comprehension**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/3a7b63b50c64f4ec3358477790e84cbd6be2a0b4)  
Influential Citation Count (447), SS-ID (3a7b63b50c64f4ec3358477790e84cbd6be2a0b4)  

**ABSTRACT**  
Machine comprehension (MC), answering a query about a given context paragraph, requires modeling complex interactions between the context and the query. Recently, attention mechanisms have been successfully extended to MC. Typically these methods use attention to focus on a small portion of the context and summarize it with a fixed-size vector, couple attentions temporally, and/or often form a uni-directional attention. In this paper we introduce the Bi-Directional Attention Flow (BIDAF) network, a multi-stage hierarchical process that represents the context at different levels of granularity and uses bi-directional attention flow mechanism to obtain a query-aware context representation without early summarization. Our experimental evaluations show that our model achieves the state-of-the-art results in Stanford Question Answering Dataset (SQuAD) and CNN/DailyMail cloze test.

{{< /ci-details >}}

{{< ci-details summary="Making Neural QA as Simple as Possible but not Simpler (Dirk Weissenborn et al., 2017)">}}

Dirk Weissenborn, Georg Wiese, Laura Seiffe. (2017)  
**Making Neural QA as Simple as Possible but not Simpler**  
CoNLL  
[Paper Link](https://www.semanticscholar.org/paper/46a7afc2b23bb3406fb64c36b6f2696145b54f24)  
Influential Citation Count (15), SS-ID (46a7afc2b23bb3406fb64c36b6f2696145b54f24)  

**ABSTRACT**  
Recent development of large-scale question answering (QA) datasets triggered a substantial amount of research into end-to-end neural architectures for QA. Increasingly complex systems have been conceived without comparison to simpler neural baseline systems that would justify their complexity. In this work, we propose a simple heuristic that guides the development of neural baseline systems for the extractive QA task. We find that there are two ingredients necessary for building a high-performing neural QA system: first, the awareness of question words while processing the context and second, a composition function that goes beyond simple bag-of-words modeling, such as recurrent neural networks. Our results show that FastQA, a system that meets these two requirements, can achieve very competitive performance compared with existing models. We argue that this surprising finding puts results of previous systems and the complexity of recent QA datasets into perspective.

{{< /ci-details >}}

{{< ci-details summary="Grammar as a Foreign Language (Oriol Vinyals et al., 2014)">}}

Oriol Vinyals, Lukasz Kaiser, Terry Koo, Slav Petrov, Ilya Sutskever, Geoffrey E. Hinton. (2014)  
**Grammar as a Foreign Language**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/47570e7f63e296f224a0e7f9a0d08b0de3cbaf40)  
Influential Citation Count (97), SS-ID (47570e7f63e296f224a0e7f9a0d08b0de3cbaf40)  

**ABSTRACT**  
Syntactic constituency parsing is a fundamental problem in natural language processing and has been the subject of intensive research and engineering for decades. As a result, the most accurate parsers are domain specific, complex, and inefficient. In this paper we show that the domain agnostic attention-enhanced sequence-to-sequence model achieves state-of-the-art results on the most widely used syntactic constituency parsing dataset, when trained on a large synthetic corpus that was annotated using existing parsers. It also matches the performance of standard parsers when trained only on a small human-annotated dataset, which shows that this model is highly data-efficient, in contrast to sequence-to-sequence models without the attention mechanism. Our parser is also fast, processing over a hundred sentences per second with an unoptimized CPU implementation.

{{< /ci-details >}}

{{< ci-details summary="MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text (Matthew Richardson et al., 2013)">}}

Matthew Richardson, C. Burges, Erin Renshaw. (2013)  
**MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/564257469fa44cdb57e4272f85253efb9acfd69d)  
Influential Citation Count (108), SS-ID (564257469fa44cdb57e4272f85253efb9acfd69d)  

**ABSTRACT**  
We present MCTest, a freely available set of stories and associated questions intended for research on the machine comprehension of text. Previous work on machine comprehension (e.g., semantic modeling) has made great strides, but primarily focuses either on limited-domain datasets, or on solving a more restricted goal (e.g., open-domain relation extraction). In contrast, MCTest requires machines to answer multiple-choice reading comprehension questions about fictional stories, directly tackling the high-level goal of open-domain machine comprehension. Reading comprehension can test advanced abilities such as causal reasoning and understanding the world, yet, by being multiple-choice, still provide a clear metric. By being fictional, the answer typically can be found only in the story itself. The stories and questions are also carefully limited to those a young child would understand, reducing the world knowledge that is required for the task. We present the scalable crowd-sourcing methods that allow us to cheaply construct a dataset of 500 stories and 2000 questions. By screening workers (with grammar tests) and stories (with grading), we have ensured that the data is the same quality as another set that we manually edited, but at one tenth the editing cost. By being open-domain, yet carefully restricted, we hope MCTest will serve to encourage research and provide a clear metric for advancement on the machine comprehension of text. 1 Reading Comprehension A major goal for NLP is for machines to be able to understand text as well as people. Several research disciplines are focused on this problem: for example, information extraction, relation extraction, semantic role labeling, and recognizing textual entailment. Yet these techniques are necessarily evaluated individually, rather than by how much they advance us towards the end goal. On the other hand, the goal of semantic parsing is the machine comprehension of text (MCT), yet its evaluation requires adherence to a specific knowledge representation, and it is currently unclear what the best representation is, for open-domain text. We believe that it is useful to directly tackle the top-level task of MCT. For this, we need a way to measure progress. One common method for evaluating someone’s understanding of text is by giving them a multiple-choice reading comprehension test. This has the advantage that it is objectively gradable (vs. essays) yet may test a range of abilities such as causal or counterfactual reasoning, inference among relations, or just basic understanding of the world in which the passage is set. Therefore, we propose a multiple-choice reading comprehension task as a way to evaluate progress on MCT. We have built a reading comprehension dataset containing 500 fictional stories, with 4 multiple choice questions per story. It was built using methods which can easily scale to at least 5000 stories, since the stories were created, and the curation was done, using crowd sourcing almost entirely, at a total of $4.00 per story. We plan to periodically update the dataset to ensure that methods are not overfitting to the existing data. The dataset is open-domain, yet restricted to concepts and words that a 7 year old is expected to understand. This task is still beyond the capability of today’s computers and algorithms.

{{< /ci-details >}}

{{< ci-details summary="Learning Natural Language Inference with LSTM (Shuohang Wang et al., 2015)">}}

Shuohang Wang, Jing Jiang. (2015)  
**Learning Natural Language Inference with LSTM**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/596c882de006e4bb4a93f1fa08a5dd467bee060a)  
Influential Citation Count (39), SS-ID (596c882de006e4bb4a93f1fa08a5dd467bee060a)  

**ABSTRACT**  
Natural language inference (NLI) is a fundamentally important task in natural language processing that has many applications. The recently released Stanford Natural Language Inference (SNLI) corpus has made it possible to develop and evaluate learning-centered methods such as deep neural networks for natural language inference (NLI). In this paper, we propose a special long short-term memory (LSTM) architecture for NLI. Our model builds on top of a recently proposed neural attention model for NLI but is based on a significantly different idea. Instead of deriving sentence embeddings for the premise and the hypothesis to be used for classification, our solution uses a match-LSTM to perform word-by-word matching of the hypothesis with the premise. This LSTM is able to place more emphasis on important word-level matching results. In particular, we observe that this LSTM remembers important mismatches that are critical for predicting the contradiction or the neutral relationship label. On the SNLI corpus, our model achieves an accuracy of 86.1%, outperforming the state of the art.

{{< /ci-details >}}

{{< ci-details summary="ROUGE: A Package for Automatic Evaluation of Summaries (Chin-Yew Lin, 2004)">}}

Chin-Yew Lin. (2004)  
**ROUGE: A Package for Automatic Evaluation of Summaries**  
ACL 2004  
[Paper Link](https://www.semanticscholar.org/paper/60b05f32c32519a809f21642ef1eb3eaf3848008)  
Influential Citation Count (1543), SS-ID (60b05f32c32519a809f21642ef1eb3eaf3848008)  

**ABSTRACT**  
ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation. It includes measures to automatically determine the quality of a summary by comparing it to other (ideal) summaries created by humans. The measures count the number of overlapping units such as n-gram, word sequences, and word pairs between the computer-generated summary to be evaluated and the ideal summaries created by humans. This paper introduces four different ROUGE measures: ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S included in the ROUGE summarization evaluation package and their evaluations. Three of them have been used in the Document Understanding Conference (DUC) 2004, a large-scale summarization evaluation sponsored by NIST.

{{< /ci-details >}}

{{< ci-details summary="ADADELTA: An Adaptive Learning Rate Method (Matthew D. Zeiler, 2012)">}}

Matthew D. Zeiler. (2012)  
**ADADELTA: An Adaptive Learning Rate Method**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/8729441d734782c3ed532a7d2d9611b438c0a09a)  
Influential Citation Count (809), SS-ID (8729441d734782c3ed532a7d2d9611b438c0a09a)  

**ABSTRACT**  
We present a novel per-dimension learning rate method for gradient descent called ADADELTA. The method dynamically adapts over time using only first order information and has minimal computational overhead beyond vanilla stochastic gradient descent. The method requires no manual tuning of a learning rate and appears robust to noisy gradient information, different model architecture choices, various data modalities and selection of hyperparameters. We show promising results compared to other methods on the MNIST digit classification task using a single machine and on a large scale voice dataset in a distributed cluster environment.

{{< /ci-details >}}

{{< ci-details summary="Effective Approaches to Attention-based Neural Machine Translation (Thang Luong et al., 2015)">}}

Thang Luong, Hieu Pham, Christopher D. Manning. (2015)  
**Effective Approaches to Attention-based Neural Machine Translation**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/93499a7c7f699b6630a86fad964536f9423bb6d0)  
Influential Citation Count (621), SS-ID (93499a7c7f699b6630a86fad964536f9423bb6d0)  

**ABSTRACT**  
An attentional mechanism has lately been used to improve neural machine translation (NMT) by selectively focusing on parts of the source sentence during translation. However, there has been little work exploring useful architectures for attention-based NMT. This paper examines two simple and effective classes of attentional mechanism: a global approach which always attends to all source words and a local one that only looks at a subset of source words at a time. We demonstrate the effectiveness of both approaches on the WMT translation tasks between English and German in both directions. With local attention, we achieve a significant gain of 5.0 BLEU points over non-attentional systems that already incorporate known techniques such as dropout. Our ensemble model using different attention architectures yields a new state-of-the-art result in the WMT’15 English to German translation task with 25.9 BLEU points, an improvement of 1.0 BLEU points over the existing best system backed by NMT and an n-gram reranker. 1

{{< /ci-details >}}

{{< ci-details summary="Learning Recurrent Span Representations for Extractive Question Answering (Kenton Lee et al., 2016)">}}

Kenton Lee, T. Kwiatkowski, Ankur P. Parikh, Dipanjan Das. (2016)  
**Learning Recurrent Span Representations for Extractive Question Answering**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/97e6ed1f7e5de0034f71c370c01f59c87aaf9a72)  
Influential Citation Count (20), SS-ID (97e6ed1f7e5de0034f71c370c01f59c87aaf9a72)  

**ABSTRACT**  
The reading comprehension task, that asks questions about a given evidence document, is a central problem in natural language understanding. Recent formulations of this task have typically focused on answer selection from a set of candidates pre-defined manually or through the use of an external NLP pipeline. However, Rajpurkar et al. (2016) recently released the SQUAD dataset in which the answers can be arbitrary strings from the supplied text. In this paper, we focus on this answer extraction task, presenting a novel model architecture that efficiently builds fixed length representations of all spans in the evidence document with a recurrent network. We show that scoring explicit span representations significantly improves performance over other approaches that factor the prediction into separate predictions about words or start and end markers. Our approach improves upon the best published results of Wang & Jiang (2016) by 5% and decreases the error of Rajpurkar et al.’s baseline by > 50%.

{{< /ci-details >}}

{{< ci-details summary="MS MARCO: A Human Generated MAchine Reading COmprehension Dataset (Daniel Fernando Campos et al., 2016)">}}

Daniel Fernando Campos, Tri Nguyen, M. Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, L. Deng, Bhaskar Mitra. (2016)  
**MS MARCO: A Human Generated MAchine Reading COmprehension Dataset**  
CoCo@NIPS  
[Paper Link](https://www.semanticscholar.org/paper/a69cf45d44a9d806d2487a1ffb9eca71ee73c2ee)  
Influential Citation Count (211), SS-ID (a69cf45d44a9d806d2487a1ffb9eca71ee73c2ee)  

**ABSTRACT**  
This paper presents our recent work on the design and development of a new, large scale dataset, which we name MS MARCO, for MAchine Reading COmprehension. This new dataset is aimed to overcome a number of well-known weaknesses of previous publicly available datasets for the same task of reading comprehension and question answering. In MS MARCO, all questions are sampled from real anonymized user queries. The context passages, from which answers in the dataset are derived, are extracted from real web documents using the most advanced version of the Bing search engine. The answers to the queries are human generated. Finally, a subset of these queries has multiple answers. We aim to release one million queries and the corresponding answers in the dataset, which, to the best of our knowledge, is the most comprehensive real-world dataset of its kind in both quantity and quality. We are currently releasing 100,000 queries with their corresponding answers to inspire work in reading comprehension and question answering along with gathering feedback from the research community.

{{< /ci-details >}}

{{< ci-details summary="End-to-End Reading Comprehension with Dynamic Answer Chunk Ranking (Yang Yu et al., 2016)">}}

Yang Yu, Wei Zhang, K. Hasan, Mo Yu, Bing Xiang, Bowen Zhou. (2016)  
**End-to-End Reading Comprehension with Dynamic Answer Chunk Ranking**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/a8c33413a626bafc67d46029ed28c2a28cc08899)  
Influential Citation Count (7), SS-ID (a8c33413a626bafc67d46029ed28c2a28cc08899)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Selective Encoding for Abstractive Sentence Summarization (Qingyu Zhou et al., 2017)">}}

Qingyu Zhou, Nan Yang, Furu Wei, M. Zhou. (2017)  
**Selective Encoding for Abstractive Sentence Summarization**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/b53907a11dc14d7d36e56212b6af71f8b22020af)  
Influential Citation Count (32), SS-ID (b53907a11dc14d7d36e56212b6af71f8b22020af)  

**ABSTRACT**  
We propose a selective encoding model to extend the sequence-to-sequence framework for abstractive sentence summarization. It consists of a sentence encoder, a selective gate network, and an attention equipped decoder. The sentence encoder and decoder are built with recurrent neural networks. The selective gate network constructs a second level sentence representation by controlling the information flow from encoder to decoder. The second level representation is tailored for sentence summarization task, which leads to better performance. We evaluate our model on the English Gigaword, DUC 2004 and MSR abstractive sentence summarization datasets. The experimental results show that the proposed selective encoding model outperforms the state-of-the-art baseline models.

{{< /ci-details >}}

{{< ci-details summary="Gated Self-Matching Networks for Reading Comprehension and Question Answering (Wenhui Wang et al., 2017)">}}

Wenhui Wang, Nan Yang, Furu Wei, Baobao Chang, M. Zhou. (2017)  
**Gated Self-Matching Networks for Reading Comprehension and Question Answering**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/b798cfd967e1a9ca5e7bc995d33a907bf65d1c7f)  
Influential Citation Count (102), SS-ID (b798cfd967e1a9ca5e7bc995d33a907bf65d1c7f)  

**ABSTRACT**  
In this paper, we present the gated self-matching networks for reading comprehension style question answering, which aims to answer questions from a given passage. We first match the question and passage with gated attention-based recurrent networks to obtain the question-aware passage representation. Then we propose a self-matching attention mechanism to refine the representation by matching the passage against itself, which effectively encodes information from the whole passage. We finally employ the pointer networks to locate the positions of answers from the passages. We conduct extensive experiments on the SQuAD dataset. The single model achieves 71.3% on the evaluation metrics of exact match on the hidden test set, while the ensemble model further boosts the results to 75.9%. At the time of submission of the paper, our model holds the first place on the SQuAD leaderboard for both single and ensemble model.

{{< /ci-details >}}

{{< ci-details summary="Words or Characters? Fine-grained Gating for Reading Comprehension (Zhilin Yang et al., 2016)">}}

Zhilin Yang, Bhuwan Dhingra, Ye Yuan, Junjie Hu, William W. Cohen, R. Salakhutdinov. (2016)  
**Words or Characters? Fine-grained Gating for Reading Comprehension**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/b7ffc8f44f7dafd7f51e4e7500842ec406b8e239)  
Influential Citation Count (8), SS-ID (b7ffc8f44f7dafd7f51e4e7500842ec406b8e239)  

**ABSTRACT**  
Previous work combines word-level and character-level representations using concatenation or scalar weighting, which is suboptimal for high-level tasks like reading comprehension. We present a fine-grained gating mechanism to dynamically combine word-level and character-level representations based on properties of the words. We also extend the idea of fine-grained gating to modeling the interaction between questions and paragraphs for reading comprehension. Experiments show that our approach can improve the performance on reading comprehension tasks, achieving new state-of-the-art results on the Children's Book Test dataset. To demonstrate the generality of our gating mechanism, we also show improved results on a social media tag prediction task.

{{< /ci-details >}}

{{< ci-details summary="Incorporating Copying Mechanism in Sequence-to-Sequence Learning (Jiatao Gu et al., 2016)">}}

Jiatao Gu, Zhengdong Lu, Hang Li, V. Li. (2016)  
**Incorporating Copying Mechanism in Sequence-to-Sequence Learning**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/ba30df190664193514d1d309cb673728ed48f449)  
Influential Citation Count (144), SS-ID (ba30df190664193514d1d309cb673728ed48f449)  

**ABSTRACT**  
We address an important problem in sequence-to-sequence (Seq2Seq) learning referred to as copying, in which certain segments in the input sequence are selectively replicated in the output sequence. A similar phenomenon is observable in human language communication. For example, humans tend to repeat entity names or even long phrases in conversation. The challenge with regard to copying in Seq2Seq is that new machinery is needed to decide when to perform the operation. In this paper, we incorporate copying into neural network-based Seq2Seq learning and propose a new model called CopyNet with encoder-decoder structure. CopyNet can nicely integrate the regular way of word generation in the decoder with the new copying mechanism which can choose sub-sequences in the input sequence and put them at proper places in the output sequence. Our empirical study on both synthetic data sets and real world data sets demonstrates the efficacy of CopyNet. For example, CopyNet can outperform regular RNN-based model with remarkable margins on text summarization tasks.

{{< /ci-details >}}

{{< ci-details summary="ReasoNet: Learning to Stop Reading in Machine Comprehension (Yelong Shen et al., 2016)">}}

Yelong Shen, Po-Sen Huang, Jianfeng Gao, Weizhu Chen. (2016)  
**ReasoNet: Learning to Stop Reading in Machine Comprehension**  
CoCo@NIPS  
[Paper Link](https://www.semanticscholar.org/paper/c636a2dd242908fe2e598a1077c0c57bfdea8633)  
Influential Citation Count (22), SS-ID (c636a2dd242908fe2e598a1077c0c57bfdea8633)  

**ABSTRACT**  
Teaching a computer to read and answer general questions pertaining to a document is a challenging yet unsolved problem. In this paper, we describe a novel neural network architecture called the Reasoning Network (ReasoNet) for machine comprehension tasks. ReasoNets make use of multiple turns to effectively exploit and then reason over the relation among queries, documents, and answers. Different from previous approaches using a fixed number of turns during inference, ReasoNets introduce a termination state to relax this constraint on the reasoning depth. With the use of reinforcement learning, ReasoNets can dynamically determine whether to continue the comprehension process after digesting intermediate results, or to terminate reading when it concludes that existing information is adequate to produce an answer. ReasoNets achieve superior performance in machine comprehension datasets, including unstructured CNN and Daily Mail datasets, the Stanford SQuAD dataset, and a structured Graph Reachability dataset.

{{< /ci-details >}}

{{< ci-details summary="Teaching Machines to Read and Comprehend (K. Hermann et al., 2015)">}}

K. Hermann, Tomás Kociský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, P. Blunsom. (2015)  
**Teaching Machines to Read and Comprehend**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/d1505c6123c102e53eb19dff312cb25cea840b72)  
Influential Citation Count (417), SS-ID (d1505c6123c102e53eb19dff312cb25cea840b72)  

**ABSTRACT**  
Teaching machines to read natural language documents remains an elusive challenge. Machine reading systems can be tested on their ability to answer questions posed on the contents of documents that they have seen, but until now large scale training and test datasets have been missing for this type of evaluation. In this work we define a new methodology that resolves this bottleneck and provides large scale supervised reading comprehension data. This allows us to develop a class of attention based deep neural networks that learn to read real documents and answer complex questions with minimal prior knowledge of language structure.

{{< /ci-details >}}

{{< ci-details summary="Bleu: a Method for Automatic Evaluation of Machine Translation (Kishore Papineni et al., 2002)">}}

Kishore Papineni, S. Roukos, T. Ward, Wei-Jing Zhu. (2002)  
**Bleu: a Method for Automatic Evaluation of Machine Translation**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/d7da009f457917aa381619facfa5ffae9329a6e9)  
Influential Citation Count (4410), SS-ID (d7da009f457917aa381619facfa5ffae9329a6e9)  

**ABSTRACT**  
Human evaluations of machine translation are extensive but expensive. Human evaluations can take months to finish and involve human labor that can not be reused. We propose a method of automatic machine translation evaluation that is quick, inexpensive, and language-independent, that correlates highly with human evaluation, and that has little marginal cost per run. We present this method as an automated understudy to skilled human judges which substitutes for them when there is need for quick or frequent evaluations.

{{< /ci-details >}}

{{< ci-details summary="FastQA: A Simple and Efficient Neural Architecture for Question Answering (Dirk Weissenborn et al., 2017)">}}

Dirk Weissenborn, Georg Wiese, Laura Seiffe. (2017)  
**FastQA: A Simple and Efficient Neural Architecture for Question Answering**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/e4600ece1f09236d082eca4537ee9c1efe687f6c)  
Influential Citation Count (11), SS-ID (e4600ece1f09236d082eca4537ee9c1efe687f6c)  

**ABSTRACT**  
Recent development of large-scale question answering (QA) datasets triggered a substantial amount of research into end-toend neural architectures for QA. Increasingly complex systems have been conceived without comparison to a simpler neural baseline system that would justify their complexity. In this work, we propose a simple heuristic that guided the development of FastQA, an efficient endto-end neural model for question answering that is very competitive with existing models. We further demonstrate, that an extended version (FastQAExt) achieves state-of-the-art results on recent benchmark datasets, namely SQuAD, NewsQA and MsMARCO, outperforming most existing models. However, we show that increasing the complexity of FastQA to FastQAExt does not yield any systematic improvements. We argue that the same holds true for most existing systems that are similar to FastQAExt. A manual analysis reveals that our proposed heuristic explains most predictions of our model, which indicates that modeling a simple heuristic is enough to achieve strong performance on extractive QA datasets. The overall strong performance of FastQA puts results of existing, more complex models into perspective.

{{< /ci-details >}}

{{< ci-details summary="Dynamic Coattention Networks For Question Answering (Caiming Xiong et al., 2016)">}}

Caiming Xiong, Victor Zhong, R. Socher. (2016)  
**Dynamic Coattention Networks For Question Answering**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/e978d832a4d86571e1b52aa1685dc32ccb250f50)  
Influential Citation Count (113), SS-ID (e978d832a4d86571e1b52aa1685dc32ccb250f50)  

**ABSTRACT**  
Several deep learning models have been proposed for question answering. However, due to their single-pass nature, they have no way to recover from local maxima corresponding to incorrect answers. To address this problem, we introduce the Dynamic Coattention Network (DCN) for question answering. The DCN first fuses co-dependent representations of the question and the document in order to focus on relevant parts of both. Then a dynamic pointing decoder iterates over potential answer spans. This iterative procedure enables the model to recover from initial local maxima corresponding to incorrect answers. On the Stanford question answering dataset, a single DCN model improves the previous state of the art from 71.0% F1 to 75.9%, while a DCN ensemble obtains 80.4% F1.

{{< /ci-details >}}

{{< ci-details summary="GloVe: Global Vectors for Word Representation (Jeffrey Pennington et al., 2014)">}}

Jeffrey Pennington, R. Socher, Christopher D. Manning. (2014)  
**GloVe: Global Vectors for Word Representation**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/f37e1b62a767a307c046404ca96bc140b3e68cb5)  
Influential Citation Count (3635), SS-ID (f37e1b62a767a307c046404ca96bc140b3e68cb5)  

**ABSTRACT**  
Recent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using vector arithmetic, but the origin of these regularities has remained opaque. We analyze and make explicit the model properties needed for such regularities to emerge in word vectors. The result is a new global logbilinear regression model that combines the advantages of the two major model families in the literature: global matrix factorization and local context window methods. Our model efficiently leverages statistical information by training only on the nonzero elements in a word-word cooccurrence matrix, rather than on the entire sparse matrix or on individual context windows in a large corpus. The model produces a vector space with meaningful substructure, as evidenced by its performance of 75% on a recent word analogy task. It also outperforms related models on similarity tasks and named entity recognition.

{{< /ci-details >}}

{{< ci-details summary="Machine Comprehension Using Match-LSTM and Answer Pointer (Shuohang Wang et al., 2016)">}}

Shuohang Wang, Jing Jiang. (2016)  
**Machine Comprehension Using Match-LSTM and Answer Pointer**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/ff1861b71eaedba46cb679bbe2c585dbe18f9b19)  
Influential Citation Count (81), SS-ID (ff1861b71eaedba46cb679bbe2c585dbe18f9b19)  

**ABSTRACT**  
Machine comprehension of text is an important problem in natural language processing. A recently released dataset, the Stanford Question Answering Dataset (SQuAD), offers a large number of real questions and their answers created by humans through crowdsourcing. SQuAD provides a challenging testbed for evaluating machine comprehension algorithms, partly because compared with previous datasets, in SQuAD the answers do not come from a small set of candidate answers and they have variable lengths. We propose an end-to-end neural architecture for the task. The architecture is based on match-LSTM, a model we proposed previously for textual entailment, and Pointer Net, a sequence-to-sequence model proposed by Vinyals et al.(2015) to constrain the output tokens to be from the input sequences. We propose two ways of using Pointer Net for our task. Our experiments show that both of our two models substantially outperform the best results obtained by Rajpurkar et al.(2016) using logistic regression and manually crafted features.

{{< /ci-details >}}

