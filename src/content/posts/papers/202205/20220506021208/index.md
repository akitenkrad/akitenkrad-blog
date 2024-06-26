---
draft: false
title: "A Primer in BERTology: What We Know About How BERT Works"
date: 2022-05-06
author: "akitenkrad"
description: ""
tags: ["At:Round-1", "published:2020", "Survey", "BERT", "BERTology"]
menu:
  sidebar:
    name: "A Primer in BERTology: What We Know About How BERT Works"
    identifier: 20220506
    parent: 202205
    weight: 10
math: true
mermaid: true
---

- [x] Round-1: Overview
- [ ] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation

{{< citation >}}
Rogers, A., Kovaleva, O., & Rumshisky, A. (2020).  
**A Primer in BERTology: What We Know About How BERT Works.**  
Transactions of the Association for Computational Linguistics, 8, 842–866.  
[Paper Link](https://doi.org/10.1162/TACL_A_00349)
{{< /citation >}}

## Abstract

> Transformer-based models have pushed state of the art in many areas of NLP, but our understanding of what is behind their success is still limited. This paper is the first survey of over 150 studies of the popular BERT model. We review the current state of knowledge about how BERT works, what kind of information it learns and how it is represented, common modifications to its training objectives and architecture, the overparameterization issue, and approaches to compression. We then outline directions for future research.


## Model History

TBD

## What Knowledge Does BERT Have?
### Syntactic Knowledge

---

{{< ci-details summary="Lin et al. (2019)">}}
Yongjie Lin, Y. Tan, R. Frank. (2019)  
**Open Sesame: Getting inside BERT’s Linguistic Knowledge**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/165d51a547cd920e6ac55660ad5c404dcb9562ed)  
Influential Citation Count (18), SS-ID (165d51a547cd920e6ac55660ad5c404dcb9562ed)  

**ABSTRACT**  
How and to what extent does BERT encode syntactically-sensitive hierarchical information or positionally-sensitive linear information? Recent work has shown that contextual representations like BERT perform well on tasks that require sensitivity to linguistic structure. We present here two studies which aim to provide a better understanding of the nature of BERT’s representations. The first of these focuses on the identification of structurally-defined elements using diagnostic classifiers, while the second explores BERT’s representation of subject-verb agreement and anaphor-antecedent dependencies through a quantitative assessment of self-attention vectors. In both cases, we find that BERT encodes positional information about word tokens well on its lower layers, but switches to a hierarchically-oriented encoding on higher layers. We conclude then that BERT’s representations do indeed model linguistically relevant aspects of hierarchical structure, though they do not appear to show the sharp sensitivity to hierarchical structure that is found in human processing of reflexive anaphora.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTの分散表現は線形というよりは階層的である．
---
BERTの分散表現には単語の順番に関する情報以外に文法構造の階層的な情報が含まれている可能性がある．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Tenney et al. (2019)" >}}
Ian Tenney, Patrick Xia, Berlin Chen, Alex Wang, Adam Poliak, R. Thomas McCoy, Najoung Kim, Benjamin Van Durme, Samuel R. Bowman, Dipanjan Das, Ellie Pavlick. (2019)  
**What do you learn from context? Probing for sentence structure in contextualized word representations**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/e2587eddd57bc4ba286d91b27c185083f16f40ee)  
Influential Citation Count (45), SS-ID (e2587eddd57bc4ba286d91b27c185083f16f40ee)  

**ABSTRACT**  
Contextualized representation models such as ELMo (Peters et al., 2018a) and BERT (Devlin et al., 2018) have recently achieved state-of-the-art results on a diverse array of downstream NLP tasks. Building on recent token-level probing work, we introduce a novel edge probing task design and construct a broad suite of sub-sentence tasks derived from the traditional structured NLP pipeline. We probe word-level contextual representations from four recent models and investigate how they encode sentence structure across a range of syntactic, semantic, local, and long-range phenomena. We find that existing models trained on language modeling and translation produce strong representations for syntactic phenomena, but only offer comparably small improvements on semantic tasks over a non-contextual baseline.
{{< /ci-details >}}

{{< ci-details summary="Liu et al. (2019)" >}}
Nelson F. Liu, Matt Gardner, Y. Belinkov, Matthew E. Peters, Noah A. Smith. (2019)  
**Linguistic Knowledge and Transferability of Contextual Representations**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  
Influential Citation Count (108), SS-ID (f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  

**ABSTRACT**  
Contextual word representations derived from large-scale neural language models are successful across a diverse set of NLP tasks, suggesting that they encode useful and transferable features of language. To shed light on the linguistic knowledge they capture, we study the representations produced by several recent pretrained contextualizers (variants of ELMo, the OpenAI transformer language model, and BERT) with a suite of sixteen diverse probing tasks. We find that linear models trained on top of frozen contextual representations are competitive with state-of-the-art task-specific models in many cases, but fail on tasks requiring fine-grained linguistic knowledge (e.g., conjunct identification). To investigate the transferability of contextual word representations, we quantify differences in the transferability of individual layers within contextualizers, especially between recurrent neural networks (RNNs) and transformers. For instance, higher layers of RNNs are more task-specific, while transformer layers do not exhibit the same monotonic trend. In addition, to better understand what makes contextual word representations transferable, we compare language model pretraining with eleven supervised pretraining tasks. For any given task, pretraining on a closely related task yields better performance than language model pretraining (which is better on average) when the pretraining dataset is fixed. However, language model pretraining on more data gives the best results.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTには品詞，文法的なチャンク・格に関する情報が一部含まれている．
---
ただし，文法階層的に遠く離れた親ノードのラベルを復元することはできなかったので，分散表現が保持しているのは単語の近辺の情報に限られる．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Htut et al. (2019)" >}}
Phu Mon Htut, Jason Phang, Shikha Bordia, Samuel R. Bowman. (2019)  
**Do Attention Heads in BERT Track Syntactic Dependencies?**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/ba8215e77f35b0d947c7cec39c45df4516e93421)  
Influential Citation Count (12), SS-ID (ba8215e77f35b0d947c7cec39c45df4516e93421)  

**ABSTRACT**  
We investigate the extent to which individual attention heads in pretrained transformer language models, such as BERT and RoBERTa, implicitly capture syntactic dependency relations. We employ two methods---taking the maximum attention weight and computing the maximum spanning tree---to extract implicit dependency relations from the attention weights of each layer/head, and compare them to the ground-truth Universal Dependency (UD) trees. We show that, for some UD relation types, there exist heads that can recover the dependency type significantly better than baselines on parsed English text, suggesting that some self-attention heads act as a proxy for syntactic structure. We also analyze BERT fine-tuned on two datasets---the syntax-oriented CoLA and the semantics-oriented MNLI---to investigate whether fine-tuning affects the patterns of their self-attention, but we do not observe substantial differences in the overall dependency relations extracted using our methods. Our results suggest that these models have some specialist attention heads that track individual dependency types, but no generalist head that performs holistic parsing significantly better than a trivial baseline, and that analyzing attention weights directly may not reveal much of the syntactic knowledge that BERT-style models are known to learn.
{{< /ci-details >}}

{{< ci-details summary="Jawahar et al. (2019)" >}}
Ganesh Jawahar, Benoît Sagot, Djamé Seddah. (2019)  
**What Does BERT Learn about the Structure of Language?**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/335613303ebc5eac98de757ed02a56377d99e03a)  
Influential Citation Count (44), SS-ID (335613303ebc5eac98de757ed02a56377d99e03a)  

**ABSTRACT**  
BERT is a recent language representation model that has surprisingly performed well in diverse language understanding benchmarks. This result indicates the possibility that BERT networks capture structural information about language. In this work, we provide novel support for this claim by performing a series of experiments to unpack the elements of English language structure learned by BERT. Our findings are fourfold. BERT’s phrasal representation captures the phrase-level information in the lower layers. The intermediate layers of BERT compose a rich hierarchy of linguistic information, starting with surface features at the bottom, syntactic features in the middle followed by semantic features at the top. BERT requires deeper layers while tracking subject-verb agreement to handle long-term dependency problem. Finally, the compositional scheme underlying BERT mimics classical, tree-like structures.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
文法構造がどのように埋め込まれるかということに関して， **Attention Weights** には文法的な構造は直接的には含まれていない．
---
Htut et al. (2019)によれば，文法ツリーのルートの正解データを与えたとしても，Attention Weightsから完全な文法ツリーを構成することはできなかった．
---
Jawahar et al. (2019)は，Attention Weightsから文法ツリーを抽出したとするが，定量的な結果は示されていない．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Hewitt and Manning (2019)" >}}
John Hewitt, Christopher D. Manning. (2019)  
**A Structural Probe for Finding Syntax in Word Representations**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/455a8838cde44f288d456d01c76ede95b56dc675)  
Influential Citation Count (30), SS-ID (455a8838cde44f288d456d01c76ede95b56dc675)  

**ABSTRACT**  
Recent work has improved our ability to detect linguistic knowledge in word representations. However, current methods for detecting syntactic knowledge do not test whether syntax trees are represented in their entirety. In this work, we propose a structural probe, which evaluates whether syntax trees are embedded in a linear transformation of a neural network’s word representation space. The probe identifies a linear transformation under which squared L2 distance encodes the distance between words in the parse tree, and one in which squared L2 norm encodes depth in the parse tree. Using our probe, we show that such transformations exist for both ELMo and BERT but not in baselines, providing evidence that entire syntax trees are embedded implicitly in deep models’ vector geometry.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
文法の情報はBERTの **トークンの分散表現** から復元することができる可能性がある．
---
Hewitt and Manning (2019)では，PenTreebankデータセットを使用して，BERTのトークンの分散表現から文法の依存構造を復元する変換行列を学習することに成功した．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Wu et al. (2020)" >}}
Zhiyong Wu, Yun Chen, B. Kao, Qun Liu. (2020)  
**Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/3aaa8aaad5ef36550a6b47d6ee000f0b346a5a1f)  
Influential Citation Count (10), SS-ID (3aaa8aaad5ef36550a6b47d6ee000f0b346a5a1f)  

**ABSTRACT**  
By introducing a small set of additional parameters, a probe learns to solve specific linguistic tasks (e.g., dependency parsing) in a supervised manner using feature representations (e.g., contextualized embeddings). The effectiveness of such probing tasks is taken as evidence that the pre-trained model encodes linguistic knowledge. However, this approach of evaluating a language model is undermined by the uncertainty of the amount of knowledge that is learned by the probe itself. Complementary to those works, we propose a parameter-free probing technique for analyzing pre-trained language models (e.g., BERT). Our method does not require direct supervision from the probing tasks, nor do we introduce additional parameters to the probing process. Our experiments on BERT show that syntactic trees recovered from BERT using our method are significantly better than linguistically-uninformed baselines. We further feed the empirically induced dependency structures into a downstream sentiment classification task and find its improvement compatible with or even superior to a human-designed dependency schema.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
MLM (Masked Language Model) タスクにおいて，ある単語が他の単語とどの程度関連性があるかを検証し，BERTは文法的な情報をある程度学習するが，それは正解データに近しいものとは限らないと結論づけた．
{{< /fa-arrow-right-list >}}

{{< split 6 6 >}}
<img src="fig-1-1.png" />
---
<img src="fig-1-2.png" style="margin-top:30%"/>
{{< /split >}}
Figure 1: Parameter-free probe for syntactic know-ledge: words sharing syntactic subtrees have largerimpact on each other in the MLM prediction (Wu et al.,2020).

---

{{< ci-details summary="Goldberg (2019)" >}}
Yoav Goldberg. (2019)  
**Assessing BERT's Syntactic Abilities**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/efeab0dcdb4c1cce5e537e57745d84774be99b9a)  
Influential Citation Count (17), SS-ID (efeab0dcdb4c1cce5e537e57745d84774be99b9a)  

**ABSTRACT**  
I assess the extent to which the recently introduced BERT model captures English syntactic phenomena, using (1) naturally-occurring subject-verb agreement stimuli; (2) "coloreless green ideas" subject-verb agreement stimuli, in which content words in natural sentences are randomly replaced with words sharing the same part-of-speech and inflection; and (3) manually crafted stimuli for subject-verb agreement and reflexive anaphora phenomena. The BERT model performs remarkably well on all cases.
{{< /ci-details >}}

{{< ci-details summary="van Schijndel et al. (2019)" >}}
Marten van Schijndel, Aaron Mueller, Tal Linzen. (2019)  
**Quantity doesn’t buy quality syntax with neural language models**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/356645552f8f40adf5a99b4e3a69f47699399010)  
Influential Citation Count (0), SS-ID (356645552f8f40adf5a99b4e3a69f47699399010)  

**ABSTRACT**  
Recurrent neural networks can learn to predict upcoming words remarkably well on average; in syntactically complex contexts, however, they often assign unexpectedly high probabilities to ungrammatical words. We investigate to what extent these shortcomings can be mitigated by increasing the size of the network and the corpus on which it is trained. We find that gains from increasing network size are minimal beyond a certain point. Likewise, expanding the training corpus yields diminishing returns; we estimate that the training corpus would need to be unrealistically large for the models to match human performance. A comparison to GPT and BERT, Transformer-based models trained on billions of words, reveals that these models perform even more poorly than our LSTMs in some constructions. Our results make the case for more data efficient architectures.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTはClozeタスクにおいて，主語-述語の関係を加味しながら学習しているが，意味のない文章や曖昧な文章に関しても一律に主語と動詞を関連づけようとしてしまう．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Warstadt et al. (2019)" >}}
Alex Warstadt, Yuning Cao, Ioana Grosu, Wei Peng, Hagen Blix, Yining Nie, Anna Alsop, Shikha Bordia, Haokun Liu, Alicia Parrish, Sheng-Fu Wang, Jason Phang, Anhad Mohananey, Phu Mon Htut, Paloma Jeretic, Samuel R. Bowman. (2019)  
**Investigating BERT’s Knowledge of Language: Five Analysis Methods with NPIs**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/3cd331c997e90f737810aad6fcce4d993315189f)  
Influential Citation Count (4), SS-ID (3cd331c997e90f737810aad6fcce4d993315189f)  

**ABSTRACT**  
Though state-of-the-art sentence representation models can perform tasks requiring significant knowledge of grammar, it is an open question how best to evaluate their grammatical knowledge. We explore five experimental methods inspired by prior work evaluating pretrained sentence representation models. We use a single linguistic phenomenon, negative polarity item (NPI) licensing, as a case study for our experiments. NPIs like any are grammatical only if they appear in a licensing environment like negation (Sue doesn’t have any cats vs. *Sue has any cats). This phenomenon is challenging because of the variety of NPI licensing environments that exist. We introduce an artificially generated dataset that manipulates key features of NPI licensing for the experiments. We find that BERT has significant knowledge of these features, but its success varies widely across different experimental methods. We conclude that a variety of methods is necessary to reveal all relevant aspects of a model’s grammatical knowledge in a given domain.  
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTは文法のscope violationsを検知するよりも，NPIs (Negative Polarity Items) の存在やNPIsに関連する単語を検知するほうが得意である．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Ettinger (2019)" >}}
Allyson Ettinger. (2019)  
**What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models**  
TACL  
[Paper Link](https://www.semanticscholar.org/paper/a0e49f65b6847437f262c59d0d399255101d0b75)  
Influential Citation Count (10), SS-ID (a0e49f65b6847437f262c59d0d399255101d0b75)  

**ABSTRACT**  
Pre-training by language modeling has become a popular and successful approach to NLP tasks, but we have yet to understand exactly what linguistic capacities these pre-training processes confer upon models. In this paper we introduce a suite of diagnostics drawn from human language experiments, which allow us to ask targeted questions about information used by language models for generating predictions in context. As a case study, we apply these diagnostics to the popular BERT model, finding that it can generally distinguish good from bad completions involving shared category or role reversal, albeit with less sensitivity than humans, and it robustly retrieves noun hypernyms, but it struggles with challenging inference and role-based event prediction— and, in particular, it shows clear insensitivity to the contextual impacts of negation.
{{< /ci-details >}}

{{< ci-details summary="Vulic (2020)" >}}
Goran Glavas, Ivan Vulic. (2020)  
**Is Supervised Syntactic Parsing Beneficial for Language Understanding Tasks? An Empirical Investigation**  
EACL  
[Paper Link](https://www.semanticscholar.org/paper/575ac3f36e9fddeb258e2f639e26a6a7ec35160a)  
Influential Citation Count (0), SS-ID (575ac3f36e9fddeb258e2f639e26a6a7ec35160a)  

**ABSTRACT**  
Traditional NLP has long held (supervised) syntactic parsing necessary for successful higher-level semantic language understanding (LU). The recent advent of end-to-end neural models, self-supervised via language modeling (LM), and their success on a wide range of LU tasks, however, questions this belief. In this work, we empirically investigate the usefulness of supervised parsing for semantic LU in the context of LM-pretrained transformer networks. Relying on the established fine-tuning paradigm, we first couple a pretrained transformer with a biaffine parsing head, aiming to infuse explicit syntactic knowledge from Universal Dependencies treebanks into the transformer. We then fine-tune the model for LU tasks and measure the effect of the intermediate parsing training (IPT) on downstream LU task performance. Results from both monolingual English and zero-shot language transfer experiments (with intermediate target-language parsing) show that explicit formalized syntax, injected into transformers through IPT, has very limited and inconsistent effect on downstream LU performance. Our results, coupled with our analysis of transformers’ representation spaces before and after intermediate parsing, make a significant step towards providing answers to an essential question: how (un)availing is supervised parsing for high-level semantic natural language understanding in the era of large neural models?
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTは「否定」を理解できず，不正な入力の検知に対しても脆弱である．  
---
単語の語順を入れ替えたり，文章を一部削除したり守護や目的語を取り除いてもBERTの出力結果が変わらなかったため．これが意味するところは，BERTが学習している文法は不完全であるか，またはタスクが文法的な知識に依存していないかのどちらかである．  
---
タスクにおける中間段階での教師ありFine-Tuningは下流タスクの性能に大きく影響しないため，後者である可能性が高いとのこと．
{{< /fa-arrow-right-list >}}

---

### Semantic Knowledge

---

{{< ci-details summary="Tenney et al. (2019)" >}}
Ian Tenney, Patrick Xia, Berlin Chen, Alex Wang, Adam Poliak, R. Thomas McCoy, Najoung Kim, Benjamin Van Durme, Samuel R. Bowman, Dipanjan Das, Ellie Pavlick. (2019)  
**What do you learn from context? Probing for sentence structure in contextualized word representations**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/e2587eddd57bc4ba286d91b27c185083f16f40ee)  
Influential Citation Count (45), SS-ID (e2587eddd57bc4ba286d91b27c185083f16f40ee)  

**ABSTRACT**  
Contextualized representation models such as ELMo (Peters et al., 2018a) and BERT (Devlin et al., 2018) have recently achieved state-of-the-art results on a diverse array of downstream NLP tasks. Building on recent token-level probing work, we introduce a novel edge probing task design and construct a broad suite of sub-sentence tasks derived from the traditional structured NLP pipeline. We probe word-level contextual representations from four recent models and investigate how they encode sentence structure across a range of syntactic, semantic, local, and long-range phenomena. We find that existing models trained on language modeling and translation produce strong representations for syntactic phenomena, but only offer comparably small improvements on semantic tasks over a non-contextual baseline.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
Classifierを分析することで，BERTがentity type，relations，semantic roles，proto-rolesに関する情報をエンコードしていることを示した
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Wallace et al. (2019)" >}}
Eric Wallace, Yizhong Wang, Sujian Li, Sameer Singh, Matt Gardner. (2019)  
**Do NLP Models Know Numbers? Probing Numeracy in Embeddings**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/0427110f0e79f41e69a8eb00a3ec8868bac26a4f)  
Influential Citation Count (18), SS-ID (0427110f0e79f41e69a8eb00a3ec8868bac26a4f)  

**ABSTRACT**  
The ability to understand and work with numbers (numeracy) is critical for many complex reasoning tasks. Currently, most NLP models treat numbers in text in the same way as other tokens—they embed them as distributed vectors. Is this enough to capture numeracy? We begin by investigating the numerical reasoning capabilities of a state-of-the-art question answering model on the DROP dataset. We find this model excels on questions that require numerical reasoning, i.e., it already captures numeracy. To understand how this capability emerges, we probe token embedding methods (e.g., BERT, GloVe) on synthetic list maximum, number decoding, and addition tasks. A surprising degree of numeracy is naturally present in standard embeddings. For example, GloVe and word2vec accurately encode magnitude for numbers up to 1,000. Furthermore, character-level embeddings are even more precise—ELMo captures numeracy the best for all pre-trained methods—but BERT, which uses sub-word units, is less exact.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTは数字に関する分散表現が苦手である．
---
加算や数値のエンコード・デコードに関するタスクではBERTはあまり良い性能を発揮できなかった．
---
BERTのTokenizerであるWordpieceでは，似た数値であってもベクトル空間上では離れた位置に写像されることがありうるため，これが原因の一部である可能性が考えられる．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Balasubramanian et al. (2020)" >}}
S. Balasubramanian, Naman Jain, G. Jindal, Abhijeet Awasthi, Sunita Sarawagi. (2020)  
**What’s in a Name? Are BERT Named Entity Representations just as Good for any other Name?**  
REPL4NLP  
[Paper Link](https://www.semanticscholar.org/paper/167f52d369b0979f27282af0f3a1a4be9c9be84b)  
Influential Citation Count (1), SS-ID (167f52d369b0979f27282af0f3a1a4be9c9be84b)  

**ABSTRACT**  
We evaluate named entity representations of BERT-based NLP models by investigating their robustness to replacements from the same typed class in the input. We highlight that on several tasks while such perturbations are natural, state of the art trained models are surprisingly brittle. The brittleness continues even with the recent entity-aware BERT models. We also try to discern the cause of this non-robustness, considering factors such as tokenization and frequency of occurrence. Then we provide a simple method that ensembles predictions from multiple replacements while jointly modeling the uncertainty of type annotations and label predictions. Experiments on three NLP tasks shows that our method enhances robustness and increases accuracy on both natural and adversarial datasets.
{{< /ci-details >}}

{{< ci-details summary="Broscheit (2019)" >}}
Samuel Broscheit. (2019)  
**Investigating Entity Knowledge in BERT with Simple Neural End-To-End Entity Linking**  
CoNLL  
[Paper Link](https://www.semanticscholar.org/paper/399308fa54ade9b1362d56628132323489ce50cd)  
Influential Citation Count (6), SS-ID (399308fa54ade9b1362d56628132323489ce50cd)  

**ABSTRACT**  
A typical architecture for end-to-end entity linking systems consists of three steps: mention detection, candidate generation and entity disambiguation. In this study we investigate the following questions: (a) Can all those steps be learned jointly with a model for contextualized text-representations, i.e. BERT? (b) How much entity knowledge is already contained in pretrained BERT? (c) Does additional entity knowledge improve BERT’s performance in downstream tasks? To this end we propose an extreme simplification of the entity linking setup that works surprisingly well: simply cast it as a per token classification over the entire entity vocabulary (over 700K classes in our case). We show on an entity linking benchmark that (i) this model improves the entity representations over plain BERT, (ii) that it outperforms entity linking architectures that optimize the tasks separately and (iii) that it only comes second to the current state-of-the-art that does mention detection and entity disambiguation jointly. Additionally, we investigate the usefulness of entity-aware token-representations in the text-understanding benchmark GLUE, as well as the question answering benchmarks SQUAD~V2 and SWAG and also the EN-DE WMT14 machine translation benchmark. To our surprise, we find that most of those benchmarks do not benefit from additional entity knowledge, except for a task with very small training data, the RTE task in GLUE, which improves by 2%.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTは固有名詞に関する分散表現が苦手である．
---
Balasubramanian et al. (2020) によれば，Coreference Taskにおいて名詞を置換すると，予測結果の85%が変化する．これによれば，NERタスクのF1スコアは高く出るものの，BERTは固有名詞の一般的な概念を理解しているとは言い難い．
---
Broscheit (2019) によれば，Wikipediaのentitiy linkingにおけるBERTのfine-tuningでは，entityに関して追加情報を与えるのみで，entityに関連する情報を全て学習しているわけではない．
{{< /fa-arrow-right-list >}}

---

### World Knowledge

---

{{< ci-details summary="Ettinger (2019)" >}}
Allyson Ettinger. (2019)  
**What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models**  
TACL  
[Paper Link](https://www.semanticscholar.org/paper/a0e49f65b6847437f262c59d0d399255101d0b75)  
Influential Citation Count (10), SS-ID (a0e49f65b6847437f262c59d0d399255101d0b75)  

**ABSTRACT**  
Pre-training by language modeling has become a popular and successful approach to NLP tasks, but we have yet to understand exactly what linguistic capacities these pre-training processes confer upon models. In this paper we introduce a suite of diagnostics drawn from human language experiments, which allow us to ask targeted questions about information used by language models for generating predictions in context. As a case study, we apply these diagnostics to the popular BERT model, finding that it can generally distinguish good from bad completions involving shared category or role reversal, albeit with less sensitivity than humans, and it robustly retrieves noun hypernyms, but it struggles with challenging inference and role-based event prediction— and, in particular, it shows clear insensitivity to the contextual impacts of negation.
{{< /ci-details >}}

{{< ci-details summary="Da and Kasai (2019)" >}}
Jeff Da, Jungo Kasai. (2019)  
**Understanding Commonsense Inference Aptitude of Deep Contextual Representations**  
Proceedings of the First Workshop on Commonsense Inference in Natural Language Processing  
[Paper Link](https://www.semanticscholar.org/paper/80dc7b0e6dbc26571672d9be57a0ae589689e410)  
Influential Citation Count (0), SS-ID (80dc7b0e6dbc26571672d9be57a0ae589689e410)  

**ABSTRACT**  
Pretrained deep contextual representations have advanced the state-of-the-art on various commonsense NLP tasks, but we lack a concrete understanding of the capability of these models. Thus, we investigate and challenge several aspects of BERT’s commonsense representation abilities. First, we probe BERT’s ability to classify various object attributes, demonstrating that BERT shows a strong ability in encoding various commonsense features in its embedding space, but is still deficient in many areas. Next, we show that, by augmenting BERT’s pretraining data with additional data related to the deficient attributes, we are able to improve performance on a downstream commonsense reasoning task while using a minimal amount of data. Finally, we develop a method of fine-tuning knowledge graphs embeddings alongside BERT and show the continued importance of explicit knowledge graphs.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTは実用的な推論や現実のイベントなどに関する知識の扱いが不得手である．
---
BERTは抽象的な物事や見た目，感覚的な特徴などの扱いも得意ではない．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Petroni et al. (2019)" >}}
Fabio Petroni, Tim Rocktäschel, Patrick Lewis, A. Bakhtin, Yuxiang Wu, Alexander H. Miller, S. Riedel. (2019)  
**Language Models as Knowledge Bases?**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d0086b86103a620a86bc918746df0aa642e2a8a3)  
Influential Citation Count (115), SS-ID (d0086b86103a620a86bc918746df0aa642e2a8a3)  

**ABSTRACT**  
Recent progress in pretraining language models on large textual corpora led to a surge of improvements for downstream NLP tasks. Whilst learning linguistic knowledge, these models may also be storing relational knowledge present in the training data, and may be able to answer queries structured as “fill-in-the-blank” cloze statements. Language models have many advantages over structured knowledge bases: they require no schema engineering, allow practitioners to query about an open class of relations, are easy to extend to more data, and require no human supervision to train. We present an in-depth analysis of the relational knowledge already present (without fine-tuning) in a wide range of state-of-the-art pretrained language models. We find that (i) without fine-tuning, BERT contains relational knowledge competitive with traditional NLP methods that have some access to oracle knowledge, (ii) BERT also does remarkably well on open-domain question answering against a supervised baseline, and (iii) certain types of factual knowledge are learned much more readily than others by standard language model pretraining approaches. The surprisingly strong ability of these models to recall factual knowledge without any fine-tuning demonstrates their potential as unsupervised open-domain QA systems. The code to reproduce our analysis is available at https://github.com/facebookresearch/LAMA.
{{< /ci-details >}}

{{< ci-details summary="Roberts et al. (2020)" >}}
Adam Roberts, Colin Raffel, Noam M. Shazeer. (2020)  
**How Much Knowledge Can You Pack into the Parameters of a Language Model?**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/80376bdec5f534be78ba82821f540590ebce5559)  
Influential Citation Count (38), SS-ID (80376bdec5f534be78ba82821f540590ebce5559)  

**ABSTRACT**  
It has recently been observed that neural language models trained on unstructured text can implicitly store and retrieve knowledge using natural language queries. In this short paper, we measure the practical utility of this approach by fine-tuning pre-trained models to answer questions without access to any external context or knowledge. We show that this approach scales surprisingly well with model size and outperforms models that explicitly look up knowledge on the open-domain variants of Natural Questions and WebQuestions. To facilitate reproducibility and future work, we release our code and trained models.
{{< /ci-details >}}

{{< ci-details summary="Davison et al. (2019)" >}}
Joshua Feldman, Joe Davison, Alexander M. Rush. (2019)  
**Commonsense Knowledge Mining from Pretrained Models**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/f98e135986414cccf29aec593d547c0656e4d82c)  
Influential Citation Count (17), SS-ID (f98e135986414cccf29aec593d547c0656e4d82c)  

**ABSTRACT**  
Inferring commonsense knowledge is a key challenge in machine learning. Due to the sparsity of training data, previous work has shown that supervised methods for commonsense knowledge mining underperform when evaluated on novel data. In this work, we develop a method for generating commonsense knowledge using a large, pre-trained bidirectional language model. By transforming relational triples into masked sentences, we can use this model to rank a triple’s validity by the estimated pointwise mutual information between the two entities. Since we do not update the weights of the bidirectional model, our approach is not biased by the coverage of any one commonsense knowledge base. Though we do worse on a held-out test set than models explicitly trained on a corresponding training set, our approach outperforms these methods when mining commonsense knowledge from new sources, suggesting that our unsupervised technique generalizes better than current supervised approaches.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTは関係性の抽出には長けており，シンプルなBERTモデルであってもKnowledge Baseによる手法と同等の精度を発揮することができる．
---
Roberts et al. (2019)はT5を使ってopen-domain QAでも同様の結果が得られることを示した．
---
Davison et al. (2020)によれば，BERTは関係性の推論に関しては汎化性能が高く，未知のデータにも良く対応できる．
{{< /fa-arrow-right-list >}}

<img src="fig-2.png">

---

{{< ci-details summary="Forbes et al. (2019)" >}}
Maxwell Forbes, Ari Holtzman, Yejin Choi. (2019)  
**Do Neural Language Representations Learn Physical Commonsense?**  
CogSci  
[Paper Link](https://www.semanticscholar.org/paper/cc02386375b1262c3a1d5525154eaea24c761d15)  
Influential Citation Count (3), SS-ID (cc02386375b1262c3a1d5525154eaea24c761d15)  

**ABSTRACT**  
Humans understand language based on the rich background knowledge about how the physical world works, which in turn allows us to reason about the physical world through language. In addition to the properties of objects (e.g., boats require fuel) and their affordances, i.e., the actions that are applicable to them (e.g., boats can be driven), we can also reason about if-then inferences between what properties of objects imply the kind of actions that are applicable to them (e.g., that if we can drive something then it likely requires fuel). In this paper, we investigate the extent to which state-of-the-art neural language representations, trained on a vast amount of natural language text, demonstrate physical commonsense reasoning. While recent advancements of neural language models have demonstrated strong performance on various types of natural language inference tasks, our study based on a dataset of over 200k newly collected annotations suggests that neural language representations still only learn associations that are explicitly written down.
{{< /ci-details >}}

{{< ci-details summary="Zhou et al. (2019)" >}}
Xuhui Zhou, Yue Zhang, Leyang Cui, Dandan Huang. (2019)  
**Evaluating Commonsense in Pre-trained Language Models**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/01f2b214962997260020279bd1fd1f8f372249d4)  
Influential Citation Count (5), SS-ID (01f2b214962997260020279bd1fd1f8f372249d4)  

**ABSTRACT**  
Contextualized representations trained over large raw text data have given remarkable improvements for NLP tasks including question answering and reading comprehension. There have been works showing that syntactic, semantic and word sense knowledge are contained in such representations, which explains why they benefit such tasks. However, relatively little work has been done investigating commonsense knowledge contained in contextualized representations, which is crucial for human question answering and reading comprehension. We study the commonsense ability of GPT, BERT, XLNet, and RoBERTa by testing them on seven challenging benchmarks, finding that language modeling and its variants are effective objectives for promoting models' commonsense ability while bi-directional context and larger training set are bonuses. We additionally find that current models do poorly on tasks require more necessary inference steps. Finally, we test the robustness of models by making dual test cases, which are correlated so that the correct prediction of one sample should lead to correct prediction of the other. Interestingly, the models show confusion on these test cases, which suggests that they learn commonsense at the surface rather than the deep level. We release a test set, named CATs publicly, for future research.
{{< /ci-details >}}

{{< ci-details summary="Richardson and Sabharwal (2019)" >}}
Kyle Richardson, Ashish Sabharwal. (2019)  
**What Does My QA Model Know? Devising Controlled Probes Using Expert Knowledge**  
Transactions of the Association for Computational Linguistics  
[Paper Link](https://www.semanticscholar.org/paper/5a9001cdccdb8b1de227a45eccc503d32d1a2464)  
Influential Citation Count (2), SS-ID (5a9001cdccdb8b1de227a45eccc503d32d1a2464)  

**ABSTRACT**  
Abstract Open-domain question answering (QA) involves many knowledge and reasoning challenges, but are successful QA models actually learning such knowledge when trained on benchmark QA tasks? We investigate this via several new diagnostic tasks probing whether multiple-choice QA models know definitions and taxonomic reasoning—two skills widespread in existing benchmarks and fundamental to more complex reasoning. We introduce a methodology for automatically building probe datasets from expert knowledge sources, allowing for systematic control and a comprehensive evaluation. We include ways to carefully control for artifacts that may arise during this process. Our evaluation confirms that transformer-based multiple-choice QA models are already predisposed to recognize certain types of structural linguistic knowledge. However, it also reveals a more nuanced picture: their performance notably degrades even with a slight increase in the number of “hops” in the underlying taxonomic hierarchy, and with more challenging distractor candidates. Further, existing models are far from perfect when assessed at the level of clusters of semantically connected probes, such as all hypernym questions about a single concept.
{{< /ci-details >}}

{{< ci-details summary="Poerner et al. (2019)" >}}
Nina Poerner, Ulli Waltinger, Hinrich Schütze. (2019)  
**BERT is Not a Knowledge Base (Yet): Factual Knowledge vs. Name-Based Reasoning in Unsupervised QA**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/7c62ac7aedacc39ca417a48f8134e0514dc6a523)  
Influential Citation Count (8), SS-ID (7c62ac7aedacc39ca417a48f8134e0514dc6a523)  

**ABSTRACT**  
The BERT language model (LM) (Devlin et al., 2019) is surprisingly good at answering cloze-style questions about relational facts. Petroni et al. (2019) take this as evidence that BERT memorizes factual knowledge during pre-training. We take issue with this interpretation and argue that the performance of BERT is partly due to reasoning about (the surface form of) entity names, e.g., guessing that a person with an Italian-sounding name speaks Italian. More specifically, we show that BERT's precision drops dramatically when we filter certain easy-to-guess facts. As a remedy, we propose E-BERT, an extension of BERT that replaces entity mentions with symbolic entity embeddings. E-BERT outperforms both BERT and ERNIE (Zhang et al., 2019) on hard-to-guess queries. We take this as evidence that E-BERT is richer in factual knowledge, and we show two ways of ensembling BERT and E-BERT.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTはWorld Knowledgeに関する因果推論を行うことはできない．
---
BERTは関係性を推論することはできるが，その関係性の因果については感知しない．例えば，BERTは「人間が家に入る」ということと，「家が大きい」ということはそれぞれ学習できるが，「家が人間よりも大きいかどうか」ということは判断できない．
---
BERTのWorld Knowledgeに関する成功は，現実世界のステレオタイプに依存している．例えば，BERTモデルではアメリカ人であろうとドイツ人であろうと，イタリア風な名前の持ち主はイタリア人であると判断される．
{{< /fa-arrow-right-list >}}

---

## Limitations

---

{{< ci-details summary="Tenney et al. (2019)" >}}
Ian Tenney, Dipanjan Das, Ellie Pavlick. (2019)  
**BERT Rediscovers the Classical NLP Pipeline**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/97906df07855b029b7aae7c2a1c6c5e8df1d531c)  
Influential Citation Count (59), SS-ID (97906df07855b029b7aae7c2a1c6c5e8df1d531c)  

**ABSTRACT**  
Pre-trained text encoders have rapidly advanced the state of the art on many NLP tasks. We focus on one such model, BERT, and aim to quantify where linguistic information is captured within the network. We find that the model represents the steps of the traditional NLP pipeline in an interpretable and localizable way, and that the regions responsible for each step appear in the expected sequence: POS tagging, parsing, NER, semantic roles, then coreference. Qualitative analysis reveals that the model can and often does adjust this pipeline dynamically, revising lower-level decisions on the basis of disambiguating information from higher-level representations.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTモデルの深掘りによって文法的・意味的な特徴がモデルから見つからなかったということは，必ずしもBERTモデルがそれらの特徴を持っていないということを意味しているわけではない．
---
よりよりモデルの調査方法が見つかれば，実はモデルがそれらの特徴を保持していたという事実が見つかる可能性は常にある．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Warstadt et al. (2019)">}}
Alex Warstadt, Yuning Cao, Ioana Grosu, Wei Peng, Hagen Blix, Yining Nie, Anna Alsop, Shikha Bordia, Haokun Liu, Alicia Parrish, Sheng-Fu Wang, Jason Phang, Anhad Mohananey, Phu Mon Htut, Paloma Jeretic, Samuel R. Bowman. (2019)  
**Investigating BERT’s Knowledge of Language: Five Analysis Methods with NPIs**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/3cd331c997e90f737810aad6fcce4d993315189f)  
Influential Citation Count (4), SS-ID (3cd331c997e90f737810aad6fcce4d993315189f)  

**ABSTRACT**  
Though state-of-the-art sentence representation models can perform tasks requiring significant knowledge of grammar, it is an open question how best to evaluate their grammatical knowledge. We explore five experimental methods inspired by prior work evaluating pretrained sentence representation models. We use a single linguistic phenomenon, negative polarity item (NPI) licensing, as a case study for our experiments. NPIs like any are grammatical only if they appear in a licensing environment like negation (Sue doesn’t have any cats vs. *Sue has any cats). This phenomenon is challenging because of the variety of NPI licensing environments that exist. We introduce an artificially generated dataset that manipulates key features of NPI licensing for the experiments. We find that BERT has significant knowledge of these features, but its success varies widely across different experimental methods. We conclude that a variety of methods is necessary to reveal all relevant aspects of a model’s grammatical knowledge in a given domain.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
モデルの調査方法によっては，互いを補完するような結果が出ることもあれば，全く矛盾する結果が出てくることもある．
---
また，一口にBERTと言っても多くの亜種が開発されているので，どのモデルを優先的に調査するかによって結果が変わってくる．
---
一つの解決策は "BERT" モデルに注目して，BERTが何に依拠して推論を実施しているのかを明らかにすることである．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Pimentel et al. (2020)" >}}
Tiago Pimentel, Josef Valvoda, Rowan Hall Maudslay, Ran Zmigrod, Adina Williams, Ryan Cotterell. (2020)  
**Information-Theoretic Probing for Linguistic Structure**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/738c6d664aa6c3854e1aa894957bd595f621fc42)  
Influential Citation Count (16), SS-ID (738c6d664aa6c3854e1aa894957bd595f621fc42)  

**ABSTRACT**  
The success of neural networks on a diverse set of NLP tasks has led researchers to question how much these networks actually “know” about natural language. Probes are a natural way of assessing this. When probing, a researcher chooses a linguistic task and trains a supervised model to predict annotations in that linguistic task from the network’s learned representations. If the probe does well, the researcher may conclude that the representations encode knowledge related to the task. A commonly held belief is that using simpler models as probes is better; the logic is that simpler models will identify linguistic structure, but not learn the task itself. We propose an information-theoretic operationalization of probing as estimating mutual information that contradicts this received wisdom: one should always select the highest performing probe one can, even if it is more complex, since it will result in a tighter estimate, and thus reveal more of the linguistic information inherent in the representation. The experimental portion of our paper focuses on empirically estimating the mutual information between a linguistic property and BERT, comparing these estimates to several baselines. We evaluate on a set of ten typologically diverse languages often underrepresented in NLP research—plus English—totalling eleven languages. Our implementation is available in https://github.com/rycolab/info-theoretic-probing.
{{< /ci-details >}}

{{< ci-details summary="Voita and Titov (2020)" >}}
Elena Voita, Ivan Titov. (2020)  
**Information-Theoretic Probing with Minimum Description Length**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/f4b585c9a79dfce0807b445a09036ea0f9cbcdce)  
Influential Citation Count (13), SS-ID (f4b585c9a79dfce0807b445a09036ea0f9cbcdce)  

**ABSTRACT**  
To measure how well pretrained representations encode some linguistic property, it is common to use accuracy of a probe, i.e. a classifier trained to predict the property from the representations. Despite widespread adoption of probes, differences in their accuracy fail to adequately reflect differences in representations. For example, they do not substantially favour pretrained representations over randomly initialized ones. Analogously, their accuracy can be similar when probing for genuine linguistic labels and probing for random synthetic tasks. To see reasonable differences in accuracy with respect to these random baselines, previous work had to constrain either the amount of probe training data or its model size. Instead, we propose an alternative to the standard probes, information-theoretic probing with minimum description length (MDL). With MDL probing, training a probe to predict labels is recast as teaching it to effectively transmit the data. Therefore, the measure of interest changes from probe accuracy to the description length of labels given representations. In addition to probe quality, the description length evaluates "the amount of effort" needed to achieve the quality. This amount of effort characterizes either (i) size of a probing model, or (ii) the amount of data needed to achieve the high quality. We consider two methods for estimating MDL which can be easily implemented on top of the standard probing pipelines: variational coding and online coding. We show that these methods agree in results and are more informative and stable than the standard probes.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
今一つの方向性は，情報理論によるモデル内部の調査である．
---
Pimentel et al. (2020)では，言語情報が与えられたときにモデルがどのような分散表現を学習するのかということに関して，相互情報量を推定するアプローチをとっている．
---
相互情報量アプローチでは，分散表現に含まれる情報の量よりも，情報をどの程度簡単に抽出できるか，ということに主眼が置かれた．
---
Voita and Titov (2020)では，モデルの分散表現から情報を取り出す場合にどの程度の労力が必要なのかを定量化している．
{{< /fa-arrow-right-list >}}

---

## Localizing Linguistic Knowledge

### BERT Embeddings

---

{{< ci-details summary="Mikolov et al. (2013)" >}}
Tomas Mikolov, Ilya Sutskever, Kai Chen, G. Corrado, J. Dean. (2013)  
**Distributed Representations of Words and Phrases and their Compositionality**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/87f40e6f3022adbc1f1905e3e506abad05a9964f)  
Influential Citation Count (3587), SS-ID (87f40e6f3022adbc1f1905e3e506abad05a9964f)  

**ABSTRACT**  
The recently introduced continuous Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships. In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word representations. We also describe a simple alternative to the hierarchical softmax called negative sampling.    An inherent limitation of word representations is their indifference to word order and their inability to represent idiomatic phrases. For example, the meanings of "Canada" and "Air" cannot be easily combined to obtain "Air Canada". Motivated by this example, we present a simple method for finding phrases in text, and show that learning good vector representations for millions of phrases is possible.
{{< /ci-details >}}

{{< ci-details summary="Kong et al. (2019)" >}}
Lingpeng Kong, Cyprien de Masson d'Autume, Wang Ling, Lei Yu, Zihang Dai, Dani Yogatama. (2019)  
**A Mutual Information Maximization Perspective of Language Representation Learning**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/b04889922aae7f799affb2ae6508bc5f5c989567)  
Influential Citation Count (11), SS-ID (b04889922aae7f799affb2ae6508bc5f5c989567)  

**ABSTRACT**  
We show state-of-the-art word representation learning methods maximize an objective function that is a lower bound on the mutual information between different parts of a word sequence (i.e., a sentence). Our formulation provides an alternative perspective that unifies classical word embedding models (e.g., Skip-gram) and modern contextual embeddings (e.g., BERT, XLNet). In addition to enhancing our theoretical understanding of these methods, our derivation leads to a principled framework that can be used to construct new self-supervised tasks. We provide an example by drawing inspirations from related methods based on mutual information maximization that have been successful in computer vision, and introduce a simple self-supervised objective that maximizes the mutual information between a global sentence representation and n-grams in the sentence. Our analysis offers a holistic view of representation learning methods to transfer knowledge and translate progress across multiple domains (e.g., natural language processing, computer vision, audio processing).
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTにおいて，`embedding` とはTransformerレイヤのアウトプット（特に最終層のアウトプット）を指す．
---
Kong et al. (2019) によれば，Word2Vec（Mikolov et al., 2013）もBERTのembeddingも相互情報量の一種であるとみなすことができるが，BERTのEmbeddingは前者と比較したときに `contextualized` されているという特徴がある．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Akbik et al. (2019)" >}}
A. Akbik, Tanja Bergmann, Roland Vollgraf. (2019)  
**Pooled Contextualized Embeddings for Named Entity Recognition**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/edfe9dd16316618e694cd087d0d418dac91eb48c)  
Influential Citation Count (40), SS-ID (edfe9dd16316618e694cd087d0d418dac91eb48c)  

**ABSTRACT**  
Contextual string embeddings are a recent type of contextualized word embedding that were shown to yield state-of-the-art results when utilized in a range of sequence labeling tasks. They are based on character-level language models which treat text as distributions over characters and are capable of generating embeddings for any string of characters within any textual context. However, such purely character-based approaches struggle to produce meaningful embeddings if a rare string is used in a underspecified context. To address this drawback, we propose a method in which we dynamically aggregate contextualized embeddings of each unique string that we encounter. We then use a pooling operation to distill a ”global” word representation from all contextualized instances. We evaluate these ”pooled contextualized embeddings” on common named entity recognition (NER) tasks such as CoNLL-03 and WNUT and show that our approach significantly improves the state-of-the-art for NER. We make all code and pre-trained models available to the research community for use and reproduction.
{{< /ci-details >}}

{{< ci-details summary="Bommasani et al. (2020)" >}}
Rishi Bommasani, Kelly Davis, Claire Cardie. (2020)  
**Interpreting Pretrained Contextualized Representations via Reductions to Static Embeddings**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/d34580c522c79d5cde620331dd9ffb18643a8090)  
Influential Citation Count (17), SS-ID (d34580c522c79d5cde620331dd9ffb18643a8090)  

**ABSTRACT**  
Contextualized representations (e.g. ELMo, BERT) have become the default pretrained representations for downstream NLP applications. In some settings, this transition has rendered their static embedding predecessors (e.g. Word2Vec, GloVe) obsolete. As a side-effect, we observe that older interpretability methods for static embeddings — while more diverse and mature than those available for their dynamic counterparts — are underutilized in studying newer contextualized representations. Consequently, we introduce simple and fully general methods for converting from contextualized representations to static lookup-table embeddings which we apply to 5 popular pretrained models and 9 sets of pretrained weights. Our analysis of the resulting static embeddings notably reveals that pooling over many contexts significantly improves representational quality under intrinsic evaluation. Complementary to analyzing representational quality, we consider social biases encoded in pretrained representations with respect to gender, race/ethnicity, and religion and find that bias is encoded disparately across pretrained models and internal layers even for models with the same training data. Concerningly, we find dramatic inconsistencies between social bias estimators for word embeddings.
{{< /ci-details >}}

{{< ci-details summary="May et al. (2019)" >}}
Chandler May, Alex Wang, Shikha Bordia, Samuel R. Bowman, Rachel Rudinger. (2019)  
**On Measuring Social Biases in Sentence Encoders**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/5e9c85235210b59a16bdd84b444a904ae271f7e7)  
Influential Citation Count (25), SS-ID (5e9c85235210b59a16bdd84b444a904ae271f7e7)  

**ABSTRACT**  
The Word Embedding Association Test shows that GloVe and word2vec word embeddings exhibit human-like implicit biases based on gender, race, and other social constructs (Caliskan et al., 2017). Meanwhile, research on learning reusable text representations has begun to explore sentence-level texts, with some sentence encoders seeing enthusiastic adoption. Accordingly, we extend the Word Embedding Association Test to measure bias in sentence encoders. We then test several sentence encoders, including state-of-the-art methods such as ELMo and BERT, for the social biases studied in prior work and two important biases that are difficult or impossible to test at the word level. We observe mixed results including suspicious patterns of sensitivity that suggest the test’s assumptions may not hold in general. We conclude by proposing directions for future work on measuring bias in sentence encoders.
{{< /ci-details >}}

{{< ci-details summary="Wang et al. (2020)" >}}
Karthikeyan K, Zihan Wang, Stephen Mayhew, D. Roth. (2019)  
**Cross-Lingual Ability of Multilingual BERT: An Empirical Study**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/3b2538f84812f434c740115c185be3e5e216c526)  
Influential Citation Count (7), SS-ID (3b2538f84812f434c740115c185be3e5e216c526)  

**ABSTRACT**  
Recent work has exhibited the surprising cross-lingual abilities of multilingual BERT (M-BERT) -- surprising since it is trained without any cross-lingual objective and with no aligned data. In this work, we provide a comprehensive study of the contribution of different components in M-BERT to its cross-lingual ability. We study the impact of linguistic properties of the languages, the architecture of the model, and the learning objectives. The experimental study is done in the context of three typologically different languages -- Spanish, Hindi, and Russian -- and using two conceptually different NLP tasks, textual entailment and named entity recognition. Among our key conclusions is the fact that the lexical overlap between languages plays a negligible role in the cross-lingual success, while the depth of the network is an integral part of it. All our models and implementations can be found on our project page: this http URL .
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
`distilled contextualized embedding` は単語レベルの伝統的なタスクにおける単語の類似度といった表現と比べて，単語の意味に関して多くの情報を含んでいるということが複数の研究によって示されている．
---
Akbik et al. (2019)，Bommasani et al. (2020)，May et al. (2019)，Wang et al. (2020)によればcontextualized representationを蒸留する手法は複数のコンテキストに渡って情報を集約するプロセスを含んでいる．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Ethayarajh (2019)" >}}
Kawin Ethayarajh. (2019)  
**How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/9d7902e834d5d1d35179962c7a5b9d16623b0d39)  
Influential Citation Count (29), SS-ID (9d7902e834d5d1d35179962c7a5b9d16623b0d39)  

**ABSTRACT**  
Replacing static word embeddings with contextualized word representations has yielded significant improvements on many NLP tasks. However, just how contextual are the contextualized representations produced by models such as ELMo and BERT? Are there infinitely many context-specific representations for each word, or are words essentially assigned one of a finite number of word-sense representations? For one, we find that the contextualized representations of all words are not isotropic in any layer of the contextualizing model. While representations of the same word in different contexts still have a greater cosine similarity than those of two different words, this self-similarity is much lower in upper layers. This suggests that upper layers of contextualizing models produce more context-specific representations, much like how upper layers of LSTMs produce more task-specific representations. In all layers of ELMo, BERT, and GPT-2, on average, less than 5% of the variance in a word’s contextualized representations can be explained by a static embedding for that word, providing some justification for the success of contextualized representations.
{{< /ci-details >}}


{{< fa-arrow-right-list >}}
単語にフォーカスした分散表現をBERT内部のレイヤ間で比較した場合，後ろの方のレイヤはよりcontext-specificな情報を含むように学習されている．
---
もし分散表現の空間が各次元に関して一様なものであるならば（directionally uniform/isotropic） BERTの分散表現においては2つのランダムな単語が思ったよりも高いコサイン類似度を示すことになる．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Wiedemann et al. (2019)" >}}
Gregor Wiedemann, Steffen Remus, Avi Chawla, Chris Biemann. (2019)  
**Does BERT Make Any Sense? Interpretable Word Sense Disambiguation with Contextualized Embeddings**  
KONVENS  
[Paper Link](https://www.semanticscholar.org/paper/ba8b3d0d2b09bc2b56c6d3f153919786d9fc3075)  
Influential Citation Count (6), SS-ID (ba8b3d0d2b09bc2b56c6d3f153919786d9fc3075)  

**ABSTRACT**  
Contextualized word embeddings (CWE) such as provided by ELMo (Peters et al., 2018), Flair NLP (Akbik et al., 2018), or BERT (Devlin et al., 2019) are a major recent innovation in NLP. CWEs provide semantic vector representations of words depending on their respective context. Their advantage over static word embeddings has been shown for a number of tasks, such as text classification, sequence tagging, or machine translation. Since vectors of the same word type can vary depending on the respective context, they implicitly provide a model for word sense disambiguation (WSD). We introduce a simple but effective approach to WSD using a nearest neighbor classification on CWEs. We compare the performance of different CWE models for the task and can report improvements above the current state of the art for two standard WSD benchmark datasets. We further show that the pre-trained BERT model is able to place polysemic words into distinct 'sense' regions of the embedding space, while ELMo and Flair NLP do not seem to possess this ability.
{{< /ci-details >}}

{{< ci-details summary="Schmidt and Hofmann (2020)" >}}
Florian Schmidt, T. Hofmann. (2020)  
**BERT as a Teacher: Contextual Embeddings for Sequence-Level Reward**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/de9e7d6319b26c0d9f0da20c79403e9b9367fff4)  
Influential Citation Count (0), SS-ID (de9e7d6319b26c0d9f0da20c79403e9b9367fff4)  

**ABSTRACT**  
Measuring the quality of a generated sequence against a set of references is a central problem in many learning frameworks, be it to compute a score, to assign a reward, or to perform discrimination. Despite great advances in model architectures, metrics that scale independently of the number of references are still based on n-gram estimates. We show that the underlying operations, counting words and comparing counts, can be lifted to embedding words and comparing embeddings. An in-depth analysis of BERT embeddings shows empirically that contextual embeddings can be employed to capture the required dependencies while maintaining the necessary scalability through appropriate pruning and smoothing techniques. We cast unconditional generation as a reinforcement learning problem and show that our reward function indeed provides a more effective learning signal than n-gram reward in this challenging setting.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTの分散表現がcontextualizedされているならば，BERTは多義語や同音異義語といったものに関してどの程度の情報を持っているのか？
---
BERTのcontextualized embeddingは単語の意味に応じてはっきりと別れたクラスター（`distinct clusters`）を形成している．これによって，BERTは語義曖昧性解消といったタスクにおいて高い精度を達成することができる．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Mickus et al. (2019) ">}}
Timothee Mickus, Denis Paperno, Mathieu Constant, Kees van Deemter. (2019)  
**What do you mean, BERT? Assessing BERT as a Distributional Semantics Model**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/4bff291cf7fa02a0dbac767aba55d43ad8c59055)  
Influential Citation Count (3), SS-ID (4bff291cf7fa02a0dbac767aba55d43ad8c59055)  

**ABSTRACT**  
Contextualized word embeddings, i.e. vector representations for words in context, are naturally seen as an extension of previous noncontextual distributional semantic models. In this work, we focus on BERT, a deep neural network that produces contextualized embeddings and has set the state-of-the-art in several semantic tasks, and study the semantic coherence of its embedding space. While showing a tendency towards coherence, BERT does not fully live up to the natural expectations for a semantic vector space. In particular, we find that the position of the sentence in which a word occurs, while having no meaning correlates, leaves a noticeable trace on the word embeddings and disturbs similarity relationships.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
Mickus et al. (2019)によれば，BERTにおける単語の分散表現は，同じ単語であったとしても，その単語がどの文章のどの位置に出現するかに依存して変動する．変動の程度によっては，同じ単語の意味がブレることになるので，これは言語学的な観点からは望ましくない．
{{< /fa-arrow-right-list >}}

---

{{< fa-arrow-right-list >}}
このセクションにおける研究成果は主に単語の分散表現に着目したものであるが，BERTは文章のエンコーダとして使われるのが一般的であることに鑑みれば，単語単位の分散表現だけでなく文章単位の分散表現に関しても詳しく調査すべきである．
{{< /fa-arrow-right-list >}}

---

### Self-Attention Heads

Attention Head にどのようなタイプが存在するか，ということについてはいくつかの研究成果がある．

---

{{< ci-details summary="Raganato and Tiedemann (2018)" >}}
Alessandro Raganato, J. Tiedemann. (2018)  
**An Analysis of Encoder Representations in Transformer-Based Machine Translation**  
BlackboxNLP@EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/94238dead40b12735d79ed63e29ead70730261a2)  
Influential Citation Count (10), SS-ID (94238dead40b12735d79ed63e29ead70730261a2)  

**ABSTRACT**  
The attention mechanism is a successful technique in modern NLP, especially in tasks like machine translation. The recently proposed network architecture of the Transformer is based entirely on attention mechanisms and achieves new state of the art results in neural machine translation, outperforming other sequence-to-sequence models. However, so far not much is known about the internal properties of the model and the representations it learns to achieve that performance. To study this question, we investigate the information that is learned by the attention mechanism in Transformer models with different translation quality. We assess the representations of the encoder by extracting dependency relations based on self-attention weights, we perform four probing tasks to study the amount of syntactic and semantic captured information and we also test attention in a transfer learning scenario. Our analysis sheds light on the relative strengths and weaknesses of the various encoder representations. We observe that specific attention heads mark syntactic dependency relations and we can also confirm that lower layers tend to learn more about syntax while higher layers tend to encode more semantics.
{{< /ci-details >}}

{{< ci-details summary="Clark et al. (2019)" >}}
Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning. (2019)  
**What Does BERT Look at? An Analysis of BERT’s Attention**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/95a251513853c6032bdecebd4b74e15795662986)  
Influential Citation Count (74), SS-ID (95a251513853c6032bdecebd4b74e15795662986)  

**ABSTRACT**  
Large pre-trained neural networks such as BERT have had great recent success in NLP, motivating a growing body of research investigating what aspects of language they are able to learn from unlabeled data. Most recent analysis has focused on model outputs (e.g., language model surprisal) or internal vector representations (e.g., probing classifiers). Complementary to these works, we propose methods for analyzing the attention mechanisms of pre-trained models and apply them to BERT. BERT’s attention heads exhibit patterns such as attending to delimiter tokens, specific positional offsets, or broadly attending over the whole sentence, with heads in the same layer often exhibiting similar behaviors. We further show that certain attention heads correspond well to linguistic notions of syntax and coreference. For example, we find heads that attend to the direct objects of verbs, determiners of nouns, objects of prepositions, and coreferent mentions with remarkably high accuracy. Lastly, we propose an attention-based probing classifier and use it to further demonstrate that substantial syntactic information is captured in BERT’s attention.
{{< /ci-details >}}

{{< ci-details summary="Kovaleva et al. (2019)" >}}
Olga Kovaleva, Alexey Romanov, Anna Rogers, Anna Rumshisky. (2019)  
**Revealing the Dark Secrets of BERT**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d78aed1dac6656affa4a04cbf225ced11a83d103)  
Influential Citation Count (34), SS-ID (d78aed1dac6656affa4a04cbf225ced11a83d103)  

**ABSTRACT**  
BERT-based architectures currently give state-of-the-art performance on many NLP tasks, but little is known about the exact mechanisms that contribute to its success. In the current work, we focus on the interpretation of self-attention, which is one of the fundamental underlying components of BERT. Using a subset of GLUE tasks and a set of handcrafted features-of-interest, we propose the methodology and carry out a qualitative and quantitative analysis of the information encoded by the individual BERT’s heads. Our findings suggest that there is a limited set of attention patterns that are repeated across different heads, indicating the overall model overparametrization. While different heads consistently use the same attention patterns, they have varying impact on performance across different tasks. We show that manually disabling attention in certain heads leads to a performance improvement over the regular fine-tuned BERT models.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
Raganato and Tiedemann (2018)によれば，Self-Attention Headsのタイプにはトークン自体に加えて，`[CLS]`，`[SEP]`トークンと文章の最後を表すトークンが含まれる．
---
Clark et al. (2019)では，`[CLS]`，`[SEP]`トークン，句読点や記号などでタイプが分かれることが議論された．
---
Kovaleva et al. (2019)はSelf-Attention Headsのタイプについて，5つのパターンがあることを示している．
{{< /fa-arrow-right-list >}}

<img src="fig-3.png" />

---

#### Heads with Linguistic Functions

---

{{< ci-details summary="Htut et al. (2019)" >}}
Phu Mon Htut, Jason Phang, Shikha Bordia, Samuel R. Bowman. (2019)  
**Do Attention Heads in BERT Track Syntactic Dependencies?**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/ba8215e77f35b0d947c7cec39c45df4516e93421)  
Influential Citation Count (12), SS-ID (ba8215e77f35b0d947c7cec39c45df4516e93421)  

**ABSTRACT**  
We investigate the extent to which individual attention heads in pretrained transformer language models, such as BERT and RoBERTa, implicitly capture syntactic dependency relations. We employ two methods---taking the maximum attention weight and computing the maximum spanning tree---to extract implicit dependency relations from the attention weights of each layer/head, and compare them to the ground-truth Universal Dependency (UD) trees. We show that, for some UD relation types, there exist heads that can recover the dependency type significantly better than baselines on parsed English text, suggesting that some self-attention heads act as a proxy for syntactic structure. We also analyze BERT fine-tuned on two datasets---the syntax-oriented CoLA and the semantics-oriented MNLI---to investigate whether fine-tuning affects the patterns of their self-attention, but we do not observe substantial differences in the overall dependency relations extracted using our methods. Our results suggest that these models have some specialist attention heads that track individual dependency types, but no generalist head that performs holistic parsing significantly better than a trivial baseline, and that analyzing attention weights directly may not reveal much of the syntactic knowledge that BERT-style models are known to learn.
{{< /ci-details >}}

{{< ci-details summary="Clark et al. (2019)" >}}
Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning. (2019)  
**What Does BERT Look at? An Analysis of BERT’s Attention**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/95a251513853c6032bdecebd4b74e15795662986)  
Influential Citation Count (74), SS-ID (95a251513853c6032bdecebd4b74e15795662986)  

**ABSTRACT**  
Large pre-trained neural networks such as BERT have had great recent success in NLP, motivating a growing body of research investigating what aspects of language they are able to learn from unlabeled data. Most recent analysis has focused on model outputs (e.g., language model surprisal) or internal vector representations (e.g., probing classifiers). Complementary to these works, we propose methods for analyzing the attention mechanisms of pre-trained models and apply them to BERT. BERT’s attention heads exhibit patterns such as attending to delimiter tokens, specific positional offsets, or broadly attending over the whole sentence, with heads in the same layer often exhibiting similar behaviors. We further show that certain attention heads correspond well to linguistic notions of syntax and coreference. For example, we find heads that attend to the direct objects of verbs, determiners of nouns, objects of prepositions, and coreferent mentions with remarkably high accuracy. Lastly, we propose an attention-based probing classifier and use it to further demonstrate that substantial syntactic information is captured in BERT’s attention.
{{< /ci-details >}}

{{< ci-details summary="Voita et al. (2019)" >}}
Elena Voita, David Talbot, F. Moiseev, Rico Sennrich, Ivan Titov. (2019)  
**Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/07a64686ce8e43ac475a8d820a8a9f1d87989583)  
Influential Citation Count (49), SS-ID (07a64686ce8e43ac475a8d820a8a9f1d87989583)  

**ABSTRACT**  
Multi-head self-attention is a key component of the Transformer, a state-of-the-art architecture for neural machine translation. In this work we evaluate the contribution made by individual attention heads to the overall performance of the model and analyze the roles played by them in the encoder. We find that the most important and confident heads play consistent and often linguistically-interpretable roles. When pruning heads using a method based on stochastic gates and a differentiable relaxation of the L0 penalty, we observe that specialized heads are last to be pruned. Our novel pruning method removes the vast majority of heads without seriously affecting performance. For example, on the English-Russian WMT dataset, pruning 38 out of 48 encoder heads results in a drop of only 0.15 BLEU.
{{< /ci-details >}}

{{< ci-details summary="Hoover et al. (2019)" >}}
Benjamin Hoover, Hendrik Strobelt, Sebastian Gehrmann. (2019)  
**exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformer Models**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/327d7e55d64cb34d55bd3a3fe58233c238a312cd)  
Influential Citation Count (4), SS-ID (327d7e55d64cb34d55bd3a3fe58233c238a312cd)  

**ABSTRACT**  
Large Transformer-based language models can route and reshape complex information via their multi-headed attention mechanism. Although the attention never receives explicit supervision, it can exhibit recognizable patterns following linguistic or positional information. Analyzing the learned representations and attentions is paramount to furthering our understanding of the inner workings of these models. However, analyses have to catch up with the rapid release of new models and the growing diversity of investigation techniques. To support analysis for a wide variety of models, we introduce exBERT, a tool to help humans conduct flexible, interactive investigations and formulate hypotheses for the model-internal reasoning process. exBERT provides insights into the meaning of the contextual representations and attention by matching a human-specified input to similar contexts in large annotated datasets. By aggregating the annotations of the matched contexts, exBERT can quickly replicate findings from literature and extend them to previously not analyzed models.
{{< /ci-details >}}

{{< ci-details summary="Zhao and Bethard (2019)" >}}
Yiyun Zhao, Steven Bethard. (2020)  
**How does BERT’s attention change when you fine-tune? An analysis methodology and a case study in negation scope**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/868349fe969bc7c6b14b5f35e118a26075b7b1f2)  
Influential Citation Count (0), SS-ID (868349fe969bc7c6b14b5f35e118a26075b7b1f2)  

**ABSTRACT**  
Large pretrained language models like BERT, after fine-tuning to a downstream task, have achieved high performance on a variety of NLP problems. Yet explaining their decisions is difficult despite recent work probing their internal representations. We propose a procedure and analysis methods that take a hypothesis of how a transformer-based model might encode a linguistic phenomenon, and test the validity of that hypothesis based on a comparison between knowledge-related downstream tasks with downstream control tasks, and measurement of cross-dataset consistency. We apply this methodology to test BERT and RoBERTa on a hypothesis that some attention heads will consistently attend from a word in negation scope to the negation cue. We find that after fine-tuning BERT and RoBERTa on a negation scope task, the average attention head improves its sensitivity to negation and its attention consistency across negation datasets compared to the pre-trained models. However, only the base models (not the large models) improve compared to a control task, indicating there is evidence for a shallow encoding of negation only in the base models.
{{< /ci-details >}}

---

{{< fa-arrow-right-list >}}
Self-Attention Headsはそれぞれ，ある文法的な側面に特化して学習されているという可能性がある．
---
Htut et al. (2019)及びClark et al. (2019)によれば，BERTのSelf-Attention Headsはランダムなベースラインに比べて単語の文法的な位置に対して敏感である．両研究で用いられたデータセットは別々であるが，どちらの研究でもSelf-Attention Headsは単語が`obj`の位置にある場合に敏感に反応すると結論づけている．
---
Hoover et al. (2019)では，`dobj`などの複雑な文法的ロールは複数のSelf-Attention Headsの組み合わせで表現されているのではないかという仮説を立てたが，今のところ十分に検証されてはいない．
---
Htut et al. (2019)，Clark et al. (2019)は，どのSelf-Attention Headも単独では文法ツリーの完全な情報を保持していないと結論づけた．
---
ただし，Clark et al (2019)では，共参照解析に単独で分類器として用いることができるSelf-Attention Headが見つかっている．このタスク自体はルールベースのものであるが，多分に文法的な情報を必要とするタスクである．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Lin et al. (2019)" >}}
Yongjie Lin, Y. Tan, R. Frank. (2019)  
**Open Sesame: Getting inside BERT’s Linguistic Knowledge**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/165d51a547cd920e6ac55660ad5c404dcb9562ed)  
Influential Citation Count (18), SS-ID (165d51a547cd920e6ac55660ad5c404dcb9562ed)  

**ABSTRACT**  
How and to what extent does BERT encode syntactically-sensitive hierarchical information or positionally-sensitive linear information? Recent work has shown that contextual representations like BERT perform well on tasks that require sensitivity to linguistic structure. We present here two studies which aim to provide a better understanding of the nature of BERT’s representations. The first of these focuses on the identification of structurally-defined elements using diagnostic classifiers, while the second explores BERT’s representation of subject-verb agreement and anaphor-antecedent dependencies through a quantitative assessment of self-attention vectors. In both cases, we find that BERT encodes positional information about word tokens well on its lower layers, but switches to a hierarchically-oriented encoding on higher layers. We conclude then that BERT’s representations do indeed model linguistically relevant aspects of hierarchical structure, though they do not appear to show the sharp sensitivity to hierarchical structure that is found in human processing of reflexive anaphora.
{{< /ci-details>}}

{{< ci-details summary="Ettinger (2019)" >}}
Allyson Ettinger. (2019)  
**What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models**  
TACL  
[Paper Link](https://www.semanticscholar.org/paper/a0e49f65b6847437f262c59d0d399255101d0b75)  
Influential Citation Count (10), SS-ID (a0e49f65b6847437f262c59d0d399255101d0b75)  

**ABSTRACT**  
Pre-training by language modeling has become a popular and successful approach to NLP tasks, but we have yet to understand exactly what linguistic capacities these pre-training processes confer upon models. In this paper we introduce a suite of diagnostics drawn from human language experiments, which allow us to ask targeted questions about information used by language models for generating predictions in context. As a case study, we apply these diagnostics to the popular BERT model, finding that it can generally distinguish good from bad completions involving shared category or role reversal, albeit with less sensitivity than humans, and it robustly retrieves noun hypernyms, but it struggles with challenging inference and role-based event prediction— and, in particular, it shows clear insensitivity to the contextual impacts of negation.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
`Attention Weights` は主語-動詞の対応関係や照応の分析にはあまり役に立たない．
---
Lin et al. (2019)によれば，BERTのSelf-Attention Headsは強く関連づけられるべき単語同士の関係を深く学習するというよりは，それぞれの単語の関係を一様に学習するというべきである．ただし，心理言語学のデータを使った実験において，一部の文法的な誤りに対しては敏感に反応したことがある．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Correia et al. (2019)" >}}
Gonçalo M. Correia, Vlad Niculae, André F. T. Martins. (2019)  
**Adaptively Sparse Transformers**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/f6390beca54411b06f3bde424fb983a451789733)  
Influential Citation Count (18), SS-ID (f6390beca54411b06f3bde424fb983a451789733)  

**ABSTRACT**  
Attention mechanisms have become ubiquitous in NLP. Recent architectures, notably the Transformer, learn powerful context-aware word representations through layered, multi-headed attention. The multiple heads learn diverse types of word relationships. However, with standard softmax attention, all attention heads are dense, assigning a non-zero weight to all context words. In this work, we introduce the adaptively sparse Transformer, wherein attention heads have flexible, context-dependent sparsity patterns. This sparsity is accomplished by replacing softmax with alpha-entmax: a differentiable generalization of softmax that allows low-scoring words to receive precisely zero weight. Moreover, we derive a method to automatically learn the alpha parameter – which controls the shape and sparsity of alpha-entmax – allowing attention heads to choose between focused or spread-out behavior. Our adaptively sparse Transformer improves interpretability and head diversity when compared to softmax Transformers on machine translation datasets. Findings of the quantitative and qualitative analysis of our approach include that heads in different layers learn different sparsity preferences and tend to be more diverse in their attention distributions than softmax Transformers. Furthermore, at no cost in accuracy, sparsity in attention heads helps to uncover different head specializations.
{{< /ci-details>}}

{{< fa-arrow-right-list >}}
BERTのSelf-Attention Headsに関して形態論的な観点からの研究はまだ実施されていない．
---
Correia et al. (2019)では，Tarnsformerの一部のSelf-Attention HeadsがBPE（Byte-Pair Encoding）でトークン化された単語の特徴を取り込んでいるようだということが指摘されている．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Clark et al. (2019)" >}}
Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning. (2019)  
**What Does BERT Look at? An Analysis of BERT’s Attention**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/95a251513853c6032bdecebd4b74e15795662986)  
Influential Citation Count (74), SS-ID (95a251513853c6032bdecebd4b74e15795662986)  

**ABSTRACT**  
Large pre-trained neural networks such as BERT have had great recent success in NLP, motivating a growing body of research investigating what aspects of language they are able to learn from unlabeled data. Most recent analysis has focused on model outputs (e.g., language model surprisal) or internal vector representations (e.g., probing classifiers). Complementary to these works, we propose methods for analyzing the attention mechanisms of pre-trained models and apply them to BERT. BERT’s attention heads exhibit patterns such as attending to delimiter tokens, specific positional offsets, or broadly attending over the whole sentence, with heads in the same layer often exhibiting similar behaviors. We further show that certain attention heads correspond well to linguistic notions of syntax and coreference. For example, we find heads that attend to the direct objects of verbs, determiners of nouns, objects of prepositions, and coreferent mentions with remarkably high accuracy. Lastly, we propose an attention-based probing classifier and use it to further demonstrate that substantial syntactic information is captured in BERT’s attention.
{{< /ci-details>}}

{{< ci-details summary="Jain and Wallance (2019)" >}}
Sarthak Jain, Byron C. Wallace. (2019)  
**Attention is not Explanation**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/1e83c20def5c84efa6d4a0d80aa3159f55cb9c3f)  
Influential Citation Count (36), SS-ID (1e83c20def5c84efa6d4a0d80aa3159f55cb9c3f)  

**ABSTRACT**  
Attention mechanisms have seen wide adoption in neural NLP models. In addition to improving predictive performance, these are often touted as affording transparency: models equipped with attention provide a distribution over attended-to input units, and this is often presented (at least implicitly) as communicating the relative importance of inputs. However, it is unclear what relationship exists between attention weights and model outputs. In this work we perform extensive experiments across a variety of NLP tasks that aim to assess the degree to which attention weights provide meaningful “explanations” for predictions. We find that they largely do not. For example, learned attention weights are frequently uncorrelated with gradient-based measures of feature importance, and one can identify very different attention distributions that nonetheless yield equivalent predictions. Our findings show that standard attention modules do not provide meaningful explanations and should not be treated as though they do.
{{< /ci-details>}}

{{< ci-details summary="Serrano and Smith (2019)" >}}
Sofia Serrano, Noah A. Smith. (2019)  
**Is Attention Interpretable?**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/135112c7ba1762d65f39b1a61777f26ae4dfd8ad)  
Influential Citation Count (16), SS-ID (135112c7ba1762d65f39b1a61777f26ae4dfd8ad)  

**ABSTRACT**  
Attention mechanisms have recently boosted performance on a range of NLP tasks. Because attention layers explicitly weight input components’ representations, it is also often assumed that attention can be used to identify information that models found important (e.g., specific contextualized word tokens). We test whether that assumption holds by manipulating attention weights in already-trained text classification models and analyzing the resulting differences in their predictions. While we observe some ways in which higher attention weights correlate with greater impact on model predictions, we also find many ways in which this does not hold, i.e., where gradient-based rankings of attention weights better predict their effects than their magnitudes. We conclude that while attention noisily predicts input components’ overall importance to a model, it is by no means a fail-safe indicator.1
{{< /ci-details>}}

{{< ci-details summary="Wiegreffe and Pinter (2019)" >}}
Sarah Wiegreffe, Yuval Pinter. (2019)  
**Attention is not not Explanation**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/ce177672b00ddf46e4906157a7e997ca9338b8b9)  
Influential Citation Count (12), SS-ID (ce177672b00ddf46e4906157a7e997ca9338b8b9)  

**ABSTRACT**  
Attention mechanisms play a central role in NLP systems, especially within recurrent neural network (RNN) models. Recently, there has been increasing interest in whether or not the intermediate representations offered by these modules may be used to explain the reasoning for a model’s prediction, and consequently reach insights regarding the model’s decision-making process. A recent paper claims that ‘Attention is not Explanation’ (Jain and Wallace, 2019). We challenge many of the assumptions underlying this work, arguing that such a claim depends on one’s definition of explanation, and that testing it needs to take into account all elements of the model. We propose four alternative tests to determine when/whether attention can be used as explanation: a simple uniform-weights baseline; a variance calibration based on multiple random seed runs; a diagnostic framework using frozen weights from pretrained models; and an end-to-end adversarial attention training protocol. Each allows for meaningful interpretation of attention mechanisms in RNN models. We show that even when reliable adversarial distributions can be found, they don’t perform well on the simple diagnostic, indicating that prior work does not disprove the usefulness of attention mechanisms for explainability.
{{< /ci-details>}}

{{< ci-details summary="Brunner et al. (2020)" >}}
Gino Brunner, Yang Liu, Damian Pascual, Oliver Richter, Massimiliano Ciaramita, Roger Wattenhofer. (2019)  
**On Identifiability in Transformers**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/9d7fbdb2e9817a6396992a1c92f75206689852d9)  
Influential Citation Count (13), SS-ID (9d7fbdb2e9817a6396992a1c92f75206689852d9)  

**ABSTRACT**  
In this paper we delve deep in the Transformer architecture by investigating two of its core components: self-attention and contextual embeddings. In particular, we study the identifiability of attention weights and token embeddings, and the aggregation of context into hidden tokens. We show that, for sequences longer than the attention head dimension, attention weights are not identifiable. We propose effective attention as a complementary tool for improving explanatory interpretations based on attention. Furthermore, we show that input tokens retain to a large degree their identity across the model. We also find evidence suggesting that identity information is mainly encoded in the angle of the embeddings and gradually decreases with depth. Finally, we demonstrate strong mixing of input information in the generation of contextual embeddings by means of a novel quantification method based on gradient attribution. Overall, we show that self-attention distributions are not directly interpretable and present tools to better understand and further investigate Transformer models.
{{< /ci-details>}}

{{< fa-arrow-right-list >}}
現在のSelf-Attention Headsへの注目度の高さは，「Attention Weightsにはある単語が別の単語に関連してどの程度重みづけられるか，ということが自動的に学習されるため，重みが持つ意味がクリアである」というアイディアによっている．
---
この点は現在活発に議論されており，今のところAttentionレイヤが非線形の活性化関数によって重ねられているような複数レイヤのモデルにおいて，各レイヤが単独で全ての情報を保持するということはない，ということが明らかになっている．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Vig (2019)" >}}
Jesse Vig, Y. Belinkov. (2019)  
**Analyzing the Structure of Attention in a Transformer Language Model**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/a039ea239e37f53a2cb60c68e0a1967994353166)  
Influential Citation Count (8), SS-ID (a039ea239e37f53a2cb60c68e0a1967994353166)  

**ABSTRACT**  
The Transformer is a fully attention-based alternative to recurrent networks that has achieved state-of-the-art results across a range of NLP tasks. In this paper, we analyze the structure of attention in a Transformer language model, the GPT-2 small pretrained model. We visualize attention for individual instances and analyze the interaction between attention and syntax over a large corpus. We find that attention targets different parts of speech at different layer depths within the model, and that attention aligns with dependency relations most strongly in the middle layers. We also find that the deepest layers of the model capture the most distant relationships. Finally, we extract exemplar sentences that reveal highly specific patterns targeted by particular attention heads.
{{< /ci-details >}}

{{< ci-details summary="Hoover et al. (2019)" >}}
Benjamin Hoover, Hendrik Strobelt, Sebastian Gehrmann. (2019)  
**exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformer Models**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/327d7e55d64cb34d55bd3a3fe58233c238a312cd)  
Influential Citation Count (4), SS-ID (327d7e55d64cb34d55bd3a3fe58233c238a312cd)  

**ABSTRACT**  
Large Transformer-based language models can route and reshape complex information via their multi-headed attention mechanism. Although the attention never receives explicit supervision, it can exhibit recognizable patterns following linguistic or positional information. Analyzing the learned representations and attentions is paramount to furthering our understanding of the inner workings of these models. However, analyses have to catch up with the rapid release of new models and the growing diversity of investigation techniques. To support analysis for a wide variety of models, we introduce exBERT, a tool to help humans conduct flexible, interactive investigations and formulate hypotheses for the model-internal reasoning process. exBERT provides insights into the meaning of the contextual representations and attention by matching a human-specified input to similar contexts in large annotated datasets. By aggregating the annotations of the matched contexts, exBERT can quickly replicate findings from literature and extend them to previously not analyzed models.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
Self-Attention Headsの可視化に関しても，様々なツールが開発されている．
---
例えば，Vig (2019)，Hoover et al. (2019)など．
{{< /fa-arrow-right-list >}}

---

#### Attention to Special Tokens

---

{{< ci-details summary="Kovaleva et al. (2019)" >}}
Olga Kovaleva, Alexey Romanov, Anna Rogers, Anna Rumshisky. (2019)  
**Revealing the Dark Secrets of BERT**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d78aed1dac6656affa4a04cbf225ced11a83d103)  
Influential Citation Count (34), SS-ID (d78aed1dac6656affa4a04cbf225ced11a83d103)  

**ABSTRACT**  
BERT-based architectures currently give state-of-the-art performance on many NLP tasks, but little is known about the exact mechanisms that contribute to its success. In the current work, we focus on the interpretation of self-attention, which is one of the fundamental underlying components of BERT. Using a subset of GLUE tasks and a set of handcrafted features-of-interest, we propose the methodology and carry out a qualitative and quantitative analysis of the information encoded by the individual BERT’s heads. Our findings suggest that there is a limited set of attention patterns that are repeated across different heads, indicating the overall model overparametrization. While different heads consistently use the same attention patterns, they have varying impact on performance across different tasks. We show that manually disabling attention in certain heads leads to a performance improvement over the regular fine-tuned BERT models.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
一つのモデルの中で言語情報を直接エンコードされているSelf-Attention Headsはほとんどない（少なくともGLUEでFine-tuningした場合においては）．なぜならば，Self-Attention Headsのうち `heterogeneous` の[パターン](#self-attention-heads)を示したHeadは50%にも満たなかったからである．
---
なお，多くのHeadは `vertical` パターンを示した．
---
この冗長性の問題はモデルの過学習と関連しているのではないかと考えられている．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Kobayashi et al. (2020) " >}}
Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, Kentaro Inui. (2020)  
**Attention Module is Not Only a Weight: Analyzing Transformers with Vector Norms**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/2a8e42995caaedadc9dc739d85bed2c57fc78568)  
Influential Citation Count (0), SS-ID (2a8e42995caaedadc9dc739d85bed2c57fc78568)  

**ABSTRACT**  
Attention is a key component of Transformers, which have recently achieved considerable success in natural language processing. Hence, attention is being extensively studied to investigate various linguistic capabilities of Transformers, focusing on analyzing the parallels between attention weights and specific linguistic phenomena. This paper shows that attention weights alone are only one of the two factors that determine the output of attention and proposes a norm-based analysis that incorporates the second factor, the norm of the transformed input vectors. The findings of our norm-based analyses of BERT and a Transformer-based neural machine translation system include the following: (i) contrary to previous studies, BERT pays poor attention to special tokens, and (ii) reasonable word alignment can be extracted from attention mechanisms of Transformer. These findings provide insights into the inner workings of Transformers.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
入力ベクトルに対してSelf-Attention Headで重み付けしたベクトルのノルムは直感的に理解しやすいアウトプットになるが，これらのノルムはspecial tokensに対してはあまり反応しなかった．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Lin et al. (2019)" >}}
Yongjie Lin, Y. Tan, R. Frank. (2019)  
**Open Sesame: Getting inside BERT’s Linguistic Knowledge**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/165d51a547cd920e6ac55660ad5c404dcb9562ed)  
Influential Citation Count (18), SS-ID (165d51a547cd920e6ac55660ad5c404dcb9562ed)  

**ABSTRACT**  
How and to what extent does BERT encode syntactically-sensitive hierarchical information or positionally-sensitive linear information? Recent work has shown that contextual representations like BERT perform well on tasks that require sensitivity to linguistic structure. We present here two studies which aim to provide a better understanding of the nature of BERT’s representations. The first of these focuses on the identification of structurally-defined elements using diagnostic classifiers, while the second explores BERT’s representation of subject-verb agreement and anaphor-antecedent dependencies through a quantitative assessment of self-attention vectors. In both cases, we find that BERT encodes positional information about word tokens well on its lower layers, but switches to a hierarchically-oriented encoding on higher layers. We conclude then that BERT’s representations do indeed model linguistically relevant aspects of hierarchical structure, though they do not appear to show the sharp sensitivity to hierarchical structure that is found in human processing of reflexive anaphora.
{{< /ci-details >}}

{{< ci-details summary="Htut et al. (2019)" >}}
Phu Mon Htut, Jason Phang, Shikha Bordia, Samuel R. Bowman. (2019)  
**Do Attention Heads in BERT Track Syntactic Dependencies?**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/ba8215e77f35b0d947c7cec39c45df4516e93421)  
Influential Citation Count (12), SS-ID (ba8215e77f35b0d947c7cec39c45df4516e93421)  

**ABSTRACT**  
We investigate the extent to which individual attention heads in pretrained transformer language models, such as BERT and RoBERTa, implicitly capture syntactic dependency relations. We employ two methods---taking the maximum attention weight and computing the maximum spanning tree---to extract implicit dependency relations from the attention weights of each layer/head, and compare them to the ground-truth Universal Dependency (UD) trees. We show that, for some UD relation types, there exist heads that can recover the dependency type significantly better than baselines on parsed English text, suggesting that some self-attention heads act as a proxy for syntactic structure. We also analyze BERT fine-tuned on two datasets---the syntax-oriented CoLA and the semantics-oriented MNLI---to investigate whether fine-tuning affects the patterns of their self-attention, but we do not observe substantial differences in the overall dependency relations extracted using our methods. Our results suggest that these models have some specialist attention heads that track individual dependency types, but no generalist head that performs holistic parsing significantly better than a trivial baseline, and that analyzing attention weights directly may not reveal much of the syntactic knowledge that BERT-style models are known to learn.

{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERT系のモデルの実装においてよく採用される方法は，単語間のAttentionを重視し，`special tokens`は除外する，というものである．
---
しかし，タスクの推論が，実は`special tokens`に強く依存するものだった場合，単語間のAttentionのみによって得られた推論結果は精度を保証されない可能性が高くなる．
---
実のところ，`special tokens`の機能については，まだあまり理解が進んでいない．`[CLS]`は`special token`のひとつかもしれないが，BERTの学習においては各Headの情報がこの`[CLS]`というトークンに集約されるため，このトークンには`special token`以外の言語情報も多く含まれていると考えられる．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Clark et al. (2019)" >}}
Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning. (2019)  
**What Does BERT Look at? An Analysis of BERT’s Attention**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/95a251513853c6032bdecebd4b74e15795662986)  
Influential Citation Count (74), SS-ID (95a251513853c6032bdecebd4b74e15795662986)  

**ABSTRACT**  
Large pre-trained neural networks such as BERT have had great recent success in NLP, motivating a growing body of research investigating what aspects of language they are able to learn from unlabeled data. Most recent analysis has focused on model outputs (e.g., language model surprisal) or internal vector representations (e.g., probing classifiers). Complementary to these works, we propose methods for analyzing the attention mechanisms of pre-trained models and apply them to BERT. BERT’s attention heads exhibit patterns such as attending to delimiter tokens, specific positional offsets, or broadly attending over the whole sentence, with heads in the same layer often exhibiting similar behaviors. We further show that certain attention heads correspond well to linguistic notions of syntax and coreference. For example, we find heads that attend to the direct objects of verbs, determiners of nouns, objects of prepositions, and coreferent mentions with remarkably high accuracy. Lastly, we propose an attention-based probing classifier and use it to further demonstrate that substantial syntactic information is captured in BERT’s attention.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
Clark et al. (2019)はWikipediaの文章を使ってBERTがどの程度spetial tokensを重視しているが実験を実施した．
---
実験結果によれば，入力に近いレイヤでは`[CLS]`が，中間レイヤでは`[SEP]`が，出力に近いレイヤではピリオドとカンマに高い重みが割り当てられていることがわかった．
---
Clark et al. (2019)では，これらのspecial tokensはある種の`no-op`的な機能を持っているのではないかという仮説を立てている．つまり，各ヘッドが今扱っているケースに対して適用可能かどうかを，これらのspecial tokensを使って判断しているのではないかということである．
---
興味深いことに，BERTは句読点（punctuation）をかなり重要視している．句読点はspecial tokensと並んで頻繁に出現するものであるため，上記と同じような扱われ方をしているのではないかと考えられている．
{{< /fa-arrow-right-list >}}

---

### BERT Layers

BERTの最初のレイヤはトークン，セグメント，単語の位置情報のEmbeddingを入力として受け取る．

---

{{< ci-details summary="Lin et al. (2019)" >}}
Yongjie Lin, Y. Tan, R. Frank. (2019)  
**Open Sesame: Getting inside BERT’s Linguistic Knowledge**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/165d51a547cd920e6ac55660ad5c404dcb9562ed)  
Influential Citation Count (18), SS-ID (165d51a547cd920e6ac55660ad5c404dcb9562ed)  

**ABSTRACT**  
How and to what extent does BERT encode syntactically-sensitive hierarchical information or positionally-sensitive linear information? Recent work has shown that contextual representations like BERT perform well on tasks that require sensitivity to linguistic structure. We present here two studies which aim to provide a better understanding of the nature of BERT’s representations. The first of these focuses on the identification of structurally-defined elements using diagnostic classifiers, while the second explores BERT’s representation of subject-verb agreement and anaphor-antecedent dependencies through a quantitative assessment of self-attention vectors. In both cases, we find that BERT encodes positional information about word tokens well on its lower layers, but switches to a hierarchically-oriented encoding on higher layers. We conclude then that BERT’s representations do indeed model linguistically relevant aspects of hierarchical structure, though they do not appear to show the sharp sensitivity to hierarchical structure that is found in human processing of reflexive anaphora.
{{</ ci-details >}}

{{< fa-arrow-right-list >}}
最初の方のレイヤは単語の語順に関して最も情報を保持している．
---
Lin et al. (2019)によれば，語順に関する情報はBERT-baseモデルでは4番目のレイヤあたりで減衰し始める．それに伴って文章の階層的な構造に関する情報が増えていく．これらはトークンのインデックスや助動詞，文章の主語を予測するタスクから明らかになっている．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Hewitt and Manning (2019)" >}}
John Hewitt, Christopher D. Manning. (2019)  
**A Structural Probe for Finding Syntax in Word Representations**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/455a8838cde44f288d456d01c76ede95b56dc675)  
Influential Citation Count (30), SS-ID (455a8838cde44f288d456d01c76ede95b56dc675)  

**ABSTRACT**  
Recent work has improved our ability to detect linguistic knowledge in word representations. However, current methods for detecting syntactic knowledge do not test whether syntax trees are represented in their entirety. In this work, we propose a structural probe, which evaluates whether syntax trees are embedded in a linear transformation of a neural network’s word representation space. The probe identifies a linear transformation under which squared L2 distance encodes the distance between words in the parse tree, and one in which squared L2 norm encodes depth in the parse tree. Using our probe, we show that such transformations exist for both ELMo and BERT but not in baselines, providing evidence that entire syntax trees are embedded implicitly in deep models’ vector geometry.
{{< /ci-details >}}

{{< ci-details summary="Goldberg (2019)" >}}
Yoav Goldberg. (2019)  
**Assessing BERT's Syntactic Abilities**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/efeab0dcdb4c1cce5e537e57745d84774be99b9a)  
Influential Citation Count (17), SS-ID (efeab0dcdb4c1cce5e537e57745d84774be99b9a)  

**ABSTRACT**  
I assess the extent to which the recently introduced BERT model captures English syntactic phenomena, using (1) naturally-occurring subject-verb agreement stimuli; (2) "coloreless green ideas" subject-verb agreement stimuli, in which content words in natural sentences are randomly replaced with words sharing the same part-of-speech and inflection; and (3) manually crafted stimuli for subject-verb agreement and reflexive anaphora phenomena. The BERT model performs remarkably well on all cases.
{{< /ci-details >}}

{{< ci-details summary="Jawahar et al. (2019)" >}}
Ganesh Jawahar, Benoît Sagot, Djamé Seddah. (2019)  
**What Does BERT Learn about the Structure of Language?**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/335613303ebc5eac98de757ed02a56377d99e03a)  
Influential Citation Count (44), SS-ID (335613303ebc5eac98de757ed02a56377d99e03a)  

**ABSTRACT**  
BERT is a recent language representation model that has surprisingly performed well in diverse language understanding benchmarks. This result indicates the possibility that BERT networks capture structural information about language. In this work, we provide novel support for this claim by performing a series of experiments to unpack the elements of English language structure learned by BERT. Our findings are fourfold. BERT’s phrasal representation captures the phrase-level information in the lower layers. The intermediate layers of BERT compose a rich hierarchy of linguistic information, starting with surface features at the bottom, syntactic features in the middle followed by semantic features at the top. BERT requires deeper layers while tracking subject-verb agreement to handle long-term dependency problem. Finally, the compositional scheme underlying BERT mimics classical, tree-like structures.
{{< /ci-details >}}

{{< ci-details summary="Liu et al (2019)" >}}
Nelson F. Liu, Matt Gardner, Y. Belinkov, Matthew E. Peters, Noah A. Smith. (2019)  
**Linguistic Knowledge and Transferability of Contextual Representations**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  
Influential Citation Count (108), SS-ID (f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  

**ABSTRACT**  
Contextual word representations derived from large-scale neural language models are successful across a diverse set of NLP tasks, suggesting that they encode useful and transferable features of language. To shed light on the linguistic knowledge they capture, we study the representations produced by several recent pretrained contextualizers (variants of ELMo, the OpenAI transformer language model, and BERT) with a suite of sixteen diverse probing tasks. We find that linear models trained on top of frozen contextual representations are competitive with state-of-the-art task-specific models in many cases, but fail on tasks requiring fine-grained linguistic knowledge (e.g., conjunct identification). To investigate the transferability of contextual word representations, we quantify differences in the transferability of individual layers within contextualizers, especially between recurrent neural networks (RNNs) and transformers. For instance, higher layers of RNNs are more task-specific, while transformer layers do not exhibit the same monotonic trend. In addition, to better understand what makes contextual word representations transferable, we compare language model pretraining with eleven supervised pretraining tasks. For any given task, pretraining on a closely related task yields better performance than language model pretraining (which is better on average) when the pretraining dataset is fixed. However, language model pretraining on more data gives the best results.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
文法に関する情報がBERTの中間レイヤで最も顕著に見られるということは，多くの研究が明らかにしているところである．
---
Hewitt and Manning (2019)ではBERTの中間レイヤから文法ツリーを再構築することに最も成功した研究である（BERT-base: 6-9，BERT-large: 14-19）．
---
Goldberg (2019)は主語と動詞の対応関係が8-9レイヤ付近で最も顕著に捉えられているということを報告している．
---
Jawahar et al. (2019)においても同様に文法に関するタスクでモデルの中間レイヤを使用することで最も精度が高くなることがわかっている．
---
BERTの中間層において文法的な情報が顕著に見られるという事実と関連する研究として，Liu et al. (2019)ではTransformerの中間レイヤが最も他のタスクに転用しやすいレイヤであるということが発見された．
{{< /fa-arrow-right-list >}}

<img src="fig-4.png" />

---

上記の主張とは矛盾する研究もある．

{{< ci-details summary="Tenney et al. (2019)" >}}
Ian Tenney, Dipanjan Das, Ellie Pavlick. (2019)  
**BERT Rediscovers the Classical NLP Pipeline**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/97906df07855b029b7aae7c2a1c6c5e8df1d531c)  
Influential Citation Count (59), SS-ID (97906df07855b029b7aae7c2a1c6c5e8df1d531c)  

**ABSTRACT**  
Pre-trained text encoders have rapidly advanced the state of the art on many NLP tasks. We focus on one such model, BERT, and aim to quantify where linguistic information is captured within the network. We find that the model represents the steps of the traditional NLP pipeline in an interpretable and localizable way, and that the regions responsible for each step appear in the expected sequence: POS tagging, parsing, NER, semantic roles, then coreference. Qualitative analysis reveals that the model can and often does adjust this pipeline dynamically, revising lower-level decisions on the basis of disambiguating information from higher-level representations.
{{< /ci-details >}}

{{< ci-details summary="Jawahar et al. (2019)" >}}
Ganesh Jawahar, Benoît Sagot, Djamé Seddah. (2019)  
**What Does BERT Learn about the Structure of Language?**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/335613303ebc5eac98de757ed02a56377d99e03a)  
Influential Citation Count (44), SS-ID (335613303ebc5eac98de757ed02a56377d99e03a)  

**ABSTRACT**  
BERT is a recent language representation model that has surprisingly performed well in diverse language understanding benchmarks. This result indicates the possibility that BERT networks capture structural information about language. In this work, we provide novel support for this claim by performing a series of experiments to unpack the elements of English language structure learned by BERT. Our findings are fourfold. BERT’s phrasal representation captures the phrase-level information in the lower layers. The intermediate layers of BERT compose a rich hierarchy of linguistic information, starting with surface features at the bottom, syntactic features in the middle followed by semantic features at the top. BERT requires deeper layers while tracking subject-verb agreement to handle long-term dependency problem. Finally, the compositional scheme underlying BERT mimics classical, tree-like structures.
{{< /ci-details >}}

{{< ci-details summary="Liu et al (2019)" >}}
Nelson F. Liu, Matt Gardner, Y. Belinkov, Matthew E. Peters, Noah A. Smith. (2019)  
**Linguistic Knowledge and Transferability of Contextual Representations**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  
Influential Citation Count (108), SS-ID (f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  

**ABSTRACT**  
Contextual word representations derived from large-scale neural language models are successful across a diverse set of NLP tasks, suggesting that they encode useful and transferable features of language. To shed light on the linguistic knowledge they capture, we study the representations produced by several recent pretrained contextualizers (variants of ELMo, the OpenAI transformer language model, and BERT) with a suite of sixteen diverse probing tasks. We find that linear models trained on top of frozen contextual representations are competitive with state-of-the-art task-specific models in many cases, but fail on tasks requiring fine-grained linguistic knowledge (e.g., conjunct identification). To investigate the transferability of contextual word representations, we quantify differences in the transferability of individual layers within contextualizers, especially between recurrent neural networks (RNNs) and transformers. For instance, higher layers of RNNs are more task-specific, while transformer layers do not exhibit the same monotonic trend. In addition, to better understand what makes contextual word representations transferable, we compare language model pretraining with eleven supervised pretraining tasks. For any given task, pretraining on a closely related task yields better performance than language model pretraining (which is better on average) when the pretraining dataset is fixed. However, language model pretraining on more data gives the best results.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
Tenney et al. (2019)では，BERTの前半のレイヤでは基本的な文法に関する情報が保持され，後半のレイヤになるほどハイレベルな意味の特徴を捉える傾向が見られると結論づけられている．
---
Jawahar et al. (2019)においても，モデルの最初の方のレイヤはchunkingなどの処理に，中間レイヤはパースなどの処理に有用であると報告されている．
---
一方，Liu et al. (2019)では，POS-taggingやchunkingなどのタスクは中間層を用いることで最も精度が良くなると報告されている．
---
このように研究によって結論がばらついているが，これらの研究は同じデータセット，パラメータで実験されているわけではないため，単純に横並びで比較することはできない．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Liu et al (2019)" >}}
Nelson F. Liu, Matt Gardner, Y. Belinkov, Matthew E. Peters, Noah A. Smith. (2019)  
**Linguistic Knowledge and Transferability of Contextual Representations**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  
Influential Citation Count (108), SS-ID (f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  

**ABSTRACT**  
Contextual word representations derived from large-scale neural language models are successful across a diverse set of NLP tasks, suggesting that they encode useful and transferable features of language. To shed light on the linguistic knowledge they capture, we study the representations produced by several recent pretrained contextualizers (variants of ELMo, the OpenAI transformer language model, and BERT) with a suite of sixteen diverse probing tasks. We find that linear models trained on top of frozen contextual representations are competitive with state-of-the-art task-specific models in many cases, but fail on tasks requiring fine-grained linguistic knowledge (e.g., conjunct identification). To investigate the transferability of contextual word representations, we quantify differences in the transferability of individual layers within contextualizers, especially between recurrent neural networks (RNNs) and transformers. For instance, higher layers of RNNs are more task-specific, while transformer layers do not exhibit the same monotonic trend. In addition, to better understand what makes contextual word representations transferable, we compare language model pretraining with eleven supervised pretraining tasks. For any given task, pretraining on a closely related task yields better performance than language model pretraining (which is better on average) when the pretraining dataset is fixed. However, language model pretraining on more data gives the best results.
{{< /ci-details >}}

{{< ci-details summary="Kovaleva et al. (2019)" >}}
Olga Kovaleva, Alexey Romanov, Anna Rogers, Anna Rumshisky. (2019)  
**Revealing the Dark Secrets of BERT**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d78aed1dac6656affa4a04cbf225ced11a83d103)  
Influential Citation Count (34), SS-ID (d78aed1dac6656affa4a04cbf225ced11a83d103)  

**ABSTRACT**  
BERT-based architectures currently give state-of-the-art performance on many NLP tasks, but little is known about the exact mechanisms that contribute to its success. In the current work, we focus on the interpretation of self-attention, which is one of the fundamental underlying components of BERT. Using a subset of GLUE tasks and a set of handcrafted features-of-interest, we propose the methodology and carry out a qualitative and quantitative analysis of the information encoded by the individual BERT’s heads. Our findings suggest that there is a limited set of attention patterns that are repeated across different heads, indicating the overall model overparametrization. While different heads consistently use the same attention patterns, they have varying impact on performance across different tasks. We show that manually disabling attention in certain heads leads to a performance improvement over the regular fine-tuned BERT models.
{{< /ci-details >}}

{{< ci-details summary="Hao et al. (2019)" >}}
Y. Hao, Li Dong, Furu Wei, Ke Xu. (2019)  
**Visualizing and Understanding the Effectiveness of BERT**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d3cacb4806886eb2fe59c90d4b6f822c24ff1822)  
Influential Citation Count (3), SS-ID (d3cacb4806886eb2fe59c90d4b6f822c24ff1822)  

**ABSTRACT**  
Language model pre-training, such as BERT, has achieved remarkable results in many NLP tasks. However, it is unclear why the pre-training-then-fine-tuning paradigm can improve performance and generalization capability across different tasks. In this paper, we propose to visualize loss landscapes and optimization trajectories of fine-tuning BERT on specific datasets. First, we find that pre-training reaches a good initial point across downstream tasks, which leads to wider optima and easier optimization compared with training from scratch. We also demonstrate that the fine-tuning procedure is robust to overfitting, even though BERT is highly over-parameterized for downstream tasks. Second, the visualization results indicate that fine-tuning BERT tends to generalize better because of the flat and wide optima, and the consistency between the training loss surface and the generalization error surface. Third, the lower layers of BERT are more invariant during fine-tuning, which suggests that the layers that are close to input learn more transferable representations of language.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTの最終層は後続のタスクに特化している．
---
これは，Liu et al. (2019)で報告されているように，モデルの中間層が最も他のタスクに転用しやすくなっているという事実とも付合する．
---
Kovaleva et al. (2019)では，Fine-Tuningにおいてモデルの最終層のパラメータが最も大きく更新されると指摘されているが，こちらも同様に上記の事実を示唆するものである．
---
同様に，Hao et al. (2019)ではFine-TuningされたBERTモデルに対して低レイヤのパラメータをオリジナルのモデルで上書きしたとしてもタスクの精度に大きな影響は与えないということが報告されている．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Tenney et al. (2019)" >}}
Ian Tenney, Dipanjan Das, Ellie Pavlick. (2019)  
**BERT Rediscovers the Classical NLP Pipeline**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/97906df07855b029b7aae7c2a1c6c5e8df1d531c)  
Influential Citation Count (59), SS-ID (97906df07855b029b7aae7c2a1c6c5e8df1d531c)  

**ABSTRACT**  
Pre-trained text encoders have rapidly advanced the state of the art on many NLP tasks. We focus on one such model, BERT, and aim to quantify where linguistic information is captured within the network. We find that the model represents the steps of the traditional NLP pipeline in an interpretable and localizable way, and that the regions responsible for each step appear in the expected sequence: POS tagging, parsing, NER, semantic roles, then coreference. Qualitative analysis reveals that the model can and often does adjust this pipeline dynamically, revising lower-level decisions on the basis of disambiguating information from higher-level representations.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
Tenney et al. (2019)によれば，文法的な情報はモデルの前半のレイヤに集約されている一方で，意味的な情報はモデルのレイヤ全体に分散しているのではないかということが指摘されている．
---
このことは，難しいタスクがモデルの前半のレイヤではうまく回答できないのに対して，モデルの後半のレイヤを使った場合には正解する場合が多いという事実からも示唆される．
---
では，レイヤを重ねれば重ねるほど意味的な情報が蓄積されていくのか，という疑問が生じるが，Tenney et al. (2019)ではBERT-baseとBERT-largeで最終層を使用した場合のタスクのスコアを比較した結果，予想されるほど大きな違いは出ないということがわかった．
---
ただし，Tenney et al. (2019)は文章レベルのsemantic relationsに関する実験の結果である．
{{< /fa-arrow-right-list >}}

## Training BERT

### Model Arcitecture Choices

---

{{< ci-details summary="Wang et al (2019)" >}}
Karthikeyan K, Zihan Wang, Stephen Mayhew, D. Roth. (2019)  
**Cross-Lingual Ability of Multilingual BERT: An Empirical Study**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/3b2538f84812f434c740115c185be3e5e216c526)  
Influential Citation Count (7), SS-ID (3b2538f84812f434c740115c185be3e5e216c526)  

**ABSTRACT**  
Recent work has exhibited the surprising cross-lingual abilities of multilingual BERT (M-BERT) -- surprising since it is trained without any cross-lingual objective and with no aligned data. In this work, we provide a comprehensive study of the contribution of different components in M-BERT to its cross-lingual ability. We study the impact of linguistic properties of the languages, the architecture of the model, and the learning objectives. The experimental study is done in the context of three typologically different languages -- Spanish, Hindi, and Russian -- and using two conceptually different NLP tasks, textual entailment and named entity recognition. Among our key conclusions is the fact that the lexical overlap between languages plays a negligible role in the cross-lingual success, while the depth of the network is an integral part of it. All our models and implementations can be found on our project page: this http URL .
{{< /ci-details >}}

{{< ci-details summary="Voita et al. (2019)" >}}
Elena Voita, David Talbot, F. Moiseev, Rico Sennrich, Ivan Titov. (2019)  
**Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/07a64686ce8e43ac475a8d820a8a9f1d87989583)  
Influential Citation Count (49), SS-ID (07a64686ce8e43ac475a8d820a8a9f1d87989583)  

**ABSTRACT**  
Multi-head self-attention is a key component of the Transformer, a state-of-the-art architecture for neural machine translation. In this work we evaluate the contribution made by individual attention heads to the overall performance of the model and analyze the roles played by them in the encoder. We find that the most important and confident heads play consistent and often linguistically-interpretable roles. When pruning heads using a method based on stochastic gates and a differentiable relaxation of the L0 penalty, we observe that specialized heads are last to be pruned. Our novel pruning method removes the vast majority of heads without seriously affecting performance. For example, on the English-Russian WMT dataset, pruning 38 out of 48 encoder heads results in a drop of only 0.15 BLEU.
{{< /ci-details >}}

{{< ci-details summary="Michel et al. (2019)" >}}
Paul Michel, Omer Levy, Graham Neubig. (2019)  
**Are Sixteen Heads Really Better than One?**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/b03c7ff961822183bab66b2e594415e585d3fd09)  
Influential Citation Count (49), SS-ID (b03c7ff961822183bab66b2e594415e585d3fd09)  

**ABSTRACT**  
Attention is a powerful and ubiquitous mechanism for allowing neural models to focus on particular salient pieces of information by taking their weighted average when making predictions. In particular, multi-headed attention is a driving force behind many recent state-of-the-art NLP models such as Transformer-based MT models and BERT. These models apply multiple attention mechanisms in parallel, with each attention "head" potentially focusing on different parts of the input, which makes it possible to express sophisticated functions beyond the simple weighted average. In this paper we make the surprising observation that even if models have been trained using multiple heads, in practice, a large percentage of attention heads can be removed at test time without significantly impacting performance. In fact, some layers can even be reduced to a single head. We further examine greedy algorithms for pruning down models, and the potential speed, memory efficiency, and accuracy improvements obtainable therefrom. Finally, we analyze the results with respect to which parts of the model are more reliant on having multiple heads, and provide precursory evidence that training dynamics play a role in the gains provided by multi-head attention.
{{< /ci-details >}}

{{< ci-details summary="Liu et al. (2019)" >}}
Nelson F. Liu, Matt Gardner, Y. Belinkov, Matthew E. Peters, Noah A. Smith. (2019)  
**Linguistic Knowledge and Transferability of Contextual Representations**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  
Influential Citation Count (108), SS-ID (f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  

**ABSTRACT**  
Contextual word representations derived from large-scale neural language models are successful across a diverse set of NLP tasks, suggesting that they encode useful and transferable features of language. To shed light on the linguistic knowledge they capture, we study the representations produced by several recent pretrained contextualizers (variants of ELMo, the OpenAI transformer language model, and BERT) with a suite of sixteen diverse probing tasks. We find that linear models trained on top of frozen contextual representations are competitive with state-of-the-art task-specific models in many cases, but fail on tasks requiring fine-grained linguistic knowledge (e.g., conjunct identification). To investigate the transferability of contextual word representations, we quantify differences in the transferability of individual layers within contextualizers, especially between recurrent neural networks (RNNs) and transformers. For instance, higher layers of RNNs are more task-specific, while transformer layers do not exhibit the same monotonic trend. In addition, to better understand what makes contextual word representations transferable, we compare language model pretraining with eleven supervised pretraining tasks. For any given task, pretraining on a closely related task yields better performance than language model pretraining (which is better on average) when the pretraining dataset is fixed. However, language model pretraining on more data gives the best results.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
論文執筆時点で，BERTのアーキテクチャに関して最もシステマチックな解析を実施したのはWang et al. (2019)である．Wang et al. (2019)では，レイヤの数，Self-Attention Heads，モデルのパラメータをそれぞれ一つずつ変化させて実験を実施した．その結果，Self-Attention Headsの数はレイヤの数に比べてモデルの精度にそれほど大きな影響を与えないという結論に達した．
---
上記の結論は，Voita et al. (2019)，Michel et al. (2019), Liu et al. (2019)，においても支持されている．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Liu et al. (2019)" >}}
Nelson F. Liu, Matt Gardner, Y. Belinkov, Matthew E. Peters, Noah A. Smith. (2019)  
**Linguistic Knowledge and Transferability of Contextual Representations**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  
Influential Citation Count (108), SS-ID (f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  

**ABSTRACT**  
Contextual word representations derived from large-scale neural language models are successful across a diverse set of NLP tasks, suggesting that they encode useful and transferable features of language. To shed light on the linguistic knowledge they capture, we study the representations produced by several recent pretrained contextualizers (variants of ELMo, the OpenAI transformer language model, and BERT) with a suite of sixteen diverse probing tasks. We find that linear models trained on top of frozen contextual representations are competitive with state-of-the-art task-specific models in many cases, but fail on tasks requiring fine-grained linguistic knowledge (e.g., conjunct identification). To investigate the transferability of contextual word representations, we quantify differences in the transferability of individual layers within contextualizers, especially between recurrent neural networks (RNNs) and transformers. For instance, higher layers of RNNs are more task-specific, while transformer layers do not exhibit the same monotonic trend. In addition, to better understand what makes contextual word representations transferable, we compare language model pretraining with eleven supervised pretraining tasks. For any given task, pretraining on a closely related task yields better performance than language model pretraining (which is better on average) when the pretraining dataset is fixed. However, language model pretraining on more data gives the best results.
{{< /ci-details >}}

{{< ci-details summary="Hao et al. (2019)" >}}
Y. Hao, Li Dong, Furu Wei, Ke Xu. (2019)  
**Visualizing and Understanding the Effectiveness of BERT**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d3cacb4806886eb2fe59c90d4b6f822c24ff1822)  
Influential Citation Count (3), SS-ID (d3cacb4806886eb2fe59c90d4b6f822c24ff1822)  

**ABSTRACT**  
Language model pre-training, such as BERT, has achieved remarkable results in many NLP tasks. However, it is unclear why the pre-training-then-fine-tuning paradigm can improve performance and generalization capability across different tasks. In this paper, we propose to visualize loss landscapes and optimization trajectories of fine-tuning BERT on specific datasets. First, we find that pre-training reaches a good initial point across downstream tasks, which leads to wider optima and easier optimization compared with training from scratch. We also demonstrate that the fine-tuning procedure is robust to overfitting, even though BERT is highly over-parameterized for downstream tasks. Second, the visualization results indicate that fine-tuning BERT tends to generalize better because of the flat and wide optima, and the consistency between the training loss surface and the generalization error surface. Third, the lower layers of BERT are more invariant during fine-tuning, which suggests that the layers that are close to input learn more transferable representations of language.
{{< /ci-details >}}

{{< ci-details summary="Brunner et al. (2020)" >}}
Gino Brunner, Yang Liu, Damian Pascual, Oliver Richter, Massimiliano Ciaramita, Roger Wattenhofer. (2019)  
**On Identifiability in Transformers**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/9d7fbdb2e9817a6396992a1c92f75206689852d9)  
Influential Citation Count (13), SS-ID (9d7fbdb2e9817a6396992a1c92f75206689852d9)  

**ABSTRACT**  
In this paper we delve deep in the Transformer architecture by investigating two of its core components: self-attention and contextual embeddings. In particular, we study the identifiability of attention weights and token embeddings, and the aggregation of context into hidden tokens. We show that, for sequences longer than the attention head dimension, attention weights are not identifiable. We propose effective attention as a complementary tool for improving explanatory interpretations based on attention. Furthermore, we show that input tokens retain to a large degree their identity across the model. We also find evidence suggesting that identity information is mainly encoded in the angle of the embeddings and gradually decreases with depth. Finally, we demonstrate strong mixing of input information in the generation of contextual embeddings by means of a novel quantification method based on gradient attribution. Overall, we show that self-attention distributions are not directly interpretable and present tools to better understand and further investigate Transformer models.
{{< /ci-details >}}

{{< ci-details summary="Kovaleva et al. (2019)" >}}
Olga Kovaleva, Alexey Romanov, Anna Rogers, Anna Rumshisky. (2019)  
**Revealing the Dark Secrets of BERT**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d78aed1dac6656affa4a04cbf225ced11a83d103)  
Influential Citation Count (34), SS-ID (d78aed1dac6656affa4a04cbf225ced11a83d103)  

**ABSTRACT**  
BERT-based architectures currently give state-of-the-art performance on many NLP tasks, but little is known about the exact mechanisms that contribute to its success. In the current work, we focus on the interpretation of self-attention, which is one of the fundamental underlying components of BERT. Using a subset of GLUE tasks and a set of handcrafted features-of-interest, we propose the methodology and carry out a qualitative and quantitative analysis of the information encoded by the individual BERT’s heads. Our findings suggest that there is a limited set of attention patterns that are repeated across different heads, indicating the overall model overparametrization. While different heads consistently use the same attention patterns, they have varying impact on performance across different tasks. We show that manually disabling attention in certain heads leads to a performance improvement over the regular fine-tuned BERT models.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
Self-Attention Headsの数とレイヤの数はそれぞれ別の機能として作用する．
---
モデルの深さは情報の流れに作用すると考えて間違いない．最終層に近いほどタスクに特化した特徴を持ち，入力層に近ければタスクに関する特徴ではなく入力されたトークンに関する情報をより多く保持するようになる．もしこの仮説が正しければ，層が深ければ深いほどモデル全体としてはタスクに特化しない情報を多く蓄えることができるようになる．
---
一方で，Self-Attention Headsは基本的にいずれのHeadも同じようなパターンを学習しているものと考えらえれる．例えば，Pruningによってモデルの精度に大きな影響が出ない理由はこのあたりにありそうである．
---
Self-Attention Headsにどの程度多様性を持たせることができるか，という点は研究に値する．今のままでは理論的には複数のレイヤで同じ情報をパラメータの重みとして保持してしまっていることになるからである．
{{< /fa-arrow-right-list >}}

---

{{< ci-details summary="Press et al. (2020)" >}}
Ofir Press, Noah A. Smith, Omer Levy. (2019)  
**Improving Transformer Models by Reordering their Sublayers**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/3ff8d265f4351e4b1fdac5b586466bee0b5d6fff)  
Influential Citation Count (2), SS-ID (3ff8d265f4351e4b1fdac5b586466bee0b5d6fff)  

**ABSTRACT**  
Multilayer transformer networks consist of interleaved self-attention and feedforward sublayers. Could ordering the sublayers in a different pattern lead to better performance? We generate randomly ordered transformers and train them with the language modeling objective. We observe that some of these models are able to achieve better performance than the interleaved baseline, and that those successful variants tend to have more self-attention at the bottom and more feedforward sublayers at the top. We propose a new transformer pattern that adheres to this property, the sandwich transformer, and show that it improves perplexity on multiple word-level and character-level language modeling benchmarks, at no cost in parameters, memory, or training time. However, the sandwich reordering pattern does not guarantee performance gains across every task, as we demonstrate on machine translation models. Instead, we suggest that further exploration of task-specific sublayer reorderings is needed in order to unlock additional gains.
{{< /ci-details >}}

{{< fa-arrow-right-list >}}
BERTはSelf-AttentionとFeed-Forwardレイヤに関して対称的でバランスが取れている（is symmetric and balanced）が，これは必ずしもそうである必要はない．
---
Press et al. (2020)によれば，Self-Attentionレイヤに関しては，最終層に近いレイヤから有益な情報が得られ，一方でFeed-Forwardレイヤに関しては入力層に近いレイヤから良い情報が得られることが多いという．
{{< /fa-arrow-right-list >}}

---

### Improvements to the Training Regime

### Pre-training BERT

### Fine-tuning BERT

## How Big Should BERT Be?

### Overparameterization

### Compression Techniques

### Pruning and Model Analysis

## Directions for Further Research

## References


{{< ci-details summary="Evaluating Commonsense in Pre-trained Language Models (Xuhui Zhou et al., 2019)">}}

Xuhui Zhou, Yue Zhang, Leyang Cui, Dandan Huang. (2019)  
**Evaluating Commonsense in Pre-trained Language Models**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/01f2b214962997260020279bd1fd1f8f372249d4)  
Influential Citation Count (5), SS-ID (01f2b214962997260020279bd1fd1f8f372249d4)  

**ABSTRACT**  
Contextualized representations trained over large raw text data have given remarkable improvements for NLP tasks including question answering and reading comprehension. There have been works showing that syntactic, semantic and word sense knowledge are contained in such representations, which explains why they benefit such tasks. However, relatively little work has been done investigating commonsense knowledge contained in contextualized representations, which is crucial for human question answering and reading comprehension. We study the commonsense ability of GPT, BERT, XLNet, and RoBERTa by testing them on seven challenging benchmarks, finding that language modeling and its variants are effective objectives for promoting models' commonsense ability while bi-directional context and larger training set are bonuses. We additionally find that current models do poorly on tasks require more necessary inference steps. Finally, we test the robustness of models by making dual test cases, which are correlated so that the correct prediction of one sample should lead to correct prediction of the other. Interestingly, the models show confusion on these test cases, which suggests that they learn commonsense at the surface rather than the deep level. We release a test set, named CATs publicly, for future research.

{{< /ci-details >}}

{{< ci-details summary="ERNIE: Enhanced Representation through Knowledge Integration (Yu Sun et al., 2019)">}}

Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, Hua Wu. (2019)  
**ERNIE: Enhanced Representation through Knowledge Integration**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/031e4e43aaffd7a479738dcea69a2d5be7957aa3)  
Influential Citation Count (63), SS-ID (031e4e43aaffd7a479738dcea69a2d5be7957aa3)  

**ABSTRACT**  
We present a novel language representation model enhanced by knowledge called ERNIE (Enhanced Representation through kNowledge IntEgration). Inspired by the masking strategy of BERT, ERNIE is designed to learn language representation enhanced by knowledge masking strategies, which includes entity-level masking and phrase-level masking. Entity-level strategy masks entities which are usually composed of multiple words.Phrase-level strategy masks the whole phrase which is composed of several words standing together as a conceptual unit.Experimental results show that ERNIE outperforms other baseline methods, achieving new state-of-the-art results on five Chinese natural language processing tasks including natural language inference, semantic similarity, named entity recognition, sentiment analysis and question answering. We also demonstrate that ERNIE has more powerful knowledge inference capacity on a cloze test.

{{< /ci-details >}}

{{< ci-details summary="Do NLP Models Know Numbers? Probing Numeracy in Embeddings (Eric Wallace et al., 2019)">}}

Eric Wallace, Yizhong Wang, Sujian Li, Sameer Singh, Matt Gardner. (2019)  
**Do NLP Models Know Numbers? Probing Numeracy in Embeddings**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/0427110f0e79f41e69a8eb00a3ec8868bac26a4f)  
Influential Citation Count (18), SS-ID (0427110f0e79f41e69a8eb00a3ec8868bac26a4f)  

**ABSTRACT**  
The ability to understand and work with numbers (numeracy) is critical for many complex reasoning tasks. Currently, most NLP models treat numbers in text in the same way as other tokens—they embed them as distributed vectors. Is this enough to capture numeracy? We begin by investigating the numerical reasoning capabilities of a state-of-the-art question answering model on the DROP dataset. We find this model excels on questions that require numerical reasoning, i.e., it already captures numeracy. To understand how this capability emerges, we probe token embedding methods (e.g., BERT, GloVe) on synthetic list maximum, number decoding, and addition tasks. A surprising degree of numeracy is naturally present in standard embeddings. For example, GloVe and word2vec accurately encode magnitude for numbers up to 1,000. Furthermore, character-level embeddings are even more precise—ELMo captures numeracy the best for all pre-trained methods—but BERT, which uses sub-word units, is less exact.

{{< /ci-details >}}

{{< ci-details summary="Emergent linguistic structure in artificial neural networks trained by self-supervision (Christopher D. Manning et al., 2020)">}}

Christopher D. Manning, Kevin Clark, John Hewitt, Urvashi Khandelwal, Omer Levy. (2020)  
**Emergent linguistic structure in artificial neural networks trained by self-supervision**  
Proceedings of the National Academy of Sciences  
[Paper Link](https://www.semanticscholar.org/paper/04ef54bd467d5e03dee7b0be601cf06d420bffa0)  
Influential Citation Count (5), SS-ID (04ef54bd467d5e03dee7b0be601cf06d420bffa0)  

**ABSTRACT**  
This paper explores the knowledge of linguistic structure learned by large artificial neural networks, trained via self-supervision, whereby the model simply tries to predict a masked word in a given context. Human language communication is via sequences of words, but language understanding requires constructing rich hierarchical structures that are never observed explicitly. The mechanisms for this have been a prime mystery of human language acquisition, while engineering work has mainly proceeded by supervised learning on treebanks of sentences hand labeled for this latent structure. However, we demonstrate that modern deep contextual language models learn major aspects of this structure, without any explicit supervision. We develop methods for identifying linguistic hierarchical structure emergent in artificial neural networks and demonstrate that components in these models focus on syntactic grammatical relationships and anaphoric coreference. Indeed, we show that a linear transformation of learned embeddings in these models captures parse tree distances to a surprising degree, allowing approximate reconstruction of the sentence tree structures normally assumed by linguists. These results help explain why these models have brought such large improvements across many language-understanding tasks.

{{< /ci-details >}}

{{< ci-details summary="RoBERTa: A Robustly Optimized BERT Pretraining Approach (Yinhan Liu et al., 2019)">}}

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, M. Lewis, Luke Zettlemoyer, Veselin Stoyanov. (2019)  
**RoBERTa: A Robustly Optimized BERT Pretraining Approach**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/077f8329a7b6fa3b7c877a57b81eb6c18b5f87de)  
Influential Citation Count (2015), SS-ID (077f8329a7b6fa3b7c877a57b81eb6c18b5f87de)  

**ABSTRACT**  
Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices, and raise questions about the source of recently reported improvements. We release our models and code.

{{< /ci-details >}}

{{< ci-details summary="Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned (Elena Voita et al., 2019)">}}

Elena Voita, David Talbot, F. Moiseev, Rico Sennrich, Ivan Titov. (2019)  
**Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/07a64686ce8e43ac475a8d820a8a9f1d87989583)  
Influential Citation Count (49), SS-ID (07a64686ce8e43ac475a8d820a8a9f1d87989583)  

**ABSTRACT**  
Multi-head self-attention is a key component of the Transformer, a state-of-the-art architecture for neural machine translation. In this work we evaluate the contribution made by individual attention heads to the overall performance of the model and analyze the roles played by them in the encoder. We find that the most important and confident heads play consistent and often linguistically-interpretable roles. When pruning heads using a method based on stochastic gates and a differentiable relaxation of the L0 penalty, we observe that specialized heads are last to be pruned. Our novel pruning method removes the vast majority of heads without seriously affecting performance. For example, on the English-Russian WMT dataset, pruning 38 out of 48 encoder heads results in a drop of only 0.15 BLEU.

{{< /ci-details >}}

{{< ci-details summary="Distilling the Knowledge in a Neural Network (Geoffrey E. Hinton et al., 2015)">}}

Geoffrey E. Hinton, Oriol Vinyals, J. Dean. (2015)  
**Distilling the Knowledge in a Neural Network**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/0c908739fbff75f03469d13d4a1a07de3414ee19)  
Influential Citation Count (1210), SS-ID (0c908739fbff75f03469d13d4a1a07de3414ee19)  

**ABSTRACT**  
A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets. Caruana and his collaborators have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.

{{< /ci-details >}}

{{< ci-details summary="TinyBERT: Distilling BERT for Natural Language Understanding (Xiaoqi Jiao et al., 2019)">}}

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, F. Wang, Qun Liu. (2019)  
**TinyBERT: Distilling BERT for Natural Language Understanding**  
FINDINGS  
[Paper Link](https://www.semanticscholar.org/paper/0cbf97173391b0430140117027edcaf1a37968c7)  
Influential Citation Count (131), SS-ID (0cbf97173391b0430140117027edcaf1a37968c7)  

**ABSTRACT**  
Language model pre-training, such as BERT, has significantly improved the performances of many natural language processing tasks. However, pre-trained language models are usually computationally expensive, so it is difficult to efficiently execute them on resource-restricted devices. To accelerate inference and reduce model size while maintaining accuracy, we first propose a novel Transformer distillation method that is specially designed for knowledge distillation (KD) of the Transformer-based models. By leveraging this new KD method, the plenty of knowledge encoded in a large “teacher” BERT can be effectively transferred to a small “student” TinyBERT. Then, we introduce a new two-stage learning framework for TinyBERT, which performs Transformer distillation at both the pre-training and task-specific learning stages. This framework ensures that TinyBERT can capture the general-domain as well as the task-specific knowledge in BERT. TinyBERT4 with 4 layers is empirically effective and achieves more than 96.8% the performance of its teacher BERT-Base on GLUE benchmark, while being 7.5x smaller and 9.4x faster on inference. TinyBERT4 is also significantly better than 4-layer state-of-the-art baselines on BERT distillation, with only ~28% parameters and ~31% inference time of them. Moreover, TinyBERT6 with 6 layers performs on-par with its teacher BERT-Base.

{{< /ci-details >}}

{{< ci-details summary="MPNet: Masked and Permuted Pre-training for Language Understanding (Kaitao Song et al., 2020)">}}

Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu. (2020)  
**MPNet: Masked and Permuted Pre-training for Language Understanding**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/0e002114cd379efaca0ec5cda6d262b5fe0be104)  
Influential Citation Count (16), SS-ID (0e002114cd379efaca0ec5cda6d262b5fe0be104)  

**ABSTRACT**  
BERT adopts masked language modeling (MLM) for pre-training and is one of the most successful pre-training models. Since BERT neglects dependency among predicted tokens, XLNet introduces permuted language modeling (PLM) for pre-training to address this problem. We argue that XLNet does not leverage the full position information of a sentence and thus suffers from position discrepancy between pre-training and fine-tuning. In this paper, we propose MPNet, a novel pre-training method that inherits the advantages of BERT and XLNet and avoids their limitations. MPNet leverages the dependency among predicted tokens through permuted language modeling (vs. MLM in BERT), and takes auxiliary position information as input to make the model see a full sentence and thus reducing the position discrepancy (vs. PLM in XLNet). We pre-train MPNet on a large-scale dataset (over 160GB text corpora) and fine-tune on a variety of down-streaming tasks (GLUE, SQuAD, etc). Experimental results show that MPNet outperforms MLM and PLM by a large margin, and achieves better results on these tasks compared with previous state-of-the-art pre-trained methods (e.g., BERT, XLNet, RoBERTa) under the same model setting. We release the code and pre-trained model in GitHub\footnote{\url{this https URL}}.

{{< /ci-details >}}

{{< ci-details summary="Can neural networks acquire a structural bias from raw linguistic data? (Alex Warstadt et al., 2020)">}}

Alex Warstadt, Samuel R. Bowman. (2020)  
**Can neural networks acquire a structural bias from raw linguistic data?**  
CogSci  
[Paper Link](https://www.semanticscholar.org/paper/0e012c2bd18236445cfbc6e3e409eb02df4691fe)  
Influential Citation Count (3), SS-ID (0e012c2bd18236445cfbc6e3e409eb02df4691fe)  

**ABSTRACT**  
We evaluate whether BERT, a widely used neural network for sentence processing, acquires an inductive bias towards forming structural generalizations through pretraining on raw data. We conduct four experiments testing its preference for structural vs. linear generalizations in different structure-dependent phenomena. We find that BERT makes a structural generalization in 3 out of 4 empirical domains---subject-auxiliary inversion, reflexive binding, and verb tense detection in embedded clauses---but makes a linear generalization when tested on NPI licensing. We argue that these results are the strongest evidence so far from artificial learners supporting the proposition that a structural bias can be acquired from raw data. If this conclusion is correct, it is tentative evidence that some linguistic universals can be acquired by learners without innate biases. However, the precise implications for human language acquisition are unclear, as humans learn language from significantly less data than BERT.

{{< /ci-details >}}

{{< ci-details summary="The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives (Elena Voita et al., 2019)">}}

Elena Voita, Rico Sennrich, Ivan Titov. (2019)  
**The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/112fd54ee193237b24f2ce7fce79e399609a29c5)  
Influential Citation Count (10), SS-ID (112fd54ee193237b24f2ce7fce79e399609a29c5)  

**ABSTRACT**  
We seek to understand how the representations of individual tokens and the structure of the learned feature space evolve between layers in deep neural networks under different learning objectives. We chose the Transformers for our analysis as they have been shown effective with various tasks, including machine translation (MT), standard left-to-right language models (LM) and masked language modeling (MLM). Previous work used black-box probing tasks to show that the representations learned by the Transformer differ significantly depending on the objective. In this work, we use canonical correlation analysis and mutual information estimators to study how information flows across Transformer layers and observe that the choice of the objective determines this process. For example, as you go from bottom to top layers, information about the past in left-to-right language models gets vanished and predictions about the future get formed. In contrast, for MLM, representations initially acquire information about the context around the token, partially forgetting the token identity and producing a more generalized token representation. The token identity then gets recreated at the top MLM layers.

{{< /ci-details >}}

{{< ci-details summary="Using Dynamic Embeddings to Improve Static Embeddings (Yile Wang et al., 2019)">}}

Yile Wang, Leyang Cui, Yue Zhang. (2019)  
**Using Dynamic Embeddings to Improve Static Embeddings**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/1257f59bd9b6bc3f3823125408c7b6e63db4a158)  
Influential Citation Count (1), SS-ID (1257f59bd9b6bc3f3823125408c7b6e63db4a158)  

**ABSTRACT**  
How to build high-quality word embeddings is a fundamental research question in the field of natural language processing. Traditional methods such as Skip-Gram and Continuous Bag-of-Words learn {\it static} embeddings by training lookup tables that translate words into dense vectors. Static embeddings are directly useful for solving lexical semantics tasks, and can be used as input representations for downstream problems. Recently, contextualized embeddings such as BERT have been shown more effective than static embeddings as NLP input embeddings. Such embeddings are {\it dynamic}, calculated according to a sentential context using a network structure. One limitation of dynamic embeddings, however, is that they cannot be used without a sentence-level context. We explore the advantages of dynamic embeddings for training static embeddings, by using contextualized embeddings to facilitate training of static embedding lookup tables. Results show that the resulting embeddings outperform existing static embedding methods on various lexical semantics tasks.

{{< /ci-details >}}

{{< ci-details summary="Is Attention Interpretable? (Sofia Serrano et al., 2019)">}}

Sofia Serrano, Noah A. Smith. (2019)  
**Is Attention Interpretable?**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/135112c7ba1762d65f39b1a61777f26ae4dfd8ad)  
Influential Citation Count (16), SS-ID (135112c7ba1762d65f39b1a61777f26ae4dfd8ad)  

**ABSTRACT**  
Attention mechanisms have recently boosted performance on a range of NLP tasks. Because attention layers explicitly weight input components’ representations, it is also often assumed that attention can be used to identify information that models found important (e.g., specific contextualized word tokens). We test whether that assumption holds by manipulating attention weights in already-trained text classification models and analyzing the resulting differences in their predictions. While we observe some ways in which higher attention weights correlate with greater impact on model predictions, we also find many ways in which this does not hold, i.e., where gradient-based rankings of attention weights better predict their effects than their magnitudes. We conclude that while attention noisily predicts input components’ overall importance to a model, it is by no means a fail-safe indicator.1

{{< /ci-details >}}

{{< ci-details summary="Open Sesame: Getting inside BERT’s Linguistic Knowledge (Yongjie Lin et al., 2019)">}}

Yongjie Lin, Y. Tan, R. Frank. (2019)  
**Open Sesame: Getting inside BERT’s Linguistic Knowledge**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/165d51a547cd920e6ac55660ad5c404dcb9562ed)  
Influential Citation Count (18), SS-ID (165d51a547cd920e6ac55660ad5c404dcb9562ed)  

**ABSTRACT**  
How and to what extent does BERT encode syntactically-sensitive hierarchical information or positionally-sensitive linear information? Recent work has shown that contextual representations like BERT perform well on tasks that require sensitivity to linguistic structure. We present here two studies which aim to provide a better understanding of the nature of BERT’s representations. The first of these focuses on the identification of structurally-defined elements using diagnostic classifiers, while the second explores BERT’s representation of subject-verb agreement and anaphor-antecedent dependencies through a quantitative assessment of self-attention vectors. In both cases, we find that BERT encodes positional information about word tokens well on its lower layers, but switches to a hierarchically-oriented encoding on higher layers. We conclude then that BERT’s representations do indeed model linguistically relevant aspects of hierarchical structure, though they do not appear to show the sharp sensitivity to hierarchical structure that is found in human processing of reflexive anaphora.

{{< /ci-details >}}

{{< ci-details summary="What’s in a Name? Are BERT Named Entity Representations just as Good for any other Name? (S. Balasubramanian et al., 2020)">}}

S. Balasubramanian, Naman Jain, G. Jindal, Abhijeet Awasthi, Sunita Sarawagi. (2020)  
**What’s in a Name? Are BERT Named Entity Representations just as Good for any other Name?**  
REPL4NLP  
[Paper Link](https://www.semanticscholar.org/paper/167f52d369b0979f27282af0f3a1a4be9c9be84b)  
Influential Citation Count (1), SS-ID (167f52d369b0979f27282af0f3a1a4be9c9be84b)  

**ABSTRACT**  
We evaluate named entity representations of BERT-based NLP models by investigating their robustness to replacements from the same typed class in the input. We highlight that on several tasks while such perturbations are natural, state of the art trained models are surprisingly brittle. The brittleness continues even with the recent entity-aware BERT models. We also try to discern the cause of this non-robustness, considering factors such as tokenization and frequency of occurrence. Then we provide a simple method that ensembles predictions from multiple replacements while jointly modeling the uncertainty of type annotations and label predictions. Experiments on three NLP tasks shows that our method enhances robustness and increases accuracy on both natural and adversarial datasets.

{{< /ci-details >}}

{{< ci-details summary="Conditional BERT Contextual Augmentation (Xing Wu et al., 2018)">}}

Xing Wu, Shangwen Lv, Liangjun Zang, Jizhong Han, Songlin Hu. (2018)  
**Conditional BERT Contextual Augmentation**  
ICCS  
[Paper Link](https://www.semanticscholar.org/paper/188024469a2443f262b3cbb5c5d4a96851949d68)  
Influential Citation Count (21), SS-ID (188024469a2443f262b3cbb5c5d4a96851949d68)  

**ABSTRACT**  
Data augmentation methods are often applied to prevent overfitting and improve generalization of deep neural network models. Recently proposed contextual augmentation augments labeled sentences by randomly replacing words with more varied substitutions predicted by language model. Bidirectional Encoder Representations from Transformers (BERT) demonstrates that a deep bidirectional language model is more powerful than either an unidirectional language model or the shallow concatenation of a forward and backward model. We propose a novel data augmentation method for labeled sentences called conditional BERT contextual augmentation. We retrofit BERT to conditional BERT by introducing a new conditional masked language model (The term “conditional masked language model” appeared once in original BERT paper, which indicates context-conditional, is equivalent to term “masked language model”. In our paper, “conditional masked language model” indicates we apply extra label-conditional constraint to the “masked language model”.) task. The well trained conditional BERT can be applied to enhance contextual augmentation. Experiments on six various different text classification tasks show that our method can be easily applied to both convolutional or recurrent neural networks classifier to obtain improvement.

{{< /ci-details >}}

{{< ci-details summary="Universal Adversarial Triggers for Attacking and Analyzing NLP (Eric Wallace et al., 2019)">}}

Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh. (2019)  
**Universal Adversarial Triggers for Attacking and Analyzing NLP**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/18a1c21f35153c45d0ef30c564bffb7d70a13ccc)  
Influential Citation Count (41), SS-ID (18a1c21f35153c45d0ef30c564bffb7d70a13ccc)  

**ABSTRACT**  
Adversarial examples highlight model vulnerabilities and are useful for evaluation and interpretation. We define universal adversarial triggers: input-agnostic sequences of tokens that trigger a model to produce a specific prediction when concatenated to any input from a dataset. We propose a gradient-guided search over tokens which finds short trigger sequences (e.g., one word for classification and four words for language modeling) that successfully trigger the target prediction. For example, triggers cause SNLI entailment accuracy to drop from 89.94% to 0.55%, 72% of “why” questions in SQuAD to be answered “to kill american people”, and the GPT-2 language model to spew racist output even when conditioned on non-racial contexts. Furthermore, although the triggers are optimized using white-box access to a specific model, they transfer to other models for all tasks we consider. Finally, since triggers are input-agnostic, they provide an analysis of global model behavior. For instance, they confirm that SNLI models exploit dataset biases and help to diagnose heuristics learned by reading comprehension models.

{{< /ci-details >}}

{{< ci-details summary="Learning and Evaluating General Linguistic Intelligence (Dani Yogatama et al., 2019)">}}

Dani Yogatama, Cyprien de Masson d'Autume, Jerome T. Connor, Tomás Kociský, Mike Chrzanowski, Lingpeng Kong, Angeliki Lazaridou, Wang Ling, Lei Yu, Chris Dyer, P. Blunsom. (2019)  
**Learning and Evaluating General Linguistic Intelligence**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/19281b9ecdb5c07a93423a506627ab9d9b0cf039)  
Influential Citation Count (6), SS-ID (19281b9ecdb5c07a93423a506627ab9d9b0cf039)  

**ABSTRACT**  
We define general linguistic intelligence as the ability to reuse previously acquired knowledge about a language's lexicon, syntax, semantics, and pragmatic conventions to adapt to new tasks quickly. Using this definition, we analyze state-of-the-art natural language understanding models and conduct an extensive empirical investigation to evaluate them against these criteria through a series of experiments that assess the task-independence of the knowledge being acquired by the learning process. In addition to task performance, we propose a new evaluation metric based on an online encoding of the test data that quantifies how quickly an existing agent (model) learns a new task. Our results show that while the field has made impressive progress in terms of model architectures that generalize to many tasks, these models still require a lot of in-domain training examples (e.g., for fine tuning, training task-specific modules), and are prone to catastrophic forgetting. Moreover, we find that far from solving general tasks (e.g., document question answering), our models are overfitting to the quirks of particular datasets (e.g., SQuAD). We discuss missing components and conjecture on how to make progress toward general linguistic intelligence.

{{< /ci-details >}}

{{< ci-details summary="GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy Efficient Inference (Ali Hadi Zadeh et al., 2020)">}}

Ali Hadi Zadeh, A. Moshovos. (2020)  
**GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy Efficient Inference**  
2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)  
[Paper Link](https://www.semanticscholar.org/paper/1b0c8b26affd13e10ace5770e85478d60dcc368e)  
Influential Citation Count (4), SS-ID (1b0c8b26affd13e10ace5770e85478d60dcc368e)  

**ABSTRACT**  
Attention-based models have demonstrated remarkable success in various natural language understanding tasks. However, efficient execution remains a challenge for these models which are memory-bound due to their massive number of parameters. We present GOBO, a model quantization technique that compresses the vast majority (typically 99.9%) of the 32-bit floating-point parameters of state-of-the-art BERT models and their variants to 3 bits while maintaining their accuracy. Unlike other quantization methods, GOBO does not require fine-tuning nor retraining to compensate for the quantization error. We present two practical hardware applications of GOBO. In the first GOBO reduces memory storage and traffic and as a result inference latency and energy consumption. This GOBO memory compression mechanism is plug-in compatible with many architectures; we demonstrate it with the TPU, Eyeriss, and an architecture using Tensor Cores-like units. Second, we present a co-designed hardware architecture that also reduces computation. Uniquely, the GOBO architecture maintains most of the weights in 3b even during computation, a property that: (i) makes the processing elements area efficient, allowing us to pack more compute power per unit area, (ii) replaces most multiply-accumulations with additions, and (iii) reduces the off-chip traffic by amplifying on-chip memory capacity.

{{< /ci-details >}}

{{< ci-details summary="SegaBERT: Pre-training of Segment-aware BERT for Language Understanding (He Bai et al., 2020)">}}

He Bai, Peng Shi, Jimmy J. Lin, Luchen Tan, Kun Xiong, Wen Gao, Ming Li. (2020)  
**SegaBERT: Pre-training of Segment-aware BERT for Language Understanding**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/1cc0b98b938b984e5da85f86c1a24099b9b4b582)  
Influential Citation Count (2), SS-ID (1cc0b98b938b984e5da85f86c1a24099b9b4b582)  

**ABSTRACT**  
Pre-trained language models have achieved state-of-the-art results in various natural language processing tasks. Most of them are based on the Transformer architecture, which distinguishes tokens with the token position index of the input sequence. However, sentence index and paragraph index are also important to indicate the token position in a document. We hypothesize that better contextual representations can be generated from the text encoder with richer positional information. To verify this, we propose a segment-aware BERT, by replacing the token position embedding of Transformer with a combination of paragraph index, sentence index, and token index embeddings. We pre-trained the SegaBERT on the masked language modeling task in BERT but without any affiliated tasks. Experimental results show that our pre-trained model can outperform the original BERT model on various NLP tasks.

{{< /ci-details >}}

{{< ci-details summary="Attention is not Explanation (Sarthak Jain et al., 2019)">}}

Sarthak Jain, Byron C. Wallace. (2019)  
**Attention is not Explanation**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/1e83c20def5c84efa6d4a0d80aa3159f55cb9c3f)  
Influential Citation Count (36), SS-ID (1e83c20def5c84efa6d4a0d80aa3159f55cb9c3f)  

**ABSTRACT**  
Attention mechanisms have seen wide adoption in neural NLP models. In addition to improving predictive performance, these are often touted as affording transparency: models equipped with attention provide a distribution over attended-to input units, and this is often presented (at least implicitly) as communicating the relative importance of inputs. However, it is unclear what relationship exists between attention weights and model outputs. In this work we perform extensive experiments across a variety of NLP tasks that aim to assess the degree to which attention weights provide meaningful “explanations” for predictions. We find that they largely do not. For example, learned attention weights are frequently uncorrelated with gradient-based measures of feature importance, and one can identify very different attention distributions that nonetheless yield equivalent predictions. Our findings show that standard attention modules do not provide meaningful explanations and should not be treated as though they do.

{{< /ci-details >}}

{{< ci-details summary="HuggingFace's Transformers: State-of-the-art Natural Language Processing (Thomas Wolf et al., 2019)">}}

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, T. Rault, Rémi Louf, Morgan Funtowicz, Jamie Brew. (2019)  
**HuggingFace's Transformers: State-of-the-art Natural Language Processing**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/1fa9ed2bea208511ae698a967875e943049f16b6)  
Influential Citation Count (208), SS-ID (1fa9ed2bea208511ae698a967875e943049f16b6)  

**ABSTRACT**  
Recent progress in natural language processing has been driven by advances in both model architecture and model pretraining. Transformer architectures have facilitated building higher-capacity models and pretraining has made it possible to effectively utilize this capacity for a wide variety of tasks. \textit{Transformers} is an open-source library with the goal of opening up these advances to the wider machine learning community. The library consists of carefully engineered state-of-the art Transformer architectures under a unified API. Backing this library is a curated collection of pretrained models made by and available for the community. \textit{Transformers} is designed to be extensible by researchers, simple for practitioners, and fast and robust in industrial deployments. The library is available at \url{this https URL}.

{{< /ci-details >}}

{{< ci-details summary="Attention is All you Need (Ashish Vaswani et al., 2017)">}}

Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. (2017)  
**Attention is All you Need**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776)  
Influential Citation Count (7570), SS-ID (204e3073870fae3d05bcbc2f6a8e263d9b72e776)  

**ABSTRACT**  
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

{{< /ci-details >}}

{{< ci-details summary="The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks (Jonathan Frankle et al., 2018)">}}

Jonathan Frankle, Michael Carbin. (2018)  
**The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/21937ecd9d66567184b83eca3d3e09eb4e6fbd60)  
Influential Citation Count (230), SS-ID (21937ecd9d66567184b83eca3d3e09eb4e6fbd60)  

**ABSTRACT**  
Neural network pruning techniques can reduce the parameter counts of trained networks by over 90%, decreasing storage requirements and improving computational performance of inference without compromising accuracy. However, contemporary experience is that the sparse architectures produced by pruning are difficult to train from the start, which would similarly improve training performance.  We find that a standard pruning technique naturally uncovers subnetworks whose initializations made them capable of training effectively. Based on these results, we articulate the "lottery ticket hypothesis:" dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations. The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective.  We present an algorithm to identify winning tickets and a series of experiments that support the lottery ticket hypothesis and the importance of these fortuitous initializations. We consistently find winning tickets that are less than 10-20% of the size of several fully-connected and convolutional feed-forward architectures for MNIST and CIFAR10. Above this size, the winning tickets that we find learn faster than the original network and reach higher test accuracy.

{{< /ci-details >}}

{{< ci-details summary="Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models (Cheolhyoung Lee et al., 2019)">}}

Cheolhyoung Lee, Kyunghyun Cho, Wanmo Kang. (2019)  
**Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/222b9a7b8038120671a1610e857d3edbc7ac5550)  
Influential Citation Count (12), SS-ID (222b9a7b8038120671a1610e857d3edbc7ac5550)  

**ABSTRACT**  
In natural language processing, it has been observed recently that generalization could be greatly improved by finetuning a large-scale language model pretrained on a large unlabeled corpus. Despite its recent success and wide adoption, finetuning a large pretrained language model on a downstream task is prone to degenerate performance when there are only a small number of training instances available. In this paper, we introduce a new regularization technique, to which we refer as "mixout", motivated by dropout. Mixout stochastically mixes the parameters of two models. We show that our mixout technique regularizes learning to minimize the deviation from one of the two models and that the strength of regularization adapts along the optimization trajectory. We empirically evaluate the proposed mixout and its variants on finetuning a pretrained language model on downstream tasks. More specifically, we demonstrate that the stability of finetuning and the average accuracy greatly increase when we use the proposed approach to regularize finetuning of BERT on downstream tasks in GLUE.

{{< /ci-details >}}

{{< ci-details summary="Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers (Zhuohan Li et al., 2020)">}}

Zhuohan Li, Eric Wallace, Sheng Shen, Kevin Lin, K. Keutzer, D. Klein, Joseph Gonzalez. (2020)  
**Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/2356781b8a98bf94e6fc73798c6cb65ac35e5f97)  
Influential Citation Count (6), SS-ID (2356781b8a98bf94e6fc73798c6cb65ac35e5f97)  

**ABSTRACT**  
Since hardware resources are limited, the objective of training deep learning models is typically to maximize accuracy subject to the time and memory constraints of training and inference. We study the impact of model size in this setting, focusing on Transformer models for NLP tasks that are limited by compute: self-supervised pretraining and high-resource machine translation. We first show that even though smaller Transformer models execute faster per iteration, wider and deeper models converge in significantly fewer steps. Moreover, this acceleration in convergence typically outpaces the additional computational overhead of using larger models. Therefore, the most compute-efficient training strategy is to counterintuitively train extremely large models but stop after a small number of iterations.  This leads to an apparent trade-off between the training efficiency of large Transformer models and the inference efficiency of small Transformer models. However, we show that large models are more robust to compression techniques such as quantization and pruning than small models. Consequently, one can get the best of both worlds: heavily compressed, large models achieve higher accuracy than lightly compressed, small models.

{{< /ci-details >}}

{{< ci-details summary="Parameter-Efficient Transfer Learning for NLP (N. Houlsby et al., 2019)">}}

N. Houlsby, A. Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, S. Gelly. (2019)  
**Parameter-Efficient Transfer Learning for NLP**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/29ddc1f43f28af7c846515e32cc167bc66886d0c)  
Influential Citation Count (112), SS-ID (29ddc1f43f28af7c846515e32cc167bc66886d0c)  

**ABSTRACT**  
Fine-tuning large pre-trained models is an effective transfer mechanism in NLP. However, in the presence of many downstream tasks, fine-tuning is parameter inefficient: an entire new model is required for every task. As an alternative, we propose transfer with adapter modules. Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones. The parameters of the original network remain fixed, yielding a high degree of parameter sharing. To demonstrate adapter's effectiveness, we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark. Adapters attain near state-of-the-art performance, whilst adding only a few parameters per task. On GLUE, we attain within 0.4% of the performance of full fine-tuning, adding only 3.6% parameters per task. By contrast, fine-tuning trains 100% of the parameters per task.

{{< /ci-details >}}

{{< ci-details summary="Attention Module is Not Only a Weight: Analyzing Transformers with Vector Norms (Goro Kobayashi et al., 2020)">}}

Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, Kentaro Inui. (2020)  
**Attention Module is Not Only a Weight: Analyzing Transformers with Vector Norms**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/2a8e42995caaedadc9dc739d85bed2c57fc78568)  
Influential Citation Count (0), SS-ID (2a8e42995caaedadc9dc739d85bed2c57fc78568)  

**ABSTRACT**  
Attention is a key component of Transformers, which have recently achieved considerable success in natural language processing. Hence, attention is being extensively studied to investigate various linguistic capabilities of Transformers, focusing on analyzing the parallels between attention weights and specific linguistic phenomena. This paper shows that attention weights alone are only one of the two factors that determine the output of attention and proposes a norm-based analysis that incorporates the second factor, the norm of the transformed input vectors. The findings of our norm-based analyses of BERT and a Transformer-based neural machine translation system include the following: (i) contrary to previous studies, BERT pays poor attention to special tokens, and (ii) reasonable word alignment can be extracted from attention mechanisms of Transformer. These findings provide insights into the inner workings of Transformers.

{{< /ci-details >}}

{{< ci-details summary="BERT-of-Theseus: Compressing BERT by Progressive Module Replacing (Canwen Xu et al., 2020)">}}

Canwen Xu, Wangchunshu Zhou, Tao Ge, Furu Wei, Ming Zhou. (2020)  
**BERT-of-Theseus: Compressing BERT by Progressive Module Replacing**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/2e27f119e6fcc5477248eb0f4a6abe8d7cf4f6e7)  
Influential Citation Count (14), SS-ID (2e27f119e6fcc5477248eb0f4a6abe8d7cf4f6e7)  

**ABSTRACT**  
In this paper, we propose a novel model compression approach to effectively compress BERT by progressive module replacing. Our approach first divides the original BERT into several modules and builds their compact substitutes. Then, we randomly replace the original modules with their substitutes to train the compact modules to mimic the behavior of the original modules. We progressively increase the probability of replacement through the training. In this way, our approach brings a deeper level of interaction between the original and compact models. Compared to the previous knowledge distillation approaches for BERT compression, our approach does not introduce any additional loss function. Our approach outperforms existing knowledge distillation approaches on GLUE benchmark, showing a new perspective of model compression.

{{< /ci-details >}}

{{< ci-details summary="Small and Practical BERT Models for Sequence Labeling (Henry Tsai et al., 2019)">}}

Henry Tsai, Jason Riesa, Melvin Johnson, N. Arivazhagan, Xin Li, Amelia Archer. (2019)  
**Small and Practical BERT Models for Sequence Labeling**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/2f9d4887d0022400fc40c774c4c78350c3bc5390)  
Influential Citation Count (6), SS-ID (2f9d4887d0022400fc40c774c4c78350c3bc5390)  

**ABSTRACT**  
We propose a practical scheme to train a single multilingual sequence labeling model that yields state of the art results and is small and fast enough to run on a single CPU. Starting from a public multilingual BERT checkpoint, our final model is 6x smaller and 27x faster, and has higher accuracy than a state-of-the-art multilingual baseline. We show that our model especially outperforms on low-resource languages, and works on codemixed input text without being explicitly trained on codemixed examples. We showcase the effectiveness of our method by reporting on part-of-speech tagging and morphological prediction on 70 treebanks and 48 languages.

{{< /ci-details >}}

{{< ci-details summary="Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT (Shijie Wu et al., 2019)">}}

Shijie Wu, Mark Dredze. (2019)  
**Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/2fa3f7ce620a1c7155daef6620dd6bb0e01934f3)  
Influential Citation Count (28), SS-ID (2fa3f7ce620a1c7155daef6620dd6bb0e01934f3)  

**ABSTRACT**  
Pretrained contextual representation models (Peters et al., 2018; Devlin et al., 2018) have pushed forward the state-of-the-art on many NLP tasks. A new release of BERT (Devlin, 2018) includes a model simultaneously pretrained on 104 languages with impressive performance for zero-shot cross-lingual transfer on a natural language inference task. This paper explores the broader cross-lingual potential of mBERT (multilingual) as a zero shot language transfer model on 5 NLP tasks covering a total of 39 languages from various language families: NLI, document classification, NER, POS tagging, and dependency parsing. We compare mBERT with the best-published methods for zero-shot cross-lingual transfer and find mBERT competitive on each task. Additionally, we investigate the most effective strategy for utilizing mBERT in this manner, determine to what extent mBERT generalizes away from language specific features, and measure factors that influence cross-lingual transfer.

{{< /ci-details >}}

{{< ci-details summary="Pre-Training With Whole Word Masking for Chinese BERT (Yiming Cui et al., 2019)">}}

Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang, Shijin Wang, Guoping Hu. (2019)  
**Pre-Training With Whole Word Masking for Chinese BERT**  
IEEE/ACM Transactions on Audio, Speech, and Language Processing  
[Paper Link](https://www.semanticscholar.org/paper/2ff41a463a374b138bb5a012e5a32bc4beefec20)  
Influential Citation Count (61), SS-ID (2ff41a463a374b138bb5a012e5a32bc4beefec20)  

**ABSTRACT**  
Bidirectional Encoder Representations from Transformers (BERT) has shown marvelous improvements across various NLP tasks, and its consecutive variants have been proposed to further improve the performance of the pre-trained language models. In this paper, we aim to first introduce the whole word masking (wwm) strategy for Chinese BERT, along with a series of Chinese pre-trained language models. Then we also propose a simple but effective model called MacBERT, which improves upon RoBERTa in several ways. Especially, we propose a new masking strategy called MLM as correction (Mac). To demonstrate the effectiveness of these models, we create a series of Chinese pre-trained language models as our baselines, including BERT, RoBERTa, ELECTRA, RBT, etc. We carried out extensive experiments on ten Chinese NLP tasks to evaluate the created Chinese pre-trained language models as well as the proposed MacBERT. Experimental results show that MacBERT could achieve state-of-the-art performances on many NLP tasks, and we also ablate details with several findings that may help future research. We open-source our pre-trained language models for further facilitating our research community.1

{{< /ci-details >}}

{{< ci-details summary="Whatcha lookin' at? DeepLIFTing BERT's Attention in Question Answering (Ekaterina Arkhangelskaia et al., 2019)">}}

Ekaterina Arkhangelskaia, Sourav Dutta. (2019)  
**Whatcha lookin' at? DeepLIFTing BERT's Attention in Question Answering**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/304b7c87e5c6e76ffcfdaa59fbd0656f9dab47d8)  
Influential Citation Count (0), SS-ID (304b7c87e5c6e76ffcfdaa59fbd0656f9dab47d8)  

**ABSTRACT**  
There has been great success recently in tackling challenging NLP tasks by neural networks which have been pre-trained and fine-tuned on large amounts of task data. In this paper, we investigate one such model, BERT for question-answering, with the aim to analyze why it is able to achieve significantly better results than other models. We run DeepLIFT on the model predictions and test the outcomes to monitor shift in the attention values for input. We also cluster the results to analyze any possible patterns similar to human reasoning depending on the kind of input paragraph and question the model is trying to answer.

{{< /ci-details >}}

{{< ci-details summary="Understanding Multi-Head Attention in Abstractive Summarization (Joris Baan et al., 2019)">}}

Joris Baan, Maartje ter Hoeve, M. V. D. Wees, Anne Schuth, M. de Rijke. (2019)  
**Understanding Multi-Head Attention in Abstractive Summarization**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/317d2ac530e1db49229d6c442f50722db85afbb7)  
Influential Citation Count (1), SS-ID (317d2ac530e1db49229d6c442f50722db85afbb7)  

**ABSTRACT**  
Attention mechanisms in deep learning architectures have often been used as a means of transparency and, as such, to shed light on the inner workings of the architectures. Recently, there has been a growing interest in whether or not this assumption is correct. In this paper we investigate the interpretability of multi-head attention in abstractive summarization, a sequence-to-sequence task for which attention does not have an intuitive alignment role, such as in machine translation. We first introduce three metrics to gain insight in the focus of attention heads and observe that these heads specialize towards relative positions, specific part-of-speech tags, and named entities. However, we also find that ablating and pruning these heads does not lead to a significant drop in performance, indicating redundancy. By replacing the softmax activation functions with sparsemax activation functions, we find that attention heads behave seemingly more transparent: we can ablate fewer heads and heads score higher on our interpretability metrics. However, if we apply pruning to the sparsemax model we find that we can prune even more heads, raising the question whether enforced sparsity actually improves transparency. Finally, we find that relative positions heads seem integral to summarization performance and persistently remain after pruning.

{{< /ci-details >}}

{{< ci-details summary="75 Languages, 1 Model: Parsing Universal Dependencies Universally (D. Kondratyuk, 2019)">}}

D. Kondratyuk. (2019)  
**75 Languages, 1 Model: Parsing Universal Dependencies Universally**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/31c872514c28a172f7f0221c8596aa5bfcdb9e98)  
Influential Citation Count (33), SS-ID (31c872514c28a172f7f0221c8596aa5bfcdb9e98)  

**ABSTRACT**  
We present UDify, a multilingual multi-task model capable of accurately predicting universal part-of-speech, morphological features, lemmas, and dependency trees simultaneously for all 124 Universal Dependencies treebanks across 75 languages. By leveraging a multilingual BERT self-attention model pretrained on 104 languages, we found that fine-tuning it on all datasets concatenated together with simple softmax classifiers for each UD task can meet or exceed state-of-the-art UPOS, UFeats, Lemmas, (and especially) UAS, and LAS scores, without requiring any recurrent or language-specific components. We evaluate UDify for multilingual learning, showing that low-resource languages benefit the most from cross-linguistic annotations. We also evaluate for zero-shot learning, with results suggesting that multilingual training provides strong UD predictions even for languages that neither UDify nor BERT have ever been trained on.

{{< /ci-details >}}

{{< ci-details summary="exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformer Models (Benjamin Hoover et al., 2019)">}}

Benjamin Hoover, Hendrik Strobelt, Sebastian Gehrmann. (2019)  
**exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformer Models**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/327d7e55d64cb34d55bd3a3fe58233c238a312cd)  
Influential Citation Count (4), SS-ID (327d7e55d64cb34d55bd3a3fe58233c238a312cd)  

**ABSTRACT**  
Large Transformer-based language models can route and reshape complex information via their multi-headed attention mechanism. Although the attention never receives explicit supervision, it can exhibit recognizable patterns following linguistic or positional information. Analyzing the learned representations and attentions is paramount to furthering our understanding of the inner workings of these models. However, analyses have to catch up with the rapid release of new models and the growing diversity of investigation techniques. To support analysis for a wide variety of models, we introduce exBERT, a tool to help humans conduct flexible, interactive investigations and formulate hypotheses for the model-internal reasoning process. exBERT provides insights into the meaning of the contextual representations and attention by matching a human-specified input to similar contexts in large annotated datasets. By aggregating the annotations of the matched contexts, exBERT can quickly replicate findings from literature and extend them to previously not analyzed models.

{{< /ci-details >}}

{{< ci-details summary="Data Augmentation using Pre-trained Transformer Models (Varun Kumar et al., 2020)">}}

Varun Kumar, Ashutosh Choudhary, Eunah Cho. (2020)  
**Data Augmentation using Pre-trained Transformer Models**  
LIFELONGNLP  
[Paper Link](https://www.semanticscholar.org/paper/33496cb3a5623925267528fa6b726f015e4dcda2)  
Influential Citation Count (17), SS-ID (33496cb3a5623925267528fa6b726f015e4dcda2)  

**ABSTRACT**  
Language model based pre-trained models such as BERT have provided significant gains across different NLP tasks. In this paper, we study different types of transformer based pre-trained models such as auto-regressive models (GPT-2), auto-encoder models (BERT), and seq2seq models (BART) for conditional data augmentation. We show that prepending the class labels to text sequences provides a simple yet effective way to condition the pre-trained models for data augmentation. Additionally, on three classification benchmarks, pre-trained Seq2Seq model outperforms other data augmentation methods in a low-resource setting. Further, we explore how different pre-trained model based data augmentation differs in-terms of data diversity, and how well such methods preserve the class-label information.

{{< /ci-details >}}

{{< ci-details summary="What Does BERT Learn about the Structure of Language? (Ganesh Jawahar et al., 2019)">}}

Ganesh Jawahar, Benoît Sagot, Djamé Seddah. (2019)  
**What Does BERT Learn about the Structure of Language?**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/335613303ebc5eac98de757ed02a56377d99e03a)  
Influential Citation Count (44), SS-ID (335613303ebc5eac98de757ed02a56377d99e03a)  

**ABSTRACT**  
BERT is a recent language representation model that has surprisingly performed well in diverse language understanding benchmarks. This result indicates the possibility that BERT networks capture structural information about language. In this work, we provide novel support for this claim by performing a series of experiments to unpack the elements of English language structure learned by BERT. Our findings are fourfold. BERT’s phrasal representation captures the phrase-level information in the lower layers. The intermediate layers of BERT compose a rich hierarchy of linguistic information, starting with surface features at the bottom, syntactic features in the middle followed by semantic features at the top. BERT requires deeper layers while tracking subject-verb agreement to handle long-term dependency problem. Finally, the compositional scheme underlying BERT mimics classical, tree-like structures.

{{< /ci-details >}}

{{< ci-details summary="Beyond Accuracy: Behavioral Testing of NLP Models with CheckList (Marco Tulio Ribeiro et al., 2020)">}}

Marco Tulio Ribeiro, Tongshuang Sherry Wu, Carlos Guestrin, Sameer Singh. (2020)  
**Beyond Accuracy: Behavioral Testing of NLP Models with CheckList**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/33ec7eb2168e37e3007d1059aa96b9a63254b4da)  
Influential Citation Count (61), SS-ID (33ec7eb2168e37e3007d1059aa96b9a63254b4da)  

**ABSTRACT**  
Although measuring held-out accuracy has been the primary approach to evaluate generalization, it often overestimates the performance of NLP models, while alternative approaches for evaluating models either focus on individual tasks or on specific behaviors. Inspired by principles of behavioral testing in software engineering, we introduce CheckList, a task-agnostic methodology for testing NLP models. CheckList includes a matrix of general linguistic capabilities and test types that facilitate comprehensive test ideation, as well as a software tool to generate a large and diverse number of test cases quickly. We illustrate the utility of CheckList with tests for three tasks, identifying critical failures in both commercial and state-of-art models. In a user study, a team responsible for a commercial sentiment analysis model found new and actionable bugs in an extensively tested model. In another user study, NLP practitioners with CheckList created twice as many tests, and found almost three times as many bugs as users without it.

{{< /ci-details >}}

{{< ci-details summary="Quantity doesn’t buy quality syntax with neural language models (Marten van Schijndel et al., 2019)">}}

Marten van Schijndel, Aaron Mueller, Tal Linzen. (2019)  
**Quantity doesn’t buy quality syntax with neural language models**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/356645552f8f40adf5a99b4e3a69f47699399010)  
Influential Citation Count (0), SS-ID (356645552f8f40adf5a99b4e3a69f47699399010)  

**ABSTRACT**  
Recurrent neural networks can learn to predict upcoming words remarkably well on average; in syntactically complex contexts, however, they often assign unexpectedly high probabilities to ungrammatical words. We investigate to what extent these shortcomings can be mitigated by increasing the size of the network and the corpus on which it is trained. We find that gains from increasing network size are minimal beyond a certain point. Likewise, expanding the training corpus yields diminishing returns; we estimate that the training corpus would need to be unrealistically large for the models to match human performance. A comparison to GPT and BERT, Transformer-based models trained on billions of words, reveals that these models perform even more poorly than our LSTMs in some constructions. Our results make the case for more data efficient architectures.

{{< /ci-details >}}

{{< ci-details summary="What do you mean? (M. Jackson, 1989)">}}

M. Jackson. (1989)  
**What do you mean?**  
Geriatric nursing  
[Paper Link](https://www.semanticscholar.org/paper/357771514cfbbdc0ddafe1dfdf54eda3c42b325e)  
Influential Citation Count (17), SS-ID (357771514cfbbdc0ddafe1dfdf54eda3c42b325e)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="The Lottery Ticket Hypothesis for Pre-trained BERT Networks (Tianlong Chen et al., 2020)">}}

Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang Wang, Michael Carbin. (2020)  
**The Lottery Ticket Hypothesis for Pre-trained BERT Networks**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/389036b1366b64579725457993c1f63a4f3370ba)  
Influential Citation Count (18), SS-ID (389036b1366b64579725457993c1f63a4f3370ba)  

**ABSTRACT**  
In natural language processing (NLP), enormous pre-trained models like BERT have become the standard starting point for training on a range of downstream tasks, and similar trends are emerging in other areas of deep learning. In parallel, work on the lottery ticket hypothesis has shown that models for NLP and computer vision contain smaller matching subnetworks capable of training in isolation to full accuracy and transferring to other tasks. In this work, we combine these observations to assess whether such trainable, transferrable subnetworks exist in pre-trained BERT models. For a range of downstream tasks, we indeed find matching subnetworks at 40% to 90% sparsity. We find these subnetworks at (pre-trained) initialization, a deviation from prior NLP research where they emerge only after some amount of training. Subnetworks found on the masked language modeling task (the same task used to pre-train the model) transfer universally; those found on other tasks transfer in a limited fashion if at all. As large-scale pre-training becomes an increasingly central paradigm in deep learning, our results demonstrate that the main lottery ticket observations remain relevant in this context. Codes available at this https URL.

{{< /ci-details >}}

{{< ci-details summary="BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (M. Lewis et al., 2019)">}}

M. Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, Luke Zettlemoyer. (2019)  
**BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/395de0bd3837fdf4b4b5e5f04835bcc69c279481)  
Influential Citation Count (482), SS-ID (395de0bd3837fdf4b4b5e5f04835bcc69c279481)  

**ABSTRACT**  
We present BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text. It uses a standard Tranformer-based neural machine translation architecture which, despite its simplicity, can be seen as generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder), and other recent pretraining schemes. We evaluate a number of noising approaches, finding the best performance by both randomly shuffling the order of sentences and using a novel in-filling scheme, where spans of text are replaced with a single mask token. BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa on GLUE and SQuAD, and achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 3.5 ROUGE. BART also provides a 1.1 BLEU increase over a back-translation system for machine translation, with only target language pretraining. We also replicate other pretraining schemes within the BART framework, to understand their effect on end-task performance.

{{< /ci-details >}}

{{< ci-details summary="Investigating Entity Knowledge in BERT with Simple Neural End-To-End Entity Linking (Samuel Broscheit, 2019)">}}

Samuel Broscheit. (2019)  
**Investigating Entity Knowledge in BERT with Simple Neural End-To-End Entity Linking**  
CoNLL  
[Paper Link](https://www.semanticscholar.org/paper/399308fa54ade9b1362d56628132323489ce50cd)  
Influential Citation Count (6), SS-ID (399308fa54ade9b1362d56628132323489ce50cd)  

**ABSTRACT**  
A typical architecture for end-to-end entity linking systems consists of three steps: mention detection, candidate generation and entity disambiguation. In this study we investigate the following questions: (a) Can all those steps be learned jointly with a model for contextualized text-representations, i.e. BERT? (b) How much entity knowledge is already contained in pretrained BERT? (c) Does additional entity knowledge improve BERT’s performance in downstream tasks? To this end we propose an extreme simplification of the entity linking setup that works surprisingly well: simply cast it as a per token classification over the entire entity vocabulary (over 700K classes in our case). We show on an entity linking benchmark that (i) this model improves the entity representations over plain BERT, (ii) that it outperforms entity linking architectures that optimize the tasks separately and (iii) that it only comes second to the current state-of-the-art that does mention detection and entity disambiguation jointly. Additionally, we investigate the usefulness of entity-aware token-representations in the text-understanding benchmark GLUE, as well as the question answering benchmarks SQUAD~V2 and SWAG and also the EN-DE WMT14 machine translation benchmark. To our surprise, we find that most of those benchmarks do not benefit from additional entity knowledge, except for a task with very small training data, the RTE task in GLUE, which improves by 2%.

{{< /ci-details >}}

{{< ci-details summary="Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT (Zhiyong Wu et al., 2020)">}}

Zhiyong Wu, Yun Chen, B. Kao, Qun Liu. (2020)  
**Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/3aaa8aaad5ef36550a6b47d6ee000f0b346a5a1f)  
Influential Citation Count (10), SS-ID (3aaa8aaad5ef36550a6b47d6ee000f0b346a5a1f)  

**ABSTRACT**  
By introducing a small set of additional parameters, a probe learns to solve specific linguistic tasks (e.g., dependency parsing) in a supervised manner using feature representations (e.g., contextualized embeddings). The effectiveness of such probing tasks is taken as evidence that the pre-trained model encodes linguistic knowledge. However, this approach of evaluating a language model is undermined by the uncertainty of the amount of knowledge that is learned by the probe itself. Complementary to those works, we propose a parameter-free probing technique for analyzing pre-trained language models (e.g., BERT). Our method does not require direct supervision from the probing tasks, nor do we introduce additional parameters to the probing process. Our experiments on BERT show that syntactic trees recovered from BERT using our method are significantly better than linguistically-uninformed baselines. We further feed the empirically induced dependency structures into a downstream sentiment classification task and find its improvement compatible with or even superior to a human-designed dependency schema.

{{< /ci-details >}}

{{< ci-details summary="Cross-Lingual Ability of Multilingual BERT: An Empirical Study (Karthikeyan K et al., 2019)">}}

Karthikeyan K, Zihan Wang, Stephen Mayhew, D. Roth. (2019)  
**Cross-Lingual Ability of Multilingual BERT: An Empirical Study**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/3b2538f84812f434c740115c185be3e5e216c526)  
Influential Citation Count (7), SS-ID (3b2538f84812f434c740115c185be3e5e216c526)  

**ABSTRACT**  
Recent work has exhibited the surprising cross-lingual abilities of multilingual BERT (M-BERT) -- surprising since it is trained without any cross-lingual objective and with no aligned data. In this work, we provide a comprehensive study of the contribution of different components in M-BERT to its cross-lingual ability. We study the impact of linguistic properties of the languages, the architecture of the model, and the learning objectives. The experimental study is done in the context of three typologically different languages -- Spanish, Hindi, and Russian -- and using two conceptually different NLP tasks, textual entailment and named entity recognition. Among our key conclusions is the fact that the lexical overlap between languages plays a negligible role in the cross-lingual success, while the depth of the network is an integral part of it. All our models and implementations can be found on our project page: this http URL .

{{< /ci-details >}}

{{< ci-details summary="Reducing BERT Pre-Training Time from 3 Days to 76 Minutes (Yang You et al., 2019)">}}

Yang You, Jing Li, Jonathan Hseu, Xiaodan Song, J. Demmel, Cho-Jui Hsieh. (2019)  
**Reducing BERT Pre-Training Time from 3 Days to 76 Minutes**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/3c6dca9041f54583aeab60587c9e6e9272104dc1)  
Influential Citation Count (13), SS-ID (3c6dca9041f54583aeab60587c9e6e9272104dc1)  

**ABSTRACT**  
Large-batch training is key to speeding up deep neural network training in large distributed systems. However, large-batch training is difficult because it produces a generalization gap. Straightforward optimization often leads to accuracy loss on the test set. BERT \cite{devlin2018bert} is a state-of-the-art deep learning model that builds on top of deep bidirectional transformers for language understanding. Previous large-batch training techniques do not perform well for BERT when we scale the batch size (e.g. beyond 8192). BERT pre-training also takes a long time to finish (around three days on 16 TPUv3 chips). To solve this problem, we propose the LAMB optimizer, which helps us to scale the batch size to 65536 without losing accuracy. LAMB is a general optimizer that works for both small and large batch sizes and does not need hyper-parameter tuning besides the learning rate. The baseline BERT-Large model needs 1 million iterations to finish pre-training, while LAMB with batch size 65536/32768 only needs 8599 iterations. We push the batch size to the memory limit of a TPUv3 pod and can finish BERT training in 76 minutes.

{{< /ci-details >}}

{{< ci-details summary="Investigating BERT’s Knowledge of Language: Five Analysis Methods with NPIs (Alex Warstadt et al., 2019)">}}

Alex Warstadt, Yuning Cao, Ioana Grosu, Wei Peng, Hagen Blix, Yining Nie, Anna Alsop, Shikha Bordia, Haokun Liu, Alicia Parrish, Sheng-Fu Wang, Jason Phang, Anhad Mohananey, Phu Mon Htut, Paloma Jeretic, Samuel R. Bowman. (2019)  
**Investigating BERT’s Knowledge of Language: Five Analysis Methods with NPIs**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/3cd331c997e90f737810aad6fcce4d993315189f)  
Influential Citation Count (4), SS-ID (3cd331c997e90f737810aad6fcce4d993315189f)  

**ABSTRACT**  
Though state-of-the-art sentence representation models can perform tasks requiring significant knowledge of grammar, it is an open question how best to evaluate their grammatical knowledge. We explore five experimental methods inspired by prior work evaluating pretrained sentence representation models. We use a single linguistic phenomenon, negative polarity item (NPI) licensing, as a case study for our experiments. NPIs like any are grammatical only if they appear in a licensing environment like negation (Sue doesn’t have any cats vs. *Sue has any cats). This phenomenon is challenging because of the variety of NPI licensing environments that exist. We introduce an artificially generated dataset that manipulates key features of NPI licensing for the experiments. We find that BERT has significant knowledge of these features, but its success varies widely across different experimental methods. We conclude that a variety of methods is necessary to reveal all relevant aspects of a model’s grammatical knowledge in a given domain.

{{< /ci-details >}}

{{< ci-details summary="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Colin Raffel et al., 2019)">}}

Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. (2019)  
**Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**  
J. Mach. Learn. Res.  
[Paper Link](https://www.semanticscholar.org/paper/3cfb319689f06bf04c2e28399361f414ca32c4b3)  
Influential Citation Count (615), SS-ID (3cfb319689f06bf04c2e28399361f414ca32c4b3)  

**ABSTRACT**  
Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new "Colossal Clean Crawled Corpus", we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.

{{< /ci-details >}}

{{< ci-details summary="Deepening Hidden Representations from Pre-trained Language Models for Natural Language Understanding (Jie Yang et al., 2019)">}}

Jie Yang, Hai Zhao. (2019)  
**Deepening Hidden Representations from Pre-trained Language Models for Natural Language Understanding**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/3ee7b17cc627ac5bc99632a22ef820dc559393e6)  
Influential Citation Count (0), SS-ID (3ee7b17cc627ac5bc99632a22ef820dc559393e6)  

**ABSTRACT**  
Transformer-based pre-trained language models have proven to be effective for learning contextualized language representation. However, current approaches only take advantage of the output of the encoder's final layer when fine-tuning the downstream tasks. We argue that only taking single layer's output restricts the power of pre-trained representation. Thus we deepen the representation learned by the model by fusing the hidden representation in terms of an explicit HIdden Representation Extractor (HIRE), which automatically absorbs the complementary representation with respect to the output from the final layer. Utilizing RoBERTa as the backbone encoder, our proposed improvement over the pre-trained models is shown effective on multiple natural language understanding tasks and help our model rival with the state-of-the-art models on the GLUE benchmark.

{{< /ci-details >}}

{{< ci-details summary="Improving Transformer Models by Reordering their Sublayers (Ofir Press et al., 2019)">}}

Ofir Press, Noah A. Smith, Omer Levy. (2019)  
**Improving Transformer Models by Reordering their Sublayers**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/3ff8d265f4351e4b1fdac5b586466bee0b5d6fff)  
Influential Citation Count (2), SS-ID (3ff8d265f4351e4b1fdac5b586466bee0b5d6fff)  

**ABSTRACT**  
Multilayer transformer networks consist of interleaved self-attention and feedforward sublayers. Could ordering the sublayers in a different pattern lead to better performance? We generate randomly ordered transformers and train them with the language modeling objective. We observe that some of these models are able to achieve better performance than the interleaved baseline, and that those successful variants tend to have more self-attention at the bottom and more feedforward sublayers at the top. We propose a new transformer pattern that adheres to this property, the sandwich transformer, and show that it improves perplexity on multiple word-level and character-level language modeling benchmarks, at no cost in parameters, memory, or training time. However, the sandwich reordering pattern does not guarantee performance gains across every task, as we demonstrate on machine translation models. Instead, we suggest that further exploration of task-specific sublayer reorderings is needed in order to unlock additional gains.

{{< /ci-details >}}

{{< ci-details summary="Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference (R. Thomas McCoy et al., 2019)">}}

R. Thomas McCoy, Ellie Pavlick, Tal Linzen. (2019)  
**Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/42ed4a9994e6121a9f325f5b901c5b3d7ce104f5)  
Influential Citation Count (103), SS-ID (42ed4a9994e6121a9f325f5b901c5b3d7ce104f5)  

**ABSTRACT**  
A machine learning system can score well on a given test set by relying on heuristics that are effective for frequent example types but break down in more challenging cases. We study this issue within natural language inference (NLI), the task of determining whether one sentence entails another. We hypothesize that statistical NLI models may adopt three fallible syntactic heuristics: the lexical overlap heuristic, the subsequence heuristic, and the constituent heuristic. To determine whether models have adopted these heuristics, we introduce a controlled evaluation set called HANS (Heuristic Analysis for NLI Systems), which contains many examples where the heuristics fail. We find that models trained on MNLI, including BERT, a state-of-the-art model, perform very poorly on HANS, suggesting that they have indeed adopted these heuristics. We conclude that there is substantial room for improvement in NLI systems, and that the HANS dataset can motivate and measure progress in this area.

{{< /ci-details >}}

{{< ci-details summary="PoWER-BERT: Accelerating BERT inference for Classification Tasks (Saurabh Goyal et al., 2020)">}}

Saurabh Goyal, Anamitra R. Choudhury, Venkatesan T. Chakaravarthy, Saurabh ManishRaje, Yogish Sabharwal, Ashish Verma. (2020)  
**PoWER-BERT: Accelerating BERT inference for Classification Tasks**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/4510d9ad22f474c30c530ae7f886ec4d42402d68)  
Influential Citation Count (1), SS-ID (4510d9ad22f474c30c530ae7f886ec4d42402d68)  

**ABSTRACT**  
BERT has emerged as a popular model for natural language understanding. Given its computeintensive nature, even for inference, many recent studies have considered optimization of two important performance characteristics: model size and inference time. We consider classification tasks and propose a novel method, called PoWER-BERT, for improving the inference time for the BERT model without significant loss in the accuracy. The method works by eliminating word-vectors (intermediate vector outputs) from the encoder pipeline. We design a strategy for measuring the significance of the word-vectors based on the self-attention mechanism of the encoders which helps us identify the word-vectors to be eliminated. Experimental evaluation on the standard GLUE benchmark shows that PoWER-BERT achieves up to 4.5x reduction in inference time over BERT with < 1% loss in accuracy. We show that compared to the prior inference time reduction methods, PoWER-BERT offers better trade-off between accuracy and inference time. Lastly, we demonstrate that our scheme can also be used in conjunction with ALBERT (a highly compressed version of BERT) and can attain up to 6.8x factor reduction in inference time with < 1% loss in accuracy.

{{< /ci-details >}}

{{< ci-details summary="A Structural Probe for Finding Syntax in Word Representations (John Hewitt et al., 2019)">}}

John Hewitt, Christopher D. Manning. (2019)  
**A Structural Probe for Finding Syntax in Word Representations**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/455a8838cde44f288d456d01c76ede95b56dc675)  
Influential Citation Count (30), SS-ID (455a8838cde44f288d456d01c76ede95b56dc675)  

**ABSTRACT**  
Recent work has improved our ability to detect linguistic knowledge in word representations. However, current methods for detecting syntactic knowledge do not test whether syntax trees are represented in their entirety. In this work, we propose a structural probe, which evaluates whether syntax trees are embedded in a linear transformation of a neural network’s word representation space. The probe identifies a linear transformation under which squared L2 distance encodes the distance between words in the parse tree, and one in which squared L2 norm encodes depth in the parse tree. Using our probe, we show that such transformations exist for both ELMo and BERT but not in baselines, providing evidence that entire syntax trees are embedded implicitly in deep models’ vector geometry.

{{< /ci-details >}}

{{< ci-details summary="What do you mean, BERT? Assessing BERT as a Distributional Semantics Model (Timothee Mickus et al., 2019)">}}

Timothee Mickus, Denis Paperno, Mathieu Constant, Kees van Deemter. (2019)  
**What do you mean, BERT? Assessing BERT as a Distributional Semantics Model**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/4bff291cf7fa02a0dbac767aba55d43ad8c59055)  
Influential Citation Count (3), SS-ID (4bff291cf7fa02a0dbac767aba55d43ad8c59055)  

**ABSTRACT**  
Contextualized word embeddings, i.e. vector representations for words in context, are naturally seen as an extension of previous noncontextual distributional semantic models. In this work, we focus on BERT, a deep neural network that produces contextualized embeddings and has set the state-of-the-art in several semantic tasks, and study the semantic coherence of its embedding space. While showing a tendency towards coherence, BERT does not fully live up to the natural expectations for a semantic vector space. In particular, we find that the position of the sentence in which a word occurs, while having no meaning correlates, leaves a noticeable trace on the word embeddings and disturbs similarity relationships.

{{< /ci-details >}}

{{< ci-details summary="Structured Pruning of a BERT-based Question Answering Model (J. Scott McCarley et al., 2019)">}}

J. Scott McCarley, Rishav Chakravarti, Avirup Sil. (2019)  
**Structured Pruning of a BERT-based Question Answering Model**  
  
[Paper Link](https://www.semanticscholar.org/paper/4d8a4509753cc91832f80ec35795064e79630ef3)  
Influential Citation Count (4), SS-ID (4d8a4509753cc91832f80ec35795064e79630ef3)  

**ABSTRACT**  
The recent trend in industry-setting Natural Language Processing (NLP) research has been to operate large scale pretrained language models like BERT under strict computational limits. While most model compression work has focused on "distilling" a general-purpose language representation using expensive pretraining distillation, much less attention has been paid to creating smaller task-specific language representations which, arguably, are more useful in an industry setting. In this paper, we investigate compressing BERT- and RoBERTa-based question answering systems by structured pruning of parameters from the underlying trained transformer model. We find that an inexpensive combination of task-specific structured pruning and task-specific distillation, without the expense of pretraining distillation, yields highly-performing models across a range of speed/accuracy tradeoff operating points. We start from full-size models trained for SQuAD 2.0 or Natural Questions and introduce gates that allow selected parts of transformers to be individually eliminated. Specifically, we investigate (1) structured pruning to reduce the number of parameters in each transformer layer, (2) applicability to both BERT- and RoBERTa-based models, (3) applicability to both SQuAD 2.0 and Natural Questions, and (4) combining structured pruning with distillation. We find that pruning a combination of attention heads and the feed-forward layer yields a near-doubling of inference speed on SQuAD 2.0, with less than a 1.5 F1-point loss in accuracy. Furthermore, we find that a combination of distillation and structured pruning almost doubles the inference speed of RoBERTa-large based model for Natural Questions, while losing less than 0.5 F1-point on short answers.

{{< /ci-details >}}

{{< ci-details summary="K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters (Ruize Wang et al., 2020)">}}

Ruize Wang, Duyu Tang, Nan Duan, Zhongyu Wei, Xuanjing Huang, Jianshu Ji, Guihong Cao, Daxin Jiang, Ming Zhou. (2020)  
**K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters**  
FINDINGS  
[Paper Link](https://www.semanticscholar.org/paper/4f03e69963b9649950ba29ae864a0de8c14f1f86)  
Influential Citation Count (32), SS-ID (4f03e69963b9649950ba29ae864a0de8c14f1f86)  

**ABSTRACT**  
We study the problem of injecting knowledge into large pre-trained models like BERT and RoBERTa. Existing methods typically update the original parameters of pre-trained models when injecting knowledge. However, when multiple kinds of knowledge are injected, they may suffer from catastrophic forgetting. To address this, we propose K-Adapter, which remains the original parameters of the pre-trained model fixed and supports continual knowledge infusion. Taking RoBERTa as the pre-trained model, K-Adapter has a neural adapter for each kind of infused knowledge, like a plug-in connected to RoBERTa. There is no information flow between different adapters, thus different adapters are efficiently trained in a distributed way. We inject two kinds of knowledge, including factual knowledge obtained from automatically aligned text-triplets on Wikipedia and Wikidata, and linguistic knowledge obtained from dependency parsing. Results on three knowledge-driven tasks (total six datasets) including relation classification, entity typing and question answering demonstrate that each adapter improves the performance, and the combination of both adapters brings further improvements. Probing experiments further indicate that K-Adapter captures richer factual and commonsense knowledge than RoBERTa.

{{< /ci-details >}}

{{< ci-details summary="Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT (Sheng Shen et al., 2019)">}}

Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Z. Yao, A. Gholami, Michael W. Mahoney, K. Keutzer. (2019)  
**Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/4fb8fd55b476909a26a8dc594e0ae98d4923ad4d)  
Influential Citation Count (22), SS-ID (4fb8fd55b476909a26a8dc594e0ae98d4923ad4d)  

**ABSTRACT**  
Transformer based architectures have become de-facto models used for a range of Natural Language Processing tasks. In particular, the BERT based models achieved significant accuracy gain for GLUE tasks, CoNLL-03 and SQuAD. However, BERT based models have a prohibitive memory footprint and latency. As a result, deploying BERT based models in resource constrained environments has become a challenging task. In this work, we perform an extensive analysis of fine-tuned BERT models using second order Hessian information, and we use our results to propose a novel method for quantizing BERT models to ultra low precision. In particular, we propose a new group-wise quantization scheme, and we use Hessian-based mix-precision method to compress the model further. We extensively test our proposed method on BERT downstream tasks of SST-2, MNLI, CoNLL-03, and SQuAD. We can achieve comparable performance to baseline with at most 2.3% performance degradation, even with ultra-low precision quantization down to 2 bits, corresponding up to 13× compression of the model parameters, and up to 4× compression of the embedding table as well as activations. Among all tasks, we observed the highest performance loss for BERT fine-tuned on SQuAD. By probing into the Hessian based analysis as well as visualization, we show that this is related to the fact that current training/fine-tuning strategy of BERT does not converge for SQuAD.

{{< /ci-details >}}

{{< ci-details summary="Parsing as Pretraining (David Vilares et al., 2020)">}}

David Vilares, Michalina Strzyz, Anders Søgaard, Carlos G'omez-Rodr'iguez. (2020)  
**Parsing as Pretraining**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/50be7d2858523d0e63174d974f380349fca0d666)  
Influential Citation Count (2), SS-ID (50be7d2858523d0e63174d974f380349fca0d666)  

**ABSTRACT**  
Recent analyses suggest that encoders pretrained for language modeling capture certain morpho-syntactic structure. However, probing frameworks for word vectors still do not report results on standard setups such as constituent and dependency parsing. This paper addresses this problem and does full parsing (on English) relying only on pretraining architectures – and no decoding. We first cast constituent and dependency parsing as sequence tagging. We then use a single feed-forward layer to directly map word vectors to labels that encode a linearized tree. This is used to: (i) see how far we can reach on syntax modelling with just pretrained encoders, and (ii) shed some light about the syntax-sensitivity of different word vectors (by freezing the weights of the pretraining network during training). For evaluation, we use bracketing F1-score and las, and analyze in-depth differences across representations for span lengths and dependency displacements. The overall results surpass existing sequence tagging parsers on the ptb (93.5%) and end-to-end en-ewt ud (78.8%).

{{< /ci-details >}}

{{< ci-details summary="Reweighted Proximal Pruning for Large-Scale Language Representation (Fu-Ming Guo et al., 2019)">}}

Fu-Ming Guo, Sijia Liu, F. Mungall, Xue Lin, Yanzhi Wang. (2019)  
**Reweighted Proximal Pruning for Large-Scale Language Representation**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/540f074cb6f16563a357741837e41c44c0a38234)  
Influential Citation Count (3), SS-ID (540f074cb6f16563a357741837e41c44c0a38234)  

**ABSTRACT**  
Recently, pre-trained language representation flourishes as the mainstay of the natural language understanding community, e.g., BERT. These pre-trained language representations can create state-of-the-art results on a wide range of downstream tasks. Along with continuous significant performance improvement, the size and complexity of these pre-trained neural models continue to increase rapidly. Is it possible to compress these large-scale language representation models? How will the pruned language representation affect the downstream multi-task transfer learning objectives? In this paper, we propose Reweighted Proximal Pruning (RPP), a new pruning method specifically designed for a large-scale language representation model. Through experiments on SQuAD and the GLUE benchmark suite, we show that proximal pruned BERT keeps high accuracy for both the pre-training task and the downstream multiple fine-tuning tasks at high prune ratio. RPP provides a new perspective to help us analyze what large-scale language representation might learn. Additionally, RPP makes it possible to deploy a large state-of-the-art language representation model such as BERT on a series of distinct devices (e.g., online servers, mobile phones, and edge devices).

{{< /ci-details >}}

{{< ci-details summary="The Berkeley FrameNet Project (Collin F. Baker et al., 1998)">}}

Collin F. Baker, C. Fillmore, J. Lowe. (1998)  
**The Berkeley FrameNet Project**  
COLING-ACL  
[Paper Link](https://www.semanticscholar.org/paper/547f23597f9ec8a93f66cedaa6fbfb73960426b1)  
Influential Citation Count (436), SS-ID (547f23597f9ec8a93f66cedaa6fbfb73960426b1)  

**ABSTRACT**  
FrameNet is a three-year NSF-supported project in corpus-based computational lexicography, now in its second year (NSF IRI-9618838, "Tools for Lexicon Building"). The project's key features are (a) a commitment to corpus evidence for semantic and syntactic generalizations, and (b) the representation of the valences of its target words (mostly nouns, adjectives, and verbs) in which the semantic portion makes use of frame semantics. The resulting database will contain (a) descriptions of the semantic frames underlying the meanings of the words described, and (b) the valence representation (semantic and syntactic) of several thousand words and phrases, each accompanied by (c) a representative collection of annotated corpus attestations, which jointly exemplify the observed linkings between "frame elements" and their syntactic realizations (e.g. grammatical function, phrase type, and other syntactic traits). This report will present the project's goals and workflow, and information about the computational tools that have been adapted or created in-house for this work.

{{< /ci-details >}}

{{< ci-details summary="SesameBERT: Attention for Anywhere (Ta-Chun Su et al., 2019)">}}

Ta-Chun Su, Hsiang-Chih Cheng. (2019)  
**SesameBERT: Attention for Anywhere**  
2020 IEEE 7th International Conference on Data Science and Advanced Analytics (DSAA)  
[Paper Link](https://www.semanticscholar.org/paper/553c1048e90e84575cad9016f367cf69c52a7fd7)  
Influential Citation Count (0), SS-ID (553c1048e90e84575cad9016f367cf69c52a7fd7)  

**ABSTRACT**  
Fine-tuning with pre-trained models has achieved exceptional results for many language tasks. In this study, we focused on one such self-attention network model, namely BERT, which has performed well in terms of stacking layers across diverse language-understanding benchmarks. However, in many downstream tasks, information between layers is ignored by BERT for fine-tuning. In addition, although self-attention networks are well-known for their ability to capture global dependencies, room for improvement remains in terms of emphasizing the importance of local contexts. In light of these advantages and disadvantages, this paper proposes SesameBERT, a generalized fine-tuning method that (1) enables the extraction of global information among all layers through Squeeze and Excitation and (2) enriches local information by capturing neighboring contexts via Gaussian blurring. Furthermore, we demonstrated the effectiveness of our approach in the HANS dataset, which is used to determine whether models have adopted shallow heuristics instead of learning underlying generalizations. The experiments revealed that SesameBERT outperformed BERT with respect to GLUE benchmark and the HANS evaluation set.

{{< /ci-details >}}

{{< ci-details summary="Association for Computational Linguistics (D. Litman et al., 2001)">}}

D. Litman, J. Hirschberg, M. Swerts, Scott Miller, L. Ramshaw, R. Weischedel, Eugene Charniak, Lillian Lee. (2001)  
**Association for Computational Linguistics**  
  
[Paper Link](https://www.semanticscholar.org/paper/566eb7be43b8a2b2daff82b03711098a84859b2a)  
Influential Citation Count (67), SS-ID (566eb7be43b8a2b2daff82b03711098a84859b2a)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation (Xiaozhi Wang et al., 2019)">}}

Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhiyuan Liu, Juan-Zi Li, Jian Tang. (2019)  
**KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation**  
Transactions of the Association for Computational Linguistics  
[Paper Link](https://www.semanticscholar.org/paper/56cafbac34f2bb3f6a9828cd228ff281b810d6bb)  
Influential Citation Count (40), SS-ID (56cafbac34f2bb3f6a9828cd228ff281b810d6bb)  

**ABSTRACT**  
Abstract Pre-trained language representation models (PLMs) cannot well capture factual knowledge from text. In contrast, knowledge embedding (KE) methods can effectively represent the relational facts in knowledge graphs (KGs) with informative entity embeddings, but conventional KE models cannot take full advantage of the abundant textual information. In this paper, we propose a unified model for Knowledge Embedding and Pre-trained LanguagERepresentation (KEPLER), which can not only better integrate factual knowledge into PLMs but also produce effective text-enhanced KE with the strong PLMs. In KEPLER, we encode textual entity descriptions with a PLM as their embeddings, and then jointly optimize the KE and language modeling objectives. Experimental results show that KEPLER achieves state-of-the-art performances on various NLP tasks, and also works remarkably well as an inductive KE model on KG link prediction. Furthermore, for pre-training and evaluating KEPLER, we construct Wikidata5M1 , a large-scale KG dataset with aligned entity descriptions, and benchmark state-of-the-art KE methods on it. It shall serve as a new KE benchmark and facilitate the research on large KG, inductive KE, and KG with text. The source code can be obtained from https://github.com/THU-KEG/KEPLER.

{{< /ci-details >}}

{{< ci-details summary="Semantics-aware BERT for Language Understanding (Zhuosheng Zhang et al., 2019)">}}

Zhuosheng Zhang, Yuwei Wu, Zhao Hai, Z. Li, Shuailiang Zhang, Xi Zhou, Xiang Zhou. (2019)  
**Semantics-aware BERT for Language Understanding**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/5744f56d3253bd7c4341d36de40a93fceaa266b3)  
Influential Citation Count (27), SS-ID (5744f56d3253bd7c4341d36de40a93fceaa266b3)  

**ABSTRACT**  
The latest work on language representations carefully integrates contextualized features into language model training, which enables a series of success especially in various machine reading comprehension and natural language inference tasks. However, the existing language representation models including ELMo, GPT and BERT only exploit plain context-sensitive features such as character or word embeddings. They rarely consider incorporating structured semantic information which can provide rich semantics for language representation. To promote natural language understanding, we propose to incorporate explicit contextual semantics from pre-trained semantic role labeling, and introduce an improved language representation model, Semantics-aware BERT (SemBERT), which is capable of explicitly absorbing contextual semantics over a BERT backbone. SemBERT keeps the convenient usability of its BERT precursor in a light fine-tuning way without substantial task-specific modifications. Compared with BERT, semantics-aware BERT is as simple in concept but more powerful. It obtains new state-of-the-art or substantially improves results on ten reading comprehension and language inference tasks.

{{< /ci-details >}}

{{< ci-details summary="Is Supervised Syntactic Parsing Beneficial for Language Understanding Tasks? An Empirical Investigation (Goran Glavas et al., 2020)">}}

Goran Glavas, Ivan Vulic. (2020)  
**Is Supervised Syntactic Parsing Beneficial for Language Understanding Tasks? An Empirical Investigation**  
EACL  
[Paper Link](https://www.semanticscholar.org/paper/575ac3f36e9fddeb258e2f639e26a6a7ec35160a)  
Influential Citation Count (0), SS-ID (575ac3f36e9fddeb258e2f639e26a6a7ec35160a)  

**ABSTRACT**  
Traditional NLP has long held (supervised) syntactic parsing necessary for successful higher-level semantic language understanding (LU). The recent advent of end-to-end neural models, self-supervised via language modeling (LM), and their success on a wide range of LU tasks, however, questions this belief. In this work, we empirically investigate the usefulness of supervised parsing for semantic LU in the context of LM-pretrained transformer networks. Relying on the established fine-tuning paradigm, we first couple a pretrained transformer with a biaffine parsing head, aiming to infuse explicit syntactic knowledge from Universal Dependencies treebanks into the transformer. We then fine-tune the model for LU tasks and measure the effect of the intermediate parsing training (IPT) on downstream LU task performance. Results from both monolingual English and zero-shot language transfer experiments (with intermediate target-language parsing) show that explicit formalized syntax, injected into transformers through IPT, has very limited and inconsistent effect on downstream LU performance. Our results, coupled with our analysis of transformers’ representation spaces before and after intermediate parsing, make a significant step towards providing answers to an essential question: how (un)availing is supervised parsing for high-level semantic natural language understanding in the era of large neural models?

{{< /ci-details >}}

{{< ci-details summary="Fixed Encoder Self-Attention Patterns in Transformer-Based Machine Translation (Alessandro Raganato et al., 2020)">}}

Alessandro Raganato, Yves Scherrer, J. Tiedemann. (2020)  
**Fixed Encoder Self-Attention Patterns in Transformer-Based Machine Translation**  
FINDINGS  
[Paper Link](https://www.semanticscholar.org/paper/57f123c95ecf9d901be3a53291f53302740451e2)  
Influential Citation Count (5), SS-ID (57f123c95ecf9d901be3a53291f53302740451e2)  

**ABSTRACT**  
Transformer-based models have brought a radical change to neural machine translation. A key feature of the Transformer architecture is the so-called multi-head attention mechanism, which allows the model to focus simultaneously on different parts of the input. However, recent works have shown that most attention heads learn simple, and often redundant, positional patterns. In this paper, we propose to replace all but one attention head of each encoder layer with simple fixed – non-learnable – attentive patterns that are solely based on position and do not require any external knowledge. Our experiments with different data sizes and multiple language pairs show that fixing the attention heads on the encoder side of the Transformer at training time does not impact the translation quality and even increases BLEU scores by up to 3 points in low-resource scenarios.

{{< /ci-details >}}

{{< ci-details summary="What does BERT Learn from Multiple-Choice Reading Comprehension Datasets? (Chenglei Si et al., 2019)">}}

Chenglei Si, Shuohang Wang, Min-Yen Kan, Jing Jiang. (2019)  
**What does BERT Learn from Multiple-Choice Reading Comprehension Datasets?**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/59abe3db26b55c8837a1f2babb87350ba95ab1c0)  
Influential Citation Count (2), SS-ID (59abe3db26b55c8837a1f2babb87350ba95ab1c0)  

**ABSTRACT**  
Multiple-Choice Reading Comprehension (MCRC) requires the model to read the passage and question, and select the correct answer among the given options. Recent state-of-the-art models have achieved impressive performance on multiple MCRC datasets. However, such performance may not reflect the model's true ability of language understanding and reasoning. In this work, we adopt two approaches to investigate what BERT learns from MCRC datasets: 1) an un-readable data attack, in which we add keywords to confuse BERT, leading to a significant performance drop; and 2) an un-answerable data training, in which we train BERT on partial or shuffled input. Under un-answerable data training, BERT achieves unexpectedly high performance. Based on our experiments on the 5 key MCRC datasets - RACE, MCTest, MCScript, MCScript2.0, DREAM - we observe that 1) fine-tuned BERT mainly learns how keywords lead to correct prediction, instead of learning semantic understanding and reasoning; and 2) BERT does not need correct syntactic information to solve the task; 3) there exists artifacts in these datasets such that they can be solved even without the full context.

{{< /ci-details >}}

{{< ci-details summary="Efficient Training of BERT by Progressively Stacking (Linyuan Gong et al., 2019)">}}

Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, Tie-Yan Liu. (2019)  
**Efficient Training of BERT by Progressively Stacking**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/5a3749929bf5fb8b1f98a7b2a43c3b957bcf6c88)  
Influential Citation Count (7), SS-ID (5a3749929bf5fb8b1f98a7b2a43c3b957bcf6c88)  

**ABSTRACT**  
Unsupervised pre-training is commonly used in natural language processing: a deep neural network trained with proper unsupervised prediction tasks are shown to be effective in many downstream tasks. Because it is easy to create a large monolingual dataset by collecting data from the Web, we can train high-capacity models. Therefore, training efficiency becomes a critical issue even when using high-performance hardware. In this paper, we explore an efficient training method for the state-of-the-art bidirectional Transformer (BERT) model. By visualizing the self-attention distributions of different layers at different positions in a well-trained BERT model, we find that in most layers, the self-attention distribution will concentrate locally around its position and the start-of-sentence token. Motivated by this, we propose the stacking algorithm to transfer knowledge from a shallow model to a deep model; then we apply stacking progressively to accelerate BERT training. Experiments showed that the models trained by our training strategy achieve similar performance to models trained from scratch, but our algorithm is much faster.

{{< /ci-details >}}

{{< ci-details summary="What Does My QA Model Know? Devising Controlled Probes Using Expert Knowledge (Kyle Richardson et al., 2019)">}}

Kyle Richardson, Ashish Sabharwal. (2019)  
**What Does My QA Model Know? Devising Controlled Probes Using Expert Knowledge**  
Transactions of the Association for Computational Linguistics  
[Paper Link](https://www.semanticscholar.org/paper/5a9001cdccdb8b1de227a45eccc503d32d1a2464)  
Influential Citation Count (2), SS-ID (5a9001cdccdb8b1de227a45eccc503d32d1a2464)  

**ABSTRACT**  
Abstract Open-domain question answering (QA) involves many knowledge and reasoning challenges, but are successful QA models actually learning such knowledge when trained on benchmark QA tasks? We investigate this via several new diagnostic tasks probing whether multiple-choice QA models know definitions and taxonomic reasoning—two skills widespread in existing benchmarks and fundamental to more complex reasoning. We introduce a methodology for automatically building probe datasets from expert knowledge sources, allowing for systematic control and a comprehensive evaluation. We include ways to carefully control for artifacts that may arise during this process. Our evaluation confirms that transformer-based multiple-choice QA models are already predisposed to recognize certain types of structural linguistic knowledge. However, it also reveals a more nuanced picture: their performance notably degrades even with a slight increase in the number of “hops” in the underlying taxonomic hierarchy, and with more challenging distractor candidates. Further, existing models are far from perfect when assessed at the level of clusters of semantically connected probes, such as all hypernym questions about a single concept.

{{< /ci-details >}}

{{< ci-details summary="How Language-Neutral is Multilingual BERT? (Jindřich Libovický et al., 2019)">}}

Jindřich Libovický, Rudolf Rosa, Alexander M. Fraser. (2019)  
**How Language-Neutral is Multilingual BERT?**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/5d8beeca1a2e3263b2796e74e2f57ffb579737ee)  
Influential Citation Count (12), SS-ID (5d8beeca1a2e3263b2796e74e2f57ffb579737ee)  

**ABSTRACT**  
Multilingual BERT (mBERT) provides sentence representations for 104 languages, which are useful for many multi-lingual tasks. Previous work probed the cross-linguality of mBERT using zero-shot transfer learning on morphological and syntactic tasks. We instead focus on the semantic properties of mBERT. We show that mBERT representations can be split into a language-specific component and a language-neutral component, and that the language-neutral component is sufficiently general in terms of modeling semantics to allow high-accuracy word-alignment and sentence retrieval but is not yet good enough for the more difficult task of MT quality estimation. Our work presents interesting challenges which must be solved to build better language-neutral representations, particularly for tasks requiring linguistic transfer of semantics.

{{< /ci-details >}}

{{< ci-details summary="Transformers to Learn Hierarchical Contexts in Multiparty Dialogue for Span-based Question Answering (Changmao Li et al., 2020)">}}

Changmao Li, Jinho D. Choi. (2020)  
**Transformers to Learn Hierarchical Contexts in Multiparty Dialogue for Span-based Question Answering**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/5dd520b6c92aae3fd76df5bb61014e50fab93817)  
Influential Citation Count (3), SS-ID (5dd520b6c92aae3fd76df5bb61014e50fab93817)  

**ABSTRACT**  
We introduce a novel approach to transformers that learns hierarchical representations in multiparty dialogue. First, three language modeling tasks are used to pre-train the transformers, token- and utterance-level language modeling and utterance order prediction, that learn both token and utterance embeddings for better understanding in dialogue contexts. Then, multi-task learning between the utterance prediction and the token span prediction is applied to fine-tune for span-based question answering (QA). Our approach is evaluated on the FriendsQA dataset and shows improvements of 3.8% and 1.4% over the two state-of-the-art transformer models, BERT and RoBERTa, respectively.

{{< /ci-details >}}

{{< ci-details summary="oLMpics-On What Language Model Pre-training Captures (Alon Talmor et al., 2019)">}}

Alon Talmor, Yanai Elazar, Yoav Goldberg, Jonathan Berant. (2019)  
**oLMpics-On What Language Model Pre-training Captures**  
Transactions of the Association for Computational Linguistics  
[Paper Link](https://www.semanticscholar.org/paper/5e0cffc51e8b64a8f11326f955fa4b4f1803e3be)  
Influential Citation Count (17), SS-ID (5e0cffc51e8b64a8f11326f955fa4b4f1803e3be)  

**ABSTRACT**  
Abstract Recent success of pre-trained language models (LMs) has spurred widespread interest in the language capabilities that they possess. However, efforts to understand whether LM representations are useful for symbolic reasoning tasks have been limited and scattered. In this work, we propose eight reasoning tasks, which conceptually require operations such as comparison, conjunction, and composition. A fundamental challenge is to understand whether the performance of a LM on a task should be attributed to the pre-trained representations or to the process of fine-tuning on the task data. To address this, we propose an evaluation protocol that includes both zero-shot evaluation (no fine-tuning), as well as comparing the learning curve of a fine-tuned LM to the learning curve of multiple controls, which paints a rich picture of the LM capabilities. Our main findings are that: (a) different LMs exhibit qualitatively different reasoning abilities, e.g., RoBERTa succeeds in reasoning tasks where BERT fails completely; (b) LMs do not reason in an abstract manner and are context-dependent, e.g., while RoBERTa can compare ages, it can do so only when the ages are in the typical range of human ages; (c) On half of our reasoning tasks all models fail completely. Our findings and infrastructure can help future work on designing new datasets, models, and objective functions for pre-training.

{{< /ci-details >}}

{{< ci-details summary="On Measuring Social Biases in Sentence Encoders (Chandler May et al., 2019)">}}

Chandler May, Alex Wang, Shikha Bordia, Samuel R. Bowman, Rachel Rudinger. (2019)  
**On Measuring Social Biases in Sentence Encoders**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/5e9c85235210b59a16bdd84b444a904ae271f7e7)  
Influential Citation Count (25), SS-ID (5e9c85235210b59a16bdd84b444a904ae271f7e7)  

**ABSTRACT**  
The Word Embedding Association Test shows that GloVe and word2vec word embeddings exhibit human-like implicit biases based on gender, race, and other social constructs (Caliskan et al., 2017). Meanwhile, research on learning reusable text representations has begun to explore sentence-level texts, with some sentence encoders seeing enthusiastic adoption. Accordingly, we extend the Word Embedding Association Test to measure bias in sentence encoders. We then test several sentence encoders, including state-of-the-art methods such as ELMo and BERT, for the social biases studied in prior work and two important biases that are difficult or impossible to test at the word level. We observe mixed results including suspicious patterns of sensitivity that suggest the test’s assumptions may not hold in general. We conclude by proposing directions for future work on measuring bias in sentence encoders.

{{< /ci-details >}}

{{< ci-details summary="ERNIE: Enhanced Language Representation with Informative Entities (Zhengyan Zhang et al., 2019)">}}

Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, Qun Liu. (2019)  
**ERNIE: Enhanced Language Representation with Informative Entities**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/5f994dc8cae24ca9d1ed629e517fcc652660ddde)  
Influential Citation Count (90), SS-ID (5f994dc8cae24ca9d1ed629e517fcc652660ddde)  

**ABSTRACT**  
Neural language representation models such as BERT pre-trained on large-scale corpora can well capture rich semantic patterns from plain text, and be fine-tuned to consistently improve the performance of various NLP tasks. However, the existing pre-trained language models rarely consider incorporating knowledge graphs (KGs), which can provide rich structured knowledge facts for better language understanding. We argue that informative entities in KGs can enhance language representation with external knowledge. In this paper, we utilize both large-scale textual corpora and KGs to train an enhanced language representation model (ERNIE), which can take full advantage of lexical, syntactic, and knowledge information simultaneously. The experimental results have demonstrated that ERNIE achieves significant improvements on various knowledge-driven tasks, and meanwhile is comparable with the state-of-the-art model BERT on other common NLP tasks. The code and datasets will be available in the future.

{{< /ci-details >}}

{{< ci-details summary="MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers (Wenhui Wang et al., 2020)">}}

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, Ming Zhou. (2020)  
**MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/60a4a3a886338d0c8e3579d392cb32f493430255)  
Influential Citation Count (42), SS-ID (60a4a3a886338d0c8e3579d392cb32f493430255)  

**ABSTRACT**  
Pre-trained language models (e.g., BERT (Devlin et al., 2018) and its variants) have achieved remarkable success in varieties of NLP tasks. However, these models usually consist of hundreds of millions of parameters which brings challenges for fine-tuning and online serving in real-life applications due to latency and capacity constraints. In this work, we present a simple and effective approach to compress large Transformer (Vaswani et al., 2017) based pre-trained models, termed as deep self-attention distillation. The small model (student) is trained by deeply mimicking the self-attention module, which plays a vital role in Transformer networks, of the large model (teacher). Specifically, we propose distilling the self-attention module of the last Transformer layer of the teacher, which is effective and flexible for the student. Furthermore, we introduce the scaled dot-product between values in the self-attention module as the new deep self-attention knowledge, in addition to the attention distributions (i.e., the scaled dot-product of queries and keys) that have been used in existing works. Moreover, we show that introducing a teacher assistant (Mirzadeh et al., 2019) also helps the distillation of large pre-trained Transformer models. Experimental results demonstrate that our monolingual model outperforms state-of-the-art baselines in different parameter size of student models. In particular, it retains more than 99% accuracy on SQuAD 2.0 and several GLUE benchmark tasks using 50% of the Transformer parameters and computations of the teacher model. We also obtain competitive results in applying deep self-attention distillation to multilingual pre-trained models.

{{< /ci-details >}}

{{< ci-details summary="Symmetric Regularization based BERT for Pair-wise Semantic Reasoning (Xingyi Cheng et al., 2019)">}}

Xingyi Cheng, Weidi Xu, Kunlong Chen, Wei Wang, Bin Bi, Ming Yan, Chen Wu, Luo Si, Wei Chu, Taifeng Wang. (2019)  
**Symmetric Regularization based BERT for Pair-wise Semantic Reasoning**  
SIGIR  
[Paper Link](https://www.semanticscholar.org/paper/63f9e2417563456f91c7e5586d43eb25c00a0c19)  
Influential Citation Count (1), SS-ID (63f9e2417563456f91c7e5586d43eb25c00a0c19)  

**ABSTRACT**  
The ability of semantic reasoning over the sentence pair is essential for many natural language understanding tasks, e.g., natural language inference and machine reading comprehension. A recent significant improvement in these tasks comes from BERT. As reported, the next sentence prediction (NSP) in BERT is of great significance for downstream problems with sentence-pair input. Despite its effectiveness, NSP still lacks the essential signal to distinguish between entailment and shallow correlation. To remedy this, we propose to augment the NSP task to a multi-class categorization task, which includes previous sentence prediction (PSP). This task encourages the model to learn the subtle semantics, thereby improves the ability of semantic understanding. Furthermore, by using a smoothing technique, the scopes of NSP and PSP are expanded into a broader range which includes close but nonsuccessive sentences. This simple method yields remarkable improvement against vanilla BERT. Our method consistently improves the performance on the NLI and MRC benchmarks by a large margin, including the challenging HANS dataset.

{{< /ci-details >}}

{{< ci-details summary="Does BERT agree? Evaluating knowledge of structure dependence through agreement relations (Geoff Bacon et al., 2019)">}}

Geoff Bacon, T. Regier. (2019)  
**Does BERT agree? Evaluating knowledge of structure dependence through agreement relations**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/645a96e5c474d919415850892880005e4ad3fb43)  
Influential Citation Count (0), SS-ID (645a96e5c474d919415850892880005e4ad3fb43)  

**ABSTRACT**  
Learning representations that accurately model semantics is an important goal of natural language processing research. Many semantic phenomena depend on syntactic structure. Recent work examines the extent to which state-of-the-art models for pre-training representations, such as BERT, capture such structure-dependent phenomena, but is largely restricted to one phenomenon in English: number agreement between subjects and verbs. We evaluate BERT's sensitivity to four types of structure-dependent agreement relations in a new semi-automatically curated dataset across 26 languages. We show that both the single-language and multilingual BERT models capture syntax-sensitive agreement patterns well in general, but we also highlight the specific linguistic contexts in which their performance degrades.

{{< /ci-details >}}

{{< ci-details summary="Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks (Haoyang Huang et al., 2019)">}}

Haoyang Huang, Yaobo Liang, Nan Duan, Ming Gong, Linjun Shou, Daxin Jiang, M. Zhou. (2019)  
**Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/65f788fb964901e3f1149a0a53317535ca85ed7d)  
Influential Citation Count (17), SS-ID (65f788fb964901e3f1149a0a53317535ca85ed7d)  

**ABSTRACT**  
We present Unicoder, a universal language encoder that is insensitive to different languages. Given an arbitrary NLP task, a model can be trained with Unicoder using training data in one language and directly applied to inputs of the same task in other languages. Comparing to similar efforts such as Multilingual BERT and XLM , three new cross-lingual pre-training tasks are proposed, including cross-lingual word recovery, cross-lingual paraphrase classification and cross-lingual masked language model. These tasks help Unicoder learn the mappings among different languages from more perspectives. We also find that doing fine-tuning on multiple languages together can bring further improvement. Experiments are performed on two tasks: cross-lingual natural language inference (XNLI) and cross-lingual question answering (XQA), where XLM is our baseline. On XNLI, 1.8% averaged accuracy improvement (on 15 languages) is obtained. On XQA, which is a new cross-lingual dataset built by us, 5.5% averaged accuracy improvement (on French and German) is obtained.

{{< /ci-details >}}

{{< ci-details summary="BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning (Asa Cooper Stickland et al., 2019)">}}

Asa Cooper Stickland, Iain Murray. (2019)  
**BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/660d3472d9c3733dedcf911187b234f2b65561b5)  
Influential Citation Count (13), SS-ID (660d3472d9c3733dedcf911187b234f2b65561b5)  

**ABSTRACT**  
Multi-task learning shares information between related tasks, sometimes reducing the number of parameters required. State-of-the-art results across multiple natural language understanding tasks in the GLUE benchmark have previously used transfer from a single large task: unsupervised pre-training with BERT, where a separate BERT model was fine-tuned for each task. We explore multi-task approaches that share a single BERT model with a small number of additional task-specific parameters. Using new adaptation modules, PALs or `projected attention layers', we match the performance of separately fine-tuned models on the GLUE benchmark with roughly 7 times fewer parameters, and obtain state-of-the-art results on the Recognizing Textual Entailment dataset.

{{< /ci-details >}}

{{< ci-details summary="Analysis Methods in Neural Language Processing: A Survey (Y. Belinkov et al., 2018)">}}

Y. Belinkov, James R. Glass. (2018)  
**Analysis Methods in Neural Language Processing: A Survey**  
TACL  
[Paper Link](https://www.semanticscholar.org/paper/668f42a4d4094f0a66d402a16087e14269b31a1f)  
Influential Citation Count (17), SS-ID (668f42a4d4094f0a66d402a16087e14269b31a1f)  

**ABSTRACT**  
The field of natural language processing has seen impressive progress in recent years, with neural network models replacing many of the traditional systems. A plethora of new models have been proposed, many of which are thought to be opaque compared to their feature-rich counterparts. This has led researchers to analyze, interpret, and evaluate neural networks in novel and more fine-grained ways. In this survey paper, we review analysis methods in neural language processing, categorize them according to prominent research trends, highlight existing limitations, and point to potential directions for future work.

{{< /ci-details >}}

{{< ci-details summary="Movement Pruning: Adaptive Sparsity by Fine-Tuning (Victor Sanh et al., 2020)">}}

Victor Sanh, Thomas Wolf, Alexander M. Rush. (2020)  
**Movement Pruning: Adaptive Sparsity by Fine-Tuning**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/66f0f35fc78bdf2af9de46093d49a428970cde2e)  
Influential Citation Count (20), SS-ID (66f0f35fc78bdf2af9de46093d49a428970cde2e)  

**ABSTRACT**  
Magnitude pruning is a widely used strategy for reducing model size in pure supervised learning; however, it is less effective in the transfer learning regime that has become standard for state-of-the-art natural language processing applications. We propose the use of movement pruning, a simple, deterministic first-order weight pruning method that is more adaptive to pretrained model fine-tuning. We give mathematical foundations to the method and compare it to existing zeroth- and first-order pruning methods. Experiments show that when pruning large pretrained language models, movement pruning shows significant improvements in high-sparsity regimes. When combined with distillation, the approach achieves minimal accuracy loss with down to only 3% of the model parameters.

{{< /ci-details >}}

{{< ci-details summary="Probing Natural Language Inference Models through Semantic Fragments (Kyle Richardson et al., 2019)">}}

Kyle Richardson, Hai Hu, L. Moss, Ashish Sabharwal. (2019)  
**Probing Natural Language Inference Models through Semantic Fragments**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/681fbcd98acf20df3355eff3585994bd1f9008b7)  
Influential Citation Count (12), SS-ID (681fbcd98acf20df3355eff3585994bd1f9008b7)  

**ABSTRACT**  
Do state-of-the-art models for language understanding already have, or can they easily learn, abilities such as boolean coordination, quantification, conditionals, comparatives, and monotonicity reasoning (i.e., reasoning about word substitutions in sentential contexts)? While such phenomena are involved in natural language inference (NLI) and go beyond basic linguistic understanding, it is unclear the extent to which they are captured in existing NLI benchmarks and effectively learned by models. To investigate this, we propose the use of semantic fragments—systematically generated datasets that each target a different semantic phenomenon—for probing, and efficiently improving, such capabilities of linguistic models. This approach to creating challenge datasets allows direct control over the semantic diversity and complexity of the targeted linguistic phenomena, and results in a more precise characterization of a model's linguistic behavior. Our experiments, using a library of 8 such semantic fragments, reveal two remarkable findings: (a) State-of-the-art models, including BERT, that are pre-trained on existing NLI benchmark datasets perform poorly on these new fragments, even though the phenomena probed here are central to the NLI task; (b) On the other hand, with only a few minutes of additional fine-tuning—with a carefully selected learning rate and a novel variation of “inoculation”—a BERT-based model can master all of these logic and monotonicity fragments while retaining its performance on established NLI benchmarks.

{{< /ci-details >}}

{{< ci-details summary="Language Models are Few-Shot Learners (Tom B. Brown et al., 2020)">}}

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, J. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. Henighan, Rewon Child, A. Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei. (2020)  
**Language Models are Few-Shot Learners**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/6b85b63579a916f705a8e10a49bd8d849d91b1fc)  
Influential Citation Count (428), SS-ID (6b85b63579a916f705a8e10a49bd8d849d91b1fc)  

**ABSTRACT**  
Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.

{{< /ci-details >}}

{{< ci-details summary="Span Selection Pre-training for Question Answering (Michael R. Glass et al., 2019)">}}

Michael R. Glass, A. Gliozzo, Rishav Chakravarti, Anthony Ferritto, Lin Pan, G P Shrivatsa Bhargav, Dinesh Garg, Avirup Sil. (2019)  
**Span Selection Pre-training for Question Answering**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/6c8503803760c5c7790f72437d0f8b874334e6f0)  
Influential Citation Count (2), SS-ID (6c8503803760c5c7790f72437d0f8b874334e6f0)  

**ABSTRACT**  
BERT (Bidirectional Encoder Representations from Transformers) and related pre-trained Transformers have provided large gains across many language understanding tasks, achieving a new state-of-the-art (SOTA). BERT is pretrained on two auxiliary tasks: Masked Language Model and Next Sentence Prediction. In this paper we introduce a new pre-training task inspired by reading comprehension to better align the pre-training from memorization to understanding. Span Selection PreTraining (SSPT) poses cloze-like training instances, but rather than draw the answer from the model’s parameters, it is selected from a relevant passage. We find significant and consistent improvements over both BERT-BASE and BERT-LARGE on multiple Machine Reading Comprehension (MRC) datasets. Specifically, our proposed model has strong empirical evidence as it obtains SOTA results on Natural Questions, a new benchmark MRC dataset, outperforming BERT-LARGE by 3 F1 points on short answer prediction. We also show significant impact in HotpotQA, improving answer prediction F1 by 4 points and supporting fact prediction F1 by 1 point and outperforming the previous best system. Moreover, we show that our pre-training approach is particularly effective when training data is limited, improving the learning curve by a large amount.

{{< /ci-details >}}

{{< ci-details summary="Unsupervised Cross-lingual Representation Learning at Scale (A. Conneau et al., 2019)">}}

A. Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov. (2019)  
**Unsupervised Cross-lingual Representation Learning at Scale**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/6fec3e579c7cd4f13bdabbee2b6ac2e8ff5941c6)  
Influential Citation Count (527), SS-ID (6fec3e579c7cd4f13bdabbee2b6ac2e8ff5941c6)  

**ABSTRACT**  
This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +14.6% average accuracy on XNLI, +13% average F1 score on MLQA, and +2.4% F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 15.7% in XNLI accuracy for Swahili and 11.4% for Urdu over previous XLM models. We also present a detailed empirical analysis of the key factors that are required to achieve these gains, including the trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing per-language performance; XLM-R is very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We will make our code and models publicly available.

{{< /ci-details >}}

{{< ci-details summary="Inducing Syntactic Trees from BERT Representations (Rudolf Rosa et al., 2019)">}}

Rudolf Rosa, D. Mareček. (2019)  
**Inducing Syntactic Trees from BERT Representations**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/71f551f0352b91ab4725c498c68610655d3b5578)  
Influential Citation Count (1), SS-ID (71f551f0352b91ab4725c498c68610655d3b5578)  

**ABSTRACT**  
We use the English model of BERT and explore how a deletion of one word in a sentence changes representations of other words. Our hypothesis is that removing a reducible word (e.g. an adjective) does not affect the representation of other words so much as removing e.g. the main verb, which makes the sentence ungrammatical and of "high surprise" for the language model. We estimate reducibilities of individual words and also of longer continuous phrases (word n-grams), study their syntax-related properties, and then also use them to induce full dependency trees.

{{< /ci-details >}}

{{< ci-details summary="Compressing Large-Scale Transformer-Based Models: A Case Study on BERT (Prakhar Ganesh et al., 2020)">}}

Prakhar Ganesh, Yao Chen, Xin Lou, Mohammad Ali Khan, Y. Yang, Deming Chen, M. Winslett, Hassan Sajjad, Preslav Nakov. (2020)  
**Compressing Large-Scale Transformer-Based Models: A Case Study on BERT**  
Transactions of the Association for Computational Linguistics  
[Paper Link](https://www.semanticscholar.org/paper/738215a396f6eee1709c6b521a6199769f0ce674)  
Influential Citation Count (5), SS-ID (738215a396f6eee1709c6b521a6199769f0ce674)  

**ABSTRACT**  
Abstract Pre-trained Transformer-based models have achieved state-of-the-art performance for various Natural Language Processing (NLP) tasks. However, these models often have billions of parameters, and thus are too resource- hungry and computation-intensive to suit low- capability devices or applications with strict latency requirements. One potential remedy for this is model compression, which has attracted considerable research attention. Here, we summarize the research in compressing Transformers, focusing on the especially popular BERT model. In particular, we survey the state of the art in compression for BERT, we clarify the current best practices for compressing large-scale Transformer models, and we provide insights into the workings of various methods. Our categorization and analysis also shed light on promising future research directions for achieving lightweight, accurate, and generic NLP models.

{{< /ci-details >}}

{{< ci-details summary="Information-Theoretic Probing for Linguistic Structure (Tiago Pimentel et al., 2020)">}}

Tiago Pimentel, Josef Valvoda, Rowan Hall Maudslay, Ran Zmigrod, Adina Williams, Ryan Cotterell. (2020)  
**Information-Theoretic Probing for Linguistic Structure**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/738c6d664aa6c3854e1aa894957bd595f621fc42)  
Influential Citation Count (16), SS-ID (738c6d664aa6c3854e1aa894957bd595f621fc42)  

**ABSTRACT**  
The success of neural networks on a diverse set of NLP tasks has led researchers to question how much these networks actually “know” about natural language. Probes are a natural way of assessing this. When probing, a researcher chooses a linguistic task and trains a supervised model to predict annotations in that linguistic task from the network’s learned representations. If the probe does well, the researcher may conclude that the representations encode knowledge related to the task. A commonly held belief is that using simpler models as probes is better; the logic is that simpler models will identify linguistic structure, but not learn the task itself. We propose an information-theoretic operationalization of probing as estimating mutual information that contradicts this received wisdom: one should always select the highest performing probe one can, even if it is more complex, since it will result in a tighter estimate, and thus reveal more of the linguistic information inherent in the representation. The experimental portion of our paper focuses on empirically estimating the mutual information between a linguistic property and BERT, comparing these estimates to several baselines. We evaluate on a set of ten typologically diverse languages often underrepresented in NLP research—plus English—totalling eleven languages. Our implementation is available in https://github.com/rycolab/info-theoretic-probing.

{{< /ci-details >}}

{{< ci-details summary="WaLDORf: Wasteless Language-model Distillation On Reading-comprehension (J. Tian et al., 2019)">}}

J. Tian, A. Kreuzer, Pai-Hung Chen, Hans-Martin Will. (2019)  
**WaLDORf: Wasteless Language-model Distillation On Reading-comprehension**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/73dd65e859d2566f3755c11cb12aff518202186a)  
Influential Citation Count (0), SS-ID (73dd65e859d2566f3755c11cb12aff518202186a)  

**ABSTRACT**  
Transformer based Very Large Language Models (VLLMs) like BERT, XLNet and RoBERTa, have recently shown tremendous performance on a large variety of Natural Language Understanding (NLU) tasks. However, due to their size, these VLLMs are extremely resource intensive and cumbersome to deploy at production time. Several recent publications have looked into various ways to distil knowledge from a transformer based VLLM (most commonly BERT-Base) into a smaller model which can run much faster at inference time. Here, we propose a novel set of techniques which together produce a task-specific hybrid convolutional and transformer model, WaLDORf, that achieves state-of-the-art inference speed while still being more accurate than previous distilled models.

{{< /ci-details >}}

{{< ci-details summary="Well-Read Students Learn Better: On the Importance of Pre-training Compact Models (Iulia Turc et al., 2019)">}}

Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019)  
**Well-Read Students Learn Better: On the Importance of Pre-training Compact Models**  
  
[Paper Link](https://www.semanticscholar.org/paper/7402b604f14b8b91c53ed6eed04af92c59636c97)  
Influential Citation Count (37), SS-ID (7402b604f14b8b91c53ed6eed04af92c59636c97)  

**ABSTRACT**  
Recent developments in natural language representations have been accompanied by large and expensive models that leverage vast amounts of general-domain text through self-supervised pre-training. Due to the cost of applying such models to down-stream tasks, several model compression techniques on pre-trained language representations have been proposed (Sun et al., 2019; Sanh, 2019). However, surprisingly, the simple baseline of just pre-training and fine-tuning compact models has been overlooked. In this paper, we first show that pre-training remains important in the context of smaller architectures, and fine-tuning pre-trained compact models can be competitive to more elaborate methods proposed in concurrent work. Starting with pre-trained compact models, we then explore transferring task knowledge from large fine-tuned models through standard knowledge distillation. The resulting simple, yet effective and general algorithm, Pre-trained Distillation, brings further improvements. Through extensive experiments, we more generally explore the interaction between pre-training and distillation under two variables that have been under-studied: model size and properties of unlabeled task data. One surprising observation is that they have a compound effect even when sequentially applied on the same data. To accelerate future research, we will make our 24 pre-trained miniature BERT models publicly available.

{{< /ci-details >}}

{{< ci-details summary="Extreme Language Model Compression with Optimal Subwords and Shared Projections (Sanqiang Zhao et al., 2019)">}}

Sanqiang Zhao, Raghav Gupta, Yang Song, Denny Zhou. (2019)  
**Extreme Language Model Compression with Optimal Subwords and Shared Projections**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/740e4599b0e3113ad804cee4394c7fa7c0e96ca5)  
Influential Citation Count (6), SS-ID (740e4599b0e3113ad804cee4394c7fa7c0e96ca5)  

**ABSTRACT**  
Pre-trained deep neural network language models such as ELMo, GPT, BERT and XLNet have recently achieved state-of-the-art performance on a variety of language understanding tasks. However, their size makes them impractical for a number of scenarios, especially on mobile and edge devices. In particular, the input word embedding matrix accounts for a significant proportion of the model's memory footprint, due to the large input vocabulary and embedding dimensions. Knowledge distillation techniques have had success at compressing large neural network models, but they are ineffective at yielding student models with vocabularies different from the original teacher models. We introduce a novel knowledge distillation technique for training a student model with a significantly smaller vocabulary as well as lower embedding and hidden state dimensions. Specifically, we employ a dual-training mechanism that trains the teacher and student models simultaneously to obtain optimal word embeddings for the student vocabulary. We combine this approach with learning shared projection matrices that transfer layer-wise knowledge from the teacher model to the student model. Our method is able to compress the BERT_BASE model by more than 60x, with only a minor drop in downstream task metrics, resulting in a language model with a footprint of under 7MB. Experimental results also demonstrate higher compression efficiency and accuracy when compared with other state-of-the-art compression techniques.

{{< /ci-details >}}

{{< ci-details summary="ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (Zhenzhong Lan et al., 2019)">}}

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. (2019)  
**ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/7a064df1aeada7e69e5173f7d4c8606f4470365b)  
Influential Citation Count (573), SS-ID (7a064df1aeada7e69e5173f7d4c8606f4470365b)  

**ABSTRACT**  
Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations and longer training times. To address these problems, we present two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT. We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and \squad benchmarks while having fewer parameters compared to BERT-large. The code and the pretrained models are available at this https URL.

{{< /ci-details >}}

{{< ci-details summary="BERT is Not a Knowledge Base (Yet): Factual Knowledge vs. Name-Based Reasoning in Unsupervised QA (Nina Poerner et al., 2019)">}}

Nina Poerner, Ulli Waltinger, Hinrich Schütze. (2019)  
**BERT is Not a Knowledge Base (Yet): Factual Knowledge vs. Name-Based Reasoning in Unsupervised QA**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/7c62ac7aedacc39ca417a48f8134e0514dc6a523)  
Influential Citation Count (8), SS-ID (7c62ac7aedacc39ca417a48f8134e0514dc6a523)  

**ABSTRACT**  
The BERT language model (LM) (Devlin et al., 2019) is surprisingly good at answering cloze-style questions about relational facts. Petroni et al. (2019) take this as evidence that BERT memorizes factual knowledge during pre-training. We take issue with this interpretation and argue that the performance of BERT is partly due to reasoning about (the surface form of) entity names, e.g., guessing that a person with an Italian-sounding name speaks Italian. More specifically, we show that BERT's precision drops dramatically when we filter certain easy-to-guess facts. As a remedy, we propose E-BERT, an extension of BERT that replaces entity mentions with symbolic entity embeddings. E-BERT outperforms both BERT and ERNIE (Zhang et al., 2019) on hard-to-guess queries. We take this as evidence that E-BERT is richer in factual knowledge, and we show two ways of ensembling BERT and E-BERT.

{{< /ci-details >}}

{{< ci-details summary="Are Pre-trained Language Models Aware of Phrases? Simple but Strong Baselines for Grammar Induction (Taeuk Kim et al., 2020)">}}

Taeuk Kim, Jihun Choi, Daniel Edmiston, Sang-goo Lee. (2020)  
**Are Pre-trained Language Models Aware of Phrases? Simple but Strong Baselines for Grammar Induction**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/7cf8510d5905bd8a63f1e098e05ab591d689e0fd)  
Influential Citation Count (7), SS-ID (7cf8510d5905bd8a63f1e098e05ab591d689e0fd)  

**ABSTRACT**  
With the recent success and popularity of pre-trained language models (LMs) in natural language processing, there has been a rise in efforts to understand their inner workings. In line with such interest, we propose a novel method that assists us in investigating the extent to which pre-trained LMs capture the syntactic notion of constituency. Our method provides an effective way of extracting constituency trees from the pre-trained LMs without training. In addition, we report intriguing findings in the induced trees, including the fact that pre-trained LMs outperform other approaches in correctly demarcating adverb phrases in sentences.

{{< /ci-details >}}

{{< ci-details summary="How Much Knowledge Can You Pack into the Parameters of a Language Model? (Adam Roberts et al., 2020)">}}

Adam Roberts, Colin Raffel, Noam M. Shazeer. (2020)  
**How Much Knowledge Can You Pack into the Parameters of a Language Model?**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/80376bdec5f534be78ba82821f540590ebce5559)  
Influential Citation Count (38), SS-ID (80376bdec5f534be78ba82821f540590ebce5559)  

**ABSTRACT**  
It has recently been observed that neural language models trained on unstructured text can implicitly store and retrieve knowledge using natural language queries. In this short paper, we measure the practical utility of this approach by fine-tuning pre-trained models to answer questions without access to any external context or knowledge. We show that this approach scales surprisingly well with model size and outperforms models that explicitly look up knowledge on the open-domain variants of Natural Questions and WebQuestions. To facilitate reproducibility and future work, we release our code and trained models.

{{< /ci-details >}}

{{< ci-details summary="How Multilingual is Multilingual BERT? (Telmo J. P. Pires et al., 2019)">}}

Telmo J. P. Pires, Eva Schlinger, Dan Garrette. (2019)  
**How Multilingual is Multilingual BERT?**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/809cc93921e4698bde891475254ad6dfba33d03b)  
Influential Citation Count (62), SS-ID (809cc93921e4698bde891475254ad6dfba33d03b)  

**ABSTRACT**  
In this paper, we show that Multilingual BERT (M-BERT), released by Devlin et al. (2018) as a single language model pre-trained from monolingual corpora in 104 languages, is surprisingly good at zero-shot cross-lingual model transfer, in which task-specific annotations in one language are used to fine-tune the model for evaluation in another language. To understand why, we present a large number of probing experiments, showing that transfer is possible even to languages in different scripts, that transfer works best between typologically similar languages, that monolingual corpora can train models for code-switching, and that the model can find translation pairs. From these results, we can conclude that M-BERT does create multilingual representations, but that these representations exhibit systematic deficiencies affecting certain language pairs.

{{< /ci-details >}}

{{< ci-details summary="Patient Knowledge Distillation for BERT Model Compression (S. Sun et al., 2019)">}}

S. Sun, Yu Cheng, Zhe Gan, Jingjing Liu. (2019)  
**Patient Knowledge Distillation for BERT Model Compression**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/80cf2a6af4200ecfca1c18fc89de16148f1cd4bf)  
Influential Citation Count (82), SS-ID (80cf2a6af4200ecfca1c18fc89de16148f1cd4bf)  

**ABSTRACT**  
Pre-trained language models such as BERT have proven to be highly effective for natural language processing (NLP) tasks. However, the high demand for computing resources in training such models hinders their application in practice. In order to alleviate this resource hunger in large-scale model training, we propose a Patient Knowledge Distillation approach to compress an original large model (teacher) into an equally-effective lightweight shallow network (student). Different from previous knowledge distillation methods, which only use the output from the last layer of the teacher network for distillation, our student model patiently learns from multiple intermediate layers of the teacher model for incremental knowledge extraction, following two strategies: (i) PKD-Last: learning from the last k layers; and (ii) PKD-Skip: learning from every k layers. These two patient distillation schemes enable the exploitation of rich information in the teacher’s hidden layers, and encourage the student model to patiently learn from and imitate the teacher through a multi-layer distillation process. Empirically, this translates into improved results on multiple NLP tasks with a significant gain in training efficiency, without sacrificing model accuracy.

{{< /ci-details >}}

{{< ci-details summary="Understanding Commonsense Inference Aptitude of Deep Contextual Representations (Jeff Da et al., 2019)">}}

Jeff Da, Jungo Kasai. (2019)  
**Understanding Commonsense Inference Aptitude of Deep Contextual Representations**  
Proceedings of the First Workshop on Commonsense Inference in Natural Language Processing  
[Paper Link](https://www.semanticscholar.org/paper/80dc7b0e6dbc26571672d9be57a0ae589689e410)  
Influential Citation Count (0), SS-ID (80dc7b0e6dbc26571672d9be57a0ae589689e410)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="ERNIE 2.0: A Continual Pre-training Framework for Language Understanding (Yu Sun et al., 2019)">}}

Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Hao Tian, Hua Wu, Haifeng Wang. (2019)  
**ERNIE 2.0: A Continual Pre-training Framework for Language Understanding**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/80f9f109d1564cb8f82aa440a5f6f3fbe220c9ef)  
Influential Citation Count (63), SS-ID (80f9f109d1564cb8f82aa440a5f6f3fbe220c9ef)  

**ABSTRACT**  
Recently pre-trained models have achieved state-of-the-art results in various language understanding tasks. Current pre-training procedures usually focus on training the model with several simple tasks to grasp the co-occurrence of words or sentences. However, besides co-occurring information, there exists other valuable lexical, syntactic and semantic information in training corpora, such as named entities, semantic closeness and discourse relations. In order to extract the lexical, syntactic and semantic information from training corpora, we propose a continual pre-training framework named ERNIE 2.0 which incrementally builds pre-training tasks and then learn pre-trained models on these constructed tasks via continual multi-task learning. Based on this framework, we construct several tasks and train the ERNIE 2.0 model to capture lexical, syntactic and semantic aspects of information in the training data. Experimental results demonstrate that ERNIE 2.0 model outperforms BERT and XLNet on 16 tasks including English tasks on GLUE benchmarks and several similar tasks in Chinese. The source codes and pre-trained models have been released at https://github.com/PaddlePaddle/ERNIE.

{{< /ci-details >}}

{{< ci-details summary="From English To Foreign Languages: Transferring Pre-trained Language Models (Ke M. Tran, 2019)">}}

Ke M. Tran. (2019)  
**From English To Foreign Languages: Transferring Pre-trained Language Models**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/8199b4c196b09d6176816e4d7db8d6f3d65e07c1)  
Influential Citation Count (2), SS-ID (8199b4c196b09d6176816e4d7db8d6f3d65e07c1)  

**ABSTRACT**  
Pre-trained models have demonstrated their effectiveness in many downstream natural language processing (NLP) tasks. The availability of multilingual pre-trained models enables zero-shot transfer of NLP tasks from high resource languages to low resource ones. However, recent research in improving pre-trained models focuses heavily on English. While it is possible to train the latest neural architectures for other languages from scratch, it is undesirable due to the required amount of compute. In this work, we tackle the problem of transferring an existing pre-trained model from English to other languages under a limited computational budget. With a single GPU, our approach can obtain a foreign BERT base model within a day and a foreign BERT large within two days. Furthermore, evaluating our models on six languages, we demonstrate that our models are better than multilingual BERT on two zero-shot tasks: natural language inference and dependency parsing.

{{< /ci-details >}}

{{< ci-details summary="How Can We Know What Language Models Know? (Zhengbao Jiang et al., 2019)">}}

Zhengbao Jiang, Frank F. Xu, J. Araki, Graham Neubig. (2019)  
**How Can We Know What Language Models Know?**  
Transactions of the Association for Computational Linguistics  
[Paper Link](https://www.semanticscholar.org/paper/81dd3faf762ad8f084ab1d7b8fc9e77e9e160f85)  
Influential Citation Count (16), SS-ID (81dd3faf762ad8f084ab1d7b8fc9e77e9e160f85)  

**ABSTRACT**  
Abstract Recent work has presented intriguing results examining the knowledge contained in language models (LMs) by having the LM fill in the blanks of prompts such as “Obama is a __ by profession”. These prompts are usually manually created, and quite possibly sub-optimal; another prompt such as “Obama worked as a __ ” may result in more accurately predicting the correct profession. Because of this, given an inappropriate prompt, we might fail to retrieve facts that the LM does know, and thus any given prompt only provides a lower bound estimate of the knowledge contained in an LM. In this paper, we attempt to more accurately estimate the knowledge contained in LMs by automatically discovering better prompts to use in this querying process. Specifically, we propose mining-based and paraphrasing-based methods to automatically generate high-quality and diverse prompts, as well as ensemble methods to combine answers from different prompts. Extensive experiments on the LAMA benchmark for extracting relational knowledge from LMs demonstrate that our methods can improve accuracy from 31.1% to 39.6%, providing a tighter lower bound on what LMs know. We have released the code and the resulting LM Prompt And Query Archive (LPAQA) at https://github.com/jzbjyb/LPAQA.

{{< /ci-details >}}

{{< ci-details summary="SpanBERT: Improving Pre-training by Representing and Predicting Spans (Mandar Joshi et al., 2019)">}}

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, Omer Levy. (2019)  
**SpanBERT: Improving Pre-training by Representing and Predicting Spans**  
TACL  
[Paper Link](https://www.semanticscholar.org/paper/81f5810fbbab9b7203b9556f4ce3c741875407bc)  
Influential Citation Count (175), SS-ID (81f5810fbbab9b7203b9556f4ce3c741875407bc)  

**ABSTRACT**  
We present SpanBERT, a pre-training method that is designed to better represent and predict spans of text. Our approach extends BERT by (1) masking contiguous random spans, rather than random tokens, and (2) training the span boundary representations to predict the entire content of the masked span, without relying on the individual token representations within it. SpanBERT consistently outperforms BERT and our better-tuned baselines, with substantial gains on span selection tasks such as question answering and coreference resolution. In particular, with the same training data and model size as BERTlarge, our single model obtains 94.6% and 88.7% F1 on SQuAD 1.1 and 2.0 respectively. We also achieve a new state of the art on the OntoNotes coreference resolution task (79.6% F1), strong performance on the TACRED relation extraction benchmark, and even gains on GLUE.1

{{< /ci-details >}}

{{< ci-details summary="Is Multilingual BERT Fluent in Language Generation? (Samuel Rönnqvist et al., 2019)">}}

Samuel Rönnqvist, Jenna Kanerva, T. Salakoski, Filip Ginter. (2019)  
**Is Multilingual BERT Fluent in Language Generation?**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/81fbf08beb80b01abaa6ad6a07b48c3034ead8a6)  
Influential Citation Count (3), SS-ID (81fbf08beb80b01abaa6ad6a07b48c3034ead8a6)  

**ABSTRACT**  
The multilingual BERT model is trained on 104 languages and meant to serve as a universal language model and tool for encoding sentences. We explore how well the model performs on several languages across several tasks: a diagnostic classification probing the embeddings for a particular syntactic property, a cloze task testing the language modelling ability to fill in gaps in a sentence, and a natural language generation task testing for the ability to produce coherent text fitting a given context. We find that the currently available multilingual BERT model is clearly inferior to the monolingual counterparts, and cannot in many cases serve as a substitute for a well-trained monolingual model. We find that the English and German models perform well at generation, whereas the multilingual model is lacking, in particular, for Nordic languages.

{{< /ci-details >}}

{{< ci-details summary="REALM: Retrieval-Augmented Language Model Pre-Training (Kelvin Guu et al., 2020)">}}

Kelvin Guu, Kenton Lee, Z. Tung, Panupong Pasupat, Ming-Wei Chang. (2020)  
**REALM: Retrieval-Augmented Language Model Pre-Training**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/832fff14d2ed50eb7969c4c4b976c35776548f56)  
Influential Citation Count (64), SS-ID (832fff14d2ed50eb7969c4c4b976c35776548f56)  

**ABSTRACT**  
Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network, requiring ever-larger networks to cover more facts.  To capture knowledge in a more modular and interpretable way, we augment language model pre-training with a latent knowledge retriever, which allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference. For the first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents.  We demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as interpretability and modularity.

{{< /ci-details >}}

{{< ci-details summary="How Does BERT Answer Questions?: A Layer-Wise Analysis of Transformer Representations (Betty van Aken et al., 2019)">}}

Betty van Aken, Benjamin Winter, Alexander Löser, F. Gers. (2019)  
**How Does BERT Answer Questions?: A Layer-Wise Analysis of Transformer Representations**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/8380ab11c120a77cbdd2053337aa52525ec0f22e)  
Influential Citation Count (4), SS-ID (8380ab11c120a77cbdd2053337aa52525ec0f22e)  

**ABSTRACT**  
Bidirectional Encoder Representations from Transformers (BERT) reach state-of-the-art results in a variety of Natural Language Processing tasks. However, understanding of their internal functioning is still insufficient and unsatisfactory. In order to better understand BERT and other Transformer-based models, we present a layer-wise analysis of BERT's hidden states. Unlike previous research, which mainly focuses on explaining Transformer models by their attention weights, we argue that hidden states contain equally valuable information. Specifically, our analysis focuses on models fine-tuned on the task of Question Answering (QA) as an example of a complex downstream task. We inspect how QA models transform token vectors in order to find the correct answer. To this end, we apply a set of general and QA-specific probing tasks that reveal the information stored in each representation layer. Our qualitative analysis of hidden state visualizations provides additional insights into BERT's reasoning process. Our results show that the transformations within BERT go through phases that are related to traditional pipeline tasks. The system can therefore implicitly incorporate task-specific information into its token representations. Furthermore, our analysis reveals that fine-tuning has little impact on the models' semantic abilities and that prediction errors can be recognized in the vector representations of even early layers.

{{< /ci-details >}}

{{< ci-details summary="Universal Text Representation from BERT: An Empirical Study (Xiaofei Ma et al., 2019)">}}

Xiaofei Ma, Zhiguo Wang, Patrick Ng, Ramesh Nallapati, Bing Xiang. (2019)  
**Universal Text Representation from BERT: An Empirical Study**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/850713961a5aa20812cf952f950f09d491fae281)  
Influential Citation Count (1), SS-ID (850713961a5aa20812cf952f950f09d491fae281)  

**ABSTRACT**  
We present a systematic investigation of layer-wise BERT activations for general-purpose text representations to understand what linguistic information they capture and how transferable they are across different tasks. Sentence-level embeddings are evaluated against two state-of-the-art models on downstream and probing tasks from SentEval, while passage-level embeddings are evaluated on four question-answering (QA) datasets under a learning-to-rank problem setting. Embeddings from the pre-trained BERT model perform poorly in semantic similarity and sentence surface information probing tasks. Fine-tuning BERT on natural language inference data greatly improves the quality of the embeddings. Combining embeddings from different BERT layers can further boost performance. BERT embeddings outperform BM25 baseline significantly on factoid QA datasets at the passage level, but fail to perform better than BM25 on non-factoid datasets. For all QA datasets, there is a gap between embedding-based method and in-domain fine-tuned BERT (we report new state-of-the-art results on two datasets), which suggests deep interactions between question and answer pairs are critical for those hard tasks.

{{< /ci-details >}}

{{< ci-details summary="To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks (Matthew E. Peters et al., 2019)">}}

Matthew E. Peters, Sebastian Ruder, Noah A. Smith. (2019)  
**To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks**  
RepL4NLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/8659bf379ca8756755125a487c43cfe8611ce842)  
Influential Citation Count (21), SS-ID (8659bf379ca8756755125a487c43cfe8611ce842)  

**ABSTRACT**  
While most previous work has focused on different pretraining objectives and architectures for transfer learning, we ask how to best adapt the pretrained model to a given target task. We focus on the two most common forms of adaptation, feature extraction (where the pretrained weights are frozen), and directly fine-tuning the pretrained model. Our empirical results across diverse NLP tasks with two state-of-the-art models show that the relative performance of fine-tuning vs. feature extraction depends on the similarity of the pretraining and target tasks. We explore possible explanations for this finding and provide a set of adaptation guidelines for the NLP practitioner.

{{< /ci-details >}}

{{< ci-details summary="How does BERT’s attention change when you fine-tune? An analysis methodology and a case study in negation scope (Yiyun Zhao et al., 2020)">}}

Yiyun Zhao, Steven Bethard. (2020)  
**How does BERT’s attention change when you fine-tune? An analysis methodology and a case study in negation scope**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/868349fe969bc7c6b14b5f35e118a26075b7b1f2)  
Influential Citation Count (0), SS-ID (868349fe969bc7c6b14b5f35e118a26075b7b1f2)  

**ABSTRACT**  
Large pretrained language models like BERT, after fine-tuning to a downstream task, have achieved high performance on a variety of NLP problems. Yet explaining their decisions is difficult despite recent work probing their internal representations. We propose a procedure and analysis methods that take a hypothesis of how a transformer-based model might encode a linguistic phenomenon, and test the validity of that hypothesis based on a comparison between knowledge-related downstream tasks with downstream control tasks, and measurement of cross-dataset consistency. We apply this methodology to test BERT and RoBERTa on a hypothesis that some attention heads will consistently attend from a word in negation scope to the negation cue. We find that after fine-tuning BERT and RoBERTa on a negation scope task, the average attention head improves its sensitivity to negation and its attention consistency across negation datasets compared to the pre-trained models. However, only the base models (not the large models) improve compared to a control task, indicating there is evidence for a shallow encoding of negation only in the base models.

{{< /ci-details >}}

{{< ci-details summary="On the Comparability of Pre-trained Language Models (M. Aßenmacher et al., 2020)">}}

M. Aßenmacher, C. Heumann. (2020)  
**On the Comparability of Pre-trained Language Models**  
SwissText/KONVENS  
[Paper Link](https://www.semanticscholar.org/paper/86bd570007c863c147eb9c13f00fc6908f6b3fc9)  
Influential Citation Count (0), SS-ID (86bd570007c863c147eb9c13f00fc6908f6b3fc9)  

**ABSTRACT**  
Recent developments in unsupervised representation learning have successfully established the concept of transfer learning in NLP. Mainly three forces are driving the improvements in this area of research: More elaborated architectures are making better use of contextual information. Instead of simply plugging in static pre-trained representations, these are learned based on surrounding context in end-to-end trainable models with more intelligently designed language modelling objectives. Along with this, larger corpora are used as resources for pre-training large language models in a self-supervised fashion which are afterwards fine-tuned on supervised tasks. Advances in parallel computing as well as in cloud computing, made it possible to train these models with growing capacities in the same or even in shorter time than previously established models. These three developments agglomerate in new state-of-the-art (SOTA) results being revealed in a higher and higher frequency. It is not always obvious where these improvements originate from, as it is not possible to completely disentangle the contributions of the three driving forces. We set ourselves to providing a clear and concise overview on several large pre-trained language models, which achieved SOTA results in the last two years, with respect to their use of new architectures and resources. We want to clarify for the reader where the differences between the models are and we furthermore attempt to gain some insight into the single contributions of lexical/computational improvements as well as of architectural changes. We explicitly do not intend to quantify these contributions, but rather see our work as an overview in order to identify potential starting points for benchmark comparisons. Furthermore, we tentatively want to point at potential possibilities for improvement in the field of open-sourcing and reproducible research.

{{< /ci-details >}}

{{< ci-details summary="Distributed Representations of Words and Phrases and their Compositionality (Tomas Mikolov et al., 2013)">}}

Tomas Mikolov, Ilya Sutskever, Kai Chen, G. Corrado, J. Dean. (2013)  
**Distributed Representations of Words and Phrases and their Compositionality**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/87f40e6f3022adbc1f1905e3e506abad05a9964f)  
Influential Citation Count (3587), SS-ID (87f40e6f3022adbc1f1905e3e506abad05a9964f)  

**ABSTRACT**  
The recently introduced continuous Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships. In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word representations. We also describe a simple alternative to the hierarchical softmax called negative sampling.    An inherent limitation of word representations is their indifference to word order and their inability to represent idiomatic phrases. For example, the meanings of "Canada" and "Air" cannot be easily combined to obtain "Air Canada". Motivated by this example, we present a simple method for finding phrases in text, and show that learning good vector representations for millions of phrases is possible.

{{< /ci-details >}}

{{< ci-details summary="HellaSwag: Can a Machine Really Finish Your Sentence? (Rowan Zellers et al., 2019)">}}

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, Yejin Choi. (2019)  
**HellaSwag: Can a Machine Really Finish Your Sentence?**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/8b0f27bb594b1eaaf493eaf1e2ee723a2b0a19ad)  
Influential Citation Count (35), SS-ID (8b0f27bb594b1eaaf493eaf1e2ee723a2b0a19ad)  

**ABSTRACT**  
Recent work by Zellers et al. (2018) introduced a new task of commonsense natural language inference: given an event description such as “A woman sits at a piano,” a machine must select the most likely followup: “She sets her fingers on the keys.” With the introduction of BERT, near human-level performance was reached. Does this mean that machines can perform human level commonsense inference? In this paper, we show that commonsense inference still proves difficult for even state-of-the-art models, by presenting HellaSwag, a new challenge dataset. Though its questions are trivial for humans (>95% accuracy), state-of-the-art models struggle (<48%). We achieve this via Adversarial Filtering (AF), a data collection paradigm wherein a series of discriminators iteratively select an adversarial set of machine-generated wrong answers. AF proves to be surprisingly robust. The key insight is to scale up the length and complexity of the dataset examples towards a critical ‘Goldilocks’ zone wherein generated text is ridiculous to humans, yet often misclassified by state-of-the-art models. Our construction of HellaSwag, and its resulting difficulty, sheds light on the inner workings of deep pretrained models. More broadly, it suggests a new path forward for NLP research, in which benchmarks co-evolve with the evolving state-of-the-art in an adversarial way, so as to present ever-harder challenges.

{{< /ci-details >}}

{{< ci-details summary="Questionable Answers in Question Answering Research: Reproducibility and Variability of Published Results (M. Crane, 2018)">}}

M. Crane. (2018)  
**Questionable Answers in Question Answering Research: Reproducibility and Variability of Published Results**  
TACL  
[Paper Link](https://www.semanticscholar.org/paper/8c0548331e02c2ead48d6c0380f9a80471ea5d80)  
Influential Citation Count (5), SS-ID (8c0548331e02c2ead48d6c0380f9a80471ea5d80)  

**ABSTRACT**  
“Based on theoretical reasoning it has been suggested that the reliability of findings published in the scientific literature decreases with the popularity of a research field” (Pfeiffer and Hoffmann, 2009). As we know, deep learning is very popular and the ability to reproduce results is an important part of science. There is growing concern within the deep learning community about the reproducibility of results that are presented. In this paper we present a number of controllable, yet unreported, effects that can substantially change the effectiveness of a sample model, and thusly the reproducibility of those results. Through these environmental effects we show that the commonly held belief that distribution of source code is all that is needed for reproducibility is not enough. Source code without a reproducible environment does not mean anything at all. In addition the range of results produced from these effects can be larger than the majority of incremental improvement reported.

{{< /ci-details >}}

{{< ci-details summary="Transfer Fine-Tuning: A BERT Case Study (Yuki Arase et al., 2019)">}}

Yuki Arase, Junichi Tsujii. (2019)  
**Transfer Fine-Tuning: A BERT Case Study**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/8e00d81ff7b1656c621f64fe72fff2356bacb29f)  
Influential Citation Count (0), SS-ID (8e00d81ff7b1656c621f64fe72fff2356bacb29f)  

**ABSTRACT**  
A semantic equivalence assessment is defined as a task that assesses semantic equivalence in a sentence pair by binary judgment (i.e., paraphrase identification) or grading (i.e., semantic textual similarity measurement). It constitutes a set of tasks crucial for research on natural language understanding. Recently, BERT realized a breakthrough in sentence representation learning (Devlin et al., 2019), which is broadly transferable to various NLP tasks. While BERT’s performance improves by increasing its model size, the required computational power is an obstacle preventing practical applications from adopting the technology. Herein, we propose to inject phrasal paraphrase relations into BERT in order to generate suitable representations for semantic equivalence assessment instead of increasing the model size. Experiments on standard natural language understanding tasks confirm that our method effectively improves a smaller BERT model while maintaining the model size. The generated model exhibits superior performance compared to a larger BERT model on semantic equivalence assessment tasks. Furthermore, it achieves larger performance gains on tasks with limited training datasets for fine-tuning, which is a property desirable for transfer learning.

{{< /ci-details >}}

{{< ci-details summary="Assessing the Benchmarking Capacity of Machine Reading Comprehension Datasets (Saku Sugawara et al., 2019)">}}

Saku Sugawara, Pontus Stenetorp, Kentaro Inui, Akiko Aizawa. (2019)  
**Assessing the Benchmarking Capacity of Machine Reading Comprehension Datasets**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/9148f4bb8ebdcc75beaddc875d6de857bbe85ba3)  
Influential Citation Count (6), SS-ID (9148f4bb8ebdcc75beaddc875d6de857bbe85ba3)  

**ABSTRACT**  
Existing analysis work in machine reading comprehension (MRC) is largely concerned with evaluating the capabilities of systems. However, the capabilities of datasets are not assessed for benchmarking language understanding precisely. We propose a semi-automated, ablation-based methodology for this challenge; By checking whether questions can be solved even after removing features associated with a skill requisite for language understanding, we evaluate to what degree the questions do not require the skill. Experiments on 10 datasets (e.g., CoQA, SQuAD v2.0, and RACE) with a strong baseline model show that, for example, the relative scores of the baseline model provided with content words only and with shuffled sentence words in the context are on average 89.2% and 78.5% of the original scores, respectively. These results suggest that most of the questions already answered correctly by the model do not necessarily require grammatical and complex reasoning. For precise benchmarking, MRC datasets will need to take extra care in their design to ensure that questions can correctly evaluate the intended skills.

{{< /ci-details >}}

{{< ci-details summary="When BERT Plays the Lottery, All Tickets Are Winning (Sai Prasanna et al., 2020)">}}

Sai Prasanna, Anna Rogers, Anna Rumshisky. (2020)  
**When BERT Plays the Lottery, All Tickets Are Winning**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/91ac65431b2dc46919e1673fde67671c29446812)  
Influential Citation Count (7), SS-ID (91ac65431b2dc46919e1673fde67671c29446812)  

**ABSTRACT**  
Much of the recent success in NLP is due to the large Transformer-based models such as BERT (Devlin et al, 2019). However, these models have been shown to be reducible to a smaller number of self-attention heads and layers. We consider this phenomenon from the perspective of the lottery ticket hypothesis. For fine-tuned BERT, we show that (a) it is possible to find a subnetwork of elements that achieves performance comparable with that of the full model, and (b) similarly-sized subnetworks sampled from the rest of the model perform worse. However, the "bad" subnetworks can be fine-tuned separately to achieve only slightly worse performance than the "good" ones, indicating that most weights in the pre-trained BERT are potentially useful. We also show that the "good" subnetworks vary considerably across GLUE tasks, opening up the possibilities to learn what knowledge BERT actually uses at inference time.

{{< /ci-details >}}

{{< ci-details summary="Well-Read Students Learn Better: The Impact of Student Initialization on Knowledge Distillation (Iulia Turc et al., 2019)">}}

Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019)  
**Well-Read Students Learn Better: The Impact of Student Initialization on Knowledge Distillation**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/93ad19fbc85360043988fa9ea7932b7fdf1fa948)  
Influential Citation Count (18), SS-ID (93ad19fbc85360043988fa9ea7932b7fdf1fa948)  

**ABSTRACT**  
Recent developments in NLP have been accompanied by large, expensive models. Knowledge distillation is the standard method to realize these gains in applications with limited resources: a compact student is trained to recover the outputs of a powerful teacher. While most prior work investigates student architectures and transfer techniques, we focus on an often-neglected aspect---student initialization. We argue that a random starting point hinders students from fully leveraging the teacher expertise, even in the presence of a large transfer set. We observe that applying language model pre-training to students unlocks their generalization potential, surprisingly even for very compact networks. We conduct experiments on 4 NLP tasks and 24 sizes of Transformer-based students; for sentiment classification on the Amazon Book Reviews dataset, pre-training boosts size reduction and TPU speed-up from 3.1x/1.25x to 31x/16x. Extensive ablation studies dissect the interaction between pre-training and distillation, revealing a compound effect even when they are applied on the same unlabeled dataset.

{{< /ci-details >}}

{{< ci-details summary="GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding (Alex Wang et al., 2018)">}}

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman. (2018)  
**GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding**  
BlackboxNLP@EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/93b8da28d006415866bf48f9a6e06b5242129195)  
Influential Citation Count (625), SS-ID (93b8da28d006415866bf48f9a6e06b5242129195)  

**ABSTRACT**  
Human ability to understand language is general, flexible, and robust. In contrast, most NLU models above the word level are designed for a specific task and struggle with out-of-domain data. If we aspire to develop models with understanding beyond the detection of superficial correspondences between inputs and outputs, then it is critical to develop a unified model that can execute a range of linguistic tasks across different domains. To facilitate research in this direction, we present the General Language Understanding Evaluation (GLUE, gluebenchmark.com): a benchmark of nine diverse NLU tasks, an auxiliary dataset for probing models for understanding of specific linguistic phenomena, and an online platform for evaluating and comparing models. For some benchmark tasks, training data is plentiful, but for others it is limited or does not match the genre of the test set. GLUE thus favors models that can represent linguistic knowledge in a way that facilitates sample-efficient learning and effective knowledge-transfer across tasks. While none of the datasets in GLUE were created from scratch for the benchmark, four of them feature privately-held test data, which is used to ensure that the benchmark is used fairly. We evaluate baselines that use ELMo (Peters et al., 2018), a powerful transfer learning technique, as well as state-of-the-art sentence representation models. The best models still achieve fairly low absolute scores. Analysis with our diagnostic dataset yields similarly weak performance over all phenomena tested, with some exceptions.

{{< /ci-details >}}

{{< ci-details summary="An Analysis of Encoder Representations in Transformer-Based Machine Translation (Alessandro Raganato et al., 2018)">}}

Alessandro Raganato, J. Tiedemann. (2018)  
**An Analysis of Encoder Representations in Transformer-Based Machine Translation**  
BlackboxNLP@EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/94238dead40b12735d79ed63e29ead70730261a2)  
Influential Citation Count (10), SS-ID (94238dead40b12735d79ed63e29ead70730261a2)  

**ABSTRACT**  
The attention mechanism is a successful technique in modern NLP, especially in tasks like machine translation. The recently proposed network architecture of the Transformer is based entirely on attention mechanisms and achieves new state of the art results in neural machine translation, outperforming other sequence-to-sequence models. However, so far not much is known about the internal properties of the model and the representations it learns to achieve that performance. To study this question, we investigate the information that is learned by the attention mechanism in Transformer models with different translation quality. We assess the representations of the encoder by extracting dependency relations based on self-attention weights, we perform four probing tasks to study the amount of syntactic and semantic captured information and we also test attention in a transfer learning scenario. Our analysis sheds light on the relative strengths and weaknesses of the various encoder representations. We observe that specific attention heads mark syntactic dependency relations and we can also confirm that lower layers tend to learn more about syntax while higher layers tend to encode more semantics.

{{< /ci-details >}}

{{< ci-details summary="Knowledge Distillation from Internal Representations (Gustavo Aguilar et al., 2019)">}}

Gustavo Aguilar, Yuan Ling, Y. Zhang, Benjamin Yao, Xing Fan, Edward Guo. (2019)  
**Knowledge Distillation from Internal Representations**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/944e7b64903bde89bfea433203d5a0e774cff354)  
Influential Citation Count (4), SS-ID (944e7b64903bde89bfea433203d5a0e774cff354)  

**ABSTRACT**  
Knowledge distillation is typically conducted by training a small model (the student) to mimic a large and cumbersome model (the teacher). The idea is to compress the knowledge from the teacher by using its output probabilities as soft-labels to optimize the student. However, when the teacher is considerably large, there is no guarantee that the internal knowledge of the teacher will be transferred into the student; even if the student closely matches the soft-labels, its internal representations may be considerably different. This internal mismatch can undermine the generalization capabilities originally intended to be transferred from the teacher to the student. In this paper, we propose to distill the internal representations of a large model such as BERT into a simplified version of it. We formulate two ways to distill such representations and various algorithms to conduct the distillation. We experiment with datasets from the GLUE benchmark and consistently show that adding knowledge distillation from internal representations is a more powerful method than only using soft-label distillation.

{{< /ci-details >}}

{{< ci-details summary="PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination (Saurabh Goyal et al., 2020)">}}

Saurabh Goyal, Anamitra R. Choudhury, S. Raje, Venkatesan T. Chakaravarthy, Yogish Sabharwal, Ashish Verma. (2020)  
**PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/94f94e8892261d0377159379ca5a166ceae19a14)  
Influential Citation Count (13), SS-ID (94f94e8892261d0377159379ca5a166ceae19a14)  

**ABSTRACT**  
We develop a novel method, called PoWER-BERT, for improving the inference time of the popular BERT model, while maintaining the accuracy. It works by: a) exploiting redundancy pertaining to word-vectors (intermediate encoder outputs) and eliminating the redundant vectors. b) determining which word-vectors to eliminate by developing a strategy for measuring their significance, based on the self-attention mechanism. c) learning how many word-vectors to eliminate by augmenting the BERT model and the loss function. Experiments on the standard GLUE benchmark shows that PoWER-BERT achieves up to 4.5x reduction in inference time over BERT with <1% loss in accuracy. We show that PoWER-BERT offers significantly better trade-off between accuracy and inference time compared to prior methods. We demonstrate that our method attains up to 6.8x reduction in inference time with <1% loss in accuracy when applied over ALBERT, a highly compressed version of BERT. The code for PoWER-BERT is publicly available at this https URL.

{{< /ci-details >}}

{{< ci-details summary="What Does BERT Look at? An Analysis of BERT’s Attention (Kevin Clark et al., 2019)">}}

Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning. (2019)  
**What Does BERT Look at? An Analysis of BERT’s Attention**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/95a251513853c6032bdecebd4b74e15795662986)  
Influential Citation Count (74), SS-ID (95a251513853c6032bdecebd4b74e15795662986)  

**ABSTRACT**  
Large pre-trained neural networks such as BERT have had great recent success in NLP, motivating a growing body of research investigating what aspects of language they are able to learn from unlabeled data. Most recent analysis has focused on model outputs (e.g., language model surprisal) or internal vector representations (e.g., probing classifiers). Complementary to these works, we propose methods for analyzing the attention mechanisms of pre-trained models and apply them to BERT. BERT’s attention heads exhibit patterns such as attending to delimiter tokens, specific positional offsets, or broadly attending over the whole sentence, with heads in the same layer often exhibiting similar behaviors. We further show that certain attention heads correspond well to linguistic notions of syntax and coreference. For example, we find heads that attend to the direct objects of verbs, determiners of nouns, objects of prepositions, and coreferent mentions with remarkably high accuracy. Lastly, we propose an attention-based probing classifier and use it to further demonstrate that substantial syntactic information is captured in BERT’s attention.

{{< /ci-details >}}

{{< ci-details summary="BERT Rediscovers the Classical NLP Pipeline (Ian Tenney et al., 2019)">}}

Ian Tenney, Dipanjan Das, Ellie Pavlick. (2019)  
**BERT Rediscovers the Classical NLP Pipeline**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/97906df07855b029b7aae7c2a1c6c5e8df1d531c)  
Influential Citation Count (59), SS-ID (97906df07855b029b7aae7c2a1c6c5e8df1d531c)  

**ABSTRACT**  
Pre-trained text encoders have rapidly advanced the state of the art on many NLP tasks. We focus on one such model, BERT, and aim to quantify where linguistic information is captured within the network. We find that the model represents the steps of the traditional NLP pipeline in an interpretable and localizable way, and that the regions responsible for each step appear in the expected sequence: POS tagging, parsing, NER, semantic roles, then coreference. Qualitative analysis reveals that the model can and often does adjust this pipeline dynamically, revising lower-level decisions on the basis of disambiguating information from higher-level representations.

{{< /ci-details >}}

{{< ci-details summary="Contextual and Non-Contextual Word Embeddings: an in-depth Linguistic Investigation (Alessio Miaschi et al., 2020)">}}

Alessio Miaschi, F. Dell’Orletta. (2020)  
**Contextual and Non-Contextual Word Embeddings: an in-depth Linguistic Investigation**  
REPL4NLP  
[Paper Link](https://www.semanticscholar.org/paper/9b1933038680b13c06b60dfe810e96a3a0ef9d37)  
Influential Citation Count (0), SS-ID (9b1933038680b13c06b60dfe810e96a3a0ef9d37)  

**ABSTRACT**  
In this paper we present a comparison between the linguistic knowledge encoded in the internal representations of a contextual Language Model (BERT) and a contextual-independent one (Word2vec). We use a wide set of probing tasks, each of which corresponds to a distinct sentence-level feature extracted from different levels of linguistic annotation. We show that, although BERT is capable of understanding the full context of each word in an input sequence, the implicit knowledge encoded in its aggregated sentence representations is still comparable to that of a contextual-independent model. We also find that BERT is able to encode sentence-level properties even within single-word embeddings, obtaining comparable or even superior results than those obtained with sentence representations.

{{< /ci-details >}}

{{< ci-details summary="A Cross-Task Analysis of Text Span Representations (Shubham Toshniwal et al., 2020)">}}

Shubham Toshniwal, Haoyue Shi, Bowen Shi, Lingyu Gao, Karen Livescu, Kevin Gimpel. (2020)  
**A Cross-Task Analysis of Text Span Representations**  
REPL4NLP  
[Paper Link](https://www.semanticscholar.org/paper/9b2b96adf4ec05b086037222a893fa778f83a985)  
Influential Citation Count (0), SS-ID (9b2b96adf4ec05b086037222a893fa778f83a985)  

**ABSTRACT**  
Many natural language processing (NLP) tasks involve reasoning with textual spans, including question answering, entity recognition, and coreference resolution. While extensive research has focused on functional architectures for representing words and sentences, there is less work on representing arbitrary spans of text within sentences. In this paper, we conduct a comprehensive empirical evaluation of six span representation methods using eight pretrained language representation models across six tasks, including two tasks that we introduce. We find that, although some simple span representations are fairly reliable across tasks, in general the optimal span representation varies by task, and can also vary within different facets of individual tasks. We also find that the choice of span representation has a bigger impact with a fixed pretrained encoder than with a fine-tuned encoder.

{{< /ci-details >}}

{{< ci-details summary="How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings (Kawin Ethayarajh, 2019)">}}

Kawin Ethayarajh. (2019)  
**How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/9d7902e834d5d1d35179962c7a5b9d16623b0d39)  
Influential Citation Count (29), SS-ID (9d7902e834d5d1d35179962c7a5b9d16623b0d39)  

**ABSTRACT**  
Replacing static word embeddings with contextualized word representations has yielded significant improvements on many NLP tasks. However, just how contextual are the contextualized representations produced by models such as ELMo and BERT? Are there infinitely many context-specific representations for each word, or are words essentially assigned one of a finite number of word-sense representations? For one, we find that the contextualized representations of all words are not isotropic in any layer of the contextualizing model. While representations of the same word in different contexts still have a greater cosine similarity than those of two different words, this self-similarity is much lower in upper layers. This suggests that upper layers of contextualizing models produce more context-specific representations, much like how upper layers of LSTMs produce more task-specific representations. In all layers of ELMo, BERT, and GPT-2, on average, less than 5% of the variance in a word’s contextualized representations can be explained by a static embedding for that word, providing some justification for the success of contextualized representations.

{{< /ci-details >}}

{{< ci-details summary="On Identifiability in Transformers (Gino Brunner et al., 2019)">}}

Gino Brunner, Yang Liu, Damian Pascual, Oliver Richter, Massimiliano Ciaramita, Roger Wattenhofer. (2019)  
**On Identifiability in Transformers**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/9d7fbdb2e9817a6396992a1c92f75206689852d9)  
Influential Citation Count (13), SS-ID (9d7fbdb2e9817a6396992a1c92f75206689852d9)  

**ABSTRACT**  
In this paper we delve deep in the Transformer architecture by investigating two of its core components: self-attention and contextual embeddings. In particular, we study the identifiability of attention weights and token embeddings, and the aggregation of context into hidden tokens. We show that, for sequences longer than the attention head dimension, attention weights are not identifiable. We propose effective attention as a complementary tool for improving explanatory interpretations based on attention. Furthermore, we show that input tokens retain to a large degree their identity across the model. We also find evidence suggesting that identity information is mainly encoded in the angle of the embeddings and gradually decreases with depth. Finally, we demonstrate strong mixing of input information in the generation of contextual embeddings by means of a novel quantification method based on gradient attribution. Overall, we show that self-attention distributions are not directly interpretable and present tools to better understand and further investigate Transformer models.

{{< /ci-details >}}

{{< ci-details summary="Products of Random Latent Variable Grammars (Slav Petrov, 2010)">}}

Slav Petrov. (2010)  
**Products of Random Latent Variable Grammars**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/9dccaf6ea0fa19772cf8067295b16df3eb7b4dda)  
Influential Citation Count (10), SS-ID (9dccaf6ea0fa19772cf8067295b16df3eb7b4dda)  

**ABSTRACT**  
We show that the automatically induced latent variable grammars of Petrov et al. (2006) vary widely in their underlying representations, depending on their EM initialization point. We use this to our advantage, combining multiple automatically learned grammars into an unweighted product model, which gives significantly improved performance over state-of-the-art individual grammars. In our model, the probability of a constituent is estimated as a product of posteriors obtained from multiple grammars that differ only in the random seed used for initialization, without any learning or tuning of combination weights. Despite its simplicity, a product of eight automatically learned grammars improves parsing accuracy from 90.2% to 91.8% on English, and from 80.3% to 84.5% on German.

{{< /ci-details >}}

{{< ci-details summary="Intermediate-Task Transfer Learning with Pretrained Language Models: When and Why Does It Work? (Yada Pruksachatkun et al., 2020)">}}

Yada Pruksachatkun, Jason Phang, Haokun Liu, Phu Mon Htut, Xiaoyi Zhang, Richard Yuanzhe Pang, Clara Vania, Katharina Kann, Samuel R. Bowman. (2020)  
**Intermediate-Task Transfer Learning with Pretrained Language Models: When and Why Does It Work?**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/9e594ae4ae9c38b6495810a8872f513ae19be29c)  
Influential Citation Count (7), SS-ID (9e594ae4ae9c38b6495810a8872f513ae19be29c)  

**ABSTRACT**  
While pretrained models such as BERT have shown large gains across natural language understanding tasks, their performance can be improved by further training the model on a data-rich intermediate task, before fine-tuning it on a target task. However, it is still poorly understood when and why intermediate-task training is beneficial for a given target task. To investigate this, we perform a large-scale study on the pretrained RoBERTa model with 110 intermediate-target task combinations. We further evaluate all trained models with 25 probing tasks meant to reveal the specific skills that drive transfer. We observe that intermediate tasks requiring high-level inference and reasoning abilities tend to work best. We also observe that target task performance is strongly correlated with higher-level abilities such as coreference resolution. However, we fail to observe more granular correlations between probing and target task performance, highlighting the need for further work on broad-coverage probing benchmarks. We also observe evidence that the forgetting of knowledge learned during pretraining may limit our analysis, highlighting the need for further work on transfer learning methods in these settings.

{{< /ci-details >}}

{{< ci-details summary="On the Cross-lingual Transferability of Monolingual Representations (Mikel Artetxe et al., 2019)">}}

Mikel Artetxe, Sebastian Ruder, Dani Yogatama. (2019)  
**On the Cross-lingual Transferability of Monolingual Representations**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/9e9d919c1de684ca42c8b581ec62c7aa685f431e)  
Influential Citation Count (68), SS-ID (9e9d919c1de684ca42c8b581ec62c7aa685f431e)  

**ABSTRACT**  
State-of-the-art unsupervised multilingual models (e.g., multilingual BERT) have been shown to generalize in a zero-shot cross-lingual setting. This generalization ability has been attributed to the use of a shared subword vocabulary and joint training across multiple languages giving rise to deep multilingual abstractions. We evaluate this hypothesis by designing an alternative approach that transfers a monolingual model to new languages at the lexical level. More concretely, we first train a transformer-based masked language model on one language, and transfer it to a new language by learning a new embedding matrix with the same masked language modeling objective, freezing parameters of all other layers. This approach does not rely on a shared vocabulary or joint training. However, we show that it is competitive with multilingual BERT on standard cross-lingual classification benchmarks and on a new Cross-lingual Question Answering Dataset (XQuAD). Our results contradict common beliefs of the basis of the generalization ability of multilingual models and suggest that deep monolingual models learn some abstractions that generalize across languages. We also release XQuAD as a more comprehensive cross-lingual benchmark, which comprises 240 paragraphs and 1190 question-answer pairs from SQuAD v1.1 translated into ten languages by professional translators.

{{< /ci-details >}}

{{< ci-details summary="BERT is Not an Interlingua and the Bias of Tokenization (Jasdeep Singh et al., 2019)">}}

Jasdeep Singh, Bryan McCann, R. Socher, Caiming Xiong. (2019)  
**BERT is Not an Interlingua and the Bias of Tokenization**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/9eb4cd1a4b4717c97c47e3dc4563a75779ae9390)  
Influential Citation Count (8), SS-ID (9eb4cd1a4b4717c97c47e3dc4563a75779ae9390)  

**ABSTRACT**  
Multilingual transfer learning can benefit both high- and low-resource languages, but the source of these improvements is not well understood. Cananical Correlation Analysis (CCA) of the internal representations of a pre- trained, multilingual BERT model reveals that the model partitions representations for each language rather than using a common, shared, interlingual space. This effect is magnified at deeper layers, suggesting that the model does not progressively abstract semantic con- tent while disregarding languages. Hierarchical clustering based on the CCA similarity scores between languages reveals a tree structure that mirrors the phylogenetic trees hand- designed by linguists. The subword tokenization employed by BERT provides a stronger bias towards such structure than character- and word-level tokenizations. We release a subset of the XNLI dataset translated into an additional 14 languages at https://www.github.com/salesforce/xnli_extension to assist further research into multilingual representations.

{{< /ci-details >}}

{{< ci-details summary="Cloze-driven Pretraining of Self-attention Networks (Alexei Baevski et al., 2019)">}}

Alexei Baevski, Sergey Edunov, Yinhan Liu, Luke Zettlemoyer, Michael Auli. (2019)  
**Cloze-driven Pretraining of Self-attention Networks**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/9f1c5777a193b2c3bb2b25e248a156348e5ba56d)  
Influential Citation Count (14), SS-ID (9f1c5777a193b2c3bb2b25e248a156348e5ba56d)  

**ABSTRACT**  
We present a new approach for pretraining a bi-directional transformer model that provides significant performance gains across a variety of language understanding problems. Our model solves a cloze-style word reconstruction task, where each word is ablated and must be predicted given the rest of the text. Experiments demonstrate large performance gains on GLUE and new state of the art results on NER as well as constituency parsing benchmarks, consistent with BERT. We also present a detailed analysis of a number of factors that contribute to effective pretraining, including data domain and size, model capacity, and variations on the cloze objective.

{{< /ci-details >}}

{{< ci-details summary="BERTRAM: Improved Word Embeddings Have Big Impact on Contextualized Model Performance (Timo Schick et al., 2019)">}}

Timo Schick, Hinrich Schütze. (2019)  
**BERTRAM: Improved Word Embeddings Have Big Impact on Contextualized Model Performance**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/9f4c37f154946e141a67ae2816c70b19241b3224)  
Influential Citation Count (3), SS-ID (9f4c37f154946e141a67ae2816c70b19241b3224)  

**ABSTRACT**  
Pretraining deep language models has led to large performance gains in NLP. Despite this success, Schick and Schütze (2020) recently showed that these models struggle to understand rare words. For static word embeddings, this problem has been addressed by separately learning representations for rare words. In this work, we transfer this idea to pretrained language models: We introduce BERTRAM, a powerful architecture based on BERT that is capable of inferring high-quality embeddings for rare words that are suitable as input representations for deep language models. This is achieved by enabling the surface form and contexts of a word to interact with each other in a deep architecture. Integrating BERTRAM into BERT leads to large performance increases due to improved representations of rare and medium frequency words on both a rare word probing task and three downstream tasks.

{{< /ci-details >}}

{{< ci-details summary="Analyzing the Structure of Attention in a Transformer Language Model (Jesse Vig et al., 2019)">}}

Jesse Vig, Y. Belinkov. (2019)  
**Analyzing the Structure of Attention in a Transformer Language Model**  
BlackboxNLP@ACL  
[Paper Link](https://www.semanticscholar.org/paper/a039ea239e37f53a2cb60c68e0a1967994353166)  
Influential Citation Count (8), SS-ID (a039ea239e37f53a2cb60c68e0a1967994353166)  

**ABSTRACT**  
The Transformer is a fully attention-based alternative to recurrent networks that has achieved state-of-the-art results across a range of NLP tasks. In this paper, we analyze the structure of attention in a Transformer language model, the GPT-2 small pretrained model. We visualize attention for individual instances and analyze the interaction between attention and syntax over a large corpus. We find that attention targets different parts of speech at different layer depths within the model, and that attention aligns with dependency relations most strongly in the middle layers. We also find that the deepest layers of the model capture the most distant relationships. Finally, we extract exemplar sentences that reveal highly specific patterns targeted by particular attention heads.

{{< /ci-details >}}

{{< ci-details summary="Distilling Task-Specific Knowledge from BERT into Simple Neural Networks (Raphael Tang et al., 2019)">}}

Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, Jimmy J. Lin. (2019)  
**Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/a08293b2c9c5bcddb023cc7eb3354d4d86bfae89)  
Influential Citation Count (26), SS-ID (a08293b2c9c5bcddb023cc7eb3354d4d86bfae89)  

**ABSTRACT**  
In the natural language processing literature, neural networks are becoming increasingly deeper and complex. The recent poster child of this trend is the deep language representation model, which includes BERT, ELMo, and GPT. These developments have led to the conviction that previous-generation, shallower neural networks for language understanding are obsolete. In this paper, however, we demonstrate that rudimentary, lightweight neural networks can still be made competitive without architecture changes, external training data, or additional input features. We propose to distill knowledge from BERT, a state-of-the-art language representation model, into a single-layer BiLSTM, as well as its siamese counterpart for sentence-pair tasks. Across multiple datasets in paraphrasing, natural language inference, and sentiment classification, we achieve comparable results with ELMo, while using roughly 100 times fewer parameters and 15 times less inference time.

{{< /ci-details >}}

{{< ci-details summary="What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models (Allyson Ettinger, 2019)">}}

Allyson Ettinger. (2019)  
**What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models**  
TACL  
[Paper Link](https://www.semanticscholar.org/paper/a0e49f65b6847437f262c59d0d399255101d0b75)  
Influential Citation Count (10), SS-ID (a0e49f65b6847437f262c59d0d399255101d0b75)  

**ABSTRACT**  
Pre-training by language modeling has become a popular and successful approach to NLP tasks, but we have yet to understand exactly what linguistic capacities these pre-training processes confer upon models. In this paper we introduce a suite of diagnostics drawn from human language experiments, which allow us to ask targeted questions about information used by language models for generating predictions in context. As a case study, we apply these diagnostics to the popular BERT model, finding that it can generally distinguish good from bad completions involving shared category or role reversal, albeit with less sensitivity than humans, and it robustly retrieves noun hypernyms, but it struggles with challenging inference and role-based event prediction— and, in particular, it shows clear insensitivity to the contextual impacts of negation.

{{< /ci-details >}}

{{< ci-details summary="A Matter of Framing: The Impact of Linguistic Formalism on Probing Results (Ilia Kuznetsov et al., 2020)">}}

Ilia Kuznetsov, Iryna Gurevych. (2020)  
**A Matter of Framing: The Impact of Linguistic Formalism on Probing Results**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/a160dbe78b0546679ec8a3140b3cf4614e3cc485)  
Influential Citation Count (0), SS-ID (a160dbe78b0546679ec8a3140b3cf4614e3cc485)  

**ABSTRACT**  
Deep pre-trained contextualized encoders like BERT (Delvin et al., 2019) demonstrate remarkable performance on a range of downstream tasks. A recent line of research in probing investigates the linguistic knowledge implicitly learned by these models during pre-training. While most work in probing operates on the task level, linguistic tasks are rarely uniform and can be represented in a variety of formalisms. Any linguistics-based probing study thereby inevitably commits to the formalism used to annotate the underlying data. Can the choice of formalism affect probing results? To investigate, we conduct an in-depth cross-formalism layer probing study in role semantics. We find linguistically meaningful differences in the encoding of semantic role- and proto-role information by BERT depending on the formalism and demonstrate that layer probing can detect subtle differences between the implementations of the same linguistic formalism. Our results suggest that linguistic formalism is an important dimension in probing studies, along with the commonly used cross-task and cross-lingual experimental settings.

{{< /ci-details >}}

{{< ci-details summary="All-but-the-Top: Simple and Effective Postprocessing for Word Representations (Jiaqi Mu et al., 2017)">}}

Jiaqi Mu, S. Bhat, P. Viswanath. (2017)  
**All-but-the-Top: Simple and Effective Postprocessing for Word Representations**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/a2d407962bb1f5fcd209114f5687d4c11bf9dfad)  
Influential Citation Count (30), SS-ID (a2d407962bb1f5fcd209114f5687d4c11bf9dfad)  

**ABSTRACT**  
Real-valued word representations have transformed NLP applications; popular examples are word2vec and GloVe, recognized for their ability to capture linguistic regularities. In this paper, we demonstrate a {\em very simple}, and yet counter-intuitive, postprocessing technique -- eliminate the common mean vector and a few top dominating directions from the word vectors -- that renders off-the-shelf representations {\em even stronger}. The postprocessing is empirically validated on a variety of lexical-level intrinsic tasks (word similarity, concept categorization, word analogy) and sentence-level tasks (semantic textural similarity and { text classification}) on multiple datasets and with a variety of representation methods and hyperparameter choices in multiple languages; in each case, the processed representations are consistently better than the original ones.

{{< /ci-details >}}

{{< ci-details summary="PERL: Pivot-based Domain Adaptation for Pre-trained Deep Contextualized Embedding Models (Eyal Ben-David et al., 2020)">}}

Eyal Ben-David, Carmel Rabinovitz, Roi Reichart. (2020)  
**PERL: Pivot-based Domain Adaptation for Pre-trained Deep Contextualized Embedding Models**  
Transactions of the Association for Computational Linguistics  
[Paper Link](https://www.semanticscholar.org/paper/a33b09d1a41db92ca14185a28f9163056ca2a115)  
Influential Citation Count (4), SS-ID (a33b09d1a41db92ca14185a28f9163056ca2a115)  

**ABSTRACT**  
Abstract Pivot-based neural representation models have led to significant progress in domain adaptation for NLP. However, previous research following this approach utilize only labeled data from the source domain and unlabeled data from the source and target domains, but neglect to incorporate massive unlabeled corpora that are not necessarily drawn from these domains. To alleviate this, we propose PERL: A representation learning model that extends contextualized word embedding models such as BERT (Devlin et al., 2019) with pivot-based fine-tuning. PERL outperforms strong baselines across 22 sentiment classification domain adaptation setups, improves in-domain model performance, yields effective reduced-size models, and increases model stability.1

{{< /ci-details >}}

{{< ci-details summary="DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (Victor Sanh et al., 2019)">}}

Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf. (2019)  
**DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/a54b56af24bb4873ed0163b77df63b92bd018ddc)  
Influential Citation Count (417), SS-ID (a54b56af24bb4873ed0163b77df63b92bd018ddc)  

**ABSTRACT**  
As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP), operating these large models in on-the-edge and/or under constrained computational training or inference budgets remains challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage knowledge distillation during the pre-training phase and show that it is possible to reduce the size of a BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive biases learned by larger models during pre-training, we introduce a triple loss combining language modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device study.

{{< /ci-details >}}

{{< ci-details summary="TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data (Pengcheng Yin et al., 2020)">}}

Pengcheng Yin, Graham Neubig, Wen-tau Yih, Sebastian Riedel. (2020)  
**TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/a5b1d1cab073cb746a990b37d42dc7b67763f881)  
Influential Citation Count (30), SS-ID (a5b1d1cab073cb746a990b37d42dc7b67763f881)  

**ABSTRACT**  
Recent years have witnessed the burgeoning of pretrained language models (LMs) for text-based natural language (NL) understanding tasks. Such models are typically trained on free-form NL text, hence may not be suitable for tasks like semantic parsing over structured data, which require reasoning over both free-form NL questions and structured tabular data (e.g., database tables). In this paper we present TaBERT, a pretrained LM that jointly learns representations for NL sentences and (semi-)structured tables. TaBERT is trained on a large corpus of 26 million tables and their English contexts. In experiments, neural semantic parsers using TaBERT as feature representation layers achieve new best results on the challenging weakly-supervised semantic parsing benchmark WikiTableQuestions, while performing competitively on the text-to-SQL dataset Spider.

{{< /ci-details >}}

{{< ci-details summary="Getting Closer to AI Complete Question Answering: A Set of Prerequisite Real Tasks (Anna Rogers et al., 2020)">}}

Anna Rogers, Olga Kovaleva, Matthew Downey, Anna Rumshisky. (2020)  
**Getting Closer to AI Complete Question Answering: A Set of Prerequisite Real Tasks**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/a87f0bac2ed58e8a79d33d0c1cf81c6407cd645f)  
Influential Citation Count (7), SS-ID (a87f0bac2ed58e8a79d33d0c1cf81c6407cd645f)  

**ABSTRACT**  
The recent explosion in question answering research produced a wealth of both factoid reading comprehension (RC) and commonsense reasoning datasets. Combining them presents a different kind of task: deciding not simply whether information is present in the text, but also whether a confident guess could be made for the missing information. We present QuAIL, the first RC dataset to combine text-based, world knowledge and unanswerable questions, and to provide question type annotation that would enable diagnostics of the reasoning strategies by a given QA system. QuAIL contains 15K multi-choice questions for 800 texts in 4 domains. Crucially, it offers both general and text-specific questions, unlikely to be found in pretraining data. We show that QuAIL poses substantial challenges to the current state-of-the-art systems, with a 30% drop in accuracy compared to the most similar existing dataset.

{{< /ci-details >}}

{{< ci-details summary="SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization (Haoming Jiang et al., 2019)">}}

Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, T. Zhao. (2019)  
**SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/ab70853cd5912c470f6ff95e95481980f0a2a41b)  
Influential Citation Count (24), SS-ID (ab70853cd5912c470f6ff95e95481980f0a2a41b)  

**ABSTRACT**  
Transfer learning has fundamentally changed the landscape of natural language processing (NLP). Many state-of-the-art models are first pre-trained on a large text corpus and then fine-tuned on downstream tasks. However, due to limited data resources from downstream tasks and the extremely high complexity of pre-trained models, aggressive fine-tuning often causes the fine-tuned model to overfit the training data of downstream tasks and fail to generalize to unseen data. To address such an issue in a principled manner, we propose a new learning framework for robust and efficient fine-tuning for pre-trained models to attain better generalization performance. The proposed framework contains two important ingredients: 1. Smoothness-inducing regularization, which effectively manages the complexity of the model; 2. Bregman proximal point optimization, which is an instance of trust-region methods and can prevent aggressive updating. Our experiments show that the proposed framework achieves new state-of-the-art performance on a number of NLP tasks including GLUE, SNLI, SciTail and ANLI. Moreover, it also outperforms the state-of-the-art T5 model, which is the largest pre-trained model containing 11 billion parameters, on GLUE.

{{< /ci-details >}}

{{< ci-details summary="Thieves on Sesame Street! Model Extraction of BERT-based APIs (Kalpesh Krishna et al., 2019)">}}

Kalpesh Krishna, Gaurav Singh Tomar, Ankur P. Parikh, Nicolas Papernot, Mohit Iyyer. (2019)  
**Thieves on Sesame Street! Model Extraction of BERT-based APIs**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/ac713aebdcc06f15f8ea61e1140bb360341fdf27)  
Influential Citation Count (15), SS-ID (ac713aebdcc06f15f8ea61e1140bb360341fdf27)  

**ABSTRACT**  
We study the problem of model extraction in natural language processing, in which an adversary with only query access to a victim model attempts to reconstruct a local copy of that model. Assuming that both the adversary and victim model fine-tune a large pretrained language model such as BERT (Devlin et al. 2019), we show that the adversary does not need any real training data to successfully mount the attack. In fact, the attacker need not even use grammatical or semantically meaningful queries: we show that random sequences of words coupled with task-specific heuristics form effective queries for model extraction on a diverse set of NLP tasks, including natural language inference and question answering. Our work thus highlights an exploit only made feasible by the shift towards transfer learning methods within the NLP community: for a query budget of a few hundred dollars, an attacker can extract a model that performs only slightly worse than the victim model. Finally, we study two defense strategies against model extraction---membership classification and API watermarking---which while successful against naive adversaries, are ineffective against more sophisticated ones.

{{< /ci-details >}}

{{< ci-details summary="Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment (Di Jin et al., 2019)">}}

Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits. (2019)  
**Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/ae04f3d011511ad8ed7ffdf9fcfb7f11e6899ca2)  
Influential Citation Count (68), SS-ID (ae04f3d011511ad8ed7ffdf9fcfb7f11e6899ca2)  

**ABSTRACT**  
Machine learning algorithms are often vulnerable to adversarial examples that have imperceptible alterations from the original counterparts but can fool the state-of-the-art models. It is helpful to evaluate or even improve the robustness of these models by exposing the maliciously crafted adversarial examples. In this paper, we present TextFooler, a simple but strong baseline to generate adversarial text. By applying it to two fundamental natural language tasks, text classification and textual entailment, we successfully attacked three target models, including the powerful pre-trained BERT, and the widely used convolutional and recurrent neural networks. We demonstrate three advantages of this framework: (1) effective—it outperforms previous attacks by success rate and perturbation rate, (2) utility-preserving—it preserves semantic content, grammaticality, and correct types classified by humans, and (3) efficient—it generates adversarial text with computational complexity linear to the text length.1

{{< /ci-details >}}

{{< ci-details summary="Are Sixteen Heads Really Better than One? (Paul Michel et al., 2019)">}}

Paul Michel, Omer Levy, Graham Neubig. (2019)  
**Are Sixteen Heads Really Better than One?**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/b03c7ff961822183bab66b2e594415e585d3fd09)  
Influential Citation Count (49), SS-ID (b03c7ff961822183bab66b2e594415e585d3fd09)  

**ABSTRACT**  
Attention is a powerful and ubiquitous mechanism for allowing neural models to focus on particular salient pieces of information by taking their weighted average when making predictions. In particular, multi-headed attention is a driving force behind many recent state-of-the-art NLP models such as Transformer-based MT models and BERT. These models apply multiple attention mechanisms in parallel, with each attention "head" potentially focusing on different parts of the input, which makes it possible to express sophisticated functions beyond the simple weighted average. In this paper we make the surprising observation that even if models have been trained using multiple heads, in practice, a large percentage of attention heads can be removed at test time without significantly impacting performance. In fact, some layers can even be reduced to a single head. We further examine greedy algorithms for pruning down models, and the potential speed, memory efficiency, and accuracy improvements obtainable therefrom. Finally, we analyze the results with respect to which parts of the model are more reliant on having multiple heads, and provide precursory evidence that training dynamics play a role in the gains provided by multi-head attention.

{{< /ci-details >}}

{{< ci-details summary="A Mutual Information Maximization Perspective of Language Representation Learning (Lingpeng Kong et al., 2019)">}}

Lingpeng Kong, Cyprien de Masson d'Autume, Wang Ling, Lei Yu, Zihang Dai, Dani Yogatama. (2019)  
**A Mutual Information Maximization Perspective of Language Representation Learning**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/b04889922aae7f799affb2ae6508bc5f5c989567)  
Influential Citation Count (11), SS-ID (b04889922aae7f799affb2ae6508bc5f5c989567)  

**ABSTRACT**  
We show state-of-the-art word representation learning methods maximize an objective function that is a lower bound on the mutual information between different parts of a word sequence (i.e., a sentence). Our formulation provides an alternative perspective that unifies classical word embedding models (e.g., Skip-gram) and modern contextual embeddings (e.g., BERT, XLNet). In addition to enhancing our theoretical understanding of these methods, our derivation leads to a principled framework that can be used to construct new self-supervised tasks. We provide an example by drawing inspirations from related methods based on mutual information maximization that have been successful in computer vision, and introduce a simple self-supervised objective that maximizes the mutual information between a global sentence representation and n-grams in the sentence. Our analysis offers a holistic view of representation learning methods to transfer knowledge and translate progress across multiple domains (e.g., natural language processing, computer vision, audio processing).

{{< /ci-details >}}

{{< ci-details summary="Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks (Jason Phang et al., 2018)">}}

Jason Phang, Thibault Févry, Samuel R. Bowman. (2018)  
**Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/b47381e04739ea3f392ba6c8faaf64105493c196)  
Influential Citation Count (38), SS-ID (b47381e04739ea3f392ba6c8faaf64105493c196)  

**ABSTRACT**  
Pretraining sentence encoders with language modeling and related unsupervised tasks has recently been shown to be very effective for language understanding tasks. By supplementing language model-style pretraining with further training on data-rich supervised tasks, such as natural language inference, we obtain additional performance improvements on the GLUE benchmark. Applying supplementary training on BERT (Devlin et al., 2018), we attain a GLUE score of 81.8---the state of the art (as of 02/24/2019) and a 1.4 point improvement over BERT. We also observe reduced variance across random restarts in this setting. Our approach yields similar improvements when applied to ELMo (Peters et al., 2018a) and Radford et al. (2018)'s model. In addition, the benefits of supplementary training are particularly pronounced in data-constrained regimes, as we show in experiments with artificially limited training data.

{{< /ci-details >}}

{{< ci-details summary="Do Attention Heads in BERT Track Syntactic Dependencies? (Phu Mon Htut et al., 2019)">}}

Phu Mon Htut, Jason Phang, Shikha Bordia, Samuel R. Bowman. (2019)  
**Do Attention Heads in BERT Track Syntactic Dependencies?**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/ba8215e77f35b0d947c7cec39c45df4516e93421)  
Influential Citation Count (12), SS-ID (ba8215e77f35b0d947c7cec39c45df4516e93421)  

**ABSTRACT**  
We investigate the extent to which individual attention heads in pretrained transformer language models, such as BERT and RoBERTa, implicitly capture syntactic dependency relations. We employ two methods---taking the maximum attention weight and computing the maximum spanning tree---to extract implicit dependency relations from the attention weights of each layer/head, and compare them to the ground-truth Universal Dependency (UD) trees. We show that, for some UD relation types, there exist heads that can recover the dependency type significantly better than baselines on parsed English text, suggesting that some self-attention heads act as a proxy for syntactic structure. We also analyze BERT fine-tuned on two datasets---the syntax-oriented CoLA and the semantics-oriented MNLI---to investigate whether fine-tuning affects the patterns of their self-attention, but we do not observe substantial differences in the overall dependency relations extracted using our methods. Our results suggest that these models have some specialist attention heads that track individual dependency types, but no generalist head that performs holistic parsing significantly better than a trivial baseline, and that analyzing attention weights directly may not reveal much of the syntactic knowledge that BERT-style models are known to learn.

{{< /ci-details >}}

{{< ci-details summary="Does BERT Make Any Sense? Interpretable Word Sense Disambiguation with Contextualized Embeddings (Gregor Wiedemann et al., 2019)">}}

Gregor Wiedemann, Steffen Remus, Avi Chawla, Chris Biemann. (2019)  
**Does BERT Make Any Sense? Interpretable Word Sense Disambiguation with Contextualized Embeddings**  
KONVENS  
[Paper Link](https://www.semanticscholar.org/paper/ba8b3d0d2b09bc2b56c6d3f153919786d9fc3075)  
Influential Citation Count (6), SS-ID (ba8b3d0d2b09bc2b56c6d3f153919786d9fc3075)  

**ABSTRACT**  
Contextualized word embeddings (CWE) such as provided by ELMo (Peters et al., 2018), Flair NLP (Akbik et al., 2018), or BERT (Devlin et al., 2019) are a major recent innovation in NLP. CWEs provide semantic vector representations of words depending on their respective context. Their advantage over static word embeddings has been shown for a number of tasks, such as text classification, sequence tagging, or machine translation. Since vectors of the same word type can vary depending on the respective context, they implicitly provide a model for word sense disambiguation (WSD). We introduce a simple but effective approach to WSD using a nearest neighbor classification on CWEs. We compare the performance of different CWE models for the task and can report improvements above the current state of the art for two standard WSD benchmark datasets. We further show that the pre-trained BERT model is able to place polysemic words into distinct 'sense' regions of the embedding space, while ELMo and Flair NLP do not seem to possess this ability.

{{< /ci-details >}}

{{< ci-details summary="Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping (Jesse Dodge et al., 2020)">}}

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, Noah A. Smith. (2020)  
**Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/baf60d13c98916b77b09bc525ede1cd610ed1db5)  
Influential Citation Count (17), SS-ID (baf60d13c98916b77b09bc525ede1cd610ed1db5)  

**ABSTRACT**  
Fine-tuning pretrained contextual word embedding models to supervised downstream tasks has become commonplace in natural language processing. This process, however, is often brittle: even with the same hyperparameter values, distinct random seeds can lead to substantially different results. To better understand this phenomenon, we experiment with four datasets from the GLUE benchmark, fine-tuning BERT hundreds of times on each while varying only the random seeds. We find substantial performance increases compared to previously reported results, and we quantify how the performance of the best-found model varies as a function of the number of fine-tuning trials. Further, we examine two factors influenced by the choice of random seed: weight initialization and training data order. We find that both contribute comparably to the variance of out-of-sample performance, and that some weight initializations perform well across all tasks explored. On small datasets, we observe that many fine-tuning trials diverge part of the way through training, and we offer best practices for practitioners to stop training less promising runs early. We publicly release all of our experimental data, including training and validation scores for 2,100 trials, to encourage further analysis of training dynamics during fine-tuning.

{{< /ci-details >}}

{{< ci-details summary="Improving BERT Fine-tuning with Embedding Normalization (Wenxuan Zhou et al., 2019)">}}

Wenxuan Zhou, Junyi Du, Xiang Ren. (2019)  
**Improving BERT Fine-tuning with Embedding Normalization**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/bb6e205f56f064ae76703b40147422483c438ef6)  
Influential Citation Count (0), SS-ID (bb6e205f56f064ae76703b40147422483c438ef6)  

**ABSTRACT**  
Large pre-trained sentence encoders like BERT start a new chapter in natural language processing. A common practice to apply pre-trained BERT to sequence classification tasks (e.g., classification of sentences or sentence pairs) is by feeding the embedding of [CLS] token (in the last layer) to a task-specific classification layer, and then fine tune the model parameters of BERT and classifier jointly. In this paper, we conduct systematic analysis over several sequence classification datasets to examine the embedding values of [CLS] token before the fine tuning phase, and present the biased embedding distribution issue---i.e., embedding values of [CLS] concentrate on a few dimensions and are non-zero centered. Such biased embedding brings challenge to the optimization process during fine-tuning as gradients of [CLS] embedding may explode and result in degraded model performance. We further propose several simple yet effective normalization methods to modify the [CLS] embedding during the fine-tuning. Compared with the previous practice, neural classification model with the normalized embedding shows improvements on several text classification tasks, demonstrates the effectiveness of our method.

{{< /ci-details >}}

{{< ci-details summary="Large Batch Optimization for Deep Learning: Training BERT in 76 minutes (Yang You et al., 2019)">}}

Yang You, Jing Li, Sashank J. Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, J. Demmel, K. Keutzer, Cho-Jui Hsieh. (2019)  
**Large Batch Optimization for Deep Learning: Training BERT in 76 minutes**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/bc789aef715498e79a74f857fa090ece9e383bf1)  
Influential Citation Count (53), SS-ID (bc789aef715498e79a74f857fa090ece9e383bf1)  

**ABSTRACT**  
Training large deep neural networks on massive datasets is computationally very challenging. There has been recent surge in interest in using large batch stochastic optimization methods to tackle this issue. The most prominent algorithm in this line of research is LARS, which by employing layerwise adaptive learning rates trains ResNet on ImageNet in a few minutes. However, LARS performs poorly for attention models like BERT, indicating that its performance gains are not consistent across tasks. In this paper, we first study a principled layerwise adaptation strategy to accelerate training of deep neural networks using large mini-batches. Using this strategy, we develop a new layerwise adaptive large batch optimization technique called LAMB; we then provide convergence analysis of LAMB as well as LARS, showing convergence to a stationary point in general nonconvex settings. Our empirical results demonstrate the superior performance of LAMB across various tasks such as BERT and ResNet-50 training with very little hyperparameter tuning. In particular, for BERT training, our optimizer enables use of very large batch sizes of 32868 without any degradation of performance. By increasing the batch size to the memory limit of a TPUv3 Pod, BERT training time can be reduced from 3 days to just 76 minutes (Table 1). The LAMB implementation is available at this https URL

{{< /ci-details >}}

{{< ci-details summary="Visualizing Attention in Transformer-Based Language Representation Models (Jesse Vig, 2019)">}}

Jesse Vig. (2019)  
**Visualizing Attention in Transformer-Based Language Representation Models**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/beb051c652f02c2d5829d783fbc4f3acce99bc3c)  
Influential Citation Count (5), SS-ID (beb051c652f02c2d5829d783fbc4f3acce99bc3c)  

**ABSTRACT**  
We present an open-source tool for visualizing multi-head self-attention in Transformer-based language representation models. The tool extends earlier work by visualizing attention at three levels of granularity: the attention-head level, the model level, and the neuron level. We describe how each of these views can help to interpret the model, and we demonstrate the tool on the BERT model and the OpenAI GPT-2 model. We also present three use cases for analyzing GPT-2: detecting model bias, identifying recurring patterns, and linking neurons to model behavior.

{{< /ci-details >}}

{{< ci-details summary="Syntax-Infused Transformer and BERT models for Machine Translation and Natural Language Understanding (Dhanasekar Sundararaman et al., 2019)">}}

Dhanasekar Sundararaman, Vivek Subramanian, Guoyin Wang, Shijing Si, Dinghan Shen, Dong Wang, L. Carin. (2019)  
**Syntax-Infused Transformer and BERT models for Machine Translation and Natural Language Understanding**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/beb91a773677872fc21f08722bdcc737bf5917b5)  
Influential Citation Count (1), SS-ID (beb91a773677872fc21f08722bdcc737bf5917b5)  

**ABSTRACT**  
Attention-based models have shown significant improvement over traditional algorithms in several NLP tasks. The Transformer, for instance, is an illustrative example that generates abstract representations of tokens inputted to an encoder based on their relationships to all tokens in a sequence. Recent studies have shown that although such models are capable of learning syntactic features purely by seeing examples, explicitly feeding this information to deep learning models can significantly enhance their performance. Leveraging syntactic information like part of speech (POS) may be particularly beneficial in limited training data settings for complex models such as the Transformer. We show that the syntax-infused Transformer with multiple features achieves an improvement of 0.7 BLEU when trained on the full WMT 14 English to German translation dataset and a maximum improvement of 1.99 BLEU points when trained on a fraction of the dataset. In addition, we find that the incorporation of syntax into BERT fine-tuning outperforms baseline on a number of downstream tasks from the GLUE benchmark.

{{< /ci-details >}}

{{< ci-details summary="Knowledge Enhanced Contextual Word Representations (Matthew E. Peters et al., 2019)">}}

Matthew E. Peters, Mark Neumann, IV RobertL.Logan, Roy Schwartz, V. Joshi, Sameer Singh, Noah A. Smith. (2019)  
**Knowledge Enhanced Contextual Word Representations**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/bfeb827d06c1a3583b5cc6d25241203a81f6af09)  
Influential Citation Count (61), SS-ID (bfeb827d06c1a3583b5cc6d25241203a81f6af09)  

**ABSTRACT**  
Contextual word representations, typically trained on unstructured, unlabeled text, do not contain any explicit grounding to real world entities and are often unable to remember facts about those entities. We propose a general method to embed multiple knowledge bases (KBs) into large scale models, and thereby enhance their representations with structured, human-curated knowledge. For each KB, we first use an integrated entity linker to retrieve relevant entity embeddings, then update contextual word representations via a form of word-to-entity attention. In contrast to previous approaches, the entity linkers and self-supervised language modeling objective are jointly trained end-to-end in a multitask setting that combines a small amount of entity linking supervision with a large amount of raw text. After integrating WordNet and a subset of Wikipedia into BERT, the knowledge enhanced BERT (KnowBert) demonstrates improved perplexity, ability to recall facts as measured in a probing task and downstream performance on relationship extraction, entity typing, and word sense disambiguation. KnowBert’s runtime is comparable to BERT’s and it scales to large KBs.

{{< /ci-details >}}

{{< ci-details summary="TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection (Siddhant Garg et al., 2019)">}}

Siddhant Garg, Thuy Vu, Alessandro Moschitti. (2019)  
**TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/c12e6c65e1de5d3993c5b65d0e234ae1f60c85ae)  
Influential Citation Count (12), SS-ID (c12e6c65e1de5d3993c5b65d0e234ae1f60c85ae)  

**ABSTRACT**  
We propose TandA, an effective technique for fine-tuning pre-trained Transformer models for natural language tasks. Specifically, we first transfer a pre-trained model into a model for a general task by fine-tuning it with a large and high-quality dataset. We then perform a second fine-tuning step to adapt the transferred model to the target domain. We demonstrate the benefits of our approach for answer sentence selection, which is a well-known inference task in Question Answering. We built a large scale dataset to enable the transfer step, exploiting the Natural Questions dataset. Our approach establishes the state of the art on two well-known benchmarks, WikiQA and TREC-QA, achieving the impressive MAP scores of 92% and 94.3%, respectively, which largely outperform the the highest scores of 83.4% and 87.5% of previous work. We empirically show that TandA generates more stable and robust models reducing the effort required for selecting optimal hyper-parameters. Additionally, we show that the transfer step of TandA makes the adaptation step more robust to noise. This enables a more effective use of noisy datasets for fine-tuning. Finally, we also confirm the positive impact of TandA in an industrial setting, using domain specific datasets subject to different types of noise.

{{< /ci-details >}}

{{< ci-details summary="When Bert Forgets How To POS: Amnesic Probing of Linguistic Properties and MLM Predictions (Yanai Elazar et al., 2020)">}}

Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg. (2020)  
**When Bert Forgets How To POS: Amnesic Probing of Linguistic Properties and MLM Predictions**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/c8b00d4706fc8979a9c5f410addccbcfe1c0d894)  
Influential Citation Count (1), SS-ID (c8b00d4706fc8979a9c5f410addccbcfe1c0d894)  

**ABSTRACT**  
A growing body of work makes use of probing in order to investigate the working of neural models, often considered black boxes. Recently, an ongoing debate emerged surrounding the limitations of the probing paradigm. In this work, we point out the inability to infer behavioral conclusions from probing results, and offer an alternative method which is focused on how the information is being used, rather than on what information is encoded. Our method, Amnesic Probing, follows the intuition that the utility of a property for a given task can be assessed by measuring the influence of a causal intervention which removes it from the representation. Equipped with this new analysis tool, we can now ask questions that were not possible before, e.g. is part-of-speech information important for word prediction? We perform a series of analyses on BERT to answer these types of questions. Our findings demonstrate that conventional probing performance is not correlated to task importance, and we call for increased scrutiny of claims that draw behavioral or causal conclusions from probing results.

{{< /ci-details >}}

{{< ci-details summary="On the use of BERT for Neural Machine Translation (S. Clinchant et al., 2019)">}}

S. Clinchant, K. Jung, Vassilina Nikoulina. (2019)  
**On the use of BERT for Neural Machine Translation**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/c93b2d64fce8737506757bbce51e17b533f9285b)  
Influential Citation Count (9), SS-ID (c93b2d64fce8737506757bbce51e17b533f9285b)  

**ABSTRACT**  
Exploiting large pretrained models for various NMT tasks have gained a lot of visibility recently. In this work we study how BERT pretrained models could be exploited for supervised Neural Machine Translation. We compare various ways to integrate pretrained BERT model with NMT model and study the impact of the monolingual data used for BERT training on the final translation quality. We use WMT-14 English-German, IWSLT15 English-German and IWSLT14 English-Russian datasets for these experiments. In addition to standard task test set evaluation, we perform evaluation on out-of-domain test sets and noise injected test sets, in order to assess how BERT pretrained representations affect model robustness.

{{< /ci-details >}}

{{< ci-details summary="Do Neural Language Representations Learn Physical Commonsense? (Maxwell Forbes et al., 2019)">}}

Maxwell Forbes, Ari Holtzman, Yejin Choi. (2019)  
**Do Neural Language Representations Learn Physical Commonsense?**  
CogSci  
[Paper Link](https://www.semanticscholar.org/paper/cc02386375b1262c3a1d5525154eaea24c761d15)  
Influential Citation Count (3), SS-ID (cc02386375b1262c3a1d5525154eaea24c761d15)  

**ABSTRACT**  
Humans understand language based on the rich background knowledge about how the physical world works, which in turn allows us to reason about the physical world through language. In addition to the properties of objects (e.g., boats require fuel) and their affordances, i.e., the actions that are applicable to them (e.g., boats can be driven), we can also reason about if-then inferences between what properties of objects imply the kind of actions that are applicable to them (e.g., that if we can drive something then it likely requires fuel).  In this paper, we investigate the extent to which state-of-the-art neural language representations, trained on a vast amount of natural language text, demonstrate physical commonsense reasoning. While recent advancements of neural language models have demonstrated strong performance on various types of natural language inference tasks, our study based on a dataset of over 200k newly collected annotations suggests that neural language representations still only learn associations that are explicitly written down.

{{< /ci-details >}}

{{< ci-details summary="Q8BERT: Quantized 8Bit BERT (Ofir Zafrir et al., 2019)">}}

Ofir Zafrir, Guy Boudoukh, Peter Izsak, Moshe Wasserblat. (2019)  
**Q8BERT: Quantized 8Bit BERT**  
2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing - NeurIPS Edition (EMC2-NIPS)  
[Paper Link](https://www.semanticscholar.org/paper/ce106590145e89ea4b621c99665862967ccf5dac)  
Influential Citation Count (17), SS-ID (ce106590145e89ea4b621c99665862967ccf5dac)  

**ABSTRACT**  
Recently, pre-trained Transformer [1] based language models such as BERT [2] and GPT [3], have shown great improvement in many Natural Language Processing (NLP) tasks. However, these models contain a large amount of parameters. The emergence of even larger and more accurate models such as GPT2 [4] and Megatron11https://github.com/NVIDIA/Megatron-LM, suggest a trend of large pre-trained Transformer models. However, using these large models in production environments is a complex task requiring a large amount of compute, memory and power resources. In this work we show how to perform quantization-aware training during the fine-tuning phase of BERT in order to compress BERT by 4x with minimal accuracy loss. Furthermore, the produced quantized model can accelerate inference speed if it is optimized for 8bit Integer supporting hardware.

{{< /ci-details >}}

{{< ci-details summary="Attention is not not Explanation (Sarah Wiegreffe et al., 2019)">}}

Sarah Wiegreffe, Yuval Pinter. (2019)  
**Attention is not not Explanation**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/ce177672b00ddf46e4906157a7e997ca9338b8b9)  
Influential Citation Count (12), SS-ID (ce177672b00ddf46e4906157a7e997ca9338b8b9)  

**ABSTRACT**  
Attention mechanisms play a central role in NLP systems, especially within recurrent neural network (RNN) models. Recently, there has been increasing interest in whether or not the intermediate representations offered by these modules may be used to explain the reasoning for a model’s prediction, and consequently reach insights regarding the model’s decision-making process. A recent paper claims that ‘Attention is not Explanation’ (Jain and Wallace, 2019). We challenge many of the assumptions underlying this work, arguing that such a claim depends on one’s definition of explanation, and that testing it needs to take into account all elements of the model. We propose four alternative tests to determine when/whether attention can be used as explanation: a simple uniform-weights baseline; a variance calibration based on multiple random seed runs; a diagnostic framework using frozen weights from pretrained models; and an end-to-end adversarial attention training protocol. Each allows for meaningful interpretation of attention mechanisms in RNN models. We show that even when reliable adversarial distributions can be found, they don’t perform well on the simple diagnostic, indicating that prior work does not disprove the usefulness of attention mechanisms for explainability.

{{< /ci-details >}}

{{< ci-details summary="Language Models as Knowledge Bases? (Fabio Petroni et al., 2019)">}}

Fabio Petroni, Tim Rocktäschel, Patrick Lewis, A. Bakhtin, Yuxiang Wu, Alexander H. Miller, S. Riedel. (2019)  
**Language Models as Knowledge Bases?**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d0086b86103a620a86bc918746df0aa642e2a8a3)  
Influential Citation Count (115), SS-ID (d0086b86103a620a86bc918746df0aa642e2a8a3)  

**ABSTRACT**  
Recent progress in pretraining language models on large textual corpora led to a surge of improvements for downstream NLP tasks. Whilst learning linguistic knowledge, these models may also be storing relational knowledge present in the training data, and may be able to answer queries structured as “fill-in-the-blank” cloze statements. Language models have many advantages over structured knowledge bases: they require no schema engineering, allow practitioners to query about an open class of relations, are easy to extend to more data, and require no human supervision to train. We present an in-depth analysis of the relational knowledge already present (without fine-tuning) in a wide range of state-of-the-art pretrained language models. We find that (i) without fine-tuning, BERT contains relational knowledge competitive with traditional NLP methods that have some access to oracle knowledge, (ii) BERT also does remarkably well on open-domain question answering against a supervised baseline, and (iii) certain types of factual knowledge are learned much more readily than others by standard language model pretraining approaches. The surprisingly strong ability of these models to recall factual knowledge without any fine-tuning demonstrates their potential as unsupervised open-domain QA systems. The code to reproduce our analysis is available at https://github.com/facebookresearch/LAMA.

{{< /ci-details >}}

{{< ci-details summary="FreeLB: Enhanced Adversarial Training for Natural Language Understanding (Chen Zhu et al., 2019)">}}

Chen Zhu, Yu Cheng, Zhe Gan, S. Sun, T. Goldstein, Jingjing Liu. (2019)  
**FreeLB: Enhanced Adversarial Training for Natural Language Understanding**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/d01fa0311e8e15b8b874b376123530c815f52852)  
Influential Citation Count (34), SS-ID (d01fa0311e8e15b8b874b376123530c815f52852)  

**ABSTRACT**  
Adversarial training, which minimizes the maximal risk for label-preserving input perturbations, has proved to be effective for improving the generalization of language models. In this work, we propose a novel adversarial training algorithm, FreeLB, that promotes higher invariance in the embedding space, by adding adversarial perturbations to word embeddings and minimizing the resultant adversarial risk inside different regions around input samples. To validate the effectiveness of the proposed approach, we apply it to Transformer-based models for natural language understanding and commonsense reasoning tasks. Experiments on the GLUE benchmark show that when applied only to the finetuning stage, it is able to improve the overall test scores of BERT-base model from 78.3 to 79.4, and RoBERTa-large model from 88.5 to 88.8. In addition, the proposed approach achieves state-of-the-art single-model test accuracies of 85.44\% and 67.75\% on ARC-Easy and ARC-Challenge. Experiments on CommonsenseQA benchmark further demonstrate that FreeLB can be generalized and boost the performance of RoBERTa-large model on other tasks as well. Code is available at \url{this https URL .

{{< /ci-details >}}

{{< ci-details summary="FreeLB: Enhanced Adversarial Training for Language Understanding (Chen Zhu et al., 2019)">}}

Chen Zhu, Yu Cheng, Zhe Gan, S. Sun, T. Goldstein, Jingjing Liu. (2019)  
**FreeLB: Enhanced Adversarial Training for Language Understanding**  
ICLR 2020  
[Paper Link](https://www.semanticscholar.org/paper/d2038ced371e45aee3651c7a595c4566f4826b9f)  
Influential Citation Count (16), SS-ID (d2038ced371e45aee3651c7a595c4566f4826b9f)  

**ABSTRACT**  
Adversarial training, which minimizes the maximal risk for label-preserving input perturbations, has proved to be effective for improving the generalization of language models. In this work, we propose a novel adversarial training algorithm - FreeLB, that promotes higher robustness and invariance in the embedding space, by adding adversarial perturbations to word embeddings and minimizing the resultant adversarial risk inside different regions around input samples. To validate the effectiveness of the proposed approach, we apply it to Transformer-based models for natural language understanding and commonsense reasoning tasks. Experiments on the GLUE benchmark show that when applied only to the finetuning stage, it is able to improve the overall test scores of BERT-based model from 78.3 to 79.4, and RoBERTa-large model from 88.5 to 88.8. In addition, the proposed approach achieves state-of-the-art test accuracies of 85.39\% and 67.32\% on ARC-Easy and ARC-Challenge. Experiments on CommonsenseQA benchmark further demonstrate that FreeLB can be generalized and boost the performance of RoBERTa-large model on other tasks as well.

{{< /ci-details >}}

{{< ci-details summary="Interpreting Pretrained Contextualized Representations via Reductions to Static Embeddings (Rishi Bommasani et al., 2020)">}}

Rishi Bommasani, Kelly Davis, Claire Cardie. (2020)  
**Interpreting Pretrained Contextualized Representations via Reductions to Static Embeddings**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/d34580c522c79d5cde620331dd9ffb18643a8090)  
Influential Citation Count (17), SS-ID (d34580c522c79d5cde620331dd9ffb18643a8090)  

**ABSTRACT**  
Contextualized representations (e.g. ELMo, BERT) have become the default pretrained representations for downstream NLP applications. In some settings, this transition has rendered their static embedding predecessors (e.g. Word2Vec, GloVe) obsolete. As a side-effect, we observe that older interpretability methods for static embeddings — while more diverse and mature than those available for their dynamic counterparts — are underutilized in studying newer contextualized representations. Consequently, we introduce simple and fully general methods for converting from contextualized representations to static lookup-table embeddings which we apply to 5 popular pretrained models and 9 sets of pretrained weights. Our analysis of the resulting static embeddings notably reveals that pooling over many contexts significantly improves representational quality under intrinsic evaluation. Complementary to analyzing representational quality, we consider social biases encoded in pretrained representations with respect to gender, race/ethnicity, and religion and find that bias is encoded disparately across pretrained models and internal layers even for models with the same training data. Concerningly, we find dramatic inconsistencies between social bias estimators for word embeddings.

{{< /ci-details >}}

{{< ci-details summary="Visualizing and Understanding the Effectiveness of BERT (Y. Hao et al., 2019)">}}

Y. Hao, Li Dong, Furu Wei, Ke Xu. (2019)  
**Visualizing and Understanding the Effectiveness of BERT**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d3cacb4806886eb2fe59c90d4b6f822c24ff1822)  
Influential Citation Count (3), SS-ID (d3cacb4806886eb2fe59c90d4b6f822c24ff1822)  

**ABSTRACT**  
Language model pre-training, such as BERT, has achieved remarkable results in many NLP tasks. However, it is unclear why the pre-training-then-fine-tuning paradigm can improve performance and generalization capability across different tasks. In this paper, we propose to visualize loss landscapes and optimization trajectories of fine-tuning BERT on specific datasets. First, we find that pre-training reaches a good initial point across downstream tasks, which leads to wider optima and easier optimization compared with training from scratch. We also demonstrate that the fine-tuning procedure is robust to overfitting, even though BERT is highly over-parameterized for downstream tasks. Second, the visualization results indicate that fine-tuning BERT tends to generalize better because of the flat and wide optima, and the consistency between the training loss surface and the generalization error surface. Third, the lower layers of BERT are more invariant during fine-tuning, which suggests that the layers that are close to input learn more transferable representations of language.

{{< /ci-details >}}

{{< ci-details summary="RNNs Implicitly Implement Tensor Product Representations (R. Thomas McCoy et al., 2018)">}}

R. Thomas McCoy, Tal Linzen, Ewan Dunbar, P. Smolensky. (2018)  
**RNNs Implicitly Implement Tensor Product Representations**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/d3ded34ff3378aadaa9a7c10e51cef6d04391a86)  
Influential Citation Count (2), SS-ID (d3ded34ff3378aadaa9a7c10e51cef6d04391a86)  

**ABSTRACT**  
Recurrent neural networks (RNNs) can learn continuous vector representations of symbolic structures such as sequences and sentences; these representations often exhibit linear regularities (analogies). Such regularities motivate our hypothesis that RNNs that show such regularities implicitly compile symbolic structures into tensor product representations (TPRs; Smolensky, 1990), which additively combine tensor products of vectors representing roles (e.g., sequence positions) and vectors representing fillers (e.g., particular words). To test this hypothesis, we introduce Tensor Product Decomposition Networks (TPDNs), which use TPRs to approximate existing vector representations. We demonstrate using synthetic data that TPDNs can successfully approximate linear and tree-based RNN autoencoder representations, suggesting that these representations exhibit interpretable compositional structure; we explore the settings that lead RNNs to induce such structure-sensitive representations. By contrast, further TPDN experiments show that the representations of four models trained to encode naturally-occurring sentences can be largely approximated with a bag of words, with only marginal improvements from more sophisticated structures. We conclude that TPDNs provide a powerful method for interpreting vector representations, and that standard RNNs can induce compositional sequence representations that are remarkably well approximated by TPRs; at the same time, existing training tasks for sentence representation learning may not be sufficient for inducing robust structural representations.

{{< /ci-details >}}

{{< ci-details summary="StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding (Wei Wang et al., 2019)">}}

Wei Wang, Bin Bi, Ming Yan, Chen Wu, Zuyi Bao, Liwei Peng, Luo Si. (2019)  
**StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/d56c1fc337fb07ec004dc846f80582c327af717c)  
Influential Citation Count (21), SS-ID (d56c1fc337fb07ec004dc846f80582c327af717c)  

**ABSTRACT**  
Recently, the pre-trained language model, BERT (and its robustly optimized version RoBERTa), has attracted a lot of attention in natural language understanding (NLU), and achieved state-of-the-art accuracy in various NLU tasks, such as sentiment classification, natural language inference, semantic textual similarity and question answering. Inspired by the linearization exploration work of Elman [8], we extend BERT to a new model, StructBERT, by incorporating language structures into pre-training. Specifically, we pre-train StructBERT with two auxiliary tasks to make the most of the sequential order of words and sentences, which leverage language structures at the word and sentence levels, respectively. As a result, the new model is adapted to different levels of language understanding required by downstream tasks. The StructBERT with structural pre-training gives surprisingly good empirical results on a variety of downstream tasks, including pushing the state-of-the-art on the GLUE benchmark to 89.0 (outperforming all published models), the F1 score on SQuAD v1.1 question answering to 93.0, the accuracy on SNLI to 91.7.

{{< /ci-details >}}

{{< ci-details summary="Multilingual Alignment of Contextual Word Representations (Steven Cao et al., 2020)">}}

Steven Cao, Nikita Kitaev, D. Klein. (2020)  
**Multilingual Alignment of Contextual Word Representations**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/d592007d1c106fe1217604eb35664c7a5f07cb32)  
Influential Citation Count (17), SS-ID (d592007d1c106fe1217604eb35664c7a5f07cb32)  

**ABSTRACT**  
We propose procedures for evaluating and strengthening contextual embedding alignment and show that they are useful in analyzing and improving multilingual BERT. In particular, after our proposed alignment procedure, BERT exhibits significantly improved zero-shot performance on XNLI compared to the base model, remarkably matching pseudo-fully-supervised translate-train models for Bulgarian and Greek. Further, to measure the degree of alignment, we introduce a contextual version of word retrieval and show that it correlates well with downstream zero-shot transfer. Using this word retrieval task, we also analyze BERT and find that it exhibits systematic deficiencies, e.g. worse alignment for open-class parts-of-speech and word pairs written in different scripts, that are corrected by the alignment procedure. These results support contextual alignment as a useful concept for understanding large multilingual pre-trained models.

{{< /ci-details >}}

{{< ci-details summary="Energy and Policy Considerations for Deep Learning in NLP (Emma Strubell et al., 2019)">}}

Emma Strubell, Ananya Ganesh, A. McCallum. (2019)  
**Energy and Policy Considerations for Deep Learning in NLP**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/d6a083dad7114f3a39adc65c09bfbb6cf3fee9ea)  
Influential Citation Count (64), SS-ID (d6a083dad7114f3a39adc65c09bfbb6cf3fee9ea)  

**ABSTRACT**  
Recent progress in hardware and methodology for training neural networks has ushered in a new generation of large networks trained on abundant data. These models have obtained notable gains in accuracy across many NLP tasks. However, these accuracy improvements depend on the availability of exceptionally large computational resources that necessitate similarly substantial energy consumption. As a result these models are costly to train and develop, both financially, due to the cost of hardware and electricity or cloud compute time, and environmentally, due to the carbon footprint required to fuel modern tensor processing hardware. In this paper we bring this issue to the attention of NLP researchers by quantifying the approximate financial and environmental costs of training a variety of recently successful neural network models for NLP. Based on these findings, we propose actionable recommendations to reduce costs and improve equity in NLP research and practice.

{{< /ci-details >}}

{{< ci-details summary="Revealing the Dark Secrets of BERT (Olga Kovaleva et al., 2019)">}}

Olga Kovaleva, Alexey Romanov, Anna Rogers, Anna Rumshisky. (2019)  
**Revealing the Dark Secrets of BERT**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d78aed1dac6656affa4a04cbf225ced11a83d103)  
Influential Citation Count (34), SS-ID (d78aed1dac6656affa4a04cbf225ced11a83d103)  

**ABSTRACT**  
BERT-based architectures currently give state-of-the-art performance on many NLP tasks, but little is known about the exact mechanisms that contribute to its success. In the current work, we focus on the interpretation of self-attention, which is one of the fundamental underlying components of BERT. Using a subset of GLUE tasks and a set of handcrafted features-of-interest, we propose the methodology and carry out a qualitative and quantitative analysis of the information encoded by the individual BERT’s heads. Our findings suggest that there is a limited set of attention patterns that are repeated across different heads, indicating the overall model overparametrization. While different heads consistently use the same attention patterns, they have varying impact on performance across different tasks. We show that manually disabling attention in certain heads leads to a performance improvement over the regular fine-tuned BERT models.

{{< /ci-details >}}

{{< ci-details summary="Does BERT Solve Commonsense Task via Commonsense Knowledge? (Leyang Cui et al., 2020)">}}

Leyang Cui, Sijie Cheng, Yu Wu, Yue Zhang. (2020)  
**Does BERT Solve Commonsense Task via Commonsense Knowledge?**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/d8ea988072efb115ee8c85e159c1fa4a816360b5)  
Influential Citation Count (1), SS-ID (d8ea988072efb115ee8c85e159c1fa4a816360b5)  

**ABSTRACT**  
The success of pre-trained contextualized language models such as BERT motivates a line of work that investigates linguistic knowledge inside such models in order to explain the huge improvement in downstream tasks. While previous work shows syntactic, semantic and word sense knowledge in BERT, little work has been done on investigating how BERT solves CommonsenseQA tasks. In particular, it is an interesting research question whether BERT relies on shallow syntactic patterns or deeper commonsense knowledge for disambiguation. We propose two attention-based methods to analyze commonsense knowledge inside BERT, and the contribution of such knowledge for the model prediction. We find that attention heads successfully capture the structured commonsense knowledge encoded in ConceptNet, which helps BERT solve commonsense tasks directly. Fine-tuning further makes BERT learn to use the commonsense knowledge on higher layers.

{{< /ci-details >}}

{{< ci-details summary="Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning (Mitchell A. Gordon et al., 2020)">}}

Mitchell A. Gordon, Kevin Duh, Nicholas Andrews. (2020)  
**Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning**  
REPL4NLP  
[Paper Link](https://www.semanticscholar.org/paper/d9b824dbecbe3a1f0b1489f9e4521a532a63818d)  
Influential Citation Count (8), SS-ID (d9b824dbecbe3a1f0b1489f9e4521a532a63818d)  

**ABSTRACT**  
Pre-trained universal feature extractors, such as BERT for natural language processing and VGG for computer vision, have become effective methods for improving deep learning models without requiring more labeled data. While effective, feature extractors like BERT may be prohibitively large for some deployment scenarios. We explore weight pruning for BERT and ask: how does compression during pre-training affect transfer learning? We find that pruning affects transfer learning in three broad regimes. Low levels of pruning (30-40%) do not affect pre-training loss or transfer to downstream tasks at all. Medium levels of pruning increase the pre-training loss and prevent useful pre-training information from being transferred to downstream tasks. High levels of pruning additionally prevent models from fitting downstream datasets, leading to further degradation. Finally, we observe that fine-tuning BERT on a specific task does not improve its prunability. We conclude that BERT can be pruned once during pre-training rather than separately for each task without affecting performance.

{{< /ci-details >}}

{{< ci-details summary="Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation (Yonghui Wu et al., 2016)">}}

Yonghui Wu, M. Schuster, Z. Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, M. Krikun, Yuan Cao, Qin Gao, Klaus Macherey, J. Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser, Stephan Gouws, Y. Kato, Taku Kudo, H. Kazawa, K. Stevens, George Kurian, Nishant Patil, Wei Wang, C. Young, Jason R. Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, G. Corrado, Macduff Hughes, J. Dean. (2016)  
**Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/dbde7dfa6cae81df8ac19ef500c42db96c3d1edd)  
Influential Citation Count (345), SS-ID (dbde7dfa6cae81df8ac19ef500c42db96c3d1edd)  

**ABSTRACT**  
Neural Machine Translation (NMT) is an end-to-end learning approach for automated translation, with the potential to overcome many of the weaknesses of conventional phrase-based translation systems. Unfortunately, NMT systems are known to be computationally expensive both in training and in translation inference. Also, most NMT systems have difficulty with rare words. These issues have hindered NMT's use in practical deployments and services, where both accuracy and speed are essential. In this work, we present GNMT, Google's Neural Machine Translation system, which attempts to address many of these issues. Our model consists of a deep LSTM network with 8 encoder and 8 decoder layers using attention and residual connections. To improve parallelism and therefore decrease training time, our attention mechanism connects the bottom layer of the decoder to the top layer of the encoder. To accelerate the final translation speed, we employ low-precision arithmetic during inference computations. To improve handling of rare words, we divide words into a limited set of common sub-word units ("wordpieces") for both input and output. This method provides a good balance between the flexibility of "character"-delimited models and the efficiency of "word"-delimited models, naturally handles translation of rare words, and ultimately improves the overall accuracy of the system. Our beam search technique employs a length-normalization procedure and uses a coverage penalty, which encourages generation of an output sentence that is most likely to cover all the words in the source sentence. On the WMT'14 English-to-French and English-to-German benchmarks, GNMT achieves competitive results to state-of-the-art. Using a human side-by-side evaluation on a set of isolated simple sentences, it reduces translation errors by an average of 60% compared to Google's phrase-based production system.

{{< /ci-details >}}

{{< ci-details summary="BERT as a Teacher: Contextual Embeddings for Sequence-Level Reward (Florian Schmidt et al., 2020)">}}

Florian Schmidt, T. Hofmann. (2020)  
**BERT as a Teacher: Contextual Embeddings for Sequence-Level Reward**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/de9e7d6319b26c0d9f0da20c79403e9b9367fff4)  
Influential Citation Count (0), SS-ID (de9e7d6319b26c0d9f0da20c79403e9b9367fff4)  

**ABSTRACT**  
Measuring the quality of a generated sequence against a set of references is a central problem in many learning frameworks, be it to compute a score, to assign a reward, or to perform discrimination. Despite great advances in model architectures, metrics that scale independently of the number of references are still based on n-gram estimates. We show that the underlying operations, counting words and comparing counts, can be lifted to embedding words and comparing embeddings. An in-depth analysis of BERT embeddings shows empirically that contextual embeddings can be employed to capture the required dependencies while maintaining the necessary scalability through appropriate pruning and smoothing techniques. We cast unconditional generation as a reinforcement learning problem and show that our reward function indeed provides a more effective learning signal than n-gram reward in this challenging setting.

{{< /ci-details >}}

{{< ci-details summary="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Jacob Devlin et al., 2019)">}}

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019)  
**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/df2b0e26d0599ce3e70df8a9da02e51594e0e992)  
Influential Citation Count (9863), SS-ID (df2b0e26d0599ce3e70df8a9da02e51594e0e992)  

**ABSTRACT**  
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5 (7.7 point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

{{< /ci-details >}}

{{< ci-details summary="Constructions at Work: The Nature of Generalization in Language (A. Goldberg, 2006)">}}

A. Goldberg. (2006)  
**Constructions at Work: The Nature of Generalization in Language**  
  
[Paper Link](https://www.semanticscholar.org/paper/dfc79017e52efb270155ce8b93337467804cb697)  
Influential Citation Count (229), SS-ID (dfc79017e52efb270155ce8b93337467804cb697)  

**ABSTRACT**  
Part One: Constructions 1. Overview 2. Surface Generalizations 3. Item Specific Knowledge and Generalizations Part Two: Learning Generalizations 4. How Generalizations are Learned 5. How Generalizations are Constrained 6. Why Generalizations are Learned Part Three: Explaining Generalizations 7. Island Constraints and Scope 8. Grammatical Categorization: Subject Auxiliary Inversion 9. Cross-linguistic Generalizations in Argument Realization 10. Variations on a Constructionist Theme 11. Conclusion References Index

{{< /ci-details >}}

{{< ci-details summary="XLNet: Generalized Autoregressive Pretraining for Language Understanding (Zhilin Yang et al., 2019)">}}

Zhilin Yang, Zihang Dai, Yiming Yang, J. Carbonell, R. Salakhutdinov, Quoc V. Le. (2019)  
**XLNet: Generalized Autoregressive Pretraining for Language Understanding**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/e0c6abdbdecf04ffac65c440da77fb9d66bb474c)  
Influential Citation Count (631), SS-ID (e0c6abdbdecf04ffac65c440da77fb9d66bb474c)  

**ABSTRACT**  
With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling. However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.

{{< /ci-details >}}

{{< ci-details summary="What do you learn from context? Probing for sentence structure in contextualized word representations (Ian Tenney et al., 2019)">}}

Ian Tenney, Patrick Xia, Berlin Chen, Alex Wang, Adam Poliak, R. Thomas McCoy, Najoung Kim, Benjamin Van Durme, Samuel R. Bowman, Dipanjan Das, Ellie Pavlick. (2019)  
**What do you learn from context? Probing for sentence structure in contextualized word representations**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/e2587eddd57bc4ba286d91b27c185083f16f40ee)  
Influential Citation Count (45), SS-ID (e2587eddd57bc4ba286d91b27c185083f16f40ee)  

**ABSTRACT**  
Contextualized representation models such as ELMo (Peters et al., 2018a) and BERT (Devlin et al., 2018) have recently achieved state-of-the-art results on a diverse array of downstream NLP tasks. Building on recent token-level probing work, we introduce a novel edge probing task design and construct a broad suite of sub-sentence tasks derived from the traditional structured NLP pipeline. We probe word-level contextual representations from four recent models and investigate how they encode sentence structure across a range of syntactic, semantic, local, and long-range phenomena. We find that existing models trained on language modeling and translation produce strong representations for syntactic phenomena, but only offer comparably small improvements on semantic tasks over a non-contextual baseline.

{{< /ci-details >}}

{{< ci-details summary="Cross-lingual Language Model Pretraining (Guillaume Lample et al., 2019)">}}

Guillaume Lample, A. Conneau. (2019)  
**Cross-lingual Language Model Pretraining**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/ec4eba83f6b3266d9ae7cabb2b2cb1518f727edc)  
Influential Citation Count (345), SS-ID (ec4eba83f6b3266d9ae7cabb2b2cb1518f727edc)  

**ABSTRACT**  
Recent studies have demonstrated the efficiency of generative pretraining for English natural language understanding. In this work, we extend this approach to multiple languages and show the effectiveness of cross-lingual pretraining. We propose two methods to learn cross-lingual language models (XLMs): one unsupervised that only relies on monolingual data, and one supervised that leverages parallel data with a new cross-lingual language model objective. We obtain state-of-the-art results on cross-lingual classification, unsupervised and supervised machine translation. On XNLI, our approach pushes the state of the art by an absolute gain of 4.9% accuracy. On unsupervised machine translation, we obtain 34.3 BLEU on WMT’16 German-English, improving the previous state of the art by more than 9 BLEU. On supervised machine translation, we obtain a new state of the art of 38.5 BLEU on WMT’16 Romanian-English, outperforming the previous best approach by more than 4 BLEU. Our code and pretrained models will be made publicly available.

{{< /ci-details >}}

{{< ci-details summary="Pooled Contextualized Embeddings for Named Entity Recognition (A. Akbik et al., 2019)">}}

A. Akbik, Tanja Bergmann, Roland Vollgraf. (2019)  
**Pooled Contextualized Embeddings for Named Entity Recognition**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/edfe9dd16316618e694cd087d0d418dac91eb48c)  
Influential Citation Count (40), SS-ID (edfe9dd16316618e694cd087d0d418dac91eb48c)  

**ABSTRACT**  
Contextual string embeddings are a recent type of contextualized word embedding that were shown to yield state-of-the-art results when utilized in a range of sequence labeling tasks. They are based on character-level language models which treat text as distributions over characters and are capable of generating embeddings for any string of characters within any textual context. However, such purely character-based approaches struggle to produce meaningful embeddings if a rare string is used in a underspecified context. To address this drawback, we propose a method in which we dynamically aggregate contextualized embeddings of each unique string that we encounter. We then use a pooling operation to distill a ”global” word representation from all contextualized instances. We evaluate these ”pooled contextualized embeddings” on common named entity recognition (NER) tasks such as CoNLL-03 and WNUT and show that our approach significantly improves the state-of-the-art for NER. We make all code and pre-trained models available to the research community for use and reproduction.

{{< /ci-details >}}

{{< ci-details summary="Document Classification by Word Embeddings of BERT (Hirotaka Tanaka et al., 2019)">}}

Hirotaka Tanaka, Hiroyuki Shinnou, Rui Cao, Jing Bai, Wen Ma. (2019)  
**Document Classification by Word Embeddings of BERT**  
PACLING  
[Paper Link](https://www.semanticscholar.org/paper/ef1041ff14c02dc9e35317916561b904d7ef8433)  
Influential Citation Count (0), SS-ID (ef1041ff14c02dc9e35317916561b904d7ef8433)  

**ABSTRACT**  
Bidirectional Encoder Representations from Transformers (BERT) is a pre-training model that uses the encoder component of a bidirectional transformer and converts an input sentence or input sentence pair into word enbeddings. The performance of various natural language processing systems has been greatly improved by BERT. However, for a real task, it is necessary to consider how BERT is used based on the type of task. The standerd method for document classification by BERT is to treat the word embedding of special token [CLS] as a feature vector of the document, and to fine-tune the entire model of the classifier, including a pre-training model. However, after normalizing each the feature vector consisting of the mean vector of word embeddings outputted by BERT for the document, and the feature vectors according to the bag-of-words model, we create a vector concatenating them. Our proposed method involves using the concatenated vector as the feature vector of the document.

{{< /ci-details >}}

{{< ci-details summary="Pay Less Attention with Lightweight and Dynamic Convolutions (Felix Wu et al., 2019)">}}

Felix Wu, Angela Fan, Alexei Baevski, Yann Dauphin, Michael Auli. (2019)  
**Pay Less Attention with Lightweight and Dynamic Convolutions**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/ef523bb9437178c50d1b1e3e3ca5fb230ab37e3f)  
Influential Citation Count (65), SS-ID (ef523bb9437178c50d1b1e3e3ca5fb230ab37e3f)  

**ABSTRACT**  
Self-attention is a useful mechanism to build generative models for language and images. It determines the importance of context elements by comparing each element to the current time step. In this paper, we show that a very lightweight convolution can perform competitively to the best reported self-attention results. Next, we introduce dynamic convolutions which are simpler and more efficient than self-attention. We predict separate convolution kernels based solely on the current time-step in order to determine the importance of context elements. The number of operations required by this approach scales linearly in the input length, whereas self-attention is quadratic. Experiments on large-scale machine translation, language modeling and abstractive summarization show that dynamic convolutions improve over strong self-attention models. On the WMT'14 English-German test set dynamic convolutions achieve a new state of the art of 29.7 BLEU.

{{< /ci-details >}}

{{< ci-details summary="Assessing BERT's Syntactic Abilities (Yoav Goldberg, 2019)">}}

Yoav Goldberg. (2019)  
**Assessing BERT's Syntactic Abilities**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/efeab0dcdb4c1cce5e537e57745d84774be99b9a)  
Influential Citation Count (17), SS-ID (efeab0dcdb4c1cce5e537e57745d84774be99b9a)  

**ABSTRACT**  
I assess the extent to which the recently introduced BERT model captures English syntactic phenomena, using (1) naturally-occurring subject-verb agreement stimuli; (2) "coloreless green ideas" subject-verb agreement stimuli, in which content words in natural sentences are randomly replaced with words sharing the same part-of-speech and inflection; and (3) manually crafted stimuli for subject-verb agreement and reflexive anaphora phenomena. The BERT model performs remarkably well on all cases.

{{< /ci-details >}}

{{< ci-details summary="Further Boosting BERT-based Models by Duplicating Existing Layers: Some Intriguing Phenomena inside BERT (Wei-Tsung Kao et al., 2020)">}}

Wei-Tsung Kao, Tsung-Han Wu, Po-Han Chi, Chun-Cheng Hsieh, Hung-yi Lee. (2020)  
**Further Boosting BERT-based Models by Duplicating Existing Layers: Some Intriguing Phenomena inside BERT**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/f18fa3728868af6c44bb1dc3e913925abc37b5c1)  
Influential Citation Count (1), SS-ID (f18fa3728868af6c44bb1dc3e913925abc37b5c1)  

**ABSTRACT**  
Although Bidirectional Encoder Representations from Transformers (BERT) have achieved tremendous success in many natural language processing (NLP) tasks, it remains a black box, so much previous work has tried to lift the veil of BERT and understand the functionality of each layer. In this paper, we found that removing or duplicating most layers in BERT would not change their outputs. This fact remains true across a wide variety of BERT-based models. Based on this observation, we propose a quite simple method to boost the performance of BERT. By duplicating some layers in the BERT-based models to make it deeper (no extra training required in this step), they obtain better performance in the down-stream tasks after fine-tuning.

{{< /ci-details >}}

{{< ci-details summary="GloVe: Global Vectors for Word Representation (Jeffrey Pennington et al., 2014)">}}

Jeffrey Pennington, R. Socher, Christopher D. Manning. (2014)  
**GloVe: Global Vectors for Word Representation**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/f37e1b62a767a307c046404ca96bc140b3e68cb5)  
Influential Citation Count (3451), SS-ID (f37e1b62a767a307c046404ca96bc140b3e68cb5)  

**ABSTRACT**  
Recent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using vector arithmetic, but the origin of these regularities has remained opaque. We analyze and make explicit the model properties needed for such regularities to emerge in word vectors. The result is a new global logbilinear regression model that combines the advantages of the two major model families in the literature: global matrix factorization and local context window methods. Our model efficiently leverages statistical information by training only on the nonzero elements in a word-word cooccurrence matrix, rather than on the entire sparse matrix or on individual context windows in a large corpus. The model produces a vector space with meaningful substructure, as evidenced by its performance of 75% on a recent word analogy task. It also outperforms related models on similarity tasks and named entity recognition.

{{< /ci-details >}}

{{< ci-details summary="Probing Neural Network Comprehension of Natural Language Arguments (Timothy Niven et al., 2019)">}}

Timothy Niven, Hung-Yu Kao. (2019)  
**Probing Neural Network Comprehension of Natural Language Arguments**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/f3b89e9a2b8ce1b6058e6984c3556bc2dded0938)  
Influential Citation Count (12), SS-ID (f3b89e9a2b8ce1b6058e6984c3556bc2dded0938)  

**ABSTRACT**  
We are surprised to find that BERT’s peak performance of 77% on the Argument Reasoning Comprehension Task reaches just three points below the average untrained human baseline. However, we show that this result is entirely accounted for by exploitation of spurious statistical cues in the dataset. We analyze the nature of these cues and demonstrate that a range of models all exploit them. This analysis informs the construction of an adversarial dataset on which all models achieve random accuracy. Our adversarial dataset provides a more robust assessment of argument comprehension and should be adopted as the standard in future work.

{{< /ci-details >}}

{{< ci-details summary="Reducing Transformer Depth on Demand with Structured Dropout (Angela Fan et al., 2019)">}}

Angela Fan, Edouard Grave, Armand Joulin. (2019)  
**Reducing Transformer Depth on Demand with Structured Dropout**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/f4a8480cffa491020bdbb8c4c4e7a7e923b1c2c1)  
Influential Citation Count (47), SS-ID (f4a8480cffa491020bdbb8c4c4e7a7e923b1c2c1)  

**ABSTRACT**  
Overparameterized transformer networks have obtained state of the art results in various natural language processing tasks, such as machine translation, language modeling, and question answering. These models contain hundreds of millions of parameters, necessitating a large amount of computation and making them prone to overfitting. In this work, we explore LayerDrop, a form of structured dropout, which has a regularization effect during training and allows for efficient pruning at inference time. In particular, we show that it is possible to select sub-networks of any depth from one large network without having to finetune them and with limited impact on performance. We demonstrate the effectiveness of our approach by improving the state of the art on machine translation, language modeling, summarization, question answering, and language understanding benchmarks. Moreover, we show that our approach leads to small BERT-like models of higher quality compared to training from scratch or using distillation.

{{< /ci-details >}}

{{< ci-details summary="Information-Theoretic Probing with Minimum Description Length (Elena Voita et al., 2020)">}}

Elena Voita, Ivan Titov. (2020)  
**Information-Theoretic Probing with Minimum Description Length**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/f4b585c9a79dfce0807b445a09036ea0f9cbcdce)  
Influential Citation Count (13), SS-ID (f4b585c9a79dfce0807b445a09036ea0f9cbcdce)  

**ABSTRACT**  
To measure how well pretrained representations encode some linguistic property, it is common to use accuracy of a probe, i.e. a classifier trained to predict the property from the representations. Despite widespread adoption of probes, differences in their accuracy fail to adequately reflect differences in representations. For example, they do not substantially favour pretrained representations over randomly initialized ones. Analogously, their accuracy can be similar when probing for genuine linguistic labels and probing for random synthetic tasks. To see reasonable differences in accuracy with respect to these random baselines, previous work had to constrain either the amount of probe training data or its model size. Instead, we propose an alternative to the standard probes, information-theoretic probing with minimum description length (MDL). With MDL probing, training a probe to predict labels is recast as teaching it to effectively transmit the data. Therefore, the measure of interest changes from probe accuracy to the description length of labels given representations. In addition to probe quality, the description length evaluates "the amount of effort" needed to achieve the quality. This amount of effort characterizes either (i) size of a probing model, or (ii) the amount of data needed to achieve the high quality. We consider two methods for estimating MDL which can be easily implemented on top of the standard probing pipelines: variational coding and online coding. We show that these methods agree in results and are more informative and stable than the standard probes.

{{< /ci-details >}}

{{< ci-details summary="Adaptively Sparse Transformers (Gonçalo M. Correia et al., 2019)">}}

Gonçalo M. Correia, Vlad Niculae, André F. T. Martins. (2019)  
**Adaptively Sparse Transformers**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/f6390beca54411b06f3bde424fb983a451789733)  
Influential Citation Count (18), SS-ID (f6390beca54411b06f3bde424fb983a451789733)  

**ABSTRACT**  
Attention mechanisms have become ubiquitous in NLP. Recent architectures, notably the Transformer, learn powerful context-aware word representations through layered, multi-headed attention. The multiple heads learn diverse types of word relationships. However, with standard softmax attention, all attention heads are dense, assigning a non-zero weight to all context words. In this work, we introduce the adaptively sparse Transformer, wherein attention heads have flexible, context-dependent sparsity patterns. This sparsity is accomplished by replacing softmax with alpha-entmax: a differentiable generalization of softmax that allows low-scoring words to receive precisely zero weight. Moreover, we derive a method to automatically learn the alpha parameter – which controls the shape and sparsity of alpha-entmax – allowing attention heads to choose between focused or spread-out behavior. Our adaptively sparse Transformer improves interpretability and head diversity when compared to softmax Transformers on machine translation datasets. Findings of the quantitative and qualitative analysis of our approach include that heads in different layers learn different sparsity preferences and tend to be more diverse in their attention distributions than softmax Transformers. Furthermore, at no cost in accuracy, sparsity in attention heads helps to uncover different head specializations.

{{< /ci-details >}}

{{< ci-details summary="UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training (Hangbo Bao et al., 2020)">}}

Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu Wang, Songhao Piao, Jianfeng Gao, Ming Zhou, H. Hon. (2020)  
**UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/f64e1d6bc13aae99aab5449fc9ae742a9ba7761e)  
Influential Citation Count (23), SS-ID (f64e1d6bc13aae99aab5449fc9ae742a9ba7761e)  

**ABSTRACT**  
We propose to pre-train a unified language model for both autoencoding and partially autoregressive language modeling tasks using a novel training procedure, referred to as a pseudo-masked language model (PMLM). Given an input text with masked tokens, we rely on conventional masks to learn inter-relations between corrupted tokens and context via autoencoding, and pseudo masks to learn intra-relations between masked spans via partially autoregressive modeling. With well-designed position embeddings and self-attention masks, the context encodings are reused to avoid redundant computation. Moreover, conventional masks used for autoencoding provide global masking information, so that all the position embeddings are accessible in partially autoregressive language modeling. In addition, the two tasks pre-train a unified language model as a bidirectional encoder and a sequence-to-sequence decoder, respectively. Our experiments show that the unified language models pre-trained using PMLM achieve new state-of-the-art results on a wide range of natural language understanding and generation tasks across several widely used benchmarks.

{{< /ci-details >}}

{{< ci-details summary="Inducing Relational Knowledge from BERT (Zied Bouraoui et al., 2019)">}}

Zied Bouraoui, José Camacho-Collados, S. Schockaert. (2019)  
**Inducing Relational Knowledge from BERT**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/f67fcbb1aec92ae293998ddfd904f61a31bef334)  
Influential Citation Count (10), SS-ID (f67fcbb1aec92ae293998ddfd904f61a31bef334)  

**ABSTRACT**  
One of the most remarkable properties of word embeddings is the fact that they capture certain types of semantic and syntactic relationships. Recently, pre-trained language models such as BERT have achieved groundbreaking results across a wide range of Natural Language Processing tasks. However, it is unclear to what extent such models capture relational knowledge beyond what is already captured by standard word embeddings. To explore this question, we propose a methodology for distilling relational knowledge from a pre-trained language model. Starting from a few seed instances of a given relation, we first use a large text corpus to find sentences that are likely to express this relation. We then use a subset of these extracted sentences as templates. Finally, we fine-tune a language model to predict whether a given word pair is likely to be an instance of some relation, when given an instantiated template for that relation as input.

{{< /ci-details >}}

{{< ci-details summary="Linguistic Knowledge and Transferability of Contextual Representations (Nelson F. Liu et al., 2019)">}}

Nelson F. Liu, Matt Gardner, Y. Belinkov, Matthew E. Peters, Noah A. Smith. (2019)  
**Linguistic Knowledge and Transferability of Contextual Representations**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  
Influential Citation Count (108), SS-ID (f6fbb6809374ca57205bd2cf1421d4f4fa04f975)  

**ABSTRACT**  
Contextual word representations derived from large-scale neural language models are successful across a diverse set of NLP tasks, suggesting that they encode useful and transferable features of language. To shed light on the linguistic knowledge they capture, we study the representations produced by several recent pretrained contextualizers (variants of ELMo, the OpenAI transformer language model, and BERT) with a suite of sixteen diverse probing tasks. We find that linear models trained on top of frozen contextual representations are competitive with state-of-the-art task-specific models in many cases, but fail on tasks requiring fine-grained linguistic knowledge (e.g., conjunct identification). To investigate the transferability of contextual word representations, we quantify differences in the transferability of individual layers within contextualizers, especially between recurrent neural networks (RNNs) and transformers. For instance, higher layers of RNNs are more task-specific, while transformer layers do not exhibit the same monotonic trend. In addition, to better understand what makes contextual word representations transferable, we compare language model pretraining with eleven supervised pretraining tasks. For any given task, pretraining on a closely related task yields better performance than language model pretraining (which is better on average) when the pretraining dataset is fixed. However, language model pretraining on more data gives the best results.

{{< /ci-details >}}

{{< ci-details summary="Commonsense Knowledge Mining from Pretrained Models (Joshua Feldman et al., 2019)">}}

Joshua Feldman, Joe Davison, Alexander M. Rush. (2019)  
**Commonsense Knowledge Mining from Pretrained Models**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/f98e135986414cccf29aec593d547c0656e4d82c)  
Influential Citation Count (17), SS-ID (f98e135986414cccf29aec593d547c0656e4d82c)  

**ABSTRACT**  
Inferring commonsense knowledge is a key challenge in machine learning. Due to the sparsity of training data, previous work has shown that supervised methods for commonsense knowledge mining underperform when evaluated on novel data. In this work, we develop a method for generating commonsense knowledge using a large, pre-trained bidirectional language model. By transforming relational triples into masked sentences, we can use this model to rank a triple’s validity by the estimated pointwise mutual information between the two entities. Since we do not update the weights of the bidirectional model, our approach is not biased by the coverage of any one commonsense knowledge base. Though we do worse on a held-out test set than models explicitly trained on a corresponding training set, our approach outperforms these methods when mining commonsense knowledge from new sources, suggesting that our unsupervised technique generalizes better than current supervised approaches.

{{< /ci-details >}}

{{< ci-details summary="Green AI (Roy Schwartz et al., 2019)">}}

Roy Schwartz, Jesse Dodge, Noah Smith, Oren Etzioni. (2019)  
**Green AI**  
Commun. ACM  
[Paper Link](https://www.semanticscholar.org/paper/fb73b93de3734a996829caf31e4310e0054e9c6b)  
Influential Citation Count (17), SS-ID (fb73b93de3734a996829caf31e4310e0054e9c6b)  

**ABSTRACT**  
Creating efficiency in AI research will decrease its carbon footprint and increase its inclusivity as deep learning study should not require the deepest pockets.

{{< /ci-details >}}

