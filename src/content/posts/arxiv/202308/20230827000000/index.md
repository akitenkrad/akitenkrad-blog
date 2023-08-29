---
draft: false
title: "arXiv @ 2023.08.27"
date: 2023-08-27
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.27"
    identifier: arxiv_20230827
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (24)](#cscl-24)
- [cs.SE (7)](#csse-7)
- [cs.LG (20)](#cslg-20)
- [cs.SI (3)](#cssi-3)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [cs.CV (29)](#cscv-29)
- [eess.IV (3)](#eessiv-3)
- [cs.HC (3)](#cshc-3)
- [cs.IR (2)](#csir-2)
- [cs.RO (2)](#csro-2)
- [cs.CR (3)](#cscr-3)
- [cs.AI (2)](#csai-2)
- [quant-ph (1)](#quant-ph-1)
- [cs.NE (1)](#csne-1)
- [eess.SY (1)](#eesssy-1)
- [eess.SP (3)](#eesssp-3)
- [cs.GT (1)](#csgt-1)
- [q-fin.TR (1)](#q-fintr-1)
- [physics.comp-ph (1)](#physicscomp-ph-1)
- [cs.SD (1)](#cssd-1)
- [cs.IT (1)](#csit-1)
- [stat.ML (1)](#statml-1)

## cs.CL (24)



### (1/111) WellXplain: Wellness Concept Extraction and Classification in Reddit Posts for Mental Health Analysis (Muskan Garg, 2023)

{{<citation>}}

Muskan Garg. (2023)  
**WellXplain: Wellness Concept Extraction and Classification in Reddit Posts for Mental Health Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13710v1)  

---


**ABSTRACT**  
During the current mental health crisis, the importance of identifying potential indicators of mental issues from social media content has surged. Overlooking the multifaceted nature of mental and social well-being can have detrimental effects on one's mental state. In traditional therapy sessions, professionals manually pinpoint the origins and outcomes of underlying mental challenges, a process both detailed and time-intensive. We introduce an approach to this intricate mental health analysis by framing the identification of wellness dimensions in Reddit content as a wellness concept extraction and categorization challenge. We've curated a unique dataset named WELLXPLAIN, comprising 3,092 entries and totaling 72,813 words. Drawing from Halbert L. Dunn's well-regarded wellness theory, our team formulated an annotation framework along with guidelines. This dataset also includes human-marked textual segments, offering clear reasoning for decisions made in the wellness concept categorization process. Our aim in publishing this dataset and analyzing initial benchmarks is to spearhead the creation of advanced language models tailored for healthcare-focused concept extraction and categorization.

{{</citation>}}


### (2/111) On the Depth between Beam Search and Exhaustive Search for Text Generation (Yuu Jinnai et al., 2023)

{{<citation>}}

Yuu Jinnai, Tetsuro Morimura, Ukyo Honda. (2023)  
**On the Depth between Beam Search and Exhaustive Search for Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2308.13696v1)  

---


**ABSTRACT**  
Beam search and exhaustive search are two extreme ends of text decoding algorithms with respect to the search depth. Beam search is limited in both search width and depth, whereas exhaustive search is a global search that has no such limitations. Surprisingly, beam search is not only computationally cheaper but also performs better than exhaustive search despite its higher search error. Plenty of research has investigated a range of beam widths, from small to large, and reported that a beam width that is neither too large nor too small is desirable. However, in terms of search depth, only the two extreme ends, beam search and exhaustive search are studied intensively. In this paper, we examine a range of search depths between the two extremes to discover the desirable search depth. To this end, we introduce Lookahead Beam Search (LBS), a multi-step lookahead search that optimizes the objective considering a fixed number of future steps. Beam search and exhaustive search are special cases of LBS where the lookahead depth is set to $0$ and $\infty$, respectively. We empirically evaluate the performance of LBS and find that it outperforms beam search overall on machine translation tasks. The result suggests there is room for improvement in beam search by searching deeper. Inspired by the analysis, we propose Lookbehind Heuristic Beam Search, a computationally feasible search algorithm that heuristically simulates LBS with 1-step lookahead. The empirical results show that the proposed method outperforms vanilla beam search on machine translation and text summarization tasks.

{{</citation>}}


### (3/111) Rethinking Language Models as Symbolic Knowledge Graphs (Vishwas Mruthyunjaya et al., 2023)

{{<citation>}}

Vishwas Mruthyunjaya, Pouya Pezeshkpour, Estevam Hruschka, Nikita Bhutani. (2023)  
**Rethinking Language Models as Symbolic Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, GPT, GPT-4, Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13676v1)  

---


**ABSTRACT**  
Symbolic knowledge graphs (KGs) play a pivotal role in knowledge-centric applications such as search, question answering and recommendation. As contemporary language models (LMs) trained on extensive textual data have gained prominence, researchers have extensively explored whether the parametric knowledge within these models can match up to that present in knowledge graphs. Various methodologies have indicated that enhancing the size of the model or the volume of training data enhances its capacity to retrieve symbolic knowledge, often with minimal or no human supervision. Despite these advancements, there is a void in comprehensively evaluating whether LMs can encompass the intricate topological and semantic attributes of KGs, attributes crucial for reasoning processes. In this work, we provide an exhaustive evaluation of language models of varying sizes and capabilities. We construct nine qualitative benchmarks that encompass a spectrum of attributes including symmetry, asymmetry, hierarchy, bidirectionality, compositionality, paths, entity-centricity, bias and ambiguity. Additionally, we propose novel evaluation metrics tailored for each of these attributes. Our extensive evaluation of various LMs shows that while these models exhibit considerable potential in recalling factual information, their ability to capture intricate topological and semantic traits of KGs remains significantly constrained. We note that our proposed evaluation metrics are more reliable in evaluating these abilities than the existing metrics. Lastly, some of our benchmarks challenge the common notion that larger LMs (e.g., GPT-4) universally outshine their smaller counterparts (e.g., BERT).

{{</citation>}}


### (4/111) ChatGPT as Data Augmentation for Compositional Generalization: A Case Study in Open Intent Detection (Yihao Fang et al., 2023)

{{<citation>}}

Yihao Fang, Xianzhi Li, Stephen W. Thomas, Xiaodan Zhu. (2023)  
**ChatGPT as Data Augmentation for Compositional Generalization: A Case Study in Open Intent Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation, ChatGPT, GPT, Intent Detection  
[Paper Link](http://arxiv.org/abs/2308.13517v1)  

---


**ABSTRACT**  
Open intent detection, a crucial aspect of natural language understanding, involves the identification of previously unseen intents in user-generated text. Despite the progress made in this field, challenges persist in handling new combinations of language components, which is essential for compositional generalization. In this paper, we present a case study exploring the use of ChatGPT as a data augmentation technique to enhance compositional generalization in open intent detection tasks. We begin by discussing the limitations of existing benchmarks in evaluating this problem, highlighting the need for constructing datasets for addressing compositional generalization in open intent detection tasks. By incorporating synthetic data generated by ChatGPT into the training process, we demonstrate that our approach can effectively improve model performance. Rigorous evaluation of multiple benchmarks reveals that our method outperforms existing techniques and significantly enhances open intent detection capabilities. Our findings underscore the potential of large language models like ChatGPT for data augmentation in natural language understanding tasks.

{{</citation>}}


### (5/111) Training and Meta-Evaluating Machine Translation Evaluation Metrics at the Paragraph Level (Daniel Deutsch et al., 2023)

{{<citation>}}

Daniel Deutsch, Juraj Juraska, Mara Finkelstein, Markus Freitag. (2023)  
**Training and Meta-Evaluating Machine Translation Evaluation Metrics at the Paragraph Level**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2308.13506v2)  

---


**ABSTRACT**  
As research on machine translation moves to translating text beyond the sentence level, it remains unclear how effective automatic evaluation metrics are at scoring longer translations. In this work, we first propose a method for creating paragraph-level data for training and meta-evaluating metrics from existing sentence-level data. Then, we use these new datasets to benchmark existing sentence-level metrics as well as train learned metrics at the paragraph level. Interestingly, our experimental results demonstrate that using sentence-level metrics to score entire paragraphs is equally as effective as using a metric designed to work at the paragraph level. We speculate this result can be attributed to properties of the task of reference-based evaluation as well as limitations of our datasets with respect to capturing all types of phenomena that occur in paragraph-level translations.

{{</citation>}}


### (6/111) Ngambay-French Neural Machine Translation (sba-Fr) (Sakayo Toadoum Sari et al., 2023)

{{<citation>}}

Sakayo Toadoum Sari, Angela Fan, Lema Logamou Seknewna. (2023)  
**Ngambay-French Neural Machine Translation (sba-Fr)**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Machine Translation, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.13497v1)  

---


**ABSTRACT**  
In Africa, and the world at large, there is an increasing focus on developing Neural Machine Translation (NMT) systems to overcome language barriers. NMT for Low-resource language is particularly compelling as it involves learning with limited labelled data. However, obtaining a well-aligned parallel corpus for low-resource languages can be challenging. The disparity between the technological advancement of a few global languages and the lack of research on NMT for local languages in Chad is striking. End-to-end NMT trials on low-resource Chad languages have not been attempted. Additionally, there is a dearth of online and well-structured data gathering for research in Natural Language Processing, unlike some African languages. However, a guided approach for data gathering can produce bitext data for many Chadian language translation pairs with well-known languages that have ample data. In this project, we created the first sba-Fr Dataset, which is a corpus of Ngambay-to-French translations, and fine-tuned three pre-trained models using this dataset. Our experiments show that the M2M100 model outperforms other models with high BLEU scores on both original and original+synthetic data. The publicly available bitext dataset can be used for research purposes.

{{</citation>}}


### (7/111) Prompting a Large Language Model to Generate Diverse Motivational Messages: A Comparison with Human-Written Messages (Samuel Rhys Cox et al., 2023)

{{<citation>}}

Samuel Rhys Cox, Ashraf Abdul, Wei Tsang Ooi. (2023)  
**Prompting a Large Language Model to Generate Diverse Motivational Messages: A Comparison with Human-Written Messages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13479v1)  

---


**ABSTRACT**  
Large language models (LLMs) are increasingly capable and prevalent, and can be used to produce creative content. The quality of content is influenced by the prompt used, with more specific prompts that incorporate examples generally producing better results. On from this, it could be seen that using instructions written for crowdsourcing tasks (that are specific and include examples to guide workers) could prove effective LLM prompts. To explore this, we used a previous crowdsourcing pipeline that gave examples to people to help them generate a collectively diverse corpus of motivational messages. We then used this same pipeline to generate messages using GPT-4, and compared the collective diversity of messages from: (1) crowd-writers, (2) GPT-4 using the pipeline, and (3 & 4) two baseline GPT-4 prompts. We found that the LLM prompts using the crowdsourcing pipeline caused GPT-4 to produce more diverse messages than the two baseline prompts. We also discuss implications from messages generated by both human writers and LLMs.

{{</citation>}}


### (8/111) Leveraging Knowledge and Reinforcement Learning for Enhanced Reliability of Language Models (Nancy Tyagi et al., 2023)

{{<citation>}}

Nancy Tyagi, Surjodeep Sarkar, Manas Gaur. (2023)  
**Leveraging Knowledge and Reinforcement Learning for Enhanced Reliability of Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: BERT, GLUE, Language Model, NLP, Natural Language Processing, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.13467v1)  

---


**ABSTRACT**  
The Natural Language Processing(NLP) community has been using crowd sourcing techniques to create benchmark datasets such as General Language Understanding and Evaluation(GLUE) for training modern Language Models such as BERT. GLUE tasks measure the reliability scores using inter annotator metrics i.e. Cohens Kappa. However, the reliability aspect of LMs has often been overlooked. To counter this problem, we explore a knowledge-guided LM ensembling approach that leverages reinforcement learning to integrate knowledge from ConceptNet and Wikipedia as knowledge graph embeddings. This approach mimics human annotators resorting to external knowledge to compensate for information deficits in the datasets. Across nine GLUE datasets, our research shows that ensembling strengthens reliability and accuracy scores, outperforming state of the art.

{{</citation>}}


### (9/111) ARTIST: ARTificial Intelligence for Simplified Text (Lorenzo Corti et al., 2023)

{{<citation>}}

Lorenzo Corti, Jie Yang. (2023)  
**ARTIST: ARTificial Intelligence for Simplified Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Generative AI, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.13458v1)  

---


**ABSTRACT**  
Complex text is a major barrier for many citizens when accessing public information and knowledge. While often done manually, Text Simplification is a key Natural Language Processing task that aims for reducing the linguistic complexity of a text while preserving the original meaning. Recent advances in Generative Artificial Intelligence (AI) have enabled automatic text simplification both on the lexical and syntactical levels. However, as applications often focus on English, little is understood about the effectiveness of Generative AI techniques on low-resource languages such as Dutch. For this reason, we carry out empirical studies to understand the benefits and limitations of applying generative technologies for text simplification and provide the following outcomes: 1) the design and implementation for a configurable text simplification pipeline that orchestrates state-of-the-art generative text simplification models, domain and reader adaptation, and visualisation modules; 2) insights and lessons learned, showing the strengths of automatic text simplification while exposing the challenges in handling cultural and commonsense knowledge. These outcomes represent a first step in the exploration of Dutch text simplification and shed light on future endeavours both for research and practice.

{{</citation>}}


### (10/111) The Poison of Alignment (Aibek Bekbayev et al., 2023)

{{<citation>}}

Aibek Bekbayev, Sungbae Chun, Yerzat Dulat, James Yamazaki. (2023)  
**The Poison of Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.13449v1)  

---


**ABSTRACT**  
From the perspective of content safety issues, alignment has shown to limit large language models' (LLMs) harmful content generation. This intentional method of reinforcing models to not respond to certain user inputs seem to be present in many modern open-source instruction tuning datasets such as OpenAssistant or Guanaco. We introduce a novel insight to an instruction-tuned model's performance affected by the presence of alignment in supervised fine-tuning dataset. To be specific, we noticed that alignment acts as if it is poisoning the instruction dataset. Experimentally, we demonstrate that aligned answers significantly worsen the performance of the resulting fine-tuned model's on various reasoning benchmarks such as Big Bench (BBH), Massive Multitask Language Understanding (MMLU), Human Eval, and Discrete Reasoning Over Paragraphs (DROP), performing worse than the counterpart tuned without alignment by 4-33%.

{{</citation>}}


### (11/111) EntropyRank: Unsupervised Keyphrase Extraction via Side-Information Optimization for Language Model-based Text Compression (Alexander Tsvetkov. Alon Kipnis, 2023)

{{<citation>}}

Alexander Tsvetkov. Alon Kipnis. (2023)  
**EntropyRank: Unsupervised Keyphrase Extraction via Side-Information Optimization for Language Model-based Text Compression**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IT, cs-LG, cs.CL, math-IT  
Keywords: Keyphrase Extraction, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13399v1)  

---


**ABSTRACT**  
We propose an unsupervised method to extract keywords and keyphrases from texts based on a pre-trained language model (LM) and Shannon's information maximization. Specifically, our method extracts phrases having the highest conditional entropy under the LM. The resulting set of keyphrases turns out to solve a relevant information-theoretic problem: if provided as side information, it leads to the expected minimal binary code length in compressing the text using the LM and an entropy encoder. Alternately, the resulting set is an approximation via a causal LM to the set of phrases that minimize the entropy of the text when conditioned upon it. Empirically, the method provides results comparable to the most commonly used methods in various keyphrase extraction benchmark challenges.

{{</citation>}}


### (12/111) Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs (Yuxia Wang et al., 2023)

{{<citation>}}

Yuxia Wang, Haonan Li, Xudong Han, Preslav Nakov, Timothy Baldwin. (2023)  
**Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.13387v1)  

---


**ABSTRACT**  
With the rapid evolution of large language models (LLMs), new and hard-to-predict harmful capabilities are emerging. This requires developers to be able to identify risks through the evaluation of "dangerous capabilities" in order to responsibly deploy LLMs. In this work, we collect the first open-source dataset to evaluate safeguards in LLMs, and deploy safer open-source LLMs at a low cost. Our dataset is curated and filtered to consist only of instructions that responsible language models should not follow. We annotate and assess the responses of six popular LLMs to these instructions. Based on our annotation, we proceed to train several BERT-like classifiers, and find that these small classifiers can achieve results that are comparable with GPT-4 on automatic safety evaluation. Warning: this paper contains example data that may be offensive, harmful, or biased.

{{</citation>}}


### (13/111) Text Style Transfer Evaluation Using Large Language Models (Phil Ostheimer et al., 2023)

{{<citation>}}

Phil Ostheimer, Mayank Nagda, Marius Kloft, Sophie Fellenz. (2023)  
**Text Style Transfer Evaluation Using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Style Transfer  
[Paper Link](http://arxiv.org/abs/2308.13577v1)  

---


**ABSTRACT**  
Text Style Transfer (TST) is challenging to evaluate because the quality of the generated text manifests itself in multiple aspects, each of which is hard to measure individually: style transfer accuracy, content preservation, and overall fluency of the text. Human evaluation is the gold standard in TST evaluation; however, it is expensive, and the results are difficult to reproduce. Numerous automated metrics are employed to assess performance in these aspects, serving as substitutes for human evaluation. However, the correlation between many of these automated metrics and human evaluations remains unclear, raising doubts about their effectiveness as reliable benchmarks. Recent advancements in Large Language Models (LLMs) have demonstrated their ability to not only match but also surpass the average human performance across a wide range of unseen tasks. This suggests that LLMs have the potential to serve as a viable alternative to human evaluation and other automated metrics. We assess the performance of different LLMs on TST evaluation by employing multiple input prompts and comparing their results. Our findings indicate that (even zero-shot) prompting correlates strongly with human evaluation and often surpasses the performance of (other) automated metrics. Additionally, we propose the ensembling of prompts and show it increases the robustness of TST evaluation.This work contributes to the ongoing efforts in evaluating LLMs on diverse tasks, which includes a discussion of failure cases and limitations.

{{</citation>}}


### (14/111) Construction Grammar and Language Models (Harish Tayyar Madabushi et al., 2023)

{{<citation>}}

Harish Tayyar Madabushi, Laurence Romain, Petar Milin, Dagmar Divjak. (2023)  
**Construction Grammar and Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.13315v1)  

---


**ABSTRACT**  
Recent progress in deep learning and natural language processing has given rise to powerful models that are primarily trained on a cloze-like task and show some evidence of having access to substantial linguistic information, including some constructional knowledge. This groundbreaking discovery presents an exciting opportunity for a synergistic relationship between computational methods and Construction Grammar research. In this chapter, we explore three distinct approaches to the interplay between computational methods and Construction Grammar: (i) computational methods for text analysis, (ii) computational Construction Grammar, and (iii) deep learning models, with a particular focus on language models. We touch upon the first two approaches as a contextual foundation for the use of computational methods before providing an accessible, yet comprehensive overview of deep learning models, which also addresses reservations construction grammarians may have. Additionally, we delve into experiments that explore the emergence of constructionally relevant information within these models while also examining the aspects of Construction Grammar that may pose challenges for these models. This chapter aims to foster collaboration between researchers in the fields of natural language processing and Construction Grammar. By doing so, we hope to pave the way for new insights and advancements in both these fields.

{{</citation>}}


### (15/111) Knowledge-Driven CoT: Exploring Faithful Reasoning in LLMs for Knowledge-intensive Question Answering (Keheng Wang et al., 2023)

{{<citation>}}

Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin, Wenge Rong, Zhang Xiong. (2023)  
**Knowledge-Driven CoT: Exploring Faithful Reasoning in LLMs for Knowledge-intensive Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.13259v1)  

---


**ABSTRACT**  
Equipped with Chain-of-Thought (CoT), Large language models (LLMs) have shown impressive reasoning ability in various downstream tasks. Even so, suffering from hallucinations and the inability to access external knowledge, LLMs often come with incorrect or unfaithful intermediate reasoning steps, especially in the context of answering knowledge-intensive tasks such as KBQA. To alleviate this issue, we propose a framework called Knowledge-Driven Chain-of-Thought (KD-CoT) to verify and modify reasoning traces in CoT via interaction with external knowledge, and thus overcome the hallucinations and error propagation. Concretely, we formulate the CoT rationale process of LLMs into a structured multi-round QA format. In each round, LLMs interact with a QA system that retrieves external knowledge and produce faithful reasoning traces based on retrieved precise answers. The structured CoT reasoning of LLMs is facilitated by our developed KBQA CoT collection, which serves as in-context learning demonstrations and can also be utilized as feedback augmentation to train a robust retriever. Extensive experiments on WebQSP and ComplexWebQuestion datasets demonstrate the effectiveness of proposed KD-CoT in task-solving reasoning generation, which outperforms the vanilla CoT ICL with an absolute success rate of 8.0% and 5.1%. Furthermore, our proposed feedback-augmented retriever outperforms the state-of-the-art baselines for retrieving knowledge, achieving significant improvement in Hit performance.

{{</citation>}}


### (16/111) LLM2KB: Constructing Knowledge Bases using instruction tuned context aware Large Language Models (Anmol Nayak et al., 2023)

{{<citation>}}

Anmol Nayak, Hari Prasad Timmapathini. (2023)  
**LLM2KB: Constructing Knowledge Bases using instruction tuned context aware Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.13207v1)  

---


**ABSTRACT**  
The advent of Large Language Models (LLM) has revolutionized the field of natural language processing, enabling significant progress in various applications. One key area of interest is the construction of Knowledge Bases (KB) using these powerful models. Knowledge bases serve as repositories of structured information, facilitating information retrieval and inference tasks. Our paper proposes LLM2KB, a system for constructing knowledge bases using large language models, with a focus on the Llama 2 architecture and the Wikipedia dataset. We perform parameter efficient instruction tuning for Llama-2-13b-chat and StableBeluga-13B by training small injection models that have only 0.05 % of the parameters of the base models using the Low Rank Adaptation (LoRA) technique. These injection models have been trained with prompts that are engineered to utilize Wikipedia page contexts of subject entities fetched using a Dense Passage Retrieval (DPR) algorithm, to answer relevant object entities for a given subject entity and relation. Our best performing model achieved an average F1 score of 0.6185 across 21 relations in the LM-KBC challenge held at the ISWC 2023 conference.

{{</citation>}}


### (17/111) Journey to the Center of the Knowledge Neurons: Discoveries of Language-Independent Knowledge Neurons and Degenerate Knowledge Neurons (Yuheng Chen et al., 2023)

{{<citation>}}

Yuheng Chen, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao. (2023)  
**Journey to the Center of the Knowledge Neurons: Discoveries of Language-Independent Knowledge Neurons and Degenerate Knowledge Neurons**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2308.13198v1)  

---


**ABSTRACT**  
Pre-trained language models (PLMs) contain vast amounts of factual knowledge, but how the knowledge is stored in the parameters remains unclear. This paper delves into the complex task of understanding how factual knowledge is stored in multilingual PLMs, and introduces the Architecture-adapted Multilingual Integrated Gradients method, which successfully localizes knowledge neurons more precisely compared to current methods, and is more universal across various architectures and languages. Moreover, we conduct an in-depth exploration of knowledge neurons, leading to the following two important discoveries: (1) The discovery of Language-Independent Knowledge Neurons, which store factual knowledge in a form that transcends language. We design cross-lingual knowledge editing experiments, demonstrating that the PLMs can accomplish this task based on language-independent neurons; (2) The discovery of Degenerate Knowledge Neurons, a novel type of neuron showing that different knowledge neurons can store the same fact. Its property of functional overlap endows the PLMs with a robust mastery of factual knowledge. We design fact-checking experiments, proving that the degenerate knowledge neurons can help the PLMs to detect wrong facts. Experiments corroborate these findings, shedding light on the mechanisms of factual knowledge storage in multilingual PLMs, and contribute valuable insights to the field. The source code will be made publicly available for further research.

{{</citation>}}


### (18/111) Chunk, Align, Select: A Simple Long-sequence Processing Method for Transformers (Jiawen Xie et al., 2023)

{{<citation>}}

Jiawen Xie, Pengyu Cheng, Xiao Liang, Yong Dai, Nan Du. (2023)  
**Chunk, Align, Select: A Simple Long-sequence Processing Method for Transformers**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.13191v1)  

---


**ABSTRACT**  
Although dominant in natural language processing, transformer-based models remain challenged by the task of long-sequence processing, because the computational cost of self-attention operations in transformers swells quadratically with the input sequence length. To alleviate the complexity of long-sequence processing, we propose a simple framework to enable the offthe-shelf pre-trained transformers to process much longer sequences, while the computation and memory costs remain growing linearly with the input sequence lengths. More specifically, our method divides each long-sequence input into a batch of chunks, then aligns the interchunk information during the encoding steps, and finally selects the most representative hidden states from the encoder for the decoding process. To extract inter-chunk semantic information, we align the start and end token embeddings among chunks in each encoding transformer block. To learn an effective hidden selection policy, we design a dual updating scheme inspired by reinforcement learning, which regards the decoders of transformers as environments, and the downstream performance metrics as the rewards to evaluate the hidden selection actions. Our empirical results on real-world long-text summarization and reading comprehension tasks demonstrate effective improvements compared to prior longsequence processing baselines.

{{</citation>}}


### (19/111) Discovering Mental Health Research Topics with Topic Modeling (Xin Gao et al., 2023)

{{<citation>}}

Xin Gao, Cem Sazara. (2023)  
**Discovering Mental Health Research Topics with Topic Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2308.13569v1)  

---


**ABSTRACT**  
Mental health significantly influences various aspects of our daily lives, and its importance has been increasingly recognized by the research community and the general public, particularly in the wake of the COVID-19 pandemic. This heightened interest is evident in the growing number of publications dedicated to mental health in the past decade. In this study, our goal is to identify general trends in the field and pinpoint high-impact research topics by analyzing a large dataset of mental health research papers. To accomplish this, we collected abstracts from various databases and trained a customized Sentence-BERT based embedding model leveraging the BERTopic framework. Our dataset comprises 96,676 research papers pertaining to mental health, enabling us to examine the relationships between different topics using their abstracts. To evaluate the effectiveness of the model, we compared it against two other state-of-the-art methods: Top2Vec model and LDA-BERT model. The model demonstrated superior performance in metrics that measure topic diversity and coherence. To enhance our analysis, we also generated word clouds to provide a comprehensive overview of the machine learning models applied in mental health research, shedding light on commonly utilized techniques and emerging trends. Furthermore, we provide a GitHub link* to the dataset used in this paper, ensuring its accessibility for further research endeavors.

{{</citation>}}


### (20/111) Measuring Spurious Correlation in Classification: 'Clever Hans' in Translationese (Angana Borah et al., 2023)

{{<citation>}}

Angana Borah, Daria Pylypenko, Cristina Espana-Bonet, Josef van Genabith. (2023)  
**Measuring Spurious Correlation in Classification: 'Clever Hans' in Translationese**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.13170v1)  

---


**ABSTRACT**  
Recent work has shown evidence of 'Clever Hans' behavior in high-performance neural translationese classifiers, where BERT-based classifiers capitalize on spurious correlations, in particular topic information, between data and target classification labels, rather than genuine translationese signals. Translationese signals are subtle (especially for professional translation) and compete with many other signals in the data such as genre, style, author, and, in particular, topic. This raises the general question of how much of the performance of a classifier is really due to spurious correlations in the data versus the signals actually targeted for by the classifier, especially for subtle target signals and in challenging (low resource) data settings. We focus on topic-based spurious correlation and approach the question from two directions: (i) where we have no knowledge about spurious topic information and its distribution in the data, (ii) where we have some indication about the nature of spurious topic correlations. For (i) we develop a measure from first principles capturing alignment of unsupervised topics with target classification labels as an indication of spurious topic information in the data. We show that our measure is the same as purity in clustering and propose a 'topic floor' (as in a 'noise floor') for classification. For (ii) we investigate masking of known spurious topic carriers in classification. Both (i) and (ii) contribute to quantifying and (ii) to mitigating spurious correlations.

{{</citation>}}


### (21/111) SciEval: A Multi-Level Large Language Model Evaluation Benchmark for Scientific Research (Liangtai Sun et al., 2023)

{{<citation>}}

Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu Chen, Kai Yu. (2023)  
**SciEval: A Multi-Level Large Language Model Evaluation Benchmark for Scientific Research**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13149v1)  

---


**ABSTRACT**  
Recently, there has been growing interest in using Large Language Models (LLMs) for scientific research. Numerous benchmarks have been proposed to evaluate the ability of LLMs for scientific research. However, current benchmarks are mostly based on pre-collected objective questions. This design suffers from data leakage problem and lacks the evaluation of subjective Q/A ability. In this paper, we propose SciEval, a comprehensive and multi-disciplinary evaluation benchmark to address these issues. Based on Bloom's taxonomy, SciEval covers four dimensions to systematically evaluate scientific research ability. In particular, we design a "dynamic" subset based on scientific principles to prevent evaluation from potential data leakage. Both objective and subjective questions are included in SciEval. These characteristics make SciEval a more effective benchmark for scientific research ability evaluation of LLMs. Comprehensive experiments on most advanced LLMs show that, although GPT-4 achieves SOTA performance compared to other LLMs, there is still substantial room for improvement, especially for dynamic questions. The data and codes are now publicly available.

{{</citation>}}


### (22/111) MatchXML: An Efficient Text-label Matching Framework for Extreme Multi-label Text Classification (Hui Ye et al., 2023)

{{<citation>}}

Hui Ye, Rajshekhar Sunderraman, Shihao Ji. (2023)  
**MatchXML: An Efficient Text-label Matching Framework for Extreme Multi-label Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Text Classification, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13139v1)  

---


**ABSTRACT**  
The eXtreme Multi-label text Classification(XMC) refers to training a classifier that assigns a text sample with relevant labels from an extremely large-scale label set (e.g., millions of labels). We propose MatchXML, an efficient text-label matching framework for XMC. We observe that the label embeddings generated from the sparse Term Frequency-Inverse Document Frequency(TF-IDF) features have several limitations. We thus propose label2vec to effectively train the semantic dense label embeddings by the Skip-gram model. The dense label embeddings are then used to build a Hierarchical Label Tree by clustering. In fine-tuning the pre-trained encoder Transformer, we formulate the multi-label text classification as a text-label matching problem in a bipartite graph. We then extract the dense text representations from the fine-tuned Transformer. Besides the fine-tuned dense text embeddings, we also extract the static dense sentence embeddings from a pre-trained Sentence Transformer. Finally, a linear ranker is trained by utilizing the sparse TF-IDF features, the fine-tuned dense text representations and static dense sentence features. Experimental results demonstrate that MatchXML achieves state-of-the-art accuracy on five out of six datasets. As for the speed, MatchXML outperforms the competing methods on all the six datasets. Our source code is publicly available at https://github.com/huiyegit/MatchXML.

{{</citation>}}


### (23/111) DARWIN Series: Domain Specific Large Language Models for Natural Science (Tong Xie et al., 2023)

{{<citation>}}

Tong Xie, Yuwei Wan, Wei Huang, Zhenyu Yin, Yixuan Liu, Shaozhou Wang, Qingyuan Linghu, Chunyu Kit, Clara Grazian, Wenjie Zhang, Imran Razzak, Bram Hoex. (2023)  
**DARWIN Series: Domain Specific Large Language Models for Natural Science**  

---
Primary Category: cs.CL  
Categories: cond-mat-mtrl-sci, cs-CL, cs.CL, physics-app-ph  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13565v1)  

---


**ABSTRACT**  
Emerging tools bring forth fresh approaches to work, and the field of natural science is no different. In natural science, traditional manual, serial, and labour-intensive work is being augmented by automated, parallel, and iterative processes driven by artificial intelligence-based experimental automation and more. To add new capabilities in natural science, enabling the acceleration and enrichment of automation of the discovery process, we present DARWIN, a series of tailored LLMs for natural science, mainly in physics, chemistry, and material science. This series relies on open-source LLM, incorporating structured and unstructured scientific knowledge from public datasets and literature. We fine-tuned the models using over 60,000 instruction data points, emphasizing factual correctness. During the fine-tuning, we introduce the Scientific Instruction Generation (SIG) model, automating instruction generation from scientific texts. This eliminates the need for manual extraction or domain-specific knowledge graphs and efficiently injects scientific knowledge into the model. We also explore multi-task training strategies, revealing interconnections between scientific tasks. DARWIN series not only achieves state-of-the-art results on various scientific tasks but also diminishes reliance on closed-source AI models. Our research showcases the ability of LLM in the scientific domain, with the overarching goal of fostering prosperity within the broader AI for science community.

{{</citation>}}


### (24/111) Large Language Models in Analyzing Crash Narratives -- A Comparative Study of ChatGPT, BARD and GPT-4 (Maroa Mumtarin et al., 2023)

{{<citation>}}

Maroa Mumtarin, Md Samiullah Chowdhury, Jonathan Wood. (2023)  
**Large Language Models in Analyzing Crash Narratives -- A Comparative Study of ChatGPT, BARD and GPT-4**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: BARD, ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13563v1)  

---


**ABSTRACT**  
In traffic safety research, extracting information from crash narratives using text analysis is a common practice. With recent advancements of large language models (LLM), it would be useful to know how the popular LLM interfaces perform in classifying or extracting information from crash narratives. To explore this, our study has used the three most popular publicly available LLM interfaces- ChatGPT, BARD and GPT4. This study investigated their usefulness and boundaries in extracting information and answering queries related to accidents from 100 crash narratives from Iowa and Kansas. During the investigation, their capabilities and limitations were assessed and their responses to the queries were compared. Five questions were asked related to the narratives: 1) Who is at-fault? 2) What is the manner of collision? 3) Has the crash occurred in a work-zone? 4) Did the crash involve pedestrians? and 5) What are the sequence of harmful events in the crash? For questions 1 through 4, the overall similarity among the LLMs were 70%, 35%, 96% and 89%, respectively. The similarities were higher while answering direct questions requiring binary responses and significantly lower for complex questions. To compare the responses to question 5, network diagram and centrality measures were analyzed. The network diagram from the three LLMs were not always similar although they sometimes have the same influencing events with high in-degree, out-degree and betweenness centrality. This study suggests using multiple models to extract viable information from narratives. Also, caution must be practiced while using these interfaces to obtain crucial safety related information.

{{</citation>}}


## cs.SE (7)



### (25/111) Human-in-the-loop online just-in-time software defect prediction (Xutong Liu et al., 2023)

{{<citation>}}

Xutong Liu, Yufei Zhou, Yutian Tang, Junyan Qian, Yuming Zhou. (2023)  
**Human-in-the-loop online just-in-time software defect prediction**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.13707v1)  

---


**ABSTRACT**  
Online Just-In-Time Software Defect Prediction (O-JIT-SDP) uses an online model to predict whether a new software change will introduce a bug or not. However, existing studies neglect the interaction of Software Quality Assurance (SQA) staff with the model, which may miss the opportunity to improve the prediction accuracy through the feedback from SQA staff. To tackle this problem, we propose Human-In-The-Loop (HITL) O-JIT-SDP that integrates feedback from SQA staff to enhance the prediction process. Furthermore, we introduce a performance evaluation framework that utilizes a k-fold distributed bootstrap method along with the Wilcoxon signed-rank test. This framework facilitates thorough pairwise comparisons of alternative classification algorithms using a prequential evaluation approach. Our proposal enables continuous statistical testing throughout the prequential process, empowering developers to make real-time decisions based on robust statistical evidence. Through experimentation across 10 GitHub projects, we demonstrate that our evaluation framework enhances the credibility of model evaluation, and the incorporation of HITL feedback elevates the prediction performance of online JIT-SDP models. These advancements hold the potential to significantly enhance the value of O-JIT-SDP for industrial applications.

{{</citation>}}


### (26/111) Does Asking Clarifying Questions Increases Confidence in Generated Code? On the Communication Skills of Large Language Models (Jie JW Wu, 2023)

{{<citation>}}

Jie JW Wu. (2023)  
**Does Asking Clarifying Questions Increases Confidence in Generated Code? On the Communication Skills of Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.13507v1)  

---


**ABSTRACT**  
Large language models (LLMs) have significantly improved the ability to perform tasks in the field of code generation. However, there is still a gap between LLMs being capable coders and being top-tier software engineers. Based on the observation that top-level software engineers often ask clarifying questions to reduce ambiguity in both requirements and coding solutions, we argue that the same should be applied to LLMs for code generation tasks. By asking probing questions in various topics before generating the final code, the challenges of programming with LLMs, such as unclear intent specification, lack of computational thinking, and undesired code quality, may be alleviated. This, in turn, increases confidence in the generated code. In this work, we explore how to leverage better communication skills to achieve greater confidence in generated code. We propose a communication-centered process that uses an LLM-generated communicator to identify issues with high ambiguity or low confidence in problem descriptions and generated code. We then ask clarifying questions to obtain responses from users for refining the code.

{{</citation>}}


### (27/111) Communicating on Security within Software Development Issue Tracking (Léon McGregor et al., 2023)

{{<citation>}}

Léon McGregor, Manuel Maarek, Hans-Wolfgang Loidl. (2023)  
**Communicating on Security within Software Development Issue Tracking**  

---
Primary Category: cs.SE  
Categories: D-2-9; K-6-3, cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.13480v1)  

---


**ABSTRACT**  
During software development, balancing security and non security issues is challenging. We focus on security awareness and approaches taken by non-security experts using software development issue trackers when considering security. We first analyse interfaces from prominent issue trackers to see how they support security communication and how they integrate security scoring. Then, we investigate through a small scale user study what criteria developers take when prioritising issues, in particular observing their attitudes to security.   We find projects make reference to CVSS summaries (Common Vulnerability Scoring System), often alongside CVE reports (Common Vulnerabilities and Exposures), but issue trackers do not often have interfaces designed for this. Users in our study were not comfortable with CVSS analysis, though were able to reason in a manner compatible with CVSS. Detailed explanations and advice were seen as helpful in making security decisions. This suggests that adding improvements to communication through CVSS-like questioning in issue tracking software can elicit better security interactions.

{{</citation>}}


### (28/111) SoTaNa: The Open-Source Software Development Assistant (Ensheng Shi et al., 2023)

{{<citation>}}

Ensheng Shi, Fengji Zhang, Yanlin Wang, Bei Chen, Lun Du, Hongyu Zhang, Shi Han, Dongmei Zhang, Hongbin Sun. (2023)  
**SoTaNa: The Open-Source Software Development Assistant**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, LLaMA  
[Paper Link](http://arxiv.org/abs/2308.13416v1)  

---


**ABSTRACT**  
Software development plays a crucial role in driving innovation and efficiency across modern societies. To meet the demands of this dynamic field, there is a growing need for an effective software development assistant. However, existing large language models represented by ChatGPT suffer from limited accessibility, including training data and model weights. Although other large open-source models like LLaMA have shown promise, they still struggle with understanding human intent. In this paper, we present SoTaNa, an open-source software development assistant. SoTaNa utilizes ChatGPT to generate high-quality instruction-based data for the domain of software engineering and employs a parameter-efficient fine-tuning approach to enhance the open-source foundation model, LLaMA. We evaluate the effectiveness of \our{} in answering Stack Overflow questions and demonstrate its capabilities. Additionally, we discuss its capabilities in code summarization and generation, as well as the impact of varying the volume of generated data on model performance. Notably, SoTaNa can run on a single GPU, making it accessible to a broader range of researchers. Our code, model weights, and data are public at \url{https://github.com/DeepSoftwareAnalytics/SoTaNa}.

{{</citation>}}


### (29/111) On the Impact of Language Selection for Training and Evaluating Programming Language Models (Jonathan Katzy et al., 2023)

{{<citation>}}

Jonathan Katzy, Maliheh Izadi, Arie van Deursen. (2023)  
**On the Impact of Language Selection for Training and Evaluating Programming Language Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-LG, cs-PL, cs-SE, cs.SE  
Keywords: BERT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13354v1)  

---


**ABSTRACT**  
The recent advancements in Transformer-based Language Models have demonstrated significant potential in enhancing the multilingual capabilities of these models. The remarkable progress made in this domain not only applies to natural language tasks but also extends to the domain of programming languages. Despite the ability of these models to learn from multiple languages, evaluations typically focus on particular combinations of the same languages. In this study, we evaluate the similarity of programming languages by analyzing their representations using a CodeBERT-based model. Our experiments reveal that token representation in languages such as C++, Python, and Java exhibit proximity to one another, whereas the same tokens in languages such as Mathematica and R display significant dissimilarity. Our findings suggest that this phenomenon can potentially result in performance challenges when dealing with diverse languages. Thus, we recommend using our similarity measure to select a diverse set of programming languages when training and evaluating future models.

{{</citation>}}


### (30/111) COCO: Testing Code Generation Systems via Concretized Instructions (Ming Yan et al., 2023)

{{<citation>}}

Ming Yan, Junjie Chen, Jie M. Zhang, Xuejie Cao, Chen Yang, Mark Harman. (2023)  
**COCO: Testing Code Generation Systems via Concretized Instructions**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.13319v1)  

---


**ABSTRACT**  
Code generation systems have been extensively developed in recent years to generate source code based on natural language instructions. However, despite their advancements, these systems still face robustness issues where even slightly different instructions can result in significantly different code semantics. Robustness is critical for code generation systems, as it can have significant impacts on software development, software quality, and trust in the generated code. Although existing testing techniques for general text-to-text software can detect some robustness issues, they are limited in effectiveness due to ignoring the characteristics of code generation systems. In this work, we propose a novel technique COCO to test the robustness of code generation systems. It exploits the usage scenario of code generation systems to make the original programming instruction more concrete by incorporating features known to be contained in the original code. A robust system should maintain code semantics for the concretized instruction, and COCO detects robustness inconsistencies when it does not. We evaluated COCO on eight advanced code generation systems, including commercial tools such as Copilot and ChatGPT, using two widely-used datasets. Our results demonstrate the effectiveness of COCO in testing the robustness of code generation systems, outperforming two techniques adopted from general text-to-text software testing by 466.66% and 104.02%, respectively. Furthermore, concretized instructions generated by COCO can help reduce robustness inconsistencies by 18.35% to 53.91% through fine-tuning.

{{</citation>}}


### (31/111) Knowledge-Based Version Incompatibility Detection for Deep Learning (Zhongkai Zhao et al., 2023)

{{<citation>}}

Zhongkai Zhao, Bonan Kou, Mohamed Yilmaz Ibrahim, Muhao Chen, Tianyi Zhang. (2023)  
**Knowledge-Based Version Incompatibility Detection for Deep Learning**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.13276v2)  

---


**ABSTRACT**  
Version incompatibility issues are rampant when reusing or reproducing deep learning models and applications. Existing techniques are limited to library dependency specifications declared in PyPI. Therefore, these techniques cannot detect version issues due to undocumented version constraints or issues involving hardware drivers or OS. To address this challenge, we propose to leverage the abundant discussions of DL version issues from Stack Overflow to facilitate version incompatibility detection. We reformulate the problem of knowledge extraction as a Question-Answering (QA) problem and use a pre-trained QA model to extract version compatibility knowledge from online discussions. The extracted knowledge is further consolidated into a weighted knowledge graph to detect potential version incompatibilities when reusing a DL project. Our evaluation results show that (1) our approach can accurately extract version knowledge with 84% accuracy, and (2) our approach can accurately identify 65% of known version issues in 10 popular DL projects with a high precision (92%), while two state-of-the-art approaches can only detect 29% and 6% of these issues with 33% and 17% precision respectively.

{{</citation>}}


## cs.LG (20)



### (32/111) PAITS: Pretraining and Augmentation for Irregularly-Sampled Time Series (Nicasia Beebe-Wang et al., 2023)

{{<citation>}}

Nicasia Beebe-Wang, Sayna Ebrahimi, Jinsung Yoon, Sercan O. Arik, Tomas Pfister. (2023)  
**PAITS: Pretraining and Augmentation for Irregularly-Sampled Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Augmentation, NLP, Time Series  
[Paper Link](http://arxiv.org/abs/2308.13703v1)  

---


**ABSTRACT**  
Real-world time series data that commonly reflect sequential human behavior are often uniquely irregularly sampled and sparse, with highly nonuniform sampling over time and entities. Yet, commonly-used pretraining and augmentation methods for time series are not specifically designed for such scenarios. In this paper, we present PAITS (Pretraining and Augmentation for Irregularly-sampled Time Series), a framework for identifying suitable pretraining strategies for sparse and irregularly sampled time series datasets. PAITS leverages a novel combination of NLP-inspired pretraining tasks and augmentations, and a random search to identify an effective strategy for a given dataset. We demonstrate that different datasets benefit from different pretraining choices. Compared with prior methods, our approach is better able to consistently improve pretraining across multiple datasets and domains. Our code is available at \url{https://github.com/google-research/google-research/tree/master/irregular_timeseries_pretraining}.

{{</citation>}}


### (33/111) Linear Oscillation: The Aesthetics of Confusion for Vision Transformer (Juyoung Yun, 2023)

{{<citation>}}

Juyoung Yun. (2023)  
**Linear Oscillation: The Aesthetics of Confusion for Vision Transformer**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-NE, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.13670v1)  

---


**ABSTRACT**  
Activation functions are the linchpins of deep learning, profoundly influencing both the representational capacity and training dynamics of neural networks. They shape not only the nature of representations but also optimize convergence rates and enhance generalization potential. Appreciating this critical role, we present the Linear Oscillation (LoC) activation function, defined as $f(x) = x \times \sin(\alpha x + \beta)$. Distinct from conventional activation functions which primarily introduce non-linearity, LoC seamlessly blends linear trajectories with oscillatory deviations. The nomenclature ``Linear Oscillation'' is a nod to its unique attribute of infusing linear activations with harmonious oscillations, capturing the essence of the 'Importance of Confusion'. This concept of ``controlled confusion'' within network activations is posited to foster more robust learning, particularly in contexts that necessitate discerning subtle patterns. Our empirical studies reveal that, when integrated into diverse neural architectures, the LoC activation function consistently outperforms established counterparts like ReLU and Sigmoid. The stellar performance exhibited by the avant-garde Vision Transformer model using LoC further validates its efficacy. This study illuminates the remarkable benefits of the LoC over other prominent activation functions. It champions the notion that intermittently introducing deliberate complexity or ``confusion'' during training can spur more profound and nuanced learning. This accentuates the pivotal role of judiciously selected activation functions in shaping the future of neural network training.

{{</citation>}}


### (34/111) Network Embedding Using Sparse Approximations of Random Walks (Paula Mercurio et al., 2023)

{{<citation>}}

Paula Mercurio, Di Liu. (2023)  
**Network Embedding Using Sparse Approximations of Random Walks**  

---
Primary Category: cs.LG  
Categories: 05C81 (Primary) 68R10, 05C62 (Secondary), cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.13663v1)  

---


**ABSTRACT**  
In this paper, we propose an efficient numerical implementation of Network Embedding based on commute times, using sparse approximation of a diffusion process on the network obtained by a modified version of the diffusion wavelet algorithm. The node embeddings are computed by optimizing the cross entropy loss via the stochastic gradient descent method with sampling of low-dimensional representations of green functions. We demonstrate the efficacy of this method for data clustering and multi-label classification through several examples, and compare its performance over existing methods in terms of efficiency and accuracy. Theoretical issues justifying the scheme are also discussed.

{{</citation>}}


### (35/111) GRASP: A Rehearsal Policy for Efficient Online Continual Learning (Md Yousuf Harun et al., 2023)

{{<citation>}}

Md Yousuf Harun, Jhair Gallardo, Christopher Kanan. (2023)  
**GRASP: A Rehearsal Policy for Efficient Online Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.13646v1)  

---


**ABSTRACT**  
Continual learning (CL) in deep neural networks (DNNs) involves incrementally accumulating knowledge in a DNN from a growing data stream. A major challenge in CL is that non-stationary data streams cause catastrophic forgetting of previously learned abilities. Rehearsal is a popular and effective way to mitigate this problem, which is storing past observations in a buffer and mixing them with new observations during learning. This leads to a question: Which stored samples should be selected for rehearsal? Choosing samples that are best for learning, rather than simply selecting them at random, could lead to significantly faster learning. For class incremental learning, prior work has shown that a simple class balanced random selection policy outperforms more sophisticated methods. Here, we revisit this question by exploring a new sample selection policy called GRASP. GRASP selects the most prototypical (class representative) samples first and then gradually selects less prototypical (harder) examples to update the DNN. GRASP has little additional compute or memory overhead compared to uniform selection, enabling it to scale to large datasets. We evaluate GRASP and other policies by conducting CL experiments on the large-scale ImageNet-1K and Places-LT image classification datasets. GRASP outperforms all other rehearsal policies. Beyond vision, we also demonstrate that GRASP is effective for CL on five text classification datasets.

{{</citation>}}


### (36/111) Unveiling the Role of Message Passing in Dual-Privacy Preservation on GNNs (Tianyi Zhao et al., 2023)

{{<citation>}}

Tianyi Zhao, Hui Hu, Lu Cheng. (2023)  
**Unveiling the Role of Message Passing in Dual-Privacy Preservation on GNNs**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.13513v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are powerful tools for learning representations on graphs, such as social networks. However, their vulnerability to privacy inference attacks restricts their practicality, especially in high-stake domains. To address this issue, privacy-preserving GNNs have been proposed, focusing on preserving node and/or link privacy. This work takes a step back and investigates how GNNs contribute to privacy leakage. Through theoretical analysis and simulations, we identify message passing under structural bias as the core component that allows GNNs to \textit{propagate} and \textit{amplify} privacy leakage. Building upon these findings, we propose a principled privacy-preserving GNN framework that effectively safeguards both node and link privacy, referred to as dual-privacy preservation. The framework comprises three major modules: a Sensitive Information Obfuscation Module that removes sensitive information from node embeddings, a Dynamic Structure Debiasing Module that dynamically corrects the structural bias, and an Adversarial Learning Module that optimizes the privacy-utility trade-off. Experimental results on four benchmark datasets validate the effectiveness of the proposed model in protecting both node and link privacy while preserving high utility for downstream tasks, such as node classification.

{{</citation>}}


### (37/111) A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance (Ian Colbert et al., 2023)

{{<citation>}}

Ian Colbert, Alessandro Pappalardo, Jakoba Petri-Koenig. (2023)  
**A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-CV, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2308.13504v1)  

---


**ABSTRACT**  
We present accumulator-aware quantization (A2Q), a novel weight quantization method designed to train quantized neural networks (QNNs) to avoid overflow when using low-precision accumulators during inference. A2Q introduces a unique formulation inspired by weight normalization that constrains the L1-norm of model weights according to accumulator bit width bounds that we derive. Thus, in training QNNs for low-precision accumulation, A2Q also inherently promotes unstructured weight sparsity to guarantee overflow avoidance. We apply our method to deep learning-based computer vision tasks to show that A2Q can train QNNs for low-precision accumulators while maintaining model accuracy competitive with a floating-point baseline. In our evaluations, we consider the impact of A2Q on both general-purpose platforms and programmable hardware. However, we primarily target model deployment on FPGAs because they can be programmed to fully exploit custom accumulator bit widths. Our experimentation shows accumulator bit width significantly impacts the resource efficiency of FPGA-based accelerators. On average across our benchmarks, A2Q offers up to a 2.3x reduction in resource utilization over 32-bit accumulator counterparts with 99.2% of the floating-point model accuracy.

{{</citation>}}


### (38/111) Escaping the Sample Trap: Fast and Accurate Epistemic Uncertainty Estimation with Pairwise-Distance Estimators (Lucas Berry et al., 2023)

{{<citation>}}

Lucas Berry, David Meger. (2023)  
**Escaping the Sample Trap: Fast and Accurate Epistemic Uncertainty Estimation with Pairwise-Distance Estimators**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2308.13498v1)  

---


**ABSTRACT**  
This work introduces a novel approach for epistemic uncertainty estimation for ensemble models using pairwise-distance estimators (PaiDEs). These estimators utilize the pairwise-distance between model components to establish bounds on entropy and uses said bounds as estimates for information-based criterion. Unlike recent deep learning methods for epistemic uncertainty estimation, which rely on sample-based Monte Carlo estimators, PaiDEs are able to estimate epistemic uncertainty up to 100$\times$ faster, over a larger space (up to 100$\times$) and perform more accurately in higher dimensions. To validate our approach, we conducted a series of experiments commonly used to evaluate epistemic uncertainty estimation: 1D sinusoidal data, Pendulum-v0, Hopper-v2, Ant-v2 and Humanoid-v2. For each experimental setting, an Active Learning framework was applied to demonstrate the advantages of PaiDEs for epistemic uncertainty estimation.

{{</citation>}}


### (39/111) TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs (Phitchaya Mangpo Phothilimthana et al., 2023)

{{<citation>}}

Phitchaya Mangpo Phothilimthana, Sami Abu-El-Haija, Kaidi Cao, Bahare Fatemi, Charith Mendis, Bryan Perozzi. (2023)  
**TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs-SI, cs.LG  
Keywords: Google, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13490v1)  

---


**ABSTRACT**  
Precise hardware performance models play a crucial role in code optimizations. They can assist compilers in making heuristic decisions or aid autotuners in identifying the optimal configuration for a given program. For example, the autotuner for XLA, a machine learning compiler, discovered 10-20% speedup on state-of-the-art models serving substantial production traffic at Google. Although there exist a few datasets for program performance prediction, they target small sub-programs such as basic blocks or kernels. This paper introduces TpuGraphs, a performance prediction dataset on full tensor programs, represented as computational graphs, running on Tensor Processing Units (TPUs). Each graph in the dataset represents the main computation of a machine learning workload, e.g., a training epoch or an inference step. Each data sample contains a computational graph, a compilation configuration, and the execution time of the graph when compiled with the configuration. The graphs in the dataset are collected from open-source machine learning programs, featuring popular model architectures, e.g., ResNet, EfficientNet, Mask R-CNN, and Transformer. TpuGraphs provides 25x more graphs than the largest graph property prediction dataset (with comparable graph sizes), and 770x larger graphs on average compared to existing performance prediction datasets on machine learning programs. This graph-level prediction task on large graphs introduces new challenges in learning, ranging from scalability, training efficiency, to model quality.

{{</citation>}}


### (40/111) Staleness-Alleviated Distributed GNN Training via Online Dynamic-Embedding Prediction (Guangji Bai et al., 2023)

{{<citation>}}

Guangji Bai, Ziyang Yu, Zheng Chai, Yue Cheng, Liang Zhao. (2023)  
**Staleness-Alleviated Distributed GNN Training via Online Dynamic-Embedding Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.13466v1)  

---


**ABSTRACT**  
Despite the recent success of Graph Neural Networks (GNNs), it remains challenging to train GNNs on large-scale graphs due to neighbor explosions. As a remedy, distributed computing becomes a promising solution by leveraging abundant computing resources (e.g., GPU). However, the node dependency of graph data increases the difficulty of achieving high concurrency in distributed GNN training, which suffers from the massive communication overhead. To address it, Historical value approximation is deemed a promising class of distributed training techniques. It utilizes an offline memory to cache historical information (e.g., node embedding) as an affordable approximation of the exact value and achieves high concurrency. However, such benefits come at the cost of involving dated training information, leading to staleness, imprecision, and convergence issues. To overcome these challenges, this paper proposes SAT (Staleness-Alleviated Training), a novel and scalable distributed GNN training framework that reduces the embedding staleness adaptively. The key idea of SAT is to model the GNN's embedding evolution as a temporal graph and build a model upon it to predict future embedding, which effectively alleviates the staleness of the cached historical embedding. We propose an online algorithm to train the embedding predictor and the distributed GNN alternatively and further provide a convergence analysis. Empirically, we demonstrate that SAT can effectively reduce embedding staleness and thus achieve better performance and convergence speed on multiple large-scale graph datasets.

{{</citation>}}


### (41/111) Nougat: Neural Optical Understanding for Academic Documents (Lukas Blecher et al., 2023)

{{<citation>}}

Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic. (2023)  
**Nougat: Neural Optical Understanding for Academic Documents**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: OCR, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13418v1)  

---


**ABSTRACT**  
Scientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (Neural Optical Understanding for Academic Documents), a Visual Transformer model that performs an Optical Character Recognition (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the accessibility of scientific knowledge in the digital age, by bridging the gap between human-readable documents and machine-readable text. We release the models and code to accelerate future work on scientific text recognition.

{{</citation>}}


### (42/111) TFDNet: Time-Frequency Enhanced Decomposed Network for Long-term Time Series Forecasting (Yuxiao Luo et al., 2023)

{{<citation>}}

Yuxiao Luo, Ziyu Lyu, Xingyu Huang. (2023)  
**TFDNet: Time-Frequency Enhanced Decomposed Network for Long-term Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.13386v1)  

---


**ABSTRACT**  
Long-term time series forecasting is a vital task and has a wide range of real applications. Recent methods focus on capturing the underlying patterns from one single domain (e.g. the time domain or the frequency domain), and have not taken a holistic view to process long-term time series from the time-frequency domains. In this paper, we propose a Time-Frequency Enhanced Decomposed Network (TFDNet) to capture both the long-term underlying patterns and temporal periodicity from the time-frequency domain. In TFDNet, we devise a multi-scale time-frequency enhanced encoder backbone and develop two separate trend and seasonal time-frequency blocks to capture the distinct patterns within the decomposed trend and seasonal components in multi-resolutions. Diverse kernel learning strategies of the kernel operations in time-frequency blocks have been explored, by investigating and incorporating the potential different channel-wise correlation patterns of multivariate time series. Experimental evaluation of eight datasets from five benchmark domains demonstrated that TFDNet is superior to state-of-the-art approaches in both effectiveness and efficiency.

{{</citation>}}


### (43/111) A Generic Machine Learning Framework for Fully-Unsupervised Anomaly Detection with Contaminated Data (Markus Ulmer et al., 2023)

{{<citation>}}

Markus Ulmer, Jannik Zgraggen, Lilach Goren Huber. (2023)  
**A Generic Machine Learning Framework for Fully-Unsupervised Anomaly Detection with Contaminated Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.13352v1)  

---


**ABSTRACT**  
Anomaly detection (AD) tasks have been solved using machine learning algorithms in various domains and applications. The great majority of these algorithms use normal data to train a residual-based model, and assign anomaly scores to unseen samples based on their dissimilarity with the learned normal regime. The underlying assumption of these approaches is that anomaly-free data is available for training. This is, however, often not the case in real-world operational settings, where the training data may be contaminated with a certain fraction of abnormal samples. Training with contaminated data, in turn, inevitably leads to a deteriorated AD performance of the residual-based algorithms.   In this paper we introduce a framework for a fully unsupervised refinement of contaminated training data for AD tasks. The framework is generic and can be applied to any residual-based machine learning model. We demonstrate the application of the framework to two public datasets of multivariate time series machine data from different application fields. We show its clear superiority over the naive approach of training with contaminated data without refinement. Moreover, we compare it to the ideal, unrealistic reference in which anomaly-free data would be available for training. Since the approach exploits information from the anomalies, and not only from the normal regime, it is comparable and often outperforms the ideal baseline as well.

{{</citation>}}


### (44/111) A Bayesian Active Learning Approach to Comparative Judgement (Andy Gray et al., 2023)

{{<citation>}}

Andy Gray, Alma Rahat, Tom Crick, Stephen Lindsay. (2023)  
**A Bayesian Active Learning Approach to Comparative Judgement**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-IR, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2308.13292v1)  

---


**ABSTRACT**  
Assessment is a crucial part of education. Traditional marking is a source of inconsistencies and unconscious bias, placing a high cognitive load on the assessors. An approach to address these issues is comparative judgement (CJ). In CJ, the assessor is presented with a pair of items and is asked to select the better one. Following a series of comparisons, a rank is derived using a ranking model, for example, the BTM, based on the results. While CJ is considered a reliable method for marking, there are concerns around transparency, and the ideal number of pairwise comparisons to generate a reliable estimation of the rank order is not known. Additionally, there have been attempts to generate a method of selecting pairs that should be compared next in an informative manner, but some existing methods are known to have created their own bias within results inflating the reliability metric used. As a result, a random selection approach is usually deployed.   We propose a novel Bayesian approach to CJ (BCJ) for determining the ranks of compared items alongside a new way to select the pairs to present to the marker(s) using active learning (AL), addressing the key shortcomings of traditional CJ. Furthermore, we demonstrate how the entire approach may provide transparency by providing the user insights into how it is making its decisions and, at the same time, being more efficient. Results from our experiments confirm that the proposed BCJ combined with entropy-driven AL pair-selection method is superior to other alternatives. We also find that the more comparisons done, the more accurate BCJ becomes, which solves the issue the current method has of the model deteriorating if too many comparisons are performed. As our approach can generate the complete predicted rank distribution for an item, we also show how this can be utilised in devising a predicted grade, guided by the assessor.

{{</citation>}}


### (45/111) Integrating LLMs and Decision Transformers for Language Grounded Generative Quality-Diversity (Achkan Salehi et al., 2023)

{{<citation>}}

Achkan Salehi, Stephane Doncieux. (2023)  
**Integrating LLMs and Decision Transformers for Language Grounded Generative Quality-Diversity**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Language Model, Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.13278v1)  

---


**ABSTRACT**  
Quality-Diversity is a branch of stochastic optimization that is often applied to problems from the Reinforcement Learning and control domains in order to construct repertoires of well-performing policies/skills that exhibit diversity with respect to a behavior space. Such archives are usually composed of a finite number of reactive agents which are each associated to a unique behavior descriptor, and instantiating behavior descriptors outside of that coarsely discretized space is not straight-forward. While a few recent works suggest solutions to that issue, the trajectory that is generated is not easily customizable beyond the specification of a target behavior descriptor. We propose to jointly solve those problems in environments where semantic information about static scene elements is available by leveraging a Large Language Model to augment the repertoire with natural language descriptions of trajectories, and training a policy conditioned on those descriptions. Thus, our method allows a user to not only specify an arbitrary target behavior descriptor, but also provide the model with a high-level textual prompt to shape the generated trajectory. We also propose an LLM-based approach to evaluating the performance of such generative agents. Furthermore, we develop a benchmark based on simulated robot navigation in a 2d maze that we use for experimental validation.

{{</citation>}}


### (46/111) Heterogeneous Decentralized Machine Unlearning with Seed Model Distillation (Guanhua Ye et al., 2023)

{{<citation>}}

Guanhua Ye, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin. (2023)  
**Heterogeneous Decentralized Machine Unlearning with Seed Model Distillation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Model Distillation  
[Paper Link](http://arxiv.org/abs/2308.13269v2)  

---


**ABSTRACT**  
As some recent information security legislation endowed users with unconditional rights to be forgotten by any trained machine learning model, personalized IoT service providers have to put unlearning functionality into their consideration. The most straightforward method to unlearn users' contribution is to retrain the model from the initial state, which is not realistic in high throughput applications with frequent unlearning requests. Though some machine unlearning frameworks have been proposed to speed up the retraining process, they fail to match decentralized learning scenarios. In this paper, we design a decentralized unlearning framework called HDUS, which uses distilled seed models to construct erasable ensembles for all clients. Moreover, the framework is compatible with heterogeneous on-device models, representing stronger scalability in real-world applications. Extensive experiments on three real-world datasets show that our HDUS achieves state-of-the-art performance.

{{</citation>}}


### (47/111) Model-free Reinforcement Learning with Stochastic Reward Stabilization for Recommender Systems (Tianchi Cai et al., 2023)

{{<citation>}}

Tianchi Cai, Shenliao Bao, Jiyan Jiang, Shiji Zhou, Wenpeng Zhang, Lihong Gu, Jinjie Gu, Guannan Zhang. (2023)  
**Model-free Reinforcement Learning with Stochastic Reward Stabilization for Recommender Systems**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.13246v1)  

---


**ABSTRACT**  
Model-free RL-based recommender systems have recently received increasing research attention due to their capability to handle partial feedback and long-term rewards. However, most existing research has ignored a critical feature in recommender systems: one user's feedback on the same item at different times is random. The stochastic rewards property essentially differs from that in classic RL scenarios with deterministic rewards, which makes RL-based recommender systems much more challenging. In this paper, we first demonstrate in a simulator environment where using direct stochastic feedback results in a significant drop in performance. Then to handle the stochastic feedback more efficiently, we design two stochastic reward stabilization frameworks that replace the direct stochastic feedback with that learned by a supervised model. Both frameworks are model-agnostic, i.e., they can effectively utilize various supervised models. We demonstrate the superiority of the proposed frameworks over different RL-based recommendation baselines with extensive experiments on a recommendation simulator as well as an industrial-level recommender system.

{{</citation>}}


### (48/111) Physics-Inspired Neural Graph ODE for Long-term Dynamical Simulation (Yang Liu et al., 2023)

{{<citation>}}

Yang Liu, Jiashun Cheng, Haihong Zhao, Tingyang Xu, Peilin Zhao, Fugee Tsung, Jia Li, Yu Rong. (2023)  
**Physics-Inspired Neural Graph ODE for Long-term Dynamical Simulation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.13212v1)  

---


**ABSTRACT**  
Simulating and modeling the long-term dynamics of multi-object physical systems is an essential and challenging task. Current studies model the physical systems utilizing Graph Neural Networks (GNNs) with equivariant properties. Specifically, they model the dynamics as a sequence of discrete states with a fixed time interval and learn a direct mapping for all the two adjacent states. However, this direct mapping overlooks the continuous nature between the two states. Namely, we have verified that there are countless possible trajectories between two discrete dynamic states in current GNN-based direct mapping models. This issue greatly hinders the model generalization ability, leading to poor performance of the long-term simulation. In this paper, to better model the latent trajectory through discrete supervision signals, we propose a Physics-Inspired Neural Graph ODE (PINGO) algorithm. In PINGO, to ensure the uniqueness of the trajectory, we construct a Physics-Inspired Neural ODE framework to update the latent trajectory. Meanwhile, to effectively capture intricate interactions among objects, we use a GNN-based model to parameterize Neural ODE in a plug-and-play manner. Furthermore, we prove that the discrepancy between the learned trajectory of PIGNO and the true trajectory can be theoretically bounded. Extensive experiments verify our theoretical findings and demonstrate that our model yields an order-of-magnitude improvement over the state-of-the-art baselines, especially on long-term predictions and roll-out errors.

{{</citation>}}


### (49/111) Stochastic Configuration Machines for Industrial Artificial Intelligence (Dianhui Wang et al., 2023)

{{<citation>}}

Dianhui Wang, Matthew J. Felicetti. (2023)  
**Stochastic Configuration Machines for Industrial Artificial Intelligence**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13570v1)  

---


**ABSTRACT**  
Real-time predictive modelling with desired accuracy is highly expected in industrial artificial intelligence (IAI), where neural networks play a key role. Neural networks in IAI require powerful, high-performance computing devices to operate a large number of floating point data. Based on stochastic configuration networks (SCNs), this paper proposes a new randomized learner model, termed stochastic configuration machines (SCMs), to stress effective modelling and data size saving that are useful and valuable for industrial applications. Compared to SCNs and random vector functional-link (RVFL) nets with binarized implementation, the model storage of SCMs can be significantly compressed while retaining favourable prediction performance. Besides the architecture of the SCM learner model and its learning algorithm, as an important part of this contribution, we also provide a theoretical basis on the learning capacity of SCMs by analysing the model's complexity. Experimental studies are carried out over some benchmark datasets and three industrial applications. The results demonstrate that SCM has great potential for dealing with industrial data analytics.

{{</citation>}}


### (50/111) OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models (Wenqi Shao et al., 2023)

{{<citation>}}

Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, Ping Luo. (2023)  
**OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: LLaMA, Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2308.13137v1)  

---


**ABSTRACT**  
Large language models (LLMs) have revolutionized natural language processing tasks. However, their practical deployment is hindered by their immense memory and computation requirements. Although recent post-training quantization (PTQ) methods are effective in reducing memory footprint and improving the computational efficiency of LLM, they hand-craft quantization parameters, which leads to low performance and fails to deal with extremely low-bit quantization. To tackle this issue, we introduce an Omnidirectionally calibrated Quantization (OmniQuant) technique for LLMs, which achieves good performance in diverse quantization settings while maintaining the computational efficiency of PTQ by efficiently optimizing various quantization parameters. OmniQuant comprises two innovative components including Learnable Weight Clipping (LWC) and Learnable Equivalent Transformation (LET). LWC modulates the extreme values of weights by optimizing the clipping threshold. Meanwhile, LET tackles activation outliers by shifting the challenge of quantization from activations to weights through a learnable equivalent transformation. Operating within a differentiable framework using block-wise error minimization, OmniQuant can optimize the quantization process efficiently for both weight-only and weight-activation quantization. For instance, the LLaMA-2 model family with the size of 7-70B can be processed with OmniQuant on a single A100-40G GPU within 1-16 hours using 128 samples. Extensive experiments validate OmniQuant's superior performance across diverse quantization configurations such as W4A4, W6A6, W4A16, W3A16, and W2A16. Additionally, OmniQuant demonstrates effectiveness in instruction-tuned models and delivers notable improvements in inference speed and memory reduction on real devices. Codes and models are available at \url{https://github.com/OpenGVLab/OmniQuant}.

{{</citation>}}


### (51/111) MLLM-DataEngine: An Iterative Refinement Approach for MLLM (Zhiyuan Zhao et al., 2023)

{{<citation>}}

Zhiyuan Zhao, Linke Ouyang, Bin Wang, Siyuan Huang, Pan Zhang, Xiaoyi Dong, Jiaqi Wang, Conghui He. (2023)  
**MLLM-DataEngine: An Iterative Refinement Approach for MLLM**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13566v1)  

---


**ABSTRACT**  
Despite the great advance of Multimodal Large Language Models (MLLMs) in both instruction dataset building and benchmarking, the independence of training and evaluation makes current MLLMs hard to further improve their capability under the guidance of evaluation results with a relatively low human cost. In this paper, we propose MLLM-DataEngine, a novel closed-loop system that bridges data generation, model training, and evaluation. Within each loop iteration, the MLLM-DataEngine first analyze the weakness of the model based on the evaluation results, then generate a proper incremental dataset for the next training iteration and enhance the model capability iteratively. Compared with previous data collection methods which are separate from the benchmarking, the data generated by MLLM-DataEngine shows better targeting, quality, and correctness. For targeting, we propose an Adaptive Bad-case Sampling module, which adjusts the ratio of different types of data within each incremental dataset based on the benchmarking results. For quality, we resort to GPT-4 to generate high-quality data with each given data type. For correctness, prompt design is critical for the data generation results. Rather than previous hand-crafted prompt, we propose an Interactive Prompt Optimization strategy, which optimizes the prompt with the multi-round interaction between human and GPT, and improve the correctness of generated data greatly. Through extensive experiments, we find our MLLM-DataEngine could boost the MLLM capability in a targeted and automatic manner, with only a few human participation. The MLLM-DataEngine will be released and we hope it could be a general solution for the following MLLMs building.

{{</citation>}}


## cs.SI (3)



### (52/111) Party Prediction for Twitter (Kellin Pelrine et al., 2023)

{{<citation>}}

Kellin Pelrine, Anne Imouza, Zachary Yang, Jacob-Junqi Tian, Sacha Lévy, Gabrielle Desrosiers-Brisebois, Aarash Feizi, Cécile Amadoro, André Blais, Jean-François Godbout, Reihaneh Rabbany. (2023)  
**Party Prediction for Twitter**  

---
Primary Category: cs.SI  
Categories: cs-LG, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2308.13699v1)  

---


**ABSTRACT**  
A large number of studies on social media compare the behaviour of users from different political parties. As a basic step, they employ a predictive model for inferring their political affiliation. The accuracy of this model can change the conclusions of a downstream analysis significantly, yet the choice between different models seems to be made arbitrarily. In this paper, we provide a comprehensive survey and an empirical comparison of the current party prediction practices and propose several new approaches which are competitive with or outperform state-of-the-art methods, yet require less computational resources. Party prediction models rely on the content generated by the users (e.g., tweet texts), the relations they have (e.g., who they follow), or their activities and interactions (e.g., which tweets they like). We examine all of these and compare their signal strength for the party prediction task. This paper lets the practitioner select from a wide range of data types that all give strong performance. Finally, we conduct extensive experiments on different aspects of these methods, such as data collection speed and transfer capabilities, which can provide further insights for both applied and methodological research.

{{</citation>}}


### (53/111) Age of Information Diffusion on Social Networks: Optimizing Multi-Stage Seeding Strategies (Songhua Li et al., 2023)

{{<citation>}}

Songhua Li, Lingjie Duan. (2023)  
**Age of Information Diffusion on Social Networks: Optimizing Multi-Stage Seeding Strategies**  

---
Primary Category: cs.SI  
Categories: cs-DM, cs-IT, cs-SI, cs.SI, math-IT  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2308.13303v1)  

---


**ABSTRACT**  
To promote viral marketing, major social platforms (e.g., Facebook Marketplace and Pinduoduo) repeatedly select and invite different users (as seeds) in online social networks to share fresh information about a product or service with their friends. Thereby, we are motivated to optimize a multi-stage seeding process of viral marketing in social networks and adopt the recent notions of the peak and the average age of information (AoI) to measure the timeliness of promotion information received by network users. Our problem is different from the literature on information diffusion in social networks, which limits to one-time seeding and overlooks AoI dynamics or information replacement over time. As a critical step, we manage to develop closed-form expressions that characterize and trace AoI dynamics over any social network. For the peak AoI problem, we first prove the NP-hardness of our multi-stage seeding problem by a highly non-straightforward reduction from the dominating set problem, and then present a new polynomial-time algorithm that achieves good approximation guarantees (e.g., less than 2 for linear network topology). To minimize the average AoI, we also prove that our problem is NP-hard by properly reducing it from the set cover problem. Benefiting from our two-side bound analysis on the average AoI objective, we build up a new framework for approximation analysis and link our problem to a much simplified sum-distance minimization problem. This intriguing connection inspires us to develop another polynomial-time algorithm that achieves a good approximation guarantee. Additionally, our theoretical results are well corroborated by experiments on a real social network.

{{</citation>}}


### (54/111) Using Adamic-Adar Index Algorithm to Predict Volunteer Collaboration: Less is More (Chao Wu et al., 2023)

{{<citation>}}

Chao Wu, Peng Chen, Baiqiao Yin, Zijuan Lin, Chen Jiang, Di Yu, Changhong Zou, Chunwang Lui. (2023)  
**Using Adamic-Adar Index Algorithm to Predict Volunteer Collaboration: Less is More**  

---
Primary Category: cs.SI  
Categories: cs-LG, cs-SI, cs.SI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13176v1)  

---


**ABSTRACT**  
Social networks exhibit a complex graph-like structure due to the uncertainty surrounding potential collaborations among participants. Machine learning algorithms possess generic outstanding performance in multiple real-world prediction tasks. However, whether machine learning algorithms outperform specific algorithms designed for graph link prediction remains unknown to us. To address this issue, the Adamic-Adar Index (AAI), Jaccard Coefficient (JC) and common neighbour centrality (CNC) as representatives of graph-specific algorithms were applied to predict potential collaborations, utilizing data from volunteer activities during the Covid-19 pandemic in Shenzhen city, along with the classical machine learning algorithms such as random forest, support vector machine, and gradient boosting as single predictors and components of ensemble learning. This paper introduces that the AAI algorithm outperformed the traditional JC and CNC, and other machine learning algorithms in analyzing graph node attributes for this task.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (55/111) 1.5 million materials narratives generated by chatbots (Yang Jeong Park et al., 2023)

{{<citation>}}

Yang Jeong Park, Sung Eun Jerng, Jin-Sung Park, Choah Kwon, Chia-Wei Hsu, Zhichu Ren, Sungroh Yoon, Ju Li. (2023)  
**1.5 million materials narratives generated by chatbots**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat.mtrl-sci, cs-CL  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.13687v1)  

---


**ABSTRACT**  
The advent of artificial intelligence (AI) has enabled a comprehensive exploration of materials for various applications. However, AI models often prioritize frequently encountered materials in the scientific literature, limiting the selection of suitable candidates based on inherent physical and chemical properties. To address this imbalance, we have generated a dataset of 1,494,017 natural language-material paragraphs based on combined OQMD, Materials Project, JARVIS, COD and AFLOW2 databases, which are dominated by ab initio calculations and tend to be much more evenly distributed on the periodic table. The generated text narratives were then polled and scored by both human experts and ChatGPT-4, based on three rubrics: technical accuracy, language and structure, and relevance and depth of content, showing similar scores but with human-scored depth of content being the most lagging. The merger of multi-modality data sources and large language model (LLM) holds immense potential for AI frameworks to help the exploration and discovery of solid-state materials for specific applications.

{{</citation>}}


## cs.CV (29)



### (56/111) ACC-UNet: A Completely Convolutional UNet model for the 2020s (Nabil Ibtehaz et al., 2023)

{{<citation>}}

Nabil Ibtehaz, Daisuke Kihara. (2023)  
**ACC-UNet: A Completely Convolutional UNet model for the 2020s**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.13680v1)  

---


**ABSTRACT**  
This decade is marked by the introduction of Vision Transformer, a radical paradigm shift in broad computer vision. A similar trend is followed in medical imaging, UNet, one of the most influential architectures, has been redesigned with transformers. Recently, the efficacy of convolutional models in vision is being reinvestigated by seminal works such as ConvNext, which elevates a ResNet to Swin Transformer level. Deriving inspiration from this, we aim to improve a purely convolutional UNet model so that it can be on par with the transformer-based models, e.g, Swin-Unet or UCTransNet. We examined several advantages of the transformer-based UNet models, primarily long-range dependencies and cross-level skip connections. We attempted to emulate them through convolution operations and thus propose, ACC-UNet, a completely convolutional UNet model that brings the best of both worlds, the inherent inductive biases of convnets with the design decisions of transformers. ACC-UNet was evaluated on 5 different medical image segmentation benchmarks and consistently outperformed convnets, transformers, and their hybrids. Notably, ACC-UNet outperforms state-of-the-art models Swin-Unet and UCTransNet by $2.64 \pm 2.54\%$ and $0.45 \pm 1.61\%$ in terms of dice score, respectively, while using a fraction of their parameters ($59.26\%$ and $24.24\%$). Our codes are available at https://github.com/kiharalab/ACC-UNet.

{{</citation>}}


### (57/111) An Open Hyperspectral Dataset with Sea-Land-Cloud Ground-Truth from the HYPSO-1 Satellite (Jon A. Justo et al., 2023)

{{<citation>}}

Jon A. Justo, Joseph Garrett, Dennis D. Langer, Marie B. Henriksen, Radu T. Ionescu, Tor A. Johansen. (2023)  
**An Open Hyperspectral Dataset with Sea-Land-Cloud Ground-Truth from the HYPSO-1 Satellite**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13679v1)  

---


**ABSTRACT**  
Hyperspectral Imaging, employed in satellites for space remote sensing, like HYPSO-1, faces constraints due to few labeled data sets, affecting the training of AI models demanding these ground-truth annotations. In this work, we introduce The HYPSO-1 Sea-Land-Cloud-Labeled Dataset, an open dataset with 200 diverse hyperspectral images from the HYPSO-1 mission, available in both raw and calibrated forms for scientific research in Earth observation. Moreover, 38 of these images from different countries include ground-truth labels at pixel-level totaling about 25 million spectral signatures labeled for sea/land/cloud categories. To demonstrate the potential of the dataset and its labeled subset, we have additionally optimized a deep learning model (1D Fully Convolutional Network), achieving superior performance to the current state of the art. The complete dataset, ground-truth labels, deep learning model, and software code are openly accessible for download at the website https://ntnu-smallsat-lab.github.io/hypso1_sea_land_clouds_dataset/ .

{{</citation>}}


### (58/111) Fusion of Infrared and Visible Images based on Spatial-Channel Attentional Mechanism (Qian Xu, 2023)

{{<citation>}}

Qian Xu. (2023)  
**Fusion of Infrared and Visible Images based on Spatial-Channel Attentional Mechanism**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.13672v1)  

---


**ABSTRACT**  
In the study, we present AMFusionNet, an innovative approach to infrared and visible image fusion (IVIF), harnessing the power of multiple kernel sizes and attention mechanisms. By assimilating thermal details from infrared images with texture features from visible sources, our method produces images enriched with comprehensive information. Distinct from prevailing deep learning methodologies, our model encompasses a fusion mechanism powered by multiple convolutional kernels, facilitating the robust capture of a wide feature spectrum. Notably, we incorporate parallel attention mechanisms to emphasize and retain pivotal target details in the resultant images. Moreover, the integration of the multi-scale structural similarity (MS-SSIM) loss function refines network training, optimizing the model for IVIF task. Experimental results demonstrate that our method outperforms state-of-the-art algorithms in terms of quality and quantity. The performance metrics on publicly available datasets also show significant improvement

{{</citation>}}


### (59/111) Enhancing Landmark Detection in Cluttered Real-World Scenarios with Vision Transformers (Mohammad Javad Rajabi et al., 2023)

{{<citation>}}

Mohammad Javad Rajabi, Morteza Mirzai, Ahmad Nickabadi. (2023)  
**Enhancing Landmark Detection in Cluttered Real-World Scenarios with Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.13671v1)  

---


**ABSTRACT**  
Visual place recognition tasks often encounter significant challenges in landmark detection due to the presence of irrelevant objects such as humans, cars, and trees, despite the remarkable progress achieved by previous models, especially in the context of transformers. To address this issue, we propose a novel method that effectively leverages the strengths of vision transformers. By employing a meticulous selection process, our approach identifies and isolates specific patches within the image that correspond to occluding objects. To evaluate the efficacy of our method, we created augmented datasets and conducted comprehensive testing. The results demonstrate the superior accuracy achieved by our proposed approach. This research contributes to the advancement of landmark detection in visual place recognition and shows the potential of leveraging vision transformers to overcome challenges posed by cluttered real-world scenarios.

{{</citation>}}


### (60/111) AdvisingNets: Learning to Distinguish Correct and Wrong Classifications via Nearest-Neighbor Explanations (Giang Nguyen et al., 2023)

{{<citation>}}

Giang Nguyen, Valerie Chen, Anh Nguyen. (2023)  
**AdvisingNets: Learning to Distinguish Correct and Wrong Classifications via Nearest-Neighbor Explanations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13651v1)  

---


**ABSTRACT**  
Besides providing insights into how an image classifier makes its predictions, nearest-neighbor examples also help humans make more accurate decisions. Yet, leveraging this type of explanation to improve both human-AI team accuracy and classifier's accuracy remains an open question. In this paper, we aim to increase both types of accuracy by (1) comparing the input image with post-hoc, nearest-neighbor explanations using a novel network (AdvisingNet), and (2) employing a new reranking algorithm. Over different baseline models, our method consistently improves the image classification accuracy on CUB-200 and Cars-196 datasets. Interestingly, we also reach the state-of-the-art human-AI team accuracy on CUB-200 where both humans and an AdvisingNet make decisions on complementary subsets of images.

{{</citation>}}


### (61/111) Open Gaze: An Open-Source Implementation Replicating Google's Eye Tracking Paper (Sushmanth reddy Mereddy et al., 2023)

{{<citation>}}

Sushmanth reddy Mereddy, Jyothi Swaroop Reddy, Somnath Sharma. (2023)  
**Open Gaze: An Open-Source Implementation Replicating Google's Eye Tracking Paper**  

---
Primary Category: cs.CV  
Categories: 68T10(primary), secondary, I-2-1; I-4-1, cs-AI, cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.13495v1)  

---


**ABSTRACT**  
Eye tracking has been a pivotal tool in diverse fields such as vision research, language analysis, and usability assessment. The majority of prior investigations, however, have concentrated on expansive desktop displays employing specialized, costly eye tracking hardware that lacks scalability. Remarkably little insight exists into ocular movement patterns on smartphones, despite their widespread adoption and significant usage. In this manuscript, we present an open-source implementation of a smartphone-based gaze tracker that emulates the methodology proposed by a GooglePaper (whose source code remains proprietary). Our focus is on attaining accuracy comparable to that attained through the GooglePaper's methodology, without the necessity for supplementary hardware. Through the integration of machine learning techniques, we unveil an accurate eye tracking solution that is native to smartphones. Our approach demonstrates precision akin to the state-of-the-art mobile eye trackers, which are characterized by a cost that is two orders of magnitude higher. Leveraging the vast MIT GazeCapture dataset, which is available through registration on the dataset's website, we successfully replicate crucial findings from previous studies concerning ocular motion behavior in oculomotor tasks and saliency analyses during natural image observation. Furthermore, we emphasize the applicability of smartphone-based gaze tracking in discerning reading comprehension challenges. Our findings exhibit the inherent potential to amplify eye movement research by significant proportions, accommodating participation from thousands of subjects with explicit consent. This scalability not only fosters advancements in vision research, but also extends its benefits to domains such as accessibility enhancement and healthcare applications.

{{</citation>}}


### (62/111) Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers (Matthew Dutson et al., 2023)

{{<citation>}}

Matthew Dutson, Yin Li, Mohit Gupta. (2023)  
**Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.13494v1)  

---


**ABSTRACT**  
Vision Transformers achieve impressive accuracy across a range of visual recognition tasks. Unfortunately, their accuracy frequently comes with high computational costs. This is a particular issue in video recognition, where models are often applied repeatedly across frames or temporal chunks. In this work, we exploit temporal redundancy between subsequent inputs to reduce the cost of Transformers for video processing. We describe a method for identifying and re-processing only those tokens that have changed significantly over time. Our proposed family of models, Eventful Transformers, can be converted from existing Transformers (often without any re-training) and give adaptive control over the compute cost at runtime. We evaluate our method on large-scale datasets for video object detection (ImageNet VID) and action recognition (EPIC-Kitchens 100). Our approach leads to significant computational savings (on the order of 2-4x) with only minor reductions in accuracy.

{{</citation>}}


### (63/111) Ultrafast-and-Ultralight ConvNet-Based Intelligent Monitoring System for Diagnosing Early-Stage Mpox Anytime and Anywhere (Yubiao Yue et al., 2023)

{{<citation>}}

Yubiao Yue, Xiaoqiang Shi, Li Qin, Xinyue Zhang, Yanmei Chen, Jialong Xu, Zipei Zheng, Yujun Cao, Di Liu, Zhenzhang Li, Yang Li. (2023)  
**Ultrafast-and-Ultralight ConvNet-Based Intelligent Monitoring System for Diagnosing Early-Stage Mpox Anytime and Anywhere**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13492v1)  

---


**ABSTRACT**  
Due to the lack of more efficient diagnostic tools for monkeypox, its spread remains unchecked, presenting a formidable challenge to global health. While the high efficacy of deep learning models for monkeypox diagnosis has been demonstrated in related studies, the overlook of inference speed, the parameter size and diagnosis performance for early-stage monkeypox renders the models inapplicable in real-world settings. To address these challenges, we proposed an ultrafast and ultralight network named Fast-MpoxNet. Fast-MpoxNet possesses only 0.27M parameters and can process input images at 68 frames per second (FPS) on the CPU. To counteract the diagnostic performance limitation brought about by the small model capacity, it integrates the attention-based feature fusion module and the multiple auxiliary losses enhancement strategy for better detecting subtle image changes and optimizing weights. Using transfer learning and five-fold cross-validation, Fast-MpoxNet achieves 94.26% Accuracy on the Mpox dataset. Notably, its recall for early-stage monkeypox achieves 93.65%. By adopting data augmentation, our model's Accuracy rises to 98.40% and attains a Practicality Score (A new metric for measuring model practicality in real-time diagnosis application) of 0.80. We also developed an application system named Mpox-AISM V2 for both personal computers and mobile phones. Mpox-AISM V2 features ultrafast responses, offline functionality, and easy deployment, enabling accurate and real-time diagnosis for both the public and individuals in various real-world settings, especially in populous settings during the outbreak. Our work could potentially mitigate future monkeypox outbreak and illuminate a fresh paradigm for developing real-time diagnostic tools in the healthcare field.

{{</citation>}}


### (64/111) RestNet: Boosting Cross-Domain Few-Shot Segmentation with Residual Transformation Network (Xinyang Huang et al., 2023)

{{<citation>}}

Xinyang Huang, Chuang Zhu, Wenkai Chen. (2023)  
**RestNet: Boosting Cross-Domain Few-Shot Segmentation with Residual Transformation Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.13469v1)  

---


**ABSTRACT**  
Cross-domain few-shot segmentation (CD-FSS) aims to achieve semantic segmentation in previously unseen domains with a limited number of annotated samples. Although existing CD-FSS models focus on cross-domain feature transformation, relying exclusively on inter-domain knowledge transfer may lead to the loss of critical intra-domain information. To this end, we propose a novel residual transformation network (RestNet) that facilitates knowledge transfer while retaining the intra-domain support-query feature information. Specifically, we propose a Semantic Enhanced Anchor Transform (SEAT) module that maps features to a stable domain-agnostic space using advanced semantics. Additionally, an Intra-domain Residual Enhancement (IRE) module is designed to maintain the intra-domain representation of the original discriminant space in the new space. We also propose a mask prediction strategy based on prototype fusion to help the model gradually learn how to segment. Our RestNet can transfer cross-domain knowledge from both inter-domain and intra-domain without requiring additional fine-tuning. Extensive experiments on ISIC, Chest X-ray, and FSS-1000 show that our RestNet achieves state-of-the-art performance. Our code will be available soon.

{{</citation>}}


### (65/111) Unlocking Fine-Grained Details with Wavelet-based High-Frequency Enhancement in Transformers (Reza Azad et al., 2023)

{{<citation>}}

Reza Azad, Amirhossein Kazerouni, Alaa Sulaiman, Afshin Bozorgpour, Ehsan Khodapanah Aghdam, Abin Jose, Dorit Merhof. (2023)  
**Unlocking Fine-Grained Details with Wavelet-based High-Frequency Enhancement in Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.13442v1)  

---


**ABSTRACT**  
Medical image segmentation is a critical task that plays a vital role in diagnosis, treatment planning, and disease monitoring. Accurate segmentation of anatomical structures and abnormalities from medical images can aid in the early detection and treatment of various diseases. In this paper, we address the local feature deficiency of the Transformer model by carefully re-designing the self-attention map to produce accurate dense prediction in medical images. To this end, we first apply the wavelet transformation to decompose the input feature map into low-frequency (LF) and high-frequency (HF) subbands. The LF segment is associated with coarse-grained features while the HF components preserve fine-grained features such as texture and edge information. Next, we reformulate the self-attention operation using the efficient Transformer to perform both spatial and context attention on top of the frequency representation. Furthermore, to intensify the importance of the boundary information, we impose an additional attention map by creating a Gaussian pyramid on top of the HF components. Moreover, we propose a multi-scale context enhancement block within skip connections to adaptively model inter-scale dependencies to overcome the semantic gap among stages of the encoder and decoder modules. Throughout comprehensive experiments, we demonstrate the effectiveness of our strategy on multi-organ and skin lesion segmentation benchmarks. The implementation code will be available upon acceptance. \href{https://github.com/mindflow-institue/WaveFormer}{GitHub}.

{{</citation>}}


### (66/111) Position-Enhanced Visual Instruction Tuning for Multimodal Large Language Models (Chi Chen et al., 2023)

{{<citation>}}

Chi Chen, Ruoyu Qin, Fuwen Luo, Xiaoyue Mi, Peng Li, Maosong Sun, Yang Liu. (2023)  
**Position-Enhanced Visual Instruction Tuning for Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.13437v1)  

---


**ABSTRACT**  
Recently, Multimodal Large Language Models (MLLMs) that enable Large Language Models (LLMs) to interpret images through visual instruction tuning have achieved significant success. However, existing visual instruction tuning methods only utilize image-language instruction data to align the language and image modalities, lacking a more fine-grained cross-modal alignment. In this paper, we propose Position-enhanced Visual Instruction Tuning (PVIT), which extends the functionality of MLLMs by integrating an additional region-level vision encoder. This integration promotes a more detailed comprehension of images for the MLLM. In addition, to efficiently achieve a fine-grained alignment between the vision modules and the LLM, we design multiple data generation strategies to construct an image-region-language instruction dataset. Finally, we present both quantitative experiments and qualitative analysis that demonstrate the superiority of the proposed model. Code and data will be released at https://github.com/THUNLP-MT/PVIT.

{{</citation>}}


### (67/111) Exploiting Diverse Feature for Multimodal Sentiment Analysis (Jia Li et al., 2023)

{{<citation>}}

Jia Li, Wei Qian, Kun Li, Qi Li, Dan Guo, Meng Wang. (2023)  
**Exploiting Diverse Feature for Multimodal Sentiment Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2308.13421v1)  

---


**ABSTRACT**  
In this paper, we present our solution to the MuSe-Personalisation sub-challenge in the MuSe 2023 Multimodal Sentiment Analysis Challenge. The task of MuSe-Personalisation aims to predict the continuous arousal and valence values of a participant based on their audio-visual, language, and physiological signal modalities data. Considering different people have personal characteristics, the main challenge of this task is how to build robustness feature presentation for sentiment prediction. To address this issue, we propose exploiting diverse features. Specifically, we proposed a series of feature extraction methods to build a robust representation and model ensemble. We empirically evaluate the performance of the utilized method on the officially provided dataset. \textbf{As a result, we achieved 3rd place in the MuSe-Personalisation sub-challenge.} Specifically, we achieve the results of 0.8492 and 0.8439 for MuSe-Personalisation in terms of arousal and valence CCC.

{{</citation>}}


### (68/111) Harvard Glaucoma Detection and Progression: A Multimodal Multitask Dataset and Generalization-Reinforced Semi-Supervised Learning (Yan Luo et al., 2023)

{{<citation>}}

Yan Luo, Min Shi, Yu Tian, Tobias Elze, Mengyu Wang. (2023)  
**Harvard Glaucoma Detection and Progression: A Multimodal Multitask Dataset and Generalization-Reinforced Semi-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.13411v1)  

---


**ABSTRACT**  
Glaucoma is the number one cause of irreversible blindness globally. A major challenge for accurate glaucoma detection and progression forecasting is the bottleneck of limited labeled patients with the state-of-the-art (SOTA) 3D retinal imaging data of optical coherence tomography (OCT). To address the data scarcity issue, this paper proposes two solutions. First, we develop a novel generalization-reinforced semi-supervised learning (SSL) model called pseudo supervisor to optimally utilize unlabeled data. Compared with SOTA models, the proposed pseudo supervisor optimizes the policy of predicting pseudo labels with unlabeled samples to improve empirical generalization. Our pseudo supervisor model is evaluated with two clinical tasks consisting of glaucoma detection and progression forecasting. The progression forecasting task is evaluated both unimodally and multimodally. Our pseudo supervisor model demonstrates superior performance than SOTA SSL comparison models. Moreover, our model also achieves the best results on the publicly available LAG fundus dataset. Second, we introduce the Harvard Glaucoma Detection and Progression (Harvard-GDP) Dataset, a multimodal multitask dataset that includes data from 1,000 patients with OCT imaging data, as well as labels for glaucoma detection and progression. This is the largest glaucoma detection dataset with 3D OCT imaging data and the first glaucoma progression forecasting dataset that is publicly available. Detailed sex and racial analysis are provided, which can be used by interested researchers for fairness learning studies. Our released dataset is benchmarked with several SOTA supervised CNN and transformer deep learning models. The dataset and code are made publicly available via \url{https://ophai.hms.harvard.edu/datasets/harvard-gdp1000}.

{{</citation>}}


### (69/111) Self-Supervised Representation Learning with Cross-Context Learning between Global and Hypercolumn Features (Zheng Gao et al., 2023)

{{<citation>}}

Zheng Gao, Chen Feng, Ioannis Patras. (2023)  
**Self-Supervised Representation Learning with Cross-Context Learning between Global and Hypercolumn Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.13392v1)  

---


**ABSTRACT**  
Whilst contrastive learning yields powerful representations by matching different augmented views of the same instance, it lacks the ability to capture the similarities between different instances. One popular way to address this limitation is by learning global features (after the global pooling) to capture inter-instance relationships based on knowledge distillation, where the global features of the teacher are used to guide the learning of the global features of the student. Inspired by cross-modality learning, we extend this existing framework that only learns from global features by encouraging the global features and intermediate layer features to learn from each other. This leads to our novel self-supervised framework: cross-context learning between global and hypercolumn features (CGH), that enforces the consistency of instance relations between low- and high-level semantics. Specifically, we stack the intermediate feature maps to construct a hypercolumn representation so that we can measure instance relations using two contexts (hypercolumn and global feature) separately, and then use the relations of one context to guide the learning of the other. This cross-context learning allows the model to learn from the differences between the two contexts. The experimental results on linear classification and downstream tasks show that our method outperforms the state-of-the-art methods.

{{</citation>}}


### (70/111) Prompting Visual-Language Models for Dynamic Facial Expression Recognition (Zengqun Zhao et al., 2023)

{{<citation>}}

Zengqun Zhao, Ioannis Patras. (2023)  
**Prompting Visual-Language Models for Dynamic Facial Expression Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ChatGPT, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13382v1)  

---


**ABSTRACT**  
This paper presents a novel visual-language model called DFER-CLIP, which is based on the CLIP model and designed for in-the-wild Dynamic Facial Expression Recognition (DFER). Specifically, the proposed DFER-CLIP consists of a visual part and a textual part. For the visual part, based on the CLIP image encoder, a temporal model consisting of several Transformer encoders is introduced for extracting temporal facial expression features, and the final feature embedding is obtained as a learnable "class" token. For the textual part, we use as inputs textual descriptions of the facial behaviour that is related to the classes (facial expressions) that we are interested in recognising -- those descriptions are generated using large language models, like ChatGPT. This, in contrast to works that use only the class names and more accurately captures the relationship between them. Alongside the textual description, we introduce a learnable token which helps the model learn relevant context information for each expression during training. Extensive experiments demonstrate the effectiveness of the proposed method and show that our DFER-CLIP also achieves state-of-the-art results compared with the current supervised DFER methods on the DFEW, FERV39k, and MAFW benchmarks. Code is publicly available at https://github.com/zengqunzhao/DFER-CLIP.

{{</citation>}}


### (71/111) Enhanced Mortality Prediction In Patients With Subarachnoid Haemorrhage Using A Deep Learning Model Based On The Initial CT Scan (Sergio Garcia-Garcia et al., 2023)

{{<citation>}}

Sergio Garcia-Garcia, Santiago Cepeda, Dominik Muller, Alejandra Mosteiro, Ramon Torne, Silvia Agudo, Natalia de la Torre, Ignacio Arrese, Rosario Sarabia. (2023)  
**Enhanced Mortality Prediction In Patients With Subarachnoid Haemorrhage Using A Deep Learning Model Based On The Initial CT Scan**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13373v1)  

---


**ABSTRACT**  
PURPOSE: Subarachnoid hemorrhage (SAH) entails high morbidity and mortality rates. Convolutional neural networks (CNN), a form of deep learning, are capable of generating highly accurate predictions from imaging data. Our objective was to predict mortality in SAH patients by processing the initial CT scan on a CNN based algorithm.   METHODS: Retrospective multicentric study of a consecutive cohort of patients with SAH between 2011-2022. Demographic, clinical and radiological variables were analyzed. Pre-processed baseline CT scan images were used as the input for training a CNN using AUCMEDI Framework. Our model's architecture leverages the DenseNet-121 structure, employing transfer learning principles. The output variable was mortality in the first three months. Performance of the model was evaluated by statistical parameters conventionally used in studies involving artificial intelligence methods.   RESULTS: Images from 219 patients were processed, 175 for training and validation of the CNN and 44 for its evaluation. 52%(115/219) of patients were female, and the median age was 58(SD=13.06) years. 18.5%(39/219) were idiopathic SAH. Mortality rate was 28.5%(63/219). The model showed good accuracy at predicting mortality in SAH patients exclusively using the images of the initial CT scan (Accuracy=74%, F1=75% and AUC=82%). CONCLUSION: Modern image processing techniques based on AI and CNN make possible to predict mortality in SAH patients with high accuracy using CT scan images as the only input. These models might be optimized by including more data and patients resulting in better training, development and performance on tasks which are beyond the skills of conventional clinical knowledge.

{{</citation>}}


### (72/111) Burnt area extraction from high-resolution satellite images based on anomaly detection (Oscar David Rafael Narvaez Luces et al., 2023)

{{<citation>}}

Oscar David Rafael Narvaez Luces, Minh-Tan Pham, Quentin Poterek, Rémi Braun. (2023)  
**Burnt area extraction from high-resolution satellite images based on anomaly detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.13367v1)  

---


**ABSTRACT**  
Wildfire detection using satellite images is a widely studied task in remote sensing with many applications to fire delineation and mapping. Recently, deep learning methods have become a scalable solution to automate this task, especially in the field of unsupervised learning where no training data is available. This is particularly important in the context of emergency risk monitoring where fast and effective detection is needed, generally based on high-resolution satellite data. Among various approaches, Anomaly Detection (AD) appears to be highly potential thanks to its broad applications in computer vision, medical imaging, as well as remote sensing. In this work, we build upon the framework of Vector Quantized Variational Autoencoder (VQ-VAE), a popular reconstruction-based AD method with discrete latent spaces, to perform unsupervised burnt area extraction. We integrate VQ-VAE into an end-to-end framework with an intensive post-processing step using dedicated vegetation, water and brightness indexes. Our experiments conducted on high-resolution SPOT-6/7 images provide promising results of the proposed technique, showing its high potential in future research on unsupervised burnt area extraction.

{{</citation>}}


### (73/111) CS-Mixer: A Cross-Scale Vision MLP Model with Spatial-Channel Mixing (Jonathan Cui et al., 2023)

{{<citation>}}

Jonathan Cui, David A. Araujo, Suman Saha, Md. Faisal Kabir. (2023)  
**CS-Mixer: A Cross-Scale Vision MLP Model with Spatial-Channel Mixing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.13363v1)  

---


**ABSTRACT**  
Despite their simpler information fusion designs compared with Vision Transformers and Convolutional Neural Networks, Vision MLP architectures have demonstrated strong performance and high data efficiency in recent research. However, existing works such as CycleMLP and Vision Permutator typically model spatial information in equal-size spatial regions and do not consider cross-scale spatial interactions. Further, their token mixers only model 1- or 2-axis correlations, avoiding 3-axis spatial-channel mixing due to its computational demands. We therefore propose CS-Mixer, a hierarchical Vision MLP that learns dynamic low-rank transformations for spatial-channel mixing through cross-scale local and global aggregation. The proposed methodology achieves competitive results on popular image recognition benchmarks without incurring substantially more compute. Our largest model, CS-Mixer-L, reaches 83.2% top-1 accuracy on ImageNet-1k with 13.7 GFLOPs and 94 M parameters.

{{</citation>}}


### (74/111) A Re-Parameterized Vision Transformer (ReVT) for Domain-Generalized Semantic Segmentation (Jan-Aike Termöhlen et al., 2023)

{{<citation>}}

Jan-Aike Termöhlen, Timo Bartels, Tim Fingscheidt. (2023)  
**A Re-Parameterized Vision Transformer (ReVT) for Domain-Generalized Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13331v1)  

---


**ABSTRACT**  
The task of semantic segmentation requires a model to assign semantic labels to each pixel of an image. However, the performance of such models degrades when deployed in an unseen domain with different data distributions compared to the training domain. We present a new augmentation-driven approach to domain generalization for semantic segmentation using a re-parameterized vision transformer (ReVT) with weight averaging of multiple models after training. We evaluate our approach on several benchmark datasets and achieve state-of-the-art mIoU performance of 47.3% (prior art: 46.3%) for small models and of 50.1% (prior art: 47.8%) for midsized models on commonly used benchmark datasets. At the same time, our method requires fewer parameters and reaches a higher frame rate than the best prior art. It is also easy to implement and, unlike network ensembles, does not add any computational complexity during inference.

{{</citation>}}


### (75/111) ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis (Yanyan Huang et al., 2023)

{{<citation>}}

Yanyan Huang, Weiqin Zhao, Shujun Wang, Yu Fu, Yuming Jiang, Lequan Yu. (2023)  
**ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.13324v1)  

---


**ABSTRACT**  
Whole slide image (WSI) analysis has become increasingly important in the medical imaging community, enabling automated and objective diagnosis, prognosis, and therapeutic-response prediction. However, in clinical practice, the ever-evolving environment hamper the utility of WSI analysis models. In this paper, we propose the FIRST continual learning framework for WSI analysis, named ConSlide, to tackle the challenges of enormous image size, utilization of hierarchical structure, and catastrophic forgetting by progressive model updating on multiple sequential datasets. Our framework contains three key components. The Hierarchical Interaction Transformer (HIT) is proposed to model and utilize the hierarchical structural knowledge of WSI. The Breakup-Reorganize (BuRo) rehearsal method is developed for WSI data replay with efficient region storing buffer and WSI reorganizing operation. The asynchronous updating mechanism is devised to encourage the network to learn generic and specific knowledge respectively during the replay stage, based on a nested cross-scale similarity learning (CSSL) module. We evaluated the proposed ConSlide on four public WSI datasets from TCGA projects. It performs best over other state-of-the-art methods with a fair WSI-based continual learning setting and achieves a better trade-off of the overall performance and forgetting on previous task

{{</citation>}}


### (76/111) SVQNet: Sparse Voxel-Adjacent Query Network for 4D Spatio-Temporal LiDAR Semantic Segmentation (Xuechao Chen et al., 2023)

{{<citation>}}

Xuechao Chen, Shuangjie Xu, Xiaoyi Zou, Tongyi Cao, Dit-Yan Yeung, Lu Fang. (2023)  
**SVQNet: Sparse Voxel-Adjacent Query Network for 4D Spatio-Temporal LiDAR Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.13323v1)  

---


**ABSTRACT**  
LiDAR-based semantic perception tasks are critical yet challenging for autonomous driving. Due to the motion of objects and static/dynamic occlusion, temporal information plays an essential role in reinforcing perception by enhancing and completing single-frame knowledge. Previous approaches either directly stack historical frames to the current frame or build a 4D spatio-temporal neighborhood using KNN, which duplicates computation and hinders realtime performance. Based on our observation that stacking all the historical points would damage performance due to a large amount of redundant and misleading information, we propose the Sparse Voxel-Adjacent Query Network (SVQNet) for 4D LiDAR semantic segmentation. To take full advantage of the historical frames high-efficiently, we shunt the historical points into two groups with reference to the current points. One is the Voxel-Adjacent Neighborhood carrying local enhancing knowledge. The other is the Historical Context completing the global knowledge. Then we propose new modules to select and extract the instructive features from the two groups. Our SVQNet achieves state-of-the-art performance in LiDAR semantic segmentation of the SemanticKITTI benchmark and the nuScenes dataset.

{{</citation>}}


### (77/111) Bridging the Gap: Fine-to-Coarse Sketch Interpolation Network for High-Quality Animation Sketch Inbetweening (Jiaming Shen et al., 2023)

{{<citation>}}

Jiaming Shen, Kun Hu, Wei Bao, Chang Wen Chen, Zhiyong Wang. (2023)  
**Bridging the Gap: Fine-to-Coarse Sketch Interpolation Network for High-Quality Animation Sketch Inbetweening**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Sketch, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13273v1)  

---


**ABSTRACT**  
The 2D animation workflow is typically initiated with the creation of keyframes using sketch-based drawing. Subsequent inbetweens (i.e., intermediate sketch frames) are crafted through manual interpolation for smooth animations, which is a labor-intensive process. Thus, the prospect of automatic animation sketch interpolation has become highly appealing. However, existing video interpolation methods are generally hindered by two key issues for sketch inbetweening: 1) limited texture and colour details in sketches, and 2) exaggerated alterations between two sketch keyframes. To overcome these issues, we propose a novel deep learning method, namely Fine-to-Coarse Sketch Interpolation Network (FC-SIN). This approach incorporates multi-level guidance that formulates region-level correspondence, sketch-level correspondence and pixel-level dynamics. A multi-stream U-Transformer is then devised to characterize sketch inbewteening patterns using these multi-level guides through the integration of both self-attention and cross-attention mechanisms. Additionally, to facilitate future research on animation sketch inbetweening, we constructed a large-scale dataset - STD-12K, comprising 30 sketch animation series in diverse artistic styles. Comprehensive experiments on this dataset convincingly show that our proposed FC-SIN surpasses the state-of-the-art interpolation methods. Our code and dataset will be publicly available.

{{</citation>}}


### (78/111) MultiCapCLIP: Auto-Encoding Prompts for Zero-Shot Multilingual Visual Captioning (Bang Yang et al., 2023)

{{<citation>}}

Bang Yang, Fenglin Liu, Xian Wu, Yaowei Wang, Xu Sun, Yuexian Zou. (2023)  
**MultiCapCLIP: Auto-Encoding Prompts for Zero-Shot Multilingual Visual Captioning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: BLEU, Multilingual, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.13218v1)  

---


**ABSTRACT**  
Supervised visual captioning models typically require a large scale of images or videos paired with descriptions in a specific language (i.e., the vision-caption pairs) for training. However, collecting and labeling large-scale datasets is time-consuming and expensive for many scenarios and languages. Therefore, sufficient labeled pairs are usually not available. To deal with the label shortage problem, we present a simple yet effective zero-shot approach MultiCapCLIP that can generate visual captions for different scenarios and languages without any labeled vision-caption pairs of downstream datasets. In the training stage, MultiCapCLIP only requires text data for input. Then it conducts two main steps: 1) retrieving concept prompts that preserve the corresponding domain knowledge of new scenarios; 2) auto-encoding the prompts to learn writing styles to output captions in a desired language. In the testing stage, MultiCapCLIP instead takes visual data as input directly to retrieve the concept prompts to generate the final visual descriptions. The extensive experiments on image and video captioning across four benchmarks and four languages (i.e., English, Chinese, German, and French) confirm the effectiveness of our approach. Compared with state-of-the-art zero-shot and weakly-supervised methods, our method achieves 4.8% and 21.5% absolute improvements in terms of BLEU@4 and CIDEr metrics. Our code is available at https://github.com/yangbang18/MultiCapCLIP.

{{</citation>}}


### (79/111) GEMTrans: A General, Echocardiography-based, Multi-Level Transformer Framework for Cardiovascular Diagnosis (Masoud Mokhtari et al., 2023)

{{<citation>}}

Masoud Mokhtari, Neda Ahmadi, Teresa S. M. Tsang, Purang Abolmaesumi, Renjie Liao. (2023)  
**GEMTrans: A General, Echocardiography-based, Multi-Level Transformer Framework for Cardiovascular Diagnosis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.13217v1)  

---


**ABSTRACT**  
Echocardiography (echo) is an ultrasound imaging modality that is widely used for various cardiovascular diagnosis tasks. Due to inter-observer variability in echo-based diagnosis, which arises from the variability in echo image acquisition and the interpretation of echo images based on clinical experience, vision-based machine learning (ML) methods have gained popularity to act as secondary layers of verification. For such safety-critical applications, it is essential for any proposed ML method to present a level of explainability along with good accuracy. In addition, such methods must be able to process several echo videos obtained from various heart views and the interactions among them to properly produce predictions for a variety of cardiovascular measurements or interpretation tasks. Prior work lacks explainability or is limited in scope by focusing on a single cardiovascular task. To remedy this, we propose a General, Echo-based, Multi-Level Transformer (GEMTrans) framework that provides explainability, while simultaneously enabling multi-video training where the inter-play among echo image patches in the same frame, all frames in the same video, and inter-video relationships are captured based on a downstream task. We show the flexibility of our framework by considering two critical tasks including ejection fraction (EF) and aortic stenosis (AS) severity detection. Our model achieves mean absolute errors of 4.15 and 4.84 for single and dual-video EF estimation and an accuracy of 96.5 % for AS detection, while providing informative task-specific attention maps and prototypical explainability.

{{</citation>}}


### (80/111) Self-supervised Scene Text Segmentation with Object-centric Layered Representations Augmented by Text Regions (Yibo Wang et al., 2023)

{{<citation>}}

Yibo Wang, Yunhu Ye, Yuanpeng Mao, Yanwei Yu, Yuanping Song. (2023)  
**Self-supervised Scene Text Segmentation with Object-centric Layered Representations Augmented by Text Regions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Text Segmentation  
[Paper Link](http://arxiv.org/abs/2308.13178v1)  

---


**ABSTRACT**  
Text segmentation tasks have a very wide range of application values, such as image editing, style transfer, watermark removal, etc.However, existing public datasets are of poor quality of pixel-level labels that have been shown to be notoriously costly to acquire, both in terms of money and time. At the same time, when pretraining is performed on synthetic datasets, the data distribution of the synthetic datasets is far from the data distribution in the real scene. These all pose a huge challenge to the current pixel-level text segmentation algorithms.To alleviate the above problems, we propose a self-supervised scene text segmentation algorithm with layered decoupling of representations derived from the object-centric manner to segment images into texts and background. In our method, we propose two novel designs which include Region Query Module and Representation Consistency Constraints adapting to the unique properties of text as complements to Auto Encoder, which improves the network's sensitivity to texts.For this unique design, we treat the polygon-level masks predicted by the text localization model as extra input information, and neither utilize any pixel-level mask annotations for training stage nor pretrain on synthetic datasets.Extensive experiments show the effectiveness of the method proposed. On several public scene text datasets, our method outperforms the state-of-the-art unsupervised segmentation algorithms.

{{</citation>}}


### (81/111) DISGO: Automatic End-to-End Evaluation for Scene Text OCR (Mei-Yuh Hwang et al., 2023)

{{<citation>}}

Mei-Yuh Hwang, Yangyang Shi, Ankit Ramchandani, Guan Pang, Praveen Krishnan, Lucas Kabela, Frank Seide, Samyak Datta, Jun Liu. (2023)  
**DISGO: Automatic End-to-End Evaluation for Scene Text OCR**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: BLEU, OCR  
[Paper Link](http://arxiv.org/abs/2308.13173v1)  

---


**ABSTRACT**  
This paper discusses the challenges of optical character recognition (OCR) on natural scenes, which is harder than OCR on documents due to the wild content and various image backgrounds. We propose to uniformly use word error rates (WER) as a new measurement for evaluating scene-text OCR, both end-to-end (e2e) performance and individual system component performances. Particularly for the e2e metric, we name it DISGO WER as it considers Deletion, Insertion, Substitution, and Grouping/Ordering errors. Finally we propose to utilize the concept of super blocks to automatically compute BLEU scores for e2e OCR machine translation. The small SCUT public test set is used to demonstrate WER performance by a modularized OCR system.

{{</citation>}}


### (82/111) IOMatch: Simplifying Open-Set Semi-Supervised Learning with Joint Inliers and Outliers Utilization (Zekun Li et al., 2023)

{{<citation>}}

Zekun Li, Lei Qi, Yinghuan Shi, Yang Gao. (2023)  
**IOMatch: Simplifying Open-Set Semi-Supervised Learning with Joint Inliers and Outliers Utilization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.13168v1)  

---


**ABSTRACT**  
Semi-supervised learning (SSL) aims to leverage massive unlabeled data when labels are expensive to obtain. Unfortunately, in many real-world applications, the collected unlabeled data will inevitably contain unseen-class outliers not belonging to any of the labeled classes. To deal with the challenging open-set SSL task, the mainstream methods tend to first detect outliers and then filter them out. However, we observe a surprising fact that such approach could result in more severe performance degradation when labels are extremely scarce, as the unreliable outlier detector may wrongly exclude a considerable portion of valuable inliers. To tackle with this issue, we introduce a novel open-set SSL framework, IOMatch, which can jointly utilize inliers and outliers, even when it is difficult to distinguish exactly between them. Specifically, we propose to employ a multi-binary classifier in combination with the standard closed-set classifier for producing unified open-set classification targets, which regard all outliers as a single new class. By adopting these targets as open-set pseudo-labels, we optimize an open-set classifier with all unlabeled samples including both inliers and outliers. Extensive experiments have shown that IOMatch significantly outperforms the baseline methods across different benchmark datasets and different settings despite its remarkable simplicity. Our code and models are available at https://github.com/nukezil/IOMatch.

{{</citation>}}


### (83/111) Diff-Retinex: Rethinking Low-light Image Enhancement with A Generative Diffusion Model (Xunpeng Yi et al., 2023)

{{<citation>}}

Xunpeng Yi, Han Xu, Hao Zhang, Linfeng Tang, Jiayi Ma. (2023)  
**Diff-Retinex: Rethinking Low-light Image Enhancement with A Generative Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.13164v1)  

---


**ABSTRACT**  
In this paper, we rethink the low-light image enhancement task and propose a physically explainable and generative diffusion model for low-light image enhancement, termed as Diff-Retinex. We aim to integrate the advantages of the physical model and the generative network. Furthermore, we hope to supplement and even deduce the information missing in the low-light image through the generative network. Therefore, Diff-Retinex formulates the low-light image enhancement problem into Retinex decomposition and conditional image generation. In the Retinex decomposition, we integrate the superiority of attention in Transformer and meticulously design a Retinex Transformer decomposition network (TDN) to decompose the image into illumination and reflectance maps. Then, we design multi-path generative diffusion networks to reconstruct the normal-light Retinex probability distribution and solve the various degradations in these components respectively, including dark illumination, noise, color deviation, loss of scene contents, etc. Owing to generative diffusion model, Diff-Retinex puts the restoration of low-light subtle detail into practice. Extensive experiments conducted on real-world low-light datasets qualitatively and quantitatively demonstrate the effectiveness, superiority, and generalization of the proposed method.

{{</citation>}}


### (84/111) A Survey of Diffusion Based Image Generation Models: Issues and Their Solutions (Tianyi Zhang et al., 2023)

{{<citation>}}

Tianyi Zhang, Zheng Wang, Jing Huang, Mohiuddin Muhammad Tasnim, Wei Shi. (2023)  
**A Survey of Diffusion Based Image Generation Models: Issues and Their Solutions**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, ChatGPT, GPT, Google  
[Paper Link](http://arxiv.org/abs/2308.13142v1)  

---


**ABSTRACT**  
Recently, there has been significant progress in the development of large models. Following the success of ChatGPT, numerous language models have been introduced, demonstrating remarkable performance. Similar advancements have also been observed in image generation models, such as Google's Imagen model, OpenAI's DALL-E 2, and stable diffusion models, which have exhibited impressive capabilities in generating images. However, similar to large language models, these models still encounter unresolved challenges. Fortunately, the availability of open-source stable diffusion models and their underlying mathematical principles has enabled the academic community to extensively analyze the performance of current image generation models and make improvements based on this stable diffusion framework. This survey aims to examine the existing issues and the current solutions pertaining to image generation models.

{{</citation>}}


## eess.IV (3)



### (85/111) AI in Thyroid Cancer Diagnosis: Techniques, Trends, and Future Directions (Yassine Habchi et al., 2023)

{{<citation>}}

Yassine Habchi, Yassine Himeur, Hamza Kheddar, Abdelkrim Boukabou, Shadi Atalla, Ammar Chouchane, Abdelmalik Ouamane, Wathiq Mansoor. (2023)  
**AI in Thyroid Cancer Diagnosis: Techniques, Trends, and Future Directions**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CY, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13592v1)  

---


**ABSTRACT**  
There has been a growing interest in creating intelligent diagnostic systems to assist medical professionals in analyzing and processing big data for the treatment of incurable diseases. One of the key challenges in this field is detecting thyroid cancer, where advancements have been made using machine learning (ML) and big data analytics to evaluate thyroid cancer prognosis and determine a patient's risk of malignancy. This review paper summarizes a large collection of articles related to artificial intelligence (AI)-based techniques used in the diagnosis of thyroid cancer. Accordingly, a new classification was introduced to classify these techniques based on the AI algorithms used, the purpose of the framework, and the computing platforms used. Additionally, this study compares existing thyroid cancer datasets based on their features. The focus of this study is on how AI-based tools can support the diagnosis and treatment of thyroid cancer, through supervised, unsupervised, or hybrid techniques. It also highlights the progress made and the unresolved challenges in this field. Finally, the future trends and areas of focus in this field are discussed.

{{</citation>}}


### (86/111) An investigation into the impact of deep learning model choice on sex and race bias in cardiac MR segmentation (Tiarna Lee et al., 2023)

{{<citation>}}

Tiarna Lee, Esther Puyol-Antón, Bram Ruijsink, Keana Aitcheson, Miaojing Shi, Andrew P. King. (2023)  
**An investigation into the impact of deep learning model choice on sex and race bias in cardiac MR segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13415v1)  

---


**ABSTRACT**  
In medical imaging, artificial intelligence (AI) is increasingly being used to automate routine tasks. However, these algorithms can exhibit and exacerbate biases which lead to disparate performances between protected groups. We investigate the impact of model choice on how imbalances in subject sex and race in training datasets affect AI-based cine cardiac magnetic resonance image segmentation. We evaluate three convolutional neural network-based models and one vision transformer model. We find significant sex bias in three of the four models and racial bias in all of the models. However, the severity and nature of the bias varies between the models, highlighting the importance of model choice when attempting to train fair AI-based segmentation models for medical imaging tasks.

{{</citation>}}


### (87/111) Enhancing Breast Cancer Classification Using Transfer ResNet with Lightweight Attention Mechanism (Suxing Liu, 2023)

{{<citation>}}

Suxing Liu. (2023)  
**Enhancing Breast Cancer Classification Using Transfer ResNet with Lightweight Attention Mechanism**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.13150v1)  

---


**ABSTRACT**  
Deep learning models have revolutionized image classification by learning complex feature hierarchies in raw pixel data. This paper introduces an image classification method based on the ResNet model, and introduces a lightweight attention mechanism framework to improve performance. The framework optimizes feature representation, enhances classification capabilities, and improves feature discriminativeness. We verified the effectiveness of the algorithm on the Breakhis dataset, showing its superior performance in many aspects. Not only in terms of conventional models, our method also shows advantages on state-of-the-art methods such as contemporary visual transformers. Significant improvements have been achieved in metrics such as precision, accuracy, recall, F1-score, and G-means, while also performing well in terms of convergence time. These results strengthen the performance of the algorithm and solidify its application prospects in practical image classification tasks. Keywords: ResNet model, Lightweight attention mechanism

{{</citation>}}


## cs.HC (3)



### (88/111) Queering the ethics of AI (Eduard Fosch-Villaronga et al., 2023)

{{<citation>}}

Eduard Fosch-Villaronga, Gianclaudio Malgieri. (2023)  
**Queering the ethics of AI**  

---
Primary Category: cs.HC  
Categories: K-2; I-2-m, cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13591v1)  

---


**ABSTRACT**  
This book chapter delves into the pressing need to "queer" the ethics of AI to challenge and re-evaluate the normative suppositions and values that underlie AI systems. The chapter emphasizes the ethical concerns surrounding the potential for AI to perpetuate discrimination, including binarism, and amplify existing inequalities due to the lack of representative datasets and the affordances and constraints depending on technology readiness. The chapter argues that a critical examination of the neoliberal conception of equality that often underpins non-discrimination law is necessary and cannot stress more the need to create alternative interdisciplinary approaches that consider the complex and intersecting factors that shape individuals' experiences of discrimination. By exploring such approaches centering on intersectionality and vulnerability-informed design, the chapter contends that designers and developers can create more ethical AI systems that are inclusive, equitable, and responsive to the needs and experiences of all individuals and communities, particularly those who are most vulnerable to discrimination and harm.

{{</citation>}}


### (89/111) WorldSmith: Iterative and Expressive Prompting for World Building with a Generative AI (Hai Dang et al., 2023)

{{<citation>}}

Hai Dang, Frederik Brudy, George Fitzmaurice, Fraser Anderson. (2023)  
**WorldSmith: Iterative and Expressive Prompting for World Building with a Generative AI**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.13355v1)  

---


**ABSTRACT**  
Crafting a rich and unique environment is crucial for fictional world-building, but can be difficult to achieve since illustrating a world from scratch requires time and significant skill. We investigate the use of recent multi-modal image generation systems to enable users iteratively visualize and modify elements of their fictional world using a combination of text input, sketching, and region-based filling. WorldSmith enables novice world builders to quickly visualize a fictional world with layered edits and hierarchical compositions. Through a formative study (4 participants) and first-use study (13 participants) we demonstrate that WorldSmith offers more expressive interactions with prompt-based models. With this work, we explore how creatives can be empowered to leverage prompt-based generative AI as a tool in their creative process, beyond current "click-once" prompting UI paradigms.

{{</citation>}}


### (90/111) Meaningful XAI Based on User-Centric Design Methodology (Winston Maxwell et al., 2023)

{{<citation>}}

Winston Maxwell, Bruno Dumas. (2023)  
**Meaningful XAI Based on User-Centric Design Methodology**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13228v1)  

---


**ABSTRACT**  
This report first takes stock of XAI-related requirements appearing in various EU directives, regulations, guidelines, and CJEU case law. This analysis of existing requirements will permit us to have a clearer vision of the purposes, the ``why'', of XAI, which we separate into five categories: contestability, empowerment/redressing information asymmetries, control over system performance, evaluation of algorithmic decisions, and public administration transparency. The analysis of legal requirements also permits us to create four categories of recipients for explainability: data science teams; human operators of the system; persons affected by algorithmic decisions, and regulators/judges/auditors. Lastly, we identify four main operational contexts for explainability: XAI for the upstream design and testing phase; XAI for human-on-the-loop control; XAI for human-in-the-loop control; and XAI for ex-post challenges and investigations.Second, we will present user-centered design methodology, which takes the purposes, the recipients and the operational context into account in order to develop optimal XAI solutions.Third, we will suggest a methodology to permit suppliers and users of high-risk AI applications to propose local XAI solutions that are effective in the sense of being ``meaningful'', for example, useful in light of the operational, safety and fundamental rights contexts. The process used to develop these ``meaningful'' XAI solutions will be based on user-centric design principles examined in the second part.Fourth, we will suggest that the European Commission issue guidelines to provide a harmonised approach to defining ``meaningful'' explanations based on the purposes, audiences and operational contexts of AI systems. These guidelines would apply to the AI Act, but also to the other EU texts requiring explanations for algorithmic systems and results.

{{</citation>}}


## cs.IR (2)



### (91/111) LSTM-based QoE Evaluation for Web Microservices' Reputation Scoring (Maha Driss, 2023)

{{<citation>}}

Maha Driss. (2023)  
**LSTM-based QoE Evaluation for Web Microservices' Reputation Scoring**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Amazon, LSTM  
[Paper Link](http://arxiv.org/abs/2308.13590v1)  

---


**ABSTRACT**  
Sentiment analysis is the task of mining the authors' opinions about specific entities. It allows organizations to monitor different services in real time and act accordingly. Reputation is what is generally said or believed about people or things. Informally, reputation combines the measure of reliability derived from feedback, reviews, and ratings gathered from users, which reflect their quality of experience (QoE) and can either increase or harm the reputation of the provided services. In this study, we propose to perform sentiment analysis on web microservices reviews to exploit the provided information to assess and score the microservices' reputation. Our proposed approach uses the Long Short-Term Memory (LSTM) model to perform sentiment analysis and the Net Brand Reputation (NBR) algorithm to assess reputation scores for microservices. This approach is tested on a set of more than 10,000 reviews related to 15 Amazon Web microservices, and the experimental results have shown that our approach is more accurate than existing approaches, with an accuracy and precision of 93% obtained after applying an oversampling strategy and a resulting reputation score of the considered microservices community of 89%.

{{</citation>}}


### (92/111) MMBAttn: Max-Mean and Bit-wise Attention for CTR Prediction (Hasan Saribas et al., 2023)

{{<citation>}}

Hasan Saribas, Cagri Yesil, Serdarcan Dilbaz, Halit Orenbas. (2023)  
**MMBAttn: Max-Mean and Bit-wise Attention for CTR Prediction**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.13187v1)  

---


**ABSTRACT**  
With the increasing complexity and scale of click-through rate (CTR) prediction tasks in online advertising and recommendation systems, accurately estimating the importance of features has become a critical aspect of developing effective models. In this paper, we propose an attention-based approach that leverages max and mean pooling operations, along with a bit-wise attention mechanism, to enhance feature importance estimation in CTR prediction. Traditionally, pooling operations such as max and mean pooling have been widely used to extract relevant information from features. However, these operations can lead to information loss and hinder the accurate determination of feature importance. To address this challenge, we propose a novel attention architecture that utilizes a bit-based attention structure that emphasizes the relationships between all bits in features, together with maximum and mean pooling. By considering the fine-grained interactions at the bit level, our method aims to capture intricate patterns and dependencies that might be overlooked by traditional pooling operations. To examine the effectiveness of the proposed method, experiments have been conducted on three public datasets. The experiments demonstrated that the proposed method significantly improves the performance of the base models to achieve state-of-the-art results.

{{</citation>}}


## cs.RO (2)



### (93/111) Towards Optimal Head-to-head Autonomous Racing with Curriculum Reinforcement Learning (Dvij Kalaria et al., 2023)

{{<citation>}}

Dvij Kalaria, Qin Lin, John M. Dolan. (2023)  
**Towards Optimal Head-to-head Autonomous Racing with Curriculum Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.13491v1)  

---


**ABSTRACT**  
Head-to-head autonomous racing is a challenging problem, as the vehicle needs to operate at the friction or handling limits in order to achieve minimum lap times while also actively looking for strategies to overtake/stay ahead of the opponent. In this work we propose a head-to-head racing environment for reinforcement learning which accurately models vehicle dynamics. Some previous works have tried learning a policy directly in the complex vehicle dynamics environment but have failed to learn an optimal policy. In this work, we propose a curriculum learning-based framework by transitioning from a simpler vehicle model to a more complex real environment to teach the reinforcement learning agent a policy closer to the optimal policy. We also propose a control barrier function-based safe reinforcement learning algorithm to enforce the safety of the agent in a more effective way while not compromising on optimality.

{{</citation>}}


### (94/111) iCub Detecting Gazed Objects: A Pipeline Estimating Human Attention (Shiva Hanifi et al., 2023)

{{<citation>}}

Shiva Hanifi, Elisa Maiettini, Maria Lombardi, Lorenzo Natale. (2023)  
**iCub Detecting Gazed Objects: A Pipeline Estimating Human Attention**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.13318v1)  

---


**ABSTRACT**  
This paper explores the role of eye gaze in human-robot interactions and proposes a novel system for detecting objects gazed by the human using solely visual feedback. The system leverages on face detection, human attention prediction, and online object detection, and it allows the robot to perceive and interpret human gaze accurately, paving the way for establishing joint attention with human partners. Additionally, a novel dataset collected with the humanoid robot iCub is introduced, comprising over 22,000 images from ten participants gazing at different annotated objects. This dataset serves as a benchmark for evaluating the performance of the proposed pipeline. The paper also includes an experimental analysis of the pipeline's effectiveness in a human-robot interaction setting, examining the performance of each component. Furthermore, the developed system is deployed on the humanoid robot iCub, and a supplementary video showcases its functionality. The results demonstrate the potential of the proposed approach to enhance social awareness and responsiveness in social robotics, as well as improve assistance and support in collaborative scenarios, promoting efficient human-robot collaboration. The code and the collected dataset will be released upon acceptance.

{{</citation>}}


## cs.CR (3)



### (95/111) Implementing Snort Intrusion Prevention System (IPS) for Network Forensic Analysis (Kashif Ishaq et al., 2023)

{{<citation>}}

Kashif Ishaq, Hafiz Ahsan Javed. (2023)  
**Implementing Snort Intrusion Prevention System (IPS) for Network Forensic Analysis**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Prevention  
[Paper Link](http://arxiv.org/abs/2308.13589v1)  

---


**ABSTRACT**  
The security trade confidentiality, integrity and availability are the main pillar of the information systems as every organization emphasize of the security. From last few decades, digital data is the main asset for every digital or non-digital organization. The proliferation of easily accessible attack software on the internet has lowered the barrier for individuals without hacking skills to engage in malicious activities. An Industrial organization operates a server that (Confluence) serves as a learning platform for newly hired employees or Management training officers, thereby making it vulnerable to potential attacks using readily available internet-based software. To mitigate this risk, it is essential to implement a security system capable of detecting and preventing attacks, as well as conducting investigations. This research project aims to develop a comprehensive security system that can detect attack attempts, initiate preventive measures, and carry out investigations by analyzing attack logs. The study adopted a survey methodology and spanned a period of four months, from March 1, 2023, to June 31, 2023. The outcome of this research is a robust security system that effectively identifies attack attempts, blocks the attacker's IP address, and employs network forensic techniques for investigation purposes. The findings indicate that deploying Snort in IPS mode on PfSense enables the detection of attacks targeting e-learning servers, triggering automatic preventive measures such as IP address blocking. The alerts generated by Snort facilitate investigative actions through network forensics, allowing for accurate reporting on the detrimental effects of the attacks.

{{</citation>}}


### (96/111) Falcon: Accelerating Homomorphically Encrypted Convolutions for Efficient Private Mobile Network Inference (Tianshi Xu et al., 2023)

{{<citation>}}

Tianshi Xu, Meng Li, Runsheng Wang, Ru Huang. (2023)  
**Falcon: Accelerating Homomorphically Encrypted Convolutions for Efficient Private Mobile Network Inference**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Falcon  
[Paper Link](http://arxiv.org/abs/2308.13189v1)  

---


**ABSTRACT**  
Efficient networks, e.g., MobileNetV2, EfficientNet, etc, achieves state-of-the-art (SOTA) accuracy with lightweight computation. However, existing homomorphic encryption (HE)-based two-party computation (2PC) frameworks are not optimized for these networks and suffer from a high inference overhead. We observe the inefficiency mainly comes from the packing algorithm, which ignores the computation characteristics and the communication bottleneck of homomorphically encrypted depthwise convolutions. Therefore, in this paper, we propose Falcon, an effective dense packing algorithm for HE-based 2PC frameworks. Falcon features a zero-aware greedy packing algorithm and a communication-aware operator tiling strategy to improve the packing density for depthwise convolutions. Compared to SOTA HE-based 2PC frameworks, e.g., CrypTFlow2, Iron and Cheetah, Falcon achieves more than 15.6x, 5.1x and 1.8x latency reduction, respectively, at operator level. Meanwhile, at network level, Falcon allows for 1.4% and 4.2% accuracy improvement over Cheetah on CIFAR-100 and TinyImagenet datasets with iso-communication, respecitvely.

{{</citation>}}


### (97/111) A Large-Scale Study of IoT Security Weaknesses and Vulnerabilities in the Wild (Madhu Selvaraj et al., 2023)

{{<citation>}}

Madhu Selvaraj, Gias Uddin. (2023)  
**A Large-Scale Study of IoT Security Weaknesses and Vulnerabilities in the Wild**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.13141v1)  

---


**ABSTRACT**  
Internet of Things (IoT) is defined as the connection between places and physical objects (i.e., things) over the internet/network via smart computing devices. We observed that IoT software developers share solutions to programming questions as code examples on three Stack Exchange Q&A sites: Stack Overflow (SO), Arduino, and Raspberry Pi. Previous research studies found vulnerabilities/weaknesses in C/C++ code examples shared in Stack Overflow. However, the studies did not investigate C/C++ code examples related to IoT. The studies investigated SO code examples only. In this paper, we conduct a large-scale empirical study of all IoT C/C++ code examples shared in the three Stack Exchange sites, i.e., SO, Arduino, and Raspberry Pi. From the 11,329 obtained code snippets from the three sites, we identify 29 distinct CWE (Common Weakness Enumeration) types in 609 snippets. These CWE types can be categorized into 8 general weakness categories, and we observe that evaluation, memory, and initialization related weaknesses are the most common to be introduced by users when posting programming solutions. Furthermore, we find that 39.58% of the vulnerable code snippets contain instances of CWE types that can be mapped to real-world occurrences of those CWE types (i.e. CVE instances). The most number vulnerable IoT code examples was found in Arduino, followed by SO, and Raspberry Pi. Memory type vulnerabilities are on the rise in the sites. For example, from the 3595 mapped CVE instances, we find that 28.99% result in Denial of Service (DoS) errors, which is particularly harmful for network reliant IoT devices such as smart cars. Our study results can guide various IoT stakeholders to be aware of such vulnerable IoT code examples and to inform IoT researchers during their development of tools that can help prevent developers the sharing of such vulnerable code examples in the sites. [Abridged].

{{</citation>}}


## cs.AI (2)



### (98/111) Representing Timed Automata and Timing Anomalies of Cyber-Physical Production Systems in Knowledge Graphs (Tom Westermann et al., 2023)

{{<citation>}}

Tom Westermann, Milapji Singh Gill, Alexander Fay. (2023)  
**Representing Timed Automata and Timing Anomalies of Cyber-Physical Production Systems in Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Anomaly Detection, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.13433v1)  

---


**ABSTRACT**  
Model-Based Anomaly Detection has been a successful approach to identify deviations from the expected behavior of Cyber-Physical Production Systems. Since manual creation of these models is a time-consuming process, it is advantageous to learn them from data and represent them in a generic formalism like timed automata. However, these models - and by extension, the detected anomalies - can be challenging to interpret due to a lack of additional information about the system. This paper aims to improve model-based anomaly detection in CPPS by combining the learned timed automaton with a formal knowledge graph about the system. Both the model and the detected anomalies are described in the knowledge graph in order to allow operators an easier interpretation of the model and the detected anomalies. The authors additionally propose an ontology of the necessary concepts. The approach was validated on a five-tank mixing CPPS and was able to formally define both automata model as well as timing anomalies in automata execution.

{{</citation>}}


### (99/111) Transforming the Output of Generative Pre-trained Transformer: The Influence of the PGI Framework on Attention Dynamics (Aline Ioste, 2023)

{{<citation>}}

Aline Ioste. (2023)  
**Transforming the Output of Generative Pre-trained Transformer: The Influence of the PGI Framework on Attention Dynamics**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Attention, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13317v1)  

---


**ABSTRACT**  
This paper presents a novel approach named Persona-Grouping-Intelligence (PGI), which has been crafted to tackle the challenges posed by GPT models when applied to real-world business issues. PGI leverages the inherent capabilities of the GPT model to comprehend intricate language structures and generate responses that are contextually relevant. The experiment occurred in a business scenario where human intelligence was being underutilized due to less optimized business processes. The primary objective of this approach is to leverage GPT models to reduce the workload on humans in tasks that are extensive, monotonous, and repetitive. Instead, the focus is redirected toward decision-making activities. Remarkably, the experiment yielded an accuracy rate of 93.81% in validating 4,000 responses generated by the model, underscoring the effectiveness of the PGI strategies. Effectively addressing the issue of underutilized human intelligence, this paradigm shift aligns business environments with dynamic machine intelligence, enabling them to navigate the intricacies of real-world challenges. This approach facilitates the practical utilization of these models to tackle actual problems. The methodology offers an opportunity to reshape the fundamental structure of business processes by seamlessly integrating human decision-making with adaptable machine intelligence. Consequently, this optimization enhances operational efficiency and elevates strategic decision-making across diverse business contexts.

{{</citation>}}


## quant-ph (1)



### (100/111) QKSAN: A Quantum Kernel Self-Attention Network (Ren-Xin Zhao et al., 2023)

{{<citation>}}

Ren-Xin Zhao, Jinjing Shi, Xuelong Li. (2023)  
**QKSAN: A Quantum Kernel Self-Attention Network**  

---
Primary Category: quant-ph  
Categories: cs-AI, quant-ph, quant-ph  
Keywords: Attention, NLP, Natural Language Processing, Self-Attention  
[Paper Link](http://arxiv.org/abs/2308.13422v1)  

---


**ABSTRACT**  
Self-Attention Mechanism (SAM) is skilled at extracting important information from the interior of data to improve the computational efficiency of models. Nevertheless, many Quantum Machine Learning (QML) models lack the ability to distinguish the intrinsic connections of information like SAM, which limits their effectiveness on massive high-dimensional quantum data. To address this issue, a Quantum Kernel Self-Attention Mechanism (QKSAM) is introduced, which combines the data representation benefit of Quantum Kernel Methods (QKM) with the efficient information extraction capability of SAM. A Quantum Kernel Self-Attention Network (QKSAN) framework is built based on QKSAM, with Deferred Measurement Principle (DMP) and conditional measurement techniques, which releases half of the quantum resources with probabilistic measurements during computation. The Quantum Kernel Self-Attention Score (QKSAS) determines the measurement conditions and reflects the probabilistic nature of quantum systems. Finally, four QKSAN models are deployed on the Pennylane platform to perform binary classification on MNIST images. The best-performing among the four models is assessed for noise immunity and learning ability. Remarkably, the potential learning benefit of partial QKSAN models over classical deep learning is that they require few parameters for a high return of 98\% $\pm$ 1\% test and train accuracy, even with highly compressed images. QKSAN lays the foundation for future quantum computers to perform machine learning on massive amounts of data, while driving advances in areas such as quantum Natural Language Processing (NLP).

{{</citation>}}


## cs.NE (1)



### (101/111) Reinforcement Learning-assisted Evolutionary Algorithm: A Survey and Research Opportunities (Yanjie Song et al., 2023)

{{<citation>}}

Yanjie Song, Yutong Wu, Yangyang Guo, Ran Yan, P. N. Suganthan, Yue Zhang, Witold Pedrycz, Yingwu Chen, Swagatam Das, Rammohan Mallipeddi, Oladayo Solomon Ajani. (2023)  
**Reinforcement Learning-assisted Evolutionary Algorithm: A Survey and Research Opportunities**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.13420v2)  

---


**ABSTRACT**  
Evolutionary algorithms (EA), a class of stochastic search methods based on the principles of natural evolution, have received widespread acclaim for their exceptional performance in various real-world optimization problems. While researchers worldwide have proposed a wide variety of EAs, certain limitations remain, such as slow convergence speed and poor generalization capabilities. Consequently, numerous scholars actively explore improvements to algorithmic structures, operators, search patterns, etc., to enhance their optimization performance. Reinforcement learning (RL) integrated as a component in the EA framework has demonstrated superior performance in recent years. This paper presents a comprehensive survey on integrating reinforcement learning into the evolutionary algorithm, referred to as reinforcement learning-assisted evolutionary algorithm (RL-EA). We begin with the conceptual outlines of reinforcement learning and the evolutionary algorithm. We then provide a taxonomy of RL-EA. Subsequently, we discuss the RL-EA integration method, the RL-assisted strategy adopted by RL-EA, and its applications according to the existing literature. The RL-assisted procedure is divided according to the implemented functions including solution generation, learnable objective function, algorithm/operator/sub-population selection, parameter adaptation, and other strategies. Finally, we analyze potential directions for future research. This survey serves as a rich resource for researchers interested in RL-EA as it overviews the current state-of-the-art and highlights the associated challenges. By leveraging this survey, readers can swiftly gain insights into RL-EA to develop efficient algorithms, thereby fostering further advancements in this emerging field.

{{</citation>}}


## eess.SY (1)



### (102/111) In-context learning for model-free system identification (Marco Forgione et al., 2023)

{{<citation>}}

Marco Forgione, Filippo Pura, Dario Piga. (2023)  
**In-context learning for model-free system identification**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: GPT, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13380v1)  

---


**ABSTRACT**  
In traditional system identification, we estimate a model of an unknown dynamical system based on given input/output sequences and available physical knowledge. Yet, is it also possible to understand the intricacies of dynamical systems not solely from their input/output patterns, but by observing the behavior of other systems within the same class? This central question drives the study presented in this paper.   In response to this query, we introduce a novel paradigm for system identification, addressing two primary tasks: one-step-ahead prediction and multi-step simulation. Unlike conventional methods, we do not directly estimate a model for the specific system. Instead, we pretrain a meta model that represents a class of dynamical systems. This meta model is trained from a potentially infinite stream of synthetic data, generated by systems randomly extracted from a certain distribution. At its core, the meta model serves as an implicit representation of the main characteristics of a class of dynamical systems. When provided with a brief context from a new system - specifically, a short input/output sequence - the meta model implicitly discerns its dynamics, enabling predictions of its behavior.   The proposed approach harnesses the power of Transformer architectures, renowned for their in-context learning capabilities in Natural Language Processing tasks. For one-step prediction, a GPT-like decoder-only architecture is utilized, whereas the simulation problem employs an encoder-decoder structure.   Initial experimental results affirmatively answer our foundational question, opening doors to fresh research avenues in system identification.

{{</citation>}}


## eess.SP (3)



### (103/111) EOG Artifact Removal from Single and Multi-channel EEG Recordings through the combination of Long Short-Term Memory Networks and Independent Component Analysis (Behrad TaghiBeyglou et al., 2023)

{{<citation>}}

Behrad TaghiBeyglou, Fatemeh Bagheri. (2023)  
**EOG Artifact Removal from Single and Multi-channel EEG Recordings through the combination of Long Short-Term Memory Networks and Independent Component Analysis**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2308.13371v1)  

---


**ABSTRACT**  
Introduction: Electroencephalogram (EEG) signals have gained significant popularity in various applications due to their rich information content. However, these signals are prone to contamination from various sources of artifacts, notably the electrooculogram (EOG) artifacts caused by eye movements. The most effective approach to mitigate EOG artifacts involves recording EOG signals simultaneously with EEG and employing blind source separation techniques, such as independent component analysis (ICA). Nevertheless, the availability of EOG recordings is not always feasible, particularly in pre-recorded datasets. Objective: In this paper, we present a novel methodology that combines a long short-term memory (LSTM)-based neural network with ICA to address the challenge of EOG artifact removal from contaminated EEG signals. Approach: Our approach aims to accomplish two primary objectives: 1) estimate the horizontal and vertical EOG signals from the contaminated EEG data, and 2) employ ICA to eliminate the estimated EOG signals from the EEG, thereby producing an artifact-free EEG signal. Main results: To evaluate the performance of our proposed method, we conducted experiments on a publicly available dataset comprising recordings from 27 participants. We employed well-established metrics such as mean squared error, mean absolute error, and mean error to assess the quality of our artifact removal technique. Significance: Furthermore, we compared the performance of our approach with two state-of-the-art deep learning-based methods reported in the literature, demonstrating the superior performance of our proposed methodology.

{{</citation>}}


### (104/111) FrFT based estimation of linear and nonlinear impairments using Vision Transformer (Ting Jiang et al., 2023)

{{<citation>}}

Ting Jiang, Zheng Gao, Yizhao Chen, Zihe Hu, Ming Tang. (2023)  
**FrFT based estimation of linear and nonlinear impairments using Vision Transformer**  

---
Primary Category: eess.SP  
Categories: cs-AI, eess-SP, eess.SP, physics-optics  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.13575v1)  

---


**ABSTRACT**  
To comprehensively assess optical fiber communication system conditions, it is essential to implement joint estimation of the following four critical impairments: nonlinear signal-to-noise ratio (SNRNL), optical signal-to-noise ratio (OSNR), chromatic dispersion (CD) and differential group delay (DGD). However, current studies only achieve identifying a limited number of impairments within a narrow range, due to limitations in network capabilities and lack of unified representation of impairments. To address these challenges, we adopt time-frequency signal processing based on fractional Fourier transform (FrFT) to achieve the unified representation of impairments, while employing a Transformer based neural networks (NN) to break through network performance limitations. To verify the effectiveness of the proposed estimation method, the numerical simulation is carried on a 5-channel polarization-division-multiplexed quadrature phase shift keying (PDM-QPSK) long haul optical transmission system with the symbol rate of 50 GBaud per channel, the mean absolute error (MAE) for SNRNL, OSNR, CD, and DGD estimation is 0.091 dB, 0.058 dB, 117 ps/nm, and 0.38 ps, and the monitoring window ranges from 0~20 dB, 10~30 dB, 0~51000 ps/nm, and 0~100 ps, respectively. Our proposed method achieves accurate estimation of linear and nonlinear impairments over a broad range, representing a significant advancement in the field of optical performance monitoring (OPM).

{{</citation>}}


### (105/111) EEATC: A Novel Calibration Approach for Low-cost Sensors (M V Narayana et al., 2023)

{{<citation>}}

M V Narayana, Devendra Jalihal, Shiva Nagendra. (2023)  
**EEATC: A Novel Calibration Approach for Low-cost Sensors**  

---
Primary Category: eess.SP  
Categories: cs-SY, eess-SP, eess-SY, eess.SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13572v1)  

---


**ABSTRACT**  
Low-cost sensors (LCS) are affordable, compact, and often portable devices designed to measure various environmental parameters, including air quality. These sensors are intended to provide accessible and cost-effective solutions for monitoring pollution levels in different settings, such as indoor, outdoor and moving vehicles. However, the data produced by LCS is prone to various sources of error that can affect accuracy. Calibration is a well-known procedure to improve the reliability of the data produced by LCS, and several developments and efforts have been made to calibrate the LCS. This work proposes a novel Estimated Error Augmented Two-phase Calibration (\textit{EEATC}) approach to calibrate the LCS in stationary and mobile deployments. In contrast to the existing approaches, the \textit{EEATC} calibrates the LCS in two phases, where the error estimated in the first phase calibration is augmented with the input to the second phase, which helps the second phase to learn the distributional features better to produce more accurate results. We show that the \textit{EEATC} outperforms well-known single-phase calibration models such as linear regression models (single variable linear regression (SLR) and multiple variable linear regression (MLR)) and Random forest (RF) in stationary and mobile deployments. To test the \textit{EEATC} in stationary deployments, we have used the Community Air Sensor Network (CAIRSENSE) data set approved by the United States Environmental Protection Agency (USEPA), and the mobile deployments are tested with the real-time data obtained from SensurAir, an LCS device developed and deployed on moving vehicle in Chennai, India.

{{</citation>}}


## cs.GT (1)



### (106/111) On Incentivizing Social Information Sharing in Routing Games (Songhua Li et al., 2023)

{{<citation>}}

Songhua Li, Lingjie Duan. (2023)  
**On Incentivizing Social Information Sharing in Routing Games**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs-MA, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13301v1)  

---


**ABSTRACT**  
We study a new incentive problem of social information sharing for location-based services (e.g., Waze and Yelp). The problem aims to crowdsource a mass of mobile users to learn massive point-of-interest (PoI) information while traveling and share it with each other as a public good. Given that crowdsourced users mind their own travel costs and possess various preferences over the PoI information along different paths, we formulate the problem as a non-atomic routing game with positive network externalities. We first show by price of anarchy (PoA) analysis that, in the absence of any incentive design, users' selfish routing on the path with the lowest cost will limit information diversity and lead to an arbitrarily large efficiency loss from the social optimum. This motivates us to explore effective incentive mechanisms to remedy while upholding individual rationality, incentive compatibility, and budget balance to ensure practical feasibility. We start by presenting an adaptive information restriction (AIR) mechanism that dynamically customizes restriction fractions, depending on the real user flows along different paths, to govern users' access to the shared PoI aggregation. We show that AIR achieves a PoA of 0.25 for homogeneous users (of identical PoI preferences over paths) and 0.125 for heterogeneous users in a typical network of two parallel paths. Further, we propose a side-payment mechanism (ASP) that adaptively charges or rewards users along certain paths. With those charges and rewards well-tailored, ASP significantly improves the PoA to 1 (optimal) and 0.5 for homogeneous and heterogeneous users in the two-path network, respectively. For a generalized network of multiple parallel paths, we further advance ASP to be able to guarantee a PoA of 0.5. Additionally, our theoretical results are well corroborated by our numerical findings.

{{</citation>}}


## q-fin.TR (1)



### (107/111) JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading (Sascha Frey et al., 2023)

{{<citation>}}

Sascha Frey, Kang Li, Peer Nagy, Silvia Sapora, Chris Lu, Stefan Zohren, Jakob Foerster, Anisoara Calinescu. (2023)  
**JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading**  

---
Primary Category: q-fin.TR  
Categories: cs-AI, cs-CE, cs-LG, q-fin-TR, q-fin.TR  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2308.13289v1)  

---


**ABSTRACT**  
Financial exchanges across the world use limit order books (LOBs) to process orders and match trades. For research purposes it is important to have large scale efficient simulators of LOB dynamics. LOB simulators have previously been implemented in the context of agent-based models (ABMs), reinforcement learning (RL) environments, and generative models, processing order flows from historical data sets and hand-crafted agents alike. For many applications, there is a requirement for processing multiple books, either for the calibration of ABMs or for the training of RL agents. We showcase the first GPU-enabled LOB simulator designed to process thousands of books in parallel, with a notably reduced per-message processing time. The implementation of our simulator - JAX-LOB - is based on design choices that aim to best exploit the powers of JAX without compromising on the realism of LOB-related mechanisms. We integrate JAX-LOB with other JAX packages, to provide an example of how one may address an optimal execution problem with reinforcement learning, and to share some preliminary results from end-to-end RL training on GPUs.

{{</citation>}}


## physics.comp-ph (1)



### (108/111) Bayesian Reasoning for Physics Informed Neural Networks (Krzysztof M. Graczyk et al., 2023)

{{<citation>}}

Krzysztof M. Graczyk, Kornel Witkowski. (2023)  
**Bayesian Reasoning for Physics Informed Neural Networks**  

---
Primary Category: physics.comp-ph  
Categories: cs-LG, physics-comp-ph, physics-flu-dyn, physics.comp-ph, stat-ML  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.13222v1)  

---


**ABSTRACT**  
Physics informed neural network (PINN) approach in Bayesian formulation is presented. We adopt the Bayesian neural network framework formulated by MacKay (Neural Computation 4 (3) (1992) 448). The posterior densities are obtained from Laplace approximation. For each model (fit), the so-called evidence is computed. It is a measure that classifies the hypothesis. The most optimal solution has the maximal value of the evidence. The Bayesian framework allows us to control the impact of the boundary contribution to the total loss. Indeed, the relative weights of loss components are fine-tuned by the Bayesian algorithm. We solve heat, wave, and Burger's equations. The obtained results are in good agreement with the exact solutions. All solutions are provided with the uncertainties computed within the Bayesian framework.

{{</citation>}}


## cs.SD (1)



### (109/111) Deep Active Audio Feature Learning in Resource-Constrained Environments (Md Mohaimenuzzaman et al., 2023)

{{<citation>}}

Md Mohaimenuzzaman, Christoph Bergmeir, Bernd Meyer. (2023)  
**Deep Active Audio Feature Learning in Resource-Constrained Environments**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-SD, cs.SD, eess-AS  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2308.13201v1)  

---


**ABSTRACT**  
The scarcity of labelled data makes training Deep Neural Network (DNN) models in bioacoustic applications challenging. In typical bioacoustics applications, manually labelling the required amount of data can be prohibitively expensive. To effectively identify both new and current classes, DNN models must continue to learn new features from a modest amount of fresh data. Active Learning (AL) is an approach that can help with this learning while requiring little labelling effort. Nevertheless, the use of fixed feature extraction approaches limits feature quality, resulting in underutilization of the benefits of AL. We describe an AL framework that addresses this issue by incorporating feature extraction into the AL loop and refining the feature extractor after each round of manual annotation. In addition, we use raw audio processing rather than spectrograms, which is a novel approach. Experiments reveal that the proposed AL framework requires 14.3%, 66.7%, and 47.4% less labelling effort on benchmark audio datasets ESC-50, UrbanSound8k, and InsectWingBeat, respectively, for a large DNN model and similar savings on a microcontroller-based counterpart. Furthermore, we showcase the practical relevance of our study by incorporating data from conservation biology projects.

{{</citation>}}


## cs.IT (1)



### (110/111) Performance Analysis of Finite Blocklength Transmissions Over Wiretap Fading Channels: An Average Information Leakage Perspective (Milad Tatar Mamaghani et al., 2023)

{{<citation>}}

Milad Tatar Mamaghani, Xiangyun Zhou, Nan Yang, A. Lee Swindlehurst, H. Vincent Poor. (2023)  
**Performance Analysis of Finite Blocklength Transmissions Over Wiretap Fading Channels: An Average Information Leakage Perspective**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13184v1)  

---


**ABSTRACT**  
Physical-layer security (PLS) is a promising technique to complement communication security in beyond-5G wireless networks. However, PLS developments in current research are often based on the ideal assumption of infinite coding blocklengths or perfect knowledge of the wiretap link's channel state information (CSI). In this work, we study the performance of finite blocklength (FBL) transmissions using a new secrecy metric - the average information leakage (AIL). We evaluate the exact and approximate AIL with arbitrary signaling and fading channels, assuming that the eavesdropper's instantaneous CSI is unknown. We then conduct case studies that use artificial noise (AN) beamforming to thoroughly analyze the AIL in both Rayleigh and Rician fading channels. The accuracy of the analytical expressions is verified through extensive simulations, and various insights regarding the impact of key system parameters on the AIL are obtained. Particularly, our results reveal that allowing a small level of AIL can potentially lead to significant reliability improvements. To improve the system performance, we formulate and solve an average secrecy throughput (AST) optimization problem via both non-adaptive and adaptive design strategies. Our findings highlight the significance of blocklength design and AN power allocation, as well as the impact of their trade-off on the AST.

{{</citation>}}


## stat.ML (1)



### (111/111) Nonparametric Additive Value Functions: Interpretable Reinforcement Learning with an Application to Surgical Recovery (Patrick Emedom-Nnamdi et al., 2023)

{{<citation>}}

Patrick Emedom-Nnamdi, Timothy R. Smith, Jukka-Pekka Onnela, Junwei Lu. (2023)  
**Nonparametric Additive Value Functions: Interpretable Reinforcement Learning with an Application to Surgical Recovery**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.13135v1)  

---


**ABSTRACT**  
We propose a nonparametric additive model for estimating interpretable value functions in reinforcement learning. Learning effective adaptive clinical interventions that rely on digital phenotyping features is a major for concern medical practitioners. With respect to spine surgery, different post-operative recovery recommendations concerning patient mobilization can lead to significant variation in patient recovery. While reinforcement learning has achieved widespread success in domains such as games, recent methods heavily rely on black-box methods, such neural networks. Unfortunately, these methods hinder the ability of examining the contribution each feature makes in producing the final suggested decision. While such interpretations are easily provided in classical algorithms such as Least Squares Policy Iteration, basic linearity assumptions prevent learning higher-order flexible interactions between features. In this paper, we present a novel method that offers a flexible technique for estimating action-value functions without making explicit parametric assumptions regarding their additive functional form. This nonparametric estimation strategy relies on incorporating local kernel regression and basis expansion to obtain a sparse, additive representation of the action-value function. Under this approach, we are able to locally approximate the action-value function and retrieve the nonlinear, independent contribution of select features as well as joint feature pairs. We validate the proposed approach with a simulation study, and, in an application to spine disease, uncover recovery recommendations that are inline with related clinical knowledge.

{{</citation>}}
