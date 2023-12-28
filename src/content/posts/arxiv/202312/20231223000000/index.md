---
draft: false
title: "arXiv @ 2023.12.23"
date: 2023-12-23
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.23"
    identifier: arxiv_20231223
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (21)](#cscl-21)
- [cs.LG (19)](#cslg-19)
- [cs.MA (1)](#csma-1)
- [cs.SI (2)](#cssi-2)
- [cs.CR (5)](#cscr-5)
- [cs.IR (2)](#csir-2)
- [cs.AI (4)](#csai-4)
- [cs.SE (2)](#csse-2)
- [cs.NI (1)](#csni-1)
- [cs.CV (30)](#cscv-30)
- [physics.comp-ph (1)](#physicscomp-ph-1)
- [cs.RO (8)](#csro-8)
- [stat.AP (1)](#statap-1)
- [cs.CY (2)](#cscy-2)
- [cs.HC (5)](#cshc-5)
- [cs.SD (5)](#cssd-5)
- [eess.IV (2)](#eessiv-2)
- [eess.AS (2)](#eessas-2)
- [q-fin.PM (1)](#q-finpm-1)
- [q-bio.GN (1)](#q-biogn-1)
- [cs.CE (1)](#csce-1)

## cs.CL (21)



### (1/116) Context-aware Decoding Reduces Hallucination in Query-focused Summarization (Zhichao Xu, 2023)

{{<citation>}}

Zhichao Xu. (2023)  
**Context-aware Decoding Reduces Hallucination in Query-focused Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2312.14335v1)  

---


**ABSTRACT**  
Query-focused summarization (QFS) aims to provide a summary of a single document/multi documents that can satisfy the information needs of a given query. It is useful for various real-world applications, such as abstractive snippet generation or more recent retrieval augmented generation (RAG). A prototypical QFS pipeline consists of a retriever (sparse or dense retrieval) and a generator (usually a large language model). However, applying large language models (LLM) potentially leads to hallucinations, especially when the evidence contradicts the prior belief of LLMs. There has been growing interest in developing new decoding methods to improve generation quality and reduce hallucination. In this work, we conduct a large-scale reproducibility on one recently proposed decoding method -- Context-aware Decoding (CAD). In addition to replicating CAD's experiments on news summarization datasets, we include experiments on QFS datasets, and conduct more rigorous analysis on computational complexity and hyperparameter sensitivity. Experiments with eight different language models show that performance-wise, CAD improves QFS quality by (1) reducing factuality errors/hallucinations while (2) mostly retaining the match of lexical patterns, measured by ROUGE scores, while also at a cost of increased inference-time FLOPs and reduced decoding speed. The code implementation based on Huggingface Library is made available https://github.com/zhichaoxu-shufe/context-aware-decoding-qfs

{{</citation>}}


### (2/116) Parameter Efficient Tuning Allows Scalable Personalization of LLMs for Text Entry: A Case Study on Abbreviation Expansion (Katrin Tomanek et al., 2023)

{{<citation>}}

Katrin Tomanek, Shanqing Cai, Subhashini Venugopalan. (2023)  
**Parameter Efficient Tuning Allows Scalable Personalization of LLMs for Text Entry: A Case Study on Abbreviation Expansion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.14327v1)  

---


**ABSTRACT**  
Abbreviation expansion is a strategy used to speed up communication by limiting the amount of typing and using a language model to suggest expansions. Here we look at personalizing a Large Language Model's (LLM) suggestions based on prior conversations to enhance the relevance of predictions, particularly when the user data is small (~1000 samples). Specifically, we compare fine-tuning, prompt-tuning, and retrieval augmented generation of expanded text suggestions for abbreviated inputs. Our case study with a deployed 8B parameter LLM on a real user living with ALS, and experiments on movie character personalization indicates that (1) customization may be necessary in some scenarios and prompt-tuning generalizes well to those, (2) fine-tuning on in-domain data (with as few as 600 samples) still shows some gains, however (3) retrieval augmented few-shot selection also outperforms fine-tuning. (4) Parameter efficient tuning allows for efficient and scalable personalization. For prompt-tuning, we also find that initializing the learned "soft-prompts" to user relevant concept tokens leads to higher accuracy than random initialization.

{{</citation>}}


### (3/116) T-Eval: Evaluating the Tool Utilization Capability Step by Step (Zehui Chen et al., 2023)

{{<citation>}}

Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, Feng Zhao. (2023)  
**T-Eval: Evaluating the Tool Utilization Capability Step by Step**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.14033v1)  

---


**ABSTRACT**  
Large language models (LLM) have achieved remarkable performance on various NLP tasks and are augmented by tools for broader applications. Yet, how to evaluate and analyze the tool-utilization capability of LLMs is still under-explored. In contrast to previous works that evaluate models holistically, we comprehensively decompose the tool utilization into multiple sub-processes, including instruction following, planning, reasoning, retrieval, understanding, and review. Based on that, we further introduce \shortname~to evaluate the tool utilization capability step by step. \shortname~disentangles the tool utilization evaluation into several sub-domains along model capabilities, facilitating the inner understanding of both holistic and isolated competency of LLMs. We conduct extensive experiments on \shortname~and in-depth analysis of various LLMs. \shortname~ not only exhibits consistency with the outcome-oriented evaluation but also provides a more fine-grained analysis of the capabilities of LLMs, providing a new perspective in LLM evaluation on tool-utilization ability. The benchmark will be available at \href{https://github.com/open-compass/T-Eval}{https://github.com/open-compass/T-Eval}.

{{</citation>}}


### (4/116) Deep de Finetti: Recovering Topic Distributions from Large Language Models (Liyi Zhang et al., 2023)

{{<citation>}}

Liyi Zhang, R. Thomas McCoy, Theodore R. Sumers, Jian-Qiao Zhu, Thomas L. Griffiths. (2023)  
**Deep de Finetti: Recovering Topic Distributions from Large Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-6; I-2-7, cs-AI, cs-CL, cs-LG, cs.CL, stat-ML  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.14226v1)  

---


**ABSTRACT**  
Large language models (LLMs) can produce long, coherent passages of text, suggesting that LLMs, although trained on next-word prediction, must represent the latent structure that characterizes a document. Prior work has found that internal representations of LLMs encode one aspect of latent structure, namely syntax; here we investigate a complementary aspect, namely the document's topic structure. We motivate the hypothesis that LLMs capture topic structure by connecting LLM optimization to implicit Bayesian inference. De Finetti's theorem shows that exchangeable probability distributions can be represented as a mixture with respect to a latent generating distribution. Although text is not exchangeable at the level of syntax, exchangeability is a reasonable starting assumption for topic structure. We thus hypothesize that predicting the next token in text will lead LLMs to recover latent topic distributions. We examine this hypothesis using Latent Dirichlet Allocation (LDA), an exchangeable probabilistic topic model, as a target, and we show that the representations formed by LLMs encode both the topics used to generate synthetic data and those used to explain natural corpus data.

{{</citation>}}


### (5/116) ChatGPT as a commenter to the news: can LLMs generate human-like opinions? (Rayden Tseng et al., 2023)

{{<citation>}}

Rayden Tseng, Suzan Verberne, Peter van der Putten. (2023)  
**ChatGPT as a commenter to the news: can LLMs generate human-like opinions?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: BERT, ChatGPT, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2312.13961v1)  

---


**ABSTRACT**  
ChatGPT, GPT-3.5, and other large language models (LLMs) have drawn significant attention since their release, and the abilities of these models have been investigated for a wide variety of tasks. In this research we investigate to what extent GPT-3.5 can generate human-like comments on Dutch news articles. We define human likeness as `not distinguishable from human comments', approximated by the difficulty of automatic classification between human and GPT comments. We analyze human likeness across multiple prompting techniques. In particular, we utilize zero-shot, few-shot and context prompts, for two generated personas. We found that our fine-tuned BERT models can easily distinguish human-written comments from GPT-3.5 generated comments, with none of the used prompting methods performing noticeably better. We further analyzed that human comments consistently showed higher lexical diversity than GPT-generated comments. This indicates that although generative LLMs can generate fluent text, their capability to create human-like opinionated comments is still limited.

{{</citation>}}


### (6/116) Typhoon: Thai Large Language Models (Kunat Pipatanakul et al., 2023)

{{<citation>}}

Kunat Pipatanakul, Phatrasek Jirabovonvisut, Potsawee Manakul, Sittipong Sripaisarnmongkol, Ruangsak Patomwong, Pathomporn Chokchainant, Kasima Tharnpipitchai. (2023)  
**Typhoon: Thai Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2312.13951v1)  

---


**ABSTRACT**  
Typhoon is a series of Thai large language models (LLMs) developed specifically for the Thai language. This technical report presents challenges and insights in developing Thai LLMs, including data preparation, pretraining, instruction-tuning, and evaluation. As one of the challenges of low-resource languages is the amount of pretraining data, we apply continual training to transfer existing world knowledge from a strong LLM. To evaluate the Thai knowledge encapsulated in each model from the pretraining stage, we develop ThaiExam, a benchmark based on examinations for high-school students and investment professionals in Thailand. In addition, we fine-tune Typhoon to follow Thai instructions, and we evaluate instruction-tuned models on Thai instruction datasets as well as translation, summarization, and question-answering tasks. Experimental results on a suite of Thai benchmarks show that Typhoon outperforms all open-source Thai language models, and its performance is on par with GPT-3.5 in Thai while having only 7 billion parameters and being 2.62 times more efficient in tokenizing Thai text.

{{</citation>}}


### (7/116) Diversifying Knowledge Enhancement of Biomedical Language Models using Adapter Modules and Knowledge Graphs (Juraj Vladika et al., 2023)

{{<citation>}}

Juraj Vladika, Alexander Fichtl, Florian Matthes. (2023)  
**Diversifying Knowledge Enhancement of Biomedical Language Models using Adapter Modules and Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Knowledge Graph, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.13881v1)  

---


**ABSTRACT**  
Recent advances in natural language processing (NLP) owe their success to pre-training language models on large amounts of unstructured data. Still, there is an increasing effort to combine the unstructured nature of LMs with structured knowledge and reasoning. Particularly in the rapidly evolving field of biomedical NLP, knowledge-enhanced language models (KELMs) have emerged as promising tools to bridge the gap between large language models and domain-specific knowledge, considering the available biomedical knowledge graphs (KGs) curated by experts over the decades. In this paper, we develop an approach that uses lightweight adapter modules to inject structured biomedical knowledge into pre-trained language models (PLMs). We use two large KGs, the biomedical knowledge system UMLS and the novel biochemical ontology OntoChem, with two prominent biomedical PLMs, PubMedBERT and BioLinkBERT. The approach includes partitioning knowledge graphs into smaller subgraphs, fine-tuning adapter modules for each subgraph, and combining the knowledge in a fusion layer. We test the performance on three downstream tasks: document classification,question answering, and natural language inference. We show that our methodology leads to performance improvements in several instances while keeping requirements in computing power low. Finally, we provide a detailed interpretation of the results and report valuable insights for future work.

{{</citation>}}


### (8/116) Evaluating Task-oriented Dialogue Systems: A Systematic Review of Measures, Constructs and their Operationalisations (Anouck Braggaar et al., 2023)

{{<citation>}}

Anouck Braggaar, Christine Liebrecht, Emiel van Miltenburg, Emiel Krahmer. (2023)  
**Evaluating Task-oriented Dialogue Systems: A Systematic Review of Measures, Constructs and their Operationalisations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.13871v1)  

---


**ABSTRACT**  
This review gives an extensive overview of evaluation methods for task-oriented dialogue systems, paying special attention to practical applications of dialogue systems, for example for customer service. The review (1) provides an overview of the used constructs and metrics in previous work, (2) discusses challenges in the context of dialogue system evaluation and (3) develops a research agenda for the future of dialogue system evaluation. We conducted a systematic review of four databases (ACL, ACM, IEEE and Web of Science), which after screening resulted in 122 studies. Those studies were carefully analysed for the constructs and methods they proposed for evaluation. We found a wide variety in both constructs and methods. Especially the operationalisation is not always clearly reported. We hope that future work will take a more critical approach to the operationalisation and specification of the used constructs. To work towards this aim, this review ends with recommendations for evaluation and suggestions for outstanding questions.

{{</citation>}}


### (9/116) Team Flow at DRC2023: Building Common Ground and Text-based Turn-taking in a Travel Agent Spoken Dialogue System (Ryu Hirai et al., 2023)

{{<citation>}}

Ryu Hirai, Shinya Iizuka, Haruhisa Iseno, Ao Guo, Jingjing Jiang, Atsumoto Ohashi, Ryuichiro Higashinaka. (2023)  
**Team Flow at DRC2023: Building Common Ground and Text-based Turn-taking in a Travel Agent Spoken Dialogue System**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-RO, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.13816v1)  

---


**ABSTRACT**  
At the Dialogue Robot Competition 2023 (DRC2023), which was held to improve the capability of dialogue robots, our team developed a system that could build common ground and take more natural turns based on user utterance texts. Our system generated queries for sightseeing spot searches using the common ground and engaged in dialogue while waiting for user comprehension.

{{</citation>}}


### (10/116) SimLM: Can Language Models Infer Parameters of Physical Systems? (Sean Memery et al., 2023)

{{<citation>}}

Sean Memery, Mirella Lapata, Kartic Subr. (2023)  
**SimLM: Can Language Models Infer Parameters of Physical Systems?**  

---
Primary Category: cs.CL  
Categories: I-2-7; I-6, cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.14215v1)  

---


**ABSTRACT**  
Recent developments in large-scale machine learning models for general-purpose understanding, translation and generation of language are driving impact across a variety of sectors including medicine, robotics, and scientific discovery. The strength of such Large Language Models (LLMs) stems from the large corpora that they are trained with. While this imbues them with a breadth of capabilities, they have been found unsuitable for some specific types of problems such as advanced mathematics. In this paper, we highlight the inability of LLMs to reason about physics tasks. We demonstrate that their ability to infer parameters of physical systems can be improved, without retraining, by augmenting their context with feedback from physical simulation.

{{</citation>}}


### (11/116) Experimenting with Large Language Models and vector embeddings in NASA SciX (Sergi Blanco-Cuaresma et al., 2023)

{{<citation>}}

Sergi Blanco-Cuaresma, Ioana Ciucă, Alberto Accomazzi, Michael J. Kurtz, Edwin A. Henneken, Kelly E. Lockhart, Felix Grezes, Thomas Allen, Golnaz Shapurian, Carolyn S. Grant, Donna M. Thompson, Timothy W. Hostetler, Matthew R. Templeton, Shinyi Chen, Jennifer Koch, Taylor Jacovich, Daniel Chivvis, Fernanda de Macedo Alves, Jean-Claude Paquin, Jennifer Bartlett, Mugdha Polimera, Stephanie Jarmak. (2023)  
**Experimenting with Large Language Models and vector embeddings in NASA SciX**  

---
Primary Category: cs.CL  
Categories: astro-ph-IM, cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.14211v1)  

---


**ABSTRACT**  
Open-source Large Language Models enable projects such as NASA SciX (i.e., NASA ADS) to think out of the box and try alternative approaches for information retrieval and data augmentation, while respecting data copyright and users' privacy. However, when large language models are directly prompted with questions without any context, they are prone to hallucination. At NASA SciX we have developed an experiment where we created semantic vectors for our large collection of abstracts and full-text content, and we designed a prompt system to ask questions using contextual chunks from our system. Based on a non-systematic human evaluation, the experiment shows a lower degree of hallucination and better responses when using Retrieval Augmented Generation. Further exploration is required to design new features and data augmentation processes at NASA SciX that leverages this technology while respecting the high level of trust and quality that the project holds.

{{</citation>}}


### (12/116) Text2Analysis: A Benchmark of Table Question Answering with Advanced Data Analysis and Unclear Queries (Xinyi He et al., 2023)

{{<citation>}}

Xinyi He, Mengyu Zhou, Xinrun Xu, Xiaojun Ma, Rui Ding, Lun Du, Yan Gao, Ran Jia, Xu Chen, Shi Han, Zejian Yuan, Dongmei Zhang. (2023)  
**Text2Analysis: A Benchmark of Table Question Answering with Advanced Data Analysis and Unclear Queries**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.13671v1)  

---


**ABSTRACT**  
Tabular data analysis is crucial in various fields, and large language models show promise in this area. However, current research mostly focuses on rudimentary tasks like Text2SQL and TableQA, neglecting advanced analysis like forecasting and chart generation. To address this gap, we developed the Text2Analysis benchmark, incorporating advanced analysis tasks that go beyond the SQL-compatible operations and require more in-depth analysis. We also develop five innovative and effective annotation methods, harnessing the capabilities of large language models to enhance data quality and quantity. Additionally, we include unclear queries that resemble real-world user questions to test how well models can understand and tackle such challenges. Finally, we collect 2249 query-result pairs with 347 tables. We evaluate five state-of-the-art models using three different metrics and the results show that our benchmark presents introduces considerable challenge in the field of tabular data analysis, paving the way for more advanced research opportunities.

{{</citation>}}


### (13/116) Argue with Me Tersely: Towards Sentence-Level Counter-Argument Generation (Jiayu Lin et al., 2023)

{{<citation>}}

Jiayu Lin, Rong Ye, Meng Han, Qi Zhang, Ruofei Lai, Xinyu Zhang, Zhao Cao, Xuanjing Huang, Zhongyu Wei. (2023)  
**Argue with Me Tersely: Towards Sentence-Level Counter-Argument Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT  
[Paper Link](http://arxiv.org/abs/2312.13608v1)  

---


**ABSTRACT**  
Counter-argument generation -- a captivating area in computational linguistics -- seeks to craft statements that offer opposing views. While most research has ventured into paragraph-level generation, sentence-level counter-argument generation beckons with its unique constraints and brevity-focused challenges. Furthermore, the diverse nature of counter-arguments poses challenges for evaluating model performance solely based on n-gram-based metrics. In this paper, we present the ArgTersely benchmark for sentence-level counter-argument generation, drawing from a manually annotated dataset from the ChangeMyView debate forum. We also propose Arg-LlaMA for generating high-quality counter-argument. For better evaluation, we trained a BERT-based evaluator Arg-Judge with human preference data. We conducted comparative experiments involving various baselines such as LlaMA, Alpaca, GPT-3, and others. The results show the competitiveness of our proposed framework and evaluator in counter-argument generation tasks. Code and data are available at https://github.com/amazingljy1206/ArgTersely.

{{</citation>}}


### (14/116) Towards More Faithful Natural Language Explanation Using Multi-Level Contrastive Learning in VQA (Chengen Lai et al., 2023)

{{<citation>}}

Chengen Lai, Shengli Song, Shiqi Meng, Jingyang Li, Sitong Yan, Guangneng Hu. (2023)  
**Towards More Faithful Natural Language Explanation Using Multi-Level Contrastive Learning in VQA**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Contrastive Learning, QA  
[Paper Link](http://arxiv.org/abs/2312.13594v1)  

---


**ABSTRACT**  
Natural language explanation in visual question answer (VQA-NLE) aims to explain the decision-making process of models by generating natural language sentences to increase users' trust in the black-box systems. Existing post-hoc methods have achieved significant progress in obtaining a plausible explanation. However, such post-hoc explanations are not always aligned with human logical inference, suffering from the issues on: 1) Deductive unsatisfiability, the generated explanations do not logically lead to the answer; 2) Factual inconsistency, the model falsifies its counterfactual explanation for answers without considering the facts in images; and 3) Semantic perturbation insensitivity, the model can not recognize the semantic changes caused by small perturbations. These problems reduce the faithfulness of explanations generated by models. To address the above issues, we propose a novel self-supervised \textbf{M}ulti-level \textbf{C}ontrastive \textbf{L}earning based natural language \textbf{E}xplanation model (MCLE) for VQA with semantic-level, image-level, and instance-level factual and counterfactual samples. MCLE extracts discriminative features and aligns the feature spaces from explanations with visual question and answer to generate more consistent explanations. We conduct extensive experiments, ablation analysis, and case study to demonstrate the effectiveness of our method on two VQA-NLE benchmarks.

{{</citation>}}


### (15/116) Speech Translation with Large Language Models: An Industrial Practice (Zhichao Huang et al., 2023)

{{<citation>}}

Zhichao Huang, Rong Ye, Tom Ko, Qianqian Dong, Shanbo Cheng, Mingxuan Wang, Hang Li. (2023)  
**Speech Translation with Large Language Models: An Industrial Practice**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.13585v1)  

---


**ABSTRACT**  
Given the great success of large language models (LLMs) across various tasks, in this paper, we introduce LLM-ST, a novel and effective speech translation model constructed upon a pre-trained LLM. By integrating the large language model (LLM) with a speech encoder and employing multi-task instruction tuning, LLM-ST can produce accurate timestamped transcriptions and translations, even from long audio inputs. Furthermore, our findings indicate that the implementation of Chain-of-Thought (CoT) prompting can yield advantages in the context of LLM-ST. Through rigorous experimentation on English and Chinese datasets, we showcase the exceptional performance of LLM-ST, establishing a new benchmark in the field of speech translation. Demo: https://speechtranslation.github.io/llm-st/.

{{</citation>}}


### (16/116) Illuminating the Black Box: A Psychometric Investigation into the Multifaceted Nature of Large Language Models (Yang Lu et al., 2023)

{{<citation>}}

Yang Lu, Jordan Yu, Shou-Hsuan Stephen Huang. (2023)  
**Illuminating the Black Box: A Psychometric Investigation into the Multifaceted Nature of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.14202v1)  

---


**ABSTRACT**  
This study explores the idea of AI Personality or AInality suggesting that Large Language Models (LLMs) exhibit patterns similar to human personalities. Assuming that LLMs share these patterns with humans, we investigate using human-centered psychometric tests such as the Myers-Briggs Type Indicator (MBTI), Big Five Inventory (BFI), and Short Dark Triad (SD3) to identify and confirm LLM personality types. By introducing role-play prompts, we demonstrate the adaptability of LLMs, showing their ability to switch dynamically between different personality types. Using projective tests, such as the Washington University Sentence Completion Test (WUSCT), we uncover hidden aspects of LLM personalities that are not easily accessible through direct questioning. Projective tests allowed for a deep exploration of LLMs cognitive processes and thought patterns and gave us a multidimensional view of AInality. Our machine learning analysis revealed that LLMs exhibit distinct AInality traits and manifest diverse personality types, demonstrating dynamic shifts in response to external instructions. This study pioneers the application of projective tests on LLMs, shedding light on their diverse and adaptable AInality traits.

{{</citation>}}


### (17/116) How to Prune Your Language Model: Recovering Accuracy on the 'Sparsity May Cry'' Benchmark (Eldar Kurtic et al., 2023)

{{<citation>}}

Eldar Kurtic, Torsten Hoefler, Dan Alistarh. (2023)  
**How to Prune Your Language Model: Recovering Accuracy on the 'Sparsity May Cry'' Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Knowledge Distillation, Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2312.13547v1)  

---


**ABSTRACT**  
Pruning large language models (LLMs) from the BERT family has emerged as a standard compression benchmark, and several pruning methods have been proposed for this task. The recent ``Sparsity May Cry'' (SMC) benchmark put into question the validity of all existing methods, exhibiting a more complex setup where many known pruning methods appear to fail. We revisit the question of accurate BERT-pruning during fine-tuning on downstream datasets, and propose a set of general guidelines for successful pruning, even on the challenging SMC benchmark. First, we perform a cost-vs-benefits analysis of pruning model components, such as the embeddings and the classification head; second, we provide a simple-yet-general way of scaling training, sparsification and learning rate schedules relative to the desired target sparsity; finally, we investigate the importance of proper parametrization for Knowledge Distillation in the context of LLMs. Our simple insights lead to state-of-the-art results, both on classic BERT-pruning benchmarks, as well as on the SMC benchmark, showing that even classic gradual magnitude pruning (GMP) can yield competitive results, with the right approach.

{{</citation>}}


### (18/116) Developing Interactive Tourism Planning: A Dialogue Robot System Powered by a Large Language Model (Katsumasa Yoshikawa et al., 2023)

{{<citation>}}

Katsumasa Yoshikawa, Takato Yamazaki, Masaya Ohagi, Tomoya Mizumoto, Keiya Sato. (2023)  
**Developing Interactive Tourism Planning: A Dialogue Robot System Powered by a Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2312.13545v2)  

---


**ABSTRACT**  
In recent years, large language models (LLMs) have rapidly proliferated and have been utilized in various tasks, including research in dialogue systems. We aimed to construct a system that not only leverages the flexible conversational abilities of LLMs but also their advanced planning capabilities to reduce the speaking load on human interlocutors and efficiently plan trips. Furthermore, we propose a method that divides the complex task of a travel agency into multiple subtasks, managing each as a separate phase to effectively accomplish the task. Our proposed system confirmed a certain level of success by achieving fourth place in the Dialogue Robot Competition 2023 preliminaries rounds. We report on the challenges identified through the competition.

{{</citation>}}


### (19/116) Automated Clinical Coding for Outpatient Departments (Viktor Schlegel et al., 2023)

{{<citation>}}

Viktor Schlegel, Abhinav Ramesh Kashyap, Thanh-Tung Nguyen, Tsung-Han Yang, Vijay Prakash Dwivedi, Wei-Hsian Yin, Jeng Wei, Stefan Winkler. (2023)  
**Automated Clinical Coding for Outpatient Departments**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2312.13533v2)  

---


**ABSTRACT**  
Computerised clinical coding approaches aim to automate the process of assigning a set of codes to medical records. While there is active research pushing the state of the art on clinical coding for hospitalized patients, the outpatient setting -- where doctors tend to non-hospitalised patients -- is overlooked. Although both settings can be formalised as a multi-label classification task, they present unique and distinct challenges, which raises the question of whether the success of inpatient clinical coding approaches translates to the outpatient setting. This paper is the first to investigate how well state-of-the-art deep learning-based clinical coding approaches work in the outpatient setting at hospital scale. To this end, we collect a large outpatient dataset comprising over 7 million notes documenting over half a million patients. We adapt four state-of-the-art clinical coding approaches to this setting and evaluate their potential to assist coders. We find evidence that clinical coding in outpatient settings can benefit from more innovations in popular inpatient coding benchmarks. A deeper analysis of the factors contributing to the success -- amount and form of data and choice of document representation -- reveals the presence of easy-to-solve examples, the coding of which can be completely automated with a low error rate.

{{</citation>}}


### (20/116) Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models (Jingwei Yi et al., 2023)

{{<citation>}}

Jingwei Yi, Yueqi Xie, Bin Zhu, Keegan Hines, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu. (2023)  
**Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.14197v1)  

---


**ABSTRACT**  
Recent remarkable advancements in large language models (LLMs) have led to their widespread adoption in various applications. A key feature of these applications is the combination of LLMs with external content, where user instructions and third-party content are combined to create prompts for LLM processing. These applications, however, are vulnerable to indirect prompt injection attacks, where malicious instructions embedded within external content compromise LLM's output, causing their responses to deviate from user expectations. Despite the discovery of this security issue, no comprehensive analysis of indirect prompt injection attacks on different LLMs is available due to the lack of a benchmark. Furthermore, no effective defense has been proposed.   In this work, we introduce the first benchmark, BIPIA, to measure the robustness of various LLMs and defenses against indirect prompt injection attacks. Our experiments reveal that LLMs with greater capabilities exhibit more vulnerable to indirect prompt injection attacks for text tasks, resulting in a higher ASR. We hypothesize that indirect prompt injection attacks are mainly due to the LLMs' inability to distinguish between instructions and external content. Based on this conjecture, we propose four black-box methods based on prompt learning and a white-box defense methods based on fine-tuning with adversarial training to enable LLMs to distinguish between instructions and external content and ignore instructions in the external content. Our experimental results show that our black-box defense methods can effectively reduce ASR but cannot completely thwart indirect prompt injection attacks, while our white-box defense method can reduce ASR to nearly zero with little adverse impact on the LLM's performance on general tasks. We hope that our benchmark and defenses can inspire future work in this important area.

{{</citation>}}


### (21/116) Decoupling Representation and Knowledge for Few-Shot Intent Classification and Slot Filling (Jie Han et al., 2023)

{{<citation>}}

Jie Han, Yixiong Zou, Haozhao Wang, Jun Wang, Wei Liu, Yao Wu, Tao Zhang, Ruixuan Li. (2023)  
**Decoupling Representation and Knowledge for Few-Shot Intent Classification and Slot Filling**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.13495v1)  

---


**ABSTRACT**  
Few-shot intent classification and slot filling are important but challenging tasks due to the scarcity of finely labeled data. Therefore, current works first train a model on source domains with sufficiently labeled data, and then transfer the model to target domains where only rarely labeled data is available. However, experience transferring as a whole usually suffers from gaps that exist among source domains and target domains. For instance, transferring domain-specific-knowledge-related experience is difficult. To tackle this problem, we propose a new method that explicitly decouples the transferring of general-semantic-representation-related experience and the domain-specific-knowledge-related experience. Specifically, for domain-specific-knowledge-related experience, we design two modules to capture intent-slot relation and slot-slot relation respectively. Extensive experiments on Snips and FewJoint datasets show that our method achieves state-of-the-art performance. The method improves the joint accuracy metric from 27.72% to 42.20% in the 1-shot setting, and from 46.54% to 60.79% in the 5-shot setting.

{{</citation>}}


## cs.LG (19)



### (22/116) DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction) (Qiaoyue Tang et al., 2023)

{{<citation>}}

Qiaoyue Tang, Frederick Shpilevskiy, Mathias Lécuyer. (2023)  
**DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction)**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.14334v1)  

---


**ABSTRACT**  
The Adam optimizer is a popular choice in contemporary deep learning, due to its strong empirical performance. However we observe that in privacy sensitive scenarios, the traditional use of Differential Privacy (DP) with the Adam optimizer leads to sub-optimal performance on several tasks. We find that this performance degradation is due to a DP bias in Adam's second moment estimator, introduced by the addition of independent noise in the gradient computation to enforce DP guarantees. This DP bias leads to a different scaling for low variance parameter updates, that is inconsistent with the behavior of non-private Adam. We propose DP-AdamBC, an optimization algorithm which removes the bias in the second moment estimation and retrieves the expected behaviour of Adam. Empirically, DP-AdamBC significantly improves the optimization performance of DP-Adam by up to 3.5% in final accuracy in image, text, and graph node classification tasks.

{{</citation>}}


### (23/116) Invariant Anomaly Detection under Distribution Shifts: A Causal Perspective (João B. S. Carvalho et al., 2023)

{{<citation>}}

João B. S. Carvalho, Mengtao Zhang, Robin Geyer, Carlos Cotrini, Joachim M. Buhmann. (2023)  
**Invariant Anomaly Detection under Distribution Shifts: A Causal Perspective**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.14329v1)  

---


**ABSTRACT**  
Anomaly detection (AD) is the machine learning task of identifying highly discrepant abnormal samples by solely relying on the consistency of the normal training samples. Under the constraints of a distribution shift, the assumption that training samples and test samples are drawn from the same distribution breaks down. In this work, by leveraging tools from causal inference we attempt to increase the resilience of anomaly detection models to different kinds of distribution shifts. We begin by elucidating a simple yet necessary statistical property that ensures invariant representations, which is critical for robust AD under both domain and covariate shifts. From this property, we derive a regularization term which, when minimized, leads to partial distribution invariance across environments. Through extensive experimental evaluation on both synthetic and real-world tasks, covering a range of six different AD methods, we demonstrated significant improvements in out-of-distribution performance. Under both covariate and domain shift, models regularized with our proposed term showed marked increased robustness. Code is available at: https://github.com/JoaoCarv/invariant-anomaly-detection.

{{</citation>}}


### (24/116) Federated Quantum Long Short-term Memory (FedQLSTM) (Mahdi Chehimi et al., 2023)

{{<citation>}}

Mahdi Chehimi, Samuel Yen-Chi Chen, Walid Saad, Shinjae Yoo. (2023)  
**Federated Quantum Long Short-term Memory (FedQLSTM)**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG, quant-ph  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.14309v1)  

---


**ABSTRACT**  
Quantum federated learning (QFL) can facilitate collaborative learning across multiple clients using quantum machine learning (QML) models, while preserving data privacy. Although recent advances in QFL span different tasks like classification while leveraging several data types, no prior work has focused on developing a QFL framework that utilizes temporal data to approximate functions useful to analyze the performance of distributed quantum sensing networks. In this paper, a novel QFL framework that is the first to integrate quantum long short-term memory (QLSTM) models with temporal data is proposed. The proposed federated QLSTM (FedQLSTM) framework is exploited for performing the task of function approximation. In this regard, three key use cases are presented: Bessel function approximation, sinusoidal delayed quantum feedback control function approximation, and Struve function approximation. Simulation results confirm that, for all considered use cases, the proposed FedQLSTM framework achieves a faster convergence rate under one local training epoch, minimizing the overall computations, and saving 25-33% of the number of communication rounds needed until convergence compared to an FL framework with classical LSTM models.

{{</citation>}}


### (25/116) Diffusion Models for Generative Artificial Intelligence: An Introduction for Applied Mathematicians (Catherine F. Higham et al., 2023)

{{<citation>}}

Catherine F. Higham, Desmond J. Higham, Peter Grindrod. (2023)  
**Diffusion Models for Generative Artificial Intelligence: An Introduction for Applied Mathematicians**  

---
Primary Category: cs.LG  
Categories: 68T07, 60J60, I-2; I-2-6, cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14977v1)  

---


**ABSTRACT**  
Generative artificial intelligence (AI) refers to algorithms that create synthetic but realistic output. Diffusion models currently offer state of the art performance in generative AI for images. They also form a key component in more general tools, including text-to-image generators and large language models. Diffusion models work by adding noise to the available training data and then learning how to reverse the process. The reverse operation may then be applied to new random data in order to produce new outputs. We provide a brief introduction to diffusion models for applied mathematicians and statisticians. Our key aims are (a) to present illustrative computational examples, (b) to give a careful derivation of the underlying mathematical formulas involved, and (c) to draw a connection with partial differential equation (PDE) diffusion models. We provide code for the computational experiments. We hope that this topic will be of interest to advanced undergraduate students and postgraduate students. Portions of the material may also provide useful motivational examples for those who teach courses in stochastic processes, inference, machine learning, PDEs or scientific computing.

{{</citation>}}


### (26/116) Elevating Defenses: Bridging Adversarial Training and Watermarking for Model Resilience (Janvi Thakkar et al., 2023)

{{<citation>}}

Janvi Thakkar, Giulio Zizzo, Sergio Maffeis. (2023)  
**Elevating Defenses: Bridging Adversarial Training and Watermarking for Model Resilience**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2312.14260v1)  

---


**ABSTRACT**  
Machine learning models are being used in an increasing number of critical applications; thus, securing their integrity and ownership is critical. Recent studies observed that adversarial training and watermarking have a conflicting interaction. This work introduces a novel framework to integrate adversarial training with watermarking techniques to fortify against evasion attacks and provide confident model verification in case of intellectual property theft. We use adversarial training together with adversarial watermarks to train a robust watermarked model. The key intuition is to use a higher perturbation budget to generate adversarial watermarks compared to the budget used for adversarial training, thus avoiding conflict. We use the MNIST and Fashion-MNIST datasets to evaluate our proposed technique on various model stealing attacks. The results obtained consistently outperform the existing baseline in terms of robustness performance and further prove the resilience of this defense against pruning and fine-tuning removal attacks.

{{</citation>}}


### (27/116) WellFactor: Patient Profiling using Integrative Embedding of Healthcare Data (Dongjin Choi et al., 2023)

{{<citation>}}

Dongjin Choi, Andy Xiang, Ozgur Ozturk, Deep Shrestha, Barry Drake, Hamid Haidarian, Faizan Javed, Haesun Park. (2023)  
**WellFactor: Patient Profiling using Integrative Embedding of Healthcare Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.14129v1)  

---


**ABSTRACT**  
In the rapidly evolving healthcare industry, platforms now have access to not only traditional medical records, but also diverse data sets encompassing various patient interactions, such as those from healthcare web portals. To address this rich diversity of data, we introduce WellFactor: a method that derives patient profiles by integrating information from these sources. Central to our approach is the utilization of constrained low-rank approximation. WellFactor is optimized to handle the sparsity that is often inherent in healthcare data. Moreover, by incorporating task-specific label information, our method refines the embedding results, offering a more informed perspective on patients. One important feature of WellFactor is its ability to compute embeddings for new, previously unobserved patient data instantaneously, eliminating the need to revisit the entire data set or recomputing the embedding. Comprehensive evaluations on real-world healthcare data demonstrate WellFactor's effectiveness. It produces better results compared to other existing methods in classification performance, yields meaningful clustering of patients, and delivers consistent results in patient similarity searches and predictions.

{{</citation>}}


### (28/116) Real-time Neural Network Inference on Extremely Weak Devices: Agile Offloading with Explainable AI (Kai Huang et al., 2023)

{{<citation>}}

Kai Huang, Wei Gao. (2023)  
**Real-time Neural Network Inference on Extremely Weak Devices: Agile Offloading with Explainable AI**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14229v1)  

---


**ABSTRACT**  
With the wide adoption of AI applications, there is a pressing need of enabling real-time neural network (NN) inference on small embedded devices, but deploying NNs and achieving high performance of NN inference on these small devices is challenging due to their extremely weak capabilities. Although NN partitioning and offloading can contribute to such deployment, they are incapable of minimizing the local costs at embedded devices. Instead, we suggest to address this challenge via agile NN offloading, which migrates the required computations in NN offloading from online inference to offline learning. In this paper, we present AgileNN, a new NN offloading technique that achieves real-time NN inference on weak embedded devices by leveraging eXplainable AI techniques, so as to explicitly enforce feature sparsity during the training phase and minimize the online computation and communication costs. Experiment results show that AgileNN's inference latency is >6x lower than the existing schemes, ensuring that sensory data on embedded devices can be timely consumed. It also reduces the local device's resource consumption by >8x, without impairing the inference accuracy.

{{</citation>}}


### (29/116) On Partial Optimal Transport: Revising the Infeasibility of Sinkhorn and Efficient Gradient Methods (Anh Duc Nguyen et al., 2023)

{{<citation>}}

Anh Duc Nguyen, Tuan Dung Nguyen, Quang Minh Nguyen, Hoang H. Nguyen, Lam M. Nguyen, Kim-Chuan Toh. (2023)  
**On Partial Optimal Transport: Revising the Infeasibility of Sinkhorn and Efficient Gradient Methods**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-OC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.13970v2)  

---


**ABSTRACT**  
This paper studies the Partial Optimal Transport (POT) problem between two unbalanced measures with at most $n$ supports and its applications in various AI tasks such as color transfer or domain adaptation. There is hence the need for fast approximations of POT with increasingly large problem sizes in arising applications. We first theoretically and experimentally investigate the infeasibility of the state-of-the-art Sinkhorn algorithm for POT due to its incompatible rounding procedure, which consequently degrades its qualitative performance in real world applications like point-cloud registration. To this end, we propose a novel rounding algorithm for POT, and then provide a feasible Sinkhorn procedure with a revised computation complexity of $\mathcal{\widetilde O}(n^2/\varepsilon^4)$. Our rounding algorithm also permits the development of two first-order methods to approximate the POT problem. The first algorithm, Adaptive Primal-Dual Accelerated Gradient Descent (APDAGD), finds an $\varepsilon$-approximate solution to the POT problem in $\mathcal{\widetilde O}(n^{2.5}/\varepsilon)$, which is better in $\varepsilon$ than revised Sinkhorn. The second method, Dual Extrapolation, achieves the computation complexity of $\mathcal{\widetilde O}(n^2/\varepsilon)$, thereby being the best in the literature. We further demonstrate the flexibility of POT compared to standard OT as well as the practicality of our algorithms on real applications where two marginal distributions are unbalanced.

{{</citation>}}


### (30/116) Comparative Evaluation of Anomaly Detection Methods for Fraud Detection in Online Credit Card Payments (Hugo Thimonier et al., 2023)

{{<citation>}}

Hugo Thimonier, Fabrice Popineau, Arpad Rimmel, Bich-Liên Doan, Fabrice Daniel. (2023)  
**Comparative Evaluation of Anomaly Detection Methods for Fraud Detection in Online Credit Card Payments**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-fin-ST  
Keywords: Anomaly Detection, Fraud Detection  
[Paper Link](http://arxiv.org/abs/2312.13896v1)  

---


**ABSTRACT**  
This study explores the application of anomaly detection (AD) methods in imbalanced learning tasks, focusing on fraud detection using real online credit card payment data. We assess the performance of several recent AD methods and compare their effectiveness against standard supervised learning methods. Offering evidence of distribution shift within our dataset, we analyze its impact on the tested models' performances. Our findings reveal that LightGBM exhibits significantly superior performance across all evaluated metrics but suffers more from distribution shifts than AD methods. Furthermore, our investigation reveals that LightGBM also captures the majority of frauds detected by AD methods. This observation challenges the potential benefits of ensemble methods to combine supervised, and AD approaches to enhance performance. In summary, this research provides practical insights into the utility of these techniques in real-world scenarios, showing LightGBM's superiority in fraud detection while highlighting challenges related to distribution shifts.

{{</citation>}}


### (31/116) Capture the Flag: Uncovering Data Insights with Large Language Models (Issam Laradji et al., 2023)

{{<citation>}}

Issam Laradji, Perouz Taslakian, Sai Rajeswar, Valentina Zantedeschi, Alexandre Lacoste, Nicolas Chapados, David Vazquez, Christopher Pal, Alexandre Drouin. (2023)  
**Capture the Flag: Uncovering Data Insights with Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.13876v1)  

---


**ABSTRACT**  
The extraction of a small number of relevant insights from vast amounts of data is a crucial component of data-driven decision-making. However, accomplishing this task requires considerable technical skills, domain expertise, and human labor. This study explores the potential of using Large Language Models (LLMs) to automate the discovery of insights in data, leveraging recent advances in reasoning and code generation techniques. We propose a new evaluation methodology based on a "capture the flag" principle, measuring the ability of such models to recognize meaningful and pertinent information (flags) in a dataset. We further propose two proof-of-concept agents, with different inner workings, and compare their ability to capture such flags in a real-world sales dataset. While the work reported here is preliminary, our results are sufficiently interesting to mandate future exploration by the community.

{{</citation>}}


### (32/116) Hierarchical Topology Isomorphism Expertise Embedded Graph Contrastive Learning (Jiangmeng Li et al., 2023)

{{<citation>}}

Jiangmeng Li, Yifan Jin, Hang Gao, Wenwen Qiang, Changwen Zheng, Fuchun Sun. (2023)  
**Hierarchical Topology Isomorphism Expertise Embedded Graph Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.14222v2)  

---


**ABSTRACT**  
Graph contrastive learning (GCL) aims to align the positive features while differentiating the negative features in the latent space by minimizing a pair-wise contrastive loss. As the embodiment of an outstanding discriminative unsupervised graph representation learning approach, GCL achieves impressive successes in various graph benchmarks. However, such an approach falls short of recognizing the topology isomorphism of graphs, resulting in that graphs with relatively homogeneous node features cannot be sufficiently discriminated. By revisiting classic graph topology recognition works, we disclose that the corresponding expertise intuitively complements GCL methods. To this end, we propose a novel hierarchical topology isomorphism expertise embedded graph contrastive learning, which introduces knowledge distillations to empower GCL models to learn the hierarchical topology isomorphism expertise, including the graph-tier and subgraph-tier. On top of this, the proposed method holds the feature of plug-and-play, and we empirically demonstrate that the proposed method is universal to multiple state-of-the-art GCL models. The solid theoretical analyses are further provided to prove that compared with conventional GCL methods, our method acquires the tighter upper bound of Bayes classification error. We conduct extensive experiments on real-world benchmarks to exhibit the performance superiority of our method over candidate GCL methods, e.g., for the real-world graph representation learning experiments, the proposed method beats the state-of-the-art method by 0.23% on unsupervised representation learning setting, 0.43% on transfer learning setting. Our code is available at https://github.com/jyf123/HTML.

{{</citation>}}


### (33/116) Sparse Training for Federated Learning with Regularized Error Correction (Ran Greidi et al., 2023)

{{<citation>}}

Ran Greidi, Kobi Cohen. (2023)  
**Sparse Training for Federated Learning with Regularized Error Correction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.13795v1)  

---


**ABSTRACT**  
Federated Learning (FL) has attracted much interest due to the significant advantages it brings to training deep neural network (DNN) models. However, since communications and computation resources are limited, training DNN models in FL systems face challenges such as elevated computational and communication costs in complex tasks. Sparse training schemes gain increasing attention in order to scale down the dimensionality of each client (i.e., node) transmission. Specifically, sparsification with error correction methods is a promising technique, where only important updates are sent to the parameter server (PS) and the rest are accumulated locally. While error correction methods have shown to achieve a significant sparsification level of the client-to-PS message without harming convergence, pushing sparsity further remains unresolved due to the staleness effect. In this paper, we propose a novel algorithm, dubbed Federated Learning with Accumulated Regularized Embeddings (FLARE), to overcome this challenge. FLARE presents a novel sparse training approach via accumulated pulling of the updated models with regularization on the embeddings in the FL process, providing a powerful solution to the staleness effect, and pushing sparsity to an exceptional level. The performance of FLARE is validated through extensive experiments on diverse and complex models, achieving a remarkable sparsity level (10 times and more beyond the current state-of-the-art) along with significantly improved accuracy. Additionally, an open-source software package has been developed for the benefit of researchers and developers in related fields.

{{</citation>}}


### (34/116) Critic-Guided Decision Transformer for Offline Reinforcement Learning (Yuanfu Wang et al., 2023)

{{<citation>}}

Yuanfu Wang, Chao Yang, Ying Wen, Yu Liu, Yu Qiao. (2023)  
**Critic-Guided Decision Transformer for Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.13716v1)  

---


**ABSTRACT**  
Recent advancements in offline reinforcement learning (RL) have underscored the capabilities of Return-Conditioned Supervised Learning (RCSL), a paradigm that learns the action distribution based on target returns for each state in a supervised manner. However, prevailing RCSL methods largely focus on deterministic trajectory modeling, disregarding stochastic state transitions and the diversity of future trajectory distributions. A fundamental challenge arises from the inconsistency between the sampled returns within individual trajectories and the expected returns across multiple trajectories. Fortunately, value-based methods offer a solution by leveraging a value function to approximate the expected returns, thereby addressing the inconsistency effectively. Building upon these insights, we propose a novel approach, termed the Critic-Guided Decision Transformer (CGDT), which combines the predictability of long-term returns from value-based methods with the trajectory modeling capability of the Decision Transformer. By incorporating a learned value function, known as the critic, CGDT ensures a direct alignment between the specified target returns and the expected returns of actions. This integration bridges the gap between the deterministic nature of RCSL and the probabilistic characteristics of value-based methods. Empirical evaluations on stochastic environments and D4RL benchmark datasets demonstrate the superiority of CGDT over traditional RCSL methods. These results highlight the potential of CGDT to advance the state of the art in offline RL and extend the applicability of RCSL to a wide range of RL tasks.

{{</citation>}}


### (35/116) Anchoring Path for Inductive Relation Prediction in Knowledge Graphs (Zhixiang Su et al., 2023)

{{<citation>}}

Zhixiang Su, Di Wang, Chunyan Miao, Lizhen Cui. (2023)  
**Anchoring Path for Inductive Relation Prediction in Knowledge Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Knowledge Graph, Transformer  
[Paper Link](http://arxiv.org/abs/2312.13596v1)  

---


**ABSTRACT**  
Aiming to accurately predict missing edges representing relations between entities, which are pervasive in real-world Knowledge Graphs (KGs), relation prediction plays a critical role in enhancing the comprehensiveness and utility of KGs. Recent research focuses on path-based methods due to their inductive and explainable properties. However, these methods face a great challenge when lots of reasoning paths do not form Closed Paths (CPs) in the KG. To address this challenge, we propose Anchoring Path Sentence Transformer (APST) by introducing Anchoring Paths (APs) to alleviate the reliance of CPs. Specifically, we develop a search-based description retrieval method to enrich entity descriptions and an assessment mechanism to evaluate the rationality of APs. APST takes both APs and CPs as the inputs of a unified Sentence Transformer architecture, enabling comprehensive predictions and high-quality explanations. We evaluate APST on three public datasets and achieve state-of-the-art (SOTA) performance in 30 of 36 transductive, inductive, and few-shot experimental settings.

{{</citation>}}


### (36/116) Fine-tuning Graph Neural Networks by Preserving Graph Generative Patterns (Yifei Sun et al., 2023)

{{<citation>}}

Yifei Sun, Qi Zhu, Yang Yang, Chunping Wang, Tianyu Fan, Jiajun Zhu, Lei Chen. (2023)  
**Fine-tuning Graph Neural Networks by Preserving Graph Generative Patterns**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.13583v1)  

---


**ABSTRACT**  
Recently, the paradigm of pre-training and fine-tuning graph neural networks has been intensively studied and applied in a wide range of graph mining tasks. Its success is generally attributed to the structural consistency between pre-training and downstream datasets, which, however, does not hold in many real-world scenarios. Existing works have shown that the structural divergence between pre-training and downstream graphs significantly limits the transferability when using the vanilla fine-tuning strategy. This divergence leads to model overfitting on pre-training graphs and causes difficulties in capturing the structural properties of the downstream graphs. In this paper, we identify the fundamental cause of structural divergence as the discrepancy of generative patterns between the pre-training and downstream graphs. Furthermore, we propose G-Tuning to preserve the generative patterns of downstream graphs. Given a downstream graph G, the core idea is to tune the pre-trained GNN so that it can reconstruct the generative patterns of G, the graphon W. However, the exact reconstruction of a graphon is known to be computationally expensive. To overcome this challenge, we provide a theoretical analysis that establishes the existence of a set of alternative graphons called graphon bases for any given graphon. By utilizing a linear combination of these graphon bases, we can efficiently approximate W. This theoretical finding forms the basis of our proposed model, as it enables effective learning of the graphon bases and their associated coefficients. Compared with existing algorithms, G-Tuning demonstrates an average improvement of 0.5% and 2.6% on in-domain and out-of-domain transfer learning experiments, respectively.

{{</citation>}}


### (37/116) The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction (Pratyusha Sharma et al., 2023)

{{<citation>}}

Pratyusha Sharma, Jordan T. Ash, Dipendra Misra. (2023)  
**The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: Language Model, Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.13558v1)  

---


**ABSTRACT**  
Transformer-based Large Language Models (LLMs) have become a fixture in modern machine learning. Correspondingly, significant resources are allocated towards research that aims to further advance this technology, typically resulting in models of increasing size that are trained on increasing amounts of data. This work, however, demonstrates the surprising result that it is often possible to significantly improve the performance of LLMs by selectively removing higher-order components of their weight matrices. This simple intervention, which we call LAyer-SElective Rank reduction (LASER), can be done on a model after training has completed, and requires no additional parameters or data. We show extensive experiments demonstrating the generality of this finding across language models and datasets, and provide in-depth analyses offering insights into both when LASER is effective and the mechanism by which it operates.

{{</citation>}}


### (38/116) CR-SAM: Curvature Regularized Sharpness-Aware Minimization (Tao Wu et al., 2023)

{{<citation>}}

Tao Wu, Tie Luo, Donald C. Wunsch. (2023)  
**CR-SAM: Curvature Regularized Sharpness-Aware Minimization**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI, ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2312.13555v2)  

---


**ABSTRACT**  
The capacity to generalize to future unseen data stands as one of the utmost crucial attributes of deep neural networks. Sharpness-Aware Minimization (SAM) aims to enhance the generalizability by minimizing worst-case loss using one-step gradient ascent as an approximation. However, as training progresses, the non-linearity of the loss landscape increases, rendering one-step gradient ascent less effective. On the other hand, multi-step gradient ascent will incur higher training cost. In this paper, we introduce a normalized Hessian trace to accurately measure the curvature of loss landscape on {\em both} training and test sets. In particular, to counter excessive non-linearity of loss landscape, we propose Curvature Regularized SAM (CR-SAM), integrating the normalized Hessian trace as a SAM regularizer. Additionally, we present an efficient way to compute the trace via finite differences with parallelism. Our theoretical analysis based on PAC-Bayes bounds establishes the regularizer's efficacy in reducing generalization error. Empirical evaluation on CIFAR and ImageNet datasets shows that CR-SAM consistently enhances classification performance for ResNet and Vision Transformer (ViT) models across various datasets. Our code is available at https://github.com/TrustAIoT/CR-SAM.

{{</citation>}}


### (39/116) Domain Adaptive Graph Classification (Siyang Luo et al., 2023)

{{<citation>}}

Siyang Luo, Ziyi Jiang, Zhenghan Chen, Xiaoxuan Liang. (2023)  
**Domain Adaptive Graph Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.13536v1)  

---


**ABSTRACT**  
Despite the remarkable accomplishments of graph neural networks (GNNs), they typically rely on task-specific labels, posing potential challenges in terms of their acquisition. Existing work have been made to address this issue through the lens of unsupervised domain adaptation, wherein labeled source graphs are utilized to enhance the learning process for target data. However, the simultaneous exploration of graph topology and reduction of domain disparities remains a substantial hurdle. In this paper, we introduce the Dual Adversarial Graph Representation Learning (DAGRL), which explore the graph topology from dual branches and mitigate domain discrepancies via dual adversarial learning. Our method encompasses a dual-pronged structure, consisting of a graph convolutional network branch and a graph kernel branch, which enables us to capture graph semantics from both implicit and explicit perspectives. Moreover, our approach incorporates adaptive perturbations into the dual branches, which align the source and target distribution to address domain discrepancies. Extensive experiments on a wild range graph classification datasets demonstrate the effectiveness of our proposed method.

{{</citation>}}


### (40/116) Optimizing Heat Alert Issuance for Public Health in the United States with Reinforcement Learning (Ellen M. Considine et al., 2023)

{{<citation>}}

Ellen M. Considine, Rachel C. Nethery, Gregory A. Wellenius, Francesca Dominici, Mauricio Tec. (2023)  
**Optimizing Heat Alert Issuance for Public Health in the United States with Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14196v1)  

---


**ABSTRACT**  
Alerting the public when heat may harm their health is a crucial service, especially considering that extreme heat events will be more frequent under climate change. Current practice for issuing heat alerts in the US does not take advantage of modern data science methods for optimizing local alert criteria. Specifically, application of reinforcement learning (RL) has the potential to inform more health-protective policies, accounting for regional and sociodemographic heterogeneity as well as sequential dependence of alerts. In this work, we formulate the issuance of heat alerts as a sequential decision making problem and develop modifications to the RL workflow to address challenges commonly encountered in environmental health settings. Key modifications include creating a simulator that pairs hierarchical Bayesian modeling of low-signal health effects with sampling of real weather trajectories (exogenous features), constraining the total number of alerts issued as well as preventing alerts on less-hot days, and optimizing location-specific policies. Post-hoc contrastive analysis offers insights into scenarios when using RL for heat alert issuance may protect public health better than the current or alternative policies. This work contributes to a broader movement of advancing data-driven policy optimization for public health and climate change adaptation.

{{</citation>}}


## cs.MA (1)



### (41/116) Behaviour Modelling of Social Animals via Causal Structure Discovery and Graph Neural Networks (Gaël Gendron et al., 2023)

{{<citation>}}

Gaël Gendron, Yang Chen, Mitchell Rogers, Yiping Liu, Mihailo Azhar, Shahrokh Heidari, David Arturo Soriano Valdez, Kobe Knowles, Padriac O'Leary, Simon Eyre, Michael Witbrock, Gillian Dobbie, Jiamou Liu, Patrice Delmas. (2023)  
**Behaviour Modelling of Social Animals via Causal Structure Discovery and Graph Neural Networks**  

---
Primary Category: cs.MA  
Categories: I-2-6; I-5-1; I-6-3; J-4, cs-LG, cs-MA, cs.MA, stat-ME  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.14333v1)  

---


**ABSTRACT**  
Better understanding the natural world is a crucial task with a wide range of applications. In environments with close proximity between humans and animals, such as zoos, it is essential to better understand the causes behind animal behaviour and what interventions are responsible for changes in their behaviours. This can help to predict unusual behaviours, mitigate detrimental effects and increase the well-being of animals. There has been work on modelling the dynamics behind swarms of birds and insects but the complex social behaviours of mammalian groups remain less explored. In this work, we propose a method to build behavioural models using causal structure discovery and graph neural networks for time series. We apply this method to a mob of meerkats in a zoo environment and study its ability to predict future actions and model the behaviour distribution at an individual-level and at a group level. We show that our method can match and outperform standard deep learning architectures and generate more realistic data, while using fewer parameters and providing increased interpretability.

{{</citation>}}


## cs.SI (2)



### (42/116) Social Recommendation through Heterogeneous Graph Modeling of the Long-term and Short-term Preference Defined by Dynamic Periods (Behafarid Mohammad Jafari et al., 2023)

{{<citation>}}

Behafarid Mohammad Jafari, Xiao Luo, Ali Jafari. (2023)  
**Social Recommendation through Heterogeneous Graph Modeling of the Long-term and Short-term Preference Defined by Dynamic Periods**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-CE, cs-SI, cs.SI  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.14306v1)  

---


**ABSTRACT**  
Social recommendations have been widely adopted in substantial domains. Recently, graph neural networks (GNN) have been employed in recommender systems due to their success in graph representation learning. However, dealing with the dynamic property of social network data is a challenge. This research presents a novel method that provides social recommendations by incorporating the dynamic property of social network data in a heterogeneous graph. The model aims to capture user preference over time without going through the complexities of a dynamic graph by adding period nodes to define users' long-term and short-term preferences and aggregating assigned edge weights. The model is applied to real-world data to argue its superior performance. Promising results demonstrate the effectiveness of this model.

{{</citation>}}


### (43/116) Designing Artificial Intelligence Equipped Social Decentralized Autonomous Organizations for Tackling Sextortion Cases Version 0.7 (Norta Alex et al., 2023)

{{<citation>}}

Norta Alex, Makrygiannis Sotiris. (2023)  
**Designing Artificial Intelligence Equipped Social Decentralized Autonomous Organizations for Tackling Sextortion Cases Version 0.7**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14090v1)  

---


**ABSTRACT**  
With the rapid diffusion of social networks in combination with mobile phones, a new social threat of sextortion has emerged, in which vulnerable young women are essentially blackmailed with their explicit shared multimedia content. The phenomenon of sextortion is now widely studied by psychologists, sociologists, criminologists, etc. The findings have been translated into scattered help from NGOs, specialized law enforcement units, and therapists, who usually do not coordinate their efforts among each other. This paper addresses the gap of lacking coordination systems to effectively and efficiently use modern information technologies that align the efforts of scattered and non-aligned sextortion help organizations. Consequently, this paper not only investigates the goals, incentives, and disincentives for a system design and development that not only governs effectively and efficiently diverse cases of sextortion victims, but also leverages artificial intelligence in a targeted manner. It explores how AI and, in particular, autonomous cognitive entities can improve victim profiles analysis, streamline support mechanisms, and provide intelligent insight into sextortion cases. Furthermore, the paper conceptually studies the extent to which such efforts can be monetized in a sustainable way. Following a novel design methodology for the design of trusted blockchain decentralized applications, the paper presents a set of conceptual requirements and system models based on which it is possible to deduce a best-practice technology stack for rapid implementation deployment.

{{</citation>}}


## cs.CR (5)



### (44/116) Exploiting Novel GPT-4 APIs (Kellin Pelrine et al., 2023)

{{<citation>}}

Kellin Pelrine, Mohammad Taufeeque, Michał Zając, Euan McLean, Adam Gleave. (2023)  
**Exploiting Novel GPT-4 APIs**  

---
Primary Category: cs.CR  
Categories: I-2-7, cs-AI, cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.14302v1)  

---


**ABSTRACT**  
Language model attacks typically assume one of two extreme threat models: full white-box access to model weights, or black-box access limited to a text generation API. However, real-world APIs are often more flexible than just text generation: these APIs expose ``gray-box'' access leading to new threat vectors. To explore this, we red-team three new functionalities exposed in the GPT-4 APIs: fine-tuning, function calling and knowledge retrieval. We find that fine-tuning a model on as few as 15 harmful examples or 100 benign examples can remove core safeguards from GPT-4, enabling a range of harmful outputs. Furthermore, we find that GPT-4 Assistants readily divulge the function call schema and can be made to execute arbitrary function calls. Finally, we find that knowledge retrieval can be hijacked by injecting instructions into retrieval documents. These vulnerabilities highlight that any additions to the functionality exposed by an API can create new vulnerabilities.

{{</citation>}}


### (45/116) Benchmark Evaluation of Anomaly-Based Intrusion Detection Systems in the Context of Smart Grids (Ömer Sen et al., 2023)

{{<citation>}}

Ömer Sen, Simon Glomb, Martin Henze, Andreas Ulbig. (2023)  
**Benchmark Evaluation of Anomaly-Based Intrusion Detection Systems in the Context of Smart Grids**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SY, cs.CR, eess-SY  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2312.13705v1)  

---


**ABSTRACT**  
The increasing digitization of smart grids has made addressing cybersecurity issues crucial in order to secure the power supply. Anomaly detection has emerged as a key technology for cybersecurity in smart grids, enabling the detection of unknown threats. Many research efforts have proposed various machine-learning-based approaches for anomaly detection in grid operations. However, there is a need for a reproducible and comprehensive evaluation environment to investigate and compare different approaches to anomaly detection. The assessment process is highly dependent on the specific application and requires an evaluation that considers representative datasets from the use case as well as the specific characteristics of the use case. In this work, we present an evaluation environment for anomaly detection methods in smart grids that facilitates reproducible and comprehensive evaluation of different anomaly detection methods.

{{</citation>}}


### (46/116) A Forecasting-Based DLP Approach for Data Security (Kishu Gupta et al., 2023)

{{<citation>}}

Kishu Gupta, Ashwani Kush. (2023)  
**A Forecasting-Based DLP Approach for Data Security**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.13704v1)  

---


**ABSTRACT**  
Sensitive data leakage is the major growing problem being faced by enterprises in this technical era. Data leakage causes severe threats for organization of data safety which badly affects the reputation of organizations. Data leakage is the flow of sensitive data/information from any data holder to an unauthorized destination. Data leak prevention (DLP) is set of techniques that try to alleviate the threats which may hinder data security. DLP unveils guilty user responsible for data leakage and ensures that user without appropriate permission cannot access sensitive data and also provides protection to sensitive data if sensitive data is shared accidentally. In this paper, data leakage prevention (DLP) model is used to restrict/grant data access permission to user, based on the forecast of their access to data. This study provides a DLP solution using data statistical analysis to forecast the data access possibilities of any user in future based on the access to data in the past. The proposed approach makes use of renowned simple piecewise linear function for learning/training to model. The results show that the proposed DLP approach with high level of precision can correctly classify between users even in cases of extreme data access.

{{</citation>}}


### (47/116) HW-V2W-Map: Hardware Vulnerability to Weakness Mapping Framework for Root Cause Analysis with GPT-assisted Mitigation Suggestion (Yu-Zheng Lin et al., 2023)

{{<citation>}}

Yu-Zheng Lin, Muntasir Mamun, Muhtasim Alam Chowdhury, Shuyu Cai, Mingyu Zhu, Banafsheh Saber Latibari, Kevin Immanuel Gubbi, Najmeh Nazari Bavarsad, Arjun Caputo, Avesta Sasan, Houman Homayoun, Setareh Rafatirad, Pratik Satam, Soheil Salehi. (2023)  
**HW-V2W-Map: Hardware Vulnerability to Weakness Mapping Framework for Root Cause Analysis with GPT-assisted Mitigation Suggestion**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR  
Keywords: GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.13530v1)  

---


**ABSTRACT**  
The escalating complexity of modern computing frameworks has resulted in a surge in the cybersecurity vulnerabilities reported to the National Vulnerability Database (NVD) by practitioners. Despite the fact that the stature of NVD is one of the most significant databases for the latest insights into vulnerabilities, extracting meaningful trends from such a large amount of unstructured data is still challenging without the application of suitable technological methodologies. Previous efforts have mostly concentrated on software vulnerabilities; however, a holistic strategy incorporates approaches for mitigating vulnerabilities, score prediction, and a knowledge-generating system that may extract relevant insights from the Common Weakness Enumeration (CWE) and Common Vulnerability Exchange (CVE) databases is notably absent. As the number of hardware attacks on Internet of Things (IoT) devices continues to rapidly increase, we present the Hardware Vulnerability to Weakness Mapping (HW-V2W-Map) Framework, which is a Machine Learning (ML) framework focusing on hardware vulnerabilities and IoT security. The architecture that we have proposed incorporates an Ontology-driven Storytelling framework, which automates the process of updating the ontology in order to recognize patterns and evolution of vulnerabilities over time and provides approaches for mitigating the vulnerabilities. The repercussions of vulnerabilities can be mitigated as a result of this, and conversely, future exposures can be predicted and prevented. Furthermore, our proposed framework utilized Generative Pre-trained Transformer (GPT) Large Language Models (LLMs) to provide mitigation suggestions.

{{</citation>}}


### (48/116) Secure Information Embedding in Images with Hybrid Firefly Algorithm (Sahil Nokhwal et al., 2023)

{{<citation>}}

Sahil Nokhwal, Manoj Chandrasekharan, Ankit Chaudhary. (2023)  
**Secure Information Embedding in Images with Hybrid Firefly Algorithm**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.13519v1)  

---


**ABSTRACT**  
Various methods have been proposed to secure access to sensitive information over time, such as the many cryptographic methods in use to facilitate secure communications on the internet. But other methods like steganography have been overlooked which may be more suitable in cases where the act of transmission of sensitive information itself should remain a secret. Multiple techniques that are commonly discussed for such scenarios suffer from low capacity and high distortion in the output signal. This research introduces a novel steganographic approach for concealing a confidential portable document format (PDF) document within a host image by employing the Hybrid Firefly algorithm (HFA) proposed to select the pixel arrangement. This algorithm combines two widely used optimization algorithms to improve their performance. The suggested methodology utilizes the HFA algorithm to conduct a search for optimal pixel placements in the spatial domain. The purpose of this search is to accomplish two main goals: increasing the host image's capacity and reducing distortion. Moreover, the proposed approach intends to reduce the time required for the embedding procedure. The findings indicate a decrease in image distortion and an accelerated rate of convergence in the search process. The resultant embeddings exhibit robustness against steganalytic assaults, hence rendering the identification of the embedded data a formidable undertaking.

{{</citation>}}


## cs.IR (2)



### (49/116) On Quantifying Sentiments of Financial News -- Are We Doing the Right Things? (Gourab Nath et al., 2023)

{{<citation>}}

Gourab Nath, Arav Sood, Aanchal Khanna, Savi Wilson, Karan Manot, Sree Kavya Durbaka. (2023)  
**On Quantifying Sentiments of Financial News -- Are We Doing the Right Things?**  

---
Primary Category: cs.IR  
Categories: I-2-7, cs-AI, cs-IR, cs-LG, cs-NE, cs.IR  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2312.14978v1)  

---


**ABSTRACT**  
Typical investors start off the day by going through the daily news to get an intuition about the performance of the market. The speculations based on the tone of the news ultimately shape their responses towards the market. Today, computers are being trained to compute the news sentiment so that it can be used as a variable to predict stock market movements and returns. Some researchers have even developed news-based market indices to forecast stock market returns. Majority of the research in the field of news sentiment analysis has focussed on using libraries like Vader, Loughran-McDonald (LM), Harvard IV and Pattern. However, are the popular approaches for measuring financial news sentiment really approaching the problem of sentiment analysis correctly? Our experiments suggest that measuring sentiments using these libraries, especially for financial news, fails to depict the true picture and hence may not be very reliable. Therefore, the question remains: What is the most effective and accurate approach to measure financial news sentiment? Our paper explores these questions and attempts to answer them through SENTInews: a one-of-its-kind financial news sentiment analyzer customized to the Indian context

{{</citation>}}


### (50/116) Empowering Few-Shot Recommender Systems with Large Language Models -- Enhanced Representations (Zhoumeng Wang, 2023)

{{<citation>}}

Zhoumeng Wang. (2023)  
**Empowering Few-Shot Recommender Systems with Large Language Models -- Enhanced Representations**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Few-Shot, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.13557v1)  

---


**ABSTRACT**  
Recommender systems utilizing explicit feedback have witnessed significant advancements and widespread applications over the past years. However, generating recommendations in few-shot scenarios remains a persistent challenge. Recently, large language models (LLMs) have emerged as a promising solution for addressing natural language processing (NLP) tasks, thereby offering novel insights into tackling the few-shot scenarios encountered by explicit feedback-based recommender systems. To bridge recommender systems and LLMs, we devise a prompting template that generates user and item representations based on explicit feedback. Subsequently, we integrate these LLM-processed representations into various recommendation models to evaluate their significance across diverse recommendation tasks. Our ablation experiments and case study analysis collectively demonstrate the effectiveness of LLMs in processing explicit feedback, highlighting that LLMs equipped with generative and logical reasoning capabilities can effectively serve as a component of recommender systems to enhance their performance in few-shot scenarios. Furthermore, the broad adaptability of LLMs augments the generalization potential of recommender models, despite certain inherent constraints. We anticipate that our study can inspire researchers to delve deeper into the multifaceted dimensions of LLMs's involvement in recommender systems and contribute to the advancement of the explicit feedback-based recommender systems field.

{{</citation>}}


## cs.AI (4)



### (51/116) Benchmarking Multi-Agent Preference-based Reinforcement Learning for Human-AI Teaming (Siddhant Bhambri et al., 2023)

{{<citation>}}

Siddhant Bhambri, Mudit Verma, Anil Murthy, Subbarao Kambhampati. (2023)  
**Benchmarking Multi-Agent Preference-based Reinforcement Learning for Human-AI Teaming**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14292v1)  

---


**ABSTRACT**  
Preference-based Reinforcement Learning (PbRL) is an active area of research, and has made significant strides in single-agent actor and in observer human-in-the-loop scenarios. However, its application within the co-operative multi-agent RL frameworks, where humans actively participate and express preferences for agent behavior, remains largely uncharted. We consider a two-agent (Human-AI) cooperative setup where both the agents are rewarded according to human's reward function for the team. However, the agent does not have access to it, and instead, utilizes preference-based queries to elicit its objectives and human's preferences for the robot in the human-robot team. We introduce the notion of Human-Flexibility, i.e. whether the human partner is amenable to multiple team strategies, with a special case being Specified Orchestration where the human has a single team policy in mind (most constrained case). We propose a suite of domains to study PbRL for Human-AI cooperative setup which explicitly require forced cooperation. Adapting state-of-the-art single-agent PbRL algorithms to our two-agent setting, we conduct a comprehensive benchmarking study across our domain suite. Our findings highlight the challenges associated with high degree of Human-Flexibility and the limited access to the human's envisioned policy in PbRL for Human-AI cooperation. Notably, we observe that PbRL algorithms exhibit effective performance exclusively in the case of Specified Orchestration which can be seen as an upper bound PbRL performance for future research.

{{</citation>}}


### (52/116) Learning Human-like Representations to Enable Learning Human Values (Andrea Wynn et al., 2023)

{{<citation>}}

Andrea Wynn, Ilia Sucholutsky, Thomas L. Griffiths. (2023)  
**Learning Human-like Representations to Enable Learning Human Values**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14106v1)  

---


**ABSTRACT**  
How can we build AI systems that are aligned with human values and objectives in order to avoid causing harm or violating societal standards for acceptable behavior? Making AI systems learn human-like representations of the world has many known benefits, including improving generalization, robustness to domain shifts, and few-shot learning performance, among others. We propose that this kind of representational alignment between machine learning (ML) models and humans is also a necessary condition for value alignment, where ML systems conform to human values and societal norms. We focus on ethics as one aspect of value alignment and train multiple ML agents (support vector regression and kernel regression) in a multi-armed bandit setting, where rewards are sampled from a distribution that reflects the morality of the chosen action. We then study the relationship between each agent's degree of representational alignment with humans and their performance when learning to take the most ethical actions.

{{</citation>}}


### (53/116) Understanding Inter-Session Intentions via Complex Logical Reasoning (Jiaxin Bai et al., 2023)

{{<citation>}}

Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Yangqiu Song. (2023)  
**Understanding Inter-Session Intentions via Complex Logical Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: QA, Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.13866v1)  

---


**ABSTRACT**  
Understanding user intentions is crucial for enhancing product recommendations, navigation suggestions, and query reformulations. However, user intentions can be complex, involving multiple sessions and attribute requirements connected by logical operators such as And, Or, and Not. For example, a user may search for Nike or Adidas running shoes across various sessions, with a preference for the color purple. In another case, a user may have purchased a mattress in a previous session and is now seeking a corresponding bed frame without intending to buy another mattress. Prior research on session understanding has not sufficiently addressed how to make product or attribute recommendations for such complex intentions. In this paper, we introduce the task of logical session complex query answering, where sessions are treated as hyperedges of items, and we formulate the problem of complex intention understanding as a task of logical session complex queries answering (LS-CQA) on an aggregated hypergraph of sessions, items, and attributes. The proposed task is a special type of complex query answering task with sessions as ordered hyperedges. We also propose a new model, the Logical Session Graph Transformer (LSGT), which captures interactions among items across different sessions and their logical connections using a transformer structure. We analyze the expressiveness of LSGT and prove the permutation invariance of the inputs for the logical operators. We evaluate LSGT on three datasets and demonstrate that it achieves state-of-the-art results.

{{</citation>}}


### (54/116) HGE: Embedding Temporal Knowledge Graphs in a Product Space of Heterogeneous Geometric Subspaces (Jiaxin Pan et al., 2023)

{{<citation>}}

Jiaxin Pan, Mojtaba Nayyeri, Yinan Li, Steffen Staab. (2023)  
**HGE: Embedding Temporal Knowledge Graphs in a Product Space of Heterogeneous Geometric Subspaces**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.13680v2)  

---


**ABSTRACT**  
Temporal knowledge graphs represent temporal facts $(s,p,o,\tau)$ relating a subject $s$ and an object $o$ via a relation label $p$ at time $\tau$, where $\tau$ could be a time point or time interval. Temporal knowledge graphs may exhibit static temporal patterns at distinct points in time and dynamic temporal patterns between different timestamps. In order to learn a rich set of static and dynamic temporal patterns and apply them for inference, several embedding approaches have been suggested in the literature. However, as most of them resort to single underlying embedding spaces, their capability to model all kinds of temporal patterns was severely limited by having to adhere to the geometric property of their one embedding space. We lift this limitation by an embedding approach that maps temporal facts into a product space of several heterogeneous geometric subspaces with distinct geometric properties, i.e.\ Complex, Dual, and Split-complex spaces. In addition, we propose a temporal-geometric attention mechanism to integrate information from different geometric subspaces conveniently according to the captured relational and temporal information. Experimental results on standard temporal benchmark datasets favorably evaluate our approach against state-of-the-art models.

{{</citation>}}


## cs.SE (2)



### (55/116) Exploring the intersection of Generative AI and Software Development (Filipe Calegario et al., 2023)

{{<citation>}}

Filipe Calegario, Vanilson Burégio, Francisco Erivaldo, Daniel Moraes Costa Andrade, Kailane Felix, Nathalia Barbosa, Pedro Lucas da Silva Lucena, César França. (2023)  
**Exploring the intersection of Generative AI and Software Development**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.14262v1)  

---


**ABSTRACT**  
In the ever-evolving landscape of Artificial Intelligence (AI), the synergy between generative AI and Software Engineering emerges as a transformative frontier. This whitepaper delves into the unexplored realm, elucidating how generative AI techniques can revolutionize software development. Spanning from project management to support and updates, we meticulously map the demands of each development stage and unveil the potential of generative AI in addressing them. Techniques such as zero-shot prompting, self-consistency, and multimodal chain-of-thought are explored, showcasing their unique capabilities in enhancing generative AI models. The significance of vector embeddings, context, plugins, tools, and code assistants is underscored, emphasizing their role in capturing semantic information and amplifying generative AI capabilities. Looking ahead, this intersection promises to elevate productivity, improve code quality, and streamline the software development process. This whitepaper serves as a guide for stakeholders, urging discussions and experiments in the application of generative AI in Software Engineering, fostering innovation and collaboration for a qualitative leap in the efficiency and effectiveness of software development.

{{</citation>}}


### (56/116) Building Your Own Product Copilot: Challenges, Opportunities, and Needs (Chris Parnin et al., 2023)

{{<citation>}}

Chris Parnin, Gustavo Soares, Rahul Pandita, Sumit Gulwani, Jessica Rich, Austin Z. Henley. (2023)  
**Building Your Own Product Copilot: Challenges, Opportunities, and Needs**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14231v1)  

---


**ABSTRACT**  
A race is underway to embed advanced AI capabilities into products. These product copilots enable users to ask questions in natural language and receive relevant responses that are specific to the user's context. In fact, virtually every large technology company is looking to add these capabilities to their software products. However, for most software engineers, this is often their first encounter with integrating AI-powered technology. Furthermore, software engineering processes and tools have not caught up with the challenges and scale involved with building AI-powered applications. In this work, we present the findings of an interview study with 26 professional software engineers responsible for building product copilots at various companies. From our interviews, we found pain points at every step of the engineering process and the challenges that strained existing development practices. We then conducted group brainstorming sessions to collaborative on opportunities and tool designs for the broader software engineering community.

{{</citation>}}


## cs.NI (1)



### (57/116) Deep Reinforcement Learning Based Placement for Integrated Access Backhauling in UAV-Assisted Wireless Networks (Yuhui Wang et al., 2023)

{{<citation>}}

Yuhui Wang, Junaid Farooq. (2023)  
**Deep Reinforcement Learning Based Placement for Integrated Access Backhauling in UAV-Assisted Wireless Networks**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs-SY, cs.NI, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14247v1)  

---


**ABSTRACT**  
The advent of fifth generation (5G) networks has opened new avenues for enhancing connectivity, particularly in challenging environments like remote areas or disaster-struck regions. Unmanned aerial vehicles (UAVs) have been identified as a versatile tool in this context, particularly for improving network performance through the Integrated access and backhaul (IAB) feature of 5G. However, existing approaches to UAV-assisted network enhancement face limitations in dynamically adapting to varying user locations and network demands. This paper introduces a novel approach leveraging deep reinforcement learning (DRL) to optimize UAV placement in real-time, dynamically adjusting to changing network conditions and user requirements. Our method focuses on the intricate balance between fronthaul and backhaul links, a critical aspect often overlooked in current solutions. The unique contribution of this work lies in its ability to autonomously position UAVs in a way that not only ensures robust connectivity to ground users but also maintains seamless integration with central network infrastructure. Through various simulated scenarios, we demonstrate how our approach effectively addresses these challenges, enhancing coverage and network performance in critical areas. This research fills a significant gap in UAV-assisted 5G networks, providing a scalable and adaptive solution for future mobile networks.

{{</citation>}}


## cs.CV (30)



### (58/116) DriveLM: Driving with Graph Visual Question Answering (Chonghao Sima et al., 2023)

{{<citation>}}

Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Ping Luo, Andreas Geiger, Hongyang Li. (2023)  
**DriveLM: Driving with Graph Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.14150v1)  

---


**ABSTRACT**  
We study how vision-language models (VLMs) trained on web-scale data can be integrated into end-to-end driving systems to boost generalization and enable interactivity with human users. While recent approaches adapt VLMs to driving via single-round visual question answering (VQA), human drivers reason about decisions in multiple steps. Starting from the localization of key objects, humans estimate object interactions before taking actions. The key insight is that with our proposed task, Graph VQA, where we model graph-structured reasoning through perception, prediction and planning question-answer pairs, we obtain a suitable proxy task to mimic the human reasoning process. We instantiate datasets (DriveLM-Data) built upon nuScenes and CARLA, and propose a VLM-based baseline approach (DriveLM-Agent) for jointly performing Graph VQA and end-to-end driving. The experiments demonstrate that Graph VQA provides a simple, principled framework for reasoning about a driving scene, and DriveLM-Data provides a challenging benchmark for this task. Our DriveLM-Agent baseline performs end-to-end autonomous driving competitively in comparison to state-of-the-art driving-specific architectures. Notably, its benefits are pronounced when it is evaluated zero-shot on unseen objects or sensor configurations. We hope this work can be the starting point to shed new light on how to apply VLMs for autonomous driving. To facilitate future research, all code, data, and models are available to the public.

{{</citation>}}


### (59/116) DUSt3R: Geometric 3D Vision Made Easy (Shuzhe Wang et al., 2023)

{{<citation>}}

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud. (2023)  
**DUSt3R: Geometric 3D Vision Made Easy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.14132v1)  

---


**ABSTRACT**  
Multi-view stereo reconstruction (MVS) in the wild requires to first estimate the camera parameters e.g. intrinsic and extrinsic parameters. These are usually tedious and cumbersome to obtain, yet they are mandatory to triangulate corresponding pixels in 3D space, which is the core of all best performing MVS algorithms. In this work, we take an opposite stance and introduce DUSt3R, a radically novel paradigm for Dense and Unconstrained Stereo 3D Reconstruction of arbitrary image collections, i.e. operating without prior information about camera calibration nor viewpoint poses. We cast the pairwise reconstruction problem as a regression of pointmaps, relaxing the hard constraints of usual projective camera models. We show that this formulation smoothly unifies the monocular and binocular reconstruction cases. In the case where more than two images are provided, we further propose a simple yet effective global alignment strategy that expresses all pairwise pointmaps in a common reference frame. We base our network architecture on standard Transformer encoders and decoders, allowing us to leverage powerful pretrained models. Our formulation directly provides a 3D model of the scene as well as depth information, but interestingly, we can seamlessly recover from it, pixel matches, relative and absolute camera. Exhaustive experiments on all these tasks showcase that the proposed DUSt3R can unify various 3D vision tasks and set new SoTAs on monocular/multi-view depth estimation as well as relative pose estimation. In summary, DUSt3R makes many geometric 3D vision tasks easy.

{{</citation>}}


### (60/116) VCoder: Versatile Vision Encoders for Multimodal Large Language Models (Jitesh Jain et al., 2023)

{{<citation>}}

Jitesh Jain, Jianwei Yang, Humphrey Shi. (2023)  
**VCoder: Versatile Vision Encoders for Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.14233v1)  

---


**ABSTRACT**  
Humans possess the remarkable skill of Visual Perception, the ability to see and understand the seen, helping them make sense of the visual world and, in turn, reason. Multimodal Large Language Models (MLLM) have recently achieved impressive performance on vision-language tasks ranging from visual question-answering and image captioning to visual reasoning and image generation. However, when prompted to identify or count (perceive) the entities in a given image, existing MLLM systems fail. Working towards developing an accurate MLLM system for perception and reasoning, we propose using Versatile vision enCoders (VCoder) as perception eyes for Multimodal LLMs. We feed the VCoder with perception modalities such as segmentation or depth maps, improving the MLLM's perception abilities. Secondly, we leverage the images from COCO and outputs from off-the-shelf vision perception models to create our COCO Segmentation Text (COST) dataset for training and evaluating MLLMs on the object perception task. Thirdly, we introduce metrics to assess the object perception abilities in MLLMs on our COST dataset. Lastly, we provide extensive experimental evidence proving the VCoder's improved object-level perception skills over existing Multimodal LLMs, including GPT-4V. We open-source our dataset, code, and models to promote research. We open-source our code at https://github.com/SHI-Labs/VCoder

{{</citation>}}


### (61/116) Entropic Open-set Active Learning (Bardia Safaei et al., 2023)

{{<citation>}}

Bardia Safaei, Vibashan VS, Celso M. de Melo, Vishal M. Patel. (2023)  
**Entropic Open-set Active Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning, ImageNet  
[Paper Link](http://arxiv.org/abs/2312.14126v1)  

---


**ABSTRACT**  
Active Learning (AL) aims to enhance the performance of deep models by selecting the most informative samples for annotation from a pool of unlabeled data. Despite impressive performance in closed-set settings, most AL methods fail in real-world scenarios where the unlabeled data contains unknown categories. Recently, a few studies have attempted to tackle the AL problem for the open-set setting. However, these methods focus more on selecting known samples and do not efficiently utilize unknown samples obtained during AL rounds. In this work, we propose an Entropic Open-set AL (EOAL) framework which leverages both known and unknown distributions effectively to select informative samples during AL rounds. Specifically, our approach employs two different entropy scores. One measures the uncertainty of a sample with respect to the known-class distributions. The other measures the uncertainty of the sample with respect to the unknown-class distributions. By utilizing these two entropy scores we effectively separate the known and unknown samples from the unlabeled data resulting in better sampling. Through extensive experiments, we show that the proposed method outperforms existing state-of-the-art methods on CIFAR-10, CIFAR-100, and TinyImageNet datasets. Code is available at \url{https://github.com/bardisafa/EOAL}.

{{</citation>}}


### (62/116) Parrot Captions Teach CLIP to Spot Text (Yiqi Lin et al., 2023)

{{<citation>}}

Yiqi Lin, Conghui He, Alex Jinpeng Wang, Bin Wang, Weijia Li, Mike Zheng Shou. (2023)  
**Parrot Captions Teach CLIP to Spot Text**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14232v1)  

---


**ABSTRACT**  
Despite CLIP being the foundation model in numerous vision-language applications, the CLIP suffers from a severe text spotting bias. Such bias causes CLIP models to `Parrot' the visual text embedded within images while disregarding the authentic visual semantics. We uncover that in the most popular image-text dataset LAION-2B, the captions also densely parrot (spell) the text embedded in images. Our analysis shows that around \textbf{50\%} of images are embedded with visual text content, and \textbf{90\%} of their captions more or less parrot the visual text. Based on such observation, we thoroughly inspect the different release d versions of CLIP models and verify that the visual text is the dominant factor in measuring the LAION-style image-text similarity for these models. To examine whether these parrot captions shape the text spotting bias, we train a series of CLIP models with LAION subsets curated by different parrot-caption-oriented criteria. We show that training with parrot captions easily shapes such bias but harms the expected visual-language representation learning in CLIP models. This suggests that it is urgent to revisit either the design of CLIP-like models or the existing image-text dataset curation pipeline built on CLIP score filtering.

{{</citation>}}


### (63/116) VideoPoet: A Large Language Model for Zero-Shot Video Generation (Dan Kondratyuk et al., 2023)

{{<citation>}}

Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, Yong Cheng, Ming-Chang Chiu, Josh Dillon, Irfan Essa, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, David Ross, Grant Schindler, Mikhail Sirotenko, Kihyuk Sohn, Krishna Somandepalli, Huisheng Wang, Jimmy Yan, Ming-Hsuan Yang, Xuan Yang, Bryan Seybold, Lu Jiang. (2023)  
**VideoPoet: A Large Language Model for Zero-Shot Video Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, Transformer, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.14125v1)  

---


**ABSTRACT**  
We present VideoPoet, a language model capable of synthesizing high-quality video, with matching audio, from a large variety of conditioning signals. VideoPoet employs a decoder-only transformer architecture that processes multimodal inputs -- including images, videos, text, and audio. The training protocol follows that of Large Language Models (LLMs), consisting of two stages: pretraining and task-specific adaptation. During pretraining, VideoPoet incorporates a mixture of multimodal generative objectives within an autoregressive Transformer framework. The pretrained LLM serves as a foundation that can be adapted for a range of video generation tasks. We present empirical results demonstrating the model's state-of-the-art capabilities in zero-shot video generation, specifically highlighting VideoPoet's ability to generate high-fidelity motions. Project page: http://sites.research.google/videopoet/

{{</citation>}}


### (64/116) HD-Painter: High-Resolution and Prompt-Faithful Text-Guided Image Inpainting with Diffusion Models (Hayk Manukyan et al., 2023)

{{<citation>}}

Hayk Manukyan, Andranik Sargsyan, Barsegh Atanyan, Zhangyang Wang, Shant Navasardyan, Humphrey Shi. (2023)  
**HD-Painter: High-Resolution and Prompt-Faithful Text-Guided Image Inpainting with Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2312.14091v2)  

---


**ABSTRACT**  
Recent progress in text-guided image inpainting, based on the unprecedented success of text-to-image diffusion models, has led to exceptionally realistic and visually plausible results. However, there is still significant potential for improvement in current text-to-image inpainting models, particularly in better aligning the inpainted area with user prompts and performing high-resolution inpainting. Therefore, in this paper we introduce HD-Painter, a completely training-free approach that accurately follows to prompts and coherently scales to high-resolution image inpainting. To this end, we design the Prompt-Aware Introverted Attention (PAIntA) layer enhancing self-attention scores by prompt information and resulting in better text alignment generations. To further improve the prompt coherence we introduce the Reweighting Attention Score Guidance (RASG) mechanism seamlessly integrating a post-hoc sampling strategy into general form of DDIM to prevent out-of-distribution latent shifts. Moreover, HD-Painter allows extension to larger scales by introducing a specialized super-resolution technique customized for inpainting, enabling the completion of missing regions in images of up to 2K resolution. Our experiments demonstrate that HD-Painter surpasses existing state-of-the-art approaches qualitatively and quantitatively, achieving an impressive generation accuracy improvement of 61.4% vs 51.9%. We will make the codes publicly available at: https://github.com/Picsart-AI-Research/HD-Painter

{{</citation>}}


### (65/116) LiDAR-LLM: Exploring the Potential of Large Language Models for 3D LiDAR Understanding (Senqiao Yang et al., 2023)

{{<citation>}}

Senqiao Yang, Jiaming Liu, Ray Zhang, Mingjie Pan, Zoey Guo, Xiaoqi Li, Zehui Chen, Peng Gao, Yandong Guo, Shanghang Zhang. (2023)  
**LiDAR-LLM: Exploring the Potential of Large Language Models for 3D LiDAR Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BLEU, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.14074v1)  

---


**ABSTRACT**  
Recently, Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have shown promise in instruction following and 2D image understanding. While these models are powerful, they have not yet been developed to comprehend the more challenging 3D physical scenes, especially when it comes to the sparse outdoor LiDAR data. In this paper, we introduce LiDAR-LLM, which takes raw LiDAR data as input and harnesses the remarkable reasoning capabilities of LLMs to gain a comprehensive understanding of outdoor 3D scenes. The central insight of our LiDAR-LLM is the reformulation of 3D outdoor scene cognition as a language modeling problem, encompassing tasks such as 3D captioning, 3D grounding, 3D question answering, etc. Specifically, due to the scarcity of 3D LiDAR-text pairing data, we introduce a three-stage training strategy and generate relevant datasets, progressively aligning the 3D modality with the language embedding space of LLM. Furthermore, we design a View-Aware Transformer (VAT) to connect the 3D encoder with the LLM, which effectively bridges the modality gap and enhances the LLM's spatial orientation comprehension of visual features. Our experiments show that LiDAR-LLM possesses favorable capabilities to comprehend various instructions regarding 3D scenes and engage in complex spatial reasoning. LiDAR-LLM attains a 40.9 BLEU-1 on the 3D captioning task and achieves a 63.1\% classification accuracy and a 14.3\% BEV mIoU on the 3D grounding task. Web page: https://sites.google.com/view/lidar-llm

{{</citation>}}


### (66/116) A Strong Baseline for Temporal Video-Text Alignment (Zeqian Li et al., 2023)

{{<citation>}}

Zeqian Li, Qirui Chen, Tengda Han, Ya Zhang, Yanfeng Wang, Weidi Xie. (2023)  
**A Strong Baseline for Temporal Video-Text Alignment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.14055v1)  

---


**ABSTRACT**  
In this paper, we consider the problem of temporally aligning the video and texts from instructional videos, specifically, given a long-term video, and associated text sentences, our goal is to determine their corresponding timestamps in the video. To this end, we establish a simple, yet strong model that adopts a Transformer-based architecture with all texts as queries, iteratively attending to the visual features, to infer the optimal timestamp. We conduct thorough experiments to investigate: (i) the effect of upgrading ASR systems to reduce errors from speech recognition, (ii) the effect of various visual-textual backbones, ranging from CLIP to S3D, to the more recent InternVideo, (iii) the effect of transforming noisy ASR transcripts into descriptive steps by prompting a large language model (LLM), to summarize the core activities within the ASR transcript as a new training dataset. As a result, our proposed simple model demonstrates superior performance on both narration alignment and procedural step grounding tasks, surpassing existing state-of-the-art methods by a significant margin on three public benchmarks, namely, 9.3% on HT-Step, 3.4% on HTM-Align and 4.7% on CrossTask. We believe the proposed model and dataset with descriptive steps can be treated as a strong baseline for future research in temporal video-text alignment. All codes, models, and the resulting dataset will be publicly released to the research community.

{{</citation>}}


### (67/116) Dual Attention U-Net with Feature Infusion: Pushing the Boundaries of Multiclass Defect Segmentation (Rasha Alshawi et al., 2023)

{{<citation>}}

Rasha Alshawi, Md Tamjidul Hoque, Md Meftahul Ferdaus, Mahdi Abdelguerfi, Kendall Niles, Ken Prathak, Joe Tom, Jordan Klein, Murtada Mousa, Johny Javier Lopez. (2023)  
**Dual Attention U-Net with Feature Infusion: Pushing the Boundaries of Multiclass Defect Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.14053v1)  

---


**ABSTRACT**  
The proposed architecture, Dual Attentive U-Net with Feature Infusion (DAU-FI Net), addresses challenges in semantic segmentation, particularly on multiclass imbalanced datasets with limited samples. DAU-FI Net integrates multiscale spatial-channel attention mechanisms and feature injection to enhance precision in object localization. The core employs a multiscale depth-separable convolution block, capturing localized patterns across scales. This block is complemented by a spatial-channel squeeze and excitation (scSE) attention unit, modeling inter-dependencies between channels and spatial regions in feature maps. Additionally, additive attention gates refine segmentation by connecting encoder-decoder pathways.   To augment the model, engineered features using Gabor filters for textural analysis, Sobel and Canny filters for edge detection are injected guided by semantic masks to expand the feature space strategically. Comprehensive experiments on a challenging sewer pipe and culvert defect dataset and a benchmark dataset validate DAU-FI Net's capabilities. Ablation studies highlight incremental benefits from attention blocks and feature injection. DAU-FI Net achieves state-of-the-art mean Intersection over Union (IoU) of 95.6% and 98.8% on the defect test set and benchmark respectively, surpassing prior methods by 8.9% and 12.6%, respectively. Ablation studies highlight incremental benefits from attention blocks and feature injection. The proposed architecture provides a robust solution, advancing semantic segmentation for multiclass problems with limited training data. Our sewer-culvert defects dataset, featuring pixel-level annotations, opens avenues for further research in this crucial domain. Overall, this work delivers key innovations in architecture, attention, and feature engineering to elevate semantic segmentation efficacy.

{{</citation>}}


### (68/116) Open-Set: ID Card Presentation Attack Detection using Neural Transfer Style (Reuben Markham et al., 2023)

{{<citation>}}

Reuben Markham, Juan M. Espin, Mario Nieto-Hidalgo, Juan E. Tapia. (2023)  
**Open-Set: ID Card Presentation Attack Detection using Neural Transfer Style**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.13993v1)  

---


**ABSTRACT**  
The accurate detection of ID card Presentation Attacks (PA) is becoming increasingly important due to the rising number of online/remote services that require the presentation of digital photographs of ID cards for digital onboarding or authentication. Furthermore, cybercriminals are continuously searching for innovative ways to fool authentication systems to gain unauthorized access to these services. Although advances in neural network design and training have pushed image classification to the state of the art, one of the main challenges faced by the development of fraud detection systems is the curation of representative datasets for training and evaluation. The handcrafted creation of representative presentation attack samples often requires expertise and is very time-consuming, thus an automatic process of obtaining high-quality data is highly desirable. This work explores ID card Presentation Attack Instruments (PAI) in order to improve the generation of samples with four Generative Adversarial Networks (GANs) based image translation models and analyses the effectiveness of the generated data for training fraud detection systems. Using open-source data, we show that synthetic attack presentations are an adequate complement for additional real attack presentations, where we obtain an EER performance increase of 0.63% points for print attacks and a loss of 0.29% for screen capture attacks.

{{</citation>}}


### (69/116) Controllable 3D Face Generation with Conditional Style Code Diffusion (Xiaolong Shen et al., 2023)

{{<citation>}}

Xiaolong Shen, Jianxin Ma, Chang Zhou, Zongxin Yang. (2023)  
**Controllable 3D Face Generation with Conditional Style Code Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Dialog  
[Paper Link](http://arxiv.org/abs/2312.13941v1)  

---


**ABSTRACT**  
Generating photorealistic 3D faces from given conditions is a challenging task. Existing methods often rely on time-consuming one-by-one optimization approaches, which are not efficient for modeling the same distribution content, e.g., faces. Additionally, an ideal controllable 3D face generation model should consider both facial attributes and expressions. Thus we propose a novel approach called TEx-Face(TExt & Expression-to-Face) that addresses these challenges by dividing the task into three components, i.e., 3D GAN Inversion, Conditional Style Code Diffusion, and 3D Face Decoding. For 3D GAN inversion, we introduce two methods which aim to enhance the representation of style codes and alleviate 3D inconsistencies. Furthermore, we design a style code denoiser to incorporate multiple conditions into the style code and propose a data augmentation strategy to address the issue of insufficient paired visual-language data. Extensive experiments conducted on FFHQ, CelebA-HQ, and CelebA-Dialog demonstrate the promising performance of our TEx-Face in achieving the efficient and controllable generation of photorealistic 3D faces. The code will be available at https://github.com/sxl142/TEx-Face.

{{</citation>}}


### (70/116) Reducing Hallucinations: Enhancing VQA for Flood Disaster Damage Assessment with Visual Contexts (Yimin Sun et al., 2023)

{{<citation>}}

Yimin Sun, Chao Wang, Yan Peng. (2023)  
**Reducing Hallucinations: Enhancing VQA for Flood Disaster Damage Assessment with Visual Contexts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.13848v1)  

---


**ABSTRACT**  
The zero-shot performance of visual question answering (VQA) models relies heavily on prompts. For example, a zero-shot VQA for disaster scenarios could leverage well-designed Chain of Thought (CoT) prompts to stimulate the model's potential. However, using CoT prompts has some problems, such as causing an incorrect answer in the end due to the hallucination in the thought process. In this paper, we propose a zero-shot VQA named Flood Disaster VQA with Two-Stage Prompt (VQA-TSP). The model generates the thought process in the first stage and then uses the thought process to generate the final answer in the second stage. In particular, visual context is added in the second stage to relieve the hallucination problem that exists in the thought process. Experimental results show that our method exceeds the performance of state-of-the-art zero-shot VQA models for flood disaster scenarios in total. Our study provides a research basis for improving the performance of CoT-based zero-shot VQA.

{{</citation>}}


### (71/116) Q-SENN: Quantized Self-Explaining Neural Networks (Thomas Norrenbrock et al., 2023)

{{<citation>}}

Thomas Norrenbrock, Marco Rudolph, Bodo Rosenhahn. (2023)  
**Q-SENN: Quantized Self-Explaining Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.13839v1)  

---


**ABSTRACT**  
Explanations in Computer Vision are often desired, but most Deep Neural Networks can only provide saliency maps with questionable faithfulness. Self-Explaining Neural Networks (SENN) extract interpretable concepts with fidelity, diversity, and grounding to combine them linearly for decision-making. While they can explain what was recognized, initial realizations lack accuracy and general applicability. We propose the Quantized-Self-Explaining Neural Network Q-SENN. Q-SENN satisfies or exceeds the desiderata of SENN while being applicable to more complex datasets and maintaining most or all of the accuracy of an uninterpretable baseline model, out-performing previous work in all considered metrics. Q-SENN describes the relationship between every class and feature as either positive, negative or neutral instead of an arbitrary number of possible relations, enforcing more binary human-friendly features. Since every class is assigned just 5 interpretable features on average, Q-SENN shows convincing local and global interpretability. Additionally, we propose a feature alignment method, capable of aligning learned features with human language-based concepts without additional supervision. Thus, what is learned can be more easily verbalized. The code is published: https://github.com/ThomasNorr/Q-SENN

{{</citation>}}


### (72/116) Universal Noise Annotation: Unveiling the Impact of Noisy annotation on Object Detection (Kwangrok Ryoo et al., 2023)

{{<citation>}}

Kwangrok Ryoo, Yeonsik Jo, Seungjun Lee, Mira Kim, Ahra Jo, Seung Hwan Kim, Seungryong Kim, Soonyoung Lee. (2023)  
**Universal Noise Annotation: Unveiling the Impact of Noisy annotation on Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.13822v1)  

---


**ABSTRACT**  
For object detection task with noisy labels, it is important to consider not only categorization noise, as in image classification, but also localization noise, missing annotations, and bogus bounding boxes. However, previous studies have only addressed certain types of noise (e.g., localization or categorization). In this paper, we propose Universal-Noise Annotation (UNA), a more practical setting that encompasses all types of noise that can occur in object detection, and analyze how UNA affects the performance of the detector. We analyzed the development direction of previous works of detection algorithms and examined the factors that impact the robustness of detection model learning method. We open-source the code for injecting UNA into the dataset and all the training log and weight are also shared.

{{</citation>}}


### (73/116) AutoAugment Input Transformation for Highly Transferable Targeted Attacks (Haobo Lu et al., 2023)

{{<citation>}}

Haobo Lu, Xin Liu, Kun He. (2023)  
**AutoAugment Input Transformation for Highly Transferable Targeted Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2312.14218v1)  

---


**ABSTRACT**  
Deep Neural Networks (DNNs) are widely acknowledged to be susceptible to adversarial examples, wherein imperceptible perturbations are added to clean examples through diverse input transformation attacks. However, these methods originally designed for non-targeted attacks exhibit low success rates in targeted attacks. Recent targeted adversarial attacks mainly pay attention to gradient optimization, attempting to find the suitable perturbation direction. However, few of them are dedicated to input transformation.In this work, we observe a positive correlation between the logit/probability of the target class and diverse input transformation methods in targeted attacks. To this end, we propose a novel targeted adversarial attack called AutoAugment Input Transformation (AAIT). Instead of relying on hand-made strategies, AAIT searches for the optimal transformation policy from a transformation space comprising various operations. Then, AAIT crafts adversarial examples using the found optimal transformation policy to boost the adversarial transferability in targeted attacks. Extensive experiments conducted on CIFAR-10 and ImageNet-Compatible datasets demonstrate that the proposed AAIT surpasses other transfer-based targeted attacks significantly.

{{</citation>}}


### (74/116) Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection (Soopil Kim et al., 2023)

{{<citation>}}

Soopil Kim, Sion An, Philip Chikontwe, Myeongkyun Kang, Ehsan Adeli, Kilian M. Pohl, Sanghyun Park. (2023)  
**Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.13783v1)  

---


**ABSTRACT**  
Logical anomalies (LA) refer to data violating underlying logical constraints e.g., the quantity, arrangement, or composition of components within an image. Detecting accurately such anomalies requires models to reason about various component types through segmentation. However, curation of pixel-level annotations for semantic segmentation is both time-consuming and expensive. Although there are some prior few-shot or unsupervised co-part segmentation algorithms, they often fail on images with industrial object. These images have components with similar textures and shapes, and a precise differentiation proves challenging. In this study, we introduce a novel component segmentation model for LA detection that leverages a few labeled samples and unlabeled images sharing logical constraints. To ensure consistent segmentation across unlabeled images, we employ a histogram matching loss in conjunction with an entropy loss. As segmentation predictions play a crucial role, we propose to enhance both local and global sample validity detection by capturing key aspects from visual semantics via three memory banks: class histograms, component composition embeddings and patch-level representations. For effective LA detection, we propose an adaptive scaling strategy to standardize anomaly scores from different memory banks in inference. Extensive experiments on the public benchmark MVTec LOCO AD reveal our method achieves 98.1% AUROC in LA detection vs. 89.6% from competing methods.

{{</citation>}}


### (75/116) 3D Points Splatting for Real-Time Dynamic Hand Reconstruction (Zheheng Jiang et al., 2023)

{{<citation>}}

Zheheng Jiang, Hossein Rahmani, Sue Black, Bryan M. Williams. (2023)  
**3D Points Splatting for Real-Time Dynamic Hand Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.13770v1)  

---


**ABSTRACT**  
We present 3D Points Splatting Hand Reconstruction (3D-PSHR), a real-time and photo-realistic hand reconstruction approach. We propose a self-adaptive canonical points upsampling strategy to achieve high-resolution hand geometry representation. This is followed by a self-adaptive deformation that deforms the hand from the canonical space to the target pose, adapting to the dynamic changing of canonical points which, in contrast to the common practice of subdividing the MANO model, offers greater flexibility and results in improved geometry fitting. To model texture, we disentangle the appearance color into the intrinsic albedo and pose-aware shading, which are learned through a Context-Attention module. Moreover, our approach allows the geometric and the appearance models to be trained simultaneously in an end-to-end manner. We demonstrate that our method is capable of producing animatable, photorealistic and relightable hand reconstructions using multiple datasets, including monocular videos captured with handheld smartphones and large-scale multi-view videos featuring various hand poses. We also demonstrate that our approach achieves real-time rendering speeds while simultaneously maintaining superior performance compared to existing state-of-the-art methods.

{{</citation>}}


### (76/116) A Semantic Space is Worth 256 Language Descriptions: Make Stronger Segmentation Models with Descriptive Properties (Junfei Xiao et al., 2023)

{{<citation>}}

Junfei Xiao, Ziqi Zhou, Wenxuan Li, Shiyi Lan, Jieru Mei, Zhiding Yu, Alan Yuille, Yuyin Zhou, Cihang Xie. (2023)  
**A Semantic Space is Worth 256 Language Descriptions: Make Stronger Segmentation Models with Descriptive Properties**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.13764v1)  

---


**ABSTRACT**  
This paper introduces ProLab, a novel approach using property-level label space for creating strong interpretable segmentation models. Instead of relying solely on category-specific annotations, ProLab uses descriptive properties grounded in common sense knowledge for supervising segmentation models. It is based on two core designs. First, we employ Large Language Models (LLMs) and carefully crafted prompts to generate descriptions of all involved categories that carry meaningful common sense knowledge and follow a structured format. Second, we introduce a description embedding model preserving semantic correlation across descriptions and then cluster them into a set of descriptive properties (e.g., 256) using K-Means. These properties are based on interpretable common sense knowledge consistent with theories of human recognition. We empirically show that our approach makes segmentation models perform stronger on five classic benchmarks (e.g., ADE20K, COCO-Stuff, Pascal Context, Cityscapes, and BDD). Our method also shows better scalability with extended training steps than category-level supervision. Our interpretable segmentation framework also emerges with the generalization ability to segment out-of-domain or unknown categories using only in-domain descriptive properties. Code is available at https://github.com/lambert-x/ProLab.

{{</citation>}}


### (77/116) DECO: Query-Based End-to-End Object Detection with ConvNets (Xinghao Chen et al., 2023)

{{<citation>}}

Xinghao Chen, Siwei Li, Yijing Yang, Yunhe Wang. (2023)  
**DECO: Query-Based End-to-End Object Detection with ConvNets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2312.13735v1)  

---


**ABSTRACT**  
Detection Transformer (DETR) and its variants have shown great potential for accurate object detection in recent years. The mechanism of object query enables DETR family to directly obtain a fixed number of object predictions and streamlines the detection pipeline. Meanwhile, recent studies also reveal that with proper architecture design, convolution networks (ConvNets) also achieve competitive performance with transformers, \eg, ConvNeXt. To this end, in this paper we explore whether we could build a query-based end-to-end object detection framework with ConvNets instead of sophisticated transformer architecture. The proposed framework, \ie, Detection ConvNet (DECO), is composed of a backbone and convolutional encoder-decoder architecture. We carefully design the DECO encoder and propose a novel mechanism for our DECO decoder to perform interaction between object queries and image features via convolutional layers. We compare the proposed DECO against prior detectors on the challenging COCO benchmark. Despite its simplicity, our DECO achieves competitive performance in terms of detection accuracy and running speed. Specifically, with the ResNet-50 and ConvNeXt-Tiny backbone, DECO obtains $38.6\%$ and $40.8\%$ AP on COCO \textit{val} set with $35$ and $28$ FPS respectively and outperforms the DETR model. Incorporated with advanced multi-scale feature module, our DECO+ achieves $47.8\%$ AP with $34$ FPS. We hope the proposed DECO brings another perspective for designing object detection framework.

{{</citation>}}


### (78/116) Free-Editor: Zero-shot Text-driven 3D Scene Editing (Nazmul Karim et al., 2023)

{{<citation>}}

Nazmul Karim, Umar Khalid, Hasan Iqbal, Jing Hua, Chen Chen. (2023)  
**Free-Editor: Zero-shot Text-driven 3D Scene Editing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.13663v1)  

---


**ABSTRACT**  
Text-to-Image (T2I) diffusion models have gained popularity recently due to their multipurpose and easy-to-use nature, e.g. image and video generation as well as editing. However, training a diffusion model specifically for 3D scene editing is not straightforward due to the lack of large-scale datasets. To date, editing 3D scenes requires either re-training the model to adapt to various 3D edited scenes or design-specific methods for each special editing type. Furthermore, state-of-the-art (SOTA) methods require multiple synchronized edited images from the same scene to facilitate the scene editing. Due to the current limitations of T2I models, it is very challenging to apply consistent editing effects to multiple images, i.e. multi-view inconsistency in editing. This in turn compromises the desired 3D scene editing performance if these images are used. In our work, we propose a novel training-free 3D scene editing technique, Free-Editor, which allows users to edit 3D scenes without further re-training the model during test time. Our proposed method successfully avoids the multi-view style inconsistency issue in SOTA methods with the help of a "single-view editing" scheme. Specifically, we show that editing a particular 3D scene can be performed by only modifying a single view. To this end, we introduce an Edit Transformer that enforces intra-view consistency and inter-view style transfer by utilizing self- and cross-attention, respectively. Since it is no longer required to re-train the model and edit every view in a scene, the editing time, as well as memory resources, are reduced significantly, e.g., the runtime being $\sim \textbf{20} \times$ faster than SOTA. We have conducted extensive experiments on a wide range of benchmark datasets and achieve diverse editing capabilities with our proposed technique.

{{</citation>}}


### (79/116) Weakly Supervised Semantic Segmentation for Driving Scenes (Dongseob Kim et al., 2023)

{{<citation>}}

Dongseob Kim, Seungho Lee, Junsuk Choe, Hyunjung Shim. (2023)  
**Weakly Supervised Semantic Segmentation for Driving Scenes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.13646v2)  

---


**ABSTRACT**  
State-of-the-art techniques in weakly-supervised semantic segmentation (WSSS) using image-level labels exhibit severe performance degradation on driving scene datasets such as Cityscapes. To address this challenge, we develop a new WSSS framework tailored to driving scene datasets. Based on extensive analysis of dataset characteristics, we employ Contrastive Language-Image Pre-training (CLIP) as our baseline to obtain pseudo-masks. However, CLIP introduces two key challenges: (1) pseudo-masks from CLIP lack in representing small object classes, and (2) these masks contain notable noise. We propose solutions for each issue as follows. (1) We devise Global-Local View Training that seamlessly incorporates small-scale patches during model training, thereby enhancing the model's capability to handle small-sized yet critical objects in driving scenes (e.g., traffic light). (2) We introduce Consistency-Aware Region Balancing (CARB), a novel technique that discerns reliable and noisy regions through evaluating the consistency between CLIP masks and segmentation predictions. It prioritizes reliable pixels over noisy pixels via adaptive loss weighting. Notably, the proposed method achieves 51.8\% mIoU on the Cityscapes test dataset, showcasing its potential as a strong WSSS baseline on driving scene datasets. Experimental results on CamVid and WildDash2 demonstrate the effectiveness of our method across diverse datasets, even with small-scale datasets or visually challenging conditions. The code is available at https://github.com/k0u-id/CARB.

{{</citation>}}


### (80/116) LLM4VG: Large Language Models Evaluation for Video Grounding (Wei Feng et al., 2023)

{{<citation>}}

Wei Feng, Xin Wang, Hong Chen, Zeyang Zhang, Zihan Song, Yuwei Zhou, Wenwu Zhu. (2023)  
**LLM4VG: Large Language Models Evaluation for Video Grounding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.14206v1)  

---


**ABSTRACT**  
Recently, researchers have attempted to investigate the capability of LLMs in handling videos and proposed several video LLM models. However, the ability of LLMs to handle video grounding (VG), which is an important time-related video task requiring the model to precisely locate the start and end timestamps of temporal moments in videos that match the given textual queries, still remains unclear and unexplored in literature. To fill the gap, in this paper, we propose the LLM4VG benchmark, which systematically evaluates the performance of different LLMs on video grounding tasks. Based on our proposed LLM4VG, we design extensive experiments to examine two groups of video LLM models on video grounding: (i) the video LLMs trained on the text-video pairs (denoted as VidLLM), and (ii) the LLMs combined with pretrained visual description models such as the video/image captioning model. We propose prompt methods to integrate the instruction of VG and description from different kinds of generators, including caption-based generators for direct visual description and VQA-based generators for information enhancement. We also provide comprehensive comparisons of various VidLLMs and explore the influence of different choices of visual models, LLMs, prompt designs, etc, as well. Our experimental evaluations lead to two conclusions: (i) the existing VidLLMs are still far away from achieving satisfactory video grounding performance, and more time-related video tasks should be included to further fine-tune these models, and (ii) the combination of LLMs and visual models shows preliminary abilities for video grounding with considerable potential for improvement by resorting to more reliable models and further guidance of prompt instructions.

{{</citation>}}


### (81/116) SPGroup3D: Superpoint Grouping Network for Indoor 3D Object Detection (Yun Zhu et al., 2023)

{{<citation>}}

Yun Zhu, Le Hui, Yaqi Shen, Jin Xie. (2023)  
**SPGroup3D: Superpoint Grouping Network for Indoor 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.13641v1)  

---


**ABSTRACT**  
Current 3D object detection methods for indoor scenes mainly follow the voting-and-grouping strategy to generate proposals. However, most methods utilize instance-agnostic groupings, such as ball query, leading to inconsistent semantic information and inaccurate regression of the proposals. To this end, we propose a novel superpoint grouping network for indoor anchor-free one-stage 3D object detection. Specifically, we first adopt an unsupervised manner to partition raw point clouds into superpoints, areas with semantic consistency and spatial similarity. Then, we design a geometry-aware voting module that adapts to the centerness in anchor-free detection by constraining the spatial relationship between superpoints and object centers. Next, we present a superpoint-based grouping module to explore the consistent representation within proposals. This module includes a superpoint attention layer to learn feature interaction between neighboring superpoints, and a superpoint-voxel fusion layer to propagate the superpoint-level information to the voxel level. Finally, we employ effective multiple matching to capitalize on the dynamic receptive fields of proposals based on superpoints during the training. Experimental results demonstrate our method achieves state-of-the-art performance on ScanNet V2, SUN RGB-D, and S3DIS datasets in the indoor one-stage 3D object detection. Source code is available at https://github.com/zyrant/SPGroup3D.

{{</citation>}}


### (82/116) A Comprehensive End-to-End Computer Vision Framework for Restoration and Recognition of Low-Quality Engineering Drawings (Lvyang Yang et al., 2023)

{{<citation>}}

Lvyang Yang, Jiankang Zhang, Huaiqiang Li, Longfei Ren, Chen Yang, Jingyu Wang, Dongyuan Shi. (2023)  
**A Comprehensive End-to-End Computer Vision Framework for Restoration and Recognition of Low-Quality Engineering Drawings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.13620v1)  

---


**ABSTRACT**  
The digitization of engineering drawings is crucial for efficient reuse, distribution, and archiving. Existing computer vision approaches for digitizing engineering drawings typically assume the input drawings have high quality. However, in reality, engineering drawings are often blurred and distorted due to improper scanning, storage, and transmission, which may jeopardize the effectiveness of existing approaches. This paper focuses on restoring and recognizing low-quality engineering drawings, where an end-to-end framework is proposed to improve the quality of the drawings and identify the graphical symbols on them. The framework uses K-means clustering to classify different engineering drawing patches into simple and complex texture patches based on their gray level co-occurrence matrix statistics. Computer vision operations and a modified Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) model are then used to improve the quality of the two types of patches, respectively. A modified Faster Region-based Convolutional Neural Network (Faster R-CNN) model is used to recognize the quality-enhanced graphical symbols. Additionally, a multi-stage task-driven collaborative learning strategy is proposed to train the modified ESRGAN and Faster R-CNN models to improve the resolution of engineering drawings in the direction that facilitates graphical symbol recognition, rather than human visual perception. A synthetic data generation method is also proposed to construct quality-degraded samples for training the framework. Experiments on real-world electrical diagrams show that the proposed framework achieves an accuracy of 98.98% and a recall of 99.33%, demonstrating its superiority over previous approaches. Moreover, the framework is integrated into a widely-used power system software application to showcase its practicality.

{{</citation>}}


### (83/116) DREAM-Talk: Diffusion-based Realistic Emotional Audio-driven Method for Single Image Talking Face Generation (Chenxu Zhang et al., 2023)

{{<citation>}}

Chenxu Zhang, Chao Wang, Jianfeng Zhang, Hongyi Xu, Guoxian Song, You Xie, Linjie Luo, Yapeng Tian, Xiaohu Guo, Jiashi Feng. (2023)  
**DREAM-Talk: Diffusion-based Realistic Emotional Audio-driven Method for Single Image Talking Face Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.13578v1)  

---


**ABSTRACT**  
The generation of emotional talking faces from a single portrait image remains a significant challenge. The simultaneous achievement of expressive emotional talking and accurate lip-sync is particularly difficult, as expressiveness is often compromised for the accuracy of lip-sync. As widely adopted by many prior works, the LSTM network often fails to capture the subtleties and variations of emotional expressions. To address these challenges, we introduce DREAM-Talk, a two-stage diffusion-based audio-driven framework, tailored for generating diverse expressions and accurate lip-sync concurrently. In the first stage, we propose EmoDiff, a novel diffusion module that generates diverse highly dynamic emotional expressions and head poses in accordance with the audio and the referenced emotion style. Given the strong correlation between lip motion and audio, we then refine the dynamics with enhanced lip-sync accuracy using audio features and emotion style. To this end, we deploy a video-to-video rendering module to transfer the expressions and lip motions from our proxy 3D avatar to an arbitrary portrait. Both quantitatively and qualitatively, DREAM-Talk outperforms state-of-the-art methods in terms of expressiveness, lip-sync accuracy and perceptual quality.

{{</citation>}}


### (84/116) ARBiBench: Benchmarking Adversarial Robustness of Binarized Neural Networks (Peng Zhao et al., 2023)

{{<citation>}}

Peng Zhao, Jiehua Zhang, Bowen Peng, Longguang Wang, YingMei Wei, Yu Liu, Li Liu. (2023)  
**ARBiBench: Benchmarking Adversarial Robustness of Binarized Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.13575v1)  

---


**ABSTRACT**  
Network binarization exhibits great potential for deployment on resource-constrained devices due to its low computational cost. Despite the critical importance, the security of binarized neural networks (BNNs) is rarely investigated. In this paper, we present ARBiBench, a comprehensive benchmark to evaluate the robustness of BNNs against adversarial perturbations on CIFAR-10 and ImageNet. We first evaluate the robustness of seven influential BNNs on various white-box and black-box attacks. The results reveal that 1) The adversarial robustness of BNNs exhibits a completely opposite performance on the two datasets under white-box attacks. 2) BNNs consistently exhibit better adversarial robustness under black-box attacks. 3) Different BNNs exhibit certain similarities in their robustness performance. Then, we conduct experiments to analyze the adversarial robustness of BNNs based on these insights. Our research contributes to inspiring future research on enhancing the robustness of BNNs and advancing their application in real-world scenarios.

{{</citation>}}


### (85/116) Efficient Architecture Search via Bi-level Data Pruning (Chongjun Tu et al., 2023)

{{<citation>}}

Chongjun Tu, Peng Ye, Weihao Lin, Hancheng Ye, Chong Yu, Tao Chen, Baopu Li, Wanli Ouyang. (2023)  
**Efficient Architecture Search via Bi-level Data Pruning**  

---
Primary Category: cs.CV  
Categories: 68T05(Primary), cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.14200v1)  

---


**ABSTRACT**  
Improving the efficiency of Neural Architecture Search (NAS) is a challenging but significant task that has received much attention. Previous works mainly adopted the Differentiable Architecture Search (DARTS) and improved its search strategies or modules to enhance search efficiency. Recently, some methods have started considering data reduction for speedup, but they are not tightly coupled with the architecture search process, resulting in sub-optimal performance. To this end, this work pioneers an exploration into the critical role of dataset characteristics for DARTS bi-level optimization, and then proposes a novel Bi-level Data Pruning (BDP) paradigm that targets the weights and architecture levels of DARTS to enhance efficiency from a data perspective. Specifically, we introduce a new progressive data pruning strategy that utilizes supernet prediction dynamics as the metric, to gradually prune unsuitable samples for DARTS during the search. An effective automatic class balance constraint is also integrated into BDP, to suppress potential class imbalances resulting from data-efficient algorithms. Comprehensive evaluations on the NAS-Bench-201 search space, DARTS search space, and MobileNet-like search space validate that BDP reduces search costs by over 50% while achieving superior performance when applied to baseline DARTS. Besides, we demonstrate that BDP can harmoniously integrate with advanced DARTS variants, like PC-DARTS and \b{eta}-DARTS, offering an approximately 2 times speedup with minimal performance compromises.

{{</citation>}}


### (86/116) MR-STGN: Multi-Residual Spatio Temporal Graph Network Using Attention Fusion for Patient Action Assessment (Youssef Mourchid et al., 2023)

{{<citation>}}

Youssef Mourchid, Rim Slama. (2023)  
**MR-STGN: Multi-Residual Spatio Temporal Graph Network Using Attention Fusion for Patient Action Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.13509v1)  

---


**ABSTRACT**  
Accurate assessment of patient actions plays a crucial role in healthcare as it contributes significantly to disease progression monitoring and treatment effectiveness. However, traditional approaches to assess patient actions often rely on manual observation and scoring, which are subjective and time-consuming. In this paper, we propose an automated approach for patient action assessment using a Multi-Residual Spatio Temporal Graph Network (MR-STGN) that incorporates both angular and positional 3D skeletons. The MR-STGN is specifically designed to capture the spatio-temporal dynamics of patient actions. It achieves this by integrating information from multiple residual layers, with each layer extracting features at distinct levels of abstraction. Furthermore, we integrate an attention fusion mechanism into the network, which facilitates the adaptive weighting of various features. This empowers the model to concentrate on the most pertinent aspects of the patient's movements, offering precise instructions regarding specific body parts or movements that require attention. Ablation studies are conducted to analyze the impact of individual components within the proposed model. We evaluate our model on the UI-PRMD dataset demonstrating its performance in accurately predicting real-time patient action scores, surpassing state-of-the-art methods.

{{</citation>}}


### (87/116) InfoVisDial: An Informative Visual Dialogue Dataset by Bridging Large Multimodal and Language Models (Bingbing Wen et al., 2023)

{{<citation>}}

Bingbing Wen, Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Bill Howe, Lijuan Wang. (2023)  
**InfoVisDial: An Informative Visual Dialogue Dataset by Bridging Large Multimodal and Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Dialog, Dialogue, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.13503v1)  

---


**ABSTRACT**  
In this paper, we build a visual dialogue dataset, named InfoVisDial, which provides rich informative answers in each round even with external knowledge related to the visual content. Different from existing datasets where the answer is compact and short, InfoVisDial contains long free-form answers with rich information in each round of dialogue. For effective data collection, the key idea is to bridge the large-scale multimodal model (e.g., GIT) and the language models (e.g., GPT-3). GIT can describe the image content even with scene text, while GPT-3 can generate informative dialogue based on the image description and appropriate prompting techniques. With such automatic pipeline, we can readily generate informative visual dialogue data at scale. Then, we ask human annotators to rate the generated dialogues to filter the low-quality conversations.Human analyses show that InfoVisDial covers informative and diverse dialogue topics: $54.4\%$ of the dialogue rounds are related to image scene texts, and $36.7\%$ require external knowledge. Each round's answer is also long and open-ended: $87.3\%$ of answers are unique with an average length of $8.9$, compared with $27.37\%$ and $2.9$ in VisDial. Last, we propose a strong baseline by adapting the GIT model for the visual dialogue task and fine-tune the model on InfoVisDial. Hopefully, our work can motivate more effort on this direction.

{{</citation>}}


## physics.comp-ph (1)



### (88/116) AI-Lorenz: A physics-data-driven framework for black-box and gray-box identification of chaotic systems with symbolic regression (Mario De Florio et al., 2023)

{{<citation>}}

Mario De Florio, Ioannis G. Kevrekidis, George Em Karniadakis. (2023)  
**AI-Lorenz: A physics-data-driven framework for black-box and gray-box identification of chaotic systems with symbolic regression**  

---
Primary Category: physics.comp-ph  
Categories: 34A34, 34A55, 70K55, J-2; G-1-7; I-2-0, cs-LG, nlin-CD, physics-comp-ph, physics-data-an, physics.comp-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14237v1)  

---


**ABSTRACT**  
Discovering mathematical models that characterize the observed behavior of dynamical systems remains a major challenge, especially for systems in a chaotic regime. The challenge is even greater when the physics underlying such systems is not yet understood, and scientific inquiry must solely rely on empirical data. Driven by the need to fill this gap, we develop a framework that learns mathematical expressions modeling complex dynamical behaviors by identifying differential equations from noisy and sparse observable data. We train a small neural network to learn the dynamics of a system, its rate of change in time, and missing model terms, which are used as input for a symbolic regression algorithm to autonomously distill the explicit mathematical terms. This, in turn, enables us to predict the future evolution of the dynamical behavior. The performance of this framework is validated by recovering the right-hand sides and unknown terms of certain complex, chaotic systems such as the well-known Lorenz system, a six-dimensional hyperchaotic system, and the non-autonomous Sprott chaotic system, and comparing them with their known analytical expressions.

{{</citation>}}


## cs.RO (8)



### (89/116) LingoQA: Video Question Answering for Autonomous Driving (Ana-Maria Marcu et al., 2023)

{{<citation>}}

Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, Oleg Sinavski. (2023)  
**LingoQA: Video Question Answering for Autonomous Driving**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-RO, cs.RO  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.14115v1)  

---


**ABSTRACT**  
Autonomous driving has long faced a challenge with public acceptance due to the lack of explainability in the decision-making process. Video question-answering (QA) in natural language provides the opportunity for bridging this gap. Nonetheless, evaluating the performance of Video QA models has proved particularly tough due to the absence of comprehensive benchmarks. To fill this gap, we introduce LingoQA, a benchmark specifically for autonomous driving Video QA. The LingoQA trainable metric demonstrates a 0.95 Spearman correlation coefficient with human evaluations. We introduce a Video QA dataset of central London consisting of 419k samples that we release with the paper. We establish a baseline vision-language model and run extensive ablation studies to understand its performance.

{{</citation>}}


### (90/116) Multi-Agent Probabilistic Ensembles with Trajectory Sampling for Connected Autonomous Vehicles (Ruoqi Wen et al., 2023)

{{<citation>}}

Ruoqi Wen, Jiahao Huang, Rongpeng Li, Guoru Ding, Zhifeng Zhao. (2023)  
**Multi-Agent Probabilistic Ensembles with Trajectory Sampling for Connected Autonomous Vehicles**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-MA, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.13910v1)  

---


**ABSTRACT**  
Autonomous Vehicles (AVs) have attracted significant attention in recent years and Reinforcement Learning (RL) has shown remarkable performance in improving the autonomy of vehicles. In that regard, the widely adopted Model-Free RL (MFRL) promises to solve decision-making tasks in connected AVs (CAVs), contingent on the readiness of a significant amount of data samples for training. Nevertheless, it might be infeasible in practice and possibly lead to learning instability. In contrast, Model-Based RL (MBRL) manifests itself in sample-efficient learning, but the asymptotic performance of MBRL might lag behind the state-of-the-art MFRL algorithms. Furthermore, most studies for CAVs are limited to the decision-making of a single AV only, thus underscoring the performance due to the absence of communications. In this study, we try to address the decision-making problem of multiple CAVs with limited communications and propose a decentralized Multi-Agent Probabilistic Ensembles with Trajectory Sampling algorithm MA-PETS. In particular, in order to better capture the uncertainty of the unknown environment, MA-PETS leverages Probabilistic Ensemble (PE) neural networks to learn from communicated samples among neighboring CAVs. Afterwards, MA-PETS capably develops Trajectory Sampling (TS)-based model-predictive control for decision-making. On this basis, we derive the multi-agent group regret bound affected by the number of agents within the communication range and mathematically validate that incorporating effective information exchange among agents into the multi-agent learning scheme contributes to reducing the group regret bound in the worst case. Finally, we empirically demonstrate the superiority of MA-PETS in terms of the sample efficiency comparable to MFBL.

{{</citation>}}


### (91/116) Domain-Specific Fine-Tuning of Large Language Models for Interactive Robot Programming (Benjamin Alt et al., 2023)

{{<citation>}}

Benjamin Alt, Urs Keßner, Aleksandar Taranovic, Darko Katic, Andreas Hermann, Rainer Jäkel, Gerhard Neumann. (2023)  
**Domain-Specific Fine-Tuning of Large Language Models for Interactive Robot Programming**  

---
Primary Category: cs.RO  
Categories: 68T40, I-2-9; I-2-5; I-2-6; I-2-7, cs-AI, cs-CL, cs-HC, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.13905v1)  

---


**ABSTRACT**  
Industrial robots are applied in a widening range of industries, but robot programming mostly remains a task limited to programming experts. We propose a natural language-based assistant for programming of advanced, industrial robotic applications and investigate strategies for domain-specific fine-tuning of foundation models with limited data and compute.

{{</citation>}}


### (92/116) A Summarized History-based Dialogue System for Amnesia-Free Prompt Updates (Hyejin Hong et al., 2023)

{{<citation>}}

Hyejin Hong, Hibiki Kawano, Takuto Maekawa, Naoki Yoshimaru, Takamasa Iio, Kenji Hatano. (2023)  
**A Summarized History-based Dialogue System for Amnesia-Free Prompt Updates**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.13891v1)  

---


**ABSTRACT**  
In today's society, information overload presents challenges in providing optimal recommendations. Consequently, the importance of dialogue systems that can discern and provide the necessary information through dialogue is increasingly recognized. However, some concerns existing dialogue systems rely on pre-trained models and need help to cope with real-time or insufficient information. To address these concerns, models that allow the addition of missing information to dialogue robots are being proposed. Yet, maintaining the integrity of previous conversation history while integrating new data remains a formidable challenge. This paper presents a novel system for dialogue robots designed to remember user-specific characteristics by retaining past conversation history even as new information is added.

{{</citation>}}


### (93/116) Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator (Zichun Xu et al., 2023)

{{<citation>}}

Zichun Xu, Yuntao Li, Xiaohang Yang, Zhiyuan Zhao, Lei Zhuang, Jingdong Zhao. (2023)  
**Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.13788v1)  

---


**ABSTRACT**  
This paper presents three open-source reinforcement learning environments developed on the MuJoCo physics engine with the Franka Emika Panda arm in MuJoCo Menagerie. Three representative tasks, push, slide, and pick-and-place, are implemented through the Gymnasium Robotics API, which inherits from the core of Gymnasium. Both the sparse binary and dense rewards are supported, and the observation space contains the keys of desired and achieved goals to follow the Multi-Goal Reinforcement Learning framework. Three different off-policy algorithms are used to validate the simulation attributes to ensure the fidelity of all tasks, and benchmark results are also given. Each environment and task are defined in a clean way, and the main parameters for modifying the environment are preserved to reflect the main difference. The repository, including all environments, is available at https://github.com/zichunxx/panda_mujoco_gym.

{{</citation>}}


### (94/116) Team Irisapu Project Description for DRC2023 (Reon Ohashi et al., 2023)

{{<citation>}}

Reon Ohashi, Shinjitsu Agatsuma, Kazuya Tsubokura, Yurie Iribe. (2023)  
**Team Irisapu Project Description for DRC2023**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI, Dialog, Dialogue, GPT  
[Paper Link](http://arxiv.org/abs/2312.13765v1)  

---


**ABSTRACT**  
This paper describes the dialog robot system designed by Team Irisapu for the preliminary round of the Dialogue Robot Competition 2023 (DRC2023). In order to generate dialogue responses flexibly while adhering to predetermined scenarios, we attempted to generate dialogue response sentences using OpenAI's GPT-3. We aimed to create a system that can appropriately respond to users by dividing the dialogue scenario into five sub-scenarios, and creating prompts for each sub-scenario. Also, we incorporated a recovery strategy that can handle dialogue breakdowns flexibly. Our research group has been working on research related to dialogue breakdown detection, and we incorporated our findings to date in this competition. As a result of the preliminary round, a bug in our system affected the outcome and we were not able to achieve a satisfactory result. However, in the evaluation category of "reliability of provided information", we ranked third among all teams.

{{</citation>}}


### (95/116) Meta-control of Dialogue Systems Using Large Language Models (Kotaro Shukuri et al., 2023)

{{<citation>}}

Kotaro Shukuri, Ryoma Ishigaki, Jundai Suzuki, Tsubasa Naganuma, Takuma Fujimoto, Daisuke Kawakubo, Masaki Shuzo, Eisaku Maeda. (2023)  
**Meta-control of Dialogue Systems Using Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2312.13715v1)  

---


**ABSTRACT**  
Utilizing Large Language Models (LLMs) facilitates the creation of flexible and natural dialogues, a task that has been challenging with traditional rule-based dialogue systems. However, LLMs also have the potential to produce unexpected responses, which may not align with the intentions of dialogue system designers. To address this issue, this paper introduces a meta-control method that employs LLMs to develop more stable and adaptable dialogue systems. The method includes dialogue flow control to ensure that utterances conform to predefined scenarios and turn-taking control to foster natural dialogues. Furthermore, we have implemented a dialogue system that utilizes this meta-control strategy and verified that the dialogue system utilizing meta-control operates as intended.

{{</citation>}}


### (96/116) Compositional Zero-Shot Learning for Attribute-Based Object Reference in Human-Robot Interaction (Peng Gao et al., 2023)

{{<citation>}}

Peng Gao, Ahmed Jaafar, Brian Reily, Christopher Reardon, Hao Zhang. (2023)  
**Compositional Zero-Shot Learning for Attribute-Based Object Reference in Human-Robot Interaction**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-CV, cs-RO, cs.RO  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.13655v1)  

---


**ABSTRACT**  
Language-enabled robots have been widely studied over the past years to enable natural human-robot interaction and teaming in various real-world applications. Language-enabled robots must be able to comprehend referring expressions to identify a particular object from visual perception using a set of referring attributes extracted from natural language. However, visual observations of an object may not be available when it is referred to, and the number of objects and attributes may also be unbounded in open worlds. To address the challenges, we implement an attribute-based compositional zero-shot learning method that uses a list of attributes to perform referring expression comprehension in open worlds. We evaluate the approach on two datasets including the MIT-States and the Clothing 16K. The preliminary experimental results show that our implemented approach allows a robot to correctly identify the objects referred to by human commands.

{{</citation>}}


## stat.AP (1)



### (97/116) RetailSynth: Synthetic Data Generation for Retail AI Systems Evaluation (Yu Xia et al., 2023)

{{<citation>}}

Yu Xia, Ali Arian, Sriram Narayanamoorthy, Joshua Mabry. (2023)  
**RetailSynth: Synthetic Data Generation for Retail AI Systems Evaluation**  

---
Primary Category: stat.AP  
Categories: cs-AI, cs-LG, econ-EM, stat-AP, stat.AP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14095v1)  

---


**ABSTRACT**  
Significant research effort has been devoted in recent years to developing personalized pricing, promotions, and product recommendation algorithms that can leverage rich customer data to learn and earn. Systematic benchmarking and evaluation of these causal learning systems remains a critical challenge, due to the lack of suitable datasets and simulation environments. In this work, we propose a multi-stage model for simulating customer shopping behavior that captures important sources of heterogeneity, including price sensitivity and past experiences. We embedded this model into a working simulation environment -- RetailSynth. RetailSynth was carefully calibrated on publicly available grocery data to create realistic synthetic shopping transactions. Multiple pricing policies were implemented within the simulator and analyzed for impact on revenue, category penetration, and customer retention. Applied researchers can use RetailSynth to validate causal demand models for multi-category retail and to incorporate realistic price sensitivity into emerging benchmarking suites for personalized pricing, promotions, and product recommendations.

{{</citation>}}


## cs.CY (2)



### (98/116) Don't slip into binary thinking about AI (Thorin Bristow et al., 2023)

{{<citation>}}

Thorin Bristow, Luke Thorburn. (2023)  
**Don't slip into binary thinking about AI**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14230v1)  

---


**ABSTRACT**  
In discussions about the development and governance of AI, a false binary is often drawn between two groups: those most concerned about the existing, social impacts of AI, and those most concerned about possible future risks of powerful AI systems taking actions that don't align with human interests. In this piece, we (i) describe the emergence of this false binary, (ii) explain why the seemingly clean distinctions drawn between these two groups don't hold up under scrutiny and (iii) highlight efforts to bridge this divide.

{{</citation>}}


### (99/116) How Does Connecting Online Activities to Advertising Inferences Impact Privacy Perceptions? (Florian M. Farke et al., 2023)

{{<citation>}}

Florian M. Farke, David G. Balash, Maximilian Golla, Adam J. Aviv. (2023)  
**How Does Connecting Online Activities to Advertising Inferences Impact Privacy Perceptions?**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs.CY  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.13813v1)  

---


**ABSTRACT**  
Data dashboards are designed to help users manage data collected about them. However, prior work showed that exposure to some dashboards, notably Google's My Activity dashboard, results in significant decreases in perceived concern and increases in perceived benefit from data collection, contrary to expectations. We theorize that this result is due to the fact that data dashboards currently do not sufficiently "connect the dots" of the data food chain, that is, by connecting data collection with the use of that data. To evaluate this, we designed a study where participants assigned advertising interest labels to their own real activities, effectively acting as a behavioral advertising engine to "connect the dots." When comparing pre- and post-labeling task responses, we find no significant difference in concern with Google's data collection practices, which indicates that participants' priors are maintained after more exposure to the data food chain (differing from prior work), suggesting that data dashboards that offer deeper perspectives of how data collection is used have potential. However, these gains are offset when participants are exposed to their true interest labels inferred by Google. Concern for data collection dropped significantly as participants viewed Google's labeling as generic compared to their own more specific labeling. This presents a possible new paradox that must be overcome when designing data dashboards, the generic paradox, which occurs when users misalign individual, generic inferences from collected data as benign compared to the totality and specificity of many generic inferences made about them.

{{</citation>}}


## cs.HC (5)



### (100/116) BANSpEmo: A Bangla Emotional Speech Recognition Dataset (Md Gulzar Hussain et al., 2023)

{{<citation>}}

Md Gulzar Hussain, Mahmuda Rahman, Babe Sultana, Ye Shiren. (2023)  
**BANSpEmo: A Bangla Emotional Speech Recognition Dataset**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-LG, cs-SD, cs.HC, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.14020v1)  

---


**ABSTRACT**  
In the field of audio and speech analysis, the ability to identify emotions from acoustic signals is essential. Human-computer interaction (HCI) and behavioural analysis are only a few of the many areas where the capacity to distinguish emotions from speech signals has an extensive range of applications. Here, we are introducing BanSpEmo, a corpus of emotional speech that only consists of audio recordings and has been created specifically for the Bangla language. This corpus contains 792 audio recordings over a duration of more than 1 hour and 23 minutes. 22 native speakers took part in the recording of two sets of sentences that represent the six desired emotions. The data set consists of 12 Bangla sentences which are uttered in 6 emotions as Disgust, Happy, Sad, Surprised, Anger, and Fear. This corpus is not also gender balanced. Ten individuals who either have experience in related field or have acting experience took part in the assessment of this corpus. It has a balanced number of audio recordings in each emotion class. BanSpEmo can be considered as a useful resource to promote emotion and speech recognition research and related applications in the Bangla language. The dataset can be found here: https://data.mendeley.com/datasets/rdwn4bs5ky and might be employed for academic research.

{{</citation>}}


### (101/116) AsyncMLD: Asynchronous Multi-LLM Framework for Dialogue Recommendation System (Naoki Yoshimaru et al., 2023)

{{<citation>}}

Naoki Yoshimaru, Motoharu Okuma, Takamasa Iio, Kenji Hatano. (2023)  
**AsyncMLD: Asynchronous Multi-LLM Framework for Dialogue Recommendation System**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-RO, cs.HC  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.13925v1)  

---


**ABSTRACT**  
We have reached a practical and realistic phase in human-support dialogue agents by developing a large language model (LLM). However, when requiring expert knowledge or anticipating the utterance content using the massive size of the dialogue database, we still need help with the utterance content's effectiveness and the efficiency of its output speed, even if using LLM. Therefore, we propose a framework that uses LLM asynchronously in the part of the system that returns an appropriate response and in the part that understands the user's intention and searches the database. In particular, noting that it takes time for the robot to speak, threading related to database searches is performed while the robot is speaking.

{{</citation>}}


### (102/116) User-adaptive Tourist Information Dialogue System with Yes/No Classifier and Sentiment Estimator (Ryo Yanagimoto et al., 2023)

{{<citation>}}

Ryo Yanagimoto, Yunosuke Kubo, Miki Oshio, Mikio Nakano, Kenta Yamamoto, Kazunori Komatani. (2023)  
**User-adaptive Tourist Information Dialogue System with Yes/No Classifier and Sentiment Estimator**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: BERT, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.13787v1)  

---


**ABSTRACT**  
We introduce our system developed for Dialogue Robot Competition 2023 (DRC2023). First, rule-based utterance selection and utterance generation using a large language model (LLM) are combined. We ensure the quality of system utterances while also being able to respond to unexpected user utterances. Second, dialogue flow is controlled by considering the results of the BERT-based yes/no classifier and sentiment estimator. These allow the system to adapt state transitions and sightseeing plans to the user.

{{</citation>}}


### (103/116) Dialogue System of Team NTT-EASE for DRC2023 (Yuki Kubo et al., 2023)

{{<citation>}}

Yuki Kubo, Tomoya Yamashita, Masanori Yamada. (2023)  
**Dialogue System of Team NTT-EASE for DRC2023**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Dialog, Dialogue, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2312.13734v1)  

---


**ABSTRACT**  
We developed a dialogue system as a team NTT-EASE in the Dialogue Robot Competition 2023 (DRC2023). We introduce a dialogue system (EASE-DRCBot) constructed for DRC2023. EASE-DRCBot incorporates a manually defined dialogue flow. The conditions for system utterances are based on keyword extraction, example-based method, and sentiment analysis. For answering a user's question, EASE-DRCBot utilizes GPT-3.5 to generate responses. We analyze the results of the preliminary round and explain future works.

{{</citation>}}


### (104/116) Understanding the Role of Large Language Models in Personalizing and Scaffolding Strategies to Combat Academic Procrastination (Ananya Bhattacharjee et al., 2023)

{{<citation>}}

Ananya Bhattacharjee, Yuchen Zeng, Sarah Yi Xu, Dana Kulzhabayeva, Minyi Ma, Rachel Kornfield, Syed Ishtiaque Ahmed, Alex Mariakakis, Mary P Czerwinski, Anastasia Kuzminykh, Michael Liut, Joseph Jay Williams. (2023)  
**Understanding the Role of Large Language Models in Personalizing and Scaffolding Strategies to Combat Academic Procrastination**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.13581v1)  

---


**ABSTRACT**  
Traditional interventions for academic procrastination often fail to capture the nuanced, individual-specific factors that underlie them. Large language models (LLMs) hold immense potential for addressing this gap by permitting open-ended inputs, including the ability to customize interventions to individuals' unique needs. However, user expectations and potential limitations of LLMs in this context remain underexplored. To address this, we conducted interviews and focus group discussions with 15 university students and 6 experts, during which a technology probe for generating personalized advice for managing procrastination was presented. Our results highlight the necessity for LLMs to provide structured, deadline-oriented steps and enhanced user support mechanisms. Additionally, our results surface the need for an adaptive approach to questioning based on factors like busyness. These findings offer crucial design implications for the development of LLM-based tools for managing procrastination while cautioning the use of LLMs for therapeutic guidance.

{{</citation>}}


## cs.SD (5)



### (105/116) On the choice of the optimal temporal support for audio classification with Pre-trained embeddings (Aurian Quelennec et al., 2023)

{{<citation>}}

Aurian Quelennec, Michel Olvera, Geoffroy Peeters, Slim Essid. (2023)  
**On the choice of the optimal temporal support for audio classification with Pre-trained embeddings**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.14005v1)  

---


**ABSTRACT**  
Current state-of-the-art audio analysis systems rely on pre-trained embedding models, often used off-the-shelf as (frozen) feature extractors. Choosing the best one for a set of tasks is the subject of many recent publications. However, one aspect often overlooked in these works is the influence of the duration of audio input considered to extract an embedding, which we refer to as Temporal Support (TS). In this work, we study the influence of the TS for well-established or emerging pre-trained embeddings, chosen to represent different types of architectures and learning paradigms. We conduct this evaluation using both musical instrument and environmental sound datasets, namely OpenMIC, TAU Urban Acoustic Scenes 2020 Mobile, and ESC-50. We especially highlight that Audio Spectrogram Transformer-based systems (PaSST and BEATs) remain effective with smaller TS, which therefore allows for a drastic reduction in memory and computational cost. Moreover, we show that by choosing the optimal TS we reach competitive results across all tasks. In particular, we improve the state-of-the-art results on OpenMIC, using BEATs and PaSST without any fine-tuning.

{{</citation>}}


### (106/116) Self-Supervised Adaptive AV Fusion Module for Pre-Trained ASR Models (Christopher Simic et al., 2023)

{{<citation>}}

Christopher Simic, Tobias Bocklet. (2023)  
**Self-Supervised Adaptive AV Fusion Module for Pre-Trained ASR Models**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: BERT, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.13873v1)  

---


**ABSTRACT**  
Automatic speech recognition (ASR) has reached a level of accuracy in recent years, that even outperforms humans in transcribing speech to text. Nevertheless, all current ASR approaches show a certain weakness against ambient noise. To reduce this weakness, audio-visual speech recognition (AVSR) approaches additionally consider visual information from lip movements for transcription. This additional modality increases the computational cost for training models from scratch. We propose an approach, that builds on a pre-trained ASR model and extends it with an adaptive upstream module, that fuses audio and visual information. Since we do not need to train the transformer structure from scratch, our approach requires a fraction of the computational resources compared to traditional AVSR models. Compared to current SOTA systems like AV-HuBERT, our approach achieves an average improvement of 8.3% in word error rate across different model sizes, noise categories and broad SNR range. The approach allows up to 21% smaller models and requires only a fraction of the computational resources for training and inference compared to common AVSR approaches.

{{</citation>}}


### (107/116) Fine-grained Disentangled Representation Learning for Multimodal Emotion Recognition (Haoqin Sun et al., 2023)

{{<citation>}}

Haoqin Sun, Shiwan Zhao, Xuechen Wang, Wenjia Zeng, Yong Chen, Yong Qin. (2023)  
**Fine-grained Disentangled Representation Learning for Multimodal Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.13567v1)  

---


**ABSTRACT**  
Multimodal emotion recognition (MMER) is an active research field that aims to accurately recognize human emotions by fusing multiple perceptual modalities. However, inherent heterogeneity across modalities introduces distribution gaps and information redundancy, posing significant challenges for MMER. In this paper, we propose a novel fine-grained disentangled representation learning (FDRL) framework to address these challenges. Specifically, we design modality-shared and modality-private encoders to project each modality into modality-shared and modality-private subspaces, respectively. In the shared subspace, we introduce a fine-grained alignment component to learn modality-shared representations, thus capturing modal consistency. Subsequently, we tailor a fine-grained disparity component to constrain the private subspaces, thereby learning modality-private representations and enhancing their diversity. Lastly, we introduce a fine-grained predictor component to ensure that the labels of the output representations from the encoders remain unchanged. Experimental results on the IEMOCAP dataset show that FDRL outperforms the state-of-the-art methods, achieving 78.34% and 79.44% on WAR and UAR, respectively.

{{</citation>}}


### (108/116) kNN-CTC: Enhancing ASR via Retrieval of CTC Pseudo Labels (Jiaming Zhou et al., 2023)

{{<citation>}}

Jiaming Zhou, Shiwan Zhao, Yaqi Liu, Wenjia Zeng, Yong Chen, Yong Qin. (2023)  
**kNN-CTC: Enhancing ASR via Retrieval of CTC Pseudo Labels**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.13560v1)  

---


**ABSTRACT**  
The success of retrieval-augmented language models in various natural language processing (NLP) tasks has been constrained in automatic speech recognition (ASR) applications due to challenges in constructing fine-grained audio-text datastores. This paper presents kNN-CTC, a novel approach that overcomes these challenges by leveraging Connectionist Temporal Classification (CTC) pseudo labels to establish frame-level audio-text key-value pairs, circumventing the need for precise ground truth alignments. We further introduce a skip-blank strategy, which strategically ignores CTC blank frames, to reduce datastore size. kNN-CTC incorporates a k-nearest neighbors retrieval mechanism into pre-trained CTC ASR systems, achieving significant improvements in performance. By incorporating a k-nearest neighbors retrieval mechanism into pre-trained CTC ASR systems and leveraging a fine-grained, pruned datastore, kNN-CTC consistently achieves substantial improvements in performance under various experimental settings. Our code is available at https://github.com/NKU-HLT/KNN-CTC.

{{</citation>}}


### (109/116) Multi-Level Knowledge Distillation for Speech Emotion Recognition in Noisy Conditions (Yang Liu et al., 2023)

{{<citation>}}

Yang Liu, Haoqin Sun, Geng Chen, Qingyue Wang, Zhen Zhao, Xugang Lu, Longbiao Wang. (2023)  
**Multi-Level Knowledge Distillation for Speech Emotion Recognition in Noisy Conditions**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.13556v1)  

---


**ABSTRACT**  
Speech emotion recognition (SER) performance deteriorates significantly in the presence of noise, making it challenging to achieve competitive performance in noisy conditions. To this end, we propose a multi-level knowledge distillation (MLKD) method, which aims to transfer the knowledge from a teacher model trained on clean speech to a simpler student model trained on noisy speech. Specifically, we use clean speech features extracted by the wav2vec-2.0 as the learning goal and train the distil wav2vec-2.0 to approximate the feature extraction ability of the original wav2vec-2.0 under noisy conditions. Furthermore, we leverage the multi-level knowledge of the original wav2vec-2.0 to supervise the single-level output of the distil wav2vec-2.0. We evaluate the effectiveness of our proposed method by conducting extensive experiments using five types of noise-contaminated speech on the IEMOCAP dataset, which show promising results compared to state-of-the-art models.

{{</citation>}}


## eess.IV (2)



### (110/116) Hunting imaging biomarkers in pulmonary fibrosis: Benchmarks of the AIIB23 challenge (Yang Nan et al., 2023)

{{<citation>}}

Yang Nan, Xiaodan Xing, Shiyi Wang, Zeyu Tang, Federico N Felder, Sheng Zhang, Roberta Eufrasia Ledda, Xiaoliu Ding, Ruiqi Yu, Weiping Liu, Feng Shi, Tianyang Sun, Zehong Cao, Minghui Zhang, Yun Gu, Hanxiao Zhang, Jian Gao, Wen Tang, Pengxin Yu, Han Kang, Junqiang Chen, Xing Lu, Boyu Zhang, Michail Mamalakis, Francesco Prinzi, Gianluca Carlini, Lisa Cuneo, Abhirup Banerjee, Zhaohu Xing, Lei Zhu, Zacharia Mesbah, Dhruv Jain, Tsiry Mayet, Hongyu Yuan, Qing Lyu, Athol Wells, Simon LF Walsh, Guang Yang. (2023)  
**Hunting imaging biomarkers in pulmonary fibrosis: Benchmarks of the AIIB23 challenge**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.13752v1)  

---


**ABSTRACT**  
Airway-related quantitative imaging biomarkers are crucial for examination, diagnosis, and prognosis in pulmonary diseases. However, the manual delineation of airway trees remains prohibitively time-consuming. While significant efforts have been made towards enhancing airway modelling, current public-available datasets concentrate on lung diseases with moderate morphological variations. The intricate honeycombing patterns present in the lung tissues of fibrotic lung disease patients exacerbate the challenges, often leading to various prediction errors. To address this issue, the 'Airway-Informed Quantitative CT Imaging Biomarker for Fibrotic Lung Disease 2023' (AIIB23) competition was organized in conjunction with the official 2023 International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI). The airway structures were meticulously annotated by three experienced radiologists. Competitors were encouraged to develop automatic airway segmentation models with high robustness and generalization abilities, followed by exploring the most correlated QIB of mortality prediction. A training set of 120 high-resolution computerised tomography (HRCT) scans were publicly released with expert annotations and mortality status. The online validation set incorporated 52 HRCT scans from patients with fibrotic lung disease and the offline test set included 140 cases from fibrosis and COVID-19 patients. The results have shown that the capacity of extracting airway trees from patients with fibrotic lung disease could be enhanced by introducing voxel-wise weighted general union loss and continuity loss. In addition to the competitive image biomarkers for prognosis, a strong airway-derived biomarker (Hazard ratio>1.5, p<0.0001) was revealed for survival prognostication compared with existing clinical measurements, clinician assessment and AI-based biomarkers.

{{</citation>}}


### (111/116) Meta Transfer of Self-Supervised Knowledge: Foundation Model in Action for Post-Traumatic Epilepsy Prediction (Wenhui Cui et al., 2023)

{{<citation>}}

Wenhui Cui, Haleh Akrami, Ganning Zhao, Anand A. Joshi, Richard M. Leahy. (2023)  
**Meta Transfer of Self-Supervised Knowledge: Foundation Model in Action for Post-Traumatic Epilepsy Prediction**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess.IV, q-bio-NC  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.14204v1)  

---


**ABSTRACT**  
Despite the impressive advancements achieved using deep-learning for functional brain activity analysis, the heterogeneity of functional patterns and scarcity of imaging data still pose challenges in tasks such as prediction of future onset of Post-Traumatic Epilepsy (PTE) from data acquired shortly after traumatic brain injury (TBI). Foundation models pre-trained on separate large-scale datasets can improve the performance from scarce and heterogeneous datasets. For functional Magnetic Resonance Imaging (fMRI), while data may be abundantly available from healthy controls, clinical data is often scarce, limiting the ability of foundation models to identify clinically-relevant features. We overcome this limitation by introducing a novel training strategy for our foundation model by integrating meta-learning with self-supervised learning to improve the generalization from normal to clinical features. In this way we enable generalization to other downstream clinical tasks, in our case prediction of PTE. To achieve this, we perform self-supervised training on the control dataset to focus on inherent features that are not limited to a particular supervised task while applying meta-learning, which strongly improves the model's generalizability using bi-level optimization. Through experiments on neurological disorder classification tasks, we demonstrate that the proposed strategy significantly improves task performance on small-scale clinical datasets. To explore the generalizability of the foundation model in downstream applications, we then apply the model to an unseen TBI dataset for prediction of PTE using zero-shot learning. Results further demonstrated the enhanced generalizability of our foundation model.

{{</citation>}}


## eess.AS (2)



### (112/116) Self-supervised Complex Network for Machine Sound Anomaly Detection (Miseul Kim et al., 2023)

{{<citation>}}

Miseul Kim, Minh Tri Ho, Hong-Goo Kang. (2023)  
**Self-supervised Complex Network for Machine Sound Anomaly Detection**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess-SP, eess.AS  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.13615v1)  

---


**ABSTRACT**  
In this paper, we propose an anomaly detection algorithm for machine sounds with a deep complex network trained by self-supervision. Using the fact that phase continuity information is crucial for detecting abnormalities in time-series signals, our proposed algorithm utilizes the complex spectrum as an input and performs complex number arithmetic throughout the entire process. Since the usefulness of phase information can vary depending on the type of machine sound, we also apply an attention mechanism to control the weights of the complex and magnitude spectrum bottleneck features depending on the machine type. We train our network to perform a self-supervised task that classifies the machine identifier (id) of normal input sounds among multiple classes. At test time, an input signal is detected as anomalous if the trained model is unable to correctly classify the id. In other words, we determine the presence of an anomality when the output cross-entropy score of the multiclass identification task is lower than a pre-defined threshold. Experiments with the MIMII dataset show that the proposed algorithm has a much higher area under the curve (AUC) score than conventional magnitude spectrum-based algorithms.

{{</citation>}}


### (113/116) BrainTalker: Low-Resource Brain-to-Speech Synthesis with Transfer Learning using Wav2Vec 2.0 (Miseul Kim et al., 2023)

{{<citation>}}

Miseul Kim, Zhenyu Piao, Jihyun Lee, Hong-Goo Kang. (2023)  
**BrainTalker: Low-Resource Brain-to-Speech Synthesis with Transfer Learning using Wav2Vec 2.0**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Low-Resource  
[Paper Link](http://arxiv.org/abs/2312.13600v1)  

---


**ABSTRACT**  
Decoding spoken speech from neural activity in the brain is a fast-emerging research topic, as it could enable communication for people who have difficulties with producing audible speech. For this task, electrocorticography (ECoG) is a common method for recording brain activity with high temporal resolution and high spatial precision. However, due to the risky surgical procedure required for obtaining ECoG recordings, relatively little of this data has been collected, and the amount is insufficient to train a neural network-based Brain-to-Speech (BTS) system. To address this problem, we propose BrainTalker-a novel BTS framework that generates intelligible spoken speech from ECoG signals under extremely low-resource scenarios. We apply a transfer learning approach utilizing a pre-trained self supervised model, Wav2Vec 2.0. Specifically, we train an encoder module to map ECoG signals to latent embeddings that match Wav2Vec 2.0 representations of the corresponding spoken speech. These embeddings are then transformed into mel-spectrograms using stacked convolutional and transformer-based layers, which are fed into a neural vocoder to synthesize speech waveform. Experimental results demonstrate our proposed framework achieves outstanding performance in terms of subjective and objective metrics, including a Pearson correlation coefficient of 0.9 between generated and ground truth mel spectrograms. We share publicly available Demos and Code.

{{</citation>}}


## q-fin.PM (1)



### (114/116) Shai: A large language model for asset management (Zhongyang Guo et al., 2023)

{{<citation>}}

Zhongyang Guo, Guanran Jiang, Zhongdan Zhang, Peng Li, Zhefeng Wang, Yinchun Wang. (2023)  
**Shai: A large language model for asset management**  

---
Primary Category: q-fin.PM  
Categories: cs-CL, cs-LG, q-fin-PM, q-fin.PM  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.14203v1)  

---


**ABSTRACT**  
This paper introduces "Shai" a 10B level large language model specifically designed for the asset management industry, built upon an open-source foundational model. With continuous pre-training and fine-tuning using a targeted corpus, Shai demonstrates enhanced performance in tasks relevant to its domain, outperforming baseline models. Our research includes the development of an innovative evaluation framework, which integrates professional qualification exams, tailored tasks, open-ended question answering, and safety assessments, to comprehensively assess Shai's capabilities. Furthermore, we discuss the challenges and implications of utilizing large language models like GPT-4 for performance assessment in asset management, suggesting a combination of automated evaluation and human judgment. Shai's development, showcasing the potential and versatility of 10B-level large language models in the financial sector with significant performance and modest computational requirements, hopes to provide practical insights and methodologies to assist industry peers in their similar endeavors.

{{</citation>}}


## q-bio.GN (1)



### (115/116) Using GPT-4 Prompts to Determine Whether Articles Contain Functional Evidence Supporting or Refuting Variant Pathogenicity (Samuel J. Aronson et al., 2023)

{{<citation>}}

Samuel J. Aronson, Kalotina Machini, Pranav Sriraman, Jiyeon Shin, Emma R. Henricks, Charlotte Mailly, Angie J. Nottage, Michael Oates, Matthew S. Lebo. (2023)  
**Using GPT-4 Prompts to Determine Whether Articles Contain Functional Evidence Supporting or Refuting Variant Pathogenicity**  

---
Primary Category: q-bio.GN  
Categories: cs-AI, q-bio-GN, q-bio.GN  
Keywords: GPT, GPT-4, Transformer  
[Paper Link](http://arxiv.org/abs/2312.13521v1)  

---


**ABSTRACT**  
Purpose: To assess Generative Pre-trained Transformer version 4's (GPT-4) ability to classify articles containing functional evidence relevant to assessments of variant pathogenicity.   Results: GPT-4 settings and prompts were trained on a set of 45 articles and genetic variants. A final test set of 72 manually classified articles and genetic variants were then processed using two prompts. The prompts asked GPT-4 to supply all functional evidence present in an article for a variant or indicate that no functional evidence is present. For articles with having functional evidence, a second prompt asked GPT-4 to classify the evidence into pathogenic, benign, intermediate, and inconclusive categories. The first prompt identified articles with variant-level functional evidence with 87% sensitivity and 89% positive predictive value (PPV). Five of 26 articles with no functional data were indicated as having functional evidence by GPT-4. For variants with functional assays present as determined by both manual review and GPT-4, the sensitivity and PPV of GPT-4 prompt concordance was: Pathogenic (92% sensitive and 73% PPV), Intermediate or Inconclusive (67% sensitive and 93% PPV), Benign (100% sensitive and 73% PPV).   Conclusion: The GPT-4 prompts detected the presence or absence of a functional assay with high sensitivity and PPV, and articles with unambiguous evidence supporting a benign or pathogenic classification with high sensitivity and reasonable PPV. Our prompts detected papers with intermediate or inconclusive evidence with lower sensitivity but high PPV. Our results support that GPT-4 may be useful in variant classification workflows by enabling prioritization of articles for review that are likely to have functional evidence supporting or refuting pathogenicity, but not that GPT-4 is capable of fully automating the genetics literature review component of variant classification.

{{</citation>}}


## cs.CE (1)



### (116/116) An integrated framework for accelerating reactive flow simulation using GPU and machine learning models (Runze Mao et al., 2023)

{{<citation>}}

Runze Mao, Yingrui Wang, Min Zhang, Han Li, Jiayang Xu, Xinyu Dong, Yan Zhang, Zhi X. Chen. (2023)  
**An integrated framework for accelerating reactive flow simulation using GPU and machine learning models**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE, physics-flu-dyn  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.13513v1)  

---


**ABSTRACT**  
Recent progress in artificial intelligence (AI) and high-performance computing (HPC) have brought potentially game-changing opportunities in accelerating reactive flow simulations. In this study, we introduce an open-source computational fluid dynamics (CFD) framework that integrates the strengths of machine learning (ML) and graphics processing unit (GPU) to demonstrate their combined capability. Within this framework, all computational operations are solely executed on GPU, including ML-accelerated chemistry integration, fully-implicit solving of PDEs, and computation of thermal and transport properties, thereby eliminating the CPU-GPU memory copy overhead. Optimisations both within the kernel functions and during the kernel launch process are conducted to enhance computational performance. Strategies such as static data reorganisation and dynamic data allocation are adopted to reduce the GPU memory footprint. The computational performance is evaluated in two turbulent flame benchmarks using quasi-DNS and LES modelling, respectively. Remarkably, while maintaining a similar level of accuracy to the conventional CPU/CVODE-based solver, the GPU/ML-accelerated approach shows an overall speedup of over two orders of magnitude for both cases. This result highlights that high-fidelity turbulent combustion simulation with finite-rate chemistry that requires normally hundreds of CPUs can now be performed on portable devices such as laptops with a medium-end GPU.

{{</citation>}}
