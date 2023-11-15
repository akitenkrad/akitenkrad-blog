---
draft: false
title: "arXiv @ 2023.11.12"
date: 2023-11-12
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.12"
    identifier: arxiv_20231112
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.SE (2)](#csse-2)
- [cs.CL (25)](#cscl-25)
- [cs.LG (11)](#cslg-11)
- [cs.CV (19)](#cscv-19)
- [eess.IV (4)](#eessiv-4)
- [cs.AI (7)](#csai-7)
- [cs.CY (1)](#cscy-1)
- [cs.RO (3)](#csro-3)
- [cs.IT (1)](#csit-1)
- [cs.CR (2)](#cscr-2)
- [cs.IR (10)](#csir-10)
- [cs.SI (2)](#cssi-2)
- [q-fin.ST (1)](#q-finst-1)
- [cs.HC (2)](#cshc-2)
- [cs.NE (1)](#csne-1)
- [cs.NI (1)](#csni-1)

## cs.SE (2)



### (1/92) Testing LLMs on Code Generation with Varying Levels of Prompt Specificity (Lincoln Murr et al., 2023)

{{<citation>}}

Lincoln Murr, Morgan Grainger, David Gao. (2023)  
**Testing LLMs on Code Generation with Varying Levels of Prompt Specificity**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.07599v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated unparalleled prowess in mimicking human-like text generation and processing. Among the myriad of applications that benefit from LLMs, automated code generation is increasingly promising. The potential to transform natural language prompts into executable code promises a major shift in software development practices and paves the way for significant reductions in manual coding efforts and the likelihood of human-induced errors. This paper reports the results of a study that evaluates the performance of various LLMs, such as Bard, ChatGPT-3.5, ChatGPT-4, and Claude-2, in generating Python for coding problems. We focus on how levels of prompt specificity impact the accuracy, time efficiency, and space efficiency of the generated code. A benchmark of 104 coding problems, each with four types of prompts with varying degrees of tests and specificity, was employed to examine these aspects comprehensively. Our results indicate significant variations in performance across different LLMs and prompt types, and its key contribution is to reveal the ideal prompting strategy for creating accurate Python functions. This study lays the groundwork for further research in LLM capabilities and suggests practical implications for utilizing LLMs in automated code generation tasks and test-driven development.

{{</citation>}}


### (2/92) TransformCode: A Contrastive Learning Framework for Code Embedding via Subtree transformation (Zixiang Xian et al., 2023)

{{<citation>}}

Zixiang Xian, Rubing Huang, Dave Towey, Chunrong Fang, Zhenyu Chen. (2023)  
**TransformCode: A Contrastive Learning Framework for Code Embedding via Subtree transformation**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Contrastive Learning, Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2311.08157v1)  

---


**ABSTRACT**  
Large-scale language models have made great progress in the field of software engineering in recent years. They can be used for many code-related tasks such as code clone detection, code-to-code search, and method name prediction. However, these large-scale language models based on each code token have several drawbacks: They are usually large in scale, heavily dependent on labels, and require a lot of computing power and time to fine-tune new datasets.Furthermore, code embedding should be performed on the entire code snippet rather than encoding each code token. The main reason for this is that encoding each code token would cause model parameter inflation, resulting in a lot of parameters storing information that we are not very concerned about. In this paper, we propose a novel framework, called TransformCode, that learns about code embeddings in a contrastive learning manner. The framework uses the Transformer encoder as an integral part of the model. We also introduce a novel data augmentation technique called abstract syntax tree transformation: This technique applies syntactic and semantic transformations to the original code snippets to generate more diverse and robust anchor samples. Our proposed framework is both flexible and adaptable: It can be easily extended to other downstream tasks that require code representation such as code clone detection and classification. The framework is also very efficient and scalable: It does not require a large model or a large amount of training data, and can support any programming language.Finally, our framework is not limited to unsupervised learning, but can also be applied to some supervised learning tasks by incorporating task-specific labels or objectives. To explore the effectiveness of our framework, we conducted extensive experiments on different software engineering tasks using different programming languages and multiple datasets.

{{</citation>}}


## cs.CL (25)



### (3/92) ChatGPT Prompting Cannot Estimate Predictive Uncertainty in High-Resource Languages (Martino Pelucchi et al., 2023)

{{<citation>}}

Martino Pelucchi, Matias Valdenegro-Toro. (2023)  
**ChatGPT Prompting Cannot Estimate Predictive Uncertainty in High-Resource Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2311.06427v1)  

---


**ABSTRACT**  
ChatGPT took the world by storm for its impressive abilities. Due to its release without documentation, scientists immediately attempted to identify its limits, mainly through its performance in natural language processing (NLP) tasks. This paper aims to join the growing literature regarding ChatGPT's abilities by focusing on its performance in high-resource languages and on its capacity to predict its answers' accuracy by giving a confidence level. The analysis of high-resource languages is of interest as studies have shown that low-resource languages perform worse than English in NLP tasks, but no study so far has analysed whether high-resource languages perform as well as English. The analysis of ChatGPT's confidence calibration has not been carried out before either and is critical to learn about ChatGPT's trustworthiness. In order to study these two aspects, five high-resource languages and two NLP tasks were chosen. ChatGPT was asked to perform both tasks in the five languages and to give a numerical confidence value for each answer. The results show that all the selected high-resource languages perform similarly and that ChatGPT does not have a good confidence calibration, often being overconfident and never giving low confidence values.

{{</citation>}}


### (4/92) Autoregressive Language Models For Estimating the Entropy of Epic EHR Audit Logs (Benjamin C. Warner et al., 2023)

{{<citation>}}

Benjamin C. Warner, Thomas Kannampallil, Seunghwan Kim. (2023)  
**Autoregressive Language Models For Estimating the Entropy of Epic EHR Audit Logs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IT, cs.CL, math-IT  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.06401v2)  

---


**ABSTRACT**  
EHR audit logs are a highly granular stream of events that capture clinician activities, and is a significant area of interest for research in characterizing clinician workflow on the electronic health record (EHR). Existing techniques to measure the complexity of workflow through EHR audit logs (audit logs) involve time- or frequency-based cross-sectional aggregations that are unable to capture the full complexity of a EHR session. We briefly evaluate the usage of transformer-based tabular language model (tabular LM) in measuring the entropy or disorderedness of action sequences within workflow and release the evaluated models publicly.

{{</citation>}}


### (5/92) Distilling Large Language Models using Skill-Occupation Graph Context for HR-Related Tasks (Pouya Pezeshkpour et al., 2023)

{{<citation>}}

Pouya Pezeshkpour, Hayate Iso, Thom Lake, Nikita Bhutani, Estevam Hruschka. (2023)  
**Distilling Large Language Models using Skill-Occupation Graph Context for HR-Related Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.06383v1)  

---


**ABSTRACT**  
Numerous HR applications are centered around resumes and job descriptions. While they can benefit from advancements in NLP, particularly large language models, their real-world adoption faces challenges due to absence of comprehensive benchmarks for various HR tasks, and lack of smaller models with competitive capabilities. In this paper, we aim to bridge this gap by introducing the Resume-Job Description Benchmark (RJDB). We meticulously craft this benchmark to cater to a wide array of HR tasks, including matching and explaining resumes to job descriptions, extracting skills and experiences from resumes, and editing resumes. To create this benchmark, we propose to distill domain-specific knowledge from a large language model (LLM). We rely on a curated skill-occupation graph to ensure diversity and provide context for LLMs generation. Our benchmark includes over 50 thousand triples of job descriptions, matched resumes and unmatched resumes. Using RJDB, we train multiple smaller student models. Our experiments reveal that the student models achieve near/better performance than the teacher model (GPT-4), affirming the effectiveness of the benchmark. Additionally, we explore the utility of RJDB on out-of-distribution data for skill extraction and resume-job description matching, in zero-shot and weak supervision manner. We release our datasets and code to foster further research and industry applications.

{{</citation>}}


### (6/92) Transfer Learning for Structured Pruning under Limited Task Data (Lucio Dery et al., 2023)

{{<citation>}}

Lucio Dery, David Grangier, Awni Hannun. (2023)  
**Transfer Learning for Structured Pruning under Limited Task Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.06382v1)  

---


**ABSTRACT**  
Large, pre-trained models are problematic to use in resource constrained applications. Fortunately, task-aware structured pruning methods offer a solution. These approaches reduce model size by dropping structural units like layers and attention heads in a manner that takes into account the end-task. However, these pruning algorithms require more task-specific data than is typically available. We propose a framework which combines structured pruning with transfer learning to reduce the need for task-specific data. Our empirical results answer questions such as: How should the two tasks be coupled? What parameters should be transferred? And, when during training should transfer learning be introduced? Leveraging these insights, we demonstrate that our framework results in pruned models with improved generalization over strong baselines.

{{</citation>}}


### (7/92) DeMuX: Data-efficient Multilingual Learning (Simran Khanuja et al., 2023)

{{<citation>}}

Simran Khanuja, Srinivas Gowriraj, Lucio Dery, Graham Neubig. (2023)  
**DeMuX: Data-efficient Multilingual Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2311.06379v1)  

---


**ABSTRACT**  
We consider the task of optimally fine-tuning pre-trained multilingual models, given small amounts of unlabelled target data and an annotation budget. In this paper, we introduce DEMUX, a framework that prescribes the exact data-points to label from vast amounts of unlabelled multilingual data, having unknown degrees of overlap with the target set. Unlike most prior works, our end-to-end framework is language-agnostic, accounts for model representations, and supports multilingual target configurations. Our active learning strategies rely upon distance and uncertainty measures to select task-specific neighbors that are most informative to label, given a model. DeMuX outperforms strong baselines in 84% of the test cases, in the zero-shot setting of disjoint source and target language sets (including multilingual target pools), across three models and four tasks. Notably, in low-budget settings (5-100 examples), we observe gains of up to 8-11 F1 points for token-level tasks, and 2-5 F1 for complex tasks. Our code is released here: https://github.com/simran-khanuja/demux.

{{</citation>}}


### (8/92) Heaps' Law in GPT-Neo Large Language Model Emulated Corpora (Uyen Lai et al., 2023)

{{<citation>}}

Uyen Lai, Gurjit S. Randhawa, Paul Sheridan. (2023)  
**Heaps' Law in GPT-Neo Large Language Model Emulated Corpora**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.06377v1)  

---


**ABSTRACT**  
Heaps' law is an empirical relation in text analysis that predicts vocabulary growth as a function of corpus size. While this law has been validated in diverse human-authored text corpora, its applicability to large language model generated text remains unexplored. This study addresses this gap, focusing on the emulation of corpora using the suite of GPT-Neo large language models. To conduct our investigation, we emulated corpora of PubMed abstracts using three different parameter sizes of the GPT-Neo model. Our emulation strategy involved using the initial five words of each PubMed abstract as a prompt and instructing the model to expand the content up to the original abstract's length. Our findings indicate that the generated corpora adhere to Heaps' law. Interestingly, as the GPT-Neo model size grows, its generated vocabulary increasingly adheres to Heaps' law as as observed in human-authored text. To further improve the richness and authenticity of GPT-Neo outputs, future iterations could emphasize enhancing model size or refining the model architecture to curtail vocabulary repetition.

{{</citation>}}


### (9/92) Relation Extraction in underexplored biomedical domains: A diversity-optimised sampling and synthetic data generation approach (Maxime Delmas et al., 2023)

{{<citation>}}

Maxime Delmas, Magdalena Wysocka, André Freitas. (2023)  
**Relation Extraction in underexplored biomedical domains: A diversity-optimised sampling and synthetic data generation approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, LLaMA, Language Model, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2311.06364v1)  

---


**ABSTRACT**  
The sparsity of labelled data is an obstacle to the development of Relation Extraction models and the completion of databases in various biomedical areas. While being of high interest in drug-discovery, the natural-products literature, reporting the identification of potential bioactive compounds from organisms, is a concrete example of such an overlooked topic. To mark the start of this new task, we created the first curated evaluation dataset and extracted literature items from the LOTUS database to build training sets. To this end, we developed a new sampler inspired by diversity metrics in ecology, named Greedy Maximum Entropy sampler, or GME-sampler (https://github.com/idiap/gme-sampler). The strategic optimization of both balance and diversity of the selected items in the evaluation set is important given the resource-intensive nature of manual curation. After quantifying the noise in the training set, in the form of discrepancies between the input abstracts text and the expected output labels, we explored different strategies accordingly. Framing the task as an end-to-end Relation Extraction, we evaluated the performance of standard fine-tuning as a generative task and few-shot learning with open Large Language Models (LLaMA 7B-65B). In addition to their evaluation in few-shot settings, we explore the potential of open Large Language Models (Vicuna-13B) as synthetic data generator and propose a new workflow for this purpose. All evaluated models exhibited substantial improvements when fine-tuned on synthetic abstracts rather than the original noisy data. We provide our best performing (f1-score=59.0) BioGPT-Large model for end-to-end RE of natural-products relationships along with all the generated synthetic data and the evaluation dataset. See more details at https://github.com/idiap/abroad-re.

{{</citation>}}


### (10/92) Word Definitions from Large Language Models (Yunting Yin et al., 2023)

{{<citation>}}

Yunting Yin, Steven Skiena. (2023)  
**Word Definitions from Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.06362v1)  

---


**ABSTRACT**  
Dictionary definitions are historically the arbitrator of what words mean, but this primacy has come under threat by recent progress in NLP, including word embeddings and generative models like ChatGPT. We present an exploratory study of the degree of alignment between word definitions from classical dictionaries and these newer computational artifacts. Specifically, we compare definitions from three published dictionaries to those generated from variants of ChatGPT. We show that (i) definitions from different traditional dictionaries exhibit more surface form similarity than do model-generated definitions, (ii) that the ChatGPT definitions are highly accurate, comparable to traditional dictionaries, and (iii) ChatGPT-based embedding definitions retain their accuracy even on low frequency words, much better than GloVE and FastText word embeddings.

{{</citation>}}


### (11/92) Schema Graph-Guided Prompt for Multi-Domain Dialogue State Tracking (Ruolin Su et al., 2023)

{{<citation>}}

Ruolin Su, Ting-Wei Wu, Biing-Hwang Juang. (2023)  
**Schema Graph-Guided Prompt for Multi-Domain Dialogue State Tracking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2311.06345v1)  

---


**ABSTRACT**  
Tracking dialogue states is an essential topic in task-oriented dialogue systems, which involve filling in the necessary information in pre-defined slots corresponding to a schema. While general pre-trained language models have been shown effective in slot-filling, their performance is limited when applied to specific domains. We propose a graph-based framework that learns domain-specific prompts by incorporating the dialogue schema. Specifically, we embed domain-specific schema encoded by a graph neural network into the pre-trained language model, which allows for relations in the schema to guide the model for better adaptation to the specific domain. Our experiments demonstrate that the proposed graph-based method outperforms other multi-domain DST approaches while using similar or fewer trainable parameters. We also conduct a comprehensive study of schema graph architectures, parameter usage, and module ablation that demonstrate the effectiveness of our model on multi-domain dialogue state tracking.

{{</citation>}}


### (12/92) Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models (Shahriar Golchin et al., 2023)

{{<citation>}}

Shahriar Golchin, Mihai Surdeanu. (2023)  
**Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.06233v1)  

---


**ABSTRACT**  
We propose the Data Contamination Quiz, a simple and effective approach to detect data contamination in large language models (LLMs) and estimate the amount of it. Specifically, we frame data contamination detection as a series of multiple-choice questions. We devise a quiz format wherein three perturbed versions of each dataset instance are created. These changes only include word-level perturbations, replacing words with their contextual synonyms, ensuring both the semantic and sentence structure remain exactly the same as the original instance. Together with the original instance, these perturbed versions constitute the choices in the quiz. Given that the only distinguishing signal among these choices is the exact wording, an LLM, when tasked with identifying the original instance from the choices, opts for the original if it has memorized it in its pre-training phase--a trait intrinsic to LLMs. A dataset partition is then marked as contaminated if the LLM's performance on the quiz surpasses what random chance suggests. Our evaluation spans seven datasets and their respective splits (train and test/validation) on two state-of-the-art LLMs: GPT-4 and GPT-3.5. While lacking access to the pre-training data, our results suggest that our approach not only enhances the detection of data contamination but also provides an accurate estimation of its extent, even when the contamination signal is weak.

{{</citation>}}


### (13/92) A Comparison of Lexicon-Based and ML-Based Sentiment Analysis: Are There Outlier Words? (Siddhant Jaydeep Mahajani et al., 2023)

{{<citation>}}

Siddhant Jaydeep Mahajani, Shashank Srivastava, Alan F. Smeaton. (2023)  
**A Comparison of Lexicon-Based and ML-Based Sentiment Analysis: Are There Outlier Words?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Azure, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.06221v1)  

---


**ABSTRACT**  
Lexicon-based approaches to sentiment analysis of text are based on each word or lexical entry having a pre-defined weight indicating its sentiment polarity. These are usually manually assigned but the accuracy of these when compared against machine leaning based approaches to computing sentiment, are not known. It may be that there are lexical entries whose sentiment values cause a lexicon-based approach to give results which are very different to a machine learning approach. In this paper we compute sentiment for more than 150,000 English language texts drawn from 4 domains using the Hedonometer, a lexicon-based technique and Azure, a contemporary machine-learning based approach which is part of the Azure Cognitive Services family of APIs which is easy to use. We model differences in sentiment scores between approaches for documents in each domain using a regression and analyse the independent variables (Hedonometer lexical entries) as indicators of each word's importance and contribution to the score differences. Our findings are that the importance of a word depends on the domain and there are no standout lexical entries which systematically cause differences in sentiment scores.

{{</citation>}}


### (14/92) BanglaBait: Semi-Supervised Adversarial Approach for Clickbait Detection on Bangla Clickbait Dataset (Md. Motahar Mahtab et al., 2023)

{{<citation>}}

Md. Motahar Mahtab, Monirul Haque, Mehedi Hasan, Farig Sadeque. (2023)  
**BanglaBait: Semi-Supervised Adversarial Approach for Clickbait Detection on Bangla Clickbait Dataset**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: LSTM, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.06204v1)  

---


**ABSTRACT**  
Intentionally luring readers to click on a particular content by exploiting their curiosity defines a title as clickbait. Although several studies focused on detecting clickbait titles in English articles, low resource language like Bangla has not been given adequate attention. To tackle clickbait titles in Bangla, we have constructed the first Bangla clickbait detection dataset containing 15,056 labeled news articles and 65,406 unlabelled news articles extracted from clickbait dense news sites. Each article has been labeled by three expert linguists and includes an article's title, body, and other metadata. By incorporating labeled and unlabelled data, we finetune a pretrained Bangla transformer model in an adversarial fashion using Semi Supervised Generative Adversarial Networks (SS GANs). The proposed model acts as a good baseline for this dataset, outperforming traditional neural network models (LSTM, GRU, CNN) and linguistic feature based models. We expect that this dataset and the detailed analysis and comparison of these clickbait detection models will provide a fundamental basis for future research into detecting clickbait titles in Bengali articles. We have released the corresponding code and dataset.

{{</citation>}}


### (15/92) Language Models can be Logical Solvers (Jiazhan Feng et al., 2023)

{{<citation>}}

Jiazhan Feng, Ruochen Xu, Junheng Hao, Hiteshi Sharma, Yelong Shen, Dongyan Zhao, Weizhu Chen. (2023)  
**Language Models can be Logical Solvers**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.06158v1)  

---


**ABSTRACT**  
Logical reasoning is a fundamental aspect of human intelligence and a key component of tasks like problem-solving and decision-making. Recent advancements have enabled Large Language Models (LLMs) to potentially exhibit reasoning capabilities, but complex logical reasoning remains a challenge. The state-of-the-art, solver-augmented language models, use LLMs to parse natural language logical questions into symbolic representations first and then adopt external logical solvers to take in the symbolic representations and output the answers. Despite their impressive performance, any parsing errors will inevitably result in the failure of the execution of the external logical solver and no answer to the logical questions. In this paper, we introduce LoGiPT, a novel language model that directly emulates the reasoning processes of logical solvers and bypasses the parsing errors by learning to strict adherence to solver syntax and grammar. LoGiPT is fine-tuned on a newly constructed instruction-tuning dataset derived from revealing and refining the invisible reasoning process of deductive solvers. Experimental results on two public deductive reasoning datasets demonstrate that LoGiPT outperforms state-of-the-art solver-augmented LMs and few-shot prompting methods on competitive LLMs like ChatGPT or GPT-4.

{{</citation>}}


### (16/92) Making LLMs Worth Every Penny: Resource-Limited Text Classification in Banking (Lefteris Loukas et al., 2023)

{{<citation>}}

Lefteris Loukas, Ilias Stogiannidis, Odysseas Diamantopoulos, Prodromos Malakasiotis, Stavros Vassos. (2023)  
**Making LLMs Worth Every Penny: Resource-Limited Text Classification in Banking**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model, NLP, Text Classification  
[Paper Link](http://arxiv.org/abs/2311.06102v1)  

---


**ABSTRACT**  
Standard Full-Data classifiers in NLP demand thousands of labeled examples, which is impractical in data-limited domains. Few-shot methods offer an alternative, utilizing contrastive learning techniques that can be effective with as little as 20 examples per class. Similarly, Large Language Models (LLMs) like GPT-4 can perform effectively with just 1-5 examples per class. However, the performance-cost trade-offs of these methods remain underexplored, a critical concern for budget-limited organizations. Our work addresses this gap by studying the aforementioned approaches over the Banking77 financial intent detection dataset, including the evaluation of cutting-edge LLMs by OpenAI, Cohere, and Anthropic in a comprehensive set of few-shot scenarios. We complete the picture with two additional methods: first, a cost-effective querying method for LLMs based on retrieval-augmented generation (RAG), able to reduce operational costs multiple times compared to classic few-shot approaches, and second, a data augmentation method using GPT-4, able to improve performance in data-limited scenarios. Finally, to inspire future research, we provide a human expert's curated subset of Banking77, along with extensive error analysis.

{{</citation>}}


### (17/92) Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration (Wenjie Fu et al., 2023)

{{<citation>}}

Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang. (2023)  
**Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.06062v1)  

---


**ABSTRACT**  
Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.

{{</citation>}}


### (18/92) ChiMed-GPT: A Chinese Medical Large Language Model with Full Training Regime and Better Alignment to Human Preferences (Yuanhe Tian et al., 2023)

{{<citation>}}

Yuanhe Tian, Ruyi Gan, Yan Song, Jiaxing Zhang, Yongdong Zhang. (2023)  
**ChiMed-GPT: A Chinese Medical Large Language Model with Full Training Regime and Better Alignment to Human Preferences**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.06025v1)  

---


**ABSTRACT**  
Recently, the increasing demand for superior medical services has highlighted the discrepancies in the medical infrastructure. With big data, especially texts, forming the foundation of medical services, there is an exigent need for effective natural language processing (NLP) solutions tailored to the healthcare domain. Conventional approaches leveraging pre-trained models present promising results in this domain and current large language models (LLMs) offer advanced foundation for medical text processing. However, most medical LLMs are trained only with supervised fine-tuning (SFT), even though it efficiently empowers LLMs to understand and respond to medical instructions but is ineffective in learning domain knowledge and aligning with human preference. Another engineering barrier that prevents current medical LLM from better text processing ability is their restricted context length (e.g., 2,048 tokens), making it hard for the LLMs to process long context, which is frequently required in the medical domain. In this work, we propose ChiMed-GPT, a new benchmark LLM designed explicitly for Chinese medical domain, with enlarged context length to 4,096 tokens and undergoes a comprehensive training regime with pre-training, SFT, and RLHF. Evaluations on real-world tasks including information extraction, question answering, and dialogue generation demonstrate ChiMed-GPT's superior performance over general domain LLMs. Furthermore, we analyze possible biases through prompting ChiMed-GPT to perform attitude scales regarding discrimination of patients, so as to contribute to further responsible development of LLMs in the medical domain. The code and model are released at https://github.com/synlp/ChiMed-GPT.

{{</citation>}}


### (19/92) Large Language Models are Zero Shot Hypothesis Proposers (Biqing Qi et al., 2023)

{{<citation>}}

Biqing Qi, Kaiyan Zhang, Haoxiang Li, Kai Tian, Sihang Zeng, Zhang-Ren Chen, Bowen Zhou. (2023)  
**Large Language Models are Zero Shot Hypothesis Proposers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05965v1)  

---


**ABSTRACT**  
Significant scientific discoveries have driven the progress of human civilisation. The explosion of scientific literature and data has created information barriers across disciplines that have slowed the pace of scientific discovery. Large Language Models (LLMs) hold a wealth of global and interdisciplinary knowledge that promises to break down these information barriers and foster a new wave of scientific discovery. However, the potential of LLMs for scientific discovery has not been formally explored. In this paper, we start from investigating whether LLMs can propose scientific hypotheses. To this end, we construct a dataset consist of background knowledge and hypothesis pairs from biomedical literature. The dataset is divided into training, seen, and unseen test sets based on the publication date to control visibility. We subsequently evaluate the hypothesis generation capabilities of various top-tier instructed models in zero-shot, few-shot, and fine-tuning settings, including both closed and open-source LLMs. Additionally, we introduce an LLM-based multi-agent cooperative framework with different role designs and external tools to enhance the capabilities related to generating hypotheses. We also design four metrics through a comprehensive review to evaluate the generated hypotheses for both ChatGPT-based and human evaluations. Through experiments and analyses, we arrive at the following findings: 1) LLMs surprisingly generate untrained yet validated hypotheses from testing literature. 2) Increasing uncertainty facilitates candidate generation, potentially enhancing zero-shot hypothesis generation capabilities. These findings strongly support the potential of LLMs as catalysts for new scientific discoveries and guide further exploration.

{{</citation>}}


### (20/92) How to Bridge the Gap between Modalities: A Comprehensive Survey on Multimodal Large Language Model (Shezheng Song et al., 2023)

{{<citation>}}

Shezheng Song, Xiaopeng Li, Shasha Li. (2023)  
**How to Bridge the Gap between Modalities: A Comprehensive Survey on Multimodal Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07594v1)  

---


**ABSTRACT**  
This review paper explores Multimodal Large Language Models (MLLMs), which integrate Large Language Models (LLMs) like GPT-4 to handle multimodal data such as text and vision. MLLMs demonstrate capabilities like generating image narratives and answering image-based questions, bridging the gap towards real-world human-computer interactions and hinting at a potential pathway to artificial general intelligence. However, MLLMs still face challenges in processing the semantic gap in multimodality, which may lead to erroneous generation, posing potential risks to society. Choosing the appropriate modality alignment method is crucial, as improper methods might require more parameters with limited performance improvement. This paper aims to explore modality alignment methods for LLMs and their existing capabilities. Implementing modality alignment allows LLMs to address environmental issues and enhance accessibility. The study surveys existing modal alignment methods in MLLMs into four groups: (1) Multimodal Converters that change data into something LLMs can understand; (2) Multimodal Perceivers to improve how LLMs perceive different types of data; (3) Tools Assistance for changing data into one common format, usually text; and (4) Data-Driven methods that teach LLMs to understand specific types of data in a dataset. This field is still in a phase of exploration and experimentation, and we will organize and update various existing research methods for multimodal information alignment.

{{</citation>}}


### (21/92) The Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer-Based Models (Anton Razzhigaev et al., 2023)

{{<citation>}}

Anton Razzhigaev, Matvey Mikhalchuk, Elizaveta Goncharova, Ivan Oseledets, Denis Dimitrov, Andrey Kuznetsov. (2023)  
**The Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer-Based Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IT, cs-LG, cs.CL, math-GN, math-IT  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.05928v1)  

---


**ABSTRACT**  
In this study, we present an investigation into the anisotropy dynamics and intrinsic dimension of embeddings in transformer architectures, focusing on the dichotomy between encoders and decoders. Our findings reveal that the anisotropy profile in transformer decoders exhibits a distinct bell-shaped curve, with the highest anisotropy concentrations in the middle layers. This pattern diverges from the more uniformly distributed anisotropy observed in encoders. In addition, we found that the intrinsic dimension of embeddings increases in the initial phases of training, indicating an expansion into higher-dimensional space. Which is then followed by a compression phase towards the end of training with dimensionality decrease, suggesting a refinement into more compact representations. Our results provide fresh insights to the understanding of encoders and decoders embedding properties.

{{</citation>}}


### (22/92) Chain of Thought with Explicit Evidence Reasoning for Few-shot Relation Extraction (Xilai Ma et al., 2023)

{{<citation>}}

Xilai Ma, Jing Li, Min Zhang. (2023)  
**Chain of Thought with Explicit Evidence Reasoning for Few-shot Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2311.05922v1)  

---


**ABSTRACT**  
Few-shot relation extraction involves identifying the type of relationship between two specific entities within a text, using a limited number of annotated samples. A variety of solutions to this problem have emerged by applying meta-learning and neural graph techniques which typically necessitate a training process for adaptation. Recently, the strategy of in-context learning has been demonstrating notable results without the need of training. Few studies have already utilized in-context learning for zero-shot information extraction. Unfortunately, the evidence for inference is either not considered or implicitly modeled during the construction of chain-of-thought prompts. In this paper, we propose a novel approach for few-shot relation extraction using large language models, named CoT-ER, chain-of-thought with explicit evidence reasoning. In particular, CoT-ER first induces large language models to generate evidences using task-specific and concept-level knowledge. Then these evidences are explicitly incorporated into chain-of-thought prompting for relation extraction. Experimental results demonstrate that our CoT-ER approach (with 0% training data) achieves competitive performance compared to the fully-supervised (with 100% training data) state-of-the-art approach on the FewRel1.0 and FewRel2.0 datasets.

{{</citation>}}


### (23/92) Follow-Up Differential Descriptions: Language Models Resolve Ambiguities for Image Classification (Reza Esfandiarpoor et al., 2023)

{{<citation>}}

Reza Esfandiarpoor, Stephen H. Bach. (2023)  
**Follow-Up Differential Descriptions: Language Models Resolve Ambiguities for Image Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: Image Classification, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07593v1)  

---


**ABSTRACT**  
A promising approach for improving the performance of vision-language models like CLIP for image classification is to extend the class descriptions (i.e., prompts) with related attributes, e.g., using brown sparrow instead of sparrow. However, current zero-shot methods select a subset of attributes regardless of commonalities between the target classes, potentially providing no useful information that would have helped to distinguish between them. For instance, they may use color instead of bill shape to distinguish between sparrows and wrens, which are both brown. We propose Follow-up Differential Descriptions (FuDD), a zero-shot approach that tailors the class descriptions to each dataset and leads to additional attributes that better differentiate the target classes. FuDD first identifies the ambiguous classes for each image, and then uses a Large Language Model (LLM) to generate new class descriptions that differentiate between them. The new class descriptions resolve the initial ambiguity and help predict the correct label. In our experiments, FuDD consistently outperforms generic description ensembles and naive LLM-generated descriptions on 12 datasets. We show that differential descriptions are an effective tool to resolve class ambiguities, which otherwise significantly degrade the performance. We also show that high quality natural language class descriptions produced by FuDD result in comparable performance to few-shot adaptation methods.

{{</citation>}}


### (24/92) Trends in Integration of Knowledge and Large Language Models: A Survey and Taxonomy of Methods, Benchmarks, and Applications (Zhangyin Feng et al., 2023)

{{<citation>}}

Zhangyin Feng, Weitao Ma, Weijiang Yu, Lei Huang, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, Ting liu. (2023)  
**Trends in Integration of Knowledge and Large Language Models: A Survey and Taxonomy of Methods, Benchmarks, and Applications**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.05876v1)  

---


**ABSTRACT**  
Large language models (LLMs) exhibit superior performance on various natural language tasks, but they are susceptible to issues stemming from outdated data and domain-specific limitations. In order to address these challenges, researchers have pursued two primary strategies, knowledge editing and retrieval augmentation, to enhance LLMs by incorporating external information from different aspects. Nevertheless, there is still a notable absence of a comprehensive survey. In this paper, we propose a review to discuss the trends in integration of knowledge and large language models, including taxonomy of methods, benchmarks, and applications. In addition, we conduct an in-depth analysis of different methods and point out potential research directions in the future. We hope this survey offers the community quick access and a comprehensive overview of this research area, with the intention of inspiring future research endeavors.

{{</citation>}}


### (25/92) Tamil-Llama: A New Tamil Language Model Based on Llama 2 (Abhinand Balachandran, 2023)

{{<citation>}}

Abhinand Balachandran. (2023)  
**Tamil-Llama: A New Tamil Language Model Based on Llama 2**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05845v1)  

---


**ABSTRACT**  
Language modeling has witnessed remarkable advancements in recent years, with Large Language Models (LLMs) like ChatGPT setting unparalleled benchmarks in human-like text generation. However, a prevailing limitation is the underrepresentation of languages like Tamil in these cutting-edge models, leading to suboptimal performance in diverse linguistic contexts. This paper addresses this lacuna, enhancing the open-source LLaMA model with an addition of 16,000 Tamil tokens, aiming to achieve superior text generation and comprehension in the Tamil language. We strategically employ the LoRA methodology for efficient model training on a comprehensive Tamil corpus, ensuring computational feasibility and model robustness. Moreover, we introduce a Tamil-translated version of the Alpaca dataset and a subset of the OpenOrca dataset tailored for instruction fine-tuning. Our results showcase significant performance improvements in Tamil text generation, with potential implications for the broader landscape of LLMs in Indian languages. We further underscore our commitment to open research by making our models, datasets, and code publicly accessible, fostering further innovations in language modeling.

{{</citation>}}


### (26/92) Let's Reinforce Step by Step (Sarah Pan et al., 2023)

{{<citation>}}

Sarah Pan, Vladislav Lialin, Sherin Muckatira, Anna Rumshisky. (2023)  
**Let's Reinforce Step by Step**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.05821v1)  

---


**ABSTRACT**  
While recent advances have boosted LM proficiency in linguistic benchmarks, LMs consistently struggle to reason correctly on complex tasks like mathematics. We turn to Reinforcement Learning from Human Feedback (RLHF) as a method with which to shape model reasoning processes. In particular, we explore two reward schemes, outcome-supervised reward models (ORMs) and process-supervised reward models (PRMs), to optimize for logical reasoning. Our results show that the fine-grained reward provided by PRM-based methods enhances accuracy on simple mathematical reasoning (GSM8K) while, unexpectedly, reducing performance in complex tasks (MATH). Furthermore, we show the critical role reward aggregation functions play in model performance. Providing promising avenues for future research, our study underscores the need for further exploration into fine-grained reward modeling for more reliable language models.

{{</citation>}}


### (27/92) CFBenchmark: Chinese Financial Assistant Benchmark for Large Language Model (Yang Lei et al., 2023)

{{<citation>}}

Yang Lei, Jiangtong Li, Ming Jiang, Junjie Hu, Dawei Cheng, Zhijun Ding, Changjun Jiang. (2023)  
**CFBenchmark: Chinese Financial Assistant Benchmark for Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Financial, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05812v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated great potential in the financial domain. Thus, it becomes important to assess the performance of LLMs in the financial tasks. In this work, we introduce CFBenchmark, to evaluate the performance of LLMs for Chinese financial assistant. The basic version of CFBenchmark is designed to evaluate the basic ability in Chinese financial text processing from three aspects~(\emph{i.e.} recognition, classification, and generation) including eight tasks, and includes financial texts ranging in length from 50 to over 1,800 characters. We conduct experiments on several LLMs available in the literature with CFBenchmark-Basic, and the experimental results indicate that while some LLMs show outstanding performance in specific tasks, overall, there is still significant room for improvement in basic tasks of financial text processing with existing models. In the future, we plan to explore the advanced version of CFBenchmark, aiming to further explore the extensive capabilities of language models in more profound dimensions as a financial assistant in Chinese. Our codes are released at https://github.com/TongjiFinLab/CFBenchmark.

{{</citation>}}


## cs.LG (11)



### (28/92) Flatness-aware Adversarial Attack (Mingyuan Fan et al., 2023)

{{<citation>}}

Mingyuan Fan, Xiaodan Li, Cen Chen, Yinggui Wang. (2023)  
**Flatness-aware Adversarial Attack**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2311.06423v1)  

---


**ABSTRACT**  
The transferability of adversarial examples can be exploited to launch black-box attacks. However, adversarial examples often present poor transferability. To alleviate this issue, by observing that the diversity of inputs can boost transferability, input regularization based methods are proposed, which craft adversarial examples by combining several transformed inputs. We reveal that input regularization based methods make resultant adversarial examples biased towards flat extreme regions. Inspired by this, we propose an attack called flatness-aware adversarial attack (FAA) which explicitly adds a flatness-aware regularization term in the optimization target to promote the resultant adversarial examples towards flat extreme regions. The flatness-aware regularization term involves gradients of samples around the resultant adversarial examples but optimizing gradients requires the evaluation of Hessian matrix in high-dimension spaces which generally is intractable. To address the problem, we derive an approximate solution to circumvent the construction of Hessian matrix, thereby making FAA practical and cheap. Extensive experiments show the transferability of adversarial examples crafted by FAA can be considerably boosted compared with state-of-the-art baselines.

{{</citation>}}


### (29/92) Knowledge Graphs are not Created Equal: Exploring the Properties and Structure of Real KGs (Nedelina Teneva et al., 2023)

{{<citation>}}

Nedelina Teneva, Estevam Hruschka. (2023)  
**Knowledge Graphs are not Created Equal: Exploring the Properties and Structure of Real KGs**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: AI, Knowledge Graph, NLP  
[Paper Link](http://arxiv.org/abs/2311.06414v1)  

---


**ABSTRACT**  
Despite the recent popularity of knowledge graph (KG) related tasks and benchmarks such as KG embeddings, link prediction, entity alignment and evaluation of the reasoning abilities of pretrained language models as KGs, the structure and properties of real KGs are not well studied. In this paper, we perform a large scale comparative study of 29 real KG datasets from diverse domains such as the natural sciences, medicine, and NLP to analyze their properties and structural patterns. Based on our findings, we make several recommendations regarding KG-based model development and evaluation. We believe that the rich structural information contained in KGs can benefit the development of better KG models across fields and we hope this study will contribute to breaking the existing data silos between different areas of research (e.g., ML, NLP, AI for sciences).

{{</citation>}}


### (30/92) FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective (Kun Yi et al., 2023)

{{<citation>}}

Kun Yi, Qi Zhang, Wei Fan, Hui He, Liang Hu, Pengyang Wang, Ning An, Longbing Cao, Zhendong Niu. (2023)  
**FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2311.06190v1)  

---


**ABSTRACT**  
Multivariate time series (MTS) forecasting has shown great importance in numerous industries. Current state-of-the-art graph neural network (GNN)-based forecasting methods usually require both graph networks (e.g., GCN) and temporal networks (e.g., LSTM) to capture inter-series (spatial) dynamics and intra-series (temporal) dependencies, respectively. However, the uncertain compatibility of the two networks puts an extra burden on handcrafted model designs. Moreover, the separate spatial and temporal modeling naturally violates the unified spatiotemporal inter-dependencies in real world, which largely hinders the forecasting performance. To overcome these problems, we explore an interesting direction of directly applying graph networks and rethink MTS forecasting from a pure graph perspective. We first define a novel data structure, hypervariate graph, which regards each series value (regardless of variates or timestamps) as a graph node, and represents sliding windows as space-time fully-connected graphs. This perspective considers spatiotemporal dynamics unitedly and reformulates classic MTS forecasting into the predictions on hypervariate graphs. Then, we propose a novel architecture Fourier Graph Neural Network (FourierGNN) by stacking our proposed Fourier Graph Operator (FGO) to perform matrix multiplications in Fourier space. FourierGNN accommodates adequate expressiveness and achieves much lower complexity, which can effectively and efficiently accomplish the forecasting. Besides, our theoretical analysis reveals FGO's equivalence to graph convolutions in the time domain, which further verifies the validity of FourierGNN. Extensive experiments on seven datasets have demonstrated our superior performance with higher efficiency and fewer parameters compared with state-of-the-art methods.

{{</citation>}}


### (31/92) Frequency-domain MLPs are More Effective Learners in Time Series Forecasting (Kun Yi et al., 2023)

{{<citation>}}

Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Defu Lian, Ning An, Longbing Cao, Zhendong Niu. (2023)  
**Frequency-domain MLPs are More Effective Learners in Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Time Series, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.06184v1)  

---


**ABSTRACT**  
Time series forecasting has played the key role in different industrial, including finance, traffic, energy, and healthcare domains. While existing literatures have designed many sophisticated architectures based on RNNs, GNNs, or Transformers, another kind of approaches based on multi-layer perceptrons (MLPs) are proposed with simple structure, low complexity, and {superior performance}. However, most MLP-based forecasting methods suffer from the point-wise mappings and information bottleneck, which largely hinders the forecasting performance. To overcome this problem, we explore a novel direction of applying MLPs in the frequency domain for time series forecasting. We investigate the learned patterns of frequency-domain MLPs and discover their two inherent characteristic benefiting forecasting, (i) global view: frequency spectrum makes MLPs own a complete view for signals and learn global dependencies more easily, and (ii) energy compaction: frequency-domain MLPs concentrate on smaller key part of frequency components with compact signal energy. Then, we propose FreTS, a simple yet effective architecture built upon Frequency-domain MLPs for Time Series forecasting. FreTS mainly involves two stages, (i) Domain Conversion, that transforms time-domain signals into complex numbers of frequency domain; (ii) Frequency Learning, that performs our redesigned MLPs for the learning of real and imaginary part of frequency components. The above stages operated on both inter-series and intra-series scales further contribute to channel-wise and time-wise dependency learning. Extensive experiments on 13 real-world benchmarks (including 7 benchmarks for short-term forecasting and 6 benchmarks for long-term forecasting) demonstrate our consistent superiority over state-of-the-art methods.

{{</citation>}}


### (32/92) Time Scale Network: A Shallow Neural Network For Time Series Data (Trevor Meyer et al., 2023)

{{<citation>}}

Trevor Meyer, Camden Shultz, Najim Dehak, Laureano Moro-Velazquez, Pedro Irazoqui. (2023)  
**Time Scale Network: A Shallow Neural Network For Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.06170v1)  

---


**ABSTRACT**  
Time series data is often composed of information at multiple time scales, particularly in biomedical data. While numerous deep learning strategies exist to capture this information, many make networks larger, require more data, are more demanding to compute, and are difficult to interpret. This limits their usefulness in real-world applications facing even modest computational or data constraints and can further complicate their translation into practice. We present a minimal, computationally efficient Time Scale Network combining the translation and dilation sequence used in discrete wavelet transforms with traditional convolutional neural networks and back-propagation. The network simultaneously learns features at many time scales for sequence classification with significantly reduced parameters and operations. We demonstrate advantages in Atrial Dysfunction detection including: superior accuracy-per-parameter and accuracy-per-operation, fast training and inference speeds, and visualization and interpretation of learned patterns in atrial dysfunction detection on ECG signals. We also demonstrate impressive performance in seizure prediction using EEG signals. Our network isolated a few time scales that could be strategically selected to achieve 90.9% accuracy using only 1,133 active parameters and consistently converged on pulsatile waveform shapes. This method does not rest on any constraints or assumptions regarding signal content and could be leveraged in any area of time series analysis dealing with signals containing features at many time scales.

{{</citation>}}


### (33/92) Interpretable Graph Anomaly Detection using Gradient Attention Maps (Yifei Yang et al., 2023)

{{<citation>}}

Yifei Yang, Peng Wang, Xiaofan He, Dongmian Zou. (2023)  
**Interpretable Graph Anomaly Detection using Gradient Attention Maps**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Attention  
[Paper Link](http://arxiv.org/abs/2311.06153v1)  

---


**ABSTRACT**  
Detecting unusual patterns in graph data is a crucial task in data mining. However, existing methods often face challenges in consistently achieving satisfactory performance and lack interpretability, which hinders our understanding of anomaly detection decisions. In this paper, we propose a novel approach to graph anomaly detection that leverages the power of interpretability to enhance performance. Specifically, our method extracts an attention map derived from gradients of graph neural networks, which serves as a basis for scoring anomalies. In addition, we conduct theoretical analysis using synthetic data to validate our method and gain insights into its decision-making process. To demonstrate the effectiveness of our method, we extensively evaluate our approach against state-of-the-art graph anomaly detection techniques. The results consistently demonstrate the superior performance of our method compared to the baselines.

{{</citation>}}


### (34/92) Going beyond persistent homology using persistent homology (Johanna Immonen et al., 2023)

{{<citation>}}

Johanna Immonen, Amauri H. Souza, Vikas Garg. (2023)  
**Going beyond persistent homology using persistent homology**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.06152v1)  

---


**ABSTRACT**  
Representational limits of message-passing graph neural networks (MP-GNNs), e.g., in terms of the Weisfeiler-Leman (WL) test for isomorphism, are well understood. Augmenting these graph models with topological features via persistent homology (PH) has gained prominence, but identifying the class of attributed graphs that PH can recognize remains open. We introduce a novel concept of color-separating sets to provide a complete resolution to this important problem. Specifically, we establish the necessary and sufficient conditions for distinguishing graphs based on the persistence of their connected components, obtained from filter functions on vertex and edge colors. Our constructions expose the limits of vertex- and edge-level PH, proving that neither category subsumes the other. Leveraging these theoretical insights, we propose RePHINE for learning topological features on graphs. RePHINE efficiently combines vertex- and edge-level PH, achieving a scheme that is provably more powerful than both. Integrating RePHINE into MP-GNNs boosts their expressive power, resulting in gains over standard PH on several benchmarks for graph classification.

{{</citation>}}


### (35/92) Enhancing Actuarial Non-Life Pricing Models via Transformers (Alexej Brauer, 2023)

{{<citation>}}

Alexej Brauer. (2023)  
**Enhancing Actuarial Non-Life Pricing Models via Transformers**  

---
Primary Category: cs.LG  
Categories: 62 68, cs-AI, cs-LG, cs.LG, q-fin-ST, stat-AP  
Keywords: GLM, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.07597v1)  

---


**ABSTRACT**  
Currently, there is a lot of research in the field of neural networks for non-life insurance pricing. The usual goal is to improve the predictive power via neural networks while building upon the generalized linear model, which is the current industry standard. Our paper contributes to this current journey via novel methods to enhance actuarial non-life models with transformer models for tabular data. We build here upon the foundation laid out by the combined actuarial neural network as well as the localGLMnet and enhance those models via the feature tokenizer transformer. The manuscript demonstrates the performance of the proposed methods on a real-world claim frequency dataset and compares them with several benchmark models such as generalized linear models, feed-forward neural networks, combined actuarial neural networks, LocalGLMnet, and pure feature tokenizer transformer. The paper shows that the new methods can achieve better results than the benchmark models while preserving certain generalized linear model advantages. The paper also discusses the practical implications and challenges of applying transformer models in actuarial settings.

{{</citation>}}


### (36/92) Multiscale Neural Operators for Solving Time-Independent PDEs (Winfried Ripken et al., 2023)

{{<citation>}}

Winfried Ripken, Lisa Coiffard, Felix Pieper, Sebastian Dziadzio. (2023)  
**Multiscale Neural Operators for Solving Time-Independent PDEs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.05964v1)  

---


**ABSTRACT**  
Time-independent Partial Differential Equations (PDEs) on large meshes pose significant challenges for data-driven neural PDE solvers. We introduce a novel graph rewiring technique to tackle some of these challenges, such as aggregating information across scales and on irregular meshes. Our proposed approach bridges distant nodes, enhancing the global interaction capabilities of GNNs. Our experiments on three datasets reveal that GNN-based methods set new performance standards for time-independent PDEs on irregular meshes. Finally, we show that our graph rewiring strategy boosts the performance of baseline methods, achieving state-of-the-art results in one of the tasks.

{{</citation>}}


### (37/92) FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores (Daniel Y. Fu et al., 2023)

{{<citation>}}

Daniel Y. Fu, Hermann Kumbong, Eric Nguyen, Christopher Ré. (2023)  
**FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT, GLUE, GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.05908v1)  

---


**ABSTRACT**  
Convolution models with long filters have demonstrated state-of-the-art reasoning abilities in many long-sequence tasks but lag behind the most optimized Transformers in wall-clock time. A major bottleneck is the Fast Fourier Transform (FFT)--which allows long convolutions to run in $O(N logN)$ time in sequence length $N$ but has poor hardware utilization. In this paper, we study how to optimize the FFT convolution. We find two key bottlenecks: the FFT does not effectively use specialized matrix multiply units, and it incurs expensive I/O between layers of the memory hierarchy. In response, we propose FlashFFTConv. FlashFFTConv uses a matrix decomposition that computes the FFT using matrix multiply units and enables kernel fusion for long sequences, reducing I/O. We also present two sparse convolution algorithms--1) partial convolutions and 2) frequency-sparse convolutions--which can be implemented simply by skipping blocks in the matrix decomposition, enabling further opportunities for memory and compute savings. FlashFFTConv speeds up exact FFT convolutions by up to 7.93$\times$ over PyTorch and achieves up to 4.4$\times$ speedup end-to-end. Given the same compute budget, FlashFFTConv allows Hyena-GPT-s to achieve 2.3 points better perplexity on the PILE and M2-BERT-base to achieve 3.3 points higher GLUE score--matching models with twice the parameter count. FlashFFTConv also achieves 96.1% accuracy on Path-512, a high-resolution vision task where no model had previously achieved better than 50%. Furthermore, partial convolutions enable longer-sequence models--yielding the first DNA model that can process the longest human genes (2.3M base pairs)--and frequency-sparse convolutions speed up pretrained models while maintaining or improving model quality.

{{</citation>}}


### (38/92) Layer-wise Auto-Weighting for Non-Stationary Test-Time Adaptation (Junyoung Park et al., 2023)

{{<citation>}}

Junyoung Park, Jin Kim, Hyeongjun Kwon, Ilhoon Yoon, Kwanghoon Sohn. (2023)  
**Layer-wise Auto-Weighting for Non-Stationary Test-Time Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.05858v1)  

---


**ABSTRACT**  
Given the inevitability of domain shifts during inference in real-world applications, test-time adaptation (TTA) is essential for model adaptation after deployment. However, the real-world scenario of continuously changing target distributions presents challenges including catastrophic forgetting and error accumulation. Existing TTA methods for non-stationary domain shifts, while effective, incur excessive computational load, making them impractical for on-device settings. In this paper, we introduce a layer-wise auto-weighting algorithm for continual and gradual TTA that autonomously identifies layers for preservation or concentrated adaptation. By leveraging the Fisher Information Matrix (FIM), we first design the learning weight to selectively focus on layers associated with log-likelihood changes while preserving unrelated ones. Then, we further propose an exponential min-max scaler to make certain layers nearly frozen while mitigating outliers. This minimizes forgetting and error accumulation, leading to efficient adaptation to non-stationary target distribution. Experiments on CIFAR-10C, CIFAR-100C, and ImageNet-C show our method outperforms conventional continual and gradual TTA approaches while significantly reducing computational load, highlighting the importance of FIM-based learning weight in adapting to continuously or gradually shifting target domains.

{{</citation>}}


## cs.CV (19)



### (39/92) Analyzing Modular Approaches for Visual Question Decomposition (Apoorv Khandelwal et al., 2023)

{{<citation>}}

Apoorv Khandelwal, Ellie Pavlick, Chen Sun. (2023)  
**Analyzing Modular Approaches for Visual Question Decomposition**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, QA  
[Paper Link](http://arxiv.org/abs/2311.06411v1)  

---


**ABSTRACT**  
Modular neural networks without additional training have recently been shown to surpass end-to-end neural networks on challenging vision-language tasks. The latest such methods simultaneously introduce LLM-based code generation to build programs and a number of skill-specific, task-oriented modules to execute them. In this paper, we focus on ViperGPT and ask where its additional performance comes from and how much is due to the (state-of-art, end-to-end) BLIP-2 model it subsumes vs. additional symbolic components. To do so, we conduct a controlled study (comparing end-to-end, modular, and prompting-based methods across several VQA benchmarks). We find that ViperGPT's reported gains over BLIP-2 can be attributed to its selection of task-specific modules, and when we run ViperGPT using a more task-agnostic selection of modules, these gains go away. Additionally, ViperGPT retains much of its performance if we make prominent alterations to its selection of modules: e.g. removing or retaining only BLIP-2. Finally, we compare ViperGPT against a prompting-based decomposition strategy and find that, on some benchmarks, modular approaches significantly benefit by representing subtasks with natural language, instead of code.

{{</citation>}}


### (40/92) Towards A Unified Neural Architecture for Visual Recognition and Reasoning (Calvin Luo et al., 2023)

{{<citation>}}

Calvin Luo, Boqing Gong, Ting Chen, Chen Sun. (2023)  
**Towards A Unified Neural Architecture for Visual Recognition and Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2311.06386v1)  

---


**ABSTRACT**  
Recognition and reasoning are two pillars of visual understanding. However, these tasks have an imbalance in focus; whereas recent advances in neural networks have shown strong empirical performance in visual recognition, there has been comparably much less success in solving visual reasoning. Intuitively, unifying these two tasks under a singular framework is desirable, as they are mutually dependent and beneficial. Motivated by the recent success of multi-task transformers for visual recognition and language understanding, we propose a unified neural architecture for visual recognition and reasoning with a generic interface (e.g., tokens) for both. Our framework enables the principled investigation of how different visual recognition tasks, datasets, and inductive biases can help enable spatiotemporal reasoning capabilities. Noticeably, we find that object detection, which requires spatial localization of individual objects, is the most beneficial recognition task for reasoning. We further demonstrate via probing that implicit object-centric representations emerge automatically inside our framework. Intriguingly, we discover that certain architectural choices such as the backbone model of the visual encoder have a significant impact on visual reasoning, but little on object detection. Given the results of our experiments, we believe that visual reasoning should be considered as a first-class citizen alongside visual recognition, as they are strongly correlated but benefit from potentially different design choices.

{{</citation>}}


### (41/92) Image Classification using Combination of Topological Features and Neural Networks (Mariana Dória Prata Lima et al., 2023)

{{<citation>}}

Mariana Dória Prata Lima, Gilson Antonio Giraldi, Gastão Florêncio Miranda Junior. (2023)  
**Image Classification using Combination of Topological Features and Neural Networks**  

---
Primary Category: cs.CV  
Categories: 57T99, 57Z25, I-4-10; I-2-10; I-5-4, cs-CV, cs-LG, cs-NE, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2311.06375v1)  

---


**ABSTRACT**  
In this work we use the persistent homology method, a technique in topological data analysis (TDA), to extract essential topological features from the data space and combine them with deep learning features for classification tasks. In TDA, the concepts of complexes and filtration are building blocks. Firstly, a filtration is constructed from some complex. Then, persistent homology classes are computed, and their evolution along the filtration is visualized through the persistence diagram. Additionally, we applied vectorization techniques to the persistence diagram to make this topological information compatible with machine learning algorithms. This was carried out with the aim of classifying images from multiple classes in the MNIST dataset. Our approach inserts topological features into deep learning approaches composed by single and two-streams neural networks architectures based on a multi-layer perceptron (MLP) and a convolutional neral network (CNN) taylored for multi-class classification in the MNIST dataset. In our analysis, we evaluated the obtained results and compared them with the outcomes achieved through the baselines that are available in the TensorFlow library. The main conclusion is that topological information may increase neural network accuracy in multi-class classification tasks with the price of computational complexity of persistent homology calculation. Up to the best of our knowledge, it is the first work that combines deep learning features and the combination of topological features for multi-class classification tasks.

{{</citation>}}


### (42/92) Harnessing Synthetic Datasets: The Role of Shape Bias in Deep Neural Network Generalization (Elior Benarous et al., 2023)

{{<citation>}}

Elior Benarous, Sotiris Anagnostidis, Luca Biggio, Thomas Hofmann. (2023)  
**Harnessing Synthetic Datasets: The Role of Shape Bias in Deep Neural Network Generalization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.06224v1)  

---


**ABSTRACT**  
Recent advancements in deep learning have been primarily driven by the use of large models trained on increasingly vast datasets. While neural scaling laws have emerged to predict network performance given a specific level of computational resources, the growing demand for expansive datasets raises concerns. To address this, a new research direction has emerged, focusing on the creation of synthetic data as a substitute. In this study, we investigate how neural networks exhibit shape bias during training on synthetic datasets, serving as an indicator of the synthetic data quality. Specifically, our findings indicate three key points: (1) Shape bias varies across network architectures and types of supervision, casting doubt on its reliability as a predictor for generalization and its ability to explain differences in model recognition compared to human capabilities. (2) Relying solely on shape bias to estimate generalization is unreliable, as it is entangled with diversity and naturalism. (3) We propose a novel interpretation of shape bias as a tool for estimating the diversity of samples within a dataset. Our research aims to clarify the implications of using synthetic data and its associated shape bias in deep learning, addressing concerns regarding generalization and dataset quality.

{{</citation>}}


### (43/92) Diffusion Models for Earth Observation Use-cases: from cloud removal to urban change detection (Fulvio Sanguigni et al., 2023)

{{<citation>}}

Fulvio Sanguigni, Mikolaj Czerkawski, Lorenzo Papa, Irene Amerini, Bertrand Le Saux. (2023)  
**Diffusion Models for Earth Observation Use-cases: from cloud removal to urban change detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06222v1)  

---


**ABSTRACT**  
The advancements in the state of the art of generative Artificial Intelligence (AI) brought by diffusion models can be highly beneficial in novel contexts involving Earth observation data. After introducing this new family of generative models, this work proposes and analyses three use cases which demonstrate the potential of diffusion-based approaches for satellite image data. Namely, we tackle cloud removal and inpainting, dataset generation for change-detection tasks, and urban replanning.

{{</citation>}}


### (44/92) Semantic-aware Video Representation for Few-shot Action Recognition (Yutao Tang et al., 2023)

{{<citation>}}

Yutao Tang, Benjamin Bejar, Rene Vidal. (2023)  
**Semantic-aware Video Representation for Few-shot Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.06218v1)  

---


**ABSTRACT**  
Recent work on action recognition leverages 3D features and textual information to achieve state-of-the-art performance. However, most of the current few-shot action recognition methods still rely on 2D frame-level representations, often require additional components to model temporal relations, and employ complex distance functions to achieve accurate alignment of these representations. In addition, existing methods struggle to effectively integrate textual semantics, some resorting to concatenation or addition of textual and visual features, and some using text merely as an additional supervision without truly achieving feature fusion and information transfer from different modalities. In this work, we propose a simple yet effective Semantic-Aware Few-Shot Action Recognition (SAFSAR) model to address these issues. We show that directly leveraging a 3D feature extractor combined with an effective feature-fusion scheme, and a simple cosine similarity for classification can yield better performance without the need of extra components for temporal modeling or complex distance functions. We introduce an innovative scheme to encode the textual semantics into the video representation which adaptively fuses features from text and video, and encourages the visual encoder to extract more semantically consistent features. In this scheme, SAFSAR achieves alignment and fusion in a compact way. Experiments on five challenging few-shot action recognition benchmarks under various settings demonstrate that the proposed SAFSAR model significantly improves the state-of-the-art performance.

{{</citation>}}


### (45/92) A Survey of AI Text-to-Image and AI Text-to-Video Generators (Aditi Singh, 2023)

{{<citation>}}

Aditi Singh. (2023)  
**A Survey of AI Text-to-Image and AI Text-to-Video Generators**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2311.06329v1)  

---


**ABSTRACT**  
Text-to-Image and Text-to-Video AI generation models are revolutionary technologies that use deep learning and natural language processing (NLP) techniques to create images and videos from textual descriptions. This paper investigates cutting-edge approaches in the discipline of Text-to-Image and Text-to-Video AI generations. The survey provides an overview of the existing literature as well as an analysis of the approaches used in various studies. It covers data preprocessing techniques, neural network types, and evaluation metrics used in the field. In addition, the paper discusses the challenges and limitations of Text-to-Image and Text-to-Video AI generations, as well as future research directions. Overall, these models have promising potential for a wide range of applications such as video production, content creation, and digital marketing.

{{</citation>}}


### (46/92) Automatic Report Generation for Histopathology images using pre-trained Vision Transformers (Saurav Sengupta et al., 2023)

{{<citation>}}

Saurav Sengupta, Donald E. Brown. (2023)  
**Automatic Report Generation for Histopathology images using pre-trained Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: LSTM, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.06176v2)  

---


**ABSTRACT**  
Deep learning for histopathology has been successfully used for disease classification, image segmentation and more. However, combining image and text modalities using current state-of-the-art methods has been a challenge due to the high resolution of histopathology images. Automatic report generation for histopathology images is one such challenge. In this work, we show that using an existing pre-trained Vision Transformer in a two-step process of first using it to encode 4096x4096 sized patches of the Whole Slide Image (WSI) and then using it as the encoder and an LSTM decoder for report generation, we can build a fairly performant and portable report generation mechanism that takes into account the whole of the high resolution image, instead of just the patches. We are also able to use representations from an existing powerful pre-trained hierarchical vision transformer and show its usefulness in not just zero shot classification but also for report generation.

{{</citation>}}


### (47/92) Federated Learning Across Decentralized and Unshared Archives for Remote Sensing Image Classification (Barış Büyüktaş et al., 2023)

{{<citation>}}

Barış Büyüktaş, Gencer Sumbul, Begüm Demir. (2023)  
**Federated Learning Across Decentralized and Unshared Archives for Remote Sensing Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2311.06141v1)  

---


**ABSTRACT**  
Federated learning (FL) enables the collaboration of multiple deep learning models to learn from decentralized data archives (i.e., clients) without accessing data on clients. Although FL offers ample opportunities in knowledge discovery from distributed image archives, it is seldom considered in remote sensing (RS). In this paper, as a first time in RS, we present a comparative study of state-of-the-art FL algorithms. To this end, we initially provide a systematic review of the FL algorithms presented in the computer vision community for image classification problems, and select several state-of-the-art FL algorithms based on their effectiveness with respect to training data heterogeneity across clients (known as non-IID data). After presenting an extensive overview of the selected algorithms, a theoretical comparison of the algorithms is conducted based on their: 1) local training complexity; 2) aggregation complexity; 3) learning efficiency; 4) communication cost; and 5) scalability in terms of number of clients. As the classification task, we consider multi-label classification (MLC) problem since RS images typically consist of multiple classes, and thus can simultaneously be associated with multi-labels. After the theoretical comparison, experimental analyses are presented to compare them under different decentralization scenarios in terms of MLC performance. Based on our comprehensive analyses, we finally derive a guideline for selecting suitable FL algorithms in RS. The code of this work will be publicly available at https://git.tu-berlin.de/rsim/FL-RS.

{{</citation>}}


### (48/92) MonoProb: Self-Supervised Monocular Depth Estimation with Interpretable Uncertainty (Rémi Marsal et al., 2023)

{{<citation>}}

Rémi Marsal, Florian Chabot, Angelique Loesch, William Grolleau, Hichem Sahbi. (2023)  
**MonoProb: Self-Supervised Monocular Depth Estimation with Interpretable Uncertainty**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.06137v1)  

---


**ABSTRACT**  
Self-supervised monocular depth estimation methods aim to be used in critical applications such as autonomous vehicles for environment analysis. To circumvent the potential imperfections of these approaches, a quantification of the prediction confidence is crucial to guide decision-making systems that rely on depth estimation. In this paper, we propose MonoProb, a new unsupervised monocular depth estimation method that returns an interpretable uncertainty, which means that the uncertainty reflects the expected error of the network in its depth predictions. We rethink the stereo or the structure-from-motion paradigms used to train unsupervised monocular depth models as a probabilistic problem. Within a single forward pass inference, this model provides a depth prediction and a measure of its confidence, without increasing the inference time. We then improve the performance on depth and uncertainty with a novel self-distillation loss for which a student is supervised by a pseudo ground truth that is a probability distribution on depth output by a teacher. To quantify the performance of our models we design new metrics that, unlike traditional ones, measure the absolute performance of uncertainty predictions. Our experiments highlight enhancements achieved by our method on standard depth and uncertainty metrics as well as on our tailored metrics. https://github.com/CEA-LIST/MonoProb

{{</citation>}}


### (49/92) Dual input stream transformer for eye-tracking line assignment (Thomas M. Mercier et al., 2023)

{{<citation>}}

Thomas M. Mercier, Marcin Budka, Martin R. Vasilev, Julie A. Kirkby, Bernhard Angele, Timothy J. Slattery. (2023)  
**Dual input stream transformer for eye-tracking line assignment**  

---
Primary Category: cs.CV  
Categories: 91Cxx, J-4, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.06095v1)  

---


**ABSTRACT**  
We introduce a novel Dual Input Stream Transformer (DIST) for the challenging problem of assigning fixation points from eye-tracking data collected during passage reading to the line of text that the reader was actually focused on. This post-processing step is crucial for analysis of the reading data due to the presence of noise in the form of vertical drift. We evaluate DIST against nine classical approaches on a comprehensive suite of nine diverse datasets, and demonstrate DIST's superiority. By combining multiple instances of the DIST model in an ensemble we achieve an average accuracy of 98.5\% across all datasets. Our approach presents a significant step towards addressing the bottleneck of manual line assignment in reading research. Through extensive model analysis and ablation studies, we identify key factors that contribute to DIST's success, including the incorporation of line overlap features and the use of a second input stream. Through evaluation on a set of diverse datasets we demonstrate that DIST is robust to various experimental setups, making it a safe first choice for practitioners in the field.

{{</citation>}}


### (50/92) Enhancing Rock Image Segmentation in Digital Rock Physics: A Fusion of Generative AI and State-of-the-Art Neural Networks (Zhaoyang Ma et al., 2023)

{{<citation>}}

Zhaoyang Ma, Xupeng He, Hyung Kwak, Jun Gao, Shuyu Sun, Bicheng Yan. (2023)  
**Enhancing Rock Image Segmentation in Digital Rock Physics: A Fusion of Generative AI and State-of-the-Art Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI, Attention, Generative AI, Transformer  
[Paper Link](http://arxiv.org/abs/2311.06079v1)  

---


**ABSTRACT**  
In digital rock physics, analysing microstructures from CT and SEM scans is crucial for estimating properties like porosity and pore connectivity. Traditional segmentation methods like thresholding and CNNs often fall short in accurately detailing rock microstructures and are prone to noise. U-Net improved segmentation accuracy but required many expert-annotated samples, a laborious and error-prone process due to complex pore shapes. Our study employed an advanced generative AI model, the diffusion model, to overcome these limitations. This model generated a vast dataset of CT/SEM and binary segmentation pairs from a small initial dataset. We assessed the efficacy of three neural networks: U-Net, Attention-U-net, and TransUNet, for segmenting these enhanced images. The diffusion model proved to be an effective data augmentation technique, improving the generalization and robustness of deep learning models. TransU-Net, incorporating Transformer structures, demonstrated superior segmentation accuracy and IoU metrics, outperforming both U-Net and Attention-U-net. Our research advances rock image segmentation by combining the diffusion model with cutting-edge neural networks, reducing dependency on extensive expert data and boosting segmentation accuracy and robustness. TransU-Net sets a new standard in digital rock physics, paving the way for future geoscience and engineering breakthroughs.

{{</citation>}}


### (51/92) Learning-Based Biharmonic Augmentation for Point Cloud Classification (Jiacheng Wei et al., 2023)

{{<citation>}}

Jiacheng Wei, Guosheng Lin, Henghui Ding, Jie Hu, Kim-Hui Yap. (2023)  
**Learning-Based Biharmonic Augmentation for Point Cloud Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.06070v1)  

---


**ABSTRACT**  
Point cloud datasets often suffer from inadequate sample sizes in comparison to image datasets, making data augmentation challenging. While traditional methods, like rigid transformations and scaling, have limited potential in increasing dataset diversity due to their constraints on altering individual sample shapes, we introduce the Biharmonic Augmentation (BA) method. BA is a novel and efficient data augmentation technique that diversifies point cloud data by imposing smooth non-rigid deformations on existing 3D structures. This approach calculates biharmonic coordinates for the deformation function and learns diverse deformation prototypes. Utilizing a CoefNet, our method predicts coefficients to amalgamate these prototypes, ensuring comprehensive deformation. Moreover, we present AdvTune, an advanced online augmentation system that integrates adversarial training. This system synergistically refines the CoefNet and the classification network, facilitating the automated creation of adaptive shape deformations contingent on the learner status. Comprehensive experimental analysis validates the superiority of Biharmonic Augmentation, showcasing notable performance improvements over prevailing point cloud augmentation techniques across varied network designs.

{{</citation>}}


### (52/92) Deep learning for 3D Object Detection and Tracking in Autonomous Driving: A Brief Survey (Yang Peng, 2023)

{{<citation>}}

Yang Peng. (2023)  
**Deep learning for 3D Object Detection and Tracking in Autonomous Driving: A Brief Survey**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.06043v1)  

---


**ABSTRACT**  
Object detection and tracking are vital and fundamental tasks for autonomous driving, aiming at identifying and locating objects from those predefined categories in a scene. 3D point cloud learning has been attracting more and more attention among all other forms of self-driving data. Currently, there are many deep learning methods for 3D object detection. However, the tasks of object detection and tracking for point clouds still need intensive study due to the unique characteristics of point cloud data. To help get a good grasp of the present situation of this research, this paper shows recent advances in deep learning methods for 3D object detection and tracking.

{{</citation>}}


### (53/92) 2D Image head pose estimation via latent space regression under occlusion settings (José Celestino et al., 2023)

{{<citation>}}

José Celestino, Manuel Marques, Jacinto C. Nascimento, João Paulo Costeira. (2023)  
**2D Image head pose estimation via latent space regression under occlusion settings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2311.06038v1)  

---


**ABSTRACT**  
Head orientation is a challenging Computer Vision problem that has been extensively researched having a wide variety of applications. However, current state-of-the-art systems still underperform in the presence of occlusions and are unreliable for many task applications in such scenarios. This work proposes a novel deep learning approach for the problem of head pose estimation under occlusions. The strategy is based on latent space regression as a fundamental key to better structure the problem for occluded scenarios. Our model surpasses several state-of-the-art methodologies for occluded HPE, and achieves similar accuracy for non-occluded scenarios. We demonstrate the usefulness of the proposed approach with: (i) two synthetically occluded versions of the BIWI and AFLW2000 datasets, (ii) real-life occlusions of the Pandora dataset, and (iii) a real-life application to human-robot interaction scenarios where face occlusions often occur. Specifically, the autonomous feeding from a robotic arm.

{{</citation>}}


### (54/92) Robust Adversarial Attacks Detection for Deep Learning based Relative Pose Estimation for Space Rendezvous (Ziwei Wang et al., 2023)

{{<citation>}}

Ziwei Wang, Nabil Aouf, Jose Pizarro, Christophe Honvault. (2023)  
**Robust Adversarial Attacks Detection for Deep Learning based Relative Pose Estimation for Space Rendezvous**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CR, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Adversarial Attack, LSTM  
[Paper Link](http://arxiv.org/abs/2311.05992v1)  

---


**ABSTRACT**  
Research on developing deep learning techniques for autonomous spacecraft relative navigation challenges is continuously growing in recent years. Adopting those techniques offers enhanced performance. However, such approaches also introduce heightened apprehensions regarding the trustability and security of such deep learning methods through their susceptibility to adversarial attacks. In this work, we propose a novel approach for adversarial attack detection for deep neural network-based relative pose estimation schemes based on the explainability concept. We develop for an orbital rendezvous scenario an innovative relative pose estimation technique adopting our proposed Convolutional Neural Network (CNN), which takes an image from the chaser's onboard camera and outputs accurately the target's relative position and rotation. We perturb seamlessly the input images using adversarial attacks that are generated by the Fast Gradient Sign Method (FGSM). The adversarial attack detector is then built based on a Long Short Term Memory (LSTM) network which takes the explainability measure namely SHapley Value from the CNN-based pose estimator and flags the detection of adversarial attacks when acting. Simulation results show that the proposed adversarial attack detector achieves a detection accuracy of 99.21%. Both the deep relative pose estimator and adversarial attack detector are then tested on real data captured from our laboratory-designed setup. The experimental results from our laboratory-designed setup demonstrate that the proposed adversarial attack detector achieves an average detection accuracy of 96.29%.

{{</citation>}}


### (55/92) Vision Big Bird: Random Sparsification for Full Attention (Zhemin Zhang et al., 2023)

{{<citation>}}

Zhemin Zhang, Xun Gong. (2023)  
**Vision Big Bird: Random Sparsification for Full Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.05988v1)  

---


**ABSTRACT**  
Recently, Transformers have shown promising performance in various vision tasks. However, the high costs of global self-attention remain challenging for Transformers, especially for high-resolution vision tasks. Inspired by one of the most successful transformers-based models for NLP: Big Bird, we propose a novel sparse attention mechanism for Vision Transformers (ViT). Specifically, we separate the heads into three groups, the first group used convolutional neural network (CNN) to extract local features and provide positional information for the model, the second group used Random Sampling Windows (RS-Win) for sparse self-attention calculation, and the third group reduces the resolution of the keys and values by average pooling for global attention. Based on these components, ViT maintains the sparsity of self-attention while maintaining the merits of Big Bird (i.e., the model is a universal approximator of sequence functions and is Turing complete). Moreover, our results show that the positional encoding, a crucial component in ViTs, can be safely removed in our model. Experiments show that Vision Big Bird demonstrates competitive performance on common vision tasks.

{{</citation>}}


### (56/92) Post-training Quantization with Progressive Calibration and Activation Relaxing for Text-to-Image Diffusion Models (Siao Tang et al., 2023)

{{<citation>}}

Siao Tang, Xin Wang, Hong Chen, Chaoyu Guan, Zewen Wu, Yansong Tang, Wenwu Zhu. (2023)  
**Post-training Quantization with Progressive Calibration and Activation Relaxing for Text-to-Image Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.06322v1)  

---


**ABSTRACT**  
Diffusion models have achieved great success due to their remarkable generation ability. However, their high computational overhead is still a troublesome problem. Recent studies have leveraged post-training quantization (PTQ) to compress diffusion models. However, most of them only focus on unconditional models, leaving the quantization of widely used large pretrained text-to-image models, e.g., Stable Diffusion, largely unexplored. In this paper, we propose a novel post-training quantization method PCR (Progressive Calibration and Relaxing) for text-to-image diffusion models, which consists of a progressive calibration strategy that considers the accumulated quantization error across timesteps, and an activation relaxing strategy that improves the performance with negligible cost. Additionally, we demonstrate the previous metrics for text-to-image diffusion model quantization are not accurate due to the distribution gap. To tackle the problem, we propose a novel QDiffBench benchmark, which utilizes data in the same domain for more accurate evaluation. Besides, QDiffBench also considers the generalization performance of the quantized model outside the calibration dataset. Extensive experiments on Stable Diffusion and Stable Diffusion XL demonstrate the superiority of our method and benchmark. Moreover, we are the first to achieve quantization for Stable Diffusion XL while maintaining the performance.

{{</citation>}}


### (57/92) Automated Heterogeneous Low-Bit Quantization of Multi-Model Deep Learning Inference Pipeline (Jayeeta Mondal et al., 2023)

{{<citation>}}

Jayeeta Mondal, Swarnava Dey, Arijit Mukherjee. (2023)  
**Automated Heterogeneous Low-Bit Quantization of Multi-Model Deep Learning Inference Pipeline**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, math-OC  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.05870v1)  

---


**ABSTRACT**  
Multiple Deep Neural Networks (DNNs) integrated into single Deep Learning (DL) inference pipelines e.g. Multi-Task Learning (MTL) or Ensemble Learning (EL), etc., albeit very accurate, pose challenges for edge deployment. In these systems, models vary in their quantization tolerance and resource demands, requiring meticulous tuning for accuracy-latency balance. This paper introduces an automated heterogeneous quantization approach for DL inference pipelines with multiple DNNs.

{{</citation>}}


## eess.IV (4)



### (58/92) A design of Convolutional Neural Network model for the Diagnosis of the COVID-19 (Xinyuan Song, 2023)

{{<citation>}}

Xinyuan Song. (2023)  
**A design of Convolutional Neural Network model for the Diagnosis of the COVID-19**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06394v1)  

---


**ABSTRACT**  
With the spread of COVID-19 around the globe over the past year, the usage of artificial intelligence (AI) algorithms and image processing methods to analyze the X-ray images of patients' chest with COVID-19 has become essential. The COVID-19 virus recognition in the lung area of a patient is one of the basic and essential needs of clicical centers and hospitals. Most research in this field has been devoted to papers on the basis of deep learning methods utilizing CNNs (Convolutional Neural Network), which mainly deal with the screening of sick and healthy people.In this study, a new structure of a 19-layer CNN has been recommended for accurately recognition of the COVID-19 from the X-ray pictures of chest. The offered CNN is developed to serve as a precise diagnosis system for a three class (viral pneumonia, Normal, COVID) and a four classclassification (Lung opacity, Normal, COVID-19, and pneumonia). A comparison is conducted among the outcomes of the offered procedure and some popular pretrained networks, including Inception, Alexnet, ResNet50, Squeezenet, and VGG19 and based on Specificity, Accuracy, Precision, Sensitivity, Confusion Matrix, and F1-score. The experimental results of the offered CNN method specify its dominance over the existing published procedures. This method can be a useful tool for clinicians in deciding properly about COVID-19.

{{</citation>}}


### (59/92) Exploring the Efficacy of Base Data Augmentation Methods in Deep Learning-Based Radiograph Classification of Knee Joint Osteoarthritis (Fabi Prezja et al., 2023)

{{<citation>}}

Fabi Prezja, Leevi Annala, Sampsa Kiiskinen, Timo Ojala. (2023)  
**Exploring the Efficacy of Base Data Augmentation Methods in Deep Learning-Based Radiograph Classification of Knee Joint Osteoarthritis**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.06118v1)  

---


**ABSTRACT**  
Diagnosing knee joint osteoarthritis (KOA), a major cause of disability worldwide, is challenging due to subtle radiographic indicators and the varied progression of the disease. Using deep learning for KOA diagnosis requires broad, comprehensive datasets. However, obtaining these datasets poses significant challenges due to patient privacy concerns and data collection restrictions. Additive data augmentation, which enhances data variability, emerges as a promising solution. Yet, it's unclear which augmentation techniques are most effective for KOA. This study explored various data augmentation methods, including adversarial augmentations, and their impact on KOA classification model performance. While some techniques improved performance, others commonly used underperformed. We identified potential confounding regions within the images using adversarial augmentation. This was evidenced by our models' ability to classify KL0 and KL4 grades accurately, with the knee joint omitted. This observation suggested a model bias, which might leverage unrelated features for classification currently present in radiographs. Interestingly, removing the knee joint also led to an unexpected improvement in KL1 classification accuracy. To better visualize these paradoxical effects, we employed Grad-CAM, highlighting the associated regions. Our study underscores the need for careful technique selection for improved model performance and identifying and managing potential confounding regions in radiographic KOA deep learning.

{{</citation>}}


### (60/92) Ulcerative Colitis Mayo Endoscopic Scoring Classification with Active Learning and Generative Data Augmentation (Ümit Mert Çağlar et al., 2023)

{{<citation>}}

Ümit Mert Çağlar, Alperen İnci, Oğuz Hanoğlu, Görkem Polat, Alptekin Temizel. (2023)  
**Ulcerative Colitis Mayo Endoscopic Scoring Classification with Active Learning and Generative Data Augmentation**  

---
Primary Category: eess.IV  
Categories: I-2-1; I-4-9, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Active Learning, Augmentation  
[Paper Link](http://arxiv.org/abs/2311.06057v1)  

---


**ABSTRACT**  
Endoscopic imaging is commonly used to diagnose Ulcerative Colitis (UC) and classify its severity. It has been shown that deep learning based methods are effective in automated analysis of these images and can potentially be used to aid medical doctors. Unleashing the full potential of these methods depends on the availability of large amount of labeled images; however, obtaining and labeling these images are quite challenging. In this paper, we propose a active learning based generative augmentation method. The method involves generating a large number of synthetic samples by training using a small dataset consisting of real endoscopic images. The resulting data pool is narrowed down by using active learning methods to select the most informative samples, which are then used to train a classifier. We demonstrate the effectiveness of our method through experiments on a publicly available endoscopic image dataset. The results show that using synthesized samples in conjunction with active learning leads to improved classification performance compared to using only the original labeled examples and the baseline classification performance of 68.1% increases to 74.5% in terms of Quadratic Weighted Kappa (QWK) Score. Another observation is that, attaining equivalent performance using only real data necessitated three times higher number of images.

{{</citation>}}


### (61/92) Polar-Net: A Clinical-Friendly Model for Alzheimer's Disease Detection in OCTA Images (Shouyue Liu et al., 2023)

{{<citation>}}

Shouyue Liu, Jinkui Hao, Yanwu Xu, Huazhu Fu, Xinyu Guo, Jiang Liu, Yalin Zheng, Yonghuai Liu, Jiong Zhang, Yitian Zhao. (2023)  
**Polar-Net: A Clinical-Friendly Model for Alzheimer's Disease Detection in OCTA Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2311.06009v1)  

---


**ABSTRACT**  
Optical Coherence Tomography Angiography (OCTA) is a promising tool for detecting Alzheimer's disease (AD) by imaging the retinal microvasculature. Ophthalmologists commonly use region-based analysis, such as the ETDRS grid, to study OCTA image biomarkers and understand the correlation with AD. However, existing studies have used general deep computer vision methods, which present challenges in providing interpretable results and leveraging clinical prior knowledge. To address these challenges, we propose a novel deep-learning framework called Polar-Net. Our approach involves mapping OCTA images from Cartesian coordinates to polar coordinates, which allows for the use of approximate sector convolution and enables the implementation of the ETDRS grid-based regional analysis method commonly used in clinical practice. Furthermore, Polar-Net incorporates clinical prior information of each sector region into the training process, which further enhances its performance. Additionally, our framework adapts to acquire the importance of the corresponding retinal region, which helps researchers and clinicians understand the model's decision-making process in detecting AD and assess its conformity to clinical observations. Through evaluations on private and public datasets, we have demonstrated that Polar-Net outperforms existing state-of-the-art methods and provides more valuable pathological evidence for the association between retinal vascular changes and AD. In addition, we also show that the two innovative modules introduced in our framework have a significant impact on improving overall performance.

{{</citation>}}


## cs.AI (7)



### (62/92) ChatGPT in the context of precision agriculture data analytics (Ilyas Potamitis, 2023)

{{<citation>}}

Ilyas Potamitis. (2023)  
**ChatGPT in the context of precision agriculture data analytics**  

---
Primary Category: cs.AI  
Categories: 68Txx, cs-AI, cs.AI, eess-SP  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.06390v1)  

---


**ABSTRACT**  
In this study we argue that integrating ChatGPT into the data processing pipeline of automated sensors in precision agriculture has the potential to bring several benefits and enhance various aspects of modern farming practices. Policy makers often face a barrier when they need to get informed about the situation in vast agricultural fields to reach to decisions. They depend on the close collaboration between agricultural experts in the field, data analysts, and technology providers to create interdisciplinary teams that cannot always be secured on demand or establish effective communication across these diverse domains to respond in real-time. In this work we argue that the speech recognition input modality of ChatGPT provides a more intuitive and natural way for policy makers to interact with the database of the server of an agricultural data processing system to which a large, dispersed network of automated insect traps and sensors probes reports. The large language models map the speech input to text, allowing the user to form its own version of unconstrained verbal query, raising the barrier of having to learn and adapt oneself to a specific data analytics software. The output of the language model can interact through Python code and Pandas with the entire database, visualize the results and use speech synthesis to engage the user in an iterative and refining discussion related to the data. We show three ways of how ChatGPT can interact with the database of the remote server to which a dispersed network of different modalities (optical counters, vibration recordings, pictures, and video), report. We examine the potential and the validity of the response of ChatGPT in analyzing, and interpreting agricultural data, providing real time insights and recommendations to stakeholders

{{</citation>}}


### (63/92) Smart Agent-Based Modeling: On the Use of Large Language Models in Computer Simulations (Zengqing Wu et al., 2023)

{{<citation>}}

Zengqing Wu, Run Peng, Xu Han, Shuyuan Zheng, Yixin Zhang, Chuan Xiao. (2023)  
**Smart Agent-Based Modeling: On the Use of Large Language Models in Computer Simulations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs-MA, cs.AI, econ-GN, q-fin-EC  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.06330v1)  

---


**ABSTRACT**  
Computer simulations offer a robust toolset for exploring complex systems across various disciplines. A particularly impactful approach within this realm is Agent-Based Modeling (ABM), which harnesses the interactions of individual agents to emulate intricate system dynamics. ABM's strength lies in its bottom-up methodology, illuminating emergent phenomena by modeling the behaviors of individual components of a system. Yet, ABM has its own set of challenges, notably its struggle with modeling natural language instructions and common sense in mathematical equations or rules. This paper seeks to transcend these boundaries by integrating Large Language Models (LLMs) like GPT into ABM. This amalgamation gives birth to a novel framework, Smart Agent-Based Modeling (SABM). Building upon the concept of smart agents -- entities characterized by their intelligence, adaptability, and computation ability -- we explore in the direction of utilizing LLM-powered agents to simulate real-world scenarios with increased nuance and realism. In this comprehensive exploration, we elucidate the state of the art of ABM, introduce SABM's potential and methodology, and present three case studies (source codes available at https://github.com/Roihn/SABM), demonstrating the SABM methodology and validating its effectiveness in modeling real-world systems. Furthermore, we cast a vision towards several aspects of the future of SABM, anticipating a broader horizon for its applications. Through this endeavor, we aspire to redefine the boundaries of computer simulations, enabling a more profound understanding of complex systems.

{{</citation>}}


### (64/92) Search-Based Fairness Testing: An Overview (Hussaini Mamman et al., 2023)

{{<citation>}}

Hussaini Mamman, Shuib Basri, Abdullateef Oluwaqbemiga Balogun, Abdullahi Abubakar Imam, Ganesh Kumar, Luiz Fernando Capretz. (2023)  
**Search-Based Fairness Testing: An Overview**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06175v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) has demonstrated remarkable capabilities in domains such as recruitment, finance, healthcare, and the judiciary. However, biases in AI systems raise ethical and societal concerns, emphasizing the need for effective fairness testing methods. This paper reviews current research on fairness testing, particularly its application through search-based testing. Our analysis highlights progress and identifies areas of improvement in addressing AI systems biases. Future research should focus on leveraging established search-based testing methodologies for fairness testing.

{{</citation>}}


### (65/92) JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models (Zihao Wang et al., 2023)

{{<citation>}}

Zihao Wang, Shaofei Cai, Anji Liu, Yonggang Jin, Jinbing Hou, Bowei Zhang, Haowei Lin, Zhaofeng He, Zilong Zheng, Yaodong Yang, Xiaojian Ma, Yitao Liang. (2023)  
**JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.05997v1)  

---


**ABSTRACT**  
Achieving human-like planning and control with multimodal observations in an open world is a key milestone for more functional generalist agents. Existing approaches can handle certain long-horizon tasks in an open world. However, they still struggle when the number of open-world tasks could potentially be infinite and lack the capability to progressively enhance task completion as game time progresses. We introduce JARVIS-1, an open-world agent that can perceive multimodal input (visual observations and human instructions), generate sophisticated plans, and perform embodied control, all within the popular yet challenging open-world Minecraft universe. Specifically, we develop JARVIS-1 on top of pre-trained multimodal language models, which map visual observations and textual instructions to plans. The plans will be ultimately dispatched to the goal-conditioned controllers. We outfit JARVIS-1 with a multimodal memory, which facilitates planning using both pre-trained knowledge and its actual game survival experiences. In our experiments, JARVIS-1 exhibits nearly perfect performances across over 200 varying tasks from the Minecraft Universe Benchmark, ranging from entry to intermediate levels. JARVIS-1 has achieved a completion rate of 12.5% in the long-horizon diamond pickaxe task. This represents a significant increase up to 5 times compared to previous records. Furthermore, we show that JARVIS-1 is able to $\textit{self-improve}$ following a life-long learning paradigm thanks to multimodal memory, sparking a more general intelligence and improved autonomy. The project page is available at https://craftjarvis-jarvis1.github.io.

{{</citation>}}


### (66/92) A Decision Support System for Liver Diseases Prediction: Integrating Batch Processing, Rule-Based Event Detection and SPARQL Query (Ritesh Chandra et al., 2023)

{{<citation>}}

Ritesh Chandra, Sadhana Tiwari, Satyam Rastogi, Sonali Agarwal. (2023)  
**A Decision Support System for Liver Diseases Prediction: Integrating Batch Processing, Rule-Based Event Detection and SPARQL Query**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Event Detection  
[Paper Link](http://arxiv.org/abs/2311.07595v1)  

---


**ABSTRACT**  
Liver diseases pose a significant global health burden, impacting a substantial number of individuals and exerting substantial economic and social consequences. Rising liver problems are considered a fatal disease in many countries, such as Egypt, Molda, etc. The objective of this study is to construct a predictive model for liver illness using Basic Formal Ontology (BFO) and detection rules derived from a decision tree algorithm. Based on these rules, events are detected through batch processing using the Apache Jena framework. Based on the event detected, queries can be directly processed using SPARQL. To make the ontology operational, these Decision Tree (DT) rules are converted into Semantic Web Rule Language (SWRL). Using this SWRL in the ontology for predicting different types of liver disease with the help of the Pellet and Drool inference engines in Protege Tools, a total of 615 records are taken from different liver diseases. After inferring the rules, the result can be generated for the patient according to the DT rules, and other patient-related details along with different precautionary suggestions can be obtained based on these results. Combining query results of batch processing and ontology-generated results can give more accurate suggestions for disease prevention and detection. This work aims to provide a comprehensive approach that is applicable for liver disease prediction, rich knowledge graph representation, and smart querying capabilities. The results show that combining RDF data, SWRL rules, and SPARQL queries for analysing and predicting liver disease can help medical professionals to learn more about liver diseases and make a Decision Support System (DSS) for health care.

{{</citation>}}


### (67/92) Cognitive Architecture Toward Common Ground Sharing Among Humans and Generative AIs: Trial on Model-Model Interactions in Tangram Naming Task (Junya Morita et al., 2023)

{{<citation>}}

Junya Morita, Tatsuya Yui, Takeru Amaya, Ryuichiro Higashinaka, Yugo Takeuchi. (2023)  
**Cognitive Architecture Toward Common Ground Sharing Among Humans and Generative AIs: Trial on Model-Model Interactions in Tangram Naming Task**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.05851v1)  

---


**ABSTRACT**  
For generative AIs to be trustworthy, establishing transparent common grounding with humans is essential. As a preparation toward human-model common grounding, this study examines the process of model-model common grounding. In this context, common ground is defined as a cognitive framework shared among agents in communication, enabling the connection of symbols exchanged between agents to the meanings inherent in each agent. This connection is facilitated by a shared cognitive framework among the agents involved. In this research, we focus on the tangram naming task (TNT) as a testbed to examine the common-ground-building process. Unlike previous models designed for this task, our approach employs generative AIs to visualize the internal processes of the model. In this task, the sender constructs a metaphorical image of an abstract figure within the model and generates a detailed description based on this image. The receiver interprets the generated description from the partner by constructing another image and reconstructing the original abstract figure. Preliminary results from the study show an improvement in task performance beyond the chance level, indicating the effect of the common cognitive framework implemented in the models. Additionally, we observed that incremental backpropagations leveraging successful communication cases for a component of the model led to a statistically significant increase in performance. These results provide valuable insights into the mechanisms of common grounding made by generative AIs, improving human communication with the evolving intelligent machines in our future society.

{{</citation>}}


### (68/92) Model-as-a-Service (MaaS): A Survey (Wensheng Gan et al., 2023)

{{<citation>}}

Wensheng Gan, Shicheng Wan, Philip S. Yu. (2023)  
**Model-as-a-Service (MaaS): A Survey**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05804v1)  

---


**ABSTRACT**  
Due to the increased number of parameters and data in the pre-trained model exceeding a certain level, a foundation model (e.g., a large language model) can significantly improve downstream task performance and emerge with some novel special abilities (e.g., deep learning, complex reasoning, and human alignment) that were not present before. Foundation models are a form of generative artificial intelligence (GenAI), and Model-as-a-Service (MaaS) has emerged as a groundbreaking paradigm that revolutionizes the deployment and utilization of GenAI models. MaaS represents a paradigm shift in how we use AI technologies and provides a scalable and accessible solution for developers and users to leverage pre-trained AI models without the need for extensive infrastructure or expertise in model training. In this paper, the introduction aims to provide a comprehensive overview of MaaS, its significance, and its implications for various industries. We provide a brief review of the development history of "X-as-a-Service" based on cloud computing and present the key technologies involved in MaaS. The development of GenAI models will become more democratized and flourish. We also review recent application studies of MaaS. Finally, we highlight several challenges and future issues in this promising area. MaaS is a new deployment and service paradigm for different AI-based models. We hope this review will inspire future research in the field of MaaS.

{{</citation>}}


## cs.CY (1)



### (69/92) Vox Populi, Vox ChatGPT: Large Language Models, Education and Democracy (Niina Zuber et al., 2023)

{{<citation>}}

Niina Zuber, Jan Gogoll. (2023)  
**Vox Populi, Vox ChatGPT: Large Language Models, Education and Democracy**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.06207v1)  

---


**ABSTRACT**  
In the era of generative AI and specifically large language models (LLMs), exemplified by ChatGPT, the intersection of artificial intelligence and human reasoning has become a focal point of global attention. Unlike conventional search engines, LLMs go beyond mere information retrieval, entering into the realm of discourse culture. Its outputs mimic well-considered, independent opinions or statements of facts, presenting a pretense of wisdom. This paper explores the potential transformative impact of LLMs on democratic societies. It delves into the concerns regarding the difficulty in distinguishing ChatGPT-generated texts from human output. The discussion emphasizes the essence of authorship, rooted in the unique human capacity for reason - a quality indispensable for democratic discourse and successful collaboration within free societies. Highlighting the potential threats to democracy, this paper presents three arguments: the Substitution argument, the Authenticity argument, and the Facts argument. These arguments highlight the potential risks that are associated with an overreliance on LLMs. The central thesis posits that widespread deployment of LLMs may adversely affect the fabric of a democracy if not comprehended and addressed proactively and properly. In proposing a solution, we advocate for an emphasis on education as a means to mitigate risks. We suggest cultivating thinking skills in children, fostering coherent thought formulation, and distinguishing between machine-generated output and genuine, i.e. human, reasoning. The focus should be on responsible development and usage of LLMs, with the goal of augmenting human capacities in thinking, deliberating and decision-making rather than substituting them.

{{</citation>}}


## cs.RO (3)



### (70/92) Multi-Agent Reinforcement Learning for the Low-Level Control of a Quadrotor UAV (Beomyeol Yu et al., 2023)

{{<citation>}}

Beomyeol Yu, Taeyoung Lee. (2023)  
**Multi-Agent Reinforcement Learning for the Low-Level Control of a Quadrotor UAV**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.06144v1)  

---


**ABSTRACT**  
This paper presents multi-agent reinforcement learning frameworks for the low-level control of a quadrotor UAV. While single-agent reinforcement learning has been successfully applied to quadrotors, training a single monolithic network is often data-intensive and time-consuming. To address this, we decompose the quadrotor dynamics into the translational dynamics and the yawing dynamics, and assign a reinforcement learning agent to each part for efficient training and performance improvements. The proposed multi-agent framework for quadrotor low-level control that leverages the underlying structures of the quadrotor dynamics is a unique contribution. Further, we introduce regularization terms to mitigate steady-state errors and to avoid aggressive control inputs. Through benchmark studies with sim-to-sim transfer, it is illustrated that the proposed multi-agent reinforcement learning substantially improves the convergence rate of the training and the stability of the controlled dynamics.

{{</citation>}}


### (71/92) RSG: Fast Learning Adaptive Skills for Quadruped Robots by Skill Graph (Hongyin Zhang et al., 2023)

{{<citation>}}

Hongyin Zhang, Diyuan Shi, Zifeng Zhuang, Han Zhao, Zhenyu Wei, Feng Zhao, Sibo Gai, Shangke Lyu, Donglin Wang. (2023)  
**RSG: Fast Learning Adaptive Skills for Quadruped Robots by Skill Graph**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.06015v1)  

---


**ABSTRACT**  
Developing robotic intelligent systems that can adapt quickly to unseen wild situations is one of the critical challenges in pursuing autonomous robotics. Although some impressive progress has been made in walking stability and skill learning in the field of legged robots, their ability to fast adaptation is still inferior to that of animals in nature. Animals are born with massive skills needed to survive, and can quickly acquire new ones, by composing fundamental skills with limited experience. Inspired by this, we propose a novel framework, named Robot Skill Graph (RSG) for organizing massive fundamental skills of robots and dexterously reusing them for fast adaptation. Bearing a structure similar to the Knowledge Graph (KG), RSG is composed of massive dynamic behavioral skills instead of static knowledge in KG and enables discovering implicit relations that exist in be-tween of learning context and acquired skills of robots, serving as a starting point for understanding subtle patterns existing in robots' skill learning. Extensive experimental results demonstrate that RSG can provide rational skill inference upon new tasks and environments and enable quadruped robots to adapt to new scenarios and learn new skills rapidly.

{{</citation>}}


### (72/92) Towards Interpretable Motion-level Skill Assessment in Robotic Surgery (Kay Hutchinson et al., 2023)

{{<citation>}}

Kay Hutchinson, Katherina Chen, Homa Alemzadeh. (2023)  
**Towards Interpretable Motion-level Skill Assessment in Robotic Surgery**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AWS  
[Paper Link](http://arxiv.org/abs/2311.05838v1)  

---


**ABSTRACT**  
Purpose: We study the relationship between surgical gestures and motion primitives in dry-lab surgical exercises towards a deeper understanding of surgical activity at fine-grained levels and interpretable feedback in skill assessment.   Methods: We analyze the motion primitive sequences of gestures in the JIGSAWS dataset and identify inverse motion primitives in those sequences. Inverse motion primitives are defined as sequential actions on the same object by the same tool that effectively negate each other. We also examine the correlation between surgical skills (measured by GRS scores) and the number and total durations of inverse motion primitives in the dry-lab trials of Suturing, Needle Passing, and Knot Tying tasks.   Results: We find that the sequence of motion primitives used to perform gestures can help detect labeling errors in surgical gestures. Inverse motion primitives are often used as recovery actions to correct the position or orientation of objects or may be indicative of other issues such as with depth perception. The number and total durations of inverse motion primitives in trials are also strongly correlated with lower GRS scores in the Suturing and Knot Tying tasks.   Conclusion: The sequence and pattern of motion primitives could be used to provide interpretable feedback in surgical skill assessment. Combined with an action recognition model, the explainability of automated skill assessment can be improved by showing video clips of the inverse motion primitives of inefficient or problematic movements.

{{</citation>}}


## cs.IT (1)



### (73/92) In-Context Learning for MIMO Equalization Using Transformer-Based Sequence Models (Matteo Zecchin et al., 2023)

{{<citation>}}

Matteo Zecchin, Kai Yu, Osvaldo Simeone. (2023)  
**In-Context Learning for MIMO Equalization Using Transformer-Based Sequence Models**  

---
Primary Category: cs.IT  
Categories: cs-AI, cs-IT, cs-LG, cs.IT, math-IT  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.06101v1)  

---


**ABSTRACT**  
Large pre-trained sequence models, such as transformer-based architectures, have been recently shown to have the capacity to carry out in-context learning (ICL). In ICL, a decision on a new input is made via a direct mapping of the input and of a few examples from the given task, serving as the task's context, to the output variable. No explicit updates of model parameters are needed to tailor the decision to a new task. Pre-training, which amounts to a form of meta-learning, is based on the observation of examples from several related tasks. Prior work has shown ICL capabilities for linear regression. In this study, we leverage ICL to address the inverse problem of multiple-input and multiple-output (MIMO) equalization based on a context given by pilot symbols. A task is defined by the unknown fading channel and by the signal-to-noise ratio (SNR) level, which may be known. To highlight the practical potential of the approach, we allow for the presence of quantization of the received signals. We demonstrate via numerical results that transformer-based ICL has a threshold behavior, whereby, as the number of pre-training tasks grows, the performance switches from that of a minimum mean squared error (MMSE) equalizer with a prior determined by the pre-trained tasks to that of an MMSE equalizer with the true data-generating prior.

{{</citation>}}


## cs.CR (2)



### (74/92) A high throughput Intrusion Detection System (IDS) to enhance the security of data transmission among research centers (Marco Grossi et al., 2023)

{{<citation>}}

Marco Grossi, Fabrizio Alfonsi, Marco Prandini, Alessandro Gabrielli. (2023)  
**A high throughput Intrusion Detection System (IDS) to enhance the security of data transmission among research centers**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SY, cs.CR, eess-SY  
Keywords: Intrusion Detection, Intrusion Prevention  
[Paper Link](http://arxiv.org/abs/2311.06082v1)  

---


**ABSTRACT**  
Data breaches and cyberattacks represent a severe problem in higher education institutions and universities that can result in illegal access to sensitive information and data loss. To enhance the security of data transmission, Intrusion Prevention Systems (IPS, i.e., firewalls) and Intrusion Detection Systems (IDS, i.e., packet sniffers) are used to detect potential threats in the exchanged data. IPSs and IDSs are usually designed as software programs running on a server machine. However, when the speed of exchanged data is too high, this solution can become unreliable. In this case, IPSs and IDSs designed on a real hardware platform, such as ASICs and FPGAs, represent a more reliable solution. This paper presents a packet sniffer that was designed using a commercial FPGA development board. The system can support a data throughput of 10 Gbit/s with preliminary results showing that the speed of data transmission can be reliably extended to 100 Gbit/s. The designed system is highly configurable by the user and can enhance the data protection of information transmitted using the Ethernet protocol. It is particularly suited for the security of universities and research centers, where point-to-point network connections are dominant and large amount of sensitive data are shared among different hosts.

{{</citation>}}


### (75/92) Watermarking Vision-Language Pre-trained Models for Multi-modal Embedding as a Service (Yuanmin Tang et al., 2023)

{{<citation>}}

Yuanmin Tang, Jing Yu, Keke Gai, Xiangyan Qu, Yue Hu, Gang Xiong, Qi Wu. (2023)  
**Watermarking Vision-Language Pre-trained Models for Multi-modal Embedding as a Service**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs.CR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.05863v1)  

---


**ABSTRACT**  
Recent advances in vision-language pre-trained models (VLPs) have significantly increased visual understanding and cross-modal analysis capabilities. Companies have emerged to provide multi-modal Embedding as a Service (EaaS) based on VLPs (e.g., CLIP-based VLPs), which cost a large amount of training data and resources for high-performance service. However, existing studies indicate that EaaS is vulnerable to model extraction attacks that induce great loss for the owners of VLPs. Protecting the intellectual property and commercial ownership of VLPs is increasingly crucial yet challenging. A major solution of watermarking model for EaaS implants a backdoor in the model by inserting verifiable trigger embeddings into texts, but it is only applicable for large language models and is unrealistic due to data and model privacy. In this paper, we propose a safe and robust backdoor-based embedding watermarking method for VLPs called VLPMarker. VLPMarker utilizes embedding orthogonal transformation to effectively inject triggers into the VLPs without interfering with the model parameters, which achieves high-quality copyright verification and minimal impact on model performance. To enhance the watermark robustness, we further propose a collaborative copyright verification strategy based on both backdoor trigger and embedding distribution, enhancing resilience against various attacks. We increase the watermark practicality via an out-of-distribution trigger selection approach, removing access to the model training data and thus making it possible for many real-world scenarios. Our extensive experiments on various datasets indicate that the proposed watermarking approach is effective and safe for verifying the copyright of VLPs for multi-modal EaaS and robust against model extraction attacks. Our code is available at https://github.com/Pter61/vlpmarker.

{{</citation>}}


## cs.IR (10)



### (76/92) Attributes Grouping and Mining Hashing for Fine-Grained Image Retrieval (Xin Lu et al., 2023)

{{<citation>}}

Xin Lu, Shikun Chen, Yichao Cao, Xin Zhou, Xiaobo Lu. (2023)  
**Attributes Grouping and Mining Hashing for Fine-Grained Image Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-CV, cs-IR, cs.IR  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.06067v1)  

---


**ABSTRACT**  
In recent years, hashing methods have been popular in the large-scale media search for low storage and strong representation capabilities. To describe objects with similar overall appearance but subtle differences, more and more studies focus on hashing-based fine-grained image retrieval. Existing hashing networks usually generate both local and global features through attention guidance on the same deep activation tensor, which limits the diversity of feature representations. To handle this limitation, we substitute convolutional descriptors for attention-guided features and propose an Attributes Grouping and Mining Hashing (AGMH), which groups and embeds the category-specific visual attributes in multiple descriptors to generate a comprehensive feature representation for efficient fine-grained image retrieval. Specifically, an Attention Dispersion Loss (ADL) is designed to force the descriptors to attend to various local regions and capture diverse subtle details. Moreover, we propose a Stepwise Interactive External Attention (SIEA) to mine critical attributes in each descriptor and construct correlations between fine-grained attributes and objects. The attention mechanism is dedicated to learning discrete attributes, which will not cost additional computations in hash codes generation. Finally, the compact binary codes are learned by preserving pairwise similarities. Experimental results demonstrate that AGMH consistently yields the best performance against state-of-the-art methods on fine-grained benchmark datasets.

{{</citation>}}


### (77/92) Reviewing Developments of Graph Convolutional Network Techniques for Recommendation Systems (Haojun Zhu et al., 2023)

{{<citation>}}

Haojun Zhu, Vikram Kapoor, Priya Sharma. (2023)  
**Reviewing Developments of Graph Convolutional Network Techniques for Recommendation Systems**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2311.06323v1)  

---


**ABSTRACT**  
The Recommender system is a vital information service on today's Internet. Recently, graph neural networks have emerged as the leading approach for recommender systems. We try to review recent literature on graph neural network-based recommender systems, covering the background and development of both recommender systems and graph neural networks. Then categorizing recommender systems by their settings and graph neural networks by spectral and spatial models, we explore the motivation behind incorporating graph neural networks into recommender systems. We also analyze challenges and open problems in graph construction, embedding propagation and aggregation, and computation efficiency. This guides us to better explore the future directions and developments in this domain.

{{</citation>}}


### (78/92) ID Embedding as Subtle Features of Content and Structure for Multimodal Recommendation (Yuting Liu et al., 2023)

{{<citation>}}

Yuting Liu, Enneng Yang, Yizhou Dang, Guibing Guo, Qiang Liu, Yuliang Liang, Linying Jiang, Xingwei Wang. (2023)  
**ID Embedding as Subtle Features of Content and Structure for Multimodal Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.05956v1)  

---


**ABSTRACT**  
Multimodal recommendation aims to model user and item representations comprehensively with the involvement of multimedia content for effective recommendations. Existing research has shown that it is beneficial for recommendation performance to combine (user- and item-) ID embeddings with multimodal salient features, indicating the value of IDs. However, there is a lack of a thorough analysis of the ID embeddings in terms of feature semantics in the literature. In this paper, we revisit the value of ID embeddings for multimodal recommendation and conduct a thorough study regarding its semantics, which we recognize as subtle features of content and structures. Then, we propose a novel recommendation model by incorporating ID embeddings to enhance the semantic features of both content and structures. Specifically, we put forward a hierarchical attention mechanism to incorporate ID embeddings in modality fusing, coupled with contrastive learning, to enhance content representations. Meanwhile, we propose a lightweight graph convolutional network for each modality to amalgamate neighborhood and ID embeddings for improving structural representations. Finally, the content and structure representations are combined to form the ultimate item embedding for recommendation. Extensive experiments on three real-world datasets (Baby, Sports, and Clothing) demonstrate the superiority of our method over state-of-the-art multimodal recommendation methods and the effectiveness of fine-grained ID embeddings.

{{</citation>}}


### (79/92) Establishing Performance Baselines in Fine-Tuning, Retrieval-Augmented Generation and Soft-Prompting for Non-Specialist LLM Users (Jennifer Dodgson et al., 2023)

{{<citation>}}

Jennifer Dodgson, Lin Nanzheng, Julian Peh, Akira Rafhael Janson Pattirane, Alfath Daryl Alhajir, Eko Ridho Dinarto, Joseph Lim, Syed Danyal Ahmad. (2023)  
**Establishing Performance Baselines in Fine-Tuning, Retrieval-Augmented Generation and Soft-Prompting for Non-Specialist LLM Users**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.05903v1)  

---


**ABSTRACT**  
Research into methods for improving the performance of large language models (LLMs) through fine-tuning, retrieval-augmented generation (RAG) and soft-prompting has tended to focus on the use of highly technical or high-cost techniques, making many of the newly discovered approaches comparatively inaccessible to non-technical users. In this paper we tested an unmodified version of GPT 3.5, a fine-tuned version, and the same unmodified model when given access to a vectorised RAG database, both in isolation and in combination with a basic, non-algorithmic soft prompt. In each case we tested the model's ability to answer a set of 100 questions relating primarily to events that occurred after September 2021 (the point at which GPT 3.5's training data set ends). We found that if commercial platforms are used and default settings are applied with no iteration in order to establish a baseline set of outputs, a fine-tuned model outperforms GPT 3.5 Turbo, while the RAG approach out-performed both. The application of a soft prompt significantly improved the performance of each approach.

{{</citation>}}


### (80/92) Citation Recommendation on Scholarly Legal Articles (Doğukan Arslan et al., 2023)

{{<citation>}}

Doğukan Arslan, Saadet Sena Erdoğan, Gülşen Eryiğit. (2023)  
**Citation Recommendation on Scholarly Legal Articles**  

---
Primary Category: cs.IR  
Categories: H-3-3; I-2-7, cs-CL, cs-IR, cs.IR  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2311.05902v1)  

---


**ABSTRACT**  
Citation recommendation is the task of finding appropriate citations based on a given piece of text. The proposed datasets for this task consist mainly of several scientific fields, lacking some core ones, such as law. Furthermore, citation recommendation is used within the legal domain to identify supporting arguments, utilizing non-scholarly legal articles. In order to alleviate the limitations of existing studies, we gather the first scholarly legal dataset for the task of citation recommendation. Also, we conduct experiments with state-of-the-art models and compare their performance on this dataset. The study suggests that, while BM25 is a strong benchmark for the legal citation recommendation task, the most effective method involves implementing a two-step process that entails pre-fetching with BM25+, followed by re-ranking with SciNCL, which enhances the performance of the baseline from 0.26 to 0.30 MAP@10. Moreover, fine-tuning leads to considerable performance increases in pre-trained models, which shows the importance of including legal articles in the training data of these models.

{{</citation>}}


### (81/92) Hiformer: Heterogeneous Feature Interactions Learning with Transformers for Recommender Systems (Huan Gui et al., 2023)

{{<citation>}}

Huan Gui, Ruoxi Wang, Ke Yin, Long Jin, Maciej Kula, Taibai Xu, Lichan Hong, Ed H. Chi. (2023)  
**Hiformer: Heterogeneous Feature Interactions Learning with Transformers for Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Google, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.05884v1)  

---


**ABSTRACT**  
Learning feature interaction is the critical backbone to building recommender systems. In web-scale applications, learning feature interaction is extremely challenging due to the sparse and large input feature space; meanwhile, manually crafting effective feature interactions is infeasible because of the exponential solution space. We propose to leverage a Transformer-based architecture with attention layers to automatically capture feature interactions. Transformer architectures have witnessed great success in many domains, such as natural language processing and computer vision. However, there has not been much adoption of Transformer architecture for feature interaction modeling in industry. We aim at closing the gap. We identify two key challenges for applying the vanilla Transformer architecture to web-scale recommender systems: (1) Transformer architecture fails to capture the heterogeneous feature interactions in the self-attention layer; (2) The serving latency of Transformer architecture might be too high to be deployed in web-scale recommender systems. We first propose a heterogeneous self-attention layer, which is a simple yet effective modification to the self-attention layer in Transformer, to take into account the heterogeneity of feature interactions. We then introduce \textsc{Hiformer} (\textbf{H}eterogeneous \textbf{I}nteraction Trans\textbf{former}) to further improve the model expressiveness. With low-rank approximation and model pruning, \hiformer enjoys fast inference for online deployment. Extensive offline experiment results corroborates the effectiveness and efficiency of the \textsc{Hiformer} model. We have successfully deployed the \textsc{Hiformer} model to a real world large scale App ranking model at Google Play, with significant improvement in key engagement metrics (up to +2.66\%).

{{</citation>}}


### (82/92) DPR: An Algorithm Mitigate Bias Accumulation in Recommendation feedback loops (Hangtong Xu et al., 2023)

{{<citation>}}

Hangtong Xu, Yuanbo Xu, Yongjian Yang, Fuzhen Zhuang, Hui Xiong. (2023)  
**DPR: An Algorithm Mitigate Bias Accumulation in Recommendation feedback loops**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.05864v1)  

---


**ABSTRACT**  
Recommendation models trained on the user feedback collected from deployed recommendation systems are commonly biased. User feedback is considerably affected by the exposure mechanism, as users only provide feedback on the items exposed to them and passively ignore the unexposed items, thus producing numerous false negative samples. Inevitably, biases caused by such user feedback are inherited by new models and amplified via feedback loops. Moreover, the presence of false negative samples makes negative sampling difficult and introduces spurious information in the user preference modeling process of the model. Recent work has investigated the negative impact of feedback loops and unknown exposure mechanisms on recommendation quality and user experience, essentially treating them as independent factors and ignoring their cross-effects. To address these issues, we deeply analyze the data exposure mechanism from the perspective of data iteration and feedback loops with the Missing Not At Random (\textbf{MNAR}) assumption, theoretically demonstrating the existence of an available stabilization factor in the transformation of the exposure mechanism under the feedback loops. We further propose Dynamic Personalized Ranking (\textbf{DPR}), an unbiased algorithm that uses dynamic re-weighting to mitigate the cross-effects of exposure mechanisms and feedback loops without additional information. Furthermore, we design a plugin named Universal Anti-False Negative (\textbf{UFN}) to mitigate the negative impact of the false negative problem. We demonstrate theoretically that our approach mitigates the negative effects of feedback loops and unknown exposure mechanisms. Experimental results on real-world datasets demonstrate that models using DPR can better handle bias accumulation and the universality of UFN in mainstream loss methods.

{{</citation>}}


### (83/92) Exploring Fine-tuning ChatGPT for News Recommendation (Xinyi Li et al., 2023)

{{<citation>}}

Xinyi Li, Yongfeng Zhang, Edward C Malthouse. (2023)  
**Exploring Fine-tuning ChatGPT for News Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: ChatGPT, GPT, Microsoft, NLP, T5  
[Paper Link](http://arxiv.org/abs/2311.05850v1)  

---


**ABSTRACT**  
News recommendation systems (RS) play a pivotal role in the current digital age, shaping how individuals access and engage with information. The fusion of natural language processing (NLP) and RS, spurred by the rise of large language models such as the GPT and T5 series, blurs the boundaries between these domains, making a tendency to treat RS as a language task. ChatGPT, renowned for its user-friendly interface and increasing popularity, has become a prominent choice for a wide range of NLP tasks. While previous studies have explored ChatGPT on recommendation tasks, this study breaks new ground by investigating its fine-tuning capability, particularly within the news domain. In this study, we design two distinct prompts: one designed to treat news RS as the ranking task and another tailored for the rating task. We evaluate ChatGPT's performance in news recommendation by eliciting direct responses through the formulation of these two tasks. More importantly, we unravel the pivotal role of fine-tuning data quality in enhancing ChatGPT's personalized recommendation capabilities, and illustrates its potential in addressing the longstanding challenge of the "cold item" problem in RS. Our experiments, conducted using the Microsoft News dataset (MIND), reveal significant improvements achieved by ChatGPT after fine-tuning, especially in scenarios where a user's topic interests remain consistent, treating news RS as a ranking task. This study illuminates the transformative potential of fine-tuning ChatGPT as a means to advance news RS, offering more effective news consumption experiences.

{{</citation>}}


### (84/92) Knowledge-Augmented Large Language Models for Personalized Contextual Query Suggestion (Jinheon Baek et al., 2023)

{{<citation>}}

Jinheon Baek, Nirupama Chandrasekaran, Silviu Cucerzan, Allen herring, Sujay Kumar Jauhar. (2023)  
**Knowledge-Augmented Large Language Models for Personalized Contextual Query Suggestion**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.06318v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) excel at tackling various natural language tasks. However, due to the significant costs involved in re-training or fine-tuning them, they remain largely static and difficult to personalize. Nevertheless, a variety of applications could benefit from generations that are tailored to users' preferences, goals, and knowledge. Among them is web search, where knowing what a user is trying to accomplish, what they care about, and what they know can lead to improved search experiences. In this work, we propose a novel and general approach that augments an LLM with relevant context from users' interaction histories with a search engine in order to personalize its outputs. Specifically, we construct an entity-centric knowledge store for each user based on their search and browsing activities on the web, which is then leveraged to provide contextually relevant LLM prompt augmentations. This knowledge store is light-weight, since it only produces user-specific aggregate projections of interests and knowledge onto public knowledge graphs, and leverages existing search log infrastructure, thereby mitigating the privacy, compliance, and scalability concerns associated with building deep user profiles for personalization. We then validate our approach on the task of contextual query suggestion, which requires understanding not only the user's current search context but also what they historically know and care about. Through a number of experiments based on human evaluation, we show that our approach is significantly better than several other LLM-powered baselines, generating query suggestions that are contextually more relevant, personalized, and useful.

{{</citation>}}


### (85/92) Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval (Nandan Thakur et al., 2023)

{{<citation>}}

Nandan Thakur, Jianmo Ni, Gustavo Hernández Ábrego, John Wieting, Jimmy Lin, Daniel Cer. (2023)  
**Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs.IR  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2311.05800v1)  

---


**ABSTRACT**  
Dense retrieval models have predominantly been studied for English, where models have shown great success, due to the availability of human-labeled training pairs. However, there has been limited success for multilingual retrieval so far, as training data is uneven or scarcely available across multiple languages. Synthetic training data generation is promising (e.g., InPars or Promptagator), but has been investigated only for English. Therefore, to study model capabilities across both cross-lingual and monolingual retrieval tasks, we develop SWIM-IR, a synthetic retrieval training dataset containing 33 (high to very-low resource) languages for training multilingual dense retrieval models without requiring any human supervision. To construct SWIM-IR, we propose SAP (summarize-then-ask prompting), where the large language model (LLM) generates a textual summary prior to the query generation step. SAP assists the LLM in generating informative queries in the target language. Using SWIM-IR, we explore synthetic fine-tuning of multilingual dense retrieval models and evaluate them robustly on three retrieval benchmarks: XOR-Retrieve (cross-lingual), XTREME-UP (cross-lingual) and MIRACL (monolingual). Our models, called SWIM-X, are competitive with human-supervised dense retrieval models, e.g., mContriever, finding that SWIM-IR can cheaply substitute for expensive human-labeled retrieval training data.

{{</citation>}}


## cs.SI (2)



### (86/92) Privacy-Preserving Individual-Level COVID-19 Infection Prediction via Federated Graph Learning (Wenjie Fu et al., 2023)

{{<citation>}}

Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang. (2023)  
**Privacy-Preserving Individual-Level COVID-19 Infection Prediction via Federated Graph Learning**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Falcon, GNN  
[Paper Link](http://arxiv.org/abs/2311.06049v1)  

---


**ABSTRACT**  
Accurately predicting individual-level infection state is of great value since its essential role in reducing the damage of the epidemic. However, there exists an inescapable risk of privacy leakage in the fine-grained user mobility trajectories required by individual-level infection prediction. In this paper, we focus on developing a framework of privacy-preserving individual-level infection prediction based on federated learning (FL) and graph neural networks (GNN). We propose Falcon, a Federated grAph Learning method for privacy-preserving individual-level infeCtion predictiON. It utilizes a novel hypergraph structure with spatio-temporal hyperedges to describe the complex interactions between individuals and locations in the contagion process. By organically combining the FL framework with hypergraph neural networks, the information propagation process of the graph machine learning is able to be divided into two stages distributed on the server and the clients, respectively, so as to effectively protect user privacy while transmitting high-level information. Furthermore, it elaborately designs a differential privacy perturbation mechanism as well as a plausible pseudo location generation approach to preserve user privacy in the graph structure. Besides, it introduces a cooperative coupling mechanism between the individual-level prediction model and an additional region-level model to mitigate the detrimental impacts caused by the injected obfuscation mechanisms. Extensive experimental results show that our methodology outperforms state-of-the-art algorithms and is able to protect user privacy against actual privacy attacks. Our code and datasets are available at the link: https://github.com/wjfu99/FL-epidemic.

{{</citation>}}


### (87/92) Signature-Based Community Detection for Time Series (Marco Gregnanin et al., 2023)

{{<citation>}}

Marco Gregnanin, Johannes De Smedt, Giorgio Gnecco, Maurizio Parton. (2023)  
**Signature-Based Community Detection for Time Series**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-data-an  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.05986v1)  

---


**ABSTRACT**  
Community detection for time series without prior knowledge poses an open challenge within complex networks theory. Traditional approaches begin by assessing time series correlations and maximizing modularity under diverse null models. These methods suffer from assuming temporal stationarity and are influenced by the granularity of observation intervals. In this study, we propose an approach based on the signature matrix, a concept from path theory for studying stochastic processes. By employing a signature-derived similarity measure, our method overcomes drawbacks of traditional correlation-based techniques. Through a series of numerical experiments, we demonstrate that our method consistently yields higher modularity compared to baseline models, when tested on the Standard and Poor's 500 dataset. Moreover, our approach showcases enhanced stability in modularity when the length of the underlying time series is manipulated. This research contributes to the field of community detection by introducing a signature-based similarity measure, offering an alternative to conventional correlation matrices.

{{</citation>}}


## q-fin.ST (1)



### (88/92) Multi-Label Topic Model for Financial Textual Data (Moritz Scherrmann, 2023)

{{<citation>}}

Moritz Scherrmann. (2023)  
**Multi-Label Topic Model for Financial Textual Data**  

---
Primary Category: q-fin.ST  
Categories: cs-CL, cs-LG, q-fin-ST, q-fin.ST  
Keywords: Financial, Topic Model  
[Paper Link](http://arxiv.org/abs/2311.07598v1)  

---


**ABSTRACT**  
This paper presents a multi-label topic model for financial texts like ad-hoc announcements, 8-K filings, finance related news or annual reports. I train the model on a new financial multi-label database consisting of 3,044 German ad-hoc announcements that are labeled manually using 20 predefined, economically motivated topics. The best model achieves a macro F1 score of more than 85%. Translating the data results in an English version of the model with similar performance. As application of the model, I investigate differences in stock market reactions across topics. I find evidence for strong positive or negative market reactions for some topics, like announcements of new Large Scale Projects or Bankruptcy Filings, while I do not observe significant price effects for some other topics. Furthermore, in contrast to previous studies, the multi-label structure of the model allows to analyze the effects of co-occurring topics on stock market reactions. For many cases, the reaction to a specific topic depends heavily on the co-occurrence with other topics. For example, if allocated capital from a Seasoned Equity Offering (SEO) is used for restructuring a company in the course of a Bankruptcy Proceeding, the market reacts positively on average. However, if that capital is used for covering unexpected, additional costs from the development of new drugs, the SEO implies negative reactions on average.

{{</citation>}}


## cs.HC (2)



### (89/92) Prompt Problems: A New Programming Exercise for the Generative AI Era (Paul Denny et al., 2023)

{{<citation>}}

Paul Denny, Juho Leinonen, James Prather, Andrew Luxton-Reilly, Thezyrie Amarouche, Brett A. Becker, Brent N. Reeves. (2023)  
**Prompt Problems: A New Programming Exercise for the Generative AI Era**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05943v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are revolutionizing the field of computing education with their powerful code-generating capabilities. Traditional pedagogical practices have focused on code writing tasks, but there is now a shift in importance towards code reading, comprehension and evaluation of LLM-generated code. Alongside this shift, an important new skill is emerging -- the ability to solve programming tasks by constructing good prompts for code-generating models. In this work we introduce a new type of programming exercise to hone this nascent skill: 'Prompt Problems'. Prompt Problems are designed to help students learn how to write effective prompts for AI code generators. A student solves a Prompt Problem by crafting a natural language prompt which, when provided as input to an LLM, outputs code that successfully solves a specified programming task. We also present a new web-based tool called Promptly which hosts a repository of Prompt Problems and supports the automated evaluation of prompt-generated code. We deploy Promptly for the first time in one CS1 and one CS2 course and describe our experiences, which include student perceptions of this new type of activity and their interactions with the tool. We find that students are enthusiastic about Prompt Problems, and appreciate how the problems engage their computational thinking skills and expose them to new programming constructs. We discuss ideas for the future development of new variations of Prompt Problems, and the need to carefully study their integration into classroom practice.

{{</citation>}}


### (90/92) PodReels: Human-AI Co-Creation of Video Podcast Teasers (Sitong Wang et al., 2023)

{{<citation>}}

Sitong Wang, Zheng Ning, Anh Truong, Mira Dontcheva, Dingzeyu Li, Lydia B. Chilton. (2023)  
**PodReels: Human-AI Co-Creation of Video Podcast Teasers**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05867v1)  

---


**ABSTRACT**  
Video podcast teasers are short videos that can be shared on social media platforms to capture interest in the full episodes of a video podcast. These teasers enable long-form podcasters to reach new audiences and gain new followers. However, creating a compelling teaser from an hour-long episode is challenging. Selecting interesting clips requires significant mental effort; editing the chosen clips into a cohesive, well-produced teaser is time-consuming. To support the creation of video podcast teasers, we first investigate what makes a good teaser. We combine insights from both audience comments and creator interviews to determine a set of essential ingredients. We also identify a common workflow shared by creators during the process. Based on these findings, we introduce a human-AI co-creative tool called PodReels to assist video podcasters in creating teasers. Our user study shows that PodReels significantly reduces creators' mental demand and improves their efficiency in producing video podcast teasers.

{{</citation>}}


## cs.NE (1)



### (91/92) Genetic Algorithm enhanced by Deep Reinforcement Learning in parent selection mechanism and mutation : Minimizing makespan in permutation flow shop scheduling problems (Maissa Irmouli et al., 2023)

{{<citation>}}

Maissa Irmouli, Nourelhouda Benazzoug, Alaa Dania Adimi, Fatma Zohra Rezkellah, Imane Hamzaoui, Thanina Hamitouche. (2023)  
**Genetic Algorithm enhanced by Deep Reinforcement Learning in parent selection mechanism and mutation : Minimizing makespan in permutation flow shop scheduling problems**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-NE, cs.NE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.05937v1)  

---


**ABSTRACT**  
This paper introduces a reinforcement learning (RL) approach to address the challenges associated with configuring and optimizing genetic algorithms (GAs) for solving difficult combinatorial or non-linear problems. The proposed RL+GA method was specifically tested on the flow shop scheduling problem (FSP). The hybrid algorithm incorporates neural networks (NN) and uses the off-policy method Q-learning or the on-policy method Sarsa(0) to control two key genetic algorithm (GA) operators: parent selection mechanism and mutation. At each generation, the RL agent's action is determining the selection method, the probability of the parent selection and the probability of the offspring mutation. This allows the RL agent to dynamically adjust the selection and mutation based on its learned policy. The results of the study highlight the effectiveness of the RL+GA approach in improving the performance of the primitive GA. They also demonstrate its ability to learn and adapt from population diversity and solution improvements over time. This adaptability leads to improved scheduling solutions compared to static parameter configurations while maintaining population diversity throughout the evolutionary process.

{{</citation>}}


## cs.NI (1)



### (92/92) AI-native Interconnect Framework for Integration of Large Language Model Technologies in 6G Systems (Sasu Tarkoma et al., 2023)

{{<citation>}}

Sasu Tarkoma, Roberto Morabito, Jaakko Sauvola. (2023)  
**AI-native Interconnect Framework for Integration of Large Language Model Technologies in 6G Systems**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI  
Keywords: AI, GPT, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.05842v1)  

---


**ABSTRACT**  
The evolution towards 6G architecture promises a transformative shift in communication networks, with artificial intelligence (AI) playing a pivotal role. This paper delves deep into the seamless integration of Large Language Models (LLMs) and Generalized Pretrained Transformers (GPT) within 6G systems. Their ability to grasp intent, strategize, and execute intricate commands will be pivotal in redefining network functionalities and interactions. Central to this is the AI Interconnect framework, intricately woven to facilitate AI-centric operations within the network. Building on the continuously evolving current state-of-the-art, we present a new architectural perspective for the upcoming generation of mobile networks. Here, LLMs and GPTs will collaboratively take center stage alongside traditional pre-generative AI and machine learning (ML) algorithms. This union promises a novel confluence of the old and new, melding tried-and-tested methods with transformative AI technologies. Along with providing a conceptual overview of this evolution, we delve into the nuances of practical applications arising from such an integration. Through this paper, we envisage a symbiotic integration where AI becomes the cornerstone of the next-generation communication paradigm, offering insights into the structural and functional facets of an AI-native 6G network.

{{</citation>}}
