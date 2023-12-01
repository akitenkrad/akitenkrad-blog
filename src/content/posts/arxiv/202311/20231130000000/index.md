---
draft: false
title: "arXiv @ 2023.11.30"
date: 2023-11-30
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.30"
    identifier: arxiv_20231130
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (17)](#cscl-17)
- [eess.SY (3)](#eesssy-3)
- [cs.CV (56)](#cscv-56)
- [cs.LG (27)](#cslg-27)
- [eess.IV (2)](#eessiv-2)
- [cs.CY (6)](#cscy-6)
- [cs.AI (4)](#csai-4)
- [math.NT (1)](#mathnt-1)
- [cs.GT (2)](#csgt-2)
- [hep-ex (1)](#hep-ex-1)
- [astro-ph.IM (1)](#astro-phim-1)
- [cs.RO (4)](#csro-4)
- [cs.DS (1)](#csds-1)
- [stat.ME (1)](#statme-1)
- [physics.optics (1)](#physicsoptics-1)
- [cs.NI (2)](#csni-2)
- [cs.HC (3)](#cshc-3)
- [cs.IR (4)](#csir-4)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [cs.CR (1)](#cscr-1)
- [eess.AS (1)](#eessas-1)
- [cs.SD (1)](#cssd-1)
- [cs.DC (1)](#csdc-1)
- [cs.AR (2)](#csar-2)
- [cs.SE (2)](#csse-2)

## cs.CL (17)



### (1/145) Does VLN Pretraining Work with Nonsensical or Irrelevant Instructions? (Wang Zhu et al., 2023)

{{<citation>}}

Wang Zhu, Ishika Singh, Yuan Huang, Robin Jia, Jesse Thomason. (2023)  
**Does VLN Pretraining Work with Nonsensical or Irrelevant Instructions?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2311.17280v1)  

---


**ABSTRACT**  
Data augmentation via back-translation is common when pretraining Vision-and-Language Navigation (VLN) models, even though the generated instructions are noisy. But: does that noise matter? We find that nonsensical or irrelevant language instructions during pretraining can have little effect on downstream performance for both HAMT and VLN-BERT on R2R, and is still better than only using clean, human data. To underscore these results, we concoct an efficient augmentation method, Unigram + Object, which generates nonsensical instructions that nonetheless improve downstream performance. Our findings suggest that what matters for VLN R2R pretraining is the quantity of visual trajectories, not the quality of instructions.

{{</citation>}}


### (2/145) General-Purpose vs. Domain-Adapted Large Language Models for Extraction of Data from Thoracic Radiology Reports (Ali H. Dhanaliwala et al., 2023)

{{<citation>}}

Ali H. Dhanaliwala, Rikhiya Ghosh, Sanjeev Kumar Karn, Poikavila Ullaskrishnan, Oladimeji Farri, Dorin Comaniciu, Charles E. Kahn. (2023)  
**General-Purpose vs. Domain-Adapted Large Language Models for Extraction of Data from Thoracic Radiology Reports**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-IV  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17213v1)  

---


**ABSTRACT**  
Radiologists produce unstructured data that could be valuable for clinical care when consumed by information systems. However, variability in style limits usage. Study compares performance of system using domain-adapted language model (RadLing) and general-purpose large language model (GPT-4) in extracting common data elements (CDE) from thoracic radiology reports. Three radiologists annotated a retrospective dataset of 1300 thoracic reports (900 training, 400 test) and mapped to 21 pre-selected relevant CDEs. RadLing was used to generate embeddings for sentences and identify CDEs using cosine-similarity, which were mapped to values using light-weight mapper. GPT-4 system used OpenAI's general-purpose embeddings to identify relevant CDEs and used GPT-4 to map to values. The output CDE:value pairs were compared to the reference standard; an identical match was considered true positive. Precision (positive predictive value) was 96% (2700/2824) for RadLing and 99% (2034/2047) for GPT-4. Recall (sensitivity) was 94% (2700/2876) for RadLing and 70% (2034/2887) for GPT-4; the difference was statistically significant (P<.001). RadLing's domain-adapted embeddings were more sensitive in CDE identification (95% vs 71%) and its light-weight mapper had comparable precision in value assignment (95.4% vs 95.0%). RadLing system exhibited higher performance than GPT-4 system in extracting CDEs from radiology reports. RadLing system's domain-adapted embeddings outperform general-purpose embeddings from OpenAI in CDE identification and its light-weight value mapper achieves comparable precision to large GPT-4. RadLing system offers operational advantages including local deployment and reduced runtime costs. Domain-adapted RadLing system surpasses GPT-4 system in extracting common data elements from radiology reports, while providing benefits of local deployment and lower costs.

{{</citation>}}


### (3/145) Pragmatic Radiology Report Generation (Dang Nguyen et al., 2023)

{{<citation>}}

Dang Nguyen, Chacha Chen, He He, Chenhao Tan. (2023)  
**Pragmatic Radiology Report Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2311.17154v1)  

---


**ABSTRACT**  
When pneumonia is not found on a chest X-ray, should the report describe this negative observation or omit it? We argue that this question cannot be answered from the X-ray alone and requires a pragmatic perspective, which captures the communicative goal that radiology reports serve between radiologists and patients. However, the standard image-to-text formulation for radiology report generation fails to incorporate such pragmatic intents. Following this pragmatic perspective, we demonstrate that the indication, which describes why a patient comes for an X-ray, drives the mentions of negative observations and introduce indications as additional input to report generation. With respect to the output, we develop a framework to identify uninferable information from the image as a source of model hallucinations, and limit them by cleaning groundtruth reports. Finally, we use indications and cleaned groundtruth reports to develop pragmatic models, and show that they outperform existing methods not only in new pragmatics-inspired metrics (+4.3 Negative F1) but also in standard metrics (+6.3 Positive F1 and +11.0 BLEU-2).

{{</citation>}}


### (4/145) ChatGPT's One-year Anniversary: Are Open-Source Large Language Models Catching up? (Hailin Chen et al., 2023)

{{<citation>}}

Hailin Chen, Fangkai Jiao, Xingxuan Li, Chengwei Qin, Mathieu Ravaut, Ruochen Zhao, Caiming Xiong, Shafiq Joty. (2023)  
**ChatGPT's One-year Anniversary: Are Open-Source Large Language Models Catching up?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16989v2)  

---


**ABSTRACT**  
Upon its release in late 2022, ChatGPT has brought a seismic shift in the entire landscape of AI, both in research and commerce. Through instruction-tuning a large language model (LLM) with supervised fine-tuning and reinforcement learning from human feedback, it showed that a model could answer human questions and follow instructions on a broad panel of tasks. Following this success, interests in LLMs have intensified, with new LLMs flourishing at frequent interval across academia and industry, including many start-ups focused on LLMs. While closed-source LLMs (e.g., OpenAI's GPT, Anthropic's Claude) generally outperform their open-source counterparts, the progress on the latter has been rapid with claims of achieving parity or even better on certain tasks. This has crucial implications not only on research but also on business. In this work, on the first anniversary of ChatGPT, we provide an exhaustive overview of this success, surveying all tasks where an open-source LLM has claimed to be on par or better than ChatGPT.

{{</citation>}}


### (5/145) Natural Language Processing Through Transfer Learning: A Case Study on Sentiment Analysis (Aman Yadav et al., 2023)

{{<citation>}}

Aman Yadav, Abhishek Vichare. (2023)  
**Natural Language Processing Through Transfer Learning: A Case Study on Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP, Natural Language Processing, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.16965v1)  

---


**ABSTRACT**  
Artificial intelligence and machine learning have significantly bolstered the technological world. This paper explores the potential of transfer learning in natural language processing focusing mainly on sentiment analysis. The models trained on the big data can also be used where data are scarce. The claim is that, compared to training models from scratch, transfer learning, using pre-trained BERT models, can increase sentiment classification accuracy. The study adopts a sophisticated experimental design that uses the IMDb dataset of sentimentally labelled movie reviews. Pre-processing includes tokenization and encoding of text data, making it suitable for NLP models. The dataset is used on a BERT based model, measuring its performance using accuracy. The result comes out to be 100 per cent accurate. Although the complete accuracy could appear impressive, it might be the result of overfitting or a lack of generalization. Further analysis is required to ensure the model's ability to handle diverse and unseen data. The findings underscore the effectiveness of transfer learning in NLP, showcasing its potential to excel in sentiment analysis tasks. However, the research calls for a cautious interpretation of perfect accuracy and emphasizes the need for additional measures to validate the model's generalization.

{{</citation>}}


### (6/145) The Falcon Series of Open Language Models (Ebtesam Almazrouei et al., 2023)

{{<citation>}}

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, Daniele Mazzotta, Badreddine Noune, Baptiste Pannier, Guilherme Penedo. (2023)  
**The Falcon Series of Open Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AWS, Falcon, GPT, GPT-4, LLaMA, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2311.16867v2)  

---


**ABSTRACT**  
We introduce the Falcon series: 7B, 40B, and 180B parameters causal decoder-only models trained on a diverse high-quality corpora predominantly assembled from web data. The largest model, Falcon-180B, has been trained on over 3.5 trillion tokens of text--the largest openly documented pretraining run. Falcon-180B significantly outperforms models such as PaLM or Chinchilla, and improves upon concurrently developed models such as LLaMA 2 or Inflection-1. It nears the performance of PaLM-2-Large at a reduced pretraining and inference cost, making it, to our knowledge, one of the three best language models in the world along with GPT-4 and PaLM-2-Large. We report detailed evaluations, as well as a deep dive into the methods and custom tooling employed to pretrain Falcon. Notably, we report on our custom distributed training codebase, allowing us to efficiently pretrain these models on up to 4,096 A100s on cloud AWS infrastructure with limited interconnect. We release a 600B tokens extract of our web dataset, as well as the Falcon-7/40/180B models under a permissive license to foster open-science and accelerate the development of an open ecosystem of large language models.

{{</citation>}}


### (7/145) A Benchmark for Evaluating Machine Translation Metrics on Dialects Without Standard Orthography (Noëmi Aepli et al., 2023)

{{<citation>}}

Noëmi Aepli, Chantal Amrhein, Florian Schottmann, Rico Sennrich. (2023)  
**A Benchmark for Evaluating Machine Translation Metrics on Dialects Without Standard Orthography**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.16865v1)  

---


**ABSTRACT**  
For sensible progress in natural language processing, it is important that we are aware of the limitations of the evaluation metrics we use. In this work, we evaluate how robust metrics are to non-standardized dialects, i.e. spelling differences in language varieties that do not have a standard orthography. To investigate this, we collect a dataset of human translations and human judgments for automatic machine translations from English to two Swiss German dialects. We further create a challenge set for dialect variation and benchmark existing metrics' performances. Our results show that existing metrics cannot reliably evaluate Swiss German text generation outputs, especially on segment level. We propose initial design adaptations that increase robustness in the face of non-standardized dialects, although there remains much room for further improvement. The dataset, code, and models are available here: https://github.com/textshuttle/dialect_eval

{{</citation>}}


### (8/145) The Claire French Dialogue Dataset (Julie Hunter et al., 2023)

{{<citation>}}

Julie Hunter, Jérôme Louradour, Virgile Rennard, Ismaïl Harrando, Guokan Shang, Jean-Pierre Lorré. (2023)  
**The Claire French Dialogue Dataset**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2311.16840v1)  

---


**ABSTRACT**  
We present the Claire French Dialogue Dataset (CFDD), a resource created by members of LINAGORA Labs in the context of the OpenLLM France initiative. CFDD is a corpus containing roughly 160 million words from transcripts and stage plays in French that we have assembled and publicly released in an effort to further the development of multilingual, open source language models. This paper describes the 24 individual corpora of which CFDD is composed and provides links and citations to their original sources. It also provides our proposed breakdown of the full CFDD dataset into eight categories of subcorpora and describes the process we followed to standardize the format of the final dataset. We conclude with a discussion of similar work and future directions.

{{</citation>}}


### (9/145) CharacterGLM: Customizing Chinese Conversational AI Characters with Large Language Models (Jinfeng Zhou et al., 2023)

{{<citation>}}

Jinfeng Zhou, Zhuang Chen, Dazhen Wan, Bosi Wen, Yi Song, Jifan Yu, Yongkang Huang, Libiao Peng, Jiaming Yang, Xiyao Xiao, Sahand Sabour, Xiaohan Zhang, Wenjing Hou, Yijia Zhang, Yuxiao Dong, Jie Tang, Minlie Huang. (2023)  
**CharacterGLM: Customizing Chinese Conversational AI Characters with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Dialog, Dialogue, GLM, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16832v1)  

---


**ABSTRACT**  
In this paper, we present CharacterGLM, a series of models built upon ChatGLM, with model sizes ranging from 6B to 66B parameters. Our CharacterGLM is designed for generating Character-based Dialogues (CharacterDial), which aims to equip a conversational AI system with character customization for satisfying people's inherent social desires and emotional needs. On top of CharacterGLM, we can customize various AI characters or social agents by configuring their attributes (identities, interests, viewpoints, experiences, achievements, social relationships, etc.) and behaviors (linguistic features, emotional expressions, interaction patterns, etc.). Our model outperforms most mainstream close-source large langauge models, including the GPT series, especially in terms of consistency, human-likeness, and engagement according to manual evaluations. We will release our 6B version of CharacterGLM and a subset of training data to facilitate further research development in the direction of character-based dialogue generation.

{{</citation>}}


### (10/145) A Survey of the Evolution of Language Model-Based Dialogue Systems (Hongru Wang et al., 2023)

{{<citation>}}

Hongru Wang, Lingzhi Wang, Yiming Du, Liang Chen, Jingyan Zhou, Yufei Wang, Kam-Fai Wong. (2023)  
**A Survey of the Evolution of Language Model-Based Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, LSTM, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16789v1)  

---


**ABSTRACT**  
Dialogue systems, including task-oriented_dialogue_system (TOD) and open-domain_dialogue_system (ODD), have undergone significant transformations, with language_models (LM) playing a central role. This survey delves into the historical trajectory of dialogue systems, elucidating their intricate relationship with advancements in language models by categorizing this evolution into four distinct stages, each marked by pivotal LM breakthroughs: 1) Early_Stage: characterized by statistical LMs, resulting in rule-based or machine-learning-driven dialogue_systems; 2) Independent development of TOD and ODD based on neural_language_models (NLM; e.g., LSTM and GRU), since NLMs lack intrinsic knowledge in their parameters; 3) fusion between different types of dialogue systems with the advert of pre-trained_language_models (PLMs), starting from the fusion between four_sub-tasks_within_TOD, and then TOD_with_ODD; and 4) current LLM-based_dialogue_system, wherein LLMs can be used to conduct TOD and ODD seamlessly. Thus, our survey provides a chronological perspective aligned with LM breakthroughs, offering a comprehensive review of state-of-the-art research outcomes. What's more, we focus on emerging topics and discuss open challenges, providing valuable insights into future directions for LLM-based_dialogue_systems. Through this exploration, we pave the way for a deeper_comprehension of the evolution, guiding future developments in LM-based dialogue_systems.

{{</citation>}}


### (11/145) Radiology-Aware Model-Based Evaluation Metric for Report Generation (Amos Calamida et al., 2023)

{{<citation>}}

Amos Calamida, Farhad Nooralahzadeh, Morteza Rohanian, Koji Fujimoto, Mizuho Nishio, Michael Krauthammer. (2023)  
**Radiology-Aware Model-Based Evaluation Metric for Report Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, BLEU  
[Paper Link](http://arxiv.org/abs/2311.16764v1)  

---


**ABSTRACT**  
We propose a new automated evaluation metric for machine-generated radiology reports using the successful COMET architecture adapted for the radiology domain. We train and publish four medically-oriented model checkpoints, including one trained on RadGraph, a radiology knowledge graph. Our results show that our metric correlates moderately to high with established metrics such as BERTscore, BLEU, and CheXbert scores. Furthermore, we demonstrate that one of our checkpoints exhibits a high correlation with human judgment, as assessed using the publicly available annotations of six board-certified radiologists, using a set of 200 reports. We also performed our own analysis gathering annotations with two radiologists on a collection of 100 reports. The results indicate the potential effectiveness of our method as a radiology-specific evaluation metric. The code, data, and model checkpoints to reproduce our findings will be publicly available.

{{</citation>}}


### (12/145) Entity-Aspect-Opinion-Sentiment Quadruple Extraction for Fine-grained Sentiment Analysis (Dan Ma et al., 2023)

{{<citation>}}

Dan Ma, Jun Xu, Zongyu Wang, Xuezhi Cao, Yunsen Xian. (2023)  
**Entity-Aspect-Opinion-Sentiment Quadruple Extraction for Fine-grained Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.16678v1)  

---


**ABSTRACT**  
Product reviews often contain a large number of implicit aspects and object-attribute co-existence cases. Unfortunately, many existing studies in Aspect-Based Sentiment Analysis (ABSA) have overlooked this issue, which can make it difficult to extract opinions comprehensively and fairly. In this paper, we propose a new task called Entity-Aspect-Opinion-Sentiment Quadruple Extraction (EASQE), which aims to hierarchically decompose aspect terms into entities and aspects to avoid information loss, non-exclusive annotations, and opinion misunderstandings in ABSA tasks. To facilitate research in this new task, we have constructed four datasets (Res14-EASQE, Res15-EASQE, Res16-EASQE, and Lap14-EASQE) based on the SemEval Restaurant and Laptop datasets. We have also proposed a novel two-stage sequence-tagging based Trigger-Opinion framework as the baseline for the EASQE task. Empirical evaluations show that our Trigger-Opinion framework can generate satisfactory EASQE results and can also be applied to other ABSA tasks, significantly outperforming state-of-the-art methods. We have made the four datasets and source code of Trigger-Opinion publicly available to facilitate further research in this area.

{{</citation>}}


### (13/145) A Distribution-Based Threshold for Determining Sentence Similarity (Gioele Cadamuro et al., 2023)

{{<citation>}}

Gioele Cadamuro, Marco Gruppo. (2023)  
**A Distribution-Based Threshold for Determining Sentence Similarity**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Sentence Similarity  
[Paper Link](http://arxiv.org/abs/2311.16675v1)  

---


**ABSTRACT**  
We hereby present a solution to a semantic textual similarity (STS) problem in which it is necessary to match two sentences containing, as the only distinguishing factor, highly specific information (such as names, addresses, identification codes), and from which we need to derive a definition for when they are similar and when they are not. The solution revolves around the use of a neural network, based on the siamese architecture, to create the distributions of the distances between similar and dissimilar pairs of sentences. The goal of these distributions is to find a discriminating factor, that we call "threshold", which represents a well-defined quantity that can be used to distinguish vector distances of similar pairs from vector distances of dissimilar pairs in new predictions and later analyses. In addition, we developed a way to score the predictions by combining attributes from both the distributions' features and the way the distance function works. Finally, we generalize the results showing that they can be transferred to a wider range of domains by applying the system discussed to a well-known and widely used benchmark dataset for STS problems.

{{</citation>}}


### (14/145) Scaling Political Texts with ChatGPT (Gaël Le Mens et al., 2023)

{{<citation>}}

Gaël Le Mens, Aina Gallego. (2023)  
**Scaling Political Texts with ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.16639v1)  

---


**ABSTRACT**  
We use GPT-4 to obtain position estimates of political texts in continuous spaces. We develop and validate a new approach by positioning British party manifestos on the economic, social, and immigration policy dimensions and tweets by members of the US Congress on the left-right ideological spectrum. For the party manifestos, the correlation between the positions produced by GPT-4 and experts is 93% or higher, a performance similar to or better than that obtained with crowdsourced position estimates. For individual tweets, the positions obtained with GPT-4 achieve a correlation of 91% with crowdsourced position estimates. For senators of the 117th US Congress, the positions obtained with GPT-4 achieve a correlation of 97% with estimates based on roll call votes and of 96% with those based on campaign funding. Correlations are also substantial within party, indicating that position estimates produced with GPT-4 capture within-party differences between senators. Overall, using GPT-4 for ideological scaling is fast, cost-efficient, and reliable. This approach provides a viable alternative to scaling by both expert raters and crowdsourcing.

{{</citation>}}


### (15/145) MedGen: A Python Natural Language Processing Toolkit for Medical Text Processing (Rui Yang et al., 2023)

{{<citation>}}

Rui Yang, Qingcheng Zeng, Keen You, Yujie Qiao, Lucas Huang, Chia-Chun Hsieh, Benjamin Rosand, Jeremy Goldwasser, Amisha D Dave, Tiarnan D. L. Keenan, Emily Y Chew, Dragomir Radev, Zhiyong Lu, Hua Xu, Qingyu Chen, Irene Li. (2023)  
**MedGen: A Python Natural Language Processing Toolkit for Medical Text Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.16588v1)  

---


**ABSTRACT**  
This study introduces MedGen, a comprehensive natural language processing (NLP) toolkit designed for medical text processing. MedGen is tailored for biomedical researchers and healthcare professionals with an easy-to-use, all-in-one solution that requires minimal programming expertise. It includes (1) Generative Functions: For the first time, MedGen includes four advanced generative functions: question answering, text summarization, text simplification, and machine translation; (2) Basic NLP Functions: MedGen integrates 12 essential NLP functions such as word tokenization and sentence segmentation; and (3) Query and Search Capabilities: MedGen provides user-friendly query and search functions on text corpora. We fine-tuned 32 domain-specific language models, evaluated them thoroughly on 24 established benchmarks and conducted manual reviews with clinicians. Additionally, we expanded our toolkit by introducing query and search functions, while also standardizing and integrating functions from third-party libraries. The toolkit, its models, and associated data are publicly available via https://github.com/Yale-LILY/MedGen.

{{</citation>}}


### (16/145) Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine (Harsha Nori et al., 2023)

{{<citation>}}

Harsha Nori, Yin Tat Lee, Sheng Zhang, Dean Carignan, Richard Edgar, Nicolo Fusi, Nicholas King, Jonathan Larson, Yuanzhi Li, Weishung Liu, Renqian Luo, Scott Mayer McKinney, Robert Osazuwa Ness, Hoifung Poon, Tao Qin, Naoto Usuyama, Chris White, Eric Horvitz. (2023)  
**Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: GPT, GPT-4, PaLM, QA  
[Paper Link](http://arxiv.org/abs/2311.16452v1)  

---


**ABSTRACT**  
Generalist foundation models such as GPT-4 have displayed surprising capabilities in a wide variety of domains and tasks. Yet, there is a prevalent assumption that they cannot match specialist capabilities of fine-tuned models. For example, most explorations to date on medical competency benchmarks have leveraged domain-specific training, as exemplified by efforts on BioGPT and Med-PaLM. We build on a prior study of GPT-4's capabilities on medical challenge benchmarks in the absence of special training. Rather than using simple prompting to highlight the model's out-of-the-box capabilities, we perform a systematic exploration of prompt engineering. We find that prompting innovation can unlock deeper specialist capabilities and show that GPT-4 easily tops prior leading results for medical benchmarks. The prompting methods we explore are general purpose, and make no specific use of domain expertise, removing the need for expert-curated content. Our experimental design carefully controls for overfitting during the prompt engineering process. We introduce Medprompt, based on a composition of several prompting strategies. With Medprompt, GPT-4 achieves state-of-the-art results on all nine of the benchmark datasets in the MultiMedQA suite. The method outperforms leading specialist models such as Med-PaLM 2 by a significant margin with an order of magnitude fewer calls to the model. Steering GPT-4 with Medprompt achieves a 27% reduction in error rate on the MedQA dataset over the best methods to date achieved with specialist models and surpasses a score of 90% for the first time. Beyond medical problems, we show the power of Medprompt to generalize to other domains and provide evidence for the broad applicability of the approach via studies of the strategy on exams in electrical engineering, machine learning, philosophy, accounting, law, nursing, and clinical psychology.

{{</citation>}}


### (17/145) CDEval: A Benchmark for Measuring the Cultural Dimensions of Large Language Models (Yuhang Wang et al., 2023)

{{<citation>}}

Yuhang Wang, Yanxu Zhu, Chao Kong, Shuyu Wei, Xiaoyuan Yi, Xing Xie, Jitao Sang. (2023)  
**CDEval: A Benchmark for Measuring the Cultural Dimensions of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16421v1)  

---


**ABSTRACT**  
As the scaling of Large Language Models (LLMs) has dramatically enhanced their capabilities, there has been a growing focus on the alignment problem to ensure their responsible and ethical use. While existing alignment efforts predominantly concentrate on universal values such as the HHH principle, the aspect of culture, which is inherently pluralistic and diverse, has not received adequate attention. This work introduces a new benchmark, CDEval, aimed at evaluating the cultural dimensions of LLMs. CDEval is constructed by incorporating both GPT-4's automated generation and human verification, covering six cultural dimensions across seven domains. Our comprehensive experiments provide intriguing insights into the culture of mainstream LLMs, highlighting both consistencies and variations across different dimensions and domains. The findings underscore the importance of integrating cultural considerations in LLM development, particularly for applications in diverse cultural settings. Through CDEval, we aim to broaden the horizon of LLM alignment research by including cultural dimensions, thus providing a more holistic framework for the future development and evaluation of LLMs. This benchmark serves as a valuable resource for cultural studies in LLMs, paving the way for more culturally aware and sensitive models.

{{</citation>}}


## eess.SY (3)



### (18/145) Advancing Attack-Resilient Scheduling of Integrated Energy Systems with Demand Response via Deep Reinforcement Learning (Yang Li et al., 2023)

{{<citation>}}

Yang Li, Wenjie Ma, Yuanzheng Li, Sen Li, Zhe Chen. (2023)  
**Advancing Attack-Resilient Scheduling of Integrated Energy Systems with Demand Response via Deep Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.17941v1)  

---


**ABSTRACT**  
Optimally scheduling multi-energy flow is an effective method to utilize renewable energy sources (RES) and improve the stability and economy of integrated energy systems (IES). However, the stable demand-supply of IES faces challenges from uncertainties that arise from RES and loads, as well as the increasing impact of cyber-attacks with advanced information and communication technologies adoption. To address these challenges, this paper proposes an innovative model-free resilience scheduling method based on state-adversarial deep reinforcement learning (DRL) for integrated demand response (IDR)-enabled IES. The proposed method designs an IDR program to explore the interaction ability of electricity-gas-heat flexible loads. Additionally, a state-adversarial Markov decision process (SA-MDP) model characterizes the energy scheduling problem of IES under cyber-attack. The state-adversarial soft actor-critic (SA-SAC) algorithm is proposed to mitigate the impact of cyber-attacks on the scheduling strategy. Simulation results demonstrate that our method is capable of adequately addressing the uncertainties resulting from RES and loads, mitigating the impact of cyber-attacks on the scheduling strategy, and ensuring a stable demand supply for various energy sources. Moreover, the proposed method demonstrates resilience against cyber-attacks. Compared to the original soft actor-critic (SAC) algorithm, it achieves a 10\% improvement in economic performance under cyber-attack scenarios.

{{</citation>}}


### (19/145) Optimization Theory Based Deep Reinforcement Learning for Resource Allocation in Ultra-Reliable Wireless Networked Control Systems (Hamida Qumber Ali et al., 2023)

{{<citation>}}

Hamida Qumber Ali, Amirhassan Babazadeh Darabi, Sinem Coleri. (2023)  
**Optimization Theory Based Deep Reinforcement Learning for Resource Allocation in Ultra-Reliable Wireless Networked Control Systems**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-IT, cs-SY, eess-SY, eess.SY, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.16895v1)  

---


**ABSTRACT**  
The design of Wireless Networked Control System (WNCS) requires addressing critical interactions between control and communication systems with minimal complexity and communication overhead while providing ultra-high reliability. This paper introduces a novel optimization theory based deep reinforcement learning (DRL) framework for the joint design of controller and communication systems. The objective of minimum power consumption is targeted while satisfying the schedulability and rate constraints of the communication system in the finite blocklength regime and stability constraint of the control system. Decision variables include the sampling period in the control system, and blocklength and packet error probability in the communication system. The proposed framework contains two stages: optimization theory and DRL. In the optimization theory stage, following the formulation of the joint optimization problem, optimality conditions are derived to find the mathematical relations between the optimal values of the decision variables. These relations allow the decomposition of the problem into multiple building blocks. In the DRL stage, the blocks that are simplified but not tractable are replaced by DRL. Via extensive simulations, the proposed optimization theory based DRL approach is demonstrated to outperform the optimization theory and pure DRL based approaches, with close to optimal performance and much lower complexity.

{{</citation>}}


### (20/145) Advancements in Arc Fault Detection for Electrical Distribution Systems: A Comprehensive Review from Artificial Intelligence Perspective (Kriti Thakur et al., 2023)

{{<citation>}}

Kriti Thakur, Divyanshi Dwivedi, K. Victor Sam Moses Babu, Alivelu Manga Parimi, Pradeep Kumar Yemula, Pratyush Chakraborty, Mayukha Pal. (2023)  
**Advancements in Arc Fault Detection for Electrical Distribution Systems: A Comprehensive Review from Artificial Intelligence Perspective**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16804v1)  

---


**ABSTRACT**  
This comprehensive review paper provides a thorough examination of current advancements and research in the field of arc fault detection for electrical distribution systems. The increasing demand for electricity, coupled with the increasing utilization of renewable energy sources, has necessitated vigilance in safeguarding electrical distribution systems against arc faults. Such faults could lead to catastrophic accidents, including fires, equipment damage, loss of human life, and other critical issues. To mitigate these risks, this review article focuses on the identification and early detection of arc faults, with a particular emphasis on the vital role of artificial intelligence (AI) in the detection and prediction of arc faults. The paper explores a wide range of methodologies for arc fault detection and highlights the superior performance of AI-based methods in accurately identifying arc faults when compared to other approaches. A thorough evaluation of existing methodologies is conducted by categorizing them into distinct groups, which provides a structured framework for understanding the current state of arc fault detection techniques. This categorization serves as a foundation for identifying the existing constraints and future research avenues in the domain of arc fault detection for electrical distribution systems. This review paper provides the state of the art in arc fault detection, aiming to enhance safety and reliability in electrical distribution systems and guide future research efforts.

{{</citation>}}


## cs.CV (56)



### (21/145) E-ViLM: Efficient Video-Language Model via Masked Video Modeling with Semantic Vector-Quantized Tokenizer (Jacob Zhiyuan Fang et al., 2023)

{{<citation>}}

Jacob Zhiyuan Fang, Skyler Zheng, Vasu Sharma, Robinson Piramuthu. (2023)  
**E-ViLM: Efficient Video-Language Model via Masked Video Modeling with Semantic Vector-Quantized Tokenizer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17267v1)  

---


**ABSTRACT**  
To build scalable models for challenging real-world tasks, it is important to learn from diverse, multi-modal data in various forms (e.g., videos, text, and images). Among the existing works, a plethora of them have focused on leveraging large but cumbersome cross-modal architectures. Regardless of their effectiveness, larger architectures unavoidably prevent the models from being extended to real-world applications, so building a lightweight VL architecture and an efficient learning schema is of great practical value. In this paper, we propose an Efficient Video-Language Model (dubbed as E-ViLM) and a masked video modeling (MVM) schema, assisted with a semantic vector-quantized tokenizer. In particular, our E-ViLM learns to reconstruct the semantic labels of masked video regions, produced by the pre-trained vector-quantized tokenizer, which discretizes the continuous visual signals into labels. We show that with our simple MVM task and regular VL pre-training modelings, our E-ViLM, despite its compactness, is able to learn expressive representations from Video-Language corpus and generalize well to extensive Video-Language tasks including video question answering, text-to-video retrieval, etc. In particular, our E-ViLM obtains obvious efficiency improvements by reaching competing performances with faster inference speed, i.e., our model reaches $39.3$% Top-$1$ accuracy on the MSRVTT benchmark, retaining $91.4$% of the accuracy of state-of-the-art larger VL architecture with only $15%$ parameters and $94.8%$ fewer GFLOPs. We also provide extensive ablative studies that validate the effectiveness of our proposed learning schema for E-ViLM.

{{</citation>}}


### (22/145) Scene Summarization: Clustering Scene Videos into Spatially Diverse Frames (Chao Chen et al., 2023)

{{<citation>}}

Chao Chen, Mingzhi Zhu, Ankush Pratap Singh, Yu Yan, Felix Juefei Xu, Chen Feng. (2023)  
**Scene Summarization: Clustering Scene Videos into Spatially Diverse Frames**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.17940v1)  

---


**ABSTRACT**  
We propose scene summarization as a new video-based scene understanding task. It aims to summarize a long video walkthrough of a scene into a small set of frames that are spatially diverse in the scene, which has many impotant applications, such as in surveillance, real estate, and robotics. It stems from video summarization but focuses on long and continuous videos from moving cameras, instead of user-edited fragmented video clips that are more commonly studied in existing video summarization works. Our solution to this task is a two-stage self-supervised pipeline named SceneSum. Its first stage uses clustering to segment the video sequence. Our key idea is to combine visual place recognition (VPR) into this clustering process to promote spatial diversity. Its second stage needs to select a representative keyframe from each cluster as the summary while respecting resource constraints such as memory and disk space limits. Additionally, if the ground truth image trajectory is available, our method can be easily augmented with a supervised loss to enhance the clustering and keyframe selection. Extensive experiments on both real-world and simulated datasets show our method outperforms common video summarization baselines by 50%

{{</citation>}}


### (23/145) LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS (Zhiwen Fan et al., 2023)

{{<citation>}}

Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang. (2023)  
**LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning, Quantization  
[Paper Link](http://arxiv.org/abs/2311.17245v1)  

---


**ABSTRACT**  
Recent advancements in real-time neural rendering using point-based techniques have paved the way for the widespread adoption of 3D representations. However, foundational approaches like 3D Gaussian Splatting come with a substantial storage overhead caused by growing the SfM points to millions, often demanding gigabyte-level disk space for a single unbounded scene, posing significant scalability challenges and hindering the splatting efficiency.   To address this challenge, we introduce LightGaussian, a novel method designed to transform 3D Gaussians into a more efficient and compact format. Drawing inspiration from the concept of Network Pruning, LightGaussian identifies Gaussians that are insignificant in contributing to the scene reconstruction and adopts a pruning and recovery process, effectively reducing redundancy in Gaussian counts while preserving visual effects. Additionally, LightGaussian employs distillation and pseudo-view augmentation to distill spherical harmonics to a lower degree, allowing knowledge transfer to more compact representations while maintaining reflectance. Furthermore, we propose a hybrid scheme, VecTree Quantization, to quantize all attributes, resulting in lower bitwidth representations with minimal accuracy losses.   In summary, LightGaussian achieves an averaged compression rate over 15x while boosting the FPS from 139 to 215, enabling an efficient representation of complex scenes on Mip-NeRF 360, Tank and Temple datasets.   Project website: https://lightgaussian.github.io/

{{</citation>}}


### (24/145) PHG-Net: Persistent Homology Guided Medical Image Classification (Yaopeng Peng et al., 2023)

{{<citation>}}

Yaopeng Peng, Hongxiao Wang, Milan Sonka, Danny Z. Chen. (2023)  
**PHG-Net: Persistent Homology Guided Medical Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Image Classification, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.17243v1)  

---


**ABSTRACT**  
Modern deep neural networks have achieved great successes in medical image analysis. However, the features captured by convolutional neural networks (CNNs) or Transformers tend to be optimized for pixel intensities and neglect key anatomical structures such as connected components and loops. In this paper, we propose a persistent homology guided approach (PHG-Net) that explores topological features of objects for medical image classification. For an input image, we first compute its cubical persistence diagram and extract topological features into a vector representation using a small neural network (called the PH module). The extracted topological features are then incorporated into the feature map generated by CNN or Transformer for feature fusion. The PH module is lightweight and capable of integrating topological features into any CNN or Transformer architectures in an end-to-end fashion. We evaluate our PHG-Net on three public datasets and demonstrate its considerable improvements on the target classification tasks over state-of-the-art methods.

{{</citation>}}


### (25/145) ReWaRD: Retinal Waves for Pre-Training Artificial Neural Networks Mimicking Real Prenatal Development (Benjamin Cappell et al., 2023)

{{<citation>}}

Benjamin Cappell, Andreas Stoll, Williams Chukwudi Umah, Bernhard Egger. (2023)  
**ReWaRD: Retinal Waves for Pre-Training Artificial Neural Networks Mimicking Real Prenatal Development**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.17232v1)  

---


**ABSTRACT**  
Computational models trained on a large amount of natural images are the state-of-the-art to study human vision - usually adult vision. Computational models of infant vision and its further development are gaining more and more attention in the community. In this work we aim at the very beginning of our visual experience - pre- and post-natal retinal waves which suggest to be a pre-training mechanism for the primate visual system at a very early stage of development. We see this approach as an instance of biologically plausible data driven inductive bias through pre-training. We built a computational model that mimics this development mechanism by pre-training different artificial convolutional neural networks with simulated retinal wave images. The resulting features of this biologically plausible pre-training closely match the V1 features of the primate visual system. We show that the performance gain by pre-training with retinal waves is similar to a state-of-the art pre-training pipeline. Our framework contains the retinal wave generator, as well as a training strategy, which can be a first step in a curriculum learning based training diet for various models of development. We release code, data and trained networks to build the basis for future work on visual development and based on a curriculum learning approach including prenatal development to support studies of innate vs. learned properties of the primate visual system. An additional benefit of our pre-trained networks for neuroscience or computer vision applications is the absence of biases inherited from datasets like ImageNet.

{{</citation>}}


### (26/145) BIM: Block-Wise Self-Supervised Learning with Masked Image Modeling (Yixuan Luo et al., 2023)

{{<citation>}}

Yixuan Luo, Mengye Ren, Sai Qian Zhang. (2023)  
**BIM: Block-Wise Self-Supervised Learning with Masked Image Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.17218v1)  

---


**ABSTRACT**  
Like masked language modeling (MLM) in natural language processing, masked image modeling (MIM) aims to extract valuable insights from image patches to enhance the feature extraction capabilities of the underlying deep neural network (DNN). Contrasted with other training paradigms like supervised learning and unsupervised contrastive learning, masked image modeling (MIM) pretraining typically demands significant computational resources in order to manage large training data batches (e.g., 4096). The significant memory and computation requirements pose a considerable challenge to its broad adoption. To mitigate this, we introduce a novel learning framework, termed~\textit{Block-Wise Masked Image Modeling} (BIM). This framework involves decomposing the MIM tasks into several sub-tasks with independent computation patterns, resulting in block-wise back-propagation operations instead of the traditional end-to-end approach. Our proposed BIM maintains superior performance compared to conventional MIM while greatly reducing peak memory consumption. Moreover, BIM naturally enables the concurrent training of numerous DNN backbones of varying depths. This leads to the creation of multiple trained DNN backbones, each tailored to different hardware platforms with distinct computing capabilities. This approach significantly reduces computational costs in comparison with training each DNN backbone individually. Our framework offers a promising solution for resource constrained training of MIM.

{{</citation>}}


### (27/145) Active Open-Vocabulary Recognition: Let Intelligent Moving Mitigate CLIP Limitations (Lei Fan et al., 2023)

{{<citation>}}

Lei Fan, Jianxiong Zhou, Xiaoying Xing, Ying Wu. (2023)  
**Active Open-Vocabulary Recognition: Let Intelligent Moving Mitigate CLIP Limitations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17938v1)  

---


**ABSTRACT**  
Active recognition, which allows intelligent agents to explore observations for better recognition performance, serves as a prerequisite for various embodied AI tasks, such as grasping, navigation and room arrangements. Given the evolving environment and the multitude of object classes, it is impractical to include all possible classes during the training stage. In this paper, we aim at advancing active open-vocabulary recognition, empowering embodied agents to actively perceive and classify arbitrary objects. However, directly adopting recent open-vocabulary classification models, like Contrastive Language Image Pretraining (CLIP), poses its unique challenges. Specifically, we observe that CLIP's performance is heavily affected by the viewpoint and occlusions, compromising its reliability in unconstrained embodied perception scenarios. Further, the sequential nature of observations in agent-environment interactions necessitates an effective method for integrating features that maintains discriminative strength for open-vocabulary classification. To address these issues, we introduce a novel agent for active open-vocabulary recognition. The proposed method leverages inter-frame and inter-concept similarities to navigate agent movements and to fuse features, without relying on class-specific knowledge. Compared to baseline CLIP model with 29.6% accuracy on ShapeNet dataset, the proposed agent could achieve 53.3% accuracy for open-vocabulary recognition, without any fine-tuning to the equipped CLIP model. Additional experiments conducted with the Habitat simulator further affirm the efficacy of our method.

{{</citation>}}


### (28/145) SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery (Konstantin Klemmer et al., 2023)

{{<citation>}}

Konstantin Klemmer, Esther Rolf, Caleb Robinson, Lester Mackey, Marc Rußwurm. (2023)  
**SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-CY, cs-LG, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.17179v1)  

---


**ABSTRACT**  
Geographic location is essential for modeling tasks in fields ranging from ecology to epidemiology to the Earth system sciences. However, extracting relevant and meaningful characteristics of a location can be challenging, often entailing expensive data fusion or data distillation from global imagery datasets. To address this challenge, we introduce Satellite Contrastive Location-Image Pretraining (SatCLIP), a global, general-purpose geographic location encoder that learns an implicit representation of locations from openly available satellite imagery. Trained location encoders provide vector embeddings summarizing the characteristics of any given location for convenient usage in diverse downstream tasks. We show that SatCLIP embeddings, pretrained on globally sampled multi-spectral Sentinel-2 satellite data, can be used in various predictive tasks that depend on location information but not necessarily satellite imagery, including temperature prediction, animal recognition in imagery, and population density estimation. Across tasks, SatCLIP embeddings consistently outperform embeddings from existing pretrained location encoders, ranging from models trained on natural images to models trained on semantic context. SatCLIP embeddings also help to improve geographic generalization. This demonstrates the potential of general-purpose location encoders and opens the door to learning meaningful representations of our planet from the vast, varied, and largely untapped modalities of geospatial data.

{{</citation>}}


### (29/145) Self-Supervised Motion Magnification by Backpropagating Through Optical Flow (Zhaoying Pan et al., 2023)

{{<citation>}}

Zhaoying Pan, Daniel Geng, Andrew Owens. (2023)  
**Self-Supervised Motion Magnification by Backpropagating Through Optical Flow**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.17056v1)  

---


**ABSTRACT**  
This paper presents a simple, self-supervised method for magnifying subtle motions in video: given an input video and a magnification factor, we manipulate the video such that its new optical flow is scaled by the desired amount. To train our model, we propose a loss function that estimates the optical flow of the generated video and penalizes how far if deviates from the given magnification factor. Thus, training involves differentiating through a pretrained optical flow network. Since our model is self-supervised, we can further improve its performance through test-time adaptation, by finetuning it on the input video. It can also be easily extended to magnify the motions of only user-selected objects. Our approach avoids the need for synthetic magnification datasets that have been used to train prior learning-based approaches. Instead, it leverages the existing capabilities of off-the-shelf motion estimators. We demonstrate the effectiveness of our method through evaluations of both visual quality and quantitative metrics on a range of real-world and synthetic videos, and we show our method works for both supervised and unsupervised optical flow methods.

{{</citation>}}


### (30/145) TLControl: Trajectory and Language Control for Human Motion Synthesis (Weilin Wan et al., 2023)

{{<citation>}}

Weilin Wan, Zhiyang Dou, Taku Komura, Wenping Wang, Dinesh Jayaraman, Lingjie Liu. (2023)  
**TLControl: Trajectory and Language Control for Human Motion Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2311.17135v1)  

---


**ABSTRACT**  
Controllable human motion synthesis is essential for applications in AR/VR, gaming, movies, and embodied AI. Existing methods often focus solely on either language or full trajectory control, lacking precision in synthesizing motions aligned with user-specified trajectories, especially for multi-joint control. To address these issues, we present TLControl, a new method for realistic human motion synthesis, incorporating both low-level trajectory and high-level language semantics controls. Specifically, we first train a VQ-VAE to learn a compact latent motion space organized by body parts. We then propose a Masked Trajectories Transformer to make coarse initial predictions of full trajectories of joints based on the learned latent motion space, with user-specified partial trajectories and text descriptions as conditioning. Finally, we introduce an efficient test-time optimization to refine these coarse predictions for accurate trajectory control. Experiments demonstrate that TLControl outperforms the state-of-the-art in trajectory accuracy and time efficiency, making it practical for interactive and high-quality animation generation.

{{</citation>}}


### (31/145) LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models (Yanwei Li et al., 2023)

{{<citation>}}

Yanwei Li, Chengyao Wang, Jiaya Jia. (2023)  
**LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17043v1)  

---


**ABSTRACT**  
In this work, we present a novel method to tackle the token generation challenge in Vision Language Models (VLMs) for video and image understanding, called LLaMA-VID. Current VLMs, while proficient in tasks like image captioning and visual question answering, face computational burdens when processing long videos due to the excessive visual tokens. LLaMA-VID addresses this issue by representing each frame with two distinct tokens, namely context token and content token. The context token encodes the overall image context based on user input, whereas the content token encapsulates visual cues in each frame. This dual-token strategy significantly reduces the overload of long videos while preserving critical information. Generally, LLaMA-VID empowers existing frameworks to support hour-long videos and pushes their upper limit with an extra context token. It is proved to surpass previous methods on most of video- or image-based benchmarks. Code is available https://github.com/dvlab-research/LLaMA-VID}{https://github.com/dvlab-research/LLaMA-VID

{{</citation>}}


### (32/145) Adversarial Diffusion Distillation (Axel Sauer et al., 2023)

{{<citation>}}

Axel Sauer, Dominik Lorenz, Andreas Blattmann, Robin Rombach. (2023)  
**Adversarial Diffusion Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17042v1)  

---


**ABSTRACT**  
We introduce Adversarial Diffusion Distillation (ADD), a novel training approach that efficiently samples large-scale foundational image diffusion models in just 1-4 steps while maintaining high image quality. We use score distillation to leverage large-scale off-the-shelf image diffusion models as a teacher signal in combination with an adversarial loss to ensure high image fidelity even in the low-step regime of one or two sampling steps. Our analyses show that our model clearly outperforms existing few-step methods (GANs, Latent Consistency Models) in a single step and reaches the performance of state-of-the-art diffusion models (SDXL) in only four steps. ADD is the first method to unlock single-step, real-time image synthesis with foundation models. Code and weights available under https://github.com/Stability-AI/generative-models and https://huggingface.co/stabilityai/ .

{{</citation>}}


### (33/145) Efficient In-Context Learning in Vision-Language Models for Egocentric Videos (Keunwoo Peter Yu et al., 2023)

{{<citation>}}

Keunwoo Peter Yu, Zheyuan Zhang, Fengyuan Hu, Joyce Chai. (2023)  
**Efficient In-Context Learning in Vision-Language Models for Egocentric Videos**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17041v2)  

---


**ABSTRACT**  
Recent advancements in text-only large language models (LLMs) have highlighted the benefit of in-context learning for adapting to new tasks with a few demonstrations. However, extending in-context learning to large vision-language models (VLMs) using a huge amount of naturalistic vision-language data has shown limited success, particularly for egocentric videos, due to high data collection costs. We propose a novel training method $\mathbb{E}$fficient $\mathbb{I}$n-context $\mathbb{L}$earning on $\mathbb{E}$gocentric $\mathbb{V}$ideos ($\mathbb{EILEV}$), which elicits in-context learning in VLMs for egocentric videos without requiring massive, naturalistic egocentric video datasets. $\mathbb{EILEV}$ involves architectural and training data adaptations to allow the model to process contexts interleaved with video clips and narrations, sampling of in-context examples with clusters of similar verbs and nouns, use of data with skewed marginal distributions with a long tail of infrequent verbs and nouns, as well as homonyms and synonyms. Our evaluations show that $\mathbb{EILEV}$-trained models outperform larger VLMs trained on a huge amount of naturalistic data in in-context learning. Furthermore, they can generalize to not only out-of-distribution, but also novel, rare egocentric videos and texts via in-context learning, demonstrating potential for applications requiring cost-effective training, and rapid post-deployment adaptability. Our code and demo are available at \url{https://github.com/yukw777/EILEV}.

{{</citation>}}


### (34/145) When the Few Outweigh the Many: Illicit Content Recognition with Few-Shot Learning (G. Cascavilla et al., 2023)

{{<citation>}}

G. Cascavilla, G. Catolino, M. Conti, D. Mellios, D. A. Tamburri. (2023)  
**When the Few Outweigh the Many: Illicit Content Recognition with Few-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CR, cs-CV, cs-CY, cs-LG, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.17026v1)  

---


**ABSTRACT**  
The anonymity and untraceability benefits of the Dark web account for the exponentially-increased potential of its popularity while creating a suitable womb for many illicit activities, to date. Hence, in collaboration with cybersecurity and law enforcement agencies, research has provided approaches for recognizing and classifying illicit activities with most exploiting textual dark web markets' content recognition; few such approaches use images that originated from dark web content. This paper investigates this alternative technique for recognizing illegal activities from images. In particular, we investigate label-agnostic learning techniques like One-Shot and Few-Shot learning featuring the use Siamese neural networks, a state-of-the-art approach in the field. Our solution manages to handle small-scale datasets with promising accuracy. In particular, Siamese neural networks reach 90.9% on 20-Shot experiments over a 10-class dataset; this leads us to conclude that such models are a promising and cheaper alternative to the definition of automated law-enforcing machinery over the dark web.

{{</citation>}}


### (35/145) Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer (Danah Yatim et al., 2023)

{{<citation>}}

Danah Yatim, Rafail Fridman, Omer Bar Tal, Yoni Kasten, Tali Dekel. (2023)  
**Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.17009v1)  

---


**ABSTRACT**  
We present a new method for text-driven motion transfer - synthesizing a video that complies with an input text prompt describing the target objects and scene while maintaining an input video's motion and scene layout. Prior methods are confined to transferring motion across two subjects within the same or closely related object categories and are applicable for limited domains (e.g., humans). In this work, we consider a significantly more challenging setting in which the target and source objects differ drastically in shape and fine-grained motion characteristics (e.g., translating a jumping dog into a dolphin). To this end, we leverage a pre-trained and fixed text-to-video diffusion model, which provides us with generative and motion priors. The pillar of our method is a new space-time feature loss derived directly from the model. This loss guides the generation process to preserve the overall motion of the input video while complying with the target object in terms of shape and fine-grained motion traits.

{{</citation>}}


### (36/145) TransNeXt: Robust Foveal Visual Perception for Vision Transformers (Dai Shi, 2023)

{{<citation>}}

Dai Shi. (2023)  
**TransNeXt: Robust Foveal Visual Perception for Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.17132v1)  

---


**ABSTRACT**  
Due to the depth degradation effect in residual connections, many efficient Vision Transformers models that rely on stacking layers for information exchange often fail to form sufficient information mixing, leading to unnatural visual perception. To address this issue, in this paper, we propose Aggregated Attention, a biomimetic design-based token mixer that simulates biological foveal vision and continuous eye movement while enabling each token on the feature map to have a global perception. Furthermore, we incorporate learnable tokens that interact with conventional queries and keys, which further diversifies the generation of affinity matrices beyond merely relying on the similarity between queries and keys. Our approach does not rely on stacking for information exchange, thus effectively avoiding depth degradation and achieving natural visual perception. Additionally, we propose Convolutional GLU, a channel mixer that bridges the gap between GLU and SE mechanism, which empowers each token to have channel attention based on its nearest neighbor image features, enhancing local modeling capability and model robustness. We combine aggregated attention and convolutional GLU to create a new visual backbone called TransNeXt. Extensive experiments demonstrate that our TransNeXt achieves state-of-the-art performance across multiple model sizes. At a resolution of $224^2$, TransNeXt-Tiny attains an ImageNet accuracy of 84.0%, surpassing ConvNeXt-B with 69% fewer parameters. Our TransNeXt-Base achieves an ImageNet accuracy of 86.2% and an ImageNet-A accuracy of 61.6% at a resolution of $384^2$, a COCO object detection mAP of 57.1, and an ADE20K semantic segmentation mIoU of 54.7.

{{</citation>}}


### (37/145) MVBench: A Comprehensive Multi-modal Video Understanding Benchmark (Kunchang Li et al., 2023)

{{<citation>}}

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, Yu Qiao. (2023)  
**MVBench: A Comprehensive Multi-modal Video Understanding Benchmark**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.17005v1)  

---


**ABSTRACT**  
With the rapid development of Multi-modal Large Language Models (MLLMs), a number of diagnostic benchmarks have recently emerged to evaluate the comprehension capabilities of these models. However, most benchmarks predominantly assess spatial understanding in the static image tasks, while overlooking temporal understanding in the dynamic video tasks. To alleviate this issue, we introduce a comprehensive Multi-modal Video understanding Benchmark, namely MVBench, which covers 20 challenging video tasks that cannot be effectively solved with a single frame. Specifically, we first introduce a novel static-to-dynamic method to define these temporal-related tasks. By transforming various static tasks into dynamic ones, we enable the systematic generation of video tasks that require a broad spectrum of temporal skills, ranging from perception to cognition. Then, guided by the task definition, we automatically convert public video annotations into multiple-choice QA to evaluate each task. On one hand, such a distinct paradigm allows us to build MVBench efficiently, without much manual intervention. On the other hand, it guarantees evaluation fairness with ground-truth video annotations, avoiding the biased scoring of LLMs. Moreover, we further develop a robust video MLLM baseline, i.e., VideoChat2, by progressive multi-modal training with diverse instruction-tuning data. The extensive results on our MVBench reveal that, the existing MLLMs are far from satisfactory in temporal understanding, while our VideoChat2 largely surpasses these leading models by over 15% on MVBench. All models and data are available at https://github.com/OpenGVLab/Ask-Anything.

{{</citation>}}


### (38/145) COLE: A Hierarchical Generation Framework for Graphic Design (Peidong Jia et al., 2023)

{{<citation>}}

Peidong Jia, Chenxuan Li, Zeyu Liu, Yichao Shen, Xingru Chen, Yuhui Yuan, Yinglin Zheng, Dong Chen, Ji Li, Xiaodong Xie, Shanghang Zhang, Baining Guo. (2023)  
**COLE: A Hierarchical Generation Framework for Graphic Design**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, NER  
[Paper Link](http://arxiv.org/abs/2311.16974v1)  

---


**ABSTRACT**  
Graphic design, which has been evolving since the 15th century, plays a crucial role in advertising. The creation of high-quality designs demands creativity, innovation, and lateral thinking. This intricate task involves understanding the objective, crafting visual elements such as the background, decoration, font, color, and shape, formulating diverse professional layouts, and adhering to fundamental visual design principles. In this paper, we introduce COLE, a hierarchical generation framework designed to comprehensively address these challenges. This COLE system can transform a straightforward intention prompt into a high-quality graphic design, while also supporting flexible editing based on user input. Examples of such input might include directives like ``design a poster for Hisaishi's concert.'' The key insight is to dissect the complex task of text-to-design generation into a hierarchy of simpler sub-tasks, each addressed by specialized models working collaboratively. The results from these models are then consolidated to produce a cohesive final output. Our hierarchical task decomposition can streamline the complex process and significantly enhance generation reliability. Our COLE system consists of multiple fine-tuned Large Language Models (LLMs), Large Multimodal Models (LMMs), and Diffusion Models (DMs), each specifically tailored for a design-aware text or image generation task. Furthermore, we construct the DESIGNERINTENTION benchmark to highlight the superiority of our COLE over existing methods in generating high-quality graphic designs from user intent. We perceive our COLE as an important step towards addressing more complex visual design generation tasks in the future.

{{</citation>}}


### (39/145) LLaFS: When Large-Language Models Meet Few-Shot Segmentation (Lanyun Zhu et al., 2023)

{{<citation>}}

Lanyun Zhu, Tianrun Chen, Deyi Ji, Jieping Ye, Jun Liu. (2023)  
**LLaFS: When Large-Language Models Meet Few-Shot Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16926v2)  

---


**ABSTRACT**  
This paper proposes LLaFS, the first attempt to leverage large language models (LLMs) in few-shot segmentation. In contrast to the conventional few-shot segmentation methods that only rely on the limited and biased information from the annotated support images, LLaFS leverages the vast prior knowledge gained by LLM as an effective supplement and directly uses the LLM to segment images in a few-shot manner. To enable the text-based LLM to handle image-related tasks, we carefully design an input instruction that allows the LLM to produce segmentation results represented as polygons, and propose a region-attribute table to simulate the human visual mechanism and provide multi-modal guidance. We also synthesize pseudo samples and use curriculum learning for pretraining to augment data and achieve better optimization. LLaFS achieves state-of-the-art results on multiple datasets, showing the potential of using LLMs for few-shot computer vision tasks. Code will be available at https://github.com/lanyunzhu99/LLaFS.

{{</citation>}}


### (40/145) Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding (Sicong Leng et al., 2023)

{{<citation>}}

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing. (2023)  
**Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16922v1)  

---


**ABSTRACT**  
Large Vision-Language Models (LVLMs) have advanced considerably, intertwining visual recognition and language understanding to generate content that is not only coherent but also contextually attuned. Despite their success, LVLMs still suffer from the issue of object hallucinations, where models generate plausible yet incorrect outputs that include objects that do not exist in the images. To mitigate this issue, we introduce Visual Contrastive Decoding (VCD), a simple and training-free method that contrasts output distributions derived from original and distorted visual inputs. The proposed VCD effectively reduces the over-reliance on statistical bias and unimodal priors, two essential causes of object hallucinations. This adjustment ensures the generated content is closely grounded to visual inputs, resulting in contextually accurate outputs. Our experiments show that VCD, without either additional training or the usage of external tools, significantly mitigates the object hallucination issue across different LVLM families. Beyond mitigating object hallucinations, VCD also excels in general LVLM benchmarks, highlighting its wide-ranging applicability.

{{</citation>}}


### (41/145) RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D (Lingteng Qiu et al., 2023)

{{<citation>}}

Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, Xiaoguang Han. (2023)  
**RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16918v1)  

---


**ABSTRACT**  
Lifting 2D diffusion for 3D generation is a challenging problem due to the lack of geometric prior and the complex entanglement of materials and lighting in natural images. Existing methods have shown promise by first creating the geometry through score-distillation sampling (SDS) applied to rendered surface normals, followed by appearance modeling. However, relying on a 2D RGB diffusion model to optimize surface normals is suboptimal due to the distribution discrepancy between natural images and normals maps, leading to instability in optimization. In this paper, recognizing that the normal and depth information effectively describe scene geometry and be automatically estimated from images, we propose to learn a generalizable Normal-Depth diffusion model for 3D generation. We achieve this by training on the large-scale LAION dataset together with the generalizable image-to-depth and normal prior models. In an attempt to alleviate the mixed illumination effects in the generated materials, we introduce an albedo diffusion model to impose data-driven constraints on the albedo component. Our experiments show that when integrated into existing text-to-3D pipelines, our models significantly enhance the detail richness, achieving state-of-the-art results. Our project page is https://lingtengqiu.github.io/RichDreamer/.

{{</citation>}}


### (42/145) Feedback RoI Features Improve Aerial Object Detection (Botao Ren et al., 2023)

{{<citation>}}

Botao Ren, Botian Xu, Tengyu Liu, Jingyi Wang, Zhidong Deng. (2023)  
**Feedback RoI Features Improve Aerial Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.17129v1)  

---


**ABSTRACT**  
Neuroscience studies have shown that the human visual system utilizes high-level feedback information to guide lower-level perception, enabling adaptation to signals of different characteristics. In light of this, we propose Feedback multi-Level feature Extractor (Flex) to incorporate a similar mechanism for object detection. Flex refines feature selection based on image-wise and instance-level feedback information in response to image quality variation and classification uncertainty. Experimental results show that Flex offers consistent improvement to a range of existing SOTA methods on the challenging aerial object detection datasets including DOTA-v1.0, DOTA-v1.5, and HRSC2016. Although the design originates in aerial image detection, further experiments on MS COCO also reveal our module's efficacy in general detection models. Quantitative and qualitative analyses indicate that the improvements are closely related to image qualities, which match our motivation.

{{</citation>}}


### (43/145) Vulnerability Analysis of Transformer-based Optical Character Recognition to Adversarial Attacks (Lucas Beerens et al., 2023)

{{<citation>}}

Lucas Beerens, Desmond J. Higham. (2023)  
**Vulnerability Analysis of Transformer-based Optical Character Recognition to Adversarial Attacks**  

---
Primary Category: cs.CV  
Categories: 65F35, I-2-10; G-1-3, cs-AI, cs-CV, cs.CV  
Keywords: AI, Adversarial Attack, OCR, Transformer  
[Paper Link](http://arxiv.org/abs/2311.17128v1)  

---


**ABSTRACT**  
Recent advancements in Optical Character Recognition (OCR) have been driven by transformer-based models. OCR systems are critical in numerous high-stakes domains, yet their vulnerability to adversarial attack remains largely uncharted territory, raising concerns about security and compliance with emerging AI regulations. In this work we present a novel framework to assess the resilience of Transformer-based OCR (TrOCR) models. We develop and assess algorithms for both targeted and untargeted attacks. For the untargeted case, we measure the Character Error Rate (CER), while for the targeted case we use the success ratio. We find that TrOCR is highly vulnerable to untargeted attacks and somewhat less vulnerable to targeted attacks. On a benchmark handwriting data set, untargeted attacks can cause a CER of more than 1 without being noticeable to the eye. With a similar perturbation size, targeted attacks can lead to success rates of around $25\%$ -- here we attacked single tokens, requiring TrOCR to output the tenth most likely token from a large vocabulary.

{{</citation>}}


### (44/145) Self-training solutions for the ICCV 2023 GeoNet Challenge (Lijun Sheng et al., 2023)

{{<citation>}}

Lijun Sheng, Zhengbo Wang, Jian Liang. (2023)  
**Self-training solutions for the ICCV 2023 GeoNet Challenge**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.16843v1)  

---


**ABSTRACT**  
GeoNet is a recently proposed domain adaptation benchmark consisting of three challenges (i.e., GeoUniDA, GeoImNet, and GeoPlaces). Each challenge contains images collected from the USA and Asia where there are huge geographical gaps. Our solution adopts a two-stage source-free domain adaptation framework with a Swin Transformer backbone to achieve knowledge transfer from the USA (source) domain to Asia (target) domain. In the first stage, we train a source model using labeled source data with a re-sampling strategy and two types of cross-entropy loss. In the second stage, we generate pseudo labels for unlabeled target data to fine-tune the model. Our method achieves an H-score of 74.56% and ultimately ranks 1st in the GeoUniDA challenge. In GeoImNet and GeoPlaces challenges, our solution also reaches a top-3 accuracy of 64.46% and 51.23%, respectively.

{{</citation>}}


### (45/145) Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization (Zhiyuan Zhao et al., 2023)

{{<citation>}}

Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, Jiaqi Wang, Conghui He. (2023)  
**Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.16839v1)  

---


**ABSTRACT**  
Multimodal large language models have made significant advancements in recent years, yet they still suffer from a common issue known as the "hallucination problem" where the models generate textual descriptions that contain inaccurate or non-existent content from the image. To address this issue, this paper introduces a novel strategy: Hallucination-Aware Direct Preference Optimization (HA-DPO). Our approach treats the hallucination problem as a unique preference selection issue, where the model is trained to favor the non-hallucinating response when presented with two responses of the same image (one accurate and one hallucinating). This paper also presents an efficient process for constructing hallucination sample pairs to ensure high-quality, style-consistent pairs for stable HA-DPO training. We applied this strategy to two mainstream multimodal models, and the results showed a significant reduction in the hallucination problem and an enhancement in the models' generalization capabilities. With HA-DPO, the MiniGPT-4 model demonstrates significant advancements: POPE accuracy increases from 51.13% to 85.66% (34.5% absolute improvement), and the MME score escalates from 968.58 to 1365.76 (41% relative improvement). The code, models, and datasets will be made publicly available.

{{</citation>}}


### (46/145) Reason out Your Layout: Evoking the Layout Master from Large Language Models for Text-to-Image Synthesis (Xiaohui Chen et al., 2023)

{{<citation>}}

Xiaohui Chen, Yongfei Liu, Yingxiang Yang, Jianbo Yuan, Quanzeng You, Li-Ping Liu, Hongxia Yang. (2023)  
**Reason out Your Layout: Evoking the Layout Master from Large Language Models for Text-to-Image Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17126v1)  

---


**ABSTRACT**  
Recent advancements in text-to-image (T2I) generative models have shown remarkable capabilities in producing diverse and imaginative visuals based on text prompts. Despite the advancement, these diffusion models sometimes struggle to translate the semantic content from the text into images entirely. While conditioning on the layout has shown to be effective in improving the compositional ability of T2I diffusion models, they typically require manual layout input. In this work, we introduce a novel approach to improving T2I diffusion models using Large Language Models (LLMs) as layout generators. Our method leverages the Chain-of-Thought prompting of LLMs to interpret text and generate spatially reasonable object layouts. The generated layout is then used to enhance the generated images' composition and spatial accuracy. Moreover, we propose an efficient adapter based on a cross-attention mechanism, which explicitly integrates the layout information into the stable diffusion models. Our experiments demonstrate significant improvements in image quality and layout accuracy, showcasing the potential of LLMs in augmenting generative image models.

{{</citation>}}


### (47/145) Unified-modal Salient Object Detection via Adaptive Prompt Learning (Kunpeng Wang et al., 2023)

{{<citation>}}

Kunpeng Wang, Chenglong Li, Zhengzheng Tu, Bin Luo. (2023)  
**Unified-modal Salient Object Detection via Adaptive Prompt Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.16835v2)  

---


**ABSTRACT**  
Existing single-modal and multi-modal salient object detection (SOD) methods focus on designing specific architectures tailored for their respective tasks. However, developing completely different models for different tasks leads to labor and time consumption, as well as high computational and practical deployment costs. In this paper, we make the first attempt to address both single-modal and multi-modal SOD in a unified framework called UniSOD. Nevertheless, assigning appropriate strategies to modality variable inputs is challenging. To this end, UniSOD learns modality-aware prompts with task-specific hints through adaptive prompt learning, which are plugged into the proposed pre-trained baseline SOD model to handle corresponding tasks, while only requiring few learnable parameters compared to training the entire model. Each modality-aware prompt is generated from a switchable prompt generation block, which performs structural switching solely relied on single-modal and multi-modal inputs. UniSOD achieves consistent performance improvement on 14 benchmark datasets for RGB, RGB-D, and RGB-T SOD, which demonstrates that our method effectively and efficiently unifies single-modal and multi-modal SOD tasks.

{{</citation>}}


### (48/145) Decomposer: Semi-supervised Learning of Image Restoration and Image Decomposition (Boris Meinardus et al., 2023)

{{<citation>}}

Boris Meinardus, Mariusz Trzeciakiewicz, Tim Herzig, Monika Kwiatkowski, Simon Matern, Olaf Hellwich. (2023)  
**Decomposer: Semi-supervised Learning of Image Restoration and Image Decomposition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.16829v1)  

---


**ABSTRACT**  
We present Decomposer, a semi-supervised reconstruction model that decomposes distorted image sequences into their fundamental building blocks - the original image and the applied augmentations, i.e., shadow, light, and occlusions. To solve this problem, we use the SIDAR dataset that provides a large number of distorted image sequences: each sequence contains images with shadows, lighting, and occlusions applied to an undistorted version. Each distortion changes the original signal in different ways, e.g., additive or multiplicative noise. We propose a transformer-based model to explicitly learn this decomposition. The sequential model uses 3D Swin-Transformers for spatio-temporal encoding and 3D U-Nets as prediction heads for individual parts of the decomposition. We demonstrate that by separately pre-training our model on weakly supervised pseudo labels, we can steer our model to optimize for our ambiguous problem definition and learn to differentiate between the different image distortions.

{{</citation>}}


### (49/145) The curse of language biases in remote sensing VQA: the role of spatial attributes, language diversity, and the need for clear evaluation (Christel Chappuis et al., 2023)

{{<citation>}}

Christel Chappuis, Eliot Walt, Vincent Mendez, Sylvain Lobry, Bertrand Le Saux, Devis Tuia. (2023)  
**The curse of language biases in remote sensing VQA: the role of spatial attributes, language diversity, and the need for clear evaluation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.16782v1)  

---


**ABSTRACT**  
Remote sensing visual question answering (RSVQA) opens new opportunities for the use of overhead imagery by the general public, by enabling human-machine interaction with natural language. Building on the recent advances in natural language processing and computer vision, the goal of RSVQA is to answer a question formulated in natural language about a remote sensing image. Language understanding is essential to the success of the task, but has not yet been thoroughly examined in RSVQA. In particular, the problem of language biases is often overlooked in the remote sensing community, which can impact model robustness and lead to wrong conclusions about the performances of the model. Thus, the present work aims at highlighting the problem of language biases in RSVQA with a threefold analysis strategy: visual blind models, adversarial testing and dataset analysis. This analysis focuses both on model and data. Moreover, we motivate the use of more informative and complementary evaluation metrics sensitive to the issue. The gravity of language biases in RSVQA is then exposed for all of these methods with the training of models discarding the image data and the manipulation of the visual input during inference. Finally, a detailed analysis of question-answer distribution demonstrates the root of the problem in the data itself. Thanks to this analytical study, we observed that biases in remote sensing are more severe than in standard VQA, likely due to the specifics of existing remote sensing datasets for the task, e.g. geographical similarities and sparsity, as well as a simpler vocabulary and question generation strategies. While new, improved and less-biased datasets appear as a necessity for the development of the promising field of RSVQA, we demonstrate that more informed, relative evaluation metrics remain much needed to transparently communicate results of future RSVQA methods.

{{</citation>}}


### (50/145) Large Model Based Referring Camouflaged Object Detection (Shupeng Cheng et al., 2023)

{{<citation>}}

Shupeng Cheng, Ge-Peng Ji, Pengda Qin, Deng-Ping Fan, Bowen Zhou, Peng Xu. (2023)  
**Large Model Based Referring Camouflaged Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Object Detection  
[Paper Link](http://arxiv.org/abs/2311.17122v1)  

---


**ABSTRACT**  
Referring camouflaged object detection (Ref-COD) is a recently-proposed problem aiming to segment out specified camouflaged objects matched with a textual or visual reference. This task involves two major challenges: the COD domain-specific perception and multimodal reference-image alignment. Our motivation is to make full use of the semantic intelligence and intrinsic knowledge of recent Multimodal Large Language Models (MLLMs) to decompose this complex task in a human-like way. As language is highly condensed and inductive, linguistic expression is the main media of human knowledge learning, and the transmission of knowledge information follows a multi-level progression from simplicity to complexity. In this paper, we propose a large-model-based Multi-Level Knowledge-Guided multimodal method for Ref-COD termed MLKG, where multi-level knowledge descriptions from MLLM are organized to guide the large vision model of segmentation to perceive the camouflage-targets and camouflage-scene progressively and meanwhile deeply align the textual references with camouflaged photos. To our knowledge, our contributions mainly include: (1) This is the first time that the MLLM knowledge is studied for Ref-COD and COD. (2) We, for the first time, propose decomposing Ref-COD into two main perspectives of perceiving the target and scene by integrating MLLM knowledge, and contribute a multi-level knowledge-guided method. (3) Our method achieves the state-of-the-art on the Ref-COD benchmark outperforming numerous strong competitors. Moreover, thanks to the injected rich knowledge, it demonstrates zero-shot generalization ability on uni-modal COD datasets. We will release our code soon.

{{</citation>}}


### (51/145) Generative Data Augmentation Improves Scribble-supervised Semantic Segmentation (Jacob Schnell et al., 2023)

{{<citation>}}

Jacob Schnell, Jieke Wang, Lu Qi, Vincent Tao Hu, Meng Tang. (2023)  
**Generative Data Augmentation Improves Scribble-supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.17121v1)  

---


**ABSTRACT**  
Recent advances in generative models, such as diffusion models, have made generating high-quality synthetic images widely accessible. Prior works have shown that training on synthetic images improves many perception tasks, such as image classification, object detection, and semantic segmentation. We are the first to explore generative data augmentations for scribble-supervised semantic segmentation. We propose a generative data augmentation method that leverages a ControlNet diffusion model conditioned on semantic scribbles to produce high-quality training data. However, naive implementations of generative data augmentations may inadvertently harm the performance of the downstream segmentor rather than improve it. We leverage classifier-free diffusion guidance to enforce class consistency and introduce encode ratios to trade off data diversity for data realism. Using the guidance scale and encode ratio, we are able to generate a spectrum of high-quality training images. We propose multiple augmentation schemes and find that these schemes significantly impact model performance, especially in the low-data regime. Our framework further reduces the gap between the performance of scribble-supervised segmentation and that of fully-supervised segmentation. We also show that our framework significantly improves segmentation performance on small datasets, even surpassing fully-supervised segmentation.

{{</citation>}}


### (52/145) Towards Full-scene Domain Generalization in Multi-agent Collaborative Bird's Eye View Segmentation for Connected and Autonomous Driving (Senkang Hu et al., 2023)

{{<citation>}}

Senkang Hu, Zhengru Fang, Xianhao Chen, Yuguang Fang, Sam Kwong. (2023)  
**Towards Full-scene Domain Generalization in Multi-agent Collaborative Bird's Eye View Segmentation for Connected and Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.16754v1)  

---


**ABSTRACT**  
Collaborative perception has recently gained significant attention in autonomous driving, improving perception quality by enabling the exchange of additional information among vehicles. However, deploying collaborative perception systems can lead to domain shifts due to diverse environmental conditions and data heterogeneity among connected and autonomous vehicles (CAVs). To address these challenges, we propose a unified domain generalization framework applicable in both training and inference stages of collaborative perception. In the training phase, we introduce an Amplitude Augmentation (AmpAug) method to augment low-frequency image variations, broadening the model's ability to learn across various domains. We also employ a meta-consistency training scheme to simulate domain shifts, optimizing the model with a carefully designed consistency loss to encourage domain-invariant representations. In the inference phase, we introduce an intra-system domain alignment mechanism to reduce or potentially eliminate the domain discrepancy among CAVs prior to inference. Comprehensive experiments substantiate the effectiveness of our method in comparison with the existing state-of-the-art works. Code will be released at https://github.com/DG-CAVs/DG-CoPerception.git.

{{</citation>}}


### (53/145) Riemannian Self-Attention Mechanism for SPD Networks (Rui Wang et al., 2023)

{{<citation>}}

Rui Wang, Xiao-Jun Wu, Hui Li, Josef Kittler. (2023)  
**Riemannian Self-Attention Mechanism for SPD Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, GLM, Self-Attention  
[Paper Link](http://arxiv.org/abs/2311.16738v1)  

---


**ABSTRACT**  
Symmetric positive definite (SPD) matrix has been demonstrated to be an effective feature descriptor in many scientific areas, as it can encode spatiotemporal statistics of the data adequately on a curved Riemannian manifold, i.e., SPD manifold. Although there are many different ways to design network architectures for SPD matrix nonlinear learning, very few solutions explicitly mine the geometrical dependencies of features at different layers. Motivated by the great success of self-attention mechanism in capturing long-range relationships, an SPD manifold self-attention mechanism (SMSA) is proposed in this paper using some manifold-valued geometric operations, mainly the Riemannian metric, Riemannian mean, and Riemannian optimization. Then, an SMSA-based geometric learning module (SMSA-GLM) is designed for the sake of improving the discrimination of the generated deep structured representations. Extensive experimental results achieved on three benchmarking datasets show that our modification against the baseline network further alleviates the information degradation problem and leads to improved accuracy.

{{</citation>}}


### (54/145) CADTalk: An Algorithm and Benchmark for Semantic Commenting of CAD Programs (Haocheng Yuan et al., 2023)

{{<citation>}}

Haocheng Yuan, Jing Xu, Hao Pan, Adrien Bousseau, Niloy Mitra, Changjian Li. (2023)  
**CADTalk: An Algorithm and Benchmark for Semantic Commenting of CAD Programs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.16703v2)  

---


**ABSTRACT**  
CAD programs are a popular way to compactly encode shapes as a sequence of operations that are easy to parametrically modify. However, without sufficient semantic comments and structure, such programs can be challenging to understand, let alone modify. We introduce the problem of semantic commenting CAD programs, wherein the goal is to segment the input program into code blocks corresponding to semantically meaningful shape parts and assign a semantic label to each block. We solve the problem by combining program parsing with visual-semantic analysis afforded by recent advances in foundational language and vision models. Specifically, by executing the input programs, we create shapes, which we use to generate conditional photorealistic images to make use of semantic annotators for such images. We then distill the information across the images and link back to the original programs to semantically comment on them. Additionally, we collected and annotated a benchmark dataset, CADTalk, consisting of 5,280 machine-made programs and 45 human-made programs with ground truth semantic comments to foster future research. We extensively evaluated our approach, compared to a GPT-based baseline approach, and an open-set shape segmentation baseline, i.e., PartSLIP, and reported an 83.24% accuracy on the new CADTalk dataset. Project page: https://enigma-li.github.io/CADTalk/.

{{</citation>}}


### (55/145) Rethinking Intermediate Layers design in Knowledge Distillation for Kidney and Liver Tumor Segmentation (Vandan Gorade et al., 2023)

{{<citation>}}

Vandan Gorade, Sparsh Mittal, Debesh Jha, Ulas Bagci. (2023)  
**Rethinking Intermediate Layers design in Knowledge Distillation for Kidney and Liver Tumor Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, q-bio-TO  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.16700v1)  

---


**ABSTRACT**  
Knowledge distillation(KD) has demonstrated remarkable success across various domains, but its application to medical imaging tasks, such as kidney and liver tumor segmentation, has encountered challenges. Many existing KD methods are not specifically tailored for these tasks. Moreover, prevalent KD methods often lack a careful consideration of what and from where to distill knowledge from the teacher to the student. This oversight may lead to issues like the accumulation of training bias within shallower student layers, potentially compromising the effectiveness of KD. To address these challenges, we propose Hierarchical Layer-selective Feedback Distillation (HLFD). HLFD strategically distills knowledge from a combination of middle layers to earlier layers and transfers final layer knowledge to intermediate layers at both the feature and pixel levels. This design allows the model to learn higher-quality representations from earlier layers, resulting in a robust and compact student model. Extensive quantitative evaluations reveal that HLFD outperforms existing methods by a significant margin. For example, in the kidney segmentation task, HLFD surpasses the student model (without KD) by over 10pp, significantly improving its focus on tumor-specific features. From a qualitative standpoint, the student model trained using HLFD excels at suppressing irrelevant information and can focus sharply on tumor-specific details, which opens a new pathway for more efficient and accurate diagnostic tools.

{{</citation>}}


### (56/145) ContextSeg: Sketch Semantic Segmentation by Querying the Context with Attention (Jiawei Wang et al., 2023)

{{<citation>}}

Jiawei Wang, Changjian Li. (2023)  
**ContextSeg: Sketch Semantic Segmentation by Querying the Context with Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Attention, Semantic Segmentation, Sketch, Transformer  
[Paper Link](http://arxiv.org/abs/2311.16682v1)  

---


**ABSTRACT**  
Sketch semantic segmentation is a well-explored and pivotal problem in computer vision involving the assignment of pre-defined part labels to individual strokes. This paper presents ContextSeg - a simple yet highly effective approach to tackling this problem with two stages. In the first stage, to better encode the shape and positional information of strokes, we propose to predict an extra dense distance field in an autoencoder network to reinforce structural information learning. In the second stage, we treat an entire stroke as a single entity and label a group of strokes within the same semantic part using an auto-regressive Transformer with the default attention mechanism. By group-based labeling, our method can fully leverage the context information when making decisions for the remaining groups of strokes. Our method achieves the best segmentation accuracy compared with state-of-the-art approaches on two representative datasets and has been extensively evaluated demonstrating its superior performance. Additionally, we offer insights into solving part imbalance in training data and the preliminary experiment on cross-category training, which can inspire future research in this field.

{{</citation>}}


### (57/145) Understanding the (Extra-)Ordinary: Validating Deep Model Decisions with Prototypical Concept-based Explanations (Maximilian Dreyer et al., 2023)

{{<citation>}}

Maximilian Dreyer, Reduan Achtibat, Wojciech Samek, Sebastian Lapuschkin. (2023)  
**Understanding the (Extra-)Ordinary: Validating Deep Model Decisions with Prototypical Concept-based Explanations**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.16681v1)  

---


**ABSTRACT**  
Ensuring both transparency and safety is critical when deploying Deep Neural Networks (DNNs) in high-risk applications, such as medicine. The field of explainable AI (XAI) has proposed various methods to comprehend the decision-making processes of opaque DNNs. However, only few XAI methods are suitable of ensuring safety in practice as they heavily rely on repeated labor-intensive and possibly biased human assessment. In this work, we present a novel post-hoc concept-based XAI framework that conveys besides instance-wise (local) also class-wise (global) decision-making strategies via prototypes. What sets our approach apart is the combination of local and global strategies, enabling a clearer understanding of the (dis-)similarities in model decisions compared to the expected (prototypical) concept use, ultimately reducing the dependence on human long-term assessment. Quantifying the deviation from prototypical behavior not only allows to associate predictions with specific model sub-strategies but also to detect outlier behavior. As such, our approach constitutes an intuitive and explainable tool for model validation. We demonstrate the effectiveness of our approach in identifying out-of-distribution samples, spurious model behavior and data quality issues across three datasets (ImageNet, CUB-200, and CIFAR-10) utilizing VGG, ResNet, and EfficientNet architectures. Code is available on https://github.com/maxdreyer/pcx.

{{</citation>}}


### (58/145) Large Language Models Meet Computer Vision: A Brief Survey (Raby Hamadi, 2023)

{{<citation>}}

Raby Hamadi. (2023)  
**Large Language Models Meet Computer Vision: A Brief Survey**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Computer Vision, Language Model, NLP, Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.16673v1)  

---


**ABSTRACT**  
Recently, the intersection of Large Language Models (LLMs) and Computer Vision (CV) has emerged as a pivotal area of research, driving significant advancements in the field of Artificial Intelligence (AI). As transformers have become the backbone of many state-of-the-art models in both Natural Language Processing (NLP) and CV, understanding their evolution and potential enhancements is crucial. This survey paper delves into the latest progressions in the domain of transformers and their subsequent successors, emphasizing their potential to revolutionize Vision Transformers (ViTs) and LLMs. This survey also presents a comparative analysis, juxtaposing the performance metrics of several leading paid and open-source LLMs, shedding light on their strengths and areas of improvement as well as a literature review on how LLMs are being used to tackle vision related tasks. Furthermore, the survey presents a comprehensive collection of datasets employed to train LLMs, offering insights into the diverse data available to achieve high performance in various pre-training and downstream tasks of LLMs. The survey is concluded by highlighting open directions in the field, suggesting potential venues for future research and development. This survey aims to underscores the profound intersection of LLMs on CV, leading to a new era of integrated and advanced AI models.

{{</citation>}}


### (59/145) MotionZero:Exploiting Motion Priors for Zero-shot Text-to-Video Generation (Sitong Su et al., 2023)

{{<citation>}}

Sitong Su, Litao Guo, Lianli Gao, Hengtao Shen, Jingkuan Song. (2023)  
**MotionZero:Exploiting Motion Priors for Zero-shot Text-to-Video Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.16635v1)  

---


**ABSTRACT**  
Zero-shot Text-to-Video synthesis generates videos based on prompts without any videos. Without motion information from videos, motion priors implied in prompts are vital guidance. For example, the prompt "airplane landing on the runway" indicates motion priors that the "airplane" moves downwards while the "runway" stays static. Whereas the motion priors are not fully exploited in previous approaches, thus leading to two nontrivial issues: 1) the motion variation pattern remains unaltered and prompt-agnostic for disregarding motion priors; 2) the motion control of different objects is inaccurate and entangled without considering the independent motion priors of different objects. To tackle the two issues, we propose a prompt-adaptive and disentangled motion control strategy coined as MotionZero, which derives motion priors from prompts of different objects by Large-Language-Models and accordingly applies motion control of different objects to corresponding regions in disentanglement. Furthermore, to facilitate videos with varying degrees of motion amplitude, we propose a Motion-Aware Attention scheme which adjusts attention among frames by motion amplitude. Extensive experiments demonstrate that our strategy could correctly control motion of different objects and support versatile applications including zero-shot video edit.

{{</citation>}}


### (60/145) Cross-level Attention with Overlapped Windows for Camouflaged Object Detection (Jiepan Li et al., 2023)

{{<citation>}}

Jiepan Li, Fangxiao Lu, Nan Xue, Zhuohong Li, Hongyan Zhang, Wei He. (2023)  
**Cross-level Attention with Overlapped Windows for Camouflaged Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2311.16618v1)  

---


**ABSTRACT**  
Camouflaged objects adaptively fit their color and texture with the environment, which makes them indistinguishable from the surroundings. Current methods revealed that high-level semantic features can highlight the differences between camouflaged objects and the backgrounds. Consequently, they integrate high-level semantic features with low-level detailed features for accurate camouflaged object detection (COD). Unlike previous designs for multi-level feature fusion, we state that enhancing low-level features is more impending for COD. In this paper, we propose an overlapped window cross-level attention (OWinCA) to achieve the low-level feature enhancement guided by the highest-level features. By sliding an aligned window pair on both the highest- and low-level feature maps, the high-level semantics are explicitly integrated into the low-level details via cross-level attention. Additionally, it employs an overlapped window partition strategy to alleviate the incoherence among windows, which prevents the loss of global information. These adoptions enable the proposed OWinCA to enhance low-level features by promoting the separability of camouflaged objects. The associated proposed OWinCANet fuses these enhanced multi-level features by simple convolution operation to achieve the final COD. Experiments conducted on three large-scale COD datasets demonstrate that our OWinCANet significantly surpasses the current state-of-the-art COD methods.

{{</citation>}}


### (61/145) Filter-Pruning of Lightweight Face Detectors Using a Geometric Median Criterion (Konstantinos Gkrispanis et al., 2023)

{{<citation>}}

Konstantinos Gkrispanis, Nikolaos Gkalelis, Vasileios Mezaris. (2023)  
**Filter-Pruning of Lightweight Face Detectors Using a Geometric Median Criterion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.16613v1)  

---


**ABSTRACT**  
Face detectors are becoming a crucial component of many applications, including surveillance, that often have to run on edge devices with limited processing power and memory. Therefore, there's a pressing demand for compact face detection models that can function efficiently across resource-constrained devices. Over recent years, network pruning techniques have attracted a lot of attention from researchers. These methods haven't been well examined in the context of face detectors, despite their expanding popularity. In this paper, we implement filter pruning on two already small and compact face detectors, named EXTD (Extremely Tiny Face Detector) and EResFD (Efficient ResNet Face Detector). The main pruning algorithm that we utilize is Filter Pruning via Geometric Median (FPGM), combined with the Soft Filter Pruning (SFP) iterative procedure. We also apply L1 Norm pruning, as a baseline to compare with the proposed approach. The experimental evaluation on the WIDER FACE dataset indicates that the proposed approach has the potential to further reduce the model size of already lightweight face detectors, with limited accuracy loss, or even with small accuracy gain for low pruning rates.

{{</citation>}}


### (62/145) DyRA: Dynamic Resolution Adjustment for Scale-robust Object Detection (Daeun Seo et al., 2023)

{{<citation>}}

Daeun Seo, Hoeseok Yang, Hyungshin Kim. (2023)  
**DyRA: Dynamic Resolution Adjustment for Scale-robust Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.17098v1)  

---


**ABSTRACT**  
In object detection, achieving constant accuracy is challenging due to the variability of object sizes. One possible solution to this problem is to optimize the input resolution, known as a multi-resolution strategy. Previous approaches for optimizing resolution are often based on pre-defined resolutions or a dynamic neural network, but there is a lack of study for run-time resolution optimization for existing architecture. In this paper, we propose an adaptive resolution scaling network called DyRA, which comprises convolutions and transformer encoder blocks, for existing detectors. Our DyRA returns a scale factor from an input image, which enables instance-specific scaling. This network is jointly trained with detectors with specially designed loss functions, namely ParetoScaleLoss and BalanceLoss. The ParetoScaleLoss produces an adaptive scale factor from the image, while the BalanceLoss optimizes the scale factor according to localization power for the dataset. The loss function is designed to minimize accuracy drop about the contrasting objective of small and large objects. Our experiments on COCO, RetinaNet, Faster-RCNN, FCOS, and Mask-RCNN achieved 1.3%, 1.1%, 1.3%, and 0.8% accuracy improvement than a multi-resolution baseline with solely resolution adjustment. The code is available at https://github.com/DaEunFullGrace/DyRA.git.

{{</citation>}}


### (63/145) Efficient Key-Based Adversarial Defense for ImageNet by Using Pre-trained Model (AprilPyone MaungMaung et al., 2023)

{{<citation>}}

AprilPyone MaungMaung, Isao Echizen, Hitoshi Kiya. (2023)  
**Efficient Key-Based Adversarial Defense for ImageNet by Using Pre-trained Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.16577v1)  

---


**ABSTRACT**  
In this paper, we propose key-based defense model proliferation by leveraging pre-trained models and utilizing recent efficient fine-tuning techniques on ImageNet-1k classification. First, we stress that deploying key-based models on edge devices is feasible with the latest model deployment advancements, such as Apple CoreML, although the mainstream enterprise edge artificial intelligence (Edge AI) has been focused on the Cloud. Then, we point out that the previous key-based defense on on-device image classification is impractical for two reasons: (1) training many classifiers from scratch is not feasible, and (2) key-based defenses still need to be thoroughly tested on large datasets like ImageNet. To this end, we propose to leverage pre-trained models and utilize efficient fine-tuning techniques to proliferate key-based models even on limited computing resources. Experiments were carried out on the ImageNet-1k dataset using adaptive and non-adaptive attacks. The results show that our proposed fine-tuned key-based models achieve a superior classification accuracy (more than 10% increase) compared to the previous key-based models on classifying clean and adversarial examples.

{{</citation>}}


### (64/145) Plug-and-Play, Dense-Label-Free Extraction of Open-Vocabulary Semantic Segmentation from Vision-Language Models (Luo Jiayun et al., 2023)

{{<citation>}}

Luo Jiayun, Siddhesh Khandelwal, Leonid Sigal, Boyang Li. (2023)  
**Plug-and-Play, Dense-Label-Free Extraction of Open-Vocabulary Semantic Segmentation from Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.17095v1)  

---


**ABSTRACT**  
From an enormous amount of image-text pairs, large-scale vision-language models (VLMs) learn to implicitly associate image regions with words, which is vital for tasks such as image captioning and visual question answering. However, leveraging such pre-trained models for open-vocabulary semantic segmentation remains a challenge. In this paper, we propose a simple, yet extremely effective, training-free technique, Plug-and-Play Open-Vocabulary Semantic Segmentation (PnP-OVSS) for this task. PnP-OVSS leverages a VLM with direct text-to-image cross-attention and an image-text matching loss to produce semantic segmentation. However, cross-attention alone tends to over-segment, whereas cross-attention plus GradCAM tend to under-segment. To alleviate this issue, we introduce Salience Dropout; by iteratively dropping patches that the model is most attentive to, we are able to better resolve the entire extent of the segmentation mask. Compared to existing techniques, the proposed method does not require any neural network training and performs hyperparameter tuning without the need for any segmentation annotations, even for a validation set. PnP-OVSS demonstrates substantial improvements over a comparable baseline (+29.4% mIoU on Pascal VOC, +13.2% mIoU on Pascal Context, +14.0% mIoU on MS COCO, +2.4% mIoU on COCO Stuff) and even outperforms most baselines that conduct additional network training on top of pretrained VLMs.

{{</citation>}}


### (65/145) Agents meet OKR: An Object and Key Results Driven Agent System with Hierarchical Self-Collaboration and Self-Evaluation (Yi Zheng et al., 2023)

{{<citation>}}

Yi Zheng, Chongyang Ma, Kanle Shi, Haibin Huang. (2023)  
**Agents meet OKR: An Object and Key Results Driven Agent System with Hierarchical Self-Collaboration and Self-Evaluation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16542v1)  

---


**ABSTRACT**  
In this study, we introduce the concept of OKR-Agent designed to enhance the capabilities of Large Language Models (LLMs) in task-solving. Our approach utilizes both self-collaboration and self-correction mechanism, facilitated by hierarchical agents, to address the inherent complexities in task-solving. Our key observations are two-fold: first, effective task-solving demands in-depth domain knowledge and intricate reasoning, for which deploying specialized agents for individual sub-tasks can markedly enhance LLM performance. Second, task-solving intrinsically adheres to a hierarchical execution structure, comprising both high-level strategic planning and detailed task execution. Towards this end, our OKR-Agent paradigm aligns closely with this hierarchical structure, promising enhanced efficacy and adaptability across a range of scenarios. Specifically, our framework includes two novel modules: hierarchical Objects and Key Results generation and multi-level evaluation, each contributing to more efficient and robust task-solving. In practical, hierarchical OKR generation decomposes Objects into multiple sub-Objects and assigns new agents based on key results and agent responsibilities. These agents subsequently elaborate on their designated tasks and may further decompose them as necessary. Such generation operates recursively and hierarchically, culminating in a comprehensive set of detailed solutions. The multi-level evaluation module of OKR-Agent refines solution by leveraging feedback from all associated agents, optimizing each step of the process. This ensures solution is accurate, practical, and effectively address intricate task requirements, enhancing the overall reliability and quality of the outcome. Experimental results also show our method outperforms the previous methods on several tasks. Code and demo are available at https://okr-agent.github.io/

{{</citation>}}


### (66/145) Improved Prototypical Semi-Supervised Learning with Foundation Models: Prototype Selection, Parametric vMF-SNE Pretraining and Multi-view Pseudolabelling (Evelyn Mannix et al., 2023)

{{<citation>}}

Evelyn Mannix, Howard Bondell. (2023)  
**Improved Prototypical Semi-Supervised Learning with Foundation Models: Prototype Selection, Parametric vMF-SNE Pretraining and Multi-view Pseudolabelling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AWS, Embedding, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.17093v1)  

---


**ABSTRACT**  
In this paper we present an improved approach to prototypical semi-supervised learning for computer vision, in the context of leveraging a frozen foundation model as the backbone of our neural network. As a general tool, we propose parametric von-Mises Fisher Stochastic Neighbour Embedding (vMF-SNE) to create mappings with neural networks between high-dimensional latent spaces that preserve local structure. This enables us to pretrain the projection head of our network using the high-quality embeddings of the foundation model with vMF-SNE. We also propose soft multi-view pseudolabels, where predictions across multiple views are combined to provide a more reliable supervision signal compared to a consistency or swapped assignment approach. We demonstrate that these ideas improve upon P}redicting View-Assignments with Support Samples (PAWS), a current state-of-the-art semi-supervised learning method, as well as Robust PAWS (RoPAWS), over a range of benchmarking datasets. We also introduce simple $k$-means prototype selection, a technique that provides superior performance to other unsupervised label selection approaches in this context. These changes improve upon PAWS by an average of +2.9% for CIFAR-10 and +5.7% for CIFAR-100 with four labels per class, and by +15.2% for DeepWeeds, a particularly challenging dataset for semi-supervised learning. We also achieve new state-of-the-art results in semi-supervised learning in this small label regime for CIFAR-10 - 95.8% (+0.7%) and CIFAR-100 - 76.6% (+12.0%).

{{</citation>}}


### (67/145) SEED-Bench-2: Benchmarking Multimodal Large Language Models (Bohao Li et al., 2023)

{{<citation>}}

Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, Ying Shan. (2023)  
**SEED-Bench-2: Benchmarking Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17092v1)  

---


**ABSTRACT**  
Multimodal large language models (MLLMs), building upon the foundation of powerful large language models (LLMs), have recently demonstrated exceptional capabilities in generating not only texts but also images given interleaved multimodal inputs (acting like a combination of GPT-4V and DALL-E 3). However, existing MLLM benchmarks remain limited to assessing only models' comprehension ability of single image-text inputs, failing to keep up with the strides made in MLLMs. A comprehensive benchmark is imperative for investigating the progress and uncovering the limitations of current MLLMs. In this work, we categorize the capabilities of MLLMs into hierarchical levels from $L_0$ to $L_4$ based on the modalities they can accept and generate, and propose SEED-Bench-2, a comprehensive benchmark that evaluates the \textbf{hierarchical} capabilities of MLLMs. Specifically, SEED-Bench-2 comprises 24K multiple-choice questions with accurate human annotations, which spans 27 dimensions, including the evaluation of both text and image generation. Multiple-choice questions with groundtruth options derived from human annotation enables an objective and efficient assessment of model performance, eliminating the need for human or GPT intervention during evaluation. We further evaluate the performance of 23 prominent open-source MLLMs and summarize valuable observations. By revealing the limitations of existing MLLMs through extensive evaluations, we aim for SEED-Bench-2 to provide insights that will motivate future research towards the goal of General Artificial Intelligence. Dataset and evaluation code are available at \href{https://github.com/AILab-CVC/SEED-Bench}

{{</citation>}}


### (68/145) Beyond Sole Strength: Customized Ensembles for Generalized Vision-Language Models (Zhihe Lu et al., 2023)

{{<citation>}}

Zhihe Lu, Jiawang Bai, Xin Li, Zeyu Xiao, Xinchao Wang. (2023)  
**Beyond Sole Strength: Customized Ensembles for Generalized Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17091v1)  

---


**ABSTRACT**  
Fine-tuning pre-trained vision-language models (VLMs), e.g., CLIP, for the open-world generalization has gained increasing popularity due to its practical value. However, performance advancements are limited when relying solely on intricate algorithmic designs for a single model, even one exhibiting strong performance, e.g., CLIP-ViT-B/16. This paper, for the first time, explores the collaborative potential of leveraging much weaker VLMs to enhance the generalization of a robust single model. The affirmative findings motivate us to address the generalization problem from a novel perspective, i.e., ensemble of pre-trained VLMs. We introduce three customized ensemble strategies, each tailored to one specific scenario. Firstly, we introduce the zero-shot ensemble, automatically adjusting the logits of different models based on their confidence when only pre-trained VLMs are available. Furthermore, for scenarios with extra few-shot samples, we propose the training-free and tuning ensemble, offering flexibility based on the availability of computing resources. The proposed ensemble strategies are evaluated on zero-shot, base-to-new, and cross-dataset generalization, achieving new state-of-the-art performance. Notably, this work represents an initial stride toward enhancing the generalization performance of VLMs via ensemble. The code is available at https://github.com/zhiheLu/Ensemble_VLM.git.

{{</citation>}}


### (69/145) AvatarGPT: All-in-One Framework for Motion Understanding, Planning, Generation and Beyond (Zixiang Zhou et al., 2023)

{{<citation>}}

Zixiang Zhou, Yu Wan, Baoyuan Wang. (2023)  
**AvatarGPT: All-in-One Framework for Motion Understanding, Planning, Generation and Beyond**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.16468v1)  

---


**ABSTRACT**  
Large Language Models(LLMs) have shown remarkable emergent abilities in unifying almost all (if not every) NLP tasks. In the human motion-related realm, however, researchers still develop siloed models for each task. Inspired by InstuctGPT, and the generalist concept behind Gato, we introduce AvatarGPT, an All-in-One framework for motion understanding, planning, generations as well as other tasks such as motion in-between synthesis. AvatarGPT treats each task as one type of instruction fine-tuned on the shared LLM. All the tasks are seamlessly interconnected with language as the universal interface, constituting a closed-loop within the framework. To achieve this, human motion sequences are first encoded as discrete tokens, which serve as the extended vocabulary of LLM. Then, an unsupervised pipeline to generate natural language descriptions of human action sequences from in-the-wild videos is developed. Finally, all tasks are jointly trained. Extensive experiments show that AvatarGPT achieves SOTA on low-level tasks, and promising results on high-level tasks, demonstrating the effectiveness of our proposed All-in-One framework. Moreover, for the first time, AvatarGPT enables a principled approach by iterative traversal of the tasks within the closed-loop for unlimited long-motion synthesis.

{{</citation>}}


### (70/145) TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering (Jingye Chen et al., 2023)

{{<citation>}}

Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, Furu Wei. (2023)  
**TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16465v1)  

---


**ABSTRACT**  
The diffusion model has been proven a powerful generative model in recent years, yet remains a challenge in generating visual text. Several methods alleviated this issue by incorporating explicit text position and content as guidance on where and what text to render. However, these methods still suffer from several drawbacks, such as limited flexibility and automation, constrained capability of layout prediction, and restricted style diversity. In this paper, we present TextDiffuser-2, aiming to unleash the power of language models for text rendering. Firstly, we fine-tune a large language model for layout planning. The large language model is capable of automatically generating keywords for text rendering and also supports layout modification through chatting. Secondly, we utilize the language model within the diffusion model to encode the position and texts at the line level. Unlike previous methods that employed tight character-level guidance, this approach generates more diverse text images. We conduct extensive experiments and incorporate user studies involving human participants as well as GPT-4V, validating TextDiffuser-2's capacity to achieve a more rational text layout and generation with enhanced diversity. The code and model will be available at \url{https://aka.ms/textdiffuser-2}.

{{</citation>}}


### (71/145) Spiking Neural Networks with Dynamic Time Steps for Vision Transformers (Gourav Datta et al., 2023)

{{<citation>}}

Gourav Datta, Zeyu Liu, Anni Li, Peter A. Beerel. (2023)  
**Spiking Neural Networks with Dynamic Time Steps for Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.16456v1)  

---


**ABSTRACT**  
Spiking Neural Networks (SNNs) have emerged as a popular spatio-temporal computing paradigm for complex vision tasks. Recently proposed SNN training algorithms have significantly reduced the number of time steps (down to 1) for improved latency and energy efficiency, however, they target only convolutional neural networks (CNN). These algorithms, when applied on the recently spotlighted vision transformers (ViT), either require a large number of time steps or fail to converge. Based on analysis of the histograms of the ANN and SNN activation maps, we hypothesize that each ViT block has a different sensitivity to the number of time steps. We propose a novel training framework that dynamically allocates the number of time steps to each ViT module depending on a trainable score assigned to each timestep. In particular, we generate a scalar binary time step mask that filters spikes emitted by each neuron in a leaky-integrate-and-fire (LIF) layer. The resulting SNNs have high activation sparsity and require only accumulate operations (AC), except for the input embedding layer, in contrast to expensive multiply-and-accumulates (MAC) needed in traditional ViTs. This yields significant improvements in energy efficiency. We evaluate our training framework and resulting SNNs on image recognition tasks including CIFAR10, CIFAR100, and ImageNet with different ViT architectures. We obtain a test accuracy of 95.97% with 4.97 time steps with direct encoding on CIFAR10.

{{</citation>}}


### (72/145) Typhoon Intensity Prediction with Vision Transformer (Huanxin Chen et al., 2023)

{{<citation>}}

Huanxin Chen, Pengshuai Yin, Huichou Huang, Qingyao Wu, Ruirui Liu, Xiatian Zhu. (2023)  
**Typhoon Intensity Prediction with Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.16450v1)  

---


**ABSTRACT**  
Predicting typhoon intensity accurately across space and time is crucial for issuing timely disaster warnings and facilitating emergency response. This has vast potential for minimizing life losses and property damages as well as reducing economic and environmental impacts. Leveraging satellite imagery for scenario analysis is effective but also introduces additional challenges due to the complex relations among clouds and the highly dynamic context. Existing deep learning methods in this domain rely on convolutional neural networks (CNNs), which suffer from limited per-layer receptive fields. This limitation hinders their ability to capture long-range dependencies and global contextual knowledge during inference. In response, we introduce a novel approach, namely "Typhoon Intensity Transformer" (Tint), which leverages self-attention mechanisms with global receptive fields per layer. Tint adopts a sequence-to-sequence feature representation learning perspective. It begins by cutting a given satellite image into a sequence of patches and recursively employs self-attention operations to extract both local and global contextual relations between all patch pairs simultaneously, thereby enhancing per-patch feature representation learning. Extensive experiments on a publicly available typhoon benchmark validate the efficacy of Tint in comparison with both state-of-the-art deep learning and conventional meteorological methods. Our code is available at https://github.com/chen-huanxin/Tint.

{{</citation>}}


### (73/145) Rethinking Mixup for Improving the Adversarial Transferability (Xiaosen Wang et al., 2023)

{{<citation>}}

Xiaosen Wang, Zeyuan Yin. (2023)  
**Rethinking Mixup for Improving the Adversarial Transferability**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.17087v1)  

---


**ABSTRACT**  
Mixup augmentation has been widely integrated to generate adversarial examples with superior adversarial transferability when immigrating from a surrogate model to other models. However, the underlying mechanism influencing the mixup's effect on transferability remains unexplored. In this work, we posit that the adversarial examples located at the convergence of decision boundaries across various categories exhibit better transferability and identify that Admix tends to steer the adversarial examples towards such regions. However, we find the constraint on the added image in Admix decays its capability, resulting in limited transferability. To address such an issue, we propose a new input transformation-based attack called Mixing the Image but Separating the gradienT (MIST). Specifically, MIST randomly mixes the input image with a randomly shifted image and separates the gradient of each loss item for each mixed image. To counteract the imprecise gradient, MIST calculates the gradient on several mixed images for each input sample. Extensive experimental results on the ImageNet dataset demonstrate that MIST outperforms existing SOTA input transformation-based attacks with a clear margin on both Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) w/wo defense mechanisms, supporting MIST's high effectiveness and generality.

{{</citation>}}


### (74/145) CLAP: Contrastive Learning with Augmented Prompts for Robustness on Pretrained Vision-Language Models (Yichao Cai et al., 2023)

{{<citation>}}

Yichao Cai, Yuhang Liu, Zhen Zhang, Javen Qinfeng Shi. (2023)  
**CLAP: Contrastive Learning with Augmented Prompts for Robustness on Pretrained Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16445v1)  

---


**ABSTRACT**  
Contrastive vision-language models, e.g., CLIP, have garnered substantial attention for their exceptional generalization capabilities. However, their robustness to perturbations has ignited concerns. Existing strategies typically reinforce their resilience against adversarial examples by enabling the image encoder to "see" these perturbed examples, often necessitating a complete retraining of the image encoder on both natural and adversarial samples. In this study, we propose a new method to enhance robustness solely through text augmentation, eliminating the need for retraining the image encoder on adversarial examples. Our motivation arises from the realization that text and image data inherently occupy a shared latent space, comprising latent content variables and style variables. This insight suggests the feasibility of learning to disentangle these latent content variables using text data exclusively. To accomplish this, we introduce an effective text augmentation method that focuses on modifying the style while preserving the content in the text data. By changing the style part of the text data, we empower the text encoder to emphasize latent content variables, ultimately enhancing the robustness of vision-language models. Our experiments across various datasets demonstrate substantial improvements in the robustness of the pre-trained CLIP model.

{{</citation>}}


### (75/145) PEA-Diffusion: Parameter-Efficient Adapter with Knowledge Distillation in non-English Text-to-Image Generation (Jian Ma et al., 2023)

{{<citation>}}

Jian Ma, Chen Chen, Qingsong Xie, Haonan Lu. (2023)  
**PEA-Diffusion: Parameter-Efficient Adapter with Knowledge Distillation in non-English Text-to-Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.17086v1)  

---


**ABSTRACT**  
Text-to-image diffusion models are well-known for their ability to generate realistic images based on textual prompts. However, the existing works have predominantly focused on English, lacking support for non-English text-to-image models. The most commonly used translation methods cannot solve the generation problem related to language culture, while training from scratch on a specific language dataset is prohibitively expensive. In this paper, we are inspired to propose a simple plug-and-play language transfer method based on knowledge distillation. All we need to do is train a lightweight MLP-like parameter-efficient adapter (PEA) with only 6M parameters under teacher knowledge distillation along with a small parallel data corpus. We are surprised to find that freezing the parameters of UNet can still achieve remarkable performance on the language-specific prompt evaluation set, demonstrating that PEA can stimulate the potential generation ability of the original UNet. Additionally, it closely approaches the performance of the English text-to-image model on a general prompt evaluation set. Furthermore, our adapter can be used as a plugin to achieve significant results in downstream tasks in cross-lingual text-to-image generation. Code will be available at: https://github.com/OPPO-Mente-Lab/PEA-Diffusion

{{</citation>}}


### (76/145) Combating the 'Sameness' in AI Art: Reflections on the Interactive AI Installation Fencing Hallucination (Weihao Qiu et al., 2023)

{{<citation>}}

Weihao Qiu, George Legrady. (2023)  
**Combating the 'Sameness' in AI Art: Reflections on the Interactive AI Installation Fencing Hallucination**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17080v1)  

---


**ABSTRACT**  
The article summarizes three types of "sameness" issues in Artificial Intelligence(AI) art, each occurring at different stages of development in AI image creation tools. Through the Fencing Hallucination project, the article reflects on the design of AI art production in alleviating the sense of uniformity, maintaining the uniqueness of images from an AI image synthesizer, and enhancing the connection between the artworks and the audience. This paper endeavors to stimulate the creation of distinctive AI art by recounting the efforts and insights derived from the Fencing Hallucination project, all dedicated to addressing the issue of "sameness".

{{</citation>}}


## cs.LG (27)



### (77/145) SoUnD Framework: Analyzing (So)cial Representation in (Un)structured (D)ata (Mark Díaz et al., 2023)

{{<citation>}}

Mark Díaz, Sunipa Dev, Emily Reif, Remi Denton, Vinodkumar Prabhakaran. (2023)  
**SoUnD Framework: Analyzing (So)cial Representation in (Un)structured (D)ata**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17259v1)  

---


**ABSTRACT**  
The unstructured nature of data used in foundation model development is a challenge to systematic analyses for making data use and documentation decisions. From a Responsible AI perspective, these decisions often rely upon understanding how people are represented in data. We propose a framework designed to guide analysis of human representation in unstructured data and identify downstream risks. We apply the framework in two toy examples using the Common Crawl web text corpus (C4) and LAION-400M. We also propose a set of hypothetical action steps in service of dataset use, development, and documentation.

{{</citation>}}


### (78/145) Optimal EEG Electrode Set for Emotion Recognition From Brain Signals: An Empirical Quest (Rumman Ahmed Prodhan et al., 2023)

{{<citation>}}

Rumman Ahmed Prodhan, Sumya Akter, Tanmoy Sarkar Pias, Md. Akhtaruzzaman Adnan. (2023)  
**Optimal EEG Electrode Set for Emotion Recognition From Brain Signals: An Empirical Quest**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2311.17204v1)  

---


**ABSTRACT**  
The human brain is a complex organ, still completely undiscovered, that controls almost all the parts of the body. Apart from survival, the human brain stimulates emotions. Recent research indicates that brain signals can be very effective for emotion recognition. However, which parts of the brain exhibit most of the emotions is still under-explored. In this study, we empirically analyze the contribution of each part of the brain in exhibiting emotions. We use the DEAP dataset to find the most optimal electrode set which eventually leads to the effective brain part associated with emotions. We use Fast Fourier Transformation for effective feature extraction and a 1D-CNN with residual connection for classification. Though 32 electrodes from the DEAP dataset got an accuracy of 97.34%, only 12 electrodes (F7, P8, O1, F8, C4, T7, PO3, Fp1, Fp2, O2, P3, and Fz) achieve 95.81% accuracy. This study also shows that adding more than 10 electrodes does not improve performance significantly. Moreover, the frontal lobe is the most important for recognizing emotion.

{{</citation>}}


### (79/145) Minimax Exploiter: A Data Efficient Approach for Competitive Self-Play (Daniel Bairamian et al., 2023)

{{<citation>}}

Daniel Bairamian, Philippe Marcotte, Joshua Romoff, Gabriel Robert, Derek Nowrouzezahrai. (2023)  
**Minimax Exploiter: A Data Efficient Approach for Competitive Self-Play**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.17190v1)  

---


**ABSTRACT**  
Recent advances in Competitive Self-Play (CSP) have achieved, or even surpassed, human level performance in complex game environments such as Dota 2 and StarCraft II using Distributed Multi-Agent Reinforcement Learning (MARL). One core component of these methods relies on creating a pool of learning agents -- consisting of the Main Agent, past versions of this agent, and Exploiter Agents -- where Exploiter Agents learn counter-strategies to the Main Agents. A key drawback of these approaches is the large computational cost and physical time that is required to train the system, making them impractical to deploy in highly iterative real-life settings such as video game productions. In this paper, we propose the Minimax Exploiter, a game theoretic approach to exploiting Main Agents that leverages knowledge of its opponents, leading to significant increases in data efficiency. We validate our approach in a diversity of settings, including simple turn based games, the arcade learning environment, and For Honor, a modern video game. The Minimax Exploiter consistently outperforms strong baselines, demonstrating improved stability and data efficiency, leading to a robust CSP-MARL method that is both flexible and easy to deploy.

{{</citation>}}


### (80/145) Scalable Extraction of Training Data from (Production) Language Models (Milad Nasr et al., 2023)

{{<citation>}}

Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, Katherine Lee. (2023)  
**Scalable Extraction of Training Data from (Production) Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: ChatGPT, Falcon, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17035v1)  

---


**ABSTRACT**  
This paper studies extractable memorization: training data that an adversary can efficiently extract by querying a machine learning model without prior knowledge of the training dataset. We show an adversary can extract gigabytes of training data from open-source language models like Pythia or GPT-Neo, semi-open models like LLaMA or Falcon, and closed models like ChatGPT. Existing techniques from the literature suffice to attack unaligned models; in order to attack the aligned ChatGPT, we develop a new divergence attack that causes the model to diverge from its chatbot-style generations and emit training data at a rate 150x higher than when behaving properly. Our methods show practical attacks can recover far more data than previously thought, and reveal that current alignment techniques do not eliminate memorization.

{{</citation>}}


### (81/145) Deployment of a Robust and Explainable Mortality Prediction Model: The COVID-19 Pandemic and Beyond (Jacob R. Epifano et al., 2023)

{{<citation>}}

Jacob R. Epifano, Stephen Glass, Ravi P. Ramachandran, Sharad Patel, Aaron J. Masino, Ghulam Rasool. (2023)  
**Deployment of a Robust and Explainable Mortality Prediction Model: The COVID-19 Pandemic and Beyond**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17133v1)  

---


**ABSTRACT**  
This study investigated the performance, explainability, and robustness of deployed artificial intelligence (AI) models in predicting mortality during the COVID-19 pandemic and beyond. The first study of its kind, we found that Bayesian Neural Networks (BNNs) and intelligent training techniques allowed our models to maintain performance amidst significant data shifts. Our results emphasize the importance of developing robust AI models capable of matching or surpassing clinician predictions, even under challenging conditions. Our exploration of model explainability revealed that stochastic models generate more diverse and personalized explanations thereby highlighting the need for AI models that provide detailed and individualized insights in real-world clinical settings. Furthermore, we underscored the importance of quantifying uncertainty in AI models which enables clinicians to make better-informed decisions based on reliable predictions. Our study advocates for prioritizing implementation science in AI research for healthcare and ensuring that AI solutions are practical, beneficial, and sustainable in real-world clinical environments. By addressing unique challenges and complexities in healthcare settings, researchers can develop AI models that effectively improve clinical practice and patient outcomes.

{{</citation>}}


### (82/145) An Investigation of Time Reversal Symmetry in Reinforcement Learning (Brett Barkley et al., 2023)

{{<citation>}}

Brett Barkley, Amy Zhang, David Fridovich-Keil. (2023)  
**An Investigation of Time Reversal Symmetry in Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.17008v1)  

---


**ABSTRACT**  
One of the fundamental challenges associated with reinforcement learning (RL) is that collecting sufficient data can be both time-consuming and expensive. In this paper, we formalize a concept of time reversal symmetry in a Markov decision process (MDP), which builds upon the established structure of dynamically reversible Markov chains (DRMCs) and time-reversibility in classical physics. Specifically, we investigate the utility of this concept in reducing the sample complexity of reinforcement learning. We observe that utilizing the structure of time reversal in an MDP allows every environment transition experienced by an agent to be transformed into a feasible reverse-time transition, effectively doubling the number of experiences in the environment. To test the usefulness of this newly synthesized data, we develop a novel approach called time symmetric data augmentation (TSDA) and investigate its application in both proprioceptive and pixel-based state within the realm of off-policy, model-free RL. Empirical evaluations showcase how these synthetic transitions can enhance the sample efficiency of RL agents in time reversible scenarios without friction or contact. We also test this method in more realistic environments where these assumptions are not globally satisfied. We find that TSDA can significantly degrade sample efficiency and policy performance, but can also improve sample efficiency under the right conditions. Ultimately we conclude that time symmetry shows promise in enhancing the sample efficiency of reinforcement learning and provide guidance when the environment and reward structures are of an appropriate form for TSDA to be employed effectively.

{{</citation>}}


### (83/145) Debiasing Multimodal Models via Causal Information Minimization (Vaidehi Patil et al., 2023)

{{<citation>}}

Vaidehi Patil, Adyasha Maharana, Mohit Bansal. (2023)  
**Debiasing Multimodal Models via Causal Information Minimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.LG, stat-ME  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.16941v1)  

---


**ABSTRACT**  
Most existing debiasing methods for multimodal models, including causal intervention and inference methods, utilize approximate heuristics to represent the biases, such as shallow features from early stages of training or unimodal features for multimodal tasks like VQA, etc., which may not be accurate. In this paper, we study bias arising from confounders in a causal graph for multimodal data and examine a novel approach that leverages causally-motivated information minimization to learn the confounder representations. Robust predictive features contain diverse information that helps a model generalize to out-of-distribution data. Hence, minimizing the information content of features obtained from a pretrained biased model helps learn the simplest predictive features that capture the underlying data distribution. We treat these features as confounder representations and use them via methods motivated by causal theory to remove bias from models. We find that the learned confounder representations indeed capture dataset biases, and the proposed debiasing methods improve out-of-distribution (OOD) performance on multiple multimodal datasets without sacrificing in-distribution performance. Additionally, we introduce a novel metric to quantify the sufficiency of spurious features in models' predictions that further demonstrates the effectiveness of our proposed methods. Our code is available at: https://github.com/Vaidehi99/CausalInfoMin

{{</citation>}}


### (84/145) Compressing the Backward Pass of Large-Scale Neural Architectures by Structured Activation Pruning (Daniel Barley et al., 2023)

{{<citation>}}

Daniel Barley, Holger Fröning. (2023)  
**Compressing the Backward Pass of Large-Scale Neural Architectures by Structured Activation Pruning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-PF, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.16883v2)  

---


**ABSTRACT**  
The rise of Deep Neural Networks (DNNs) has led to an increase in model size and complexity, straining the memory capacity of GPUs. Sparsity in DNNs, characterized as structural or ephemeral, has gained attention as a solution. This work focuses on ephemeral sparsity, aiming to reduce memory consumption during training. It emphasizes the significance of activations, an often overlooked component, and their role in memory usage. This work employs structured pruning in Block Sparse Compressed Row (BSR) format in combination with a magnitude-based criterion to efficiently prune activations. We furthermore introduce efficient block-sparse operators for GPUs and showcase their effectiveness, as well as the superior compression offered by block sparsity. We report the effectiveness of activation pruning by evaluating training speed, accuracy, and memory usage of large-scale neural architectures on the example of ResMLP on image classification tasks. As a result, we observe a memory reduction of up to 32% while maintaining accuracy. Ultimately, our approach aims to democratize large-scale model training, reduce GPU requirements, and address ecological concerns.

{{</citation>}}


### (85/145) Power Hungry Processing: Watts Driving the Cost of AI Deployment? (Alexandra Sasha Luccioni et al., 2023)

{{<citation>}}

Alexandra Sasha Luccioni, Yacine Jernite, Emma Strubell. (2023)  
**Power Hungry Processing: Watts Driving the Cost of AI Deployment?**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16863v1)  

---


**ABSTRACT**  
Recent years have seen a surge in the popularity of commercial AI products based on generative, multi-purpose AI systems promising a unified approach to building machine learning (ML) models into technology. However, this ambition of "generality" comes at a steep cost to the environment, given the amount of energy these systems require and the amount of carbon that they emit. In this work, we propose the first systematic comparison of the ongoing inference cost of various categories of ML systems, covering both task-specific (i.e. finetuned models that carry out a single task) and `general-purpose' models, (i.e. those trained for multiple tasks). We measure deployment cost as the amount of energy and carbon required to perform 1,000 inferences on representative benchmark dataset using these models. We find that multi-purpose, generative architectures are orders of magnitude more expensive than task-specific systems for a variety of tasks, even when controlling for the number of model parameters. We conclude with a discussion around the current trend of deploying multi-purpose generative ML systems, and caution that their utility should be more intentionally weighed against increased costs in terms of energy and emissions. All the data from our study can be accessed via an interactive demo to carry out further exploration and analysis.

{{</citation>}}


### (86/145) Attentional Graph Neural Networks for Robust Massive Network Localization (Wenzhong Yan et al., 2023)

{{<citation>}}

Wenzhong Yan, Juntao Wang, Feng Yin, Abdelhak M. Zoubir. (2023)  
**Attentional Graph Neural Networks for Robust Massive Network Localization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP, stat-ML  
Keywords: Attention, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.16856v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have gained significant popularity for classification tasks in machine learning, yet their applications to regression problems remain limited. Concurrently, attention mechanisms have emerged as powerful tools in sequential learning tasks. In this paper, we employ GNNs and attention mechanisms to address a classical but challenging nonlinear regression problem: network localization. We propose a novel GNN-based network localization method that achieves exceptional stability and accuracy in the presence of severe non-line-of-sight (NLOS) propagations, while eliminating the need for laborious offline calibration or NLOS identification. Extensive experimental results validate the effectiveness and high accuracy of our GNN-based localization model, particularly in challenging NLOS scenarios. However, the proposed GNN-based model exhibits limited flexibility, and its accuracy is highly sensitive to a specific hyperparameter that determines the graph structure. To address the limitations and extend the applicability of the GNN-based model to real scenarios, we introduce two attentional graph neural networks (AGNNs) that offer enhanced flexibility and the ability to automatically learn the optimal hyperparameter for each node. Experimental results confirm that the AGNN models are able to enhance localization accuracy, providing a promising solution for real-world applications. We also provide some analyses of the improved performance achieved by the AGNN models from the perspectives of dynamic attention and signal denoising characteristics.

{{</citation>}}


### (87/145) Modular Neural Networks for Time Series Forecasting: Interpretability and Feature Selection using Attention (Qiqi Su et al., 2023)

{{<citation>}}

Qiqi Su, Christos Kloukinas, Artur d'Avila Garcez. (2023)  
**Modular Neural Networks for Time Series Forecasting: Interpretability and Feature Selection using Attention**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2311.16834v2)  

---


**ABSTRACT**  
Multivariate time series have many applications, from healthcare and meteorology to life science. Although deep learning models have shown excellent predictive performance for time series, they have been criticised for being "black-boxes" or non-interpretable. This paper proposes a novel modular neural network model for multivariate time series prediction that is interpretable by construction. A recurrent neural network learns the temporal dependencies in the data while an attention-based feature selection component selects the most relevant features and suppresses redundant features used in the learning of the temporal dependencies. A modular deep network is trained from the selected features independently to show the users how features influence outcomes, making the model interpretable. Experimental results show that this approach can outperform state-of-the-art interpretable Neural Additive Models (NAM) and variations thereof in both regression and classification of time series tasks, achieving a predictive performance that is comparable to the top non-interpretable methods for time series, LSTM and XGBoost.

{{</citation>}}


### (88/145) Large Language Models Suffer From Their Own Output: An Analysis of the Self-Consuming Training Loop (Martin Briesch et al., 2023)

{{<citation>}}

Martin Briesch, Dominik Sobania, Franz Rothlauf. (2023)  
**Large Language Models Suffer From Their Own Output: An Analysis of the Self-Consuming Training Loop**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs-NE, cs.LG  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16822v1)  

---


**ABSTRACT**  
Large language models (LLM) have become state of the art in many benchmarks and conversational LLM applications like ChatGPT are now widely used by the public. Those LLMs can be used to generate large amounts of content which is posted on the internet to various platforms. As LLMs are trained on datasets usually collected from the internet, this LLM-generated content might be used to train the next generation of LLMs. Therefore, a self-consuming training loop emerges in which new LLM generations are trained on the output from the previous generations. We empirically study this self-consuming training loop using a novel dataset to analytically and accurately measure quality and diversity of generated outputs. We find that this self-consuming training loop initially improves both quality and diversity. However, after a few generations the output inevitably degenerates in diversity. We find that the rate of degeneration depends on the proportion of real and generated data.

{{</citation>}}


### (89/145) XAI for time-series classification leveraging image highlight methods (Georgios Makridis et al., 2023)

{{<citation>}}

Georgios Makridis, Georgios Fatouros, Vasileios Koukos, Dimitrios Kotios, Dimosthenis Kyriazis, Ioannis Soldatos. (2023)  
**XAI for time-series classification leveraging image highlight methods**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2311.17110v1)  

---


**ABSTRACT**  
Although much work has been done on explainability in the computer vision and natural language processing (NLP) fields, there is still much work to be done to explain methods applied to time series as time series by nature can not be understood at first sight. In this paper, we present a Deep Neural Network (DNN) in a teacher-student architecture (distillation model) that offers interpretability in time-series classification tasks. The explainability of our approach is based on transforming the time series to 2D plots and applying image highlight methods (such as LIME and GradCam), making the predictions interpretable. At the same time, the proposed approach offers increased accuracy competing with the baseline model with the trade-off of increasing the training time.

{{</citation>}}


### (90/145) PyTorch Geometric High Order: A Unified Library for High Order Graph Neural Network (Xiyuan Wang et al., 2023)

{{<citation>}}

Xiyuan Wang, Muhan Zhang. (2023)  
**PyTorch Geometric High Order: A Unified Library for High Order Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.16670v1)  

---


**ABSTRACT**  
We introduce PyTorch Geometric High Order (PyGHO), a library for High Order Graph Neural Networks (HOGNNs) that extends PyTorch Geometric (PyG). Unlike ordinary Message Passing Neural Networks (MPNNs) that exchange messages between nodes, HOGNNs, encompassing subgraph GNNs and k-WL GNNs, encode node tuples, a method previously lacking a standardized framework and often requiring complex coding. PyGHO's main objective is to provide an unified and user-friendly interface for various HOGNNs. It accomplishes this through streamlined data structures for node tuples, comprehensive data processing utilities, and a flexible suite of operators for high-order GNN methodologies. In this work, we present a detailed in-depth of PyGHO and compare HOGNNs implemented with PyGHO with their official implementation on real-world tasks. PyGHO achieves up to $50\%$ acceleration and reduces the code needed for implementation by an order of magnitude. Our library is available at \url{https://github.com/GraphPKU/PygHO}.

{{</citation>}}


### (91/145) MultiModal-Learning for Predicting Molecular Properties: A Framework Based on Image and Graph Structures (Zhuoyuan Wang et al., 2023)

{{<citation>}}

Zhuoyuan Wang, Jiacong Mi, Shan Lu, Jieyue He. (2023)  
**MultiModal-Learning for Predicting Molecular Properties: A Framework Based on Image and Graph Structures**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, physics-chem-ph, q-bio-BM  
Keywords: AI, GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.16666v1)  

---


**ABSTRACT**  
The quest for accurate prediction of drug molecule properties poses a fundamental challenge in the realm of Artificial Intelligence Drug Discovery (AIDD). An effective representation of drug molecules emerges as a pivotal component in this pursuit. Contemporary leading-edge research predominantly resorts to self-supervised learning (SSL) techniques to extract meaningful structural representations from large-scale, unlabeled molecular data, subsequently fine-tuning these representations for an array of downstream tasks. However, an inherent shortcoming of these studies lies in their singular reliance on one modality of molecular information, such as molecule image or SMILES representations, thus neglecting the potential complementarity of various molecular modalities. In response to this limitation, we propose MolIG, a novel MultiModaL molecular pre-training framework for predicting molecular properties based on Image and Graph structures. MolIG model innovatively leverages the coherence and correlation between molecule graph and molecule image to execute self-supervised tasks, effectively amalgamating the strengths of both molecular representation forms. This holistic approach allows for the capture of pivotal molecular structural characteristics and high-level semantic information. Upon completion of pre-training, Graph Neural Network (GNN) Encoder is used for the prediction of downstream tasks. In comparison to advanced baseline models, MolIG exhibits enhanced performance in downstream tasks pertaining to molecular property prediction within benchmark groups such as MoleculeNet Benchmark Group and ADMET Benchmark Group.

{{</citation>}}


### (92/145) ClimateX: Do LLMs Accurately Assess Human Expert Confidence in Climate Statements? (Romain Lacombe et al., 2023)

{{<citation>}}

Romain Lacombe, Kerrie Wu, Eddie Dilworth. (2023)  
**ClimateX: Do LLMs Accurately Assess Human Expert Confidence in Climate Statements?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CY, cs-IR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.17107v1)  

---


**ABSTRACT**  
Evaluating the accuracy of outputs generated by Large Language Models (LLMs) is especially important in the climate science and policy domain. We introduce the Expert Confidence in Climate Statements (ClimateX) dataset, a novel, curated, expert-labeled dataset consisting of 8094 climate statements collected from the latest Intergovernmental Panel on Climate Change (IPCC) reports, labeled with their associated confidence levels. Using this dataset, we show that recent LLMs can classify human expert confidence in climate-related statements, especially in a few-shot learning setting, but with limited (up to 47%) accuracy. Overall, models exhibit consistent and significant over-confidence on low and medium confidence statements. We highlight implications of our results for climate communication, LLMs evaluation strategies, and the use of LLMs in information retrieval systems.

{{</citation>}}


### (93/145) Elucidating Discrepancy in Explanations of Predictive Models Developed using EMR (Aida Brankovic et al., 2023)

{{<citation>}}

Aida Brankovic, Wenjie Huang, David Cook, Sankalp Khanna, Konstanty Bialkowski. (2023)  
**Elucidating Discrepancy in Explanations of Predictive Models Developed using EMR**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16654v1)  

---


**ABSTRACT**  
The lack of transparency and explainability hinders the clinical adoption of Machine learning (ML) algorithms. While explainable artificial intelligence (XAI) methods have been proposed, little research has focused on the agreement between these methods and expert clinical knowledge. This study applies current state-of-the-art explainability methods to clinical decision support algorithms developed for Electronic Medical Records (EMR) data to analyse the concordance between these factors and discusses causes for identified discrepancies from a clinical and technical perspective. Important factors for achieving trustworthy XAI solutions for clinical decision support are also discussed.

{{</citation>}}


### (94/145) On the Long Range Abilities of Transformers (Itamar Zimerman et al., 2023)

{{<citation>}}

Itamar Zimerman, Lior Wolf. (2023)  
**On the Long Range Abilities of Transformers**  

---
Primary Category: cs.LG  
Categories: F-2-2; I-2-7, cs-CL, cs-LG, cs.LG  
Keywords: NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.16620v1)  

---


**ABSTRACT**  
Despite their dominance in modern DL and, especially, NLP domains, transformer architectures exhibit sub-optimal performance on long-range tasks compared to recent layers that are specifically designed for this purpose. In this work, drawing inspiration from key attributes of long-range layers, such as state-space layers, linear RNN layers, and global convolution layers, we demonstrate that minimal modifications to the transformer architecture can significantly enhance performance on the Long Range Arena (LRA) benchmark, thus narrowing the gap with these specialized layers. We identify that two key principles for long-range tasks are (i) incorporating an inductive bias towards smoothness, and (ii) locality. As we show, integrating these ideas into the attention mechanism improves results with a negligible amount of additional computation and without any additional trainable parameters. Our theory and experiments also shed light on the reasons for the inferior performance of transformers on long-range tasks and identify critical properties that are essential for successfully capturing long-range dependencies.

{{</citation>}}


### (95/145) Adversarial Distribution Balancing for Counterfactual Reasoning (Stefan Schrod et al., 2023)

{{<citation>}}

Stefan Schrod, Fabian Sinz, Michael Altenbuchinger. (2023)  
**Adversarial Distribution Balancing for Counterfactual Reasoning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2311.16616v1)  

---


**ABSTRACT**  
The development of causal prediction models is challenged by the fact that the outcome is only observable for the applied (factual) intervention and not for its alternatives (the so-called counterfactuals); in medicine we only know patients' survival for the administered drug and not for other therapeutic options. Machine learning approaches for counterfactual reasoning have to deal with both unobserved outcomes and distributional differences due to non-random treatment administration. Unsupervised domain adaptation (UDA) addresses similar issues; one has to deal with unobserved outcomes -- the labels of the target domain -- and distributional differences between source and target domain. We propose Adversarial Distribution Balancing for Counterfactual Reasoning (ADBCR), which directly uses potential outcome estimates of the counterfactuals to remove spurious causal relations. We show that ADBCR outcompetes state-of-the-art methods on three benchmark datasets, and demonstrate that ADBCR's performance can be further improved if unlabeled validation data are included in the training procedure to better adapt the model to the validation domain.

{{</citation>}}


### (96/145) LasTGL: An Industrial Framework for Large-Scale Temporal Graph Learning (Jintang Li et al., 2023)

{{<citation>}}

Jintang Li, Jiawang Dan, Ruofan Wu, Jing Zhou, Sheng Tian, Yunfei Liu, Baokun Wang, Changhua Meng, Weiqiang Wang, Yuchang Zhu, Liang Chen, Zibin Zheng. (2023)  
**LasTGL: An Industrial Framework for Large-Scale Temporal Graph Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.16605v2)  

---


**ABSTRACT**  
Over the past few years, graph neural networks (GNNs) have become powerful and practical tools for learning on (static) graph-structure data. However, many real-world applications, such as social networks and e-commerce, involve temporal graphs where nodes and edges are dynamically evolving. Temporal graph neural networks (TGNNs) have progressively emerged as an extension of GNNs to address time-evolving graphs and have gradually become a trending research topic in both academics and industry. Advancing research and application in such an emerging field necessitates the development of new tools to compose TGNN models and unify their different schemes for dealing with temporal graphs. In this work, we introduce LasTGL, an industrial framework that integrates unified and extensible implementations of common temporal graph learning algorithms for various advanced tasks. The purpose of LasTGL is to provide the essential building blocks for solving temporal graph learning tasks, focusing on the guiding principles of user-friendliness and quick prototyping on which PyTorch is based. In particular, LasTGL provides comprehensive temporal graph datasets, TGNN models and utilities along with well-documented tutorials, making it suitable for both absolute beginners and expert deep learning practitioners alike.

{{</citation>}}


### (97/145) FedAL: Black-Box Federated Knowledge Distillation Enabled by Adversarial Learning (Pengchao Han et al., 2023)

{{<citation>}}

Pengchao Han, Xingyan Shi, Jianwei Huang. (2023)  
**FedAL: Black-Box Federated Knowledge Distillation Enabled by Adversarial Learning**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.16584v1)  

---


**ABSTRACT**  
Knowledge distillation (KD) can enable collaborative learning among distributed clients that have different model architectures and do not share their local data and model parameters with others. Each client updates its local model using the average model output/feature of all client models as the target, known as federated KD. However, existing federated KD methods often do not perform well when clients' local models are trained with heterogeneous local datasets. In this paper, we propose Federated knowledge distillation enabled by Adversarial Learning (FedAL) to address the data heterogeneity among clients. First, to alleviate the local model output divergence across clients caused by data heterogeneity, the server acts as a discriminator to guide clients' local model training to achieve consensus model outputs among clients through a min-max game between clients and the discriminator. Moreover, catastrophic forgetting may happen during the clients' local training and global knowledge transfer due to clients' heterogeneous local data. Towards this challenge, we design the less-forgetting regularization for both local training and global knowledge transfer to guarantee clients' ability to transfer/learn knowledge to/from others. Experimental results show that FedAL and its variants achieve higher accuracy than other federated KD baselines.

{{</citation>}}


### (98/145) Anonymous Jamming Detection in 5G with Bayesian Network Model Based Inference Analysis (Ying Wang et al., 2023)

{{<citation>}}

Ying Wang, Shashank Jere, Soumya Banerjee, Lingjia Liu, Sachin Shetty, Shehadi Dayekh. (2023)  
**Anonymous Jamming Detection in 5G with Bayesian Network Model Based Inference Analysis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs-NI, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.17097v1)  

---


**ABSTRACT**  
Jamming and intrusion detection are critical in 5G research, aiming to maintain reliability, prevent user experience degradation, and avoid infrastructure failure. This paper introduces an anonymous jamming detection model for 5G based on signal parameters from the protocol stacks. The system uses supervised and unsupervised learning for real-time, high-accuracy detection of jamming, including unknown types. Supervised models reach an AUC of 0.964 to 1, compared to LSTM models with an AUC of 0.923 to 1. However, the need for data annotation limits the supervised approach. To address this, an unsupervised auto-encoder-based anomaly detection is presented with an AUC of 0.987. The approach is resistant to adversarial training samples. For transparency and domain knowledge injection, a Bayesian network-based causation analysis is introduced.

{{</citation>}}


### (99/145) Evaluation of dynamic characteristics of power grid based on GNN and application on knowledge graph (Hao Pei et al., 2023)

{{<citation>}}

Hao Pei, Si Lin, Chuanfu Li, Che Wang, Haoming Chen, Sizhe Li. (2023)  
**Evaluation of dynamic characteristics of power grid based on GNN and application on knowledge graph**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, eess-SP  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.16522v1)  

---


**ABSTRACT**  
A novel method for detecting faults in power grids using a graph neural network (GNN) has been developed, aimed at enhancing intelligent fault diagnosis in network operation and maintenance. This GNN-based approach identifies faulty nodes within the power grid through a specialized electrical feature extraction model coupled with a knowledge graph. Incorporating temporal data, the method leverages the status of nodes from preceding and subsequent time periods to aid in current fault detection. To validate the effectiveness of this GNN in extracting node features, a correlation analysis of the output features from each node within the neural network layer was conducted. The results from experiments show that this method can accurately locate fault nodes in simulated scenarios with a remarkable 99.53% accuracy. Additionally, the graph neural network's feature modeling allows for a qualitative examination of how faults spread across nodes, providing valuable insights for analyzing fault nodes.

{{</citation>}}


### (100/145) B-LSTM-MIONet: Bayesian LSTM-based Neural Operators for Learning the Response of Complex Dynamical Systems to Length-Variant Multiple Input Functions (Zhihao Kong et al., 2023)

{{<citation>}}

Zhihao Kong, Amirhossein Mollaali, Christian Moya, Na Lu, Guang Lin. (2023)  
**B-LSTM-MIONet: Bayesian LSTM-based Neural Operators for Learning the Response of Complex Dynamical Systems to Length-Variant Multiple Input Functions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NA, cs.LG, math-NA  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.16519v2)  

---


**ABSTRACT**  
Deep Operator Network (DeepONet) is a neural network framework for learning nonlinear operators such as those from ordinary differential equations (ODEs) describing complex systems. Multiple-input deep neural operators (MIONet) extended DeepONet to allow multiple input functions in different Banach spaces. MIONet offers flexibility in training dataset grid spacing, without constraints on output location. However, it requires offline inputs and cannot handle varying sequence lengths in testing datasets, limiting its real-time application in dynamic complex systems. This work redesigns MIONet, integrating Long Short Term Memory (LSTM) to learn neural operators from time-dependent data. This approach overcomes data discretization constraints and harnesses LSTM's capability with variable-length, real-time data. Factors affecting learning performance, like algorithm extrapolation ability are presented. The framework is enhanced with uncertainty quantification through a novel Bayesian method, sampling from MIONet parameter distributions. Consequently, we develop the B-LSTM-MIONet, incorporating LSTM's temporal strengths with Bayesian robustness, resulting in a more precise and reliable model for noisy datasets.

{{</citation>}}


### (101/145) Enabling Fast 2-bit LLM on GPUs: Memory Alignment, Sparse Outlier, and Asynchronous Dequantization (Jinhao Li et al., 2023)

{{<citation>}}

Jinhao Li, Shiyao Li, Jiaming Xu, Shan Huang, Yaoxiu Lian, Jun Liu, Yu Wang, Guohao Dai. (2023)  
**Enabling Fast 2-bit LLM on GPUs: Memory Alignment, Sparse Outlier, and Asynchronous Dequantization**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.16442v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated impressive abilities in various domains while the inference cost is expensive. The state-of-the-art methods use 2-bit quantization for mainstream LLMs. However, challenges still exist: (1) Nonnegligible accuracy loss for 2-bit quantization. Weights are quantized by groups, while the ranges of weights are large in some groups, resulting in large quantization errors and nonnegligible accuracy loss (e.g. >3% for Llama2-7b with 2-bit quantization in GPTQ and Greenbit). (2) Limited accuracy improvement by adding 4-bit weights. Increasing 10% extra average bit more 4-bit weights only leads to <0.5% accuracy improvement on a quantized Llama2-7b. (3) Time-consuming dequantization operations on GPUs. The dequantization operations lead to >50% execution time, hindering the potential of reducing LLM inference cost. To tackle these challenges, we propose the following techniques: (1) We only quantize a small fraction of groups with the larger range using 4-bit with memory alignment consideration on GPUs. (2) We point out that the distribution of the sparse outliers with larger weights is different in 2-bit and 4-bit groups, and only a small fraction of outliers require 16-bit quantization. Such design leads to >0.5% accuracy improvement with <3% average increased bit for Llama2-7b. (3) We design the asynchronous dequantization on GPUs, leading to up to 3.92X speedup. We conduct extensive experiments on different model families and model sizes. We achieve 2.85-bit for each weight and the end-to-end speedup for Llama2-7b is 1.74X over the original model, and we reduce both runtime cost and hardware cost by up to 2.70X and 2.81X with less GPU requirements.

{{</citation>}}


### (102/145) Model-free Test Time Adaptation for Out-Of-Distribution Detection (YiFan Zhang et al., 2023)

{{<citation>}}

YiFan Zhang, Xue Wang, Tian Zhou, Kun Yuan, Zhang Zhang, Liang Wang, Rong Jin, Tieniu Tan. (2023)  
**Model-free Test Time Adaptation for Out-Of-Distribution Detection**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.16420v1)  

---


**ABSTRACT**  
Out-of-distribution (OOD) detection is essential for the reliability of ML models. Most existing methods for OOD detection learn a fixed decision criterion from a given in-distribution dataset and apply it universally to decide if a data point is OOD. Recent work~\cite{fang2022is} shows that given only in-distribution data, it is impossible to reliably detect OOD data without extra assumptions. Motivated by the theoretical result and recent exploration of test-time adaptation methods, we propose a Non-Parametric Test Time \textbf{Ada}ptation framework for \textbf{O}ut-Of-\textbf{D}istribution \textbf{D}etection (\abbr). Unlike conventional methods, \abbr utilizes online test samples for model adaptation during testing, enhancing adaptability to changing data distributions. The framework incorporates detected OOD instances into decision-making, reducing false positive rates, particularly when ID and OOD distributions overlap significantly. We demonstrate the effectiveness of \abbr through comprehensive experiments on multiple OOD detection benchmarks, extensive empirical studies show that \abbr significantly improves the performance of OOD detection over state-of-the-art methods. Specifically, \abbr reduces the false positive rate (FPR95) by $23.23\%$ on the CIFAR-10 benchmarks and $38\%$ on the ImageNet-1k benchmarks compared to the advanced methods. Lastly, we theoretically verify the effectiveness of \abbr.

{{</citation>}}


### (103/145) Deep Learning for Time Series Classification of Parkinson's Disease Eye Tracking Data (Gonzalo Uribarri et al., 2023)

{{<citation>}}

Gonzalo Uribarri, Simon Ekman von Huth, Josefine Waldthaler, Per Svenningsson, Erik Fransén. (2023)  
**Deep Learning for Time Series Classification of Parkinson's Disease Eye Tracking Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.16381v1)  

---


**ABSTRACT**  
Eye-tracking is an accessible and non-invasive technology that provides information about a subject's motor and cognitive abilities. As such, it has proven to be a valuable resource in the study of neurodegenerative diseases such as Parkinson's disease. Saccade experiments, in particular, have proven useful in the diagnosis and staging of Parkinson's disease. However, to date, no single eye-movement biomarker has been found to conclusively differentiate patients from healthy controls. In the present work, we investigate the use of state-of-the-art deep learning algorithms to perform Parkinson's disease classification using eye-tracking data from saccade experiments. In contrast to previous work, instead of using hand-crafted features from the saccades, we use raw $\sim1.5\,s$ long fixation intervals recorded during the preparatory phase before each trial. Using these short time series as input we implement two different classification models, InceptionTime and ROCKET. We find that the models are able to learn the classification task and generalize to unseen subjects. InceptionTime achieves $78\%$ accuracy, while ROCKET achieves $88\%$ accuracy. We also employ a novel method for pruning the ROCKET model to improve interpretability and generalizability, achieving an accuracy of $96\%$. Our results suggest that fixation data has low inter-subject variability and potentially carries useful information about brain cognitive and motor conditions, making it suitable for use with machine learning in the discovery of disease-relevant biomarkers.

{{</citation>}}


## eess.IV (2)



### (104/145) SubZero: Subspace Zero-Shot MRI Reconstruction (Heng Yu et al., 2023)

{{<citation>}}

Heng Yu, Yamin Arefeen, Berkin Bilgic. (2023)  
**SubZero: Subspace Zero-Shot MRI Reconstruction**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.17251v1)  

---


**ABSTRACT**  
Recently introduced zero-shot self-supervised learning (ZS-SSL) has shown potential in accelerated MRI in a scan-specific scenario, which enabled high-quality reconstructions without access to a large training dataset. ZS-SSL has been further combined with the subspace model to accelerate 2D T2-shuffling acquisitions. In this work, we propose a parallel network framework and introduce an attention mechanism to improve subspace-based zero-shot self-supervised learning and enable higher acceleration factors. We name our method SubZero and demonstrate that it can achieve improved performance compared with current methods in T1 and T2 mapping acquisitions.

{{</citation>}}


### (105/145) TopoSemiSeg: Enforcing Topological Consistency for Semi-Supervised Segmentation of Histopathology Images (Meilong Xu et al., 2023)

{{<citation>}}

Meilong Xu, Xiaoling Hu, Saumya Gupta, Shahira Abousamra, Chao Chen. (2023)  
**TopoSemiSeg: Enforcing Topological Consistency for Semi-Supervised Segmentation of Histopathology Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.16447v1)  

---


**ABSTRACT**  
In computational pathology, segmenting densely distributed objects like glands and nuclei is crucial for downstream analysis. To alleviate the burden of obtaining pixel-wise annotations, semi-supervised learning methods learn from large amounts of unlabeled data. Nevertheless, existing semi-supervised methods overlook the topological information hidden in the unlabeled images and are thus prone to topological errors, e.g., missing or incorrectly merged/separated glands or nuclei. To address this issue, we propose TopoSemiSeg, the first semi-supervised method that learns the topological representation from unlabeled data. In particular, we propose a topology-aware teacher-student approach in which the teacher and student networks learn shared topological representations. To achieve this, we introduce topological consistency loss, which contains signal consistency and noise removal losses to ensure the learned representation is robust and focuses on true topological signals. Extensive experiments on public pathology image datasets show the superiority of our method, especially on topology-wise evaluation metrics. Code is available at https://github.com/Melon-Xu/TopoSemiSeg.

{{</citation>}}


## cs.CY (6)



### (106/145) Survey on AI Ethics: A Socio-technical Perspective (Dave Mbiazi et al., 2023)

{{<citation>}}

Dave Mbiazi, Meghana Bhange, Maryam Babaei, Ivaxi Sheth, Patrik Joslin Kenfack. (2023)  
**Survey on AI Ethics: A Socio-technical Perspective**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17228v1)  

---


**ABSTRACT**  
The past decade has observed a great advancement in AI with deep learning-based models being deployed in diverse scenarios including safety-critical applications. As these AI systems become deeply embedded in our societal infrastructure, the repercussions of their decisions and actions have significant consequences, making the ethical implications of AI deployment highly relevant and important. The ethical concerns associated with AI are multifaceted, including challenging issues of fairness, privacy and data protection, responsibility and accountability, safety and robustness, transparency and explainability, and environmental impact. These principles together form the foundations of ethical AI considerations that concern every stakeholder in the AI system lifecycle. In light of the present ethical and future x-risk concerns, governments have shown increasing interest in establishing guidelines for the ethical deployment of AI. This work unifies the current and future ethical concerns of deploying AI into society. While we acknowledge and appreciate the technical surveys for each of the ethical principles concerned, in this paper, we aim to provide a comprehensive overview that not only addresses each principle from a technical point of view but also discusses them from a social perspective.

{{</citation>}}


### (107/145) Foundational Moral Values for AI Alignment (Betty Li Hou et al., 2023)

{{<citation>}}

Betty Li Hou, Brian Patrick Green. (2023)  
**Foundational Moral Values for AI Alignment**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17017v1)  

---


**ABSTRACT**  
Solving the AI alignment problem requires having clear, defensible values towards which AI systems can align. Currently, targets for alignment remain underspecified and do not seem to be built from a philosophically robust structure. We begin the discussion of this problem by presenting five core, foundational values, drawn from moral philosophy and built on the requisites for human existence: survival, sustainable intergenerational existence, society, education, and truth. We show that these values not only provide a clearer direction for technical alignment work, but also serve as a framework to highlight threats and opportunities from AI systems to both obtain and sustain these values.

{{</citation>}}


### (108/145) Counter-terrorism in cyber-physical spaces: Best practices and technologies from the state of the art (Giuseppe Cascavilla et al., 2023)

{{<citation>}}

Giuseppe Cascavilla, Damian A. Tamburri, Francesco Leotta, Massimo Mecella, WillemJan Van Den Heuvel. (2023)  
**Counter-terrorism in cyber-physical spaces: Best practices and technologies from the state of the art**  

---
Primary Category: cs.CY  
Categories: cs-CR, cs-CY, cs-IT, cs-SE, cs.CY, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17012v1)  

---


**ABSTRACT**  
Context: The demand for protection and security of physical spaces and urban areas increased with the escalation of terroristic attacks in recent years. We envision with the proposed cyber-physical systems and spaces, a city that would indeed become a smarter urbanistic object, proactively providing alerts and being protective against any threat. Objectives: This survey intend to provide a systematic multivocal literature survey comprised of an updated, comprehensive and timely overview of state of the art in counter-terrorism cyber-physical systems, hence aimed at the protection of cyber-physical spaces. Hence, provide guidelines to law enforcement agencies and practitioners providing a description of technologies and best practices for the protection of public spaces. Methods: We analyzed 112 papers collected from different online sources, both from the academic field and from websites and blogs ranging from 2004 till mid-2022. Results: a) There is no one single bullet-proof solution available for the protection of public spaces. b) From our analysis we found three major active fields for the protection of public spaces: Information Technologies, Architectural approaches, Organizational field. c) While the academic suggest best practices and methodologies for the protection of urban areas, the market did not provide any type of implementation of such suggested approaches, which shows a lack of fertilization between academia and industry. Conclusion: The overall analysis has led us to state that there is no one single solution available, conversely, multiple methods and techniques can be put in place to guarantee safety and security in public spaces. The techniques range from architectural design to rethink the design of public spaces keeping security into account in continuity, to emerging technologies such as AI and predictive surveillance.

{{</citation>}}


### (109/145) Analyzing the Influence of Language Model-Generated Responses in Mitigating Hate Speech on Social Media Directed at Ukrainian Refugees in Poland (Jakub Podolak et al., 2023)

{{<citation>}}

Jakub Podolak, Szymon Łukasik, Paweł Balawender, Jan Ossowski, Katarzyna Bąkowicz, Piotr Sankowski. (2023)  
**Analyzing the Influence of Language Model-Generated Responses in Mitigating Hate Speech on Social Media Directed at Ukrainian Refugees in Poland**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Language Model, Social Media  
[Paper Link](http://arxiv.org/abs/2311.16905v1)  

---


**ABSTRACT**  
In the context of escalating hate speech and polarization on social media, this study investigates the potential of employing responses generated by Large Language Models (LLM), complemented with pertinent verified knowledge links, to counteract such trends. Through extensive A/B testing involving the posting of 753 automatically generated responses, the goal was to minimize the propagation of hate speech directed at Ukrainian refugees in Poland.   The results indicate that deploying LLM-generated responses as replies to harmful tweets effectively diminishes user engagement, as measured by likes/impressions. When we respond to an original tweet, i.e., which is not a reply, we reduce the engagement of users by over 20\% without increasing the number of impressions. On the other hand, our responses increase the ratio of the number of replies to a harmful tweet to impressions, especially if the harmful tweet is not original. Additionally, the study examines how generated responses influence the overall sentiment of tweets in the discussion, revealing that our intervention does not significantly alter the mean sentiment.   This paper suggests the implementation of an automatic moderation system to combat hate speech on social media and provides an in-depth analysis of the A/B experiment, covering methodology, data collection, and statistical outcomes. Ethical considerations and challenges are also discussed, offering guidance for the development of discourse moderation systems leveraging the capabilities of generative AI.

{{</citation>}}


### (110/145) Tracking a Year of Polarized Twitter Discourse on Abortion (Ashwin Rao et al., 2023)

{{<citation>}}

Ashwin Rao, Rong-Ching Chang, Qiankun Zhong, Kristina Lerman, Magdalena Wojcieszak. (2023)  
**Tracking a Year of Polarized Twitter Discourse on Abortion**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2311.16831v1)  

---


**ABSTRACT**  
Abortion is one of the most contentious issues in American politics. The Dobbs v. Jackson Women's Health Organization ruling in 2022, which shifted the authority to regulate abortion from the federal government to the states, triggering intense protests and emotional debates across the nation. Yet, little is known about how online discourse about abortion rights fluctuated on social media platforms. This study analyzes a corpus of over 57M abortion-related tweets from January 2022 to January 2023 to show how emotions, hateful rhetoric, toxic speech, use of obscenities and insults, and also framing strategies fluctuated over the span of one year among liberal and conservative users. We offer three key findings. (1) Fluctuations in emotions were temporary; key events during the analyzed period did not bring about lasting shifts in expressed emotions. (2) We observe significant ideological differences in the use of hate speech: conservatives resorted to hateful rhetoric more than liberals. Yet, liberals were especially likely to use obscenities and insults, especially on the days the ruling was leaked and after the Dobbs decision. In turn, toxic language sharply increased among both groups following the leak and after the SCOTUS ruling. (3) Conservatives employ religious and fetal personhood frames, while liberals emphasize women's health and bodily autonomy, with each group reacting negatively to the other group's frames. Our results offer an in-depth insight into the dynamics of online discourse on one of the most contentious issues in contemporary America.

{{</citation>}}


### (111/145) Finnish 5th and 6th graders' misconceptions about Artificial Intelligence (Pekka Mertala et al., 2023)

{{<citation>}}

Pekka Mertala, Janne Fagerlund. (2023)  
**Finnish 5th and 6th graders' misconceptions about Artificial Intelligence**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16644v1)  

---


**ABSTRACT**  
Research on children's initial conceptions of AI is in an emerging state, which, from a constructivist viewpoint, challenges the development of pedagogically sound AI-literacy curricula, methods, and materials. To contribute to resolving this need in the present paper, qualitative survey data from 195 children were analyzed abductively to answer the following three research questions: What kind of misconceptions do Finnish 5th and 6th graders' have about the essence AI?; 2) How do these misconceptions relate to common misconception types?; and 3) How profound are these misconceptions? As a result, three misconception categories were identified: 1) Non-technological AI, in which AI was conceptualized as peoples' cognitive processes (factual misconception); 2) Anthropomorphic AI, in which AI was conceptualized as a human-like entity (vernacular, non-scientific, and conceptual misconception); and 3) AI as a machine with a pre-installed intelligence or knowledge (factual misconception). Majority of the children evaluated their AI-knowledge low, which implies that the misconceptions are more superficial than profound. The findings suggest that context-specific linguistic features can contribute to students' AI misconceptions. Implications for future research and AI literacy education are discussed.

{{</citation>}}


## cs.AI (4)



### (112/145) War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars (Wenyue Hua et al., 2023)

{{<citation>}}

Wenyue Hua, Lizhou Fan, Lingyao Li, Kai Mei, Jianchao Ji, Yingqiang Ge, Libby Hemphill, Yongfeng Zhang. (2023)  
**War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CY, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.17227v1)  

---


**ABSTRACT**  
Can we avoid wars at the crossroads of history? This question has been pursued by individuals, scholars, policymakers, and organizations throughout human history. In this research, we attempt to answer the question based on the recent advances of Artificial Intelligence (AI) and Large Language Models (LLMs). We propose \textbf{WarAgent}, an LLM-powered multi-agent AI system, to simulate the participating countries, their decisions, and the consequences, in historical international conflicts, including the World War I (WWI), the World War II (WWII), and the Warring States Period (WSP) in Ancient China. By evaluating the simulation effectiveness, we examine the advancements and limitations of cutting-edge AI systems' abilities in studying complex collective human behaviors such as international conflicts under diverse settings. In these simulations, the emergent interactions among agents also offer a novel perspective for examining the triggers and conditions that lead to war. Our findings offer data-driven and AI-augmented insights that can redefine how we approach conflict resolution and peacekeeping strategies. The implications stretch beyond historical analysis, offering a blueprint for using AI to understand human history and possibly prevent future international conflicts. Code and data are available at \url{https://github.com/agiresearch/WarAgent}.

{{</citation>}}


### (113/145) (Ir)rationality in AI: State of the Art, Research Challenges and Open Questions (Olivia Macmillan-Scott et al., 2023)

{{<citation>}}

Olivia Macmillan-Scott, Mirco Musolesi. (2023)  
**(Ir)rationality in AI: State of the Art, Research Challenges and Open Questions**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-HC, cs-LG, cs-MA, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.17165v1)  

---


**ABSTRACT**  
The concept of rationality is central to the field of artificial intelligence. Whether we are seeking to simulate human reasoning, or the goal is to achieve bounded optimality, we generally seek to make artificial agents as rational as possible. Despite the centrality of the concept within AI, there is no unified definition of what constitutes a rational agent. This article provides a survey of rationality and irrationality in artificial intelligence, and sets out the open questions in this area. The understanding of rationality in other fields has influenced its conception within artificial intelligence, in particular work in economics, philosophy and psychology. Focusing on the behaviour of artificial agents, we consider irrational behaviours that can prove to be optimal in certain scenarios. Some methods have been developed to deal with irrational agents, both in terms of identification and interaction, however work in this area remains limited. Methods that have up to now been developed for other purposes, namely adversarial scenarios, may be adapted to suit interactions with artificial agents. We further discuss the interplay between human and artificial agents, and the role that rationality plays within this interaction; many questions remain in this area, relating to potentially irrational behaviour of both humans and artificial agents.

{{</citation>}}


### (114/145) Agent-Aware Training for Agent-Agnostic Action Advising in Deep Reinforcement Learning (Yaoquan Wei et al., 2023)

{{<citation>}}

Yaoquan Wei, Shunyu Liu, Jie Song, Tongya Zheng, Kaixuan Chen, Yong Wang, Mingli Song. (2023)  
**Agent-Aware Training for Agent-Agnostic Action Advising in Deep Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.16807v1)  

---


**ABSTRACT**  
Action advising endeavors to leverage supplementary guidance from expert teachers to alleviate the issue of sampling inefficiency in Deep Reinforcement Learning (DRL). Previous agent-specific action advising methods are hindered by imperfections in the agent itself, while agent-agnostic approaches exhibit limited adaptability to the learning agent. In this study, we propose a novel framework called Agent-Aware trAining yet Agent-Agnostic Action Advising (A7) to strike a balance between the two. The underlying concept of A7 revolves around utilizing the similarity of state features as an indicator for soliciting advice. However, unlike prior methodologies, the measurement of state feature similarity is performed by neither the error-prone learning agent nor the agent-agnostic advisor. Instead, we employ a proxy model to extract state features that are both discriminative (adaptive to the agent) and generally applicable (robust to agent noise). Furthermore, we utilize behavior cloning to train a model for reusing advice and introduce an intrinsic reward for the advised samples to incentivize the utilization of expert guidance. Experiments are conducted on the GridWorld, LunarLander, and six prominent scenarios from Atari games. The results demonstrate that A7 significantly accelerates the learning process and surpasses existing methods (both agent-specific and agent-agnostic) by a substantial margin. Our code will be made publicly available.

{{</citation>}}


### (115/145) Hyper-Relational Knowledge Graph Neural Network for Next POI (Jixiao Zhang et al., 2023)

{{<citation>}}

Jixiao Zhang, Yongkang Li, Ruotong Zou, Jingyuan Zhang, Zipei Fan, Xuan Song. (2023)  
**Hyper-Relational Knowledge Graph Neural Network for Next POI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IR, cs-LG, cs.AI  
Keywords: GNN, Graph Neural Network, Knowledge Graph, Social Network  
[Paper Link](http://arxiv.org/abs/2311.16683v1)  

---


**ABSTRACT**  
With the advancement of mobile technology, Point of Interest (POI) recommendation systems in Location-based Social Networks (LBSN) have brought numerous benefits to both users and companies. Many existing works employ Knowledge Graph (KG) to alleviate the data sparsity issue in LBSN. These approaches primarily focus on modeling the pair-wise relations in LBSN to enrich the semantics and thereby relieve the data sparsity issue. However, existing approaches seldom consider the hyper-relations in LBSN, such as the mobility relation (a 3-ary relation: user-POI-time). This makes the model hard to exploit the semantics accurately. In addition, prior works overlook the rich structural information inherent in KG, which consists of higher-order relations and can further alleviate the impact of data sparsity.To this end, we propose a Hyper-Relational Knowledge Graph Neural Network (HKGNN) model. In HKGNN, a Hyper-Relational Knowledge Graph (HKG) that models the LBSN data is constructed to maintain and exploit the rich semantics of hyper-relations. Then we proposed a Hypergraph Neural Network to utilize the structural information of HKG in a cohesive way. In addition, a self-attention network is used to leverage sequential information and make personalized recommendations. Furthermore, side information, essential in reducing data sparsity by providing background knowledge of POIs, is not fully utilized in current methods. In light of this, we extended the current dataset with available side information to further lessen the impact of data sparsity. Results of experiments on four real-world LBSN datasets demonstrate the effectiveness of our approach compared to existing state-of-the-art methods.

{{</citation>}}


## math.NT (1)



### (116/145) Applications of Moments of Dirichlet Coefficients in Elliptic Curve Families (Zoë Batterman et al., 2023)

{{<citation>}}

Zoë Batterman, Aditya Jambhale, Steven J. Miller, Akash L. Narayanan, Kishan Sharma, Andrew Yang, Chris Yao. (2023)  
**Applications of Moments of Dirichlet Coefficients in Elliptic Curve Families**  

---
Primary Category: math.NT  
Categories: 11G05, 11G40, cs-NA, math-NA, math-NT, math.NT  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.17215v1)  

---


**ABSTRACT**  
The moments of the coefficients of elliptic curve L-functions are related to numerous arithmetic problems. Rosen and Silverman proved a conjecture of Nagao relating the first moment of one-parameter families satisfying Tate's conjecture to the rank of the corresponding elliptic surface over Q(T); one can also construct families of moderate rank by finding families with large first moments. Michel proved that if j(T) is not constant, then the second moment of the family is of size p^2 + O(p^(3/2)); these two moments show that for suitably small support the behavior of zeros near the central point agree with that of eigenvalues from random matrix ensembles, with the higher moments impacting the rate of convergence.   In his thesis, Miller noticed a negative bias in the second moment of every one-parameter family of elliptic curves over the rationals whose second moment had a calculable closed-form expression, specifically the first lower order term which does not average to zero is on average negative. This Bias Conjecture is confirmed for many families; however, these are highly non-generic families whose resulting Legendre sums can be determined. Inspired by the recent successes by Yang-Hui He, Kyu-Hwan Lee, Thomas Oliver, Alexey Pozdnyakov and others in investigations of murmurations of elliptic curve coefficients with machine learning techniques, we pose a similar problem for trying to understand the Bias Conjecture. As a start to this program, we numerically investigate the Bias Conjecture for a family whose bias is positive for half the primes. Since the numerics do not offer conclusive evidence that negative bias for the other half is enough to overwhelm the positive bias, the Bias Conjecture cannot be verified for the family.

{{</citation>}}


## cs.GT (2)



### (117/145) LSTM model predicting outcome of strategic thinking task exhibits representations of level-k thinking (Mario Stepanik, 2023)

{{<citation>}}

Mario Stepanik. (2023)  
**LSTM model predicting outcome of strategic thinking task exhibits representations of level-k thinking**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.17211v1)  

---


**ABSTRACT**  
Which neural mechanisms underlie strategic thinking in the human brain? Neuroeconomic research has not yet bridged the gap between theoretical models of higher-order reasoning and the precise mechanisms implemented in neural networks in the human brain. In this paper, I demonstrate that a recurrent neural network model can learn to perform strongly in the simple strategic game Rock-Paper-Scissors. In doing so, it develops implicit representations of strategically important variables (the levels $k$ of reasoning) which economists have postulated in theoretical models. These representations can be extracted from the hidden activations of the neural network. These findings hint at a connection between the mechanisms implicit in recurrent neural networks and models of strategic thinking in economic theory. Future empirical brain research can investigate whether these mechanisms correspond to mechanisms implicit in biological neural networks.

{{</citation>}}


### (118/145) Multi-defender Security Games with Schedules (Zimeng Song et al., 2023)

{{<citation>}}

Zimeng Song, Chun Kai Ling, Fei Fang. (2023)  
**Multi-defender Security Games with Schedules**  

---
Primary Category: cs.GT  
Categories: cs-AI, cs-GT, cs.GT  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.16392v1)  

---


**ABSTRACT**  
Stackelberg Security Games are often used to model strategic interactions in high-stakes security settings. The majority of existing models focus on single-defender settings where a single entity assumes command of all security assets. However, many realistic scenarios feature multiple heterogeneous defenders with their own interests and priorities embedded in a more complex system. Furthermore, defenders rarely choose targets to protect. Instead, they have a multitude of defensive resources or schedules at its disposal, each with different protective capabilities. In this paper, we study security games featuring multiple defenders and schedules simultaneously. We show that unlike prior work on multi-defender security games, the introduction of schedules can cause non-existence of equilibrium even under rather restricted environments. We prove that under the mild restriction that any subset of a schedule is also a schedule, non-existence of equilibrium is not only avoided, but can be computed in polynomial time in games with two defenders. Under additional assumptions, our algorithm can be extended to games with more than two defenders and its computation scaled up in special classes of games with compactly represented schedules such as those used in patrolling applications. Experimental results suggest that our methods scale gracefully with game size, making our algorithms amongst the few that can tackle multiple heterogeneous defenders.

{{</citation>}}


## hep-ex (1)



### (119/145) Fast Particle-based Anomaly Detection Algorithm with Variational Autoencoder (Ryan Liu et al., 2023)

{{<citation>}}

Ryan Liu, Abhijith Gandrakota, Jennifer Ngadiuba, Maria Spiropulu, Jean-Roch Vlimant. (2023)  
**Fast Particle-based Anomaly Detection Algorithm with Variational Autoencoder**  

---
Primary Category: hep-ex  
Categories: cs-LG, hep-ex, hep-ex  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.17162v1)  

---


**ABSTRACT**  
Model-agnostic anomaly detection is one of the promising approaches in the search for new beyond the standard model physics. In this paper, we present Set-VAE, a particle-based variational autoencoder (VAE) anomaly detection algorithm. We demonstrate a 2x signal efficiency gain compared with traditional subjettiness-based jet selection. Furthermore, with an eye to the future deployment to trigger systems, we propose the CLIP-VAE, which reduces the inference-time cost of anomaly detection by using the KL-divergence loss as the anomaly score, resulting in a 2x acceleration in latency and reducing the caching requirement.

{{</citation>}}


## astro-ph.IM (1)



### (120/145) Predicting the Age of Astronomical Transients from Real-Time Multivariate Time Series (Hali Huang et al., 2023)

{{<citation>}}

Hali Huang, Daniel Muthukrishna, Prajna Nair, Zimi Zhang, Michael Fausnaugh, Torsha Majumder, Ryan J. Foley, George R. Ricker. (2023)  
**Predicting the Age of Astronomical Transients from Real-Time Multivariate Time Series**  

---
Primary Category: astro-ph.IM  
Categories: astro-ph-HE, astro-ph-IM, astro-ph.IM, cs-LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.17143v1)  

---


**ABSTRACT**  
Astronomical transients, such as supernovae and other rare stellar explosions, have been instrumental in some of the most significant discoveries in astronomy. New astronomical sky surveys will soon record unprecedented numbers of transients as sparsely and irregularly sampled multivariate time series. To improve our understanding of the physical mechanisms of transients and their progenitor systems, early-time measurements are necessary. Prioritizing the follow-up of transients based on their age along with their class is crucial for new surveys. To meet this demand, we present the first method of predicting the age of transients in real-time from multi-wavelength time-series observations. We build a Bayesian probabilistic recurrent neural network. Our method can accurately predict the age of a transient with robust uncertainties as soon as it is initially triggered by a survey telescope. This work will be essential for the advancement of our understanding of the numerous young transients being detected by ongoing and upcoming astronomical surveys.

{{</citation>}}


## cs.RO (4)



### (121/145) Mission-driven Exploration for Accelerated Deep Reinforcement Learning with Temporal Logic Task Specifications (Jun Wang et al., 2023)

{{<citation>}}

Jun Wang, Hosein Hasanbeig, Kaiyuan Tan, Zihe Sun, Yiannis Kantaros. (2023)  
**Mission-driven Exploration for Accelerated Deep Reinforcement Learning with Temporal Logic Task Specifications**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.17059v1)  

---


**ABSTRACT**  
This paper addresses the problem of designing optimal control policies for mobile robots with mission and safety requirements specified using Linear Temporal Logic (LTL). We consider robots with unknown stochastic dynamics operating in environments with unknown geometric structure. The robots are equipped with sensors allowing them to detect obstacles. Our goal is to synthesize a control policy that maximizes the probability of satisfying an LTL-encoded task in the presence of motion and environmental uncertainty. Several deep reinforcement learning (DRL) algorithms have been proposed recently to address similar problems. A common limitation in related works is that of slow learning performance. In order to address this issue, we propose a novel DRL algorithm, which has the capability to learn control policies at a notably faster rate compared to similar methods. Its sample efficiency is due to a mission-driven exploration strategy that prioritizes exploration towards directions that may contribute to mission accomplishment. Identifying these directions relies on an automaton representation of the LTL task as well as a learned neural network that (partially) models the unknown system dynamics. We provide comparative experiments demonstrating the efficiency of our algorithm on robot navigation tasks in unknown environments.

{{</citation>}}


### (122/145) End-to-end Reinforcement Learning for Time-Optimal Quadcopter Flight (Robin Ferede et al., 2023)

{{<citation>}}

Robin Ferede, Christophe De Wagter, Dario Izzo, Guido C. H. E. de Croon. (2023)  
**End-to-end Reinforcement Learning for Time-Optimal Quadcopter Flight**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.16948v1)  

---


**ABSTRACT**  
Aggressive time-optimal control of quadcopters poses a significant challenge in the field of robotics. The state-of-the-art approach leverages reinforcement learning (RL) to train optimal neural policies. However, a critical hurdle is the sim-to-real gap, often addressed by employing a robust inner loop controller -an abstraction that, in theory, constrains the optimality of the trained controller, necessitating margins to counter potential disturbances. In contrast, our novel approach introduces high-speed quadcopter control using end-to-end RL (E2E) that gives direct motor commands. To bridge the reality gap, we incorporate a learned residual model and an adaptive method that can compensate for modeling errors in thrust and moments. We compare our E2E approach against a state-of-the-art network that commands thrust and body rates to an INDI inner loop controller, both in simulated and real-world flight. E2E showcases a significant 1.39-second advantage in simulation and a 0.17-second edge in real-world testing, highlighting end-to-end reinforcement learning's potential. The performance drop observed from simulation to reality shows potential for further improvement, including refining strategies to address the reality gap or exploring offline reinforcement learning with real flight data.

{{</citation>}}


### (123/145) Increasing Transparency of Reinforcement Learning using Shielding for Human Preferences and Explanations (Georgios Angelopoulos et al., 2023)

{{<citation>}}

Georgios Angelopoulos, Luigi Mangiacapra, Alessandra Rossi, Claudia Di Napoli, Silvia Rossi. (2023)  
**Increasing Transparency of Reinforcement Learning using Shielding for Human Preferences and Explanations**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.16838v1)  

---


**ABSTRACT**  
The adoption of Reinforcement Learning (RL) in several human-centred applications provides robots with autonomous decision-making capabilities and adaptability based on the observations of the operating environment. In such scenarios, however, the learning process can make robots' behaviours unclear and unpredictable to humans, thus preventing a smooth and effective Human-Robot Interaction (HRI). As a consequence, it becomes crucial to avoid robots performing actions that are unclear to the user. In this work, we investigate whether including human preferences in RL (concerning the actions the robot performs during learning) improves the transparency of a robot's behaviours. For this purpose, a shielding mechanism is included in the RL algorithm to include human preferences and to monitor the learning agent's decisions. We carried out a within-subjects study involving 26 participants to evaluate the robot's transparency in terms of Legibility, Predictability, and Expectability in different settings. Results indicate that considering human preferences during learning improves Legibility with respect to providing only Explanations, and combining human preferences with explanations elucidating the rationale behind the robot's decisions further amplifies transparency. Results also confirm that an increase in transparency leads to an increase in the safety, comfort, and reliability of the robot. These findings show the importance of transparency during learning and suggest a paradigm for robotic applications with human in the loop.

{{</citation>}}


### (124/145) ROSO: Improving Robotic Policy Inference via Synthetic Observations (Yusuke Miyashita et al., 2023)

{{<citation>}}

Yusuke Miyashita, Dimitris Gahtidis, Colin La, Jeremy Rabinowicz, Jurgen Leitner. (2023)  
**ROSO: Improving Robotic Policy Inference via Synthetic Observations**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16680v2)  

---


**ABSTRACT**  
In this paper, we propose the use of generative artificial intelligence (AI) to improve zero-shot performance of a pre-trained policy by altering observations during inference. Modern robotic systems, powered by advanced neural networks, have demonstrated remarkable capabilities on pre-trained tasks. However, generalizing and adapting to new objects and environments is challenging, and fine-tuning visuomotor policies is time-consuming. To overcome these issues we propose Robotic Policy Inference via Synthetic Observations (ROSO). ROSO uses stable diffusion to pre-process a robot's observation of novel objects during inference time to fit within its distribution of observations of the pre-trained policies. This novel paradigm allows us to transfer learned knowledge from known tasks to previously unseen scenarios, enhancing the robot's adaptability without requiring lengthy fine-tuning. Our experiments show that incorporating generative AI into robotic inference significantly improves successful outcomes, finishing up to 57% of tasks otherwise unsuccessful with the pre-trained policy.

{{</citation>}}


## cs.DS (1)



### (125/145) Node Connectivity Augmentation of Highly Connected Graphs (Waldo Galvez et al., 2023)

{{<citation>}}

Waldo Galvez, Dylan Hyatt-Denesik, Afrouz Jabal Ameli, Laura Sanita. (2023)  
**Node Connectivity Augmentation of Highly Connected Graphs**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs.DS  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.17010v1)  

---


**ABSTRACT**  
Node-connectivity augmentation is a fundamental network design problem. We are given a $k$-node connected graph $G$ together with an additional set of links, and the goal is to add a cheap subset of links to $G$ to make it $(k+1)$-node connected.   In this work, we characterize completely the computational complexity status of the problem, by showing hardness for all values of $k$ which were not addressed previously in the literature.   We then focus on $k$-node connectivity augmentation for $k=n-4$, which corresponds to the highest value of $k$ for which the problem is NP-hard. We improve over the previously best known approximation bounds for this problem, by developing a $\frac{3}{2}$-approximation algorithm for the weighted setting, and a $\frac{4}{3}$-approximation algorithm for the unweighted setting.

{{</citation>}}


## stat.ME (1)



### (126/145) FedECA: A Federated External Control Arm Method for Causal Inference with Time-To-Event Data in Distributed Settings (Jean Ogier du Terrail et al., 2023)

{{<citation>}}

Jean Ogier du Terrail, Quentin Klopfenstein, Honghao Li, Imke Mayer, Nicolas Loiseau, Mohammad Hallal, Félix Balazard, Mathieu Andreux. (2023)  
**FedECA: A Federated External Control Arm Method for Causal Inference with Time-To-Event Data in Distributed Settings**  

---
Primary Category: stat.ME  
Categories: cs-DC, cs-LG, stat-ME, stat.ME  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16984v1)  

---


**ABSTRACT**  
External control arms (ECA) can inform the early clinical development of experimental drugs and provide efficacy evidence for regulatory approval in non-randomized settings. However, the main challenge of implementing ECA lies in accessing real-world data or historical clinical trials. Indeed, data sharing is often not feasible due to privacy considerations related to data leaving the original collection centers, along with pharmaceutical companies' competitive motives. In this paper, we leverage a privacy-enhancing technology called federated learning (FL) to remove some of the barriers to data sharing. We introduce a federated learning inverse probability of treatment weighted (IPTW) method for time-to-event outcomes called FedECA which eases the implementation of ECA by limiting patients' data exposure. We show with extensive experiments that FedECA outperforms its closest competitor, matching-adjusted indirect comparison (MAIC), in terms of statistical power and ability to balance the treatment and control groups. To encourage the use of such methods, we publicly release our code which relies on Substra, an open-source FL software with proven experience in privacy-sensitive contexts.

{{</citation>}}


## physics.optics (1)



### (127/145) 65 GOPS/neuron Photonic Tensor Core with Thin-film Lithium Niobate Photonics (Zhongjin Lin et al., 2023)

{{<citation>}}

Zhongjin Lin, Bhavin J. Shastri, Shangxuan Yu, Jingxiang Song, Yuntao Zhu, Arman Safarnejadian, Wangning Cai, Yanmei Lin, Wei Ke, Mustafa Hammood, Tianye Wang, Mengyue Xu, Zibo Zheng, Mohammed Al-Qadasi, Omid Esmaeeli, Mohamed Rahim, Grzegorz Pakulski, Jens Schmid, Pedro Barrios, Weihong Jiang, Hugh Morison, Matthew Mitchell, Xiaogang Qiang, Xun Guan, Nicolas A. F. Jaeger, Leslie A. n Rusch, Sudip Shekhar, Wei Shi, Siyuan Yu, Xinlun Cai, Lukas Chrostowski. (2023)  
**65 GOPS/neuron Photonic Tensor Core with Thin-film Lithium Niobate Photonics**  

---
Primary Category: physics.optics  
Categories: 78A05, cs-ET, physics-app-ph, physics-optics, physics.optics  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16896v1)  

---


**ABSTRACT**  
Photonics offers a transformative approach to artificial intelligence (AI) and neuromorphic computing by providing low latency, high bandwidth, and energy-efficient computations. Here, we introduce a photonic tensor core processor enabled by time-multiplexed inputs and charge-integrated outputs. This fully integrated processor, comprising only two thin-film lithium niobate (TFLN) modulators, a III-V laser, and a charge-integration photoreceiver, can implement an entire layer of a neural network. It can execute 65 billion operations per second (GOPS) per neuron, including simultaneous weight updates-a hitherto unachieved speed. Our processor stands out from conventional photonic processors, which have static weights set during training, as it supports fast "hardware-in-the-loop" training, and can dynamically adjust the inputs (fan-in) and outputs (fan-out) within a layer, thereby enhancing its versatility. Our processor can perform large-scale dot-product operations with vector dimensions up to 131,072. Furthermore, it successfully classifies (supervised learning) and clusters (unsupervised learning) 112*112-pixel images after "hardware-in-the-loop" training. To handle "hardware-in-the-loop" training for clustering AI tasks, we provide a solution for multiplications involving two negative numbers based on our processor.

{{</citation>}}


## cs.NI (2)



### (128/145) Digital Twin-Enhanced Deep Reinforcement Learning for Resource Management in Networks Slicing (Zhengming Zhang et al., 2023)

{{<citation>}}

Zhengming Zhang, Yongming Huang, Cheng Zhang, Qingbi Zheng, Luxi Yang, Xiaohu You. (2023)  
**Digital Twin-Enhanced Deep Reinforcement Learning for Resource Management in Networks Slicing**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.16876v1)  

---


**ABSTRACT**  
Network slicing-based communication systems can dynamically and efficiently allocate resources for diversified services. However, due to the limitation of the network interface on channel access and the complexity of the resource allocation, it is challenging to achieve an acceptable solution in the practical system without precise prior knowledge of the dynamics probability model of the service requests. Existing work attempts to solve this problem using deep reinforcement learning (DRL), however, such methods usually require a lot of interaction with the real environment in order to achieve good results. In this paper, a framework consisting of a digital twin and reinforcement learning agents is present to handle the issue. Specifically, we propose to use the historical data and the neural networks to build a digital twin model to simulate the state variation law of the real environment. Then, we use the data generated by the network slicing environment to calibrate the digital twin so that it is in sync with the real environment. Finally, DRL for slice optimization optimizes its own performance in this virtual pre-verification environment. We conducted an exhaustive verification of the proposed digital twin framework to confirm its scalability. Specifically, we propose to use loss landscapes to visualize the generalization of DRL solutions. We explore a distillation-based optimization scheme for lightweight slicing strategies. In addition, we also extend the framework to offline reinforcement learning, where solutions can be used to obtain intelligent decisions based solely on historical data. Numerical simulation experiments show that the proposed digital twin can significantly improve the performance of the slice optimization strategy.

{{</citation>}}


### (129/145) Edge AI for Internet of Energy: Challenges and Perspectives (Yassine Himeur et al., 2023)

{{<citation>}}

Yassine Himeur, Aya Nabil Sayed, Abdullah Alsalemi, Faycal Bensaali, Abbes Amira. (2023)  
**Edge AI for Internet of Energy: Challenges and Perspectives**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-CY, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16851v1)  

---


**ABSTRACT**  
The digital landscape of the Internet of Energy (IoE) is on the brink of a revolutionary transformation with the integration of edge Artificial Intelligence (AI). This comprehensive review elucidates the promise and potential that edge AI holds for reshaping the IoE ecosystem. Commencing with a meticulously curated research methodology, the article delves into the myriad of edge AI techniques specifically tailored for IoE. The myriad benefits, spanning from reduced latency and real-time analytics to the pivotal aspects of information security, scalability, and cost-efficiency, underscore the indispensability of edge AI in modern IoE frameworks. As the narrative progresses, readers are acquainted with pragmatic applications and techniques, highlighting on-device computation, secure private inference methods, and the avant-garde paradigms of AI training on the edge. A critical analysis follows, offering a deep dive into the present challenges including security concerns, computational hurdles, and standardization issues. However, as the horizon of technology ever expands, the review culminates in a forward-looking perspective, envisaging the future symbiosis of 5G networks, federated edge AI, deep reinforcement learning, and more, painting a vibrant panorama of what the future beholds. For anyone vested in the domains of IoE and AI, this review offers both a foundation and a visionary lens, bridging the present realities with future possibilities.

{{</citation>}}


## cs.HC (3)



### (130/145) RELIC: Investigating Large Language Model Responses using Self-Consistency (Furui Cheng et al., 2023)

{{<citation>}}

Furui Cheng, Vilém Zouhar, Simran Arora, Mrinmaya Sachan, Hendrik Strobelt, Mennatallah El-Assady. (2023)  
**RELIC: Investigating Large Language Model Responses using Self-Consistency**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16842v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are notorious for blending fact with fiction and generating non-factual content, known as hallucinations. To tackle this challenge, we propose an interactive system that helps users obtain insights into the reliability of the generated text. Our approach is based on the idea that the self-consistency of multiple samples generated by the same LLM relates to its confidence in individual claims in the generated texts. Using this idea, we design RELIC, an interactive system that enables users to investigate and verify semantic-level variations in multiple long-form responses. This allows users to recognize potentially inaccurate information in the generated text and make necessary corrections. From a user study with ten participants, we demonstrate that our approach helps users better verify the reliability of the generated text. We further summarize the design implications and lessons learned from this research for inspiring future studies on reliable human-LLM interactions.

{{</citation>}}


### (131/145) Inspo: Writing Stories with a Flock of AIs and Humans (Chieh-Yang Huang et al., 2023)

{{<citation>}}

Chieh-Yang Huang, Sanjana Gautam, Shannon McClellan Brooks, Ya-Fang Lin, Ting-Hao 'Kenneth' Huang. (2023)  
**Inspo: Writing Stories with a Flock of AIs and Humans**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16521v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have advanced automated writing assistance, enabling complex tasks like co-writing novels and poems. However, real-world writing typically requires various support and collaboration across stages and scenarios. Existing research mainly examines how writers utilize single text generators, neglecting this broader context. This paper introduces Inspo, a web-based editor that incorporates various text generators and online crowd workers. Through a three-phase user study, we examine writers' interactions with Inspo for novel writing. Quantitative analyses of writing logs highlight changes in participants' writing progress and the influence of various text-generation models. Complementing this with qualitative insights from semi-structured interviews, we illustrate participants' perceptions of these models and the crowd. Based on the findings, we provide design recommendations for the next generation of intelligent writing tools and discuss the potential sociocultural implications of integrating AI and human input in the writing process.

{{</citation>}}


### (132/145) Enhancing Human Persuasion With Large Language Models (Minkyu Shin et al., 2023)

{{<citation>}}

Minkyu Shin, Jin Kim. (2023)  
**Enhancing Human Persuasion With Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: AI, ChatGPT, Financial, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16466v1)  

---


**ABSTRACT**  
Although large language models (LLMs) are reshaping various aspects of human life, our current understanding of their impacts remains somewhat constrained. Here we investigate the impact of LLMs on human communication, in the context of consumer complaints in the financial industry. Employing an AI detection tool on more than 780K complaints gathered by the Consumer Financial Protection Bureau (CFPB), we find evidence of LLM usage in the writing of complaints - shortly after the release of ChatGPT. Our analyses reveal that LLM usage is positively correlated with the likelihood of obtaining desirable outcomes (i.e., offer of relief from financial firms) and suggest that this positive correlation may be partly due to the linguistic features improved by LLMs. We test this conjecture with a preregistered experiment, which reveals results consistent with those from observational studies: Consumer complaints written with ChatGPT for improved linguistic qualities were more likely to receive hypothetical relief offers than the original consumer complaints, demonstrating the LLM's ability to enhance message persuasiveness in human communication. Being some of the earliest empirical evidence on LLM usage for enhancing persuasion, our results highlight the transformative potential of LLMs in human communication.

{{</citation>}}


## cs.IR (4)



### (133/145) MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation (Yunshan Ma et al., 2023)

{{<citation>}}

Yunshan Ma, Yingzhi He, Xiang Wang, Yinwei Wei, Xiaoyu Du, Yuyangzi Fu, Tat-Seng Chua. (2023)  
**MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.16751v1)  

---


**ABSTRACT**  
Bundle recommendation seeks to recommend a bundle of related items to users to improve both user experience and the profits of platform. Existing bundle recommendation models have progressed from capturing only user-bundle interactions to the modeling of multiple relations among users, bundles and items. CrossCBR, in particular, incorporates cross-view contrastive learning into a two-view preference learning framework, significantly improving SOTA performance. It does, however, have two limitations: 1) the two-view formulation does not fully exploit all the heterogeneous relations among users, bundles and items; and 2) the "early contrast and late fusion" framework is less effective in capturing user preference and difficult to generalize to multiple views. In this paper, we present MultiCBR, a novel Multi-view Contrastive learning framework for Bundle Recommendation. First, we devise a multi-view representation learning framework capable of capturing all the user-bundle, user-item and bundle-item relations, especially better utilizing the bundle-item affiliations to enhance sparse bundles' representations. Second, we innovatively adopt an "early fusion and late contrast" design that first fuses the multi-view representations before performing self-supervised contrastive learning. In comparison to existing approaches, our framework reverses the order of fusion and contrast, introducing the following advantages: 1)our framework is capable of modeling both cross-view and ego-view preferences, allowing us to achieve enhanced user preference modeling; and 2) instead of requiring quadratic number of cross-view contrastive losses, we only require two self-supervised contrastive losses, resulting in minimal extra costs. Experimental results on three public datasets indicate that our method outperforms SOTA methods.

{{</citation>}}


### (134/145) RankingGPT: Empowering Large Language Models in Text Ranking with Progressive Enhancement (Longhui Zhang et al., 2023)

{{<citation>}}

Longhui Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, Min Zhang. (2023)  
**RankingGPT: Empowering Large Language Models in Text Ranking with Progressive Enhancement**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16720v1)  

---


**ABSTRACT**  
Text ranking is a critical task in various information retrieval applications, and the recent success of Large Language Models (LLMs) in natural language processing has sparked interest in their application to text ranking. These methods primarily involve combining query and candidate documents and leveraging prompt learning to determine query-document relevance using the LLM's output probabilities for specific tokens or by directly generating a ranked list of candidate documents. Although these approaches have demonstrated promise, a noteworthy disparity arises between the training objective of LLMs, which typically centers around next token prediction, and the objective of evaluating query-document relevance. To address this gap and fully leverage LLM potential in text ranking tasks, we propose a progressive multi-stage training strategy. Firstly, we introduce a large-scale weakly supervised dataset of relevance texts to enable the LLMs to acquire the ability to predict relevant tokens without altering their original training objective. Subsequently, we incorporate supervised training to further enhance LLM ranking capability. Our experimental results on multiple benchmarks demonstrate the superior performance of our proposed method compared to previous competitive approaches, both in in-domain and out-of-domain scenarios.

{{</citation>}}


### (135/145) Graph Pre-training and Prompt Learning for Recommendation (Yuhao Yang et al., 2023)

{{<citation>}}

Yuhao Yang, Lianghao Xia, Da Luo, Kangyi Lin, Chao Huang. (2023)  
**Graph Pre-training and Prompt Learning for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.16716v1)  

---


**ABSTRACT**  
GNN-based recommenders have excelled in modeling intricate user-item interactions through multi-hop message passing. However, existing methods often overlook the dynamic nature of evolving user-item interactions, which impedes the adaption to changing user preferences and distribution shifts in newly arriving data. Thus, their scalability and performances in real-world dynamic environments are limited. In this study, we propose GraphPL, a framework that incorporates parameter-efficient and dynamic graph pre-training with prompt learning. This novel combination empowers GNNs to effectively capture both long-term user preferences and short-term behavior dynamics, enabling the delivery of accurate and timely recommendations. Our GraphPL framework addresses the challenge of evolving user preferences by seamlessly integrating a temporal prompt mechanism and a graph-structural prompt learning mechanism into the pre-trained GNN model. The temporal prompt mechanism encodes time information on user-item interaction, allowing the model to naturally capture temporal context, while the graph-structural prompt learning mechanism enables the transfer of pre-trained knowledge to adapt to behavior dynamics without the need for continuous incremental training. We further bring in a dynamic evaluation setting for recommendation to mimic real-world dynamic scenarios and bridge the offline-online gap to a better level. Our extensive experiments including a large-scale industrial deployment showcases the lightweight plug-in scalability of our GraphPL when integrated with various state-of-the-art recommenders, emphasizing the advantages of GraphPL in terms of effectiveness, robustness and efficiency.

{{</citation>}}


### (136/145) ControlRec: Bridging the Semantic Gap between Language Model and Personalized Recommendation (Junyan Qiu et al., 2023)

{{<citation>}}

Junyan Qiu, Haitao Wang, Zhaolin Hong, Yiping Yang, Qiang Liu, Xingxing Wang. (2023)  
**ControlRec: Bridging the Semantic Gap between Language Model and Personalized Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Contrastive Learning, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16441v1)  

---


**ABSTRACT**  
The successful integration of large language models (LLMs) into recommendation systems has proven to be a major breakthrough in recent studies, paving the way for more generic and transferable recommendations. However, LLMs struggle to effectively utilize user and item IDs, which are crucial identifiers for successful recommendations. This is mainly due to their distinct representation in a semantic space that is different from the natural language (NL) typically used to train LLMs. To tackle such issue, we introduce ControlRec, an innovative Contrastive prompt learning framework for Recommendation systems. ControlRec treats user IDs and NL as heterogeneous features and encodes them individually. To promote greater alignment and integration between them in the semantic space, we have devised two auxiliary contrastive objectives: (1) Heterogeneous Feature Matching (HFM) aligning item description with the corresponding ID or user's next preferred ID based on their interaction sequence, and (2) Instruction Contrastive Learning (ICL) effectively merging these two crucial data sources by contrasting probability distributions of output sequences generated by diverse tasks. Experimental results on four public real-world datasets demonstrate the effectiveness of the proposed method on improving model performance.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (137/145) Sluggish and Chemically-Biased Interstitial Diffusion in Concentrated Solid Solution Alloys: Mechanisms and Methods (Biao Xu et al., 2023)

{{<citation>}}

Biao Xu, Haijun Fu, Shasha Huang, Shihua Ma, Yaoxu Xiong, Jun Zhang, Xuepeng Xiang, Wenyu Lu, Ji-Jung Kai, Shijun Zhao. (2023)  
**Sluggish and Chemically-Biased Interstitial Diffusion in Concentrated Solid Solution Alloys: Mechanisms and Methods**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat.mtrl-sci, cs-LG, physics-atm-clus  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.16727v1)  

---


**ABSTRACT**  
Interstitial diffusion is a pivotal process that governs the phase stability and irradiation response of materials in non-equilibrium conditions. In this work, we study sluggish and chemically-biased interstitial diffusion in Fe-Ni concentrated solid solution alloys (CSAs) by combining machine learning (ML) and kinetic Monte Carlo (kMC), where ML is used to accurately and efficiently predict the migration energy barriers on-the-fly. The ML-kMC reproduces the diffusivity that was reported by molecular dynamics results at high temperatures. With this powerful tool, we find that the observed sluggish diffusion and the "Ni-Ni-Ni"-biased diffusion in Fe-Ni alloys are ascribed to a unique "Barrier Lock" mechanism, whereas the "Fe-Fe-Fe"-biased diffusion is influenced by a "Component Dominance" mechanism. Inspired by the mentioned mechanisms, a practical AvgS-kMC method is proposed for conveniently and swiftly determining interstitial-mediated diffusivity by only relying on the mean energy barriers of migration patterns. Combining the AvgS-kMC with the differential evolutionary algorithm, an inverse design strategy for optimizing sluggish diffusion properties is applied to emphasize the crucial role of favorable migration patterns.

{{</citation>}}


## cs.CR (1)



### (138/145) A Unified Hardware-based Threat Detector for AI Accelerators (Xiaobei Yan et al., 2023)

{{<citation>}}

Xiaobei Yan, Han Qiu, Tianwei Zhang. (2023)  
**A Unified Hardware-based Threat Detector for AI Accelerators**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16684v1)  

---


**ABSTRACT**  
The proliferation of AI technology gives rise to a variety of security threats, which significantly compromise the confidentiality and integrity of AI models and applications. Existing software-based solutions mainly target one specific attack, and require the implementation into the models, rendering them less practical. We design UniGuard, a novel unified and non-intrusive detection methodology to safeguard FPGA-based AI accelerators. The core idea of UniGuard is to harness power side-channel information generated during model inference to spot any anomaly. We employ a Time-to-Digital Converter to capture power fluctuations and train a supervised machine learning model to identify various types of threats. Evaluations demonstrate that UniGuard can achieve 94.0% attack detection accuracy, with high generalization over unknown or adaptive attacks and robustness against varied configurations (e.g., sensor frequency and location).

{{</citation>}}


## eess.AS (1)



### (139/145) LC4SV: A Denoising Framework Learning to Compensate for Unseen Speaker Verification Models (Chi-Chang Lee et al., 2023)

{{<citation>}}

Chi-Chang Lee, Hong-Wei Chen, Chu-Song Chen, Hsin-Min Wang, Tsung-Te Liu, Yu Tsao. (2023)  
**LC4SV: A Denoising Framework Learning to Compensate for Unseen Speaker Verification Models**  

---
Primary Category: eess.AS  
Categories: cs-LG, eess-AS, eess.AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2311.16604v1)  

---


**ABSTRACT**  
The performance of speaker verification (SV) models may drop dramatically in noisy environments. A speech enhancement (SE) module can be used as a front-end strategy. However, existing SE methods may fail to bring performance improvements to downstream SV systems due to artifacts in the predicted signals of SE models. To compensate for artifacts, we propose a generic denoising framework named LC4SV, which can serve as a pre-processor for various unknown downstream SV models. In LC4SV, we employ a learning-based interpolation agent to automatically generate the appropriate coefficients between the enhanced signal and its noisy input to improve SV performance in noisy environments. Our experimental results demonstrate that LC4SV consistently improves the performance of various unseen SV systems. To the best of our knowledge, this work is the first attempt to develop a learning-based interpolation scheme aiming at improving SV performance in noisy environments.

{{</citation>}}


## cs.SD (1)



### (140/145) D4AM: A General Denoising Framework for Downstream Acoustic Models (Chi-Chang Lee et al., 2023)

{{<citation>}}

Chi-Chang Lee, Yu Tsao, Hsin-Min Wang, Chu-Song Chen. (2023)  
**D4AM: A General Denoising Framework for Downstream Acoustic Models**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.16595v1)  

---


**ABSTRACT**  
The performance of acoustic models degrades notably in noisy environments. Speech enhancement (SE) can be used as a front-end strategy to aid automatic speech recognition (ASR) systems. However, existing training objectives of SE methods are not fully effective at integrating speech-text and noisy-clean paired data for training toward unseen ASR systems. In this study, we propose a general denoising framework, D4AM, for various downstream acoustic models. Our framework fine-tunes the SE model with the backward gradient according to a specific acoustic model and the corresponding classification objective. In addition, our method aims to consider the regression objective as an auxiliary loss to make the SE model generalize to other unseen acoustic models. To jointly train an SE unit with regression and classification objectives, D4AM uses an adjustment scheme to directly estimate suitable weighting coefficients rather than undergoing a grid search process with additional training costs. The adjustment scheme consists of two parts: gradient calibration and regression objective weighting. The experimental results show that D4AM can consistently and effectively provide improvements to various unseen acoustic models and outperforms other combination setups. Specifically, when evaluated on the Google ASR API with real noisy data completely unseen during SE training, D4AM achieves a relative WER reduction of 24.65% compared with the direct feeding of noisy input. To our knowledge, this is the first work that deploys an effective combination scheme of regression (denoising) and classification (ASR) objectives to derive a general pre-processor applicable to various unseen ASR systems. Our code is available at https://github.com/ChangLee0903/D4AM.

{{</citation>}}


## cs.DC (1)



### (141/145) Wireless Powered Metaverse: Joint Task Scheduling and Trajectory Design for Multi-Devices and Multi-UAVs (Xiaojie Wang et al., 2023)

{{<citation>}}

Xiaojie Wang, Jiameng Li, Zhaolong Ning, Qingyang Song, Lei Guo, Abbas Jamalipour. (2023)  
**Wireless Powered Metaverse: Joint Task Scheduling and Trajectory Design for Multi-Devices and Multi-UAVs**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.16576v1)  

---


**ABSTRACT**  
To support the running of human-centric metaverse applications on mobile devices, Unmanned Aerial Vehicle (UAV)-assisted Wireless Powered Mobile Edge Computing (WPMEC) is promising to compensate for limited computational capabilities and energy supplies of mobile devices. The high-speed computational processing demands and significant energy consumption of metaverse applications require joint resource scheduling of multiple devices and UAVs, but existing WPMEC solutions address either device or UAV scheduling due to the complexity of combinatorial optimization. To solve the above challenge, we propose a two-stage alternating optimization algorithm based on multi-task Deep Reinforcement Learning (DRL) to jointly allocate charging time, schedule computation tasks, and optimize trajectory of UAVs and mobile devices in a wireless powered metaverse scenario. First, considering energy constraints of both UAVs and mobile devices, we formulate an optimization problem to maximize the computation efficiency of the system. Second, we propose a heuristic algorithm to efficiently perform time allocation and charging scheduling for mobile devices. Following this, we design a multi-task DRL scheme to make charging scheduling and trajectory design decisions for UAVs. Finally, theoretical analysis and performance results demonstrate that our algorithm exhibits significant advantages over representative methods in terms of convergence speed and average computation efficiency.

{{</citation>}}


## cs.AR (2)



### (142/145) RTLFixer: Automatically Fixing RTL Syntax Errors with Large Language Models (YunDa Tsai et al., 2023)

{{<citation>}}

YunDa Tsai, Mingjie Liu, Haoxing Ren. (2023)  
**RTLFixer: Automatically Fixing RTL Syntax Errors with Large Language Models**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.16543v1)  

---


**ABSTRACT**  
This paper presents RTLFixer, a novel framework enabling automatic syntax errors fixing for Verilog code with Large Language Models (LLMs). Despite LLM's promising capabilities, our analysis indicates that approximately 55% of errors in LLM-generated Verilog are syntax-related, leading to compilation failures. To tackle this issue, we introduce a novel debugging framework that employs Retrieval-Augmented Generation (RAG) and ReAct prompting, enabling LLMs to act as autonomous agents in interactively debugging the code with feedback. This framework demonstrates exceptional proficiency in resolving syntax errors, successfully correcting about 98.5% of compilation errors in our debugging dataset, comprising 212 erroneous implementations derived from the VerilogEval benchmark. Our method leads to 32.3% and 10.1% increase in pass@1 success rates in the VerilogEval-Machine and VerilogEval-Human benchmarks, respectively.

{{</citation>}}


### (143/145) Challenges and Opportunities to Enable Large-Scale Computing via Heterogeneous Chiplets (Zhuoping Yang et al., 2023)

{{<citation>}}

Zhuoping Yang, Shixin Ji, Xingzhen Chen, Jinming Zhuang, Weifeng Zhang, Dharmesh Jani, Peipei Zhou. (2023)  
**Challenges and Opportunities to Enable Large-Scale Computing via Heterogeneous Chiplets**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.16417v1)  

---


**ABSTRACT**  
Fast-evolving artificial intelligence (AI) algorithms such as large language models have been driving the ever-increasing computing demands in today's data centers. Heterogeneous computing with domain-specific architectures (DSAs) brings many opportunities when scaling up and scaling out the computing system. In particular, heterogeneous chiplet architecture is favored to keep scaling up and scaling out the system as well as to reduce the design complexity and the cost stemming from the traditional monolithic chip design. However, how to interconnect computing resources and orchestrate heterogeneous chiplets is the key to success. In this paper, we first discuss the diversity and evolving demands of different AI workloads. We discuss how chiplet brings better cost efficiency and shorter time to market. Then we discuss the challenges in establishing chiplet interface standards, packaging, and security issues. We further discuss the software programming challenges in chiplet systems.

{{</citation>}}


## cs.SE (2)



### (144/145) The Transformative Influence of Large Language Models on Software Development (Sajed Jalil, 2023)

{{<citation>}}

Sajed Jalil. (2023)  
**The Transformative Influence of Large Language Models on Software Development**  

---
Primary Category: cs.SE  
Categories: 68T07, D-2-3; I-2-5; I-2-7, cs-HC, cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.16429v1)  

---


**ABSTRACT**  
The increasing adoption and commercialization of generalized Large Language Models (LLMs) have profoundly impacted various aspects of our daily lives. Initially embraced by the computer science community, the versatility of LLMs has found its way into diverse domains. In particular, the software engineering realm has witnessed the most transformative changes. With LLMs increasingly serving as AI Pair Programming Assistants spurred the development of specialized models aimed at aiding software engineers. Although this new paradigm offers numerous advantages, it also presents critical challenges and open problems. To identify the potential and prevailing obstacles, we systematically reviewed contemporary scholarly publications, emphasizing the perspectives of software developers and usability concerns. Preliminary findings underscore pressing concerns about data privacy, bias, and misinformation. Additionally, we identified several usability challenges, including prompt engineering, increased cognitive demands, and mistrust. Finally, we introduce 12 open problems that we have identified through our survey, covering these various domains.

{{</citation>}}


### (145/145) Toward Effective Secure Code Reviews: An Empirical Study of Security-Related Coding Weaknesses (Wachiraphan Charoenwet et al., 2023)

{{<citation>}}

Wachiraphan Charoenwet, Patanamon Thongtanunam, Van-Thuan Pham, Christoph Treude. (2023)  
**Toward Effective Secure Code Reviews: An Empirical Study of Security-Related Coding Weaknesses**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.16396v1)  

---


**ABSTRACT**  
Identifying security issues early is encouraged to reduce the latent negative impacts on software systems. Code review is a widely-used method that allows developers to manually inspect modified code, catching security issues during a software development cycle. However, existing code review studies often focus on known vulnerabilities, neglecting coding weaknesses, which can introduce real-world security issues that are more visible through code review. The practices of code reviews in identifying such coding weaknesses are not yet fully investigated.   To better understand this, we conducted an empirical case study in two large open-source projects, OpenSSL and PHP. Based on 135,560 code review comments, we found that reviewers raised security concerns in 35 out of 40 coding weakness categories. Surprisingly, some coding weaknesses related to past vulnerabilities, such as memory errors and resource management, were discussed less often than the vulnerabilities. Developers attempted to address raised security concerns in many cases (39%-41%), but a substantial portion was merely acknowledged (30%-36%), and some went unfixed due to disagreements about solutions (18%-20%). This highlights that coding weaknesses can slip through code review even when identified. Our findings suggest that reviewers can identify various coding weaknesses leading to security issues during code reviews. However, these results also reveal shortcomings in current code review practices, indicating the need for more effective mechanisms or support for increasing awareness of security issue management in code reviews.

{{</citation>}}
