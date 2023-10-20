---
draft: false
title: "arXiv @ 2023.10.16"
date: 2023-10-16
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.16"
    identifier: arxiv_20231016
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (13)](#cscl-13)
- [cs.CV (12)](#cscv-12)
- [cs.LG (13)](#cslg-13)
- [cs.RO (2)](#csro-2)
- [cs.DC (1)](#csdc-1)
- [cs.CR (2)](#cscr-2)
- [cs.SE (3)](#csse-3)
- [cs.HC (1)](#cshc-1)
- [cs.NI (1)](#csni-1)
- [cs.AI (3)](#csai-3)
- [cs.IR (3)](#csir-3)
- [cs.AR (1)](#csar-1)
- [q-bio.PE (1)](#q-biope-1)
- [cs.SD (1)](#cssd-1)
- [stat.ML (1)](#statml-1)

## cs.CL (13)



### (1/58) Improved Contextual Recognition In Automatic Speech Recognition Systems By Semantic Lattice Rescoring (Ankitha Sudarshan et al., 2023)

{{<citation>}}

Ankitha Sudarshan, Vinay Samuel, Parth Patwa, Ibtihel Amara, Aman Chadha. (2023)  
**Improved Contextual Recognition In Automatic Speech Recognition Systems By Semantic Lattice Rescoring**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.09680v2)  

---


**ABSTRACT**  
Automatic Speech Recognition (ASR) has witnessed a profound research interest. Recent breakthroughs have given ASR systems different prospects such as faithfully transcribing spoken language, which is a pivotal advancement in building conversational agents. However, there is still an imminent challenge of accurately discerning context-dependent words and phrases. In this work, we propose a novel approach for enhancing contextual recognition within ASR systems via semantic lattice processing leveraging the power of deep learning models in accurately delivering spot-on transcriptions across a wide variety of vocabularies and speaking styles. Our solution consists of using Hidden Markov Models and Gaussian Mixture Models (HMM-GMM) along with Deep Neural Networks (DNN) models integrating both language and acoustic modeling for better accuracy. We infused our network with the use of a transformer-based model to properly rescore the word lattice achieving remarkable capabilities with a palpable reduction in Word Error Rate (WER). We demonstrate the effectiveness of our proposed framework on the LibriSpeech dataset with empirical analyses.

{{</citation>}}


### (2/58) Beyond Testers' Biases: Guiding Model Testing with Knowledge Bases using LLMs (Chenyang Yang et al., 2023)

{{<citation>}}

Chenyang Yang, Rishabh Rustogi, Rachel Brower-Sinning, Grace A. Lewis, Christian Kästner, Tongshuang Wu. (2023)  
**Beyond Testers' Biases: Guiding Model Testing with Knowledge Bases using LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SE, cs.CL  
Keywords: Bias, ChatGPT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2310.09668v1)  

---


**ABSTRACT**  
Current model testing work has mostly focused on creating test cases. Identifying what to test is a step that is largely ignored and poorly supported. We propose Weaver, an interactive tool that supports requirements elicitation for guiding model testing. Weaver uses large language models to generate knowledge bases and recommends concepts from them interactively, allowing testers to elicit requirements for further testing. Weaver provides rich external knowledge to testers and encourages testers to systematically explore diverse concepts beyond their own biases. In a user study, we show that both NLP experts and non-experts identified more, as well as more diverse concepts worth testing when using Weaver. Collectively, they found more than 200 failing test cases for stance detection with zero-shot ChatGPT. Our case studies further show that Weaver can help practitioners test models in real-world settings, where developers define more nuanced application scenarios (e.g., code understanding and transcript summarization) using LLMs.

{{</citation>}}


### (3/58) Legend at ArAIEval Shared Task: Persuasion Technique Detection using a Language-Agnostic Text Representation Model (Olumide E. Ojo et al., 2023)

{{<citation>}}

Olumide E. Ojo, Olaronke O. Adebanji, Hiram Calvo, Damian O. Dieke, Olumuyiwa E. Ojo, Seye E. Akinsanya, Tolulope O. Abiola, Anna Feldman. (2023)  
**Legend at ArAIEval Shared Task: Persuasion Technique Detection using a Language-Agnostic Text Representation Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, BERT, NLP  
[Paper Link](http://arxiv.org/abs/2310.09661v1)  

---


**ABSTRACT**  
In this paper, we share our best performing submission to the Arabic AI Tasks Evaluation Challenge (ArAIEval) at ArabicNLP 2023. Our focus was on Task 1, which involves identifying persuasion techniques in excerpts from tweets and news articles. The persuasion technique in Arabic texts was detected using a training loop with XLM-RoBERTa, a language-agnostic text representation model. This approach proved to be potent, leveraging fine-tuning of a multilingual language model. In our evaluation of the test set, we achieved a micro F1 score of 0.64 for subtask A of the competition.

{{</citation>}}


### (4/58) ASSERT: Automated Safety Scenario Red Teaming for Evaluating the Robustness of Large Language Models (Alex Mei et al., 2023)

{{<citation>}}

Alex Mei, Sharon Levy, William Yang Wang. (2023)  
**ASSERT: Automated Safety Scenario Red Teaming for Evaluating the Robustness of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.09624v1)  

---


**ABSTRACT**  
As large language models are integrated into society, robustness toward a suite of prompts is increasingly important to maintain reliability in a high-variance environment.Robustness evaluations must comprehensively encapsulate the various settings in which a user may invoke an intelligent system. This paper proposes ASSERT, Automated Safety Scenario Red Teaming, consisting of three methods -- semantically aligned augmentation, target bootstrapping, and adversarial knowledge injection. For robust safety evaluation, we apply these methods in the critical domain of AI safety to algorithmically generate a test suite of prompts covering diverse robustness settings -- semantic equivalence, related scenarios, and adversarial. We partition our prompts into four safety domains for a fine-grained analysis of how the domain affects model performance. Despite dedicated safeguards in existing state-of-the-art models, we find statistically significant performance differences of up to 11% in absolute classification accuracy among semantically related scenarios and error rates of up to 19% absolute error in zero-shot adversarial settings, raising concerns for users' physical safety.

{{</citation>}}


### (5/58) A decoder-only foundation model for time-series forecasting (Abhimanyu Das et al., 2023)

{{<citation>}}

Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou. (2023)  
**A decoder-only foundation model for time-series forecasting**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.10688v1)  

---


**ABSTRACT**  
Motivated by recent advances in large language models for Natural Language Processing (NLP), we design a time-series foundation model for forecasting whose out-of-the-box zero-shot performance on a variety of public datasets comes close to the accuracy of state-of-the-art supervised forecasting models for each individual dataset. Our model is based on pretraining a patched-decoder style attention model on a large time-series corpus, and can work well across different forecasting history lengths, prediction lengths and temporal granularities.

{{</citation>}}


### (6/58) Autonomous Tree-search Ability of Large Language Models (Zheyu Zhang et al., 2023)

{{<citation>}}

Zheyu Zhang, Zhuorui Ye, Yikang Shen, Chuang Gan. (2023)  
**Autonomous Tree-search Ability of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10686v1)  

---


**ABSTRACT**  
Large Language Models have excelled in remarkable reasoning capabilities with advanced prompting techniques, but they fall short on tasks that require exploration, strategic foresight, and sequential decision-making. Recent works propose to utilize external programs to define search logic, such that LLMs can perform passive tree search to solve more challenging reasoning tasks. Though impressive results have been achieved, there are several fundamental limitations of these approaches. First, passive tree searches are not efficient as they usually require multiple rounds of LLM API calls to solve one single problem. Moreover, passive search methods are not flexible since they need task-specific program designs. Then a natural question arises: can we maintain the tree-search capability of LLMs without the aid of external programs, and can still generate responses that clearly demonstrate the process of a tree-structure search? To this end, we propose a new concept called autonomous tree-search ability of LLM, which can automatically generate a response containing search trajectories for the correct answer. Concretely, we perform search trajectories using capable LLM API via a fixed system prompt, allowing them to perform autonomous tree-search (ATS) right out of the box. Experiments on 4 puzzle games demonstrate our method can achieve huge improvements. The ATS-BFS method outperforms the Chain of Thought approach by achieving an average accuracy improvement of 33%. Compared to Tree of Thoughts, it requires 65.6% or 47.7% less GPT-api cost to attain a comparable level of accuracy. Moreover, we have collected data using the ATS prompt method and fine-tuned LLaMA. This approach yield a greater improvement compared to the ones fine-tuned on CoT data. Specifically, it outperforms CoT-tuned LLaMAs by an average of 40.6% and 38.5% for LLaMA2-7B and LLaMA2-13B, respectively.

{{</citation>}}


### (7/58) Self-Detoxifying Language Models via Toxification Reversal (Chak Tou Leong et al., 2023)

{{<citation>}}

Chak Tou Leong, Yi Cheng, Jiashuo Wang, Jian Wang, Wenjie Li. (2023)  
**Self-Detoxifying Language Models via Toxification Reversal**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.09573v1)  

---


**ABSTRACT**  
Language model detoxification aims to minimize the risk of generating offensive or harmful content in pretrained language models (PLMs) for safer deployment. Existing methods can be roughly categorized as finetuning-based and decoding-based. However, the former is often resource-intensive, while the latter relies on additional components and potentially compromises the generation fluency. In this paper, we propose a more lightweight approach that enables the PLM itself to achieve "self-detoxification". Our method is built upon the observation that prepending a negative steering prompt can effectively induce PLMs to generate toxic content. At the same time, we are inspired by the recent research in the interpretability field, which formulates the evolving contextualized representations within the PLM as an information stream facilitated by the attention layers. Drawing on this idea, we devise a method to identify the toxification direction from the normal generation process to the one prompted with the negative prefix, and then steer the generation to the reversed direction by manipulating the information movement within the attention layers. Experimental results show that our approach, without any fine-tuning or extra components, can achieve comparable performance with state-of-the-art methods.

{{</citation>}}


### (8/58) Can Large Language Model Comprehend Ancient Chinese? A Preliminary Test on ACLUE (Yixuan Zhang et al., 2023)

{{<citation>}}

Yixuan Zhang, Haonan Li. (2023)  
**Can Large Language Model Comprehend Ancient Chinese? A Preliminary Test on ACLUE**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLM, Language Model  
[Paper Link](http://arxiv.org/abs/2310.09550v1)  

---


**ABSTRACT**  
Large language models (LLMs) have showcased remarkable capabilities in understanding and generating language. However, their ability in comprehending ancient languages, particularly ancient Chinese, remains largely unexplored. To bridge this gap, we present ACLUE, an evaluation benchmark designed to assess the capability of language models in comprehending ancient Chinese. ACLUE consists of 15 tasks cover a range of skills, spanning phonetic, lexical, syntactic, semantic, inference and knowledge. Through the evaluation of eight state-of-the-art LLMs, we observed a noticeable disparity in their performance between modern Chinese and ancient Chinese. Among the assessed models, ChatGLM2 demonstrates the most remarkable performance, achieving an average score of 37.4%. We have made our code and data public available.

{{</citation>}}


### (9/58) CarExpert: Leveraging Large Language Models for In-Car Conversational Question Answering (Md Rashad Al Hasan Rony et al., 2023)

{{<citation>}}

Md Rashad Al Hasan Rony, Christian Suess, Sinchana Ramakanth Bhat, Viju Sudhi, Julia Schneider, Maximilian Vogel, Roman Teucher, Ken E. Friedl, Soumya Sahoo. (2023)  
**CarExpert: Leveraging Large Language Models for In-Car Conversational Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.09536v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated remarkable performance by following natural language instructions without fine-tuning them on domain-specific tasks and data. However, leveraging LLMs for domain-specific question answering suffers from severe limitations. The generated answer tends to hallucinate due to the training data collection time (when using off-the-shelf), complex user utterance and wrong retrieval (in retrieval-augmented generation). Furthermore, due to the lack of awareness about the domain and expected output, such LLMs may generate unexpected and unsafe answers that are not tailored to the target domain. In this paper, we propose CarExpert, an in-car retrieval-augmented conversational question-answering system leveraging LLMs for different tasks. Specifically, CarExpert employs LLMs to control the input, provide domain-specific documents to the extractive and generative answering components, and controls the output to ensure safe and domain-specific answers. A comprehensive empirical evaluation exhibits that CarExpert outperforms state-of-the-art LLMs in generating natural, safe and car-specific answers.

{{</citation>}}


### (10/58) Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model (Haikang Deng et al., 2023)

{{<citation>}}

Haikang Deng, Colin Raffel. (2023)  
**Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2310.09520v2)  

---


**ABSTRACT**  
While large language models have proven effective in a huge range of downstream applications, they often generate text that is problematic or lacks a desired attribute. In this paper, we introduce Reward-Augmented Decoding (RAD), a text generation procedure that uses a small unidirectional reward model to encourage a language model to generate text that has certain properties. Specifically, RAD uses the reward model to score generations as they are produced and rescales sampling probabilities to favor high-reward tokens. By using a unidirectional reward model, RAD can cache activations from prior generation steps to decrease computational overhead. Through experiments on generating non-toxic and sentiment-controlled text, we demonstrate that RAD performs best among methods that change only the generation procedure and matches the performance of state-of-the-art methods that involve re-training the language model. We further validate that RAD is effective on very large language models while incurring a minimal computational overhead.

{{</citation>}}


### (11/58) Instruction Tuning with Human Curriculum (Bruce W. Lee et al., 2023)

{{<citation>}}

Bruce W. Lee, Hyunsoo Cho, Kang Min Yoo. (2023)  
**Instruction Tuning with Human Curriculum**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.09518v1)  

---


**ABSTRACT**  
The dominant paradigm for instruction tuning is the random-shuffled training of maximally diverse instruction-response pairs. This paper explores the potential benefits of applying a structured cognitive learning approach to instruction tuning in contemporary large language models like ChatGPT and GPT-4. Unlike the previous conventional randomized instruction dataset, we propose a highly structured synthetic dataset that mimics the progressive and organized nature of human education. We curate our dataset by aligning it with educational frameworks, incorporating meta information including its topic and cognitive rigor level for each sample. Our dataset covers comprehensive fine-grained topics spanning diverse educational stages (from middle school to graduate school) with various questions for each topic to enhance conceptual depth using Bloom's taxonomy-a classification framework distinguishing various levels of human cognition for each concept. The results demonstrate that this cognitive rigorous training approach yields significant performance enhancements - +3.06 on the MMLU benchmark and an additional +1.28 on AI2 Reasoning Challenge (hard set) - compared to conventional randomized training, all while avoiding additional computational costs. This research highlights the potential of leveraging human learning principles to enhance the capabilities of language models in comprehending and responding to complex instructions and tasks.

{{</citation>}}


### (12/58) One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models (Hang Shao et al., 2023)

{{<citation>}}

Hang Shao, Bei Liu, Yanmin Qian. (2023)  
**One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model, Pruning, Transformer  
[Paper Link](http://arxiv.org/abs/2310.09499v1)  

---


**ABSTRACT**  
Various Large Language Models(LLMs) from the Generative Pretrained Transformer~(GPT) family have achieved outstanding performances in a wide range of text generation tasks. However, the enormous model sizes have hindered their practical use in real-world applications due to high inference latency. Therefore, improving the efficiencies of LLMs through quantization, pruning, and other means has been a key issue in LLM studies. In this work, we propose a method based on Hessian sensitivity-aware mixed sparsity pruning to prune LLMs to at least 50\% sparsity without the need of any retraining. It allocates sparsity adaptively based on sensitivity, allowing us to reduce pruning-induced error while maintaining the overall sparsity level. The advantages of the proposed method exhibit even more when the sparsity is extremely high. Furthermore, our method is compatible with quantization, enabling further compression of LLMs.

{{</citation>}}


### (13/58) Large Language Model Unlearning (Yuanshun Yao et al., 2023)

{{<citation>}}

Yuanshun Yao, Xiaojun Xu, Yang Liu. (2023)  
**Large Language Model Unlearning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.10683v1)  

---


**ABSTRACT**  
We study how to perform unlearning, i.e. forgetting undesirable (mis)behaviors, on large language models (LLMs). We show at least three scenarios of aligning LLMs with human preferences can benefit from unlearning: (1) removing harmful responses, (2) erasing copyright-protected content as requested, and (3) eliminating hallucinations. Unlearning, as an alignment technique, has three advantages. (1) It only requires negative (e.g. harmful) examples, which are much easier and cheaper to collect (e.g. via red teaming or user reporting) than positive (e.g. helpful and often human-written) examples required in RLHF (RL from human feedback). (2) It is computationally efficient. (3) It is especially effective when we know which training samples cause the misbehavior. To the best of our knowledge, our work is among the first to explore LLM unlearning. We are also among the first to formulate the settings, goals, and evaluations in LLM unlearning. We show that if practitioners only have limited resources, and therefore the priority is to stop generating undesirable outputs rather than to try to generate desirable outputs, unlearning is particularly appealing. Despite only having negative samples, our ablation study shows that unlearning can still achieve better alignment performance than RLHF with just 2% of its computational time.

{{</citation>}}


## cs.CV (12)



### (14/58) What Do Deep Saliency Models Learn about Visual Attention? (Shi Chen et al., 2023)

{{<citation>}}

Shi Chen, Ming Jiang, Qi Zhao. (2023)  
**What Do Deep Saliency Models Learn about Visual Attention?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.09679v1)  

---


**ABSTRACT**  
In recent years, deep saliency models have made significant progress in predicting human visual attention. However, the mechanisms behind their success remain largely unexplained due to the opaque nature of deep neural networks. In this paper, we present a novel analytic framework that sheds light on the implicit features learned by saliency models and provides principled interpretation and quantification of their contributions to saliency prediction. Our approach decomposes these implicit features into interpretable bases that are explicitly aligned with semantic attributes and reformulates saliency prediction as a weighted combination of probability maps connecting the bases and saliency. By applying our framework, we conduct extensive analyses from various perspectives, including the positive and negative weights of semantics, the impact of training data and architectural designs, the progressive influences of fine-tuning, and common failure patterns of state-of-the-art deep saliency models. Additionally, we demonstrate the effectiveness of our framework by exploring visual attention characteristics in various application scenarios, such as the atypical attention of people with autism spectrum disorder, attention to emotion-eliciting stimuli, and attention evolution over time. Our code is publicly available at \url{https://github.com/szzexpoi/saliency_analysis}.

{{</citation>}}


### (15/58) Does CLIP's Generalization Performance Mainly Stem from High Train-Test Similarity? (Prasanna Mayilvahanan et al., 2023)

{{<citation>}}

Prasanna Mayilvahanan, Thaddäus Wiedemer, Evgenia Rusak, Matthias Bethge, Wieland Brendel. (2023)  
**Does CLIP's Generalization Performance Mainly Stem from High Train-Test Similarity?**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.09562v1)  

---


**ABSTRACT**  
Foundation models like CLIP are trained on hundreds of millions of samples and effortlessly generalize to new tasks and inputs. Out of the box, CLIP shows stellar zero-shot and few-shot capabilities on a wide range of out-of-distribution (OOD) benchmarks, which prior works attribute mainly to today's large and comprehensive training dataset (like LAION). However, it is questionable how meaningful terms like out-of-distribution generalization are for CLIP as it seems likely that web-scale datasets like LAION simply contain many samples that are similar to common OOD benchmarks originally designed for ImageNet. To test this hypothesis, we retrain CLIP on pruned LAION splits that replicate ImageNet's train-test similarity with respect to common OOD benchmarks. While we observe a performance drop on some benchmarks, surprisingly, CLIP's overall performance remains high. This shows that high train-test similarity is insufficient to explain CLIP's OOD performance, and other properties of the training data must drive CLIP to learn more generalizable representations. Additionally, by pruning data points that are dissimilar to the OOD benchmarks, we uncover a 100M split of LAION ($\frac{1}{4}$th of its original size) on which CLIP can be trained to match its original OOD performance.

{{</citation>}}


### (16/58) UNIQA: A Unified Framework for Both Full-Reference and No-Reference Image Quality Assessment (Yi Ke Yun et al., 2023)

{{<citation>}}

Yi Ke Yun, Weisi Lin. (2023)  
**UNIQA: A Unified Framework for Both Full-Reference and No-Reference Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV, eess-IV  
Keywords: Attention, QA, Self-Attention  
[Paper Link](http://arxiv.org/abs/2310.09560v1)  

---


**ABSTRACT**  
The human visual system (HVS) is effective at distinguishing low-quality images due to its ability to sense the distortion level and the resulting semantic impact. Prior research focuses on developing dedicated networks based on the presence and absence of pristine images, respectively, and this results in limited application scope and potential performance inconsistency when switching from NR to FR IQA. In addition, most methods heavily rely on spatial distortion modeling through difference maps or weighted features, and this may not be able to well capture the correlations between distortion and the semantic impact it causes. To this end, we aim to design a unified network for both Full-Reference (FR) and No-Reference (NR) IQA via semantic impact modeling. Specifically, we employ an encoder to extract multi-level features from input images. Then a Hierarchical Self-Attention (HSA) module is proposed as a universal adapter for both FR and NR inputs to model the spatial distortion level at each encoder stage. Furthermore, considering that distortions contaminate encoder stages and damage image semantic meaning differently, a Cross-Scale Cross-Attention (CSCA) module is proposed to examine correlations between distortion at shallow stages and deep ones. By adopting HSA and CSCA, the proposed network can effectively perform both FR and NR IQA. Extensive experiments demonstrate that the proposed simple network is effective and outperforms the relevant state-of-the-art FR and NR methods on four synthetic-distorted datasets and three authentic-distorted datasets.

{{</citation>}}


### (17/58) Scene Text Recognition Models Explainability Using Local Features (Mark Vincent Ty et al., 2023)

{{<citation>}}

Mark Vincent Ty, Rowel Atienza. (2023)  
**Scene Text Recognition Models Explainability Using Local Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09549v1)  

---


**ABSTRACT**  
Explainable AI (XAI) is the study on how humans can be able to understand the cause of a model's prediction. In this work, the problem of interest is Scene Text Recognition (STR) Explainability, using XAI to understand the cause of an STR model's prediction. Recent XAI literatures on STR only provide a simple analysis and do not fully explore other XAI methods. In this study, we specifically work on data explainability frameworks, called attribution-based methods, that explain the important parts of an input data in deep learning models. However, integrating them into STR produces inconsistent and ineffective explanations, because they only explain the model in the global context. To solve this problem, we propose a new method, STRExp, to take into consideration the local explanations, i.e. the individual character prediction explanations. This is then benchmarked across different attribution-based methods on different STR datasets and evaluated across different STR models.

{{</citation>}}


### (18/58) Towards End-to-End Unsupervised Saliency Detection with Self-Supervised Top-Down Context (Yicheng Song et al., 2023)

{{<citation>}}

Yicheng Song, Shuyong Gao, Haozhe Xing, Yiting Cheng, Yan Wang, Wenqiang Zhang. (2023)  
**Towards End-to-End Unsupervised Saliency Detection with Self-Supervised Top-Down Context**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.09533v1)  

---


**ABSTRACT**  
Unsupervised salient object detection aims to detect salient objects without using supervision signals eliminating the tedious task of manually labeling salient objects. To improve training efficiency, end-to-end methods for USOD have been proposed as a promising alternative. However, current solutions rely heavily on noisy handcraft labels and fail to mine rich semantic information from deep features. In this paper, we propose a self-supervised end-to-end salient object detection framework via top-down context. Specifically, motivated by contrastive learning, we exploit the self-localization from the deepest feature to construct the location maps which are then leveraged to learn the most instructive segmentation guidance. Further considering the lack of detailed information in deepest features, we exploit the detail-boosting refiner module to enrich the location labels with details. Moreover, we observe that due to lack of supervision, current unsupervised saliency models tend to detect non-salient objects that are salient in some other samples of corresponding scenarios. To address this widespread issue, we design a novel Unsupervised Non-Salient Suppression (UNSS) method developing the ability to ignore non-salient objects. Extensive experiments on benchmark datasets demonstrate that our method achieves leading performance among the recent end-to-end methods and most of the multi-stage solutions. The code is available.

{{</citation>}}


### (19/58) TS-ENAS:Two-Stage Evolution for Cell-based Network Architecture Search (Juan Zou et al., 2023)

{{<citation>}}

Juan Zou, Shenghong Wu, Yizhang Xia, Weiwei Jiang, Zeping Wu, Jinhua Zheng. (2023)  
**TS-ENAS:Two-Stage Evolution for Cell-based Network Architecture Search**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.09525v1)  

---


**ABSTRACT**  
Neural network architecture search provides a solution to the automatic design of network structures. However, it is difficult to search the whole network architecture directly. Although using stacked cells to search neural network architectures is an effective way to reduce the complexity of searching, these methods do not able find the global optimal neural network structure since the number of layers, cells and connection methods is fixed. In this paper, we propose a Two-Stage Evolution for cell-based Network Architecture Search(TS-ENAS), including one-stage searching based on stacked cells and second-stage adjusting these cells. In our algorithm, a new cell-based search space and an effective two-stage encoding method are designed to represent cells and neural network structures. In addition, a cell-based weight inheritance strategy is designed to initialize the weight of the network, which significantly reduces the running time of the algorithm. The proposed methods are extensively tested and compared on four image classification dataset, Fashion-MNIST, CIFAR10, CIFAR100 and ImageNet and compared with 22 state-of-the-art algorithms including hand-designed networks and NAS networks. The experimental results show that TS-ENAS can more effectively find the neural network architecture with comparative performance.

{{</citation>}}


### (20/58) Foundation Ark: Accruing and Reusing Knowledge for Superior and Robust Performance (DongAo Ma et al., 2023)

{{<citation>}}

DongAo Ma, Jiaxuan Pang, Michael B. Gotway, Jianming Liang. (2023)  
**Foundation Ark: Accruing and Reusing Knowledge for Superior and Robust Performance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.09507v1)  

---


**ABSTRACT**  
Deep learning nowadays offers expert-level and sometimes even super-expert-level performance, but achieving such performance demands massive annotated data for training (e.g., Google's proprietary CXR Foundation Model (CXR-FM) was trained on 821,544 labeled and mostly private chest X-rays (CXRs)). Numerous datasets are publicly available in medical imaging but individually small and heterogeneous in expert labels. We envision a powerful and robust foundation model that can be trained by aggregating numerous small public datasets. To realize this vision, we have developed Ark, a framework that accrues and reuses knowledge from heterogeneous expert annotations in various datasets. As a proof of concept, we have trained two Ark models on 335,484 and 704,363 CXRs, respectively, by merging several datasets including ChestX-ray14, CheXpert, MIMIC-II, and VinDr-CXR, evaluated them on a wide range of imaging tasks covering both classification and segmentation via fine-tuning, linear-probing, and gender-bias analysis, and demonstrated our Ark's superior and robust performance over the SOTA fully/self-supervised baselines and Google's proprietary CXR-FM. This enhanced performance is attributed to our simple yet powerful observation that aggregating numerous public datasets diversifies patient populations and accrues knowledge from diverse experts, yielding unprecedented performance yet saving annotation cost. With all codes and pretrained models released at GitHub.com/JLiangLab/Ark, we hope that Ark exerts an important impact on open science, as accruing and reusing knowledge from expert annotations in public datasets can potentially surpass the performance of proprietary models trained on unusually large data, inspiring many more researchers worldwide to share codes and datasets to build open foundation models, accelerate open science, and democratize deep learning for medical imaging.

{{</citation>}}


### (21/58) Perception Reinforcement Using Auxiliary Learning Feature Fusion: A Modified Yolov8 for Head Detection (Jiezhou Chen et al., 2023)

{{<citation>}}

Jiezhou Chen, Guankun Wang, Weixiang Liu, Xiaopin Zhong, Yibin Tian, ZongZe Wu. (2023)  
**Perception Reinforcement Using Auxiliary Learning Feature Fusion: A Modified Yolov8 for Head Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: LSTM, Yolo  
[Paper Link](http://arxiv.org/abs/2310.09492v1)  

---


**ABSTRACT**  
Head detection provides distribution information of pedestrian, which is crucial for scene statistical analysis, traffic management, and risk assessment and early warning. However, scene complexity and large-scale variation in the real world make accurate detection more difficult. Therefore, we present a modified Yolov8 which improves head detection performance through reinforcing target perception. An Auxiliary Learning Feature Fusion (ALFF) module comprised of LSTM and convolutional blocks is used as the auxiliary task to help the model perceive targets. In addition, we introduce Noise Calibration into Distribution Focal Loss to facilitate model fitting and improve the accuracy of detection. Considering the requirements of high accuracy and speed for the head detection task, our method is adapted with two kinds of backbone, namely Yolov8n and Yolov8m. The results demonstrate the superior performance of our approach in improving detection accuracy and robustness.

{{</citation>}}


### (22/58) Unified High-binding Watermark for Unconditional Image Generation Models (Ruinan Ma et al., 2023)

{{<citation>}}

Ruinan Ma, Yu-an Tan, Shangbo Wu, Tian Chen, Yajie Wang, Yuanzhang Li. (2023)  
**Unified High-binding Watermark for Unconditional Image Generation Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09479v1)  

---


**ABSTRACT**  
Deep learning techniques have implemented many unconditional image generation (UIG) models, such as GAN, Diffusion model, etc. The extremely realistic images (also known as AI-Generated Content, AIGC for short) produced by these models bring urgent needs for intellectual property protection such as data traceability and copyright certification. An attacker can steal the output images of the target model and use them as part of the training data to train a private surrogate UIG model. The implementation mechanisms of UIG models are diverse and complex, and there is no unified and effective protection and verification method at present. To address these issues, we propose a two-stage unified watermark verification mechanism with high-binding effects for such models. In the first stage, we use an encoder to invisibly write the watermark image into the output images of the original AIGC tool, and reversely extract the watermark image through the corresponding decoder. In the second stage, we design the decoder fine-tuning process, and the fine-tuned decoder can make correct judgments on whether the suspicious model steals the original AIGC tool data. Experiments demonstrate our method can complete the verification work with almost zero false positive rate under the condition of only using the model output images. Moreover, the proposed method can achieve data steal verification across different types of UIG models, which further increases the practicality of the method.

{{</citation>}}


### (23/58) MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning (Jun Chen et al., 2023)

{{<citation>}}

Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, Mohamed Elhoseiny. (2023)  
**MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.09478v1)  

---


**ABSTRACT**  
Large language models have shown their remarkable capabilities as a general interface for various language-related applications. Motivated by this, we target to build a unified interface for completing many vision-language tasks including image description, visual question answering, and visual grounding, among others. The challenge is to use a single model for performing diverse vision-language tasks effectively with simple multi-modal instructions. Towards this objective, we introduce MiniGPT-v2, a model that can be treated as a unified interface for better handling various vision-language tasks. We propose using unique identifiers for different tasks when training the model. These identifiers enable our model to better distinguish each task instruction effortlessly and also improve the model learning efficiency for each task. After the three-stage training, the experimental results show that MiniGPT-v2 achieves strong performance on many visual question-answering and visual grounding benchmarks compared to other vision-language generalist models. Our model and codes are available at https://minigpt-v2.github.io/

{{</citation>}}


### (24/58) Plug-and-Play Feature Generation for Few-Shot Medical Image Classification (Qianyu Guo et al., 2023)

{{<citation>}}

Qianyu Guo, Huifang Du, Xing Jia, Shuyong Gao, Yan Teng, Haofen Wang, Wenqiang Zhang. (2023)  
**Plug-and-Play Feature Generation for Few-Shot Medical Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Image Classification  
[Paper Link](http://arxiv.org/abs/2310.09471v1)  

---


**ABSTRACT**  
Few-shot learning (FSL) presents immense potential in enhancing model generalization and practicality for medical image classification with limited training data; however, it still faces the challenge of severe overfitting in classifier training due to distribution bias caused by the scarce training samples. To address the issue, we propose MedMFG, a flexible and lightweight plug-and-play method designed to generate sufficient class-distinctive features from limited samples. Specifically, MedMFG first re-represents the limited prototypes to assign higher weights for more important information features. Then, the prototypes are variationally generated into abundant effective features. Finally, the generated features and prototypes are together to train a more generalized classifier. Experiments demonstrate that MedMFG outperforms the previous state-of-the-art methods on cross-domain benchmarks involving the transition from natural images to medical images, as well as medical images with different lesions. Notably, our method achieves over 10% performance improvement compared to several baselines. Fusion experiments further validate the adaptability of MedMFG, as it seamlessly integrates into various backbones and baselines, consistently yielding improvements of over 2.9% across all results.

{{</citation>}}


### (25/58) MAC: ModAlity Calibration for Object Detection (Yutian Lei et al., 2023)

{{<citation>}}

Yutian Lei, Jun Liu, Dong Huang. (2023)  
**MAC: ModAlity Calibration for Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.09461v1)  

---


**ABSTRACT**  
The flourishing success of Deep Neural Networks(DNNs) on RGB-input perception tasks has opened unbounded possibilities for non-RGB-input perception tasks, such as object detection from wireless signals, lidar scans, and infrared images. Compared to the matured development pipeline of RGB-input (source modality) models, developing non-RGB-input (target-modality) models from scratch poses excessive challenges in the modality-specific network design/training tricks and labor in the target-modality annotation. In this paper, we propose ModAlity Calibration (MAC), an efficient pipeline for calibrating target-modality inputs to the DNN object detection models developed on the RGB (source) modality. We compose a target-modality-input model by adding a small calibrator module ahead of a source-modality model and introduce MAC training techniques to impose dense supervision on the calibrator. By leveraging (1) prior knowledge synthesized from the source-modality model and (2) paired {target, source} data with zero manual annotations, our target-modality models reach comparable or better metrics than baseline models that require 100% manual annotations. We demonstrate the effectiveness of MAC by composing the WiFi-input, Lidar-input, and Thermal-Infrared-input models upon the pre-trained RGB-input models respectively.

{{</citation>}}


## cs.LG (13)



### (26/58) Efficient Model-Agnostic Multi-Group Equivariant Networks (Razan Baltaji et al., 2023)

{{<citation>}}

Razan Baltaji, Sourya Basu, Lav R. Varshney. (2023)  
**Efficient Model-Agnostic Multi-Group Equivariant Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.09675v1)  

---


**ABSTRACT**  
Constructing model-agnostic group equivariant networks, such as equitune (Basu et al., 2023b) and its generalizations (Kim et al., 2023), can be computationally expensive for large product groups. We address this by providing efficient model-agnostic equivariant designs for two related problems: one where the network has multiple inputs each with potentially different groups acting on them, and another where there is a single input but the group acting on it is a large product group. For the first design, we initially consider a linear model and characterize the entire equivariant space that satisfies this constraint. This characterization gives rise to a novel fusion layer between different channels that satisfies an invariance-symmetry (IS) constraint, which we call an IS layer. We then extend this design beyond linear models, similar to equitune, consisting of equivariant and IS layers. We also show that the IS layer is a universal approximator of invariant-symmetric functions. Inspired by the first design, we use the notion of the IS property to design a second efficient model-agnostic equivariant design for large product groups acting on a single input. For the first design, we provide experiments on multi-image classification where each view is transformed independently with transformations such as rotations. We find equivariant models are robust to such transformations and perform competitively otherwise. For the second design, we consider three applications: language compositionality on the SCAN dataset to product groups; fairness in natural language generation from GPT-2 to address intersectionality; and robust zero-shot image classification with CLIP. Overall, our methods are simple and general, competitive with equitune and its variants, while also being computationally more efficient.

{{</citation>}}


### (27/58) Towards Semi-Structured Automatic ICD Coding via Tree-based Contrastive Learning (Chang Lu et al., 2023)

{{<citation>}}

Chang Lu, Chandan K. Reddy, Ping Wang, Yue Ning. (2023)  
**Towards Semi-Structured Automatic ICD Coding via Tree-based Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning, NLP  
[Paper Link](http://arxiv.org/abs/2310.09672v1)  

---


**ABSTRACT**  
Automatic coding of International Classification of Diseases (ICD) is a multi-label text categorization task that involves extracting disease or procedure codes from clinical notes. Despite the application of state-of-the-art natural language processing (NLP) techniques, there are still challenges including limited availability of data due to privacy constraints and the high variability of clinical notes caused by different writing habits of medical professionals and various pathological features of patients. In this work, we investigate the semi-structured nature of clinical notes and propose an automatic algorithm to segment them into sections. To address the variability issues in existing ICD coding models with limited data, we introduce a contrastive pre-training approach on sections using a soft multi-label similarity metric based on tree edit distance. Additionally, we design a masked section training strategy to enable ICD coding models to locate sections related to ICD codes. Extensive experimental results demonstrate that our proposed training strategies effectively enhance the performance of existing ICD coding methods.

{{</citation>}}


### (28/58) Topology-guided Hypergraph Transformer Network: Unveiling Structural Insights for Improved Representation (Khaled Mohammed Saifuddin et al., 2023)

{{<citation>}}

Khaled Mohammed Saifuddin, Mehmet Emin Aktas, Esra Akbas. (2023)  
**Topology-guided Hypergraph Transformer Network: Unveiling Structural Insights for Improved Representation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2310.09657v1)  

---


**ABSTRACT**  
Hypergraphs, with their capacity to depict high-order relationships, have emerged as a significant extension of traditional graphs. Although Graph Neural Networks (GNNs) have remarkable performance in graph representation learning, their extension to hypergraphs encounters challenges due to their intricate structures. Furthermore, current hypergraph transformers, a special variant of GNN, utilize semantic feature-based self-attention, ignoring topological attributes of nodes and hyperedges. To address these challenges, we propose a Topology-guided Hypergraph Transformer Network (THTN). In this model, we first formulate a hypergraph from a graph while retaining its structural essence to learn higher-order relations within the graph. Then, we design a simple yet effective structural and spatial encoding module to incorporate the topological and spatial information of the nodes into their representation. Further, we present a structure-aware self-attention mechanism that discovers the important nodes and hyperedges from both semantic and structural viewpoints. By leveraging these two modules, THTN crafts an improved node representation, capturing both local and global topological expressions. Extensive experiments conducted on node classification tasks demonstrate that the performance of the proposed model consistently exceeds that of the existing approaches.

{{</citation>}}


### (29/58) Multimodal Federated Learning in Healthcare: a review (Jacob Thrasher et al., 2023)

{{<citation>}}

Jacob Thrasher, Alina Devkota, Prasiddha Siwakotai, Rohit Chivukula, Pranav Poudel, Chaunbo Hu, Binod Bhattarai, Prashnna Gyawali. (2023)  
**Multimodal Federated Learning in Healthcare: a review**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09650v1)  

---


**ABSTRACT**  
Recent advancements in multimodal machine learning have empowered the development of accurate and robust AI systems in the medical domain, especially within centralized database systems. Simultaneously, Federated Learning (FL) has progressed, providing a decentralized mechanism where data need not be consolidated, thereby enhancing the privacy and security of sensitive healthcare data. The integration of these two concepts supports the ongoing progress of multimodal learning in healthcare while ensuring the security and privacy of patient records within local data-holding agencies. This paper offers a concise overview of the significance of FL in healthcare and outlines the current state-of-the-art approaches to Multimodal Federated Learning (MMFL) within the healthcare domain. It comprehensively examines the existing challenges in the field, shedding light on the limitations of present models. Finally, the paper outlines potential directions for future advancements in the field, aiming to bridge the gap between cutting-edge AI technology and the imperative need for patient data privacy in healthcare applications.

{{</citation>}}


### (30/58) Generative Adversarial Training for Text-to-Speech Synthesis Based on Raw Phonetic Input and Explicit Prosody Modelling (Tiberiu Boros et al., 2023)

{{<citation>}}

Tiberiu Boros, Stefan Daniel Dumitrescu, Ionut Mironica, Radu Chivereanu. (2023)  
**Generative Adversarial Training for Text-to-Speech Synthesis Based on Raw Phonetic Input and Explicit Prosody Modelling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2310.09636v1)  

---


**ABSTRACT**  
We describe an end-to-end speech synthesis system that uses generative adversarial training. We train our Vocoder for raw phoneme-to-audio conversion, using explicit phonetic, pitch and duration modeling. We experiment with several pre-trained models for contextualized and decontextualized word embeddings and we introduce a new method for highly expressive character voice matching, based on discreet style tokens.

{{</citation>}}


### (31/58) STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning (Weipu Zhang et al., 2023)

{{<citation>}}

Weipu Zhang, Gang Wang, Jian Sun, Yetian Yuan, Gao Huang. (2023)  
**STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.09615v1)  

---


**ABSTRACT**  
Recently, model-based reinforcement learning algorithms have demonstrated remarkable efficacy in visual input environments. These approaches begin by constructing a parameterized simulation world model of the real environment through self-supervised learning. By leveraging the imagination of the world model, the agent's policy is enhanced without the constraints of sampling from the real environment. The performance of these algorithms heavily relies on the sequence modeling and generation capabilities of the world model. However, constructing a perfectly accurate model of a complex unknown environment is nearly impossible. Discrepancies between the model and reality may cause the agent to pursue virtual goals, resulting in subpar performance in the real environment. Introducing random noise into model-based reinforcement learning has been proven beneficial. In this work, we introduce Stochastic Transformer-based wORld Model (STORM), an efficient world model architecture that combines the strong sequence modeling and generation capabilities of Transformers with the stochastic nature of variational autoencoders. STORM achieves a mean human performance of $126.7\%$ on the Atari $100$k benchmark, setting a new record among state-of-the-art methods that do not employ lookahead search techniques. Moreover, training an agent with $1.85$ hours of real-time interaction experience on a single NVIDIA GeForce RTX 3090 graphics card requires only $4.3$ hours, showcasing improved efficiency compared to previous methodologies.

{{</citation>}}


### (32/58) Causality and Independence Enhancement for Biased Node Classification (Guoxin Chen et al., 2023)

{{<citation>}}

Guoxin Chen, Yongqing Wang, Fangda Guo, Qinglang Guo, Jiangli Shao, Huawei Shen, Xueqi Cheng. (2023)  
**Causality and Independence Enhancement for Biased Node Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias, GNN  
[Paper Link](http://arxiv.org/abs/2310.09586v1)  

---


**ABSTRACT**  
Most existing methods that address out-of-distribution (OOD) generalization for node classification on graphs primarily focus on a specific type of data biases, such as label selection bias or structural bias. However, anticipating the type of bias in advance is extremely challenging, and designing models solely for one specific type may not necessarily improve overall generalization performance. Moreover, limited research has focused on the impact of mixed biases, which are more prevalent and demanding in real-world scenarios. To address these limitations, we propose a novel Causality and Independence Enhancement (CIE) framework, applicable to various graph neural networks (GNNs). Our approach estimates causal and spurious features at the node representation level and mitigates the influence of spurious correlations through the backdoor adjustment. Meanwhile, independence constraint is introduced to improve the discriminability and stability of causal and spurious features in complex biased environments. Essentially, CIE eliminates different types of data biases from a unified perspective, without the need to design separate methods for each bias as before. To evaluate the performance under specific types of data biases, mixed biases, and low-resource scenarios, we conducted comprehensive experiments on five publicly available datasets. Experimental results demonstrate that our approach CIE not only significantly enhances the performance of GNNs but outperforms state-of-the-art debiased node classification methods.

{{</citation>}}


### (33/58) Graph Neural Network approaches for single-cell data: A recent overview (Konstantinos Lazaros et al., 2023)

{{<citation>}}

Konstantinos Lazaros, Dimitris E. Koumadorakis, Panagiotis Vlamos, Aristidis G. Vrahatis. (2023)  
**Graph Neural Network approaches for single-cell data: A recent overview**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-GN  
Keywords: Attention, GNN, Graph Attention Network, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.09561v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNN) are reshaping our understanding of biomedicine and diseases by revealing the deep connections among genes and cells. As both algorithmic and biomedical technologies have advanced significantly, we're entering a transformative phase of personalized medicine. While pioneering tools like Graph Attention Networks (GAT) and Graph Convolutional Neural Networks (Graph CNN) are advancing graph-based learning, the rise of single-cell sequencing techniques is reshaping our insights on cellular diversity and function. Numerous studies have combined GNNs with single-cell data, showing promising results. In this work, we highlight the GNN methodologies tailored for single-cell data over the recent years. We outline the diverse range of graph deep learning architectures that center on GAT methodologies. Furthermore, we underscore the several objectives of GNN strategies in single-cell data contexts, ranging from cell-type annotation, data integration and imputation, gene regulatory network reconstruction, clustering and many others. This review anticipates a future where GNNs become central to single-cell analysis efforts, particularly as vast omics datasets are continuously generated and the interconnectedness of cells and genes enhances our depth of knowledge in biomedicine.

{{</citation>}}


### (34/58) Protein 3D Graph Structure Learning for Robust Structure-based Protein Property Prediction (Yufei Huang et al., 2023)

{{<citation>}}

Yufei Huang, Siyuan Li, Jin Su, Lirong Wu, Odin Zhang, Haitao Lin, Jingqi Qi, Zihan Liu, Zhangyang Gao, Yuyang Liu, Jiangbin Zheng, Stan. ZQ. Li. (2023)  
**Protein 3D Graph Structure Learning for Robust Structure-based Protein Property Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-QM  
Keywords: AI, Computer Vision  
[Paper Link](http://arxiv.org/abs/2310.11466v2)  

---


**ABSTRACT**  
Protein structure-based property prediction has emerged as a promising approach for various biological tasks, such as protein function prediction and sub-cellular location estimation. The existing methods highly rely on experimental protein structure data and fail in scenarios where these data are unavailable. Predicted protein structures from AI tools (e.g., AlphaFold2) were utilized as alternatives. However, we observed that current practices, which simply employ accurately predicted structures during inference, suffer from notable degradation in prediction accuracy. While similar phenomena have been extensively studied in general fields (e.g., Computer Vision) as model robustness, their impact on protein property prediction remains unexplored. In this paper, we first investigate the reason behind the performance decrease when utilizing predicted structures, attributing it to the structure embedding bias from the perspective of structure representation learning. To study this problem, we identify a Protein 3D Graph Structure Learning Problem for Robust Protein Property Prediction (PGSL-RP3), collect benchmark datasets, and present a protein Structure embedding Alignment Optimization framework (SAO) to mitigate the problem of structure embedding bias between the predicted and experimental protein structures. Extensive experiments have shown that our framework is model-agnostic and effective in improving the property prediction of both predicted structures and experimental structures. The benchmark datasets and codes will be released to benefit the community.

{{</citation>}}


### (35/58) Efficient Link Prediction via GNN Layers Induced by Negative Sampling (Yuxin Wang et al., 2023)

{{<citation>}}

Yuxin Wang, Xiannian Hu, Quan Gan, Xuanjing Huang, Xipeng Qiu, David Wipf. (2023)  
**Efficient Link Prediction via GNN Layers Induced by Negative Sampling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.09516v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) for link prediction can loosely be divided into two broad categories. First, \emph{node-wise} architectures pre-compute individual embeddings for each node that are later combined by a simple decoder to make predictions. While extremely efficient at inference time (since node embeddings are only computed once and repeatedly reused), model expressiveness is limited such that isomorphic nodes contributing to candidate edges may not be distinguishable, compromising accuracy. In contrast, \emph{edge-wise} methods rely on the formation of edge-specific subgraph embeddings to enrich the representation of pair-wise relationships, disambiguating isomorphic nodes to improve accuracy, but with the cost of increased model complexity. To better navigate this trade-off, we propose a novel GNN architecture whereby the \emph{forward pass} explicitly depends on \emph{both} positive (as is typical) and negative (unique to our approach) edges to inform more flexible, yet still cheap node-wise embeddings. This is achieved by recasting the embeddings themselves as minimizers of a forward-pass-specific energy function (distinct from the actual training loss) that favors separation of positive and negative samples. As demonstrated by extensive empirical evaluations, the resulting architecture retains the inference speed of node-wise models, while producing competitive accuracy with edge-wise alternatives.

{{</citation>}}


### (36/58) Mirage: Model-Agnostic Graph Distillation for Graph Classification (Mridul Gupta et al., 2023)

{{<citation>}}

Mridul Gupta, Sahil Manchanda, Hariprasad Kodamana, Sayan Ranu. (2023)  
**Mirage: Model-Agnostic Graph Distillation for Graph Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.09486v2)  

---


**ABSTRACT**  
GNNs, like other deep learning models, are data and computation hungry. There is a pressing need to scale training of GNNs on large datasets to enable their usage on low-resource environments. Graph distillation is an effort in that direction with the aim to construct a smaller synthetic training set from the original training data without significantly compromising model performance. While initial efforts are promising, this work is motivated by two key observations: (1) Existing graph distillation algorithms themselves rely on training with the full dataset, which undermines the very premise of graph distillation. (2) The distillation process is specific to the target GNN architecture and hyper-parameters and thus not robust to changes in the modeling pipeline. We circumvent these limitations by designing a distillation algorithm called Mirage for graph classification. Mirage is built on the insight that a message-passing GNN decomposes the input graph into a multiset of computation trees. Furthermore, the frequency distribution of computation trees is often skewed in nature, enabling us to condense this data into a concise distilled summary. By compressing the computation data itself, as opposed to emulating gradient flows on the original training set-a prevalent approach to date-Mirage transforms into an unsupervised and architecture-agnostic distillation algorithm. Extensive benchmarking on real-world datasets underscores Mirage's superiority, showcasing enhanced generalization accuracy, data compression, and distillation efficiency when compared to state-of-the-art baselines.

{{</citation>}}


### (37/58) Applying Bayesian Ridge Regression AI Modeling in Virus Severity Prediction (Jai Pal et al., 2023)

{{<citation>}}

Jai Pal, Bryan Hong. (2023)  
**Applying Bayesian Ridge Regression AI Modeling in Virus Severity Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09485v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) is a powerful tool for reshaping healthcare systems. In healthcare, AI is invaluable for its capacity to manage vast amounts of data, which can lead to more accurate and speedy diagnoses, ultimately easing the workload on healthcare professionals. As a result, AI has proven itself to be a power tool across various industries, simplifying complex tasks and pattern recognition that would otherwise be overwhelming for humans or traditional computer algorithms. In this paper, we review the strengths and weaknesses of Bayesian Ridge Regression, an AI model that can be used to bring cutting edge virus analysis to healthcare professionals around the world. The model's accuracy assessment revealed promising results, with room for improvement primarily related to data organization. In addition, the severity index serves as a valuable tool to gain a broad overview of patient care needs, aligning with healthcare professionals' preference for broader categorizations.

{{</citation>}}


### (38/58) Can CNNs Accurately Classify Human Emotions? A Deep-Learning Facial Expression Recognition Study (Ashley Jisue Hong et al., 2023)

{{<citation>}}

Ashley Jisue Hong, David DiStefano, Sejal Dua. (2023)  
**Can CNNs Accurately Classify Human Emotions? A Deep-Learning Facial Expression Recognition Study**  

---
Primary Category: cs.LG  
Categories: I-2-6; I-4-m, cs-LG, cs-NE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09473v1)  

---


**ABSTRACT**  
Emotional Artificial Intelligences are currently one of the most anticipated developments of AI. If successful, these AIs will be classified as one of the most complex, intelligent nonhuman entities as they will possess sentience, the primary factor that distinguishes living humans and mechanical machines. For AIs to be classified as "emotional," they should be able to empathize with others and classify their emotions because without such abilities they cannot normally interact with humans. This study investigates the CNN model's ability to recognize and classify human facial expressions (positive, neutral, negative). The CNN model made for this study is programmed in Python and trained with preprocessed data from the Chicago Face Database. The model is intentionally designed with less complexity to further investigate its ability. We hypothesized that the model will perform better than chance (33.3%) in classifying each emotion class of input data. The model accuracy was tested with novel images. Accuracy was summarized in a percentage report, comparative plot, and confusion matrix. Results of this study supported the hypothesis as the model had 75% accuracy over 10,000 images (data), highlighting the possibility of AIs that accurately analyze human emotions and the prospect of viable Emotional AIs.

{{</citation>}}


## cs.RO (2)



### (39/58) A Framework For Automated Dissection Along Tissue Boundary (Ki-Hwan Oh et al., 2023)

{{<citation>}}

Ki-Hwan Oh, Leonardo Borgioli, Milos Zefran, Liaohai Chen, Pier Cristoforo Giulianotti. (2023)  
**A Framework For Automated Dissection Along Tissue Boundary**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09669v1)  

---


**ABSTRACT**  
Robotic surgery promises enhanced precision and adaptability over traditional surgical methods. It also offers the possibility of automating surgical interventions, resulting in reduced stress on the surgeon, better surgical outcomes, and lower costs. Cholecystectomy, the removal of the gallbladder, serves as an ideal model procedure for automation due to its distinct and well-contrasted anatomical features between the gallbladder and liver, along with standardized surgical maneuvers. Dissection is a frequently used subtask in cholecystectomy where the surgeon delivers the energy on the hook to detach the gallbladder from the liver. Hence, dissection along tissue boundaries is a good candidate for surgical automation. For the da Vinci surgical robot to perform the same procedure as a surgeon automatically, it needs to have the ability to (1) recognize and distinguish between the two different tissues (e.g. the liver and the gallbladder), (2) understand where the boundary between the two tissues is located in the 3D workspace, (3) locate the instrument tip relative to the boundary in the 3D space using visual feedback, and (4) move the instrument along the boundary. This paper presents a novel framework that addresses these challenges through AI-assisted image processing and vision-based robot control. We also present the ex-vivo evaluation of the automated procedure on chicken and pork liver specimens that demonstrates the effectiveness of the proposed framework.

{{</citation>}}


### (40/58) Airborne Sense and Detect of Drones using LiDAR and adapted PointPillars DNN (Manduhu Manduhu et al., 2023)

{{<citation>}}

Manduhu Manduhu, Alexander Dow, Petar Trslic, Gerard Dooly, Benjamin Blanck, James Riordan. (2023)  
**Airborne Sense and Detect of Drones using LiDAR and adapted PointPillars DNN**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.09589v1)  

---


**ABSTRACT**  
The safe operation of drone swarms beyond visual line of sight requires multiple safeguards to mitigate the risk of collision between drones flying in hyper localised scenarios. Cooperative navigation and flight coordination strategies that rely on pre-planned trajectories and require constant network connectivity are brittle to failure. Drone embedded sense and detect offers a comprehensive mode of separation between drones for deconfliction and collision avoidance. This paper presents the first airborne LiDAR based solution for drone-swarm detection and localisation using 3D deep learning. It adapts and embeds the PointPillars deep learning neural network on the drone. To collect training data of close-quarter multi drone operations and safety critical scenarios, a scenario Digital Twin is used to augment real datasets with high fidelity synthetic data. The method has been validated in real-world tests. The trained model achieves over 80% recall and 96% precision when tested on real datasets. By incorporating a detection-by-tracking algorithm the system can reliably monitor the separation distance of multiple drones in challenging environments.

{{</citation>}}


## cs.DC (1)



### (41/58) A Blockchain-empowered Multi-Aggregator Federated Learning Architecture in Edge Computing with Deep Reinforcement Learning Optimization (Xiao Li et al., 2023)

{{<citation>}}

Xiao Li, Weili Wu. (2023)  
**A Blockchain-empowered Multi-Aggregator Federated Learning Architecture in Edge Computing with Deep Reinforcement Learning Optimization**  

---
Primary Category: cs.DC  
Categories: cs-CR, cs-DC, cs-LG, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.09665v1)  

---


**ABSTRACT**  
Federated learning (FL) is emerging as a sought-after distributed machine learning architecture, offering the advantage of model training without direct exposure of raw data. With advancements in network infrastructure, FL has been seamlessly integrated into edge computing. However, the limited resources on edge devices introduce security vulnerabilities to FL in the context. While blockchain technology promises to bolster security, practical deployment on resource-constrained edge devices remains a challenge. Moreover, the exploration of FL with multiple aggregators in edge computing is still new in the literature. Addressing these gaps, we introduce the Blockchain-empowered Heterogeneous Multi-Aggregator Federated Learning Architecture (BMA-FL). We design a novel light-weight Byzantine consensus mechanism, namely PBCM, to enable secure and fast model aggregation and synchronization in BMA-FL. We also dive into the heterogeneity problem in BMA-FL that the aggregators are associated with varied number of connected trainers with Non-IID data distributions and diverse training speed. We proposed a multi-agent deep reinforcement learning algorithm to help aggregators decide the best training strategies. The experiments on real-word datasets demonstrate the efficiency of BMA-FL to achieve better models faster than baselines, showing the efficacy of PBCM and proposed deep reinforcement learning algorithm.

{{</citation>}}


## cs.CR (2)



### (42/58) BufferSearch: Generating Black-Box Adversarial Texts With Lower Queries (Wenjie Lv et al., 2023)

{{<citation>}}

Wenjie Lv, Zhen Wang, Yitao Zheng, Zhehua Zhong, Qi Xuan, Tianyi Chen. (2023)  
**BufferSearch: Generating Black-Box Adversarial Texts With Lower Queries**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.09652v1)  

---


**ABSTRACT**  
Machine learning security has recently become a prominent topic in the natural language processing (NLP) area. The existing black-box adversarial attack suffers prohibitively from the high model querying complexity, resulting in easily being captured by anti-attack monitors. Meanwhile, how to eliminate redundant model queries is rarely explored. In this paper, we propose a query-efficient approach BufferSearch to effectively attack general intelligent NLP systems with the minimal number of querying requests. In general, BufferSearch makes use of historical information and conducts statistical test to avoid incurring model queries frequently. Numerically, we demonstrate the effectiveness of BufferSearch on various benchmark text-classification experiments by achieving the competitive attacking performance but with a significant reduction of query quantity. Furthermore, BufferSearch performs multiple times better than competitors within restricted query budget. Our work establishes a strong benchmark for the future study of query-efficiency in NLP adversarial attacks.

{{</citation>}}


### (43/58) Survey on Security Attacks in Connected and Autonomous Vehicular Systems (S M Mostaq Hossain et al., 2023)

{{<citation>}}

S M Mostaq Hossain, Shampa Banik, Trapa Banik, Ashfak Md Shibli. (2023)  
**Survey on Security Attacks in Connected and Autonomous Vehicular Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.09510v1)  

---


**ABSTRACT**  
Connected and autonomous vehicles, also known as CAVs, are a general trend in the evolution of the automotive industry that can be utilized to make transportation safer, improve the number of mobility options available, user costs will go down and new jobs will be created. However, as our society grows more automated and networked, criminal actors will have additional opportunities to conduct a variety of attacks, putting CAV security in danger. By providing a brief review of the state of cyber security in the CAVs environment, this study aims to draw attention to the issues and concerns associated with security. The first thing it does is categorize the multiple cybersecurity threats and weaknesses in the context of CAVs into three groups: attacks on the vehicles network, attacks on the Internet at large, and other attacks. This is done in accordance with the various communication networks and targets under attack. Next, it considers the possibility of cyber attacks to be an additional form of threat posed by the environment of CAVs. After that, it details the most uptodate defense tactics for securing CAVs and analyzes how effective they are. In addition, it draws some conclusions about the various cyber security and safety requirements of CAVs that are now available, which is beneficial for the use of CAVs in the real world. At the end, we discussed some implications on Adversary Attacks on Autonomous Vehicles. In conclusion, a number of difficulties and unsolved issues for future research are analyzed and explored.

{{</citation>}}


## cs.SE (3)



### (44/58) Enhancing Binary Code Comment Quality Classification: Integrating Generative AI for Improved Accuracy (Rohith Arumugam S et al., 2023)

{{<citation>}}

Rohith Arumugam S, Angel Deborah S. (2023)  
**Enhancing Binary Code Comment Quality Classification: Integrating Generative AI for Improved Accuracy**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: AI, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11467v1)  

---


**ABSTRACT**  
This report focuses on enhancing a binary code comment quality classification model by integrating generated code and comment pairs, to improve model accuracy. The dataset comprises 9048 pairs of code and comments written in the C programming language, each annotated as "Useful" or "Not Useful." Additionally, code and comment pairs are generated using a Large Language Model Architecture, and these generated pairs are labeled to indicate their utility. The outcome of this effort consists of two classification models: one utilizing the original dataset and another incorporating the augmented dataset with the newly generated code comment pairs and labels.

{{</citation>}}


### (45/58) An Exploration Into Web Session Security- A Systematic Literature Review (Md. Imtiaz Habib et al., 2023)

{{<citation>}}

Md. Imtiaz Habib, Abdullah Al Maruf, Md. Jobair Ahmed Nabil. (2023)  
**An Exploration Into Web Session Security- A Systematic Literature Review**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.10687v1)  

---


**ABSTRACT**  
The most common attacks against web sessions are reviewed in this paper, for example, some attacks against web browsers' honest users attempting to create session with trusted web browser application legally. We have assessed with four different ways to judge the viability of a certain solution by reviewing existing security solutions which prevent or halt the different attacks. Then we have pointed out some guidelines that have been taken into account by the designers of the proposals we reviewed. The guidelines we have identified will be helpful for the creative solutions proceeding web security in a more structured and holistic way.

{{</citation>}}


### (46/58) Common Challenges of Deep Reinforcement Learning Applications Development: An Empirical Study (Mohammad Mehdi Morovati et al., 2023)

{{<citation>}}

Mohammad Mehdi Morovati, Florian Tambon, Mina Taraghi, Amin Nikanjam, Foutse Khomh. (2023)  
**Common Challenges of Deep Reinforcement Learning Applications Development: An Empirical Study**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.09575v1)  

---


**ABSTRACT**  
Machine Learning (ML) is increasingly being adopted in different industries. Deep Reinforcement Learning (DRL) is a subdomain of ML used to produce intelligent agents. Despite recent developments in DRL technology, the main challenges that developers face in the development of DRL applications are still unknown. To fill this gap, in this paper, we conduct a large-scale empirical study of 927 DRL-related posts extracted from Stack Overflow, the most popular Q&A platform in the software community. Through the process of labeling and categorizing extracted posts, we created a taxonomy of common challenges encountered in the development of DRL applications, along with their corresponding popularity levels. This taxonomy has been validated through a survey involving 59 DRL developers. Results show that at least 45% of developers experienced 18 of the 21 challenges identified in the taxonomy. The most frequent source of difficulty during the development of DRL applications are Comprehension, API usage, and Design problems, while Parallel processing, and DRL libraries/frameworks are classified as the most difficult challenges to address, with respect to the time required to receive an accepted answer. We hope that the research community will leverage this taxonomy to develop efficient strategies to address the identified challenges and improve the quality of DRL applications.

{{</citation>}}


## cs.HC (1)



### (47/58) How Good is ChatGPT in Giving Advice on Your Visualization Design? (Nam Wook Kim et al., 2023)

{{<citation>}}

Nam Wook Kim, Grace Myers, Benjamin Bach. (2023)  
**How Good is ChatGPT in Giving Advice on Your Visualization Design?**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.09617v1)  

---


**ABSTRACT**  
Data visualization practitioners often lack formal training, resulting in a knowledge gap in visualization design best practices. Large-language models like ChatGPT, with their vast internet-scale training data, offer transformative potential in addressing this gap. To explore this potential, we adopted a mixed-method approach. Initially, we analyzed the VisGuide forum, a repository of data visualization questions, by comparing ChatGPT-generated responses to human replies. Subsequently, our user study delved into practitioners' reactions and attitudes toward ChatGPT as a visualization assistant. Participants, who brought their visualizations and questions, received feedback from both human experts and ChatGPT in a randomized order. They filled out experience surveys and shared deeper insights through post-interviews. The results highlight the unique advantages and disadvantages of ChatGPT, such as its ability to quickly provide a wide range of design options based on a broad knowledge base, while also revealing its limitations in terms of depth and critical thinking capabilities.

{{</citation>}}


## cs.NI (1)



### (48/58) Towards Intelligent Network Management: Leveraging AI for Network Service Detection (Khuong N. Nguyen et al., 2023)

{{<citation>}}

Khuong N. Nguyen, Abhishek Sehgal, Yuming Zhu, Junsu Choi, Guanbo Chen, Hao Chen, Boon Loong Ng, Charlie Zhang. (2023)  
**Towards Intelligent Network Management: Leveraging AI for Network Service Detection**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09609v1)  

---


**ABSTRACT**  
As the complexity and scale of modern computer networks continue to increase, there has emerged an urgent need for precise traffic analysis, which plays a pivotal role in cutting-edge wireless connectivity technologies. This study focuses on leveraging Machine Learning methodologies to create an advanced network traffic classification system. We introduce a novel data-driven approach that excels in identifying various network service types in real-time, by analyzing patterns within the network traffic. Our method organizes similar kinds of network traffic into distinct categories, referred to as network services, based on latency requirement. Furthermore, it decomposes the network traffic stream into multiple, smaller traffic flows, with each flow uniquely carrying a specific service. Our ML models are trained on a dataset comprised of labeled examples representing different network service types collected on various Wi-Fi network conditions. Upon evaluation, our system demonstrates a remarkable accuracy in distinguishing the network services. These results emphasize the substantial promise of integrating Artificial Intelligence in wireless technologies. Such an approach encourages more efficient energy consumption, enhances Quality of Service assurance, and optimizes the allocation of network resources, thus laying a solid groundwork for the development of advanced intelligent networks.

{{</citation>}}


## cs.AI (3)



### (49/58) Penetrative AI: Making LLMs Comprehend the Physical World (Huatao Xu et al., 2023)

{{<citation>}}

Huatao Xu, Liying Han, Mo Li, Mani Srivastava. (2023)  
**Penetrative AI: Making LLMs Comprehend the Physical World**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.09605v1)  

---


**ABSTRACT**  
Recent developments in Large Language Models (LLMs) have demonstrated their remarkable capabilities across a range of tasks. Questions, however, persist about the nature of LLMs and their potential to integrate common-sense human knowledge when performing tasks involving information about the real physical world. This paper delves into these questions by exploring how LLMs can be extended to interact with and reason about the physical world through IoT sensors and actuators, a concept that we term "\textit{Penetrative AI}". The paper explores such an extension at two levels of LLMs' ability to penetrate into the physical world via the processing of sensory signals. Our preliminary findings indicate that LLMs, with ChatGPT being the representative example in our exploration, have considerable and unique proficiency in employing the knowledge they learned during training for interpreting IoT sensor data and reasoning over them about tasks in the physical realm. Not only this opens up new applications for LLMs beyond traditional text-based tasks, but also enables new ways of incorporating human knowledge in cyber-physical systems.

{{</citation>}}


### (50/58) A Framework for Empowering Reinforcement Learning Agents with Causal Analysis: Enhancing Automated Cryptocurrency Trading (Rasoul Amirzadeh et al., 2023)

{{<citation>}}

Rasoul Amirzadeh, Dhananjay Thiruvady, Asef Nazari, Mong Shan Ee. (2023)  
**A Framework for Empowering Reinforcement Learning Agents with Causal Analysis: Enhancing Automated Cryptocurrency Trading**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.09462v1)  

---


**ABSTRACT**  
Despite advances in artificial intelligence-enhanced trading methods, developing a profitable automated trading system remains challenging in the rapidly evolving cryptocurrency market. This study aims to address these challenges by developing a reinforcement learning-based automated trading system for five popular altcoins~(cryptocurrencies other than Bitcoin): Binance Coin, Ethereum, Litecoin, Ripple, and Tether. To this end, we present CausalReinforceNet, a framework framed as a decision support system. Designed as the foundational architecture of the trading system, the CausalReinforceNet framework enhances the capabilities of the reinforcement learning agent through causal analysis. Within this framework, we use Bayesian networks in the feature engineering process to identify the most relevant features with causal relationships that influence cryptocurrency price movements. Additionally, we incorporate probabilistic price direction signals from dynamic Bayesian networks to enhance our reinforcement learning agent's decision-making. Due to the high volatility of the cryptocurrency market, we design our framework to adopt a conservative approach that limits sell and buy position sizes to manage risk. We develop two agents using the CausalReinforceNet framework, each based on distinct reinforcement learning algorithms. The results indicate that our framework substantially surpasses the Buy-and-Hold benchmark strategy in profitability. Additionally, both agents generated notable returns on investment for Binance Coin and Ethereum.

{{</citation>}}


### (51/58) LgTS: Dynamic Task Sampling using LLM-generated sub-goals for Reinforcement Learning Agents (Yash Shukla et al., 2023)

{{<citation>}}

Yash Shukla, Wenchang Gao, Vasanth Sarathy, Alvaro Velasquez, Robert Wright, Jivko Sinapov. (2023)  
**LgTS: Dynamic Task Sampling using LLM-generated sub-goals for Reinforcement Learning Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.09454v1)  

---


**ABSTRACT**  
Recent advancements in reasoning abilities of Large Language Models (LLM) has promoted their usage in problems that require high-level planning for robots and artificial agents. However, current techniques that utilize LLMs for such planning tasks make certain key assumptions such as, access to datasets that permit finetuning, meticulously engineered prompts that only provide relevant and essential information to the LLM, and most importantly, a deterministic approach to allow execution of the LLM responses either in the form of existing policies or plan operators. In this work, we propose LgTS (LLM-guided Teacher-Student learning), a novel approach that explores the planning abilities of LLMs to provide a graphical representation of the sub-goals to a reinforcement learning (RL) agent that does not have access to the transition dynamics of the environment. The RL agent uses Teacher-Student learning algorithm to learn a set of successful policies for reaching the goal state from the start state while simultaneously minimizing the number of environmental interactions. Unlike previous methods that utilize LLMs, our approach does not assume access to a propreitary or a fine-tuned LLM, nor does it require pre-trained policies that achieve the sub-goals proposed by the LLM. Through experiments on a gridworld based DoorKey domain and a search-and-rescue inspired domain, we show that generating a graphical structure of sub-goals helps in learning policies for the LLM proposed sub-goals and the Teacher-Student learning algorithm minimizes the number of environment interactions when the transition dynamics are unknown.

{{</citation>}}


## cs.IR (3)



### (52/58) Context-aware Session-based Recommendation with Graph Neural Networks (Zhihui Zhang et al., 2023)

{{<citation>}}

Zhihui Zhang, JianXiang Yu, Xiang Li. (2023)  
**Context-aware Session-based Recommendation with Graph Neural Networks**  

---
Primary Category: cs.IR  
Categories: F-4-1, cs-AI, cs-IR, cs.IR  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.09593v1)  

---


**ABSTRACT**  
Session-based recommendation (SBR) is a task that aims to predict items based on anonymous sequences of user behaviors in a session. While there are methods that leverage rich context information in sessions for SBR, most of them have the following limitations: 1) they fail to distinguish the item-item edge types when constructing the global graph for exploiting cross-session contexts; 2) they learn a fixed embedding vector for each item, which lacks the flexibility to reflect the variation of user interests across sessions; 3) they generally use the one-hot encoded vector of the target item as the hard label to predict, thus failing to capture the true user preference. To solve these issues, we propose CARES, a novel context-aware session-based recommendation model with graph neural networks, which utilizes different types of contexts in sessions to capture user interests. Specifically, we first construct a multi-relation cross-session graph to connect items according to intra- and cross-session item-level contexts. Further, to encode the variation of user interests, we design personalized item representations. Finally, we employ a label collaboration strategy for generating soft user preference distribution as labels. Experiments on three benchmark datasets demonstrate that CARES consistently outperforms state-of-the-art models in terms of P@20 and MRR@20. Our data and codes are publicly available at https://github.com/brilliantZhang/CARES.

{{</citation>}}


### (53/58) Findability: A Novel Measure of Information Accessibility (Aman Sinha et al., 2023)

{{<citation>}}

Aman Sinha, Priyanshu Raj Mall, Dwaipayan Roy. (2023)  
**Findability: A Novel Measure of Information Accessibility**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2310.09508v1)  

---


**ABSTRACT**  
The overwhelming volume of data generated and indexed by search engines poses a significant challenge in retrieving documents from the index efficiently and effectively. Even with a well-crafted query, several relevant documents often get buried among a multitude of competing documents, resulting in reduced accessibility or `findability' of the desired document. Consequently, it is crucial to develop a robust methodology for assessing this dimension of Information Retrieval (IR) system performance. While previous studies have focused on measuring document accessibility disregarding user queries and document relevance, there exists no metric to quantify the findability of a document within a given IR system without resorting to manual labor. This paper aims to address this gap by defining and deriving a metric to evaluate the findability of documents as perceived by end-users. Through experiments, we demonstrate the varying impact of different retrieval models and collections on the findability of documents. Furthermore, we establish the findability measure as an independent metric distinct from retrievability, an accessibility measure introduced in prior literature.

{{</citation>}}


### (54/58) A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models (Shengyao Zhuang et al., 2023)

{{<citation>}}

Shengyao Zhuang, Honglei Zhuang, Bevan Koopman, Guido Zuccon. (2023)  
**A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.09497v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) demonstrate impressive effectiveness in zero-shot document ranking tasks. Pointwise, Pairwise, and Listwise prompting approaches have been proposed for LLM-based zero-shot ranking. Our study begins by thoroughly evaluating these existing approaches within a consistent experimental framework, considering factors like model size, token consumption, latency, among others. This first-of-its-kind comparative evaluation of these approaches allows us to identify the trade-offs between effectiveness and efficiency inherent in each approach. We find that while Pointwise approaches score high on efficiency, they suffer from poor effectiveness. Conversely, Pairwise approaches demonstrate superior effectiveness but incur high computational overhead. To further enhance the efficiency of LLM-based zero-shot ranking, we propose a novel Setwise prompting approach. Our approach reduces the number of LLM inferences and the amount of prompt token consumption during the ranking procedure, significantly improving the efficiency of LLM-based zero-shot ranking. We test our method using the TREC DL datasets and the BEIR zero-shot document ranking benchmark. The empirical results indicate that our approach considerably reduces computational costs while also retaining high zero-shot ranking effectiveness.

{{</citation>}}


## cs.AR (1)



### (55/58) Wafer-scale Computing: Advancements, Challenges, and Future Perspectives (Yang Hu et al., 2023)

{{<citation>}}

Yang Hu, Xinhan Lin, Huizheng Wang, Zhen He, Xingmao Yu, Jiahao Zhang, Qize Yang, Zheng Xu, Sihan Guan, Jiahao Fang, Haoran Shang, Xinru Tang, Xu Dai, Shaojun Wei, Shouyi Yin. (2023)  
**Wafer-scale Computing: Advancements, Challenges, and Future Perspectives**  

---
Primary Category: cs.AR  
Categories: B-7-0; C-1, cs-AR, cs.AR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09568v1)  

---


**ABSTRACT**  
Nowadays, artificial intelligence (AI) technology with large models plays an increasingly important role in both academia and industry. It also brings a rapidly increasing demand for the computing power of the hardware. As the computing demand for AI continues to grow, the growth of hardware computing power has failed to keep up. This has become a significant factor restricting the development of AI. The augmentation of hardware computing power is mainly propelled by the escalation of transistor density and chip area. However, the former is impeded by the termination of the Moore's Law and Dennard scaling, and the latter is significantly restricted by the challenge of disrupting the legacy fabrication equipment and process.   In recent years, advanced packaging technologies that have gradually matured are increasingly used to implement bigger chips that integrate multiple chiplets, while still providing interconnections with chip-level density and bandwidth. Compared to conventional high-performance computing paradigms such as multi-accelerator and datacenter-scale computing, Wafer-scale Computing shows remarkable advantages in communication bandwidth, integration density, and programmability potential. Not surprisingly, disruptive Wafer-scale Computing also brings unprecedented design challenges for hardware architecture, design-system-technology co-optimization, power and cooling systems, and compiler tool chain. At present, there are no comprehensive surveys summarizing the current state and design insights of Wafer-scale Computing. This paper aims to take the first step to help academia and industry review existing wafer-scale chips and essential technologies in a one-stop manner. So that people can conveniently grasp the basic knowledge and key points, understand the achievements and shortcomings of existing research, and contribute to this promising research direction.

{{</citation>}}


## q-bio.PE (1)



### (56/58) ARTree: A Deep Autoregressive Model for Phylogenetic Inference (Tianyu Xie et al., 2023)

{{<citation>}}

Tianyu Xie, Cheng Zhang. (2023)  
**ARTree: A Deep Autoregressive Model for Phylogenetic Inference**  

---
Primary Category: q-bio.PE  
Categories: cs-LG, q-bio-PE, q-bio.PE, stat-ML  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.09553v1)  

---


**ABSTRACT**  
Designing flexible probabilistic models over tree topologies is important for developing efficient phylogenetic inference methods. To do that, previous works often leverage the similarity of tree topologies via hand-engineered heuristic features which would require pre-sampled tree topologies and may suffer from limited approximation capability. In this paper, we propose a deep autoregressive model for phylogenetic inference based on graph neural networks (GNNs), called ARTree. By decomposing a tree topology into a sequence of leaf node addition operations and modeling the involved conditional distributions based on learnable topological features via GNNs, ARTree can provide a rich family of distributions over the entire tree topology space that have simple sampling algorithms and density estimation procedures, without using heuristic features. We demonstrate the effectiveness and efficiency of our method on a benchmark of challenging real data tree topology density estimation and variational Bayesian phylogenetic inference problems.

{{</citation>}}


## cs.SD (1)



### (57/58) Dynamic Prediction of Full-Ocean Depth SSP by Hierarchical LSTM: An Experimental Result (Jiajun Lu et al., 2023)

{{<citation>}}

Jiajun Lu, Wei Huang, Hao Zhang. (2023)  
**Dynamic Prediction of Full-Ocean Depth SSP by Hierarchical LSTM: An Experimental Result**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.09522v1)  

---


**ABSTRACT**  
SSP distribution is an important parameter for underwater positioning, navigation and timing (PNT) because it affects the propagation mode of underwater acoustic signals. To accurate predict future sound speed distribution, we propose a hierarchical long short--term memory (H--LSTM) neural network for future sound speed prediction, which explore the distribution pattern of sound velocity in the time dimension. To verify the feasibility and effectiveness, we conducted both simulations and real experiments. The ocean experiment was held in the South China Sea in April, 2023. Results show that the accuracy of the proposed method outperforms the state--of--the--art methods.

{{</citation>}}


## stat.ML (1)



### (58/58) ARM: Refining Multivariate Forecasting with Adaptive Temporal-Contextual Learning (Jiecheng Lu et al., 2023)

{{<citation>}}

Jiecheng Lu, Xu Han, Shihao Yang. (2023)  
**ARM: Refining Multivariate Forecasting with Adaptive Temporal-Contextual Learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.09488v1)  

---


**ABSTRACT**  
Long-term time series forecasting (LTSF) is important for various domains but is confronted by challenges in handling the complex temporal-contextual relationships. As multivariate input models underperforming some recent univariate counterparts, we posit that the issue lies in the inefficiency of existing multivariate LTSF Transformers to model series-wise relationships: the characteristic differences between series are often captured incorrectly. To address this, we introduce ARM: a multivariate temporal-contextual adaptive learning method, which is an enhanced architecture specifically designed for multivariate LTSF modelling. ARM employs Adaptive Univariate Effect Learning (AUEL), Random Dropping (RD) training strategy, and Multi-kernel Local Smoothing (MKLS), to better handle individual series temporal patterns and correctly learn inter-series dependencies. ARM demonstrates superior performance on multiple benchmarks without significantly increasing computational costs compared to vanilla Transformer, thereby advancing the state-of-the-art in LTSF. ARM is also generally applicable to other LTSF architecture beyond vanilla Transformer.

{{</citation>}}
