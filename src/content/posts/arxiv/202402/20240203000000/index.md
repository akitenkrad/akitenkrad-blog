---
draft: false
title: "arXiv @ 2024.02.03"
date: 2024-02-03
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.02.03"
    identifier: arxiv_20240203
    parent: 202402_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (26)](#cscl-26)
- [cs.LG (23)](#cslg-23)
- [cs.CR (4)](#cscr-4)
- [cs.CY (2)](#cscy-2)
- [cs.DS (1)](#csds-1)
- [eess.AS (2)](#eessas-2)
- [cs.MA (1)](#csma-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.HC (1)](#cshc-1)
- [cs.SD (2)](#cssd-2)
- [cs.SE (6)](#csse-6)
- [cs.RO (7)](#csro-7)
- [cs.AI (3)](#csai-3)
- [cs.CV (16)](#cscv-16)
- [cs.DC (1)](#csdc-1)
- [cs.LO (1)](#cslo-1)
- [q-fin.PM (1)](#q-finpm-1)
- [cs.AR (2)](#csar-2)
- [cs.IT (1)](#csit-1)
- [cs.NE (1)](#csne-1)
- [eess.IV (1)](#eessiv-1)
- [q-bio.OT (1)](#q-bioot-1)
- [cs.IR (2)](#csir-2)
- [q-fin.GN (1)](#q-fingn-1)
- [cs.DB (1)](#csdb-1)

## cs.CL (26)



### (1/108) Evaluating Large Language Models for Generalization and Robustness via Data Compression (Yucheng Li et al., 2024)

{{<citation>}}

Yucheng Li, Yunhao Guo, Frank Guerin, Chenghua Lin. (2024)  
**Evaluating Large Language Models for Generalization and Robustness via Data Compression**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00861v1)  

---


**ABSTRACT**  
Existing methods for evaluating large language models face challenges such as data contamination, sensitivity to prompts, and the high cost of benchmark creation. To address this, we propose a lossless data compression based evaluation approach that tests how models' predictive abilities generalize after their training cutoff. Specifically, we collect comprehensive test data spanning 83 months from 2017 to 2023 and split the data into training and testing periods according to models' training data cutoff. We measure: 1) the compression performance on the testing period as a measure of generalization on unseen data; and 2) the performance gap between the training and testing period as a measure of robustness. Our experiments test 14 representative large language models with various sizes on sources including Wikipedia, news articles, code, arXiv papers, and multi-modal data. We find that the compression rate of many models reduces significantly after their cutoff date, but models such as Mistral and Llama-2 demonstrate a good balance between performance and robustness. Results also suggest that models struggle to generalize on news and code data, but work especially well on arXiv papers. We also find the context size and tokenization implementation have a big impact of on the overall compression performance.

{{</citation>}}


### (2/108) Can Large Language Models Understand Context? (Yilun Zhu et al., 2024)

{{<citation>}}

Yilun Zhu, Joel Ruben Antony Moniz, Shruti Bhargava, Jiarui Lu, Dhivya Piraviperumal, Site Li, Yuan Zhang, Hong Yu, Bo-Hsiang Tseng. (2024)  
**Can Large Language Models Understand Context?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2402.00858v1)  

---


**ABSTRACT**  
Understanding context is key to understanding human language, an ability which Large Language Models (LLMs) have been increasingly seen to demonstrate to an impressive extent. However, though the evaluation of LLMs encompasses various domains within the realm of Natural Language Processing, limited attention has been paid to probing their linguistic capability of understanding contextual features. This paper introduces a context understanding benchmark by adapting existing datasets to suit the evaluation of generative models. This benchmark comprises of four distinct tasks and nine datasets, all featuring prompts designed to assess the models' ability to understand context. First, we evaluate the performance of LLMs under the in-context learning pretraining scenario. Experimental results indicate that pre-trained dense models struggle with understanding more nuanced contextual features when compared to state-of-the-art fine-tuned models. Second, as LLM compression holds growing significance in both research and real-world applications, we assess the context understanding of quantized models under in-context-learning settings. We find that 3-bit post-training quantization leads to varying degrees of performance reduction on our benchmark. We conduct an extensive analysis of these scenarios to substantiate our experimental results.

{{</citation>}}


### (3/108) Towards Efficient and Exact Optimization of Language Model Alignment (Haozhe Ji et al., 2024)

{{<citation>}}

Haozhe Ji, Cheng Lu, Yilin Niu, Pei Ke, Hongning Wang, Jun Zhu, Jie Tang, Minlie Huang. (2024)  
**Towards Efficient and Exact Optimization of Language Model Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00856v1)  

---


**ABSTRACT**  
The alignment of language models with human preferences is vital for their application in real-world tasks. The problem is formulated as optimizing the model's policy to maximize the expected reward that reflects human preferences with minimal deviation from the initial policy. While considered as a straightforward solution, reinforcement learning (RL) suffers from high variance in policy updates, which impedes efficient policy improvement. Recently, direct preference optimization (DPO) was proposed to directly optimize the policy from preference data. Though simple to implement, DPO is derived based on the optimal policy that is not assured to be achieved in practice, which undermines its convergence to the intended solution.   In this paper, we propose efficient exact optimization (EXO) of the alignment objective. We prove that EXO is guaranteed to optimize in the same direction as the RL algorithms asymptotically for arbitary parametrization of the policy, while enables efficient optimization by circumventing the complexities associated with RL algorithms. We compare our method to DPO with both theoretical and empirical analyses, and further demonstrate the advantages of our method over existing approaches on realistic human preference data.

{{</citation>}}


### (4/108) Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization? (Xue-Yong Fu et al., 2024)

{{<citation>}}

Xue-Yong Fu, Md Tahmid Rahman Laskar, Elena Khasanova, Cheng Chen, Shashi Bhushan TN. (2024)  
**Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, LLaMA, Language Model, PaLM, Summarization, T5  
[Paper Link](http://arxiv.org/abs/2402.00841v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated impressive capabilities to solve a wide range of tasks without being explicitly fine-tuned on task-specific datasets. However, deploying LLMs in the real world is not trivial, as it requires substantial computing resources. In this paper, we investigate whether smaller, compact LLMs are a good alternative to the comparatively Larger LLMs2 to address significant costs associated with utilizing LLMs in the real world. In this regard, we study the meeting summarization task in a real-world industrial environment and conduct extensive experiments by comparing the performance of fine-tuned compact LLMs (e.g., FLAN-T5, TinyLLaMA, LiteLLaMA) with zero-shot larger LLMs (e.g., LLaMA-2, GPT-3.5, PaLM-2). We observe that most smaller LLMs, even after fine-tuning, fail to outperform larger zero-shot LLMs in meeting summarization datasets. However, a notable exception is FLAN-T5 (780M parameters), which performs on par or even better than many zero-shot Larger LLMs (from 7B to above 70B parameters), while being significantly smaller. This makes compact LLMs like FLAN-T5 a suitable cost-efficient solution for real-world industrial deployment.

{{</citation>}}


### (5/108) OLMo: Accelerating the Science of Language Models (Dirk Groeneveld et al., 2024)

{{<citation>}}

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, Hannaneh Hajishirzi. (2024)  
**OLMo: Accelerating the Science of Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2402.00838v1)  

---


**ABSTRACT**  
Language models (LMs) have become ubiquitous in both NLP research and in commercial product offerings. As their commercial importance has surged, the most powerful models have become closed off, gated behind proprietary interfaces, with important details of their training data, architectures, and development undisclosed. Given the importance of these details in scientifically studying these models, including their biases and potential risks, we believe it is essential for the research community to have access to powerful, truly open LMs. To this end, this technical report details the first release of OLMo, a state-of-the-art, truly Open Language Model and its framework to build and study the science of language modeling. Unlike most prior efforts that have only released model weights and inference code, we release OLMo and the whole framework, including training data and training and evaluation code. We hope this release will empower and strengthen the open research community and inspire a new wave of innovation.

{{</citation>}}


### (6/108) ALISON: Fast and Effective Stylometric Authorship Obfuscation (Eric Xing et al., 2024)

{{<citation>}}

Eric Xing, Saranya Venkatraman, Thai Le, Dongwon Lee. (2024)  
**ALISON: Fast and Effective Stylometric Authorship Obfuscation**  

---
Primary Category: cs.CL  
Categories: I-2-7; I-2-0, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2402.00835v1)  

---


**ABSTRACT**  
Authorship Attribution (AA) and Authorship Obfuscation (AO) are two competing tasks of increasing importance in privacy research. Modern AA leverages an author's consistent writing style to match a text to its author using an AA classifier. AO is the corresponding adversarial task, aiming to modify a text in such a way that its semantics are preserved, yet an AA model cannot correctly infer its authorship. To address privacy concerns raised by state-of-the-art (SOTA) AA methods, new AO methods have been proposed but remain largely impractical to use due to their prohibitively slow training and obfuscation speed, often taking hours. To this challenge, we propose a practical AO method, ALISON, that (1) dramatically reduces training/obfuscation time, demonstrating more than 10x faster obfuscation than SOTA AO methods, (2) achieves better obfuscation success through attacking three transformer-based AA methods on two benchmark datasets, typically performing 15% better than competing methods, (3) does not require direct signals from a target AA classifier during obfuscation, and (4) utilizes unique stylometric features, allowing sound model interpretation for explainable obfuscation. We also demonstrate that ALISON can effectively prevent four SOTA AA methods from accurately determining the authorship of ChatGPT-generated texts, all while minimally changing the original text semantics. To ensure the reproducibility of our findings, our code and data are available at: https://github.com/EricX003/ALISON.

{{</citation>}}


### (7/108) ReAGent: Towards A Model-agnostic Feature Attribution Method for Generative Language Models (Zhixue Zhao et al., 2024)

{{<citation>}}

Zhixue Zhao, Boxuan Shan. (2024)  
**ReAGent: Towards A Model-agnostic Feature Attribution Method for Generative Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2402.00794v1)  

---


**ABSTRACT**  
Feature attribution methods (FAs), such as gradients and attention, are widely employed approaches to derive the importance of all input features to the model predictions. Existing work in natural language processing has mostly focused on developing and testing FAs for encoder-only language models (LMs) in classification tasks. However, it is unknown if it is faithful to use these FAs for decoder-only models on text generation, due to the inherent differences between model architectures and task settings respectively. Moreover, previous work has demonstrated that there is no `one-wins-all' FA across models and tasks. This makes the selection of a FA computationally expensive for large LMs since input importance derivation often requires multiple forward and backward passes including gradient computations that might be prohibitive even with access to large compute. To address these issues, we present a model-agnostic FA for generative LMs called Recursive Attribution Generator (ReAGent). Our method updates the token importance distribution in a recursive manner. For each update, we compute the difference in the probability distribution over the vocabulary for predicting the next token between using the original input and using a modified version where a part of the input is replaced with RoBERTa predictions. Our intuition is that replacing an important token in the context should have resulted in a larger change in the model's confidence in predicting the token than replacing an unimportant token. Our method can be universally applied to any generative LM without accessing internal model weights or additional training and fine-tuning, as most other FAs require. We extensively compare the faithfulness of ReAGent with seven popular FAs across six decoder-only LMs of various sizes. The results show that our method consistently provides more faithful token importance distributions.

{{</citation>}}


### (8/108) CroissantLLM: A Truly Bilingual French-English Language Model (Manuel Faysse et al., 2024)

{{<citation>}}

Manuel Faysse, Patrick Fernandes, Nuno Guerreiro, António Loison, Duarte Alves, Caio Corro, Nicolas Boizard, João Alves, Ricardo Rei, Pedro Martins, Antoni Bigata Casademunt, François Yvon, André Martins, Gautier Viaud, Céline Hudelot, Pierre Colombo. (2024)  
**CroissantLLM: A Truly Bilingual French-English Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2402.00786v1)  

---


**ABSTRACT**  
We introduce CroissantLLM, a 1.3B language model pretrained on a set of 3T English and French tokens, to bring to the research and industrial community a high-performance, fully open-sourced bilingual model that runs swiftly on consumer-grade local hardware. To that end, we pioneer the approach of training an intrinsically bilingual model with a 1:1 English-to-French pretraining data ratio, a custom tokenizer, and bilingual finetuning datasets. We release the training dataset, notably containing a French split with manually curated, high-quality, and varied data sources. To assess performance outside of English, we craft a novel benchmark, FrenchBench, consisting of an array of classification and generation tasks, covering various orthogonal aspects of model performance in the French Language. Additionally, rooted in transparency and to foster further Large Language Model research, we release codebases, and dozens of checkpoints across various model sizes, training data distributions, and training steps, as well as fine-tuned Chat models, and strong translation models. We evaluate our model through the FMTI framework, and validate 81 % of the transparency criteria, far beyond the scores of even most open initiatives. This work enriches the NLP landscape, breaking away from previous English-centric work in order to strengthen our understanding of multilinguality in language models.

{{</citation>}}


### (9/108) Health-LLM: Personalized Retrieval-Augmented Disease Prediction Model (Mingyu Jin et al., 2024)

{{<citation>}}

Mingyu Jin, Qinkai Yu, Chong Zhang, Dong Shu, Suiyuan Zhu, Mengnan Du, Yongfeng Zhang, Yanda Meng. (2024)  
**Health-LLM: Personalized Retrieval-Augmented Disease Prediction Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00746v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) in healthcare has significantly advanced intelligent medical treatment. However, traditional intelligent healthcare is limited by static data and unified standards, preventing full integration with individual situations and other challenges. Hence, a more professional and detailed intelligent healthcare method is needed for development. To this end, we propose an innovative framework named Heath-LLM, which combines large-scale feature extraction and medical knowledge trade-off scoring. Compared to traditional health management methods, our approach has three main advantages. First, our method integrates health reports into a large model to provide detailed task information. Second, professional medical expertise is used to adjust the weighted scores of health characteristics. Third, we use a semi-automated feature extraction framework to enhance the analytical power of language models and incorporate expert insights to improve the accuracy of disease prediction. We have conducted disease prediction experiments on a large number of health reports to assess the effectiveness of Health-LLM. The results of the experiments indicate that the proposed method surpasses traditional methods and has the potential to revolutionize disease prediction and personalized health management. The code is available at https://github.com/jmyissb/HealthLLM.

{{</citation>}}


### (10/108) Enhancing Ethical Explanations of Large Language Models through Iterative Symbolic Refinement (Xin Quan et al., 2024)

{{<citation>}}

Xin Quan, Marco Valentino, Louise A. Dennis, André Freitas. (2024)  
**Enhancing Ethical Explanations of Large Language Models through Iterative Symbolic Refinement**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLI, Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2402.00745v1)  

---


**ABSTRACT**  
An increasing amount of research in Natural Language Inference (NLI) focuses on the application and evaluation of Large Language Models (LLMs) and their reasoning capabilities. Despite their success, however, LLMs are still prone to factual errors and inconsistencies in their explanations, offering limited control and interpretability for inference in complex domains. In this paper, we focus on ethical NLI, investigating how hybrid neuro-symbolic techniques can enhance the logical validity and alignment of ethical explanations produced by LLMs. Specifically, we present an abductive-deductive framework named Logic-Explainer, which integrates LLMs with an external backward-chaining solver to refine step-wise natural language explanations and jointly verify their correctness, reduce incompleteness and minimise redundancy. An extensive empirical analysis demonstrates that Logic-Explainer can improve explanations generated via in-context learning methods and Chain-of-Thought (CoT) on challenging ethical NLI tasks, while, at the same time, producing formal proofs describing and supporting models' reasoning. As ethical NLI requires commonsense reasoning to identify underlying moral violations, our results suggest the effectiveness of neuro-symbolic methods for multi-step NLI more broadly, opening new opportunities to enhance the logical consistency, reliability, and alignment of LLMs.

{{</citation>}}


### (11/108) Transforming and Combining Rewards for Aligning Large Language Models (Zihao Wang et al., 2024)

{{<citation>}}

Zihao Wang, Chirag Nagpal, Jonathan Berant, Jacob Eisenstein, Alex D'Amour, Sanmi Koyejo, Victor Veitch. (2024)  
**Transforming and Combining Rewards for Aligning Large Language Models**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2, cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00742v1)  

---


**ABSTRACT**  
A common approach for aligning language models to human preferences is to first learn a reward model from preference data, and then use this reward model to update the language model. We study two closely related problems that arise in this approach. First, any monotone transformation of the reward model preserves preference ranking; is there a choice that is ``better'' than others? Second, we often wish to align language models to multiple properties: how should we combine multiple reward models? Using a probabilistic interpretation of the alignment procedure, we identify a natural choice for transformation for (the common case of) rewards learned from Bradley-Terry preference models. This derived transformation has two important properties. First, it emphasizes improving poorly-performing outputs, rather than outputs that already score well. This mitigates both underfitting (where some prompts are not improved) and reward hacking (where the model learns to exploit misspecification of the reward model). Second, it enables principled aggregation of rewards by linking summation to logical conjunction: the sum of transformed rewards corresponds to the probability that the output is ``good'' in all measured properties, in a sense we make precise. Experiments aligning language models to be both helpful and harmless using RLHF show substantial improvements over the baseline (non-transformed) approach.

{{</citation>}}


### (12/108) Improving Semantic Control in Discrete Latent Spaces with Transformer Quantized Variational Autoencoders (Yingji Zhang et al., 2024)

{{<citation>}}

Yingji Zhang, Danilo S. Carvalho, Marco Valentino, Ian Pratt-Hartmann, Andre Freitas. (2024)  
**Improving Semantic Control in Discrete Latent Spaces with Transformer Quantized Variational Autoencoders**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2402.00723v1)  

---


**ABSTRACT**  
Achieving precise semantic control over the latent spaces of Variational AutoEncoders (VAEs) holds significant value for downstream tasks in NLP as the underlying generative mechanisms could be better localised, explained and improved upon. Recent research, however, has struggled to achieve consistent results, primarily due to the inevitable loss of semantic information in the variational bottleneck and limited control over the decoding mechanism. To overcome these challenges, we investigate discrete latent spaces in Vector Quantized Variational AutoEncoders (VQVAEs) to improve semantic control and generation in Transformer-based VAEs. In particular, We propose T5VQVAE, a novel model that leverages the controllability of VQVAEs to guide the self-attention mechanism in T5 at the token-level, exploiting its full generalization capabilities. Experimental results indicate that T5VQVAE outperforms existing state-of-the-art VAE models, including Optimus, in terms of controllability and preservation of semantic information across different tasks such as auto-encoding of sentences and mathematical expressions, text transfer, and inference. Moreover, T5VQVAE exhibits improved inference capabilities, suggesting potential applications for downstream natural language and symbolic reasoning tasks.

{{</citation>}}


### (13/108) Improving Weak-to-Strong Generalization with Scalable Oversight and Ensemble Learning (Jitao Sang et al., 2024)

{{<citation>}}

Jitao Sang, Yuhang Wang, Jing Zhang, Yanxu Zhu, Chao Kong, Junhong Ye, Shuyu Wei, Jinlin Xiao. (2024)  
**Improving Weak-to-Strong Generalization with Scalable Oversight and Ensemble Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00667v1)  

---


**ABSTRACT**  
This paper presents a follow-up study to OpenAI's recent superalignment work on Weak-to-Strong Generalization (W2SG). Superalignment focuses on ensuring that high-level AI systems remain consistent with human values and intentions when dealing with complex, high-risk tasks. The W2SG framework has opened new possibilities for empirical research in this evolving field. Our study simulates two phases of superalignment under the W2SG framework: the development of general superhuman models and the progression towards superintelligence. In the first phase, based on human supervision, the quality of weak supervision is enhanced through a combination of scalable oversight and ensemble learning, reducing the capability gap between weak teachers and strong students. In the second phase, an automatic alignment evaluator is employed as the weak supervisor. By recursively updating this auto aligner, the capabilities of the weak teacher models are synchronously enhanced, achieving weak-to-strong supervision over stronger student models.We also provide an initial validation of the proposed approach for the first phase. Using the SciQ task as example, we explore ensemble learning for weak teacher models through bagging and boosting. Scalable oversight is explored through two auxiliary settings: human-AI interaction and AI-AI debate. Additionally, the paper discusses the impact of improved weak supervision on enhancing weak-to-strong generalization based on in-context learning. Experiment code and dataset will be released at https://github.com/ADaM-BJTU/W2SG.

{{</citation>}}


### (14/108) Actor Identification in Discourse: A Challenge for LLMs? (Ana Barić et al., 2024)

{{<citation>}}

Ana Barić, Sean Papay, Sebastian Padó. (2024)  
**Actor Identification in Discourse: A Challenge for LLMs?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2402.00620v1)  

---


**ABSTRACT**  
The identification of political actors who put forward claims in public debate is a crucial step in the construction of discourse networks, which are helpful to analyze societal debates. Actor identification is, however, rather challenging: Often, the locally mentioned speaker of a claim is only a pronoun ("He proposed that [claim]"), so recovering the canonical actor name requires discourse understanding. We compare a traditional pipeline of dedicated NLP components (similar to those applied to the related task of coreference) with a LLM, which appears a good match for this generation task. Evaluating on a corpus of German actors in newspaper reports, we find surprisingly that the LLM performs worse. Further analysis reveals that the LLM is very good at identifying the right reference, but struggles to generate the correct canonical form. This points to an underlying issue in LLMs with controlling generated output. Indeed, a hybrid model combining the LLM with a classifier to normalize its output substantially outperforms both initial models.

{{</citation>}}


### (15/108) A Chain-of-Thought Is as Strong as Its Weakest Link: A Benchmark for Verifiers of Reasoning Chains (Alon Jacovi et al., 2024)

{{<citation>}}

Alon Jacovi, Yonatan Bitton, Bernd Bohnet, Jonathan Herzig, Or Honovich, Michael Tseng, Michael Collins, Roee Aharoni, Mor Geva. (2024)  
**A Chain-of-Thought Is as Strong as Its Weakest Link: A Benchmark for Verifiers of Reasoning Chains**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2402.00559v1)  

---


**ABSTRACT**  
Prompting language models to provide step-by-step answers (e.g., "Chain-of-Thought") is the prominent approach for complex reasoning tasks, where more accurate reasoning chains typically improve downstream task performance. Recent literature discusses automatic methods to verify reasoning steps to evaluate and improve their correctness. However, no fine-grained step-level datasets are available to enable thorough evaluation of such verification methods, hindering progress in this direction. We introduce Reveal: Reasoning Verification Evaluation, a new dataset to benchmark automatic verifiers of complex Chain-of-Thought reasoning in open-domain question answering settings. Reveal includes comprehensive labels for the relevance, attribution to evidence passages, and logical correctness of each reasoning step in a language model's answer, across a wide variety of datasets and state-of-the-art language models.

{{</citation>}}


### (16/108) SA-MDKIF: A Scalable and Adaptable Medical Domain Knowledge Injection Framework for Large Language Models (Tianhan Xu et al., 2024)

{{<citation>}}

Tianhan Xu, Zhe Hu, Ling Chen, Bin Li. (2024)  
**SA-MDKIF: A Scalable and Adaptable Medical Domain Knowledge Injection Framework for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2402.00474v1)  

---


**ABSTRACT**  
Recent advances in large language models (LLMs) have demonstrated exceptional performance in various natural language processing (NLP) tasks. However, their effective application in the medical domain is hampered by a lack of medical domain knowledge. In this study, we present SA-MDKIF, a scalable and adaptable framework that aims to inject medical knowledge into general-purpose LLMs through instruction tuning, thereby enabling adaptability for various downstream tasks. SA-MDKIF consists of two stages: skill training and skill adaptation. In the first stage, we define 12 basic medical skills and use AdaLoRA to train these skills based on uniformly formatted instructional datasets that we have constructed. In the next stage, we train the skill router using task-specific downstream data and use this router to integrate the acquired skills with LLMs during inference. Experimental results on 9 different medical tasks show that SA-MDKIF improves performance by 10-20% compared to the original LLMs. Notably, this improvement is particularly pronounced for unseen medical tasks, showing an improvement of up to 30%.

{{</citation>}}


### (17/108) Improving Dialog Safety using Socially Aware Contrastive Learning (Souvik Das et al., 2024)

{{<citation>}}

Souvik Das, Rohini K. Srihari. (2024)  
**Improving Dialog Safety using Socially Aware Contrastive Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Contrastive Learning, Dialog  
[Paper Link](http://arxiv.org/abs/2402.00446v1)  

---


**ABSTRACT**  
State-of-the-art conversational AI systems raise concerns due to their potential risks of generating unsafe, toxic, unethical, or dangerous content. Previous works have developed datasets to teach conversational agents the appropriate social paradigms to respond effectively to specifically designed hazardous content. However, models trained on these adversarial datasets still struggle to recognize subtle unsafe situations that appear naturally in conversations or introduce an inappropriate response in a casual context. To understand the extent of this problem, we study prosociality in both adversarial and casual dialog contexts and audit the response quality of general-purpose language models in terms of propensity to produce unsafe content. We propose a dual-step fine-tuning process to address these issues using a socially aware n-pair contrastive loss. Subsequently, we train a base model that integrates prosocial behavior by leveraging datasets like Moral Integrity Corpus (MIC) and ProsocialDialog. Experimental results on several dialog datasets demonstrate the effectiveness of our approach in generating socially appropriate responses.

{{</citation>}}


### (18/108) From PARIS to LE-PARIS: Toward Patent Response Automation with Recommender Systems and Collaborative Large Language Models (Jung-Mei Chu et al., 2024)

{{<citation>}}

Jung-Mei Chu, Hao-Cheng Lo, Jieh Hsiang, Chun-Chieh Cho. (2024)  
**From PARIS to LE-PARIS: Toward Patent Response Automation with Recommender Systems and Collaborative Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs-IR, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2402.00421v1)  

---


**ABSTRACT**  
In patent prosecution, timely and effective responses to Office Actions (OAs) are crucial for acquiring patents, yet past automation and AI research have scarcely addressed this aspect. To address this gap, our study introduces the Patent Office Action Response Intelligence System (PARIS) and its advanced version, the Large Language Model Enhanced PARIS (LE-PARIS). These systems are designed to expedite the efficiency of patent attorneys in collaboratively handling OA responses. The systems' key features include the construction of an OA Topics Database, development of Response Templates, and implementation of Recommender Systems and LLM-based Response Generation. Our validation involves a multi-paradigmatic analysis using the USPTO Office Action database and longitudinal data of attorney interactions with our systems over six years. Through five studies, we examine the constructiveness of OA topics (studies 1 and 2) using topic modeling and the proposed Delphi process, the efficacy of our proposed hybrid recommender system tailored for OA (both LLM-based and non-LLM-based) (study 3), the quality of response generation (study 4), and the practical value of the systems in real-world scenarios via user studies (study 5). Results demonstrate that both PARIS and LE-PARIS significantly meet key metrics and positively impact attorney performance.

{{</citation>}}


### (19/108) Prompt-Time Symbolic Knowledge Capture with Large Language Models (Tolga Çöplü et al., 2024)

{{<citation>}}

Tolga Çöplü, Arto Bendiken, Andrii Skomorokhov, Eduard Bateiko, Stephen Cobb, Joshua J. Bouw. (2024)  
**Prompt-Time Symbolic Knowledge Capture with Large Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2402.00414v1)  

---


**ABSTRACT**  
Augmenting large language models (LLMs) with user-specific knowledge is crucial for real-world applications, such as personal AI assistants. However, LLMs inherently lack mechanisms for prompt-driven knowledge capture. This paper investigates utilizing the existing LLM capabilities to enable prompt-driven knowledge capture, with a particular emphasis on knowledge graphs. We address this challenge by focusing on prompt-to-triple (P2T) generation. We explore three methods: zero-shot prompting, few-shot prompting, and fine-tuning, and then assess their performance via a specialized synthetic dataset. Our code and datasets are publicly available at https://github.com/HaltiaAI/paper-PTSKC.

{{</citation>}}


### (20/108) Hidding the Ghostwriters: An Adversarial Evaluation of AI-Generated Student Essay Detection (Xinlin Peng et al., 2024)

{{<citation>}}

Xinlin Peng, Ying Zhou, Ben He, Le Sun, Yingfei Sun. (2024)  
**Hidding the Ghostwriters: An Adversarial Evaluation of AI-Generated Student Essay Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00412v1)  

---


**ABSTRACT**  
Large language models (LLMs) have exhibited remarkable capabilities in text generation tasks. However, the utilization of these models carries inherent risks, including but not limited to plagiarism, the dissemination of fake news, and issues in educational exercises. Although several detectors have been proposed to address these concerns, their effectiveness against adversarial perturbations, specifically in the context of student essay writing, remains largely unexplored. This paper aims to bridge this gap by constructing AIG-ASAP, an AI-generated student essay dataset, employing a range of text perturbation methods that are expected to generate high-quality essays while evading detection. Through empirical experiments, we assess the performance of current AIGC detectors on the AIG-ASAP dataset. The results reveal that the existing detectors can be easily circumvented using straightforward automatic adversarial attacks. Specifically, we explore word substitution and sentence substitution perturbation methods that effectively evade detection while maintaining the quality of the generated essays. This highlights the urgent need for more accurate and robust methods to detect AI-generated student essays in the education domain.

{{</citation>}}


### (21/108) Investigating Bias Representations in Llama 2 Chat via Activation Steering (Dawn Lu et al., 2024)

{{<citation>}}

Dawn Lu, Nina Rimsky. (2024)  
**Investigating Bias Representations in Llama 2 Chat via Activation Steering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, GPT, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.00402v1)  

---


**ABSTRACT**  
We address the challenge of societal bias in Large Language Models (LLMs), focusing on the Llama 2 7B Chat model. As LLMs are increasingly integrated into decision-making processes with substantial societal impact, it becomes imperative to ensure these models do not reinforce existing biases. Our approach employs activation steering to probe for and mitigate biases related to gender, race, and religion. This method manipulates model activations to direct responses towards or away from biased outputs, utilizing steering vectors derived from the StereoSet dataset and custom GPT4 generated gender bias prompts. Our findings reveal inherent gender bias in Llama 2 7B Chat, persisting even after Reinforcement Learning from Human Feedback (RLHF). We also observe a predictable negative correlation between bias and the model's tendency to refuse responses. Significantly, our study uncovers that RLHF tends to increase the similarity in the model's representation of different forms of societal biases, which raises questions about the model's nuanced understanding of different forms of bias. This work also provides valuable insights into effective red-teaming strategies for LLMs using activation steering, particularly emphasizing the importance of integrating a refusal vector.

{{</citation>}}


### (22/108) What Does the Bot Say? Opportunities and Risks of Large Language Models in Social Media Bot Detection (Shangbin Feng et al., 2024)

{{<citation>}}

Shangbin Feng, Herun Wan, Ningnan Wang, Zhaoxuan Tan, Minnan Luo, Yulia Tsvetkov. (2024)  
**What Does the Bot Say? Opportunities and Risks of Large Language Models in Social Media Bot Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Social Media  
[Paper Link](http://arxiv.org/abs/2402.00371v1)  

---


**ABSTRACT**  
Social media bot detection has always been an arms race between advancements in machine learning bot detectors and adversarial bot strategies to evade detection. In this work, we bring the arms race to the next level by investigating the opportunities and risks of state-of-the-art large language models (LLMs) in social bot detection. To investigate the opportunities, we design novel LLM-based bot detectors by proposing a mixture-of-heterogeneous-experts framework to divide and conquer diverse user information modalities. To illuminate the risks, we explore the possibility of LLM-guided manipulation of user textual and structured information to evade detection. Extensive experiments with three LLMs on two datasets demonstrate that instruction tuning on merely 1,000 annotated examples produces specialized LLMs that outperform state-of-the-art baselines by up to 9.1% on both datasets, while LLM-guided manipulation strategies could significantly bring down the performance of existing bot detectors by up to 29.6% and harm the calibration and reliability of bot detection systems.

{{</citation>}}


### (23/108) Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration (Shangbin Feng et al., 2024)

{{<citation>}}

Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran, Yulia Tsvetkov. (2024)  
**Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2402.00367v1)  

---


**ABSTRACT**  
Despite efforts to expand the knowledge of large language models (LLMs), knowledge gaps -- missing or outdated information in LLMs -- might always persist given the evolving nature of knowledge. In this work, we study approaches to identify LLM knowledge gaps and abstain from answering questions when knowledge gaps are present. We first adapt existing approaches to model calibration or adaptation through fine-tuning/prompting and analyze their ability to abstain from generating low-confidence outputs. Motivated by their failures in self-reflection and over-reliance on held-out sets, we propose two novel approaches that are based on model collaboration, i.e., LLMs probing other LLMs for knowledge gaps, either cooperatively or competitively. Extensive experiments with three LLMs on four QA tasks featuring diverse knowledge domains demonstrate that both cooperative and competitive approaches to unveiling LLM knowledge gaps achieve up to 19.3% improvements on abstain accuracy against the strongest baseline. Further analysis reveals that our proposed mechanisms could help identify failure cases in retrieval augmentation and pinpoint knowledge gaps in multi-hop reasoning.

{{</citation>}}


### (24/108) IndiVec: An Exploration of Leveraging Large Language Models for Media Bias Detection with Fine-Grained Bias Indicators (Luyang Lin et al., 2024)

{{<citation>}}

Luyang Lin, Lingzhi Wang, Xiaoyan Zhao, Jing Li, Kam-Fai Wong. (2024)  
**IndiVec: An Exploration of Leveraging Large Language Models for Media Bias Detection with Fine-Grained Bias Indicators**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2402.00345v1)  

---


**ABSTRACT**  
This study focuses on media bias detection, crucial in today's era of influential social media platforms shaping individual attitudes and opinions. In contrast to prior work that primarily relies on training specific models tailored to particular datasets, resulting in limited adaptability and subpar performance on out-of-domain data, we introduce a general bias detection framework, IndiVec, built upon large language models. IndiVec begins by constructing a fine-grained media bias database, leveraging the robust instruction-following capabilities of large language models and vector database techniques. When confronted with new input for bias detection, our framework automatically selects the most relevant indicator from the vector database and employs majority voting to determine the input's bias label. IndiVec excels compared to previous methods due to its adaptability (demonstrating consistent performance across diverse datasets from various sources) and explainability (providing explicit top-k indicators to interpret bias predictions). Experimental results on four political bias datasets highlight IndiVec's significant superiority over baselines. Furthermore, additional experiments and analysis provide profound insights into the framework's effectiveness.

{{</citation>}}


### (25/108) Bias in Opinion Summarisation from Pre-training to Adaptation: A Case Study in Political Bias (Nannan Huang et al., 2024)

{{<citation>}}

Nannan Huang, Haytham Fayek, Xiuzhen Zhang. (2024)  
**Bias in Opinion Summarisation from Pre-training to Adaptation: A Case Study in Political Bias**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2402.00322v1)  

---


**ABSTRACT**  
Opinion summarisation aims to summarise the salient information and opinions presented in documents such as product reviews, discussion forums, and social media texts into short summaries that enable users to effectively understand the opinions therein. Generating biased summaries has the risk of potentially swaying public opinion. Previous studies focused on studying bias in opinion summarisation using extractive models, but limited research has paid attention to abstractive summarisation models. In this study, using political bias as a case study, we first establish a methodology to quantify bias in abstractive models, then trace it from the pre-trained models to the task of summarising social media opinions using different models and adaptation methods. We find that most models exhibit intrinsic bias. Using a social media text summarisation dataset and contrasting various adaptation methods, we find that tuning a smaller number of parameters is less biased compared to standard fine-tuning; however, the diversity of topics in training data used for fine-tuning is critical.

{{</citation>}}


### (26/108) Does \textsc{DetectGPT} Fully Utilize Perturbation? Selective Perturbation on Model-Based Contrastive Learning Detector would be Better (Shengchao Liu et al., 2024)

{{<citation>}}

Shengchao Liu, Xiaoming Liu, Yichen Wang, Zehua Cheng, Chengzhengxu Li, Zhaohan Zhang, Yu Lan, Chao Shen. (2024)  
**Does \textsc{DetectGPT} Fully Utilize Perturbation? Selective Perturbation on Model-Based Contrastive Learning Detector would be Better**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, GPT  
[Paper Link](http://arxiv.org/abs/2402.00263v1)  

---


**ABSTRACT**  
The burgeoning capabilities of large language models (LLMs) have raised growing concerns about abuse. DetectGPT, a zero-shot metric-based unsupervised machine-generated text detector, first introduces perturbation and shows great performance improvement. However, DetectGPT's random perturbation strategy might introduce noise, limiting the distinguishability and further performance improvements. Moreover, its logit regression module relies on setting the threshold, which harms the generalizability and applicability of individual or small-batch inputs. Hence, we propose a novel detector, \modelname{}, which uses selective strategy perturbation to relieve the important information loss caused by random masking, and multi-pair contrastive learning to capture the implicit pattern information during perturbation, facilitating few-shot performance. The experiments show that \modelname{} outperforms the SOTA method by 1.20\% in accuracy on average on four public datasets. We further analyze the effectiveness, robustness, and generalization of our perturbation method.

{{</citation>}}


## cs.LG (23)



### (27/108) SymbolicAI: A framework for logic-based approaches combining generative models and solvers (Marius-Constantin Dinu et al., 2024)

{{<citation>}}

Marius-Constantin Dinu, Claudiu Leoveanu-Condrei, Markus Holzleitner, Werner Zellinger, Sepp Hochreiter. (2024)  
**SymbolicAI: A framework for logic-based approaches combining generative models and solvers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SC, cs-SE, cs.LG  
Keywords: AI, Embedding  
[Paper Link](http://arxiv.org/abs/2402.00854v1)  

---


**ABSTRACT**  
We introduce SymbolicAI, a versatile and modular framework employing a logic-based approach to concept learning and flow management in generative processes. SymbolicAI enables the seamless integration of generative models with a diverse range of solvers by treating large language models (LLMs) as semantic parsers that execute tasks based on both natural and formal language instructions, thus bridging the gap between symbolic reasoning and generative AI. We leverage probabilistic programming principles to tackle complex tasks, and utilize differentiable and classical programming paradigms with their respective strengths. The framework introduces a set of polymorphic, compositional, and self-referential operations for data stream manipulation, aligning LLM outputs with user objectives. As a result, we can transition between the capabilities of various foundation models endowed with zero- and few-shot learning capabilities and specialized, fine-tuned models or solvers proficient in addressing specific problems. In turn, the framework facilitates the creation and evaluation of explainable computational graphs. We conclude by introducing a quality measure and its empirical score for evaluating these computational graphs, and propose a benchmark that compares various state-of-the-art LLMs across a set of complex workflows. We refer to the empirical score as the "Vector Embedding for Relational Trajectory Evaluation through Cross-similarity", or VERTEX score for short. The framework codebase and benchmark are linked below.

{{</citation>}}


### (28/108) Data Augmentation Scheme for Raman Spectra with Highly Correlated Annotations (Christoph Lange et al., 2024)

{{<citation>}}

Christoph Lange, Isabel Thiele, Lara Santolin, Sebastian L. Riedel, Maxim Borisyak, Peter Neubauer, M. Nicolas Cruz Bournazou. (2024)  
**Data Augmentation Scheme for Raman Spectra with Highly Correlated Annotations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2402.00851v1)  

---


**ABSTRACT**  
In biotechnology Raman Spectroscopy is rapidly gaining popularity as a process analytical technology (PAT) that measures cell densities, substrate- and product concentrations. As it records vibrational modes of molecules it provides that information non-invasively in a single spectrum. Typically, partial least squares (PLS) is the model of choice to infer information about variables of interest from the spectra. However, biological processes are known for their complexity where convolutional neural networks (CNN) present a powerful alternative. They can handle non-Gaussian noise and account for beam misalignment, pixel malfunctions or the presence of additional substances. However, they require a lot of data during model training, and they pick up non-linear dependencies in the process variables. In this work, we exploit the additive nature of spectra in order to generate additional data points from a given dataset that have statistically independent labels so that a network trained on such data exhibits low correlations between the model predictions. We show that training a CNN on these generated data points improves the performance on datasets where the annotations do not bear the same correlation as the dataset that was used for model training. This data augmentation technique enables us to reuse spectra as training data for new contexts that exhibit different correlations. The additional data allows for building a better and more robust model. This is of interest in scenarios where large amounts of historical data are available but are currently not used for model training. We demonstrate the capabilities of the proposed method using synthetic spectra of Ralstonia eutropha batch cultivations to monitor substrate, biomass and polyhydroxyalkanoate (PHA) biopolymer concentrations during of the experiments.

{{</citation>}}


### (29/108) Score-based Causal Representation Learning: Linear and General Transformations (Burak Varıcı et al., 2024)

{{<citation>}}

Burak Varıcı, Emre Acartürk, Karthikeyan Shanmugam, Abhishek Kumar, Ali Tajer. (2024)  
**Score-based Causal Representation Learning: Linear and General Transformations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2402.00849v1)  

---


**ABSTRACT**  
This paper addresses intervention-based causal representation learning (CRL) under a general nonparametric latent causal model and an unknown transformation that maps the latent variables to the observed variables. Linear and general transformations are investigated. The paper addresses both the \emph{identifiability} and \emph{achievability} aspects. Identifiability refers to determining algorithm-agnostic conditions that ensure recovering the true latent causal variables and the latent causal graph underlying them. Achievability refers to the algorithmic aspects and addresses designing algorithms that achieve identifiability guarantees. By drawing novel connections between \emph{score functions} (i.e., the gradients of the logarithm of density functions) and CRL, this paper designs a \emph{score-based class of algorithms} that ensures both identifiability and achievability. First, the paper focuses on \emph{linear} transformations and shows that one stochastic hard intervention per node suffices to guarantee identifiability. It also provides partial identifiability guarantees for soft interventions, including identifiability up to ancestors for general causal models and perfect latent graph recovery for sufficiently non-linear causal models. Secondly, it focuses on \emph{general} transformations and shows that two stochastic hard interventions per node suffice for identifiability. Notably, one does \emph{not} need to know which pair of interventional environments have the same node intervened.

{{</citation>}}


### (30/108) Position Paper: Bayesian Deep Learning in the Age of Large-Scale AI (Theodore Papamarkou et al., 2024)

{{<citation>}}

Theodore Papamarkou, Maria Skoularidou, Konstantina Palla, Laurence Aitchison, Julyan Arbel, David Dunson, Maurizio Filippone, Vincent Fortuin, Philipp Hennig, Aliaksandr Hubin, Alexander Immer, Theofanis Karaletsos, Mohammad Emtiyaz Khan, Agustinus Kristiadi, Yingzhen Li, Jose Miguel Hernandez Lobato, Stephan Mandt, Christopher Nemeth, Michael A. Osborne, Tim G. J. Rudner, David Rügamer, Yee Whye Teh, Max Welling, Andrew Gordon Wilson, Ruqi Zhang. (2024)  
**Position Paper: Bayesian Deep Learning in the Age of Large-Scale AI**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00809v1)  

---


**ABSTRACT**  
In the current landscape of deep learning research, there is a predominant emphasis on achieving high predictive accuracy in supervised tasks involving large image and language datasets. However, a broader perspective reveals a multitude of overlooked metrics, tasks, and data types, such as uncertainty, active and continual learning, and scientific data, that demand attention. Bayesian deep learning (BDL) constitutes a promising avenue, offering advantages across these diverse settings. This paper posits that BDL can elevate the capabilities of deep learning. It revisits the strengths of BDL, acknowledges existing challenges, and highlights some exciting research avenues aimed at addressing these obstacles. Looking ahead, the discussion focuses on possible ways to combine large-scale foundation models with BDL to unlock their full potential.

{{</citation>}}


### (31/108) Distilling Conditional Diffusion Models for Offline Reinforcement Learning through Trajectory Stitching (Shangzhe Li et al., 2024)

{{<citation>}}

Shangzhe Li, Xinhua Zhang. (2024)  
**Distilling Conditional Diffusion Models for Offline Reinforcement Learning through Trajectory Stitching**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.00807v1)  

---


**ABSTRACT**  
Deep generative models have recently emerged as an effective approach to offline reinforcement learning. However, their large model size poses challenges in computation. We address this issue by proposing a knowledge distillation method based on data augmentation. In particular, high-return trajectories are generated from a conditional diffusion model, and they are blended with the original trajectories through a novel stitching algorithm that leverages a new reward generator. Applying the resulting dataset to behavioral cloning, the learned shallow policy whose size is much smaller outperforms or nearly matches deep generative planners on several D4RL benchmarks.

{{</citation>}}


### (32/108) Signal Quality Auditing for Time-series Data (Chufan Gao et al., 2024)

{{<citation>}}

Chufan Gao, Nicholas Gisolfi, Artur Dubrawski. (2024)  
**Signal Quality Auditing for Time-series Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2402.00803v1)  

---


**ABSTRACT**  
Signal quality assessment (SQA) is required for monitoring the reliability of data acquisition systems, especially in AI-driven Predictive Maintenance (PMx) application contexts. SQA is vital for addressing "silent failures" of data acquisition hardware and software, which when unnoticed, misinform the users of data, creating the risk for incorrect decisions with unintended or even catastrophic consequences. We have developed an open-source software implementation of signal quality indices (SQIs) for the analysis of time-series data. We codify a range of SQIs, demonstrate them using established benchmark data, and show that they can be effective for signal quality assessment. We also study alternative approaches to denoising time-series data in an attempt to improve the quality of the already degraded signal, and evaluate them empirically on relevant real-world data. To our knowledge, our software toolkit is the first to provide an open source implementation of a broad range of signal quality assessment and improvement techniques validated on publicly available benchmark data for ease of reproducibility. The generality of our framework can be easily extended to assessing reliability of arbitrary time-series measurements in complex systems, especially when morphological patterns of the waveform shapes and signal periodicity are of key interest in downstream analyses.

{{</citation>}}


### (33/108) Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents (Zelong Li et al., 2024)

{{<citation>}}

Zelong Li, Wenyue Hua, Hao Wang, He Zhu, Yongfeng Zhang. (2024)  
**Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-FL, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2402.00798v1)  

---


**ABSTRACT**  
Recent advancements on Large Language Models (LLMs) enable AI Agents to automatically generate and execute multi-step plans to solve complex tasks. However, since LLM's content generation process is hardly controllable, current LLM-based agents frequently generate invalid or non-executable plans, which jeopardizes the performance of the generated plans and corrupts users' trust in LLM-based agents. In response, this paper proposes a novel ``Formal-LLM'' framework for LLM-based agents by integrating the expressiveness of natural language and the precision of formal language. Specifically, the framework allows human users to express their requirements or constraints for the planning process as an automaton. A stack-based LLM plan generation process is then conducted under the supervision of the automaton to ensure that the generated plan satisfies the constraints, making the planning process controllable. We conduct experiments on both benchmark tasks and practical real-life tasks, and our framework achieves over 50% overall performance increase, which validates the feasibility and effectiveness of employing Formal-LLM to guide the plan generation of agents, preventing the agents from generating invalid and unsuccessful plans. Further, more controllable LLM-based agents can facilitate the broader utilization of LLM in application scenarios where high validity of planning is essential. The work is open-sourced at https://github.com/agiresearch/Formal-LLM.

{{</citation>}}


### (34/108) LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law (Toni J. B. Liu et al., 2024)

{{<citation>}}

Toni J. B. Liu, Nicolas Boullé, Raphaël Sarfati, Christopher J. Earls. (2024)  
**LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2402.00795v1)  

---


**ABSTRACT**  
Pretrained large language models (LLMs) are surprisingly effective at performing zero-shot tasks, including time-series forecasting. However, understanding the mechanisms behind such capabilities remains highly challenging due to the complexity of the models. In this paper, we study LLMs' ability to extrapolate the behavior of dynamical systems whose evolution is governed by principles of physical interest. Our results show that LLaMA 2, a language model trained primarily on texts, achieves accurate predictions of dynamical system time series without fine-tuning or prompt engineering. Moreover, the accuracy of the learned physical rules increases with the length of the input context window, revealing an in-context version of neural scaling law. Along the way, we present a flexible and efficient algorithm for extracting probability density functions of multi-digit numbers directly from LLMs.

{{</citation>}}


### (35/108) Distinguishing the Indistinguishable: Human Expertise in Algorithmic Prediction (Rohan Alur et al., 2024)

{{<citation>}}

Rohan Alur, Manish Raghavan, Devavrat Shah. (2024)  
**Distinguishing the Indistinguishable: Human Expertise in Algorithmic Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-HC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00793v1)  

---


**ABSTRACT**  
We introduce a novel framework for incorporating human expertise into algorithmic predictions. Our approach focuses on the use of human judgment to distinguish inputs which `look the same' to any feasible predictive algorithm. We argue that this framing clarifies the problem of human/AI collaboration in prediction tasks, as experts often have access to information -- particularly subjective information -- which is not encoded in the algorithm's training data. We use this insight to develop a set of principled algorithms for selectively incorporating human feedback only when it improves the performance of any feasible predictor. We find empirically that although algorithms often outperform their human counterparts on average, human judgment can significantly improve algorithmic predictions on specific instances (which can be identified ex-ante). In an X-ray classification task, we find that this subset constitutes nearly 30% of the patient population. Our approach provides a natural way of uncovering this heterogeneity and thus enabling effective human-AI collaboration.

{{</citation>}}


### (36/108) Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces (Chloe Wang et al., 2024)

{{<citation>}}

Chloe Wang, Oleksii Tsepa, Jun Ma, Bo Wang. (2024)  
**Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.00789v1)  

---


**ABSTRACT**  
Attention mechanisms have been widely used to capture long-range dependencies among nodes in Graph Transformers. Bottlenecked by the quadratic computational cost, attention mechanisms fail to scale in large graphs. Recent improvements in computational efficiency are mainly achieved by attention sparsification with random or heuristic-based graph subsampling, which falls short in data-dependent context reasoning. State space models (SSMs), such as Mamba, have gained prominence for their effectiveness and efficiency in modeling long-range dependencies in sequential data. However, adapting SSMs to non-sequential graph data presents a notable challenge. In this work, we introduce Graph-Mamba, the first attempt to enhance long-range context modeling in graph networks by integrating a Mamba block with the input-dependent node selection mechanism. Specifically, we formulate graph-centric node prioritization and permutation strategies to enhance context-aware reasoning, leading to a substantial improvement in predictive performance. Extensive experiments on ten benchmark datasets demonstrate that Graph-Mamba outperforms state-of-the-art methods in long-range graph prediction tasks, with a fraction of the computational cost in both FLOPs and GPU memory consumption. The code and models are publicly available at https://github.com/bowang-lab/Graph-Mamba.

{{</citation>}}


### (37/108) Dense Reward for Free in Reinforcement Learning from Human Feedback (Alex J. Chan et al., 2024)

{{<citation>}}

Alex J. Chan, Hao Sun, Samuel Holt, Mihaela van der Schaar. (2024)  
**Dense Reward for Free in Reinforcement Learning from Human Feedback**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.00782v1)  

---


**ABSTRACT**  
Reinforcement Learning from Human Feedback (RLHF) has been credited as the key advance that has allowed Large Language Models (LLMs) to effectively follow instructions and produce useful assistance. Classically, this involves generating completions from the LLM in response to a query before using a separate reward model to assign a score to the full completion. As an auto-regressive process, the LLM has to take many "actions" (selecting individual tokens) and only receives a single, sparse reward at the end of an episode, a setup that is known to be difficult to optimise in traditional reinforcement learning. In this work we leverage the fact that the reward model contains more information than just its scalar output, in particular, it calculates an attention map over tokens as part of the transformer architecture. We use these attention weights to redistribute the reward along the whole completion, effectively densifying the signal and highlighting the most important tokens, all without incurring extra computational cost or requiring any additional modelling. We demonstrate that, theoretically, this approach is equivalent to potential-based reward shaping, ensuring that the optimal policy remains unchanged. Empirically, we show that it stabilises training, accelerates the rate of learning, and, in practical cases, may lead to better local optima.

{{</citation>}}


### (38/108) Benefits of Transformer: In-Context Learning in Linear Regression Tasks with Unstructured Data (Yue Xing et al., 2024)

{{<citation>}}

Yue Xing, Xiaofeng Lin, Namjoon Suh, Qifan Song, Guang Cheng. (2024)  
**Benefits of Transformer: In-Context Learning in Linear Regression Tasks with Unstructured Data**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.00743v1)  

---


**ABSTRACT**  
In practice, it is observed that transformer-based models can learn concepts in context in the inference stage. While existing literature, e.g., \citet{zhang2023trained,huang2023context}, provide theoretical explanations on this in-context learning ability, they assume the input $x_i$ and the output $y_i$ for each sample are embedded in the same token (i.e., structured data). However, in reality, they are presented in two tokens (i.e., unstructured data \cite{wibisono2023role}). In this case, this paper conducts experiments in linear regression tasks to study the benefits of the architecture of transformers and provides some corresponding theoretical intuitions to explain why the transformer can learn from unstructured data. We study the exact components in a transformer that facilitate the in-context learning. In particular, we observe that (1) a transformer with two layers of softmax (self-)attentions with look-ahead attention mask can learn from the prompt if $y_i$ is in the token next to $x_i$ for each example; (2) positional encoding can further improve the performance; and (3) multi-head attention with a high input embedding dimension has a better prediction performance than single-head attention.

{{</citation>}}


### (39/108) Tropical Decision Boundaries for Neural Networks Are Robust Against Adversarial Attacks (Kurt Pasque et al., 2024)

{{<citation>}}

Kurt Pasque, Christopher Teska, Ruriko Yoshida, Keiji Miura, Jefferson Huang. (2024)  
**Tropical Decision Boundaries for Neural Networks Are Robust Against Adversarial Attacks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG, math-CO  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2402.00576v1)  

---


**ABSTRACT**  
We introduce a simple, easy to implement, and computationally efficient tropical convolutional neural network architecture that is robust against adversarial attacks. We exploit the tropical nature of piece-wise linear neural networks by embedding the data in the tropical projective torus in a single hidden layer which can be added to any model. We study the geometry of its decision boundary theoretically and show its robustness against adversarial attacks on image datasets using computational experiments.

{{</citation>}}


### (40/108) Understanding the Expressive Power and Mechanisms of Transformer for Sequence Modeling (Mingze Wang et al., 2024)

{{<citation>}}

Mingze Wang, Weinan E. (2024)  
**Understanding the Expressive Power and Mechanisms of Transformer for Sequence Modeling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.00522v1)  

---


**ABSTRACT**  
We conduct a systematic study of the approximation properties of Transformer for sequence modeling with long, sparse and complicated memory. We investigate the mechanisms through which different components of Transformer, such as the dot-product self-attention, positional encoding and feed-forward layer, affect its expressive power, and we study their combined effects through establishing explicit approximation rates. Our study reveals the roles of critical parameters in the Transformer, such as the number of layers and the number of attention heads, and these insights also provide natural suggestions for alternative architectures.

{{</citation>}}


### (41/108) EE-Tuning: An Economical yet Scalable Solution for Tuning Early-Exit Large Language Models (Xuchen Pan et al., 2024)

{{<citation>}}

Xuchen Pan, Yanxi Chen, Yaliang Li, Bolin Ding, Jingren Zhou. (2024)  
**EE-Tuning: An Economical yet Scalable Solution for Tuning Early-Exit Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00518v1)  

---


**ABSTRACT**  
This work introduces EE-Tuning, a lightweight and economical solution to training/tuning early-exit large language models (LLMs). In contrast to the common approach of full-parameter pre-training, EE-Tuning augments any pre-trained (and possibly fine-tuned) standard LLM with additional early-exit layers that are tuned in a parameter-efficient manner, which requires significantly less computational resources and training data. Our implementation of EE-Tuning achieves outstanding training efficiency via extensive performance optimizations, as well as scalability due to its full compatibility with 3D parallelism. Results of systematic experiments validate the efficacy of EE-Tuning, confirming that effective early-exit LLM inference can be achieved with a limited training budget. In hope of making early-exit LLMs accessible to the community, we release the source code of our implementation of EE-Tuning at https://github.com/pan-x-c/EE-LLM.

{{</citation>}}


### (42/108) CPT: Competence-progressive Training Strategy for Few-shot Node Classification (Qilong Yan et al., 2024)

{{<citation>}}

Qilong Yan, Yufeng Zhang, Jinghao Zhang, Jingpu Duan, Jian Yin. (2024)  
**CPT: Competence-progressive Training Strategy for Few-shot Node Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2402.00450v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have made significant advancements in node classification, but their success relies on sufficient labeled nodes per class in the training data. Real-world graph data often exhibits a long-tail distribution with sparse labels, emphasizing the importance of GNNs' ability in few-shot node classification, which entails categorizing nodes with limited data. Traditional episodic meta-learning approaches have shown promise in this domain, but they face an inherent limitation: it might lead the model to converge to suboptimal solutions because of random and uniform task assignment, ignoring task difficulty levels. This could lead the meta-learner to face complex tasks too soon, hindering proper learning. Ideally, the meta-learner should start with simple concepts and advance to more complex ones, like human learning. So, we introduce CPT, a novel two-stage curriculum learning method that aligns task difficulty with the meta-learner's progressive competence, enhancing overall performance. Specifically, in CPT's initial stage, the focus is on simpler tasks, fostering foundational skills for engaging with complex tasks later. Importantly, the second stage dynamically adjusts task difficulty based on the meta-learner's growing competence, aiming for optimal knowledge acquisition. Extensive experiments on popular node classification datasets demonstrate significant improvements of our strategy over existing methods.

{{</citation>}}


### (43/108) Merging Multi-Task Models via Weight-Ensembling Mixture of Experts (Anke Tang et al., 2024)

{{<citation>}}

Anke Tang, Li Shen, Yong Luo, Nan Yin, Lefei Zhang, Dacheng Tao. (2024)  
**Merging Multi-Task Models via Weight-Ensembling Mixture of Experts**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.00433v1)  

---


**ABSTRACT**  
Merging various task-specific Transformer-based models trained on different tasks into a single unified model can execute all the tasks concurrently. Previous methods, exemplified by task arithmetic, have been proven to be both effective and scalable. Existing methods have primarily focused on seeking a static optimal solution within the original model parameter space. A notable challenge is mitigating the interference between parameters of different models, which can substantially deteriorate performance. In this paper, we propose to merge most of the parameters while upscaling the MLP of the Transformer layers to a weight-ensembling mixture of experts (MoE) module, which can dynamically integrate shared and task-specific knowledge based on the input, thereby providing a more flexible solution that can adapt to the specific needs of each instance. Our key insight is that by identifying and separating shared knowledge and task-specific knowledge, and then dynamically integrating them, we can mitigate the parameter interference problem to a great extent. We conduct the conventional multi-task model merging experiments and evaluate the generalization and robustness of our method. The results demonstrate the effectiveness of our method and provide a comprehensive understanding of our method. The code is available at https://anonymous.4open.science/r/weight-ensembling_MoE-67C9/

{{</citation>}}


### (44/108) Adaptive Primal-Dual Method for Safe Reinforcement Learning (Weiqin Chen et al., 2024)

{{<citation>}}

Weiqin Chen, James Onyejizu, Long Vu, Lan Hoang, Dharmashankar Subramanian, Koushik Kar, Sandipan Mishra, Santiago Paternain. (2024)  
**Adaptive Primal-Dual Method for Safe Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.00355v1)  

---


**ABSTRACT**  
Primal-dual methods have a natural application in Safe Reinforcement Learning (SRL), posed as a constrained policy optimization problem. In practice however, applying primal-dual methods to SRL is challenging, due to the inter-dependency of the learning rate (LR) and Lagrangian multipliers (dual variables) each time an embedded unconstrained RL problem is solved. In this paper, we propose, analyze and evaluate adaptive primal-dual (APD) methods for SRL, where two adaptive LRs are adjusted to the Lagrangian multipliers so as to optimize the policy in each iteration. We theoretically establish the convergence, optimality and feasibility of the APD algorithm. Finally, we conduct numerical evaluation of the practical APD algorithm with four well-known environments in Bullet-Safey-Gym employing two state-of-the-art SRL algorithms: PPO-Lagrangian and DDPG-Lagrangian. All experiments show that the practical APD algorithm outperforms (or achieves comparable performance) and attains more stable training than the constant LR cases. Additionally, we substantiate the robustness of selecting the two adaptive LRs by empirical evidence.

{{</citation>}}


### (45/108) Machine Unlearning for Image-to-Image Generative Models (Guihong Li et al., 2024)

{{<citation>}}

Guihong Li, Hsiang Hsu, Chun-Fu, Chen, Radu Marculescu. (2024)  
**Machine Unlearning for Image-to-Image Generative Models**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2402.00351v1)  

---


**ABSTRACT**  
Machine unlearning has emerged as a new paradigm to deliberately forget data samples from a given model in order to adhere to stringent regulations. However, existing machine unlearning methods have been primarily focused on classification models, leaving the landscape of unlearning for generative models relatively unexplored. This paper serves as a bridge, addressing the gap by providing a unifying framework of machine unlearning for image-to-image generative models. Within this framework, we propose a computationally-efficient algorithm, underpinned by rigorous theoretical analysis, that demonstrates negligible performance degradation on the retain samples, while effectively removing the information from the forget samples. Empirical studies on two large-scale datasets, ImageNet-1K and Places-365, further show that our algorithm does not rely on the availability of the retain samples, which further complies with data retention policy. To our best knowledge, this work is the first that represents systemic, theoretical, empirical explorations of machine unlearning specifically tailored for image-to-image generative models. Our code is available at https://github.com/jpmorganchase/l2l-generator-unlearning.

{{</citation>}}


### (46/108) Diverse Explanations from Data-driven and Domain-driven Perspectives for Machine Learning Models (Sichao Li et al., 2024)

{{<citation>}}

Sichao Li, Amanda Barnard. (2024)  
**Diverse Explanations from Data-driven and Domain-driven Perspectives for Machine Learning Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00347v1)  

---


**ABSTRACT**  
Explanations of machine learning models are important, especially in scientific areas such as chemistry, biology, and physics, where they guide future laboratory experiments and resource requirements. These explanations can be derived from well-trained machine learning models (data-driven perspective) or specific domain knowledge (domain-driven perspective). However, there exist inconsistencies between these perspectives due to accurate yet misleading machine learning models and various stakeholders with specific needs, wants, or aims. This paper calls attention to these inconsistencies and suggests a way to find an accurate model with expected explanations that reinforce physical laws and meet stakeholders' requirements from a set of equally-good models, also known as Rashomon sets. Our goal is to foster a comprehensive understanding of these inconsistencies and ultimately contribute to the integration of eXplainable Artificial Intelligence (XAI) into scientific domains.

{{</citation>}}


### (47/108) Comparing Spectral Bias and Robustness For Two-Layer Neural Networks: SGD vs Adaptive Random Fourier Features (Aku Kammonen et al., 2024)

{{<citation>}}

Aku Kammonen, Lisi Liang, Anamika Pandey, Raúl Tempone. (2024)  
**Comparing Spectral Bias and Robustness For Two-Layer Neural Networks: SGD vs Adaptive Random Fourier Features**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2402.00332v1)  

---


**ABSTRACT**  
We present experimental results highlighting two key differences resulting from the choice of training algorithm for two-layer neural networks. The spectral bias of neural networks is well known, while the spectral bias dependence on the choice of training algorithm is less studied. Our experiments demonstrate that an adaptive random Fourier features algorithm (ARFF) can yield a spectral bias closer to zero compared to the stochastic gradient descent optimizer (SGD). Additionally, we train two identically structured classifiers, employing SGD and ARFF, to the same accuracy levels and empirically assess their robustness against adversarial noise attacks.

{{</citation>}}


### (48/108) Control in Stochastic Environment with Delays: A Model-based Reinforcement Learning Approach (Zhiyuan Yao et al., 2024)

{{<citation>}}

Zhiyuan Yao, Ionut Florescu, Chihoon Lee. (2024)  
**Control in Stochastic Environment with Delays: A Model-based Reinforcement Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.00313v1)  

---


**ABSTRACT**  
In this paper we are introducing a new reinforcement learning method for control problems in environments with delayed feedback. Specifically, our method employs stochastic planning, versus previous methods that used deterministic planning. This allows us to embed risk preference in the policy optimization problem. We show that this formulation can recover the optimal policy for problems with deterministic transitions. We contrast our policy with two prior methods from literature. We apply the methodology to simple tasks to understand its features. Then, we compare the performance of the methods in controlling multiple Atari games.

{{</citation>}}


### (49/108) Efficient Non-Parametric Uncertainty Quantification for Black-Box Large Language Models and Decision Planning (Yao-Hung Hubert Tsai et al., 2024)

{{<citation>}}

Yao-Hung Hubert Tsai, Walter Talbott, Jian Zhang. (2024)  
**Efficient Non-Parametric Uncertainty Quantification for Black-Box Large Language Models and Decision Planning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2402.00251v1)  

---


**ABSTRACT**  
Step-by-step decision planning with large language models (LLMs) is gaining attention in AI agent development. This paper focuses on decision planning with uncertainty estimation to address the hallucination problem in language models. Existing approaches are either white-box or computationally demanding, limiting use of black-box proprietary LLMs within budgets. The paper's first contribution is a non-parametric uncertainty quantification method for LLMs, efficiently estimating point-wise dependencies between input-decision on the fly with a single inference, without access to token logits. This estimator informs the statistical interpretation of decision trustworthiness. The second contribution outlines a systematic design for a decision-making agent, generating actions like ``turn on the bathroom light'' based on user prompts such as ``take a bath''. Users will be asked to provide preferences when more than one action has high estimated point-wise dependencies. In conclusion, our uncertainty estimation and decision-making agent design offer a cost-efficient approach for AI agent development.

{{</citation>}}


## cs.CR (4)



### (50/108) X-CBA: Explainability Aided CatBoosted Anomal-E for Intrusion Detection System (Kiymet Kaya et al., 2024)

{{<citation>}}

Kiymet Kaya, Elif Ak, Sumeyye Bas, Berk Canberk, Sule Gunduz Oguducu. (2024)  
**X-CBA: Explainability Aided CatBoosted Anomal-E for Intrusion Detection System**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs-NI, cs.CR  
Keywords: AI, GNN, Graph Neural Network, Graph Neural Networks, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2402.00839v1)  

---


**ABSTRACT**  
The effectiveness of Intrusion Detection Systems (IDS) is critical in an era where cyber threats are becoming increasingly complex. Machine learning (ML) and deep learning (DL) models provide an efficient and accurate solution for identifying attacks and anomalies in computer networks. However, using ML and DL models in IDS has led to a trust deficit due to their non-transparent decision-making. This transparency gap in IDS research is significant, affecting confidence and accountability. To address, this paper introduces a novel Explainable IDS approach, called X-CBA, that leverages the structural advantages of Graph Neural Networks (GNNs) to effectively process network traffic data, while also adapting a new Explainable AI (XAI) methodology. Unlike most GNN-based IDS that depend on labeled network traffic and node features, thereby overlooking critical packet-level information, our approach leverages a broader range of traffic data through network flows, including edge attributes, to improve detection capabilities and adapt to novel threats. Through empirical testing, we establish that our approach not only achieves high accuracy with 99.47% in threat detection but also advances the field by providing clear, actionable explanations of its analytical outcomes. This research also aims to bridge the current gap and facilitate the broader integration of ML/DL technologies in cybersecurity defenses by offering a local and global explainability solution that is both precise and interpretable.

{{</citation>}}


### (51/108) From Pre-Quantum to Post-Quantum IoT Security: A Survey on Quantum-Resistant Cryptosystems for the Internet of Things (Tiago M. Fernandez-Carames, 2024)

{{<citation>}}

Tiago M. Fernandez-Carames. (2024)  
**From Pre-Quantum to Post-Quantum IoT Security: A Survey on Quantum-Resistant Cryptosystems for the Internet of Things**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2402.00790v1)  

---


**ABSTRACT**  
This article provides a survey on what can be called post-quantum IoT systems (IoT systems protected from the currently known quantum computing attacks): the main post-quantum cryptosystems and initiatives are reviewed, the most relevant IoT architectures and challenges are analyzed, and the expected future trends are indicated. Thus, this paper is aimed at providing a wide view of post-quantum IoT security and give useful guidelines to the future post-quantum IoT developers.

{{</citation>}}


### (52/108) Ocassionally Secure: A Comparative Analysis of Code Generation Assistants (Ran Elgedawy et al., 2024)

{{<citation>}}

Ran Elgedawy, John Sadik, Senjuti Dutta, Anuj Gautam, Konstantinos Georgiou, Farzin Gholamrezae, Fujiao Ji, Kyungchan Lim, Qian Liu, Scott Ruoti. (2024)  
**Ocassionally Secure: A Comparative Analysis of Code Generation Assistants**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2402.00689v1)  

---


**ABSTRACT**  
$ $Large Language Models (LLMs) are being increasingly utilized in various applications, with code generations being a notable example. While previous research has shown that LLMs have the capability to generate both secure and insecure code, the literature does not take into account what factors help generate secure and effective code. Therefore in this paper we focus on identifying and understanding the conditions and contexts in which LLMs can be effectively and safely deployed in real-world scenarios to generate quality code. We conducted a comparative analysis of four advanced LLMs--GPT-3.5 and GPT-4 using ChatGPT and Bard and Gemini from Google--using 9 separate tasks to assess each model's code generation capabilities. We contextualized our study to represent the typical use cases of a real-life developer employing LLMs for everyday tasks as work. Additionally, we place an emphasis on security awareness which is represented through the use of two distinct versions of our developer persona. In total, we collected 61 code outputs and analyzed them across several aspects: functionality, security, performance, complexity, and reliability. These insights are crucial for understanding the models' capabilities and limitations, guiding future development and practical applications in the field of automated code generation.

{{</citation>}}


### (53/108) An Investigation of Hardware Security Bug Characteristics in Open-Source Projects (Joey Ah-kiow et al., 2024)

{{<citation>}}

Joey Ah-kiow, Benjamin Tan. (2024)  
**An Investigation of Hardware Security Bug Characteristics in Open-Source Projects**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2402.00684v1)  

---


**ABSTRACT**  
Hardware security is an important concern of system security as vulnerabilities can arise from design errors introduced throughout the development lifecycle. Recent works have proposed techniques to detect hardware security bugs, such as static analysis, fuzzing, and symbolic execution. However, the fundamental properties of hardware security bugs remain relatively unexplored. To gain a better understanding of hardware security bugs, we perform a deep dive into the popular OpenTitan project, including its bug reports and bug fixes. We manually classify the bugs as relevant to functionality or security and analyze characteristics, such as the impact and location of security bugs, and the size of their bug fixes. We also investigate relationships between security impact and bug management during development. Finally, we propose an abstract syntax tree-based analysis to identify the syntactic characteristics of bug fixes. Our results show that 53% of the bugs in OpenTitan have potential security implications and that 55% of all bug fixes modify only one file. Our findings underscore the importance of security-aware development practices and tools and motivate the development of techniques that leverage the highly localized nature of hardware bugs.

{{</citation>}}


## cs.CY (2)



### (54/108) Common errors in Generative AI systems used for knowledge extraction in the climate action domain (Denis Havlik et al., 2024)

{{<citation>}}

Denis Havlik, Marcelo Pias. (2024)  
**Common errors in Generative AI systems used for knowledge extraction in the climate action domain**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, GPT, Generative AI, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.00830v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) and, more specifically, the Generative Pre-Trained Transformers (GPT) can help stakeholders in climate action explore digital knowledge bases and extract and utilize climate action knowledge in a sustainable manner. However, LLMs are "probabilistic models of knowledge bases" that excel at generating convincing texts but cannot be entirely relied upon due to the probabilistic nature of the information produced. This brief report illustrates the problem space with examples of LLM responses to some of the questions of relevance to climate action.

{{</citation>}}


### (55/108) Responsible developments and networking research: a reflection beyond a paper ethical statement (Daphne Tuncer et al., 2024)

{{<citation>}}

Daphne Tuncer, Marc Bruyere. (2024)  
**Responsible developments and networking research: a reflection beyond a paper ethical statement**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00442v1)  

---


**ABSTRACT**  
Several recent initiatives have proposed new directions for research practices and their operations in the computer science community, from updated codes of conduct that clarify the use of AI-assisted tools to the inclusion of ethical statements and the organization of working groups on the environmental footprint of digitalization. In this position paper, we focus on the specific case of networking research. We reflect on the technical realization of the community and its incidence beyond techno-centric contributions. In particular, we structure the discussion around two frameworks that were recently developed in different contexts to describe the sense of engagement and responsibilities to which the practitioner of a computing-related area may be confronted.

{{</citation>}}


## cs.DS (1)



### (56/108) The En Route Truck-Drone Delivery Problem (Danny Krizanc et al., 2024)

{{<citation>}}

Danny Krizanc, Lata Narayanan, Jaroslav Opatrny, Denis Pankratov. (2024)  
**The En Route Truck-Drone Delivery Problem**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs.DS  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2402.00829v1)  

---


**ABSTRACT**  
We study the truck-drone cooperative delivery problem in a setting where a single truck carrying a drone travels at constant speed on a straight-line trajectory/street. Delivery to clients located in the plane and not on the truck's trajectory is performed by the drone, which has limited carrying capacity and flying range, and whose battery can be recharged when on the truck. We show that the problem of maximizing the number of deliveries is strongly NP-hard even in this simple setting. We present a 2-approximation algorithm for the problem, and an optimal algorithm for a non-trivial family of instances.

{{</citation>}}


## eess.AS (2)



### (57/108) Efficient Fine-tuning of Audio Spectrogram Transformers via Soft Mixture of Adapters (Umberto Cappellazzo et al., 2024)

{{<citation>}}

Umberto Cappellazzo, Daniele Falavigna, Alessio Brutti. (2024)  
**Efficient Fine-tuning of Audio Spectrogram Transformers via Soft Mixture of Adapters**  

---
Primary Category: eess.AS  
Categories: cs-AI, eess-AS, eess.AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.00828v1)  

---


**ABSTRACT**  
Mixture of Experts (MoE) architectures have recently started burgeoning due to their ability to scale model's capacity while maintaining the computational cost affordable. Furthermore, they can be applied to both Transformers and State Space Models, the current state-of-the-art models in numerous fields. While MoE has been mostly investigated for the pre-training stage, its use in parameter-efficient transfer learning settings is under-explored. To narrow this gap, this paper attempts to demystify the use of MoE for parameter-efficient fine-tuning of Audio Spectrogram Transformers to audio and speech downstream tasks. Specifically, we propose Soft Mixture of Adapters (Soft-MoA). It exploits adapters as the experts and, leveraging the recent Soft MoE method, it relies on a soft assignment between the input tokens and experts to keep the computational time limited. Extensive experiments across 4 benchmarks demonstrate that Soft-MoA outperforms the single adapter method and performs on par with the dense MoA counterpart. We finally present ablation studies on key elements of Soft-MoA, showing for example that Soft-MoA achieves better scaling with more experts, as well as ensuring that all experts contribute to the computation of the output tokens, thus dispensing with the expert imbalance issue.

{{</citation>}}


### (58/108) PAM: Prompting Audio-Language Models for Audio Quality Assessment (Soham Deshmukh et al., 2024)

{{<citation>}}

Soham Deshmukh, Dareen Alharthi, Benjamin Elizalde, Hannes Gamper, Mahmoud Al Ismail, Rita Singh, Bhiksha Raj, Huaming Wang. (2024)  
**PAM: Prompting Audio-Language Models for Audio Quality Assessment**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00282v1)  

---


**ABSTRACT**  
While audio quality is a key performance metric for various audio processing tasks, including generative modeling, its objective measurement remains a challenge. Audio-Language Models (ALMs) are pre-trained on audio-text pairs that may contain information about audio quality, the presence of artifacts, or noise. Given an audio input and a text prompt related to quality, an ALM can be used to calculate a similarity score between the two. Here, we exploit this capability and introduce PAM, a no-reference metric for assessing audio quality for different audio processing tasks. Contrary to other "reference-free" metrics, PAM does not require computing embeddings on a reference dataset nor training a task-specific model on a costly set of human listening scores. We extensively evaluate the reliability of PAM against established metrics and human listening scores on four tasks: text-to-audio (TTA), text-to-music generation (TTM), text-to-speech (TTS), and deep noise suppression (DNS). We perform multiple ablation studies with controlled distortions, in-the-wild setups, and prompt choices. Our evaluation shows that PAM correlates well with existing metrics and human listening scores. These results demonstrate the potential of ALMs for computing a general-purpose audio quality metric.

{{</citation>}}


## cs.MA (1)



### (59/108) Learning and Calibrating Heterogeneous Bounded Rational Market Behaviour with Multi-Agent Reinforcement Learning (Benjamin Patrick Evans et al., 2024)

{{<citation>}}

Benjamin Patrick Evans, Sumitra Ganesh. (2024)  
**Learning and Calibrating Heterogeneous Bounded Rational Market Behaviour with Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-CE, cs-GT, cs-LG, cs-MA, cs.MA, econ-GN, q-fin-EC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.00787v1)  

---


**ABSTRACT**  
Agent-based models (ABMs) have shown promise for modelling various real world phenomena incompatible with traditional equilibrium analysis. However, a critical concern is the manual definition of behavioural rules in ABMs. Recent developments in multi-agent reinforcement learning (MARL) offer a way to address this issue from an optimisation perspective, where agents strive to maximise their utility, eliminating the need for manual rule specification. This learning-focused approach aligns with established economic and financial models through the use of rational utility-maximising agents. However, this representation departs from the fundamental motivation for ABMs: that realistic dynamics emerging from bounded rationality and agent heterogeneity can be modelled. To resolve this apparent disparity between the two approaches, we propose a novel technique for representing heterogeneous processing-constrained agents within a MARL framework. The proposed approach treats agents as constrained optimisers with varying degrees of strategic skills, permitting departure from strict utility maximisation. Behaviour is learnt through repeated simulations with policy gradients to adjust action likelihoods. To allow efficient computation, we use parameterised shared policy learning with distributions of agent skill levels. Shared policy learning avoids the need for agents to learn individual policies yet still enables a spectrum of bounded rational behaviours. We validate our model's effectiveness using real-world data on a range of canonical $n$-agent settings, demonstrating significantly improved predictive capability.

{{</citation>}}


## quant-ph (1)



### (60/108) Hybrid Quantum Vision Transformers for Event Classification in High Energy Physics (Eyup B. Unlu et al., 2024)

{{<citation>}}

Eyup B. Unlu, Marçal Comajoan Cara, Gopal Ramesh Dahale, Zhongtian Dong, Roy T. Forestano, Sergei Gleyzer, Daniel Justice, Kyoungchul Kong, Tom Magorsch, Konstantin T. Matchev, Katia Matcheva. (2024)  
**Hybrid Quantum Vision Transformers for Event Classification in High Energy Physics**  

---
Primary Category: quant-ph  
Categories: cs-LG, hep-ph, quant-ph, quant-ph, stat-ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.00776v1)  

---


**ABSTRACT**  
Models based on vision transformer architectures are considered state-of-the-art when it comes to image classification tasks. However, they require extensive computational resources both for training and deployment. The problem is exacerbated as the amount and complexity of the data increases. Quantum-based vision transformer models could potentially alleviate this issue by reducing the training and operating time while maintaining the same predictive power. Although current quantum computers are not yet able to perform high-dimensional tasks yet, they do offer one of the most efficient solutions for the future. In this work, we construct several variations of a quantum hybrid vision transformer for a classification problem in high energy physics (distinguishing photons and electrons in the electromagnetic calorimeter). We test them against classical vision transformer architectures. Our findings indicate that the hybrid models can achieve comparable performance to their classical analogues with a similar number of parameters.

{{</citation>}}


## cs.HC (1)



### (61/108) To Search or To Gen? Exploring the Synergy between Generative AI and Web Search in Programming (Ryan Yen et al., 2024)

{{<citation>}}

Ryan Yen, Nicole Sultanum, Jian Zhao. (2024)  
**To Search or To Gen? Exploring the Synergy between Generative AI and Web Search in Programming**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2402.00764v1)  

---


**ABSTRACT**  
The convergence of generative AI and web search is reshaping problem-solving for programmers. However, the lack of understanding regarding their interplay in the information-seeking process often leads programmers to perceive them as alternatives rather than complementary tools. To analyze this interaction and explore their synergy, we conducted an interview study with eight experienced programmers. Drawing from the results and literature, we have identified three major challenges and proposed three decision-making stages, each with its own relevant factors. Additionally, we present a comprehensive process model that captures programmers' interaction patterns. This model encompasses decision-making stages, the information-foraging loop, and cognitive activities during system interaction, offering a holistic framework to comprehend and optimize the use of these convergent tools in programming.

{{</citation>}}


## cs.SD (2)



### (62/108) BATON: Aligning Text-to-Audio Model with Human Preference Feedback (Huan Liao et al., 2024)

{{<citation>}}

Huan Liao, Haonan Han, Kai Yang, Tianjiao Du, Rui Yang, Zunnan Xu, Qinmei Xu, Jingquan Liu, Jiasheng Lu, Xiu Li. (2024)  
**BATON: Aligning Text-to-Audio Model with Human Preference Feedback**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00744v1)  

---


**ABSTRACT**  
With the development of AI-Generated Content (AIGC), text-to-audio models are gaining widespread attention. However, it is challenging for these models to generate audio aligned with human preference due to the inherent information density of natural language and limited model understanding ability. To alleviate this issue, we formulate the BATON, a framework designed to enhance the alignment between generated audio and text prompt using human preference feedback. Our BATON comprises three key stages: Firstly, we curated a dataset containing both prompts and the corresponding generated audio, which was then annotated based on human feedback. Secondly, we introduced a reward model using the constructed dataset, which can mimic human preference by assigning rewards to input text-audio pairs. Finally, we employed the reward model to fine-tune an off-the-shelf text-to-audio model. The experiment results demonstrate that our BATON can significantly improve the generation quality of the original text-to-audio models, concerning audio integrity, temporal relationship, and alignment with human preference.

{{</citation>}}


### (63/108) Can you Remove the Downstream Model for Speaker Recognition with Self-Supervised Speech Features? (Zakaria Aldeneh et al., 2024)

{{<citation>}}

Zakaria Aldeneh, Takuya Higuchi, Jee-weon Jung, Skyler Seto, Tatiana Likhomanenko, Stephen Shum, Ahmed Hussen Abdelaziz, Shinji Watanabe, Barry-John Theobald. (2024)  
**Can you Remove the Downstream Model for Speaker Recognition with Self-Supervised Speech Features?**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2402.00340v1)  

---


**ABSTRACT**  
Self-supervised features are typically used in place of filter-banks in speaker verification models. However, these models were originally designed to ingest filter-banks as inputs, and thus, training them on top of self-supervised features assumes that both feature types require the same amount of learning for the task. In this work, we observe that pre-trained self-supervised speech features inherently include information required for downstream speaker verification task, and therefore, we can simplify the downstream model without sacrificing performance. To this end, we revisit the design of the downstream model for speaker verification using self-supervised features. We show that we can simplify the model to use 97.51% fewer parameters while achieving a 29.93% average improvement in performance on SUPERB. Consequently, we show that the simplified downstream model is more data efficient compared to baseline--it achieves better performance with only 60% of the training data.

{{</citation>}}


## cs.SE (6)



### (64/108) BIOMERO: BioImage analysis in OMERO (Torec T. Luik et al., 2024)

{{<citation>}}

Torec T. Luik, Rodrigo Rosas-Bertolini, Eric A. J. Reits, Ron A. Hoebe, Przemek M. Krawczyk. (2024)  
**BIOMERO: BioImage analysis in OMERO**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00734v1)  

---


**ABSTRACT**  
In the rapidly evolving field of bioimaging, the integration and orchestration of Findable, Accessible, Interoperable, and Reusable (FAIR) image analysis workflows remains a challenge. We introduce BIOMERO, a bridge connecting OMERO, a renowned bioimaging data management platform, FAIR workflows and high-performance computing (HPC) environments. BIOMERO, featuring our opensource Python library "OMERO Slurm Client", facilitates seamless execution of FAIR workflows, particularly for large datasets from High Content or High Throughput Screening. BIOMERO empowers researchers by eliminating the need for specialized knowledge, enabling scalable image processing directly from OMERO. BIOMERO notably supports the sharing and utilization of FAIR workflows between OMERO, Cytomine/BIAFLOWS, and other bioimaging communities. BIOMERO will promote the widespread adoption of FAIR workflows, emphasizing reusability, across the realm of bioimaging research. Its user-friendly interface will empower users, including those without technical expertise, to seamlessly apply these workflows to their datasets, democratizing the utilization of AI by the broader research community.

{{</citation>}}


### (65/108) PeaTMOSS: A Dataset and Initial Analysis of Pre-Trained Models in Open-Source Software (Wenxin Jiang et al., 2024)

{{<citation>}}

Wenxin Jiang, Jerin Yasmin, Jason Jones, Nicholas Synovic, Jiashen Kuo, Nathaniel Bielanski, Yuan Tian, George K. Thiruvathukal, James C. Davis. (2024)  
**PeaTMOSS: A Dataset and Initial Analysis of Pre-Trained Models in Open-Source Software**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-DB, cs-LG, cs-SE, cs.SE  
Keywords: Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2402.00699v1)  

---


**ABSTRACT**  
The development and training of deep learning models have become increasingly costly and complex. Consequently, software engineers are adopting pre-trained models (PTMs) for their downstream applications. The dynamics of the PTM supply chain remain largely unexplored, signaling a clear need for structured datasets that document not only the metadata but also the subsequent applications of these models. Without such data, the MSR community cannot comprehensively understand the impact of PTM adoption and reuse. This paper presents the PeaTMOSS dataset, which comprises metadata for 281,638 PTMs and detailed snapshots for all PTMs with over 50 monthly downloads (14,296 PTMs), along with 28,575 open-source software repositories from GitHub that utilize these models. Additionally, the dataset includes 44,337 mappings from 15,129 downstream GitHub repositories to the 2,530 PTMs they use. To enhance the dataset's comprehensiveness, we developed prompts for a large language model to automatically extract model metadata, including the model's training datasets, parameters, and evaluation metrics. Our analysis of this dataset provides the first summary statistics for the PTM supply chain, showing the trend of PTM development and common shortcomings of PTM package documentation. Our example application reveals inconsistencies in software licenses across PTMs and their dependent projects. PeaTMOSS lays the foundation for future research, offering rich opportunities to investigate the PTM supply chain. We outline mining opportunities on PTMs, their downstream usage, and cross-cutting questions.

{{</citation>}}


### (66/108) Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks (Zhongxin Liu et al., 2024)

{{<citation>}}

Zhongxin Liu, Zhijie Tang, Junwei Zhang, Xin Xia, Xiaohu Yang. (2024)  
**Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2402.00657v1)  

---


**ABSTRACT**  
Vulnerability analysis is crucial for software security. This work focuses on using pre-training techniques to enhance the understanding of vulnerable code and boost vulnerability analysis. The code understanding ability of a pre-trained model is highly related to its pre-training objectives. The semantic structure, e.g., control and data dependencies, of code is important for vulnerability analysis. However, existing pre-training objectives either ignore such structure or focus on learning to use it. The feasibility and benefits of learning the knowledge of analyzing semantic structure have not been investigated. To this end, this work proposes two novel pre-training objectives, namely Control Dependency Prediction (CDP) and Data Dependency Prediction (DDP), which aim to predict the statement-level control dependencies and token-level data dependencies, respectively, in a code snippet only based on its source code. During pre-training, CDP and DDP can guide the model to learn the knowledge required for analyzing fine-grained dependencies in code. After pre-training, the pre-trained model can boost the understanding of vulnerable code during fine-tuning and can directly be used to perform dependence analysis for both partial and complete functions. To demonstrate the benefits of our pre-training objectives, we pre-train a Transformer model named PDBERT with CDP and DDP, fine-tune it on three vulnerability analysis tasks, i.e., vulnerability detection, vulnerability classification, and vulnerability assessment, and also evaluate it on program dependence analysis. Experimental results show that PDBERT benefits from CDP and DDP, leading to state-of-the-art performance on the three downstream tasks. Also, PDBERT achieves F1-scores of over 99% and 94% for predicting control and data dependencies, respectively, in partial and complete functions.

{{</citation>}}


### (67/108) Towards Summarizing Code Snippets Using Pre-Trained Transformers (Antonio Mastropaolo et al., 2024)

{{<citation>}}

Antonio Mastropaolo, Matteo Ciniselli, Luca Pascarella, Rosalia Tufano, Emad Aghajani, Gabriele Bavota. (2024)  
**Towards Summarizing Code Snippets Using Pre-Trained Transformers**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.00519v1)  

---


**ABSTRACT**  
When comprehending code, a helping hand may come from the natural language comments documenting it that, unfortunately, are not always there. To support developers in such a scenario, several techniques have been presented to automatically generate natural language summaries for a given code. Most recent approaches exploit deep learning (DL) to automatically document classes or functions, while little effort has been devoted to more fine-grained documentation (e.g., documenting code snippets or even a single statement). Such a design choice is dictated by the availability of training data: For example, in the case of Java, it is easy to create datasets composed of pairs <Method, Javadoc> that can be fed to DL models to teach them how to summarize a method. Such a comment-to-code linking is instead non-trivial when it comes to inner comments documenting a few statements. In this work, we take all the steps needed to train a DL model to document code snippets. First, we manually built a dataset featuring 6.6k comments that have been (i) classified based on their type (e.g., code summary, TODO), and (ii) linked to the code statements they document. Second, we used such a dataset to train a multi-task DL model, taking as input a comment and being able to (i) classify whether it represents a "code summary" or not and (ii) link it to the code statements it documents. Our model identifies code summaries with 84% accuracy and is able to link them to the documented lines of code with recall and precision higher than 80%. Third, we run this model on 10k projects, identifying and linking code summaries to the documented code. This unlocked the possibility of building a large-scale dataset of documented code snippets that have then been used to train a new DL model able to document code snippets. A comparison with state-of-the-art baselines shows the superiority of the proposed approach.

{{</citation>}}


### (68/108) Large Language Models Based Fuzzing Techniques: A Survey (Linghan Huang et al., 2024)

{{<citation>}}

Linghan Huang, Peizhou Zhao, Huaming Chen, Lei Ma. (2024)  
**Large Language Models Based Fuzzing Techniques: A Survey**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00350v1)  

---


**ABSTRACT**  
In the modern era where software plays a pivotal role, software security and vulnerability analysis have become essential for software development. Fuzzing test, as an efficient software testing method, are widely used in various domains. Moreover, the rapid development of Large Language Models (LLMs) has facilitated their application in the field of software testing, demonstrating remarkable performance. Considering that existing fuzzing test techniques are not entirely automated and software vulnerabilities continue to evolve, there is a growing trend towards employing fuzzing test generated based on large language models. This survey provides a systematic overview of the approaches that fuse LLMs and fuzzing tests for software testing. In this paper, a statistical analysis and discussion of the literature in three areas, namely LLMs, fuzzing test, and fuzzing test generated based on LLMs, are conducted by summarising the state-of-the-art methods up until 2024. Our survey also investigates the potential for widespread deployment and application of fuzzing test techniques generated by LLMs in the future.

{{</citation>}}


### (69/108) Towards AI-Assisted Synthesis of Verified Dafny Methods (Md Rakib Hossain Misu et al., 2024)

{{<citation>}}

Md Rakib Hossain Misu, Cristina V. Lopes, Iris Ma, James Noble. (2024)  
**Towards AI-Assisted Synthesis of Verified Dafny Methods**  

---
Primary Category: cs.SE  
Categories: cs-PL, cs-SE, cs.SE  
Keywords: AI, GPT, GPT-4, PaLM  
[Paper Link](http://arxiv.org/abs/2402.00247v1)  

---


**ABSTRACT**  
Large stochastic language models show great promise in many domains, including programming. A promise is easy to make but hard to keep, and language models often fail to keep their promises when applied to programming, generating erroneous code. One promising avenue to keep models honest is to have them generate code in a language that supports formal verification: if and when that is adopted, the model would provide proof along with the code, and that proof would be automatically verified. Unfortunately, existing large language models show a severe lack of proficiency in verified programming languages. In this paper we demonstrate how to improve two pretrained models' proficiency in the Dafny verified programming language. Using 178 programming problems from the MBPP dataset, we prompt two contemporary models (GPT-4 and PaLM-2) to generate methods in Dafny. We use three different types of prompts: a direct contextless prompt, a second one that includes a signature of the method and test cases, and a third one that decomposes the problem into steps and includes dynamically chosen similar examples. Our results show that GPT-4 is better than PaLM-2, but that, in both models, the third prompt greatly improves the success of the generation task for the direct prompt. With the third prompt, GPT-4 was able to generate verified (and human-evaluated) Dafny methods in 58% of the cases, while the first prompt generated verified (and human-evaluated) methods in only 19% of the cases. Surprisingly, the second prompt had the worst performance, with only 10%. One tangible contribution of our work is a collection of 153 MBPP problems that are implemented and formally verified in Dafny, 50 of which were written by us and 103 were automatically synthesized by GPT-4. Additionally, our results demonstrate that the benefits of formal program verification (proof of correctness) are now within reach...

{{</citation>}}


## cs.RO (7)



### (70/108) Neural Style Transfer with Twin-Delayed DDPG for Shared Control of Robotic Manipulators (Raul Fernandez-Fernandez et al., 2024)

{{<citation>}}

Raul Fernandez-Fernandez, Marco Aggravi, Paolo Robuffo Giordano, Juan G. Victores, Claudio Pacchierotti. (2024)  
**Neural Style Transfer with Twin-Delayed DDPG for Shared Control of Robotic Manipulators**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-NE, cs-RO, cs.RO  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2402.00722v1)  

---


**ABSTRACT**  
Neural Style Transfer (NST) refers to a class of algorithms able to manipulate an element, most often images, to adopt the appearance or style of another one. Each element is defined as a combination of Content and Style: the Content can be conceptually defined as the what and the Style as the how of said element. In this context, we propose a custom NST framework for transferring a set of styles to the motion of a robotic manipulator, e.g., the same robotic task can be carried out in an angry, happy, calm, or sad way. An autoencoder architecture extracts and defines the Content and the Style of the target robot motions. A Twin Delayed Deep Deterministic Policy Gradient (TD3) network generates the robot control policy using the loss defined by the autoencoder. The proposed Neural Policy Style Transfer TD3 (NPST3) alters the robot motion by introducing the trained style. Such an approach can be implemented either offline, for carrying out autonomous robot motions in dynamic environments, or online, for adapting at runtime the style of a teleoperated robot. The considered styles can be learned online from human demonstrations. We carried out an evaluation with human subjects enrolling 73 volunteers, asking them to recognize the style behind some representative robotic motions. Results show a good recognition rate, proving that it is possible to convey different styles to a robot using this approach.

{{</citation>}}


### (71/108) WayFASTER: a Self-Supervised Traversability Prediction for Increased Navigation Awareness (Mateus Valverde Gasparino et al., 2024)

{{<citation>}}

Mateus Valverde Gasparino, Arun Narenthiran Sivakumar, Girish Chowdhary. (2024)  
**WayFASTER: a Self-Supervised Traversability Prediction for Increased Navigation Awareness**  

---
Primary Category: cs.RO  
Categories: I-2-9; I-2-6; I-2-10, cs-RO, cs.RO  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2402.00683v1)  

---


**ABSTRACT**  
Accurate and robust navigation in unstructured environments requires fusing data from multiple sensors. Such fusion ensures that the robot is better aware of its surroundings, including areas of the environment that are not immediately visible, but were visible at a different time. To solve this problem, we propose a method for traversability prediction in challenging outdoor environments using a sequence of RGB and depth images fused with pose estimations. Our method, termed WayFASTER (Waypoints-Free Autonomous System for Traversability with Enhanced Robustness), uses experience data recorded from a receding horizon estimator to train a self-supervised neural network for traversability prediction, eliminating the need for heuristics. Our experiments demonstrate that our method excels at avoiding geometric obstacles, and correctly detects that traversable terrains, such as tall grass, can be navigable. By using a sequence of images, WayFASTER significantly enhances the robot's awareness of its surroundings, enabling it to predict the traversability of terrains that are not immediately visible. This enhanced awareness contributes to better navigation performance in environments where such predictive capabilities are essential.

{{</citation>}}


### (72/108) Neural Policy Style Transfer (Raul Fernandez-Fernandez et al., 2024)

{{<citation>}}

Raul Fernandez-Fernandez, Juan G. Victores, Jennifer J. Gago, David Estevez, Carlos Balaguer. (2024)  
**Neural Policy Style Transfer**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-NE, cs-RO, cs.RO  
Keywords: Reinforcement Learning, Style Transfer  
[Paper Link](http://arxiv.org/abs/2402.00677v1)  

---


**ABSTRACT**  
Style Transfer has been proposed in a number of fields: fine arts, natural language processing, and fixed trajectories. We scale this concept up to control policies within a Deep Reinforcement Learning infrastructure. Each network is trained to maximize the expected reward, which typically encodes the goal of an action, and can be described as the content. The expressive power of deep neural networks enables encoding a secondary task, which can be described as the style. The Neural Policy Style Transfer (NPST) algorithm is proposed to transfer the style of one policy to another, while maintaining the content of the latter. Different policies are defined via Deep Q-Network architectures. These models are trained using demonstrations through Inverse Reinforcement Learning. Two different sets of user demonstrations are performed, one for content and other for style. Different styles are encoded as defined by user demonstrations. The generated policy is the result of feeding a content policy and a style policy to the NPST algorithm. Experiments are performed in a catch-ball game inspired by the Deep Reinforcement Learning classical Atari games; and a real-world painting scenario with a full-sized humanoid robot, based on previous works of the authors. The implementation of three different Q-Network architectures (Shallow, Deep and Deep Recurrent Q-Network) to encode the policies within the NPST framework is proposed and the results obtained in the experiments with each of these architectures compared.

{{</citation>}}


### (73/108) Deep Robot Sketching: An application of Deep Q-Learning Networks for human-like sketching (Raul Fernandez-Fernandez et al., 2024)

{{<citation>}}

Raul Fernandez-Fernandez, Juan G. Victores, Carlos Balaguer. (2024)  
**Deep Robot Sketching: An application of Deep Q-Learning Networks for human-like sketching**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-LG, cs-NE, cs-RO, cs.RO  
Keywords: Reinforcement Learning, Sketch  
[Paper Link](http://arxiv.org/abs/2402.00676v1)  

---


**ABSTRACT**  
The current success of Reinforcement Learning algorithms for its performance in complex environments has inspired many recent theoretical approaches to cognitive science. Artistic environments are studied within the cognitive science community as rich, natural, multi-sensory, multi-cultural environments. In this work, we propose the introduction of Reinforcement Learning for improving the control of artistic robot applications. Deep Q-learning Neural Networks (DQN) is one of the most successful algorithms for the implementation of Reinforcement Learning in robotics. DQN methods generate complex control policies for the execution of complex robot applications in a wide set of environments. Current art painting robot applications use simple control laws that limits the adaptability of the frameworks to a set of simple environments. In this work, the introduction of DQN within an art painting robot application is proposed. The goal is to study how the introduction of a complex control policy impacts the performance of a basic art painting robot application. The main expected contribution of this work is to serve as a first baseline for future works introducing DQN methods for complex art painting robot frameworks. Experiments consist of real world executions of human drawn sketches using the DQN generated policy and TEO, the humanoid robot. Results are compared in terms of similarity and obtained reward with respect to the reference inputs

{{</citation>}}


### (74/108) Transferring human emotions to robot motions using Neural Policy Style Transfer (Raul Fernandez-Fernandez et al., 2024)

{{<citation>}}

Raul Fernandez-Fernandez, Bartek Łukawski, Juan G. Victores, Claudio Pacchierotti. (2024)  
**Transferring human emotions to robot motions using Neural Policy Style Transfer**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2402.00663v1)  

---


**ABSTRACT**  
Neural Style Transfer (NST) was originally proposed to use feature extraction capabilities of Neural Networks as a way to perform Style Transfer with images. Pre-trained image classification architectures were selected for feature extraction, leading to new images showing the same content as the original but with a different style. In robotics, Style Transfer can be employed to transfer human motion styles to robot motions. The challenge lies in the lack of pre-trained classification architectures for robot motions that could be used for feature extraction. Neural Policy Style Transfer TD3 (NPST3) is proposed for the transfer of human motion styles to robot motions. This framework allows the same robot motion to be executed in different human-centered motion styles, such as in an angry, happy, calm, or sad fashion. The Twin Delayed Deep Deterministic Policy Gradient (TD3) network is introduced for the generation of control policies. An autoencoder network is in charge of feature extraction for the Style Transfer step. The Style Transfer step can be performed both offline and online: offline for the autonomous executions of human-style robot motions, and online for adapting at runtime the style of e.g., a teleoperated robot. The framework is tested using two different robotic platforms: a robotic manipulator designed for telemanipulation tasks, and a humanoid robot designed for social interaction. The proposed approach was evaluated for both platforms, performing a total of 147 questionnaires asking human subjects to recognize the human motion style transferred to the robot motion for a predefined set of actions.

{{</citation>}}


### (75/108) Robust Path Planning via Learning from Demonstrations for Robotic Catheters in Deformable Environments (Zhen Li et al., 2024)

{{<citation>}}

Zhen Li, Chiara Lambranzi, Di Wu, Alice Segato, Federico De Marco, Emmanuel Vander Poorten, Jenny Dankelman, Elena De Momi. (2024)  
**Robust Path Planning via Learning from Demonstrations for Robotic Catheters in Deformable Environments**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00537v1)  

---


**ABSTRACT**  
Navigation through tortuous and deformable vessels using catheters with limited steering capability underscores the need for reliable path planning. State-of-the-art path planners do not fully account for the deformable nature of the environment. This work proposes a robust path planner via a learning from demonstrations method, named Curriculum Generative Adversarial Imitation Learning (C-GAIL). This path planning framework takes into account the interaction between steerable catheters and vessel walls and the deformable property of vessels. In-silico comparative experiments show that the proposed network achieves smaller targeting errors, and a higher success rate, compared to a state-of-the-art approach based on GAIL. The in-vitro validation experiments demonstrate that the path generated by the proposed C-GAIL path planner aligns better with the actual steering capability of the pneumatic artificial muscle-driven catheter utilized in this study. Therefore, the proposed approach can provide enhanced support to the user in navigating the catheter towards the target with greater precision, in contrast to the conventional centerline-following technique. The targeting and tracking errors are 1.26$\pm$0.55mm and 5.18$\pm$3.48mm, respectively. The proposed path planning framework exhibits superior performance in managing uncertainty associated with vessel deformation, thereby resulting in lower tracking errors.

{{</citation>}}


### (76/108) Towards scalable robotic intervention of children with Autism Spectrum Disorder using LLMs (Ruchik Mishra et al., 2024)

{{<citation>}}

Ruchik Mishra, Karla Conn Welch. (2024)  
**Towards scalable robotic intervention of children with Autism Spectrum Disorder using LLMs**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: BERT, GPT, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2402.00260v1)  

---


**ABSTRACT**  
In this paper, we propose a social robot capable of verbally interacting with children with Autism Spectrum Disorder (ASD). This communication is meant to teach perspective-taking using text generated using a Large Language Model (LLM) pipeline. The social robot NAO acts as a stimulator (verbally describes a social situation and asks a question), prompter (presents three options to choose from), and reinforcer (praises when the answer is correct). For the role of the stimulator, the social situation, questions, and options are generated using our LLM pipeline. We compare two approaches: GPT-2 + BART and GPT-2 + GPT-2, where the first GPT-2 common between the pipelines is used for unsupervised social situation generation. We use the SOCIALIQA dataset to fine-tune all of our LLM pipelines. We found that the GPT-2 + BART pipeline had a better BERTscore for generating the questions and the options by combining their individual loss functions. This observation was also consistent with the human evaluations. Lastly, the unsupervised generation of social situations was visualized using T-SNE plots, and the entire pipeline was evaluated for appropriateness for children with ASD by human experts.

{{</citation>}}


## cs.AI (3)



### (77/108) Intent Assurance using LLMs guided by Intent Drift (Kristina Dzeparoska et al., 2024)

{{<citation>}}

Kristina Dzeparoska, Ali Tizghadam, Alberto Leon-Garcia. (2024)  
**Intent Assurance using LLMs guided by Intent Drift**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-NI, cs.AI, stat-ME  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2402.00715v1)  

---


**ABSTRACT**  
Intent-Based Networking (IBN) presents a paradigm shift for network management, by promising to align intents and business objectives with network operations--in an automated manner. However, its practical realization is challenging: 1) processing intents, i.e., translate, decompose and identify the logic to fulfill the intent, and 2) intent conformance, that is, considering dynamic networks, the logic should be adequately adapted to assure intents. To address the latter, intent assurance is tasked with continuous verification and validation, including taking the necessary actions to align the operational and target states. In this paper, we define an assurance framework that allows us to detect and act when intent drift occurs. To do so, we leverage AI-driven policies, generated by Large Language Models (LLMs) which can quickly learn the necessary in-context requirements, and assist with the fulfillment and assurance of intents.

{{</citation>}}


### (78/108) Learning Planning-based Reasoning by Trajectories Collection and Process Reward Synthesizing (Fangkai Jiao et al., 2024)

{{<citation>}}

Fangkai Jiao, Chengwei Qin, Zhengyuan Liu, Nancy F. Chen, Shafiq Joty. (2024)  
**Learning Planning-based Reasoning by Trajectories Collection and Process Reward Synthesizing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: GPT, GPT-3.5, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2402.00658v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated significant potential in handling complex reasoning tasks through step-by-step rationale generation. However, recent studies have raised concerns regarding the hallucination and flaws in their reasoning process. Substantial efforts are being made to improve the reliability and faithfulness of the generated rationales. Some approaches model reasoning as planning, while others focus on annotating for process supervision. Nevertheless, the planning-based search process often results in high latency due to the frequent assessment of intermediate reasoning states and the extensive exploration space. Additionally, supervising the reasoning process with human annotation is costly and challenging to scale for LLM training. To address these issues, in this paper, we propose a framework to learn planning-based reasoning through direct preference optimization (DPO) on collected trajectories, which are ranked according to synthesized process rewards. Our results on challenging logical reasoning benchmarks demonstrate the effectiveness of our learning framework, showing that our 7B model can surpass the strong counterparts like GPT-3.5-Turbo.

{{</citation>}}


### (79/108) Computational Experiments Meet Large Language Model Based Agents: A Survey and Perspective (Qun Ma et al., 2024)

{{<citation>}}

Qun Ma, Xiao Xue, Deyu Zhou, Xiangning Yu, Donghua Liu, Xuwen Zhang, Zihan Zhao, Yifan Shen, Peilin Ji, Juanjuan Li, Gang Wang, Wanpeng Ma. (2024)  
**Computational Experiments Meet Large Language Model Based Agents: A Survey and Perspective**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00262v1)  

---


**ABSTRACT**  
Computational experiments have emerged as a valuable method for studying complex systems, involving the algorithmization of counterfactuals. However, accurately representing real social systems in Agent-based Modeling (ABM) is challenging due to the diverse and intricate characteristics of humans, including bounded rationality and heterogeneity. To address this limitation, the integration of Large Language Models (LLMs) has been proposed, enabling agents to possess anthropomorphic abilities such as complex reasoning and autonomous learning. These agents, known as LLM-based Agent, offer the potential to enhance the anthropomorphism lacking in ABM. Nonetheless, the absence of explicit explainability in LLMs significantly hinders their application in the social sciences. Conversely, computational experiments excel in providing causal analysis of individual behaviors and complex phenomena. Thus, combining computational experiments with LLM-based Agent holds substantial research potential. This paper aims to present a comprehensive exploration of this fusion. Primarily, it outlines the historical development of agent structures and their evolution into artificial societies, emphasizing their importance in computational experiments. Then it elucidates the advantages that computational experiments and LLM-based Agents offer each other, considering the perspectives of LLM-based Agent for computational experiments and vice versa. Finally, this paper addresses the challenges and future trends in this research domain, offering guidance for subsequent related studies.

{{</citation>}}


## cs.CV (16)



### (80/108) A Framework for Building Point Cloud Cleaning, Plane Detection and Semantic Segmentation (Ilyass Abouelaziz et al., 2024)

{{<citation>}}

Ilyass Abouelaziz, Youssef Mourchid. (2024)  
**A Framework for Building Point Cloud Cleaning, Plane Detection and Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2402.00692v1)  

---


**ABSTRACT**  
This paper presents a framework to address the challenges involved in building point cloud cleaning, plane detection, and semantic segmentation, with the ultimate goal of enhancing building modeling. We focus in the cleaning stage on removing outliers from the acquired point cloud data by employing an adaptive threshold technique based on z-score measure. Following the cleaning process, we perform plane detection using the robust RANSAC paradigm. The goal is to carry out multiple plane segmentations, and to classify segments into distinct categories, such as floors, ceilings, and walls. The resulting segments can generate accurate and detailed point clouds representing the building's architectural elements. Moreover, we address the problem of semantic segmentation, which plays a vital role in the identification and classification of different components within the building, such as walls, windows, doors, roofs, and objects. Inspired by the PointNet architecture, we propose a deep learning architecture for efficient semantic segmentation in buildings. The results demonstrate the effectiveness of the proposed framework in handling building modeling tasks, paving the way for improved accuracy and efficiency in the field of building modelization.

{{</citation>}}


### (81/108) Exploring Homogeneous and Heterogeneous Consistent Label Associations for Unsupervised Visible-Infrared Person ReID (Lingfeng He et al., 2024)

{{<citation>}}

Lingfeng He, De Cheng, Nannan Wang, Xinbo Gao. (2024)  
**Exploring Homogeneous and Heterogeneous Consistent Label Associations for Unsupervised Visible-Infrared Person ReID**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2402.00672v1)  

---


**ABSTRACT**  
Unsupervised visible-infrared person re-identification (USL-VI-ReID) aims to retrieve pedestrian images of the same identity from different modalities without annotations. While prior work focuses on establishing cross-modality pseudo-label associations to bridge the modality-gap, they ignore maintaining the instance-level homogeneous and heterogeneous consistency in pseudo-label space, resulting in coarse associations. In response, we introduce a Modality-Unified Label Transfer (MULT) module that simultaneously accounts for both homogeneous and heterogeneous fine-grained instance-level structures, yielding high-quality cross-modality label associations. It models both homogeneous and heterogeneous affinities, leveraging them to define the inconsistency for the pseudo-labels and then minimize it, leading to pseudo-labels that maintain alignment across modalities and consistency within intra-modality structures. Additionally, a straightforward plug-and-play Online Cross-memory Label Refinement (OCLR) module is proposed to further mitigate the impact of noisy pseudo-labels while simultaneously aligning different modalities, coupled with a Modality-Invariant Representation Learning (MIRL) framework. Experiments demonstrate that our proposed method outperforms existing USL-VI-ReID methods, highlighting the superiority of our MULT in comparison to other cross-modality association methods. The code will be available.

{{</citation>}}


### (82/108) Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks (Maan Qraitem et al., 2024)

{{<citation>}}

Maan Qraitem, Nazia Tasnim, Kate Saenko, Bryan A. Plummer. (2024)  
**Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2402.00626v1)  

---


**ABSTRACT**  
Recently, significant progress has been made on Large Vision-Language Models (LVLMs); a new class of VL models that make use of large pre-trained language models. Yet, their vulnerability to Typographic attacks, which involve superimposing misleading text onto an image remain unstudied. Furthermore, prior work typographic attacks rely on sampling a random misleading class from a predefined set of classes. However, the random chosen class might not be the most effective attack. To address these issues, we first introduce a novel benchmark uniquely designed to test LVLMs vulnerability to typographic attacks. Furthermore, we introduce a new and more effective typographic attack: Self-Generated typographic attacks. Indeed, our method, given an image, make use of the strong language capabilities of models like GPT-4V by simply prompting them to recommend a typographic attack. Using our novel benchmark, we uncover that typographic attacks represent a significant threat against LVLM(s). Furthermore, we uncover that typographic attacks recommended by GPT-4V using our new method are not only more effective against GPT-4V itself compared to prior work attacks, but also against a host of less capable yet popular open source models like LLaVA, InstructBLIP, and MiniGPT4.

{{</citation>}}


### (83/108) Dynamic Texture Transfer using PatchMatch and Transformers (Guo Pu et al., 2024)

{{<citation>}}

Guo Pu, Shiyao Xu, Xixin Cao, Zhouhui Lian. (2024)  
**Dynamic Texture Transfer using PatchMatch and Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.00606v1)  

---


**ABSTRACT**  
How to automatically transfer the dynamic texture of a given video to the target still image is a challenging and ongoing problem. In this paper, we propose to handle this task via a simple yet effective model that utilizes both PatchMatch and Transformers. The key idea is to decompose the task of dynamic texture transfer into two stages, where the start frame of the target video with the desired dynamic texture is synthesized in the first stage via a distance map guided texture transfer module based on the PatchMatch algorithm. Then, in the second stage, the synthesized image is decomposed into structure-agnostic patches, according to which their corresponding subsequent patches can be predicted by exploiting the powerful capability of Transformers equipped with VQ-VAE for processing long discrete sequences. After getting all those patches, we apply a Gaussian weighted average merging strategy to smoothly assemble them into each frame of the target stylized video. Experimental results demonstrate the effectiveness and superiority of the proposed method in dynamic texture transfer compared to the state of the art.

{{</citation>}}


### (84/108) A Single Graph Convolution Is All You Need: Efficient Grayscale Image Classification (Jacob Fein-Ashley et al., 2024)

{{<citation>}}

Jacob Fein-Ashley, Tian Ye, Sachini Wickramasinghe, Bingyi Zhang, Rajgopal Kannan, Viktor Prasanna. (2024)  
**A Single Graph Convolution Is All You Need: Efficient Grayscale Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2402.00564v1)  

---


**ABSTRACT**  
Image classifiers often rely on convolutional neural networks (CNN) for their tasks, which are inherently more heavyweight than multilayer perceptrons (MLPs), which can be problematic in real-time applications. Additionally, many image classification models work on both RGB and grayscale datasets. Classifiers that operate solely on grayscale images are much less common. Grayscale image classification has diverse applications, including but not limited to medical image classification and synthetic aperture radar (SAR) automatic target recognition (ATR). Thus, we present a novel grayscale (single channel) image classification approach using a vectorized view of images. We exploit the lightweightness of MLPs by viewing images as a vector and reducing our problem setting to the grayscale image classification setting. We find that using a single graph convolutional layer batch-wise increases accuracy and reduces variance in the performance of our model. Moreover, we develop a customized accelerator on FPGA for the proposed model with several optimizations to improve its performance. Our experimental results on benchmark grayscale image datasets demonstrate the effectiveness of the proposed model, achieving vastly lower latency (up to 16$\times$ less) and competitive or leading performance compared to other state-of-the-art image classification models on various domain-specific grayscale image classification datasets.

{{</citation>}}


### (85/108) A Manifold Representation of the Key in Vision Transformers (Li Meng et al., 2024)

{{<citation>}}

Li Meng, Morten Goodwin, Anis Yazidi, Paal Engelstad. (2024)  
**A Manifold Representation of the Key in Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.00534v1)  

---


**ABSTRACT**  
Vision Transformers implement multi-head self-attention (MSA) via stacking multiple attention blocks. The query, key, and value are often intertwined and generated within those blocks via a single, shared linear transformation. This paper explores the concept of disentangling the key from the query and value, and adopting a manifold representation for the key. Our experiments reveal that decoupling and endowing the key with a manifold structure can enhance the model performance. Specifically, ViT-B exhibits a 0.87% increase in top-1 accuracy, while Swin-T sees a boost of 0.52% in top-1 accuracy on the ImageNet-1K dataset, with eight charts in the manifold key. Our approach also yields positive results in object detection and instance segmentation tasks on the COCO dataset. Through detailed ablation studies, we establish that these performance gains are not merely due to the simplicity of adding more parameters and computations. Future research may investigate strategies for cutting the budget of such representations and aim for further performance improvements based on our findings.

{{</citation>}}


### (86/108) Bias Mitigating Few-Shot Class-Incremental Learning (Li-Jun Zhao et al., 2024)

{{<citation>}}

Li-Jun Zhao, Zhen-Duo Chen, Zi-Chao Zhang, Xin Luo, Xin-Shun Xu. (2024)  
**Bias Mitigating Few-Shot Class-Incremental Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, Few-Shot  
[Paper Link](http://arxiv.org/abs/2402.00481v1)  

---


**ABSTRACT**  
Few-shot class-incremental learning (FSCIL) aims at recognizing novel classes continually with limited novel class samples. A mainstream baseline for FSCIL is first to train the whole model in the base session, then freeze the feature extractor in the incremental sessions. Despite achieving high overall accuracy, most methods exhibit notably low accuracy for incremental classes. Some recent methods somewhat alleviate the accuracy imbalance between base and incremental classes by fine-tuning the feature extractor in the incremental sessions, but they further cause the accuracy imbalance between past and current incremental classes. In this paper, we study the causes of such classification accuracy imbalance for FSCIL, and abstract them into a unified model bias problem. Based on the analyses, we propose a novel method to mitigate model bias of the FSCIL problem during training and inference processes, which includes mapping ability stimulation, separately dual-feature classification, and self-optimizing classifiers. Extensive experiments on three widely-used FSCIL benchmark datasets show that our method significantly mitigates the model bias problem and achieves state-of-the-art performance.

{{</citation>}}


### (87/108) Instruction Makes a Difference (Tosin Adewumi et al., 2024)

{{<citation>}}

Tosin Adewumi, Nudrat Habib, Lama Alkhaled, Elisa Barney. (2024)  
**Instruction Makes a Difference**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2402.00453v1)  

---


**ABSTRACT**  
We introduce Instruction Document Visual Question Answering (iDocVQA) dataset and Large Language Document (LLaDoc) model, for training Language-Vision (LV) models for document analysis and predictions on document images, respectively. Usually, deep neural networks for the DocVQA task are trained on datasets lacking instructions. We show that using instruction-following datasets improves performance. We compare performance across document-related datasets using the recent state-of-the-art (SotA) Large Language and Vision Assistant (LLaVA)1.5 as the base model. We also evaluate the performance of the derived models for object hallucination using the Polling-based Object Probing Evaluation (POPE) dataset. The results show that instruction-tuning performance ranges from 11X to 32X of zero-shot performance and from 0.1% to 4.2% over non-instruction (traditional task) finetuning. Despite the gains, these still fall short of human performance (94.36%), implying there's much room for improvement.

{{</citation>}}


### (88/108) Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection (Liyi Yao et al., 2024)

{{<citation>}}

Liyi Yao, Shaobing Gao. (2024)  
**Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2402.00448v1)  

---


**ABSTRACT**  
Due to the data imbalance and the diversity of defects, student-teacher networks (S-T) are favored in unsupervised anomaly detection, which explores the discrepancy in feature representation derived from the knowledge distillation process to recognize anomalies. However, vanilla S-T network is not stable. Employing identical structures to construct the S-T network may weaken the representative discrepancy on anomalies. But using different structures can increase the likelihood of divergent performance on normal data. To address this problem, we propose a novel dual-student knowledge distillation (DSKD) architecture. Different from other S-T networks, we use two student networks a single pre-trained teacher network, where the students have the same scale but inverted structures. This framework can enhance the distillation effect to improve the consistency in recognition of normal data, and simultaneously introduce diversity for anomaly representation. To explore high-dimensional semantic information to capture anomaly clues, we employ two strategies. First, a pyramid matching mode is used to perform knowledge distillation on multi-scale feature maps in the intermediate layers of networks. Second, an interaction is facilitated between the two student networks through a deep feature embedding module, which is inspired by real-world group discussions. In terms of classification, we obtain pixel-wise anomaly segmentation maps by measuring the discrepancy between the output feature maps of the teacher and student networks, from which an anomaly score is computed for sample-wise determination. We evaluate DSKD on three benchmark datasets and probe the effects of internal modules through ablation experiments. The results demonstrate that DSKD can achieve exceptional performance on small models like ResNet18 and effectively improve vanilla S-T networks.

{{</citation>}}


### (89/108) Lightweight Pixel Difference Networks for Efficient Visual Representation Learning (Zhuo Su et al., 2024)

{{<citation>}}

Zhuo Su, Jiehua Zhang, Longguang Wang, Hua Zhang, Zhen Liu, Matti Pietikäinen, Li Liu. (2024)  
**Lightweight Pixel Difference Networks for Efficient Visual Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Representation Learning  
[Paper Link](http://arxiv.org/abs/2402.00422v1)  

---


**ABSTRACT**  
Recently, there have been tremendous efforts in developing lightweight Deep Neural Networks (DNNs) with satisfactory accuracy, which can enable the ubiquitous deployment of DNNs in edge devices. The core challenge of developing compact and efficient DNNs lies in how to balance the competing goals of achieving high accuracy and high efficiency. In this paper we propose two novel types of convolutions, dubbed \emph{Pixel Difference Convolution (PDC) and Binary PDC (Bi-PDC)} which enjoy the following benefits: capturing higher-order local differential information, computationally efficient, and able to be integrated with existing DNNs. With PDC and Bi-PDC, we further present two lightweight deep networks named \emph{Pixel Difference Networks (PiDiNet)} and \emph{Binary PiDiNet (Bi-PiDiNet)} respectively to learn highly efficient yet more accurate representations for visual tasks including edge detection and object recognition. Extensive experiments on popular datasets (BSDS500, ImageNet, LFW, YTF, \emph{etc.}) show that PiDiNet and Bi-PiDiNet achieve the best accuracy-efficiency trade-off. For edge detection, PiDiNet is the first network that can be trained without ImageNet, and can achieve the human-level performance on BSDS500 at 100 FPS and with $<$1M parameters. For object recognition, among existing Binary DNNs, Bi-PiDiNet achieves the best accuracy and a nearly $2\times$ reduction of computational cost on ResNet18. Code available at \href{https://github.com/hellozhuo/pidinet}{https://github.com/hellozhuo/pidinet}.

{{</citation>}}


### (90/108) Short: Benchmarking transferable adversarial attacks (Zhibo Jin et al., 2024)

{{<citation>}}

Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Huaming Chen. (2024)  
**Short: Benchmarking transferable adversarial attacks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semantic Similarity  
[Paper Link](http://arxiv.org/abs/2402.00418v1)  

---


**ABSTRACT**  
The robustness of deep learning models against adversarial attacks remains a pivotal concern. This study presents, for the first time, an exhaustive review of the transferability aspect of adversarial attacks. It systematically categorizes and critically evaluates various methodologies developed to augment the transferability of adversarial attacks. This study encompasses a spectrum of techniques, including Generative Structure, Semantic Similarity, Gradient Editing, Target Modification, and Ensemble Approach. Concurrently, this paper introduces a benchmark framework \textit{TAA-Bench}, integrating ten leading methodologies for adversarial attack transferability, thereby providing a standardized and systematic platform for comparative analysis across diverse model architectures. Through comprehensive scrutiny, we delineate the efficacy and constraints of each method, shedding light on their underlying operational principles and practical utility. This review endeavors to be a quintessential resource for both scholars and practitioners in the field, charting the complex terrain of adversarial transferability and setting a foundation for future explorations in this vital sector. The associated codebase is accessible at: https://github.com/KxPlaug/TAA-Bench

{{</citation>}}


### (91/108) Safety of Multimodal Large Language Models on Images and Text (Xin Liu et al., 2024)

{{<citation>}}

Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao. (2024)  
**Safety of Multimodal Large Language Models on Images and Text**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00357v1)  

---


**ABSTRACT**  
Attracted by the impressive power of Multimodal Large Language Models (MLLMs), the public is increasingly utilizing them to improve the efficiency of daily work. Nonetheless, the vulnerabilities of MLLMs to unsafe instructions bring huge safety risks when these models are deployed in real-world scenarios. In this paper, we systematically survey current efforts on the evaluation, attack, and defense of MLLMs' safety on images and text. We begin with introducing the overview of MLLMs on images and text and understanding of safety, which helps researchers know the detailed scope of our survey. Then, we review the evaluation datasets and metrics for measuring the safety of MLLMs. Next, we comprehensively present attack and defense techniques related to MLLMs' safety. Finally, we analyze several unsolved issues and discuss promising research directions.

{{</citation>}}


### (92/108) High-Quality Medical Image Generation from Free-hand Sketch (Quan Huu Cap et al., 2024)

{{<citation>}}

Quan Huu Cap, Atsushi Fukuda. (2024)  
**High-Quality Medical Image Generation from Free-hand Sketch**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2402.00353v1)  

---


**ABSTRACT**  
Generating medical images from human-drawn free-hand sketches holds promise for various important medical imaging applications. Due to the extreme difficulty in collecting free-hand sketch data in the medical domain, most deep learning-based methods have been proposed to generate medical images from the synthesized sketches (e.g., edge maps or contours of segmentation masks from real images). However, these models often fail to generalize on the free-hand sketches, leading to unsatisfactory results. In this paper, we propose a practical free-hand sketch-to-image generation model called Sketch2MedI that learns to represent sketches in StyleGAN's latent space and generate medical images from it. Thanks to the ability to encode sketches into this meaningful representation space, Sketch2MedI only requires synthesized sketches for training, enabling a cost-effective learning process. Our Sketch2MedI demonstrates a robust generalization to free-hand sketches, resulting in high-quality and realistic medical image generations. Comparative evaluations of Sketch2MedI against the pix2pix, CycleGAN, UNIT, and U-GAT-IT models show superior performance in generating pharyngeal images, both quantitative and qualitative across various metrics.

{{</citation>}}


### (93/108) SCO-VIST: Social Interaction Commonsense Knowledge-based Visual Storytelling (Eileen Wang et al., 2024)

{{<citation>}}

Eileen Wang, Soyeon Caren Han, Josiah Poon. (2024)  
**SCO-VIST: Social Interaction Commonsense Knowledge-based Visual Storytelling**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Commonsense Knowledge  
[Paper Link](http://arxiv.org/abs/2402.00319v1)  

---


**ABSTRACT**  
Visual storytelling aims to automatically generate a coherent story based on a given image sequence. Unlike tasks like image captioning, visual stories should contain factual descriptions, worldviews, and human social commonsense to put disjointed elements together to form a coherent and engaging human-writeable story. However, most models mainly focus on applying factual information and using taxonomic/lexical external knowledge when attempting to create stories. This paper introduces SCO-VIST, a framework representing the image sequence as a graph with objects and relations that includes human action motivation and its social interaction commonsense knowledge. SCO-VIST then takes this graph representing plot points and creates bridges between plot points with semantic and occurrence-based edge weights. This weighted story graph produces the storyline in a sequence of events using Floyd-Warshall's algorithm. Our proposed framework produces stories superior across multiple metrics in terms of visual grounding, coherence, diversity, and humanness, per both automatic and human evaluations.

{{</citation>}}


### (94/108) A Survey on Hallucination in Large Vision-Language Models (Hanchao Liu et al., 2024)

{{<citation>}}

Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, Wei Peng. (2024)  
**A Survey on Hallucination in Large Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2402.00253v1)  

---


**ABSTRACT**  
Recent development of Large Vision-Language Models (LVLMs) has attracted growing attention within the AI landscape for its practical implementation potential. However, ``hallucination'', or more specifically, the misalignment between factual visual content and corresponding textual generation, poses a significant challenge of utilizing LVLMs. In this comprehensive survey, we dissect LVLM-related hallucinations in an attempt to establish an overview and facilitate future mitigation. Our scrutiny starts with a clarification of the concept of hallucinations in LVLMs, presenting a variety of hallucination symptoms and highlighting the unique challenges inherent in LVLM hallucinations. Subsequently, we outline the benchmarks and methodologies tailored specifically for evaluating hallucinations unique to LVLMs. Additionally, we delve into an investigation of the root causes of these hallucinations, encompassing insights from the training data and model components. We also critically review existing methods for mitigating hallucinations. The open questions and future directions pertaining to hallucinations within LVLMs are discussed to conclude this survey.

{{</citation>}}


### (95/108) LRDif: Diffusion Models for Under-Display Camera Emotion Recognition (Zhifeng Wang et al., 2024)

{{<citation>}}

Zhifeng Wang, Kaihao Zhang, Ramesh Sankaranarayana. (2024)  
**LRDif: Diffusion Models for Under-Display Camera Emotion Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2402.00250v1)  

---


**ABSTRACT**  
This study introduces LRDif, a novel diffusion-based framework designed specifically for facial expression recognition (FER) within the context of under-display cameras (UDC). To address the inherent challenges posed by UDC's image degradation, such as reduced sharpness and increased noise, LRDif employs a two-stage training strategy that integrates a condensed preliminary extraction network (FPEN) and an agile transformer network (UDCformer) to effectively identify emotion labels from UDC images. By harnessing the robust distribution mapping capabilities of Diffusion Models (DMs) and the spatial dependency modeling strength of transformers, LRDif effectively overcomes the obstacles of noise and distortion inherent in UDC environments. Comprehensive experiments on standard FER datasets including RAF-DB, KDEF, and FERPlus, LRDif demonstrate state-of-the-art performance, underscoring its potential in advancing FER applications. This work not only addresses a significant gap in the literature by tackling the UDC challenge in FER but also sets a new benchmark for future research in the field.

{{</citation>}}


## cs.DC (1)



### (96/108) Comparative Study of Large Language Model Architectures on Frontier (Junqi Yin et al., 2024)

{{<citation>}}

Junqi Yin, Avishek Bose, Guojing Cong, Isaac Lyngaas, Quentin Anthony. (2024)  
**Comparative Study of Large Language Model Architectures on Frontier**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI, GPT, LLaMA, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2402.00691v1)  

---


**ABSTRACT**  
Large language models (LLMs) have garnered significant attention in both the AI community and beyond. Among these, the Generative Pre-trained Transformer (GPT) has emerged as the dominant architecture, spawning numerous variants. However, these variants have undergone pre-training under diverse conditions, including variations in input data, data preprocessing, and training methodologies, resulting in a lack of controlled comparative studies. Here we meticulously examine two prominent open-sourced GPT architectures, GPT-NeoX and LLaMA, leveraging the computational power of Frontier, the world's first Exascale supercomputer. Employing the same materials science text corpus and a comprehensive end-to-end pipeline, we conduct a comparative analysis of their training and downstream performance. Our efforts culminate in achieving state-of-the-art performance on a challenging materials science benchmark. Furthermore, we investigate the computation and energy efficiency, and propose a computationally efficient method for architecture design. To our knowledge, these pre-trained models represent the largest available for materials science. Our findings provide practical guidance for building LLMs on HPC platforms.

{{</citation>}}


## cs.LO (1)



### (97/108) Bialgebraic Reasoning on Higher-Order Program Equivalence (Sergey Goncharov et al., 2024)

{{<citation>}}

Sergey Goncharov, Stefan Milius, Stelios Tsampas, Henning Urbat. (2024)  
**Bialgebraic Reasoning on Higher-Order Program Equivalence**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs-PL, cs.LO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2402.00625v1)  

---


**ABSTRACT**  
Logical relations constitute a key method for reasoning about contextual equivalence of programs in higher-order languages. They are usually developed on a per-case basis, with a new theory required for each variation of the language or of the desired notion of equivalence. In the present paper we introduce a general construction of (step-indexed) logical relations at the level of Higher-Order Mathematical Operational Semantics, a highly parametric categorical framework for modeling the operational semantics of higher-order languages. Our main result asserts that for languages whose weak operational model forms a lax bialgebra, the logical relation is automatically sound for contextual equivalence. Our abstract theory is shown to instantiate to combinatory logics and $\lambda$-calculi with recursive types, and to different flavours of contextual equivalence.

{{</citation>}}


## q-fin.PM (1)



### (98/108) Developing A Multi-Agent and Self-Adaptive Framework with Deep Reinforcement Learning for Dynamic Portfolio Risk Management (Zhenglong Li et al., 2024)

{{<citation>}}

Zhenglong Li, Vincent Tam, Kwan L. Yeung. (2024)  
**Developing A Multi-Agent and Self-Adaptive Framework with Deep Reinforcement Learning for Dynamic Portfolio Risk Management**  

---
Primary Category: q-fin.PM  
Categories: cs-LG, q-fin-PM, q-fin.PM  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.00515v1)  

---


**ABSTRACT**  
Deep or reinforcement learning (RL) approaches have been adapted as reactive agents to quickly learn and respond with new investment strategies for portfolio management under the highly turbulent financial market environments in recent years. In many cases, due to the very complex correlations among various financial sectors, and the fluctuating trends in different financial markets, a deep or reinforcement learning based agent can be biased in maximising the total returns of the newly formulated investment portfolio while neglecting its potential risks under the turmoil of various market conditions in the global or regional sectors. Accordingly, a multi-agent and self-adaptive framework namely the MASA is proposed in which a sophisticated multi-agent reinforcement learning (RL) approach is adopted through two cooperating and reactive agents to carefully and dynamically balance the trade-off between the overall portfolio returns and their potential risks. Besides, a very flexible and proactive agent as the market observer is integrated into the MASA framework to provide some additional information on the estimated market trends as valuable feedbacks for multi-agent RL approach to quickly adapt to the ever-changing market conditions. The obtained empirical results clearly reveal the potential strengths of our proposed MASA framework based on the multi-agent RL approach against many well-known RL-based approaches on the challenging data sets of the CSI 300, Dow Jones Industrial Average and S&P 500 indexes over the past 10 years. More importantly, our proposed MASA framework shed lights on many possible directions for future investigation.

{{</citation>}}


## cs.AR (2)



### (99/108) Optimization of a Line Detection Algorithm for Autonomous Vehicles on a RISC-V with Accelerator (María José Belda et al., 2024)

{{<citation>}}

María José Belda, Katzalin Olcoz, Fernando Castro, Francisco Tirado. (2024)  
**Optimization of a Line Detection Algorithm for Autonomous Vehicles on a RISC-V with Accelerator**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2402.00496v1)  

---


**ABSTRACT**  
In recent years, autonomous vehicles have attracted the attention of many research groups, both in academia and business, including researchers from leading companies such as Google, Uber and Tesla. This type of vehicles are equipped with systems that are subject to very strict requirements, essentially aimed at performing safe operations -- both for potential passengers and pedestrians -- as well as carrying out the processing needed for decision making in real time. In many instances, general-purpose processors alone cannot ensure that these safety, reliability and real-time requirements are met, so it is common to implement heterogeneous systems by including accelerators. This paper explores the acceleration of a line detection application in the autonomous car environment using a heterogeneous system consisting of a general-purpose RISC-V core and a domain-specific accelerator. In particular, the application is analyzed to identify the most computationally intensive parts of the code and it is adapted accordingly for more efficient processing. Furthermore, the code is executed on the aforementioned hardware platform to verify that the execution effectively meets the existing requirements in autonomous vehicles, experiencing a 3.7x speedup with respect to running without accelerator.

{{</citation>}}


### (100/108) AssertLLM: Generating and Evaluating Hardware Verification Assertions from Design Specifications via Multi-LLMs (Wenji Fang et al., 2024)

{{<citation>}}

Wenji Fang, Mengming Li, Min Li, Zhiyuan Yan, Shang Liu, Hongce Zhang, Zhiyao Xie. (2024)  
**AssertLLM: Generating and Evaluating Hardware Verification Assertions from Design Specifications via Multi-LLMs**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00386v1)  

---


**ABSTRACT**  
Assertion-based verification (ABV) is a critical method for ensuring design circuits comply with their architectural specifications, which are typically described in natural language. This process often requires significant interpretation by engineers to convert these specifications into functional verification assertions. Existing methods for generating assertions from natural language specifications are limited to sentences extracted by engineers, discouraging the practical application. In this work, we present AssertLLM, an automatic assertion generation framework for complete specification files. AssertLLM breaks down the complex task into three phases, incorporating three customized Large Language Models (LLMs) for extracting structural specifications, mapping signal definitions, and generating assertions. Additionally, we provide an open-source benchmark for assessing assertion generation capabilities. Our evaluation of AssertLLM on a full design, encompassing 23 signals, demonstrates that 89% of the generated assertions are both syntactically and functionally accurate.

{{</citation>}}


## cs.IT (1)



### (101/108) Coded Multi-User Information Retrieval with a Multi-Antenna Helper Node (Milad Abolpour et al., 2024)

{{<citation>}}

Milad Abolpour, MohammadJavad Salehi, Soheil Mohajer, Seyed Pooya Shariatpanahi, Antti Tölli. (2024)  
**Coded Multi-User Information Retrieval with a Multi-Antenna Helper Node**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2402.00465v1)  

---


**ABSTRACT**  
A novel coding design is proposed to enhance information retrieval in a wireless network of users with partial access to the data, in the sense of observation, measurement, computation, or storage. Information exchange in the network is assisted by a multi-antenna base station (BS), with no direct access to the data. Accordingly, the missing parts of data are exchanged among users through an uplink (UL) step followed by a downlink (DL) step. In this paper, new coding strategies, inspired by coded caching (CC) techniques, are devised to enhance both UL and DL steps. In the UL step, users transmit encoded and properly combined parts of their accessible data to the BS. Then, during the DL step, the BS carries out the required processing on its received signals and forwards a proper combination of the resulting signal terms back to the users, enabling each user to retrieve the desired information. Using the devised coded data retrieval strategy, the data exchange in both UL and DL steps requires the same communication delay, measured by normalized delivery time (NDT). Furthermore, the NDT of the UL/DL step is shown to coincide with the optimal NDT of the original DL multi-input single-output CC scheme, in which the BS is connected to a centralized data library.

{{</citation>}}


## cs.NE (1)



### (102/108) Genetic Programming Theory and Practice: A Fifteen-Year Trajectory (Moshe Sipper et al., 2024)

{{<citation>}}

Moshe Sipper, Jason H. Moore. (2024)  
**Genetic Programming Theory and Practice: A Fifteen-Year Trajectory**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2402.00425v1)  

---


**ABSTRACT**  
The GPTP workshop series, which began in 2003, has served over the years as a focal meeting for genetic programming (GP) researchers. As such, we think it provides an excellent source for studying the development of GP over the past fifteen years. We thus present herein a trajectory of the thematic developments in the field of GP.

{{</citation>}}


## eess.IV (1)



### (103/108) Disentangled Multimodal Brain MR Image Translation via Transformer-based Modality Infuser (Jihoon Cho et al., 2024)

{{<citation>}}

Jihoon Cho, Xiaofeng Liu, Fangxu Xing, Jinsong Ouyang, Georges El Fakhri, Jinah Park, Jonghye Woo. (2024)  
**Disentangled Multimodal Brain MR Image Translation via Transformer-based Modality Infuser**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.00375v1)  

---


**ABSTRACT**  
Multimodal Magnetic Resonance (MR) Imaging plays a crucial role in disease diagnosis due to its ability to provide complementary information by analyzing a relationship between multimodal images on the same subject. Acquiring all MR modalities, however, can be expensive, and, during a scanning session, certain MR images may be missed depending on the study protocol. The typical solution would be to synthesize the missing modalities from the acquired images such as using generative adversarial networks (GANs). Yet, GANs constructed with convolutional neural networks (CNNs) are likely to suffer from a lack of global relationships and mechanisms to condition the desired modality. To address this, in this work, we propose a transformer-based modality infuser designed to synthesize multimodal brain MR images. In our method, we extract modality-agnostic features from the encoder and then transform them into modality-specific features using the modality infuser. Furthermore, the modality infuser captures long-range relationships among all brain structures, leading to the generation of more realistic images. We carried out experiments on the BraTS 2018 dataset, translating between four MR modalities, and our experimental results demonstrate the superiority of our proposed method in terms of synthesis quality. In addition, we conducted experiments on a brain tumor segmentation task and different conditioning methods.

{{</citation>}}


## q-bio.OT (1)



### (104/108) The whack-a-mole governance challenge for AI-enabled synthetic biology: literature review and emerging frameworks (Trond Arne Undheim, 2024)

{{<citation>}}

Trond Arne Undheim. (2024)  
**The whack-a-mole governance challenge for AI-enabled synthetic biology: literature review and emerging frameworks**  

---
Primary Category: q-bio.OT  
Categories: cs-AI, q-bio-OT, q-bio.OT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.00312v1)  

---


**ABSTRACT**  
AI-enabled synthetic biology has tremendous potential but also significantly increases biorisks and brings about a new set of dual use concerns. The picture is complicated given the vast innovations envisioned to emerge by combining emerging technologies, as AI-enabled synthetic biology potentially scales up bioengineering into industrial biomanufacturing. However, the literature review indicates that goals such as maintaining a reasonable scope for innovation, or more ambitiously to foster a huge bioeconomy don't necessarily contrast with biosafety, but need to go hand in hand. This paper presents a literature review of the issues and describes emerging frameworks for policy and practice that transverse the options of command-and control, stewardship, bottom-up, and laissez-faire governance. How to achieve early warning systems that enable prevention and mitigation of future AI-enabled biohazards from the lab, from deliberate misuse, or from the public realm, will constantly need to evolve, and adaptive, interactive approaches should emerge. Although biorisk is subject to an established governance regime, and scientists generally adhere to biosafety protocols, even experimental, but legitimate use by scientists could lead to unexpected developments. Recent advances in chatbots enabled by generative AI have revived fears that advanced biological insight can more easily get into the hands of malignant individuals or organizations. Given these sets of issues, society needs to rethink how AI-enabled synthetic biology should be governed. The suggested way to visualize the challenge at hand is whack-a-mole governance, although the emerging solutions are perhaps not so different either.

{{</citation>}}


## cs.IR (2)



### (105/108) An Exam-based Evaluation Approach Beyond Traditional Relevance Judgments (Naghmeh Farzi et al., 2024)

{{<citation>}}

Naghmeh Farzi, Laura Dietz. (2024)  
**An Exam-based Evaluation Approach Beyond Traditional Relevance Judgments**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00309v1)  

---


**ABSTRACT**  
Current IR evaluation is based on relevance judgments, created either manually or automatically, with decisions outsourced to Large Language Models (LLMs). We offer an alternative paradigm, that never relies on relevance judgments in any form. Instead, a text is defined as relevant if it contains information that enables the answering of key questions. We use this idea to design the EXAM Answerability Metric to evaluate information retrieval/generation systems for their ability to provide topically relevant information.   We envision the role of a human judge to edit and define an exam question bank that will test for the presence of relevant information in text. We support this step by generating an initial set of exam questions. In the next phase, an LLM-based question answering system will automatically grade system responses by tracking which exam questions are answerable with which system responses. We propose two evaluation measures, the recall-oriented EXAM Cover metric, and the precision-oriented EXAM Qrels metric, the latter which can be implemented with trec_eval. This paradigm not only allows for the expansion of the exam question set post-hoc but also facilitates the ongoing evaluation of future information systems, whether they focus on retrieval, generation, or both.

{{</citation>}}


### (106/108) PAP-REC: Personalized Automatic Prompt for Recommendation Language Model (Zelong Li et al., 2024)

{{<citation>}}

Zelong Li, Jianchao Ji, Yingqiang Ge, Wenyue Hua, Yongfeng Zhang. (2024)  
**PAP-REC: Personalized Automatic Prompt for Recommendation Language Model**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.00284v1)  

---


**ABSTRACT**  
Recently emerged prompt-based Recommendation Language Models (RLM) can solve multiple recommendation tasks uniformly. The RLMs make full use of the inherited knowledge learned from the abundant pre-training data to solve the downstream recommendation tasks by prompts, without introducing additional parameters or network training. However, handcrafted prompts require significant expertise and human effort since slightly rewriting prompts may cause massive performance changes. In this paper, we propose PAP-REC, a framework to generate the Personalized Automatic Prompt for RECommendation language models to mitigate the inefficiency and ineffectiveness problems derived from manually designed prompts. Specifically, personalized automatic prompts allow different users to have different prompt tokens for the same task, automatically generated using a gradient-based method. One challenge for personalized automatic prompt generation for recommendation language models is the extremely large search space, leading to a long convergence time. To effectively and efficiently address the problem, we develop surrogate metrics and leverage an alternative updating schedule for prompting recommendation language models. Experimental results show that our PAP-REC framework manages to generate personalized prompts, and the automatically generated prompts outperform manually constructed prompts and also outperform various baseline recommendation models. The source code of the work is available at https://github.com/rutgerswiselab/PAP-REC.

{{</citation>}}


## q-fin.GN (1)



### (107/108) Attention-based Dynamic Multilayer Graph Neural Networks for Loan Default Prediction (Sahab Zandi et al., 2024)

{{<citation>}}

Sahab Zandi, Kamesh Korangi, María Óskarsdóttir, Christophe Mues, Cristián Bravo. (2024)  
**Attention-based Dynamic Multilayer Graph Neural Networks for Loan Default Prediction**  

---
Primary Category: q-fin.GN  
Categories: cs-LG, q-fin-GN, q-fin.GN  
Keywords: Attention, Graph Neural Network, Graph Neural Networks, LSTM  
[Paper Link](http://arxiv.org/abs/2402.00299v1)  

---


**ABSTRACT**  
Whereas traditional credit scoring tends to employ only individual borrower- or loan-level predictors, it has been acknowledged for some time that connections between borrowers may result in default risk propagating over a network. In this paper, we present a model for credit risk assessment leveraging a dynamic multilayer network built from a Graph Neural Network and a Recurrent Neural Network, each layer reflecting a different source of network connection. We test our methodology in a behavioural credit scoring context using a dataset provided by U.S. mortgage financier Freddie Mac, in which different types of connections arise from the geographical location of the borrower and their choice of mortgage provider. The proposed model considers both types of connections and the evolution of these connections over time. We enhance the model by using a custom attention mechanism that weights the different time snapshots according to their importance. After testing multiple configurations, a model with GAT, LSTM, and the attention mechanism provides the best results. Empirical results demonstrate that, when it comes to predicting probability of default for the borrowers, our proposed model brings both better results and novel insights for the analysis of the importance of connections and timestamps, compared to traditional methods.

{{</citation>}}


## cs.DB (1)



### (108/108) Effective Bug Detection in Graph Database Engines: An LLM-based Approach (Jiayi Wu et al., 2024)

{{<citation>}}

Jiayi Wu, Zhengyu Wu, Ronghua Li, Hongchao Qin, Guoren Wang. (2024)  
**Effective Bug Detection in Graph Database Engines: An LLM-based Approach**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2402.00292v1)  

---


**ABSTRACT**  
Graph database engines play a pivotal role in efficiently storing and managing graph data across various domains, including bioinformatics, knowledge graphs, and recommender systems. Ensuring data accuracy within graph database engines is paramount, as inaccuracies can yield unreliable analytical outcomes. Current bug-detection approaches are confined to specific graph query languages, limiting their applicabilities when handling graph database engines that use various graph query languages across various domains. Moreover, they require extensive prior knowledge to generate queries for detecting bugs. To address these challenges, we introduces DGDB, a novel paradigm harnessing large language models(LLM), such as ChatGPT, for comprehensive bug detection in graph database engines. DGDB leverages ChatGPT to generate high-quality queries for different graph query languages. It subsequently employs differential testing to identify bugs in graph database engines. We applied this paradigm to graph database engines using the Gremlin query language and those using the Cypher query language, generating approximately 4,000 queries each. In the latest versions of Neo4j, Agensgraph, and JanusGraph databases, we detected 2, 5, and 3 wrong-result bugs, respectively.

{{</citation>}}
