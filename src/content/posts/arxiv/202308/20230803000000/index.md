---
draft: false
title: "arXiv @ 2023.08.03"
date: 2023-08-03
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.03"
    identifier: arxiv_20230803
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (12)](#cscl-12)
- [quant-ph (2)](#quant-ph-2)
- [cs.AI (9)](#csai-9)
- [cs.CY (1)](#cscy-1)
- [cs.CV (24)](#cscv-24)
- [cs.HC (3)](#cshc-3)
- [cs.NE (2)](#csne-2)
- [cs.RO (4)](#csro-4)
- [cs.IR (3)](#csir-3)
- [cs.LG (11)](#cslg-11)
- [cs.GT (1)](#csgt-1)
- [cs.CR (5)](#cscr-5)
- [eess.SY (3)](#eesssy-3)
- [eess.IV (4)](#eessiv-4)
- [cs.DC (1)](#csdc-1)
- [cs.IT (1)](#csit-1)
- [cs.SE (2)](#csse-2)

## cs.CL (12)



### (1/88) DiactTOD: Learning Generalizable Latent Dialogue Acts for Controllable Task-Oriented Dialogue Systems (Qingyang Wu et al., 2023)

{{<citation>}}

Qingyang Wu, James Gung, Raphael Shu, Yi Zhang. (2023)  
**DiactTOD: Learning Generalizable Latent Dialogue Acts for Controllable Task-Oriented Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.00878v1)  

---


**ABSTRACT**  
Dialogue act annotations are important to improve response generation quality in task-oriented dialogue systems. However, it can be challenging to use dialogue acts to control response generation in a generalizable way because different datasets and tasks may have incompatible annotations. While alternative methods that utilize latent action spaces or reinforcement learning do not require explicit annotations, they may lack interpretability or face difficulties defining task-specific rewards. In this work, we present a novel end-to-end latent dialogue act model (DiactTOD) that represents dialogue acts in a latent space. DiactTOD, when pre-trained on a large corpus, is able to predict and control dialogue acts to generate controllable responses using these latent representations in a zero-shot fashion. Our approach demonstrates state-of-the-art performance across a wide range of experimental settings on the MultiWOZ dataset, including zero-shot, few-shot, and full data fine-tuning with both end-to-end and policy optimization configurations.

{{</citation>}}


### (2/88) GRDD: A Dataset for Greek Dialectal NLP (Stergios Chatzikyriakidis et al., 2023)

{{<citation>}}

Stergios Chatzikyriakidis, Chatrine Qwaider, Ilias Kolokousis, Christina Koula, Dimitris Papadakis, Efthymia Sakellariou. (2023)  
**GRDD: A Dataset for Greek Dialectal NLP**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.00802v1)  

---


**ABSTRACT**  
In this paper, we present a dataset for the computational study of a number of Modern Greek dialects. It consists of raw text data from four dialects of Modern Greek, Cretan, Pontic, Northern Greek and Cypriot Greek. The dataset is of considerable size, albeit imbalanced, and presents the first attempt to create large scale dialectal resources of this type for Modern Greek dialects. We then use the dataset to perform dialect idefntification. We experiment with traditional ML algorithms, as well as simple DL architectures. The results show very good performance on the task, potentially revealing that the dialects in question have distinct enough characteristics allowing even simple ML models to perform well on the task. Error analysis is performed for the top performing algorithms showing that in a number of cases the errors are due to insufficient dataset cleaning.

{{</citation>}}


### (3/88) Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models (Cheng-Yu Hsieh et al., 2023)

{{<citation>}}

Cheng-Yu Hsieh, Si-An Chen, Chun-Liang Li, Yasuhisa Fujii, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, Tomas Pfister. (2023)  
**Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.00675v1)  

---


**ABSTRACT**  
Today, large language models (LLMs) are taught to use new tools by providing a few demonstrations of the tool's usage. Unfortunately, demonstrations are hard to acquire, and can result in undesirable biased usage if the wrong demonstration is chosen. Even in the rare scenario that demonstrations are readily available, there is no principled selection protocol to determine how many and which ones to provide. As tasks grow more complex, the selection search grows combinatorially and invariably becomes intractable. Our work provides an alternative to demonstrations: tool documentation. We advocate the use of tool documentation, descriptions for the individual tool usage, over demonstrations. We substantiate our claim through three main empirical findings on 6 tasks across both vision and language modalities. First, on existing benchmarks, zero-shot prompts with only tool documentation are sufficient for eliciting proper tool usage, achieving performance on par with few-shot prompts. Second, on a newly collected realistic tool-use dataset with hundreds of available tool APIs, we show that tool documentation is significantly more valuable than demonstrations, with zero-shot documentation significantly outperforming few-shot without documentation. Third, we highlight the benefits of tool documentations by tackling image generation and video tracking using just-released unseen state-of-the-art models as tools. Finally, we highlight the possibility of using tool documentation to automatically enable new applications: by using nothing more than the documentation of GroundingDino, Stable Diffusion, XMem, and SAM, LLMs can re-invent the functionalities of the just-released Grounded-SAM and Track Anything models.

{{</citation>}}


### (4/88) JIANG: Chinese Open Foundation Language Model (Qinhua Duan et al., 2023)

{{<citation>}}

Qinhua Duan, Wenchao Gu, Yujia Chen, Wenxin Mao, Zewen Tian, Hui Cao. (2023)  
**JIANG: Chinese Open Foundation Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.00624v1)  

---


**ABSTRACT**  
With the advancements in large language model technology, it has showcased capabilities that come close to those of human beings across various tasks. This achievement has garnered significant interest from companies and scientific research institutions, leading to substantial investments in the research and development of these models. While numerous large models have emerged during this period, the majority of them have been trained primarily on English data. Although they exhibit decent performance in other languages, such as Chinese, their potential remains limited due to factors like vocabulary design and training corpus. Consequently, their ability to fully express their capabilities in Chinese falls short. To address this issue, we introduce the model named JIANG (Chinese pinyin of ginger) specifically designed for the Chinese language. We have gathered a substantial amount of Chinese corpus to train the model and have also optimized its structure. The extensive experimental results demonstrate the excellent performance of our model.

{{</citation>}}


### (5/88) Retrieval Augmented Generation and Representative Vector Summarization for large unstructured textual data in Medical Education (S. S. Manathunga et al., 2023)

{{<citation>}}

S. S. Manathunga, Y. A. Illangasekara. (2023)  
**Retrieval Augmented Generation and Representative Vector Summarization for large unstructured textual data in Medical Education**  

---
Primary Category: cs.CL  
Categories: H-3-1; J-3, cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2308.00479v1)  

---


**ABSTRACT**  
Large Language Models are increasingly being used for various tasks including content generation and as chatbots. Despite their impressive performances in general tasks, LLMs need to be aligned when applying for domain specific tasks to mitigate the problems of hallucination and producing harmful answers. Retrieval Augmented Generation (RAG) allows to easily attach and manipulate a non-parametric knowledgebases to LLMs. Applications of RAG in the field of medical education are discussed in this paper. A combined extractive and abstractive summarization method for large unstructured textual data using representative vectors is proposed.

{{</citation>}}


### (6/88) Discourse-Aware Text Simplification: From Complex Sentences to Linked Propositions (Christina Niklaus et al., 2023)

{{<citation>}}

Christina Niklaus, Matthias Cetto, André Freitas, Siegfried Handschuh. (2023)  
**Discourse-Aware Text Simplification: From Complex Sentences to Linked Propositions**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.00425v1)  

---


**ABSTRACT**  
Sentences that present a complex syntax act as a major stumbling block for downstream Natural Language Processing applications whose predictive quality deteriorates with sentence length and complexity. The task of Text Simplification (TS) may remedy this situation. It aims to modify sentences in order to make them easier to process, using a set of rewriting operations, such as reordering, deletion, or splitting. State-of-the-art syntactic TS approaches suffer from two major drawbacks: first, they follow a very conservative approach in that they tend to retain the input rather than transforming it, and second, they ignore the cohesive nature of texts, where context spread across clauses or sentences is needed to infer the true meaning of a statement. To address these problems, we present a discourse-aware TS approach that splits and rephrases complex English sentences within the semantic context in which they occur. Based on a linguistically grounded transformation stage that uses clausal and phrasal disembedding mechanisms, complex sentences are transformed into shorter utterances with a simple canonical structure that can be easily analyzed by downstream applications. With sentence splitting, we thus address a TS task that has hardly been explored so far. Moreover, we introduce the notion of minimality in this context, as we aim to decompose source sentences into a set of self-contained minimal semantic units. To avoid breaking down the input into a disjointed sequence of statements that is difficult to interpret because important contextual information is missing, we incorporate the semantic context between the split propositions in the form of hierarchical structures and semantic relationships. In that way, we generate a semantic hierarchy of minimal propositions that leads to a novel representation of complex assertions that puts a semantic layer on top of the simplified sentences.

{{</citation>}}


### (7/88) ZRIGF: An Innovative Multimodal Framework for Zero-Resource Image-Grounded Dialogue Generation (Bo Zhang et al., 2023)

{{<citation>}}

Bo Zhang, Jian Wang, Hui Ma, Bo Xu, Hongfei Lin. (2023)  
**ZRIGF: An Innovative Multimodal Framework for Zero-Resource Image-Grounded Dialogue Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-MM, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.00400v2)  

---


**ABSTRACT**  
Image-grounded dialogue systems benefit greatly from integrating visual information, resulting in high-quality response generation. However, current models struggle to effectively utilize such information in zero-resource scenarios, mainly due to the disparity between image and text modalities. To overcome this challenge, we propose an innovative multimodal framework, called ZRIGF, which assimilates image-grounded information for dialogue generation in zero-resource situations. ZRIGF implements a two-stage learning strategy, comprising contrastive pre-training and generative pre-training. Contrastive pre-training includes a text-image matching module that maps images and texts into a unified encoded vector space, along with a text-assisted masked image modeling module that preserves pre-training visual features and fosters further multimodal feature alignment. Generative pre-training employs a multimodal fusion module and an information transfer module to produce insightful responses based on harmonized multimodal representations. Comprehensive experiments conducted on both text-based and image-grounded dialogue datasets demonstrate ZRIGF's efficacy in generating contextually pertinent and informative responses. Furthermore, we adopt a fully zero-resource scenario in the image-grounded dialogue dataset to demonstrate our framework's robust generalization capabilities in novel domains. The code is available at https://github.com/zhangbo-nlp/ZRIGF.

{{</citation>}}


### (8/88) Tackling Hallucinations in Neural Chart Summarization (Saad Obaid ul Islam et al., 2023)

{{<citation>}}

Saad Obaid ul Islam, Iza Škrjanec, Ondřej Dušek, Vera Demberg. (2023)  
**Tackling Hallucinations in Neural Chart Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: NLI, Summarization  
[Paper Link](http://arxiv.org/abs/2308.00399v1)  

---


**ABSTRACT**  
Hallucinations in text generation occur when the system produces text that is not grounded in the input. In this work, we tackle the problem of hallucinations in neural chart summarization. Our analysis shows that the target side of chart summarization training datasets often contains additional information, leading to hallucinations. We propose a natural language inference (NLI) based method to preprocess the training data and show through human evaluation that our method significantly reduces hallucinations. We also found that shortening long-distance dependencies in the input sequence and adding chart-related information like title and legends improves the overall performance.

{{</citation>}}


### (9/88) LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack (Hai Zhu et al., 2023)

{{<citation>}}

Hai Zhu, Zhaoqing Yang, Weiwei Shang, Yuren Wu. (2023)  
**LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2308.00319v1)  

---


**ABSTRACT**  
Natural language processing models are vulnerable to adversarial examples. Previous textual adversarial attacks adopt gradients or confidence scores to calculate word importance ranking and generate adversarial examples. However, this information is unavailable in the real world. Therefore, we focus on a more realistic and challenging setting, named hard-label attack, in which the attacker can only query the model and obtain a discrete prediction label. Existing hard-label attack algorithms tend to initialize adversarial examples by random substitution and then utilize complex heuristic algorithms to optimize the adversarial perturbation. These methods require a lot of model queries and the attack success rate is restricted by adversary initialization. In this paper, we propose a novel hard-label attack algorithm named LimeAttack, which leverages a local explainable method to approximate word importance ranking, and then adopts beam search to find the optimal solution. Extensive experiments show that LimeAttack achieves the better attacking performance compared with existing hard-label attack under the same query budget. In addition, we evaluate the effectiveness of LimeAttack on large language models, and results indicate that adversarial examples remain a significant threat to large language models. The adversarial examples crafted by LimeAttack are highly transferable and effectively improve model robustness in adversarial training.

{{</citation>}}


### (10/88) Skills-in-Context Prompting: Unlocking Compositionality in Large Language Models (Jiaao Chen et al., 2023)

{{<citation>}}

Jiaao Chen, Xiaoman Pan, Dian Yu, Kaiqiang Song, Xiaoyang Wang, Dong Yu, Jianshu Chen. (2023)  
**Skills-in-Context Prompting: Unlocking Compositionality in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.00304v1)  

---


**ABSTRACT**  
We consider the problem of eliciting compositional generalization capabilities in large language models (LLMs) with a novel type of prompting strategy. Compositional generalization empowers the LLMs to solve problems that are harder than the ones they have seen (i.e., easy-to-hard generalization), which is a critical reasoning capability of human-like intelligence. However, even the current state-of-the-art LLMs still struggle with this form of reasoning. To bridge this gap, we propose skills-in-context (SKiC) prompting, which instructs LLMs how to compose basic skills to resolve more complex problems. We find that it is crucial to demonstrate both the skills and the compositional examples within the same prompting context. With as few as two examplars, our SKiC prompting initiates strong synergies between skills and their composition capabilities. Notably, it empowers LLMs to solve unseen problems that require innovative skill compositions, achieving near-perfect generalization on a broad range of challenging compositionality tasks. Intriguingly, SKiC prompting unlocks the latent potential of LLMs, enabling them to leverage pre-existing internal skills acquired during earlier pretraining and alignment stages, even when these skills are not explicitly presented in the prompting context. This results in the capability of LLMs to solve unseen complex problems by activating and composing these internal competencies.

{{</citation>}}


### (11/88) Towards Effective Ancient Chinese Translation: Dataset, Model, and Evaluation (Geyang Guo et al., 2023)

{{<citation>}}

Geyang Guo, Jiarong Yang, Fengyuan Lu, Jiaxin Qin, Tianyi Tang, Wayne Xin Zhao. (2023)  
**Towards Effective Ancient Chinese Translation: Dataset, Model, and Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, BLEU, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2308.00240v1)  

---


**ABSTRACT**  
Interpreting ancient Chinese has been the key to comprehending vast Chinese literature, tradition, and civilization. In this paper, we propose Erya for ancient Chinese translation. From a dataset perspective, we collect, clean, and classify ancient Chinese materials from various sources, forming the most extensive ancient Chinese resource to date. From a model perspective, we devise Erya training method oriented towards ancient Chinese. We design two jointly-working tasks: disyllabic aligned substitution (DAS) and dual masked language model (DMLM). From an evaluation perspective, we build a benchmark to judge ancient Chinese translation quality in different scenarios and evaluate the ancient Chinese translation capacities of various existing models. Our model exhibits remarkable zero-shot performance across five domains, with over +12.0 BLEU against GPT-3.5 models and better human evaluation results than ERNIE Bot. Subsequent fine-tuning further shows the superior transfer capability of Erya model with +6.2 BLEU gain. We release all the above-mentioned resources at https://github.com/RUCAIBox/Erya.

{{</citation>}}


### (12/88) Advancing Beyond Identification: Multi-bit Watermark for Language Models (KiYoon Yoo et al., 2023)

{{<citation>}}

KiYoon Yoo, Wonhyuk Ahn, Nojun Kwak. (2023)  
**Advancing Beyond Identification: Multi-bit Watermark for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.00221v1)  

---


**ABSTRACT**  
This study aims to proactively tackle misuse of large language models beyond identification of machine-generated text. While existing methods focus on detection, some malicious misuses demand tracing the adversary user for counteracting them. To address this, we propose "Multi-bit Watermark through Color-listing" (COLOR), embedding traceable multi-bit information during language model generation. Leveraging the benefits of zero-bit watermarking (Kirchenbauer et al., 2023a), COLOR enables extraction without model access, on-the-fly embedding, and maintains text quality, while allowing zero-bit detection all at the same time. Preliminary experiments demonstrates successful embedding of 32-bit messages with 91.9% accuracy in moderate-length texts ($\sim$500 tokens). This work advances strategies to counter language model misuse effectively.

{{</citation>}}


## quant-ph (2)



### (13/88) Single-Qubit Gates Matter for Optimising Quantum Circuit Depth in Qubit Mapping (Sanjiang Li et al., 2023)

{{<citation>}}

Sanjiang Li, Ky Dan Nguyen, Zachary Clare, Yuan Feng. (2023)  
**Single-Qubit Gates Matter for Optimising Quantum Circuit Depth in Qubit Mapping**  

---
Primary Category: quant-ph  
Categories: cs-AR, cs-ET, quant-ph, quant-ph  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.00876v1)  

---


**ABSTRACT**  
Quantum circuit transformation (QCT, a.k.a. qubit mapping) is a critical step in quantum circuit compilation. Typically, QCT is achieved by finding an appropriate initial mapping and using SWAP gates to route the qubits such that all connectivity constraints are satisfied. The objective of QCT can be to minimise circuit size or depth. Most existing QCT algorithms prioritise minimising circuit size, potentially overlooking the impact of single-qubit gates on circuit depth. In this paper, we first point out that a single SWAP gate insertion can double the circuit depth, and then propose a simple and effective method that takes into account the impact of single-qubit gates on circuit depth. Our method can be combined with many existing QCT algorithms to optimise circuit depth. The Qiskit SABRE algorithm has been widely accepted as the state-of-the-art algorithm for optimising both circuit size and depth. We demonstrate the effectiveness of our method by embedding it in SABRE, showing that it can reduce circuit depth by up to 50% and 27% on average on, for instance, Google Sycamore and 117 real quantum circuits from MQTBench.

{{</citation>}}


### (14/88) Semisupervised Anomaly Detection using Support Vector Regression with Quantum Kernel (Kilian Tscharke et al., 2023)

{{<citation>}}

Kilian Tscharke, Sebastian Issel, Pascal Debus. (2023)  
**Semisupervised Anomaly Detection using Support Vector Regression with Quantum Kernel**  

---
Primary Category: quant-ph  
Categories: cs-CR, cs-LG, quant-ph, quant-ph  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.00583v1)  

---


**ABSTRACT**  
Anomaly detection (AD) involves identifying observations or events that deviate in some way from the rest of the data. Machine learning techniques have shown success in automating this process by detecting hidden patterns and deviations in large-scale data. The potential of quantum computing for machine learning has been widely recognized, leading to extensive research efforts to develop suitable quantum machine learning (QML) algorithms. In particular, the search for QML algorithms for near-term NISQ devices is in full swing. However, NISQ devices pose additional challenges due to their limited qubit coherence times, low number of qubits, and high error rates. Kernel methods based on quantum kernel estimation have emerged as a promising approach to QML on NISQ devices, offering theoretical guarantees, versatility, and compatibility with NISQ constraints. Especially support vector machines (SVM) utilizing quantum kernel estimation have shown success in various supervised learning tasks. However, in the context of AD, semisupervised learning is of great relevance, and yet there is limited research published in this area. This paper introduces an approach to semisupervised AD based on the reconstruction loss of a support vector regression (SVR) with quantum kernel. This novel model is an alternative to the variational quantum and quantum kernel one-class classifiers, and is compared to a quantum autoencoder as quantum baseline and a SVR with radial-basis-function (RBF) kernel as well as a classical autoencoder as classical baselines. The models are benchmarked extensively on 10 real-world AD data sets and one toy data set, and it is shown that our SVR model with quantum kernel performs better than the SVR with RBF kernel as well as all other models, achieving highest mean AUC over all data sets. In addition, our QSVR outperforms the quantum autoencoder on 9 out of 11 data sets.

{{</citation>}}


## cs.AI (9)



### (15/88) Beneficent Intelligence: A Capability Approach to Modeling Benefit, Assistance, and Associated Moral Failures through AI Systems (Alex John London et al., 2023)

{{<citation>}}

Alex John London, Hoda heidari. (2023)  
**Beneficent Intelligence: A Capability Approach to Modeling Benefit, Assistance, and Associated Moral Failures through AI Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00868v1)  

---


**ABSTRACT**  
The prevailing discourse around AI ethics lacks the language and formalism necessary to capture the diverse ethical concerns that emerge when AI systems interact with individuals. Drawing on Sen and Nussbaum's capability approach, we present a framework formalizing a network of ethical concepts and entitlements necessary for AI systems to confer meaningful benefit or assistance to stakeholders. Such systems enhance stakeholders' ability to advance their life plans and well-being while upholding their fundamental rights. We characterize two necessary conditions for morally permissible interactions between AI systems and those impacted by their functioning, and two sufficient conditions for realizing the ideal of meaningful benefit. We then contrast this ideal with several salient failure modes, namely, forms of social interactions that constitute unjustified paternalism, coercion, deception, exploitation and domination. The proliferation of incidents involving AI in high-stakes domains underscores the gravity of these issues and the imperative to take an ethics-led approach to AI systems from their inception.

{{</citation>}}


### (16/88) A Knowledge-Oriented Approach to Enhance Integration and Communicability in the Polkadot Ecosystem (Marcio Ferreira Moreno et al., 2023)

{{<citation>}}

Marcio Ferreira Moreno, Rafael Rossi de Mello Brandão. (2023)  
**A Knowledge-Oriented Approach to Enhance Integration and Communicability in the Polkadot Ecosystem**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DC, cs-IR, cs-NI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00735v1)  

---


**ABSTRACT**  
The Polkadot ecosystem is a disruptive and highly complex multi-chain architecture that poses challenges in terms of data analysis and communicability. Currently, there is a lack of standardized and holistic approaches to retrieve and analyze data across parachains and applications, making it difficult for general users and developers to access ecosystem data consistently. This paper proposes a conceptual framework that includes a domain ontology called POnto (a Polkadot Ontology) to address these challenges. POnto provides a structured representation of the ecosystem's concepts and relationships, enabling a formal understanding of the platform. The proposed knowledge-oriented approach enhances integration and communicability, enabling a wider range of users to participate in the ecosystem and facilitating the development of AI-based applications. The paper presents a case study methodology to validate the proposed framework, which includes expert feedback and insights from the Polkadot community. The POnto ontology and the roadmap for a query engine based on a Controlled Natural Language using the ontology, provide valuable contributions to the growth and adoption of the Polkadot ecosystem in heterogeneous socio-technical environments.

{{</citation>}}


### (17/88) Reinforcement Learning-based Non-Autoregressive Solver for Traveling Salesman Problems (Yubin Xiao et al., 2023)

{{<citation>}}

Yubin Xiao, Di Wang, Huanhuan Chen, Boyang Li, Wei Pang, Xuan Wu, Hao Li, Dong Xu, Yanchun Liang, You Zhou. (2023)  
**Reinforcement Learning-based Non-Autoregressive Solver for Traveling Salesman Problems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GNN, Graph Neural Network, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.00560v1)  

---


**ABSTRACT**  
The Traveling Salesman Problem (TSP) is a well-known problem in combinatorial optimization with applications in various domains. However, existing TSP solvers face challenges in producing high-quality solutions with low latency. To address this issue, we propose NAR4TSP, which produces TSP solutions in a Non-Autoregressive (NAR) manner using a specially designed Graph Neural Network (GNN), achieving faster inference speed. Moreover, NAR4TSP is trained using an enhanced Reinforcement Learning (RL) strategy, eliminating the dependency on costly labels used to train conventional supervised learning-based NAR models. To the best of our knowledge, NAR4TSP is the first TSP solver that successfully combines RL and NAR decoding. The experimental results on both synthetic and real-world TSP instances demonstrate that NAR4TSP outperforms four state-of-the-art models in terms of solution quality, inference latency, and generalization ability. Lastly, we present visualizations of NAR4TSP's decoding process and its overall path planning to showcase the feasibility of implementing NAR4TSP in an end-to-end manner and its effectiveness, respectively.

{{</citation>}}


### (18/88) SurveyLM: A platform to explore emerging value perspectives in augmented language models' behaviors (Steve J. Bickley et al., 2023)

{{<citation>}}

Steve J. Bickley, Ho Fai Chan, Bang Dao, Benno Torgler, Son Tran. (2023)  
**SurveyLM: A platform to explore emerging value perspectives in augmented language models' behaviors**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SI, cs.AI, econ-GN, q-fin-EC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00521v1)  

---


**ABSTRACT**  
This white paper presents our work on SurveyLM, a platform for analyzing augmented language models' (ALMs) emergent alignment behaviors through their dynamically evolving attitude and value perspectives in complex social contexts. Social Artificial Intelligence (AI) systems, like ALMs, often function within nuanced social scenarios where there is no singular correct response, or where an answer is heavily dependent on contextual factors, thus necessitating an in-depth understanding of their alignment dynamics. To address this, we apply survey and experimental methodologies, traditionally used in studying social behaviors, to evaluate ALMs systematically, thus providing unprecedented insights into their alignment and emergent behaviors. Moreover, the SurveyLM platform leverages the ALMs' own feedback to enhance survey and experiment designs, exploiting an underutilized aspect of ALMs, which accelerates the development and testing of high-quality survey frameworks while conserving resources. Through SurveyLM, we aim to shed light on factors influencing ALMs' emergent behaviors, facilitate their alignment with human intentions and expectations, and thereby contributed to the responsible development and deployment of advanced social AI systems. This white paper underscores the platform's potential to deliver robust results, highlighting its significance to alignment research and its implications for future social AI systems.

{{</citation>}}


### (19/88) Structural Embeddings of Tools for Large Language Models (Eren Unlu, 2023)

{{<citation>}}

Eren Unlu. (2023)  
**Structural Embeddings of Tools for Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2308.00447v1)  

---


**ABSTRACT**  
It is evident that the current state of Large Language Models (LLMs) necessitates the incorporation of external tools. The lack of straightforward algebraic and logical reasoning is well documented and prompted researchers to develop frameworks which allow LLMs to operate via external tools. The ontological nature of tool utilization for a specific task can be well formulated with a Directed Acyclic Graph (DAG). The central aim of the paper is to highlight the importance of graph based approaches to LLM-tool interaction in near future. We propose an exemplary framework to guide the orchestration of exponentially increasing numbers of external tools with LLMs,where objectives and functionalities of tools are graph encoded hierarchically. Assuming that textual segments of a Chain-of-Thought (CoT) can be imagined as a tool as defined here, the graph based framework can pave new avenues in that particular direction as well.

{{</citation>}}


### (20/88) SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning (Ning Miao et al., 2023)

{{<citation>}}

Ning Miao, Yee Whye Teh, Tom Rainforth. (2023)  
**SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: QA, Reasoning, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.00436v2)  

---


**ABSTRACT**  
The recent progress in large language models (LLMs), especially the invention of chain-of-thoughts (CoT) prompting, makes it possible to solve reasoning problems. However, even the strongest LLMs are still struggling with more complicated problems that require non-linear thinking and multi-step reasoning. In this work, we explore whether LLMs have the ability to recognize their own errors, without resorting to external resources. In particular, we investigate whether they can be used to identify individual errors within a step-by-step reasoning. To this end, we propose a zero-shot verification scheme to recognize such errors. We then use this verification scheme to improve question-answering performance, by using it to perform weighted voting on different generated answers. We test the method on three math datasets-GSM8K, MathQA, and MATH-and find that it successfully recognizes errors and, in turn, increases final predictive performance.

{{</citation>}}


### (21/88) MetaGPT: Meta Programming for Multi-Agent Collaborative Framework (Sirui Hong et al., 2023)

{{<citation>}}

Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu. (2023)  
**MetaGPT: Meta Programming for Multi-Agent Collaborative Framework**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MA, cs.AI  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2308.00352v2)  

---


**ABSTRACT**  
Recently, remarkable progress has been made in automated task-solving through the use of multi-agents driven by large language models (LLMs). However, existing works primarily focuses on simple tasks lacking exploration and investigation in complicated tasks mainly due to the hallucination problem. This kind of hallucination gets amplified infinitely as multiple intelligent agents interact with each other, resulting in failures when tackling complicated problems.Therefore, we introduce MetaGPT, an innovative framework that infuses effective human workflows as a meta programming approach into LLM-driven multi-agent collaboration. In particular, MetaGPT first encodes Standardized Operating Procedures (SOPs) into prompts, fostering structured coordination. And then, it further mandates modular outputs, bestowing agents with domain expertise paralleling human professionals to validate outputs and reduce compounded errors. In this way, MetaGPT leverages the assembly line work model to assign diverse roles to various agents, thus establishing a framework that can effectively and cohesively deconstruct complex multi-agent collaborative problems. Our experiments conducted on collaborative software engineering tasks illustrate MetaGPT's capability in producing comprehensive solutions with higher coherence relative to existing conversational and chat-based multi-agent systems. This underscores the potential of incorporating human domain knowledge into multi-agents, thus opening up novel avenues for grappling with intricate real-world challenges. The GitHub repository of this project is made publicly available on: https://github.com/geekan/MetaGPT

{{</citation>}}


### (22/88) Monitoring Algorithmic Fairness under Partial Observations (Thomas A. Henzinger et al., 2023)

{{<citation>}}

Thomas A. Henzinger, Konstantin Kueffner, Kaushik Mallik. (2023)  
**Monitoring Algorithmic Fairness under Partial Observations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00341v1)  

---


**ABSTRACT**  
As AI and machine-learned software are used increasingly for making decisions that affect humans, it is imperative that they remain fair and unbiased in their decisions. To complement design-time bias mitigation measures, runtime verification techniques have been introduced recently to monitor the algorithmic fairness of deployed systems. Previous monitoring techniques assume full observability of the states of the (unknown) monitored system. Moreover, they can monitor only fairness properties that are specified as arithmetic expressions over the probabilities of different events. In this work, we extend fairness monitoring to systems modeled as partially observed Markov chains (POMC), and to specifications containing arithmetic expressions over the expected values of numerical functions on event sequences. The only assumptions we make are that the underlying POMC is aperiodic and starts in the stationary distribution, with a bound on its mixing time being known. These assumptions enable us to estimate a given property for the entire distribution of possible executions of the monitored POMC, by observing only a single execution. Our monitors observe a long run of the system and, after each new observation, output updated PAC-estimates of how fair or biased the system is. The monitors are computationally lightweight and, using a prototype implementation, we demonstrate their effectiveness on several real-world examples.

{{</citation>}}


### (23/88) Instructed to Bias: Instruction-Tuned Language Models Exhibit Emergent Cognitive Bias (Itay Itzhak et al., 2023)

{{<citation>}}

Itay Itzhak, Gabriel Stanovsky, Nir Rosenfeld, Yonatan Belinkov. (2023)  
**Instructed to Bias: Instruction-Tuned Language Models Exhibit Emergent Cognitive Bias**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-LG, cs.AI  
Keywords: Bias, GPT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2308.00225v1)  

---


**ABSTRACT**  
Recent studies show that instruction tuning and learning from human feedback improve the abilities of large language models (LMs) dramatically. While these tuning methods can make models generate high-quality text, we conjecture that more implicit cognitive biases may arise in these fine-tuned models. Our work provides evidence that these fine-tuned models exhibit biases that were absent or less pronounced in their pretrained predecessors. We examine the extent of this phenomenon in three cognitive biases - the decoy effect, the certainty effect, and the belief bias - all of which are known to influence human decision-making and reasoning. Our findings highlight the presence of these biases in various models, especially those that have undergone instruction tuning, such as Flan-T5, GPT3.5, and GPT4. This research constitutes a step toward comprehending cognitive biases in instruction-tuned LMs, which is crucial for the development of more reliable and unbiased language models.

{{</citation>}}


## cs.CY (1)



### (24/88) Confidence-Building Measures for Artificial Intelligence: Workshop Proceedings (Sarah Shoker et al., 2023)

{{<citation>}}

Sarah Shoker, Andrew Reddie, Sarah Barrington, Miles Brundage, Husanjot Chahal, Michael Depp, Bill Drexel, Ritwik Gupta, Marina Favaro, Jake Hecla, Alan Hickey, Margarita Konaev, Kirthi Kumar, Nathan Lambert, Andrew Lohn, Cullen O'Keefe, Nazneen Rajani, Michael Sellitto, Robert Trager, Leah Walker, Alexa Wehsener, Jessica Young. (2023)  
**Confidence-Building Measures for Artificial Intelligence: Workshop Proceedings**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2308.00862v1)  

---


**ABSTRACT**  
Foundation models could eventually introduce several pathways for undermining state security: accidents, inadvertent escalation, unintentional conflict, the proliferation of weapons, and the interference with human diplomacy are just a few on a long list. The Confidence-Building Measures for Artificial Intelligence workshop hosted by the Geopolitics Team at OpenAI and the Berkeley Risk and Security Lab at the University of California brought together a multistakeholder group to think through the tools and strategies to mitigate the potential risks introduced by foundation models to international security. Originating in the Cold War, confidence-building measures (CBMs) are actions that reduce hostility, prevent conflict escalation, and improve trust between parties. The flexibility of CBMs make them a key instrument for navigating the rapid changes in the foundation model landscape. Participants identified the following CBMs that directly apply to foundation models and which are further explained in this conference proceedings: 1. crisis hotlines 2. incident sharing 3. model, transparency, and system cards 4. content provenance and watermarks 5. collaborative red teaming and table-top exercises and 6. dataset and evaluation sharing. Because most foundation model developers are non-government entities, many CBMs will need to involve a wider stakeholder community. These measures can be implemented either by AI labs or by relevant government actors.

{{</citation>}}


## cs.CV (24)



### (25/88) Training on Foveated Images Improves Robustness to Adversarial Attacks (Muhammad A. Shah et al., 2023)

{{<citation>}}

Muhammad A. Shah, Bhiksha Raj. (2023)  
**Training on Foveated Images Improves Robustness to Adversarial Attacks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2308.00854v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) have been shown to be vulnerable to adversarial attacks -- subtle, perceptually indistinguishable perturbations of inputs that change the response of the model. In the context of vision, we hypothesize that an important contributor to the robustness of human visual perception is constant exposure to low-fidelity visual stimuli in our peripheral vision. To investigate this hypothesis, we develop \RBlur, an image transform that simulates the loss in fidelity of peripheral vision by blurring the image and reducing its color saturation based on the distance from a given fixation point. We show that compared to DNNs trained on the original images, DNNs trained on images transformed by \RBlur are substantially more robust to adversarial attacks, as well as other, non-adversarial, corruptions, achieving up to 25\% higher accuracy on perturbed data.

{{</citation>}}


### (26/88) Addressing Uncertainty in Imbalanced Histopathology Image Classification of HER2 Breast Cancer: An interpretable Ensemble Approach with Threshold Filtered Single Instance Evaluation (SIE) (Md Sakib Hossain Shovon et al., 2023)

{{<citation>}}

Md Sakib Hossain Shovon, M. F. Mridha, Khan Md Hasib, Sultan Alfarhood, Mejdl Safran, Dunren Che. (2023)  
**Addressing Uncertainty in Imbalanced Histopathology Image Classification of HER2 Breast Cancer: An interpretable Ensemble Approach with Threshold Filtered Single Instance Evaluation (SIE)**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Clinical, Image Classification  
[Paper Link](http://arxiv.org/abs/2308.00806v1)  

---


**ABSTRACT**  
Breast Cancer (BC) is among women's most lethal health concerns. Early diagnosis can alleviate the mortality rate by helping patients make efficient treatment decisions. Human Epidermal Growth Factor Receptor (HER2) has become one the most lethal subtype of BC. According to the College of American Pathologists/American Society of Clinical Oncology (CAP/ASCO), the severity level of HER2 expression can be classified between 0 and 3+ range. HER2 can be detected effectively from immunohistochemical (IHC) and, hematoxylin \& eosin (HE) images of different classes such as 0, 1+, 2+, and 3+. An ensemble approach integrated with threshold filtered single instance evaluation (SIE) technique has been proposed in this study to diagnose BC from the multi-categorical expression of HER2 subtypes. Initially, DenseNet201 and Xception have been ensembled into a single classifier as feature extractors with an effective combination of global average pooling, dropout layer, dense layer with a swish activation function, and l2 regularizer, batch normalization, etc. After that, extracted features has been processed through single instance evaluation (SIE) to determine different confidence levels and adjust decision boundary among the imbalanced classes. This study has been conducted on the BC immunohistochemical (BCI) dataset, which is classified by pathologists into four stages of HER2 BC. This proposed approach known as DenseNet201-Xception-SIE with a threshold value of 0.7 surpassed all other existing state-of-art models with an accuracy of 97.12\%, precision of 97.15\%, and recall of 97.68\% on H\&E data and, accuracy of 97.56\%, precision of 97.57\%, and recall of 98.00\% on IHC data respectively, maintaining momentous improvement. Finally, Grad-CAM and Guided Grad-CAM have been employed in this study to interpret, how TL-based model works on the histopathology dataset and make decisions from the data.

{{</citation>}}


### (27/88) LISA: Reasoning Segmentation via Large Language Model (Xin Lai et al., 2023)

{{<citation>}}

Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia. (2023)  
**LISA: Reasoning Segmentation via Large Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.00692v1)  

---


**ABSTRACT**  
Although perception systems have made remarkable advancements in recent years, they still rely on explicit human instruction to identify the target objects or categories before executing visual recognition tasks. Such systems lack the ability to actively reason and comprehend implicit user intentions. In this work, we propose a new segmentation task -- reasoning segmentation. The task is designed to output a segmentation mask given a complex and implicit query text. Furthermore, we establish a benchmark comprising over one thousand image-instruction pairs, incorporating intricate reasoning and world knowledge for evaluation purposes. Finally, we present LISA: large Language Instructed Segmentation Assistant, which inherits the language generation capabilities of the multi-modal Large Language Model (LLM) while also possessing the ability to produce segmentation masks. We expand the original vocabulary with a <SEG> token and propose the embedding-as-mask paradigm to unlock the segmentation capability. Remarkably, LISA can handle cases involving: 1) complex reasoning; 2) world knowledge; 3) explanatory answers; 4) multi-turn conversation. Also, it demonstrates robust zero-shot capability when trained exclusively on reasoning-free datasets. In addition, fine-tuning the model with merely 239 reasoning segmentation image-instruction pairs results in further performance enhancement. Experiments show our method not only unlocks new reasoning segmentation capabilities but also proves effective in both complex reasoning segmentation and standard referring segmentation tasks. Code, models, and demo are at https://github.com/dvlab-research/LISA.

{{</citation>}}


### (28/88) Toward Zero-shot Character Recognition: A Gold Standard Dataset with Radical-level Annotations (Xiaolei Diao et al., 2023)

{{<citation>}}

Xiaolei Diao, Daqian Shi, Jian Li, Lida Shi, Mingzhe Yue, Ruihua Qi, Chuntao Li, Hao Xu. (2023)  
**Toward Zero-shot Character Recognition: A Gold Standard Dataset with Radical-level Annotations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2308.00655v1)  

---


**ABSTRACT**  
Optical character recognition (OCR) methods have been applied to diverse tasks, e.g., street view text recognition and document analysis. Recently, zero-shot OCR has piqued the interest of the research community because it considers a practical OCR scenario with unbalanced data distribution. However, there is a lack of benchmarks for evaluating such zero-shot methods that apply a divide-and-conquer recognition strategy by decomposing characters into radicals. Meanwhile, radical recognition, as another important OCR task, also lacks radical-level annotation for model training. In this paper, we construct an ancient Chinese character image dataset that contains both radical-level and character-level annotations to satisfy the requirements of the above-mentioned methods, namely, ACCID, where radical-level annotations include radical categories, radical locations, and structural relations. To increase the adaptability of ACCID, we propose a splicing-based synthetic character algorithm to augment the training samples and apply an image denoising method to improve the image quality. By introducing character decomposition and recombination, we propose a baseline method for zero-shot OCR. The experimental results demonstrate the validity of ACCID and the baseline model quantitatively and qualitatively.

{{</citation>}}


### (29/88) Ada-DQA: Adaptive Diverse Quality-aware Feature Acquisition for Video Quality Assessment (Hongbo Liu et al., 2023)

{{<citation>}}

Hongbo Liu, Mingda Wu, Kun Yuan, Ming Sun, Yansong Tang, Chuanchuan Zheng, Xing Wen, Xiu Li. (2023)  
**Ada-DQA: Adaptive Diverse Quality-aware Feature Acquisition for Video Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.00729v1)  

---


**ABSTRACT**  
Video quality assessment (VQA) has attracted growing attention in recent years. While the great expense of annotating large-scale VQA datasets has become the main obstacle for current deep-learning methods. To surmount the constraint of insufficient training data, in this paper, we first consider the complete range of video distribution diversity (\ie content, distortion, motion) and employ diverse pretrained models (\eg architecture, pretext task, pre-training dataset) to benefit quality representation. An Adaptive Diverse Quality-aware feature Acquisition (Ada-DQA) framework is proposed to capture desired quality-related features generated by these frozen pretrained models. By leveraging the Quality-aware Acquisition Module (QAM), the framework is able to extract more essential and relevant features to represent quality. Finally, the learned quality representation is utilized as supplementary supervisory information, along with the supervision of the labeled quality score, to guide the training of a relatively lightweight VQA model in a knowledge distillation manner, which largely reduces the computational cost during inference. Experimental results on three mainstream no-reference VQA benchmarks clearly show the superior performance of Ada-DQA in comparison with current state-of-the-art approaches without using extra training data of VQA.

{{</citation>}}


### (30/88) Explainable Cost-Sensitive Deep Neural Networks for Brain Tumor Detection from Brain MRI Images considering Data Imbalance (Md Tanvir Rouf Shawon et al., 2023)

{{<citation>}}

Md Tanvir Rouf Shawon, G. M. Shahariar Shibli, Farzad Ahmed, Sajib Kumar Saha Joy. (2023)  
**Explainable Cost-Sensitive Deep Neural Networks for Brain Tumor Detection from Brain MRI Images considering Data Imbalance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00608v1)  

---


**ABSTRACT**  
This paper presents a research study on the use of Convolutional Neural Network (CNN), ResNet50, InceptionV3, EfficientNetB0 and NASNetMobile models to efficiently detect brain tumors in order to reduce the time required for manual review of the report and create an automated system for classifying brain tumors. An automated pipeline is proposed, which encompasses five models: CNN, ResNet50, InceptionV3, EfficientNetB0 and NASNetMobile. The performance of the proposed architecture is evaluated on a balanced dataset and found to yield an accuracy of 99.33% for fine-tuned InceptionV3 model. Furthermore, Explainable AI approaches are incorporated to visualize the model's latent behavior in order to understand its black box behavior. To further optimize the training process, a cost-sensitive neural network approach has been proposed in order to work with imbalanced datasets which has achieved almost 4% more accuracy than the conventional models used in our experiments. The cost-sensitive InceptionV3 (CS-InceptionV3) and CNN (CS-CNN) show a promising accuracy of 92.31% and a recall value of 1.00 respectively on an imbalanced dataset. The proposed models have shown great potential in improving tumor detection accuracy and must be further developed for application in practical solutions. We have provided the datasets and made our implementations publicly available at - https://github.com/shahariar-shibli/Explainable-Cost-Sensitive-Deep-Neural-Networks-for-Brain-Tumor-Detection-from-Brain-MRI-Images

{{</citation>}}


### (31/88) MonoNext: A 3D Monocular Object Detection with ConvNext (Marcelo Eduardo Pederiva et al., 2023)

{{<citation>}}

Marcelo Eduardo Pederiva, José Mario De Martino, Alessandro Zimmer. (2023)  
**MonoNext: A 3D Monocular Object Detection with ConvNext**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.00596v1)  

---


**ABSTRACT**  
Autonomous driving perception tasks rely heavily on cameras as the primary sensor for Object Detection, Semantic Segmentation, Instance Segmentation, and Object Tracking. However, RGB images captured by cameras lack depth information, which poses a significant challenge in 3D detection tasks. To supplement this missing data, mapping sensors such as LIDAR and RADAR are used for accurate 3D Object Detection. Despite their significant accuracy, the multi-sensor models are expensive and require a high computational demand. In contrast, Monocular 3D Object Detection models are becoming increasingly popular, offering a faster, cheaper, and easier-to-implement solution for 3D detections. This paper introduces a different Multi-Tasking Learning approach called MonoNext that utilizes a spatial grid to map objects in the scene. MonoNext employs a straightforward approach based on the ConvNext network and requires only 3D bounding box annotated data. In our experiments with the KITTI dataset, MonoNext achieved high precision and competitive performance comparable with state-of-the-art approaches. Furthermore, by adding more training data, MonoNext surpassed itself and achieved higher accuracies.

{{</citation>}}


### (32/88) PVG: Progressive Vision Graph for Vision Recognition (Jiafu Wu et al., 2023)

{{<citation>}}

Jiafu Wu, Jian Li, Jiangning Zhang, Boshen Zhang, Mingmin Chi, Yabiao Wang, Chengjie Wang. (2023)  
**PVG: Progressive Vision Graph for Vision Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GNN, ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2308.00574v1)  

---


**ABSTRACT**  
Convolution-based and Transformer-based vision backbone networks process images into the grid or sequence structures, respectively, which are inflexible for capturing irregular objects. Though Vision GNN (ViG) adopts graph-level features for complex images, it has some issues, such as inaccurate neighbor node selection, expensive node information aggregation calculation, and over-smoothing in the deep layers. To address the above problems, we propose a Progressive Vision Graph (PVG) architecture for vision recognition task. Compared with previous works, PVG contains three main components: 1) Progressively Separated Graph Construction (PSGC) to introduce second-order similarity by gradually increasing the channel of the global graph branch and decreasing the channel of local branch as the layer deepens; 2) Neighbor nodes information aggregation and update module by using Max pooling and mathematical Expectation (MaxE) to aggregate rich neighbor information; 3) Graph error Linear Unit (GraphLU) to enhance low-value information in a relaxed form to reduce the compression of image detail information for alleviating the over-smoothing. Extensive experiments on mainstream benchmarks demonstrate the superiority of PVG over state-of-the-art methods, e.g., our PVG-S obtains 83.0% Top-1 accuracy on ImageNet-1K that surpasses GNN-based ViG-S by +0.9 with the parameters reduced by 18.5%, while the largest PVG-B obtains 84.2% that has +0.5 improvement than ViG-B. Furthermore, our PVG-S obtains +1.3 box AP and +0.4 mask AP gains than ViG-S on COCO dataset.

{{</citation>}}


### (33/88) Detecting Cloud Presence in Satellite Images Using the RGB-based CLIP Vision-Language Model (Mikolaj Czerkawski et al., 2023)

{{<citation>}}

Mikolaj Czerkawski, Robert Atkinson, Christos Tachtatzis. (2023)  
**Detecting Cloud Presence in Satellite Images Using the RGB-based CLIP Vision-Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.00541v1)  

---


**ABSTRACT**  
This work explores capabilities of the pre-trained CLIP vision-language model to identify satellite images affected by clouds. Several approaches to using the model to perform cloud presence detection are proposed and evaluated, including a purely zero-shot operation with text prompts and several fine-tuning approaches. Furthermore, the transferability of the methods across different datasets and sensor types (Sentinel-2 and Landsat-8) is tested. The results that CLIP can achieve non-trivial performance on the cloud presence detection task with apparent capability to generalise across sensing modalities and sensing bands. It is also found that a low-cost fine-tuning stage leads to a strong increase in true negative rate. The results demonstrate that the representations learned by the CLIP model can be useful for satellite image processing tasks involving clouds.

{{</citation>}}


### (34/88) NormKD: Normalized Logits for Knowledge Distillation (Zhihao Chi et al., 2023)

{{<citation>}}

Zhihao Chi, Tu Zheng, Hengjia Li, Zheng Yang, Boxi Wu, Binbin Lin, Deng Cai. (2023)  
**NormKD: Normalized Logits for Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.00520v1)  

---


**ABSTRACT**  
Logit based knowledge distillation gets less attention in recent years since feature based methods perform better in most cases. Nevertheless, we find it still has untapped potential when we re-investigate the temperature, which is a crucial hyper-parameter to soften the logit outputs. For most of the previous works, it was set as a fixed value for the entire distillation procedure. However, as the logits from different samples are distributed quite variously, it is not feasible to soften all of them to an equal degree by just a single temperature, which may make the previous work transfer the knowledge of each sample inadequately. In this paper, we restudy the hyper-parameter temperature and figure out its incapability to distill the knowledge from each sample sufficiently when it is a single value. To address this issue, we propose Normalized Knowledge Distillation (NormKD), with the purpose of customizing the temperature for each sample according to the characteristic of the sample's logit distribution. Compared to the vanilla KD, NormKD barely has extra computation or storage cost but performs significantly better on CIRAR-100 and ImageNet for image classification. Furthermore, NormKD can be easily applied to the other logit based methods and achieve better performance which can be closer to or even better than the feature based method.

{{</citation>}}


### (35/88) Relational Contrastive Learning for Scene Text Recognition (Jinglei Zhang et al., 2023)

{{<citation>}}

Jinglei Zhang, Tiancheng Lin, Yi Xu, Kai Chen, Rui Zhang. (2023)  
**Relational Contrastive Learning for Scene Text Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.00508v1)  

---


**ABSTRACT**  
Context-aware methods achieved great success in supervised scene text recognition via incorporating semantic priors from words. We argue that such prior contextual information can be interpreted as the relations of textual primitives due to the heterogeneous text and background, which can provide effective self-supervised labels for representation learning. However, textual relations are restricted to the finite size of dataset due to lexical dependencies, which causes the problem of over-fitting and compromises representation robustness. To this end, we propose to enrich the textual relations via rearrangement, hierarchy and interaction, and design a unified framework called RCLSTR: Relational Contrastive Learning for Scene Text Recognition. Based on causality, we theoretically explain that three modules suppress the bias caused by the contextual prior and thus guarantee representation robustness. Experiments on representation quality show that our method outperforms state-of-the-art self-supervised STR methods. Code is available at https://github.com/ThunderVVV/RCLSTR.

{{</citation>}}


### (36/88) ViT2EEG: Leveraging Hybrid Pretrained Vision Transformers for EEG Data (Ruiqi Yang et al., 2023)

{{<citation>}}

Ruiqi Yang, Eric Modesitt. (2023)  
**ViT2EEG: Leveraging Hybrid Pretrained Vision Transformers for EEG Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-SP  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.00454v1)  

---


**ABSTRACT**  
In this study, we demonstrate the application of a hybrid Vision Transformer (ViT) model, pretrained on ImageNet, on an electroencephalogram (EEG) regression task. Despite being originally trained for image classification tasks, when fine-tuned on EEG data, this model shows a notable increase in performance compared to other models, including an identical architecture ViT trained without the ImageNet weights. This discovery challenges the traditional understanding of model generalization, suggesting that Transformer models pretrained on seemingly unrelated image data can provide valuable priors for EEG regression tasks with an appropriate fine-tuning pipeline.   The success of this approach suggests that the features extracted by ViT models in the context of visual tasks can be readily transformed for the purpose of EEG predictive modeling. We recommend utilizing this methodology not only in neuroscience and related fields, but generally for any task where data collection is limited by practical, financial, or ethical constraints. Our results illuminate the potential of pretrained models on tasks that are clearly distinct from their original purpose.

{{</citation>}}


### (37/88) FLatten Transformer: Vision Transformer using Focused Linear Attention (Dongchen Han et al., 2023)

{{<citation>}}

Dongchen Han, Xuran Pan, Yizeng Han, Shiji Song, Gao Huang. (2023)  
**FLatten Transformer: Vision Transformer using Focused Linear Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.00442v1)  

---


**ABSTRACT**  
The quadratic computation complexity of self-attention has been a persistent challenge when applying Transformer models to vision tasks. Linear attention, on the other hand, offers a much more efficient alternative with its linear complexity by approximating the Softmax operation through carefully designed mapping functions. However, current linear attention approaches either suffer from significant performance degradation or introduce additional computation overhead from the mapping functions. In this paper, we propose a novel Focused Linear Attention module to achieve both high efficiency and expressiveness. Specifically, we first analyze the factors contributing to the performance degradation of linear attention from two perspectives: the focus ability and feature diversity. To overcome these limitations, we introduce a simple yet effective mapping function and an efficient rank restoration module to enhance the expressiveness of self-attention while maintaining low computation complexity. Extensive experiments show that our linear attention module is applicable to a variety of advanced vision Transformers, and achieves consistently improved performances on multiple benchmarks. Code is available at https://github.com/LeapLabTHU/FLatten-Transformer.

{{</citation>}}


### (38/88) Patch-wise Auto-Encoder for Visual Anomaly Detection (Yajie Cui et al., 2023)

{{<citation>}}

Yajie Cui, Zhaoxiang Liu, Shiguo Lian. (2023)  
**Patch-wise Auto-Encoder for Visual Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.00429v1)  

---


**ABSTRACT**  
Anomaly detection without priors of the anomalies is challenging. In the field of unsupervised anomaly detection, traditional auto-encoder (AE) tends to fail based on the assumption that by training only on normal images, the model will not be able to reconstruct abnormal images correctly. On the contrary, we propose a novel patch-wise auto-encoder (Patch AE) framework, which aims at enhancing the reconstruction ability of AE to anomalies instead of weakening it. Each patch of image is reconstructed by corresponding spatially distributed feature vector of the learned feature representation, i.e., patch-wise reconstruction, which ensures anomaly-sensitivity of AE. Our method is simple and efficient. It advances the state-of-the-art performances on Mvtec AD benchmark, which proves the effectiveness of our model. It shows great potential in practical industrial application scenarios.

{{</citation>}}


### (39/88) Deep Image Harmonization with Learnable Augmentation (Li Niu et al., 2023)

{{<citation>}}

Li Niu, Junyan Cao, Wenyan Cong, Liqing Zhang. (2023)  
**Deep Image Harmonization with Learnable Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.00376v1)  

---


**ABSTRACT**  
The goal of image harmonization is adjusting the foreground appearance in a composite image to make the whole image harmonious. To construct paired training images, existing datasets adopt different ways to adjust the illumination statistics of foregrounds of real images to produce synthetic composite images. However, different datasets have considerable domain gap and the performances on small-scale datasets are limited by insufficient training data. In this work, we explore learnable augmentation to enrich the illumination diversity of small-scale datasets for better harmonization performance. In particular, our designed SYthetic COmposite Network (SycoNet) takes in a real image with foreground mask and a random vector to learn suitable color transformation, which is applied to the foreground of this real image to produce a synthetic composite image. Comprehensive experiments demonstrate the effectiveness of our proposed learnable augmentation for image harmonization. The code of SycoNet is released at https://github.com/bcmi/SycoNet-Adaptive-Image-Harmonization.

{{</citation>}}


### (40/88) Zero-Shot Learning by Harnessing Adversarial Samples (Zhi Chen et al., 2023)

{{<citation>}}

Zhi Chen, Pengfei Zhang, Jingjing Li, Sen Wang, Zi Huang. (2023)  
**Zero-Shot Learning by Harnessing Adversarial Samples**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.00313v1)  

---


**ABSTRACT**  
Zero-Shot Learning (ZSL) aims to recognize unseen classes by generalizing the knowledge, i.e., visual and semantic relationships, obtained from seen classes, where image augmentation techniques are commonly applied to improve the generalization ability of a model. However, this approach can also cause adverse effects on ZSL since the conventional augmentation techniques that solely depend on single-label supervision is not able to maintain semantic information and result in the semantic distortion issue consequently. In other words, image argumentation may falsify the semantic (e.g., attribute) information of an image. To take the advantage of image augmentations while mitigating the semantic distortion issue, we propose a novel ZSL approach by Harnessing Adversarial Samples (HAS). HAS advances ZSL through adversarial training which takes into account three crucial aspects: (1) robust generation by enforcing augmentations to be similar to negative classes, while maintaining correct labels, (2) reliable generation by introducing a latent space constraint to avert significant deviations from the original data manifold, and (3) diverse generation by incorporating attribute-based perturbation by adjusting images according to each semantic attribute's localization. Through comprehensive experiments on three prominent zero-shot benchmark datasets, we demonstrate the effectiveness of our adversarial samples approach in both ZSL and Generalized Zero-Shot Learning (GZSL) scenarios. Our source code is available at https://github.com/uqzhichen/HASZSL.

{{</citation>}}


### (41/88) Diffusion Model for Camouflaged Object Detection (Zhennan Chen et al., 2023)

{{<citation>}}

Zhennan Chen, Rongrong Gao, Tian-Zhu Xiang, Fan Lin. (2023)  
**Diffusion Model for Camouflaged Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.00303v1)  

---


**ABSTRACT**  
Camouflaged object detection is a challenging task that aims to identify objects that are highly similar to their background. Due to the powerful noise-to-image denoising capability of denoising diffusion models, in this paper, we propose a diffusion-based framework for camouflaged object detection, termed diffCOD, a new framework that considers the camouflaged object segmentation task as a denoising diffusion process from noisy masks to object masks. Specifically, the object mask diffuses from the ground-truth masks to a random distribution, and the designed model learns to reverse this noising process. To strengthen the denoising learning, the input image prior is encoded and integrated into the denoising diffusion model to guide the diffusion process. Furthermore, we design an injection attention module (IAM) to interact conditional semantic features extracted from the image with the diffusion noise embedding via the cross-attention mechanism to enhance denoising learning. Extensive experiments on four widely used COD benchmark datasets demonstrate that the proposed method achieves favorable performance compared to the existing 11 state-of-the-art methods, especially in the detailed texture segmentation of camouflaged objects. Our code will be made publicly available at: https://github.com/ZNan-Chen/diffCOD.

{{</citation>}}


### (42/88) Making the V in Text-VQA Matter (Shamanthak Hegde et al., 2023)

{{<citation>}}

Shamanthak Hegde, Soumya Jahagirdar, Shankar Gangisetty. (2023)  
**Making the V in Text-VQA Matter**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: OCR, QA  
[Paper Link](http://arxiv.org/abs/2308.00295v1)  

---


**ABSTRACT**  
Text-based VQA aims at answering questions by reading the text present in the images. It requires a large amount of scene-text relationship understanding compared to the VQA task. Recent studies have shown that the question-answer pairs in the dataset are more focused on the text present in the image but less importance is given to visual features and some questions do not require understanding the image. The models trained on this dataset predict biased answers due to the lack of understanding of visual context. For example, in questions like "What is written on the signboard?", the answer predicted by the model is always "STOP" which makes the model to ignore the image. To address these issues, we propose a method to learn visual features (making V matter in TextVQA) along with the OCR features and question features using VQA dataset as external knowledge for Text-based VQA. Specifically, we combine the TextVQA dataset and VQA dataset and train the model on this combined dataset. Such a simple, yet effective approach increases the understanding and correlation between the image features and text present in the image, which helps in the better answering of questions. We further test the model on different datasets and compare their qualitative and quantitative results.

{{</citation>}}


### (43/88) A Study of Unsupervised Evaluation Metrics for Practical and Automatic Domain Adaptation (Minghao Chen et al., 2023)

{{<citation>}}

Minghao Chen, Zepeng Gao, Shuai Zhao, Qibo Qiu, Wenxiao Wang, Binbin Lin, Xiaofei He. (2023)  
**A Study of Unsupervised Evaluation Metrics for Practical and Automatic Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.00287v1)  

---


**ABSTRACT**  
Unsupervised domain adaptation (UDA) methods facilitate the transfer of models to target domains without labels. However, these methods necessitate a labeled target validation set for hyper-parameter tuning and model selection. In this paper, we aim to find an evaluation metric capable of assessing the quality of a transferred model without access to target validation labels. We begin with the metric based on mutual information of the model prediction. Through empirical analysis, we identify three prevalent issues with this metric: 1) It does not account for the source structure. 2) It can be easily attacked. 3) It fails to detect negative transfer caused by the over-alignment of source and target features. To address the first two issues, we incorporate source accuracy into the metric and employ a new MLP classifier that is held out during training, significantly improving the result. To tackle the final issue, we integrate this enhanced metric with data augmentation, resulting in a novel unsupervised UDA metric called the Augmentation Consistency Metric (ACM). Additionally, we empirically demonstrate the shortcomings of previous experiment settings and conduct large-scale experiments to validate the effectiveness of our proposed metric. Furthermore, we employ our metric to automatically search for the optimal hyper-parameter set, achieving superior performance compared to manually tuned sets across four common benchmarks. Codes will be available soon.

{{</citation>}}


### (44/88) Benchmarking Ultra-High-Definition Image Reflection Removal (Zhenyuan Zhang et al., 2023)

{{<citation>}}

Zhenyuan Zhang, Zhenbo Song, Kaihao Zhang, Wenhan Luo, Zhaoxin Fan, Jianfeng Lu. (2023)  
**Benchmarking Ultra-High-Definition Image Reflection Removal**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.00265v1)  

---


**ABSTRACT**  
Deep learning based methods have achieved significant success in the task of single image reflection removal (SIRR). However, the majority of these methods are focused on High-Definition/Standard-Definition (HD/SD) images, while ignoring higher resolution images such as Ultra-High-Definition (UHD) images. With the increasing prevalence of UHD images captured by modern devices, in this paper, we aim to address the problem of UHD SIRR. Specifically, we first synthesize two large-scale UHD datasets, UHDRR4K and UHDRR8K. The UHDRR4K dataset consists of $2,999$ and $168$ quadruplets of images for training and testing respectively, and the UHDRR8K dataset contains $1,014$ and $105$ quadruplets. To the best of our knowledge, these two datasets are the first largest-scale UHD datasets for SIRR. Then, we conduct a comprehensive evaluation of six state-of-the-art SIRR methods using the proposed datasets. Based on the results, we provide detailed discussions regarding the strengths and limitations of these methods when applied to UHD images. Finally, we present a transformer-based architecture named RRFormer for reflection removal. RRFormer comprises three modules, namely the Prepossessing Embedding Module, Self-attention Feature Extraction Module, and Multi-scale Spatial Feature Extraction Module. These modules extract hypercolumn features, global and partial attention features, and multi-scale spatial features, respectively. To ensure effective training, we utilize three terms in our loss function: pixel loss, feature loss, and adversarial loss. We demonstrate through experimental results that RRFormer achieves state-of-the-art performance on both the non-UHD dataset and our proposed UHDRR datasets. The code and datasets are publicly available at https://github.com/Liar-zzy/Benchmarking-Ultra-High-Definition-Single-Image-Reflection-Removal.

{{</citation>}}


### (45/88) Improving Pixel-based MIM by Reducing Wasted Modeling Capability (Yuan Liu et al., 2023)

{{<citation>}}

Yuan Liu, Songyang Zhang, Jiacheng Chen, Zhaohui Yu, Kai Chen, Dahua Lin. (2023)  
**Improving Pixel-based MIM by Reducing Wasted Modeling Capability**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.00261v1)  

---


**ABSTRACT**  
There has been significant progress in Masked Image Modeling (MIM). Existing MIM methods can be broadly categorized into two groups based on the reconstruction target: pixel-based and tokenizer-based approaches. The former offers a simpler pipeline and lower computational cost, but it is known to be biased toward high-frequency details. In this paper, we provide a set of empirical studies to confirm this limitation of pixel-based MIM and propose a new method that explicitly utilizes low-level features from shallow layers to aid pixel reconstruction. By incorporating this design into our base method, MAE, we reduce the wasted modeling capability of pixel-based MIM, improving its convergence and achieving non-trivial improvements across various downstream tasks. To the best of our knowledge, we are the first to systematically investigate multi-level feature fusion for isotropic architectures like the standard Vision Transformer (ViT). Notably, when applied to a smaller model (e.g., ViT-S), our method yields significant performance gains, such as 1.2\% on fine-tuning, 2.8\% on linear probing, and 2.6\% on semantic segmentation. Code and models are available at https://github.com/open-mmlab/mmpretrain.

{{</citation>}}


### (46/88) LGViT: Dynamic Early Exiting for Accelerating Vision Transformer (Guanyu Xu et al., 2023)

{{<citation>}}

Guanyu Xu, Jiawei Hao, Li Shen, Han Hu, Yong Luo, Hui Lin, Jialie Shen. (2023)  
**LGViT: Dynamic Early Exiting for Accelerating Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2308.00255v1)  

---


**ABSTRACT**  
Recently, the efficient deployment and acceleration of powerful vision transformers (ViTs) on resource-limited edge devices for providing multimedia services have become attractive tasks. Although early exiting is a feasible solution for accelerating inference, most works focus on convolutional neural networks (CNNs) and transformer models in natural language processing (NLP).Moreover, the direct application of early exiting methods to ViTs may result in substantial performance degradation. To tackle this challenge, we systematically investigate the efficacy of early exiting in ViTs and point out that the insufficient feature representations in shallow internal classifiers and the limited ability to capture target semantic information in deep internal classifiers restrict the performance of these methods. We then propose an early exiting framework for general ViTs termed LGViT, which incorporates heterogeneous exiting heads, namely, local perception head and global aggregation head, to achieve an efficiency-accuracy trade-off. In particular, we develop a novel two-stage training scheme, including end-to-end training and self-distillation with the backbone frozen to generate early exiting ViTs, which facilitates the fusion of global and local information extracted by the two types of heads. We conduct extensive experiments using three popular ViT backbones on three vision datasets. Results demonstrate that our LGViT can achieve competitive performance with approximately 1.8 $\times$ speed-up.

{{</citation>}}


### (47/88) Partitioned Saliency Ranking with Dense Pyramid Transformers (Chengxiao Sun et al., 2023)

{{<citation>}}

Chengxiao Sun, Yan Xu, Jialun Pei, Haopeng Fang, He Tang. (2023)  
**Partitioned Saliency Ranking with Dense Pyramid Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.00236v1)  

---


**ABSTRACT**  
In recent years, saliency ranking has emerged as a challenging task focusing on assessing the degree of saliency at instance-level. Being subjective, even humans struggle to identify the precise order of all salient instances. Previous approaches undertake the saliency ranking by directly sorting the rank scores of salient instances, which have not explicitly resolved the inherent ambiguities. To overcome this limitation, we propose the ranking by partition paradigm, which segments unordered salient instances into partitions and then ranks them based on the correlations among these partitions. The ranking by partition paradigm alleviates ranking ambiguities in a general sense, as it consistently improves the performance of other saliency ranking models. Additionally, we introduce the Dense Pyramid Transformer (DPT) to enable global cross-scale interactions, which significantly enhances feature interactions with reduced computational burden. Extensive experiments demonstrate that our approach outperforms all existing methods. The code for our method is available at \url{https://github.com/ssecv/PSR}.

{{</citation>}}


### (48/88) Using Scene and Semantic Features for Multi-modal Emotion Recognition (Zhifeng Wang et al., 2023)

{{<citation>}}

Zhifeng Wang, Ramesh Sankaranarayana. (2023)  
**Using Scene and Semantic Features for Multi-modal Emotion Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.00228v1)  

---


**ABSTRACT**  
Automatic emotion recognition is a hot topic with a wide range of applications. Much work has been done in the area of automatic emotion recognition in recent years. The focus has been mainly on using the characteristics of a person such as speech, facial expression and pose for this purpose. However, the processing of scene and semantic features for emotion recognition has had limited exploration. In this paper, we propose to use combined scene and semantic features, along with personal features, for multi-modal emotion recognition. Scene features will describe the environment or context in which the target person is operating. The semantic feature can include objects that are present in the environment, as well as their attributes and relationships with the target person. In addition, we use a modified EmbraceNet to extract features from the images, which is trained to learn both the body and pose features simultaneously. By fusing both body and pose features, the EmbraceNet can improve the accuracy and robustness of the model, particularly when dealing with partially missing data. This is because having both body and pose features provides a more complete representation of the subject in the images, which can help the model to make more accurate predictions even when some parts of body are missing. We demonstrate the efficiency of our method on the benchmark EMOTIC dataset. We report an average precision of 40.39\% across the 26 emotion categories, which is a 5\% improvement over previous approaches.

{{</citation>}}


## cs.HC (3)



### (49/88) Designing a Communication Bridge between Communities: Participatory Design for a Question-Answering AI Agent (Jeonghyun Lee et al., 2023)

{{<citation>}}

Jeonghyun Lee, Vrinda Nandan, Harshvardhan Sikka, Spencer Rugaber, Ashok Goal. (2023)  
**Designing a Communication Bridge between Communities: Participatory Design for a Question-Answering AI Agent**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00813v1)  

---


**ABSTRACT**  
How do we design an AI system that is intended to act as a communication bridge between two user communities with different mental models and vocabularies? Skillsync is an interactive environment that engages employers (companies) and training providers (colleges) in a sustained dialogue to help them achieve the goal of building a training proposal that successfully meets the needs of the employers and employees. We used a variation of participatory design to elicit requirements for developing AskJill, a question-answering agent that explains how Skillsync works and thus acts as a communication bridge between company and college users. Our study finds that participatory design was useful in guiding the requirements gathering and eliciting user questions for the development of AskJill. Our results also suggest that the two Skillsync user communities perceived glossary assistance as a key feature that AskJill needs to offer, and they would benefit from such a shared vocabulary.

{{</citation>}}


### (50/88) TimePool: Visually Answer 'Which and When' Questions On Univariate Time Series (Tinghao Feng et al., 2023)

{{<citation>}}

Tinghao Feng, Yueqi Hu, Jing Yang, Tom Polk, Ye Zhao, Shixia Liu, Zhaocong Yang. (2023)  
**TimePool: Visually Answer 'Which and When' Questions On Univariate Time Series**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-IR, cs.HC  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.00682v1)  

---


**ABSTRACT**  
When exploring time series datasets, analysts often pose "which and when" questions. For example, with world life expectancy data over one hundred years, they may inquire about the top 10 countries in life expectancy and the time period when they achieved this status, or which countries have had longer life expectancy than Ireland and when. This paper proposes TimePool, a new visualization prototype, to address this need for univariate time series analysis. It allows users to construct interactive "which and when" queries and visually explore the results for insights.

{{</citation>}}


### (51/88) Experiments on Generative AI-Powered Parametric Modeling and BIM for Architectural Design (Jaechang Ko et al., 2023)

{{<citation>}}

Jaechang Ko, John Ajibefun, Wei Yan. (2023)  
**Experiments on Generative AI-Powered Parametric Modeling and BIM for Architectural Design**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.00227v1)  

---


**ABSTRACT**  
This paper introduces a new architectural design framework that utilizes generative AI tools including ChatGPT and Veras with parametric modeling and Building Information Modeling (BIM) to enhance the design process. The study experiments with the potential of ChatGPT and generative AI in 3D architectural design, extending beyond its use in text and 2D image generation. The proposed framework promotes collaboration between architects and AI, facilitating a quick exploration of design ideas and producing context-sensitive, creative design generation. By integrating ChatGPT for scripting and Veras for generating design ideas with widely used parametric modeling and BIM tools, the framework provides architects with an intuitive and powerful method to convey design intent, leading to more efficient, creative, and collaborative design processes.

{{</citation>}}


## cs.NE (2)



### (52/88) Evaluating Spiking Neural Network On Neuromorphic Platform For Human Activity Recognition (Sizhen Bian et al., 2023)

{{<citation>}}

Sizhen Bian, Michele Magno. (2023)  
**Evaluating Spiking Neural Network On Neuromorphic Platform For Human Activity Recognition**  

---
Primary Category: cs.NE  
Categories: cs-HC, cs-LG, cs-NE, cs.NE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00787v1)  

---


**ABSTRACT**  
Energy efficiency and low latency are crucial requirements for designing wearable AI-empowered human activity recognition systems, due to the hard constraints of battery operations and closed-loop feedback. While neural network models have been extensively compressed to match the stringent edge requirements, spiking neural networks and event-based sensing are recently emerging as promising solutions to further improve performance due to their inherent energy efficiency and capacity to process spatiotemporal data in very low latency. This work aims to evaluate the effectiveness of spiking neural networks on neuromorphic processors in human activity recognition for wearable applications. The case of workout recognition with wrist-worn wearable motion sensors is used as a study. A multi-threshold delta modulation approach is utilized for encoding the input sensor data into spike trains to move the pipeline into the event-based approach. The spikes trains are then fed to a spiking neural network with direct-event training, and the trained model is deployed on the research neuromorphic platform from Intel, Loihi, to evaluate energy and latency efficiency. Test results show that the spike-based workouts recognition system can achieve a comparable accuracy (87.5\%) comparable to the popular milliwatt RISC-V bases multi-core processor GAP8 with a traditional neural network ( 88.1\%) while achieving two times better energy-delay product (0.66 \si{\micro\joule\second} vs. 1.32 \si{\micro\joule\second}).

{{</citation>}}


### (53/88) BiERL: A Meta Evolutionary Reinforcement Learning Framework via Bilevel Optimization (Junyi Wang et al., 2023)

{{<citation>}}

Junyi Wang, Yuanyang Zhu, Zhi Wang, Yan Zheng, Jianye Hao, Chunlin Chen. (2023)  
**BiERL: A Meta Evolutionary Reinforcement Learning Framework via Bilevel Optimization**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-LG, cs-NE, cs.NE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01207v1)  

---


**ABSTRACT**  
Evolutionary reinforcement learning (ERL) algorithms recently raise attention in tackling complex reinforcement learning (RL) problems due to high parallelism, while they are prone to insufficient exploration or model collapse without carefully tuning hyperparameters (aka meta-parameters). In the paper, we propose a general meta ERL framework via bilevel optimization (BiERL) to jointly update hyperparameters in parallel to training the ERL model within a single agent, which relieves the need for prior domain knowledge or costly optimization procedure before model deployment. We design an elegant meta-level architecture that embeds the inner-level's evolving experience into an informative population representation and introduce a simple and feasible evaluation of the meta-level fitness function to facilitate learning efficiency. We perform extensive experiments in MuJoCo and Box2D tasks to verify that as a general framework, BiERL outperforms various baselines and consistently improves the learning performance for a diversity of ERL algorithms.

{{</citation>}}


## cs.RO (4)



### (54/88) An ensemble of online estimation methods for one degree-of-freedom models of unmanned surface vehicles: applied theory and preliminary field results with eight vehicles (Tyler M. Paine et al., 2023)

{{<citation>}}

Tyler M. Paine, Michael R. Benjamin. (2023)  
**An ensemble of online estimation methods for one degree-of-freedom models of unmanned surface vehicles: applied theory and preliminary field results with eight vehicles**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00782v1)  

---


**ABSTRACT**  
In this paper we report an experimental evaluation of three popular methods for online system identification of unmanned surface vehicles (USVs) which were implemented as an ensemble: certifiably stable shallow recurrent neural network (RNN), adaptive identification (AID), and recursive least squares (RLS). The algorithms were deployed on eight USVs for a total of 30 hours of online estimation. During online training the loss function for the RNN was augmented to include a cost for violating a sufficient condition for the RNN to be stable in the sense of contraction stability. Additionally we described an efficient method to calculate the equilibrium points of the RNN and classify the associated stability properties about these points. We found the AID method had lowest mean absolute error in the online prediction setting, but a weighted ensemble had lower error in offline processing.

{{</citation>}}


### (55/88) Sliding Touch-based Exploration for Modeling Unknown Object Shape with Multi-fingered Hands (Yiting Chen et al., 2023)

{{<citation>}}

Yiting Chen, Ahmet Ercan Tekden, Marc Peter Deisenroth, Yasemin Bekiroglu. (2023)  
**Sliding Touch-based Exploration for Modeling Unknown Object Shape with Multi-fingered Hands**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2308.00576v1)  

---


**ABSTRACT**  
Efficient and accurate 3D object shape reconstruction contributes significantly to the success of a robot's physical interaction with its environment. Acquiring accurate shape information about unknown objects is challenging, especially in unstructured environments, e.g. the vision sensors may only be able to provide a partial view. To address this issue, tactile sensors could be employed to extract local surface information for more robust unknown object shape estimation. In this paper, we propose a novel approach for efficient unknown 3D object shape exploration and reconstruction using a multi-fingered hand equipped with tactile sensors and a depth camera only providing a partial view. We present a multi-finger sliding touch strategy for efficient shape exploration using a Bayesian Optimization approach and a single-leader-multi-follower strategy for multi-finger smooth local surface perception. We evaluate our proposed method by estimating the 3D shape of objects from the YCB and OCRTOC datasets based on simulation and real robot experiments. The proposed approach yields successful reconstruction results relying on only a few continuous sliding touches. Experimental results demonstrate that our method is able to model unknown objects in an efficient and accurate way.

{{</citation>}}


### (56/88) UVIO: An UWB-Aided Visual-Inertial Odometry Framework with Bias-Compensated Anchors Initialization (Giulio Delama et al., 2023)

{{<citation>}}

Giulio Delama, Farhad Shamsfakhr, Stephan Weiss, Daniele Fontanelli, Alessandro Fornasier. (2023)  
**UVIO: An UWB-Aided Visual-Inertial Odometry Framework with Bias-Compensated Anchors Initialization**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.00513v1)  

---


**ABSTRACT**  
This paper introduces UVIO, a multi-sensor framework that leverages Ultra Wide Band (UWB) technology and Visual-Inertial Odometry (VIO) to provide robust and low-drift localization. In order to include range measurements in state estimation, the position of the UWB anchors must be known. This study proposes a multi-step initialization procedure to map multiple unknown anchors by an Unmanned Aerial Vehicle (UAV), in a fully autonomous fashion. To address the limitations of initializing UWB anchors via a random trajectory, this paper uses the Geometric Dilution of Precision (GDOP) as a measure of optimality in anchor position estimation, to compute a set of optimal waypoints and synthesize a trajectory that minimizes the mapping uncertainty. After the initialization is complete, the range measurements from multiple anchors, including measurement biases, are tightly integrated into the VIO system. While in range of the initialized anchors, the VIO drift in position and heading is eliminated. The effectiveness of UVIO and our initialization procedure has been validated through a series of simulations and real-world experiments.

{{</citation>}}


### (57/88) Target Search and Navigation in Heterogeneous Robot Systems with Deep Reinforcement Learning (Yun Chen et al., 2023)

{{<citation>}}

Yun Chen, Jiaping Xiao. (2023)  
**Target Search and Navigation in Heterogeneous Robot Systems with Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.00331v1)  

---


**ABSTRACT**  
Collaborative heterogeneous robot systems can greatly improve the efficiency of target search and navigation tasks. In this paper, we design a heterogeneous robot system consisting of a UAV and a UGV for search and rescue missions in unknown environments. The system is able to search for targets and navigate to them in a maze-like mine environment with the policies learned through deep reinforcement learning algorithms. During the training process, if two robots are trained simultaneously, the rewards related to their collaboration may not be properly obtained. Hence, we introduce a multi-stage reinforcement learning framework and a curiosity module to encourage agents to explore unvisited environments. Experiments in simulation environments show that our framework can train the heterogeneous robot system to achieve the search and navigation with unknown target locations while existing baselines may not, and accelerate the training speed.

{{</citation>}}


## cs.IR (3)



### (58/88) Self-Supervised Contrastive BERT Fine-tuning for Fusion-based Reviewed-Item Retrieval (Mohammad Mahdi Abdollah Pour et al., 2023)

{{<citation>}}

Mohammad Mahdi Abdollah Pour, Parsa Farinneya, Armin Toroghi, Anton Korikov, Ali Pesaranghader, Touqir Sajed, Manasa Bharadwaj, Borislav Mavrin, Scott Sanner. (2023)  
**Self-Supervised Contrastive BERT Fine-tuning for Fusion-based Reviewed-Item Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: BERT, Information Retrieval, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.00762v1)  

---


**ABSTRACT**  
As natural language interfaces enable users to express increasingly complex natural language queries, there is a parallel explosion of user review content that can allow users to better find items such as restaurants, books, or movies that match these expressive queries. While Neural Information Retrieval (IR) methods have provided state-of-the-art results for matching queries to documents, they have not been extended to the task of Reviewed-Item Retrieval (RIR), where query-review scores must be aggregated (or fused) into item-level scores for ranking. In the absence of labeled RIR datasets, we extend Neural IR methodology to RIR by leveraging self-supervised methods for contrastive learning of BERT embeddings for both queries and reviews. Specifically, contrastive learning requires a choice of positive and negative samples, where the unique two-level structure of our item-review data combined with meta-data affords us a rich structure for the selection of these samples. For contrastive learning in a Late Fusion scenario, we investigate the use of positive review samples from the same item and/or with the same rating, selection of hard positive samples by choosing the least similar reviews from the same anchor item, and selection of hard negative samples by choosing the most similar reviews from different items. We also explore anchor sub-sampling and augmenting with meta-data. For a more end-to-end Early Fusion approach, we introduce contrastive item embedding learning to fuse reviews into single item embeddings. Experimental results show that Late Fusion contrastive learning for Neural RIR outperforms all other contrastive IR configurations, Neural IR, and sparse retrieval baselines, thus demonstrating the power of exploiting the two-level structure in Neural RIR approaches as well as the importance of preserving the nuance of individual review content via Late Fusion methods.

{{</citation>}}


### (59/88) Adaptive Collaborative Filtering with Personalized Time Decay Functions for Financial Product Recommendation (Ashraf Ghiye et al., 2023)

{{<citation>}}

Ashraf Ghiye, Baptiste Barreau, Laurent Carlier, Michalis Vazirgiannis. (2023)  
**Adaptive Collaborative Filtering with Personalized Time Decay Functions for Financial Product Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR, q-fin-CP, stat-ML  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2308.01208v1)  

---


**ABSTRACT**  
Classical recommender systems often assume that historical data are stationary and fail to account for the dynamic nature of user preferences, limiting their ability to provide reliable recommendations in time-sensitive settings. This assumption is particularly problematic in finance, where financial products exhibit continuous changes in valuations, leading to frequent shifts in client interests. These evolving interests, summarized in the past client-product interactions, see their utility fade over time with a degree that might differ from one client to another. To address this challenge, we propose a time-dependent collaborative filtering algorithm that can adaptively discount distant client-product interactions using personalized decay functions. Our approach is designed to handle the non-stationarity of financial data and produce reliable recommendations by modeling the dynamic collaborative signals between clients and products. We evaluate our method using a proprietary dataset from BNP Paribas and demonstrate significant improvements over state-of-the-art benchmarks from relevant literature. Our findings emphasize the importance of incorporating time explicitly in the model to enhance the accuracy of financial product recommendation.

{{</citation>}}


### (60/88) Challenging the Myth of Graph Collaborative Filtering: a Reasoned and Reproducibility-driven Analysis (Vito Walter Anelli et al., 2023)

{{<citation>}}

Vito Walter Anelli, Daniele Malitesta, Claudio Pomo, Alejandro Bellogín, Tommaso Di Noia, Eugenio Di Sciascio. (2023)  
**Challenging the Myth of Graph Collaborative Filtering: a Reasoned and Reproducibility-driven Analysis**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Amazon, GNN  
[Paper Link](http://arxiv.org/abs/2308.00404v1)  

---


**ABSTRACT**  
The success of graph neural network-based models (GNNs) has significantly advanced recommender systems by effectively modeling users and items as a bipartite, undirected graph. However, many original graph-based works often adopt results from baseline papers without verifying their validity for the specific configuration under analysis. Our work addresses this issue by focusing on the replicability of results. We present a code that successfully replicates results from six popular and recent graph recommendation models (NGCF, DGCF, LightGCN, SGL, UltraGCN, and GFCF) on three common benchmark datasets (Gowalla, Yelp 2018, and Amazon Book). Additionally, we compare these graph models with traditional collaborative filtering models that historically performed well in offline evaluations. Furthermore, we extend our study to two new datasets (Allrecipes and BookCrossing) that lack established setups in existing literature. As the performance on these datasets differs from the previous benchmarks, we analyze the impact of specific dataset characteristics on recommendation accuracy. By investigating the information flow from users' neighborhoods, we aim to identify which models are influenced by intrinsic features in the dataset structure. The code to reproduce our experiments is available at: https://github.com/sisinflab/Graph-RSs-Reproducibility.

{{</citation>}}


## cs.LG (11)



### (61/88) The Bias Amplification Paradox in Text-to-Image Generation (Preethi Seshadri et al., 2023)

{{<citation>}}

Preethi Seshadri, Sameer Singh, Yanai Elazar. (2023)  
**The Bias Amplification Paradox in Text-to-Image Generation**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-CY, cs-LG, cs.LG  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2308.00755v1)  

---


**ABSTRACT**  
Bias amplification is a phenomenon in which models increase imbalances present in the training data. In this paper, we study bias amplification in the text-to-image domain using Stable Diffusion by comparing gender ratios in training vs. generated images. We find that the model appears to amplify gender-occupation biases found in the training data (LAION). However, we discover that amplification can largely be attributed to discrepancies between training captions and model prompts. For example, an inherent difference is that captions from the training data often contain explicit gender information while the prompts we use do not, which leads to a distribution shift and consequently impacts bias measures. Once we account for various distributional differences between texts used for training and generation, we observe that amplification decreases considerably. Our findings illustrate the challenges of comparing biases in models and the data they are trained on, and highlight confounding factors that contribute to bias amplification.

{{</citation>}}


### (62/88) CodeBPE: Investigating Subtokenization Options for Large Language Model Pretraining on Source Code (Nadezhda Chirkova et al., 2023)

{{<citation>}}

Nadezhda Chirkova, Sergey Troshin. (2023)  
**CodeBPE: Investigating Subtokenization Options for Large Language Model Pretraining on Source Code**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs-SE, cs.LG  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2308.00683v1)  

---


**ABSTRACT**  
Recent works have widely adopted large language model pretraining for source code, suggested source code-specific pretraining objectives and investigated the applicability of various Transformer-based language model architectures for source code. This work investigates another important aspect of such models, namely the effect of different subtokenization options, and aims at identifying most effective and length-efficient subtokenizations, taking into account code specifics. We propose subtokenziation that reduces average length by 17% without downstream performance drop, and show that a carefully chosen subtokenization may improve quality by 0.5-2%, possibly with some length increase.

{{</citation>}}


### (63/88) Graph Contrastive Learning with Generative Adversarial Network (Cheng Wu et al., 2023)

{{<citation>}}

Cheng Wu, Chaokun Wang, Jingcao Xu, Ziyang Liu, Kai Zheng, Xiaowei Wang, Yang Song, Kun Gai. (2023)  
**Graph Contrastive Learning with Generative Adversarial Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.00535v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have demonstrated promising results on exploiting node representations for many downstream tasks through supervised end-to-end training. To deal with the widespread label scarcity issue in real-world applications, Graph Contrastive Learning (GCL) is leveraged to train GNNs with limited or even no labels by maximizing the mutual information between nodes in its augmented views generated from the original graph. However, the distribution of graphs remains unconsidered in view generation, resulting in the ignorance of unseen edges in most existing literature, which is empirically shown to be able to improve GCL's performance in our experiments. To this end, we propose to incorporate graph generative adversarial networks (GANs) to learn the distribution of views for GCL, in order to i) automatically capture the characteristic of graphs for augmentations, and ii) jointly train the graph GAN model and the GCL model. Specifically, we present GACN, a novel Generative Adversarial Contrastive learning Network for graph representation learning. GACN develops a view generator and a view discriminator to generate augmented views automatically in an adversarial style. Then, GACN leverages these views to train a GNN encoder with two carefully designed self-supervised learning losses, including the graph contrastive loss and the Bayesian personalized ranking Loss. Furthermore, we design an optimization framework to train all GACN modules jointly. Extensive experiments on seven real-world datasets show that GACN is able to generate high-quality augmented views for GCL and is superior to twelve state-of-the-art baseline methods. Noticeably, our proposed GACN surprisingly discovers that the generated views in data augmentation finally conform to the well-known preferential attachment rule in online networks.

{{</citation>}}


### (64/88) A Novel Temporal Multi-Gate Mixture-of-Experts Approach for Vehicle Trajectory and Driving Intention Prediction (Renteng Yuan et al., 2023)

{{<citation>}}

Renteng Yuan, Mohamed Abdel-Aty, Qiaojun Xiang, Zijin Wang, Ou Zheng. (2023)  
**A Novel Temporal Multi-Gate Mixture-of-Experts Approach for Vehicle Trajectory and Driving Intention Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2308.00533v1)  

---


**ABSTRACT**  
Accurate Vehicle Trajectory Prediction is critical for automated vehicles and advanced driver assistance systems. Vehicle trajectory prediction consists of two essential tasks, i.e., longitudinal position prediction and lateral position prediction. There is a significant correlation between driving intentions and vehicle motion. In existing work, the three tasks are often conducted separately without considering the relationships between the longitudinal position, lateral position, and driving intention. In this paper, we propose a novel Temporal Multi-Gate Mixture-of-Experts (TMMOE) model for simultaneously predicting the vehicle trajectory and driving intention. The proposed model consists of three layers: a shared layer, an expert layer, and a fully connected layer. In the model, the shared layer utilizes Temporal Convolutional Networks (TCN) to extract temporal features. Then the expert layer is built to identify different information according to the three tasks. Moreover, the fully connected layer is used to integrate and export prediction results. To achieve better performance, uncertainty algorithm is used to construct the multi-task loss function. Finally, the publicly available CitySim dataset validates the TMMOE model, demonstrating superior performance compared to the LSTM model, achieving the highest classification and regression results. Keywords: Vehicle trajectory prediction, driving intentions Classification, Multi-task

{{</citation>}}


### (65/88) A Survey of Time Series Anomaly Detection Methods in the AIOps Domain (Zhenyu Zhong et al., 2023)

{{<citation>}}

Zhenyu Zhong, Qiliang Fan, Jiacheng Zhang, Minghua Ma, Shenglin Zhang, Yongqian Sun, Qingwei Lin, Yuzhi Zhang, Dan Pei. (2023)  
**A Survey of Time Series Anomaly Detection Methods in the AIOps Domain**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: AI, Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2308.00393v1)  

---


**ABSTRACT**  
Internet-based services have seen remarkable success, generating vast amounts of monitored key performance indicators (KPIs) as univariate or multivariate time series. Monitoring and analyzing these time series are crucial for researchers, service operators, and on-call engineers to detect outliers or anomalies indicating service failures or significant events. Numerous advanced anomaly detection methods have emerged to address availability and performance issues. This review offers a comprehensive overview of time series anomaly detection in Artificial Intelligence for IT operations (AIOps), which uses AI capabilities to automate and optimize operational workflows. Additionally, it explores future directions for real-world and next-generation time-series anomaly detection based on recent advancements.

{{</citation>}}


### (66/88) Counterfactual Graph Transformer for Traffic Flow Prediction (Ying Yang et al., 2023)

{{<citation>}}

Ying Yang, Kai Du, Xingyuan Dai, Jianwu Fang. (2023)  
**Counterfactual Graph Transformer for Traffic Flow Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.00391v1)  

---


**ABSTRACT**  
Traffic flow prediction (TFP) is a fundamental problem of the Intelligent Transportation System (ITS), as it models the latent spatial-temporal dependency of traffic flow for potential congestion prediction. Recent graph-based models with multiple kinds of attention mechanisms have achieved promising performance. However, existing methods for traffic flow prediction tend to inherit the bias pattern from the dataset and lack interpretability. To this end, we propose a Counterfactual Graph Transformer (CGT) model with an instance-level explainer (e.g., finding the important subgraphs) specifically designed for TFP. We design a perturbation mask generator over input sensor features at the time dimension and the graph structure on the graph transformer module to obtain spatial and temporal counterfactual explanations. By searching the optimal perturbation masks on the input data feature and graph structures, we can obtain the concise and dominant data or graph edge links for the subsequent TFP task. After re-training the utilized graph transformer model after counterfactual perturbation, we can obtain improved and interpretable traffic flow prediction. Extensive results on three real-world public datasets show that CGT can produce reliable explanations and is promising for traffic flow prediction.

{{</citation>}}


### (67/88) Pixel to policy: DQN Encoders for within & cross-game reinforcement learning (Ashrya Agrawal et al., 2023)

{{<citation>}}

Ashrya Agrawal, Priyanshi Shah, Sourabh Prakash. (2023)  
**Pixel to policy: DQN Encoders for within & cross-game reinforcement learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.00318v1)  

---


**ABSTRACT**  
Reinforcement Learning can be applied to various tasks, and environments. Many of these environments have a similar shared structure, which can be exploited to improve RL performance on other tasks. Transfer learning can be used to take advantage of this shared structure, by learning policies that are transferable across different tasks and environments and can lead to more efficient learning as well as improved performance on a wide range of tasks. This work explores as well as compares the performance between RL models being trained from the scratch and on different approaches of transfer learning. Additionally, the study explores the performance of a model trained on multiple game environments, with the goal of developing a universal game-playing agent as well as transfer learning a pre-trained encoder using DQN, and training it on the same game or a different game. Our DQN model achieves a mean episode reward of 46.16 which even beats the human-level performance with merely 20k episodes which is significantly lower than deepmind's 1M episodes. The achieved mean rewards of 533.42 and 402.17 on the Assault and Space Invader environments respectively, represent noteworthy performance on these challenging environments.

{{</citation>}}


### (68/88) Doubly Robust Instance-Reweighted Adversarial Training (Daouda Sow et al., 2023)

{{<citation>}}

Daouda Sow, Sen Lin, Zhangyang Wang, Yingbin Liang. (2023)  
**Doubly Robust Instance-Reweighted Adversarial Training**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2308.00311v1)  

---


**ABSTRACT**  
Assigning importance weights to adversarial data has achieved great success in training adversarially robust networks under limited model capacity. However, existing instance-reweighted adversarial training (AT) methods heavily depend on heuristics and/or geometric interpretations to determine those importance weights, making these algorithms lack rigorous theoretical justification/guarantee. Moreover, recent research has shown that adversarial training suffers from a severe non-uniform robust performance across the training distribution, e.g., data points belonging to some classes can be much more vulnerable to adversarial attacks than others. To address both issues, in this paper, we propose a novel doubly-robust instance reweighted AT framework, which allows to obtain the importance weights via exploring distributionally robust optimization (DRO) techniques, and at the same time boosts the robustness on the most vulnerable examples. In particular, our importance weights are obtained by optimizing the KL-divergence regularized loss function, which allows us to devise new algorithms with a theoretical convergence guarantee. Experiments on standard classification datasets demonstrate that our proposed approach outperforms related state-of-the-art baseline methods in terms of average robust performance, and at the same time improves the robustness against attacks on the weakest data points. Codes will be available soon.

{{</citation>}}


### (69/88) ZADU: A Python Library for Evaluating the Reliability of Dimensionality Reduction Embeddings (Hyeon Jeon et al., 2023)

{{<citation>}}

Hyeon Jeon, Aeri Cho, Jinhwa Jang, Soohyun Lee, Jake Hyun, Hyung-Kwon Ko, Jaemin Jo, Jinwook Seo. (2023)  
**ZADU: A Python Library for Evaluating the Reliability of Dimensionality Reduction Embeddings**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.00282v1)  

---


**ABSTRACT**  
Dimensionality reduction (DR) techniques inherently distort the original structure of input high-dimensional data, producing imperfect low-dimensional embeddings. Diverse distortion measures have thus been proposed to evaluate the reliability of DR embeddings. However, implementing and executing distortion measures in practice has so far been time-consuming and tedious. To address this issue, we present ZADU, a Python library that provides distortion measures. ZADU is not only easy to install and execute but also enables comprehensive evaluation of DR embeddings through three key features. First, the library covers a wide range of distortion measures. Second, it automatically optimizes the execution of distortion measures, substantially reducing the running time required to execute multiple measures. Last, the library informs how individual points contribute to the overall distortions, facilitating the detailed analysis of DR embeddings. By simulating a real-world scenario of optimizing DR embeddings, we verify that our optimization scheme substantially reduces the time required to execute distortion measures. Finally, as an application of ZADU, we present another library called ZADUVis that allows users to easily create distortion visualizations that depict the extent to which each region of an embedding suffers from distortions.

{{</citation>}}


### (70/88) Asynchronous Federated Learning with Bidirectional Quantized Communications and Buffered Aggregation (Tomas Ortega et al., 2023)

{{<citation>}}

Tomas Ortega, Hamid Jafarkhani. (2023)  
**Asynchronous Federated Learning with Bidirectional Quantized Communications and Buffered Aggregation**  

---
Primary Category: cs.LG  
Categories: 68W10, 68W15, 68W40, 90C06, 90C35, 90C26, G-1-6; F-2-1; E-4, cs-LG, cs.LG, eess-SP, math-OC  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.00263v1)  

---


**ABSTRACT**  
Asynchronous Federated Learning with Buffered Aggregation (FedBuff) is a state-of-the-art algorithm known for its efficiency and high scalability. However, it has a high communication cost, which has not been examined with quantized communications. To tackle this problem, we present a new algorithm (QAFeL), with a quantization scheme that establishes a shared "hidden" state between the server and clients to avoid the error propagation caused by direct quantization. This approach allows for high precision while significantly reducing the data transmitted during client-server interactions. We provide theoretical convergence guarantees for QAFeL and corroborate our analysis with experiments on a standard benchmark.

{{</citation>}}


### (71/88) AQUILA: Communication Efficient Federated Learning with Adaptive Quantization of Lazily-Aggregated Gradients (Zihao Zhao et al., 2023)

{{<citation>}}

Zihao Zhao, Yuzhu Mao, Zhenpeng Shi, Yang Liu, Tian Lan, Wenbo Ding, Xiao-Ping Zhang. (2023)  
**AQUILA: Communication Efficient Federated Learning with Adaptive Quantization of Lazily-Aggregated Gradients**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2308.00258v1)  

---


**ABSTRACT**  
The widespread adoption of Federated Learning (FL), a privacy-preserving distributed learning methodology, has been impeded by the challenge of high communication overheads, typically arising from the transmission of large-scale models. Existing adaptive quantization methods, designed to mitigate these overheads, operate under the impractical assumption of uniform device participation in every training round. Additionally, these methods are limited in their adaptability due to the necessity of manual quantization level selection and often overlook biases inherent in local devices' data, thereby affecting the robustness of the global model. In response, this paper introduces AQUILA (adaptive quantization of lazily-aggregated gradients), a novel adaptive framework devised to effectively handle these issues, enhancing the efficiency and robustness of FL. AQUILA integrates a sophisticated device selection method that prioritizes the quality and usefulness of device updates. Utilizing the exact global model stored by devices, it enables a more precise device selection criterion, reduces model deviation, and limits the need for hyperparameter adjustments. Furthermore, AQUILA presents an innovative quantization criterion, optimized to improve communication efficiency while assuring model convergence. Our experiments demonstrate that AQUILA significantly decreases communication costs compared to existing methods, while maintaining comparable model performance across diverse non-homogeneous FL settings, such as Non-IID data and heterogeneous model architectures.

{{</citation>}}


## cs.GT (1)



### (72/88) Game Theoretic Modelling of a Ransom and Extortion Attack on Ethereum Validators (Alpesh Bhudia et al., 2023)

{{<citation>}}

Alpesh Bhudia, Anna Cartwright, Edward Cartwright, Darren Hurley-Smith, Julio Hernandez-Castro. (2023)  
**Game Theoretic Modelling of a Ransom and Extortion Attack on Ethereum Validators**  

---
Primary Category: cs.GT  
Categories: cs-CR, cs-GT, cs.GT  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2308.00590v1)  

---


**ABSTRACT**  
Consensus algorithms facilitate agreement on and resolution of blockchain functions, such as smart contracts and transactions. Ethereum uses a Proof-of-Stake (PoS) consensus mechanism, which depends on financial incentives to ensure that validators perform certain duties and do not act maliciously. Should a validator attempt to defraud the system, legitimate validators will identify this and then staked cryptocurrency is `burned' through a process of slashing.   In this paper, we show that an attacker who has compromised a set of validators could threaten to perform malicious actions that would result in slashing and thus, hold those validators to ransom. We use game theory to study how an attacker can coerce payment from a victim, for example by deploying a smart contract to provide a root of trust shared between attacker and victim during the extortion process. Our game theoretic model finds that it is in the interests of the validators to fully pay the ransom due to a lack of systemic protections for validators. Financial risk is solely placed on the victim during such an attack, with no mitigations available to them aside from capitulation (payment of ransom) in many scenarios. Such attacks could be disruptive to Ethereum and, likely, to many other PoS networks, if public trust in the validator system is eroded. We also discuss and evaluate potential mitigation measures arising from our analysis of the game theoretic model.

{{</citation>}}


## cs.CR (5)



### (73/88) FLAIRS: FPGA-Accelerated Inference-Resistant & Secure Federated Learning (Huimin Li et al., 2023)

{{<citation>}}

Huimin Li, Phillip Rieger, Shaza Zeitouni, Stjepan Picek, Ahmad-Reza Sadeghi. (2023)  
**FLAIRS: FPGA-Accelerated Inference-Resistant & Secure Federated Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00553v1)  

---


**ABSTRACT**  
Federated Learning (FL) has become very popular since it enables clients to train a joint model collaboratively without sharing their private data. However, FL has been shown to be susceptible to backdoor and inference attacks. While in the former, the adversary injects manipulated updates into the aggregation process; the latter leverages clients' local models to deduce their private data. Contemporary solutions to address the security concerns of FL are either impractical for real-world deployment due to high-performance overheads or are tailored towards addressing specific threats, for instance, privacy-preserving aggregation or backdoor defenses. Given these limitations, our research delves into the advantages of harnessing the FPGA-based computing paradigm to overcome performance bottlenecks of software-only solutions while mitigating backdoor and inference attacks. We utilize FPGA-based enclaves to address inference attacks during the aggregation process of FL. We adopt an advanced backdoor-aware aggregation algorithm on the FPGA to counter backdoor attacks. We implemented and evaluated our method on Xilinx VMK-180, yielding a significant speed-up of around 300 times on the IoT-Traffic dataset and more than 506 times on the CIFAR-10 dataset.

{{</citation>}}


### (74/88) SF-IDS: An Imbalanced Semi-Supervised Learning Framework for Fine-grained Intrusion Detection (Xinran Zheng et al., 2023)

{{<citation>}}

Xinran Zheng, Shuo Yang, Xingjun Wang. (2023)  
**SF-IDS: An Imbalanced Semi-Supervised Learning Framework for Fine-grained Intrusion Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.00542v1)  

---


**ABSTRACT**  
Deep learning-based fine-grained network intrusion detection systems (NIDS) enable different attacks to be responded to in a fast and targeted manner with the help of large-scale labels. However, the cost of labeling causes insufficient labeled samples. Also, the real fine-grained traffic shows a long-tailed distribution with great class imbalance. These two problems often appear simultaneously, posing serious challenges to fine-grained NIDS. In this work, we propose a novel semi-supervised fine-grained intrusion detection framework, SF-IDS, to achieve attack classification in the label-limited and highly class imbalanced case. We design a self-training backbone model called RI-1DCNN to boost the feature extraction by reconstructing the input samples into a multichannel image format. The uncertainty of the generated pseudo-labels is evaluated and used as a reference for pseudo-label filtering in combination with the prediction probability. To mitigate the effects of fine-grained class imbalance, we propose a hybrid loss function combining supervised contrastive loss and multi-weighted classification loss to obtain more compact intra-class features and clearer inter-class intervals. Experiments show that the proposed SF-IDS achieves 3.01% and 2.71% Marco-F1 improvement on two classical datasets with 1% labeled, respectively.

{{</citation>}}


### (75/88) A First Look at Digital Rights Management Systems for Secure Mobile Content Delivery (Amir Rafi et al., 2023)

{{<citation>}}

Amir Rafi, Carlton Shepherd, Konstantinos Markantonakis. (2023)  
**A First Look at Digital Rights Management Systems for Secure Mobile Content Delivery**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Amazon, Google, Microsoft  
[Paper Link](http://arxiv.org/abs/2308.00437v2)  

---


**ABSTRACT**  
Digital rights management (DRM) solutions aim to prevent the copying or distribution of copyrighted material. On mobile devices, a variety of DRM technologies have become widely deployed. However, a detailed security study comparing their internal workings, and their strengths and weaknesses, remains missing in the existing literature. In this paper, we present the first detailed security analysis of mobile DRM systems, addressing the modern paradigm of cloud-based content delivery followed by major platforms, such as Netflix, Disney+, and Amazon Prime. We extensively analyse the security of three widely used DRM solutions -- Google Widevine, Apple FairPlay, and Microsoft PlayReady -- deployed on billions of devices worldwide. We then consolidate their features and capabilities, deriving common features and security properties for their evaluation. Furthermore, we identify some design-level shortcomings that render them vulnerable to emerging attacks within the state of the art, including micro-architectural side-channel vulnerabilities and an absence of post-quantum security. Lastly, we propose mitigations and suggest future directions of research.

{{</citation>}}


### (76/88) VulMatch: Binary-level Vulnerability Detection Through Signature (Zian Liu et al., 2023)

{{<citation>}}

Zian Liu, Lei Pan, Chao Chen, Ejaz Ahmed, Shigang Liu, Jun Zhang, Dongxi Liu. (2023)  
**VulMatch: Binary-level Vulnerability Detection Through Signature**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2308.00288v1)  

---


**ABSTRACT**  
Similar vulnerability repeats in real-world software products because of code reuse, especially in wildly reused third-party code and libraries. Detecting repeating vulnerabilities like 1-day and N-day vulnerabilities is an important cyber security task. Unfortunately, the state-of-the-art methods suffer from poor performance because they detect patch existence instead of vulnerability existence and infer the vulnerability signature directly from binary code. In this paper, we propose VulMatch to extract precise vulnerability-related binary instructions to generate the vulnerability-related signature. VulMatch detects vulnerability existence based on binary signatures. Unlike previous approaches, VulMatch accurately locates vulnerability-related instructions by utilizing source and binary codes. Our experiments were conducted using over 1000 vulnerable instances across seven open-source projects. VulMatch significantly outperformed the baseline tools Asm2vec and Palmtree. Besides the performance advantages over the baseline tools, VulMatch offers a better feature by providing explainable reasons during vulnerability detection. Our empirical studies demonstrate that VulMatch detects fine-grained vulnerability that the state-of-the-art tools struggle with. Our experiment on commercial firmware demonstrates VulMatch is able to find vulnerabilities in real-world scenario.

{{</citation>}}


### (77/88) Enhanced Security with Encrypted Vision Transformer in Federated Learning (Rei Aso et al., 2023)

{{<citation>}}

Rei Aso, Sayaka Shiota, Hitoshi Kiya. (2023)  
**Enhanced Security with Encrypted Vision Transformer in Federated Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security, Transformer  
[Paper Link](http://arxiv.org/abs/2308.00271v1)  

---


**ABSTRACT**  
Federated learning is a learning method for training models over multiple participants without directly sharing their raw data, and it has been expected to be a privacy protection method for training data. In contrast, attack methods have been studied to restore learning data from model information shared with clients, so enhanced security against attacks has become an urgent problem. Accordingly, in this article, we propose a novel framework of federated learning on the bases of the embedded structure of the vision transformer by using the model information encrypted with a random sequence. In image classification experiments, we verify the effectiveness of the proposed method on the CIFAR-10 dataset in terms of classification accuracy and robustness against attacks.

{{</citation>}}


## eess.SY (3)



### (78/88) Graph Embedding Dynamic Feature-based Supervised Contrastive Learning of Transient Stability for Changing Power Grid Topologies (Zijian Lv et al., 2023)

{{<citation>}}

Zijian Lv, Xin Chen, Zijian Feng. (2023)  
**Graph Embedding Dynamic Feature-based Supervised Contrastive Learning of Transient Stability for Changing Power Grid Topologies**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Contrastive Learning, Embedding  
[Paper Link](http://arxiv.org/abs/2308.00537v1)  

---


**ABSTRACT**  
Accurate online transient stability prediction is critical for ensuring power system stability when facing disturbances. While traditional transient stablity analysis replies on the time domain simulations can not be quickly adapted to the power grid toplogy change. In order to vectorize high-dimensional power grid topological structure information into low-dimensional node-based graph embedding streaming data, graph embedding dynamic feature (GEDF) has been proposed. The transient stability GEDF-based supervised contrastive learning (GEDF-SCL) model uses supervised contrastive learning to predict transient stability with GEDFs, considering power grid topology information. To evaluate the performance of the proposed GEDF-SCL model, power grids of varying topologies were generated based on the IEEE 39-bus system model. Transient operational data was obtained by simulating N-1 and N-$\bm{m}$-1 contingencies on these generated power system topologies. Test result demonstrated that the GEDF-SCL model can achieve high accuracy in transient stability prediction and adapt well to changing power grid topologies.

{{</citation>}}


### (79/88) Artificial-Intelligence-Based Triple Phase Shift Modulation for Dual Active Bridge Converter with Minimized Current Stress (Xinze Li et al., 2023)

{{<citation>}}

Xinze Li, Xin Zhang, Fanfan Lin, Changjiang Sun, Kezhi Mao. (2023)  
**Artificial-Intelligence-Based Triple Phase Shift Modulation for Dual Active Bridge Converter with Minimized Current Stress**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-SY, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00382v1)  

---


**ABSTRACT**  
The dual active bridge (DAB) converter has been popular in many applications for its outstanding power density and bidirectional power transfer capacity. Up to now, triple phase shift (TPS) can be considered as one of the most advanced modulation techniques for DAB converter. It can widen zero voltage switching range and improve power efficiency significantly. Currently, current stress of the DAB converter has been an important performance indicator when TPS modulation is applied for smaller size and higher efficiency. However, to minimize the current stress when the DAB converter is under TPS modulation, two difficulties exist in analysis process and realization process, respectively. Firstly, three degrees of modulation variables in TPS modulation bring challenges to the analysis of current stress in different operating modes. This analysis and deduction process leads to heavy computational burden and also suffers from low accuracy. Secondly, to realize TPS modulation, if a lookup table is adopted after the optimization of modulation variables, modulation performance will be unsatisfactory because of the discrete nature of lookup table. Therefore, an AI-based TPS modulation (AI-TPSM) strategy is proposed in this paper. Neural network (NN) and fuzzy inference system (FIS) are utilized to deal with the two difficulties mentioned above. With the proposed AI-TPSM, the optimization of TPS modulation for minimized current stress will enjoy high degree of automation which can relieve engineers' working burden and improve accuracy. In the end of this paper, the effectiveness of the proposed AI-TPSM has been experimentally verified with a 1 kW prototype.

{{</citation>}}


### (80/88) Deep Reinforcement Learning-Based Battery Conditioning Hierarchical V2G Coordination for Multi-Stakeholder Benefits (Yubao Zhang et al., 2023)

{{<citation>}}

Yubao Zhang, Xin Chen, Yi Gu, Zhicheng Li, Wu Kai. (2023)  
**Deep Reinforcement Learning-Based Battery Conditioning Hierarchical V2G Coordination for Multi-Stakeholder Benefits**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.00218v1)  

---


**ABSTRACT**  
With the growing prevalence of electric vehicles (EVs) and advancements in EV electronics, vehicle-to-grid (V2G) techniques and large-scale scheduling strategies have emerged to promote renewable energy utilization and power grid stability. This study proposes a multi-stakeholder hierarchical V2G coordination based on deep reinforcement learning (DRL) and the Proof of Stake algorithm. Furthermore, the multi-stakeholders include the power grid, EV aggregators (EVAs), and users, and the proposed strategy can achieve multi-stakeholder benefits. On the grid side, load fluctuations and renewable energy consumption are considered, while on the EVA side, energy constraints and charging costs are considered. The three critical battery conditioning parameters of battery SOX are considered on the user side, including state of charge, state of power, and state of health. Compared with four typical baselines, the multi-stakeholder hierarchical coordination strategy can enhance renewable energy consumption, mitigate load fluctuations, meet the energy demands of EVA, and reduce charging costs and battery degradation under realistic operating conditions.

{{</citation>}}


## eess.IV (4)



### (81/88) Improved Prognostic Prediction of Pancreatic Cancer Using Multi-Phase CT by Integrating Neural Distance and Texture-Aware Transformer (Hexin Dong et al., 2023)

{{<citation>}}

Hexin Dong, Jiawen Yao, Yuxing Tang, Mingze Yuan, Yingda Xia, Jian Zhou, Hong Lu, Jingren Zhou, Bin Dong, Le Lu, Li Zhang, Zaiyi Liu, Yu Shi, Ling Zhang. (2023)  
**Improved Prognostic Prediction of Pancreatic Cancer Using Multi-Phase CT by Integrating Neural Distance and Texture-Aware Transformer**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2308.00507v1)  

---


**ABSTRACT**  
Pancreatic ductal adenocarcinoma (PDAC) is a highly lethal cancer in which the tumor-vascular involvement greatly affects the resectability and, thus, overall survival of patients. However, current prognostic prediction methods fail to explicitly and accurately investigate relationships between the tumor and nearby important vessels. This paper proposes a novel learnable neural distance that describes the precise relationship between the tumor and vessels in CT images of different patients, adopting it as a major feature for prognosis prediction. Besides, different from existing models that used CNNs or LSTMs to exploit tumor enhancement patterns on dynamic contrast-enhanced CT imaging, we improved the extraction of dynamic tumor-related texture features in multi-phase contrast-enhanced CT by fusing local and global features using CNN and transformer modules, further enhancing the features extracted across multi-phase CT images. We extensively evaluated and compared the proposed method with existing methods in the multi-center (n=4) dataset with 1,070 patients with PDAC, and statistical analysis confirmed its clinical effectiveness in the external test set consisting of three centers. The developed risk marker was the strongest predictor of overall survival among preoperative factors and it has the potential to be combined with established clinical factors to select patients at higher risk who might benefit from neoadjuvant therapy.

{{</citation>}}


### (82/88) An L2-Normalized Spatial Attention Network For Accurate And Fast Classification Of Brain Tumors In 2D T1-Weighted CE-MRI Images (Grace Billingsley et al., 2023)

{{<citation>}}

Grace Billingsley, Julia Dietlmeier, Vivek Narayanaswamy, Andreas Spanias, Noel E. OConnor. (2023)  
**An L2-Normalized Spatial Attention Network For Accurate And Fast Classification Of Brain Tumors In 2D T1-Weighted CE-MRI Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.00491v1)  

---


**ABSTRACT**  
We propose an accurate and fast classification network for classification of brain tumors in MRI images that outperforms all lightweight methods investigated in terms of accuracy. We test our model on a challenging 2D T1-weighted CE-MRI dataset containing three types of brain tumors: Meningioma, Glioma and Pituitary. We introduce an l2-normalized spatial attention mechanism that acts as a regularizer against overfitting during training. We compare our results against the state-of-the-art on this dataset and show that by integrating l2-normalized spatial attention into a baseline network we achieve a performance gain of 1.79 percentage points. Even better accuracy can be attained by combining our model in an ensemble with the pretrained VGG16 at the expense of execution speed. Our code is publicly available at https://github.com/juliadietlmeier/MRI_image_classification

{{</citation>}}


### (83/88) Space Debris: Are Deep Learning-based Image Enhancements part of the Solution? (Michele Jamrozik et al., 2023)

{{<citation>}}

Michele Jamrozik, Vincent Gaudillière, Mohamed Adel Musallam, Djamila Aouada. (2023)  
**Space Debris: Are Deep Learning-based Image Enhancements part of the Solution?**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, physics-space-ph  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.00408v1)  

---


**ABSTRACT**  
The volume of space debris currently orbiting the Earth is reaching an unsustainable level at an accelerated pace. The detection, tracking, identification, and differentiation between orbit-defined, registered spacecraft, and rogue/inactive space ``objects'', is critical to asset protection. The primary objective of this work is to investigate the validity of Deep Neural Network (DNN) solutions to overcome the limitations and image artefacts most prevalent when captured with monocular cameras in the visible light spectrum. In this work, a hybrid UNet-ResNet34 Deep Learning (DL) architecture pre-trained on the ImageNet dataset, is developed. Image degradations addressed include blurring, exposure issues, poor contrast, and noise. The shortage of space-generated data suitable for supervised DL is also addressed. A visual comparison between the URes34P model developed in this work and the existing state of the art in deep learning image enhancement methods, relevant to images captured in space, is presented. Based upon visual inspection, it is determined that our UNet model is capable of correcting for space-related image degradations and merits further investigation to reduce its computational complexity.

{{</citation>}}


### (84/88) Unleashing the Power of Self-Supervised Image Denoising: A Comprehensive Review (Dan Zhang et al., 2023)

{{<citation>}}

Dan Zhang, Fangfang Zhou, Yuanzhou Wei, Xiao Yang, Yuan Gu. (2023)  
**Unleashing the Power of Self-Supervised Image Denoising: A Comprehensive Review**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2308.00247v1)  

---


**ABSTRACT**  
The advent of deep learning has brought a revolutionary transformation to image denoising techniques. However, the persistent challenge of acquiring noise-clean pairs for supervised methods in real-world scenarios remains formidable, necessitating the exploration of more practical self-supervised image denoising. This paper focuses on self-supervised image denoising methods that offer effective solutions to address this challenge. Our comprehensive review thoroughly analyzes the latest advancements in self-supervised image denoising approaches, categorizing them into three distinct classes: General methods, Blind Spot Network (BSN)-based methods, and Transformer-based methods. For each class, we provide a concise theoretical analysis along with their practical applications. To assess the effectiveness of these methods, we present both quantitative and qualitative experimental results on various datasets, utilizing classical algorithms as benchmarks. Additionally, we critically discuss the current limitations of these methods and propose promising directions for future research. By offering a detailed overview of recent developments in self-supervised image denoising, this review serves as an invaluable resource for researchers and practitioners in the field, facilitating a deeper understanding of this emerging domain and inspiring further advancements.

{{</citation>}}


## cs.DC (1)



### (85/88) Computation Offloading with Multiple Agents in Edge-Computing-Supported IoT (Shihao Shen et al., 2023)

{{<citation>}}

Shihao Shen, Yiwen Han, Xiaofei Wang, Yan Wang. (2023)  
**Computation Offloading with Multiple Agents in Edge-Computing-Supported IoT**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.00463v1)  

---


**ABSTRACT**  
With the development of the Internet of Things (IoT) and the birth of various new IoT devices, the capacity of massive IoT devices is facing challenges. Fortunately, edge computing can optimize problems such as delay and connectivity by offloading part of the computational tasks to edge nodes close to the data source. Using this feature, IoT devices can save more resources while still maintaining the quality of service. However, since computation offloading decisions concern joint and complex resource management, we use multiple Deep Reinforcement Learning (DRL) agents deployed on IoT devices to guide their own decisions. Besides, Federated Learning (FL) is utilized to train DRL agents in a distributed fashion, aiming to make the DRL-based decision making practical and further decrease the transmission cost between IoT devices and Edge Nodes. In this article, we first study the problem of computation offloading optimization and prove the problem is an NP-hard problem. Then, based on DRL and FL, we propose an offloading algorithm that is different from the traditional method. Finally, we studied the effects of various parameters on the performance of the algorithm and verified the effectiveness of both the DRL and FL in the IoT system.

{{</citation>}}


## cs.IT (1)



### (86/88) Coded Modulation Schemes for Voronoi Constellations (S. Li et al., 2023)

{{<citation>}}

S. Li, A. Mirani, M. Karlsson, E. Agrell. (2023)  
**Coded Modulation Schemes for Voronoi Constellations**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.00407v1)  

---


**ABSTRACT**  
Multidimensional Voronoi constellations (VCs) are shown to be more power-efficient than quadrature amplitude modulation (QAM) formats given the same uncoded bit error rate, and also have higher achievable information rates. However, a coded modulation scheme to sustain these gains after forward error correction (FEC) coding is still lacking. This paper designs coded modulation schemes with soft-decision FEC codes for VCs, including bit-interleaved coded modulation (BICM) and multilevel coded modulation (MLCM), together with three bit-to-integer mapping algorithms and log-likelihood ratio calculation algorithms. Simulation results show that VCs can achieve up to 1.84 dB signal-to-noise ratio (SNR) gains over QAM with BICM, and up to 0.99 dB SNR gains over QAM with MLCM for the additive white Gaussian noise channel, with a surprisingly low complexity.

{{</citation>}}


## cs.SE (2)



### (87/88) The Hitchhiker's Guide to Program Analysis: A Journey with Large Language Models (Haonan Li et al., 2023)

{{<citation>}}

Haonan Li, Yu Hao, Yizhuo Zhai, Zhiyun Qian. (2023)  
**The Hitchhiker's Guide to Program Analysis: A Journey with Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.00245v1)  

---


**ABSTRACT**  
Static analysis is a widely used technique in software engineering for identifying and mitigating bugs. However, a significant hurdle lies in achieving a delicate balance between precision and scalability. Large Language Models (LLMs) offer a promising alternative, as recent advances demonstrate remarkable capabilities in comprehending, generating, and even debugging code. Yet, the logic of bugs can be complex and require sophisticated reasoning and a large analysis scope spanning multiple functions. Therefore, at this point, LLMs are better used in an assistive role to complement static analysis. In this paper, we take a deep dive into the open space of LLM-assisted static analysis, using use-before-initialization (UBI) bugs as a case study. To this end, we develop LLift, a fully automated agent that interfaces with both a static analysis tool and an LLM. By carefully designing the agent and the prompts, we are able to overcome a number of challenges, including bug-specific modeling, the large problem scope, the non-deterministic nature of LLMs, etc. Tested in a real-world scenario analyzing nearly a thousand potential UBI bugs produced by static analysis, LLift demonstrates an extremely potent capability, showcasing a high precision (50%) and recall rate (100%). It even identified 13 previously unknown UBI bugs in the Linux kernel. This research paves the way for new opportunities and methodologies in the use of LLMs for bug discovery in extensive, real-world datasets.

{{</citation>}}


### (88/88) Prompts Matter: Insights and Strategies for Prompt Engineering in Automated Software Traceability (Alberto D. Rodriguez et al., 2023)

{{<citation>}}

Alberto D. Rodriguez, Katherine R. Dearstyne, Jane Cleland-Huang. (2023)  
**Prompts Matter: Insights and Strategies for Prompt Engineering in Automated Software Traceability**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.00229v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have the potential to revolutionize automated traceability by overcoming the challenges faced by previous methods and introducing new possibilities. However, the optimal utilization of LLMs for automated traceability remains unclear. This paper explores the process of prompt engineering to extract link predictions from an LLM. We provide detailed insights into our approach for constructing effective prompts, offering our lessons learned. Additionally, we propose multiple strategies for leveraging LLMs to generate traceability links, improving upon previous zero-shot methods on the ranking of candidate links after prompt refinement. The primary objective of this paper is to inspire and assist future researchers and engineers by highlighting the process of constructing traceability prompts to effectively harness LLMs for advancing automatic traceability.

{{</citation>}}
