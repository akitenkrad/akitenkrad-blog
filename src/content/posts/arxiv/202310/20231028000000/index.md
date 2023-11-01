---
draft: false
title: "arXiv @ 2023.10.28"
date: 2023-10-28
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.28"
    identifier: arxiv_20231028
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AI (10)](#csai-10)
- [cs.CV (23)](#cscv-23)
- [cs.CL (40)](#cscl-40)
- [cs.LG (35)](#cslg-35)
- [eess.AS (2)](#eessas-2)
- [cs.IR (2)](#csir-2)
- [econ.GN (1)](#econgn-1)
- [stat.ML (1)](#statml-1)
- [cs.NI (1)](#csni-1)
- [cs.CY (5)](#cscy-5)
- [physics.plasm-ph (1)](#physicsplasm-ph-1)
- [cs.PL (1)](#cspl-1)
- [cs.HC (1)](#cshc-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.RO (2)](#csro-2)
- [eess.SY (3)](#eesssy-3)
- [cs.SD (1)](#cssd-1)
- [cs.IT (1)](#csit-1)
- [cs.CR (4)](#cscr-4)
- [cs.SE (1)](#csse-1)
- [cs.SI (1)](#cssi-1)
- [cs.MM (1)](#csmm-1)
- [astro-ph.EP (1)](#astro-phep-1)

## cs.AI (10)



### (1/139) Style-Aware Radiology Report Generation with RadGraph and Few-Shot Prompting (Benjamin Yan et al., 2023)

{{<citation>}}

Benjamin Yan, Ruochen Liu, David E. Kuo, Subathra Adithan, Eduardo Pontes Reis, Stephen Kwak, Vasantha Kumar Venugopal, Chloe P. O'Connell, Agustina Saenz, Pranav Rajpurkar, Michael Moor. (2023)  
**Style-Aware Radiology Report Generation with RadGraph and Few-Shot Prompting**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.17811v2)  

---


**ABSTRACT**  
Automatically generated reports from medical images promise to improve the workflow of radiologists. Existing methods consider an image-to-report modeling task by directly generating a fully-fledged report from an image. However, this conflates the content of the report (e.g., findings and their attributes) with its style (e.g., format and choice of words), which can lead to clinically inaccurate reports. To address this, we propose a two-step approach for radiology report generation. First, we extract the content from an image; then, we verbalize the extracted content into a report that matches the style of a specific radiologist. For this, we leverage RadGraph -- a graph representation of reports -- together with large language models (LLMs). In our quantitative evaluations, we find that our approach leads to beneficial performance. Our human evaluation with clinical raters highlights that the AI-generated reports are indistinguishably tailored to the style of individual radiologist despite leveraging only a few examples as context.

{{</citation>}}


### (2/139) Utilizing Language Models for Energy Load Forecasting (Hao Xue et al., 2023)

{{<citation>}}

Hao Xue, Flora D. Salim. (2023)  
**Utilizing Language Models for Energy Load Forecasting**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17788v1)  

---


**ABSTRACT**  
Energy load forecasting plays a crucial role in optimizing resource allocation and managing energy consumption in buildings and cities. In this paper, we propose a novel approach that leverages language models for energy load forecasting. We employ prompting techniques to convert energy consumption data into descriptive sentences, enabling fine-tuning of language models. By adopting an autoregressive generating approach, our proposed method enables predictions of various horizons of future energy load consumption. Through extensive experiments on real-world datasets, we demonstrate the effectiveness and accuracy of our proposed method. Our results indicate that utilizing language models for energy load forecasting holds promise for enhancing energy efficiency and facilitating intelligent decision-making in energy systems.

{{</citation>}}


### (3/139) In-Context Learning Dynamics with Random Binary Sequences (Eric J. Bigelow et al., 2023)

{{<citation>}}

Eric J. Bigelow, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka, Tomer D. Ullman. (2023)  
**In-Context Learning Dynamics with Random Binary Sequences**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2310.17639v1)  

---


**ABSTRACT**  
Large language models (LLMs) trained on huge corpora of text datasets demonstrate complex, emergent capabilities, achieving state-of-the-art performance on tasks they were not explicitly trained for. The precise nature of LLM capabilities is often mysterious, and different prompts can elicit different capabilities through in-context learning. We propose a Cognitive Interpretability framework that enables us to analyze in-context learning dynamics to understand latent concepts in LLMs underlying behavioral patterns. This provides a more nuanced understanding than success-or-failure evaluation benchmarks, but does not require observing internal activations as a mechanistic interpretation of circuits would. Inspired by the cognitive science of human randomness perception, we use random binary sequences as context and study dynamics of in-context learning by manipulating properties of context data, such as sequence length. In the latest GPT-3.5+ models, we find emergent abilities to generate pseudo-random numbers and learn basic formal languages, with striking in-context learning dynamics where model outputs transition sharply from pseudo-random behaviors to deterministic repetition.

{{</citation>}}


### (4/139) CompeteAI: Understanding the Competition Behaviors in Large Language Model-based Agents (Qinlin Zhao et al., 2023)

{{<citation>}}

Qinlin Zhao, Jindong Wang, Yixuan Zhang, Yiqiao Jin, Kaijie Zhu, Hao Chen, Xing Xie. (2023)  
**CompeteAI: Understanding the Competition Behaviors in Large Language Model-based Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-HC, cs-MA, cs.AI  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.17512v1)  

---


**ABSTRACT**  
Large language models (LLMs) have been widely used as agents to complete different tasks, such as personal assistance or event planning. While most work has focused on cooperation and collaboration between agents, little work explores competition, another important mechanism that fosters the development of society and economy. In this paper, we seek to examine the competition behaviors in LLM-based agents. We first propose a general framework to study the competition between agents. Then, we implement a practical competitive environment using GPT-4 to simulate a virtual town with two types of agents, including restaurant agents and customer agents. Specifically, restaurant agents compete with each other to attract more customers, where the competition fosters them to transform, such as cultivating new operating strategies. The results of our experiments reveal several interesting findings ranging from social learning to Matthew Effect, which aligns well with existing sociological and economic theories. We believe that competition between agents deserves further investigation to help us understand society better. The code will be released soon.

{{</citation>}}


### (5/139) Orchestration of Emulator Assisted Mobile Edge Tuning for AI Foundation Models: A Multi-Agent Deep Reinforcement Learning Approach (Wenhan Yu et al., 2023)

{{<citation>}}

Wenhan Yu, Terence Jie Chua, Jun Zhao. (2023)  
**Orchestration of Emulator Assisted Mobile Edge Tuning for AI Foundation Models: A Multi-Agent Deep Reinforcement Learning Approach**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DC, cs-LG, cs-NI, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.17492v1)  

---


**ABSTRACT**  
The efficient deployment and fine-tuning of foundation models are pivotal in contemporary artificial intelligence. In this study, we present a groundbreaking paradigm integrating Mobile Edge Computing (MEC) with foundation models, specifically designed to enhance local task performance on user equipment (UE). Central to our approach is the innovative Emulator-Adapter architecture, segmenting the foundation model into two cohesive modules. This design not only conserves computational resources but also ensures adaptability and fine-tuning efficiency for downstream tasks. Additionally, we introduce an advanced resource allocation mechanism that is fine-tuned to the needs of the Emulator-Adapter structure in decentralized settings. To address the challenges presented by this system, we employ a hybrid multi-agent Deep Reinforcement Learning (DRL) strategy, adept at handling mixed discrete-continuous action spaces, ensuring dynamic and optimal resource allocations. Our comprehensive simulations and validations underscore the practical viability of our approach, demonstrating its robustness, efficiency, and scalability. Collectively, this work offers a fresh perspective on deploying foundation models and balancing computational efficiency with task proficiency.

{{</citation>}}


### (6/139) Goals are Enough: Inducing AdHoc cooperation among unseen Multi-Agent systems in IMFs (Kaushik Dey et al., 2023)

{{<citation>}}

Kaushik Dey, Satheesh K. Perepu, Abir Das. (2023)  
**Goals are Enough: Inducing AdHoc cooperation among unseen Multi-Agent systems in IMFs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MA, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17416v1)  

---


**ABSTRACT**  
Intent-based management will play a critical role in achieving customers' expectations in the next-generation mobile networks. Traditional methods cannot perform efficient resource management since they tend to handle each expectation independently. Existing approaches, e.g., based on multi-agent reinforcement learning (MARL) allocate resources in an efficient fashion when there are conflicting expectations on the network slice. However, in reality, systems are often far more complex to be addressed by a standalone MARL formulation. Often there exists a hierarchical structure of intent fulfilment where multiple pre-trained, self-interested agents may need to be further orchestrated by a supervisor or controller agent. Such agents may arrive in the system adhoc, which then needs to be orchestrated along with other available agents. Retraining the whole system every time is often infeasible given the associated time and cost. Given the challenges, such adhoc coordination of pre-trained systems could be achieved through an intelligent supervisor agent which incentivizes pre-trained RL/MARL agents through sets of dynamic contracts (goals or bonuses) and encourages them to act as a cohesive unit towards fulfilling a global expectation. Some approaches use a rule-based supervisor agent and deploy the hierarchical constituent agents sequentially, based on human-coded rules.   In the current work, we propose a framework whereby pre-trained agents can be orchestrated in parallel leveraging an AI-based supervisor agent. For this, we propose to use Adhoc-Teaming approaches which assign optimal goals to the MARL agents and incentivize them to exhibit certain desired behaviours. Results on the network emulator show that the proposed approach results in faster and improved fulfilment of expectations when compared to rule-based approaches and even generalizes to changes in environments.

{{</citation>}}


### (7/139) Dialogue-based generation of self-driving simulation scenarios using Large Language Models (Antonio Valerio Miceli-Barone et al., 2023)

{{<citation>}}

Antonio Valerio Miceli-Barone, Alex Lascarides, Craig Innes. (2023)  
**Dialogue-based generation of self-driving simulation scenarios using Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-RO, cs.AI  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2310.17372v1)  

---


**ABSTRACT**  
Simulation is an invaluable tool for developing and evaluating controllers for self-driving cars. Current simulation frameworks are driven by highly-specialist domain specific languages, and so a natural language interface would greatly enhance usability. But there is often a gap, consisting of tacit assumptions the user is making, between a concise English utterance and the executable code that captures the user's intent. In this paper we describe a system that addresses this issue by supporting an extended multimodal interaction: the user can follow up prior instructions with refinements or revisions, in reaction to the simulations that have been generated from their utterances so far. We use Large Language Models (LLMs) to map the user's English utterances in this interaction into domain-specific code, and so we explore the extent to which LLMs capture the context sensitivity that's necessary for computing the speaker's intended message in discourse.

{{</citation>}}


### (8/139) Exploring the Potential of Generative AI for the World Wide Web (Nouar AlDahoul et al., 2023)

{{<citation>}}

Nouar AlDahoul, Joseph Hong, Matteo Varvello, Yasir Zaki. (2023)  
**Exploring the Potential of Generative AI for the World Wide Web**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-IR, cs.AI  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.17370v1)  

---


**ABSTRACT**  
Generative Artificial Intelligence (AI) is a cutting-edge technology capable of producing text, images, and various media content leveraging generative models and user prompts. Between 2022 and 2023, generative AI surged in popularity with a plethora of applications spanning from AI-powered movies to chatbots. In this paper, we delve into the potential of generative AI within the realm of the World Wide Web, specifically focusing on image generation. Web developers already harness generative AI to help crafting text and images, while Web browsers might use it in the future to locally generate images for tasks like repairing broken webpages, conserving bandwidth, and enhancing privacy. To explore this research area, we have developed WebDiffusion, a tool that allows to simulate a Web powered by stable diffusion, a popular text-to-image model, from both a client and server perspective. WebDiffusion further supports crowdsourcing of user opinions, which we use to evaluate the quality and accuracy of 409 AI-generated images sourced from 60 webpages. Our findings suggest that generative AI is already capable of producing pertinent and high-quality Web images, even without requiring Web designers to manually input prompts, just by leveraging contextual information available within the webpages. However, we acknowledge that direct in-browser image generation remains a challenge, as only highly powerful GPUs, such as the A40 and A100, can (partially) compete with classic image downloads. Nevertheless, this approach could be valuable for a subset of the images, for example when fixing broken webpages or handling highly private content.

{{</citation>}}


### (9/139) FormaT5: Abstention and Examples for Conditional Table Formatting with Natural Language (Mukul Singh et al., 2023)

{{<citation>}}

Mukul Singh, José Cambronero, Sumit Gulwani, Vu Le, Carina Negreanu, Elnaz Nouri, Mohammad Raza, Gust Verbruggen. (2023)  
**FormaT5: Abstention and Examples for Conditional Table Formatting with Natural Language**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-DB, cs-PL, cs.AI  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2310.17306v2)  

---


**ABSTRACT**  
Formatting is an important property in tables for visualization, presentation, and analysis. Spreadsheet software allows users to automatically format their tables by writing data-dependent conditional formatting (CF) rules. Writing such rules is often challenging for users as it requires them to understand and implement the underlying logic. We present FormaT5, a transformer-based model that can generate a CF rule given the target table and a natural language description of the desired formatting logic. We find that user descriptions for these tasks are often under-specified or ambiguous, making it harder for code generation systems to accurately learn the desired rule in a single step. To tackle this problem of under-specification and minimise argument errors, FormaT5 learns to predict placeholders though an abstention objective. These placeholders can then be filled by a second model or, when examples of rows that should be formatted are available, by a programming-by-example system. To evaluate FormaT5 on diverse and real scenarios, we create an extensive benchmark of 1053 CF tasks, containing real-world descriptions collected from four different sources. We release our benchmarks to encourage research in this area. Abstention and filling allow FormaT5 to outperform 8 different neural approaches on our benchmarks, both with and without examples. Our results illustrate the value of building domain-specific learning systems.

{{</citation>}}


### (10/139) Content-based Controls For Music Large Language Modeling (Liwei Lin et al., 2023)

{{<citation>}}

Liwei Lin, Gus Xia, Junyan Jiang, Yixiao Zhang. (2023)  
**Content-based Controls For Music Large Language Modeling**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SD, cs.AI, eess-AS  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.17162v1)  

---


**ABSTRACT**  
Recent years have witnessed a rapid growth of large-scale language models in the domain of music audio. Such models enable end-to-end generation of higher-quality music, and some allow conditioned generation using text descriptions. However, the control power of text controls on music is intrinsically limited, as they can only describe music indirectly through meta-data (such as singers and instruments) or high-level representations (such as genre and emotion). We aim to further equip the models with direct and content-based controls on innate music languages such as pitch, chords and drum track. To this end, we contribute Coco-Mulla, a content-based control method for music large language modeling. It uses a parameter-efficient fine-tuning (PEFT) method tailored for Transformer-based audio models. Experiments show that our approach achieved high-quality music generation with low-resource semi-supervised learning, tuning with less than 4% parameters compared to the original model and training on a small dataset with fewer than 300 songs. Moreover, our approach enables effective content-based controls, and we illustrate the control power via chords and rhythms, two of the most salient features of music audio. Furthermore, we show that by combining content-based controls and text descriptions, our system achieves flexible music variation generation and style transfer. Our source codes and demos are available online.

{{</citation>}}


## cs.CV (23)



### (11/139) ControlLLM: Augment Language Models with Tools by Searching on Graphs (Zhaoyang Liu et al., 2023)

{{<citation>}}

Zhaoyang Liu, Zeqiang Lai, Zhangwei Gao, Erfei Cui, Zhiheng Li, Xizhou Zhu, Lewei Lu, Qifeng Chen, Yu Qiao, Jifeng Dai, Wenhai Wang. (2023)  
**ControlLLM: Augment Language Models with Tools by Searching on Graphs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17796v2)  

---


**ABSTRACT**  
We present ControlLLM, a novel framework that enables large language models (LLMs) to utilize multi-modal tools for solving complex real-world tasks. Despite the remarkable performance of LLMs, they still struggle with tool invocation due to ambiguous user prompts, inaccurate tool selection and parameterization, and inefficient tool scheduling. To overcome these challenges, our framework comprises three key components: (1) a \textit{task decomposer} that breaks down a complex task into clear subtasks with well-defined inputs and outputs; (2) a \textit{Thoughts-on-Graph (ToG) paradigm} that searches the optimal solution path on a pre-built tool graph, which specifies the parameter and dependency relations among different tools; and (3) an \textit{execution engine with a rich toolbox} that interprets the solution path and runs the tools efficiently on different computational devices. We evaluate our framework on diverse tasks involving image, audio, and video processing, demonstrating its superior accuracy, efficiency, and versatility compared to existing methods. The code is at https://github.com/OpenGVLab/ControlLLM .

{{</citation>}}


### (12/139) Graph Convolutional Networks for Complex Traffic Scenario Classification (Tobias Hoek et al., 2023)

{{<citation>}}

Tobias Hoek, Holger Caesar, Andreas Falkovén, Tommy Johansson. (2023)  
**Graph Convolutional Networks for Complex Traffic Scenario Classification**  

---
Primary Category: cs.CV  
Categories: I-2; I-4; I-5, cs-AI, cs-CV, cs-LG, cs-MA, cs.CV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2310.17773v1)  

---


**ABSTRACT**  
A scenario-based testing approach can reduce the time required to obtain statistically significant evidence of the safety of Automated Driving Systems (ADS). Identifying these scenarios in an automated manner is a challenging task. Most methods on scenario classification do not work for complex scenarios with diverse environments (highways, urban) and interaction with other traffic agents. This is mirrored in their approaches which model an individual vehicle in relation to its environment, but neglect the interaction between multiple vehicles (e.g. cut-ins, stationary lead vehicle). Furthermore, existing datasets lack diversity and do not have per-frame annotations to accurately learn the start and end time of a scenario. We propose a method for complex traffic scenario classification that is able to model the interaction of a vehicle with the environment, as well as other agents. We use Graph Convolutional Networks to model spatial and temporal aspects of these scenarios. Expanding the nuScenes and Argoverse 2 driving datasets, we introduce a scenario-labeled dataset, which covers different driving environments and is annotated per frame. Training our method on this dataset, we present a promising baseline for future research on per-frame complex scenario classification.

{{</citation>}}


### (13/139) A Coarse-to-Fine Pseudo-Labeling (C2FPL) Framework for Unsupervised Video Anomaly Detection (Anas Al-lahham et al., 2023)

{{<citation>}}

Anas Al-lahham, Nurbek Tastan, Zaigham Zaheer, Karthik Nandakumar. (2023)  
**A Coarse-to-Fine Pseudo-Labeling (C2FPL) Framework for Unsupervised Video Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.17650v1)  

---


**ABSTRACT**  
Detection of anomalous events in videos is an important problem in applications such as surveillance. Video anomaly detection (VAD) is well-studied in the one-class classification (OCC) and weakly supervised (WS) settings. However, fully unsupervised (US) video anomaly detection methods, which learn a complete system without any annotation or human supervision, have not been explored in depth. This is because the lack of any ground truth annotations significantly increases the magnitude of the VAD challenge. To address this challenge, we propose a simple-but-effective two-stage pseudo-label generation framework that produces segment-level (normal/anomaly) pseudo-labels, which can be further used to train a segment-level anomaly detector in a supervised manner. The proposed coarse-to-fine pseudo-label (C2FPL) generator employs carefully-designed hierarchical divisive clustering and statistical hypothesis testing to identify anomalous video segments from a set of completely unlabeled videos. The trained anomaly detector can be directly applied on segments of an unseen test video to obtain segment-level, and subsequently, frame-level anomaly predictions. Extensive studies on two large-scale public-domain datasets, UCF-Crime and XD-Violence, demonstrate that the proposed unsupervised approach achieves superior performance compared to all existing OCC and US methods , while yielding comparable performance to the state-of-the-art WS methods.

{{</citation>}}


### (14/139) Evaluating Bias and Fairness in Gender-Neutral Pretrained Vision-and-Language Models (Laura Cabello et al., 2023)

{{<citation>}}

Laura Cabello, Emanuele Bugliarello, Stephanie Brandl, Desmond Elliott. (2023)  
**Evaluating Bias and Fairness in Gender-Neutral Pretrained Vision-and-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Bias, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.17530v1)  

---


**ABSTRACT**  
Pretrained machine learning models are known to perpetuate and even amplify existing biases in data, which can result in unfair outcomes that ultimately impact user experience. Therefore, it is crucial to understand the mechanisms behind those prejudicial biases to ensure that model performance does not result in discriminatory behaviour toward certain groups or populations. In this work, we define gender bias as our case study. We quantify bias amplification in pretraining and after fine-tuning on three families of vision-and-language models. We investigate the connection, if any, between the two learning stages, and evaluate how bias amplification reflects on model performance. Overall, we find that bias amplification in pretraining and after fine-tuning are independent. We then examine the effect of continued pretraining on gender-neutral data, finding that this reduces group disparities, i.e., promotes fairness, on VQAv2 and retrieval tasks without significantly compromising task performance.

{{</citation>}}


### (15/139) A Hybrid Graph Network for Complex Activity Detection in Video (Salman Khan et al., 2023)

{{<citation>}}

Salman Khan, Izzeddin Teeti, Andrew Bradley, Mohamed Elhoseiny, Fabio Cuzzolin. (2023)  
**A Hybrid Graph Network for Complex Activity Detection in Video**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.17493v2)  

---


**ABSTRACT**  
Interpretation and understanding of video presents a challenging computer vision task in numerous fields - e.g. autonomous driving and sports analytics. Existing approaches to interpreting the actions taking place within a video clip are based upon Temporal Action Localisation (TAL), which typically identifies short-term actions. The emerging field of Complex Activity Detection (CompAD) extends this analysis to long-term activities, with a deeper understanding obtained by modelling the internal structure of a complex activity taking place within the video. We address the CompAD problem using a hybrid graph neural network which combines attention applied to a graph encoding the local (short-term) dynamic scene with a temporal graph modelling the overall long-duration activity. Our approach is as follows: i) Firstly, we propose a novel feature extraction technique which, for each video snippet, generates spatiotemporal `tubes' for the active elements (`agents') in the (local) scene by detecting individual objects, tracking them and then extracting 3D features from all the agent tubes as well as the overall scene. ii) Next, we construct a local scene graph where each node (representing either an agent tube or the scene) is connected to all other nodes. Attention is then applied to this graph to obtain an overall representation of the local dynamic scene. iii) Finally, all local scene graph representations are interconnected via a temporal graph, to estimate the complex activity class together with its start and end time. The proposed framework outperforms all previous state-of-the-art methods on all three datasets including ActivityNet-1.3, Thumos-14, and ROAD.

{{</citation>}}


### (16/139) OTMatch: Improving Semi-Supervised Learning with Optimal Transport (Zhiquan Tan et al., 2023)

{{<citation>}}

Zhiquan Tan, Kaipeng Zheng, Weiran Huang. (2023)  
**OTMatch: Improving Semi-Supervised Learning with Optimal Transport**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.17455v1)  

---


**ABSTRACT**  
Semi-supervised learning has made remarkable strides by effectively utilizing a limited amount of labeled data while capitalizing on the abundant information present in unlabeled data. However, current algorithms often prioritize aligning image predictions with specific classes generated through self-training techniques, thereby neglecting the inherent relationships that exist within these classes. In this paper, we present a new approach called OTMatch, which leverages semantic relationships among classes by employing an optimal transport loss function. By utilizing optimal transport, our proposed method consistently outperforms established state-of-the-art methods. Notably, we observed a substantial improvement of a certain percentage in accuracy compared to the current state-of-the-art method, FreeMatch. OTMatch achieves 3.18%, 3.46%, and 1.28% error rate reduction over FreeMatch on CIFAR-10 with 1 label per class, STL-10 with 4 labels per class, and ImageNet with 100 labels per class, respectively. This demonstrates the effectiveness and superiority of our approach in harnessing semantic relationships to enhance learning performance in a semi-supervised setting.

{{</citation>}}


### (17/139) Uncertainty-weighted Loss Functions for Improved Adversarial Attacks on Semantic Segmentation (Kira Maag et al., 2023)

{{<citation>}}

Kira Maag, Asja Fischer. (2023)  
**Uncertainty-weighted Loss Functions for Improved Adversarial Attacks on Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.17436v1)  

---


**ABSTRACT**  
State-of-the-art deep neural networks have been shown to be extremely powerful in a variety of perceptual tasks like semantic segmentation. However, these networks are vulnerable to adversarial perturbations of the input which are imperceptible for humans but lead to incorrect predictions. Treating image segmentation as a sum of pixel-wise classifications, adversarial attacks developed for classification models were shown to be applicable to segmentation models as well. In this work, we present simple uncertainty-based weighting schemes for the loss functions of such attacks that (i) put higher weights on pixel classifications which can more easily perturbed and (ii) zero-out the pixel-wise losses corresponding to those pixels that are already confidently misclassified. The weighting schemes can be easily integrated into the loss function of a range of well-known adversarial attackers with minimal additional computational overhead, but lead to significant improved perturbation performance, as we demonstrate in our empirical analysis on several datasets and models.

{{</citation>}}


### (18/139) AntifakePrompt: Prompt-Tuned Vision-Language Models are Fake Image Detectors (You-Ming Chang et al., 2023)

{{<citation>}}

You-Ming Chang, Chen Yeh, Wei-Chen Chiu, Ning Yu. (2023)  
**AntifakePrompt: Prompt-Tuned Vision-Language Models are Fake Image Detectors**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17419v1)  

---


**ABSTRACT**  
Deep generative models can create remarkably photorealistic fake images while raising concerns about misinformation and copyright infringement, known as deepfake threats. Deepfake detection technique is developed to distinguish between real and fake images, where the existing methods typically learn classifiers in the image domain or various feature domains. However, the generalizability of deepfake detection against emerging and more advanced generative models remains challenging. In this paper, being inspired by the zero-shot advantages of Vision-Language Models (VLMs), we propose a novel approach using VLMs (e.g. InstructBLIP) and prompt tuning techniques to improve the deepfake detection accuracy over unseen data. We formulate deepfake detection as a visual question answering problem, and tune soft prompts for InstructBLIP to answer the real/fake information of a query image. We conduct full-spectrum experiments on datasets from 3 held-in and 13 held-out generative models, covering modern text-to-image generation, image editing and image attacks. Results demonstrate that (1) the deepfake detection accuracy can be significantly and consistently improved (from 58.8% to 91.31%, in average accuracy over unseen data) using pretrained vision-language models with prompt tuning; (2) our superior performance is at less cost of trainable parameters, resulting in an effective and efficient solution for deepfake detection. Code and models can be found at https://github.com/nctu-eva-lab/AntifakePrompt.

{{</citation>}}


### (19/139) Circuit as Set of Points (Jialv Zou et al., 2023)

{{<citation>}}

Jialv Zou, Xinggang Wang, Jiahao Guo, Wenyu Liu, Qian Zhang, Chang Huang. (2023)  
**Circuit as Set of Points**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2310.17418v1)  

---


**ABSTRACT**  
As the size of circuit designs continues to grow rapidly, artificial intelligence technologies are being extensively used in Electronic Design Automation (EDA) to assist with circuit design. Placement and routing are the most time-consuming parts of the physical design process, and how to quickly evaluate the placement has become a hot research topic. Prior works either transformed circuit designs into images using hand-crafted methods and then used Convolutional Neural Networks (CNN) to extract features, which are limited by the quality of the hand-crafted methods and could not achieve end-to-end training, or treated the circuit design as a graph structure and used Graph Neural Networks (GNN) to extract features, which require time-consuming preprocessing. In our work, we propose a novel perspective for circuit design by treating circuit components as point clouds and using Transformer-based point cloud perception methods to extract features from the circuit. This approach enables direct feature extraction from raw data without any preprocessing, allows for end-to-end training, and results in high performance. Experimental results show that our method achieves state-of-the-art performance in congestion prediction tasks on both the CircuitNet and ISPD2015 datasets, as well as in design rule check (DRC) violation prediction tasks on the CircuitNet dataset. Our method establishes a bridge between the relatively mature point cloud perception methods and the fast-developing EDA algorithms, enabling us to leverage more collective intelligence to solve this task. To facilitate the research of open EDA design, source codes and pre-trained models are released at https://github.com/hustvl/circuitformer.

{{</citation>}}


### (20/139) YOLO-BEV: Generating Bird's-Eye View in the Same Way as 2D Object Detection (Chang Liu et al., 2023)

{{<citation>}}

Chang Liu, Liguo Zhou, Yanliang Huang, Alois Knoll. (2023)  
**YOLO-BEV: Generating Bird's-Eye View in the Same Way as 2D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.17379v1)  

---


**ABSTRACT**  
Vehicle perception systems strive to achieve comprehensive and rapid visual interpretation of their surroundings for improved safety and navigation. We introduce YOLO-BEV, an efficient framework that harnesses a unique surrounding cameras setup to generate a 2D bird's-eye view of the vehicular environment. By strategically positioning eight cameras, each at a 45-degree interval, our system captures and integrates imagery into a coherent 3x3 grid format, leaving the center blank, providing an enriched spatial representation that facilitates efficient processing. In our approach, we employ YOLO's detection mechanism, favoring its inherent advantages of swift response and compact model structure. Instead of leveraging the conventional YOLO detection head, we augment it with a custom-designed detection head, translating the panoramically captured data into a unified bird's-eye view map of ego car. Preliminary results validate the feasibility of YOLO-BEV in real-time vehicular perception tasks. With its streamlined architecture and potential for rapid deployment due to minimized parameters, YOLO-BEV poses as a promising tool that may reshape future perspectives in autonomous driving systems.

{{</citation>}}


### (21/139) CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling (Seyedmorteza Sadat et al., 2023)

{{<citation>}}

Seyedmorteza Sadat, Jakob Buhmann, Derek Bradely, Otmar Hilliges, Romann M. Weber. (2023)  
**CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.17347v1)  

---


**ABSTRACT**  
While conditional diffusion models are known to have good coverage of the data distribution, they still face limitations in output diversity, particularly when sampled with a high classifier-free guidance scale for optimal image quality or when trained on small datasets. We attribute this problem to the role of the conditioning signal in inference and offer an improved sampling strategy for diffusion models that can increase generation diversity, especially at high guidance scales, with minimal loss of sample quality. Our sampling strategy anneals the conditioning signal by adding scheduled, monotonically decreasing Gaussian noise to the conditioning vector during inference to balance diversity and condition alignment. Our Condition-Annealed Diffusion Sampler (CADS) can be used with any pretrained model and sampling algorithm, and we show that it boosts the diversity of diffusion models in various conditional generation tasks. Further, using an existing pretrained diffusion model, CADS achieves a new state-of-the-art FID of 1.70 and 2.31 for class-conditional ImageNet generation at 256$\times$256 and 512$\times$512 respectively.

{{</citation>}}


### (22/139) RIO: A Benchmark for Reasoning Intention-Oriented Objects in Open Environments (Mengxue Qu et al., 2023)

{{<citation>}}

Mengxue Qu, Yu Wu, Wu Liu, Xiaodan Liang, Jingkuan Song, Yao Zhao, Yunchao Wei. (2023)  
**RIO: A Benchmark for Reasoning Intention-Oriented Objects in Open Environments**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.17290v1)  

---


**ABSTRACT**  
Intention-oriented object detection aims to detect desired objects based on specific intentions or requirements. For instance, when we desire to "lie down and rest", we instinctively seek out a suitable option such as a "bed" or a "sofa" that can fulfill our needs. Previous work in this area is limited either by the number of intention descriptions or by the affordance vocabulary available for intention objects. These limitations make it challenging to handle intentions in open environments effectively. To facilitate this research, we construct a comprehensive dataset called Reasoning Intention-Oriented Objects (RIO). In particular, RIO is specifically designed to incorporate diverse real-world scenarios and a wide range of object categories. It offers the following key features: 1) intention descriptions in RIO are represented as natural sentences rather than a mere word or verb phrase, making them more practical and meaningful; 2) the intention descriptions are contextually relevant to the scene, enabling a broader range of potential functionalities associated with the objects; 3) the dataset comprises a total of 40,214 images and 130,585 intention-object pairs. With the proposed RIO, we evaluate the ability of some existing models to reason intention-oriented objects in open environments.

{{</citation>}}


### (23/139) Prototypical Contrastive Learning-based CLIP Fine-tuning for Object Re-identification (Jiachen Li et al., 2023)

{{<citation>}}

Jiachen Li, Xiaojin Gong. (2023)  
**Prototypical Contrastive Learning-based CLIP Fine-tuning for Object Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.17218v1)  

---


**ABSTRACT**  
This work aims to adapt large-scale pre-trained vision-language models, such as contrastive language-image pretraining (CLIP), to enhance the performance of object reidentification (Re-ID) across various supervision settings. Although prompt learning has enabled a recent work named CLIP-ReID to achieve promising performance, the underlying mechanisms and the necessity of prompt learning remain unclear due to the absence of semantic labels in ReID tasks. In this work, we first analyze the role prompt learning in CLIP-ReID and identify its limitations. Based on our investigations, we propose a simple yet effective approach to adapt CLIP for supervised object Re-ID. Our approach directly fine-tunes the image encoder of CLIP using a prototypical contrastive learning (PCL) loss, eliminating the need for prompt learning. Experimental results on both person and vehicle Re-ID datasets demonstrate the competitiveness of our method compared to CLIP-ReID. Furthermore, we extend our PCL-based CLIP fine-tuning approach to unsupervised scenarios, where we achieve state-of-the art performance.

{{</citation>}}


### (24/139) Emotion Recognition by Video: A review (Junxiao Xue et al., 2023)

{{<citation>}}

Junxiao Xue, Jie Wang, Xuecheng Wu, Liangyu Fu. (2023)  
**Emotion Recognition by Video: A review**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2310.17212v1)  

---


**ABSTRACT**  
Video emotion recognition is an important branch of affective computing, and its solutions can be applied in different fields such as human-computer interaction (HCI) and intelligent medical treatment. Although the number of papers published in the field of emotion recognition is increasing, there are few comprehensive literature reviews covering related research on video emotion recognition. Therefore, this paper selects articles published from 2015 to 2023 to systematize the existing trends in video emotion recognition in related studies. In this paper, we first talk about two typical emotion models, then we talk about databases that are frequently utilized for video emotion recognition, including unimodal databases and multimodal databases. Next, we look at and classify the specific structure and performance of modern unimodal and multimodal video emotion recognition methods, talk about the benefits and drawbacks of each, and then we compare them in detail in the tables. Further, we sum up the primary difficulties right now looked by video emotion recognition undertakings and point out probably the most encouraging future headings, such as establishing an open benchmark database and better multimodal fusion strategys. The essential objective of this paper is to assist scholarly and modern scientists with keeping up to date with the most recent advances and new improvements in this speedy, high-influence field of video emotion recognition.

{{</citation>}}


### (25/139) Understanding the Effects of Projectors in Knowledge Distillation (Yudong Chen et al., 2023)

{{<citation>}}

Yudong Chen, Sen Wang, Jiajun Liu, Xuwei Xu, Frank de Hoog, Brano Kusy, Zi Huang. (2023)  
**Understanding the Effects of Projectors in Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.17183v1)  

---


**ABSTRACT**  
Conventionally, during the knowledge distillation process (e.g. feature distillation), an additional projector is often required to perform feature transformation due to the dimension mismatch between the teacher and the student networks. Interestingly, we discovered that even if the student and the teacher have the same feature dimensions, adding a projector still helps to improve the distillation performance. In addition, projectors even improve logit distillation if we add them to the architecture too. Inspired by these surprising findings and the general lack of understanding of the projectors in the knowledge distillation process from existing literature, this paper investigates the implicit role that projectors play but so far have been overlooked. Our empirical study shows that the student with a projector (1) obtains a better trade-off between the training accuracy and the testing accuracy compared to the student without a projector when it has the same feature dimensions as the teacher, (2) better preserves its similarity to the teacher beyond shallow and numeric resemblance, from the view of Centered Kernel Alignment (CKA), and (3) avoids being over-confident as the teacher does at the testing phase. Motivated by the positive effects of projectors, we propose a projector ensemble-based feature distillation method to further improve distillation performance. Despite the simplicity of the proposed strategy, empirical results from the evaluation of classification tasks on benchmark datasets demonstrate the superior classification performance of our method on a broad range of teacher-student pairs and verify from the aspects of CKA and model calibration that the student's features are of improved quality with the projector ensemble design.

{{</citation>}}


### (26/139) Bridging The Gaps Between Token Pruning and Full Pre-training via Masked Fine-tuning (Fengyuan Shi et al., 2023)

{{<citation>}}

Fengyuan Shi, Limin Wang. (2023)  
**Bridging The Gaps Between Token Pruning and Full Pre-training via Masked Fine-tuning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2310.17177v1)  

---


**ABSTRACT**  
Despite the success of transformers on various computer vision tasks, they suffer from excessive memory and computational cost. Some works present dynamic vision transformers to accelerate inference by pruning redundant tokens. A key to improving token pruning is using well-trained models as initialization for faster convergence and better performance. However, current base models usually adopt full image training, i.e., using full images as inputs and keeping the whole feature maps through the forward process, which causes inconsistencies with dynamic models that gradually reduce tokens, including calculation pattern, information amount and token selection strategy inconsistencies. Inspired by MAE which performs masking and reconstruction self-supervised task, we devise masked fine-tuning to bridge the gaps between pre-trained base models used for initialization and token pruning based dynamic vision transformers, by masking image patches and predicting the image class label based on left unmasked patches. Extensive experiments on ImageNet demonstrate that base models via masked fine-tuning gain strong occlusion robustness and ability against information loss. With this better initialization, Dynamic ViT achieves higher accuracies, especially under large token pruning ratios (e.g., 81.9% vs. 81.3%, and 62.3% vs. 58.9% for DeiT based Dynamic ViT/0.8 and Dynamic ViT/0.3). Moreover, we apply our method into different token pruning based dynamic vision transformers, different pre-trained models and randomly initialized models to demonstrate the generalization ability.

{{</citation>}}


### (27/139) MO-YOLO: End-to-End Multiple-Object Tracking Method with YOLO and MOTR (Liao Pan et al., 2023)

{{<citation>}}

Liao Pan, Yang Feng, Wu Di, Liu Bo, Zhang Xingle. (2023)  
**MO-YOLO: End-to-End Multiple-Object Tracking Method with YOLO and MOTR**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.17170v1)  

---


**ABSTRACT**  
This paper aims to address critical issues in the field of Multi-Object Tracking (MOT) by proposing an efficient and computationally resource-efficient end-to-end multi-object tracking model, named MO-YOLO. Traditional MOT methods typically involve two separate steps: object detection and object tracking, leading to computational complexity and error propagation issues. Recent research has demonstrated outstanding performance in end-to-end MOT models based on Transformer architectures, but they require substantial hardware support. MO-YOLO combines the strengths of YOLO and RT-DETR models to construct a high-efficiency, lightweight, and resource-efficient end-to-end multi-object tracking network, offering new opportunities in the multi-object tracking domain. On the MOT17 dataset, MOTR\cite{zeng2022motr} requires training with 8 GeForce 2080 Ti GPUs for 4 days to achieve satisfactory results, while MO-YOLO only requires 1 GeForce 2080 Ti GPU and 12 hours of training to achieve comparable performance.

{{</citation>}}


### (28/139) Low-Dimensional Gradient Helps Out-of-Distribution Detection (Yingwen Wu et al., 2023)

{{<citation>}}

Yingwen Wu, Tao Li, Xinwen Cheng, Jie Yang, Xiaolin Huang. (2023)  
**Low-Dimensional Gradient Helps Out-of-Distribution Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.17163v1)  

---


**ABSTRACT**  
Detecting out-of-distribution (OOD) samples is essential for ensuring the reliability of deep neural networks (DNNs) in real-world scenarios. While previous research has predominantly investigated the disparity between in-distribution (ID) and OOD data through forward information analysis, the discrepancy in parameter gradients during the backward process of DNNs has received insufficient attention. Existing studies on gradient disparities mainly focus on the utilization of gradient norms, neglecting the wealth of information embedded in gradient directions. To bridge this gap, in this paper, we conduct a comprehensive investigation into leveraging the entirety of gradient information for OOD detection. The primary challenge arises from the high dimensionality of gradients due to the large number of network parameters. To solve this problem, we propose performing linear dimension reduction on the gradient using a designated subspace that comprises principal components. This innovative technique enables us to obtain a low-dimensional representation of the gradient with minimal information loss. Subsequently, by integrating the reduced gradient with various existing detection score functions, our approach demonstrates superior performance across a wide range of detection tasks. For instance, on the ImageNet benchmark, our method achieves an average reduction of 11.15% in the false positive rate at 95% recall (FPR95) compared to the current state-of-the-art approach. The code would be released.

{{</citation>}}


### (29/139) Simple Baselines for Projection-based Full-reference and No-reference Point Cloud Quality Assessment (Zicheng Zhang et al., 2023)

{{<citation>}}

Zicheng Zhang, Yingjie Zhou, Wei Sun, Xiongkuo Min, Guangtao Zhai. (2023)  
**Simple Baselines for Projection-based Full-reference and No-reference Point Cloud Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.17147v1)  

---


**ABSTRACT**  
Point clouds are widely used in 3D content representation and have various applications in multimedia. However, compression and simplification processes inevitably result in the loss of quality-aware information under storage and bandwidth constraints. Therefore, there is an increasing need for effective methods to quantify the degree of distortion in point clouds. In this paper, we propose simple baselines for projection-based point cloud quality assessment (PCQA) to tackle this challenge. We use multi-projections obtained via a common cube-like projection process from the point clouds for both full-reference (FR) and no-reference (NR) PCQA tasks. Quality-aware features are extracted with popular vision backbones. The FR quality representation is computed as the similarity between the feature maps of reference and distorted projections while the NR quality representation is obtained by simply squeezing the feature maps of distorted projections with average pooling The corresponding quality representations are regressed into visual quality scores by fully-connected layers. Taking part in the ICIP 2023 PCVQA Challenge, we succeeded in achieving the top spot in four out of the five competition tracks.

{{</citation>}}


### (30/139) LP-OVOD: Open-Vocabulary Object Detection by Linear Probing (Chau Pham et al., 2023)

{{<citation>}}

Chau Pham, Truong Vu, Khoi Nguyen. (2023)  
**LP-OVOD: Open-Vocabulary Object Detection by Linear Probing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.17109v1)  

---


**ABSTRACT**  
This paper addresses the challenging problem of open-vocabulary object detection (OVOD) where an object detector must identify both seen and unseen classes in test images without labeled examples of the unseen classes in training. A typical approach for OVOD is to use joint text-image embeddings of CLIP to assign box proposals to their closest text label. However, this method has a critical issue: many low-quality boxes, such as over- and under-covered-object boxes, have the same similarity score as high-quality boxes since CLIP is not trained on exact object location information. To address this issue, we propose a novel method, LP-OVOD, that discards low-quality boxes by training a sigmoid linear classifier on pseudo labels retrieved from the top relevant region proposals to the novel text. Experimental results on COCO affirm the superior performance of our approach over the state of the art, achieving $\textbf{40.5}$ in $\text{AP}_{novel}$ using ResNet50 as the backbone and without external datasets or knowing novel classes during training. Our code will be available at https://github.com/VinAIResearch/LP-OVOD.

{{</citation>}}


### (31/139) Navigating Data Heterogeneity in Federated Learning A Semi-Supervised Approach for Object Detection (Taehyeon Kim et al., 2023)

{{<citation>}}

Taehyeon Kim, Eric Lin, Junu Lee, Christian Lau, Vaikkunth Mugunthan. (2023)  
**Navigating Data Heterogeneity in Federated Learning A Semi-Supervised Approach for Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-DC, cs.CV  
Keywords: Object Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.17097v2)  

---


**ABSTRACT**  
Federated Learning (FL) has emerged as a potent framework for training models across distributed data sources while maintaining data privacy. Nevertheless, it faces challenges with limited high-quality labels and non-IID client data, particularly in applications like autonomous driving. To address these hurdles, we navigate the uncharted waters of Semi-Supervised Federated Object Detection (SSFOD). We present a pioneering SSFOD framework, designed for scenarios where labeled data reside only at the server while clients possess unlabeled data. Notably, our method represents the inaugural implementation of SSFOD for clients with 0% labeled non-IID data, a stark contrast to previous studies that maintain some subset of labels at each client. We propose FedSTO, a two-stage strategy encompassing Selective Training followed by Orthogonally enhanced full-parameter training, to effectively address data shift (e.g. weather conditions) between server and clients. Our contributions include selectively refining the backbone of the detector to avert overfitting, orthogonality regularization to boost representation divergence, and local EMA-driven pseudo label assignment to yield high-quality pseudo labels. Extensive validation on prominent autonomous driving datasets (BDD100K, Cityscapes, and SODA10M) attests to the efficacy of our approach, demonstrating state-of-the-art results. Remarkably, FedSTO, using just 20-30% of labels, performs nearly as well as fully-supervised centralized training methods.

{{</citation>}}


### (32/139) HCT: Hybrid Convnet-Transformer for Parkinson's disease detection and severity prediction from gait (Safwen Naimi et al., 2023)

{{<citation>}}

Safwen Naimi, Wassim Bouachir, Guillaume-Alexandre Bilodeau. (2023)  
**HCT: Hybrid Convnet-Transformer for Parkinson's disease detection and severity prediction from gait**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.17078v1)  

---


**ABSTRACT**  
In this paper, we propose a novel deep learning method based on a new Hybrid ConvNet-Transformer architecture to detect and stage Parkinson's disease (PD) from gait data. We adopt a two-step approach by dividing the problem into two sub-problems. Our Hybrid ConvNet-Transformer model first distinguishes healthy versus parkinsonian patients. If the patient is parkinsonian, a multi-class Hybrid ConvNet-Transformer model determines the Hoehn and Yahr (H&Y) score to assess the PD severity stage. Our hybrid architecture exploits the strengths of both Convolutional Neural Networks (ConvNets) and Transformers to accurately detect PD and determine the severity stage. In particular, we take advantage of ConvNets to capture local patterns and correlations in the data, while we exploit Transformers for handling long-term dependencies in the input signal. We show that our hybrid method achieves superior performance when compared to other state-of-the-art methods, with a PD detection accuracy of 97% and a severity staging accuracy of 87%. Our source code is available at: https://github.com/SafwenNaimi

{{</citation>}}


### (33/139) HyperFields: Towards Zero-Shot Generation of NeRFs from Text (Sudarshan Babu et al., 2023)

{{<citation>}}

Sudarshan Babu, Richard Liu, Avery Zhou, Michael Maire, Greg Shakhnarovich, Rana Hanocka. (2023)  
**HyperFields: Towards Zero-Shot Generation of NeRFs from Text**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.17075v2)  

---


**ABSTRACT**  
We introduce HyperFields, a method for generating text-conditioned Neural Radiance Fields (NeRFs) with a single forward pass and (optionally) some fine-tuning. Key to our approach are: (i) a dynamic hypernetwork, which learns a smooth mapping from text token embeddings to the space of NeRFs; (ii) NeRF distillation training, which distills scenes encoded in individual NeRFs into one dynamic hypernetwork. These techniques enable a single network to fit over a hundred unique scenes. We further demonstrate that HyperFields learns a more general map between text and NeRFs, and consequently is capable of predicting novel in-distribution and out-of-distribution scenes -- either zero-shot or with a few finetuning steps. Finetuning HyperFields benefits from accelerated convergence thanks to the learned general map, and is capable of synthesizing novel scenes 5 to 10 times faster than existing neural optimization-based methods. Our ablation experiments show that both the dynamic architecture and NeRF distillation are critical to the expressivity of HyperFields.

{{</citation>}}


## cs.CL (40)



### (34/139) 'You Are An Expert Linguistic Annotator': Limits of LLMs as Analyzers of Abstract Meaning Representation (Allyson Ettinger et al., 2023)

{{<citation>}}

Allyson Ettinger, Jena D. Hwang, Valentina Pyatkin, Chandra Bhagavatula, Yejin Choi. (2023)  
**'You Are An Expert Linguistic Annotator': Limits of LLMs as Analyzers of Abstract Meaning Representation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Abstract Meaning Representation, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.17793v1)  

---


**ABSTRACT**  
Large language models (LLMs) show amazing proficiency and fluency in the use of language. Does this mean that they have also acquired insightful linguistic knowledge about the language, to an extent that they can serve as an "expert linguistic annotator"? In this paper, we examine the successes and limitations of the GPT-3, ChatGPT, and GPT-4 models in analysis of sentence meaning structure, focusing on the Abstract Meaning Representation (AMR; Banarescu et al. 2013) parsing formalism, which provides rich graphical representations of sentence meaning structure while abstracting away from surface forms. We compare models' analysis of this semantic structure across two settings: 1) direct production of AMR parses based on zero- and few-shot prompts, and 2) indirect partial reconstruction of AMR via metalinguistic natural language queries (e.g., "Identify the primary event of this sentence, and the predicate corresponding to that event."). Across these settings, we find that models can reliably reproduce the basic format of AMR, and can often capture core event, argument, and modifier structure -- however, model outputs are prone to frequent and major errors, and holistic analysis of parse acceptability shows that even with few-shot demonstrations, models have virtually 0% success in producing fully accurate parses. Eliciting natural language responses produces similar patterns of errors. Overall, our findings indicate that these models out-of-the-box can capture aspects of semantic structure, but there remain key limitations in their ability to support fully accurate semantic analyses or parses.

{{</citation>}}


### (35/139) Evaluation of large language models using an Indian language LGBTI+ lexicon (Aditya Joshi et al., 2023)

{{<citation>}}

Aditya Joshi, Shruta Rawat, Alpana Dange. (2023)  
**Evaluation of large language models using an Indian language LGBTI+ lexicon**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.17787v1)  

---


**ABSTRACT**  
Large language models (LLMs) are typically evaluated on the basis of task-based benchmarks such as MMLU. Such benchmarks do not examine responsible behaviour of LLMs in specific contexts. This is particularly true in the LGBTI+ context where social stereotypes may result in variation in LGBTI+ terminology. Therefore, domain-specific lexicons or dictionaries may be useful as a representative list of words against which the LLM's behaviour needs to be evaluated. This paper presents a methodology for evaluation of LLMs using an LGBTI+ lexicon in Indian languages. The methodology consists of four steps: formulating NLP tasks relevant to the expected behaviour, creating prompts that test LLMs, using the LLMs to obtain the output and, finally, manually evaluating the results. Our qualitative analysis shows that the three LLMs we experiment on are unable to detect underlying hateful content. Similarly, we observe limitations in using machine translation as means to evaluate natural language understanding in languages other than English. The methodology presented in this paper can be useful for LGBTI+ lexicons in other languages as well as other domain-specific lexicons. The work done in this paper opens avenues for responsible behaviour of LLMs, as demonstrated in the context of prevalent social perception of the LGBTI+ community.

{{</citation>}}


### (36/139) Social Contract AI: Aligning AI Assistants with Implicit Group Norms (Jan-Philipp Fränken et al., 2023)

{{<citation>}}

Jan-Philipp Fränken, Sam Kwok, Peixuan Ye, Kanishk Gandhi, Dilip Arumugam, Jared Moore, Alex Tamkin, Tobias Gerstenberg, Noah D. Goodman. (2023)  
**Social Contract AI: Aligning AI Assistants with Implicit Group Norms**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17769v1)  

---


**ABSTRACT**  
We explore the idea of aligning an AI assistant by inverting a model of users' (unknown) preferences from observed interactions. To validate our proposal, we run proof-of-concept simulations in the economic ultimatum game, formalizing user preferences as policies that guide the actions of simulated players. We find that the AI assistant accurately aligns its behavior to match standard policies from the economic literature (e.g., selfish, altruistic). However, the assistant's learned policies lack robustness and exhibit limited generalization in an out-of-distribution setting when confronted with a currency (e.g., grams of medicine) that was not included in the assistant's training distribution. Additionally, we find that when there is inconsistency in the relationship between language use and an unknown policy (e.g., an altruistic policy combined with rude language), the assistant's learning of the policy is slowed. Overall, our preliminary results suggest that developing simulation frameworks in which AI assistants need to infer preferences from diverse users can provide a valuable approach for studying practical alignment questions.

{{</citation>}}


### (37/139) A Framework for Automated Measurement of Responsible AI Harms in Generative AI Applications (Ahmed Magooda et al., 2023)

{{<citation>}}

Ahmed Magooda, Alec Helyar, Kyle Jackson, David Sullivan, Chad Atalla, Emily Sheng, Dan Vann, Richard Edgar, Hamid Palangi, Roman Lutz, Hongliang Kong, Vincent Yun, Eslam Kamal, Federico Zarfati, Hanna Wallach, Sarah Bird, Mei Chen. (2023)  
**A Framework for Automated Measurement of Responsible AI Harms in Generative AI Applications**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.17750v1)  

---


**ABSTRACT**  
We present a framework for the automated measurement of responsible AI (RAI) metrics for large language models (LLMs) and associated products and services. Our framework for automatically measuring harms from LLMs builds on existing technical and sociotechnical expertise and leverages the capabilities of state-of-the-art LLMs, such as GPT-4. We use this framework to run through several case studies investigating how different LLMs may violate a range of RAI-related principles. The framework may be employed alongside domain-specific sociotechnical expertise to create measurements for new harm areas in the future. By implementing this framework, we aim to enable more advanced harm measurement efforts and further the responsible use of LLMs.

{{</citation>}}


### (38/139) ArchBERT: Bi-Modal Understanding of Neural Architectures and Natural Languages (Mohammad Akbari et al., 2023)

{{<citation>}}

Mohammad Akbari, Saeed Ranjbar Alvar, Behnam Kamranian, Amin Banitalebi-Dehkordi, Yong Zhang. (2023)  
**ArchBERT: Bi-Modal Understanding of Neural Architectures and Natural Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.17737v1)  

---


**ABSTRACT**  
Building multi-modal language models has been a trend in the recent years, where additional modalities such as image, video, speech, etc. are jointly learned along with natural languages (i.e., textual information). Despite the success of these multi-modal language models with different modalities, there is no existing solution for neural network architectures and natural languages. Providing neural architectural information as a new modality allows us to provide fast architecture-2-text and text-2-architecture retrieval/generation services on the cloud with a single inference. Such solution is valuable in terms of helping beginner and intermediate ML users to come up with better neural architectures or AutoML approaches with a simple text query. In this paper, we propose ArchBERT, a bi-modal model for joint learning and understanding of neural architectures and natural languages, which opens up new avenues for research in this area. We also introduce a pre-training strategy named Masked Architecture Modeling (MAM) for a more generalized joint learning. Moreover, we introduce and publicly release two new bi-modal datasets for training and validating our methods. The ArchBERT's performance is verified through a set of numerical experiments on different downstream tasks such as architecture-oriented reasoning, question answering, and captioning (summarization). Datasets, codes, and demos are available supplementary materials.

{{</citation>}}


### (39/139) Investigating Multilingual Coreference Resolution by Universal Annotations (Haixia Chai et al., 2023)

{{<citation>}}

Haixia Chai, Michael Strube. (2023)  
**Investigating Multilingual Coreference Resolution by Universal Annotations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2310.17734v1)  

---


**ABSTRACT**  
Multilingual coreference resolution (MCR) has been a long-standing and challenging task. With the newly proposed multilingual coreference dataset, CorefUD (Nedoluzhko et al., 2022), we conduct an investigation into the task by using its harmonized universal morphosyntactic and coreference annotations. First, we study coreference by examining the ground truth data at different linguistic levels, namely mention, entity and document levels, and across different genres, to gain insights into the characteristics of coreference across multiple languages. Second, we perform an error analysis of the most challenging cases that the SotA system fails to resolve in the CRAC 2022 shared task using the universal annotations. Last, based on this analysis, we extract features from universal morphosyntactic annotations and integrate these features into a baseline system to assess their potential benefits for the MCR task. Our results show that our best configuration of features improves the baseline by 0.9% F1 score.

{{</citation>}}


### (40/139) Nearest Neighbor Search over Vectorized Lexico-Syntactic Patterns for Relation Extraction from Financial Documents (Pawan Kumar Rajpoot et al., 2023)

{{<citation>}}

Pawan Kumar Rajpoot, Ankur Parikh. (2023)  
**Nearest Neighbor Search over Vectorized Lexico-Syntactic Patterns for Relation Extraction from Financial Documents**  

---
Primary Category: cs.CL  
Categories: cs-CE, cs-CL, cs.CL  
Keywords: Financial, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2310.17714v1)  

---


**ABSTRACT**  
Relation extraction (RE) has achieved remarkable progress with the help of pre-trained language models. However, existing RE models are usually incapable of handling two situations: implicit expressions and long-tail relation classes, caused by language complexity and data sparsity. Further, these approaches and models are largely inaccessible to users who don't have direct access to large language models (LLMs) and/or infrastructure for supervised training or fine-tuning. Rule-based systems also struggle with implicit expressions. Apart from this, Real world financial documents such as various 10-X reports (including 10-K, 10-Q, etc.) of publicly traded companies pose another challenge to rule-based systems in terms of longer and complex sentences. In this paper, we introduce a simple approach that consults training relations at test time through a nearest-neighbor search over dense vectors of lexico-syntactic patterns and provides a simple yet effective means to tackle the above issues. We evaluate our approach on REFinD and show that our method achieves state-of-the-art performance. We further show that it can provide a good start for human in the loop setup when a small number of annotations are available and it is also beneficial when domain experts can provide high quality patterns.

{{</citation>}}


### (41/139) Is Explanation the Cure? Misinformation Mitigation in the Short Term and Long Term (Yi-Li Hsu et al., 2023)

{{<citation>}}

Yi-Li Hsu, Shih-Chieh Dai, Aiping Xiong, Lun-Wei Ku. (2023)  
**Is Explanation the Cure? Misinformation Mitigation in the Short Term and Long Term**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, NLP  
[Paper Link](http://arxiv.org/abs/2310.17711v1)  

---


**ABSTRACT**  
With advancements in natural language processing (NLP) models, automatic explanation generation has been proposed to mitigate misinformation on social media platforms in addition to adding warning labels to identified fake news. While many researchers have focused on generating good explanations, how these explanations can really help humans combat fake news is under-explored. In this study, we compare the effectiveness of a warning label and the state-of-the-art counterfactual explanations generated by GPT-4 in debunking misinformation. In a two-wave, online human-subject study, participants (N = 215) were randomly assigned to a control group in which false contents are shown without any intervention, a warning tag group in which the false claims were labeled, or an explanation group in which the false contents were accompanied by GPT-4 generated explanations. Our results show that both interventions significantly decrease participants' self-reported belief in fake claims in an equivalent manner for the short-term and long-term. We discuss the implications of our findings and directions for future NLP-based misinformation debunking strategies.

{{</citation>}}


### (42/139) The impact of using an AI chatbot to respond to patient messages (Shan Chen et al., 2023)

{{<citation>}}

Shan Chen, Marco Guevara, Shalini Moningi, Frank Hoebers, Hesham Elhalawani, Benjamin H. Kann, Fallon E. Chipidza, Jonathan Leeman, Hugo J. W. L. Aerts, Timothy Miller, Guergana K. Savova, Raymond H. Mak, Maryam Lustberg, Majid Afshar, Danielle S. Bitterman. (2023)  
**The impact of using an AI chatbot to respond to patient messages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.17703v1)  

---


**ABSTRACT**  
Documentation burden is a major contributor to clinician burnout, which is rising nationally and is an urgent threat to our ability to care for patients. Artificial intelligence (AI) chatbots, such as ChatGPT, could reduce clinician burden by assisting with documentation. Although many hospitals are actively integrating such systems into electronic medical record systems, AI chatbots utility and impact on clinical decision-making have not been studied for this intended use. We are the first to examine the utility of large language models in assisting clinicians draft responses to patient questions. In our two-stage cross-sectional study, 6 oncologists responded to 100 realistic synthetic cancer patient scenarios and portal messages developed to reflect common medical situations, first manually, then with AI assistance.   We find AI-assisted responses were longer, less readable, but provided acceptable drafts without edits 58% of time. AI assistance improved efficiency 77% of time, with low harm risk (82% safe). However, 7.7% unedited AI responses could severely harm. In 31% cases, physicians thought AI drafts were human-written. AI assistance led to more patient education recommendations, fewer clinical actions than manual responses. Results show promise for AI to improve clinician efficiency and patient care through assisting documentation, if used judiciously. Monitoring model outputs and human-AI interaction remains crucial for safe implementation.

{{</citation>}}


### (43/139) torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP (Yoshitomo Matsubara, 2023)

{{<citation>}}

Yoshitomo Matsubara. (2023)  
**torchdistill Meets Hugging Face Libraries for Reproducible, Coding-Free Deep Learning Studies: A Case Study on NLP**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: BERT, GLUE, NLP  
[Paper Link](http://arxiv.org/abs/2310.17644v1)  

---


**ABSTRACT**  
Reproducibility in scientific work has been becoming increasingly important in research communities such as machine learning, natural language processing, and computer vision communities due to the rapid development of the research domains supported by recent advances in deep learning. In this work, we present a significantly upgraded version of torchdistill, a modular-driven coding-free deep learning framework significantly upgraded from the initial release, which supports only image classification and object detection tasks for reproducible knowledge distillation experiments. To demonstrate that the upgraded framework can support more tasks with third-party libraries, we reproduce the GLUE benchmark results of BERT models using a script based on the upgraded torchdistill, harmonizing with various Hugging Face libraries. All the 27 fine-tuned BERT models and configurations to reproduce the results are published at Hugging Face, and the model weights have already been widely used in research communities. We also reimplement popular small-sized models and new knowledge distillation methods and perform additional experiments for computer vision tasks.

{{</citation>}}


### (44/139) JudgeLM: Fine-tuned Large Language Models are Scalable Judges (Lianghui Zhu et al., 2023)

{{<citation>}}

Lianghui Zhu, Xinggang Wang, Xinlong Wang. (2023)  
**JudgeLM: Fine-tuned Large Language Models are Scalable Judges**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.17631v1)  

---


**ABSTRACT**  
Evaluating Large Language Models (LLMs) in open-ended scenarios is challenging because existing benchmarks and metrics can not measure them comprehensively. To address this problem, we propose to fine-tune LLMs as scalable judges (JudgeLM) to evaluate LLMs efficiently and effectively in open-ended benchmarks. We first propose a comprehensive, large-scale, high-quality dataset containing task seeds, LLMs-generated answers, and GPT-4-generated judgments for fine-tuning high-performance judges, as well as a new benchmark for evaluating the judges. We train JudgeLM at different scales from 7B, 13B, to 33B parameters, and conduct a systematic analysis of its capabilities and behaviors. We then analyze the key biases in fine-tuning LLM as a judge and consider them as position bias, knowledge bias, and format bias. To address these issues, JudgeLM introduces a bag of techniques including swap augmentation, reference support, and reference drop, which clearly enhance the judge's performance. JudgeLM obtains the state-of-the-art judge performance on both the existing PandaLM benchmark and our proposed new benchmark. Our JudgeLM is efficient and the JudgeLM-7B only needs 3 minutes to judge 5K samples with 8 A100 GPUs. JudgeLM obtains high agreement with the teacher judge, achieving an agreement exceeding 90% that even surpasses human-to-human agreement. JudgeLM also demonstrates extended capabilities in being judges of the single answer, multimodal models, multiple answers, and multi-turn chat.

{{</citation>}}


### (45/139) InstOptima: Evolutionary Multi-objective Instruction Optimization via Large Language Model-based Instruction Operators (Heng Yang et al., 2023)

{{<citation>}}

Heng Yang, Ke Li. (2023)  
**InstOptima: Evolutionary Multi-objective Instruction Optimization via Large Language Model-based Instruction Operators**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17630v1)  

---


**ABSTRACT**  
Instruction-based language modeling has received significant attention in pretrained language models. However, the efficiency of instruction engineering remains low and hinders the development of instruction studies. Recent studies have focused on automating instruction generation, but they primarily aim to improve performance without considering other crucial objectives that impact instruction quality, such as instruction length and perplexity. Therefore, we propose a novel approach (i.e., InstOptima) that treats instruction generation as an evolutionary multi-objective optimization problem. In contrast to text edition-based methods, our approach utilizes a large language model (LLM) to simulate instruction operators, including mutation and crossover. Furthermore, we introduce an objective-guided mechanism for these operators, allowing the LLM to comprehend the objectives and enhance the quality of the generated instructions. Experimental results demonstrate improved fine-tuning performance and the generation of a diverse set of high-quality instructions.

{{</citation>}}


### (46/139) Proving Test Set Contamination in Black Box Language Models (Yonatan Oren et al., 2023)

{{<citation>}}

Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori B. Hashimoto. (2023)  
**Proving Test Set Contamination in Black Box Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17623v1)  

---


**ABSTRACT**  
Large language models are trained on vast amounts of internet data, prompting concerns and speculation that they have memorized public benchmarks. Going from speculation to proof of contamination is challenging, as the pretraining data used by proprietary models are often not publicly accessible. We show that it is possible to provide provable guarantees of test set contamination in language models without access to pretraining data or model weights. Our approach leverages the fact that when there is no data contamination, all orderings of an exchangeable benchmark should be equally likely. In contrast, the tendency for language models to memorize example order means that a contaminated language model will find certain canonical orderings to be much more likely than others. Our test flags potential contamination whenever the likelihood of a canonically ordered benchmark dataset is significantly higher than the likelihood after shuffling the examples. We demonstrate that our procedure is sensitive enough to reliably prove test set contamination in challenging situations, including models as small as 1.4 billion parameters, on small test sets of only 1000 examples, and datasets that appear only a few times in the pretraining corpus. Using our test, we audit five popular publicly accessible language models for test set contamination and find little evidence for pervasive contamination.

{{</citation>}}


### (47/139) LeCaRDv2: A Large-Scale Chinese Legal Case Retrieval Dataset (Haitao Li et al., 2023)

{{<citation>}}

Haitao Li, Yunqiu Shao, Yueyue Wu, Qingyao Ai, Yixiao Ma, Yiqun Liu. (2023)  
**LeCaRDv2: A Large-Scale Chinese Legal Case Retrieval Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2310.17609v1)  

---


**ABSTRACT**  
As an important component of intelligent legal systems, legal case retrieval plays a critical role in ensuring judicial justice and fairness. However, the development of legal case retrieval technologies in the Chinese legal system is restricted by three problems in existing datasets: limited data size, narrow definitions of legal relevance, and naive candidate pooling strategies used in data sampling. To alleviate these issues, we introduce LeCaRDv2, a large-scale Legal Case Retrieval Dataset (version 2). It consists of 800 queries and 55,192 candidates extracted from 4.3 million criminal case documents. To the best of our knowledge, LeCaRDv2 is one of the largest Chinese legal case retrieval datasets, providing extensive coverage of criminal charges. Additionally, we enrich the existing relevance criteria by considering three key aspects: characterization, penalty, procedure. This comprehensive criteria enriches the dataset and may provides a more holistic perspective. Furthermore, we propose a two-level candidate set pooling strategy that effectively identify potential candidates for each query case. It's important to note that all cases in the dataset have been annotated by multiple legal experts specializing in criminal law. Their expertise ensures the accuracy and reliability of the annotations. We evaluate several state-of-the-art retrieval models at LeCaRDv2, demonstrating that there is still significant room for improvement in legal case retrieval. The details of LeCaRDv2 can be found at the anonymous website https://github.com/anonymous1113243/LeCaRDv2.

{{</citation>}}


### (48/139) Lil-Bevo: Explorations of Strategies for Training Language Models in More Humanlike Ways (Venkata S Govindarajan et al., 2023)

{{<citation>}}

Venkata S Govindarajan, Juan Diego Rodriguez, Kaj Bostrom, Kyle Mahowald. (2023)  
**Lil-Bevo: Explorations of Strategies for Training Language Models in More Humanlike Ways**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17591v1)  

---


**ABSTRACT**  
We present Lil-Bevo, our submission to the BabyLM Challenge. We pretrained our masked language models with three ingredients: an initial pretraining with music data, training on shorter sequences before training on longer ones, and masking specific tokens to target some of the BLiMP subtasks. Overall, our baseline models performed above chance, but far below the performance levels of larger LLMs trained on more data. We found that training on short sequences performed better than training on longer sequences.Pretraining on music may help performance marginally, but, if so, the effect seems small. Our targeted Masked Language Modeling augmentation did not seem to improve model performance in general, but did seem to help on some of the specific BLiMP tasks that we were targeting (e.g., Negative Polarity Items). Training performant LLMs on small amounts of data is a difficult but potentially informative task. While some of our techniques showed some promise, more work is needed to explore whether they can improve performance more than the modest gains here. Our code is available at https://github.com/venkatasg/Lil-Bevo and out models at https://huggingface.co/collections/venkatasg/babylm-653591cdb66f4bf68922873a

{{</citation>}}


### (49/139) An Open Source Data Contamination Report for Llama Series Models (Yucheng Li, 2023)

{{<citation>}}

Yucheng Li. (2023)  
**An Open Source Data Contamination Report for Llama Series Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.17589v1)  

---


**ABSTRACT**  
Data contamination in language model evaluation is increasingly prevalent as the popularity of large language models. It allows models to "cheat" via memorisation instead of displaying true capabilities. Therefore, contamination analysis has became an crucial part of reliable model evaluation to validate results. However, existing contamination analysis is usually conducted internally by LLM developers and often lacks transparency and completeness. This paper present an open source data contamination reports for the Llama series models. We analyse six popular multi-choice QA benchmarks and quantify their overlapping with the training set of Llama. Various levels of contamination ranging from 1\% to 8.7\% are found across benchmarks. Our comparison also reveals that Llama models can gain over 5\% higher accuracy on contaminated subsets versus clean subsets. Data and code are available at: https://github.com/liyucheng09/Contamination_Detector.

{{</citation>}}


### (50/139) Global Voices, Local Biases: Socio-Cultural Prejudices across Languages (Anjishnu Mukherjee et al., 2023)

{{<citation>}}

Anjishnu Mukherjee, Chahat Raj, Ziwei Zhu, Antonios Anastasopoulos. (2023)  
**Global Voices, Local Biases: Socio-Cultural Prejudices across Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Embedding, Word Embedding  
[Paper Link](http://arxiv.org/abs/2310.17586v1)  

---


**ABSTRACT**  
Human biases are ubiquitous but not uniform: disparities exist across linguistic, cultural, and societal borders. As large amounts of recent literature suggest, language models (LMs) trained on human data can reflect and often amplify the effects of these social biases. However, the vast majority of existing studies on bias are heavily skewed towards Western and European languages. In this work, we scale the Word Embedding Association Test (WEAT) to 24 languages, enabling broader studies and yielding interesting findings about LM bias. We additionally enhance this data with culturally relevant information for each language, capturing local contexts on a global scale. Further, to encompass more widely prevalent societal biases, we examine new bias dimensions across toxicity, ableism, and more. Moreover, we delve deeper into the Indian linguistic landscape, conducting a comprehensive regional bias analysis across six prevalent Indian languages. Finally, we highlight the significance of these social biases and the new dimensions through an extensive comparison of embedding methods, reinforcing the need to address them in pursuit of more equitable language models. All code, data and results are available here: https://github.com/iamshnoo/weathub.

{{</citation>}}


### (51/139) Can LLMs Grade Short-answer Reading Comprehension Questions : Foundational Literacy Assessment in LMICs (Owen Henkel et al., 2023)

{{<citation>}}

Owen Henkel, Libby Hills, Bill Roberts, Joshua McGrane. (2023)  
**Can LLMs Grade Short-answer Reading Comprehension Questions : Foundational Literacy Assessment in LMICs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.18373v1)  

---


**ABSTRACT**  
This paper presents emerging evidence of using generative large language models (i.e., GPT-4) to reliably evaluate short-answer reading comprehension questions. Specifically, we explore how various configurations of generative (LLMs) are able to evaluate student responses from a new dataset, drawn from a battery of reading assessments conducted with over 150 students in Ghana. As this dataset is novel and hence not used in training runs of GPT, it offers an opportunity to test for domain shift and evaluate the generalizability of generative LLMs, which are predominantly designed and trained on data from high-income North American countries. We found that GPT-4, with minimal prompt engineering performed extremely well on evaluating the novel dataset (Quadratic Weighted Kappa 0.923, F1 0.88), substantially outperforming transfer-learning based approaches, and even exceeding expert human raters (Quadratic Weighted Kappa 0.915, F1 0.87). To the best of our knowledge, our work is the first to empirically evaluate the performance of generative LLMs on short-answer reading comprehension questions, using real student data, and suggests that generative LLMs have the potential to reliably evaluate foundational literacy. Currently the assessment of formative literacy and numeracy is infrequent in many low and middle-income countries (LMICs) due to the cost and operational complexities of conducting them at scale. Automating the grading process for reading assessment could enable wider usage, and in turn improve decision-making regarding curricula, school management, and teaching practice at the classroom level. Importantly, in contrast transfer learning based approaches, generative LLMs generalize well and the technical barriers to their use are low, making them more feasible to implement and scale in lower resource educational contexts.

{{</citation>}}


### (52/139) Skill-Mix: a Flexible and Expandable Family of Evaluations for AI models (Dingli Yu et al., 2023)

{{<citation>}}

Dingli Yu, Simran Kaur, Arushi Gupta, Jonah Brown-Cohen, Anirudh Goyal, Sanjeev Arora. (2023)  
**Skill-Mix: a Flexible and Expandable Family of Evaluations for AI models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-NE, cs.CL  
Keywords: AI, GPT, GPT-4, LLaMA  
[Paper Link](http://arxiv.org/abs/2310.17567v1)  

---


**ABSTRACT**  
With LLMs shifting their role from statistical modeling of language to serving as general-purpose AI agents, how should LLM evaluations change? Arguably, a key ability of an AI agent is to flexibly combine, as needed, the basic skills it has learned. The capability to combine skills plays an important role in (human) pedagogy and also in a paper on emergence phenomena (Arora & Goyal, 2023).   This work introduces Skill-Mix, a new evaluation to measure ability to combine skills. Using a list of $N$ skills the evaluator repeatedly picks random subsets of $k$ skills and asks the LLM to produce text combining that subset of skills. Since the number of subsets grows like $N^k$, for even modest $k$ this evaluation will, with high probability, require the LLM to produce text significantly different from any text in the training set. The paper develops a methodology for (a) designing and administering such an evaluation, and (b) automatic grading (plus spot-checking by humans) of the results using GPT-4 as well as the open LLaMA-2 70B model.   Administering a version of to popular chatbots gave results that, while generally in line with prior expectations, contained surprises. Sizeable differences exist among model capabilities that are not captured by their ranking on popular LLM leaderboards ("cramming for the leaderboard"). Furthermore, simple probability calculations indicate that GPT-4's reasonable performance on $k=5$ is suggestive of going beyond "stochastic parrot" behavior (Bender et al., 2021), i.e., it combines skills in ways that it had not seen during training.   We sketch how the methodology can lead to a Skill-Mix based eco-system of open evaluations for AI capabilities of future models.

{{</citation>}}


### (53/139) Can large language models replace humans in the systematic review process? Evaluating GPT-4's efficacy in screening and extracting data from peer-reviewed and grey literature in multiple languages (Qusai Khraisha et al., 2023)

{{<citation>}}

Qusai Khraisha, Sophie Put, Johanna Kappenberg, Azza Warraitch, Kristin Hadfield. (2023)  
**Can large language models replace humans in the systematic review process? Evaluating GPT-4's efficacy in screening and extracting data from peer-reviewed and grey literature in multiple languages**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.17526v2)  

---


**ABSTRACT**  
Systematic reviews are vital for guiding practice, research, and policy, yet they are often slow and labour-intensive. Large language models (LLMs) could offer a way to speed up and automate systematic reviews, but their performance in such tasks has not been comprehensively evaluated against humans, and no study has tested GPT-4, the biggest LLM so far. This pre-registered study evaluates GPT-4's capability in title/abstract screening, full-text review, and data extraction across various literature types and languages using a 'human-out-of-the-loop' approach. Although GPT-4 had accuracy on par with human performance in most tasks, results were skewed by chance agreement and dataset imbalance. After adjusting for these, there was a moderate level of performance for data extraction, and - barring studies that used highly reliable prompts - screening performance levelled at none to moderate for different stages and languages. When screening full-text literature using highly reliable prompts, GPT-4's performance was 'almost perfect.' Penalising GPT-4 for missing key studies using highly reliable prompts improved its performance even more. Our findings indicate that, currently, substantial caution should be used if LLMs are being used to conduct systematic reviews, but suggest that, for certain systematic review tasks delivered under reliable prompts, LLMs can rival human performance.

{{</citation>}}


### (54/139) The Validity of Evaluation Results: Assessing Concurrence Across Compositionality Benchmarks (Kaiser Sun et al., 2023)

{{<citation>}}

Kaiser Sun, Adina Williams, Dieuwke Hupkes. (2023)  
**The Validity of Evaluation Results: Assessing Concurrence Across Compositionality Benchmarks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.17514v1)  

---


**ABSTRACT**  
NLP models have progressed drastically in recent years, according to numerous datasets proposed to evaluate performance. Questions remain, however, about how particular dataset design choices may impact the conclusions we draw about model capabilities. In this work, we investigate this question in the domain of compositional generalization. We examine the performance of six modeling approaches across 4 datasets, split according to 8 compositional splitting strategies, ranking models by 18 compositional generalization splits in total. Our results show that: i) the datasets, although all designed to evaluate compositional generalization, rank modeling approaches differently; ii) datasets generated by humans align better with each other than they with synthetic datasets, or than synthetic datasets among themselves; iii) generally, whether datasets are sampled from the same source is more predictive of the resulting model ranking than whether they maintain the same interpretation of compositionality; and iv) which lexical items are used in the data can strongly impact conclusions. Overall, our results demonstrate that much work remains to be done when it comes to assessing whether popular evaluation datasets measure what they intend to measure, and suggest that elucidating more rigorous standards for establishing the validity of evaluation sets could benefit the field.

{{</citation>}}


### (55/139) Improving Zero-shot Reader by Reducing Distractions from Irrelevant Documents in Open-Domain Question Answering (Sukmin Cho et al., 2023)

{{<citation>}}

Sukmin Cho, Jeong yeon Seo, Soyeong Jeong, Jong C. Park. (2023)  
**Improving Zero-shot Reader by Reducing Distractions from Irrelevant Documents in Open-Domain Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.17490v1)  

---


**ABSTRACT**  
Large language models (LLMs) enable zero-shot approaches in open-domain question answering (ODQA), yet with limited advancements as the reader is compared to the retriever. This study aims at the feasibility of a zero-shot reader that addresses the challenges of computational cost and the need for labeled data. We find that LLMs are distracted due to irrelevant documents in the retrieved set and the overconfidence of the generated answers when they are exploited as zero-shot readers. To tackle these problems, we mitigate the impact of such documents via Distraction-aware Answer Selection (DAS) with a negation-based instruction and score adjustment for proper answer selection. Experimental results show that our approach successfully handles distraction across diverse scenarios, enhancing the performance of zero-shot readers. Furthermore, unlike supervised readers struggling with unseen data, zero-shot readers demonstrate outstanding transferability without any training.

{{</citation>}}


### (56/139) Dialect Adaptation and Data Augmentation for Low-Resource ASR: TalTech Systems for the MADASR 2023 Challenge (Tanel Alumäe et al., 2023)

{{<citation>}}

Tanel Alumäe, Jiaming Kong, Daniil Robnikov. (2023)  
**Dialect Adaptation and Data Augmentation for Low-Resource ASR: TalTech Systems for the MADASR 2023 Challenge**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: Augmentation, Low-Resource  
[Paper Link](http://arxiv.org/abs/2310.17448v1)  

---


**ABSTRACT**  
This paper describes Tallinn University of Technology (TalTech) systems developed for the ASRU MADASR 2023 Challenge. The challenge focuses on automatic speech recognition of dialect-rich Indian languages with limited training audio and text data. TalTech participated in two tracks of the challenge: Track 1 that allowed using only the provided training data and Track 3 which allowed using additional audio data. In both tracks, we relied on wav2vec2.0 models. Our methodology diverges from the traditional procedure of finetuning pretrained wav2vec2.0 models in two key points: firstly, through the implementation of the aligned data augmentation technique to enhance the linguistic diversity of the training data, and secondly, via the application of deep prefix tuning for dialect adaptation of wav2vec2.0 models. In both tracks, our approach yielded significant improvements over the provided baselines, achieving the lowest word error rates across all participating teams.

{{</citation>}}


### (57/139) ''Fifty Shades of Bias'': Normative Ratings of Gender Bias in GPT Generated English Text (Rishav Hada et al., 2023)

{{<citation>}}

Rishav Hada, Agrima Seth, Harshita Diddee, Kalika Bali. (2023)  
**''Fifty Shades of Bias'': Normative Ratings of Gender Bias in GPT Generated English Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, GPT  
[Paper Link](http://arxiv.org/abs/2310.17428v1)  

---


**ABSTRACT**  
Language serves as a powerful tool for the manifestation of societal belief systems. In doing so, it also perpetuates the prevalent biases in our society. Gender bias is one of the most pervasive biases in our society and is seen in online and offline discourses. With LLMs increasingly gaining human-like fluency in text generation, gaining a nuanced understanding of the biases these systems can generate is imperative. Prior work often treats gender bias as a binary classification task. However, acknowledging that bias must be perceived at a relative scale; we investigate the generation and consequent receptivity of manual annotators to bias of varying degrees. Specifically, we create the first dataset of GPT-generated English text with normative ratings of gender bias. Ratings were obtained using Best--Worst Scaling -- an efficient comparative annotation framework. Next, we systematically analyze the variation of themes of gender biases in the observed ranking and show that identity-attack is most closely related to gender bias. Finally, we show the performance of existing automated models trained on related concepts on our dataset.

{{</citation>}}


### (58/139) Harnessing GPT-3.5-turbo for Rhetorical Role Prediction in Legal Cases (Anas Belfathi et al., 2023)

{{<citation>}}

Anas Belfathi, Nicolas Hernandez, Laura Monceaux. (2023)  
**Harnessing GPT-3.5-turbo for Rhetorical Role Prediction in Legal Cases**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-3.5, Legal  
[Paper Link](http://arxiv.org/abs/2310.17413v1)  

---


**ABSTRACT**  
We propose a comprehensive study of one-stage elicitation techniques for querying a large pre-trained generative transformer (GPT-3.5-turbo) in the rhetorical role prediction task of legal cases. This task is known as requiring textual context to be addressed. Our study explores strategies such as zero-few shots, task specification with definitions and clarification of annotation ambiguities, textual context and reasoning with general prompts and specific questions. We show that the number of examples, the definition of labels, the presentation of the (labelled) textual context and specific questions about this context have a positive influence on the performance of the model. Given non-equivalent test set configurations, we observed that prompting with a few labelled examples from direct context can lead the model to a better performance than a supervised fined-tuned multi-class classifier based on the BERT encoder (weighted F1 score of = 72%). But there is still a gap to reach the performance of the best systems = 86%) in the LegalEval 2023 task which, on the other hand, require dedicated resources, architectures and training.

{{</citation>}}


### (59/139) ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation (Zi Lin et al., 2023)

{{<citation>}}

Zi Lin, Zihan Wang, Yongqi Tong, Yangkun Wang, Yuxin Guo, Yujia Wang, Jingbo Shang. (2023)  
**ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17389v1)  

---


**ABSTRACT**  
Despite remarkable advances that large language models have achieved in chatbots, maintaining a non-toxic user-AI interactive environment has become increasingly critical nowadays. However, previous efforts in toxicity detection have been mostly based on benchmarks derived from social media content, leaving the unique challenges inherent to real-world user-AI interactions insufficiently explored. In this work, we introduce ToxicChat, a novel benchmark based on real user queries from an open-source chatbot. This benchmark contains the rich, nuanced phenomena that can be tricky for current toxicity detection models to identify, revealing a significant domain difference compared to social media content. Our systematic evaluation of models trained on existing toxicity datasets has shown their shortcomings when applied to this unique domain of ToxicChat. Our work illuminates the potentially overlooked challenges of toxicity detection in real-world user-AI conversations. In the future, ToxicChat can be a valuable resource to drive further advancements toward building a safe and healthy environment for user-AI interactions.

{{</citation>}}


### (60/139) Cultural Adaptation of Recipes (Yong Cao et al., 2023)

{{<citation>}}

Yong Cao, Yova Kementchedjhieva, Ruixiang Cui, Antonia Karamolegkou, Li Zhou, Megan Dare, Lucia Donatelli, Daniel Hershcovich. (2023)  
**Cultural Adaptation of Recipes**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.17353v1)  

---


**ABSTRACT**  
Building upon the considerable advances in Large Language Models (LLMs), we are now equipped to address more sophisticated tasks demanding a nuanced understanding of cross-cultural contexts. A key example is recipe adaptation, which goes beyond simple translation to include a grasp of ingredients, culinary techniques, and dietary preferences specific to a given culture. We introduce a new task involving the translation and cultural adaptation of recipes between Chinese and English-speaking cuisines. To support this investigation, we present CulturalRecipes, a unique dataset comprised of automatically paired recipes written in Mandarin Chinese and English. This dataset is further enriched with a human-written and curated test set. In this intricate task of cross-cultural recipe adaptation, we evaluate the performance of various methods, including GPT-4 and other LLMs, traditional machine translation, and information retrieval techniques. Our comprehensive analysis includes both automatic and human evaluation metrics. While GPT-4 exhibits impressive abilities in adapting Chinese recipes into English, it still lags behind human expertise when translating English recipes into Chinese. This underscores the multifaceted nature of cultural adaptations. We anticipate that these insights will significantly contribute to future research on culturally-aware language models and their practical application in culturally diverse contexts.

{{</citation>}}


### (61/139) ACT-SQL: In-Context Learning for Text-to-SQL with Automatically-Generated Chain-of-Thought (Hanchong Zhang et al., 2023)

{{<citation>}}

Hanchong Zhang, Ruisheng Cao, Lu Chen, Hongshen Xu, Kai Yu. (2023)  
**ACT-SQL: In-Context Learning for Text-to-SQL with Automatically-Generated Chain-of-Thought**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17342v1)  

---


**ABSTRACT**  
Recently Large Language Models (LLMs) have been proven to have strong abilities in various domains and tasks. We study the problem of prompt designing in the text-to-SQL task and attempt to improve the LLMs' reasoning ability when generating SQL queries. Besides the trivial few-shot in-context learning setting, we design our chain-of-thought (CoT) prompt with a similar method to schema linking. We provide a method named ACT-SQL to automatically generate auto-CoT exemplars and thus the whole process doesn't need manual labeling. Our approach is cost-saving since we only use the LLMs' API call once when generating one SQL query. Furthermore, we extend our in-context learning method to the multi-turn text-to-SQL task. The experiment results show that the LLMs' performance can benefit from our ACT-SQL approach. Our approach achieves SOTA performance on the Spider dev set among existing in-context learning approaches.

{{</citation>}}


### (62/139) Arabic Fine-Grained Entity Recognition (Haneen Liqreina et al., 2023)

{{<citation>}}

Haneen Liqreina, Mustafa Jarrar, Mohammed Khalilia, Ahmed Oumar El-Shangiti, Muhammad AbdulMageed. (2023)  
**Arabic Fine-Grained Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NER  
[Paper Link](http://arxiv.org/abs/2310.17333v1)  

---


**ABSTRACT**  
Traditional NER systems are typically trained to recognize coarse-grained entities, and less attention is given to classifying entities into a hierarchy of fine-grained lower-level subtypes. This article aims to advance Arabic NER with fine-grained entities. We chose to extend Wojood (an open-source Nested Arabic Named Entity Corpus) with subtypes. In particular, four main entity types in Wojood, geopolitical entity (GPE), location (LOC), organization (ORG), and facility (FAC), are extended with 31 subtypes. To do this, we first revised Wojood's annotations of GPE, LOC, ORG, and FAC to be compatible with the LDC's ACE guidelines, which yielded 5, 614 changes. Second, all mentions of GPE, LOC, ORG, and FAC (~44K) in Wojood are manually annotated with the LDC's ACE sub-types. We refer to this extended version of Wojood as WojoodF ine. To evaluate our annotations, we measured the inter-annotator agreement (IAA) using both Cohen's Kappa and F1 score, resulting in 0.9861 and 0.9889, respectively. To compute the baselines of WojoodF ine, we fine-tune three pre-trained Arabic BERT encoders in three settings: flat NER, nested NER and nested NER with subtypes and achieved F1 score of 0.920, 0.866, and 0.885, respectively. Our corpus and models are open-source and available at https://sina.birzeit.edu/wojood/.

{{</citation>}}


### (63/139) An Ensemble Method Based on the Combination of Transformers with Convolutional Neural Networks to Detect Artificially Generated Text (Vijini Liyanage et al., 2023)

{{<citation>}}

Vijini Liyanage, Davide Buscaldi. (2023)  
**An Ensemble Method Based on the Combination of Transformers with Convolutional Neural Networks to Detect Artificially Generated Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, Natural Language Generation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.17312v1)  

---


**ABSTRACT**  
Thanks to the state-of-the-art Large Language Models (LLMs), language generation has reached outstanding levels. These models are capable of generating high quality content, thus making it a challenging task to detect generated text from human-written content. Despite the advantages provided by Natural Language Generation, the inability to distinguish automatically generated text can raise ethical concerns in terms of authenticity. Consequently, it is important to design and develop methodologies to detect artificial content. In our work, we present some classification models constructed by ensembling transformer models such as Sci-BERT, DeBERTa and XLNet, with Convolutional Neural Networks (CNNs). Our experiments demonstrate that the considered ensemble architectures surpass the performance of the individual transformer models for classification. Furthermore, the proposed SciBERT-CNN ensemble model produced an F1-score of 98.36% on the ALTA shared task 2023 data.

{{</citation>}}


### (64/139) In-Context Ability Transfer for Question Decomposition in Complex QA (Venktesh V et al., 2023)

{{<citation>}}

Venktesh V, Sourangshu Bhattacharya, Avishek Anand. (2023)  
**In-Context Ability Transfer for Question Decomposition in Complex QA**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.18371v1)  

---


**ABSTRACT**  
Answering complex questions is a challenging task that requires question decomposition and multistep reasoning for arriving at the solution. While existing supervised and unsupervised approaches are specialized to a certain task and involve training, recently proposed prompt-based approaches offer generalizable solutions to tackle a wide variety of complex question-answering (QA) tasks. However, existing prompt-based approaches that are effective for complex QA tasks involve expensive hand annotations from experts in the form of rationales and are not generalizable to newer complex QA scenarios and tasks. We propose, icat (In-Context Ability Transfer) which induces reasoning capabilities in LLMs without any LLM fine-tuning or manual annotation of in-context samples. We transfer the ability to decompose complex questions to simpler questions or generate step-by-step rationales to LLMs, by careful selection from available data sources of related tasks. We also propose an automated uncertainty-aware exemplar selection approach for selecting examples from transfer data sources. Finally, we conduct large-scale experiments on a variety of complex QA tasks involving numerical reasoning, compositional complex QA, and heterogeneous complex QA which require decomposed reasoning. We show that ICAT convincingly outperforms existing prompt-based solutions without involving any model training, showcasing the benefits of re-using existing abilities.

{{</citation>}}


### (65/139) Learning to Abstract with Nonparametric Variational Information Bottleneck (Melika Behjati et al., 2023)

{{<citation>}}

Melika Behjati, Fabio Fehr, James Henderson. (2023)  
**Learning to Abstract with Nonparametric Variational Information Bottleneck**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2310.17284v1)  

---


**ABSTRACT**  
Learned representations at the level of characters, sub-words, words and sentences, have each contributed to advances in understanding different NLP tasks and linguistic phenomena. However, learning textual embeddings is costly as they are tokenization specific and require different models to be trained for each level of abstraction. We introduce a novel language representation model which can learn to compress to different levels of abstraction at different layers of the same model. We apply Nonparametric Variational Information Bottleneck (NVIB) to stacked Transformer self-attention layers in the encoder, which encourages an information-theoretic compression of the representations through the model. We find that the layers within the model correspond to increasing levels of abstraction and that their representations are more linguistically informed. Finally, we show that NVIB compression results in a model which is more robust to adversarial perturbations.

{{</citation>}}


### (66/139) Understanding the Role of Input Token Characters in Language Models: How Does Information Loss Affect Performance? (Ahmed Alajrami et al., 2023)

{{<citation>}}

Ahmed Alajrami, Katerina Margatina, Nikolaos Aletras. (2023)  
**Understanding the Role of Input Token Characters in Language Models: How Does Information Loss Affect Performance?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE, Language Model, NLU, SuperGLUE  
[Paper Link](http://arxiv.org/abs/2310.17271v1)  

---


**ABSTRACT**  
Understanding how and what pre-trained language models (PLMs) learn about language is an open challenge in natural language processing. Previous work has focused on identifying whether they capture semantic and syntactic information, and how the data or the pre-training objective affects their performance. However, to the best of our knowledge, no previous work has specifically examined how information loss in input token characters affects the performance of PLMs. In this study, we address this gap by pre-training language models using small subsets of characters from individual tokens. Surprisingly, we find that pre-training even under extreme settings, i.e. using only one character of each token, the performance retention in standard NLU benchmarks and probing tasks compared to full-token models is high. For instance, a model pre-trained only on single first characters from tokens achieves performance retention of approximately $90$\% and $77$\% of the full-token model in SuperGLUE and GLUE tasks, respectively.

{{</citation>}}


### (67/139) Joint Entity and Relation Extraction with Span Pruning and Hypergraph Neural Networks (Zhaohui Yan et al., 2023)

{{<citation>}}

Zhaohui Yan, Songlin Yang, Wei Liu, Kewei Tu. (2023)  
**Joint Entity and Relation Extraction with Span Pruning and Hypergraph Neural Networks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NER, Pruning, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2310.17238v1)  

---


**ABSTRACT**  
Entity and Relation Extraction (ERE) is an important task in information extraction. Recent marker-based pipeline models achieve state-of-the-art performance, but still suffer from the error propagation issue. Also, most of current ERE models do not take into account higher-order interactions between multiple entities and relations, while higher-order modeling could be beneficial.In this work, we propose HyperGraph neural network for ERE ($\hgnn{}$), which is built upon the PL-marker (a state-of-the-art marker-based pipleline model). To alleviate error propagation,we use a high-recall pruner mechanism to transfer the burden of entity identification and labeling from the NER module to the joint module of our model. For higher-order modeling, we build a hypergraph, where nodes are entities (provided by the span pruner) and relations thereof, and hyperedges encode interactions between two different relations or between a relation and its associated subject and object entities. We then run a hypergraph neural network for higher-order inference by applying message passing over the built hypergraph. Experiments on three widely used benchmarks (\acef{}, \ace{} and \scierc{}) for ERE task show significant improvements over the previous state-of-the-art PL-marker.

{{</citation>}}


### (68/139) EMMA-X: An EM-like Multilingual Pre-training Algorithm for Cross-lingual Representation Learning (Ping Guo et al., 2023)

{{<citation>}}

Ping Guo, Xiangpeng Wei, Yue Hu, Baosong Yang, Dayiheng Liu, Fei Huang, Jun Xie. (2023)  
**EMMA-X: An EM-like Multilingual Pre-training Algorithm for Cross-lingual Representation Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.17233v1)  

---


**ABSTRACT**  
Expressing universal semantics common to all languages is helpful in understanding the meanings of complex and culture-specific sentences. The research theme underlying this scenario focuses on learning universal representations across languages with the usage of massive parallel corpora. However, due to the sparsity and scarcity of parallel data, there is still a big challenge in learning authentic ``universals'' for any two languages. In this paper, we propose EMMA-X: an EM-like Multilingual pre-training Algorithm, to learn (X)Cross-lingual universals with the aid of excessive multilingual non-parallel data. EMMA-X unifies the cross-lingual representation learning task and an extra semantic relation prediction task within an EM framework. Both the extra semantic classifier and the cross-lingual sentence encoder approximate the semantic relation of two sentences, and supervise each other until convergence. To evaluate EMMA-X, we conduct experiments on XRETE, a newly introduced benchmark containing 12 widely studied cross-lingual tasks that fully depend on sentence-level representations. Results reveal that EMMA-X achieves state-of-the-art performance. Further geometric analysis of the built representation space with three requirements demonstrates the superiority of EMMA-X over advanced models.

{{</citation>}}


### (69/139) Beyond MLE: Convex Learning for Text Generation (Chenze Shao et al., 2023)

{{<citation>}}

Chenze Shao, Zhengrui Ma, Min Zhang, Yang Feng. (2023)  
**Beyond MLE: Convex Learning for Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Text Generation  
[Paper Link](http://arxiv.org/abs/2310.17217v1)  

---


**ABSTRACT**  
Maximum likelihood estimation (MLE) is a statistical method used to estimate the parameters of a probability distribution that best explain the observed data. In the context of text generation, MLE is often used to train generative language models, which can then be used to generate new text. However, we argue that MLE is not always necessary and optimal, especially for closed-ended text generation tasks like machine translation. In these tasks, the goal of model is to generate the most appropriate response, which does not necessarily require it to estimate the entire data distribution with MLE. To this end, we propose a novel class of training objectives based on convex functions, which enables text generation models to focus on highly probable outputs without having to estimate the entire data distribution. We investigate the theoretical properties of the optimal predicted distribution when applying convex functions to the loss, demonstrating that convex functions can sharpen the optimal distribution, thereby enabling the model to better capture outputs with high probabilities. Experiments on various text generation tasks and models show the effectiveness of our approach. It enables autoregressive models to bridge the gap between greedy and beam search, and facilitates the learning of non-autoregressive models with a maximum improvement of 9+ BLEU points. Moreover, our approach also exhibits significant impact on large language models (LLMs), substantially enhancing their generative capability on various tasks. Source code is available at \url{https://github.com/ictnlp/Convex-Learning}.

{{</citation>}}


### (70/139) Symbolic Planning and Code Generation for Grounded Dialogue (Justin T. Chiu et al., 2023)

{{<citation>}}

Justin T. Chiu, Wenting Zhao, Derek Chen, Saujas Vaduguru, Alexander M. Rush, Daniel Fried. (2023)  
**Symbolic Planning and Code Generation for Grounded Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2310.17140v1)  

---


**ABSTRACT**  
Large language models (LLMs) excel at processing and generating both text and code. However, LLMs have had limited applicability in grounded task-oriented dialogue as they are difficult to steer toward task objectives and fail to handle novel grounding. We present a modular and interpretable grounded dialogue system that addresses these shortcomings by composing LLMs with a symbolic planner and grounded code execution. Our system consists of a reader and planner: the reader leverages an LLM to convert partner utterances into executable code, calling functions that perform grounding. The translated code's output is stored to track dialogue state, while a symbolic planner determines the next appropriate response. We evaluate our system's performance on the demanding OneCommon dialogue task, involving collaborative reference resolution on abstract images of scattered dots. Our system substantially outperforms the previous state-of-the-art, including improving task success in human evaluations from 56% to 69% in the most challenging setting.

{{</citation>}}


### (71/139) Incorporating Probing Signals into Multimodal Machine Translation via Visual Question-Answering Pairs (Yuxin Zuo et al., 2023)

{{<citation>}}

Yuxin Zuo, Bei Li, Chuanhao Lv, Tong Zheng, Tong Xiao, Jingbo Zhu. (2023)  
**Incorporating Probing Signals into Multimodal Machine Translation via Visual Question-Answering Pairs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Machine Translation, QA  
[Paper Link](http://arxiv.org/abs/2310.17133v1)  

---


**ABSTRACT**  
This paper presents an in-depth study of multimodal machine translation (MMT), examining the prevailing understanding that MMT systems exhibit decreased sensitivity to visual information when text inputs are complete. Instead, we attribute this phenomenon to insufficient cross-modal interaction, rather than image information redundancy. A novel approach is proposed to generate parallel Visual Question-Answering (VQA) style pairs from the source text, fostering more robust cross-modal interaction. Using Large Language Models (LLMs), we explicitly model the probing signal in MMT to convert it into VQA-style data to create the Multi30K-VQA dataset. An MMT-VQA multitask learning framework is introduced to incorporate explicit probing signals from the dataset into the MMT training process. Experimental results on two widely-used benchmarks demonstrate the effectiveness of this novel approach. Our code and data would be available at: \url{https://github.com/libeineu/MMT-VQA}.

{{</citation>}}


### (72/139) Test-time Augmentation for Factual Probing (Go Kamoda et al., 2023)

{{<citation>}}

Go Kamoda, Benjamin Heinzerling, Keisuke Sakaguchi, Kentaro Inui. (2023)  
**Test-time Augmentation for Factual Probing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.17121v1)  

---


**ABSTRACT**  
Factual probing is a method that uses prompts to test if a language model "knows" certain world knowledge facts. A problem in factual probing is that small changes to the prompt can lead to large changes in model output. Previous work aimed to alleviate this problem by optimizing prompts via text mining or fine-tuning. However, such approaches are relation-specific and do not generalize to unseen relation types. Here, we propose to use test-time augmentation (TTA) as a relation-agnostic method for reducing sensitivity to prompt variations by automatically augmenting and ensembling prompts at test time. Experiments show improved model calibration, i.e., with TTA, model confidence better reflects prediction accuracy. Improvements in prediction accuracy are observed for some models, but for other models, TTA leads to degradation. Error analysis identifies the difficulty of producing high-quality prompt variations as the main challenge for TTA.

{{</citation>}}


### (73/139) Topic Segmentation of Semi-Structured and Unstructured Conversational Datasets using Language Models (Reshmi Ghosh et al., 2023)

{{<citation>}}

Reshmi Ghosh, Harjeet Singh Kajal, Sharanya Kamath, Dhuri Shrivastava, Samyadeep Basu, Hansi Zeng, Soundararajan Srinivasan. (2023)  
**Topic Segmentation of Semi-Structured and Unstructured Conversational Datasets using Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.17120v1)  

---


**ABSTRACT**  
Breaking down a document or a conversation into multiple contiguous segments based on its semantic structure is an important and challenging problem in NLP, which can assist many downstream tasks. However, current works on topic segmentation often focus on segmentation of structured texts. In this paper, we comprehensively analyze the generalization capabilities of state-of-the-art topic segmentation models on unstructured texts. We find that: (a) Current strategies of pre-training on a large corpus of structured text such as Wiki-727K do not help in transferability to unstructured conversational data. (b) Training from scratch with only a relatively small-sized dataset of the target unstructured domain improves the segmentation results by a significant margin. We stress-test our proposed Topic Segmentation approach by experimenting with multiple loss functions, in order to mitigate effects of imbalance in unstructured conversational datasets. Our empirical evaluation indicates that Focal Loss function is a robust alternative to Cross-Entropy and re-weighted Cross-Entropy loss function when segmenting unstructured and semi-structured chats.

{{</citation>}}


## cs.LG (35)



### (74/139) Understanding when Dynamics-Invariant Data Augmentations Benefit Model-Free Reinforcement Learning Updates (Nicholas E. Corrado et al., 2023)

{{<citation>}}

Nicholas E. Corrado, Josiah P. Hanna. (2023)  
**Understanding when Dynamics-Invariant Data Augmentations Benefit Model-Free Reinforcement Learning Updates**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.17786v1)  

---


**ABSTRACT**  
Recently, data augmentation (DA) has emerged as a method for leveraging domain knowledge to inexpensively generate additional data in reinforcement learning (RL) tasks, often yielding substantial improvements in data efficiency. While prior work has demonstrated the utility of incorporating augmented data directly into model-free RL updates, it is not well-understood when a particular DA strategy will improve data efficiency. In this paper, we seek to identify general aspects of DA responsible for observed learning improvements. Our study focuses on sparse-reward tasks with dynamics-invariant data augmentation functions, serving as an initial step towards a more general understanding of DA and its integration into RL training. Experimentally, we isolate three relevant aspects of DA: state-action coverage, reward density, and the number of augmented transitions generated per update (the augmented replay ratio). From our experiments, we draw two conclusions: (1) increasing state-action coverage often has a much greater impact on data efficiency than increasing reward density, and (2) decreasing the augmented replay ratio substantially improves data efficiency. In fact, certain tasks in our empirical study are solvable only when the replay ratio is sufficiently low.

{{</citation>}}


### (75/139) Making the End-User a Priority in Benchmarking: OrionBench for Unsupervised Time Series Anomaly Detection (Sarah Alnegheimish et al., 2023)

{{<citation>}}

Sarah Alnegheimish, Laure Berti-Equille, Kalyan Veeramachaneni. (2023)  
**Making the End-User a Priority in Benchmarking: OrionBench for Unsupervised Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2310.17748v1)  

---


**ABSTRACT**  
Time series anomaly detection is a prevalent problem in many application domains such as patient monitoring in healthcare, forecasting in finance, or predictive maintenance in energy. This has led to the emergence of a plethora of anomaly detection methods, including more recently, deep learning based methods. Although several benchmarks have been proposed to compare newly developed models, they usually rely on one-time execution over a limited set of datasets and the comparison is restricted to a few models. We propose OrionBench -- a user centric continuously maintained benchmark for unsupervised time series anomaly detection. The framework provides universal abstractions to represent models, extensibility to add new pipelines and datasets, hyperparameter standardization, pipeline verification, and frequent releases with published benchmarks. We demonstrate the usage of OrionBench, and the progression of pipelines across 15 releases published over the course of three years. Moreover, we walk through two real scenarios we experienced with OrionBench that highlight the importance of continuous benchmarks in unsupervised time series anomaly detection.

{{</citation>}}


### (76/139) Improving Traffic Density Forecasting in Intelligent Transportation Systems Using Gated Graph Neural Networks (Razib Hayat Khan et al., 2023)

{{<citation>}}

Razib Hayat Khan, Jonayet Miah, S M Yasir Arafat, M M Mahbubul Syeed, Duc M Ca. (2023)  
**Improving Traffic Density Forecasting in Intelligent Transportation Systems Using Gated Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: GNN, Graph Convolutional Network, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.17729v1)  

---


**ABSTRACT**  
This study delves into the application of graph neural networks in the realm of traffic forecasting, a crucial facet of intelligent transportation systems. Accurate traffic predictions are vital for functions like trip planning, traffic control, and vehicle routing in such systems. Three prominent GNN architectures Graph Convolutional Networks (Graph Sample and Aggregation) and Gated Graph Neural Networks are explored within the context of traffic prediction. Each architecture's methodology is thoroughly examined, including layer configurations, activation functions,and hyperparameters. The primary goal is to minimize prediction errors, with GGNNs emerging as the most effective choice among the three models. The research outlines outcomes for each architecture, elucidating their predictive performance through root mean squared error and mean absolute error (MAE). Hypothetical results reveal intriguing insights: GCNs display an RMSE of 9.10 and an MAE of 8.00, while GraphSAGE shows improvement with an RMSE of 8.3 and an MAE of 7.5. Gated Graph Neural Networks (GGNNs) exhibit the lowest RMSE at 9.15 and an impressive MAE of 7.1, positioning them as the frontrunner.

{{</citation>}}


### (77/139) ZeroQuant-HERO: Hardware-Enhanced Robust Optimized Post-Training Quantization Framework for W8A8 Transformers (Zhewei Yao et al., 2023)

{{<citation>}}

Zhewei Yao, Reza Yazdani Aminabadi, Stephen Youn, Xiaoxia Wu, Elton Zheng, Yuxiong He. (2023)  
**ZeroQuant-HERO: Hardware-Enhanced Robust Optimized Post-Training Quantization Framework for W8A8 Transformers**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: BERT, GPT, Quantization, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.17723v1)  

---


**ABSTRACT**  
Quantization techniques are pivotal in reducing the memory and computational demands of deep neural network inference. Existing solutions, such as ZeroQuant, offer dynamic quantization for models like BERT and GPT but overlook crucial memory-bounded operators and the complexities of per-token quantization. Addressing these gaps, we present a novel, fully hardware-enhanced robust optimized post-training W8A8 quantization framework, ZeroQuant-HERO. This framework uniquely integrates both memory bandwidth and compute-intensive operators, aiming for optimal hardware performance. Additionally, it offers flexibility by allowing specific INT8 modules to switch to FP16/BF16 mode, enhancing accuracy.

{{</citation>}}


### (78/139) Large Language Models as Generalizable Policies for Embodied Tasks (Andrew Szot et al., 2023)

{{<citation>}}

Andrew Szot, Max Schwarzer, Harsh Agrawal, Bogdan Mazoure, Walter Talbott, Katherine Metcalf, Natalie Mackraz, Devon Hjelm, Alexander Toshev. (2023)  
**Large Language Models as Generalizable Policies for Embodied Tasks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: AI, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.17722v1)  

---


**ABSTRACT**  
We show that large language models (LLMs) can be adapted to be generalizable policies for embodied visual tasks. Our approach, called Large LAnguage model Reinforcement Learning Policy (LLaRP), adapts a pre-trained frozen LLM to take as input text instructions and visual egocentric observations and output actions directly in the environment. Using reinforcement learning, we train LLaRP to see and act solely through environmental interactions. We show that LLaRP is robust to complex paraphrasings of task instructions and can generalize to new tasks that require novel optimal behavior. In particular, on 1,000 unseen tasks it achieves 42% success rate, 1.7x the success rate of other common learned baselines or zero-shot applications of LLMs. Finally, to aid the community in studying language conditioned, massively multi-task, embodied AI problems we release a novel benchmark, Language Rearrangement, consisting of 150,000 training and 1,000 testing tasks for language-conditioned rearrangement. Video examples of LLaRP in unseen Language Rearrangement instructions are at https://llm-rl.github.io.

{{</citation>}}


### (79/139) Fantastic Gains and Where to Find Them: On the Existence and Prospect of General Knowledge Transfer between Any Pretrained Model (Karsten Roth et al., 2023)

{{<citation>}}

Karsten Roth, Lukas Thede, Almut Sophia Koepke, Oriol Vinyals, Olivier Hénaff, Zeynep Akata. (2023)  
**Fantastic Gains and Where to Find Them: On the Existence and Prospect of General Knowledge Transfer between Any Pretrained Model**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.17653v1)  

---


**ABSTRACT**  
Training deep networks requires various design decisions regarding for instance their architecture, data augmentation, or optimization. In this work, we find these training variations to result in networks learning unique feature sets from the data. Using public model libraries comprising thousands of models trained on canonical datasets like ImageNet, we observe that for arbitrary pairings of pretrained models, one model extracts significant data context unavailable in the other -- independent of overall performance. Given any arbitrary pairing of pretrained models and no external rankings (such as separate test sets, e.g. due to data privacy), we investigate if it is possible to transfer such "complementary" knowledge from one model to another without performance degradation -- a task made particularly difficult as additional knowledge can be contained in stronger, equiperformant or weaker models. Yet facilitating robust transfer in scenarios agnostic to pretrained model pairings would unlock auxiliary gains and knowledge fusion from any model repository without restrictions on model and problem specifics - including from weaker, lower-performance models. This work therefore provides an initial, in-depth exploration on the viability of such general-purpose knowledge transfer. Across large-scale experiments, we first reveal the shortcomings of standard knowledge distillation techniques, and then propose a much more general extension through data partitioning for successful transfer between nearly all pretrained models, which we show can also be done unsupervised. Finally, we assess both the scalability and impact of fundamental model properties on successful model-agnostic knowledge transfer.

{{</citation>}}


### (80/139) Defending Against Transfer Attacks From Public Models (Chawin Sitawarin et al., 2023)

{{<citation>}}

Chawin Sitawarin, Jaewon Chang, David Huang, Wesson Altoyan, David Wagner. (2023)  
**Defending Against Transfer Attacks From Public Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.17645v1)  

---


**ABSTRACT**  
Adversarial attacks have been a looming and unaddressed threat in the industry. However, through a decade-long history of the robustness evaluation literature, we have learned that mounting a strong or optimal attack is challenging. It requires both machine learning and domain expertise. In other words, the white-box threat model, religiously assumed by a large majority of the past literature, is unrealistic. In this paper, we propose a new practical threat model where the adversary relies on transfer attacks through publicly available surrogate models. We argue that this setting will become the most prevalent for security-sensitive applications in the future. We evaluate the transfer attacks in this setting and propose a specialized defense method based on a game-theoretic perspective. The defenses are evaluated under 24 public models and 11 attack algorithms across three datasets (CIFAR-10, CIFAR-100, and ImageNet). Under this threat model, our defense, PubDef, outperforms the state-of-the-art white-box adversarial training by a large margin with almost no loss in the normal accuracy. For instance, on ImageNet, our defense achieves 62% accuracy under the strongest transfer attack vs only 36% of the best adversarially trained model. Its accuracy when not under attack is only 2% lower than that of an undefended model (78% vs 80%). We release our code at https://github.com/wagner-group/pubdef.

{{</citation>}}


### (81/139) Combating Representation Learning Disparity with Geometric Harmonization (Zhihan Zhou et al., 2023)

{{<citation>}}

Zhihan Zhou, Jiangchao Yao, Feng Hong, Ya Zhang, Bo Han, Yanfeng Wang. (2023)  
**Combating Representation Learning Disparity with Geometric Harmonization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.17622v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) as an effective paradigm of representation learning has achieved tremendous success on various curated datasets in diverse scenarios. Nevertheless, when facing the long-tailed distribution in real-world applications, it is still hard for existing methods to capture transferable and robust representation. Conventional SSL methods, pursuing sample-level uniformity, easily leads to representation learning disparity where head classes dominate the feature regime but tail classes passively collapse. To address this problem, we propose a novel Geometric Harmonization (GH) method to encourage category-level uniformity in representation learning, which is more benign to the minority and almost does not hurt the majority under long-tailed distribution. Specially, GH measures the population statistics of the embedding space on top of self-supervised learning, and then infer an fine-grained instance-wise calibration to constrain the space expansion of head classes and avoid the passive collapse of tail classes. Our proposal does not alter the setting of SSL and can be easily integrated into existing methods in a low-cost manner. Extensive results on a range of benchmark datasets show the effectiveness of GH with high tolerance to the distribution skewness. Our code is available at https://github.com/MediaBrain-SJTU/Geometric-Harmonization.

{{</citation>}}


### (82/139) Uncovering Meanings of Embeddings via Partial Orthogonality (Yibo Jiang et al., 2023)

{{<citation>}}

Yibo Jiang, Bryon Aragam, Victor Veitch. (2023)  
**Uncovering Meanings of Embeddings via Partial Orthogonality**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.17611v1)  

---


**ABSTRACT**  
Machine learning tools often rely on embedding text as vectors of real numbers. In this paper, we study how the semantic structure of language is encoded in the algebraic structure of such embeddings. Specifically, we look at a notion of ``semantic independence'' capturing the idea that, e.g., ``eggplant'' and ``tomato'' are independent given ``vegetable''. Although such examples are intuitive, it is difficult to formalize such a notion of semantic independence. The key observation here is that any sensible formalization should obey a set of so-called independence axioms, and thus any algebraic encoding of this structure should also obey these axioms. This leads us naturally to use partial orthogonality as the relevant algebraic structure. We develop theory and methods that allow us to demonstrate that partial orthogonality does indeed capture semantic independence. Complementary to this, we also introduce the concept of independence preserving embeddings where embeddings preserve the conditional independence structures of a distribution, and we prove the existence of such embeddings and approximations to them.

{{</citation>}}


### (83/139) PAC-tuning:Fine-tuning Pretrained Language Models with PAC-driven Perturbed Gradient Descent (Guangliang Liu et al., 2023)

{{<citation>}}

Guangliang Liu, Zhiyu Xue, Xitong Zhang, Kristen Marie Johnson, Rongrong Wang. (2023)  
**PAC-tuning:Fine-tuning Pretrained Language Models with PAC-driven Perturbed Gradient Descent**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GLUE, Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2310.17588v1)  

---


**ABSTRACT**  
Fine-tuning pretrained language models (PLMs) for downstream tasks is a large-scale optimization problem, in which the choice of the training algorithm critically determines how well the trained model can generalize to unseen test data, especially in the context of few-shot learning. To achieve good generalization performance and avoid overfitting, techniques such as data augmentation and pruning are often applied. However, adding these regularizations necessitates heavy tuning of the hyperparameters of optimization algorithms, such as the popular Adam optimizer. In this paper, we propose a two-stage fine-tuning method, PAC-tuning, to address this optimization challenge. First, based on PAC-Bayes training, PAC-tuning directly minimizes the PAC-Bayes generalization bound to learn proper parameter distribution. Second, PAC-tuning modifies the gradient by injecting noise with the variance learned in the first stage into the model parameters during training, resulting in a variant of perturbed gradient descent (PGD). In the past, the few-shot scenario posed difficulties for PAC-Bayes training because the PAC-Bayes bound, when applied to large models with limited training data, might not be stringent. Our experimental results across 5 GLUE benchmark tasks demonstrate that PAC-tuning successfully handles the challenges of fine-tuning tasks and outperforms strong baseline methods by a visible margin, further confirming the potential to apply PAC training for any other settings where the Adam optimizer is currently used for training.

{{</citation>}}


### (84/139) BLIS-Net: Classifying and Analyzing Signals on Graphs (Charles Xu et al., 2023)

{{<citation>}}

Charles Xu, Laney Goldman, Valentina Guo, Benjamin Hollander-Bodie, Maedee Trank-Greene, Ian Adelstein, Edward De Brouwer, Rex Ying, Smita Krishnaswamy, Michael Perlmutter. (2023)  
**BLIS-Net: Classifying and Analyzing Signals on Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.17579v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have emerged as a powerful tool for tasks such as node classification and graph classification. However, much less work has been done on signal classification, where the data consists of many functions (referred to as signals) defined on the vertices of a single graph. These tasks require networks designed differently from those designed for traditional GNN tasks. Indeed, traditional GNNs rely on localized low-pass filters, and signals of interest may have intricate multi-frequency behavior and exhibit long range interactions. This motivates us to introduce the BLIS-Net (Bi-Lipschitz Scattering Net), a novel GNN that builds on the previously introduced geometric scattering transform. Our network is able to capture both local and global signal structure and is able to capture both low-frequency and high-frequency information. We make several crucial changes to the original geometric scattering architecture which we prove increase the ability of our network to capture information about the input signal and show that BLIS-Net achieves superior performance on both synthetic and real-world data sets based on traffic flow and fMRI data.

{{</citation>}}


### (85/139) Hierarchical Ensemble-Based Feature Selection for Time Series Forecasting (Aysin Tumay et al., 2023)

{{<citation>}}

Aysin Tumay, Mustafa E. Aydin, Suleyman S. Kozat. (2023)  
**Hierarchical Ensemble-Based Feature Selection for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.17544v1)  

---


**ABSTRACT**  
We study a novel ensemble approach for feature selection based on hierarchical stacking in cases of non-stationarity and limited number of samples with large number of features. Our approach exploits the co-dependency between features using a hierarchical structure. Initially, a machine learning model is trained using a subset of features, and then the model's output is updated using another algorithm with the remaining features to minimize the target loss. This hierarchical structure allows for flexible depth and feature selection. By exploiting feature co-dependency hierarchically, our proposed approach overcomes the limitations of traditional feature selection methods and feature importance scores. The effectiveness of the approach is demonstrated on synthetic and real-life datasets, indicating improved performance with scalability and stability compared to the traditional methods and state-of-the-art approaches.

{{</citation>}}


### (86/139) The Expressive Power of Low-Rank Adaptation (Yuchen Zeng et al., 2023)

{{<citation>}}

Yuchen Zeng, Kangwook Lee. (2023)  
**The Expressive Power of Low-Rank Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.17513v2)  

---


**ABSTRACT**  
Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning method that leverages low-rank adaptation of weight matrices, has emerged as a prevalent technique for fine-tuning pre-trained models such as large language models and diffusion models. Despite its huge success in practice, the theoretical underpinnings of LoRA have largely remained unexplored. This paper takes the first step to bridge this gap by theoretically analyzing the expressive power of LoRA. We prove that, for fully connected neural networks, LoRA can adapt any model $f$ to accurately represent any smaller target model $\overline{f}$ if LoRA-rank $\geq(\text{width of }f) \times \frac{\text{depth of }\overline{f}}{\text{depth of }f}$. We also quantify the approximation error when LoRA-rank is lower than the threshold. For Transformer networks, we show any model can be adapted to a target model of the same size with rank-$(\frac{\text{embedding size}}{2})$ LoRA adapters.

{{</citation>}}


### (87/139) CBD: A Certified Backdoor Detector Based on Local Dominant Probability (Zhen Xiang et al., 2023)

{{<citation>}}

Zhen Xiang, Zidi Xiong, Bo Li. (2023)  
**CBD: A Certified Backdoor Detector Based on Local Dominant Probability**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.17498v1)  

---


**ABSTRACT**  
Backdoor attack is a common threat to deep neural networks. During testing, samples embedded with a backdoor trigger will be misclassified as an adversarial target by a backdoored model, while samples without the backdoor trigger will be correctly classified. In this paper, we present the first certified backdoor detector (CBD), which is based on a novel, adjustable conformal prediction scheme based on our proposed statistic local dominant probability. For any classifier under inspection, CBD provides 1) a detection inference, 2) the condition under which the attacks are guaranteed to be detectable for the same classification domain, and 3) a probabilistic upper bound for the false positive rate. Our theoretical results show that attacks with triggers that are more resilient to test-time noise and have smaller perturbation magnitudes are more likely to be detected with guarantees. Moreover, we conduct extensive experiments on four benchmark datasets considering various backdoor types, such as BadNet, CB, and Blend. CBD achieves comparable or even higher detection accuracy than state-of-the-art detectors, and it in addition provides detection certification. Notably, for backdoor attacks with random perturbation triggers bounded by $\ell_2\leq0.75$ which achieves more than 90\% attack success rate, CBD achieves 100\% (98\%), 100\% (84\%), 98\% (98\%), and 72\% (40\%) empirical (certified) detection true positive rates on the four benchmark datasets GTSRB, SVHN, CIFAR-10, and TinyImageNet, respectively, with low false positive rates.

{{</citation>}}


### (88/139) FedPEAT: Convergence of Federated Learning, Parameter-Efficient Fine Tuning, and Emulator Assisted Tuning for Artificial Intelligence Foundation Models with Mobile Edge Computing (Terence Jie Chua et al., 2023)

{{<citation>}}

Terence Jie Chua, Wenhan Yu, Jun Zhao, Kwok-Yan Lam. (2023)  
**FedPEAT: Convergence of Federated Learning, Parameter-Efficient Fine Tuning, and Emulator Assisted Tuning for Artificial Intelligence Foundation Models with Mobile Edge Computing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG  
Keywords: AI, BERT, GPT  
[Paper Link](http://arxiv.org/abs/2310.17491v1)  

---


**ABSTRACT**  
The emergence of foundation models, including language and vision models, has reshaped AI's landscape, offering capabilities across various applications. Deploying and fine-tuning these large models, like GPT-3 and BERT, presents challenges, especially in the current foundation model era. We introduce Emulator-Assisted Tuning (EAT) combined with Parameter-Efficient Fine-Tuning (PEFT) to form Parameter-Efficient Emulator-Assisted Tuning (PEAT). Further, we expand this into federated learning as Federated PEAT (FedPEAT). FedPEAT uses adapters, emulators, and PEFT for federated model tuning, enhancing model privacy and memory efficiency. Adapters adjust pre-trained models, while emulators give a compact representation of original models, addressing both privacy and efficiency. Adaptable to various neural networks, our approach also uses deep reinforcement learning for hyper-parameter optimization. We tested FedPEAT in a unique scenario with a server participating in collaborative federated tuning, showcasing its potential in tackling foundation model challenges.

{{</citation>}}


### (89/139) Coalitional Bargaining via Reinforcement Learning: An Application to Collaborative Vehicle Routing (Stephen Mak et al., 2023)

{{<citation>}}

Stephen Mak, Liming Xu, Tim Pearce, Michael Ostroumov, Alexandra Brintrup. (2023)  
**Coalitional Bargaining via Reinforcement Learning: An Application to Collaborative Vehicle Routing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.17458v1)  

---


**ABSTRACT**  
Collaborative Vehicle Routing is where delivery companies cooperate by sharing their delivery information and performing delivery requests on behalf of each other. This achieves economies of scale and thus reduces cost, greenhouse gas emissions, and road congestion. But which company should partner with whom, and how much should each company be compensated? Traditional game theoretic solution concepts, such as the Shapley value or nucleolus, are difficult to calculate for the real-world problem of Collaborative Vehicle Routing due to the characteristic function scaling exponentially with the number of agents. This would require solving the Vehicle Routing Problem (an NP-Hard problem) an exponential number of times. We therefore propose to model this problem as a coalitional bargaining game where - crucially - agents are not given access to the characteristic function. Instead, we implicitly reason about the characteristic function, and thus eliminate the need to evaluate the VRP an exponential number of times - we only need to evaluate it once. Our contribution is that our decentralised approach is both scalable and considers the self-interested nature of companies. The agents learn using a modified Independent Proximal Policy Optimisation. Our RL agents outperform a strong heuristic bot. The agents correctly identify the optimal coalitions 79% of the time with an average optimality gap of 4.2% and reduction in run-time of 62%.

{{</citation>}}


### (90/139) Sliceformer: Make Multi-head Attention as Simple as Sorting in Discriminative Tasks (Shen Yuan et al., 2023)

{{<citation>}}

Shen Yuan, Hongteng Xu. (2023)  
**Sliceformer: Make Multi-head Attention as Simple as Sorting in Discriminative Tasks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, BERT, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2310.17683v1)  

---


**ABSTRACT**  
As one of the most popular neural network modules, Transformer plays a central role in many fundamental deep learning models, e.g., the ViT in computer vision and the BERT and GPT in natural language processing. The effectiveness of the Transformer is often attributed to its multi-head attention (MHA) mechanism. In this study, we discuss the limitations of MHA, including the high computational complexity due to its ``query-key-value'' architecture and the numerical issue caused by its softmax operation. Considering the above problems and the recent development tendency of the attention layer, we propose an effective and efficient surrogate of the Transformer, called Sliceformer. Our Sliceformer replaces the classic MHA mechanism with an extremely simple ``slicing-sorting'' operation, i.e., projecting inputs linearly to a latent space and sorting them along different feature dimensions (or equivalently, called channels). For each feature dimension, the sorting operation implicitly generates an implicit attention map with sparse, full-rank, and doubly-stochastic structures. We consider different implementations of the slicing-sorting operation and analyze their impacts on the Sliceformer. We test the Sliceformer in the Long-Range Arena benchmark, image classification, text classification, and molecular property prediction, demonstrating its advantage in computational complexity and universal effectiveness in discriminative tasks. Our Sliceformer achieves comparable or better performance with lower memory cost and faster speed than the Transformer and its variants. Moreover, the experimental results reveal that applying our Sliceformer can empirically suppress the risk of mode collapse when representing data. The code is available at \url{https://github.com/SDS-Lab/sliceformer}.

{{</citation>}}


### (91/139) Enhancing Graph Neural Networks with Structure-Based Prompt (Qingqing Ge et al., 2023)

{{<citation>}}

Qingqing Ge, Zeyuan Zhao, Yiding Liu, Anfeng Cheng, Xiang Li, Shuaiqiang Wang, Dawei Yin. (2023)  
**Enhancing Graph Neural Networks with Structure-Based Prompt**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.17394v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are powerful in learning semantics of graph data. Recently, a new paradigm "pre-train, prompt" has shown promising results in adapting GNNs to various tasks with less supervised data. The success of such paradigm can be attributed to the more consistent objectives of pre-training and task-oriented prompt tuning, where the pre-trained knowledge can be effectively transferred to downstream tasks. However, an overlooked issue of existing studies is that the structure information of graph is usually exploited during pre-training for learning node representations, while neglected in the prompt tuning stage for learning task-specific parameters. To bridge this gap, we propose a novel structure-based prompting method for GNNs, namely SAP, which consistently exploits structure information in both pre-training and prompt tuning stages. In particular, SAP 1) employs a dual-view contrastive learning to align the latent semantic spaces of node attributes and graph structure, and 2) incorporates structure information in prompted graph to elicit more pre-trained knowledge in prompt tuning. We conduct extensive experiments on node classification and graph classification tasks to show the effectiveness of SAP. Moreover, we show that SAP can lead to better performance in more challenging few-shot scenarios on both homophilous and heterophilous graphs.

{{</citation>}}


### (92/139) Towards Unifying Diffusion Models for Probabilistic Spatio-Temporal Graph Learning (Junfeng Hu et al., 2023)

{{<citation>}}

Junfeng Hu, Xu Liu, Zhencheng Fan, Yuxuan Liang, Roger Zimmermann. (2023)  
**Towards Unifying Diffusion Models for Probabilistic Spatio-Temporal Graph Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.17360v1)  

---


**ABSTRACT**  
Spatio-temporal graph learning is a fundamental problem in the Web of Things era, which enables a plethora of Web applications such as smart cities, human mobility and climate analysis. Existing approaches tackle different learning tasks independently, tailoring their models to unique task characteristics. These methods, however, fall short of modeling intrinsic uncertainties in the spatio-temporal data. Meanwhile, their specialized designs limit their universality as general spatio-temporal learning solutions. In this paper, we propose to model the learning tasks in a unified perspective, viewing them as predictions based on conditional information with shared spatio-temporal patterns. Based on this proposal, we introduce Unified Spatio-Temporal Diffusion Models (USTD) to address the tasks uniformly within the uncertainty-aware diffusion framework. USTD is holistically designed, comprising a shared spatio-temporal encoder and attention-based denoising networks that are task-specific. The shared encoder, optimized by a pre-training strategy, effectively captures conditional spatio-temporal patterns. The denoising networks, utilizing both cross- and self-attention, integrate conditional dependencies and generate predictions. Opting for forecasting and kriging as downstream tasks, we design Gated Attention (SGA) and Temporal Gated Attention (TGA) for each task, with different emphases on the spatial and temporal dimensions, respectively. By combining the advantages of deterministic encoders and probabilistic diffusion models, USTD achieves state-of-the-art performances compared to deterministic and probabilistic baselines in both tasks, while also providing valuable uncertainty estimates.

{{</citation>}}


### (93/139) De-novo Chemical Reaction Generation by Means of Temporarily Convolutional Neural Networks (Andrei Buin et al., 2023)

{{<citation>}}

Andrei Buin, Hung Yi Chiang, S. Andrew Gadsden, Faraz A. Alderson. (2023)  
**De-novo Chemical Reaction Generation by Means of Temporarily Convolutional Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.17341v2)  

---


**ABSTRACT**  
We present here a combination of two networks, Recurrent Neural Networks (RNN) and Temporarily Convolutional Neural Networks (TCN) in de novo reaction generation using the novel Reaction Smiles-like representation of reactions (CGRSmiles) with atom mapping directly incorporated. Recurrent Neural Networks are known for their autoregressive properties and are frequently used in language modelling with direct application to SMILES generation. The relatively novel TCNs possess similar properties with wide receptive field while obeying the causality required for natural language processing (NLP). The combination of both latent representations expressed through TCN and RNN results in an overall better performance compared to RNN alone. Additionally, it is shown that different fine-tuning protocols have a profound impact on generative scope of the model when applied on a dataset of interest via transfer learning.

{{</citation>}}


### (94/139) CQM: Curriculum Reinforcement Learning with a Quantized World Model (Seungjae Lee et al., 2023)

{{<citation>}}

Seungjae Lee, Daesol Cho, Jonghae Park, H. Jin Kim. (2023)  
**CQM: Curriculum Reinforcement Learning with a Quantized World Model**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.17330v1)  

---


**ABSTRACT**  
Recent curriculum Reinforcement Learning (RL) has shown notable progress in solving complex tasks by proposing sequences of surrogate tasks. However, the previous approaches often face challenges when they generate curriculum goals in a high-dimensional space. Thus, they usually rely on manually specified goal spaces. To alleviate this limitation and improve the scalability of the curriculum, we propose a novel curriculum method that automatically defines the semantic goal space which contains vital information for the curriculum process, and suggests curriculum goals over it. To define the semantic goal space, our method discretizes continuous observations via vector quantized-variational autoencoders (VQ-VAE) and restores the temporal relations between the discretized observations by a graph. Concurrently, ours suggests uncertainty and temporal distance-aware curriculum goals that converges to the final goals over the automatically composed goal space. We demonstrate that the proposed method allows efficient explorations in an uninformed environment with raw goal examples only. Also, ours outperforms the state-of-the-art curriculum RL methods on data efficiency and performance, in various goal-reaching tasks even with ego-centric visual inputs.

{{</citation>}}


### (95/139) C-Disentanglement: Discovering Causally-Independent Generative Factors under an Inductive Bias of Confounder (Xiaoyu Liu et al., 2023)

{{<citation>}}

Xiaoyu Liu, Jiaxin Yuan, Bang An, Yuancheng Xu, Yifan Yang, Furong Huang. (2023)  
**C-Disentanglement: Discovering Causally-Independent Generative Factors under an Inductive Bias of Confounder**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.17325v1)  

---


**ABSTRACT**  
Representation learning assumes that real-world data is generated by a few semantically meaningful generative factors (i.e., sources of variation) and aims to discover them in the latent space. These factors are expected to be causally disentangled, meaning that distinct factors are encoded into separate latent variables, and changes in one factor will not affect the values of the others. Compared to statistical independence, causal disentanglement allows more controllable data generation, improved robustness, and better generalization. However, most existing work assumes unconfoundedness in the discovery process, that there are no common causes to the generative factors and thus obtain only statistical independence. In this paper, we recognize the importance of modeling confounders in discovering causal generative factors. Unfortunately, such factors are not identifiable without proper inductive bias. We fill the gap by introducing a framework entitled Confounded-Disentanglement (C-Disentanglement), the first framework that explicitly introduces the inductive bias of confounder via labels from domain expertise. In addition, we accordingly propose an approach to sufficiently identify the causally disentangled factors under any inductive bias of the confounder. We conduct extensive experiments on both synthetic and real-world datasets. Our method demonstrates competitive results compared to various SOTA baselines in obtaining causally disentangled features and downstream tasks under domain shifts.

{{</citation>}}


### (96/139) Looping in the Human: Collaborative and Explainable Bayesian Optimization (Masaki Adachi et al., 2023)

{{<citation>}}

Masaki Adachi, Brady Planden, David A. Howey, Krikamol Maundet, Michael A. Osborne, Siu Lun Chau. (2023)  
**Looping in the Human: Collaborative and Explainable Bayesian Optimization**  

---
Primary Category: cs.LG  
Categories: 62C10, 62F15, cs-HC, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17273v2)  

---


**ABSTRACT**  
Like many optimizers, Bayesian optimization often falls short of gaining user trust due to opacity. While attempts have been made to develop human-centric optimizers, they typically assume user knowledge is well-specified and error-free, employing users mainly as supervisors of the optimization process. We relax these assumptions and propose a more balanced human-AI partnership with our Collaborative and Explainable Bayesian Optimization (CoExBO) framework. Instead of explicitly requiring a user to provide a knowledge model, CoExBO employs preference learning to seamlessly integrate human insights into the optimization, resulting in algorithmic suggestions that resonate with user preference. CoExBO explains its candidate selection every iteration to foster trust, empowering users with a clearer grasp of the optimization. Furthermore, CoExBO offers a no-harm guarantee, allowing users to make mistakes; even with extreme adversarial interventions, the algorithm converges asymptotically to a vanilla Bayesian optimization. We validate CoExBO's efficacy through human-AI teaming experiments in lithium-ion battery design, highlighting substantial improvements over conventional methods.

{{</citation>}}


### (97/139) Codebook Features: Sparse and Discrete Interpretability for Neural Networks (Alex Tamkin et al., 2023)

{{<citation>}}

Alex Tamkin, Mohammad Taufeeque, Noah D. Goodman. (2023)  
**Codebook Features: Sparse and Discrete Interpretability for Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.17230v1)  

---


**ABSTRACT**  
Understanding neural networks is challenging in part because of the dense, continuous nature of their hidden states. We explore whether we can train neural networks to have hidden states that are sparse, discrete, and more interpretable by quantizing their continuous features into what we call codebook features. Codebook features are produced by finetuning neural networks with vector quantization bottlenecks at each layer, producing a network whose hidden features are the sum of a small number of discrete vector codes chosen from a larger codebook. Surprisingly, we find that neural networks can operate under this extreme bottleneck with only modest degradation in performance. This sparse, discrete bottleneck also provides an intuitive way of controlling neural network behavior: first, find codes that activate when the desired behavior is present, then activate those same codes during generation to elicit that behavior. We validate our approach by training codebook Transformers on several different datasets. First, we explore a finite state machine dataset with far more hidden states than neurons. In this setting, our approach overcomes the superposition problem by assigning states to distinct codes, and we find that we can make the neural network behave as if it is in a different state by activating the code for that state. Second, we train Transformer language models with up to 410M parameters on two natural language datasets. We identify codes in these models representing diverse, disentangled concepts (ranging from negative emotions to months of the year) and find that we can guide the model to generate different topics by activating the appropriate codes during inference. Overall, codebook features appear to be a promising unit of analysis and control for neural networks and interpretability. Our codebase and models are open-sourced at https://github.com/taufeeque9/codebook-features.

{{</citation>}}


### (98/139) miditok: A Python package for MIDI file tokenization (Nathan Fradet et al., 2023)

{{<citation>}}

Nathan Fradet, Jean-Pierre Briot, Fabien Chhel, Amal El Fallah Seghrouchni, Nicolas Gutowski. (2023)  
**miditok: A Python package for MIDI file tokenization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.17202v1)  

---


**ABSTRACT**  
Recent progress in natural language processing has been adapted to the symbolic music modality. Language models, such as Transformers, have been used with symbolic music for a variety of tasks among which music generation, modeling or transcription, with state-of-the-art performances. These models are beginning to be used in production products. To encode and decode music for the backbone model, they need to rely on tokenizers, whose role is to serialize music into sequences of distinct elements called tokens. MidiTok is an open-source library allowing to tokenize symbolic music with great flexibility and extended features. It features the most popular music tokenizations, under a unified API. It is made to be easily used and extensible for everyone.

{{</citation>}}


### (99/139) How do Language Models Bind Entities in Context? (Jiahai Feng et al., 2023)

{{<citation>}}

Jiahai Feng, Jacob Steinhardt. (2023)  
**How do Language Models Bind Entities in Context?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.17191v1)  

---


**ABSTRACT**  
To correctly use in-context information, language models (LMs) must bind entities to their attributes. For example, given a context describing a "green square" and a "blue circle", LMs must bind the shapes to their respective colors. We analyze LM representations and identify the binding ID mechanism: a general mechanism for solving the binding problem, which we observe in every sufficiently large model from the Pythia and LLaMA families. Using causal interventions, we show that LMs' internal activations represent binding information by attaching binding ID vectors to corresponding entities and attributes. We further show that binding ID vectors form a continuous subspace, in which distances between binding ID vectors reflect their discernability. Overall, our results uncover interpretable strategies in LMs for representing symbolic knowledge in-context, providing a step towards understanding general in-context reasoning in large-scale LMs.

{{</citation>}}


### (100/139) Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time (Zichang Liu et al., 2023)

{{<citation>}}

Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Re, Beidi Chen. (2023)  
**Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2310.17157v1)  

---


**ABSTRACT**  
Large language models (LLMs) with hundreds of billions of parameters have sparked a new wave of exciting AI applications. However, they are computationally expensive at inference time. Sparsity is a natural approach to reduce this cost, but existing methods either require costly retraining, have to forgo LLM's in-context learning ability, or do not yield wall-clock time speedup on modern hardware. We hypothesize that contextual sparsity, which are small, input-dependent sets of attention heads and MLP parameters that yield approximately the same output as the dense model for a given input, can address these issues. We show that contextual sparsity exists, that it can be accurately predicted, and that we can exploit it to speed up LLM inference in wall-clock time without compromising LLM's quality or in-context learning ability. Based on these insights, we propose DejaVu, a system that uses a low-cost algorithm to predict contextual sparsity on the fly given inputs to each layer, along with an asynchronous and hardware-aware implementation that speeds up LLM inference. We validate that DejaVu can reduce the inference latency of OPT-175B by over 2X compared to the state-of-the-art FasterTransformer, and over 6X compared to the widely used Hugging Face implementation, without compromising model quality. The code is available at https://github.com/FMInference/DejaVu.

{{</citation>}}


### (101/139) Spatio-Temporal Meta Contrastive Learning (Jiabin Tang et al., 2023)

{{<citation>}}

Jiabin Tang, Lianghao Xia, Jie Hu, Chao Huang. (2023)  
**Spatio-Temporal Meta Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning, GNN  
[Paper Link](http://arxiv.org/abs/2310.17678v1)  

---


**ABSTRACT**  
Spatio-temporal prediction is crucial in numerous real-world applications, including traffic forecasting and crime prediction, which aim to improve public transportation and safety management. Many state-of-the-art models demonstrate the strong capability of spatio-temporal graph neural networks (STGNN) to capture complex spatio-temporal correlations. However, despite their effectiveness, existing approaches do not adequately address several key challenges. Data quality issues, such as data scarcity and sparsity, lead to data noise and a lack of supervised signals, which significantly limit the performance of STGNN. Although recent STGNN models with contrastive learning aim to address these challenges, most of them use pre-defined augmentation strategies that heavily depend on manual design and cannot be customized for different Spatio-Temporal Graph (STG) scenarios. To tackle these challenges, we propose a new spatio-temporal contrastive learning (CL4ST) framework to encode robust and generalizable STG representations via the STG augmentation paradigm. Specifically, we design the meta view generator to automatically construct node and edge augmentation views for each disentangled spatial and temporal graph in a data-driven manner. The meta view generator employs meta networks with parameterized generative model to customize the augmentations for each input. This personalizes the augmentation strategies for every STG and endows the learning framework with spatio-temporal-aware information. Additionally, we integrate a unified spatio-temporal graph attention network with the proposed meta view generator and two-branch graph contrastive learning paradigms. Extensive experiments demonstrate that our CL4ST significantly improves performance over various state-of-the-art baselines in traffic and crime prediction.

{{</citation>}}


### (102/139) Explainable Spatio-Temporal Graph Neural Networks (Jiabin Tang et al., 2023)

{{<citation>}}

Jiabin Tang, Lianghao Xia, Chao Huang. (2023)  
**Explainable Spatio-Temporal Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.17149v1)  

---


**ABSTRACT**  
Spatio-temporal graph neural networks (STGNNs) have gained popularity as a powerful tool for effectively modeling spatio-temporal dependencies in diverse real-world urban applications, including intelligent transportation and public safety. However, the black-box nature of STGNNs limits their interpretability, hindering their application in scenarios related to urban resource allocation and policy formulation. To bridge this gap, we propose an Explainable Spatio-Temporal Graph Neural Networks (STExplainer) framework that enhances STGNNs with inherent explainability, enabling them to provide accurate predictions and faithful explanations simultaneously. Our framework integrates a unified spatio-temporal graph attention network with a positional information fusion layer as the STG encoder and decoder, respectively. Furthermore, we propose a structure distillation approach based on the Graph Information Bottleneck (GIB) principle with an explainable objective, which is instantiated by the STG encoder and decoder. Through extensive experiments, we demonstrate that our STExplainer outperforms state-of-the-art baselines in terms of predictive accuracy and explainability metrics (i.e., sparsity and fidelity) on traffic and crime prediction tasks. Furthermore, our model exhibits superior representation ability in alleviating data missing and sparsity issues. The implementation code is available at: https://github.com/HKUDS/STExplainer.

{{</citation>}}


### (103/139) Understanding and Addressing the Pitfalls of Bisimulation-based Representations in Offline Reinforcement Learning (Hongyu Zang et al., 2023)

{{<citation>}}

Hongyu Zang, Xin Li, Leiji Zhang, Yang Liu, Baigui Sun, Riashat Islam, Remi Tachet des Combes, Romain Laroche. (2023)  
**Understanding and Addressing the Pitfalls of Bisimulation-based Representations in Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.17139v1)  

---


**ABSTRACT**  
While bisimulation-based approaches hold promise for learning robust state representations for Reinforcement Learning (RL) tasks, their efficacy in offline RL tasks has not been up to par. In some instances, their performance has even significantly underperformed alternative methods. We aim to understand why bisimulation methods succeed in online settings, but falter in offline tasks. Our analysis reveals that missing transitions in the dataset are particularly harmful to the bisimulation principle, leading to ineffective estimation. We also shed light on the critical role of reward scaling in bounding the scale of bisimulation measurements and of the value error they induce. Based on these findings, we propose to apply the expectile operator for representation learning to our offline RL setting, which helps to prevent overfitting to incomplete data. Meanwhile, by introducing an appropriate reward scaling strategy, we avoid the risk of feature collapse in representation space. We implement these recommendations on two state-of-the-art bisimulation-based algorithms, MICo and SimSR, and demonstrate performance gains on two benchmark suites: D4RL and Visual D4RL. Codes are provided at \url{https://github.com/zanghyu/Offline_Bisimulation}.

{{</citation>}}


### (104/139) Unleashing the potential of GNNs via Bi-directional Knowledge Transfer (Shuai Zheng et al., 2023)

{{<citation>}}

Shuai Zheng, Zhizhe Liu, Zhenfeng Zhu, Xingxing Zhang, Jianxin Li, Yao Zhao. (2023)  
**Unleashing the potential of GNNs via Bi-directional Knowledge Transfer**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.17132v1)  

---


**ABSTRACT**  
Based on the message-passing paradigm, there has been an amount of research proposing diverse and impressive feature propagation mechanisms to improve the performance of GNNs. However, less focus has been put on feature transformation, another major operation of the message-passing framework. In this paper, we first empirically investigate the performance of the feature transformation operation in several typical GNNs. Unexpectedly, we notice that GNNs do not completely free up the power of the inherent feature transformation operation. By this observation, we propose the Bi-directional Knowledge Transfer (BiKT), a plug-and-play approach to unleash the potential of the feature transformation operations without modifying the original architecture. Taking the feature transformation operation as a derived representation learning model that shares parameters with the original GNN, the direct prediction by this model provides a topological-agnostic knowledge feedback that can further instruct the learning of GNN and the feature transformations therein. On this basis, BiKT not only allows us to acquire knowledge from both the GNN and its derived model but promotes each other by injecting the knowledge into the other. In addition, a theoretical analysis is further provided to demonstrate that BiKT improves the generalization bound of the GNNs from the perspective of domain adaption. An extensive group of experiments on up to 7 datasets with 5 typical GNNs demonstrates that BiKT brings up to 0.5% - 4% performance gain over the original GNN, which means a boosted GNN is obtained. Meanwhile, the derived model also shows a powerful performance to compete with or even surpass the original GNN, enabling us to flexibly apply it independently to some other specific downstream tasks.

{{</citation>}}


### (105/139) LLM4DyG: Can Large Language Models Solve Problems on Dynamic Graphs? (Zeyang Zhang et al., 2023)

{{<citation>}}

Zeyang Zhang, Xin Wang, Ziwei Zhang, Haoyang Li, Yijian Qin, Simin Wu, Wenwu Zhu. (2023)  
**LLM4DyG: Can Large Language Models Solve Problems on Dynamic Graphs?**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17110v1)  

---


**ABSTRACT**  
In an era marked by the increasing adoption of Large Language Models (LLMs) for various tasks, there is a growing focus on exploring LLMs' capabilities in handling web data, particularly graph data. Dynamic graphs, which capture temporal network evolution patterns, are ubiquitous in real-world web data. Evaluating LLMs' competence in understanding spatial-temporal information on dynamic graphs is essential for their adoption in web applications, which remains unexplored in the literature. In this paper, we bridge the gap via proposing to evaluate LLMs' spatial-temporal understanding abilities on dynamic graphs, to the best of our knowledge, for the first time. Specifically, we propose the LLM4DyG benchmark, which includes nine specially designed tasks considering the capability evaluation of LLMs from both temporal and spatial dimensions. Then, we conduct extensive experiments to analyze the impacts of different data generators, data statistics, prompting techniques, and LLMs on the model performance. Finally, we propose Disentangled Spatial-Temporal Thoughts (DST2) for LLMs on dynamic graphs to enhance LLMs' spatial-temporal understanding abilities. Our main observations are: 1) LLMs have preliminary spatial-temporal understanding abilities on dynamic graphs, 2) Dynamic graph tasks show increasing difficulties for LLMs as the graph size and density increase, while not sensitive to the time span and data generation mechanism, 3) the proposed DST2 prompting method can help to improve LLMs' spatial-temporal understanding abilities on dynamic graphs for most tasks. The data and codes will be open-sourced at publication time.

{{</citation>}}


### (106/139) MIM-GAN-based Anomaly Detection for Multivariate Time Series Data (Shan Lu et al., 2023)

{{<citation>}}

Shan Lu, Zhicheng Dong, Donghong Cai, Fang Fang, Dongcai Zhao. (2023)  
**MIM-GAN-based Anomaly Detection for Multivariate Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2310.18257v1)  

---


**ABSTRACT**  
The loss function of Generative adversarial network(GAN) is an important factor that affects the quality and diversity of the generated samples for anomaly detection. In this paper, we propose an unsupervised multiple time series anomaly detection algorithm based on the GAN with message importance measure(MIM-GAN). In particular, the time series data is divided into subsequences using a sliding window. Then a generator and a discriminator designed based on the Long Short-Term Memory (LSTM) are employed to capture the temporal correlations of the time series data. To avoid the local optimal solution of loss function and the model collapse, we introduce an exponential information measure into the loss function of GAN. Additionally, a discriminant reconstruction score consisting on discrimination and reconstruction loss is taken into account. The global optimal solution for the loss function is derived and the model collapse is proved to be avoided in our proposed MIM-GAN-based anomaly detection algorithm. Experimental results show that the proposed MIM-GAN-based anomaly detection algorithm has superior performance in terms of precision, recall, and F1 score.

{{</citation>}}


### (107/139) Network Design through Graph Neural Networks: Identifying Challenges and Improving Performance (Donald Loveland et al., 2023)

{{<citation>}}

Donald Loveland, Rajmonda Caceres. (2023)  
**Network Design through Graph Neural Networks: Identifying Challenges and Improving Performance**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.17100v1)  

---


**ABSTRACT**  
Graph Neural Network (GNN) research has produced strategies to modify a graph's edges using gradients from a trained GNN, with the goal of network design. However, the factors which govern gradient-based editing are understudied, obscuring why edges are chosen and if edits are grounded in an edge's importance. Thus, we begin by analyzing the gradient computation in previous works, elucidating the factors that influence edits and highlighting the potential over-reliance on structural properties. Specifically, we find that edges can achieve high gradients due to structural biases, rather than importance, leading to erroneous edits when the factors are unrelated to the design task. To improve editing, we propose ORE, an iterative editing method that (a) edits the highest scoring edges and (b) re-embeds the edited graph to refresh gradients, leading to less biased edge choices. We empirically study ORE through a set of proposed design tasks, each with an external validation method, demonstrating that ORE improves upon previous methods by up to 50%.

{{</citation>}}


### (108/139) Transformers Learn Higher-Order Optimization Methods for In-Context Learning: A Study with Linear Models (Deqing Fu et al., 2023)

{{<citation>}}

Deqing Fu, Tian-Qi Chen, Robin Jia, Vatsal Sharan. (2023)  
**Transformers Learn Higher-Order Optimization Methods for In-Context Learning: A Study with Linear Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.17086v1)  

---


**ABSTRACT**  
Transformers are remarkably good at in-context learning (ICL) -- learning from demonstrations without parameter updates -- but how they perform ICL remains a mystery. Recent work suggests that Transformers may learn in-context by internally running Gradient Descent, a first-order optimization method. In this paper, we instead demonstrate that Transformers learn to implement higher-order optimization methods to perform ICL. Focusing on in-context linear regression, we show that Transformers learn to implement an algorithm very similar to Iterative Newton's Method, a higher-order optimization method, rather than Gradient Descent. Empirically, we show that predictions from successive Transformer layers closely match different iterations of Newton's Method linearly, with each middle layer roughly computing 3 iterations. In contrast, exponentially more Gradient Descent steps are needed to match an additional Transformers layer; this suggests that Transformers have an comparable rate of convergence with high-order methods such as Iterative Newton, which are exponentially faster than Gradient Descent. We also show that Transformers can learn in-context on ill-conditioned data, a setting where Gradient Descent struggles but Iterative Newton succeeds. Finally, we show theoretical results which support our empirical findings and have a close correspondence with them: we prove that Transformers can implement $k$ iterations of Newton's method with $\mathcal{O}(k)$ layers.

{{</citation>}}


## eess.AS (2)



### (109/139) BERT-PIN: A BERT-based Framework for Recovering Missing Data Segments in Time-series Load Profiles (Yi Hu et al., 2023)

{{<citation>}}

Yi Hu, Kai Ye, Hyeonjin Kim, Ning Lu. (2023)  
**BERT-PIN: A BERT-based Framework for Recovering Missing Data Segments in Time-series Load Profiles**  

---
Primary Category: eess.AS  
Categories: cs-LG, eess-AS, eess-SP, eess.AS  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.17742v1)  

---


**ABSTRACT**  
Inspired by the success of the Transformer model in natural language processing and computer vision, this paper introduces BERT-PIN, a Bidirectional Encoder Representations from Transformers (BERT) powered Profile Inpainting Network. BERT-PIN recovers multiple missing data segments (MDSs) using load and temperature time-series profiles as inputs. To adopt a standard Transformer model structure for profile inpainting, we segment the load and temperature profiles into line segments, treating each segment as a word and the entire profile as a sentence. We incorporate a top candidates selection process in BERT-PIN, enabling it to produce a sequence of probability distributions, based on which users can generate multiple plausible imputed data sets, each reflecting different confidence levels. We develop and evaluate BERT-PIN using real-world dataset for two applications: multiple MDSs recovery and demand response baseline estimation. Simulation results show that BERT-PIN outperforms the existing methods in accuracy while is capable of restoring multiple MDSs within a longer window. BERT-PIN, served as a pre-trained model, can be fine-tuned for conducting many downstream tasks, such as classification and super resolution.

{{</citation>}}


### (110/139) Multi-Speaker Expressive Speech Synthesis via Semi-supervised Contrastive Learning (Xinfa Zhu et al., 2023)

{{<citation>}}

Xinfa Zhu, Yuke Li, Yi Lei, Ning Jiang, Guoqing Zhao, Lei Xie. (2023)  
**Multi-Speaker Expressive Speech Synthesis via Semi-supervised Contrastive Learning**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.17101v1)  

---


**ABSTRACT**  
This paper aims to build an expressive TTS system for multi-speakers, synthesizing a target speaker's speech with multiple styles and emotions. To this end, we propose a novel contrastive learning-based TTS approach to transfer style and emotion across speakers. Specifically, we construct positive-negative sample pairs at both utterance and category (such as emotion-happy or style-poet or speaker A) levels and leverage contrastive learning to better extract disentangled style, emotion, and speaker representations from speech. Furthermore, we introduce a semi-supervised training strategy to the proposed approach to effectively leverage multi-domain data, including style-labeled data, emotion-labeled data, and unlabeled data. We integrate the learned representations into an improved VITS model, enabling it to synthesize expressive speech with diverse styles and emotions for a target speaker. Experiments on multi-domain data demonstrate the good design of our model.

{{</citation>}}


## cs.IR (2)



### (111/139) GNN-GMVO: Graph Neural Networks for Optimizing Gross Merchandise Value in Similar Item Recommendation (Ramin Giahi et al., 2023)

{{<citation>}}

Ramin Giahi, Reza Yousefi Maragheh, Nima Farrokhsiar, Jianpeng Xu, Jason Cho, Evren Korpeoglu, Sushant Kumar, Kannan Achan. (2023)  
**GNN-GMVO: Graph Neural Networks for Optimizing Gross Merchandise Value in Similar Item Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.17732v1)  

---


**ABSTRACT**  
Similar item recommendation is a critical task in the e-Commerce industry, which helps customers explore similar and relevant alternatives based on their interested products. Despite the traditional machine learning models, Graph Neural Networks (GNNs), by design, can understand complex relations like similarity between products. However, in contrast to their wide usage in retrieval tasks and their focus on optimizing the relevance, the current GNN architectures are not tailored toward maximizing revenue-related objectives such as Gross Merchandise Value (GMV), which is one of the major business metrics for e-Commerce companies. In addition, defining accurate edge relations in GNNs is non-trivial in large-scale e-Commerce systems, due to the heterogeneity nature of the item-item relationships. This work aims to address these issues by designing a new GNN architecture called GNN-GMVO (Graph Neural Network - Gross Merchandise Value Optimizer). This model directly optimizes GMV while considering the complex relations between items. In addition, we propose a customized edge construction method to tailor the model toward similar item recommendation task and alleviate the noisy and complex item-item relations. In our comprehensive experiments on three real-world datasets, we show higher prediction performance and expected GMV for top ranked items recommended by our model when compared with selected state-of-the-art benchmark models.

{{</citation>}}


### (112/139) LightLM: A Lightweight Deep and Narrow Language Model for Generative Recommendation (Kai Mei et al., 2023)

{{<citation>}}

Kai Mei, Yongfeng Zhang. (2023)  
**LightLM: A Lightweight Deep and Narrow Language Model for Generative Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: AI, GPT, LLaMA, Language Model, NLP, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2310.17488v2)  

---


**ABSTRACT**  
This paper presents LightLM, a lightweight Transformer-based language model for generative recommendation. While Transformer-based generative modeling has gained importance in various AI sub-fields such as NLP and vision, generative recommendation is still in its infancy due to its unique demand on personalized generative modeling. Existing works on generative recommendation often use NLP-oriented Transformer architectures such as T5, GPT, LLaMA and M6, which are heavy-weight and are not specifically designed for recommendation tasks. LightLM tackles the issue by introducing a light-weight deep and narrow Transformer architecture, which is specifically tailored for direct generation of recommendation items. This structure is especially apt for straightforward generative recommendation and stems from the observation that language model does not have to be too wide for this task, as the input predominantly consists of short tokens that are well-suited for the model's capacity. We also show that our devised user and item ID indexing methods, i.e., Spectral Collaborative Indexing (SCI) and Graph Collaborative Indexing (GCI), enables the deep and narrow Transformer architecture to outperform large-scale language models for recommendation. Besides, to address the hallucination problem of generating items as output, we propose the constrained generation process for generative recommenders. Experiments on real-world datasets show that LightLM outperforms various competitive baselines in terms of both recommendation accuracy and efficiency. The code can be found at https://github.com/dongyuanjushi/LightLM.

{{</citation>}}


## econ.GN (1)



### (113/139) From Transcripts to Insights: Uncovering Corporate Risks Using Generative AI (Alex Kim et al., 2023)

{{<citation>}}

Alex Kim, Maximilian Muhn, Valeri Nikolaev. (2023)  
**From Transcripts to Insights: Uncovering Corporate Risks Using Generative AI**  

---
Primary Category: econ.GN  
Categories: cs-AI, cs-CL, econ-GN, econ.GN, q-fin-EC  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.17721v1)  

---


**ABSTRACT**  
We explore the value of generative AI tools, such as ChatGPT, in helping investors uncover dimensions of corporate risk. We develop and validate firm-level measures of risk exposure to political, climate, and AI-related risks. Using the GPT 3.5 model to generate risk summaries and assessments from the context provided by earnings call transcripts, we show that GPT-based measures possess significant information content and outperform the existing risk measures in predicting (abnormal) firm-level volatility and firms' choices such as investment and innovation. Importantly, information in risk assessments dominates that in risk summaries, establishing the value of general AI knowledge. We also find that generative AI is effective at detecting emerging risks, such as AI risk, which has soared in recent quarters. Our measures perform well both within and outside the GPT's training window and are priced in equity markets. Taken together, an AI-based approach to risk measurement provides useful insights to users of corporate disclosures at a low cost.

{{</citation>}}


## stat.ML (1)



### (114/139) Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec (Andrew Davison et al., 2023)

{{<citation>}}

Andrew Davison, S. Carlyle Morgan, Owen G. Ward. (2023)  
**Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec**  

---
Primary Category: stat.ML  
Categories: cs-LG, cs-SI, stat-ME, stat-ML, stat.ML  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.17712v1)  

---


**ABSTRACT**  
Embedding the nodes of a large network into an Euclidean space is a common objective in modern machine learning, with a variety of tools available. These embeddings can then be used as features for tasks such as community detection/node clustering or link prediction, where they achieve state of the art performance. With the exception of spectral clustering methods, there is little theoretical understanding for other commonly used approaches to learning embeddings. In this work we examine the theoretical properties of the embeddings learned by node2vec. Our main result shows that the use of k-means clustering on the embedding vectors produced by node2vec gives weakly consistent community recovery for the nodes in (degree corrected) stochastic block models. We also discuss the use of these embeddings for node and link prediction tasks. We demonstrate this result empirically, and examine how this relates to other embedding tools for network data.

{{</citation>}}


## cs.NI (1)



### (115/139) A Wireless AI-Generated Content (AIGC) Provisioning Framework Empowered by Semantic Communication (Runze Cheng et al., 2023)

{{<citation>}}

Runze Cheng, Yao Sun, Dusit Niyato, Lan Zhang, Lei Zhang, Muhammad Ali Imran. (2023)  
**A Wireless AI-Generated Content (AIGC) Provisioning Framework Empowered by Semantic Communication**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-LG, cs-NI, cs.NI, eess-IV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.17705v1)  

---


**ABSTRACT**  
Generative AI applications are recently catering to a vast user base by creating diverse and high-quality AI-generated content (AIGC). With the proliferation of mobile devices and rapid growth of mobile traffic, providing ubiquitous access to high-quality AIGC services via wireless communication networks is becoming the future direction for AIGC products. However, it is challenging to provide optimal AIGC services in wireless networks with unstable channels, limited bandwidth resources, and unevenly distributed computational resources. To tackle these challenges, we propose a semantic communication (SemCom)-empowered AIGC (SemAIGC) generation and transmission framework, where only semantic information of the content rather than all the binary bits should be extracted and transmitted by using SemCom. Specifically, SemAIGC integrates diffusion-based models within the semantic encoder and decoder for efficient content generation and flexible adjustment of the computing workload of both transmitter and receiver. Meanwhile, we devise a resource-aware workload trade-off (ROOT) scheme into the SemAIGC framework to intelligently decide transmitter/receiver workload, thus adjusting the utilization of computational resource according to service requirements. Simulations verify the superiority of our proposed SemAIGC framework in terms of latency and content quality compared to conventional approaches.

{{</citation>}}


## cs.CY (5)



### (116/139) Managing AI Risks in an Era of Rapid Progress (Yoshua Bengio et al., 2023)

{{<citation>}}

Yoshua Bengio, Geoffrey Hinton, Andrew Yao, Dawn Song, Pieter Abbeel, Yuval Noah Harari, Ya-Qin Zhang, Lan Xue, Shai Shalev-Shwartz, Gillian Hadfield, Jeff Clune, Tegan Maharaj, Frank Hutter, Atılım Güneş Baydin, Sheila McIlraith, Qiqi Gao, Ashwin Acharya, David Krueger, Anca Dragan, Philip Torr, Stuart Russell, Daniel Kahneman, Jan Brauner, Sören Mindermann. (2023)  
**Managing AI Risks in an Era of Rapid Progress**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17688v1)  

---


**ABSTRACT**  
In this short consensus paper, we outline risks from upcoming, advanced AI systems. We examine large-scale social harms and malicious uses, as well as an irreversible loss of human control over autonomous AI systems. In light of rapid and continuing AI progress, we propose priorities for AI R&D and governance.

{{</citation>}}


### (117/139) Unpacking the Ethical Value Alignment in Big Models (Xiaoyuan Yi et al., 2023)

{{<citation>}}

Xiaoyuan Yi, Jing Yao, Xiting Wang, Xing Xie. (2023)  
**Unpacking the Ethical Value Alignment in Big Models**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17551v1)  

---


**ABSTRACT**  
Big models have greatly advanced AI's ability to understand, generate, and manipulate information and content, enabling numerous applications. However, as these models become increasingly integrated into everyday life, their inherent ethical values and potential biases pose unforeseen risks to society. This paper provides an overview of the risks and challenges associated with big models, surveys existing AI ethics guidelines, and examines the ethical implications arising from the limitations of these models. Taking a normative ethics perspective, we propose a reassessment of recent normative guidelines, highlighting the importance of collaborative efforts in academia to establish a unified and universal AI ethics framework. Furthermore, we investigate the moral inclinations of current mainstream LLMs using the Moral Foundation theory, analyze existing alignment algorithms, and outline the unique challenges encountered in aligning ethical values within them. To address these challenges, we introduce a novel conceptual paradigm for aligning the ethical values of big models and discuss promising research directions for alignment criteria, evaluation, and method, representing an initial step towards the interdisciplinary construction of the ethically aligned AI   This paper is a modified English version of our Chinese paper https://crad.ict.ac.cn/cn/article/doi/10.7544/issn1000-1239.202330553, intended to help non-Chinese native speakers better understand our work.

{{</citation>}}


### (118/139) Decoding The Digital Fuku: Deciphering Colonial Legacies to Critically Assess ChatGPT in Dominican Education (Anaelia Ovalle, 2023)

{{<citation>}}

Anaelia Ovalle. (2023)  
**Decoding The Digital Fuku: Deciphering Colonial Legacies to Critically Assess ChatGPT in Dominican Education**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.17533v2)  

---


**ABSTRACT**  
Educational disparities within the Dominican Republic (DR) have long-standing origins rooted in economic, political, and social inequity. Addressing these challenges has necessarily called for capacity building with respect to educational materials, high-quality instruction, and structural resourcing. Generative AI tools like ChatGPT have begun to pique the interest of Dominican educators due to their perceived potential to bridge these educational gaps. However, a substantial body of AI fairness literature has documented ways AI disproportionately reinforces power dynamics reflective of jurisdictions driving AI development and deployment policies, collectively termed the AI Global North. As such, indiscriminate adoption of this technology for DR education, even in part, risks perpetuating forms of digital coloniality. Therefore, this paper centers embracing AI-facilitated educational reform by critically examining how AI-driven tools like ChatGPT in DR education may replicate facets of digital colonialism. We provide a concise overview of 20th-century Dominican education reforms following the 1916 US occupation. Then, we employ identified neocolonial aspects historically shaping Dominican education to interrogate the perceived advantages of ChatGPT for contemporary Dominican education, as outlined by a Dominican scholar. This work invites AI Global North & South developers, stakeholders, and Dominican leaders alike to exercise a relational contextualization of data-centric epistemologies like ChatGPT to reap its transformative benefits while remaining vigilant of safeguarding Dominican digital sovereignty.

{{</citation>}}


### (119/139) Bias in Evaluation Processes: An Optimization-Based Model (L. Elisa Celis et al., 2023)

{{<citation>}}

L. Elisa Celis, Amit Kumar, Anay Mehrotra, Nisheeth K. Vishnoi. (2023)  
**Bias in Evaluation Processes: An Optimization-Based Model**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-LG, cs.CY, stat-ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.17489v1)  

---


**ABSTRACT**  
Biases with respect to socially-salient attributes of individuals have been well documented in evaluation processes used in settings such as admissions and hiring. We view such an evaluation process as a transformation of a distribution of the true utility of an individual for a task to an observed distribution and model it as a solution to a loss minimization problem subject to an information constraint. Our model has two parameters that have been identified as factors leading to biases: the resource-information trade-off parameter in the information constraint and the risk-averseness parameter in the loss function. We characterize the distributions that arise from our model and study the effect of the parameters on the observed distribution. The outputs of our model enrich the class of distributions that can be used to capture variation across groups in the observed evaluations. We empirically validate our model by fitting real-world datasets and use it to study the effect of interventions in a downstream selection task. These results contribute to an understanding of the emergence of bias in evaluation processes and provide tools to guide the deployment of interventions to mitigate biases.

{{</citation>}}


### (120/139) Supercharging academic writing with generative AI: framework, techniques, and caveats (Zhicheng Lin, 2023)

{{<citation>}}

Zhicheng Lin. (2023)  
**Supercharging academic writing with generative AI: framework, techniques, and caveats**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17143v1)  

---


**ABSTRACT**  
Academic writing is an indispensable yet laborious part of the research enterprise. This Perspective maps out principles and methods for using generative artificial intelligence (AI), specifically large language models (LLMs), to elevate the quality and efficiency of academic writing. We introduce a human-AI collaborative framework that delineates the rationale (why), process (how), and nature (what) of AI engagement in writing. The framework pinpoints both short-term and long-term reasons for engagement and their underlying mechanisms (e.g., cognitive offloading and imaginative stimulation). It reveals the role of AI throughout the writing process, conceptualized through a two-stage model for human-AI collaborative writing, and the nature of AI assistance in writing, represented through a model of writing-assistance types and levels. Building on this framework, we describe effective prompting techniques for incorporating AI into the writing routine (outlining, drafting, and editing) as well as strategies for maintaining rigorous scholarship, adhering to varied journal policies, and avoiding overreliance on AI. Ultimately, the prudent integration of AI into academic writing can ease the communication burden, empower authors, accelerate discovery, and promote diversity in science.

{{</citation>}}


## physics.plasm-ph (1)



### (121/139) Do Graph Neural Networks Dream of Landau Damping? Insights from Kinetic Simulations of a Plasma Sheet Model (Diogo D Carvalho et al., 2023)

{{<citation>}}

Diogo D Carvalho, Diogo R Ferreira, Luis O Silva. (2023)  
**Do Graph Neural Networks Dream of Landau Damping? Insights from Kinetic Simulations of a Plasma Sheet Model**  

---
Primary Category: physics.plasm-ph  
Categories: cs-LG, physics-comp-ph, physics-plasm-ph, physics.plasm-ph  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.17646v1)  

---


**ABSTRACT**  
We explore the possibility of fully replacing a plasma physics kinetic simulator with a graph neural network-based simulator. We focus on this class of surrogate models given the similarity between their message-passing update mechanism and the traditional physics solver update, and the possibility of enforcing known physical priors into the graph construction and update. We show that our model learns the kinetic plasma dynamics of the one-dimensional plasma model, a predecessor of contemporary kinetic plasma simulation codes, and recovers a wide range of well-known kinetic plasma processes, including plasma thermalization, electrostatic fluctuations about thermal equilibrium, and the drag on a fast sheet and Landau damping. We compare the performance against the original plasma model in terms of run-time, conservation laws, and temporal evolution of key physical quantities. The limitations of the model are presented and possible directions for higher-dimensional surrogate models for kinetic plasmas are discussed.

{{</citation>}}


## cs.PL (1)



### (122/139) Verifying Programs with Logic and Extended Proof Rules: Deep Embedding v.s. Shallow Embedding (Zhongye Wang et al., 2023)

{{<citation>}}

Zhongye Wang, Qinxiang Cao, Yichen Tao. (2023)  
**Verifying Programs with Logic and Extended Proof Rules: Deep Embedding v.s. Shallow Embedding**  

---
Primary Category: cs.PL  
Categories: cs-PL, cs.PL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.17616v1)  

---


**ABSTRACT**  
Many foundational program verification tools have been developed to build machine-checked program correctness proofs, a majority of which are based on Hoare logic. Their program logics, their assertion languages, and their underlying programming languages can be formalized by either a shallow embedding or a deep embedding. Tools like Iris and early versions of Verified Software Toolchain (VST) choose different shallow embeddings to formalize their program logics. But the pros and cons of these different embeddings were not yet well studied. Therefore, we want to study the impact of the program logic's embedding on logic's proof rules in this paper. This paper considers a set of useful extended proof rules, and four different logic embeddings: one deep embedding and three common shallow embeddings. We prove the validity of these extended rules under these embeddings and discuss their main challenges. Furthermore, we propose a method to lift existing shallowly embedded logics to deeply embedded ones to greatly simplify proofs of extended rules in specific proof systems. We evaluate our results on two existing verification tools. We lift the originally shallowly embedded VST to our deeply embedded VST to support extended rules, and we implement Iris-CF and deeply embedded Iris-Imp based on the Iris framework to evaluate our theory in real verification projects.

{{</citation>}}


## cs.HC (1)



### (123/139) 1D-Touch: NLP-Assisted Coarse Text Selection via a Semi-Direct Gesture (Peiling Jiang et al., 2023)

{{<citation>}}

Peiling Jiang, Li Feng, Fuling Sun, Parakrant Sarkar, Haijun Xia, Can Liu. (2023)  
**1D-Touch: NLP-Assisted Coarse Text Selection via a Semi-Direct Gesture**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs.HC  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.17576v1)  

---


**ABSTRACT**  
Existing text selection techniques on touchscreen focus on improving the control for moving the carets. Coarse-grained text selection on word and phrase levels has not received much support beyond word-snapping and entity recognition. We introduce 1D-Touch, a novel text selection method that complements the carets-based sub-word selection by facilitating the selection of semantic units of words and above. This method employs a simple vertical slide gesture to expand and contract a selection area from a word. The expansion can be by words or by semantic chunks ranging from sub-phrases to sentences. This technique shifts the concept of text selection, from defining a range by locating the first and last words, towards a dynamic process of expanding and contracting a textual semantic entity. To understand the effects of our approach, we prototyped and tested two variants: WordTouch, which offers a straightforward word-by-word expansion, and ChunkTouch, which leverages NLP to chunk text into syntactic units, allowing the selection to grow by semantically meaningful units in response to the sliding gesture. Our evaluation, focused on the coarse-grained selection tasks handled by 1D-Touch, shows a 20% improvement over the default word-snapping selection method on Android.

{{</citation>}}


## quant-ph (1)



### (124/139) Effective Prime Factorization via Quantum Annealing by Modular Locally-structured Embedding (Jingwen Ding et al., 2023)

{{<citation>}}

Jingwen Ding, Giuseppe Spallitta, Roberto Sebastiani. (2023)  
**Effective Prime Factorization via Quantum Annealing by Modular Locally-structured Embedding**  

---
Primary Category: quant-ph  
Categories: cs-ET, quant-ph, quant-ph  
Keywords: Embedding, QA  
[Paper Link](http://arxiv.org/abs/2310.17574v1)  

---


**ABSTRACT**  
This paper investigates novel techniques to solve prime factorization by quantum annealing (QA). Our contribution is twofold. First, we present a novel and very compact modular encoding of a binary multiplier circuit into the Pegasus architecture of current D-Wave QA devices. The key contribution is a compact encoding of a controlled full-adder into an 8-qubit module in the Pegasus topology, which we synthesized offline by means of Optimization Modulo Theories. This allows us to encode up to a 21*12-bit multiplier (and a 22*8-bit one) into the Pegasus 5760-qubit topology of current annealers. To the best of our knowledge, these are the largest factorization problems ever encoded into a quantum annealer. Second, we have investigated the problem of actually solving encoded PF problems by running an extensive experimental evaluation on a D-Wave Advantage 4.1 quantum annealer. In order to help the annealer in reaching the global minimum, in the experiments we introduced different approaches to initialize the multiplier qubits and adopted several performance enhancement techniques. Overall, exploiting all the encoding and solving techniques described in this paper, 8, 219, 999 = 32, 749 * 251 was the highest prime product we were able to factorize within the limits of our QPU resources. To the best of our knowledge, this is the largest number which was ever factorized by means of a quantum annealer, and, more generally, by a quantum device.

{{</citation>}}


## cs.RO (2)



### (125/139) Test Bench Study on Attitude Estimation in Ground Effect Region Based on Motor Current for In-Flight Inductive Power Transfer of Drones (Kota Fujimoto et al., 2023)

{{<citation>}}

Kota Fujimoto, Sakahisa Nagai, Nguyen Binh Minh, Hiroshi Fujimoto. (2023)  
**Test Bench Study on Attitude Estimation in Ground Effect Region Based on Motor Current for In-Flight Inductive Power Transfer of Drones**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.17541v1)  

---


**ABSTRACT**  
To overcome the short flight duration of drones, research on in-flight inductive power transfer has been recognized as an essential solution. Thus, it is important to accurately estimate and control the attitude of the drones which operate close to the charging surface. To this end, this paper proposes an attitude estimation method based solely on the motor current for precision flight control in the ground effect region. The model for the estimation is derived based on the motor equation when it rotates at a constant rotational speed. The proposed method is verified on the simulations and experiments. It allows simultaneous estimation of altitude and pitch angle with the accuracy of 0.30$\hspace{0.5mm}$m and 0.04 rad, respectively. The minimum transmission efficiency of the in-flight power transfer system based on the proposed estimation is calculated as 95.3 %, which is sufficient for the efficient system.

{{</citation>}}


### (126/139) Optimal Robotic Assembly Sequence Planning: A Sequential Decision-Making Approach (Kartik Nagpal et al., 2023)

{{<citation>}}

Kartik Nagpal, Negar Mehr. (2023)  
**Optimal Robotic Assembly Sequence Planning: A Sequential Decision-Making Approach**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.17115v1)  

---


**ABSTRACT**  
The optimal robot assembly planning problem is challenging due to the necessity of finding the optimal solution amongst an exponentially vast number of possible plans, all while satisfying a selection of constraints. Traditionally, robotic assembly planning problems have been solved using heuristics, but these methods are specific to a given objective structure or set of problem parameters. In this paper, we propose a novel approach to robotic assembly planning that poses assembly sequencing as a sequential decision making problem, enabling us to harness methods that far outperform the state-of-the-art. We formulate the problem as a Markov Decision Process (MDP) and utilize Dynamic Programming (DP) to find optimal assembly policies for moderately sized strictures. We further expand our framework to exploit the deterministic nature of assembly planning and introduce a class of optimal Graph Exploration Assembly Planners (GEAPs). For larger structures, we show how Reinforcement Learning (RL) enables us to learn policies that generate high reward assembly sequences. We evaluate our approach on a variety of robotic assembly problems, such as the assembly of the Hubble Space Telescope, the International Space Station, and the James Webb Space Telescope. We further showcase how our DP, GEAP, and RL implementations are capable of finding optimal solutions under a variety of different objective functions and how our formulation allows us to translate precedence constraints to branch pruning and thus further improve performance. We have published our code at https://github.com/labicon/ORASP-Code.

{{</citation>}}


## eess.SY (3)



### (127/139) Adaptive Resource Management for Edge Network Slicing using Incremental Multi-Agent Deep Reinforcement Learning (Haiyuan Li et al., 2023)

{{<citation>}}

Haiyuan Li, Yuelin Liu, Xueqing Zhou, Xenofon Vasilakos, Reza Nejabati, Shuangyi Yan, Dimitra Simeonidou. (2023)  
**Adaptive Resource Management for Edge Network Slicing using Incremental Multi-Agent Deep Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.17523v2)  

---


**ABSTRACT**  
Multi-access edge computing provides local resources in mobile networks as the essential means for meeting the demands of emerging ultra-reliable low-latency communications. At the edge, dynamic computing requests require advanced resource management for adaptive network slicing, including resource allocations, function scaling and load balancing to utilize only the necessary resources in resource-constraint networks. Recent solutions are designed for a static number of slices. Therefore, the painful process of optimization is required again with any update on the number of slices. In addition, these solutions intend to maximize instant rewards, neglecting long-term resource scheduling. Unlike these efforts, we propose an algorithmic approach based on multi-agent deep deterministic policy gradient (MADDPG) for optimizing resource management for edge network slicing. Our objective is two-fold: (i) maximizing long-term network slicing benefits in terms of delay and energy consumption, and (ii) adapting to slice number changes. Through simulations, we demonstrate that MADDPG outperforms benchmark solutions including a static slicing-based one from the literature, achieving stable and high long-term performance. Additionally, we leverage incremental learning to facilitate a dynamic number of edge slices, with enhanced performance compared to pre-trained base models. Remarkably, this approach yields superior reward performance while saving approximately 90% of training time costs.

{{</citation>}}


### (128/139) Proposal on Model Based Current Overshoot Suppression of Receiver Side Coil in Drone Wireless Power Transfer System (Kota Fujimoto et al., 2023)

{{<citation>}}

Kota Fujimoto, Takumi Hamada, Hiroshi Fujimoto. (2023)  
**Proposal on Model Based Current Overshoot Suppression of Receiver Side Coil in Drone Wireless Power Transfer System**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.17522v1)  

---


**ABSTRACT**  
This paper proposes a model-based control method in the wireless power transfer (WPT) system by operating a semi-bridgeless active rectifier (SBAR) to suppress the secondary coil current overshoot. By damping the current overshoot, it is possible to reduce the rectifier's rated current and decrease the rectifier's size, which is beneficial for the lightweight-oriented system such as drones. In the control method, an inverse of the plant model is used to calculate the reference input to the system. The current overshoot is reduced by operating the SBAR under the duty ratio calculated from the model. To confirm the performance of the proposed method, the simulation and the experiment using the WPT prototype are conducted. The experimental results show that the proposed method can suppress the secondary coil current overshoot. The results suggest it is possible to realize the lighter secondary system by applying the proposed method.

{{</citation>}}


### (129/139) LEI2JSON: Schema-based Validation and Conversion of Livestock Event Information (Mahir Habib et al., 2023)

{{<citation>}}

Mahir Habib, Muhammad Ashad Kabir, Lihong Zheng. (2023)  
**LEI2JSON: Schema-based Validation and Conversion of Livestock Event Information**  

---
Primary Category: eess.SY  
Categories: cs-SE, cs-SY, eess-SY, eess.SY  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.17414v1)  

---


**ABSTRACT**  
Livestock producers often need help in standardising (i.e., converting and validating) their livestock event data. This article introduces a novel solution, LEI2JSON (Livestock Event Information To JSON). The tool is an add-on for Google Sheets, adhering to the livestock event information (LEI) schema. The core objective of LEI2JSON is to provide livestock producers with an efficient mechanism to standardise their data, leading to substantial savings in time and resources. This is achieved by building the spreadsheet template with the appropriate column headers, notes, and validation rules, converting the spreadsheet data into JSON format, and validating the output against the schema. LEI2JSON facilitates the seamless storage of livestock event information locally or on Google Drive in JSON. Additionally, we have conducted an extensive experimental evaluation to assess the effectiveness of the tool.

{{</citation>}}


## cs.SD (1)



### (130/139) Controllable Generation of Artificial Speaker Embeddings through Discovery of Principal Directions (Florian Lux et al., 2023)

{{<citation>}}

Florian Lux, Pascal Tilli, Sarina Meyer, Ngoc Thang Vu. (2023)  
**Controllable Generation of Artificial Speaker Embeddings through Discovery of Principal Directions**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.17502v1)  

---


**ABSTRACT**  
Customizing voice and speaking style in a speech synthesis system with intuitive and fine-grained controls is challenging, given that little data with appropriate labels is available. Furthermore, editing an existing human's voice also comes with ethical concerns. In this paper, we propose a method to generate artificial speaker embeddings that cannot be linked to a real human while offering intuitive and fine-grained control over the voice and speaking style of the embeddings, without requiring any labels for speaker or style. The artificial and controllable embeddings can be fed to a speech synthesis system, conditioned on embeddings of real humans during training, without sacrificing privacy during inference.

{{</citation>}}


## cs.IT (1)



### (131/139) Foundation Model Based Native AI Framework in 6G with Cloud-Edge-End Collaboration (Xiang Chen et al., 2023)

{{<citation>}}

Xiang Chen, Zhiheng Guo, Xijun Wang, Howard H. Yang, Chenyuan Feng, Junshen Su, Sihui Zheng, Tony Q. S. Quek. (2023)  
**Foundation Model Based Native AI Framework in 6G with Cloud-Edge-End Collaboration**  

---
Primary Category: cs.IT  
Categories: cs-DC, cs-IT, cs-LG, cs-NI, cs.IT, eess-SP, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17471v1)  

---


**ABSTRACT**  
Future wireless communication networks are in a position to move beyond data-centric, device-oriented connectivity and offer intelligent, immersive experiences based on task-oriented connections, especially in the context of the thriving development of pre-trained foundation models (PFM) and the evolving vision of 6G native artificial intelligence (AI). Therefore, redefining modes of collaboration between devices and servers and constructing native intelligence libraries become critically important in 6G. In this paper, we analyze the challenges of achieving 6G native AI from the perspectives of data, intelligence, and networks. Then, we propose a 6G native AI framework based on foundation models, provide a customization approach for intent-aware PFM, present a construction of a task-oriented AI toolkit, and outline a novel cloud-edge-end collaboration paradigm. As a practical use case, we apply this framework for orchestration, achieving the maximum sum rate within a wireless communication system, and presenting preliminary evaluation results. Finally, we outline research directions for achieving native AI in 6G.

{{</citation>}}


## cs.CR (4)



### (132/139) A near-autonomous and incremental intrusion detection system through active learning of known and unknown attacks (Lynda Boukela et al., 2023)

{{<citation>}}

Lynda Boukela, Gongxuan Zhang, Meziane Yacoub, Samia Bouzefrane. (2023)  
**A near-autonomous and incremental intrusion detection system through active learning of known and unknown attacks**  

---
Primary Category: cs.CR  
Categories: 68, cs-CR, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2310.17430v1)  

---


**ABSTRACT**  
Intrusion detection is a traditional practice of security experts, however, there are several issues which still need to be tackled. Therefore, in this paper, after highlighting these issues, we present an architecture for a hybrid Intrusion Detection System (IDS) for an adaptive and incremental detection of both known and unknown attacks. The IDS is composed of supervised and unsupervised modules, namely, a Deep Neural Network (DNN) and the K-Nearest Neighbors (KNN) algorithm, respectively. The proposed system is near-autonomous since the intervention of the expert is minimized through the active learning (AL) approach. A query strategy for the labeling process is presented, it aims at teaching the supervised module to detect unknown attacks and improve the detection of the already-known attacks. This teaching is achieved through sliding windows (SW) in an incremental fashion where the DNN is retrained when the data is available over time, thus rendering the IDS adaptive to cope with the evolutionary aspect of the network traffic. A set of experiments was conducted on the CICIDS2017 dataset in order to evaluate the performance of the IDS, promising results were obtained.

{{</citation>}}


### (133/139) Network Intrusion Detection with Edge-Directed Graph Multi-Head Attention Networks (Xiang Li et al., 2023)

{{<citation>}}

Xiang Li, Jing Zhang, Yali Yuan, Cangqi Zhou. (2023)  
**Network Intrusion Detection with Edge-Directed Graph Multi-Head Attention Networks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Attention, GNN, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2310.17348v1)  

---


**ABSTRACT**  
A network intrusion usually involves a number of network locations. Data flow (including the data generated by intrusion behaviors) among these locations (usually represented by IP addresses) naturally forms a graph. Thus, graph neural networks (GNNs) have been used in the construction of intrusion detection models in recent years since they have an excellent ability to capture graph topological features of intrusion data flow. However, existing GNN models treat node mean aggregation equally in node information aggregation. In reality, the correlations of nodes and their neighbors as well as the linked edges are different. Assigning higher weights to nodes and edges with high similarity can highlight the correlation among them, which will enhance the accuracy and expressiveness of the model. To this end, this paper proposes novel Edge-Directed Graph Multi-Head Attention Networks (EDGMAT) for network intrusion detection. The proposed EDGMAT model introduces a multi-head attention mechanism into the intrusion detection model. Additional weight learning is realized through the combination of a multi-head attention mechanism and edge features. Weighted aggregation makes better use of the relationship between different network traffic data. Experimental results on four recent NIDS benchmark datasets show that the performance of EDGMAT in terms of weighted F1-Score is significantly better than that of four state-of-the-art models in multi-class detection tasks.

{{</citation>}}


### (134/139) Static Semantics Reconstruction for Enhancing JavaScript-WebAssembly Multilingual Malware Detection (Yifan Xia et al., 2023)

{{<citation>}}

Yifan Xia, Ping He, Xuhong Zhang, Peiyu Liu, Shouling Ji, Wenhai Wang. (2023)  
**Static Semantics Reconstruction for Enhancing JavaScript-WebAssembly Multilingual Malware Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2310.17304v1)  

---


**ABSTRACT**  
The emergence of WebAssembly allows attackers to hide the malicious functionalities of JavaScript malware in cross-language interoperations, termed JavaScript-WebAssembly multilingual malware (JWMM). However, existing anti-virus solutions based on static program analysis are still limited to monolingual code. As a result, their detection effectiveness decreases significantly against JWMM. The detection of JWMM is challenging due to the complex interoperations and semantic diversity between JavaScript and WebAssembly. To bridge this gap, we present JWBinder, the first technique aimed at enhancing the static detection of JWMM. JWBinder performs a language-specific data-flow analysis to capture the cross-language interoperations and then characterizes the functionalities of JWMM through a unified high-level structure called Inter-language Program Dependency Graph. The extensive evaluation on one of the most representative real-world anti-virus platforms, VirusTotal, shows that \system effectively enhances anti-virus systems from various vendors and increases the overall successful detection rate against JWMM from 49.1\% to 86.2\%. Additionally, we assess the side effects and runtime overhead of JWBinder, corroborating its practical viability in real-world applications.

{{</citation>}}


### (135/139) A Method for Network Intrusion Detection Using Flow Sequence and BERT Framework (Loc Gia Nguyen et al., 2023)

{{<citation>}}

Loc Gia Nguyen, Kohei Watabe. (2023)  
**A Method for Network Intrusion Detection Using Flow Sequence and BERT Framework**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: BERT, Intrusion Detection, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.17127v1)  

---


**ABSTRACT**  
A Network Intrusion Detection System (NIDS) is a tool that identifies potential threats to a network. Recently, different flow-based NIDS designs utilizing Machine Learning (ML) algorithms have been proposed as solutions to detect intrusions efficiently. However, conventional ML-based classifiers have not seen widespread adoption in the real world due to their poor domain adaptation capability. In this research, our goal is to explore the possibility of using sequences of flows to improve the domain adaptation capability of network intrusion detection systems. Our proposal employs natural language processing techniques and Bidirectional Encoder Representations from Transformers framework, which is an effective technique for modeling data with respect to its context. Early empirical results show that our approach has improved domain adaptation capability compared to previous approaches. The proposed approach provides a new research method for building a robust intrusion detection system.

{{</citation>}}


## cs.SE (1)



### (136/139) CodeFusion: A Pre-trained Diffusion Model for Code Generation (Mukul Singh et al., 2023)

{{<citation>}}

Mukul Singh, José Cambronero, Sumit Gulwani, Vu Le, Carina Negreanu, Gust Verbruggen. (2023)  
**CodeFusion: A Pre-trained Diffusion Model for Code Generation**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-PL, cs-SE, cs.SE  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2310.17680v2)  

---


**ABSTRACT**  
Imagine a developer who can only change their last line of code, how often would they have to start writing a function from scratch before it is correct? Auto-regressive models for code generation from natural language have a similar limitation: they do not easily allow reconsidering earlier tokens generated. We introduce CodeFusion, a pre-trained diffusion code generation model that addresses this limitation by iteratively denoising a complete program conditioned on the encoded natural language. We evaluate CodeFusion on the task of natural language to code generation for Bash, Python, and Microsoft Excel conditional formatting (CF) rules. Experiments show that CodeFusion (75M parameters) performs on par with state-of-the-art auto-regressive systems (350M-175B parameters) in top-1 accuracy and outperforms them in top-3 and top-5 accuracy due to its better balance in diversity versus quality.

{{</citation>}}


## cs.SI (1)



### (137/139) Validating Digital Traces with Survey Data: The Use Case of Religiosity (M. Fuat Kına et al., 2023)

{{<citation>}}

M. Fuat Kına, Erdem Yörük, Ali Hürriyetoğlu, Melih Can Yardı, Şükrü Atsızelti, Fırat Duruşan, Oğuz Gürerk, Tolga Etgü, Zübeyir Nişancı, Osman Mutlu, Gizem Bacaksızlar Turbic, Yusuf Akbulut. (2023)  
**Validating Digital Traces with Survey Data: The Use Case of Religiosity**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.17220v1)  

---


**ABSTRACT**  
This paper tests the validity of a digital trace database (Politus) obtained from Twitter, with a recently conducted representative social survey, focusing on the use case of religiosity in Turkey. Religiosity scores in the research are extracted using supervised machine learning under the Politus project. The validation analysis depends on two steps. First, we compare the performances of two alternative tweet-to-user transformation strategies, and second, test for the impact of resampling via the MRP technique. Estimates of the Politus are examined at both aggregate and region-level. The results are intriguing for future research on measuring public opinion via social media data.

{{</citation>}}


## cs.MM (1)



### (138/139) Automatic Edge Error Judgment in Figure Skating Using 3D Pose Estimation from a Monocular Camera and IMUs (Ryota Tanaka et al., 2023)

{{<citation>}}

Ryota Tanaka, Tomohiro Suzuki, Kazuya Takeda, Keisuke Fujii. (2023)  
**Automatic Edge Error Judgment in Figure Skating Using 3D Pose Estimation from a Monocular Camera and IMUs**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17193v1)  

---


**ABSTRACT**  
Automatic evaluating systems are fundamental issues in sports technologies. In many sports, such as figure skating, automated evaluating methods based on pose estimation have been proposed. However, previous studies have evaluated skaters' skills in 2D analysis. In this paper, we propose an automatic edge error judgment system with a monocular smartphone camera and inertial sensors, which enable us to analyze 3D motions. Edge error is one of the most significant scoring items and is challenging to automatically judge due to its 3D motion. The results show that the model using 3D joint position coordinates estimated from the monocular camera as the input feature had the highest accuracy at 83% for unknown skaters' data. We also analyzed the detailed motion analysis for edge error judgment. These results indicate that the monocular camera can be used to judge edge errors automatically. We will provide the figure skating single Lutz jump dataset, including pre-processed videos and labels, at https://github.com/ryota-takedalab/JudgeAI-LutzEdge.

{{</citation>}}


## astro-ph.EP (1)



### (139/139) CosmosDSR -- a methodology for automated detection and tracking of orbital debris using the Unscented Kalman Filter (Daniel S. Roll et al., 2023)

{{<citation>}}

Daniel S. Roll, Zeyneb Kurt, Wai Lok Woo. (2023)  
**CosmosDSR -- a methodology for automated detection and tracking of orbital debris using the Unscented Kalman Filter**  

---
Primary Category: astro-ph.EP  
Categories: 68, I-2-6; K-3-2, astro-ph-EP, astro-ph-IM, astro-ph.EP, cs-AI, cs-CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17158v2)  

---


**ABSTRACT**  
The Kessler syndrome refers to the escalating space debris from frequent space activities, threatening future space exploration. Addressing this issue is vital. Several AI models, including Convolutional Neural Networks, Kernel Principal Component Analysis, and Model-Agnostic Meta- Learning have been assessed with various data types. Earlier studies highlighted the combination of the YOLO object detector and a linear Kalman filter (LKF) for object detection and tracking. Advancing this, the current paper introduces a novel methodology for the Comprehensive Orbital Surveillance and Monitoring Of Space by Detecting Satellite Residuals (CosmosDSR) by combining YOLOv3 with an Unscented Kalman Filter (UKF) for tracking satellites in sequential images. Using the Spacecraft Recognition Leveraging Knowledge of Space Environment (SPARK) dataset for training and testing, the YOLOv3 precisely detected and classified all satellite categories (Mean Average Precision=97.18%, F1=0.95) with few errors (TP=4163, FP=209, FN=237). Both CosmosDSR and an implemented LKF used for comparison tracked satellites accurately for a mean squared error (MSE) and root mean squared error (RME) of MSE=2.83/RMSE=1.66 for UKF and MSE=2.84/RMSE=1.66 for LKF. The current study is limited to images generated in a space simulation environment, but the CosmosDSR methodology shows great potential in detecting and tracking satellites, paving the way for solutions to the Kessler syndrome.

{{</citation>}}
