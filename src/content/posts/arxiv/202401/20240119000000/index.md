---
draft: false
title: "arXiv @ 2024.01.19"
date: 2024-01-19
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.19"
    identifier: arxiv_20240119
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.SI (2)](#cssi-2)
- [cs.LG (30)](#cslg-30)
- [eess.IV (4)](#eessiv-4)
- [cs.HC (3)](#cshc-3)
- [cs.DB (1)](#csdb-1)
- [cs.CL (17)](#cscl-17)
- [cs.CV (22)](#cscv-22)
- [cs.CR (7)](#cscr-7)
- [cs.CY (4)](#cscy-4)
- [cs.IT (2)](#csit-2)
- [cs.RO (6)](#csro-6)
- [cs.DC (2)](#csdc-2)
- [cs.SD (3)](#cssd-3)
- [cs.NE (1)](#csne-1)
- [cs.NI (1)](#csni-1)
- [cs.AI (3)](#csai-3)
- [eess.AS (1)](#eessas-1)
- [stat.AP (1)](#statap-1)
- [cs.AR (1)](#csar-1)
- [cs.OS (1)](#csos-1)

## cs.SI (2)



### (1/112) Characterizing Online Eating Disorder Communities with Large Language Models (Minh Duc Chu et al., 2024)

{{<citation>}}

Minh Duc Chu, Aryan Karnati, Zihao He, Kristina Lerman. (2024)  
**Characterizing Online Eating Disorder Communities with Large Language Models**  

---
Primary Category: cs.SI  
Categories: cs-CL, cs-CY, cs-SI, cs.SI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09647v1)  

---


**ABSTRACT**  
The rise in eating disorders, a dangerous mental health condition with high mortality and morbidity, has been linked to the proliferation of idealized body images on social media. However, the link between social media and eating disorders is far more complex. We argue that social media platforms create a feedback loop that amplifies the growth of content and communities that promote eating disorders like anorexia and bulimia. Specifically, social media platforms make it easy for vulnerable individuals to find and connect to like-minded others, while group dynamic processes encourage them to stay engaged within communities that promote and glorify harmful behaviors linked to eating disorders. We characterize this dynamic empirically through a combination of network and language analysis. We describe a novel framework that leverages large language models to analyze the discourse within online communities and probe their attitudes on topics related to eating disorders to identify potentially harmful content. Our work emphasizes the need for better social media moderation to disrupt harmful feedback loops and protect vulnerable individuals.

{{</citation>}}


### (2/112) Calibrate-Extrapolate: Rethinking Prevalence Estimation with Black Box Classifiers (Siqi Wu et al., 2024)

{{<citation>}}

Siqi Wu, Paul Resnick. (2024)  
**Calibrate-Extrapolate: Rethinking Prevalence Estimation with Black Box Classifiers**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.09329v1)  

---


**ABSTRACT**  
In computational social science, researchers often use a pre-trained, black box classifier to estimate the frequency of each class in unlabeled datasets. A variety of prevalence estimation techniques have been developed in the literature, each yielding an unbiased estimate if certain stability assumption holds. This work introduces a framework to rethink the prevalence estimation process as calibrating the classifier outputs against ground truth labels to obtain the joint distribution of a base dataset and then extrapolating to the joint distribution of a target dataset. We call this framework "Calibrate-Extrapolate". Visualizing the joint distribution makes the stability assumption needed for a prevalence estimation technique clear and easy to understand. In the calibration phase, the techniques assume only a stable calibration curve between a calibration dataset and the full base dataset. This allows for the classifier outputs to be used for purposive sampling, thus improving the efficiency of calibration. In the extrapolation phase, some techniques assume a stable calibration curve while some assume stable class-conditional densities. We discuss the stability assumptions from a causal perspective. By specifying base and target joint distributions, we can generate simulated datasets, as a way to build intuitions about the impacts of assumption violations. This also leads to a better understanding of how the classifier predictive power affects the accuracy of prevalence estimates: the greater the predictive power, the lower the sensitivity to violations of stability assumptions in the extrapolation phase. We illustrate the framework with an application that estimates the prevalence of toxic news comments over time on Reddit, Twitter, and YouTube, using Jigsaw's Perspective API as a black box classifier.

{{</citation>}}


## cs.LG (30)



### (3/112) ClimateGPT: Towards AI Synthesizing Interdisciplinary Research on Climate Change (David Thulke et al., 2024)

{{<citation>}}

David Thulke, Yingbo Gao, Petrus Pelser, Rein Brune, Rricha Jalota, Floris Fok, Michael Ramos, Ian van Wyk, Abdallah Nasir, Hayden Goldstein, Taylor Tragemann, Katie Nguyen, Ariana Fowler, Andrew Stanco, Jon Gabriel, Jordan Taylor, Dean Moro, Evgenii Tsymbalov, Juliette de Waal, Evgeny Matusov, Mudar Yaghi, Mohammad Shihadah, Hermann Ney, Christian Dugast, Jonathan Dotan, Daniel Erasmus. (2024)  
**ClimateGPT: Towards AI Synthesizing Interdisciplinary Research on Climate Change**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2401.09646v1)  

---


**ABSTRACT**  
This paper introduces ClimateGPT, a model family of domain-specific large language models that synthesize interdisciplinary research on climate change. We trained two 7B models from scratch on a science-oriented dataset of 300B tokens. For the first model, the 4.2B domain-specific tokens were included during pre-training and the second was adapted to the climate domain after pre-training. Additionally, ClimateGPT-7B, 13B and 70B are continuously pre-trained from Llama~2 on a domain-specific dataset of 4.2B tokens. Each model is instruction fine-tuned on a high-quality and human-generated domain-specific dataset that has been created in close cooperation with climate scientists. To reduce the number of hallucinations, we optimize the model for retrieval augmentation and propose a hierarchical retrieval strategy. To increase the accessibility of our model to non-English speakers, we propose to make use of cascaded machine translation and show that this approach can perform comparably to natively multilingual models while being easier to scale to a large number of languages. Further, to address the intrinsic interdisciplinary aspect of climate change we consider different research perspectives. Therefore, the model can produce in-depth answers focusing on different perspectives in addition to an overall answer. We propose a suite of automatic climate-specific benchmarks to evaluate LLMs. On these benchmarks, ClimateGPT-7B performs on par with the ten times larger Llama-2-70B Chat model while not degrading results on general domain benchmarks. Our human evaluation confirms the trends we saw in our benchmarks. All models were trained and evaluated using renewable energy and are released publicly.

{{</citation>}}


### (4/112) Bilevel Optimization under Unbounded Smoothness: A New Algorithm and Convergence Analysis (Jie Hao et al., 2024)

{{<citation>}}

Jie Hao, Xiaochuan Gong, Mingrui Liu. (2024)  
**Bilevel Optimization under Unbounded Smoothness: A New Algorithm and Convergence Analysis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.09587v1)  

---


**ABSTRACT**  
Bilevel optimization is an important formulation for many machine learning problems. Current bilevel optimization algorithms assume that the gradient of the upper-level function is Lipschitz. However, recent studies reveal that certain neural networks such as recurrent neural networks (RNNs) and long-short-term memory networks (LSTMs) exhibit potential unbounded smoothness, rendering conventional bilevel optimization algorithms unsuitable. In this paper, we design a new bilevel optimization algorithm, namely BO-REP, to address this challenge. This algorithm updates the upper-level variable using normalized momentum and incorporates two novel techniques for updating the lower-level variable: \textit{initialization refinement} and \textit{periodic updates}. Specifically, once the upper-level variable is initialized, a subroutine is invoked to obtain a refined estimate of the corresponding optimal lower-level variable, and the lower-level variable is updated only after every specific period instead of each iteration. When the upper-level problem is nonconvex and unbounded smooth, and the lower-level problem is strongly convex, we prove that our algorithm requires $\widetilde{\mathcal{O}}(1/\epsilon^4)$ iterations to find an $\epsilon$-stationary point in the stochastic setting, where each iteration involves calling a stochastic gradient or Hessian-vector product oracle. Notably, this result matches the state-of-the-art complexity results under the bounded smoothness setting and without mean-squared smoothness of the stochastic gradient, up to logarithmic factors. Our proof relies on novel technical lemmas for the periodically updated lower-level variable, which are of independent interest. Our experiments on hyper-representation learning, hyperparameter optimization, and data hyper-cleaning for text classification tasks demonstrate the effectiveness of our proposed algorithm.

{{</citation>}}


### (5/112) Sharing Knowledge in Multi-Task Deep Reinforcement Learning (Carlo D'Eramo et al., 2024)

{{<citation>}}

Carlo D'Eramo, Davide Tateo, Andrea Bonarini, Marcello Restelli, Jan Peters. (2024)  
**Sharing Knowledge in Multi-Task Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09561v1)  

---


**ABSTRACT**  
We study the benefit of sharing representations among tasks to enable the effective use of deep neural networks in Multi-Task Reinforcement Learning. We leverage the assumption that learning from different tasks, sharing common properties, is helpful to generalize the knowledge of them resulting in a more effective feature extraction compared to learning a single task. Intuitively, the resulting set of features offers performance benefits when used by Reinforcement Learning algorithms. We prove this by providing theoretical guarantees that highlight the conditions for which is convenient to share representations among tasks, extending the well-known finite-time bounds of Approximate Value-Iteration to the multi-task setting. In addition, we complement our analysis by proposing multi-task extensions of three Reinforcement Learning algorithms that we empirically evaluate on widely used Reinforcement Learning benchmarks showing significant improvements over the single-task counterparts in terms of sample efficiency and performance.

{{</citation>}}


### (6/112) Improving Classification Performance With Human Feedback: Label a few, we label the rest (Natan Vidra et al., 2024)

{{<citation>}}

Natan Vidra, Thomas Clifford, Katherine Jijo, Eden Chung, Liang Zhang. (2024)  
**Improving Classification Performance With Human Feedback: Label a few, we label the rest**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: AI, Amazon, BERT, Financial, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09555v1)  

---


**ABSTRACT**  
In the realm of artificial intelligence, where a vast majority of data is unstructured, obtaining substantial amounts of labeled data to train supervised machine learning models poses a significant challenge. To address this, we delve into few-shot and active learning, where are goal is to improve AI models with human feedback on a few labeled examples. This paper focuses on understanding how a continuous feedback loop can refine models, thereby enhancing their accuracy, recall, and precision through incremental human input. By employing Large Language Models (LLMs) such as GPT-3.5, BERT, and SetFit, we aim to analyze the efficacy of using a limited number of labeled examples to substantially improve model accuracy. We benchmark this approach on the Financial Phrasebank, Banking, Craigslist, Trec, Amazon Reviews datasets to prove that with just a few labeled examples, we are able to surpass the accuracy of zero shot large language models to provide enhanced text classification performance. We demonstrate that rather than needing to manually label millions of rows of data, we just need to label a few and the model can effectively predict the rest.

{{</citation>}}


### (7/112) BENO: Boundary-embedded Neural Operators for Elliptic PDEs (Haixin Wang et al., 2024)

{{<citation>}}

Haixin Wang, Jiaxin Li, Anubhav Dwivedi, Kentaro Hara, Tailin Wu. (2024)  
**BENO: Boundary-embedded Neural Operators for Elliptic PDEs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2401.09323v1)  

---


**ABSTRACT**  
Elliptic partial differential equations (PDEs) are a major class of time-independent PDEs that play a key role in many scientific and engineering domains such as fluid dynamics, plasma physics, and solid mechanics. Recently, neural operators have emerged as a promising technique to solve elliptic PDEs more efficiently by directly mapping the input to solutions. However, existing networks typically cannot handle complex geometries and inhomogeneous boundary values present in the real world. Here we introduce Boundary-Embedded Neural Operators (BENO), a novel neural operator architecture that embeds the complex geometries and inhomogeneous boundary values into the solving of elliptic PDEs. Inspired by classical Green's function, BENO consists of two branches of Graph Neural Networks (GNNs) for interior source term and boundary values, respectively. Furthermore, a Transformer encoder maps the global boundary geometry into a latent vector which influences each message passing layer of the GNNs. We test our model extensively in elliptic PDEs with various boundary conditions. We show that all existing baseline methods fail to learn the solution operator. In contrast, our model, endowed with boundary-embedded architecture, outperforms state-of-the-art neural operators and strong baselines by an average of 60.96\%. Our source code can be found https://github.com/AI4Science-WestlakeU/beno.git.

{{</citation>}}


### (8/112) MSHyper: Multi-Scale Hypergraph Transformer for Long-Range Time Series Forecasting (Zongjiang Shang et al., 2024)

{{<citation>}}

Zongjiang Shang, Ling Chen. (2024)  
**MSHyper: Multi-Scale Hypergraph Transformer for Long-Range Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2401.09261v1)  

---


**ABSTRACT**  
Demystifying interactions between temporal patterns of different scales is fundamental to precise long-range time series forecasting. However, previous works lack the ability to model high-order interactions. To promote more comprehensive pattern interaction modeling for long-range time series forecasting, we propose a Multi-Scale Hypergraph Transformer (MSHyper) framework. Specifically, a multi-scale hypergraph is introduced to provide foundations for modeling high-order pattern interactions. Then by treating hyperedges as nodes, we also build a hyperedge graph to enhance hypergraph modeling. In addition, a tri-stage message passing mechanism is introduced to aggregate pattern information and learn the interaction strength between temporal patterns of different scales. Extensive experiments on five real-world datasets demonstrate that MSHyper achieves state-of-the-art performance, reducing prediction errors by an average of 8.73% and 7.15% over the best baseline in MSE and MAE, respectively.

{{</citation>}}


### (9/112) Space and Time Continuous Physics Simulation From Partial Observations (Janny Steeven et al., 2024)

{{<citation>}}

Janny Steeven, Nadri Madiha, Digne Julie, Wolf Christian. (2024)  
**Space and Time Continuous Physics Simulation From Partial Observations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.09198v1)  

---


**ABSTRACT**  
Modern techniques for physical simulations rely on numerical schemes and mesh-refinement methods to address trade-offs between precision and complexity, but these handcrafted solutions are tedious and require high computational power. Data-driven methods based on large-scale machine learning promise high adaptivity by integrating long-range dependencies more directly and efficiently. In this work, we focus on fluid dynamics and address the shortcomings of a large part of the literature, which are based on fixed support for computations and predictions in the form of regular or irregular grids. We propose a novel setup to perform predictions in a continuous spatial and temporal domain while being trained on sparse observations. We formulate the task as a double observation problem and propose a solution with two interlinked dynamical systems defined on, respectively, the sparse positions and the continuous domain, which allows to forecast and interpolate a solution from the initial condition. Our practical implementation involves recurrent GNNs and a spatio-temporal attention observer capable of interpolating the solution at arbitrary locations. Our model not only generalizes to new initial conditions (as standard auto-regressive models do) but also performs evaluation at arbitrary space and time locations. We evaluate on three standard datasets in fluid dynamics and compare to strong baselines, which are outperformed both in classical settings and in the extended new task requiring continuous predictions.

{{</citation>}}


### (10/112) GNN-LoFI: a Novel Graph Neural Network through Localized Feature-based Histogram Intersection (Alessandro Bicciato et al., 2024)

{{<citation>}}

Alessandro Bicciato, Luca Cosmo, Giorgia Minello, Luca Rossi, Andrea Torsello. (2024)  
**GNN-LoFI: a Novel Graph Neural Network through Localized Feature-based Histogram Intersection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2401.09193v1)  

---


**ABSTRACT**  
Graph neural networks are increasingly becoming the framework of choice for graph-based machine learning. In this paper, we propose a new graph neural network architecture that substitutes classical message passing with an analysis of the local distribution of node features. To this end, we extract the distribution of features in the egonet for each local neighbourhood and compare them against a set of learned label distributions by taking the histogram intersection kernel. The similarity information is then propagated to other nodes in the network, effectively creating a message passing-like mechanism where the message is determined by the ensemble of the features. We perform an ablation study to evaluate the network's performance under different choices of its hyper-parameters. Finally, we test our model on standard graph classification and regression benchmarks, and we find that it outperforms widely used alternative approaches, including both graph kernels and graph neural networks.

{{</citation>}}


### (11/112) Preparing Lessons for Progressive Training on Language Models (Yu Pan et al., 2024)

{{<citation>}}

Yu Pan, Ye Yuan, Yichun Yin, Jiaxin Shi, Zenglin Xu, Ming Zhang, Lifeng Shang, Xin Jiang, Qun Liu. (2024)  
**Preparing Lessons for Progressive Training on Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.09192v2)  

---


**ABSTRACT**  
The rapid progress of Transformers in artificial intelligence has come at the cost of increased resource consumption and greenhouse gas emissions due to growing model sizes. Prior work suggests using pretrained small models to improve training efficiency, but this approach may not be suitable for new model structures. On the other hand, training from scratch can be slow, and progressively stacking layers often fails to achieve significant acceleration. To address these challenges, we propose a novel method called Apollo, which prep\textbf{a}res lessons for ex\textbf{p}anding \textbf{o}perations by \textbf{l}earning high-\textbf{l}ayer functi\textbf{o}nality during training of low layers. Our approach involves low-value-prioritized sampling (LVPS) to train different depths and weight sharing to facilitate efficient expansion. We also introduce an interpolation method for stable model depth extension. Experiments demonstrate that Apollo achieves state-of-the-art acceleration ratios, even rivaling methods using pretrained models, making it a universal and efficient solution for training deep models while reducing time, financial, and environmental costs.

{{</citation>}}


### (12/112) An Optimal Transport Approach for Computing Adversarial Training Lower Bounds in Multiclass Classification (Nicolas Garcia Trillos et al., 2024)

{{<citation>}}

Nicolas Garcia Trillos, Matt Jacobs, Jakwang Kim, Matthew Werenski. (2024)  
**An Optimal Transport Approach for Computing Adversarial Training Lower Bounds in Multiclass Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC, stat-ML  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2401.09191v1)  

---


**ABSTRACT**  
Despite the success of deep learning-based algorithms, it is widely known that neural networks may fail to be robust. A popular paradigm to enforce robustness is adversarial training (AT), however, this introduces many computational and theoretical difficulties. Recent works have developed a connection between AT in the multiclass classification setting and multimarginal optimal transport (MOT), unlocking a new set of tools to study this problem. In this paper, we leverage the MOT connection to propose computationally tractable numerical algorithms for computing universal lower bounds on the optimal adversarial risk and identifying optimal classifiers. We propose two main algorithms based on linear programming (LP) and entropic regularization (Sinkhorn). Our key insight is that one can harmlessly truncate the higher order interactions between classes, preventing the combinatorial run times typically encountered in MOT problems. We validate these results with experiments on MNIST and CIFAR-$10$, which demonstrate the tractability of our approach.

{{</citation>}}


### (13/112) Beyond Anti-Forgetting: Multimodal Continual Instruction Tuning with Positive Forward Transfer (Junhao Zheng et al., 2024)

{{<citation>}}

Junhao Zheng, Qianli Ma, Zhen Liu, Binquan Wu, Huawen Feng. (2024)  
**Beyond Anti-Forgetting: Multimodal Continual Instruction Tuning with Positive Forward Transfer**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09181v1)  

---


**ABSTRACT**  
Multimodal Continual Instruction Tuning (MCIT) enables Multimodal Large Language Models (MLLMs) to meet continuously emerging requirements without expensive retraining. MCIT faces two major obstacles: catastrophic forgetting (where old knowledge is forgotten) and negative forward transfer (where the performance of future tasks is degraded). Although existing methods have greatly alleviated catastrophic forgetting, they still suffer from negative forward transfer. By performing singular value decomposition (SVD) on input embeddings, we discover a large discrepancy in different input embeddings. The discrepancy results in the model learning irrelevant information for old and pre-trained tasks, which leads to catastrophic forgetting and negative forward transfer. To address these issues, we propose Fwd-Prompt, a prompt-based method projecting prompt gradient to the residual space to minimize the interference between tasks and to the pre-trained subspace for reusing pre-trained knowledge. Our experiments demonstrate that Fwd-Prompt achieves state-of-the-art performance while updating fewer parameters and requiring no old samples. Our research sheds light on the potential of continuously adapting MLLMs to new tasks under the instruction tuning paradigm and encourages future studies to explore MCIT. The code will soon be publicly available.

{{</citation>}}


### (14/112) ADCNet: a unified framework for predicting the activity of antibody-drug conjugates (Liye Chen et al., 2024)

{{<citation>}}

Liye Chen, Biaoshun Li, Yihao Chen, Mujie Lin, Shipeng Zhang, Chenxin Li, Yu Pang, Ling Wang. (2024)  
**ADCNet: a unified framework for predicting the activity of antibody-drug conjugates**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.09176v1)  

---


**ABSTRACT**  
Antibody-drug conjugate (ADC) has revolutionized the field of cancer treatment in the era of precision medicine due to their ability to precisely target cancer cells and release highly effective drug. Nevertheless, the realization of rational design of ADC is very difficult because the relationship between their structures and activities is difficult to understand. In the present study, we introduce a unified deep learning framework called ADCNet to help design potential ADCs. The ADCNet highly integrates the protein representation learning language model ESM-2 and small-molecule representation learning language model FG-BERT models to achieve activity prediction through learning meaningful features from antigen and antibody protein sequences of ADC, SMILES strings of linker and payload, and drug-antibody ratio (DAR) value. Based on a carefully designed and manually tailored ADC data set, extensive evaluation results reveal that ADCNet performs best on the test set compared to baseline machine learning models across all evaluation metrics. For example, it achieves an average prediction accuracy of 87.12%, a balanced accuracy of 0.8689, and an area under receiver operating characteristic curve of 0.9293 on the test set. In addition, cross-validation, ablation experiments, and external independent testing results further prove the stability, advancement, and robustness of the ADCNet architecture. For the convenience of the community, we develop the first online platform (https://ADCNet.idruglab.cn) for the prediction of ADCs activity based on the optimal ADCNet model, and the source code is publicly available at https://github.com/idrugLab/ADCNet.

{{</citation>}}


### (15/112) Asynchronous Local-SGD Training for Language Modeling (Bo Liu et al., 2024)

{{<citation>}}

Bo Liu, Rachita Chhaparia, Arthur Douillard, Satyen Kale, Andrei A. Rusu, Jiajun Shen, Arthur Szlam, Marc'Aurelio Ranzato. (2024)  
**Asynchronous Local-SGD Training for Language Modeling**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09135v1)  

---


**ABSTRACT**  
Local stochastic gradient descent (Local-SGD), also referred to as federated averaging, is an approach to distributed optimization where each device performs more than one SGD update per communication. This work presents an empirical study of {\it asynchronous} Local-SGD for training language models; that is, each worker updates the global parameters as soon as it has finished its SGD steps. We conduct a comprehensive investigation by examining how worker hardware heterogeneity, model size, number of workers, and optimizer could impact the learning performance. We find that with naive implementations, asynchronous Local-SGD takes more iterations to converge than its synchronous counterpart despite updating the (global) model parameters more frequently. We identify momentum acceleration on the global parameters when worker gradients are stale as a key challenge. We propose a novel method that utilizes a delayed Nesterov momentum update and adjusts the workers' local training steps based on their computation speed. This approach, evaluated with models up to 150M parameters on the C4 dataset, matches the performance of synchronous Local-SGD in terms of perplexity per update step, and significantly surpasses it in terms of wall clock time.

{{</citation>}}


### (16/112) Understanding Heterophily for Graph Neural Networks (Junfu Wang et al., 2024)

{{<citation>}}

Junfu Wang, Yuanfang Guo, Liang Yang, Yunhong Wang. (2024)  
**Understanding Heterophily for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.09125v1)  

---


**ABSTRACT**  
Graphs with heterophily have been regarded as challenging scenarios for Graph Neural Networks (GNNs), where nodes are connected with dissimilar neighbors through various patterns. In this paper, we present theoretical understandings of the impacts of different heterophily patterns for GNNs by incorporating the graph convolution (GC) operations into fully connected networks via the proposed Heterophilous Stochastic Block Models (HSBM), a general random graph model that can accommodate diverse heterophily patterns. Firstly, we show that by applying a GC operation, the separability gains are determined by two factors, i.e., the Euclidean distance of the neighborhood distributions and $\sqrt{\mathbb{E}\left[\operatorname{deg}\right]}$, where $\mathbb{E}\left[\operatorname{deg}\right]$ is the averaged node degree. It reveals that the impact of heterophily on classification needs to be evaluated alongside the averaged node degree. Secondly, we show that the topological noise has a detrimental impact on separability, which is equivalent to degrading $\mathbb{E}\left[\operatorname{deg}\right]$. Finally, when applying multiple GC operations, we show that the separability gains are determined by the normalized distance of the $l$-powered neighborhood distributions. It indicates that the nodes still possess separability as $l$ goes to infinity in a wide range of regimes. Extensive experiments on both synthetic and real-world data verify the effectiveness of our theory.

{{</citation>}}


### (17/112) RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks (Haowen Hou et al., 2024)

{{<citation>}}

Haowen Hou, F. Richard Yu. (2024)  
**RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM, Time Series, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.09093v1)  

---


**ABSTRACT**  
Traditional Recurrent Neural Network (RNN) architectures, such as LSTM and GRU, have historically held prominence in time series tasks. However, they have recently seen a decline in their dominant position across various time series tasks. As a result, recent advancements in time series forecasting have seen a notable shift away from RNNs towards alternative architectures such as Transformers, MLPs, and CNNs. To go beyond the limitations of traditional RNNs, we design an efficient RNN-based model for time series tasks, named RWKV-TS, with three distinctive features: (i) A novel RNN architecture characterized by $O(L)$ time complexity and memory usage. (ii) An enhanced ability to capture long-term sequence information compared to traditional RNNs. (iii) High computational efficiency coupled with the capacity to scale up effectively. Through extensive experimentation, our proposed RWKV-TS model demonstrates competitive performance when compared to state-of-the-art Transformer-based or CNN-based models. Notably, RWKV-TS exhibits not only comparable performance but also demonstrates reduced latency and memory utilization. The success of RWKV-TS encourages further exploration and innovation in leveraging RNN-based approaches within the domain of Time Series. The combination of competitive performance, low latency, and efficient memory usage positions RWKV-TS as a promising avenue for future research in time series tasks. Code is available at:\href{https://github.com/howard-hou/RWKV-TS}{ https://github.com/howard-hou/RWKV-TS}

{{</citation>}}


### (18/112) Code Simulation Challenges for Large Language Models (Emanuele La Malfa et al., 2024)

{{<citation>}}

Emanuele La Malfa, Christoph Weinhuber, Orazio Torre, Fangru Lin, Anthony Cohn, Nigel Shadbolt, Michael Wooldridge. (2024)  
**Code Simulation Challenges for Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs-PL, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09074v1)  

---


**ABSTRACT**  
We investigate the extent to which Large Language Models (LLMs) can simulate the execution of computer code and algorithms. We begin by looking straight line programs, and show that current LLMs demonstrate poor performance even with such simple programs -- performance rapidly degrades with the length of code. We then investigate the ability of LLMs to simulate programs that contain critical paths and redundant instructions. We also go beyond straight line program simulation with sorting algorithms and nested loops, and we show the computational complexity of a routine directly affects the ability of an LLM to simulate its execution. We observe that LLMs execute instructions sequentially and with a low error margin only for short programs or standard procedures. LLMs' code simulation is in tension with their pattern recognition and memorisation capabilities: on tasks where memorisation is detrimental, we propose a novel prompting method to simulate code execution line by line. Empirically, our new Chain of Simulation (CoSm) method improves on the standard Chain of Thought prompting approach by avoiding the pitfalls of memorisation.

{{</citation>}}


### (19/112) Fixed-Budget Differentially Private Best Arm Identification (Zhirui Chen et al., 2024)

{{<citation>}}

Zhirui Chen, P. N. Karthik, Yeow Meng Chee, Vincent Y. F. Tan. (2024)  
**Fixed-Budget Differentially Private Best Arm Identification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IT, cs-LG, cs.LG, math-IT, math-ST, stat-ML, stat-TH  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09073v1)  

---


**ABSTRACT**  
We study best arm identification (BAI) in linear bandits in the fixed-budget regime under differential privacy constraints, when the arm rewards are supported on the unit interval. Given a finite budget $T$ and a privacy parameter $\varepsilon>0$, the goal is to minimise the error probability in finding the arm with the largest mean after $T$ sampling rounds, subject to the constraint that the policy of the decision maker satisfies a certain {\em $\varepsilon$-differential privacy} ($\varepsilon$-DP) constraint. We construct a policy satisfying the $\varepsilon$-DP constraint (called {\sc DP-BAI}) by proposing the principle of {\em maximum absolute determinants}, and derive an upper bound on its error probability. Furthermore, we derive a minimax lower bound on the error probability, and demonstrate that the lower and the upper bounds decay exponentially in $T$, with exponents in the two bounds matching order-wise in (a) the sub-optimality gaps of the arms, (b) $\varepsilon$, and (c) the problem complexity that is expressible as the sum of two terms, one characterising the complexity of standard fixed-budget BAI (without privacy constraints), and the other accounting for the $\varepsilon$-DP constraint. Additionally, we present some auxiliary results that contribute to the derivation of the lower bound on the error probability. These results, we posit, may be of independent interest and could prove instrumental in proving lower bounds on error probabilities in several other bandit problems. Whereas prior works provide results for BAI in the fixed-budget regime without privacy constraints or in the fixed-confidence regime with privacy constraints, our work fills the gap in the literature by providing the results for BAI in the fixed-budget regime under the $\varepsilon$-DP constraint.

{{</citation>}}


### (20/112) Rethinking Spectral Graph Neural Networks with Spatially Adaptive Filtering (Jingwei Guo et al., 2024)

{{<citation>}}

Jingwei Guo, Kaizhu Huang, Xinping Yi, Zixian Su, Rui Zhang. (2024)  
**Rethinking Spectral Graph Neural Networks with Spatially Adaptive Filtering**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.09071v1)  

---


**ABSTRACT**  
Whilst spectral Graph Neural Networks (GNNs) are theoretically well-founded in the spectral domain, their practical reliance on polynomial approximation implies a profound linkage to the spatial domain. As previous studies rarely examine spectral GNNs from the spatial perspective, their spatial-domain interpretability remains elusive, e.g., what information is essentially encoded by spectral GNNs in the spatial domain? In this paper, to answer this question, we establish a theoretical connection between spectral filtering and spatial aggregation, unveiling an intrinsic interaction that spectral filtering implicitly leads the original graph to an adapted new graph, explicitly computed for spatial aggregation. Both theoretical and empirical investigations reveal that the adapted new graph not only exhibits non-locality but also accommodates signed edge weights to reflect label consistency between nodes. These findings thus highlight the interpretable role of spectral GNNs in the spatial domain and inspire us to rethink graph spectral filters beyond the fixed-order polynomials, which neglect global information. Built upon the theoretical findings, we revisit the state-of-the-art spectral GNNs and propose a novel Spatially Adaptive Filtering (SAF) framework, which leverages the adapted new graph by spectral filtering for an auxiliary non-local aggregation. Notably, our proposed SAF comprehensively models both node similarity and dissimilarity from a global perspective, therefore alleviating persistent deficiencies of GNNs related to long-range dependencies and graph heterophily. Extensive experiments over 13 node classification benchmarks demonstrate the superiority of our proposed framework to the state-of-the-art models.

{{</citation>}}


### (21/112) DTMM: Deploying TinyML Models on Extremely Weak IoT Devices with Pruning (Lixiang Han et al., 2024)

{{<citation>}}

Lixiang Han, Zhen Xiao, Zhenjiang Li. (2024)  
**DTMM: Deploying TinyML Models on Extremely Weak IoT Devices with Pruning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2401.09068v1)  

---


**ABSTRACT**  
DTMM is a library designed for efficient deployment and execution of machine learning models on weak IoT devices such as microcontroller units (MCUs). The motivation for designing DTMM comes from the emerging field of tiny machine learning (TinyML), which explores extending the reach of machine learning to many low-end IoT devices to achieve ubiquitous intelligence. Due to the weak capability of embedded devices, it is necessary to compress models by pruning enough weights before deploying. Although pruning has been studied extensively on many computing platforms, two key issues with pruning methods are exacerbated on MCUs: models need to be deeply compressed without significantly compromising accuracy, and they should perform efficiently after pruning. Current solutions only achieve one of these objectives, but not both. In this paper, we find that pruned models have great potential for efficient deployment and execution on MCUs. Therefore, we propose DTMM with pruning unit selection, pre-execution pruning optimizations, runtime acceleration, and post-execution low-cost storage to fill the gap for efficient deployment and execution of pruned models. It can be integrated into commercial ML frameworks for practical deployment, and a prototype system has been developed. Extensive experiments on various models show promising gains compared to state-of-the-art methods.

{{</citation>}}


### (22/112) Towards Continual Learning Desiderata via HSIC-Bottleneck Orthogonalization and Equiangular Embedding (Depeng Li et al., 2024)

{{<citation>}}

Depeng Li, Tianqi Wang, Junwei Chen, Qining Ren, Kenji Kawaguchi, Zhigang Zeng. (2024)  
**Towards Continual Learning Desiderata via HSIC-Bottleneck Orthogonalization and Equiangular Embedding**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.09067v1)  

---


**ABSTRACT**  
Deep neural networks are susceptible to catastrophic forgetting when trained on sequential tasks. Various continual learning (CL) methods often rely on exemplar buffers or/and network expansion for balancing model stability and plasticity, which, however, compromises their practical value due to privacy and memory concerns. Instead, this paper considers a strict yet realistic setting, where the training data from previous tasks is unavailable and the model size remains relatively constant during sequential training. To achieve such desiderata, we propose a conceptually simple yet effective method that attributes forgetting to layer-wise parameter overwriting and the resulting decision boundary distortion. This is achieved by the synergy between two key components: HSIC-Bottleneck Orthogonalization (HBO) implements non-overwritten parameter updates mediated by Hilbert-Schmidt independence criterion in an orthogonal space and EquiAngular Embedding (EAE) enhances decision boundary adaptation between old and new tasks with predefined basis vectors. Extensive experiments demonstrate that our method achieves competitive accuracy performance, even with absolute superiority of zero exemplar buffer and 1.02x the base model.

{{</citation>}}


### (23/112) Functional Autoencoder for Smoothing and Representation Learning (Sidi Wu et al., 2024)

{{<citation>}}

Sidi Wu, Cédric Beaulac, Jiguo Cao. (2024)  
**Functional Autoencoder for Smoothing and Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.09499v1)  

---


**ABSTRACT**  
A common pipeline in functional data analysis is to first convert the discretely observed data to smooth functions, and then represent the functions by a finite-dimensional vector of coefficients summarizing the information. Existing methods for data smoothing and dimensional reduction mainly focus on learning the linear mappings from the data space to the representation space, however, learning only the linear representations may not be sufficient. In this study, we propose to learn the nonlinear representations of functional data using neural network autoencoders designed to process data in the form it is usually collected without the need of preprocessing. We design the encoder to employ a projection layer computing the weighted inner product of the functional data and functional weights over the observed timestamp, and the decoder to apply a recovery layer that maps the finite-dimensional vector extracted from the functional data back to functional space using a set of predetermined basis functions. The developed architecture can accommodate both regularly and irregularly spaced data. Our experiments demonstrate that the proposed method outperforms functional principal component analysis in terms of prediction and classification, and maintains superior smoothing ability and better computational efficiency in comparison to the conventional autoencoders under both linear and nonlinear settings.

{{</citation>}}


### (24/112) Data Attribution for Diffusion Models: Timestep-induced Bias in Influence Estimation (Tong Xie et al., 2024)

{{<citation>}}

Tong Xie, Haoyu Li, Andrew Bai, Cho-Jui Hsieh. (2024)  
**Data Attribution for Diffusion Models: Timestep-induced Bias in Influence Estimation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.09031v1)  

---


**ABSTRACT**  
Data attribution methods trace model behavior back to its training dataset, offering an effective approach to better understand ``black-box'' neural networks. While prior research has established quantifiable links between model output and training data in diverse settings, interpreting diffusion model outputs in relation to training samples remains underexplored. In particular, diffusion models operate over a sequence of timesteps instead of instantaneous input-output relationships in previous contexts, posing a significant challenge to extend existing frameworks to diffusion models directly. Notably, we present Diffusion-TracIn that incorporates this temporal dynamics and observe that samples' loss gradient norms are highly dependent on timestep. This trend leads to a prominent bias in influence estimation, and is particularly noticeable for samples trained on large-norm-inducing timesteps, causing them to be generally influential. To mitigate this effect, we introduce Diffusion-ReTrac as a re-normalized adaptation that enables the retrieval of training samples more targeted to the test sample of interest, facilitating a localized measurement of influence and considerably more intuitive visualization. We demonstrate the efficacy of our approach through various evaluation metrics and auxiliary tasks, reducing the amount of generally influential samples to $\frac{1}{3}$ of its original quantity.

{{</citation>}}


### (25/112) A novel hybrid time-varying graph neural network for traffic flow forecasting (Ben Ao Dai et al., 2024)

{{<citation>}}

Ben Ao Dai, Bao-Lin Ye. (2024)  
**A novel hybrid time-varying graph neural network for traffic flow forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.10155v1)  

---


**ABSTRACT**  
Real-time and accurate traffic flow prediction is the foundation for ensuring the efficient operation of intelligent transportation systems.In existing traffic flow prediction methods based on graph neural networks (GNNs), pre-defined graphs were usually used to describe the spatial correlations of different traffic nodes in urban road networks. However, the ability of pre-defined graphs used to describe spatial correlation was limited by prior knowledge and graph generation methods. Although time-varying graphs based on data-driven learning can partially overcome the drawbacks of pre-defined graphs, the learning ability of existing adaptive graphs was limited. For example, time-varying graphs cannot adequately capture the inherent spatial correlations in traffic flow data.In order to solve these problems, we have proposed a hybrid time-varying graph neural network (HTVGNN) for traffic flow prediction.

{{</citation>}}


### (26/112) Inductive Models for Artificial Intelligence Systems are Insufficient without Good Explanations (Udesh Habaraduwa, 2024)

{{<citation>}}

Udesh Habaraduwa. (2024)  
**Inductive Models for Artificial Intelligence Systems are Insufficient without Good Explanations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09011v1)  

---


**ABSTRACT**  
This paper discusses the limitations of machine learning (ML), particularly deep artificial neural networks (ANNs), which are effective at approximating complex functions but often lack transparency and explanatory power. It highlights the `problem of induction' : the philosophical issue that past observations may not necessarily predict future events, a challenge that ML models face when encountering new, unseen data. The paper argues for the importance of not just making predictions but also providing good explanations, a feature that current models often fail to deliver. It suggests that for AI to progress, we must seek models that offer insights and explanations, not just predictions.

{{</citation>}}


### (27/112) MicroNAS: Zero-Shot Neural Architecture Search for MCUs (Ye Qiao et al., 2024)

{{<citation>}}

Ye Qiao, Haocheng Xu, Yifan Zhang, Sitao Huang. (2024)  
**MicroNAS: Zero-Shot Neural Architecture Search for MCUs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.08996v1)  

---


**ABSTRACT**  
Neural Architecture Search (NAS) effectively discovers new Convolutional Neural Network (CNN) architectures, particularly for accuracy optimization. However, prior approaches often require resource-intensive training on super networks or extensive architecture evaluations, limiting practical applications. To address these challenges, we propose MicroNAS, a hardware-aware zero-shot NAS framework designed for microcontroller units (MCUs) in edge computing. MicroNAS considers target hardware optimality during the search, utilizing specialized performance indicators to identify optimal neural architectures without high computational costs. Compared to previous works, MicroNAS achieves up to 1104x improvement in search efficiency and discovers models with over 3.23x faster MCU inference while maintaining similar accuracy

{{</citation>}}


### (28/112) FedLoGe: Joint Local and Generic Federated Learning under Long-tailed Data (Zikai Xiao et al., 2024)

{{<citation>}}

Zikai Xiao, Zihan Chen, Liyinglan Liu, Yang Feng, Jian Wu, Wanlu Liu, Joey Tianyi Zhou, Howard Hao Yang, Zuozhu Liu. (2024)  
**FedLoGe: Joint Local and Generic Federated Learning under Long-tailed Data**  

---
Primary Category: cs.LG  
Categories: I-2-0, cs-AI, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.08977v1)  

---


**ABSTRACT**  
Federated Long-Tailed Learning (Fed-LT), a paradigm wherein data collected from decentralized local clients manifests a globally prevalent long-tailed distribution, has garnered considerable attention in recent times. In the context of Fed-LT, existing works have predominantly centered on addressing the data imbalance issue to enhance the efficacy of the generic global model while neglecting the performance at the local level. In contrast, conventional Personalized Federated Learning (pFL) techniques are primarily devised to optimize personalized local models under the presumption of a balanced global data distribution. This paper introduces an approach termed Federated Local and Generic Model Training in Fed-LT (FedLoGe), which enhances both local and generic model performance through the integration of representation learning and classifier alignment within a neural collapse framework. Our investigation reveals the feasibility of employing a shared backbone as a foundational framework for capturing overarching global trends, while concurrently employing individualized classifiers to encapsulate distinct refinements stemming from each client's local features. Building upon this discovery, we establish the Static Sparse Equiangular Tight Frame Classifier (SSE-C), inspired by neural collapse principles that naturally prune extraneous noisy features and foster the acquisition of potent data representations. Furthermore, leveraging insights from imbalance neural collapse's classifier norm patterns, we develop Global and Local Adaptive Feature Realignment (GLA-FR) via an auxiliary global classifier and personalized Euclidean norm transfer to align global features with client preferences. Extensive experimental results on CIFAR-10/100-LT, ImageNet, and iNaturalist demonstrate the advantage of our method over state-of-the-art pFL and Fed-LT approaches.

{{</citation>}}


### (29/112) ACT-GAN: Radio map construction based on generative adversarial networks with ACT blocks (Chen Qi et al., 2024)

{{<citation>}}

Chen Qi, Yang Jingjing, Huang Ming, Zhou Qiang. (2024)  
**ACT-GAN: Radio map construction based on generative adversarial networks with ACT blocks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.08976v1)  

---


**ABSTRACT**  
The radio map, serving as a visual representation of electromagnetic spatial characteristics, plays a pivotal role in assessment of wireless communication networks and radio monitoring coverage. Addressing the issue of low accuracy existing in the current radio map construction, this paper presents a novel radio map construction method based on generative adversarial network (GAN) in which the Aggregated Contextual-Transformation (AOT) block, Convolutional Block Attention Module (CBAM), and Transposed Convolution (T-Conv) block are applied to the generator, and we name it as ACT-GAN. It significantly improves the reconstruction accuracy and local texture of the radio maps. The performance of ACT-GAN across three different scenarios is demonstrated. Experiment results reveal that in the scenario without sparse discrete observations, the proposed method reduces the root mean square error (RMSE) by 14.6% in comparison to the state-of-the-art models. In the scenario with sparse discrete observations, the RMSE is diminished by 13.2%. Furthermore, the predictive results of the proposed model show a more lucid representation of electromagnetic spatial field distribution. To verify the universality of this model in radio map construction tasks, the scenario of unknown radio emission source is investigated. The results indicate that the proposed model is robust radio map construction and accurate in predicting the location of the emission source.

{{</citation>}}


### (30/112) Cascading Reinforcement Learning (Yihan Du et al., 2024)

{{<citation>}}

Yihan Du, R. Srikant, Wei Chen. (2024)  
**Cascading Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.08961v1)  

---


**ABSTRACT**  
Cascading bandits have gained popularity in recent years due to their applicability to recommendation systems and online advertising. In the cascading bandit model, at each timestep, an agent recommends an ordered subset of items (called an item list) from a pool of items, each associated with an unknown attraction probability. Then, the user examines the list, and clicks the first attractive item (if any), and after that, the agent receives a reward. The goal of the agent is to maximize the expected cumulative reward. However, the prior literature on cascading bandits ignores the influences of user states (e.g., historical behaviors) on recommendations and the change of states as the session proceeds. Motivated by this fact, we propose a generalized cascading RL framework, which considers the impact of user states and state transition into decisions. In cascading RL, we need to select items not only with large attraction probabilities but also leading to good successor states. This imposes a huge computational challenge due to the combinatorial action space. To tackle this challenge, we delve into the properties of value functions, and design an oracle BestPerm to efficiently find the optimal item list. Equipped with BestPerm, we develop two algorithms CascadingVI and CascadingBPI, which are both computationally-efficient and sample-efficient, and provide near-optimal regret and sample complexity guarantees. Furthermore, we present experiments to show the improved computational and sample efficiencies of our algorithms compared to straightforward adaptations of existing RL algorithms in practice.

{{</citation>}}


### (31/112) Towards Off-Policy Reinforcement Learning for Ranking Policies with Human Feedback (Teng Xiao et al., 2024)

{{<citation>}}

Teng Xiao, Suhang Wang. (2024)  
**Towards Off-Policy Reinforcement Learning for Ranking Policies with Human Feedback**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.08959v1)  

---


**ABSTRACT**  
Probabilistic learning to rank (LTR) has been the dominating approach for optimizing the ranking metric, but cannot maximize long-term rewards. Reinforcement learning models have been proposed to maximize user long-term rewards by formulating the recommendation as a sequential decision-making problem, but could only achieve inferior accuracy compared to LTR counterparts, primarily due to the lack of online interactions and the characteristics of ranking. In this paper, we propose a new off-policy value ranking (VR) algorithm that can simultaneously maximize user long-term rewards and optimize the ranking metric offline for improved sample efficiency in a unified Expectation-Maximization (EM) framework. We theoretically and empirically show that the EM process guides the leaned policy to enjoy the benefit of integration of the future reward and ranking metric, and learn without any online interactions. Extensive offline and online experiments demonstrate the effectiveness of our methods.

{{</citation>}}


### (32/112) CEL: A Continual Learning Model for Disease Outbreak Prediction by Leveraging Domain Adaptation via Elastic Weight Consolidation (Saba Aslam et al., 2024)

{{<citation>}}

Saba Aslam, Abdur Rasool, Hongyan Wu, Xiaoli Li. (2024)  
**CEL: A Continual Learning Model for Disease Outbreak Prediction by Leveraging Domain Adaptation via Elastic Weight Consolidation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.08940v1)  

---


**ABSTRACT**  
Continual learning, the ability of a model to learn over time without forgetting previous knowledge and, therefore, be adaptive to new data, is paramount in dynamic fields such as disease outbreak prediction. Deep neural networks, i.e., LSTM, are prone to error due to catastrophic forgetting. This study introduces a novel CEL model for continual learning by leveraging domain adaptation via Elastic Weight Consolidation (EWC). This model aims to mitigate the catastrophic forgetting phenomenon in a domain incremental setting. The Fisher Information Matrix (FIM) is constructed with EWC to develop a regularization term that penalizes changes to important parameters, namely, the important previous knowledge. CEL's performance is evaluated on three distinct diseases, Influenza, Mpox, and Measles, with different metrics. The high R-squared values during evaluation and reevaluation outperform the other state-of-the-art models in several contexts, indicating that CEL adapts to incremental data well. CEL's robustness and reliability are underscored by its minimal 65% forgetting rate and 18% higher memory stability compared to existing benchmark studies. This study highlights CEL's versatility in disease outbreak prediction, addressing evolving data with temporal patterns. It offers a valuable model for proactive disease control with accurate, timely predictions.

{{</citation>}}


## eess.IV (4)



### (33/112) Uncertainty Modeling in Ultrasound Image Segmentation for Precise Fetal Biometric Measurements (Shuge Lei, 2024)

{{<citation>}}

Shuge Lei. (2024)  
**Uncertainty Modeling in Ultrasound Image Segmentation for Precise Fetal Biometric Measurements**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.09639v1)  

---


**ABSTRACT**  
Medical image segmentation, particularly in the context of ultrasound data, is a crucial aspect of computer vision and medical imaging. This paper delves into the complexities of uncertainty in the segmentation process, focusing on fetal head and femur ultrasound images. The proposed methodology involves extracting target contours and exploring techniques for precise parameter measurement. Uncertainty modeling methods are employed to enhance the training and testing processes of the segmentation network. The study reveals that the average absolute error in fetal head circumference measurement is 8.0833mm, with a relative error of 4.7347%. Similarly, the average absolute error in fetal femur measurement is 2.6163mm, with a relative error of 6.3336%. Uncertainty modeling experiments employing Test-Time Augmentation (TTA) demonstrate effective interpretability of data uncertainty on both datasets. This suggests that incorporating data uncertainty based on the TTA method can support clinical practitioners in making informed decisions and obtaining more reliable measurement results in practical clinical applications. The paper contributes to the advancement of ultrasound image segmentation, addressing critical challenges and improving the reliability of biometric measurements.

{{</citation>}}


### (34/112) CT Liver Segmentation via PVT-based Encoding and Refined Decoding (Debesh Jha et al., 2024)

{{<citation>}}

Debesh Jha, Nikhil Kumar Tomar, Koushik Biswas, Gorkem Durak, Alpay Medetalibeyoglu, Matthew Antalek, Yury Velichko, Daniela Ladner, Amir Borhani, Ulas Bagci. (2024)  
**CT Liver Segmentation via PVT-based Encoding and Refined Decoding**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.09630v1)  

---


**ABSTRACT**  
Accurate liver segmentation from CT scans is essential for computer-aided diagnosis and treatment planning. Recently, Vision Transformers achieved a competitive performance in computer vision tasks compared to convolutional neural networks due to their exceptional ability to learn global representations. However, they often struggle with scalability, memory constraints, and computational inefficiency, particularly in handling high-resolution medical images. To overcome scalability and efficiency issues, we propose a novel deep learning approach, \textit{\textbf{PVTFormer}}, that is built upon a pretrained pyramid vision transformer (PVT v2) combined with advanced residual upsampling and decoder block. By integrating a refined feature channel approach with hierarchical decoding strategy, PVTFormer generates high quality segmentation masks by enhancing semantic features. Rigorous evaluation of the proposed method on Liver Tumor Segmentation Benchmark (LiTS) 2017 demonstrates that our proposed architecture not only achieves a high dice coefficient of 86.78\%, mIoU of 78.46\%, but also obtains a low HD of 3.50. The results underscore PVTFormer's efficacy in setting a new benchmark for state-of-the-art liver segmentation methods. The source code of the proposed PVTFormer is available at \url{https://github.com/DebeshJha/PVTFormer}.

{{</citation>}}


### (35/112) SymTC: A Symbiotic Transformer-CNN Net for Instance Segmentation of Lumbar Spine MRI (Jiasong Chen et al., 2024)

{{<citation>}}

Jiasong Chen, Linchen Qian, Linhai Ma, Timur Urakov, Weiyong Gu, Liang Liang. (2024)  
**SymTC: A Symbiotic Transformer-CNN Net for Instance Segmentation of Lumbar Spine MRI**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.09627v1)  

---


**ABSTRACT**  
Intervertebral disc disease, a prevalent ailment, frequently leads to intermittent or persistent low back pain, and diagnosing and assessing of this disease rely on accurate measurement of vertebral bone and intervertebral disc geometries from lumbar MR images. Deep neural network (DNN) models may assist clinicians with more efficient image segmentation of individual instances (disks and vertebrae) of the lumbar spine in an automated way, which is termed as instance image segmentation. In this work, we proposed SymTC, an innovative lumbar spine MR image segmentation model that combines the strengths of Transformer and Convolutional Neural Network (CNN). Specifically, we designed a parallel dual-path architecture to merge CNN layers and Transformer layers, and we integrated a novel position embedding into the self-attention module of Transformer, enhancing the utilization of positional information for more accurate segmentation. To further improves model performance, we introduced a new data augmentation technique to create synthetic yet realistic MR image dataset, named SSMSpine, which is made publicly available. We evaluated our SymTC and the other 15 existing image segmentation models on our private in-house dataset and the public SSMSpine dataset, using two metrics, Dice Similarity Coefficient and 95% Hausdorff Distance. The results show that our SymTC has the best performance for segmenting vertebral bones and intervertebral discs in lumbar spine MR images. The SymTC code and SSMSpine dataset are available at https://github.com/jiasongchen/SymTC.

{{</citation>}}


### (36/112) To deform or not: treatment-aware longitudinal registration for breast DCE-MRI during neoadjuvant chemotherapy via unsupervised keypoints detection (Luyi Han et al., 2024)

{{<citation>}}

Luyi Han, Tao Tan, Tianyu Zhang, Yuan Gao, Xin Wang, Valentina Longo, Sofía Ventura-Díaz, Anna D'Angelo, Jonas Teuwen, Ritse Mann. (2024)  
**To deform or not: treatment-aware longitudinal registration for breast DCE-MRI during neoadjuvant chemotherapy via unsupervised keypoints detection**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2401.09336v1)  

---


**ABSTRACT**  
Clinicians compare breast DCE-MRI after neoadjuvant chemotherapy (NAC) with pre-treatment scans to evaluate the response to NAC. Clinical evidence supports that accurate longitudinal deformable registration without deforming treated tumor regions is key to quantifying tumor changes. We propose a conditional pyramid registration network based on unsupervised keypoint detection and selective volume-preserving to quantify changes over time. In this approach, we extract the structural and the abnormal keypoints from DCE-MRI, apply the structural keypoints for the registration algorithm to restrict large deformation, and employ volume-preserving loss based on abnormal keypoints to keep the volume of the tumor unchanged after registration. We use a clinical dataset with 1630 MRI scans from 314 patients treated with NAC. The results demonstrate that our method registers with better performance and better volume preservation of the tumors. Furthermore, a local-global-combining biomarker based on the proposed method achieves high accuracy in pathological complete response (pCR) prediction, indicating that predictive information exists outside tumor regions. The biomarkers could potentially be used to avoid unnecessary surgeries for certain patients. It may be valuable for clinicians and/or computer systems to conduct follow-up tumor segmentation and response prediction on images registered by our method. Our code is available on \url{https://github.com/fiy2W/Treatment-aware-Longitudinal-Registration}.

{{</citation>}}


## cs.HC (3)



### (37/112) Impact of Large Language Model Assistance on Patients Reading Clinical Notes: A Mixed-Methods Study (Niklas Mannhardt et al., 2024)

{{<citation>}}

Niklas Mannhardt, Elizabeth Bondi-Kelly, Barbara Lam, Chloe O'Connell, Mercy Asiedu, Hussein Mozannar, Monica Agrawal, Alejandro Buendia, Tatiana Urman, Irbaz B. Riaz, Catherine E. Ricciardi, Marzyeh Ghassemi, David Sontag. (2024)  
**Impact of Large Language Model Assistance on Patients Reading Clinical Notes: A Mixed-Methods Study**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: Augmentation, Clinical, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09637v1)  

---


**ABSTRACT**  
Patients derive numerous benefits from reading their clinical notes, including an increased sense of control over their health and improved understanding of their care plan. However, complex medical concepts and jargon within clinical notes hinder patient comprehension and may lead to anxiety. We developed a patient-facing tool to make clinical notes more readable, leveraging large language models (LLMs) to simplify, extract information from, and add context to notes. We prompt engineered GPT-4 to perform these augmentation tasks on real clinical notes donated by breast cancer survivors and synthetic notes generated by a clinician, a total of 12 notes with 3868 words. In June 2023, 200 female-identifying US-based participants were randomly assigned three clinical notes with varying levels of augmentations using our tool. Participants answered questions about each note, evaluating their understanding of follow-up actions and self-reported confidence. We found that augmentations were associated with a significant increase in action understanding score (0.63 $\pm$ 0.04 for select augmentations, compared to 0.54 $\pm$ 0.02 for the control) with p=0.002. In-depth interviews of self-identifying breast cancer patients (N=7) were also conducted via video conferencing. Augmentations, especially definitions, elicited positive responses among the seven participants, with some concerns about relying on LLMs. Augmentations were evaluated for errors by clinicians, and we found misleading errors occur, with errors more common in real donated notes than synthetic notes, illustrating the importance of carefully written clinical notes. Augmentations improve some but not all readability metrics. This work demonstrates the potential of LLMs to improve patients' experience with clinical notes at a lower burden to clinicians. However, having a human in the loop is important to correct potential model errors.

{{</citation>}}


### (38/112) Evidence-centered Assessment for Writing with Generative AI (Yixin Cheng et al., 2024)

{{<citation>}}

Yixin Cheng, Kayley Lyons, Guanliang Chen, Dragan Gasevic, Zachari Swiecki. (2024)  
**Evidence-centered Assessment for Writing with Generative AI**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.08964v1)  

---


**ABSTRACT**  
We propose a learning analytics-based methodology for assessing the collaborative writing of humans and generative artificial intelligence. Framed by the evidence-centered design, we used elements of knowledge-telling, knowledge transformation, and cognitive presence to identify assessment claims; we used data collected from the CoAuthor writing tool as potential evidence for these claims; and we used epistemic network analysis to make inferences from the data about the claims. Our findings revealed significant differences in the writing processes of different groups of CoAuthor users, suggesting that our method is a plausible approach to assessing human-AI collaborative writing.

{{</citation>}}


### (39/112) From User Surveys to Telemetry-Driven Agents: Exploring the Potential of Personalized Productivity Solutions (Subigya Nepal et al., 2024)

{{<citation>}}

Subigya Nepal, Javier Hernandez, Talie Massachi, Kael Rowan, Judith Amores, Jina Suh, Gonzalo Ramos, Brian Houck, Shamsi T. Iqbal, Mary Czerwinski. (2024)  
**From User Surveys to Telemetry-Driven Agents: Exploring the Potential of Personalized Productivity Solutions**  

---
Primary Category: cs.HC  
Categories: H-5-0; H-5-3; H-5-m; J-0, cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.08960v1)  

---


**ABSTRACT**  
We present a comprehensive, user-centric approach to understand preferences in AI-based productivity agents and develop personalized solutions tailored to users' needs. Utilizing a two-phase method, we first conducted a survey with 363 participants, exploring various aspects of productivity, communication style, agent approach, personality traits, personalization, and privacy. Drawing on the survey insights, we developed a GPT-4 powered personalized productivity agent that utilizes telemetry data gathered via Viva Insights from information workers to provide tailored assistance. We compared its performance with alternative productivity-assistive tools, such as dashboard and narrative, in a study involving 40 participants. Our findings highlight the importance of user-centric design, adaptability, and the balance between personalization and privacy in AI-assisted productivity tools. By building on the insights distilled from our study, we believe that our work can enable and guide future research to further enhance productivity solutions, ultimately leading to optimized efficiency and user experiences for information workers.

{{</citation>}}


## cs.DB (1)



### (40/112) XTable in Action: Seamless Interoperability in Data Lakes (Ashvin Agrawal et al., 2024)

{{<citation>}}

Ashvin Agrawal, Tim Brown, Anoop Johnson, Jesús Camacho-Rodríguez, Kyle Weller, Carlo Curino, Raghu Ramakrishnan. (2024)  
**XTable in Action: Seamless Interoperability in Data Lakes**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09621v1)  

---


**ABSTRACT**  
Contemporary approaches to data management are increasingly relying on unified analytics and AI platforms to foster collaboration, interoperability, seamless access to reliable data, and high performance. Data Lakes featuring open standard table formats such as Delta Lake, Apache Hudi, and Apache Iceberg are central components of these data architectures. Choosing the right format for managing a table is crucial for achieving the objectives mentioned above. The challenge lies in selecting the best format, a task that is onerous and can yield temporary results, as the ideal choice may shift over time with data growth, evolving workloads, and the competitive development of table formats and processing engines. Moreover, restricting data access to a single format can hinder data sharing resulting in diminished business value over the long term. The ability to seamlessly interoperate between formats and with negligible overhead can effectively address these challenges. Our solution in this direction is an innovative omni-directional translator, XTable, that facilitates writing data in one format and reading it in any format, thus achieving the desired format interoperability. In this work, we demonstrate the effectiveness of XTable through application scenarios inspired by real-world use cases.

{{</citation>}}


## cs.CL (17)



### (41/112) Learning Shortcuts: On the Misleading Promise of NLU in Language Models (Geetanjali Bihani et al., 2024)

{{<citation>}}

Geetanjali Bihani, Julia Taylor Rayz. (2024)  
**Learning Shortcuts: On the Misleading Promise of NLU in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLU  
[Paper Link](http://arxiv.org/abs/2401.09615v1)  

---


**ABSTRACT**  
The advent of large language models (LLMs) has enabled significant performance gains in the field of natural language processing. However, recent studies have found that LLMs often resort to shortcuts when performing tasks, creating an illusion of enhanced performance while lacking generalizability in their decision rules. This phenomenon introduces challenges in accurately assessing natural language understanding in LLMs. Our paper provides a concise survey of relevant research in this area and puts forth a perspective on the implications of shortcut learning in the evaluation of language models, specifically for NLU tasks. This paper urges more research efforts to be put towards deepening our comprehension of shortcut learning, contributing to the development of more robust language models, and raising the standards of NLU evaluation in real-world scenarios.

{{</citation>}}


### (42/112) Aligning Large Language Models with Counterfactual DPO (Bradley Butcher, 2024)

{{<citation>}}

Bradley Butcher. (2024)  
**Aligning Large Language Models with Counterfactual DPO**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09566v1)  

---


**ABSTRACT**  
Advancements in large language models (LLMs) have demonstrated remarkable capabilities across a diverse range of applications. These models excel in generating text completions that are contextually coherent and cover an extensive array of subjects. However, the vast datasets required for their training make aligning response styles during the pretraining and instruction tuning phases challenging. Consequently, an additional alignment phase is typically employed, wherein the model is further trained with human preference data to better align its outputs with human expectations. While this process doesn't introduce new capabilities per se, it does accentuate generation styles innate to the model. This paper explores the utilization of counterfactual prompting within the framework of Direct Preference Optimization (DPO) to align the model's style without relying on human intervention. We demonstrate that this method effectively instils desirable behaviour, mitigates undesirable ones, and encourages the model to disregard inappropriate instructions. Our findings suggest that counterfactual prompting with DPO presents a low-resource way to fine-tune LLMs to meet the demands for responsible and ethically aligned AI systems.

{{</citation>}}


### (43/112) BERTologyNavigator: Advanced Question Answering with BERT-based Semantics (Shreya Rajpal et al., 2024)

{{<citation>}}

Shreya Rajpal, Ricardo Usbeck. (2024)  
**BERTologyNavigator: Advanced Question Answering with BERT-based Semantics**  

---
Primary Category: cs.CL  
Categories: I-2-4; I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: BERT, BERTology, Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.09553v1)  

---


**ABSTRACT**  
The development and integration of knowledge graphs and language models has significance in artificial intelligence and natural language processing. In this study, we introduce the BERTologyNavigator -- a two-phased system that combines relation extraction techniques and BERT embeddings to navigate the relationships within the DBLP Knowledge Graph (KG). Our approach focuses on extracting one-hop relations and labelled candidate pairs in the first phases. This is followed by employing BERT's CLS embeddings and additional heuristics for relation selection in the second phase. Our system reaches an F1 score of 0.2175 on the DBLP QuAD Final test dataset for Scholarly QALD and 0.98 F1 score on the subset of the DBLP QuAD test dataset during the QA phase.

{{</citation>}}


### (44/112) Deciphering Textual Authenticity: A Generalized Strategy through the Lens of Large Language Semantics for Detecting Human vs. Machine-Generated Text (Mazal Bethany et al., 2024)

{{<citation>}}

Mazal Bethany, Brandon Wherry, Emet Bethany, Nishant Vishwamitra, Peyman Najafirad. (2024)  
**Deciphering Textual Authenticity: A Generalized Strategy through the Lens of Large Language Semantics for Detecting Human vs. Machine-Generated Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2401.09407v1)  

---


**ABSTRACT**  
With the recent proliferation of Large Language Models (LLMs), there has been an increasing demand for tools to detect machine-generated text. The effective detection of machine-generated text face two pertinent problems: First, they are severely limited in generalizing against real-world scenarios, where machine-generated text is produced by a variety of generators, including but not limited to GPT-4 and Dolly, and spans diverse domains, ranging from academic manuscripts to social media posts. Second, existing detection methodologies treat texts produced by LLMs through a restrictive binary classification lens, neglecting the nuanced diversity of artifacts generated by different LLMs. In this work, we undertake a systematic study on the detection of machine-generated text in real-world scenarios. We first study the effectiveness of state-of-the-art approaches and find that they are severely limited against text produced by diverse generators and domains in the real world. Furthermore, t-SNE visualizations of the embeddings from a pretrained LLM's encoder show that they cannot reliably distinguish between human and machine-generated text. Based on our findings, we introduce a novel system, T5LLMCipher, for detecting machine-generated text using a pretrained T5 encoder combined with LLM embedding sub-clustering to address the text produced by diverse generators and domains in the real world. We evaluate our approach across 9 machine-generated text systems and 9 domains and find that our approach provides state-of-the-art generalization ability, with an average increase in F1 score on machine-generated text of 19.6\% on unseen generators and domains compared to the top performing existing approaches and correctly attributes the generator of text with an accuracy of 93.6\%.

{{</citation>}}


### (45/112) Stuck in the Quicksand of Numeracy, Far from AGI Summit: Evaluating LLMs' Mathematical Competency through Ontology-guided Perturbations (Pengfei Hong et al., 2024)

{{<citation>}}

Pengfei Hong, Deepanway Ghosal, Navonil Majumder, Somak Aditya, Rada Mihalcea, Soujanya Poria. (2024)  
**Stuck in the Quicksand of Numeracy, Far from AGI Summit: Evaluating LLMs' Mathematical Competency through Ontology-guided Perturbations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09395v1)  

---


**ABSTRACT**  
Recent advancements in Large Language Models (LLMs) have showcased striking results on existing logical reasoning benchmarks, with some models even surpassing human performance. However, the true depth of their competencies and robustness, in mathematical reasoning tasks, remains an open question. In response, we develop (i) an ontology of perturbations of maths questions, (ii) a semi-automatic method of perturbation, and (iii) a dataset of perturbed maths questions to probe the limits of LLM capabilities in mathematical reasoning tasks. These controlled perturbations span across multiple fine dimensions of the structural and representational aspects of maths questions. Using GPT-4, we generated the MORE dataset by perturbing randomly selected five seed questions from GSM8K. This process was guided by our ontology and involved a thorough automatic and manual filtering process, yielding a set of 216 maths problems. We conducted comprehensive evaluation of both closed-source and open-source LLMs on MORE. The results show a significant performance drop across all the models against the perturbed questions. This strongly suggests that current LLMs lack robust mathematical skills and deep reasoning abilities. This research not only identifies multiple gaps in the capabilities of current models, but also highlights multiple potential directions for future development. Our dataset will be made publicly available at https://huggingface.co/datasets/declare-lab/GSM8k_MORE.

{{</citation>}}


### (46/112) Efficient slot labelling (Vladimir Vlasov, 2024)

{{<citation>}}

Vladimir Vlasov. (2024)  
**Efficient slot labelling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.09343v1)  

---


**ABSTRACT**  
Slot labelling is an essential component of any dialogue system, aiming to find important arguments in every user turn. Common approaches involve large pre-trained language models (PLMs) like BERT or RoBERTa, but they face challenges such as high computational requirements and dependence on pre-training data. In this work, we propose a lightweight method which performs on par or better than the state-of-the-art PLM-based methods, while having almost 10x less trainable parameters. This makes it especially applicable for real-life industry scenarios.

{{</citation>}}


### (47/112) Large Language Models Are Neurosymbolic Reasoners (Meng Fang et al., 2024)

{{<citation>}}

Meng Fang, Shilong Deng, Yudi Zhang, Zijing Shi, Ling Chen, Mykola Pechenizkiy, Jun Wang. (2024)  
**Large Language Models Are Neurosymbolic Reasoners**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09334v1)  

---


**ABSTRACT**  
A wide range of real-world applications is characterized by their symbolic nature, necessitating a strong capability for symbolic reasoning. This paper investigates the potential application of Large Language Models (LLMs) as symbolic reasoners. We focus on text-based games, significant benchmarks for agents with natural language capabilities, particularly in symbolic tasks like math, map reading, sorting, and applying common sense in text-based worlds. To facilitate these agents, we propose an LLM agent designed to tackle symbolic challenges and achieve in-game objectives. We begin by initializing the LLM agent and informing it of its role. The agent then receives observations and a set of valid actions from the text-based games, along with a specific symbolic module. With these inputs, the LLM agent chooses an action and interacts with the game environments. Our experimental results demonstrate that our method significantly enhances the capability of LLMs as automated agents for symbolic reasoning, and our LLM agent is effective in text-based games involving symbolic tasks, achieving an average performance of 88% across all tasks.

{{</citation>}}


### (48/112) Machines Do See Color: A Guideline to Classify Different Forms of Racist Discourse in Large Corpora (Diana Davila Gordillo et al., 2024)

{{<citation>}}

Diana Davila Gordillo, Joan Timoneda, Sebastian Vallejo Vera. (2024)  
**Machines Do See Color: A Guideline to Classify Different Forms of Racist Discourse in Large Corpora**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.09333v1)  

---


**ABSTRACT**  
Current methods to identify and classify racist language in text rely on small-n qualitative approaches or large-n approaches focusing exclusively on overt forms of racist discourse. This article provides a step-by-step generalizable guideline to identify and classify different forms of racist discourse in large corpora. In our approach, we start by conceptualizing racism and its different manifestations. We then contextualize these racist manifestations to the time and place of interest, which allows researchers to identify their discursive form. Finally, we apply XLM-RoBERTa (XLM-R), a cross-lingual model for supervised text classification with a cutting-edge contextual understanding of text. We show that XLM-R and XLM-R-Racismo, our pretrained model, outperform other state-of-the-art approaches in classifying racism in large corpora. We illustrate our approach using a corpus of tweets relating to the Ecuadorian ind\'igena community between 2018 and 2021.

{{</citation>}}


### (49/112) Learning from Emotions, Demographic Information and Implicit User Feedback in Task-Oriented Document-Grounded Dialogues (Dominic Petrak et al., 2024)

{{<citation>}}

Dominic Petrak, Thy Thy Tran, Iryna Gurevych. (2024)  
**Learning from Emotions, Demographic Information and Implicit User Feedback in Task-Oriented Document-Grounded Dialogues**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Dialog, Dialogue, GPT, LLaMA, Natural Language Processing, T5  
[Paper Link](http://arxiv.org/abs/2401.09248v1)  

---


**ABSTRACT**  
The success of task-oriented and document-grounded dialogue systems depends on users accepting and enjoying using them. To achieve this, recently published work in the field of Human-Computer Interaction suggests that the combination of considering demographic information, user emotions and learning from the implicit feedback in their utterances, is particularly important. However, these findings have not yet been transferred to the field of Natural Language Processing, where these data are primarily studied separately. Accordingly, no sufficiently annotated dataset is available. To address this gap, we introduce FEDI, the first English dialogue dataset for task-oriented document-grounded dialogues annotated with demographic information, user emotions and implicit feedback. Our experiments with FLAN-T5, GPT-2 and LLaMA-2 show that these data have the potential to improve task completion and the factual consistency of the generated responses and user acceptance.

{{</citation>}}


### (50/112) UniVIE: A Unified Label Space Approach to Visual Information Extraction from Form-like Documents (Kai Hu et al., 2024)

{{<citation>}}

Kai Hu, Jiawei Wang, Weihong Lin, Zhuoyao Zhong, Lei Sun, Qiang Huo. (2024)  
**UniVIE: A Unified Label Space Approach to Visual Information Extraction from Form-like Documents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2401.09220v1)  

---


**ABSTRACT**  
Existing methods for Visual Information Extraction (VIE) from form-like documents typically fragment the process into separate subtasks, such as key information extraction, key-value pair extraction, and choice group extraction. However, these approaches often overlook the hierarchical structure of form documents, including hierarchical key-value pairs and hierarchical choice groups. To address these limitations, we present a new perspective, reframing VIE as a relation prediction problem and unifying labels of different tasks into a single label space. This unified approach allows for the definition of various relation types and effectively tackles hierarchical relationships in form-like documents. In line with this perspective, we present UniVIE, a unified model that addresses the VIE problem comprehensively. UniVIE functions using a coarse-to-fine strategy. It initially generates tree proposals through a tree proposal network, which are subsequently refined into hierarchical trees by a relation decoder module. To enhance the relation prediction capabilities of UniVIE, we incorporate two novel tree constraints into the relation decoder: a tree attention mask and a tree level embedding. Extensive experimental evaluations on both our in-house dataset HierForms and a publicly available dataset SIBR, substantiate that our method achieves state-of-the-art results, underscoring the effectiveness and potential of our unified approach in advancing the field of VIE.

{{</citation>}}


### (51/112) QAnswer: Towards Question Answering Search over Websites (Kunpeng Guo et al., 2024)

{{<citation>}}

Kunpeng Guo, Clement Defretiere, Dennis Diefenbach, Christophe Gravier, Antoine Gourru. (2024)  
**QAnswer: Towards Question Answering Search over Websites**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Google, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.09175v1)  

---


**ABSTRACT**  
Question Answering (QA) is increasingly used by search engines to provide results to their end-users, yet very few websites currently use QA technologies for their search functionality. To illustrate the potential of QA technologies for the website search practitioner, we demonstrate web searches that combine QA over knowledge graphs and QA over free text -- each being usually tackled separately. We also discuss the different benefits and drawbacks of both approaches for web site searches. We use the case studies made of websites hosted by the Wikimedia Foundation (namely Wikipedia and Wikidata). Differently from a search engine (e.g. Google, Bing, etc), the data are indexed integrally, i.e. we do not index only a subset, and they are indexed exclusively, i.e. we index only data available on the corresponding website.

{{</citation>}}


### (52/112) Fine-tuning Strategies for Domain Specific Question Answering under Low Annotation Budget Constraints (Kunpeng Guo et al., 2024)

{{<citation>}}

Kunpeng Guo, Dennis Diefenbach, Antoine Gourru, Christophe Gravier. (2024)  
**Fine-tuning Strategies for Domain Specific Question Answering under Low Annotation Budget Constraints**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.09168v1)  

---


**ABSTRACT**  
The progress introduced by pre-trained language models and their fine-tuning has resulted in significant improvements in most downstream NLP tasks. The unsupervised training of a language model combined with further target task fine-tuning has become the standard QA fine-tuning procedure. In this work, we demonstrate that this strategy is sub-optimal for fine-tuning QA models, especially under a low QA annotation budget, which is a usual setting in practice due to the extractive QA labeling cost. We draw our conclusions by conducting an exhaustive analysis of the performance of the alternatives of the sequential fine-tuning strategy on different QA datasets. Based on the experiments performed, we observed that the best strategy to fine-tune the QA model in low-budget settings is taking a pre-trained language model (PLM) and then fine-tuning PLM with a dataset composed of the target dataset and SQuAD dataset. With zero extra annotation effort, the best strategy outperforms the standard strategy by 2.28% to 6.48%. Our experiments provide one of the first investigations on how to best fine-tune a QA system under a low budget and are therefore of the utmost practical interest to the QA practitioners.

{{</citation>}}


### (53/112) Bridging Research and Readers: A Multi-Modal Automated Academic Papers Interpretation System (Feng Jiang et al., 2024)

{{<citation>}}

Feng Jiang, Kuang Wang, Haizhou Li. (2024)  
**Bridging Research and Readers: A Multi-Modal Automated Academic Papers Interpretation System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09150v1)  

---


**ABSTRACT**  
In the contemporary information era, significantly accelerated by the advent of Large-scale Language Models, the proliferation of scientific literature is reaching unprecedented levels. Researchers urgently require efficient tools for reading and summarizing academic papers, uncovering significant scientific literature, and employing diverse interpretative methodologies. To address this burgeoning demand, the role of automated scientific literature interpretation systems has become paramount. However, prevailing models, both commercial and open-source, confront notable challenges: they often overlook multimodal data, grapple with summarizing over-length texts, and lack diverse user interfaces. In response, we introduce an open-source multi-modal automated academic paper interpretation system (MMAPIS) with three-step process stages, incorporating LLMs to augment its functionality. Our system first employs the hybrid modality preprocessing and alignment module to extract plain text, and tables or figures from documents separately. It then aligns this information based on the section names they belong to, ensuring that data with identical section names are categorized under the same section. Following this, we introduce a hierarchical discourse-aware summarization method. It utilizes the extracted section names to divide the article into shorter text segments, facilitating specific summarizations both within and between sections via LLMs with specific prompts. Finally, we have designed four types of diversified user interfaces, including paper recommendation, multimodal Q\&A, audio broadcasting, and interpretation blog, which can be widely applied across various scenarios. Our qualitative and quantitative evaluations underscore the system's superiority, especially in scientific summarization, where it outperforms solutions relying solely on GPT-4.

{{</citation>}}


### (54/112) AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models (Dong shu et al., 2024)

{{<citation>}}

Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang. (2024)  
**AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09002v1)  

---


**ABSTRACT**  
In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.

{{</citation>}}


### (55/112) Efficient Adapter Finetuning for Tail Languages in Streaming Multilingual ASR (Junwen Bai et al., 2024)

{{<citation>}}

Junwen Bai, Bo Li, Qiujia Li, Tara N. Sainath, Trevor Strohman. (2024)  
**Efficient Adapter Finetuning for Tail Languages in Streaming Multilingual ASR**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2401.08992v1)  

---


**ABSTRACT**  
The end-to-end ASR model is often desired in the streaming multilingual scenario since it is easier to deploy and can benefit from pre-trained speech models such as powerful foundation models. Meanwhile, the heterogeneous nature and imbalanced data abundance of different languages may cause performance degradation, leading to asynchronous peak performance for different languages during training, especially on tail ones. Sometimes even the data itself may become unavailable as a result of the enhanced privacy protection. Existing work tend to significantly increase the model size or learn language-specific decoders to accommodate each language separately. In this study, we explore simple yet effective Language-Dependent Adapter (LDA) finetuning under a cascaded Conformer transducer framework enhanced by teacher pseudo-labeling for tail languages in the streaming multilingual ASR. The adapter only accounts for 0.4% of the full model per language. It is plugged into the frozen foundation model and is the only trainable module during the finetuning process with noisy student training. The final model merges the adapter parameters from different checkpoints for different languages. The model performance is validated on a challenging multilingual dictation dataset, which includes 39 tail languages across Latin, Greek, Arabic, etc. Our proposed method brings 12.2% word error rate reduction on average and up to 37.5% on a single locale. Furthermore, we show that our parameter-efficient LDA can match the quality of the full model finetuning, thus greatly alleviating the asynchronous peak performance issue.

{{</citation>}}


### (56/112) ReFT: Reasoning with Reinforced Fine-Tuning (Trung Quoc Luong et al., 2024)

{{<citation>}}

Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, Hang Li. (2024)  
**ReFT: Reasoning with Reinforced Fine-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.08967v1)  

---


**ABSTRACT**  
One way to enhance the reasoning capability of Large Language Models (LLMs) is to conduct Supervised Fine-Tuning (SFT) using Chain-of-Thought (CoT) annotations. This approach does not show sufficiently strong generalization ability, however, because the training only relies on the given CoT data. In math problem-solving, for example, there is usually only one annotated reasoning path for each question in the training data. Intuitively, it would be better for the algorithm to learn from multiple annotated reasoning paths given a question. To address this issue, we propose a simple yet effective approach called Reinforced Fine-Tuning (ReFT) to enhance the generalizability of learning LLMs for reasoning, with math problem-solving as an example. ReFT first warmups the model with SFT, and then employs on-line reinforcement learning, specifically the PPO algorithm in this paper, to further fine-tune the model, where an abundance of reasoning paths are automatically sampled given the question and the rewards are naturally derived from the ground-truth answers. Extensive experiments on GSM8K, MathQA, and SVAMP datasets show that ReFT significantly outperforms SFT, and the performance can be potentially further boosted by combining inference-time strategies such as majority voting and re-ranking. Note that ReFT obtains the improvement by learning from the same training questions as SFT, without relying on extra or augmented training questions. This indicates a superior generalization ability for ReFT.

{{</citation>}}


### (57/112) Partial Diacritization: A Context-Contrastive Inference Approach (Muhammad ElNokrashy et al., 2024)

{{<citation>}}

Muhammad ElNokrashy, Badr AlKhamissi. (2024)  
**Partial Diacritization: A Context-Contrastive Inference Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.08919v1)  

---


**ABSTRACT**  
Diacritization plays a pivotal role in improving readability and disambiguating the meaning of Arabic texts. Efforts have so far focused on marking every eligible character (Full Diacritization). Comparatively overlooked, Partial Diacritzation (PD) is the selection of a subset of characters to be marked to aid comprehension where needed. Research has indicated that excessive diacritic marks can hinder skilled readers--reducing reading speed and accuracy. We conduct a behavioral experiment and show that partially marked text is often easier to read than fully marked text, and sometimes easier than plain text. In this light, we introduce Context-Contrastive Partial Diacritization (CCPD)--a novel approach to PD which integrates seamlessly with existing Arabic diacritization systems. CCPD processes each word twice, once with context and once without, and diacritizes only the characters with disparities between the two inferences. Further, we introduce novel indicators for measuring partial diacritization quality (SR, PDER, HDER, ERE), essential for establishing this as a machine learning task. Lastly, we introduce TD2, a Transformer-variant of an established model which offers a markedly different per formance profile on our proposed indicators compared to all other known systems.

{{</citation>}}


## cs.CV (22)



### (58/112) Land Cover Image Classification (Antonio Rangel et al., 2024)

{{<citation>}}

Antonio Rangel, Juan Terven, Diana M. Cordova-Esparza, E. A. Chavez-Urbiola. (2024)  
**Land Cover Image Classification**  

---
Primary Category: cs.CV  
Categories: I-2-10, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2401.09607v1)  

---


**ABSTRACT**  
Land Cover (LC) image classification has become increasingly significant in understanding environmental changes, urban planning, and disaster management. However, traditional LC methods are often labor-intensive and prone to human error. This paper explores state-of-the-art deep learning models for enhanced accuracy and efficiency in LC analysis. We compare convolutional neural networks (CNN) against transformer-based methods, showcasing their applications and advantages in LC studies. We used EuroSAT, a patch-based LC classification data set based on Sentinel-2 satellite images and achieved state-of-the-art results using current transformer models.

{{</citation>}}


### (59/112) Efficient generative adversarial networks using linear additive-attention Transformers (Emilio Morales-Juarez et al., 2024)

{{<citation>}}

Emilio Morales-Juarez, Gibran Fuentes-Pineda. (2024)  
**Efficient generative adversarial networks using linear additive-attention Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.09596v1)  

---


**ABSTRACT**  
Although the capacity of deep generative models for image generation, such as Diffusion Models (DMs) and Generative Adversarial Networks (GANs), has dramatically improved in recent years, much of their success can be attributed to computationally expensive architectures. This has limited their adoption and use to research laboratories and companies with large resources, while significantly raising the carbon footprint for training, fine-tuning, and inference. In this work, we present LadaGAN, an efficient generative adversarial network that is built upon a novel Transformer block named Ladaformer. The main component of this block is a linear additive-attention mechanism that computes a single attention vector per head instead of the quadratic dot-product attention. We employ Ladaformer in both the generator and discriminator, which reduces the computational complexity and overcomes the training instabilities often associated with Transformer GANs. LadaGAN consistently outperforms existing convolutional and Transformer GANs on benchmark datasets at different resolutions while being significantly more efficient. Moreover, LadaGAN shows competitive performance compared to state-of-the-art multi-step generative models (e.g. DMs) using orders of magnitude less computational resources.

{{</citation>}}


### (60/112) Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model (Lianghui Zhu et al., 2024)

{{<citation>}}

Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, Xinggang Wang. (2024)  
**Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Representation Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2401.09417v1)  

---


**ABSTRACT**  
Recently the state space models (SSMs) with efficient hardware-aware designs, i.e., Mamba, have shown great potential for long sequence modeling. Building efficient and generic vision backbones purely upon SSMs is an appealing direction. However, representing visual data is challenging for SSMs due to the position-sensitivity of visual data and the requirement of global context for visual understanding. In this paper, we show that the reliance of visual representation learning on self-attention is not necessary and propose a new generic vision backbone with bidirectional Mamba blocks (Vim), which marks the image sequences with position embeddings and compresses the visual representation with bidirectional state space models. On ImageNet classification, COCO object detection, and ADE20k semantic segmentation tasks, Vim achieves higher performance compared to well-established vision transformers like DeiT, while also demonstrating significantly improved computation & memory efficiency. For example, Vim is 2.8$\times$ faster than DeiT and saves 86.8% GPU memory when performing batch inference to extract features on images with a resolution of 1248$\times$1248. The results demonstrate that Vim is capable of overcoming the computation & memory constraints on performing Transformer-style understanding for high-resolution images and it has great potential to become the next-generation backbone for vision foundation models. Code is available at https://github.com/hustvl/Vim.

{{</citation>}}


### (61/112) Vlogger: Make Your Dream A Vlog (Shaobin Zhuang et al., 2024)

{{<citation>}}

Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, Yali Wang. (2024)  
**Vlogger: Make Your Dream A Vlog**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09414v1)  

---


**ABSTRACT**  
In this work, we present Vlogger, a generic AI system for generating a minute-level video blog (i.e., vlog) of user descriptions. Different from short videos with a few seconds, vlog often contains a complex storyline with diversified scenes, which is challenging for most existing video generation approaches. To break through this bottleneck, our Vlogger smartly leverages Large Language Model (LLM) as Director and decomposes a long video generation task of vlog into four key stages, where we invoke various foundation models to play the critical roles of vlog professionals, including (1) Script, (2) Actor, (3) ShowMaker, and (4) Voicer. With such a design of mimicking human beings, our Vlogger can generate vlogs through explainable cooperation of top-down planning and bottom-up shooting. Moreover, we introduce a novel video diffusion model, ShowMaker, which serves as a videographer in our Vlogger for generating the video snippet of each shooting scene. By incorporating Script and Actor attentively as textual and visual prompts, it can effectively enhance spatial-temporal coherence in the snippet. Besides, we design a concise mixed training paradigm for ShowMaker, boosting its capacity for both T2V generation and prediction. Finally, the extensive experiments show that our method achieves state-of-the-art performance on zero-shot T2V generation and prediction tasks. More importantly, Vlogger can generate over 5-minute vlogs from open-world descriptions, without loss of video coherence on script and actor. The code and model is all available at https://github.com/zhuangshaobin/Vlogger.

{{</citation>}}


### (62/112) Siamese Meets Diffusion Network: SMDNet for Enhanced Change Detection in High-Resolution RS Imagery (Jia Jia et al., 2024)

{{<citation>}}

Jia Jia, Geunho Lee, Zhibo Wang, Lyu Zhi, Yuchu He. (2024)  
**Siamese Meets Diffusion Network: SMDNet for Enhanced Change Detection in High-Resolution RS Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.09325v1)  

---


**ABSTRACT**  
Recently, the application of deep learning to change detection (CD) has significantly progressed in remote sensing images. In recent years, CD tasks have mostly used architectures such as CNN and Transformer to identify these changes. However, these architectures have shortcomings in representing boundary details and are prone to false alarms and missed detections under complex lighting and weather conditions. For that, we propose a new network, Siamese Meets Diffusion Network (SMDNet). This network combines the Siam-U2Net Feature Differential Encoder (SU-FDE) and the denoising diffusion implicit model to improve the accuracy of image edge change detection and enhance the model's robustness under environmental changes. First, we propose an innovative SU-FDE module that utilizes shared weight features to capture differences between time series images and identify similarities between features to enhance edge detail detection. Furthermore, we add an attention mechanism to identify key coarse features to improve the model's sensitivity and accuracy. Finally, the diffusion model of progressive sampling is used to fuse key coarse features, and the noise reduction ability of the diffusion model and the advantages of capturing the probability distribution of image data are used to enhance the adaptability of the model in different environments. Our method's combination of feature extraction and diffusion models demonstrates effectiveness in change detection in remote sensing images. The performance evaluation of SMDNet on LEVIR-CD, DSIFN-CD, and CDD datasets yields validated F1 scores of 90.99%, 88.40%, and 88.47%, respectively. This substantiates the advanced capabilities of our model in accurately identifying variations and intricate details.

{{</citation>}}


### (63/112) PixelDINO: Semi-Supervised Semantic Segmentation for Detecting Permafrost Disturbances (Konrad Heidler et al., 2024)

{{<citation>}}

Konrad Heidler, Ingmar Nitze, Guido Grosse, Xiao Xiang Zhu. (2024)  
**PixelDINO: Semi-Supervised Semantic Segmentation for Detecting Permafrost Disturbances**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.09271v1)  

---


**ABSTRACT**  
Arctic Permafrost is facing significant changes due to global climate change. As these regions are largely inaccessible, remote sensing plays a crucial rule in better understanding the underlying processes not just on a local scale, but across the Arctic. In this study, we focus on the remote detection of retrogressive thaw slumps (RTS), a permafrost disturbance comparable to landslides induced by thawing. For such analyses from space, deep learning has become an indispensable tool, but limited labelled training data remains a challenge for training accurate models. To improve model generalization across the Arctic without the need for additional labelled data, we present a semi-supervised learning approach to train semantic segmentation models to detect RTS. Our framework called PixelDINO is trained in parallel on labelled data as well as unlabelled data. For the unlabelled data, the model segments the imagery into self-taught pseudo-classes and the training procedure ensures consistency of these pseudo-classes across strong augmentations of the input data. Our experimental results demonstrate that PixelDINO can improve model performance both over supervised baseline methods as well as existing semi-supervised semantic segmentation approaches, highlighting its potential for training robust models that generalize well to regions that were not included in the training data. The project page containing code and other materials for this study can be found at \url{https://khdlr.github.io/PixelDINO/}.

{{</citation>}}


### (64/112) P$^2$OT: Progressive Partial Optimal Transport for Deep Imbalanced Clustering (Chuyu Zhang et al., 2024)

{{<citation>}}

Chuyu Zhang, Hui Ren, Xuming He. (2024)  
**P$^2$OT: Progressive Partial Optimal Transport for Deep Imbalanced Clustering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.09266v1)  

---


**ABSTRACT**  
Deep clustering, which learns representation and semantic clustering without labels information, poses a great challenge for deep learning-based approaches. Despite significant progress in recent years, most existing methods focus on uniformly distributed datasets, significantly limiting the practical applicability of their methods. In this paper, we first introduce a more practical problem setting named deep imbalanced clustering, where the underlying classes exhibit an imbalance distribution. To tackle this problem, we propose a novel pseudo-labeling-based learning framework. Our framework formulates pseudo-label generation as a progressive partial optimal transport problem, which progressively transports each sample to imbalanced clusters under prior distribution constraints, thus generating imbalance-aware pseudo-labels and learning from high-confident samples. In addition, we transform the initial formulation into an unbalanced optimal transport problem with augmented constraints, which can be solved efficiently by a fast matrix scaling algorithm. Experiments on various datasets, including a human-curated long-tailed CIFAR100, challenging ImageNet-R, and large-scale subsets of fine-grained iNaturalist2018 datasets, demonstrate the superiority of our method.

{{</citation>}}


### (65/112) Dynamic Relation Transformer for Contextual Text Block Detection (Jiawei Wang et al., 2024)

{{<citation>}}

Jiawei Wang, Shunchi Zhang, Kai Hu, Chixiang Ma, Zhuoyao Zhong, Lei Sun, Qiang Huo. (2024)  
**Dynamic Relation Transformer for Contextual Text Block Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.09232v1)  

---


**ABSTRACT**  
Contextual Text Block Detection (CTBD) is the task of identifying coherent text blocks within the complexity of natural scenes. Previous methodologies have treated CTBD as either a visual relation extraction challenge within computer vision or as a sequence modeling problem from the perspective of natural language processing. We introduce a new framework that frames CTBD as a graph generation problem. This methodology consists of two essential procedures: identifying individual text units as graph nodes and discerning the sequential reading order relationships among these units as graph edges. Leveraging the cutting-edge capabilities of DQ-DETR for node detection, our framework innovates further by integrating a novel mechanism, a Dynamic Relation Transformer (DRFormer), dedicated to edge generation. DRFormer incorporates a dual interactive transformer decoder that deftly manages a dynamic graph structure refinement process. Through this iterative process, the model systematically enhances the graph's fidelity, ultimately resulting in improved precision in detecting contextual text blocks. Comprehensive experimental evaluations conducted on both SCUT-CTW-Context and ReCTS-Context datasets substantiate that our method achieves state-of-the-art results, underscoring the effectiveness and potential of our graph generation framework in advancing the field of CTBD.

{{</citation>}}


### (66/112) Continuous Piecewise-Affine Based Motion Model for Image Animation (Hexiang Wang et al., 2024)

{{<citation>}}

Hexiang Wang, Fengqi Liu, Qianyu Zhou, Ran Yi, Xin Tan, Lizhuang Ma. (2024)  
**Continuous Piecewise-Affine Based Motion Model for Image Animation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09146v1)  

---


**ABSTRACT**  
Image animation aims to bring static images to life according to driving videos and create engaging visual content that can be used for various purposes such as animation, entertainment, and education. Recent unsupervised methods utilize affine and thin-plate spline transformations based on keypoints to transfer the motion in driving frames to the source image. However, limited by the expressive power of the transformations used, these methods always produce poor results when the gap between the motion in the driving frame and the source image is large. To address this issue, we propose to model motion from the source image to the driving frame in highly-expressive diffeomorphism spaces. Firstly, we introduce Continuous Piecewise-Affine based (CPAB) transformation to model the motion and present a well-designed inference algorithm to generate CPAB transformation from control keypoints. Secondly, we propose a SAM-guided keypoint semantic loss to further constrain the keypoint extraction process and improve the semantic consistency between the corresponding keypoints on the source and driving images. Finally, we design a structure alignment loss to align the structure-related features extracted from driving and generated images, thus helping the generator generate results that are more consistent with the driving action. Extensive experiments on four datasets demonstrate the effectiveness of our method against state-of-the-art competitors quantitatively and qualitatively. Code will be publicly available at: https://github.com/DevilPG/AAAI2024-CPABMM.

{{</citation>}}


### (67/112) SM$^3$: Self-Supervised Multi-task Modeling with Multi-view 2D Images for Articulated Objects (Haowen Wang et al., 2024)

{{<citation>}}

Haowen Wang, Zhen Zhao, Zhao Jin, Zhengping Che, Liang Qiao, Yakun Huang, Zhipeng Fan, Xiuquan Qiao, Jian Tang. (2024)  
**SM$^3$: Self-Supervised Multi-task Modeling with Multi-view 2D Images for Articulated Objects**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.09133v1)  

---


**ABSTRACT**  
Reconstructing real-world objects and estimating their movable joint structures are pivotal technologies within the field of robotics. Previous research has predominantly focused on supervised approaches, relying on extensively annotated datasets to model articulated objects within limited categories. However, this approach falls short of effectively addressing the diversity present in the real world. To tackle this issue, we propose a self-supervised interaction perception method, referred to as SM$^3$, which leverages multi-view RGB images captured before and after interaction to model articulated objects, identify the movable parts, and infer the parameters of their rotating joints. By constructing 3D geometries and textures from the captured 2D images, SM$^3$ achieves integrated optimization of movable part and joint parameters during the reconstruction process, obviating the need for annotations. Furthermore, we introduce the MMArt dataset, an extension of PartNet-Mobility, encompassing multi-view and multi-modal data of articulated objects spanning diverse categories. Evaluations demonstrate that SM$^3$ surpasses existing benchmarks across various categories and objects, while its adaptability in real-world scenarios has been thoroughly validated.

{{</citation>}}


### (68/112) Trapped in texture bias? A large scale comparison of deep instance segmentation (Johannes Theodoridis et al., 2024)

{{<citation>}}

Johannes Theodoridis, Jessica Hofmann, Johannes Maucher, Andreas Schilling. (2024)  
**Trapped in texture bias? A large scale comparison of deep instance segmentation**  

---
Primary Category: cs.CV  
Categories: I-2; I-2-10; I-4; I-4-6; I-4-7; I-5, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.09109v1)  

---


**ABSTRACT**  
Do deep learning models for instance segmentation generalize to novel objects in a systematic way? For classification, such behavior has been questioned. In this study, we aim to understand if certain design decisions such as framework, architecture or pre-training contribute to the semantic understanding of instance segmentation. To answer this question, we consider a special case of robustness and compare pre-trained models on a challenging benchmark for object-centric, out-of-distribution texture. We do not introduce another method in this work. Instead, we take a step back and evaluate a broad range of existing literature. This includes Cascade and Mask R-CNN, Swin Transformer, BMask, YOLACT(++), DETR, BCNet, SOTR and SOLOv2. We find that YOLACT++, SOTR and SOLOv2 are significantly more robust to out-of-distribution texture than other frameworks. In addition, we show that deeper and dynamic architectures improve robustness whereas training schedules, data augmentation and pre-training have only a minor impact. In summary we evaluate 68 models on 61 versions of MS COCO for a total of 4148 evaluations.

{{</citation>}}


### (69/112) UniVG: Towards UNIfied-modal Video Generation (Ludan Ruan et al., 2024)

{{<citation>}}

Ludan Ruan, Lei Tian, Chuanwei Huang, Xu Zhang, Xinyan Xiao. (2024)  
**UniVG: Towards UNIfied-modal Video Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Bias  
[Paper Link](http://arxiv.org/abs/2401.09084v1)  

---


**ABSTRACT**  
Diffusion based video generation has received extensive attention and achieved considerable success within both the academic and industrial communities. However, current efforts are mainly concentrated on single-objective or single-task video generation, such as generation driven by text, by image, or by a combination of text and image. This cannot fully meet the needs of real-world application scenarios, as users are likely to input images and text conditions in a flexible manner, either individually or in combination. To address this, we propose a Unified-modal Video Genearation system that is capable of handling multiple video generation tasks across text and image modalities. To this end, we revisit the various video generation tasks within our system from the perspective of generative freedom, and classify them into high-freedom and low-freedom video generation categories. For high-freedom video generation, we employ Multi-condition Cross Attention to generate videos that align with the semantics of the input images or text. For low-freedom video generation, we introduce Biased Gaussian Noise to replace the pure random Gaussian Noise, which helps to better preserve the content of the input conditions. Our method achieves the lowest Fr\'echet Video Distance (FVD) on the public academic benchmark MSR-VTT, surpasses the current open-source methods in human evaluations, and is on par with the current close-source method Gen2. For more samples, visit https://univg-baidu.github.io.

{{</citation>}}


### (70/112) Remote Sensing ChatGPT: Solving Remote Sensing Tasks with ChatGPT and Visual Models (Haonan Guo et al., 2024)

{{<citation>}}

Haonan Guo, Xin Su, Chen Wu, Bo Du, Liangpei Zhang, Deren Li. (2024)  
**Remote Sensing ChatGPT: Solving Remote Sensing Tasks with ChatGPT and Visual Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.09083v1)  

---


**ABSTRACT**  
Recently, the flourishing large language models(LLM), especially ChatGPT, have shown exceptional performance in language understanding, reasoning, and interaction, attracting users and researchers from multiple fields and domains. Although LLMs have shown great capacity to perform human-like task accomplishment in natural language and natural image, their potential in handling remote sensing interpretation tasks has not yet been fully explored. Moreover, the lack of automation in remote sensing task planning hinders the accessibility of remote sensing interpretation techniques, especially to non-remote sensing experts from multiple research fields. To this end, we present Remote Sensing ChatGPT, an LLM-powered agent that utilizes ChatGPT to connect various AI-based remote sensing models to solve complicated interpretation tasks. More specifically, given a user request and a remote sensing image, we utilized ChatGPT to understand user requests, perform task planning according to the tasks' functions, execute each subtask iteratively, and generate the final response according to the output of each subtask. Considering that LLM is trained with natural language and is not capable of directly perceiving visual concepts as contained in remote sensing images, we designed visual cues that inject visual information into ChatGPT. With Remote Sensing ChatGPT, users can simply send a remote sensing image with the corresponding request, and get the interpretation results as well as language feedback from Remote Sensing ChatGPT. Experiments and examples show that Remote Sensing ChatGPT can tackle a wide range of remote sensing tasks and can be extended to more tasks with more sophisticated models such as the remote sensing foundation model. The code and demo of Remote Sensing ChatGPT is publicly available at https://github.com/HaonanGuo/Remote-Sensing-ChatGPT .

{{</citation>}}


### (71/112) CrossVideo: Self-supervised Cross-modal Contrastive Learning for Point Cloud Video Understanding (Yunze Liu et al., 2024)

{{<citation>}}

Yunze Liu, Changxi Chen, Zifan Wang, Li Yi. (2024)  
**CrossVideo: Self-supervised Cross-modal Contrastive Learning for Point Cloud Video Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.09057v1)  

---


**ABSTRACT**  
This paper introduces a novel approach named CrossVideo, which aims to enhance self-supervised cross-modal contrastive learning in the field of point cloud video understanding. Traditional supervised learning methods encounter limitations due to data scarcity and challenges in label acquisition. To address these issues, we propose a self-supervised learning method that leverages the cross-modal relationship between point cloud videos and image videos to acquire meaningful feature representations. Intra-modal and cross-modal contrastive learning techniques are employed to facilitate effective comprehension of point cloud video. We also propose a multi-level contrastive approach for both modalities. Through extensive experiments, we demonstrate that our method significantly surpasses previous state-of-the-art approaches, and we conduct comprehensive ablation studies to validate the effectiveness of our proposed designs.

{{</citation>}}


### (72/112) Enhancing Lidar-based Object Detection in Adverse Weather using Offset Sequences in Time (Raphael van Kempen et al., 2024)

{{<citation>}}

Raphael van Kempen, Tim Rehbronn, Abin Jose, Johannes Stegmaier, Bastian Lampe, Timo Woopen, Lutz Eckstein. (2024)  
**Enhancing Lidar-based Object Detection in Adverse Weather using Offset Sequences in Time**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.09049v1)  

---


**ABSTRACT**  
Automated vehicles require an accurate perception of their surroundings for safe and efficient driving. Lidar-based object detection is a widely used method for environment perception, but its performance is significantly affected by adverse weather conditions such as rain and fog. In this work, we investigate various strategies for enhancing the robustness of lidar-based object detection by processing sequential data samples generated by lidar sensors. Our approaches leverage temporal information to improve a lidar object detection model, without the need for additional filtering or pre-processing steps. We compare $10$ different neural network architectures that process point cloud sequences including a novel augmentation strategy introducing a temporal offset between frames of a sequence during training and evaluate the effectiveness of all strategies on lidar point clouds under adverse weather conditions through experiments. Our research provides a comprehensive study of effective methods for mitigating the effects of adverse weather on the reliability of lidar-based object detection using sequential data that are evaluated using public datasets such as nuScenes, Dense, and the Canadian Adverse Driving Conditions Dataset. Our findings demonstrate that our novel method, involving temporal offset augmentation through randomized frame skipping in sequences, enhances object detection accuracy compared to both the baseline model (Pillar-based Object Detection) and no augmentation.

{{</citation>}}


### (73/112) Cross-modality Guidance-aided Multi-modal Learning with Dual Attention for MRI Brain Tumor Grading (Dunyuan Xu et al., 2024)

{{<citation>}}

Dunyuan Xu, Xi Wang, Jinyue Cai, Pheng-Ann Heng. (2024)  
**Cross-modality Guidance-aided Multi-modal Learning with Dual Attention for MRI Brain Tumor Grading**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.09029v1)  

---


**ABSTRACT**  
Brain tumor represents one of the most fatal cancers around the world, and is very common in children and the elderly. Accurate identification of the type and grade of tumor in the early stages plays an important role in choosing a precise treatment plan. The Magnetic Resonance Imaging (MRI) protocols of different sequences provide clinicians with important contradictory information to identify tumor regions. However, manual assessment is time-consuming and error-prone due to big amount of data and the diversity of brain tumor types. Hence, there is an unmet need for MRI automated brain tumor diagnosis. We observe that the predictive capability of uni-modality models is limited and their performance varies widely across modalities, and the commonly used modality fusion methods would introduce potential noise, which results in significant performance degradation. To overcome these challenges, we propose a novel cross-modality guidance-aided multi-modal learning with dual attention for addressing the task of MRI brain tumor grading. To balance the tradeoff between model efficiency and efficacy, we employ ResNet Mix Convolution as the backbone network for feature extraction. Besides, dual attention is applied to capture the semantic interdependencies in spatial and slice dimensions respectively. To facilitate information interaction among modalities, we design a cross-modality guidance-aided module where the primary modality guides the other secondary modalities during the process of training, which can effectively leverage the complementary information of different MRI modalities and meanwhile alleviate the impact of the possible noise.

{{</citation>}}


### (74/112) Hybrid of DiffStride and Spectral Pooling in Convolutional Neural Networks (Sulthan Rafif et al., 2024)

{{<citation>}}

Sulthan Rafif, Mochamad Arfan Ravy Wahyu Pratama, Mohammad Faris Azhar, Ahmad Mustafidul Ibad, Lailil Muflikhah, Novanto Yudistira. (2024)  
**Hybrid of DiffStride and Spectral Pooling in Convolutional Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2401.09008v1)  

---


**ABSTRACT**  
Stride determines the distance between adjacent filter positions as the filter moves across the input. A fixed stride causes important information contained in the image can not be captured, so that important information is not classified. Therefore, in previous research, the DiffStride Method was applied, namely the Strided Convolution Method with which it can learn its own stride value. Severe Quantization and a constraining lower bound on preserved information are arises with Max Pooling Downsampling Method. Spectral Pooling reduce the constraint lower bound on preserved information by cutting off the representation in the frequency domain. In this research a CNN Model is proposed with the Downsampling Learnable Stride Technique performed by Backpropagation combined with the Spectral Pooling Technique. Diffstride and Spectral Pooling techniques are expected to maintain most of the information contained in the image. In this study, we compare the Hybrid Method, which is a combined implementation of Spectral Pooling and DiffStride against the Baseline Method, which is the DiffStride implementation on ResNet 18. The accuracy result of the DiffStride combination with Spectral Pooling improves over DiffStride which is baseline method by 0.0094. This shows that the Hybrid Method can maintain most of the information by cutting of the representation in the frequency domain and determine the stride of the learning result through Backpropagation.

{{</citation>}}


### (75/112) COCO is 'ALL'' You Need for Visual Instruction Fine-tuning (Xiaotian Han et al., 2024)

{{<citation>}}

Xiaotian Han, Yiqi Wang, Bohan Zhai, Quanzeng You, Hongxia Yang. (2024)  
**COCO is 'ALL'' You Need for Visual Instruction Fine-tuning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.08968v1)  

---


**ABSTRACT**  
Multi-modal Large Language Models (MLLMs) are increasingly prominent in the field of artificial intelligence. Visual instruction fine-tuning (IFT) is a vital process for aligning MLLMs' output with user's intentions. High-quality and diversified instruction following data is the key to this fine-tuning process. Recent studies propose to construct visual IFT datasets through a multifaceted approach: transforming existing datasets with rule-based templates, employing GPT-4 for rewriting annotations, and utilizing GPT-4V for visual dataset pseudo-labeling. LLaVA-1.5 adopted similar approach and construct LLaVA-mix-665k, which is one of the simplest, most widely used, yet most effective IFT datasets today. Notably, when properly fine-tuned with this dataset, MLLMs can achieve state-of-the-art performance on several benchmarks. However, we noticed that models trained with this dataset often struggle to follow user instructions properly in multi-round dialog. In addition, tradition caption and VQA evaluation benchmarks, with their closed-form evaluation structure, are not fully equipped to assess the capabilities of modern open-ended generative MLLMs. This problem is not unique to the LLaVA-mix-665k dataset, but may be a potential issue in all IFT datasets constructed from image captioning or VQA sources, though the extent of this issue may vary. We argue that datasets with diverse and high-quality detailed instruction following annotations are essential and adequate for MLLMs IFT. In this work, we establish a new IFT dataset, with images sourced from the COCO dataset along with more diverse instructions. Our experiments show that when fine-tuned with out proposed dataset, MLLMs achieve better performance on open-ended evaluation benchmarks in both single-round and multi-round dialog setting.

{{</citation>}}


### (76/112) Dynamic DNNs and Runtime Management for Efficient Inference on Mobile/Embedded Devices (Lei Xun et al., 2024)

{{<citation>}}

Lei Xun, Jonathon Hare, Geoff V. Merrett. (2024)  
**Dynamic DNNs and Runtime Management for Efficient Inference on Mobile/Embedded Devices**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.08965v1)  

---


**ABSTRACT**  
Deep neural network (DNN) inference is increasingly being executed on mobile and embedded platforms due to several key advantages in latency, privacy and always-on availability. However, due to limited computing resources, efficient DNN deployment on mobile and embedded platforms is challenging. Although many hardware accelerators and static model compression methods were proposed by previous works, at system runtime, multiple applications are typically executed concurrently and compete for hardware resources. This raises two main challenges: Runtime Hardware Availability and Runtime Application Variability. Previous works have addressed these challenges through either dynamic neural networks that contain sub-networks with different performance trade-offs or runtime hardware resource management. In this thesis, we proposed a combined method, a system was developed for DNN performance trade-off management, combining the runtime trade-off opportunities in both algorithms and hardware to meet dynamically changing application performance targets and hardware constraints in real time. We co-designed novel Dynamic Super-Networks to maximise runtime system-level performance and energy efficiency on heterogeneous hardware platforms. Compared with SOTA, our experimental results using ImageNet on the GPU of Jetson Xavier NX show our model is 2.4x faster for similar ImageNet Top-1 accuracy, or 5.1% higher accuracy at similar latency. We also designed a hierarchical runtime resource manager that tunes both dynamic neural networks and DVFS at runtime. Compared with the Linux DVFS governor schedutil, our runtime approach achieves up to a 19% energy reduction and a 9% latency reduction in single model deployment scenario, and an 89% energy reduction and a 23% latency reduction in a two concurrent model deployment scenario.

{{</citation>}}


### (77/112) Uncertainty-aware No-Reference Point Cloud Quality Assessment (Songlin Fan et al., 2024)

{{<citation>}}

Songlin Fan, Zixuan Guo, Wei Gao, Ge Li. (2024)  
**Uncertainty-aware No-Reference Point Cloud Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.08926v1)  

---


**ABSTRACT**  
The evolution of compression and enhancement algorithms necessitates an accurate quality assessment for point clouds. Previous works consistently regard point cloud quality assessment (PCQA) as a MOS regression problem and devise a deterministic mapping, ignoring the stochasticity in generating MOS from subjective tests. Besides, the viewpoint switching of 3D point clouds in subjective tests reinforces the judging stochasticity of different subjects compared with traditional images. This work presents the first probabilistic architecture for no-reference PCQA, motivated by the labeling process of existing datasets. The proposed method can model the quality judging stochasticity of subjects through a tailored conditional variational autoencoder (CVAE) and produces multiple intermediate quality ratings. These intermediate ratings simulate the judgments from different subjects and are then integrated into an accurate quality prediction, mimicking the generation process of a ground truth MOS. Specifically, our method incorporates a Prior Module, a Posterior Module, and a Quality Rating Generator, where the former two modules are introduced to model the judging stochasticity in subjective tests, while the latter is developed to generate diverse quality ratings. Extensive experiments indicate that our approach outperforms previous cutting-edge methods by a large margin and exhibits gratifying cross-dataset robustness.

{{</citation>}}


### (78/112) Efficient Image Super-Resolution via Symmetric Visual Attention Network (Chengxu Wu et al., 2024)

{{<citation>}}

Chengxu Wu, Qinrui Fan, Shu Hu, Xi Wu, Xin Wang, Jing Hu. (2024)  
**Efficient Image Super-Resolution via Symmetric Visual Attention Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.08913v1)  

---


**ABSTRACT**  
An important development direction in the Single-Image Super-Resolution (SISR) algorithms is to improve the efficiency of the algorithms. Recently, efficient Super-Resolution (SR) research focuses on reducing model complexity and improving efficiency through improved deep small kernel convolution, leading to a small receptive field. The large receptive field obtained by large kernel convolution can significantly improve image quality, but the computational cost is too high. To improve the reconstruction details of efficient super-resolution reconstruction, we propose a Symmetric Visual Attention Network (SVAN) by applying large receptive fields. The SVAN decomposes a large kernel convolution into three different combinations of convolution operations and combines them with an attention mechanism to form a Symmetric Large Kernel Attention Block (SLKAB), which forms a symmetric attention block with a bottleneck structure by the size of the receptive field in the convolution combination to extract depth features effectively as the basic component of the SVAN. Our network gets a large receptive field while minimizing the number of parameters and improving the perceptual ability of the model. The experimental results show that the proposed SVAN can obtain high-quality super-resolution reconstruction results using only about 30% of the parameters of existing SOTA methods.

{{</citation>}}


### (79/112) PPR: Enhancing Dodging Attacks while Maintaining Impersonation Attacks on Face Recognition Systems (Fengfan Zhou et al., 2024)

{{<citation>}}

Fengfan Zhou, Heifei Ling. (2024)  
**PPR: Enhancing Dodging Attacks while Maintaining Impersonation Attacks on Face Recognition Systems**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack, Pruning  
[Paper Link](http://arxiv.org/abs/2401.08903v1)  

---


**ABSTRACT**  
Adversarial Attacks on Face Recognition (FR) encompass two types: impersonation attacks and evasion attacks. We observe that achieving a successful impersonation attack on FR does not necessarily ensure a successful dodging attack on FR in the black-box setting. Introducing a novel attack method named Pre-training Pruning Restoration Attack (PPR), we aim to enhance the performance of dodging attacks whilst avoiding the degradation of impersonation attacks. Our method employs adversarial example pruning, enabling a portion of adversarial perturbations to be set to zero, while tending to maintain the attack performance. By utilizing adversarial example pruning, we can prune the pre-trained adversarial examples and selectively free up certain adversarial perturbations. Thereafter, we embed adversarial perturbations in the pruned area, which enhances the dodging performance of the adversarial face examples. The effectiveness of our proposed attack method is demonstrated through our experimental results, showcasing its superior performance.

{{</citation>}}


## cs.CR (7)



### (80/112) MedBlindTuner: Towards Privacy-preserving Fine-tuning on Biomedical Images with Transformers and Fully Homomorphic Encryption (Prajwal Panzade et al., 2024)

{{<citation>}}

Prajwal Panzade, Daniel Takabi, Zhipeng Cai. (2024)  
**MedBlindTuner: Towards Privacy-preserving Fine-tuning on Biomedical Images with Transformers and Fully Homomorphic Encryption**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.09604v1)  

---


**ABSTRACT**  
Advancements in machine learning (ML) have significantly revolutionized medical image analysis, prompting hospitals to rely on external ML services. However, the exchange of sensitive patient data, such as chest X-rays, poses inherent privacy risks when shared with third parties. Addressing this concern, we propose MedBlindTuner, a privacy-preserving framework leveraging fully homomorphic encryption (FHE) and a data-efficient image transformer (DEiT). MedBlindTuner enables the training of ML models exclusively on FHE-encrypted medical images. Our experimental evaluation demonstrates that MedBlindTuner achieves comparable accuracy to models trained on non-encrypted images, offering a secure solution for outsourcing ML computations while preserving patient data privacy. To the best of our knowledge, this is the first work that uses data-efficient image transformers and fully homomorphic encryption in this domain.

{{</citation>}}


### (81/112) Zero Trust Implementation in the Emerging Technologies Era: Survey (Abraham Itzhak Weinberg et al., 2024)

{{<citation>}}

Abraham Itzhak Weinberg, Kelly Cohen. (2024)  
**Zero Trust Implementation in the Emerging Technologies Era: Survey**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09575v1)  

---


**ABSTRACT**  
This paper presents a comprehensive analysis of the shift from the traditional perimeter model of security to the Zero Trust (ZT) framework, emphasizing the key points in the transition and the practical application of ZT. It outlines the differences between ZT policies and legacy security policies, along with the significant events that have impacted the evolution of ZT. Additionally, the paper explores the potential impacts of emerging technologies, such as Artificial Intelligence (AI) and quantum computing, on the policy and implementation of ZT. The study thoroughly examines how AI can enhance ZT by utilizing Machine Learning (ML) algorithms to analyze patterns, detect anomalies, and predict threats, thereby improving real-time decision-making processes. Furthermore, the paper demonstrates how a chaos theory-based approach, in conjunction with other technologies like eXtended Detection and Response (XDR), can effectively mitigate cyberattacks. As quantum computing presents new challenges to ZT and cybersecurity as a whole, the paper delves into the intricacies of ZT migration, automation, and orchestration, addressing the complexities associated with these aspects. Finally, the paper provides a best practice approach for the seamless implementation of ZT in organizations, laying out the proposed guidelines to facilitate organizations in their transition towards a more secure ZT model. The study aims to support organizations in successfully implementing ZT and enhancing their cybersecurity measures.

{{</citation>}}


### (82/112) Username Squatting on Online Social Networks: A Study on X (Anastasios Lepipas et al., 2024)

{{<citation>}}

Anastasios Lepipas, Anastasia Borovykh, Soteris Demetriou. (2024)  
**Username Squatting on Online Social Networks: A Study on X**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SI, cs.CR  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2401.09209v1)  

---


**ABSTRACT**  
Adversaries have been targeting unique identifiers to launch typo-squatting, mobile app squatting and even voice squatting attacks. Anecdotal evidence suggest that online social networks (OSNs) are also plagued with accounts that use similar usernames. This can be confusing to users but can also be exploited by adversaries. However, to date no study characterizes this problem on OSNs. In this work, we define the username squatting problem and design the first multi-faceted measurement study to characterize it on X. We develop a username generation tool (UsernameCrazy) to help us analyze hundreds of thousands of username variants derived from celebrity accounts. Our study reveals that thousands of squatted usernames have been suspended by X, while tens of thousands that still exist on the network are likely bots. Out of these, a large number share similar profile pictures and profile names to the original account signalling impersonation attempts. We found that squatted accounts are being mentioned by mistake in tweets hundreds of thousands of times and are even being prioritized in searches by the network's search recommendation algorithm exacerbating the negative impact squatted accounts can have in OSNs. We use our insights and take the first step to address this issue by designing a framework (SQUAD) that combines UsernameCrazy with a new classifier to efficiently detect suspicious squatted accounts. Our evaluation of SQUAD's prototype implementation shows that it can achieve 94% F1-score when trained on a small dataset.

{{</citation>}}


### (83/112) Cross-Domain AI for Early Attack Detection and Defense Against Malicious Flows in O-RAN (Bruno Missi Xavier et al., 2024)

{{<citation>}}

Bruno Missi Xavier, Merim Dzaferagic, Irene Vilà, Magnos Martinello, Marco Ruffini. (2024)  
**Cross-Domain AI for Early Attack Detection and Defense Against Malicious Flows in O-RAN**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09204v1)  

---


**ABSTRACT**  
Only the chairs can edit In the fight against cyber attacks, Network Softwarization (NS) is a flexible and adaptable shield, using advanced software to spot malicious activity in regular network traffic. However, the availability of comprehensive datasets for mobile networks, which are fundamental for the development of Machine Learning (ML) solutions for attack detection near their source, is still limited. Cross-Domain Artificial Intelligence (AI) can be the key to address this, although its application in Open Radio Access Network (O-RAN) is still at its infancy. To address these challenges, we deployed an end-to-end O-RAN network, that was used to collect data from the RAN and the transport network. These datasets allow us to combine the knowledge from an in-network ML traffic classifier for attack detection to bolster the training of an ML-based traffic classifier specifically tailored for the RAN. Our results demonstrate the potential of the proposed approach, achieving an accuracy rate of 93%. This approach not only bridges critical gaps in mobile network security but also showcases the potential of cross-domain AI in enhancing the efficacy of network security measures.

{{</citation>}}


### (84/112) Machine Learning for Healthcare-IoT Security: A Review and Risk Mitigation (Mirza Akhi Khatun et al., 2024)

{{<citation>}}

Mirza Akhi Khatun, Sanober Farheen Memon, Ciarán Eising, Lubna Luxmi Dhirani. (2024)  
**Machine Learning for Healthcare-IoT Security: A Review and Risk Mitigation**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2401.09124v1)  

---


**ABSTRACT**  
The Healthcare Internet-of-Things (H-IoT), commonly known as Digital Healthcare, is a data-driven infrastructure that highly relies on smart sensing devices (i.e., blood pressure monitors, temperature sensors, etc.) for faster response time, treatments, and diagnosis. However, with the evolving cyber threat landscape, IoT devices have become more vulnerable to the broader risk surface (e.g., risks associated with generative AI, 5G-IoT, etc.), which, if exploited, may lead to data breaches, unauthorized access, and lack of command and control and potential harm. This paper reviews the fundamentals of healthcare IoT, its privacy, and data security challenges associated with machine learning and H-IoT devices. The paper further emphasizes the importance of monitoring healthcare IoT layers such as perception, network, cloud, and application. Detecting and responding to anomalies involves various cyber-attacks and protocols such as Wi-Fi 6, Narrowband Internet of Things (NB-IoT), Bluetooth, ZigBee, LoRa, and 5G New Radio (5G NR). A robust authentication mechanism based on machine learning and deep learning techniques is required to protect and mitigate H-IoT devices from increasing cybersecurity vulnerabilities. Hence, in this review paper, security and privacy challenges and risk mitigation strategies for building resilience in H-IoT are explored and reported.

{{</citation>}}


### (85/112) GPT in Sheep's Clothing: The Risk of Customized GPTs (Sagiv Antebi et al., 2024)

{{<citation>}}

Sagiv Antebi, Noam Azulay, Edan Habler, Ben Ganon, Asaf Shabtai, Yuval Elovici. (2024)  
**GPT in Sheep's Clothing: The Risk of Customized GPTs**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.09075v1)  

---


**ABSTRACT**  
In November 2023, OpenAI introduced a new service allowing users to create custom versions of ChatGPT (GPTs) by using specific instructions and knowledge to guide the model's behavior. We aim to raise awareness of the fact that GPTs can be used maliciously, posing privacy and security risks to their users.

{{</citation>}}


### (86/112) AntiPhishStack: LSTM-based Stacked Generalization Model for Optimized Phishing URLs Detection (Saba Aslam et al., 2024)

{{<citation>}}

Saba Aslam, Hafsa Aslam, Arslan Manzoor, Chen Hui, Abdur Rasool. (2024)  
**AntiPhishStack: LSTM-based Stacked Generalization Model for Optimized Phishing URLs Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.08947v1)  

---


**ABSTRACT**  
The escalating reliance on revolutionary online web services has introduced heightened security risks, with persistent challenges posed by phishing despite extensive security measures. Traditional phishing systems, reliant on machine learning and manual features, struggle with evolving tactics. Recent advances in deep learning offer promising avenues for tackling novel phishing challenges and malicious URLs. This paper introduces a two-phase stack generalized model named AntiPhishStack, designed to detect phishing sites. The model leverages the learning of URLs and character-level TF-IDF features symmetrically, enhancing its ability to combat emerging phishing threats. In Phase I, features are trained on a base machine learning classifier, employing K-fold cross-validation for robust mean prediction. Phase II employs a two-layered stacked-based LSTM network with five adaptive optimizers for dynamic compilation, ensuring premier prediction on these features. Additionally, the symmetrical predictions from both phases are optimized and integrated to train a meta-XGBoost classifier, contributing to a final robust prediction. The significance of this work lies in advancing phishing detection with AntiPhishStack, operating without prior phishing-specific feature knowledge. Experimental validation on two benchmark datasets, comprising benign and phishing or malicious URLs, demonstrates the model's exceptional performance, achieving a notable 96.04% accuracy compared to existing studies. This research adds value to the ongoing discourse on symmetry and asymmetry in information security and provides a forward-thinking solution for enhancing network security in the face of evolving cyber threats.

{{</citation>}}


## cs.CY (4)



### (87/112) Bringing Social Computing to Secondary School Classrooms (Kianna Bolante et al., 2024)

{{<citation>}}

Kianna Bolante, Kevin Chen, Quan Ze Chen, Amy Zhang. (2024)  
**Bringing Social Computing to Secondary School Classrooms**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.09591v1)  

---


**ABSTRACT**  
Social computing is the study of how technology shapes human social interactions. This topic has become increasingly relevant to secondary school students (ages 11--18) as more of young people's everyday social experiences take place online, particularly with the continuing effects of the COVID-19 pandemic. However, social computing topics are rarely touched upon in existing middle and high school curricula. We seek to introduce concepts from social computing to secondary school students so they can understand how computing has wide-ranging social implications that touch upon their everyday lives, as well as think critically about both the positive and negative sides of different social technology designs.   In this report, we present a series of six lessons combining presentations and hands-on activities covering topics within social computing and detail our experience teaching these lessons to approximately 1,405 students across 13 middle and high schools in our local school district. We developed lessons covering how social computing relates to the topics of Data Management, Encrypted Messaging, Human-Computer Interaction Careers, Machine Learning and Bias, Misinformation, and Online Behavior. We found that 81.13% of students expressed greater interest in the content of our lessons compared to their interest in STEM overall. We also found from pre- and post-lesson comprehension questions that 63.65% learned new concepts from the main activity. We release all lesson materials on a website for public use. From our experience, we observed that students were engaged in these topics and found enjoyment in finding connections between computing and their own lives.

{{</citation>}}


### (88/112) Through the Looking-Glass: Transparency Implications and Challenges in Enterprise AI Knowledge Systems (Karina Cortiñas-Lorenzo et al., 2024)

{{<citation>}}

Karina Cortiñas-Lorenzo, Siân Lindley, Ida Larsen-Ledet, Bhaskar Mitra. (2024)  
**Through the Looking-Glass: Transparency Implications and Challenges in Enterprise AI Knowledge Systems**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09410v1)  

---


**ABSTRACT**  
Knowledge can't be disentangled from people. As AI knowledge systems mine vast volumes of work-related data, the knowledge that's being extracted and surfaced is intrinsically linked to the people who create and use it. When these systems get embedded in organizational settings, the information that is brought to the foreground and the information that's pushed to the periphery can influence how individuals see each other and how they see themselves at work. In this paper, we present the looking-glass metaphor and use it to conceptualize AI knowledge systems as systems that reflect and distort, expanding our view on transparency requirements, implications and challenges. We formulate transparency as a key mediator in shaping different ways of seeing, including seeing into the system, which unveils its capabilities, limitations and behavior, and seeing through the system, which shapes workers' perceptions of their own contributions and others within the organization. Recognizing the sociotechnical nature of these systems, we identify three transparency dimensions necessary to realize the value of AI knowledge systems, namely system transparency, procedural transparency and transparency of outcomes. We discuss key challenges hindering the implementation of these forms of transparency, bringing to light the wider sociotechnical gap and highlighting directions for future Computer-supported Cooperative Work (CSCW) research.

{{</citation>}}


### (89/112) Algorithmic amplification of biases on Google Search (Hussam Habib et al., 2024)

{{<citation>}}

Hussam Habib, Ryan Stoldt, Andrew High, Brian Ekdale, Ashley Peterson, Katy Biddle, Javie Ssozi, Rishab Nithyanand. (2024)  
**Algorithmic amplification of biases on Google Search**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs-IR, cs.CY  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.09044v1)  

---


**ABSTRACT**  
The evolution of information-seeking processes, driven by search engines like Google, has transformed the access to information people have. This paper investigates how individuals' preexisting attitudes influence the modern information-seeking process, specifically the results presented by Google Search. Through a comprehensive study involving surveys and information-seeking tasks focusing on the topic of abortion, the paper provides four crucial insights: 1) Individuals with opposing attitudes on abortion receive different search results. 2) Individuals express their beliefs in their choice of vocabulary used in formulating the search queries, shaping the outcome of the search. 3) Additionally, the user's search history contributes to divergent results among those with opposing attitudes. 4) Google Search engine reinforces preexisting beliefs in search results. Overall, this study provides insights into the interplay between human biases and algorithmic processes, highlighting the potential for information polarization in modern information-seeking processes.

{{</citation>}}


### (90/112) Landscape of Generative AI in Global News: Topics, Sentiments, and Spatiotemporal Analysis (Lu Xian et al., 2024)

{{<citation>}}

Lu Xian, Lingyao Li, Yiwei Xu, Ben Zefeng Zhang, Libby Hemphill. (2024)  
**Landscape of Generative AI in Global News: Topics, Sentiments, and Spatiotemporal Analysis**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, BERT, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.08899v1)  

---


**ABSTRACT**  
Generative AI has exhibited considerable potential to transform various industries and public life. The role of news media coverage of generative AI is pivotal in shaping public perceptions and judgments about this significant technological innovation. This paper provides in-depth analysis and rich insights into the temporal and spatial distribution of topics, sentiment, and substantive themes within global news coverage focusing on the latest emerging technology --generative AI. We collected a comprehensive dataset of news articles (January 2018 to November 2023, N = 24,827). For topic modeling, we employed the BERTopic technique and combined it with qualitative coding to identify semantic themes. Subsequently, sentiment analysis was conducted using the RoBERTa-base model. Analysis of temporal patterns in the data reveals notable variability in coverage across key topics--business, corporate technological development, regulation and security, and education--with spikes in articles coinciding with major AI developments and policy discussions. Sentiment analysis shows a predominantly neutral to positive media stance, with the business-related articles exhibiting more positive sentiment, while regulation and security articles receive a reserved, neutral to negative sentiment. Our study offers a valuable framework to investigate global news discourse and evaluate news attitudes and themes related to emerging technologies.

{{</citation>}}


## cs.IT (2)



### (91/112) Weakly-Private Information Retrieval From MDS-Coded Distributed Storage (Asbjørn O. Orvedal et al., 2024)

{{<citation>}}

Asbjørn O. Orvedal, Hsuan-Yin Lin, Eirik Rosnes. (2024)  
**Weakly-Private Information Retrieval From MDS-Coded Distributed Storage**  

---
Primary Category: cs.IT  
Categories: cs-CR, cs-IT, cs.IT, math-IT  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2401.09412v1)  

---


**ABSTRACT**  
We consider the problem of weakly-private information retrieval (WPIR) when data is encoded by a maximum distance separable code and stored across multiple servers. In WPIR, a user wishes to retrieve a piece of data from a set of servers without leaking too much information about which piece of data she is interested in. We study and provide the first WPIR protocols for this scenario and present results on their optimal trade-off between download rate and information leakage using the maximal leakage privacy metric.

{{</citation>}}


### (92/112) AI Empowered Channel Semantic Acquisition for 6G Integrated Sensing and Communication Networks (Yifei Zhang et al., 2024)

{{<citation>}}

Yifei Zhang, Zhen Gao, Jingjing Zhao, Ziming He, Yunsheng Zhang, Chen Lu, Pei Xiao. (2024)  
**AI Empowered Channel Semantic Acquisition for 6G Integrated Sensing and Communication Networks**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09127v1)  

---


**ABSTRACT**  
Motivated by the need for increased spectral efficiency and the proliferation of intelligent applications, the sixth-generation (6G) mobile network is anticipated to integrate the dual-functions of communication and sensing (C&S). Although the millimeter wave (mmWave) communication and mmWave radar share similar multiple-input multiple-output (MIMO) architecture for integration, the full potential of dual-function synergy remains to be exploited. In this paper, we commence by overviewing state-of-the-art schemes from the aspects of waveform design and signal processing. Nevertheless, these approaches face the dilemma of mutual compromise between C&S performance. To this end, we reveal and exploit the synergy between C&S. In the proposed framework, we introduce a two-stage frame structure and resort artificial intelligence (AI) to achieve the synergistic gain by designing a joint C&S channel semantic extraction and reconstruction network (JCASCasterNet). With just a cost-effective and energy-efficient single sensing antenna, the proposed scheme achieves enhanced overall performance while requiring only limited pilot and feedback signaling overhead. In the end, we outline the challenges that lie ahead in the future development of integrated sensing and communication networks, along with promising directions for further research.

{{</citation>}}


## cs.RO (6)



### (93/112) A Multi-Agent Security Testbed for the Analysis of Attacks and Defenses in Collaborative Sensor Fusion (R. Spencer Hallyburton et al., 2024)

{{<citation>}}

R. Spencer Hallyburton, David Hunt, Shaocheng Luo, Miroslav Pajic. (2024)  
**A Multi-Agent Security Testbed for the Analysis of Attacks and Defenses in Collaborative Sensor Fusion**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.09387v1)  

---


**ABSTRACT**  
The performance and safety of autonomous vehicles (AVs) deteriorates under adverse environments and adversarial actors. The investment in multi-sensor, multi-agent (MSMA) AVs is meant to promote improved efficiency of travel and mitigate safety risks. Unfortunately, minimal investment has been made to develop security-aware MSMA sensor fusion pipelines leaving them vulnerable to adversaries. To advance security analysis of AVs, we develop the Multi-Agent Security Testbed, MAST, in the Robot Operating System (ROS2). Our framework is scalable for general AV scenarios and is integrated with recent multi-agent datasets. We construct the first bridge between AVstack and ROS and develop automated AV pipeline builds to enable rapid AV prototyping. We tackle the challenge of deploying variable numbers of agent/adversary nodes at launch-time with dynamic topic remapping. Using this testbed, we motivate the need for security-aware AV architectures by exposing the vulnerability of centralized multi-agent fusion pipelines to (un)coordinated adversary models in case studies and Monte Carlo analysis.

{{</citation>}}


### (94/112) Vision-driven Autonomous Flight of UAV Along River Using Deep Reinforcement Learning with Dynamic Expert Guidance (Zihan Wang et al., 2024)

{{<citation>}}

Zihan Wang, Jianwen Li, Nina Mahmoudian. (2024)  
**Vision-driven Autonomous Flight of UAV Along River Using Deep Reinforcement Learning with Dynamic Expert Guidance**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09332v1)  

---


**ABSTRACT**  
Vision-driven autonomous flight and obstacle avoidance of Unmanned Aerial Vehicles (UAVs) along complex riverine environments for tasks like rescue and surveillance requires a robust control policy, which is yet difficult to obtain due to the shortage of trainable river environment simulators and reward sparsity in such environments. To easily verify the navigation controller performance for the river following task before real-world deployment, we developed a trainable photo-realistic dynamics-free riverine simulation environment using Unity. Successful river following trajectories in the environment are manually collected and Behavior Clone (BC) is used to train an Imitation Learning (IL) agent to mimic expert behavior and generate expert guidance. Finally, a framework is proposed to train a Deep Reinforcement Learning (DRL) agent using BC expert guidance and improve the expert policy online by sampling good demonstrations produced by the DRL to increase convergence rate and policy performance. This framework is able to solve the along-river autonomous navigation task and outperform baseline RL and IL methods. The code and trainable environments are available.

{{</citation>}}


### (95/112) Deployable Reinforcement Learning with Variable Control Rate (Dong Wang et al., 2024)

{{<citation>}}

Dong Wang, Giovanni Beltrame. (2024)  
**Deployable Reinforcement Learning with Variable Control Rate**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09286v1)  

---


**ABSTRACT**  
Deploying controllers trained with Reinforcement Learning (RL) on real robots can be challenging: RL relies on agents' policies being modeled as Markov Decision Processes (MDPs), which assume an inherently discrete passage of time. The use of MDPs results in that nearly all RL-based control systems employ a fixed-rate control strategy with a period (or time step) typically chosen based on the developer's experience or specific characteristics of the application environment. Unfortunately, the system should be controlled at the highest, worst-case frequency to ensure stability, which can demand significant computational and energy resources and hinder the deployability of the controller on onboard hardware. Adhering to the principles of reactive programming, we surmise that applying control actions only when necessary enables the use of simpler hardware and helps reduce energy consumption. We challenge the fixed frequency assumption by proposing a variant of RL with variable control rate. In this approach, the policy decides the action the agent should take as well as the duration of the time step associated with that action. In our new setting, we expand Soft Actor-Critic (SAC) to compute the optimal policy with a variable control rate, introducing the Soft Elastic Actor-Critic (SEAC) algorithm. We show the efficacy of SEAC through a proof-of-concept simulation driving an agent with Newtonian kinematics. Our experiments show higher average returns, shorter task completion times, and reduced computational resources when compared to fixed rate policies.

{{</citation>}}


### (96/112) An Efficient Generalizable Framework for Visuomotor Policies via Control-aware Augmentation and Privilege-guided Distillation (Yinuo Zhao et al., 2024)

{{<citation>}}

Yinuo Zhao, Kun Wu, Tianjiao Yi, Zhiyuan Xu, Xiaozhu Ju, Zhengping Che, Qinru Qiu, Chi Harold Liu, Jian Tang. (2024)  
**An Efficient Generalizable Framework for Visuomotor Policies via Control-aware Augmentation and Privilege-guided Distillation**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.09258v1)  

---


**ABSTRACT**  
Visuomotor policies, which learn control mechanisms directly from high-dimensional visual observations, confront challenges in adapting to new environments with intricate visual variations. Data augmentation emerges as a promising method for bridging these generalization gaps by enriching data variety. However, straightforwardly augmenting the entire observation shall impose excessive burdens on policy learning and may even result in performance degradation. In this paper, we propose to improve the generalization ability of visuomotor policies as well as preserve training stability from two aspects: 1) We learn a control-aware mask through a self-supervised reconstruction task with three auxiliary losses and then apply strong augmentation only to those control-irrelevant regions based on the mask to reduce the generalization gaps. 2) To address training instability issues prevalent in visual reinforcement learning (RL), we distill the knowledge from a pretrained RL expert processing low-level environment states, to the student visuomotor policy. The policy is subsequently deployed to unseen environments without any further finetuning. We conducted comparison and ablation studies across various benchmarks: the DMControl Generalization Benchmark (DMC-GB), the enhanced Robot Manipulation Distraction Benchmark (RMDB), and a specialized long-horizontal drawer-opening robotic task. The extensive experimental results well demonstrate the effectiveness of our method, e.g., showing a 17\% improvement over previous methods in the video-hard setting of DMC-GB.

{{</citation>}}


### (97/112) Biased-MPPI: Informing Sampling-Based Model Predictive Control by Fusing Ancillary Controllers (Elia Trevisan et al., 2024)

{{<citation>}}

Elia Trevisan, Javier Alonso-Mora. (2024)  
**Biased-MPPI: Informing Sampling-Based Model Predictive Control by Fusing Ancillary Controllers**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.09241v1)  

---


**ABSTRACT**  
Motion planning for autonomous robots in human-populated environments poses numerous challenges due to uncertainties in the robot's dynamics, environment, and interaction with other agents. Sampling-based MPC approaches, such as Model Predictive Path Integral (MPPI) control, have shown promise in addressing these complex motion planning problems. However, the performance of MPPI heavily relies on the choice of the sampling distribution. Existing literature often uses the previously computed input sequence as the mean of a Gaussian distribution for sampling, leading to potential failures and local minima. In this paper, we propose novel derivations of the MPPI method to enhance its efficiency, robustness, and convergence. Our approach includes a mathematical formulation allowing for arbitrary sampling distributions, addressing numerical issues, and alleviating the problem of local minima. We present an efficient importance sampling scheme that combines classical and learning-based ancillary controllers simultaneously, resulting in more informative sampling and control fusion. We demonstrate our proposed scheme's superior efficiency and robustness through experiments by handling model uncertainties, rapid environmental changes and reducing susceptibility to local minima.

{{</citation>}}


### (98/112) SWBT: Similarity Weighted Behavior Transformer with the Imperfect Demonstration for Robotic Manipulation (Kun Wu et al., 2024)

{{<citation>}}

Kun Wu, Ning Liu, Zhen Zhao, Di Qiu, Jinming Li, Zhengping Che, Zhiyuan Xu, Qinru Qiu, Jian Tang. (2024)  
**SWBT: Similarity Weighted Behavior Transformer with the Imperfect Demonstration for Robotic Manipulation**  

---
Primary Category: cs.RO  
Categories: I-2-9, cs-AI, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.08957v1)  

---


**ABSTRACT**  
Imitation learning (IL), aiming to learn optimal control policies from expert demonstrations, has been an effective method for robot manipulation tasks. However, previous IL methods either only use expensive expert demonstrations and omit imperfect demonstrations or rely on interacting with the environment and learning from online experiences. In the context of robotic manipulation, we aim to conquer the above two challenges and propose a novel framework named Similarity Weighted Behavior Transformer (SWBT). SWBT effectively learn from both expert and imperfect demonstrations without interaction with environments. We reveal that the easy-to-get imperfect demonstrations, such as forward and inverse dynamics, significantly enhance the network by learning fruitful information. To the best of our knowledge, we are the first to attempt to integrate imperfect demonstrations into the offline imitation learning setting for robot manipulation tasks. Extensive experiments on the ManiSkill2 benchmark built on the high-fidelity Sapien simulator and real-world robotic manipulation tasks demonstrated that the proposed method can extract better features and improve the success rates for all tasks. Our code will be released upon acceptance of the paper.

{{</citation>}}


## cs.DC (2)



### (99/112) Swing: Short-cutting Rings for Higher Bandwidth Allreduce (Daniele De Sensi et al., 2024)

{{<citation>}}

Daniele De Sensi, Tommaso Bonato, David Saam, Torsten Hoefler. (2024)  
**Swing: Short-cutting Rings for Higher Bandwidth Allreduce**  

---
Primary Category: cs.DC  
Categories: C-2-4; C-2-2, cs-DC, cs-LG, cs-NI, cs-PF, cs.DC  
Keywords: Amazon, Google  
[Paper Link](http://arxiv.org/abs/2401.09356v1)  

---


**ABSTRACT**  
The allreduce collective operation accounts for a significant fraction of the runtime of workloads running on distributed systems. One factor determining its performance is the distance between communicating nodes, especially on networks like torus, where a higher distance implies multiple messages being forwarded on the same link, thus reducing the allreduce bandwidth. Torus networks are widely used on systems optimized for machine learning workloads (e.g., Google TPUs and Amazon Trainium devices), as well as on some of the Top500 supercomputers. To improve allreduce performance on torus networks we introduce Swing, a new algorithm that keeps a low distance between communicating nodes by swinging between torus directions. Our analysis and experimental evaluation show that Swing outperforms by up to 3x existing allreduce algorithms for vectors ranging from 32B to 128MiB, on different types of torus and torus-like topologies, regardless of their shape and size.

{{</citation>}}


### (100/112) InternEvo: Efficient Long-sequence Large Language Model Training via Hybrid Parallelism and Redundant Sharding (Qiaoling Chen et al., 2024)

{{<citation>}}

Qiaoling Chen, Diandian Gu, Guoteng Wang, Xun Chen, YingTong Xiong, Ting Huang, Qinghao Hu, Xin Jin, Yonggang Wen, Tianwei Zhang, Peng Sun. (2024)  
**InternEvo: Efficient Long-sequence Large Language Model Training via Hybrid Parallelism and Redundant Sharding**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09149v1)  

---


**ABSTRACT**  
Large language models (LLMs) with long sequences begin to power more and more fundamentally new applications we use every day. Existing methods for long-sequence LLM training are neither efficient nor compatible with commonly-used training algorithms such as FlashAttention. We design Buff to address these issues. Buff decouples all of the sharding dimensions into a new hierarchical space, and systematically analyzes the memory and communication cost of LLM training. Then, it generates an effective hybrid parallelism strategy. We design a new selective overlap mechanism to mitigate the communication overhead introduced by the hybrid parallelism. We also implement memory management techniques to reduce GPU memory fragmentation. Evaluation results show that Buff generates parallelization strategies that match or outperform existing methods in model FLOPs utilization.

{{</citation>}}


## cs.SD (3)



### (101/112) MLAAD: The Multi-Language Audio Anti-Spoofing Dataset (Nicolas M. Müller et al., 2024)

{{<citation>}}

Nicolas M. Müller, Piotr Kawa, Wei Herng Choong, Edresson Casanova, Eren Gölge, Thorsten Müller, Piotr Syga, Philip Sperl, Konstantin Böttinger. (2024)  
**MLAAD: The Multi-Language Audio Anti-Spoofing Dataset**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09512v1)  

---


**ABSTRACT**  
Text-to-Speech (TTS) technology brings significant advantages, such as giving a voice to those with speech impairments, but also enables audio deepfakes and spoofs. The former mislead individuals and may propagate misinformation, while the latter undermine voice biometric security systems. AI-based detection can help to address these challenges by automatically differentiating between genuine and fabricated voice recordings. However, these models are only as good as their training data, which currently is severely limited due to an overwhelming concentration on English and Chinese audio in anti-spoofing databases, thus restricting its worldwide effectiveness. In response, this paper presents the Multi-Language Audio Anti-Spoof Dataset (MLAAD), created using 52 TTS models, comprising 19 different architectures, to generate 160.1 hours of synthetic voice in 23 different languages. We train and evaluate three state-of-the-art deepfake detection models with MLAAD, and observe that MLAAD demonstrates superior performance over comparable datasets like InTheWild or FakeOrReal when used as a training resource. Furthermore, in comparison with the renowned ASVspoof 2019 dataset, MLAAD proves to be a complementary resource. In tests across eight datasets, MLAAD and ASVspoof 2019 alternately outperformed each other, both excelling on four datasets. By publishing MLAAD and making trained models accessible via an interactive webserver , we aim to democratize antispoofing technology, making it accessible beyond the realm of specialists, thus contributing to global efforts against audio spoofing and deepfakes.

{{</citation>}}


### (102/112) Similar but Faster: Manipulation of Tempo in Music Audio Embeddings for Tempo Prediction and Search (Matthew C. McCallum et al., 2024)

{{<citation>}}

Matthew C. McCallum, Florian Henkel, Jaehun Kim, Samuel E. Sandberg, Matthew E. P. Davies. (2024)  
**Similar but Faster: Manipulation of Tempo in Music Audio Embeddings for Tempo Prediction and Search**  

---
Primary Category: cs.SD  
Categories: cs-DL, cs-IR, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.08902v1)  

---


**ABSTRACT**  
Audio embeddings enable large scale comparisons of the similarity of audio files for applications such as search and recommendation. Due to the subjectivity of audio similarity, it can be desirable to design systems that answer not only whether audio is similar, but similar in what way (e.g., wrt. tempo, mood or genre). Previous works have proposed disentangled embedding spaces where subspaces representing specific, yet possibly correlated, attributes can be weighted to emphasize those attributes in downstream tasks. However, no research has been conducted into the independence of these subspaces, nor their manipulation, in order to retrieve tracks that are similar but different in a specific way. Here, we explore the manipulation of tempo in embedding spaces as a case-study towards this goal. We propose tempo translation functions that allow for efficient manipulation of tempo within a pre-existing embedding space whilst maintaining other properties such as genre. As this translation is specific to tempo it enables retrieval of tracks that are similar but have specifically different tempi. We show that such a function can be used as an efficient data augmentation strategy for both training of downstream tempo predictors, and improved nearest neighbor retrieval of properties largely independent of tempo.

{{</citation>}}


### (103/112) On the Effect of Data-Augmentation on Local Embedding Properties in the Contrastive Learning of Music Audio Representations (Matthew C. McCallum et al., 2024)

{{<citation>}}

Matthew C. McCallum, Matthew E. P. Davies, Florian Henkel, Jaehun Kim, Samuel E. Sandberg. (2024)  
**On the Effect of Data-Augmentation on Local Embedding Properties in the Contrastive Learning of Music Audio Representations**  

---
Primary Category: cs.SD  
Categories: cs-IR, cs-LG, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Augmentation, Contrastive Learning, Embedding  
[Paper Link](http://arxiv.org/abs/2401.08889v1)  

---


**ABSTRACT**  
Audio embeddings are crucial tools in understanding large catalogs of music. Typically embeddings are evaluated on the basis of the performance they provide in a wide range of downstream tasks, however few studies have investigated the local properties of the embedding spaces themselves which are important in nearest neighbor algorithms, commonly used in music search and recommendation. In this work we show that when learning audio representations on music datasets via contrastive learning, musical properties that are typically homogeneous within a track (e.g., key and tempo) are reflected in the locality of neighborhoods in the resulting embedding space. By applying appropriate data augmentation strategies, localisation of such properties can not only be reduced but the localisation of other attributes is increased. For example, locality of features such as pitch and tempo that are less relevant to non-expert listeners, may be mitigated while improving the locality of more salient features such as genre and mood, achieving state-of-the-art performance in nearest neighbor retrieval accuracy. Similarly, we show that the optimal selection of data augmentation strategies for contrastive learning of music audio embeddings is dependent on the downstream task, highlighting this as an important embedding design decision.

{{</citation>}}


## cs.NE (1)



### (104/112) Information flow and Laplacian dynamics on local optima networks (Hendrik Richter et al., 2024)

{{<citation>}}

Hendrik Richter, Sarah L. Thomson. (2024)  
**Information flow and Laplacian dynamics on local optima networks**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.09229v1)  

---


**ABSTRACT**  
We propose a new way of looking at local optima networks (LONs). LONs represent fitness landscapes; the nodes are local optima, and the edges are search transitions between them. Many metrics computed on LONs have been proposed and shown to be linked to metaheuristic search difficulty. These have typically considered LONs as describing static structures. In contrast to this, Laplacian dynamics (LD) is an approach to consider the information flow across a network as a dynamical process. We adapt and apply LD to the context of LONs. As a testbed, we consider instances from the quadratic assignment problem (QAP) library. Metrics related to LD are proposed and these are compared with existing LON metrics. The results show that certain LD metrics are strong predictors of metaheuristic performance for iterated local search and tabu search.

{{</citation>}}


## cs.NI (1)



### (105/112) The Mikado Filesystem: An experimental RPC filesystem running over gRPC (John D. Dougrez-Lewis, 2024)

{{<citation>}}

John D. Dougrez-Lewis. (2024)  
**The Mikado Filesystem: An experimental RPC filesystem running over gRPC**  

---
Primary Category: cs.NI  
Categories: H-3-2; H-3-3; H-3-5; D-4-3, cs-NI, cs-SI, cs.NI  
Keywords: Google, Security  
[Paper Link](http://arxiv.org/abs/2401.09186v1)  

---


**ABSTRACT**  
Computer applications seeking to persist files remotely across the Internet are faced with a bewildering choice of mechanisms which tend to boil down to monolithic proprietary closed-source Vendor solutions. We introduce The Mikado Filesystem (mikfs), which provides an open simple lightweight interoperable portable extensible remote filesystem that is open source. mikfs consists of client applications accessing remote servers via RPC running over TCP/IP connections. mikfs is defined as a concrete set of API method calls over gRPC expressed in Google's Protocol Buffers' IDL. gRPC supports a wide variety of programming languages & platforms. For a given language + platform, the gRPC toolset can generate client- & server-side stubs from the IDL callable from client & server code in the selected languages, e.g., a client written in C# or java running on a Windows PC can access a server written in C++ running on Linux. mikfs consists of a virtual hierarchical tree of files & directories. This logical filesystem is not constrained to the limits and file naming conventions of the host's own physical native filesystem. API methods are provided for authentication; for atomic file-level operations on files & directories; for clients to register to receive notifications of file & directory changes on a server. The public API allows developers to write their own new servers and clients; allowing migration of hosted files between different implementations; extension with new methods & features; is Open Source code available for inspection and adaptation. gRPC provides secure authenticated connection & communication over HTTP/2; End-to-End Privacy & Security against eavesdropping of data in transit; support for multiple alternate user login mechanisms. mikfs is provided as source code, 'The Bootstrap Distribution', consisting of an ecosystem of clients, servers, tools and utilities.

{{</citation>}}


## cs.AI (3)



### (106/112) Knowledge Pyramid: A Novel Hierarchical Reasoning Structure for Generalized Knowledge Augmentation and Inference (Qinghua Huang et al., 2024)

{{<citation>}}

Qinghua Huang, Yongzhen Wang. (2024)  
**Knowledge Pyramid: A Novel Hierarchical Reasoning Structure for Generalized Knowledge Augmentation and Inference**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IR, cs.AI  
Keywords: Augmentation, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.09070v1)  

---


**ABSTRACT**  
Knowledge graph (KG) based reasoning has been regarded as an effective means for the analysis of semantic networks and is of great usefulness in areas of information retrieval, recommendation, decision-making, and man-machine interaction. It is widely used in recommendation, decision-making, question-answering, search, and other fields. However, previous studies mainly used low-level knowledge in the KG for reasoning, which may result in insufficient generalization and poor robustness of reasoning. To this end, this paper proposes a new inference approach using a novel knowledge augmentation strategy to improve the generalization capability of KG. This framework extracts high-level pyramidal knowledge from low-level knowledge and applies it to reasoning in a multi-level hierarchical KG, called knowledge pyramid in this paper. We tested some medical data sets using the proposed approach, and the experimental results show that the proposed knowledge pyramid has improved the knowledge inference performance with better generalization. Especially, when there are fewer training samples, the inference accuracy can be significantly improved.

{{</citation>}}


### (107/112) LLMs for Relational Reasoning: How Far are We? (Zhiming Li et al., 2024)

{{<citation>}}

Zhiming Li, Yushi Cao, Xiufeng Xu, Junzhe Jiang, Xu Liu, Yon Shin Teo, Shang-wei Lin, Yang Liu. (2024)  
**LLMs for Relational Reasoning: How Far are We?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.09042v1)  

---


**ABSTRACT**  
Large language models (LLMs) have revolutionized many areas (e.g. natural language processing, software engineering, etc.) by achieving state-of-the-art performance on extensive downstream tasks. Aiming to achieve robust and general artificial intelligence, there has been a surge of interest in investigating the reasoning ability of the LLMs. Whereas the textual and numerical reasoning benchmarks adopted by previous works are rather shallow and simple, it is hard to conclude that the LLMs possess strong reasoning ability by merely achieving positive results on these benchmarks. Recent efforts have demonstrated that the LLMs are poor at solving sequential decision-making problems that require common-sense planning by evaluating their performance on the reinforcement learning benchmarks. In this work, we conduct an in-depth assessment of several state-of-the-art LLMs' reasoning ability based on the inductive logic programming (ILP) benchmark, which is broadly recognized as a representative and challenging measurement for evaluating logic program induction/synthesis systems as it requires inducing strict cause-effect logic to achieve robust deduction on independent and identically distributed (IID) and out-of-distribution (OOD) test samples. Our evaluations illustrate that compared with the neural program induction systems which are much smaller in model size, the state-of-the-art LLMs are much poorer in terms of reasoning ability by achieving much lower performance and generalization using either natural language prompting or truth-value matrix prompting.

{{</citation>}}


### (108/112) Continuous Time Continuous Space Homeostatic Reinforcement Learning (CTCS-HRRL) : Towards Biological Self-Autonomous Agent (Hugo Laurencon et al., 2024)

{{<citation>}}

Hugo Laurencon, Yesoda Bhargava, Riddhi Zantye, Charbel-Raphaël Ségerie, Johann Lussange, Veeky Baths, Boris Gutkin. (2024)  
**Continuous Time Continuous Space Homeostatic Reinforcement Learning (CTCS-HRRL) : Towards Biological Self-Autonomous Agent**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.08999v1)  

---


**ABSTRACT**  
Homeostasis is a biological process by which living beings maintain their internal balance. Previous research suggests that homeostasis is a learned behaviour. Recently introduced Homeostatic Regulated Reinforcement Learning (HRRL) framework attempts to explain this learned homeostatic behavior by linking Drive Reduction Theory and Reinforcement Learning. This linkage has been proven in the discrete time-space, but not in the continuous time-space. In this work, we advance the HRRL framework to a continuous time-space environment and validate the CTCS-HRRL (Continuous Time Continuous Space HRRL) framework. We achieve this by designing a model that mimics the homeostatic mechanisms in a real-world biological agent. This model uses the Hamilton-Jacobian Bellman Equation, and function approximation based on neural networks and Reinforcement Learning. Through a simulation-based experiment we demonstrate the efficacy of this model and uncover the evidence linked to the agent's ability to dynamically choose policies that favor homeostasis in a continuously changing internal-state milieu. Results of our experiments demonstrate that agent learns homeostatic behaviour in a CTCS environment, making CTCS-HRRL a promising framework for modellng animal dynamics and decision-making.

{{</citation>}}


## eess.AS (1)



### (109/112) Two-pass Endpoint Detection for Speech Recognition (Anirudh Raju et al., 2024)

{{<citation>}}

Anirudh Raju, Aparna Khare, Di He, Ilya Sklyar, Long Chen, Sam Alptekin, Viet Anh Trinh, Zhe Zhang, Colin Vaz, Venkatesh Ravichandran, Roland Maas, Ariya Rastrow. (2024)  
**Two-pass Endpoint Detection for Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.08916v1)  

---


**ABSTRACT**  
Endpoint (EP) detection is a key component of far-field speech recognition systems that assist the user through voice commands. The endpoint detector has to trade-off between accuracy and latency, since waiting longer reduces the cases of users being cut-off early. We propose a novel two-pass solution for endpointing, where the utterance endpoint detected from a first pass endpointer is verified by a 2nd-pass model termed EP Arbitrator. Our method improves the trade-off between early cut-offs and latency over a baseline endpointer, as tested on datasets including voice-assistant transactional queries, conversational speech, and the public SLURP corpus. We demonstrate that our method shows improvements regardless of the first-pass EP model used.

{{</citation>}}


## stat.AP (1)



### (110/112) How do transportation professionals perceive the impacts of AI applications in transportation? A latent class cluster analysis (Yiheng Qian et al., 2024)

{{<citation>}}

Yiheng Qian, Tejaswi Polimetla, Thomas W. Sanchez, Xiang Yan. (2024)  
**How do transportation professionals perceive the impacts of AI applications in transportation? A latent class cluster analysis**  

---
Primary Category: stat.AP  
Categories: cs-CY, stat-AP, stat.AP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08915v1)  

---


**ABSTRACT**  
Recent years have witnessed an increasing number of artificial intelligence (AI) applications in transportation. As a new and emerging technology, AI's potential to advance transportation goals and the full extent of its impacts on the transportation sector is not yet well understood. As the transportation community explores these topics, it is critical to understand how transportation professionals, the driving force behind AI Transportation applications, perceive AI's potential efficiency and equity impacts. Toward this goal, we surveyed transportation professionals in the United States and collected a total of 354 responses. Based on the survey responses, we conducted both descriptive analysis and latent class cluster analysis (LCCA). The former provides an overview of prevalent attitudes among transportation professionals, while the latter allows the identification of distinct segments based on their latent attitudes toward AI. We find widespread optimism regarding AI's potential to improve many aspects of transportation (e.g., efficiency, cost reduction, and traveler experience); however, responses are mixed regarding AI's potential to advance equity. Moreover, many respondents are concerned that AI ethics are not well understood in the transportation community and that AI use in transportation could exaggerate existing inequalities. Through LCCA, we have identified four latent segments: AI Neutral, AI Optimist, AI Pessimist, and AI Skeptic. The latent class membership is significantly associated with respondents' age, education level, and AI knowledge level. Overall, the study results shed light on the extent to which the transportation community as a whole is ready to leverage AI systems to transform current practices and inform targeted education to improve the understanding of AI among transportation professionals.

{{</citation>}}


## cs.AR (1)



### (111/112) VeriBug: An Attention-based Framework for Bug-Localization in Hardware Designs (Giuseppe Stracquadanio et al., 2024)

{{<citation>}}

Giuseppe Stracquadanio, Sourav Medya, Stefano Quer, Debjit Pal. (2024)  
**VeriBug: An Attention-based Framework for Bug-Localization in Hardware Designs**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-LG, cs.AR  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.09494v1)  

---


**ABSTRACT**  
In recent years, there has been an exponential growth in the size and complexity of System-on-Chip designs targeting different specialized applications. The cost of an undetected bug in these systems is much higher than in traditional processor systems as it may imply the loss of property or life. The problem is further exacerbated by the ever-shrinking time-to-market and ever-increasing demand to churn out billions of devices. Despite decades of research in simulation and formal methods for debugging and verification, it is still one of the most time-consuming and resource intensive processes in contemporary hardware design cycle. In this work, we propose VeriBug, which leverages recent advances in deep learning to accelerate debugging at the Register-Transfer Level and generates explanations of likely root causes. First, VeriBug uses control-data flow graph of a hardware design and learns to execute design statements by analyzing the context of operands and their assignments. Then, it assigns an importance score to each operand in a design statement and uses that score for generating explanations for failures. Finally, VeriBug produces a heatmap highlighting potential buggy source code portions. Our experiments show that VeriBug can achieve an average bug localization coverage of 82.5% on open-source designs and different types of injected bugs.

{{</citation>}}


## cs.OS (1)



### (112/112) Herding LLaMaS: Using LLMs as an OS Module (Aditya K Kamath et al., 2024)

{{<citation>}}

Aditya K Kamath, Sujay Yadalam. (2024)  
**Herding LLaMaS: Using LLMs as an OS Module**  

---
Primary Category: cs.OS  
Categories: cs-LG, cs-OS, cs.OS  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08908v1)  

---


**ABSTRACT**  
Computer systems are becoming increasingly heterogeneous with the emergence of new memory technologies and compute devices. GPUs alongside CPUs have become commonplace and CXL is poised to be a mainstay of cloud systems. The operating system is responsible for managing these hardware resources, requiring modification every time a new device is released. Years of research and development are sunk into tuning the OS for high performance with each new heterogeneous device. With the recent explosion in memory technologies and domain-specific accelerators, it would be beneficial to have an OS that could provide high performance for new devices without significant effort.   We propose LLaMaS which can adapt to new devices easily. LLaMaS uses Large Language Models (LLMs) to extract the useful features of new devices from their textual description and uses these features to make operating system decisions at runtime. Adding support to LLaMaS for a new device is as simple as describing the system and new device properties in plaintext.   LLaMaS reduces the burden on system administrators to enable easy integration of new devices into production systems.   Preliminary evaluation using ChatGPT shows that LLMs are capable of extracting device features from text and make correct OS decisions based on those features.

{{</citation>}}
