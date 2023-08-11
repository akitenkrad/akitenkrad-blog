---
draft: false
title: "arXiv @ 2023.08.08"
date: 2023-08-08
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.08"
    identifier: arxiv_20230808
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [eess.AS (1)](#eessas-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.CL (11)](#cscl-11)
- [cs.LG (6)](#cslg-6)
- [eess.IV (2)](#eessiv-2)
- [cs.AI (6)](#csai-6)
- [cs.CV (15)](#cscv-15)
- [physics.ao-ph (1)](#physicsao-ph-1)
- [cs.DC (1)](#csdc-1)
- [cs.SE (1)](#csse-1)
- [cs.IT (1)](#csit-1)
- [cs.HC (1)](#cshc-1)
- [cs.NI (1)](#csni-1)
- [cs.IR (1)](#csir-1)
- [cs.CR (1)](#cscr-1)

## eess.AS (1)



### (1/50) Investigation of Self-supervised Pre-trained Models for Classification of Voice Quality from Speech and Neck Surface Accelerometer Signals (Sudarsana Reddy Kadiri et al., 2023)

{{<citation>}}

Sudarsana Reddy Kadiri, Farhad Javanmardi, Paavo Alku. (2023)  
**Investigation of Self-supervised Pre-trained Models for Classification of Voice Quality from Speech and Neck Surface Accelerometer Signals**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CL, cs-MM, cs-SD, eess-AS, eess.AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.03226v1)  

---


**ABSTRACT**  
Prior studies in the automatic classification of voice quality have mainly studied the use of the acoustic speech signal as input. Recently, a few studies have been carried out by jointly using both speech and neck surface accelerometer (NSA) signals as inputs, and by extracting MFCCs and glottal source features. This study examines simultaneously-recorded speech and NSA signals in the classification of voice quality (breathy, modal, and pressed) using features derived from three self-supervised pre-trained models (wav2vec2-BASE, wav2vec2-LARGE, and HuBERT) and using a SVM as well as CNNs as classifiers. Furthermore, the effectiveness of the pre-trained models is compared in feature extraction between glottal source waveforms and raw signal waveforms for both speech and NSA inputs. Using two signal processing methods (quasi-closed phase (QCP) glottal inverse filtering and zero frequency filtering (ZFF)), glottal source waveforms are estimated from both speech and NSA signals. The study has three main goals: (1) to study whether features derived from pre-trained models improve classification accuracy compared to conventional features (spectrogram, mel-spectrogram, MFCCs, i-vector, and x-vector), (2) to investigate which of the two modalities (speech vs. NSA) is more effective in the classification task with pre-trained model-based features, and (3) to evaluate whether the deep learning-based CNN classifier can enhance the classification accuracy in comparison to the SVM classifier. The results revealed that the use of the NSA input showed better classification performance compared to the speech signal. Between the features, the pre-trained model-based features showed better classification accuracies, both for speech and NSA inputs compared to the conventional features. It was also found that the HuBERT features performed better than the wav2vec2-BASE and wav2vec2-LARGE features.

{{</citation>}}


## quant-ph (1)



### (2/50) Enabling High Performance Debugging for Variational Quantum Algorithms using Compressed Sensing (Kun Liu et al., 2023)

{{<citation>}}

Kun Liu, Tianyi Hao, Swamit Tannu. (2023)  
**Enabling High Performance Debugging for Variational Quantum Algorithms using Compressed Sensing**  

---
Primary Category: quant-ph  
Categories: cs-AR, cs-ET, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.03213v1)  

---


**ABSTRACT**  
Variational quantum algorithms (VQAs) can potentially solve practical problems using contemporary Noisy Intermediate Scale Quantum (NISQ) computers. VQAs find near-optimal solutions in the presence of qubit errors by classically optimizing a loss function computed by parameterized quantum circuits. However, developing and testing VQAs is challenging due to the limited availability of quantum hardware, their high error rates, and the significant overhead of classical simulations. Furthermore, VQA researchers must pick the right initialization for circuit parameters, utilize suitable classical optimizer configurations, and deploy appropriate error mitigation methods. Unfortunately, these tasks are done in an ad-hoc manner today, as there are no software tools to configure and tune the VQA hyperparameters.   In this paper, we present OSCAR (cOmpressed Sensing based Cost lAndscape Reconstruction) to help configure: 1) correct initialization, 2) noise mitigation techniques, and 3) classical optimizers to maximize the quality of the solution on NISQ hardware. OSCAR enables efficient debugging and performance tuning by providing users with the loss function landscape without running thousands of quantum circuits as required by the grid search. Using OSCAR, we can accurately reconstruct the complete cost landscape with up to 100X speedup. Furthermore, OSCAR can compute an optimizer function query in an instant by interpolating a computed landscape, thus enabling the trial run of a VQA configuration with considerably reduced overhead.

{{</citation>}}


## cs.CL (11)



### (3/50) Average-Hard Attention Transformers are Constant-Depth Uniform Threshold Circuits (Lena Strobl, 2023)

{{<citation>}}

Lena Strobl. (2023)  
**Average-Hard Attention Transformers are Constant-Depth Uniform Threshold Circuits**  

---
Primary Category: cs.CL  
Categories: cs-CC, cs-CL, cs-LG, cs.CL  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.03212v1)  

---


**ABSTRACT**  
Transformers have emerged as a widely used neural network model for various natural language processing tasks. Previous research explored their relationship with constant-depth threshold circuits, making two assumptions: average-hard attention and logarithmic precision for internal computations relative to input length. Merrill et al. (2022) prove that average-hard attention transformers recognize languages that fall within the complexity class TC0, denoting the set of languages that can be recognized by constant-depth polynomial-size threshold circuits. Likewise, Merrill and Sabharwal (2023) show that log-precision transformers recognize languages within the class of uniform TC0. This shows that both transformer models can be simulated by constant-depth threshold circuits, with the latter being more robust due to generating a uniform circuit family. Our paper shows that the first result can be extended to yield uniform circuits as well.

{{</citation>}}


### (4/50) Automatically Correcting Large Language Models: Surveying the landscape of diverse self-correction strategies (Liangming Pan et al., 2023)

{{<citation>}}

Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, William Yang Wang. (2023)  
**Automatically Correcting Large Language Models: Surveying the landscape of diverse self-correction strategies**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.03188v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated remarkable performance across a wide array of NLP tasks. However, their efficacy is undermined by undesired and inconsistent behaviors, including hallucination, unfaithful reasoning, and toxic content. A promising approach to rectify these flaws is self-correction, where the LLM itself is prompted or guided to fix problems in its own output. Techniques leveraging automated feedback -- either produced by the LLM itself or some external system -- are of particular interest as they are a promising way to make LLM-based solutions more practical and deployable with minimal human feedback. This paper presents a comprehensive review of this emerging class of techniques. We analyze and taxonomize a wide array of recent work utilizing these strategies, including training-time, generation-time, and post-hoc correction. We also summarize the major applications of this strategy and conclude by discussing future directions and challenges.

{{</citation>}}


### (5/50) Towards Multiple References Era -- Addressing Data Leakage and Limited Reference Diversity in NLG Evaluation (Xianfeng Zeng et al., 2023)

{{<citation>}}

Xianfeng Zeng, Yijin Liu, Fandong Meng, Jie Zhou. (2023)  
**Towards Multiple References Era -- Addressing Data Leakage and Limited Reference Diversity in NLG Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, BLEU  
[Paper Link](http://arxiv.org/abs/2308.03131v4)  

---


**ABSTRACT**  
N-gram matching-based evaluation metrics, such as BLEU and chrF, are widely utilized across a range of natural language generation (NLG) tasks. However, recent studies have revealed a weak correlation between these matching-based metrics and human evaluations, especially when compared with neural-based metrics like BLEURT. In this paper, we conjecture that the performance bottleneck in matching-based metrics may be caused by the limited diversity of references. To address this issue, we propose to utilize \textit{multiple references} to enhance the consistency between these metrics and human evaluations. Within the WMT Metrics benchmarks, we observe that the multi-references F200spBLEU surpasses the conventional single-reference one by an accuracy improvement of 7.2\%. Remarkably, it also exceeds the neural-based BERTscore by an accuracy enhancement of 3.9\%. Moreover, we observe that the data leakage issue in large language models (LLMs) can be mitigated to a large extent by our multi-reference metric. We release the code and data at \url{https://github.com/SefaZeng/LLM-Ref}

{{</citation>}}


### (6/50) 'Kurosawa': A Script Writer's Assistant (Prerak Gandhi et al., 2023)

{{<citation>}}

Prerak Gandhi, Vishal Pramanik, Pushpak Bhattacharyya. (2023)  
**'Kurosawa': A Script Writer's Assistant**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2308.03122v1)  

---


**ABSTRACT**  
Storytelling is the lifeline of the entertainment industry -- movies, TV shows, and stand-up comedies, all need stories. A good and gripping script is the lifeline of storytelling and demands creativity and resource investment. Good scriptwriters are rare to find and often work under severe time pressure. Consequently, entertainment media are actively looking for automation. In this paper, we present an AI-based script-writing workbench called KUROSAWA which addresses the tasks of plot generation and script generation. Plot generation aims to generate a coherent and creative plot (600-800 words) given a prompt (15-40 words). Script generation, on the other hand, generates a scene (200-500 words) in a screenplay format from a brief description (15-40 words). Kurosawa needs data to train. We use a 4-act structure of storytelling to annotate the plot dataset manually. We create a dataset of 1000 manually annotated plots and their corresponding prompts/storylines and a gold-standard dataset of 1000 scenes with four main elements -- scene headings, action lines, dialogues, and character names -- tagged individually. We fine-tune GPT-3 with the above datasets to generate plots and scenes. These plots and scenes are first evaluated and then used by the scriptwriters of a large and famous media platform ErosNow. We release the annotated datasets and the models trained on these datasets as a working benchmark for automatic movie plot and script generation.

{{</citation>}}


### (7/50) PromptSum: Parameter-Efficient Controllable Abstractive Summarization (Mathieu Ravaut et al., 2023)

{{<citation>}}

Mathieu Ravaut, Hailin Chen, Ruochen Zhao, Chengwei Qin, Shafiq Joty, Nancy Chen. (2023)  
**PromptSum: Parameter-Efficient Controllable Abstractive Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2308.03117v1)  

---


**ABSTRACT**  
Prompt tuning (PT), a parameter-efficient technique that only tunes the additional prompt embeddings while keeping the backbone pre-trained language model (PLM) frozen, has shown promising results in language understanding tasks, especially in low-resource scenarios. However, effective prompt design methods suitable for generation tasks such as summarization are still lacking. At the same time, summarization guided through instructions (discrete prompts) can achieve a desirable double objective of high quality and controllability in summary generation. Towards a goal of strong summarization performance under the triple conditions of parameter-efficiency, data-efficiency, and controllability, we introduce PromptSum, a method combining PT with a multi-task objective and discrete entity prompts for abstractive summarization. Our model achieves competitive ROUGE results on popular abstractive summarization benchmarks coupled with a strong level of controllability through entities, all while only tuning several orders of magnitude less parameters.

{{</citation>}}


### (8/50) Improving Domain-Specific Retrieval by NLI Fine-Tuning (Roman Dušek et al., 2023)

{{<citation>}}

Roman Dušek, Aleksander Wawer, Christopher Galias, Lidia Wojciechowska. (2023)  
**Improving Domain-Specific Retrieval by NLI Fine-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: NLI  
[Paper Link](http://arxiv.org/abs/2308.03103v1)  

---


**ABSTRACT**  
The aim of this article is to investigate the fine-tuning potential of natural language inference (NLI) data to improve information retrieval and ranking. We demonstrate this for both English and Polish languages, using data from one of the largest Polish e-commerce sites and selected open-domain datasets. We employ both monolingual and multilingual sentence encoders fine-tuned by a supervised method utilizing contrastive loss and NLI data. Our results point to the fact that NLI fine-tuning increases the performance of the models in both tasks and both languages, with the potential to improve mono- and multilingual models. Finally, we investigate uniformity and alignment of the embeddings to explain the effect of NLI-based fine-tuning for an out-of-domain use-case.

{{</citation>}}


### (9/50) LARCH: Large Language Model-based Automatic Readme Creation with Heuristics (Yuta Koreeda et al., 2023)

{{<citation>}}

Yuta Koreeda, Terufumi Morishita, Osamu Imaichi, Yasuhiro Sogawa. (2023)  
**LARCH: Large Language Model-based Automatic Readme Creation with Heuristics**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SE, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.03099v1)  

---


**ABSTRACT**  
Writing a readme is a crucial aspect of software development as it plays a vital role in managing and reusing program code. Though it is a pain point for many developers, automatically creating one remains a challenge even with the recent advancements in large language models (LLMs), because it requires generating abstract description from thousands of lines of code. In this demo paper, we show that LLMs are capable of generating a coherent and factually correct readmes if we can identify a code fragment that is representative of the repository. Building upon this finding, we developed LARCH (LLM-based Automatic Readme Creation with Heuristics) which leverages representative code identification with heuristics and weak supervision. Through human and automated evaluations, we illustrate that LARCH can generate coherent and factually correct readmes in the majority of cases, outperforming a baseline that does not rely on representative code identification. We have made LARCH open-source and provided a cross-platform Visual Studio Code interface and command-line interface, accessible at https://github.com/hitachi-nlp/larch . A demo video showcasing LARCH's capabilities is available at https://youtu.be/ZUKkh5ED-O4 .

{{</citation>}}


### (10/50) System-Initiated Transitions from Chit-Chat to Task-Oriented Dialogues with Transition Info Extractor and Transition Sentence Generator (Ye Liu et al., 2023)

{{<citation>}}

Ye Liu, Stefan Ultes, Wolfgang Minker, Wolfgang Maier. (2023)  
**System-Initiated Transitions from Chit-Chat to Task-Oriented Dialogues with Transition Info Extractor and Transition Sentence Generator**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.03098v1)  

---


**ABSTRACT**  
In this work, we study dialogue scenarios that start from chit-chat but eventually switch to task-related services, and investigate how a unified dialogue model, which can engage in both chit-chat and task-oriented dialogues, takes the initiative during the dialogue mode transition from chit-chat to task-oriented in a coherent and cooperative manner. We firstly build a {transition info extractor} (TIE) that keeps track of the preceding chit-chat interaction and detects the potential user intention to switch to a task-oriented service. Meanwhile, in the unified model, a {transition sentence generator} (TSG) is extended through efficient Adapter tuning and transition prompt learning. When the TIE successfully finds task-related information from the preceding chit-chat, such as a transition domain, then the TSG is activated automatically in the unified model to initiate this transition by generating a transition sentence under the guidance of transition information extracted by TIE. The experimental results show promising performance regarding the proactive transitions. We achieve an additional large improvement on TIE model by utilizing Conditional Random Fields (CRF). The TSG can flexibly generate transition sentences while maintaining the unified capabilities of normal chit-chat and task-oriented response generation.

{{</citation>}}


### (11/50) TARJAMAT: Evaluation of Bard and ChatGPT on Machine Translation of Ten Arabic Varieties (Karima Kadaoui et al., 2023)

{{<citation>}}

Karima Kadaoui, Samar M. Magdy, Abdul Waheed, Md Tawkat Islam Khondaker, Ahmed Oumar El-Shangiti, El Moatez Billah Nagoudi, Muhammad Abdul-Mageed. (2023)  
**TARJAMAT: Evaluation of Bard and ChatGPT on Machine Translation of Ten Arabic Varieties**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-3.5, GPT-4, Google, Machine Translation  
[Paper Link](http://arxiv.org/abs/2308.03051v1)  

---


**ABSTRACT**  
Large language models (LLMs) finetuned to follow human instructions have recently emerged as a breakthrough in AI. Models such as Google Bard and OpenAI ChatGPT, for example, are surprisingly powerful tools for question answering, code debugging, and dialogue generation. Despite the purported multilingual proficiency of these models, their linguistic inclusivity remains insufficiently explored. Considering this constraint, we present a thorough assessment of Bard and ChatGPT (encompassing both GPT-3.5 and GPT-4) regarding their machine translation proficiencies across ten varieties of Arabic. Our evaluation covers diverse Arabic varieties such as Classical Arabic, Modern Standard Arabic, and several nuanced dialectal variants. Furthermore, we undertake a human-centric study to scrutinize the efficacy of the most recent model, Bard, in following human instructions during translation tasks. Our exhaustive analysis indicates that LLMs may encounter challenges with certain Arabic dialects, particularly those for which minimal public data exists, such as Algerian and Mauritanian dialects. However, they exhibit satisfactory performance with more prevalent dialects, albeit occasionally trailing behind established commercial systems like Google Translate. Additionally, our analysis reveals a circumscribed capability of Bard in aligning with human instructions in translation contexts. Collectively, our findings underscore that prevailing LLMs remain far from inclusive, with only limited ability to cater for the linguistic and cultural intricacies of diverse communities.

{{</citation>}}


### (12/50) 3D-EX : A Unified Dataset of Definitions and Dictionary Examples (Fatemah Almeman et al., 2023)

{{<citation>}}

Fatemah Almeman, Hadi Sheikhi, Luis Espinosa-Anke. (2023)  
**3D-EX : A Unified Dataset of Definitions and Dictionary Examples**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.03043v1)  

---


**ABSTRACT**  
Definitions are a fundamental building block in lexicography, linguistics and computational semantics. In NLP, they have been used for retrofitting word embeddings or augmenting contextual representations in language models. However, lexical resources containing definitions exhibit a wide range of properties, which has implications in the behaviour of models trained and evaluated on them. In this paper, we introduce 3D- EX , a dataset that aims to fill this gap by combining well-known English resources into one centralized knowledge repository in the form of <term, definition, example> triples. 3D- EX is a unified evaluation framework with carefully pre-computed train/validation/test splits to prevent memorization. We report experimental results that suggest that this dataset could be effectively leveraged in downstream NLP tasks. Code and data are available at https://github.com/F-Almeman/3D-EX .

{{</citation>}}


### (13/50) Spanish Pre-trained BERT Model and Evaluation Data (José Cañete et al., 2023)

{{<citation>}}

José Cañete, Gabriel Chaperon, Rodrigo Fuentes, Jou-Hui Ho, Hojin Kang, Jorge Pérez. (2023)  
**Spanish Pre-trained BERT Model and Evaluation Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, GLUE  
[Paper Link](http://arxiv.org/abs/2308.02976v1)  

---


**ABSTRACT**  
The Spanish language is one of the top 5 spoken languages in the world. Nevertheless, finding resources to train or evaluate Spanish language models is not an easy task. In this paper we help bridge this gap by presenting a BERT-based language model pre-trained exclusively on Spanish data. As a second contribution, we also compiled several tasks specifically for the Spanish language in a single repository much in the spirit of the GLUE benchmark. By fine-tuning our pre-trained Spanish model, we obtain better results compared to other BERT-based models pre-trained on multilingual corpora for most of the tasks, even achieving a new state-of-the-art on some of them. We have publicly released our model, the pre-training data, and the compilation of the Spanish benchmarks.

{{</citation>}}


## cs.LG (6)



### (14/50) Time-Parameterized Convolutional Neural Networks for Irregularly Sampled Time Series (Chrysoula Kosma et al., 2023)

{{<citation>}}

Chrysoula Kosma, Giannis Nikolentzos, Michalis Vazirgiannis. (2023)  
**Time-Parameterized Convolutional Neural Networks for Irregularly Sampled Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.03210v2)  

---


**ABSTRACT**  
Irregularly sampled multivariate time series are ubiquitous in several application domains, leading to sparse, not fully-observed and non-aligned observations across different variables. Standard sequential neural network architectures, such as recurrent neural networks (RNNs) and convolutional neural networks (CNNs), consider regular spacing between observation times, posing significant challenges to irregular time series modeling. While most of the proposed architectures incorporate RNN variants to handle irregular time intervals, convolutional neural networks have not been adequately studied in the irregular sampling setting. In this paper, we parameterize convolutional layers by employing time-explicitly initialized kernels. Such general functions of time enhance the learning process of continuous-time hidden dynamics and can be efficiently incorporated into convolutional kernel weights. We, thus, propose the time-parameterized convolutional neural network (TPCNN), which shares similar properties with vanilla convolutions but is carefully designed for irregularly sampled time series. We evaluate TPCNN on both interpolation and classification tasks involving real-world irregularly sampled multivariate time series datasets. Our experimental results indicate the competitive performance of the proposed TPCNN model which is also significantly more efficient than other state-of-the-art methods. At the same time, the proposed architecture allows the interpretability of the input series by leveraging the combination of learnable time functions that improve the network performance in subsequent tasks and expedite the inaugural application of convolutions in this field.

{{</citation>}}


### (15/50) Communication-Free Distributed GNN Training with Vertex Cut (Kaidi Cao et al., 2023)

{{<citation>}}

Kaidi Cao, Rui Deng, Shirley Wu, Edward W Huang, Karthik Subbian, Jure Leskovec. (2023)  
**Communication-Free Distributed GNN Training with Vertex Cut**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.03209v1)  

---


**ABSTRACT**  
Training Graph Neural Networks (GNNs) on real-world graphs consisting of billions of nodes and edges is quite challenging, primarily due to the substantial memory needed to store the graph and its intermediate node and edge features, and there is a pressing need to speed up the training process. A common approach to achieve speed up is to divide the graph into many smaller subgraphs, which are then distributed across multiple GPUs in one or more machines and processed in parallel. However, existing distributed methods require frequent and substantial cross-GPU communication, leading to significant time overhead and progressively diminishing scalability. Here, we introduce CoFree-GNN, a novel distributed GNN training framework that significantly speeds up the training process by implementing communication-free training. The framework utilizes a Vertex Cut partitioning, i.e., rather than partitioning the graph by cutting the edges between partitions, the Vertex Cut partitions the edges and duplicates the node information to preserve the graph structure. Furthermore, the framework maintains high model accuracy by incorporating a reweighting mechanism to handle a distorted graph distribution that arises from the duplicated nodes. We also propose a modified DropEdge technique to further speed up the training process. Using an extensive set of experiments on real-world networks, we demonstrate that CoFree-GNN speeds up the GNN training process by up to 10 times over the existing state-of-the-art GNN training approaches.

{{</citation>}}


### (16/50) Adapting Machine Learning Diagnostic Models to New Populations Using a Small Amount of Data: Results from Clinical Neuroscience (Rongguang Wang et al., 2023)

{{<citation>}}

Rongguang Wang, Guray Erus, Pratik Chaudhari, Christos Davatzikos. (2023)  
**Adapting Machine Learning Diagnostic Models to New Populations Using a Small Amount of Data: Results from Clinical Neuroscience**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-IV, q-bio-QM  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.03175v1)  

---


**ABSTRACT**  
Machine learning (ML) has shown great promise for revolutionizing a number of areas, including healthcare. However, it is also facing a reproducibility crisis, especially in medicine. ML models that are carefully constructed from and evaluated on a training set might not generalize well on data from different patient populations or acquisition instrument settings and protocols. We tackle this problem in the context of neuroimaging of Alzheimer's disease (AD), schizophrenia (SZ) and brain aging. We develop a weighted empirical risk minimization approach that optimally combines data from a source group, e.g., subjects are stratified by attributes such as sex, age group, race and clinical cohort to make predictions on a target group, e.g., other sex, age group, etc. using a small fraction (10%) of data from the target group. We apply this method to multi-source data of 15,363 individuals from 20 neuroimaging studies to build ML models for diagnosis of AD and SZ, and estimation of brain age. We found that this approach achieves substantially better accuracy than existing domain adaptation techniques: it obtains area under curve greater than 0.95 for AD classification, area under curve greater than 0.7 for SZ classification and mean absolute error less than 5 years for brain age prediction on all target groups, achieving robustness to variations of scanners, protocols, and demographic or clinical characteristics. In some cases, it is even better than training on all data from the target group, because it leverages the diversity and size of a larger training set. We also demonstrate the utility of our models for prognostic tasks such as predicting disease progression in individuals with mild cognitive impairment. Critically, our brain age prediction models lead to new clinical insights regarding correlations with neurophysiological tests.

{{</citation>}}


### (17/50) Detection of Anomalies in Multivariate Time Series Using Ensemble Techniques (Anastasios Iliopoulos et al., 2023)

{{<citation>}}

Anastasios Iliopoulos, John Violos, Christos Diou, Iraklis Varlamis. (2023)  
**Detection of Anomalies in Multivariate Time Series Using Ensemble Techniques**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Anomaly Detection, LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2308.03171v1)  

---


**ABSTRACT**  
Anomaly Detection in multivariate time series is a major problem in many fields. Due to their nature, anomalies sparsely occur in real data, thus making the task of anomaly detection a challenging problem for classification algorithms to solve. Methods that are based on Deep Neural Networks such as LSTM, Autoencoders, Convolutional Autoencoders etc., have shown positive results in such imbalanced data. However, the major challenge that algorithms face when applied to multivariate time series is that the anomaly can arise from a small subset of the feature set. To boost the performance of these base models, we propose a feature-bagging technique that considers only a subset of features at a time, and we further apply a transformation that is based on nested rotation computed from Principal Component Analysis (PCA) to improve the effectiveness and generalization of the approach. To further enhance the prediction performance, we propose an ensemble technique that combines multiple base models toward the final decision. In addition, a semi-supervised approach using a Logistic Regressor to combine the base models' outputs is proposed. The proposed methodology is applied to the Skoltech Anomaly Benchmark (SKAB) dataset, which contains time series data related to the flow of water in a closed circuit, and the experimental results show that the proposed ensemble technique outperforms the basic algorithms. More specifically, the performance improvement in terms of anomaly detection accuracy reaches 2% for the unsupervised and at least 10% for the semi-supervised models.

{{</citation>}}


### (18/50) Iterative Magnitude Pruning as a Renormalisation Group: A Study in The Context of The Lottery Ticket Hypothesis (Abu-Al Hassan, 2023)

{{<citation>}}

Abu-Al Hassan. (2023)  
**Iterative Magnitude Pruning as a Renormalisation Group: A Study in The Context of The Lottery Ticket Hypothesis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.03128v1)  

---


**ABSTRACT**  
This thesis delves into the intricate world of Deep Neural Networks (DNNs), focusing on the exciting concept of the Lottery Ticket Hypothesis (LTH). The LTH posits that within extensive DNNs, smaller, trainable subnetworks termed "winning tickets", can achieve performance comparable to the full model. A key process in LTH, Iterative Magnitude Pruning (IMP), incrementally eliminates minimal weights, emulating stepwise learning in DNNs. Once we identify these winning tickets, we further investigate their "universality". In other words, we check if a winning ticket that works well for one specific problem could also work well for other, similar problems. We also bridge the divide between the IMP and the Renormalisation Group (RG) theory in physics, promoting a more rigorous understanding of IMP.

{{</citation>}}


### (19/50) Weakly Supervised Multi-Task Representation Learning for Human Activity Analysis Using Wearables (Taoran Sheng et al., 2023)

{{<citation>}}

Taoran Sheng, Manfred Huber. (2023)  
**Weakly Supervised Multi-Task Representation Learning for Human Activity Analysis Using Wearables**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.03805v1)  

---


**ABSTRACT**  
Sensor data streams from wearable devices and smart environments are widely studied in areas like human activity recognition (HAR), person identification, or health monitoring. However, most of the previous works in activity and sensor stream analysis have been focusing on one aspect of the data, e.g. only recognizing the type of the activity or only identifying the person who performed the activity. We instead propose an approach that uses a weakly supervised multi-output siamese network that learns to map the data into multiple representation spaces, where each representation space focuses on one aspect of the data. The representation vectors of the data samples are positioned in the space such that the data with the same semantic meaning in that aspect are closely located to each other. Therefore, as demonstrated with a set of experiments, the trained model can provide metrics for clustering data based on multiple aspects, allowing it to address multiple tasks simultaneously and even to outperform single task supervised methods in many situations. In addition, further experiments are presented that in more detail analyze the effect of the architecture and of using multiple tasks within this framework, that investigate the scalability of the model to include additional tasks, and that demonstrate the ability of the framework to combine data for which only partial relationship information with respect to the target tasks is available.

{{</citation>}}


## eess.IV (2)



### (20/50) Microvasculature Segmentation in Human BioMolecular Atlas Program (HuBMAP) (Youssef Sultan et al., 2023)

{{<citation>}}

Youssef Sultan, Yongqiang Wang, James Scanlon, Lisa D'lima. (2023)  
**Microvasculature Segmentation in Human BioMolecular Atlas Program (HuBMAP)**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03203v1)  

---


**ABSTRACT**  
Image segmentation serves as a critical tool across a range of applications, encompassing autonomous driving's pedestrian detection and pre-operative tumor delineation in the medical sector. Among these applications, we focus on the National Institutes of Health's (NIH) Human BioMolecular Atlas Program (HuBMAP), a significant initiative aimed at creating detailed cellular maps of the human body. In this study, we concentrate on segmenting various microvascular structures in human kidneys, utilizing 2D Periodic Acid-Schiff (PAS)-stained histology images. Our methodology begins with a foundational FastAI U-Net model, upon which we investigate alternative backbone architectures, delve into deeper models, and experiment with Feature Pyramid Networks. We rigorously evaluate these varied approaches by benchmarking their performance against our baseline U-Net model. This study thus offers a comprehensive exploration of cutting-edge segmentation techniques, providing valuable insights for future research in the field.

{{</citation>}}


### (21/50) Early Detection and Localization of Pancreatic Cancer by Label-Free Tumor Synthesis (Bowen Li et al., 2023)

{{<citation>}}

Bowen Li, Yu-Cheng Chou, Shuwen Sun, Hualin Qiao, Alan Yuille, Zongwei Zhou. (2023)  
**Early Detection and Localization of Pancreatic Cancer by Label-Free Tumor Synthesis**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03008v1)  

---


**ABSTRACT**  
Early detection and localization of pancreatic cancer can increase the 5-year survival rate for patients from 8.5% to 20%. Artificial intelligence (AI) can potentially assist radiologists in detecting pancreatic tumors at an early stage. Training AI models require a vast number of annotated examples, but the availability of CT scans obtaining early-stage tumors is constrained. This is because early-stage tumors may not cause any symptoms, which can delay detection, and the tumors are relatively small and may be almost invisible to human eyes on CT scans. To address this issue, we develop a tumor synthesis method that can synthesize enormous examples of small pancreatic tumors in the healthy pancreas without the need for manual annotation. Our experiments demonstrate that the overall detection rate of pancreatic tumors, measured by Sensitivity and Specificity, achieved by AI trained on synthetic tumors is comparable to that of real tumors. More importantly, our method shows a much higher detection rate for small tumors. We further investigate the per-voxel segmentation performance of pancreatic tumors if AI is trained on a combination of CT scans with synthetic tumors and CT scans with annotated large tumors at an advanced stage. Finally, we show that synthetic tumors improve AI generalizability in tumor detection and localization when processing CT scans from different hospitals. Overall, our proposed tumor synthesis method has immense potential to improve the early detection of pancreatic cancer, leading to better patient outcomes.

{{</citation>}}


## cs.AI (6)



### (22/50) Empirical Optimal Risk to Quantify Model Trustworthiness for Failure Detection (Shuang Ao et al., 2023)

{{<citation>}}

Shuang Ao, Stefan Rueger, Advaith Siddharthan. (2023)  
**Empirical Optimal Risk to Quantify Model Trustworthiness for Failure Detection**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03179v1)  

---


**ABSTRACT**  
Failure detection (FD) in AI systems is a crucial safeguard for the deployment for safety-critical tasks. The common evaluation method of FD performance is the Risk-coverage (RC) curve, which reveals the trade-off between the data coverage rate and the performance on accepted data. One common way to quantify the RC curve by calculating the area under the RC curve. However, this metric does not inform on how suited any method is for FD, or what the optimal coverage rate should be. As FD aims to achieve higher performance with fewer data discarded, evaluating with partial coverage excluding the most uncertain samples is more intuitive and meaningful than full coverage. In addition, there is an optimal point in the coverage where the model could achieve ideal performance theoretically. We propose the Excess Area Under the Optimal RC Curve (E-AUoptRC), with the area in coverage from the optimal point to the full coverage. Further, the model performance at this optimal point can represent both model learning ability and calibration. We propose it as the Trust Index (TI), a complementary evaluation metric to the overall model accuracy. We report extensive experiments on three benchmark image datasets with ten variants of transformer and CNN models. Our results show that our proposed methods can better reflect the model trustworthiness than existing evaluation metrics. We further observe that the model with high overall accuracy does not always yield the high TI, which indicates the necessity of the proposed Trust Index as a complementary metric to the model overall accuracy. The code are available at \url{https://github.com/AoShuang92/optimal_risk}.

{{</citation>}}


### (23/50) Building Safe and Reliable AI systems for Safety Critical Tasks with Vision-Language Processing (Shuang Ao, 2023)

{{<citation>}}

Shuang Ao. (2023)  
**Building Safe and Reliable AI systems for Safety Critical Tasks with Vision-Language Processing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03176v1)  

---


**ABSTRACT**  
Although AI systems have been applied in various fields and achieved impressive performance, their safety and reliability are still a big concern. This is especially important for safety-critical tasks. One shared characteristic of these critical tasks is their risk sensitivity, where small mistakes can cause big consequences and even endanger life. There are several factors that could be guidelines for the successful deployment of AI systems in sensitive tasks: (i) failure detection and out-of-distribution (OOD) detection; (ii) overfitting identification; (iii) uncertainty quantification for predictions; (iv) robustness to data perturbations. These factors are also challenges of current AI systems, which are major blocks for building safe and reliable AI. Specifically, the current AI algorithms are unable to identify common causes for failure detection. Furthermore, additional techniques are required to quantify the quality of predictions. All these contribute to inaccurate uncertainty quantification, which lowers trust in predictions. Hence obtaining accurate model uncertainty quantification and its further improvement are challenging. To address these issues, many techniques have been proposed, such as regularization methods and learning strategies. As vision and language are the most typical data type and have many open source benchmark datasets, this thesis will focus on vision-language data processing for tasks like classification, image captioning, and vision question answering. In this thesis, we aim to build a safeguard by further developing current techniques to ensure the accurate model uncertainty for safety-critical tasks.

{{</citation>}}


### (24/50) Precise Benchmarking of Explainable AI Attribution Methods (Rafaël Brandt et al., 2023)

{{<citation>}}

Rafaël Brandt, Daan Raatjens, Georgi Gaydadjiev. (2023)  
**Precise Benchmarking of Explainable AI Attribution Methods**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03161v1)  

---


**ABSTRACT**  
The rationale behind a deep learning model's output is often difficult to understand by humans. EXplainable AI (XAI) aims at solving this by developing methods that improve interpretability and explainability of machine learning models. Reliable evaluation metrics are needed to assess and compare different XAI methods. We propose a novel evaluation approach for benchmarking state-of-the-art XAI attribution methods. Our proposal consists of a synthetic classification model accompanied by its derived ground truth explanations allowing high precision representation of input nodes contributions. We also propose new high-fidelity metrics to quantify the difference between explanations of the investigated XAI method and those derived from the synthetic model. Our metrics allow assessment of explanations in terms of precision and recall separately. Also, we propose metrics to independently evaluate negative or positive contributions of inputs. Our proposal provides deeper insights into XAI methods output. We investigate our proposal by constructing a synthetic convolutional image classification model and benchmarking several widely used XAI attribution methods using our evaluation approach. We compare our results with established prior XAI evaluation metrics. By deriving the ground truth directly from the constructed model in our method, we ensure the absence of bias, e.g., subjective either based on the training set. Our experimental results provide novel insights into the performance of Guided-Backprop and Smoothgrad XAI methods that are widely in use. Both have good precision and recall scores among positively contributing pixels (0.7, 0.76 and 0.7, 0.77, respectively), but poor precision scores among negatively contributing pixels (0.44, 0.61 and 0.47, 0.75, resp.). The recall scores in the latter case remain close. We show that our metrics are among the fastest in terms of execution time.

{{</citation>}}


### (25/50) 'We care': Improving Code Mixed Speech Emotion Recognition in Customer-Care Conversations (N V S Abhishek et al., 2023)

{{<citation>}}

N V S Abhishek, Pushpak Bhattacharyya. (2023)  
**'We care': Improving Code Mixed Speech Emotion Recognition in Customer-Care Conversations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.03150v1)  

---


**ABSTRACT**  
Speech Emotion Recognition (SER) is the task of identifying the emotion expressed in a spoken utterance. Emotion recognition is essential in building robust conversational agents in domains such as law, healthcare, education, and customer support. Most of the studies published on SER use datasets created by employing professional actors in a noise-free environment. In natural settings such as a customer care conversation, the audio is often noisy with speakers regularly switching between different languages as they see fit. We have worked in collaboration with a leading unicorn in the Conversational AI sector to develop Natural Speech Emotion Dataset (NSED). NSED is a natural code-mixed speech emotion dataset where each utterance in a conversation is annotated with emotion, sentiment, valence, arousal, and dominance (VAD) values. In this paper, we show that by incorporating word-level VAD value we improve on the task of SER by 2%, for negative emotions, over the baseline value for NSED. High accuracy for negative emotion recognition is essential because customers expressing negative opinions/views need to be pacified with urgency, lest complaints and dissatisfaction snowball and get out of hand. Escalation of negative opinions speedily is crucial for business interests. Our study then can be utilized to develop conversational agents which are more polite and empathetic in such situations.

{{</citation>}}


### (26/50) Embedding-based Retrieval with LLM for Effective Agriculture Information Extracting from Unstructured Data (Ruoling Peng et al., 2023)

{{<citation>}}

Ruoling Peng, Kang Liu, Po Yang, Zhipeng Yuan, Shunbao Li. (2023)  
**Embedding-based Retrieval with LLM for Effective Agriculture Information Extracting from Unstructured Data**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.03107v1)  

---


**ABSTRACT**  
Pest identification is a crucial aspect of pest control in agriculture. However, most farmers are not capable of accurately identifying pests in the field, and there is a limited number of structured data sources available for rapid querying. In this work, we explored using domain-agnostic general pre-trained large language model(LLM) to extract structured data from agricultural documents with minimal or no human intervention. We propose a methodology that involves text retrieval and filtering using embedding-based retrieval, followed by LLM question-answering to automatically extract entities and attributes from the documents, and transform them into structured data. In comparison to existing methods, our approach achieves consistently better accuracy in the benchmark while maintaining efficiency.

{{</citation>}}


### (27/50) Pre-Trained Large Language Models for Industrial Control (Lei Song et al., 2023)

{{<citation>}}

Lei Song, Chuheng Zhang, Li Zhao, Jiang Bian. (2023)  
**Pre-Trained Large Language Models for Industrial Control**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03028v1)  

---


**ABSTRACT**  
For industrial control, developing high-performance controllers with few samples and low technical debt is appealing. Foundation models, possessing rich prior knowledge obtained from pre-training with Internet-scale corpus, have the potential to be a good controller with proper prompts. In this paper, we take HVAC (Heating, Ventilation, and Air Conditioning) building control as an example to examine the ability of GPT-4 (one of the first-tier foundation models) as the controller. To control HVAC, we wrap the task as a language game by providing text including a short description for the task, several selected demonstrations, and the current observation to GPT-4 on each step and execute the actions responded by GPT-4. We conduct series of experiments to answer the following questions: 1)~How well can GPT-4 control HVAC? 2)~How well can GPT-4 generalize to different scenarios for HVAC control? 3) How different parts of the text context affect the performance? In general, we found GPT-4 achieves the performance comparable to RL methods with few samples and low technical debt, indicating the potential of directly applying foundation models to industrial control tasks.

{{</citation>}}


## cs.CV (15)



### (28/50) CGBA: Curvature-aware Geometric Black-box Attack (Md Farhamdur Reza et al., 2023)

{{<citation>}}

Md Farhamdur Reza, Ali Rahmati, Tianfu Wu, Huaiyu Dai. (2023)  
**CGBA: Curvature-aware Geometric Black-box Attack**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.03163v1)  

---


**ABSTRACT**  
Decision-based black-box attacks often necessitate a large number of queries to craft an adversarial example. Moreover, decision-based attacks based on querying boundary points in the estimated normal vector direction often suffer from inefficiency and convergence issues. In this paper, we propose a novel query-efficient curvature-aware geometric decision-based black-box attack (CGBA) that conducts boundary search along a semicircular path on a restricted 2D plane to ensure finding a boundary point successfully irrespective of the boundary curvature. While the proposed CGBA attack can work effectively for an arbitrary decision boundary, it is particularly efficient in exploiting the low curvature to craft high-quality adversarial examples, which is widely seen and experimentally verified in commonly used classifiers under non-targeted attacks. In contrast, the decision boundaries often exhibit higher curvature under targeted attacks. Thus, we develop a new query-efficient variant, CGBA-H, that is adapted for the targeted attack. In addition, we further design an algorithm to obtain a better initial boundary point at the expense of some extra queries, which considerably enhances the performance of the targeted attack. Extensive experiments are conducted to evaluate the performance of our proposed methods against some well-known classifiers on the ImageNet and CIFAR10 datasets, demonstrating the superiority of CGBA and CGBA-H over state-of-the-art non-targeted and targeted attacks, respectively. The source code is available at https://github.com/Farhamdur/CGBA.

{{</citation>}}


### (29/50) Food-500 Cap: A Fine-Grained Food Caption Benchmark for Evaluating Vision-Language Models (Zheng Ma et al., 2023)

{{<citation>}}

Zheng Ma, Mianzhi Pan, Wenhan Wu, Kanzhi Cheng, Jianbing Zhang, Shujian Huang, Jiajun Chen. (2023)  
**Food-500 Cap: A Fine-Grained Food Caption Benchmark for Evaluating Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.03151v1)  

---


**ABSTRACT**  
Vision-language models (VLMs) have shown impressive performance in substantial downstream multi-modal tasks. However, only comparing the fine-tuned performance on downstream tasks leads to the poor interpretability of VLMs, which is adverse to their future improvement. Several prior works have identified this issue and used various probing methods under a zero-shot setting to detect VLMs' limitations, but they all examine VLMs using general datasets instead of specialized ones. In practical applications, VLMs are usually applied to specific scenarios, such as e-commerce and news fields, so the generalization of VLMs in specific domains should be given more attention. In this paper, we comprehensively investigate the capabilities of popular VLMs in a specific field, the food domain. To this end, we build a food caption dataset, Food-500 Cap, which contains 24,700 food images with 494 categories. Each image is accompanied by a detailed caption, including fine-grained attributes of food, such as the ingredient, shape, and color. We also provide a culinary culture taxonomy that classifies each food category based on its geographic origin in order to better analyze the performance differences of VLM in different regions. Experiments on our proposed datasets demonstrate that popular VLMs underperform in the food domain compared with their performance in the general domain. Furthermore, our research reveals severe bias in VLMs' ability to handle food items from different geographic regions. We adopt diverse probing methods and evaluate nine VLMs belonging to different architectures to verify the aforementioned observations. We hope that our study will bring researchers' attention to VLM's limitations when applying them to the domain of food or culinary cultures, and spur further investigations to address this issue.

{{</citation>}}


### (30/50) SAAM: Stealthy Adversarial Attack on Monoculor Depth Estimation (Amira Guesmi et al., 2023)

{{<citation>}}

Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique. (2023)  
**SAAM: Stealthy Adversarial Attack on Monoculor Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2308.03108v1)  

---


**ABSTRACT**  
In this paper, we investigate the vulnerability of MDE to adversarial patches. We propose a novel \underline{S}tealthy \underline{A}dversarial \underline{A}ttacks on \underline{M}DE (SAAM) that compromises MDE by either corrupting the estimated distance or causing an object to seamlessly blend into its surroundings. Our experiments, demonstrate that the designed stealthy patch successfully causes a DNN-based MDE to misestimate the depth of objects. In fact, our proposed adversarial patch achieves a significant 60\% depth error with 99\% ratio of the affected region. Importantly, despite its adversarial nature, the patch maintains a naturalistic appearance, making it inconspicuous to human observers. We believe that this work sheds light on the threat of adversarial attacks in the context of MDE on edge devices. We hope it raises awareness within the community about the potential real-life harm of such attacks and encourages further research into developing more robust and adaptive defense mechanisms.

{{</citation>}}


### (31/50) Incorporating Pre-training Data Matters in Unsupervised Domain Adaptation (Yinsong Xu et al., 2023)

{{<citation>}}

Yinsong Xu, Aidong Men, Yang Liu, Qingchao Chen. (2023)  
**Incorporating Pre-training Data Matters in Unsupervised Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.03097v1)  

---


**ABSTRACT**  
Unsupervised domain adaptation(UDA) and Source-free UDA(SFUDA) methods formulate the problem involving two domains: source and target. They typically employ a standard training approach that begins with models pre-trained on large-scale datasets e.g., ImageNet, while rarely discussing its effect. Recognizing this gap, we investigate the following research questions: (1) What is the correlation among ImageNet, the source, and the target domain? (2) How does pre-training on ImageNet influence the target risk? To answer the first question, we empirically observed an interesting Spontaneous Pulling (SP) Effect in fine-tuning where the discrepancies between any two of the three domains (ImageNet, Source, Target) decrease but at the cost of the impaired semantic structure of the pre-train domain. For the second question, we put forward a theory to explain SP and quantify that the target risk is bound by gradient disparities among the three domains. Our observations reveal a key limitation of existing methods: it hinders the adaptation performance if the semantic cluster structure of the pre-train dataset (i.e.ImageNet) is impaired. To address it, we incorporate ImageNet as the third domain and redefine the UDA/SFUDA as a three-player game. Specifically, inspired by the theory and empirical findings, we present a novel framework termed TriDA which additionally preserves the semantic structure of the pre-train dataset during fine-tuning. Experimental results demonstrate that it achieves state-of-the-art performance across various UDA and SFUDA benchmarks.

{{</citation>}}


### (32/50) TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment (Chaofeng Chen et al., 2023)

{{<citation>}}

Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin. (2023)  
**TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.03060v1)  

---


**ABSTRACT**  
Image Quality Assessment (IQA) is a fundamental task in computer vision that has witnessed remarkable progress with deep neural networks. Inspired by the characteristics of the human visual system, existing methods typically use a combination of global and local representations (\ie, multi-scale features) to achieve superior performance. However, most of them adopt simple linear fusion of multi-scale features, and neglect their possibly complex relationship and interaction. In contrast, humans typically first form a global impression to locate important regions and then focus on local details in those regions. We therefore propose a top-down approach that uses high-level semantics to guide the IQA network to focus on semantically important local distortion regions, named as \emph{TOPIQ}. Our approach to IQA involves the design of a heuristic coarse-to-fine network (CFANet) that leverages multi-scale features and progressively propagates multi-level semantic information to low-level representations in a top-down manner. A key component of our approach is the proposed cross-scale attention mechanism, which calculates attention maps for lower level features guided by higher level features. This mechanism emphasizes active semantic regions for low-level distortions, thereby improving performance. CFANet can be used for both Full-Reference (FR) and No-Reference (NR) IQA. We use ResNet50 as its backbone and demonstrate that CFANet achieves better or competitive performance on most public FR and NR benchmarks compared with state-of-the-art methods based on vision transformers, while being much more efficient (with only ${\sim}13\%$ FLOPS of the current best FR method). Codes are released at \url{https://github.com/chaofengc/IQA-PyTorch}.

{{</citation>}}


### (33/50) Multi-scale Alternated Attention Transformer for Generalized Stereo Matching (Wei Miao et al., 2023)

{{<citation>}}

Wei Miao, Hong Zhao, Tongjia Chen, Wei Huang, Changyan Xiao. (2023)  
**Multi-scale Alternated Attention Transformer for Generalized Stereo Matching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03048v1)  

---


**ABSTRACT**  
Recent stereo matching networks achieves dramatic performance by introducing epipolar line constraint to limit the matching range of dual-view. However, in complicated real-world scenarios, the feature information based on intra-epipolar line alone is too weak to facilitate stereo matching. In this paper, we present a simple but highly effective network called Alternated Attention U-shaped Transformer (AAUformer) to balance the impact of epipolar line in dual and single view respectively for excellent generalization performance. Compared to other models, our model has several main designs: 1) to better liberate the local semantic features of the single-view at pixel level, we introduce window self-attention to break the limits of intra-row self-attention and completely replace the convolutional network for denser features before cross-matching; 2) the multi-scale alternated attention backbone network was designed to extract invariant features in order to achieves the coarse-to-fine matching process for hard-to-discriminate regions. We performed a series of both comparative studies and ablation studies on several mainstream stereo matching datasets. The results demonstrate that our model achieves state-of-the-art on the Scene Flow dataset, and the fine-tuning performance is competitive on the KITTI 2015 dataset. In addition, for cross generalization experiments on synthetic and real-world datasets, our model outperforms several state-of-the-art works.

{{</citation>}}


### (34/50) Prototypes-oriented Transductive Few-shot Learning with Conditional Transport (Long Tian et al., 2023)

{{<citation>}}

Long Tian, Jingyi Feng, Wenchao Chen, Xiaoqiang Chai, Liming Wang, Xiyang Liu, Bo Chen. (2023)  
**Prototypes-oriented Transductive Few-shot Learning with Conditional Transport**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, ImageNet  
[Paper Link](http://arxiv.org/abs/2308.03047v1)  

---


**ABSTRACT**  
Transductive Few-Shot Learning (TFSL) has recently attracted increasing attention since it typically outperforms its inductive peer by leveraging statistics of query samples. However, previous TFSL methods usually encode uniform prior that all the classes within query samples are equally likely, which is biased in imbalanced TFSL and causes severe performance degradation.   Given this pivotal issue, in this work, we propose a novel Conditional Transport (CT) based imbalanced TFSL model called {\textbf P}rototypes-oriented {\textbf U}nbiased {\textbf T}ransfer {\textbf M}odel (PUTM) to fully exploit unbiased statistics of imbalanced query samples, which employs forward and backward navigators as transport matrices to balance the prior of query samples per class between uniform and adaptive data-driven distributions. For efficiently transferring statistics learned by CT, we further derive a closed form solution to refine prototypes based on MAP given the learned navigators. The above two steps of discovering and transferring unbiased statistics follow an iterative manner, formulating our EM-based solver.   Experimental results on four standard benchmarks including miniImageNet, tieredImageNet, CUB, and CIFAR-FS demonstrate superiority of our model in class-imbalanced generalization.

{{</citation>}}


### (35/50) FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency Information (Chenxi Wang et al., 2023)

{{<citation>}}

Chenxi Wang, Hongjun Wu, Zhi Jin. (2023)  
**FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency Information**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.03033v1)  

---


**ABSTRACT**  
Recently, Fourier frequency information has attracted much attention in Low-Light Image Enhancement (LLIE). Some researchers noticed that, in the Fourier space, the lightness degradation mainly exists in the amplitude component and the rest exists in the phase component. By incorporating both the Fourier frequency and the spatial information, these researchers proposed remarkable solutions for LLIE. In this work, we further explore the positive correlation between the magnitude of amplitude and the magnitude of lightness, which can be effectively leveraged to improve the lightness of low-light images in the Fourier space. Moreover, we find that the Fourier transform can extract the global information of the image, and does not introduce massive neural network parameters like Multi-Layer Perceptrons (MLPs) or Transformer. To this end, a two-stage Fourier-based LLIE network (FourLLIE) is proposed. In the first stage, we improve the lightness of low-light images by estimating the amplitude transform map in the Fourier space. In the second stage, we introduce the Signal-to-Noise-Ratio (SNR) map to provide the prior for integrating the global Fourier frequency and the local spatial information, which recovers image details in the spatial space. With this ingenious design, FourLLIE outperforms the existing state-of-the-art (SOTA) LLIE methods on four representative datasets while maintaining good model efficiency.

{{</citation>}}


### (36/50) High-Resolution Vision Transformers for Pixel-Level Identification of Structural Components and Damage (Kareem Eltouny et al., 2023)

{{<citation>}}

Kareem Eltouny, Seyedomid Sajedi, Xiao Liang. (2023)  
**High-Resolution Vision Transformers for Pixel-Level Identification of Structural Components and Damage**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.03006v1)  

---


**ABSTRACT**  
Visual inspection is predominantly used to evaluate the state of civil structures, but recent developments in unmanned aerial vehicles (UAVs) and artificial intelligence have increased the speed, safety, and reliability of the inspection process. In this study, we develop a semantic segmentation network based on vision transformers and Laplacian pyramids scaling networks for efficiently parsing high-resolution visual inspection images. The massive amounts of collected high-resolution images during inspections can slow down the investigation efforts. And while there have been extensive studies dedicated to the use of deep learning models for damage segmentation, processing high-resolution visual data can pose major computational difficulties. Traditionally, images are either uniformly downsampled or partitioned to cope with computational demands. However, the input is at risk of losing local fine details, such as thin cracks, or global contextual information. Inspired by super-resolution architectures, our vision transformer model learns to resize high-resolution images and masks to retain both the valuable local features and the global semantics without sacrificing computational efficiency. The proposed framework has been evaluated through comprehensive experiments on a dataset of bridge inspection report images using multiple metrics for pixel-wise materials detection.

{{</citation>}}


### (37/50) MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation (Lian Xu et al., 2023)

{{<citation>}}

Lian Xu, Mohammed Bennamoun, Farid Boussaid, Hamid Laga, Wanli Ouyang, Dan Xu. (2023)  
**MCTformer+: Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2308.03005v1)  

---


**ABSTRACT**  
This paper proposes a novel transformer-based framework that aims to enhance weakly supervised semantic segmentation (WSSS) by generating accurate class-specific object localization maps as pseudo labels. Building upon the observation that the attended regions of the one-class token in the standard vision transformer can contribute to a class-agnostic localization map, we explore the potential of the transformer model to capture class-specific attention for class-discriminative object localization by learning multiple class tokens. We introduce a Multi-Class Token transformer, which incorporates multiple class tokens to enable class-aware interactions with the patch tokens. To achieve this, we devise a class-aware training strategy that establishes a one-to-one correspondence between the output class tokens and the ground-truth class labels. Moreover, a Contrastive-Class-Token (CCT) module is proposed to enhance the learning of discriminative class tokens, enabling the model to better capture the unique characteristics and properties of each class. As a result, class-discriminative object localization maps can be effectively generated by leveraging the class-to-patch attentions associated with different class tokens. To further refine these localization maps, we propose the utilization of patch-level pairwise affinity derived from the patch-to-patch transformer attention. Furthermore, the proposed framework seamlessly complements the Class Activation Mapping (CAM) method, resulting in significantly improved WSSS performance on the PASCAL VOC 2012 and MS COCO 2014 datasets. These results underline the importance of the class token for WSSS.

{{</citation>}}


### (38/50) Cal-SFDA: Source-Free Domain-adaptive Semantic Segmentation with Differentiable Expected Calibration Error (Zixin Wang et al., 2023)

{{<citation>}}

Zixin Wang, Yadan Luo, Zhi Chen, Sen Wang, Zi Huang. (2023)  
**Cal-SFDA: Source-Free Domain-adaptive Semantic Segmentation with Differentiable Expected Calibration Error**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.03003v1)  

---


**ABSTRACT**  
The prevalence of domain adaptive semantic segmentation has prompted concerns regarding source domain data leakage, where private information from the source domain could inadvertently be exposed in the target domain. To circumvent the requirement for source data, source-free domain adaptation has emerged as a viable solution that leverages self-training methods to pseudo-label high-confidence regions and adapt the model to the target data. However, the confidence scores obtained are often highly biased due to over-confidence and class-imbalance issues, which render both model selection and optimization problematic. In this paper, we propose a novel calibration-guided source-free domain adaptive semantic segmentation (Cal-SFDA) framework. The core idea is to estimate the expected calibration error (ECE) from the segmentation predictions, serving as a strong indicator of the model's generalization capability to the unlabeled target domain. The estimated ECE scores, in turn, assist the model training and fair selection in both source training and target adaptation stages. During model pre-training on the source domain, we ensure the differentiability of the ECE objective by leveraging the LogSumExp trick and using ECE scores to select the best source checkpoints for adaptation. To enable ECE estimation on the target domain without requiring labels, we train a value net for ECE estimation and apply statistic warm-up on its BatchNorm layers for stability. The estimated ECE scores assist in determining the reliability of prediction and enable class-balanced pseudo-labeling by positively guiding the adaptation progress and inhibiting potential error accumulation. Extensive experiments on two widely-used synthetic-to-real transfer tasks show that the proposed approach surpasses previous state-of-the-art by up to 5.25% of mIoU with fair model selection criteria.

{{</citation>}}


### (39/50) StyleEDL: Style-Guided High-order Attention Network for Image Emotion Distribution Learning (Peiguang Jing et al., 2023)

{{<citation>}}

Peiguang Jing, Xianyi Liu, Ji Wang, Yinwei Wei, Liqiang Nie, Yuting Su. (2023)  
**StyleEDL: Style-Guided High-order Attention Network for Image Emotion Distribution Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.03000v1)  

---


**ABSTRACT**  
Emotion distribution learning has gained increasing attention with the tendency to express emotions through images. As for emotion ambiguity arising from humans' subjectivity, substantial previous methods generally focused on learning appropriate representations from the holistic or significant part of images. However, they rarely consider establishing connections with the stylistic information although it can lead to a better understanding of images. In this paper, we propose a style-guided high-order attention network for image emotion distribution learning termed StyleEDL, which interactively learns stylistic-aware representations of images by exploring the hierarchical stylistic information of visual contents. Specifically, we consider exploring the intra- and inter-layer correlations among GRAM-based stylistic representations, and meanwhile exploit an adversary-constrained high-order attention mechanism to capture potential interactions between subtle visual parts. In addition, we introduce a stylistic graph convolutional network to dynamically generate the content-dependent emotion representations to benefit the final emotion distribution learning. Extensive experiments conducted on several benchmark datasets demonstrate the effectiveness of our proposed StyleEDL compared to state-of-the-art methods. The implementation is released at: https://github.com/liuxianyi/StyleEDL.

{{</citation>}}


### (40/50) Novel Class Discovery for Long-tailed Recognition (Zhang Chuyu et al., 2023)

{{<citation>}}

Zhang Chuyu, Xu Ruijie, He Xuming. (2023)  
**Novel Class Discovery for Long-tailed Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.02989v1)  

---


**ABSTRACT**  
While the novel class discovery has achieved great success, existing methods usually evaluate their algorithms on balanced datasets. However, in real-world visual recognition tasks, the class distribution of a dataset is often long-tailed, making it challenging to apply those methods. In this paper, we propose a more realistic setting for novel class discovery where the distribution of novel and known classes is long-tailed. The challenge of this new problem is to discover novel classes with the help of known classes under an imbalanced class scenario. To discover imbalanced novel classes efficiently, we propose an adaptive self-labeling strategy based on an equiangular prototype representation. Our method infers better pseudo-labels for the novel classes by solving a relaxed optimal transport problem and effectively mitigates the biases in learning the known and novel classes. The extensive results on CIFAR100, ImageNet100, and the challenging Herbarium19 and large-scale iNaturalist18 datasets demonstrate the superiority of our method.

{{</citation>}}


### (41/50) Introducing Feature Attention Module on Convolutional Neural Network for Diabetic Retinopathy Detection (Susmita Ghosh et al., 2023)

{{<citation>}}

Susmita Ghosh, Abhiroop Chatterjee. (2023)  
**Introducing Feature Attention Module on Convolutional Neural Network for Diabetic Retinopathy Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.02985v1)  

---


**ABSTRACT**  
Diabetic retinopathy (DR) is a leading cause of blindness among diabetic patients. Deep learning models have shown promising results in automating the detection of DR. In the present work, we propose a new methodology that integrates a feature attention module with a pretrained VGG19 convolutional neural network (CNN) for more accurate DR detection. Here, the pretrained net is fine-tuned with the proposed feature attention block. The proposed module aims to leverage the complementary information from various regions of fundus images to enhance the discriminative power of the CNN. The said feature attention module incorporates an attention mechanism which selectively highlights salient features from images and fuses them with the original input. The simultaneous learning of attention weights for the features and thereupon the combination of attention-modulated features within the feature attention block facilitates the network's ability to focus on relevant information while reducing the impact of noisy or irrelevant features. Performance of the proposed method has been evaluated on a widely used dataset for diabetic retinopathy classification e.g., the APTOS (Asia Pacific Tele-Ophthalmology Society) DR Dataset. Results are compared with/without attention module, as well as with other state-of-the-art approaches. Results confirm that the introduction of the fusion module (fusing of feature attention module with CNN) improves the accuracy of DR detection achieving an accuracy of 95.70%.

{{</citation>}}


### (42/50) Focus the Discrepancy: Intra- and Inter-Correlation Learning for Image Anomaly Detection (Xincheng Yao et al., 2023)

{{<citation>}}

Xincheng Yao, Ruoqi Li, Zefeng Qian, Yan Luo, Chongyang Zhang. (2023)  
**Focus the Discrepancy: Intra- and Inter-Correlation Learning for Image Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2308.02983v1)  

---


**ABSTRACT**  
Humans recognize anomalies through two aspects: larger patch-wise representation discrepancies and weaker patch-to-normal-patch correlations. However, the previous AD methods didn't sufficiently combine the two complementary aspects to design AD models. To this end, we find that Transformer can ideally satisfy the two aspects as its great power in the unified modeling of patch-wise representations and patch-to-patch correlations. In this paper, we propose a novel AD framework: FOcus-the-Discrepancy (FOD), which can simultaneously spot the patch-wise, intra- and inter-discrepancies of anomalies. The major characteristic of our method is that we renovate the self-attention maps in transformers to Intra-Inter-Correlation (I2Correlation). The I2Correlation contains a two-branch structure to first explicitly establish intra- and inter-image correlations, and then fuses the features of two-branch to spotlight the abnormal patterns. To learn the intra- and inter-correlations adaptively, we propose the RBF-kernel-based target-correlations as learning targets for self-supervised learning. Besides, we introduce an entropy constraint strategy to solve the mode collapse issue in optimization and further amplify the normal-abnormal distinguishability. Extensive experiments on three unsupervised real-world AD benchmarks show the superior performance of our approach. Code will be available at https://github.com/xcyao00/FOD.

{{</citation>}}


## physics.ao-ph (1)



### (43/50) AI-GOMS: Large AI-Driven Global Ocean Modeling System (Wei Xiong et al., 2023)

{{<citation>}}

Wei Xiong, Yanfei Xiang, Hao Wu, Shuyi Zhou, Yuze Sun, Muyuan Ma, Xiaomeng Huang. (2023)  
**AI-GOMS: Large AI-Driven Global Ocean Modeling System**  

---
Primary Category: physics.ao-ph  
Categories: cs-AI, cs-LG, physics-ao-ph, physics.ao-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03152v2)  

---


**ABSTRACT**  
Ocean modeling is a powerful tool for simulating the physical, chemical, and biological processes of the ocean, which is the foundation for marine science research and operational oceanography. Modern numerical ocean modeling mainly consists of governing equations and numerical algorithms. Nonlinear instability, computational expense, low reusability efficiency and high coupling costs have gradually become the main bottlenecks for the further development of numerical ocean modeling. Recently, artificial intelligence-based modeling in scientific computing has shown revolutionary potential for digital twins and scientific simulations, but the bottlenecks of numerical ocean modeling have not been further solved. Here, we present AI-GOMS, a large AI-driven global ocean modeling system, for accurate and efficient global ocean daily prediction. AI-GOMS consists of a backbone model with the Fourier-based Masked Autoencoder structure for basic ocean variable prediction and lightweight fine-tuning models incorporating regional downscaling, wave decoding, and biochemistry coupling modules. AI-GOMS has achieved the best performance in 30 days of prediction for the global ocean basic variables with 15 depth layers at 1/4{\deg} spatial resolution. Beyond the good performance in statistical metrics, AI-GOMS realizes the simulation of mesoscale eddies in the Kuroshio region at 1/12{\deg} spatial resolution and ocean stratification in the tropical Pacific Ocean. AI-GOMS provides a new backbone-downstream paradigm for Earth system modeling, which makes the system transferable, scalable and reusable.

{{</citation>}}


## cs.DC (1)



### (44/50) Autonomous Choreography of WebAssembly Workloads in the Federated Cloud-Edge-IoT Continuum (Piotr Sowinski et al., 2023)

{{<citation>}}

Piotr Sowinski, Ignacio Lacalle, Rafael Vano, Carlos E. Palau. (2023)  
**Autonomous Choreography of WebAssembly Workloads in the Federated Cloud-Edge-IoT Continuum**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03119v1)  

---


**ABSTRACT**  
The concept of the federated Cloud-Edge-IoT continuum promises to alleviate many woes of current systems, improving resource use, energy efficiency, quality of service, and more. However, this continuum is still far from being realized in practice, with no comprehensive solutions for developing, deploying, and managing continuum-native applications. Breakthrough innovations and novel system architectures are needed to cope with the ever-increasing heterogeneity and the multi-stakeholder nature of computing resources. This work proposes a novel architecture for choreographing workloads in the continuum, attempting to address these challenges. The architecture that tackles this issue comprehensively, spanning from the workloads themselves, through networking and data exchange, up to the orchestration and choreography mechanisms. The concept emphasizes the use of varied AI techniques, enabling autonomous and intelligent management of resources and workloads. Open standards are also a key part of the proposition, making it possible to fully engage third parties in multi-stakeholder scenarios. Although the presented architecture is promising, much work is required to realize it in practice. To this end, the key directions for future research are outlined.

{{</citation>}}


## cs.SE (1)



### (45/50) Understanding the Effectiveness of Large Language Models in Code Translation (Rangeet Pan et al., 2023)

{{<citation>}}

Rangeet Pan, Ali Reza Ibrahimzada, Rahul Krishna, Divya Sankar, Lambert Pouguem Wassi, Michele Merler, Boris Sobolev, Raju Pavuluri, Saurabh Sinha, Reyhaneh Jabbarvand. (2023)  
**Understanding the Effectiveness of Large Language Models in Code Translation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.03109v1)  

---


**ABSTRACT**  
Code translation aims to convert source code from one programming language (PL) to another. Given the promising abilities of large language models (LLMs) in code synthesis, researchers are actively exploring their potential to automate code translation, i.e., generating code in target PL from its equivalent in another PL. The pre-requisite for advancing the state of LLM-based code translation is to understand their limitations. To that end, we present a large-scale empirical study to investigate the ability of LLMs, including general LLMs and code LLMs, for code translation across pairs of different languages, including C, C++, Go, Java, and Python. Our analysis involves the translation of 1,700 code samples from three distinct benchmarks and real-world projects, revealing LLMs are yet to be reliably used to automate code translation -- with incorrect translations ranging from 52.7% to 97.9% across the studied LLMs. Further manual investigation of unsuccessful translations among all PLs identifies 14 root causes for translation bugs. Based on the insights from the empirical study, we propose a prompt-crafting approach to provide additional context for LLMs, improving the performance of LLM-based code translation by 5.5% on average across different PLs, LLMs, and benchmarks. Our study is the first of its kind, in terms of its scale and breadth, that provides insights into the current limitations of LLMs in code translation and opportunities for improving them. Our collected extensive dataset -- consisting of 1,700 code samples written in five PLs with 10K+ tests, 43K+ translated code, 1,725 manually labeled bugs, and 1,365 bug-fix pairs generated using LLMs -- can help drive research in this area.

{{</citation>}}


## cs.IT (1)



### (46/50) Achievable Information Rate Analysis in Diffusive Channels with Memory and Markov Source (Fardad Vakilipoor et al., 2023)

{{<citation>}}

Fardad Vakilipoor, Luca Barletta, Stefano Bregni, Maurizio Magarini. (2023)  
**Achievable Information Rate Analysis in Diffusive Channels with Memory and Markov Source**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.03042v1)  

---


**ABSTRACT**  
This paper explores the Achievable Information Rate (AIR) of a diffusive Molecular Communication (MC) channel featuring a fully absorbing receiver that counts the absorbed particles during symbol time intervals (STIs) and resets the counter at the start of each interval. The MC channel, influenced by memory effect, experiences inter-symbol interference (ISI) arising from the molecules' delayed arrival. The channel's memory is quantified as an integer multiple of the STI and a single-sample memoryless detector is employed to mitigate complexity in computing the mutual information (MI). To maximize MI, the detector threshold is optimized under Gaussian approximation of its input. The channel's MI is calculated, considering the influence of ISI, in the context of binary concentration shift keying modulation. Two distinct scenarios were considered; independent and correlated source-generated symbols, the latter modeled as a first-order Markov process. For each communication scenario, two degrees of knowledge: ISI-Aware and ISI-Unaware were considered. Remarkably, it is demonstrated that employing a correlated source enables the attainment of higher capacity. The results indicate that the capacity-achieving input distribution is not necessarily uniform. Notably, when the STI is small, corresponding to the case of strong ISI, the maximum AIR is not achieved through equiprobable symbol transmission.

{{</citation>}}


## cs.HC (1)



### (47/50) SAPIEN: Affective Virtual Agents Powered by Large Language Models (Masum Hasan et al., 2023)

{{<citation>}}

Masum Hasan, Cengiz Ozel, Sammy Potter, Ehsan Hoque. (2023)  
**SAPIEN: Affective Virtual Agents Powered by Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.03022v1)  

---


**ABSTRACT**  
In this demo paper, we introduce SAPIEN, a platform for high-fidelity virtual agents driven by large language models that can hold open domain conversations with users in 13 different languages, and display emotions through facial expressions and voice. The platform allows users to customize their virtual agent's personality, background, and conversation premise, thus providing a rich, immersive interaction experience. Furthermore, after the virtual meeting, the user can choose to get the conversation analyzed and receive actionable feedback on their communication skills. This paper illustrates an overview of the platform and discusses the various application domains of this technology, ranging from entertainment to mental health, communication training, language learning, education, healthcare, and beyond. Additionally, we consider the ethical implications of such realistic virtual agent representations and the potential challenges in ensuring responsible use.

{{</citation>}}


## cs.NI (1)



### (48/50) A Review of Gaps between Web 4.0 and Web 3.0 Intelligent Network Infrastructure (Zihan Zhou et al., 2023)

{{<citation>}}

Zihan Zhou, Zihao Li, Xiaoshuai Zhang, Yunqing Sun, Hao Xu. (2023)  
**A Review of Gaps between Web 4.0 and Web 3.0 Intelligent Network Infrastructure**  

---
Primary Category: cs.NI  
Categories: cs-CY, cs-DC, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02996v1)  

---


**ABSTRACT**  
World Wide Web is speeding up its pace into an intelligent and decentralized ecosystem, as seen in the campaign of Web 3.0 and forthcoming Web 4.0. Marked by the Europe Commission's latest mention of Web 4.0, a race towards strategic Web 4.0 success has started. Web 4.0 is committed to bringing the next technological transition with an open, secure, trustworthy fairness and digital ecosystem for individuals and businesses in private and public sectors. Despite overlapping scopes and objectives of Web 3.0 and Web 4.0 from academic and industrial perspectives, there are distinct and definitive features and gaps for the next generation of WWW. In this review, a brief introduction to WWW development unravels the entangled but consistent requirement of a more vivid web experience, enhancing human-centric experience in both societal and technical aspects. Moreover, the review brings a decentralized intelligence prospect of view on native AI entities for Web 4.0, envisioning sustainable, autonomous and decentralized AI services for the entire Web 4.0 environment, powering a self-sustainable Decentralized Physical and Software Infrastructure for Computing Force Network, Semantic Network, Virtual/Mixed Reality, and Privacy-preserving content presumption.   The review aims to reveal that Web 4.0 offers native intelligence with focused thinking on utilizing decentralized physical infrastructure, in addition to sole requirements on decentralization, bridging the gap between Web 4.0 and Web 3.0 advances with the latest future-shaping blockchain-enabled computing and network routing protocols.

{{</citation>}}


## cs.IR (1)



### (49/50) Decision Knowledge Graphs: Construction of and Usage in Question Answering for Clinical Practice Guidelines (Vasudhan Varma Kandula et al., 2023)

{{<citation>}}

Vasudhan Varma Kandula, Pushpak Bhattacharyya. (2023)  
**Decision Knowledge Graphs: Construction of and Usage in Question Answering for Clinical Practice Guidelines**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Clinical, Knowledge Graph, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.02984v1)  

---


**ABSTRACT**  
In the medical domain, several disease treatment procedures have been documented properly as a set of instructions known as Clinical Practice Guidelines (CPGs). CPGs have been developed over the years on the basis of past treatments, and are updated frequently. A doctor treating a particular patient can use these CPGs to know how past patients with similar conditions were treated successfully and can find the recommended treatment procedure. In this paper, we present a Decision Knowledge Graph (DKG) representation to store CPGs and to perform question-answering on CPGs. CPGs are very complex and no existing representation is suitable to perform question-answering and searching tasks on CPGs. As a result, doctors and practitioners have to manually wade through the guidelines, which is inefficient. Representation of CPGs is challenging mainly due to frequent updates on CPGs and decision-based structure. Our proposed DKG has a decision dimension added to a Knowledge Graph (KG) structure, purported to take care of decision based behavior of CPGs. Using this DKG has shown 40\% increase in accuracy compared to fine-tuned BioBert model in performing question-answering on CPGs. To the best of our knowledge, ours is the first attempt at creating DKGs and using them for representing CPGs.

{{</citation>}}


## cs.CR (1)



### (50/50) A Security and Usability Analysis of Local Attacks Against FIDO2 (Tarun Kumar Yadav et al., 2023)

{{<citation>}}

Tarun Kumar Yadav, Kent Seamons. (2023)  
**A Security and Usability Analysis of Local Attacks Against FIDO2**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.02973v1)  

---


**ABSTRACT**  
The FIDO2 protocol aims to strengthen or replace password authentication using public-key cryptography. FIDO2 has primarily focused on defending against attacks from afar by remote attackers that compromise a password or attempt to phish the user. In this paper, we explore threats from local attacks on FIDO2 that have received less attention -- a browser extension compromise and attackers gaining physical access to an HSK. Our systematic analysis of current implementations of FIDO2 reveals four underlying flaws, and we demonstrate the feasibility of seven attacks that exploit those flaws. The flaws include (1) Lack of confidentiality/integrity of FIDO2 messages accessible to browser extensions, (2) Broken clone detection algorithm, (3) Potential for user misunderstanding from social engineering and notification/error messages, and (4) Cookie life cycle. We build malicious browser extensions and demonstrate the attacks on ten popular web servers that use FIDO2. We also show that many browser extensions have sufficient permissions to conduct the attacks if they were compromised. A static and dynamic analysis of current browser extensions finds no evidence of the attacks in the wild. We conducted two user studies confirming that participants do not detect the attacks with current error messages, email notifications, and UX responses to the attacks. We provide an improved clone detection algorithm and recommendations for relying part

{{</citation>}}
