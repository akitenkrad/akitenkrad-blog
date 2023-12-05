---
draft: false
title: "arXiv @ 2023.12.05"
date: 2023-12-05
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.05"
    identifier: arxiv_20231205
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (10)](#cslg-10)
- [cs.CV (12)](#cscv-12)
- [cs.HC (1)](#cshc-1)
- [stat.ML (1)](#statml-1)
- [q-bio.BM (2)](#q-biobm-2)
- [cs.CL (9)](#cscl-9)
- [cs.CY (1)](#cscy-1)
- [cs.RO (4)](#csro-4)
- [cs.SD (1)](#cssd-1)
- [cs.DB (2)](#csdb-2)
- [cs.NI (1)](#csni-1)
- [eess.SY (1)](#eesssy-1)
- [cs.AI (2)](#csai-2)
- [cs.CR (1)](#cscr-1)
- [math.OC (1)](#mathoc-1)
- [cs.CE (1)](#csce-1)

## cs.LG (10)



### (1/50) Revisiting Non-separable Binary Classification and its Applications in Anomaly Detection (Matthew Lau et al., 2023)

{{<citation>}}

Matthew Lau, Ismaila Seck, Athanasios P Meliopoulos, Wenke Lee, Eugene Ndiaye. (2023)  
**Revisiting Non-separable Binary Classification and its Applications in Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: 68T37 (Primary), 68T07 (Secondary), I-2-6; I-5-1, cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.01541v1)  

---


**ABSTRACT**  
The inability to linearly classify XOR has motivated much of deep learning. We revisit this age-old problem and show that linear classification of XOR is indeed possible. Instead of separating data between halfspaces, we propose a slightly different paradigm, equality separation, that adapts the SVM objective to distinguish data within or outside the margin. Our classifier can then be integrated into neural network pipelines with a smooth approximation. From its properties, we intuit that equality separation is suitable for anomaly detection. To formalize this notion, we introduce closing numbers, a quantitative measure on the capacity for classifiers to form closed decision regions for anomaly detection. Springboarding from this theoretical connection between binary classification and anomaly detection, we test our hypothesis on supervised anomaly detection experiments, showing that equality separation can detect both seen and unseen anomalies.

{{</citation>}}


### (2/50) Recurrent Distance-Encoding Neural Networks for Graph Representation Learning (Yuhui Ding et al., 2023)

{{<citation>}}

Yuhui Ding, Antonio Orvieto, Bobby He, Thomas Hofmann. (2023)  
**Recurrent Distance-Encoding Neural Networks for Graph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.01538v1)  

---


**ABSTRACT**  
Graph neural networks based on iterative one-hop message passing have been shown to struggle in harnessing information from distant nodes effectively. Conversely, graph transformers allow each node to attend to all other nodes directly, but suffer from high computational complexity and have to rely on ad-hoc positional encoding to bake in the graph inductive bias. In this paper, we propose a new architecture to reconcile these challenges. Our approach stems from the recent breakthroughs in long-range modeling provided by deep state-space models on sequential data: for a given target node, our model aggregates other nodes by their shortest distances to the target and uses a parallelizable linear recurrent network over the chain of distances to provide a natural encoding of its neighborhood structure. With no need for positional encoding, we empirically show that the performance of our model is highly competitive compared with that of state-of-the-art graph transformers on various benchmarks, at a drastically reduced computational complexity. In addition, we show that our model is theoretically more expressive than one-hop message passing neural networks.

{{</citation>}}


### (3/50) Normed Spaces for Graph Embedding (Diaaeldin Taha et al., 2023)

{{<citation>}}

Diaaeldin Taha, Wei Zhao, J. Maxwell Riestenberg, Michael Strube. (2023)  
**Normed Spaces for Graph Embedding**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.01502v1)  

---


**ABSTRACT**  
Theoretical results from discrete geometry suggest that normed spaces can abstractly embed finite metric spaces with surprisingly low theoretical bounds on distortion in low dimensions. In this paper, inspired by this theoretical insight, we highlight normed spaces as a more flexible and computationally efficient alternative to several popular Riemannian manifolds for learning graph embeddings. Normed space embeddings significantly outperform several popular manifolds on a large range of synthetic and real-world graph reconstruction benchmark datasets while requiring significantly fewer computational resources. We also empirically verify the superiority of normed space embeddings on growing families of graphs associated with negative, zero, and positive curvature, further reinforcing the flexibility of normed spaces in capturing diverse graph structures as graph sizes increase. Lastly, we demonstrate the utility of normed space embeddings on two applied graph embedding tasks, namely, link prediction and recommender systems. Our work highlights the potential of normed spaces for geometric graph representation learning, raises new research questions, and offers a valuable tool for experimental mathematics in the field of finite metric space embeddings. We make our code and data publically available.

{{</citation>}}


### (4/50) ADT: Agent-based Dynamic Thresholding for Anomaly Detection (Xue Yang et al., 2023)

{{<citation>}}

Xue Yang, Enda Howley, Micheal Schukat. (2023)  
**ADT: Agent-based Dynamic Thresholding for Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.01488v1)  

---


**ABSTRACT**  
The complexity and scale of IT systems are increasing dramatically, posing many challenges to real-world anomaly detection. Deep learning anomaly detection has emerged, aiming at feature learning and anomaly scoring, which has gained tremendous success. However, little work has been done on the thresholding problem despite it being a critical factor for the effectiveness of anomaly detection. In this paper, we model thresholding in anomaly detection as a Markov Decision Process and propose an agent-based dynamic thresholding (ADT) framework based on a deep Q-network. The proposed method can be integrated into many systems that require dynamic thresholding. An auto-encoder is utilized in this study to obtain feature representations and produce anomaly scores for complex input data. ADT can adjust thresholds adaptively by utilizing the anomaly scores from the auto-encoder and significantly improve anomaly detection performance. The properties of ADT are studied through experiments on three real-world datasets and compared with benchmarks, hence demonstrating its thresholding capability, data-efficient learning, stability, and robustness. Our study validates the effectiveness of reinforcement learning in optimal thresholding control in anomaly detection.

{{</citation>}}


### (5/50) BenchMARL: Benchmarking Multi-Agent Reinforcement Learning (Matteo Bettini et al., 2023)

{{<citation>}}

Matteo Bettini, Amanda Prorok, Vincent Moens. (2023)  
**BenchMARL: Benchmarking Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.01472v1)  

---


**ABSTRACT**  
The field of Multi-Agent Reinforcement Learning (MARL) is currently facing a reproducibility crisis. While solutions for standardized reporting have been proposed to address the issue, we still lack a benchmarking tool that enables standardization and reproducibility, while leveraging cutting-edge Reinforcement Learning (RL) implementations. In this paper, we introduce BenchMARL, the first MARL training library created to enable standardized benchmarking across different algorithms, models, and environments. BenchMARL uses TorchRL as its backend, granting it high performance and maintained state-of-the-art implementations while addressing the broad community of MARL PyTorch users. Its design enables systematic configuration and reporting, thus allowing users to create and run complex benchmarks from simple one-line inputs. BenchMARL is open-sourced on GitHub: https://github.com/facebookresearch/BenchMARL

{{</citation>}}


### (6/50) Transformers are uninterpretable with myopic methods: a case study with bounded Dyck grammars (Kaiyue Wen et al., 2023)

{{<citation>}}

Kaiyue Wen, Yuchen Li, Bingbin Liu, Andrej Risteski. (2023)  
**Transformers are uninterpretable with myopic methods: a case study with bounded Dyck grammars**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.01429v1)  

---


**ABSTRACT**  
Interpretability methods aim to understand the algorithm implemented by a trained model (e.g., a Transofmer) by examining various aspects of the model, such as the weight matrices or the attention patterns. In this work, through a combination of theoretical results and carefully controlled experiments on synthetic data, we take a critical view of methods that exclusively focus on individual parts of the model, rather than consider the network as a whole. We consider a simple synthetic setup of learning a (bounded) Dyck language. Theoretically, we show that the set of models that (exactly or approximately) solve this task satisfy a structural characterization derived from ideas in formal languages (the pumping lemma). We use this characterization to show that the set of optima is qualitatively rich; in particular, the attention pattern of a single layer can be ``nearly randomized'', while preserving the functionality of the network. We also show via extensive experiments that these constructions are not merely a theoretical artifact: even after severely constraining the architecture of the model, vastly different solutions can be reached via standard training. Thus, interpretability claims based on inspecting individual heads or weight matrices in the Transformer can be misleading.

{{</citation>}}


### (7/50) Graph Coordinates and Conventional Neural Networks -- An Alternative for Graph Neural Networks (Zheyi Qin et al., 2023)

{{<citation>}}

Zheyi Qin, Randy Paffenroth, Anura P. Jayasumana. (2023)  
**Graph Coordinates and Conventional Neural Networks -- An Alternative for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.01342v1)  

---


**ABSTRACT**  
Graph-based data present unique challenges and opportunities for machine learning. Graph Neural Networks (GNNs), and especially those algorithms that capture graph topology through message passing for neighborhood aggregation, have been a leading solution. However, these networks often require substantial computational resources and may not optimally leverage the information contained in the graph's topology, particularly for large-scale or complex graphs. We propose Topology Coordinate Neural Network (TCNN) and Directional Virtual Coordinate Neural Network (DVCNN) as novel and efficient alternatives to message passing GNNs, that directly leverage the graph's topology, sidestepping the computational challenges presented by competing algorithms. Our proposed methods can be viewed as a reprise of classic techniques for graph embedding for neural network feature engineering, but they are novel in that our embedding techniques leverage ideas in Graph Coordinates (GC) that are lacking in current practice. Experimental results, benchmarked against the Open Graph Benchmark Leaderboard, demonstrate that TCNN and DVCNN achieve competitive or superior performance to message passing GNNs. For similar levels of accuracy and ROC-AUC, TCNN and DVCNN need far fewer trainable parameters than contenders of the OGBN Leaderboard. The proposed TCNN architecture requires fewer parameters than any neural network method currently listed in the OGBN Leaderboard for both OGBN-Proteins and OGBN-Products datasets. Conversely, our methods achieve higher performance for a similar number of trainable parameters. By providing an efficient and effective alternative to message passing GNNs, our work expands the toolbox of techniques for graph-based machine learning.

{{</citation>}}


### (8/50) Churn Prediction via Multimodal Fusion Learning:Integrating Customer Financial Literacy, Voice, and Behavioral Data (David Hason Rudd et al., 2023)

{{<citation>}}

David Hason Rudd, Huan Huo, Md Rafiqul Islam, Guandong Xu. (2023)  
**Churn Prediction via Multimodal Fusion Learning:Integrating Customer Financial Literacy, Voice, and Behavioral Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-CV, cs-HC, cs-LG, cs.LG  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2312.01301v1)  

---


**ABSTRACT**  
In todays competitive landscape, businesses grapple with customer retention. Churn prediction models, although beneficial, often lack accuracy due to the reliance on a single data source. The intricate nature of human behavior and high dimensional customer data further complicate these efforts. To address these concerns, this paper proposes a multimodal fusion learning model for identifying customer churn risk levels in financial service providers. Our multimodal approach integrates customer sentiments financial literacy (FL) level, and financial behavioral data, enabling more accurate and bias-free churn prediction models. The proposed FL model utilizes a SMOGN COREG supervised model to gauge customer FL levels from their financial data. The baseline churn model applies an ensemble artificial neural network and oversampling techniques to predict churn propensity in high-dimensional financial data. We also incorporate a speech emotion recognition model employing a pre-trained CNN-VGG16 to recognize customer emotions based on pitch, energy, and tone. To integrate these diverse features while retaining unique insights, we introduced late and hybrid fusion techniques that complementary boost coordinated multimodal co learning. Robust metrics were utilized to evaluate the proposed multimodal fusion model and hence the approach validity, including mean average precision and macro-averaged F1 score. Our novel approach demonstrates a marked improvement in churn prediction, achieving a test accuracy of 91.2%, a Mean Average Precision (MAP) score of 66, and a Macro-Averaged F1 score of 54 through the proposed hybrid fusion learning technique compared with late fusion and baseline models. Furthermore, the analysis demonstrates a positive correlation between negative emotions, low FL scores, and high-risk customers.

{{</citation>}}


### (9/50) Deep Ensembles Meets Quantile Regression: Uncertainty-aware Imputation for Time Series (Ying Liu et al., 2023)

{{<citation>}}

Ying Liu, Peng Cui, Wenbo Hu, Richang Hong. (2023)  
**Deep Ensembles Meets Quantile Regression: Uncertainty-aware Imputation for Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.01294v1)  

---


**ABSTRACT**  
Multivariate time series are everywhere. Nevertheless, real-world time series data often exhibit numerous missing values, which is the time series imputation task. Although previous deep learning methods have been shown to be effective for time series imputation, they are shown to produce overconfident imputations, which might be a potentially overlooked threat to the reliability of the intelligence system. Score-based diffusion method(i.e., CSDI) is effective for the time series imputation task but computationally expensive due to the nature of the generative diffusion model framework. In this paper, we propose a non-generative time series imputation method that produces accurate imputations with inherent uncertainty and meanwhile is computationally efficient. Specifically, we incorporate deep ensembles into quantile regression with a shared model backbone and a series of quantile discrimination functions.This framework combines the merits of accurate uncertainty estimation of deep ensembles and quantile regression and above all, the shared model backbone tremendously reduces most of the computation overhead of the multiple ensembles. We examine the performance of the proposed method on two real-world datasets: air quality and health-care datasets and conduct extensive experiments to show that our method excels at making deterministic and probabilistic predictions. Compared with the score-based diffusion method: CSDI, we can obtain comparable forecasting results and is better when more data is missing. Furthermore, as a non-generative model compared with CSDI, the proposed method consumes a much smaller computation overhead, yielding much faster training speed and fewer model parameters.

{{</citation>}}


### (10/50) Distributed Reinforcement Learning for Molecular Design: Antioxidant case (Huanyi Qin et al., 2023)

{{<citation>}}

Huanyi Qin, Denis Akhiyarov, Sophie Loehle, Kenneth Chiu, Mauricio Araya-Polo. (2023)  
**Distributed Reinforcement Learning for Molecular Design: Antioxidant case**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG, q-bio-BM  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.01267v1)  

---


**ABSTRACT**  
Deep reinforcement learning has successfully been applied for molecular discovery as shown by the Molecule Deep Q-network (MolDQN) algorithm. This algorithm has challenges when applied to optimizing new molecules: training such a model is limited in terms of scalability to larger datasets and the trained model cannot be generalized to different molecules in the same dataset. In this paper, a distributed reinforcement learning algorithm for antioxidants, called DA-MolDQN is proposed to address these problems. State-of-the-art bond dissociation energy (BDE) and ionization potential (IP) predictors are integrated into DA-MolDQN, which are critical chemical properties while optimizing antioxidants. Training time is reduced by algorithmic improvements for molecular modifications. The algorithm is distributed, scalable for up to 512 molecules, and generalizes the model to a diverse set of molecules. The proposed models are trained with a proprietary antioxidant dataset. The results have been reproduced with both proprietary and public datasets. The proposed molecules have been validated with DFT simulations and a subset of them confirmed in public "unseen" datasets. In summary, DA-MolDQN is up to 100x faster than previous algorithms and can discover new optimized molecules from proprietary and public antioxidants.

{{</citation>}}


## cs.CV (12)



### (11/50) Robust Computer Vision in an Ever-Changing World: A Survey of Techniques for Tackling Distribution Shifts (Eashan Adhikarla et al., 2023)

{{<citation>}}

Eashan Adhikarla, Kai Zhang, Jun Yu, Lichao Sun, John Nicholson, Brian D. Davison. (2023)  
**Robust Computer Vision in an Ever-Changing World: A Survey of Techniques for Tackling Distribution Shifts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.01540v1)  

---


**ABSTRACT**  
AI applications are becoming increasingly visible to the general public. There is a notable gap between the theoretical assumptions researchers make about computer vision models and the reality those models face when deployed in the real world. One of the critical reasons for this gap is a challenging problem known as distribution shift. Distribution shifts tend to vary with complexity of the data, dataset size, and application type. In our paper, we discuss the identification of such a prominent gap, exploring the concept of distribution shift and its critical significance. We provide an in-depth overview of various types of distribution shifts, elucidate their distinctions, and explore techniques within the realm of the data-centric domain employed to address them. Distribution shifts can occur during every phase of the machine learning pipeline, from the data collection stage to the stage of training a machine learning model to the stage of final model deployment. As a result, it raises concerns about the overall robustness of the machine learning techniques for computer vision applications that are deployed publicly for consumers. Different deep learning models each tailored for specific type of data and tasks, architectural pipelines; highlighting how variations in data preprocessing and feature extraction can impact robustness., data augmentation strategies (e.g. geometric, synthetic and learning-based); demonstrating their role in enhancing model generalization, and training mechanisms (e.g. transfer learning, zero-shot) fall under the umbrella of data-centric methods. Each of these components form an integral part of the neural-network we analyze contributing uniquely to strengthening model robustness against distribution shifts. We compare and contrast numerous AI models that are built for mitigating shifts in hidden stratification and spurious correlations, ...

{{</citation>}}


### (12/50) G2D: From Global to Dense Radiography Representation Learning via Vision-Language Pre-training (Che Liu et al., 2023)

{{<citation>}}

Che Liu, Cheng Ouyang, Sibo Cheng, Anand Shah, Wenjia Bai, Rossella Arcucci. (2023)  
**G2D: From Global to Dense Radiography Representation Learning via Vision-Language Pre-training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.01522v1)  

---


**ABSTRACT**  
Recently, medical vision-language pre-training (VLP) has reached substantial progress to learn global visual representation from medical images and their paired radiology reports. However, medical imaging tasks in real world usually require finer granularity in visual features. These tasks include visual localization tasks (e.g., semantic segmentation, object detection) and visual grounding task. Yet, current medical VLP methods face challenges in learning these fine-grained features, as they primarily focus on brute-force alignment between image patches and individual text tokens for local visual feature learning, which is suboptimal for downstream dense prediction tasks. In this work, we propose a new VLP framework, named \textbf{G}lobal to \textbf{D}ense level representation learning (G2D) that achieves significantly improved granularity and more accurate grounding for the learned features, compared to existing medical VLP approaches. In particular, G2D learns dense and semantically-grounded image representations via a pseudo segmentation task parallel with the global vision-language alignment. Notably, generating pseudo segmentation targets does not incur extra trainable parameters: they are obtained on the fly during VLP with a parameter-free processor. G2D achieves superior performance across 6 medical imaging tasks and 25 diseases, particularly in semantic segmentation, which necessitates fine-grained, semantically-grounded image features. In this task, G2D surpasses peer models even when fine-tuned with just 1\% of the training data, compared to the 100\% used by these models. The code will be released upon acceptance.

{{</citation>}}


### (13/50) Effectively Fine-tune to Improve Large Multimodal Models for Radiology Report Generation (Yuzhe Lu et al., 2023)

{{<citation>}}

Yuzhe Lu, Sungmin Hong, Yash Shah, Panpan Xu. (2023)  
**Effectively Fine-tune to Improve Large Multimodal Models for Radiology Report Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.01504v1)  

---


**ABSTRACT**  
Writing radiology reports from medical images requires a high level of domain expertise. It is time-consuming even for trained radiologists and can be error-prone for inexperienced radiologists. It would be appealing to automate this task by leveraging generative AI, which has shown drastic progress in vision and language understanding. In particular, Large Language Models (LLM) have demonstrated impressive capabilities recently and continued to set new state-of-the-art performance on almost all natural language tasks. While many have proposed architectures to combine vision models with LLMs for multimodal tasks, few have explored practical fine-tuning strategies. In this work, we proposed a simple yet effective two-stage fine-tuning protocol to align visual features to LLM's text embedding space as soft visual prompts. Our framework with OpenLLaMA-7B achieved state-of-the-art level performance without domain-specific pretraining. Moreover, we provide detailed analyses of soft visual prompts and attention mechanisms, shedding light on future research directions.

{{</citation>}}


### (14/50) GAPS: Geometry-Aware, Physics-Based, Self-Supervised Neural Garment Draping (Ruochen Chen et al., 2023)

{{<citation>}}

Ruochen Chen, Liming Chen, Shaifali Parashar. (2023)  
**GAPS: Geometry-Aware, Physics-Based, Self-Supervised Neural Garment Draping**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.01490v1)  

---


**ABSTRACT**  
Recent neural, physics-based modeling of garment deformations allows faster and visually aesthetic results as opposed to the existing methods. Material-specific parameters are used by the formulation to control the garment inextensibility. This delivers unrealistic results with physically implausible stretching. Oftentimes, the draped garment is pushed inside the body which is either corrected by an expensive post-processing, thus adding to further inconsistent stretching; or by deploying a separate training regime for each body type, restricting its scalability. Additionally, the flawed skinning process deployed by existing methods produces incorrect results on loose garments.   In this paper, we introduce a geometrical constraint to the existing formulation that is collision-aware and imposes garment inextensibility wherever possible. Thus, we obtain realistic results where draped clothes stretch only while covering bigger body regions. Furthermore, we propose a geometry-aware garment skinning method by defining a body-garment closeness measure which works for all garment types, especially the loose ones.

{{</citation>}}


### (15/50) Looking Inside Out: Anticipating Driver Intent From Videos (Yung-chi Kung et al., 2023)

{{<citation>}}

Yung-chi Kung, Arthur Zhang, Junmin Wang, Joydeep Biswas. (2023)  
**Looking Inside Out: Anticipating Driver Intent From Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.01444v1)  

---


**ABSTRACT**  
Anticipating driver intention is an important task when vehicles of mixed and varying levels of human/machine autonomy share roadways. Driver intention can be leveraged to improve road safety, such as warning surrounding vehicles in the event the driver is attempting a dangerous maneuver. In this work, we propose a novel method of utilizing in-cabin and external camera data to improve state-of-the-art (SOTA) performance in predicting future driver actions. Compared to existing methods, our approach explicitly extracts object and road-level features from external camera data, which we demonstrate are important features for predicting driver intention. Using our handcrafted features as inputs for both a transformer and an LSTM-based architecture, we empirically show that jointly utilizing in-cabin and external features improves performance compared to using in-cabin features alone. Furthermore, our models predict driver maneuvers more accurately and earlier than existing approaches, with an accuracy of 87.5% and an average prediction time of 4.35 seconds before the maneuver takes place. We release our model configurations and training scripts on https://github.com/ykung83/Driver-Intent-Prediction

{{</citation>}}


### (16/50) Automatic Report Generation for Histopathology images using pre-trained Vision Transformers and BERT (Saurav Sengupta et al., 2023)

{{<citation>}}

Saurav Sengupta, Donald E. Brown. (2023)  
**Automatic Report Generation for Histopathology images using pre-trained Vision Transformers and BERT**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BERT, BLEU, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.01435v1)  

---


**ABSTRACT**  
Deep learning for histopathology has been successfully used for disease classification, image segmentation and more. However, combining image and text modalities using current state-of-the-art methods has been a challenge due to the high resolution of histopathology images. Automatic report generation for histopathology images is one such challenge. In this work, we show that using an existing pre-trained Vision Transformer in a two-step process of first using it to encode 4096x4096 sized patches of the Whole Slide Image (WSI) and then using it as the encoder and a pre-trained Bidirectional Encoder Representations from Transformers (BERT) model for language modeling-based decoder for report generation, we can build a fairly performant and portable report generation mechanism that takes into account the whole of the high resolution image, instead of just the patches. Our method allows us to not only generate and evaluate captions that describe the image, but also helps us classify the image into tissue types and the gender of the patient as well. Our best performing model achieves a 79.98% accuracy in Tissue Type classification and 66.36% accuracy in classifying the sex of the patient the tissue came from, with a BLEU-4 score of 0.5818 in our caption generation task.

{{</citation>}}


### (17/50) D$^2$ST-Adapter: Disentangled-and-Deformable Spatio-Temporal Adapter for Few-shot Action Recognition (Wenjie Pei et al., 2023)

{{<citation>}}

Wenjie Pei, Qizhong Tan, Guangming Lu, Jiandong Tian. (2023)  
**D$^2$ST-Adapter: Disentangled-and-Deformable Spatio-Temporal Adapter for Few-shot Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.01431v1)  

---


**ABSTRACT**  
Adapting large pre-trained image models to few-shot action recognition has proven to be an effective and efficient strategy for learning robust feature extractors, which is essential for few-shot learning. Typical fine-tuning based adaptation paradigm is prone to overfitting in the few-shot learning scenarios and offers little modeling flexibility for learning temporal features in video data. In this work we present the Disentangled-and-Deformable Spatio-Temporal Adapter (D$^2$ST-Adapter), a novel adapter tuning framework for few-shot action recognition, which is designed in a dual-pathway architecture to encode spatial and temporal features in a disentangled manner. Furthermore, we devise the Deformable Spatio-Temporal Attention module as the core component of D$^2$ST-Adapter, which can be tailored to model both spatial and temporal features in corresponding pathways, allowing our D$^2$ST-Adapter to encode features in a global view in 3D spatio-temporal space while maintaining a lightweight design. Extensive experiments with instantiations of our method on both pre-trained ResNet and ViT demonstrate the superiority of our method over state-of-the-art methods for few-shot action recognition. Our method is particularly well-suited to challenging scenarios where temporal dynamics are critical for action recognition.

{{</citation>}}


### (18/50) Facial Emotion Recognition Under Mask Coverage Using a Data Augmentation Technique (Aref Farhadipour et al., 2023)

{{<citation>}}

Aref Farhadipour, Pouya Taghipour. (2023)  
**Facial Emotion Recognition Under Mask Coverage Using a Data Augmentation Technique**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-HC, cs.CV  
Keywords: AI, Augmentation, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2312.01335v1)  

---


**ABSTRACT**  
Identifying human emotions using AI-based computer vision systems, when individuals wear face masks, presents a new challenge in the current Covid-19 pandemic. In this study, we propose a facial emotion recognition system capable of recognizing emotions from individuals wearing different face masks. A novel data augmentation technique was utilized to improve the performance of our model using four mask types for each face image. We evaluated the effectiveness of four convolutional neural networks, Alexnet, Squeezenet, Resnet50 and VGGFace2 that were trained using transfer learning. The experimental findings revealed that our model works effectively in multi-mask mode compared to single-mask mode. The VGGFace2 network achieved the highest accuracy rate, with 97.82% for the person-dependent mode and 74.21% for the person-independent mode using the JAFFE dataset. However, we evaluated our proposed model using the UIBVFED dataset. The Resnet50 has demonstrated superior performance, with accuracies of 73.68% for the person-dependent mode and 59.57% for the person-independent mode. Moreover, we employed metrics such as precision, sensitivity, specificity, AUC, F1 score, and confusion matrix to measure our system's efficiency in detail. Additionally, the LIME algorithm was used to visualize CNN's decision-making strategy.

{{</citation>}}


### (19/50) MABViT -- Modified Attention Block Enhances Vision Transformers (Mahesh Ramesh et al., 2023)

{{<citation>}}

Mahesh Ramesh, Aswinkumar Ramkumar. (2023)  
**MABViT -- Modified Attention Block Enhances Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention, ImageNet, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.01324v1)  

---


**ABSTRACT**  
Recent studies have demonstrated the effectiveness of Gated Linear Units (GLU) in enhancing transformer models, particularly in Large Language Models (LLMs). Additionally, utilizing a parallel configuration within each Transformer block rather than the conventional serialized method has been revealed to accelerate the training of LLMs without significantly impacting performance. However, when the MLP and attention block were run in parallel for the image classification task, we observed a noticeable decline in performance. We propose a novel transformer variant that integrates non-linearity within the attention block to tackle this problem. We implemented the GLU-based activation function on the Value tensor, and this new technique surpasses the current state-of-the-art S/16 variant of Vision Transformers by 0.6% on the ImageNet-1K dataset while utilizing fewer parameters. It also supersedes the B/16 variant while using only half the parameters. Furthermore, we provide results with the GELU activation function variant to confirm our assertions. Lastly, we showcase that the MABViT variants exhibit greater potential when utilized in deep transformers compared to the standard architecture.

{{</citation>}}


### (20/50) Deeper into Self-Supervised Monocular Indoor Depth Estimation (Chao Fan et al., 2023)

{{<citation>}}

Chao Fan, Zhenyu Yin, Yue Li, Feiqing Zhang. (2023)  
**Deeper into Self-Supervised Monocular Indoor Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.01283v1)  

---


**ABSTRACT**  
Monocular depth estimation using Convolutional Neural Networks (CNNs) has shown impressive performance in outdoor driving scenes. However, self-supervised learning of indoor depth from monocular sequences is quite challenging for researchers because of the following two main reasons. One is the large areas of low-texture regions and the other is the complex ego-motion on indoor training datasets. In this work, our proposed method, named IndoorDepth, consists of two innovations. In particular, we first propose a novel photometric loss with improved structural similarity (SSIM) function to tackle the challenge from low-texture regions. Moreover, in order to further mitigate the issue of inaccurate ego-motion prediction, multiple photometric losses at different stages are used to train a deeper pose network with two residual pose blocks. Subsequent ablation study can validate the effectiveness of each new idea. Experiments on the NYUv2 benchmark demonstrate that our IndoorDepth outperforms the previous state-of-the-art methods by a large margin. In addition, we also validate the generalization ability of our method on ScanNet dataset. Code is availabe at https://github.com/fcntes/IndoorDepth.

{{</citation>}}


### (21/50) Learning to Compose SuperWeights for Neural Parameter Allocation Search (Piotr Teterwak et al., 2023)

{{<citation>}}

Piotr Teterwak, Soren Nelson, Nikoli Dryden, Dina Bashkirova, Kate Saenko, Bryan A. Plummer. (2023)  
**Learning to Compose SuperWeights for Neural Parameter Allocation Search**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.01274v1)  

---


**ABSTRACT**  
Neural parameter allocation search (NPAS) automates parameter sharing by obtaining weights for a network given an arbitrary, fixed parameter budget. Prior work has two major drawbacks we aim to address. First, there is a disconnect in the sharing pattern between the search and training steps, where weights are warped for layers of different sizes during the search to measure similarity, but not during training, resulting in reduced performance. To address this, we generate layer weights by learning to compose sets of SuperWeights, which represent a group of trainable parameters. These SuperWeights are created to be large enough so they can be used to represent any layer in the network, but small enough that they are computationally efficient. The second drawback we address is the method of measuring similarity between shared parameters. Whereas prior work compared the weights themselves, we argue this does not take into account the amount of conflict between the shared weights. Instead, we use gradient information to identify layers with shared weights that wish to diverge from each other. We demonstrate that our SuperWeight Networks consistently boost performance over the state-of-the-art on the ImageNet and CIFAR datasets in the NPAS setting. We further show that our approach can generate parameters for many network architectures using the same set of weights. This enables us to support tasks like efficient ensembling and anytime prediction, outperforming fully-parameterized ensembles with 17% fewer parameters.

{{</citation>}}


### (22/50) TIBET: Identifying and Evaluating Biases in Text-to-Image Generative Models (Aditya Chinchure et al., 2023)

{{<citation>}}

Aditya Chinchure, Pushkar Shukla, Gaurav Bhatt, Kiri Salij, Kartik Hosanagar, Leonid Sigal, Matthew Turk. (2023)  
**TIBET: Identifying and Evaluating Biases in Text-to-Image Generative Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-CY, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.01261v1)  

---


**ABSTRACT**  
Text-to-Image (TTI) generative models have shown great progress in the past few years in terms of their ability to generate complex and high-quality imagery. At the same time, these models have been shown to suffer from harmful biases, including exaggerated societal biases (e.g., gender, ethnicity), as well as incidental correlations that limit such model's ability to generate more diverse imagery. In this paper, we propose a general approach to study and quantify a broad spectrum of biases, for any TTI model and for any prompt, using counterfactual reasoning. Unlike other works that evaluate generated images on a predefined set of bias axes, our approach automatically identifies potential biases that might be relevant to the given prompt, and measures those biases. In addition, our paper extends quantitative scores with post-hoc explanations in terms of semantic concepts in the images generated. We show that our method is uniquely capable of explaining complex multi-dimensional biases through semantic concepts, as well as the intersectionality between different biases for any given prompt. We perform extensive user studies to illustrate that the results of our method and analysis are consistent with human judgements.

{{</citation>}}


## cs.HC (1)



### (23/50) Using Large Language Models to Accelerate Communication for Users with Severe Motor Impairments (Shanqing Cai et al., 2023)

{{<citation>}}

Shanqing Cai, Subhashini Venugopalan, Katie Seaver, Xiang Xiao, Katrin Tomanek, Sri Jalasutram, Meredith Ringel Morris, Shaun Kane, Ajit Narayanan, Robert L. MacDonald, Emily Kornman, Daniel Vance, Blair Casey, Steve M. Gleason, Philip Q. Nelson, Michael P. Brenner. (2023)  
**Using Large Language Models to Accelerate Communication for Users with Severe Motor Impairments**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.01532v1)  

---


**ABSTRACT**  
Finding ways to accelerate text input for individuals with profound motor impairments has been a long-standing area of research. Closing the speed gap for augmentative and alternative communication (AAC) devices such as eye-tracking keyboards is important for improving the quality of life for such individuals. Recent advances in neural networks of natural language pose new opportunities for re-thinking strategies and user interfaces for enhanced text-entry for AAC users. In this paper, we present SpeakFaster, consisting of large language models (LLMs) and a co-designed user interface for text entry in a highly-abbreviated form, allowing saving 57% more motor actions than traditional predictive keyboards in offline simulation. A pilot study with 19 non-AAC participants typing on a mobile device by hand demonstrated gains in motor savings in line with the offline simulation, while introducing relatively small effects on overall typing speed. Lab and field testing on two eye-gaze typing users with amyotrophic lateral sclerosis (ALS) demonstrated text-entry rates 29-60% faster than traditional baselines, due to significant saving of expensive keystrokes achieved through phrase and word predictions from context-aware LLMs. These findings provide a strong foundation for further exploration of substantially-accelerated text communication for motor-impaired users and demonstrate a direction for applying LLMs to text-based user interfaces.

{{</citation>}}


## stat.ML (1)



### (24/50) Evaluation of Active Feature Acquisition Methods for Time-varying Feature Settings (Henrik von Kleist et al., 2023)

{{<citation>}}

Henrik von Kleist, Alireza Zamanian, Ilya Shpitser, Narges Ahmidi. (2023)  
**Evaluation of Active Feature Acquisition Methods for Time-varying Feature Settings**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.01530v1)  

---


**ABSTRACT**  
Machine learning methods often assume input features are available at no cost. However, in domains like healthcare, where acquiring features could be expensive or harmful, it is necessary to balance a feature's acquisition cost against its predictive value. The task of training an AI agent to decide which features to acquire is called active feature acquisition (AFA). By deploying an AFA agent, we effectively alter the acquisition strategy and trigger a distribution shift. To safely deploy AFA agents under this distribution shift, we present the problem of active feature acquisition performance evaluation (AFAPE). We examine AFAPE under i) a no direct effect (NDE) assumption, stating that acquisitions don't affect the underlying feature values; and ii) a no unobserved confounding (NUC) assumption, stating that retrospective feature acquisition decisions were only based on observed features. We show that one can apply offline reinforcement learning under the NUC assumption and missing data methods under the NDE assumption. When NUC and NDE hold, we propose a novel semi-offline reinforcement learning framework, which requires a weaker positivity assumption and yields more data-efficient estimators. We introduce three novel estimators: a direct method (DM), an inverse probability weighting (IPW), and a double reinforcement learning (DRL) estimator.

{{</citation>}}


## q-bio.BM (2)



### (25/50) NovoMol: Recurrent Neural Network for Orally Bioavailable Drug Design and Validation on PDGFRα Receptor (Ishir Rao, 2023)

{{<citation>}}

Ishir Rao. (2023)  
**NovoMol: Recurrent Neural Network for Orally Bioavailable Drug Design and Validation on PDGFRα Receptor**  

---
Primary Category: q-bio.BM  
Categories: cs-AI, q-bio-BM, q-bio-QM, q-bio.BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.01527v1)  

---


**ABSTRACT**  
Longer timelines and lower success rates of drug candidates limit the productivity of clinical trials in the pharmaceutical industry. Promising de novo drug design techniques help solve this by exploring a broader chemical space, efficiently generating new molecules, and providing improved therapies. However, optimizing for molecular characteristics found in approved oral drugs remains a challenge, limiting de novo usage. In this work, we propose NovoMol, a novel de novo method using recurrent neural networks to mass-generate drug molecules with high oral bioavailability, increasing clinical trial time efficiency. Molecules were optimized for desirable traits and ranked using the quantitative estimate of drug-likeness (QED). Generated molecules meeting QED's oral bioavailability threshold were used to retrain the neural network, and, after five training cycles, 76% of generated molecules passed this strict threshold and 96% passed the traditionally used Lipinski's Rule of Five. The trained model was then used to generate specific drug candidates for the cancer-related PDGFR{\alpha} receptor and 44% of generated candidates had better binding affinity than the current state-of-the-art drug, Imatinib (with a receptor binding affinity of -9.4 kcal/mol), and the best-generated candidate at -12.9 kcal/mol. NovoMol provides a time/cost-efficient AI-based de novo method offering promising drug candidates for clinical trials.

{{</citation>}}


### (26/50) Multiscale Topology in Interactomic Network: From Transcriptome to Antiaddiction Drug Repurposing (Hongyan Du et al., 2023)

{{<citation>}}

Hongyan Du, Guo-Wei Wei, Tingjun Hou. (2023)  
**Multiscale Topology in Interactomic Network: From Transcriptome to Antiaddiction Drug Repurposing**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio-GN, q-bio.BM  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.01272v1)  

---


**ABSTRACT**  
The escalating drug addiction crisis in the United States underscores the urgent need for innovative therapeutic strategies. This study embarked on an innovative and rigorous strategy to unearth potential drug repurposing candidates for opioid and cocaine addiction treatment, bridging the gap between transcriptomic data analysis and drug discovery. We initiated our approach by conducting differential gene expression analysis on addiction-related transcriptomic data to identify key genes. We propose a novel topological differentiation to identify key genes from a protein-protein interaction (PPI) network derived from DEGs. This method utilizes persistent Laplacians to accurately single out pivotal nodes within the network, conducting this analysis in a multiscale manner to ensure high reliability. Through rigorous literature validation, pathway analysis, and data-availability scrutiny, we identified three pivotal molecular targets, mTOR, mGluR5, and NMDAR, for drug repurposing from DrugBank. We crafted machine learning models employing two natural language processing (NLP)-based embeddings and a traditional 2D fingerprint, which demonstrated robust predictive ability in gauging binding affinities of DrugBank compounds to selected targets. Furthermore, we elucidated the interactions of promising drugs with the targets and evaluated their drug-likeness. This study delineates a multi-faceted and comprehensive analytical framework, amalgamating bioinformatics, topological data analysis and machine learning, for drug repurposing in addiction treatment, setting the stage for subsequent experimental validation. The versatility of the methods we developed allows for applications across a range of diseases and transcriptomic datasets.

{{</citation>}}


## cs.CL (9)



### (27/50) SymNoise: Advancing Language Model Fine-tuning with Symmetric Noise (Arjun Singh et al., 2023)

{{<citation>}}

Arjun Singh, Abhay Kumar Yadav. (2023)  
**SymNoise: Advancing Language Model Fine-tuning with Symmetric Noise**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.01523v1)  

---


**ABSTRACT**  
In this paper, we introduce a novel fine-tuning technique for language models, which involves incorporating symmetric noise into the embedding process. This method aims to enhance the model's function by more stringently regulating its local curvature, demonstrating superior performance over the current method, NEFTune. When fine-tuning the LLaMA-2-7B model using Alpaca, standard techniques yield a 29.79% score on AlpacaEval. However, our approach, SymNoise, increases this score significantly to 69.04%, using symmetric noisy embeddings. This is a 6.7% improvement over the state-of-the-art method, NEFTune~(64.69%). Furthermore, when tested on various models and stronger baseline instruction datasets, such as Evol-Instruct, ShareGPT, OpenPlatypus, SymNoise consistently outperforms NEFTune. The current literature, including NEFTune, has underscored the importance of more in-depth research into the application of noise-based strategies in the fine-tuning of language models. Our approach, SymNoise, is another significant step towards this direction, showing notable improvement over the existing state-of-the-art method.

{{</citation>}}


### (28/50) Unsupervised Approach to Evaluate Sentence-Level Fluency: Do We Really Need Reference? (Gopichand Kanumolu et al., 2023)

{{<citation>}}

Gopichand Kanumolu, Lokesh Madasu, Pavan Baswani, Ananya Mukherjee, Manish Shrivastava. (2023)  
**Unsupervised Approach to Evaluate Sentence-Level Fluency: Do We Really Need Reference?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2312.01500v1)  

---


**ABSTRACT**  
Fluency is a crucial goal of all Natural Language Generation (NLG) systems. Widely used automatic evaluation metrics fall short in capturing the fluency of machine-generated text. Assessing the fluency of NLG systems poses a challenge since these models are not limited to simply reusing words from the input but may also generate abstractions. Existing reference-based fluency evaluations, such as word overlap measures, often exhibit weak correlations with human judgments. This paper adapts an existing unsupervised technique for measuring text fluency without the need for any reference. Our approach leverages various word embeddings and trains language models using Recurrent Neural Network (RNN) architectures. We also experiment with other available multilingual Language Models (LMs). To assess the performance of the models, we conduct a comparative analysis across 10 Indic languages, correlating the obtained fluency scores with human judgments. Our code and human-annotated benchmark test-set for fluency is available at https://github.com/AnanyaCoder/TextFluencyForIndicLanaguges.

{{</citation>}}


### (29/50) Towards Mitigating Perceived Unfairness in Contracts from a Non-Legal Stakeholder's Perspective (Anmol Singhal et al., 2023)

{{<citation>}}

Anmol Singhal, Preethu Rose Anish, Shirish Karande, Smita Ghaisas. (2023)  
**Towards Mitigating Perceived Unfairness in Contracts from a Non-Legal Stakeholder's Perspective**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2312.01398v1)  

---


**ABSTRACT**  
Commercial contracts are known to be a valuable source for deriving project-specific requirements. However, contract negotiations mainly occur among the legal counsel of the parties involved. The participation of non-legal stakeholders, including requirement analysts, engineers, and solution architects, whose primary responsibility lies in ensuring the seamless implementation of contractual terms, is often indirect and inadequate. Consequently, a significant number of sentences in contractual clauses, though legally accurate, can appear unfair from an implementation perspective to non-legal stakeholders. This perception poses a problem since requirements indicated in the clauses are obligatory and can involve punitive measures and penalties if not implemented as committed in the contract. Therefore, the identification of potentially unfair clauses in contracts becomes crucial. In this work, we conduct an empirical study to analyze the perspectives of different stakeholders regarding contractual fairness. We then investigate the ability of Pre-trained Language Models (PLMs) to identify unfairness in contractual sentences by comparing chain of thought prompting and semi-supervised fine-tuning approaches. Using BERT-based fine-tuning, we achieved an accuracy of 84% on a dataset consisting of proprietary contracts. It outperformed chain of thought prompting using Vicuna-13B by a margin of 9%.

{{</citation>}}


### (30/50) CEScore: Simple and Efficient Confidence Estimation Model for Evaluating Split and Rephrase (AlMotasem Bellah Al Ajlouni et al., 2023)

{{<citation>}}

AlMotasem Bellah Al Ajlouni, Jinlong Li. (2023)  
**CEScore: Simple and Efficient Confidence Estimation Model for Evaluating Split and Rephrase**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.01356v1)  

---


**ABSTRACT**  
The split and rephrase (SR) task aims to divide a long, complex sentence into a set of shorter, simpler sentences that convey the same meaning. This challenging problem in NLP has gained increased attention recently because of its benefits as a pre-processing step in other NLP tasks. Evaluating quality of SR is challenging, as there no automatic metric fit to evaluate this task. In this work, we introduce CEScore, as novel statistical model to automatically evaluate SR task. By mimicking the way humans evaluate SR, CEScore provides 4 metrics (Sscore, Gscore, Mscore, and CEscore) to assess simplicity, grammaticality, meaning preservation, and overall quality, respectively. In experiments with 26 models, CEScore correlates strongly with human evaluations, achieving 0.98 in Spearman correlations at model-level. This underscores the potential of CEScore as a simple and effective metric for assessing the overall quality of SR models.

{{</citation>}}


### (31/50) AI-Powered Arabic Crossword Puzzle Generation for Educational Applications (Kamyar Zeinalipour et al., 2023)

{{<citation>}}

Kamyar Zeinalipour, Mohamed Zaky Saad, Marco Maggini, Marco Gori. (2023)  
**AI-Powered Arabic Crossword Puzzle Generation for Educational Applications**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, BERT, GPT  
[Paper Link](http://arxiv.org/abs/2312.01339v1)  

---


**ABSTRACT**  
This paper presents the first Arabic crossword puzzle generator driven by advanced AI technology. Leveraging cutting-edge large language models including GPT4, GPT3-Davinci, GPT3-Curie, GPT3-Babbage, GPT3-Ada, and BERT, the system generates distinctive and challenging clues. Based on a dataset comprising over 50,000 clue-answer pairs, the generator employs fine-tuning, few/zero-shot learning strategies, and rigorous quality-checking protocols to enforce the generation of high-quality clue-answer pairs. Importantly, educational crosswords contribute to enhancing memory, expanding vocabulary, and promoting problem-solving skills, thereby augmenting the learning experience through a fun and engaging approach, reshaping the landscape of traditional learning methods. The overall system can be exploited as a powerful educational tool that amalgamates AI and innovative learning techniques, heralding a transformative era for Arabic crossword puzzles and the intersection of technology and education.

{{</citation>}}


### (32/50) NLEBench+NorGLM: A Comprehensive Empirical Analysis and Benchmark Dataset for Generative Language Models in Norwegian (Peng Liu et al., 2023)

{{<citation>}}

Peng Liu, Lemei Zhang, Terje Nissen Farup, Even W. Lauvrak, Jon Espen Ingvaldsen, Simen Eide, Jon Atle Gulla, Zhirong Yang. (2023)  
**NLEBench+NorGLM: A Comprehensive Empirical Analysis and Benchmark Dataset for Generative Language Models in Norwegian**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GLM, Language Model, NLP, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2312.01314v1)  

---


**ABSTRACT**  
Recent advancements in Generative Language Models (GLMs) have transformed Natural Language Processing (NLP) by showcasing the effectiveness of the "pre-train, prompt, and predict" paradigm in utilizing pre-trained GLM knowledge for diverse applications. Despite their potential, these capabilities lack adequate quantitative characterization due to the absence of comprehensive benchmarks, particularly for low-resource languages. Existing low-resource benchmarks focus on discriminative language models like BERT, neglecting the evaluation of generative language models. Moreover, current benchmarks often overlook measuring generalization performance across multiple tasks, a crucial metric for GLMs.   To bridge these gaps, we introduce NLEBench, a comprehensive benchmark tailored for evaluating natural language generation capabilities in Norwegian, a low-resource language. We use Norwegian as a case study to explore whether current GLMs and benchmarks in mainstream languages like English can reveal the unique characteristics of underrepresented languages. NLEBench encompasses a suite of real-world NLP tasks ranging from news storytelling, summarization, open-domain conversation, natural language understanding, instruction fine-tuning, toxicity and bias evaluation, to self-curated Chain-of-Thought investigation. It features two high-quality, human-annotated datasets: an instruction dataset covering traditional Norwegian cultures, idioms, slang, and special expressions, and a document-grounded multi-label dataset for topic classification, question answering, and summarization. This paper also introduces foundational Norwegian Generative Language Models (NorGLMs) developed with diverse parameter scales and Transformer-based architectures. Systematic evaluations on the proposed benchmark suite provide insights into the capabilities and scalability of NorGLMs across various downstream tasks.

{{</citation>}}


### (33/50) Bridging Background Knowledge Gaps in Translation with Automatic Explicitation (HyoJung Han et al., 2023)

{{<citation>}}

HyoJung Han, Jordan Lee Boyd-Graber, Marine Carpuat. (2023)  
**Bridging Background Knowledge Gaps in Translation with Automatic Explicitation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.01308v1)  

---


**ABSTRACT**  
Translations help people understand content written in another language. However, even correct literal translations do not fulfill that goal when people lack the necessary background to understand them. Professional translators incorporate explicitations to explain the missing context by considering cultural differences between source and target audiences. Despite its potential to help users, NLP research on explicitation is limited because of the dearth of adequate evaluation methods. This work introduces techniques for automatically generating explicitations, motivated by WikiExpl: a dataset that we collect from Wikipedia and annotate with human translators. The resulting explicitations are useful as they help answer questions more accurately in a multilingual question answering framework.

{{</citation>}}


### (34/50) On Significance of Subword tokenization for Low Resource and Efficient Named Entity Recognition: A case study in Marathi (Harsh Chaudhari et al., 2023)

{{<citation>}}

Harsh Chaudhari, Anuja Patil, Dhanashree Lavekar, Pranav Khairnar, Raviraj Joshi, Sachin Pande. (2023)  
**On Significance of Subword tokenization for Low Resource and Efficient Named Entity Recognition: A case study in Marathi**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, GPT, LSTM, NER, NLP, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2312.01306v1)  

---


**ABSTRACT**  
Named Entity Recognition (NER) systems play a vital role in NLP applications such as machine translation, summarization, and question-answering. These systems identify named entities, which encompass real-world concepts like locations, persons, and organizations. Despite extensive research on NER systems for the English language, they have not received adequate attention in the context of low resource languages. In this work, we focus on NER for low-resource language and present our case study in the context of the Indian language Marathi. The advancement of NLP research revolves around the utilization of pre-trained transformer models such as BERT for the development of NER models. However, we focus on improving the performance of shallow models based on CNN, and LSTM by combining the best of both worlds. In the era of transformers, these traditional deep learning models are still relevant because of their high computational efficiency. We propose a hybrid approach for efficient NER by integrating a BERT-based subword tokenizer into vanilla CNN/LSTM models. We show that this simple approach of replacing a traditional word-based tokenizer with a BERT-tokenizer brings the accuracy of vanilla single-layer models closer to that of deep pre-trained models like BERT. We show the importance of using sub-word tokenization for NER and present our study toward building efficient NLP systems. The evaluation is performed on L3Cube-MahaNER dataset using tokenizers from MahaBERT, MahaGPT, IndicBERT, and mBERT.

{{</citation>}}


### (35/50) TextGenSHAP: Scalable Post-hoc Explanations in Text Generation with Long Documents (James Enouen et al., 2023)

{{<citation>}}

James Enouen, Hootan Nakhost, Sayna Ebrahimi, Sercan O Arik, Yan Liu, Tomas Pfister. (2023)  
**TextGenSHAP: Scalable Post-hoc Explanations in Text Generation with Long Documents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2312.01279v1)  

---


**ABSTRACT**  
Large language models (LLMs) have attracted huge interest in practical applications given their increasingly accurate responses and coherent reasoning abilities. Given their nature as black-boxes using complex reasoning processes on their inputs, it is inevitable that the demand for scalable and faithful explanations for LLMs' generated content will continue to grow. There have been major developments in the explainability of neural network models over the past decade. Among them, post-hoc explainability methods, especially Shapley values, have proven effective for interpreting deep learning models. However, there are major challenges in scaling up Shapley values for LLMs, particularly when dealing with long input contexts containing thousands of tokens and autoregressively generated output sequences. Furthermore, it is often unclear how to effectively utilize generated explanations to improve the performance of LLMs. In this paper, we introduce TextGenSHAP, an efficient post-hoc explanation method incorporating LM-specific techniques. We demonstrate that this leads to significant increases in speed compared to conventional Shapley value computations, reducing processing times from hours to minutes for token-level explanations, and to just seconds for document-level explanations. In addition, we demonstrate how real-time Shapley values can be utilized in two important scenarios, providing better understanding of long-document question answering by localizing important words and sentences; and improving existing document retrieval systems through enhancing the accuracy of selected passages and ultimately the final responses.

{{</citation>}}


## cs.CY (1)



### (36/50) Tackling Bias in Pre-trained Language Models: Current Trends and Under-represented Societies (Vithya Yogarajan et al., 2023)

{{<citation>}}

Vithya Yogarajan, Gillian Dobbie, Te Taka Keegan, Rostam J. Neuwirth. (2023)  
**Tackling Bias in Pre-trained Language Models: Current Trends and Under-represented Societies**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs.CY  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2312.01509v1)  

---


**ABSTRACT**  
The benefits and capabilities of pre-trained language models (LLMs) in current and future innovations are vital to any society. However, introducing and using LLMs comes with biases and discrimination, resulting in concerns about equality, diversity and fairness, and must be addressed. While understanding and acknowledging bias in LLMs and developing mitigation strategies are crucial, the generalised assumptions towards societal needs can result in disadvantages towards under-represented societies and indigenous populations. Furthermore, the ongoing changes to actual and proposed amendments to regulations and laws worldwide also impact research capabilities in tackling the bias problem. This research presents a comprehensive survey synthesising the current trends and limitations in techniques used for identifying and mitigating bias in LLMs, where the overview of methods for tackling bias are grouped into metrics, benchmark datasets, and mitigation strategies. The importance and novelty of this survey are that it explores the perspective of under-represented societies. We argue that current practices tackling the bias problem cannot simply be 'plugged in' to address the needs of under-represented societies. We use examples from New Zealand to present requirements for adopting existing techniques to under-represented societies.

{{</citation>}}


## cs.RO (4)



### (37/50) Learning Neural Traffic Rules (Xuan Zhang et al., 2023)

{{<citation>}}

Xuan Zhang, Xifeng Gao, Kui Wu, Zherong Pan. (2023)  
**Learning Neural Traffic Rules**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.01498v1)  

---


**ABSTRACT**  
Extensive research has been devoted to the field of multi-agent navigation. Recently, there has been remarkable progress attributed to the emergence of learning-based techniques with substantially elevated intelligence and realism. Nonetheless, prevailing learned models face limitations in terms of scalability and effectiveness, primarily due to their agent-centric nature, i.e., the learned neural policy is individually deployed on each agent. Inspired by the efficiency observed in real-world traffic networks, we present an environment-centric navigation policy. Our method learns a set of traffic rules to coordinate a vast group of unintelligent agents that possess only basic collision-avoidance capabilities. Our method segments the environment into distinct blocks and parameterizes the traffic rule using a Graph Recurrent Neural Network (GRNN) over the block network. Each GRNN node is trained to modulate the velocities of agents as they traverse through. Using either Imitation Learning (IL) or Reinforcement Learning (RL) schemes, we demonstrate the efficacy of our neural traffic rules in resolving agent congestion, closely resembling real-world traffic regulations. Our method handles up to $240$ agents at real-time and generalizes across diverse agent and environment configurations.

{{</citation>}}


### (38/50) Distilling Functional Rearrangement Priors from Large Models (Yiming Zeng et al., 2023)

{{<citation>}}

Yiming Zeng, Mingdong Wu, Long Yang, Jiyao Zhang, Hao Ding, Hui Cheng, Hao Dong. (2023)  
**Distilling Functional Rearrangement Priors from Large Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.01474v1)  

---


**ABSTRACT**  
Object rearrangement, a fundamental challenge in robotics, demands versatile strategies to handle diverse objects, configurations, and functional needs. To achieve this, the AI robot needs to learn functional rearrangement priors in order to specify precise goals that meet the functional requirements. Previous methods typically learn such priors from either laborious human annotations or manually designed heuristics, which limits scalability and generalization. In this work, we propose a novel approach that leverages large models to distill functional rearrangement priors. Specifically, our approach collects diverse arrangement examples using both LLMs and VLMs and then distills the examples into a diffusion model. During test time, the learned diffusion model is conditioned on the initial configuration and guides the positioning of objects to meet functional requirements. In this manner, we create a handshaking point that combines the strengths of conditional generative models and large models. Extensive experiments on multiple domains, including real-world scenarios, demonstrate the effectiveness of our approach in generating compatible goals for object rearrangement tasks, significantly outperforming baseline methods.

{{</citation>}}


### (39/50) RobotGPT: Robot Manipulation Learning from ChatGPT (Yixiang Jin et al., 2023)

{{<citation>}}

Yixiang Jin, Dingzhe Li, Yong A, Jun Shi, Peng Hao, Fuchun Sun, Jianwei Zhang, Bin Fang. (2023)  
**RobotGPT: Robot Manipulation Learning from ChatGPT**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.01421v1)  

---


**ABSTRACT**  
We present RobotGPT, an innovative decision framework for robotic manipulation that prioritizes stability and safety. The execution code generated by ChatGPT cannot guarantee the stability and safety of the system. ChatGPT may provide different answers for the same task, leading to unpredictability. This instability prevents the direct integration of ChatGPT into the robot manipulation loop. Although setting the temperature to 0 can generate more consistent outputs, it may cause ChatGPT to lose diversity and creativity. Our objective is to leverage ChatGPT's problem-solving capabilities in robot manipulation and train a reliable agent. The framework includes an effective prompt structure and a robust learning model. Additionally, we introduce a metric for measuring task difficulty to evaluate ChatGPT's performance in robot manipulation. Furthermore, we evaluate RobotGPT in both simulation and real-world environments. Compared to directly using ChatGPT to generate code, our framework significantly improves task success rates, with an average increase from 38.5% to 91.5%. Therefore, training a RobotGPT by utilizing ChatGPT as an expert is a more stable approach compared to directly using ChatGPT as a task planner.

{{</citation>}}


### (40/50) SAGE: Bridging Semantic and Actionable Parts for GEneralizable Articulated-Object Manipulation under Language Instructions (Haoran Geng et al., 2023)

{{<citation>}}

Haoran Geng, Songlin Wei, Congyue Deng, Bokui Shen, He Wang, Leonidas Guibas. (2023)  
**SAGE: Bridging Semantic and Actionable Parts for GEneralizable Articulated-Object Manipulation under Language Instructions**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.01307v1)  

---


**ABSTRACT**  
Generalizable manipulation of articulated objects remains a challenging problem in many real-world scenarios, given the diverse object structures, functionalities, and goals. In these tasks, both semantic interpretations and physical plausibilities are crucial for a policy to succeed. To address this problem, we propose SAGE, a novel framework that bridges the understanding of semantic and actionable parts of articulated objects to achieve generalizable manipulation under language instructions. Given a manipulation goal specified by natural language, an instruction interpreter with Large Language Models (LLMs) first translates them into programmatic actions on the object's semantic parts. This process also involves a scene context parser for understanding the visual inputs, which is designed to generate scene descriptions with both rich information and accurate interaction-related facts by joining the forces of generalist Visual-Language Models (VLMs) and domain-specialist part perception models. To further convert the action programs into executable policies, a part grounding module then maps the object semantic parts suggested by the instruction interpreter into so-called Generalizable Actionable Parts (GAParts). Finally, an interactive feedback module is incorporated to respond to failures, which greatly increases the robustness of the overall framework. Experiments both in simulation environments and on real robots show that our framework can handle a large variety of articulated objects with diverse language-instructed goals. We also provide a new benchmark for language-guided articulated-object manipulation in realistic scenarios.

{{</citation>}}


## cs.SD (1)



### (41/50) OpenVoice: Versatile Instant Voice Cloning (Zengyi Qin et al., 2023)

{{<citation>}}

Zengyi Qin, Wenliang Zhao, Xumin Yu, Xin Sun. (2023)  
**OpenVoice: Versatile Instant Voice Cloning**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.01479v1)  

---


**ABSTRACT**  
We introduce OpenVoice, a versatile voice cloning approach that requires only a short audio clip from the reference speaker to replicate their voice and generate speech in multiple languages. OpenVoice represents a significant advancement in addressing the following open challenges in the field: 1) Flexible Voice Style Control. OpenVoice enables granular control over voice styles, including emotion, accent, rhythm, pauses, and intonation, in addition to replicating the tone color of the reference speaker. The voice styles are not directly copied from and constrained by the style of the reference speaker. Previous approaches lacked the ability to flexibly manipulate voice styles after cloning. 2) Zero-Shot Cross-Lingual Voice Cloning. OpenVoice achieves zero-shot cross-lingual voice cloning for languages not included in the massive-speaker training set. Unlike previous approaches, which typically require extensive massive-speaker multi-lingual (MSML) dataset for all languages, OpenVoice can clone voices into a new language without any massive-speaker training data for that language. OpenVoice is also computationally efficient, costing tens of times less than commercially available APIs that offer even inferior performance. To foster further research in the field, we have made the source code and trained model publicly accessible. We also provide qualitative results in our demo website. Prior to its public release, our internal version of OpenVoice was used tens of millions of times by users worldwide between May and October 2023, serving as the backend of MyShell.ai.

{{</citation>}}


## cs.DB (2)



### (42/50) Context-Enhanced Relational Operators with Vector Embeddings (Viktor Sanca et al., 2023)

{{<citation>}}

Viktor Sanca, Manos Chatzakis, Anastasia Ailamaki. (2023)  
**Context-Enhanced Relational Operators with Vector Embeddings**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-DB, cs-LG, cs.DB  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.01476v1)  

---


**ABSTRACT**  
Collecting data, extracting value, and combining insights from relational and context-rich multi-modal sources in data processing pipelines presents a challenge for traditional relational DBMS. While relational operators allow declarative and optimizable query specification, they are limited to data transformations unsuitable for capturing or analyzing context. On the other hand, representation learning models can map context-rich data into embeddings, allowing machine-automated context processing but requiring imperative data transformation integration with the analytical query.   To bridge this dichotomy, we present a context-enhanced relational join and introduce an embedding operator composable with relational operators. This enables hybrid relational and context-rich vector data processing, with algebraic equivalences compatible with relational algebra and corresponding logical and physical optimizations. We investigate model-operator interaction with vector data processing and study the characteristics of the E-join operator. Using an example of string embeddings, we demonstrate enabling hybrid context-enhanced processing on relational join operators with vector embeddings. The importance of holistic optimization, from logical to physical, is demonstrated in an order of magnitude execution time improvement.

{{</citation>}}


### (43/50) D-Bot: Database Diagnosis System using Large Language Models (Xuanhe Zhou et al., 2023)

{{<citation>}}

Xuanhe Zhou, Guoliang Li, Zhaoyan Sun, Zhiyuan Liu, Weize Chen, Jianming Wu, Jiesi Liu, Ruohang Feng, Guoyang Zeng. (2023)  
**D-Bot: Database Diagnosis System using Large Language Models**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-CL, cs-DB, cs-LG, cs.DB  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.01454v1)  

---


**ABSTRACT**  
Database administrators (DBAs) play an important role in managing, maintaining and optimizing database systems. However, it is hard and tedious for DBAs to manage a large number of databases and give timely response (waiting for hours is intolerable in many online cases). In addition, existing empirical methods only support limited diagnosis scenarios, which are also labor-intensive to update the diagnosis rules for database version updates. Recently large language models (LLMs) have shown great potential in various fields. Thus, we propose D-Bot, an LLM-based database diagnosis system that can automatically acquire knowledge from diagnosis documents, and generate reasonable and well-founded diagnosis report (i.e., identifying the root causes and solutions) within acceptable time (e.g., under 10 minutes compared to hours by a DBA). The techniques in D-Bot include (i) offline knowledge extraction from documents, (ii) automatic prompt generation (e.g., knowledge matching, tool retrieval), (iii) root cause analysis using tree search algorithm, and (iv) collaborative mechanism for complex anomalies with multiple root causes. We verify D-Bot on real benchmarks (including 539 anomalies of six typical applications), and the results show that D-Bot can effectively analyze the root causes of unseen anomalies and significantly outperforms traditional methods and vanilla models like GPT-4.

{{</citation>}}


## cs.NI (1)



### (44/50) Classification of Home Network Problems with Transformers (Jeremias Dötterl et al., 2023)

{{<citation>}}

Jeremias Dötterl, Zahra Hemmati Fard. (2023)  
**Classification of Home Network Problems with Transformers**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.01445v1)  

---


**ABSTRACT**  
We propose a classifier that can identify ten common home network problems based on the raw textual output of networking tools such as ping, dig, and ip. Our deep learning model uses an encoder-only transformer architecture with a particular pre-tokenizer that we propose for splitting the tool output into token sequences. The use of transformers distinguishes our approach from related work on network problem classification, which still primarily relies on non-deep-learning methods. Our model achieves high accuracy in our experiments, demonstrating the high potential of transformer-based problem classification for the home network.

{{</citation>}}


## eess.SY (1)



### (45/50) OplixNet: Towards Area-Efficient Optical Split-Complex Networks with Real-to-Complex Data Assignment and Knowledge Distillation (Ruidi Qiu et al., 2023)

{{<citation>}}

Ruidi Qiu, Amro Eldebiky, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Ulf Schlichtmann, Bing Li. (2023)  
**OplixNet: Towards Area-Efficient Optical Split-Complex Networks with Real-to-Complex Data Assignment and Knowledge Distillation**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.01403v1)  

---


**ABSTRACT**  
Having the potential for high speed, high throughput, and low energy cost, optical neural networks (ONNs) have emerged as a promising candidate for accelerating deep learning tasks. In conventional ONNs, light amplitudes are modulated at the input and detected at the output. However, the light phases are still ignored in conventional structures, although they can also carry information for computing. To address this issue, in this paper, we propose a framework called OplixNet to compress the areas of ONNs by modulating input image data into the amplitudes and phase parts of light signals. The input and output parts of the ONNs are redesigned to make full use of both amplitude and phase information. Moreover, mutual learning across different ONN structures is introduced to maintain the accuracy. Experimental results demonstrate that the proposed framework significantly reduces the areas of ONNs with the accuracy within an acceptable range. For instance, 75.03% area is reduced with a 0.33% accuracy decrease on fully connected neural network (FCNN) and 74.88% area is reduced with a 2.38% accuracy decrease on ResNet-32.

{{</citation>}}


## cs.AI (2)



### (46/50) Honesty Is the Best Policy: Defining and Mitigating AI Deception (Francis Rhys Ward et al., 2023)

{{<citation>}}

Francis Rhys Ward, Francesco Belardinelli, Francesca Toni, Tom Everitt. (2023)  
**Honesty Is the Best Policy: Defining and Mitigating AI Deception**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.01350v1)  

---


**ABSTRACT**  
Deceptive agents are a challenge for the safety, trustworthiness, and cooperation of AI systems. We focus on the problem that agents might deceive in order to achieve their goals (for instance, in our experiments with language models, the goal of being evaluated as truthful). There are a number of existing definitions of deception in the literature on game theory and symbolic AI, but there is no overarching theory of deception for learning agents in games. We introduce a formal definition of deception in structural causal games, grounded in the philosophy literature, and applicable to real-world machine learning systems. Several examples and results illustrate that our formal definition aligns with the philosophical and commonsense meaning of deception. Our main technical result is to provide graphical criteria for deception. We show, experimentally, that these results can be used to mitigate deception in reinforcement learning agents and language models.

{{</citation>}}


### (47/50) Running cognitive evaluations on large language models: The do's and the don'ts (Anna A. Ivanova, 2023)

{{<citation>}}

Anna A. Ivanova. (2023)  
**Running cognitive evaluations on large language models: The do's and the don'ts**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.01276v1)  

---


**ABSTRACT**  
In this paper, I describe methodological considerations for studies that aim to evaluate the cognitive capacities of large language models (LLMs) using language-based behavioral assessments. Drawing on three case studies from the literature (a commonsense knowledge benchmark, a theory of mind evaluation, and a test of syntactic agreement), I describe common pitfalls that might arise when applying a cognitive test to an LLM. I then list 10 do's and don'ts that should help design high-quality cognitive evaluations for AI systems. I conclude by discussing four areas where the do's and don'ts are currently under active discussion -- prompt sensitivity, cultural and linguistic diversity, using LLMs as research assistants, and running evaluations on open vs. closed LLMs. Overall, the goal of the paper is to contribute to the broader discussion of best practices in the rapidly growing field of AI Psychology.

{{</citation>}}


## cs.CR (1)



### (48/50) Evaluating the Security of Satellite Systems (Roy Peled et al., 2023)

{{<citation>}}

Roy Peled, Eran Aizikovich, Edan Habler, Yuval Elovici, Asaf Shabtai. (2023)  
**Evaluating the Security of Satellite Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.01330v1)  

---


**ABSTRACT**  
Satellite systems are facing an ever-increasing amount of cybersecurity threats as their role in communications, navigation, and other services expands. Recent papers have examined attacks targeting satellites and space systems; however, they did not comprehensively analyze the threats to satellites and systematically identify adversarial techniques across the attack lifecycle. This paper presents a comprehensive taxonomy of adversarial tactics, techniques, and procedures explicitly targeting LEO satellites. First, we analyze the space ecosystem including the ground, space, Communication, and user segments, highlighting their architectures, functions, and vulnerabilities. Then, we examine the threat landscape, including adversary types, and capabilities, and survey historical and recent attacks such as jamming, spoofing, and supply chain. Finally, we propose a novel extension of the MITRE ATT&CK framework to categorize satellite attack techniques across the adversary lifecycle from reconnaissance to impact. The taxonomy is demonstrated by modeling high-profile incidents, including the Viasat attack that disrupted Ukraine's communications. The taxonomy provides the foundation for the development of defenses against emerging cyber risks to space assets. The proposed threat model will advance research in the space domain and contribute to the security of the space domain against sophisticated attacks.

{{</citation>}}


## math.OC (1)



### (49/50) Anomaly Detection Under Uncertainty Using Distributionally Robust Optimization Approach (Amir Hossein Noormohammadia et al., 2023)

{{<citation>}}

Amir Hossein Noormohammadia, Seyed Ali MirHassania, Farnaz Hooshmand Khaligh. (2023)  
**Anomaly Detection Under Uncertainty Using Distributionally Robust Optimization Approach**  

---
Primary Category: math.OC  
Categories: cs-LG, math-OC, math.OC  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.01296v1)  

---


**ABSTRACT**  
Anomaly detection is defined as the problem of finding data points that do not follow the patterns of the majority. Among the various proposed methods for solving this problem, classification-based methods, including one-class Support Vector Machines (SVM) are considered effective and state-of-the-art. The one-class SVM method aims to find a decision boundary to distinguish between normal data points and anomalies using only the normal data. On the other hand, most real-world problems involve some degree of uncertainty, where the true probability distribution of each data point is unknown, and estimating it is often difficult and costly. Assuming partial distribution information such as the first and second-order moments is known, a distributionally robust chance-constrained model is proposed in which the probability of misclassification is low. By utilizing a mapping function to a higher dimensional space, the proposed model will be capable of classifying origin-inseparable datasets. Also, by adopting the kernel idea, the need for explicitly knowing the mapping is eliminated, computations can be performed in the input space, and computational complexity is reduced. Computational results validate the robustness of the proposed model under different probability distributions and also the superiority of the proposed model compared to the standard one-class SVM in terms of various evaluation metrics.

{{</citation>}}


## cs.CE (1)



### (50/50) Opportunities for Retrieval and Tool Augmented Large Language Models in Scientific Facilities (Michael H. Prince et al., 2023)

{{<citation>}}

Michael H. Prince, Henry Chan, Aikaterini Vriza, Tao Zhou, Varuni K. Sastry, Matthew T. Dearing, Ross J. Harder, Rama K. Vasudevan, Mathew J. Cherukara. (2023)  
**Opportunities for Retrieval and Tool Augmented Large Language Models in Scientific Facilities**  

---
Primary Category: cs.CE  
Categories: cond-mat-mtrl-sci, cs-CE, cs.CE, physics-acc-ph, physics-app-ph, physics-ins-det  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.01291v1)  

---


**ABSTRACT**  
Upgrades to advanced scientific user facilities such as next-generation x-ray light sources, nanoscience centers, and neutron facilities are revolutionizing our understanding of materials across the spectrum of the physical sciences, from life sciences to microelectronics. However, these facility and instrument upgrades come with a significant increase in complexity. Driven by more exacting scientific needs, instruments and experiments become more intricate each year. This increased operational complexity makes it ever more challenging for domain scientists to design experiments that effectively leverage the capabilities of and operate on these advanced instruments. Large language models (LLMs) can perform complex information retrieval, assist in knowledge-intensive tasks across applications, and provide guidance on tool usage. Using x-ray light sources, leadership computing, and nanoscience centers as representative examples, we describe preliminary experiments with a Context-Aware Language Model for Science (CALMS) to assist scientists with instrument operations and complex experimentation. With the ability to retrieve relevant information from facility documentation, CALMS can answer simple questions on scientific capabilities and other operational procedures. With the ability to interface with software tools and experimental hardware, CALMS can conversationally operate scientific instruments. By making information more accessible and acting on user needs, LLMs could expand and diversify scientific facilities' users and accelerate scientific output.

{{</citation>}}
