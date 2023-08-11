---
draft: false
title: "arXiv @ 2023.08.11"
date: 2023-08-11
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.11"
    identifier: arxiv_20230811
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AI (7)](#csai-7)
- [cs.CV (31)](#cscv-31)
- [cs.SI (3)](#cssi-3)
- [cs.SE (7)](#csse-7)
- [quant-ph (1)](#quant-ph-1)
- [cs.CR (8)](#cscr-8)
- [cs.HC (1)](#cshc-1)
- [cs.CL (19)](#cscl-19)
- [cs.SD (3)](#cssd-3)
- [cs.AR (1)](#csar-1)
- [eess.IV (5)](#eessiv-5)
- [math.PR (1)](#mathpr-1)
- [cs.CY (2)](#cscy-2)
- [eess.SP (2)](#eesssp-2)
- [cs.LG (10)](#cslg-10)
- [cs.NI (3)](#csni-3)
- [cs.RO (2)](#csro-2)
- [cs.PL (1)](#cspl-1)
- [physics.geo-ph (1)](#physicsgeo-ph-1)
- [cs.IR (1)](#csir-1)

## cs.AI (7)



### (1/109) AI4GCC -- Track 3: Consumption and the Challenges of Multi-Agent RL (Marco Jiralerspong et al., 2023)

{{<citation>}}

Marco Jiralerspong, Gauthier Gidel. (2023)  
**AI4GCC -- Track 3: Consumption and the Challenges of Multi-Agent RL**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.05260v1)  

---


**ABSTRACT**  
The AI4GCC competition presents a bold step forward in the direction of integrating machine learning with traditional economic policy analysis. Below, we highlight two potential areas for improvement that could enhance the competition's ability to identify and evaluate proposed negotiation protocols. Firstly, we suggest the inclusion of an additional index that accounts for consumption/utility as part of the evaluation criteria. Secondly, we recommend further investigation into the learning dynamics of agents in the simulator and the game theoretic properties of outcomes from proposed negotiation protocols. We hope that these suggestions can be of use for future iterations of the competition/simulation.

{{</citation>}}


### (2/109) 'Generate' the Future of Work through AI: Empirical Evidence from Online Labor Markets (Jin Liu et al., 2023)

{{<citation>}}

Jin Liu, Xingchen Xu, Yongjun Li, Yong Tan. (2023)  
**'Generate' the Future of Work through AI: Empirical Evidence from Online Labor Markets**  

---
Primary Category: cs.AI  
Categories: J-4, cs-AI, cs-HC, cs.AI, econ-GN, q-fin-EC  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.05201v1)  

---


**ABSTRACT**  
With the advent of general-purpose Generative AI, the interest in discerning its impact on the labor market escalates. In an attempt to bridge the extant empirical void, we interpret the launch of ChatGPT as an exogenous shock, and implement a Difference-in-Differences (DID) approach to quantify its influence on text-related jobs and freelancers within an online labor marketplace. Our results reveal a significant decrease in transaction volume for gigs and freelancers directly exposed to ChatGPT. Additionally, this decline is particularly marked in units of relatively higher past transaction volume or lower quality standards. Yet, the negative effect is not universally experienced among service providers. Subsequent analyses illustrate that freelancers proficiently adapting to novel advancements and offering services that augment AI technologies can yield substantial benefits amidst this transformative period. Consequently, even though the advent of ChatGPT could conceivably substitute existing occupations, it also unfolds immense opportunities and carries the potential to reconfigure the future of work. This research contributes to the limited empirical repository exploring the profound influence of LLM-based generative AI on the labor market, furnishing invaluable insights for workers, job intermediaries, and regulatory bodies navigating this evolving landscape.

{{</citation>}}


### (3/109) Competitions in AI -- Robustly Ranking Solvers Using Statistical Resampling (Chris Fawcett et al., 2023)

{{<citation>}}

Chris Fawcett, Mauro Vallati, Holger H. Hoos, Alfonso E. Gerevini. (2023)  
**Competitions in AI -- Robustly Ranking Solvers Using Statistical Resampling**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.05062v1)  

---


**ABSTRACT**  
Solver competitions play a prominent role in assessing and advancing the state of the art for solving many problems in AI and beyond. Notably, in many areas of AI, competitions have had substantial impact in guiding research and applications for many years, and for a solver to be ranked highly in a competition carries considerable weight. But to which extent can we expect competition results to generalise to sets of problem instances different from those used in a particular competition? This is the question we investigate here, using statistical resampling techniques. We show that the rankings resulting from the standard interpretation of competition results can be very sensitive to even minor changes in the benchmark instance set used as the basis for assessment and can therefore not be expected to carry over to other samples from the same underlying instance distribution. To address this problem, we introduce a novel approach to statistically meaningful analysis of competition results based on resampling performance data. Our approach produces confidence intervals of competition scores as well as statistically robust solver rankings with bounded error. Applied to recent SAT, AI planning and computer vision competitions, our analysis reveals frequent statistical ties in solver performance as well as some inversions of ranks compared to the official results based on simple scoring.

{{</citation>}}


### (4/109) Expert load matters: operating networks at high accuracy and low manual effort (Sara Sangalli et al., 2023)

{{<citation>}}

Sara Sangalli, Ertunc Erdil, Ender Konukoglu. (2023)  
**Expert load matters: operating networks at high accuracy and low manual effort**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.05035v1)  

---


**ABSTRACT**  
In human-AI collaboration systems for critical applications, in order to ensure minimal error, users should set an operating point based on model confidence to determine when the decision should be delegated to human experts. Samples for which model confidence is lower than the operating point would be manually analysed by experts to avoid mistakes. Such systems can become truly useful only if they consider two aspects: models should be confident only for samples for which they are accurate, and the number of samples delegated to experts should be minimized. The latter aspect is especially crucial for applications where available expert time is limited and expensive, such as healthcare. The trade-off between the model accuracy and the number of samples delegated to experts can be represented by a curve that is similar to an ROC curve, which we refer to as confidence operating characteristic (COC) curve. In this paper, we argue that deep neural networks should be trained by taking into account both accuracy and expert load and, to that end, propose a new complementary loss function for classification that maximizes the area under this COC curve. This promotes simultaneously the increase in network accuracy and the reduction in number of samples delegated to humans. We perform experiments on multiple computer vision and medical image datasets for classification. Our results demonstrate that the proposed loss improves classification accuracy and delegates less number of decisions to experts, achieves better out-of-distribution samples detection and on par calibration performance compared to existing loss functions.

{{</citation>}}


### (5/109) MetRoBERTa: Leveraging Traditional Customer Relationship Management Data to Develop a Transit-Topic-Aware Language Model (Michael Leong et al., 2023)

{{<citation>}}

Michael Leong, Awad Abdelhalim, Jude Ha, Dianne Patterson, Gabriel L. Pincus, Anthony B. Harris, Michael Eichler, Jinhua Zhao. (2023)  
**MetRoBERTa: Leveraging Traditional Customer Relationship Management Data to Develop a Transit-Topic-Aware Language Model**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: BERT, Language Model, Twitter  
[Paper Link](http://arxiv.org/abs/2308.05012v1)  

---


**ABSTRACT**  
Transit riders' feedback provided in ridership surveys, customer relationship management (CRM) channels, and in more recent times, through social media is key for transit agencies to better gauge the efficacy of their services and initiatives. Getting a holistic understanding of riders' experience through the feedback shared in those instruments is often challenging, mostly due to the open-ended, unstructured nature of text feedback. In this paper, we propose leveraging traditional transit CRM feedback to develop and deploy a transit-topic-aware large language model (LLM) capable of classifying open-ended text feedback to relevant transit-specific topics. First, we utilize semi-supervised learning to engineer a training dataset of 11 broad transit topics detected in a corpus of 6 years of customer feedback provided to the Washington Metropolitan Area Transit Authority (WMATA). We then use this dataset to train and thoroughly evaluate a language model based on the RoBERTa architecture. We compare our LLM, MetRoBERTa, to classical machine learning approaches utilizing keyword-based and lexicon representations. Our model outperforms those methods across all evaluation metrics, providing an average topic classification accuracy of 90%. Finally, we provide a value proposition of this work demonstrating how the language model, alongside additional text processing tools, can be applied to add structure to open-ended text sources of feedback like Twitter. The framework and results we present provide a pathway for an automated, generalizable approach for ingesting, visualizing, and reporting transit riders' feedback at scale, enabling agencies to better understand and improve customer experience.

{{</citation>}}


### (6/109) Improving Autonomous Separation Assurance through Distributed Reinforcement Learning with Attention Networks (Marc W. Brittain et al., 2023)

{{<citation>}}

Marc W. Brittain, Luis E. Alvarez, Kara Breeden. (2023)  
**Improving Autonomous Separation Assurance through Distributed Reinforcement Learning with Attention Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Attention, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.04958v1)  

---


**ABSTRACT**  
Advanced Air Mobility (AAM) introduces a new, efficient mode of transportation with the use of vehicle autonomy and electrified aircraft to provide increasingly autonomous transportation between previously underserved markets. Safe and efficient navigation of low altitude aircraft through highly dense environments requires the integration of a multitude of complex observations, such as surveillance, knowledge of vehicle dynamics, and weather. The processing and reasoning on these observations pose challenges due to the various sources of uncertainty in the information while ensuring cooperation with a variable number of aircraft in the airspace. These challenges coupled with the requirement to make safety-critical decisions in real-time rule out the use of conventional separation assurance techniques. We present a decentralized reinforcement learning framework to provide autonomous self-separation capabilities within AAM corridors with the use of speed and vertical maneuvers. The problem is formulated as a Markov Decision Process and solved by developing a novel extension to the sample-efficient, off-policy soft actor-critic (SAC) algorithm. We introduce the use of attention networks for variable-length observation processing and a distributed computing architecture to achieve high training sample throughput as compared to existing approaches. A comprehensive numerical study shows that the proposed framework can ensure safe and efficient separation of aircraft in high density, dynamic environments with various sources of uncertainty.

{{</citation>}}


### (7/109) Explainable AI in Orthopedics: Challenges, Opportunities, and Prospects (Soheyla Amirian et al., 2023)

{{<citation>}}

Soheyla Amirian, Luke A. Carlson, Matthew F. Gong, Ines Lohse, Kurt R. Weiss, Johannes F. Plate, Ahmad P. Tafti. (2023)  
**Explainable AI in Orthopedics: Challenges, Opportunities, and Prospects**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04696v1)  

---


**ABSTRACT**  
While artificial intelligence (AI) has made many successful applications in various domains, its adoption in healthcare lags a little bit behind other high-stakes settings. Several factors contribute to this slower uptake, including regulatory frameworks, patient privacy concerns, and data heterogeneity. However, one significant challenge that impedes the implementation of AI in healthcare, particularly in orthopedics, is the lack of explainability and interpretability around AI models. Addressing the challenge of explainable AI (XAI) in orthopedics requires developing AI models and algorithms that prioritize transparency and interpretability, allowing clinicians, surgeons, and patients to understand the contributing factors behind any AI-powered predictive or descriptive models. The current contribution outlines several key challenges and opportunities that manifest in XAI in orthopedic practice. This work emphasizes the need for interdisciplinary collaborations between AI practitioners, orthopedic specialists, and regulatory entities to establish standards and guidelines for the adoption of XAI in orthopedics.

{{</citation>}}


## cs.CV (31)



### (8/109) Advancing Early Detection of Virus Yellows: Developing a Hybrid Convolutional Neural Network for Automatic Aphid Counting in Sugar Beet Fields (Xumin Gao et al., 2023)

{{<citation>}}

Xumin Gao, Wenxin Xue, Callum Lennox, Mark Stevens, Junfeng Gao. (2023)  
**Advancing Early Detection of Virus Yellows: Developing a Hybrid Convolutional Neural Network for Automatic Aphid Counting in Sugar Beet Fields**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Yolo  
[Paper Link](http://arxiv.org/abs/2308.05257v1)  

---


**ABSTRACT**  
Aphids are efficient vectors to transmit virus yellows in sugar beet fields. Timely monitoring and control of their populations are thus critical to prevent the large-scale outbreak of virus yellows. However, the manual counting of aphids, which is the most common practice, is labor-intensive and time-consuming. Additionally, two of the biggest challenges in aphid counting are that aphids are small objects and their density distributions are varied in different areas of the field. To address these challenges, we proposed a hybrid automatic aphid counting network architecture which integrates the detection network and the density map estimation network. When the distribution density of aphids is low, it utilizes an improved Yolov5 to count aphids. Conversely, when the distribution density of aphids is high, its witches to CSRNet to count aphids. To the best of our knowledge, this is the first framework integrating the detection network and the density map estimation network for counting tasks. Through comparison experiments of counting aphids, it verified that our proposed approach outperforms all other methods in counting aphids. It achieved the lowest MAE and RMSE values for both the standard and high-density aphid datasets: 2.93 and 4.01 (standard), and 34.19 and 38.66 (high-density), respectively. Moreover, the AP of the improved Yolov5 is 5% higher than that of the original Yolov5. Especially for extremely small aphids and densely distributed aphids, the detection performance of the improved Yolov5 is significantly better than the original Yolov5. This work provides an effective early warning for the virus yellows risk caused by aphids in sugar beet fields, offering protection for sugar beet growth and ensuring sugar beet yield. The datasets and project code are released at: https://github.com/JunfengGaolab/Counting-Aphids.

{{</citation>}}


### (9/109) Leveraging the Edge and Cloud for V2X-Based Real-Time Object Detection in Autonomous Driving (Faisal Hawlader et al., 2023)

{{<citation>}}

Faisal Hawlader, François Robinet, Raphaël Frank. (2023)  
**Leveraging the Edge and Cloud for V2X-Based Real-Time Object Detection in Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-DC, cs-LG, cs-NI, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.05234v1)  

---


**ABSTRACT**  
Environmental perception is a key element of autonomous driving because the information received from the perception module influences core driving decisions. An outstanding challenge in real-time perception for autonomous driving lies in finding the best trade-off between detection quality and latency. Major constraints on both computation and power have to be taken into account for real-time perception in autonomous vehicles. Larger object detection models tend to produce the best results, but are also slower at runtime. Since the most accurate detectors cannot run in real-time locally, we investigate the possibility of offloading computation to edge and cloud platforms, which are less resource-constrained. We create a synthetic dataset to train object detection models and evaluate different offloading strategies. Using real hardware and network simulations, we compare different trade-offs between prediction quality and end-to-end delay. Since sending raw frames over the network implies additional transmission delays, we also explore the use of JPEG and H.265 compression at varying qualities and measure their impact on prediction metrics. We show that models with adequate compression can be run in real-time on the cloud while outperforming local detection performance.

{{</citation>}}


### (10/109) SegMatch: A semi-supervised learning method for surgical instrument segmentation (Meng Wei et al., 2023)

{{<citation>}}

Meng Wei, Charlie Budd, Luis C. Garcia-Peraza-Herrera, Reuben Dorent, Miaojing Shi, Tom Vercauteren. (2023)  
**SegMatch: A semi-supervised learning method for surgical instrument segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.05232v1)  

---


**ABSTRACT**  
Surgical instrument segmentation is recognised as a key enabler to provide advanced surgical assistance and improve computer assisted interventions. In this work, we propose SegMatch, a semi supervised learning method to reduce the need for expensive annotation for laparoscopic and robotic surgical images. SegMatch builds on FixMatch, a widespread semi supervised classification pipeline combining consistency regularization and pseudo labelling, and adapts it for the purpose of segmentation. In our proposed SegMatch, the unlabelled images are weakly augmented and fed into the segmentation model to generate a pseudo-label to enforce the unsupervised loss against the output of the model for the adversarial augmented image on the pixels with a high confidence score. Our adaptation for segmentation tasks includes carefully considering the equivariance and invariance properties of the augmentation functions we rely on. To increase the relevance of our augmentations, we depart from using only handcrafted augmentations and introduce a trainable adversarial augmentation strategy. Our algorithm was evaluated on the MICCAI Instrument Segmentation Challenge datasets Robust-MIS 2019 and EndoVis 2017. Our results demonstrate that adding unlabelled data for training purposes allows us to surpass the performance of fully supervised approaches which are limited by the availability of training data in these challenges. SegMatch also outperforms a range of state-of-the-art semi-supervised learning semantic segmentation models in different labelled to unlabelled data ratios.

{{</citation>}}


### (11/109) Hierarchical Representations for Spatio-Temporal Visual Attention Modeling and Understanding (Miguel-Ángel Fernández-Torres, 2023)

{{<citation>}}

Miguel-Ángel Fernández-Torres. (2023)  
**Hierarchical Representations for Spatio-Temporal Visual Attention Modeling and Understanding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.05189v1)  

---


**ABSTRACT**  
This PhD. Thesis concerns the study and development of hierarchical representations for spatio-temporal visual attention modeling and understanding in video sequences. More specifically, we propose two computational models for visual attention. First, we present a generative probabilistic model for context-aware visual attention modeling and understanding. Secondly, we develop a deep network architecture for visual attention modeling, which first estimates top-down spatio-temporal visual attention, and ultimately serves for modeling attention in the temporal domain.

{{</citation>}}


### (12/109) A Unified Interactive Model Evaluation for Classification, Object Detection, and Instance Segmentation in Computer Vision (Changjian Chen et al., 2023)

{{<citation>}}

Changjian Chen, Yukai Guo, Fengyuan Tian, Shilong Liu, Weikai Yang, Zhaowei Wang, Jing Wu, Hang Su, Hanspeter Pfister, Shixia Liu. (2023)  
**A Unified Interactive Model Evaluation for Classification, Object Detection, and Instance Segmentation in Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs.CV  
Keywords: Computer Vision, Object Detection  
[Paper Link](http://arxiv.org/abs/2308.05168v1)  

---


**ABSTRACT**  
Existing model evaluation tools mainly focus on evaluating classification models, leaving a gap in evaluating more complex models, such as object detection. In this paper, we develop an open-source visual analysis tool, Uni-Evaluator, to support a unified model evaluation for classification, object detection, and instance segmentation in computer vision. The key idea behind our method is to formulate both discrete and continuous predictions in different tasks as unified probability distributions. Based on these distributions, we develop 1) a matrix-based visualization to provide an overview of model performance; 2) a table visualization to identify the problematic data subsets where the model performs poorly; 3) a grid visualization to display the samples of interest. These visualizations work together to facilitate the model evaluation from a global overview to individual samples. Two case studies demonstrate the effectiveness of Uni-Evaluator in evaluating model performance and making informed improvements.

{{</citation>}}


### (13/109) LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation (Leigang Qu et al., 2023)

{{<citation>}}

Leigang Qu, Shengqiong Wu, Hao Fei, Liqiang Nie, Tat-Seng Chua. (2023)  
**LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.05095v1)  

---


**ABSTRACT**  
In the text-to-image generation field, recent remarkable progress in Stable Diffusion makes it possible to generate rich kinds of novel photorealistic images. However, current models still face misalignment issues (e.g., problematic spatial relation understanding and numeration failure) in complex natural scenes, which impedes the high-faithfulness text-to-image generation. Although recent efforts have been made to improve controllability by giving fine-grained guidance (e.g., sketch and scribbles), this issue has not been fundamentally tackled since users have to provide such guidance information manually. In this work, we strive to synthesize high-fidelity images that are semantically aligned with a given textual prompt without any guidance. Toward this end, we propose a coarse-to-fine paradigm to achieve layout planning and image generation. Concretely, we first generate the coarse-grained layout conditioned on a given textual prompt via in-context learning based on Large Language Models. Afterward, we propose a fine-grained object-interaction diffusion method to synthesize high-faithfulness images conditioned on the prompt and the automatically generated layout. Extensive experiments demonstrate that our proposed method outperforms the state-of-the-art models in terms of layout and image generation. Our code and settings are available at \url{https://layoutllm-t2i.github.io}.

{{</citation>}}


### (14/109) PAT: Position-Aware Transformer for Dense Multi-Label Action Detection (Faegheh Sardari et al., 2023)

{{<citation>}}

Faegheh Sardari, Armin Mustafa, Philip J. B. Jackson, Adrian Hilton. (2023)  
**PAT: Position-Aware Transformer for Dense Multi-Label Action Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.05051v1)  

---


**ABSTRACT**  
We present PAT, a transformer-based network that learns complex temporal co-occurrence action dependencies in a video by exploiting multi-scale temporal features. In existing methods, the self-attention mechanism in transformers loses the temporal positional information, which is essential for robust action detection. To address this issue, we (i) embed relative positional encoding in the self-attention mechanism and (ii) exploit multi-scale temporal relationships by designing a novel non hierarchical network, in contrast to the recent transformer-based approaches that use a hierarchical structure. We argue that joining the self-attention mechanism with multiple sub-sampling processes in the hierarchical approaches results in increased loss of positional information. We evaluate the performance of our proposed approach on two challenging dense multi-label benchmark datasets, and show that PAT improves the current state-of-the-art result by 1.1% and 0.6% mAP on the Charades and MultiTHUMOS datasets, respectively, thereby achieving the new state-of-the-art mAP at 26.5% and 44.6%, respectively. We also perform extensive ablation studies to examine the impact of the different components of our proposed network.

{{</citation>}}


### (15/109) Density Crop-guided Semi-supervised Object Detection in Aerial Images (Akhil Meethal et al., 2023)

{{<citation>}}

Akhil Meethal, Eric Granger, Marco Pedersoli. (2023)  
**Density Crop-guided Semi-supervised Object Detection in Aerial Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Drone, Object Detection  
[Paper Link](http://arxiv.org/abs/2308.05032v1)  

---


**ABSTRACT**  
One of the important bottlenecks in training modern object detectors is the need for labeled images where bounding box annotations have to be produced for each object present in the image. This bottleneck is further exacerbated in aerial images where the annotators have to label small objects often distributed in clusters on high-resolution images. In recent days, the mean-teacher approach trained with pseudo-labels and weak-strong augmentation consistency is gaining popularity for semi-supervised object detection. However, a direct adaptation of such semi-supervised detectors for aerial images where small clustered objects are often present, might not lead to optimal results. In this paper, we propose a density crop-guided semi-supervised detector that identifies the cluster of small objects during training and also exploits them to improve performance at inference. During training, image crops of clusters identified from labeled and unlabeled images are used to augment the training set, which in turn increases the chance of detecting small objects and creating good pseudo-labels for small objects on the unlabeled images. During inference, the detector is not only able to detect the objects of interest but also regions with a high density of small objects (density crops) so that detections from the input image and detections from image crops are combined, resulting in an overall more accurate object prediction, especially for small objects. Empirical studies on the popular benchmarks of VisDrone and DOTA datasets show the effectiveness of our density crop-guided semi-supervised detector with an average improvement of more than 2\% over the basic mean-teacher method in COCO style AP. Our code is available at: https://github.com/akhilpm/DroneSSOD.

{{</citation>}}


### (16/109) Feature Modulation Transformer: Cross-Refinement of Global Representation via High-Frequency Prior for Image Super-Resolution (Ao Li et al., 2023)

{{<citation>}}

Ao Li, Le Zhang, Yun Liu, Ce Zhu. (2023)  
**Feature Modulation Transformer: Cross-Refinement of Global Representation via High-Frequency Prior for Image Super-Resolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.05022v1)  

---


**ABSTRACT**  
Transformer-based methods have exhibited remarkable potential in single image super-resolution (SISR) by effectively extracting long-range dependencies. However, most of the current research in this area has prioritized the design of transformer blocks to capture global information, while overlooking the importance of incorporating high-frequency priors, which we believe could be beneficial. In our study, we conducted a series of experiments and found that transformer structures are more adept at capturing low-frequency information, but have limited capacity in constructing high-frequency representations when compared to their convolutional counterparts. Our proposed solution, the cross-refinement adaptive feature modulation transformer (CRAFT), integrates the strengths of both convolutional and transformer structures. It comprises three key components: the high-frequency enhancement residual block (HFERB) for extracting high-frequency information, the shift rectangle window attention block (SRWAB) for capturing global information, and the hybrid fusion block (HFB) for refining the global representation. Our experiments on multiple datasets demonstrate that CRAFT outperforms state-of-the-art methods by up to 0.29dB while using fewer parameters. The source code will be made available at: https://github.com/AVC2-UESTC/CRAFT-SR.git.

{{</citation>}}


### (17/109) Robust Object Modeling for Visual Tracking (Yidong Cai et al., 2023)

{{<citation>}}

Yidong Cai, Jie Liu, Jie Tang, Gangshan Wu. (2023)  
**Robust Object Modeling for Visual Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.05140v1)  

---


**ABSTRACT**  
Object modeling has become a core part of recent tracking frameworks. Current popular tackers use Transformer attention to extract the template feature separately or interactively with the search region. However, separate template learning lacks communication between the template and search regions, which brings difficulty in extracting discriminative target-oriented features. On the other hand, interactive template learning produces hybrid template features, which may introduce potential distractors to the template via the cluttered search regions. To enjoy the merits of both methods, we propose a robust object modeling framework for visual tracking (ROMTrack), which simultaneously models the inherent template and the hybrid template features. As a result, harmful distractors can be suppressed by combining the inherent features of target objects with search regions' guidance. Target-related features can also be extracted using the hybrid template, thus resulting in a more robust object modeling framework. To further enhance robustness, we present novel variation tokens to depict the ever-changing appearance of target objects. Variation tokens are adaptable to object deformation and appearance variations, which can boost overall performance with negligible computation. Experiments show that our ROMTrack sets a new state-of-the-art on multiple benchmarks.

{{</citation>}}


### (18/109) Discrepancy-based Active Learning for Weakly Supervised Bleeding Segmentation in Wireless Capsule Endoscopy Images (Fan Bai et al., 2023)

{{<citation>}}

Fan Bai, Xiaohan Xing, Yutian Shen, Han Ma, Max Q. -H. Meng. (2023)  
**Discrepancy-based Active Learning for Weakly Supervised Bleeding Segmentation in Wireless Capsule Endoscopy Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2308.05137v1)  

---


**ABSTRACT**  
Weakly supervised methods, such as class activation maps (CAM) based, have been applied to achieve bleeding segmentation with low annotation efforts in Wireless Capsule Endoscopy (WCE) images. However, the CAM labels tend to be extremely noisy, and there is an irreparable gap between CAM labels and ground truths for medical images. This paper proposes a new Discrepancy-basEd Active Learning (DEAL) approach to bridge the gap between CAMs and ground truths with a few annotations. Specifically, to liberate labor, we design a novel discrepancy decoder model and a CAMPUS (CAM, Pseudo-label and groUnd-truth Selection) criterion to replace the noisy CAMs with accurate model predictions and a few human labels. The discrepancy decoder model is trained with a unique scheme to generate standard, coarse and fine predictions. And the CAMPUS criterion is proposed to predict the gaps between CAMs and ground truths based on model divergence and CAM divergence. We evaluate our method on the WCE dataset and results show that our method outperforms the state-of-the-art active learning methods and reaches comparable performance to those trained with full annotated datasets with only 10% of the training data labeled.

{{</citation>}}


### (19/109) Prototypical Kernel Learning and Open-set Foreground Perception for Generalized Few-shot Semantic Segmentation (Kai Huang et al., 2023)

{{<citation>}}

Kai Huang, Feigege Wang, Ye Xi, Yutao Gao. (2023)  
**Prototypical Kernel Learning and Open-set Foreground Perception for Generalized Few-shot Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.04952v2)  

---


**ABSTRACT**  
Generalized Few-shot Semantic Segmentation (GFSS) extends Few-shot Semantic Segmentation (FSS) to simultaneously segment unseen classes and seen classes during evaluation. Previous works leverage additional branch or prototypical aggregation to eliminate the constrained setting of FSS. However, representation division and embedding prejudice, which heavily results in poor performance of GFSS, have not been synthetical considered. We address the aforementioned problems by jointing the prototypical kernel learning and open-set foreground perception. Specifically, a group of learnable kernels is proposed to perform segmentation with each kernel in charge of a stuff class. Then, we explore to merge the prototypical learning to the update of base-class kernels, which is consistent with the prototype knowledge aggregation of few-shot novel classes. In addition, a foreground contextual perception module cooperating with conditional bias based inference is adopted to perform class-agnostic as well as open-set foreground detection, thus to mitigate the embedding prejudice and prevent novel targets from being misclassified as background. Moreover, we also adjust our method to the Class Incremental Few-shot Semantic Segmentation (CIFSS) which takes the knowledge of novel classes in a incremental stream. Extensive experiments on PASCAL-5i and COCO-20i datasets demonstrate that our method performs better than previous state-of-the-art.

{{</citation>}}


### (20/109) Branches Mutual Promotion for End-to-End Weakly Supervised Semantic Segmentation (Lei Zhu et al., 2023)

{{<citation>}}

Lei Zhu, Hangzhou He, Xinliang Zhang, Qian Chen, Shuang Zeng, Qiushi Ren, Yanye Lu. (2023)  
**Branches Mutual Promotion for End-to-End Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.04949v1)  

---


**ABSTRACT**  
End-to-end weakly supervised semantic segmentation aims at optimizing a segmentation model in a single-stage training process based on only image annotations. Existing methods adopt an online-trained classification branch to provide pseudo annotations for supervising the segmentation branch. However, this strategy makes the classification branch dominate the whole concurrent training process, hindering these two branches from assisting each other. In our work, we treat these two branches equally by viewing them as diverse ways to generate the segmentation map, and add interactions on both their supervision and operation to achieve mutual promotion. For this purpose, a bidirectional supervision mechanism is elaborated to force the consistency between the outputs of these two branches. Thus, the segmentation branch can also give feedback to the classification branch to enhance the quality of localization seeds. Moreover, our method also designs interaction operations between these two branches to exchange their knowledge to assist each other. Experiments indicate our work outperforms existing end-to-end weakly supervised segmentation methods.

{{</citation>}}


### (21/109) SelectNAdapt: Support Set Selection for Few-Shot Domain Adaptation (Youssef Dawoud et al., 2023)

{{<citation>}}

Youssef Dawoud, Gustavo Carneiro, Vasileios Belagiannis. (2023)  
**SelectNAdapt: Support Set Selection for Few-Shot Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.04946v1)  

---


**ABSTRACT**  
Generalisation of deep neural networks becomes vulnerable when distribution shifts are encountered between train (source) and test (target) domain data. Few-shot domain adaptation mitigates this issue by adapting deep neural networks pre-trained on the source domain to the target domain using a randomly selected and annotated support set from the target domain. This paper argues that randomly selecting the support set can be further improved for effectively adapting the pre-trained source models to the target domain. Alternatively, we propose SelectNAdapt, an algorithm to curate the selection of the target domain samples, which are then annotated and included in the support set. In particular, for the K-shot adaptation problem, we first leverage self-supervision to learn features of the target domain data. Then, we propose a per-class clustering scheme of the learned target domain features and select K representative target samples using a distance-based scoring function. Finally, we bring our selection setup towards a practical ground by relying on pseudo-labels for clustering semantically similar target domain samples. Our experiments show promising results on three few-shot domain adaptation benchmarks for image recognition compared to related approaches and the standard random selection.

{{</citation>}}


### (22/109) Gaussian Image Anomaly Detection with Greedy Eigencomponent Selection (Tetiana Gula et al., 2023)

{{<citation>}}

Tetiana Gula, João P C Bertoldo. (2023)  
**Gaussian Image Anomaly Detection with Greedy Eigencomponent Selection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.04944v1)  

---


**ABSTRACT**  
Anomaly detection (AD) in images, identifying significant deviations from normality, is a critical issue in computer vision. This paper introduces a novel approach to dimensionality reduction for AD using pre-trained convolutional neural network (CNN) that incorporate EfficientNet models. We investigate the importance of component selection and propose two types of tree search approaches, both employing a greedy strategy, for optimal eigencomponent selection. Our study conducts three main experiments to evaluate the effectiveness of our approach. The first experiment explores the influence of test set performance on component choice, the second experiment examines the performance when we train on one anomaly type and evaluate on all other types, and the third experiment investigates the impact of using a minimum number of images for training and selecting them based on anomaly types. Our approach aims to find the optimal subset of components that deliver the highest performance score, instead of focusing solely on the proportion of variance explained by each component and also understand the components behaviour in different settings. Our results indicate that the proposed method surpasses both Principal Component Analysis (PCA) and Negated Principal Component Analysis (NPCA) in terms of detection accuracy, even when using fewer components. Thus, our approach provides a promising alternative to conventional dimensionality reduction techniques in AD, and holds potential to enhance the efficiency and effectiveness of AD systems.

{{</citation>}}


### (23/109) JEDI: Joint Expert Distillation in a Semi-Supervised Multi-Dataset Student-Teacher Scenario for Video Action Recognition (Lucian Bicsi et al., 2023)

{{<citation>}}

Lucian Bicsi, Bogdan Alexe, Radu Tudor Ionescu, Marius Leordeanu. (2023)  
**JEDI: Joint Expert Distillation in a Semi-Supervised Multi-Dataset Student-Teacher Scenario for Video Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.04934v1)  

---


**ABSTRACT**  
We propose JEDI, a multi-dataset semi-supervised learning method, which efficiently combines knowledge from multiple experts, learned on different datasets, to train and improve the performance of individual, per dataset, student models. Our approach achieves this by addressing two important problems in current machine learning research: generalization across datasets and limitations of supervised training due to scarcity of labeled data. We start with an arbitrary number of experts, pretrained on their own specific dataset, which form the initial set of student models. The teachers are immediately derived by concatenating the feature representations from the penultimate layers of the students. We then train all models in a student-teacher semi-supervised learning scenario until convergence. In our efficient approach, student-teacher training is carried out jointly and end-to-end, showing that both students and teachers improve their generalization capacity during training. We validate our approach on four video action recognition datasets. By simultaneously considering all datasets within a unified semi-supervised setting, we demonstrate significant improvements over the initial experts.

{{</citation>}}


### (24/109) StableVQA: A Deep No-Reference Quality Assessment Model for Video Stability (Tengchuan Kou et al., 2023)

{{<citation>}}

Tengchuan Kou, Xiaohong Liu, Wei Sun, Jun Jia, Xiongkuo Min, Guangtao Zhai, Ning Liu. (2023)  
**StableVQA: A Deep No-Reference Quality Assessment Model for Video Stability**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.04904v2)  

---


**ABSTRACT**  
Video shakiness is an unpleasant distortion of User Generated Content (UGC) videos, which is usually caused by the unstable hold of cameras. In recent years, many video stabilization algorithms have been proposed, yet no specific and accurate metric enables comprehensively evaluating the stability of videos. Indeed, most existing quality assessment models evaluate video quality as a whole without specifically taking the subjective experience of video stability into consideration. Therefore, these models cannot measure the video stability explicitly and precisely when severe shakes are present. In addition, there is no large-scale video database in public that includes various degrees of shaky videos with the corresponding subjective scores available, which hinders the development of Video Quality Assessment for Stability (VQA-S). To this end, we build a new database named StableDB that contains 1,952 diversely-shaky UGC videos, where each video has a Mean Opinion Score (MOS) on the degree of video stability rated by 34 subjects. Moreover, we elaborately design a novel VQA-S model named StableVQA, which consists of three feature extractors to acquire the optical flow, semantic, and blur features respectively, and a regression layer to predict the final stability score. Extensive experiments demonstrate that the StableVQA achieves a higher correlation with subjective opinions than the existing VQA-S models and generic VQA models. The database and codes are available at https://github.com/QMME/StableVQA.

{{</citation>}}


### (25/109) VAST: Vivify Your Talking Avatar via Zero-Shot Expressive Facial Style Transfer (Liyang Chen et al., 2023)

{{<citation>}}

Liyang Chen, Zhiyong Wu, Runnan Li, Weihong Bao, Jun Ling, Xu Tan, Sheng Zhao. (2023)  
**VAST: Vivify Your Talking Avatar via Zero-Shot Expressive Facial Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.04830v1)  

---


**ABSTRACT**  
Current talking face generation methods mainly focus on speech-lip synchronization. However, insufficient investigation on the facial talking style leads to a lifeless and monotonous avatar. Most previous works fail to imitate expressive styles from arbitrary video prompts and ensure the authenticity of the generated video. This paper proposes an unsupervised variational style transfer model (VAST) to vivify the neutral photo-realistic avatars. Our model consists of three key components: a style encoder that extracts facial style representations from the given video prompts; a hybrid facial expression decoder to model accurate speech-related movements; a variational style enhancer that enhances the style space to be highly expressive and meaningful. With our essential designs on facial style learning, our model is able to flexibly capture the expressive facial style from arbitrary video prompts and transfer it onto a personalized image renderer in a zero-shot manner. Experimental results demonstrate the proposed approach contributes to a more vivid talking avatar with higher authenticity and richer expressiveness.

{{</citation>}}


### (26/109) MixReorg: Cross-Modal Mixed Patch Reorganization is a Good Mask Learner for Open-World Semantic Segmentation (Kaixin Cai et al., 2023)

{{<citation>}}

Kaixin Cai, Pengzhen Ren, Yi Zhu, Hang Xu, Jianzhuang Liu, Changlin Li, Guangrun Wang, Xiaodan Liang. (2023)  
**MixReorg: Cross-Modal Mixed Patch Reorganization is a Good Mask Learner for Open-World Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.04829v1)  

---


**ABSTRACT**  
Recently, semantic segmentation models trained with image-level text supervision have shown promising results in challenging open-world scenarios. However, these models still face difficulties in learning fine-grained semantic alignment at the pixel level and predicting accurate object masks. To address this issue, we propose MixReorg, a novel and straightforward pre-training paradigm for semantic segmentation that enhances a model's ability to reorganize patches mixed across images, exploring both local visual relevance and global semantic coherence. Our approach involves generating fine-grained patch-text pairs data by mixing image patches while preserving the correspondence between patches and text. The model is then trained to minimize the segmentation loss of the mixed images and the two contrastive losses of the original and restored features. With MixReorg as a mask learner, conventional text-supervised semantic segmentation models can achieve highly generalizable pixel-semantic alignment ability, which is crucial for open-world segmentation. After training with large-scale image-text data, MixReorg models can be applied directly to segment visual objects of arbitrary categories, without the need for further fine-tuning. Our proposed framework demonstrates strong performance on popular zero-shot semantic segmentation benchmarks, outperforming GroupViT by significant margins of 5.0%, 6.2%, 2.5%, and 3.4% mIoU on PASCAL VOC2012, PASCAL Context, MS COCO, and ADE20K, respectively.

{{</citation>}}


### (27/109) Joint-Relation Transformer for Multi-Person Motion Prediction (Qingyao Xu et al., 2023)

{{<citation>}}

Qingyao Xu, Weibo Mao, Jingze Gong, Chenxin Xu, Siheng Chen, Weidi Xie, Ya Zhang, Yanfeng Wang. (2023)  
**Joint-Relation Transformer for Multi-Person Motion Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.04808v1)  

---


**ABSTRACT**  
Multi-person motion prediction is a challenging problem due to the dependency of motion on both individual past movements and interactions with other people. Transformer-based methods have shown promising results on this task, but they miss the explicit relation representation between joints, such as skeleton structure and pairwise distance, which is crucial for accurate interaction modeling. In this paper, we propose the Joint-Relation Transformer, which utilizes relation information to enhance interaction modeling and improve future motion prediction. Our relation information contains the relative distance and the intra-/inter-person physical constraints. To fuse relation and joint information, we design a novel joint-relation fusion layer with relation-aware attention to update both features. Additionally, we supervise the relation information by forecasting future distance. Experiments show that our method achieves a 13.4% improvement of 900ms VIM on 3DPW-SoMoF/RC and 17.8%/12.0% improvement of 3s MPJPE on CMU-Mpcap/MuPoTS-3D dataset.

{{</citation>}}


### (28/109) High-Level Features Parallelization for Inference Cost Reduction Through Selective Attention (André Peter Kelm et al., 2023)

{{<citation>}}

André Peter Kelm, Lucas Schmidt, Tim Rolff, Christian Wilms, Ehsan Yaghoubi, Simone Frintrop. (2023)  
**High-Level Features Parallelization for Inference Cost Reduction Through Selective Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.05128v1)  

---


**ABSTRACT**  
In this work, we parallelize high-level features in deep networks to selectively skip or select class-specific features to reduce inference costs. This challenges most deep learning methods due to their limited ability to efficiently and effectively focus on selected class-specific features without retraining. We propose a serial-parallel hybrid architecture with serial generic low-level features and parallel high-level features. This accounts for the fact that many high-level features are class-specific rather than generic, and has connections to recent neuroscientific findings that observe spatially and contextually separated neural activations in the human brain. Our approach provides the unique functionality of cutouts: selecting parts of the network to focus on only relevant subsets of classes without requiring retraining. High performance is maintained, but the cost of inference can be significantly reduced. In some of our examples, up to $75\,\%$ of parameters are skipped and $35\,\%$ fewer GMACs (Giga multiply-accumulate) operations are used as the approach adapts to a change in task complexity. This is important for mobile, industrial, and robotic applications where reducing the number of parameters, the computational complexity, and thus the power consumption can be paramount. Another unique functionality is that it allows processing to be directly influenced by enhancing or inhibiting high-level class-specific features, similar to the mechanism of selective attention in the human brain. This can be relevant for cross-modal applications, the use of semantic prior knowledge, and/or context-aware processing.

{{</citation>}}


### (29/109) Enhancing Mobile Privacy and Security: A Face Skin Patch-Based Anti-Spoofing Approach (Qiushi Guo, 2023)

{{<citation>}}

Qiushi Guo. (2023)  
**Enhancing Mobile Privacy and Security: A Face Skin Patch-Based Anti-Spoofing Approach**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.04798v1)  

---


**ABSTRACT**  
As Facial Recognition System(FRS) is widely applied in areas such as access control and mobile payments due to its convenience and high accuracy. The security of facial recognition is also highly regarded. The Face anti-spoofing system(FAS) for face recognition is an important component used to enhance the security of face recognition systems. Traditional FAS used images containing identity information to detect spoofing traces, however there is a risk of privacy leakage during the transmission and storage of these images. Besides, the encryption and decryption of these privacy-sensitive data takes too long compared to inference time by FAS model. To address the above issues, we propose a face anti-spoofing algorithm based on facial skin patches leveraging pure facial skin patch images as input, which contain no privacy information, no encryption or decryption is needed for these images. We conduct experiments on several public datasets, the results prove that our algorithm has demonstrated superiority in both accuracy and speed.

{{</citation>}}


### (30/109) Multi-Scale Memory Comparison for Zero-/Few-Shot Anomaly Detection (Chaoqin Huang et al., 2023)

{{<citation>}}

Chaoqin Huang, Aofan Jiang, Ya Zhang, Yanfeng Wang. (2023)  
**Multi-Scale Memory Comparison for Zero-/Few-Shot Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.04789v1)  

---


**ABSTRACT**  
Anomaly detection has gained considerable attention due to its broad range of applications, particularly in industrial defect detection. To address the challenges of data collection, researchers have introduced zero-/few-shot anomaly detection techniques that require minimal normal images for each category. However, complex industrial scenarios often involve multiple objects, presenting a significant challenge. In light of this, we propose a straightforward yet powerful multi-scale memory comparison framework for zero-/few-shot anomaly detection. Our approach employs a global memory bank to capture features across the entire image, while an individual memory bank focuses on simplified scenes containing a single object. The efficacy of our method is validated by its remarkable achievement of 4th place in the zero-shot track and 2nd place in the few-shot track of the Visual Anomaly and Novelty Detection (VAND) competition.

{{</citation>}}


### (31/109) Objects do not disappear: Video object detection by single-frame object location anticipation (Xin Liu et al., 2023)

{{<citation>}}

Xin Liu, Fatemeh Karimi Nejadasl, Jan C. van Gemert, Olaf Booij, Silvia L. Pintea. (2023)  
**Objects do not disappear: Video object detection by single-frame object location anticipation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.04770v1)  

---


**ABSTRACT**  
Objects in videos are typically characterized by continuous smooth motion. We exploit continuous smooth motion in three ways. 1) Improved accuracy by using object motion as an additional source of supervision, which we obtain by anticipating object locations from a static keyframe. 2) Improved efficiency by only doing the expensive feature computations on a small subset of all frames. Because neighboring video frames are often redundant, we only compute features for a single static keyframe and predict object locations in subsequent frames. 3) Reduced annotation cost, where we only annotate the keyframe and use smooth pseudo-motion between keyframes. We demonstrate computational efficiency, annotation efficiency, and improved mean average precision compared to the state-of-the-art on four datasets: ImageNet VID, EPIC KITCHENS-55, YouTube-BoundingBoxes, and Waymo Open dataset. Our source code is available at https://github.com/L-KID/Videoobject-detection-by-location-anticipation.

{{</citation>}}


### (32/109) Induction Network: Audio-Visual Modality Gap-Bridging for Self-Supervised Sound Source Localization (Tianyu Liu et al., 2023)

{{<citation>}}

Tianyu Liu, Peng Zhang, Wei Huang, Yufei Zha, Tao You, Yanning Zhang. (2023)  
**Induction Network: Audio-Visual Modality Gap-Bridging for Self-Supervised Sound Source Localization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.04767v1)  

---


**ABSTRACT**  
Self-supervised sound source localization is usually challenged by the modality inconsistency. In recent studies, contrastive learning based strategies have shown promising to establish such a consistent correspondence between audio and sound sources in visual scenarios. Unfortunately, the insufficient attention to the heterogeneity influence in the different modality features still limits this scheme to be further improved, which also becomes the motivation of our work. In this study, an Induction Network is proposed to bridge the modality gap more effectively. By decoupling the gradients of visual and audio modalities, the discriminative visual representations of sound sources can be learned with the designed Induction Vector in a bootstrap manner, which also enables the audio modality to be aligned with the visual modality consistently. In addition to a visual weighted contrastive loss, an adaptive threshold selection strategy is introduced to enhance the robustness of the Induction Network. Substantial experiments conducted on SoundNet-Flickr and VGG-Sound Source datasets have demonstrated a superior performance compared to other state-of-the-art works in different challenging scenarios. The code is available at https://github.com/Tahy1/AVIN

{{</citation>}}


### (33/109) Self-supervised Learning of Rotation-invariant 3D Point Set Features using Transformer and its Self-distillation (Takahiko Furuya et al., 2023)

{{<citation>}}

Takahiko Furuya, Zhoujie Chen, Ryutarou Ohbuchi, Zhenzhong Kuang. (2023)  
**Self-supervised Learning of Rotation-invariant 3D Point Set Features using Transformer and its Self-distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-IR, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.04725v1)  

---


**ABSTRACT**  
Invariance against rotations of 3D objects is an important property in analyzing 3D point set data. Conventional 3D point set DNNs having rotation invariance typically obtain accurate 3D shape features via supervised learning by using labeled 3D point sets as training samples. However, due to the rapid increase in 3D point set data and the high cost of labeling, a framework to learn rotation-invariant 3D shape features from numerous unlabeled 3D point sets is required. This paper proposes a novel self-supervised learning framework for acquiring accurate and rotation-invariant 3D point set features at object-level. Our proposed lightweight DNN architecture decomposes an input 3D point set into multiple global-scale regions, called tokens, that preserve the spatial layout of partial shapes composing the 3D object. We employ a self-attention mechanism to refine the tokens and aggregate them into an expressive rotation-invariant feature per 3D point set. Our DNN is effectively trained by using pseudo-labels generated by a self-distillation framework. To facilitate the learning of accurate features, we propose to combine multi-crop and cut-mix data augmentation techniques to diversify 3D point sets for training. Through a comprehensive evaluation, we empirically demonstrate that, (1) existing rotation-invariant DNN architectures designed for supervised learning do not necessarily learn accurate 3D shape features under a self-supervised learning scenario, and (2) our proposed algorithm learns rotation-invariant 3D point set features that are more accurate than those learned by existing algorithms. Code will be available at https://github.com/takahikof/RIPT_SDMM

{{</citation>}}


### (34/109) Continual Road-Scene Semantic Segmentation via Feature-Aligned Symmetric Multi-Modal Network (Francesco Barbato et al., 2023)

{{<citation>}}

Francesco Barbato, Elena Camuffo, Simone Milani, Pietro Zanuttigh. (2023)  
**Continual Road-Scene Semantic Segmentation via Feature-Aligned Symmetric Multi-Modal Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.04702v1)  

---


**ABSTRACT**  
State-of-the-art multimodal semantic segmentation approaches combining LiDAR and color data are usually designed on top of asymmetric information-sharing schemes and assume that both modalities are always available. Regrettably, this strong assumption may not hold in real-world scenarios, where sensors are prone to failure or can face adverse conditions (night-time, rain, fog, etc.) that make the acquired information unreliable. Moreover, these architectures tend to fail in continual learning scenarios. In this work, we re-frame the task of multimodal semantic segmentation by enforcing a tightly-coupled feature representation and a symmetric information-sharing scheme, which allows our approach to work even when one of the input modalities is missing. This makes our model reliable even in safety-critical settings, as is the case of autonomous driving. We evaluate our approach on the SemanticKITTI dataset, comparing it with our closest competitor. We also introduce an ad-hoc continual learning scheme and show results in a class-incremental continual learning scenario that prove the effectiveness of the approach also in this setting.

{{</citation>}}


### (35/109) Rapid Training Data Creation by Synthesizing Medical Images for Classification and Localization (Abhishek Kushwaha et al., 2023)

{{<citation>}}

Abhishek Kushwaha, Sarthak Gupta, Anish Bhanushali, Tathagato Rai Dastidar. (2023)  
**Rapid Training Data Creation by Synthesizing Medical Images for Classification and Localization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04687v1)  

---


**ABSTRACT**  
While the use of artificial intelligence (AI) for medical image analysis is gaining wide acceptance, the expertise, time and cost required to generate annotated data in the medical field are significantly high, due to limited availability of both data and expert annotation. Strongly supervised object localization models require data that is exhaustively annotated, meaning all objects of interest in an image are identified. This is difficult to achieve and verify for medical images. We present a method for the transformation of real data to train any Deep Neural Network to solve the above problems. We show the efficacy of this approach on both a weakly supervised localization model and a strongly supervised localization model. For the weakly supervised model, we show that the localization accuracy increases significantly using the generated data. For the strongly supervised model, this approach overcomes the need for exhaustive annotation on real images. In the latter model, we show that the accuracy, when trained with generated images, closely parallels the accuracy when trained with exhaustively annotated real images. The results are demonstrated on images of human urine samples obtained using microscopy.

{{</citation>}}


### (36/109) Addressing Racial Bias in Facial Emotion Recognition (Alex Fan et al., 2023)

{{<citation>}}

Alex Fan, Xingshuo Xiao, Peter Washington. (2023)  
**Addressing Racial Bias in Facial Emotion Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-CY, cs.CV  
Keywords: Bias, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.04674v1)  

---


**ABSTRACT**  
Fairness in deep learning models trained with high-dimensional inputs and subjective labels remains a complex and understudied area. Facial emotion recognition, a domain where datasets are often racially imbalanced, can lead to models that yield disparate outcomes across racial groups. This study focuses on analyzing racial bias by sub-sampling training sets with varied racial distributions and assessing test performance across these simulations. Our findings indicate that smaller datasets with posed faces improve on both fairness and performance metrics as the simulations approach racial balance. Notably, the F1-score increases by $27.2\%$ points, and demographic parity increases by $15.7\%$ points on average across the simulations. However, in larger datasets with greater facial variation, fairness metrics generally remain constant, suggesting that racial balance by itself is insufficient to achieve parity in test performance across different racial groups.

{{</citation>}}


### (37/109) Which Tokens to Use? Investigating Token Reduction in Vision Transformers (Joakim Bruslund Haurum et al., 2023)

{{<citation>}}

Joakim Bruslund Haurum, Sergio Escalera, Graham W. Taylor, Thomas B. Moeslund. (2023)  
**Which Tokens to Use? Investigating Token Reduction in Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.04657v1)  

---


**ABSTRACT**  
Since the introduction of the Vision Transformer (ViT), researchers have sought to make ViTs more efficient by removing redundant information in the processed tokens. While different methods have been explored to achieve this goal, we still lack understanding of the resulting reduction patterns and how those patterns differ across token reduction methods and datasets. To close this gap, we set out to understand the reduction patterns of 10 different token reduction methods using four image classification datasets. By systematically comparing these methods on the different classification tasks, we find that the Top-K pruning method is a surprisingly strong baseline. Through in-depth analysis of the different methods, we determine that: the reduction patterns are generally not consistent when varying the capacity of the backbone model, the reduction patterns of pruning-based methods significantly differ from fixed radial patterns, and the reduction patterns of pruning-based methods are correlated across classification datasets. Finally we report that the similarity of reduction patterns is a moderate-to-strong proxy for model performance. Project page at https://vap.aau.dk/tokens.

{{</citation>}}


### (38/109) GeoAdapt: Self-Supervised Test-Time Adaption in LiDAR Place Recognition Using Geometric Priors (Joshua Knights et al., 2023)

{{<citation>}}

Joshua Knights, Stephen Hausler, Sridha Sridharan, Clinton Fookes, Peyman Moghadam. (2023)  
**GeoAdapt: Self-Supervised Test-Time Adaption in LiDAR Place Recognition Using Geometric Priors**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.04638v1)  

---


**ABSTRACT**  
LiDAR place recognition approaches based on deep learning suffer a significant degradation in performance when there is a shift between the distribution of the training and testing datasets, with re-training often required to achieve top performance. However, obtaining accurate ground truth on new environments can be prohibitively expensive, especially in complex or GPS-deprived environments. To address this issue we propose GeoAdapt, which introduces a novel auxiliary classification head to generate pseudo-labels for re-training on unseen environments in a self-supervised manner. GeoAdapt uses geometric consistency as a prior to improve the robustness of our generated pseudo-labels against domain shift, improving the performance and reliability of our Test-Time Adaptation approach. Comprehensive experiments show that GeoAdapt significantly boosts place recognition performance across moderate to severe domain shifts, and is competitive with fully supervised test-time adaptation approaches. Our code will be available at https://github.com/csiro-robotics/GeoAdapt.

{{</citation>}}


## cs.SI (3)



### (39/109) Social Network Analysis and Validation of an Agent-Based Model (Karleigh Pine et al., 2023)

{{<citation>}}

Karleigh Pine, Joel Klipfel, Jared Bennett, Nathaniel Bade, Christian Manasseh. (2023)  
**Social Network Analysis and Validation of an Agent-Based Model**  

---
Primary Category: cs.SI  
Categories: cs-DM, cs-MA, cs-SI, cs.SI, q-bio-QM  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2308.05256v1)  

---


**ABSTRACT**  
Agent-based models (ABMs) simulate the formation and evolution of social processes at a fundamental level by decoupling agent behavior from global observations. In the case where ABM networks evolve over time as a result of (or in conjunction with) agent states, there is a need for understanding the relationship between the dynamic processes and network structure. Social networks provide a natural set of tools for understanding the emergent relationships of these systems. This work examines the utility of a collection of network comparison methods for the purpose of tracking network changes in an ABM over time or between model parameters. Among the techniques examined is a novel graph pseudometric based on heat content asymptotics, which have been shown to distinguish many isospectral graphs which are not isomorphic. Additionally, we establish the use of observations about real-world networks from network science (e.g. fat-tailed degree distribution, small-world property) for ABM validation in the case where empirical population data is unavailable. These methods are all demonstrated on systematic perturbations of an original model simulating the formation of friendships in a population of 20,000 agents in Cincinnati, OH.

{{</citation>}}


### (40/109) TUBERAIDER: Attributing Coordinated Hate Attacks on YouTube Videos to their Source Communities (Mohammad Hammas Saeed et al., 2023)

{{<citation>}}

Mohammad Hammas Saeed, Kostantinos Papadamou, Jeremy Blackburn, Emiliano De Cristofaro, Gianluca Stringhini. (2023)  
**TUBERAIDER: Attributing Coordinated Hate Attacks on YouTube Videos to their Source Communities**  

---
Primary Category: cs.SI  
Categories: cs-CR, cs-SI, cs.SI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.05247v1)  

---


**ABSTRACT**  
Alas, coordinated hate attacks, or raids, are becoming increasingly common online. In a nutshell, these are perpetrated by a group of aggressors who organize and coordinate operations on a platform (e.g., 4chan) to target victims on another community (e.g., YouTube). In this paper, we focus on attributing raids to their source community, paving the way for moderation approaches that take the context (and potentially the motivation) of an attack into consideration. We present TUBERAIDER, an attribution system achieving over 75% accuracy in detecting and attributing coordinated hate attacks on YouTube videos. We instantiate it using links to YouTube videos shared on 4chan's /pol/ board, r/The_Donald, and 16 Incels-related subreddits. We use a peak detector to identify a rise in the comment activity of a YouTube video, which signals that an attack may be occurring. We then train a machine learning classifier based on the community language (i.e., TF-IDF scores of relevant keywords) to perform the attribution. We test TUBERAIDER in the wild and present a few case studies of actual aggression attacks identified by it to showcase its effectiveness.

{{</citation>}}


### (41/109) CasCIFF: A Cross-Domain Information Fusion Framework Tailored for Cascade Prediction in Social Networks (Hongjun Zhu et al., 2023)

{{<citation>}}

Hongjun Zhu, Shun Yuan, Xin Liu, Kuo Chen, Chaolong Jia, Ying Qian. (2023)  
**CasCIFF: A Cross-Domain Information Fusion Framework Tailored for Cascade Prediction in Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-LG, cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2308.04961v1)  

---


**ABSTRACT**  
Existing approaches for information cascade prediction fall into three main categories: feature-driven methods, point process-based methods, and deep learning-based methods. Among them, deep learning-based methods, characterized by its superior learning and representation capabilities, mitigates the shortcomings inherent of the other methods. However, current deep learning methods still face several persistent challenges. In particular, accurate representation of user attributes remains problematic due to factors such as fake followers and complex network configurations. Previous algorithms that focus on the sequential order of user activations often neglect the rich insights offered by activation timing. Furthermore, these techniques often fail to holistically integrate temporal and structural aspects, thus missing the nuanced propagation trends inherent in information cascades.To address these issues, we propose the Cross-Domain Information Fusion Framework (CasCIFF), which is tailored for information cascade prediction. This framework exploits multi-hop neighborhood information to make user embeddings robust. When embedding cascades, the framework intentionally incorporates timestamps, endowing it with the ability to capture evolving patterns of information diffusion. In particular, the CasCIFF seamlessly integrates the tasks of user classification and cascade prediction into a consolidated framework, thereby allowing the extraction of common features that prove useful for all tasks, a strategy anchored in the principles of multi-task learning.

{{</citation>}}


## cs.SE (7)



### (42/109) AI-Enabled Software and System Architecture Frameworks: Focusing on smart Cyber-Physical Systems (CPS) (Armin Moin et al., 2023)

{{<citation>}}

Armin Moin, Atta Badii, Stephan Günnemann, Moharram Challenger. (2023)  
**AI-Enabled Software and System Architecture Frameworks: Focusing on smart Cyber-Physical Systems (CPS)**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.05239v1)  

---


**ABSTRACT**  
Several architecture frameworks for software, systems, and enterprises have been proposed in the literature. They identified various stakeholders and defined architecture viewpoints and views to frame and address stakeholder concerns. However, the stakeholders with data science and Machine Learning (ML) related concerns, such as data scientists and data engineers, are yet to be included in existing architecture frameworks. Therefore, they failed to address the architecture viewpoints and views responsive to the concerns of the data science community. In this paper, we address this gap by establishing the architecture frameworks adapted to meet the requirements of modern applications and organizations where ML artifacts are both prevalent and crucial. In particular, we focus on ML-enabled Cyber-Physical Systems (CPSs) and propose two sets of merit criteria for their efficient development and performance assessment, namely the criteria for evaluating and benchmarking ML-enabled CPSs, and the criteria for evaluation and benchmarking of the tools intended to support users through the modeling and development pipeline. In this study, we deploy multiple empirical and qualitative research methods based on literature review and survey instruments including expert interviews and an online questionnaire. We collect, analyze, and integrate the opinions of 77 experts from more than 25 organizations in over 10 countries to devise and validate the proposed framework.

{{</citation>}}


### (43/109) Fixing Rust Compilation Errors using LLMs (Pantazis Deligiannis et al., 2023)

{{<citation>}}

Pantazis Deligiannis, Akash Lal, Nikita Mehrotra, Aseem Rastogi. (2023)  
**Fixing Rust Compilation Errors using LLMs**  

---
Primary Category: cs.SE  
Categories: cs-PL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.05177v1)  

---


**ABSTRACT**  
The Rust programming language, with its safety guarantees, has established itself as a viable choice for low-level systems programming language over the traditional, unsafe alternatives like C/C++. These guarantees come from a strong ownership-based type system, as well as primitive support for features like closures, pattern matching, etc., that make the code more concise and amenable to reasoning. These unique Rust features also pose a steep learning curve for programmers.   This paper presents a tool called RustAssistant that leverages the emergent capabilities of Large Language Models (LLMs) to automatically suggest fixes for Rust compilation errors. RustAssistant uses a careful combination of prompting techniques as well as iteration with an LLM to deliver high accuracy of fixes. RustAssistant is able to achieve an impressive peak accuracy of roughly 74% on real-world compilation errors in popular open-source Rust repositories. We plan to release our dataset of Rust compilation errors to enable further research.

{{</citation>}}


### (44/109) No Need to Lift a Finger Anymore? Assessing the Quality of Code Generation by ChatGPT (Zhijie Liu et al., 2023)

{{<citation>}}

Zhijie Liu, Yutian Tang, Xiapu Luo, Yuming Zhou, Liang Feng Zhang. (2023)  
**No Need to Lift a Finger Anymore? Assessing the Quality of Code Generation by ChatGPT**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2308.04838v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated impressive capabilities across various natural language processing (NLP) tasks, such as machine translation, question answering, summarization, and so on. Additionally, LLMs are also highly valuable in supporting software engineering tasks, particularly in the field of code generation. Automatic code generation is a process of automatically generating source code or executable code based on given specifications or requirements, improving developer productivity. In this study, we perform a systematic empirical assessment of code generation using ChatGPT, a recent and popular LLM. Our evaluation encompasses a comprehensive analysis of code snippets generated by ChatGPT, focusing on three critical aspects: correctness, understandability, and security. We also specifically investigate ChatGPT's ability to engage in multi-round process (i.e., ChatGPT's dialog ability) of facilitating code generation. By delving into the generated code and examining the experimental results, this work provides valuable insights into the performance of ChatGPT in tackling code generation tasks. Overall, our findings uncover potential issues and limitations that arise in the ChatGPT-based code generation and lay the groundwork for improving AI and LLM-based code generation techniques.

{{</citation>}}


### (45/109) Adaptive Intellect Unleashed: The Feasibility of Knowledge Transfer in Large Language Models (Qing Huang et al., 2023)

{{<citation>}}

Qing Huang, Yishun Wu, Zhenchang Xing, He Jiang, Yu Cheng, Huan Jin. (2023)  
**Adaptive Intellect Unleashed: The Feasibility of Knowledge Transfer in Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.04788v1)  

---


**ABSTRACT**  
We conduct the first empirical study on using knowledge transfer to improve the generalization ability of large language models (LLMs) in software engineering tasks, which often require LLMs to generalize beyond their training data. Our proposed general knowledge transfer approach guides the LLM towards a similar and familiar API or code snippet it has encountered before, improving the model's generalization ability for unseen knowledge. We apply this approach to three software engineering tasks: API inference, code example generation, and FQN inference, and find transfer span, transfer strategy, and transfer architecture as key factors affecting the method. Our findings demonstrate the feasibility of knowledge transfer and its potential to enhance LLMs' performance in various software engineering tasks. The effectiveness of knowledge transfer varies depending on the target domain and task, with the hierarchical strategy being more effective than direct transfer, and AI-Chain outperforming CoT in prompt design. The implications of these findings extend beyond software engineering tasks and suggest that knowledge transfer can enhance LLMs' ability to handle unknowns in any natural language task.

{{</citation>}}


### (46/109) Universal Fuzzing via Large Language Models (Chunqiu Steven Xia et al., 2023)

{{<citation>}}

Chunqiu Steven Xia, Matteo Paltenghi, Jia Le Tian, Michael Pradel, Lingming Zhang. (2023)  
**Universal Fuzzing via Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.04748v1)  

---


**ABSTRACT**  
Fuzzing has achieved tremendous success in discovering bugs and vulnerabilities in various software systems. Systems under test (SUTs) that take in programming or formal language as inputs, e.g., compilers, runtime engines, constraint solvers, and software libraries with accessible APIs, are especially important as they are fundamental building blocks of software development. However, existing fuzzers for such systems often target a specific language, and thus cannot be easily applied to other languages or even other versions of the same language. Moreover, the inputs generated by existing fuzzers are often limited to specific features of the input language, and thus can hardly reveal bugs related to other or new features. This paper presents Fuzz4All, the first fuzzer that is universal in the sense that it can target many different input languages and many different features of these languages. The key idea behind Fuzz4All is to leverage large language models (LLMs) as an input generation and mutation engine, which enables the approach to produce diverse and realistic inputs for any practically relevant language. To realize this potential, we present a novel autoprompting technique, which creates LLM prompts that are wellsuited for fuzzing, and a novel LLM-powered fuzzing loop, which iteratively updates the prompt to create new fuzzing inputs. We evaluate Fuzz4All on nine systems under test that take in six different languages (C, C++, Go, SMT2, Java and Python) as inputs. The evaluation shows, across all six languages, that universal fuzzing achieves higher coverage than existing, language-specific fuzzers. Furthermore, Fuzz4All has identified 76 bugs in widely used systems, such as GCC, Clang, Z3, CVC5, OpenJDK, and the Qiskit quantum computing platform, with 47 bugs already confirmed by developers as previously unknown.

{{</citation>}}


### (47/109) Case Study: Using AI-Assisted Code Generation In Mobile Teams (Mircea-Serban Vasiliniuc et al., 2023)

{{<citation>}}

Mircea-Serban Vasiliniuc, Adrian Groza. (2023)  
**Case Study: Using AI-Assisted Code Generation In Mobile Teams**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04736v1)  

---


**ABSTRACT**  
The aim of this study is to evaluate the performance of AI-assisted programming in actual mobile development teams that are focused on native mobile languages like Kotlin and Swift. The extensive case study involves 16 participants and 2 technical reviewers, from a software development department designed to understand the impact of using LLMs trained for code generation in specific phases of the team, more specifically, technical onboarding and technical stack switch. The study uses technical problems dedicated to each phase and requests solutions from the participants with and without using AI-Code generators. It measures time, correctness, and technical integration using ReviewerScore, a metric specific to the paper and extracted from actual industry standards, the code reviewers of merge requests. The output is converted and analyzed together with feedback from the participants in an attempt to determine if using AI-assisted programming tools will have an impact on getting developers onboard in a project or helping them with a smooth transition between the two native development environments of mobile development, Android and iOS. The study was performed between May and June 2023 with members of the mobile department of a software development company based in Cluj-Napoca, with Romanian ownership and management.

{{</citation>}}


### (48/109) Evaluating and Optimizing the Effectiveness of Neural Machine Translation in Supporting Code Retrieval Models: A Study on the CAT Benchmark (Hung Phan et al., 2023)

{{<citation>}}

Hung Phan, Ali Jannesari. (2023)  
**Evaluating and Optimizing the Effectiveness of Neural Machine Translation in Supporting Code Retrieval Models: A Study on the CAT Benchmark**  

---
Primary Category: cs.SE  
Categories: cs-IR, cs-SE, cs.SE  
Keywords: BERT, BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2308.04693v1)  

---


**ABSTRACT**  
Neural Machine Translation (NMT) is widely applied in software engineering tasks. The effectiveness of NMT for code retrieval relies on the ability to learn from the sequence of tokens in the source language to the sequence of tokens in the target language. While NMT performs well in pseudocode-to-code translation, it might have challenges in learning to translate from natural language query to source code in newly curated real-world code documentation/ implementation datasets. In this work, we analyze the performance of NMT in natural language-to-code translation in the newly curated CAT benchmark that includes the optimized versions of three Java datasets TLCodeSum, CodeSearchNet, Funcom, and a Python dataset PCSD. Our evaluation shows that NMT has low accuracy, measured by CrystalBLEU and Meteor metrics in this task. To alleviate the duty of NMT in learning complex representation of source code, we propose ASTTrans Representation, a tailored representation of an Abstract Syntax Tree (AST) using a subset of non-terminal nodes. We show that the classical approach NMT performs significantly better in learning ASTTrans Representation over code tokens with up to 36% improvement on Meteor score. Moreover, we leverage ASTTrans Representation to conduct combined code search processes from the state-of-the-art code search processes using GraphCodeBERT and UniXcoder. Our NMT models of learning ASTTrans Representation can boost the Mean Reciprocal Rank of these state-of-the-art code search processes by up to 3.08% and improve 23.08% of queries' results over the CAT benchmark.

{{</citation>}}


## quant-ph (1)



### (49/109) Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models (Nouhaila Innan et al., 2023)

{{<citation>}}

Nouhaila Innan, Muhammad Al-Zafar Khan, Mohamed Bennai. (2023)  
**Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models**  

---
Primary Category: quant-ph  
Categories: cs-LG, q-fin-GN, quant-ph, quant-ph  
Keywords: Financial, Fraud Detection  
[Paper Link](http://arxiv.org/abs/2308.05237v1)  

---


**ABSTRACT**  
In this research, a comparative study of four Quantum Machine Learning (QML) models was conducted for fraud detection in finance. We proved that the Quantum Support Vector Classifier model achieved the highest performance, with F1 scores of 0.98 for fraud and non-fraud classes. Other models like the Variational Quantum Classifier, Estimator Quantum Neural Network (QNN), and Sampler QNN demonstrate promising results, propelling the potential of QML classification for financial applications. While they exhibit certain limitations, the insights attained pave the way for future enhancements and optimisation strategies. However, challenges exist, including the need for more efficient Quantum algorithms and larger and more complex datasets. The article provides solutions to overcome current limitations and contributes new insights to the field of Quantum Machine Learning in fraud detection, with important implications for its future development.

{{</citation>}}


## cs.CR (8)



### (50/109) IoT Security: On-Chip Secure Deletion Scheme using ECC Modulation in IoT Appliances (Na Young Ahn et al., 2023)

{{<citation>}}

Na Young Ahn, Dong Hoon Lee. (2023)  
**IoT Security: On-Chip Secure Deletion Scheme using ECC Modulation in IoT Appliances**  

---
Primary Category: cs.CR  
Categories: 68M15, C-5-1, cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.05225v1)  

---


**ABSTRACT**  
NAND flash memory-based IoT devices inherently suffer from data retention issues. In IoT security, these retention issues are significant and require a robust solution for secure deletion. Secure deletion methods can be categorized into off-chip and on-chip schemes. Off-chip secure deletion schemes, based on block-level erasure operations, are unable to perform real-time trim operations. Consequently, they are vulnerable to hacking threats. On the other hand, on-chip secure deletion schemes enable real-time trim operations by performing deletion on a page-by-page basis. However, the on-chip scheme introduces a challenge of program disturbance for neighboring page data. The proposed on-chip deletion scheme tackles this problem by utilizing ECC code modulation through a partial program operation. This approach significantly reduces the program disturbance issue associated with neighboring page data. Moreover, the proposed code modulation secure deletion scheme allows for real-time verification of the deletion of original data.

{{</citation>}}


### (51/109) Kairos: : Practical Intrusion Detection and Investigation using Whole-system Provenance (Zijun Cheng et al., 2023)

{{<citation>}}

Zijun Cheng, Qiujian Lv, Jinyuan Liang, Yan Wang, Degang Sun, Thomas Pasquier, Xueyuan Han. (2023)  
**Kairos: : Practical Intrusion Detection and Investigation using Whole-system Provenance**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2308.05034v1)  

---


**ABSTRACT**  
Provenance graphs are structured audit logs that describe the history of a system's execution. Recent studies have explored a variety of techniques to analyze provenance graphs for automated host intrusion detection, focusing particularly on advanced persistent threats. Sifting through their design documents, we identify four common dimensions that drive the development of provenance-based intrusion detection systems (PIDSes): scope (can PIDSes detect modern attacks that infiltrate across application boundaries?), attack agnosticity (can PIDSes detect novel attacks without a priori knowledge of attack characteristics?), timeliness (can PIDSes efficiently monitor host systems as they run?), and attack reconstruction (can PIDSes distill attack activity from large provenance graphs so that sysadmins can easily understand and quickly respond to system intrusion?). We present KAIROS, the first PIDS that simultaneously satisfies the desiderata in all four dimensions, whereas existing approaches sacrifice at least one and struggle to achieve comparable detection performance.   Kairos leverages a novel graph neural network-based encoder-decoder architecture that learns the temporal evolution of a provenance graph's structural changes to quantify the degree of anomalousness for each system event. Then, based on this fine-grained information, Kairos reconstructs attack footprints, generating compact summary graphs that accurately describe malicious activity over a stream of system audit logs. Using state-of-the-art benchmark datasets, we demonstrate that Kairos outperforms previous approaches.

{{</citation>}}


### (52/109) An Empirical Study on Using Large Language Models to Analyze Software Supply Chain Security Failures (Tanmay Singla et al., 2023)

{{<citation>}}

Tanmay Singla, Dharun Anandayuvaraj, Kelechi G. Kalu, Taylor R. Schorlemmer, James C. Davis. (2023)  
**An Empirical Study on Using Large Language Models to Analyze Software Supply Chain Security Failures**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs-SE, cs.CR  
Keywords: GPT, Language Model, NLP, Natural Language Processing, Security  
[Paper Link](http://arxiv.org/abs/2308.04898v1)  

---


**ABSTRACT**  
As we increasingly depend on software systems, the consequences of breaches in the software supply chain become more severe. High-profile cyber attacks like those on SolarWinds and ShadowHammer have resulted in significant financial and data losses, underlining the need for stronger cybersecurity. One way to prevent future breaches is by studying past failures. However, traditional methods of analyzing these failures require manually reading and summarizing reports about them. Automated support could reduce costs and allow analysis of more failures. Natural Language Processing (NLP) techniques such as Large Language Models (LLMs) could be leveraged to assist the analysis of failures. In this study, we assessed the ability of Large Language Models (LLMs) to analyze historical software supply chain breaches. We used LLMs to replicate the manual analysis of 69 software supply chain security failures performed by members of the Cloud Native Computing Foundation (CNCF). We developed prompts for LLMs to categorize these by four dimensions: type of compromise, intent, nature, and impact. GPT 3.5s categorizations had an average accuracy of 68% and Bard had an accuracy of 58% over these dimensions. We report that LLMs effectively characterize software supply chain failures when the source articles are detailed enough for consensus among manual analysts, but cannot yet replace human analysts. Future work can improve LLM performance in this context, and study a broader range of articles and failures.

{{</citation>}}


### (53/109) can-train-and-test: A Curated CAN Dataset for Automotive Intrusion Detection (Brooke Lampe et al., 2023)

{{<citation>}}

Brooke Lampe, Weizhi Meng. (2023)  
**can-train-and-test: A Curated CAN Dataset for Automotive Intrusion Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2308.04972v1)  

---


**ABSTRACT**  
When it comes to in-vehicle networks (IVNs), the controller area network -- CAN -- bus dominates the market; automobiles manufactured and sold around the world depend on the CAN bus for safety-critical communications between various components of the vehicle (e.g., the engine, the transmission, the steering column). Unfortunately, the CAN bus is inherently insecure; in fact, it completely lacks controls such as authentication, authorization, and confidentiality (i.e., encryption). Therefore, researchers have travailed to develop automotive security enhancements. The automotive intrusion detection system (IDS) is especially popular in the literature -- due to its relatively low cost in terms of money, resource utilization, and implementation effort. That said, developing and evaluating an automotive IDS is often challenging; if researchers do not have access to a test vehicle, then they are forced to depend on publicly available CAN data -- which is not without limitations. Lack of access to adequate CAN data, then, becomes a barrier to entry into automotive security research.   We seek to lower that barrier to entry by introducing a new CAN dataset to facilitate the development and evaluation of automotive IDSs. Our dataset, dubbed can-train-and-test, provides CAN data from four different vehicles produced by two different manufacturers. The attack captures for each vehicle model are equivalent, enabling researchers to assess the ability of a given IDS to generalize to different vehicle models and even different vehicle manufacturers. Our dataset contains replayable .log files as well as labeled and unlabeled .csv files, thereby meeting a variety of development and evaluation needs. Furthermore, can-train-and-test offers nine unique attacks, ranging from denial of service (DoS) to gear spoofing to standstill...

{{</citation>}}


### (54/109) Adversarial Deep Reinforcement Learning for Cyber Security in Software Defined Networks (Luke Borchjes et al., 2023)

{{<citation>}}

Luke Borchjes, Clement Nyirenda, Louise Leenen. (2023)  
**Adversarial Deep Reinforcement Learning for Cyber Security in Software Defined Networks**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Cyber Security, Reinforcement Learning, Security  
[Paper Link](http://arxiv.org/abs/2308.04909v1)  

---


**ABSTRACT**  
This paper focuses on the impact of leveraging autonomous offensive approaches in Deep Reinforcement Learning (DRL) to train more robust agents by exploring the impact of applying adversarial learning to DRL for autonomous security in Software Defined Networks (SDN). Two algorithms, Double Deep Q-Networks (DDQN) and Neural Episodic Control to Deep Q-Network (NEC2DQN or N2D), are compared. NEC2DQN was proposed in 2018 and is a new member of the deep q-network (DQN) family of algorithms. The attacker has full observability of the environment and access to a causative attack that uses state manipulation in an attempt to poison the learning process. The implementation of the attack is done under a white-box setting, in which the attacker has access to the defender's model and experiences. Two games are played; in the first game, DDQN is a defender and N2D is an attacker, and in second game, the roles are reversed. The games are played twice; first, without an active causative attack and secondly, with an active causative attack. For execution, three sets of game results are recorded in which a single set consists of 10 game runs. The before and after results are then compared in order to see if there was actually an improvement or degradation. The results show that with minute parameter changes made to the algorithms, there was growth in the attacker's role, since it is able to win games. Implementation of the adversarial learning by the introduction of the causative attack showed the algorithms are still able to defend the network according to their strengths.

{{</citation>}}


### (55/109) Data-Free Model Extraction Attacks in the Context of Object Detection (Harshit Shah et al., 2023)

{{<citation>}}

Harshit Shah, Aravindhan G, Pavan Kulkarni, Yuvaraj Govidarajulu, Manojkumar Parmar. (2023)  
**Data-Free Model Extraction Attacks in the Context of Object Detection**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.05127v1)  

---


**ABSTRACT**  
A significant number of machine learning models are vulnerable to model extraction attacks, which focus on stealing the models by using specially curated queries against the target model. This task is well accomplished by using part of the training data or a surrogate dataset to train a new model that mimics a target model in a white-box environment. In pragmatic situations, however, the target models are trained on private datasets that are inaccessible to the adversary. The data-free model extraction technique replaces this problem when it comes to using queries artificially curated by a generator similar to that used in Generative Adversarial Nets. We propose for the first time, to the best of our knowledge, an adversary black box attack extending to a regression problem for predicting bounding box coordinates in object detection. As part of our study, we found that defining a loss function and using a novel generator setup is one of the key aspects in extracting the target model. We find that the proposed model extraction method achieves significant results by using reasonable queries. The discovery of this object detection vulnerability will support future prospects for securing such models.

{{</citation>}}


### (56/109) Data-Driven Intelligence can Revolutionize Today's Cybersecurity World: A Position Paper (Iqbal H. Sarker et al., 2023)

{{<citation>}}

Iqbal H. Sarker, Helge Janicke, Leandros Maglaras, Seyit Camtepe. (2023)  
**Data-Driven Intelligence can Revolutionize Today's Cybersecurity World: A Position Paper**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.05126v1)  

---


**ABSTRACT**  
As cyber threats evolve and grow progressively more sophisticated, cyber security is becoming a more significant concern in today's digital era. Traditional security measures tend to be insufficient to defend against these persistent and dynamic threats because they are mainly intuitional. One of the most promising ways to handle this ongoing problem is utilizing the potential of data-driven intelligence, by leveraging AI and machine learning techniques. It can improve operational efficiency and saves response times by automating repetitive operations, enabling real-time threat detection, and facilitating incident response. In addition, it augments human expertise with insightful information, predictive analytics, and enhanced decision-making, enabling them to better understand and address evolving problems. Thus, data-driven intelligence could significantly improve real-world cybersecurity solutions in a wide range of application areas like critical infrastructure, smart cities, digital twin, industrial control systems and so on. In this position paper, we argue that data-driven intelligence can revolutionize the realm of cybersecurity, offering not only large-scale task automation but also assist human experts for better situation awareness and decision-making in real-world scenarios.

{{</citation>}}


### (57/109) VulLibGen: Identifying Vulnerable Third-Party Libraries via Generative Pre-Trained Model (Tianyu Chen et al., 2023)

{{<citation>}}

Tianyu Chen, Lin Li, Liuchuan Zhu, Zongyang Li, Guangtai Liang, Ding Li, Qianxiang Wang, Tao Xie. (2023)  
**VulLibGen: Identifying Vulnerable Third-Party Libraries via Generative Pre-Trained Model**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model, Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2308.04662v1)  

---


**ABSTRACT**  
To avoid potential risks posed by vulnerabilities in third-party libraries, security researchers maintain vulnerability databases (e.g., NVD) containing vulnerability reports, each of which records the description of a vulnerability and the name list of libraries affected by the vulnerability (a.k.a. vulnerable libraries). However, recent studies on about 200,000 vulnerability reports in NVD show that 53.3% of these reports do not include the name list of vulnerable libraries, and 59.82% of the included name lists of vulnerable libraries are incomplete or incorrect.   To address the preceding issue, in this paper, we propose the first generative approach named VulLibGen to generate the name list of vulnerable libraries (out of all the existing libraries) for the given vulnerability by utilizing recent enormous advances in Large Language Models (LLMs), in order to achieve high accuracy. VulLibGen takes only the description of a vulnerability as input and achieves high identification accuracy based on LLMs' prior knowledge of all the existing libraries. VulLibGen also includes the input augmentation technique to help identify zero-shot vulnerable libraries (those not occurring during training) and the post-processing technique to help address VulLibGen's hallucinations. We evaluate VulLibGen using three state-of-the-art/practice approaches (LightXML, Chronos, and VulLibMiner) that identify vulnerable libraries on an open-source dataset (VulLib). Our evaluation results show that VulLibGen can accurately identify vulnerable libraries with an average F1 score of 0.626 while the state-of-the-art/practice approaches achieve only 0.561. The post-processing technique helps VulLibGen achieve an average improvement of F1@1 by 9.3%. The input augmentation technique helps VulLibGen achieve an average improvement of F1@1 by 39% in identifying zero-shot libraries.

{{</citation>}}


## cs.HC (1)



### (58/109) Alexa, play with robot: Introducing the First Alexa Prize SimBot Challenge on Embodied AI (Hangjie Shi et al., 2023)

{{<citation>}}

Hangjie Shi, Leslie Ball, Govind Thattai, Desheng Zhang, Lucy Hu, Qiaozi Gao, Suhaila Shakiah, Xiaofeng Gao, Aishwarya Padmakumar, Bofei Yang, Cadence Chung, Dinakar Guthy, Gaurav Sukhatme, Karthika Arumugam, Matthew Wen, Osman Ipek, Patrick Lange, Rohan Khanna, Shreyas Pansare, Vasu Sharma, Chao Zhang, Cris Flagg, Daniel Pressel, Lavina Vaz, Luke Dai, Prasoon Goyal, Sattvik Sahai, Shaohua Liu, Yao Lu, Anna Gottardi, Shui Hu, Yang Liu, Dilek Hakkani-Tur, Kate Bland, Heather Rocker, James Jeun, Yadunandana Rao, Michael Johnston, Akshaya Iyengar, Arindam Mandal, Prem Natarajan, Reza Ghanadan. (2023)  
**Alexa, play with robot: Introducing the First Alexa Prize SimBot Challenge on Embodied AI**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-RO, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.05221v1)  

---


**ABSTRACT**  
The Alexa Prize program has empowered numerous university students to explore, experiment, and showcase their talents in building conversational agents through challenges like the SocialBot Grand Challenge and the TaskBot Challenge. As conversational agents increasingly appear in multimodal and embodied contexts, it is important to explore the affordances of conversational interaction augmented with computer vision and physical embodiment. This paper describes the SimBot Challenge, a new challenge in which university teams compete to build robot assistants that complete tasks in a simulated physical environment. This paper provides an overview of the SimBot Challenge, which included both online and offline challenge phases. We describe the infrastructure and support provided to the teams including Alexa Arena, the simulated environment, and the ML toolkit provided to teams to accelerate their building of vision and language models. We summarize the approaches the participating teams took to overcome research challenges and extract key lessons learned. Finally, we provide analysis of the performance of the competing SimBots during the competition.

{{</citation>}}


## cs.CL (19)



### (59/109) Decoding Layer Saliency in Language Transformers (Elizabeth M. Hou et al., 2023)

{{<citation>}}

Elizabeth M. Hou, Gregory Castanon. (2023)  
**Decoding Layer Saliency in Language Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.05219v1)  

---


**ABSTRACT**  
In this paper, we introduce a strategy for identifying textual saliency in large-scale language models applied to classification tasks. In visual networks where saliency is more well-studied, saliency is naturally localized through the convolutional layers of the network; however, the same is not true in modern transformer-stack networks used to process natural language. We adapt gradient-based saliency methods for these networks, propose a method for evaluating the degree of semantic coherence of each layer, and demonstrate consistent improvement over numerous other methods for textual saliency on multiple benchmark classification datasets. Our approach requires no additional training or access to labelled data, and is comparatively very computationally efficient.

{{</citation>}}


### (60/109) RadGraph2: Modeling Disease Progression in Radiology Reports via Hierarchical Information Extraction (Sameer Khanna et al., 2023)

{{<citation>}}

Sameer Khanna, Adam Dejl, Kibo Yoon, Quoc Hung Truong, Hanh Duong, Agustina Saenz, Pranav Rajpurkar. (2023)  
**RadGraph2: Modeling Disease Progression in Radiology Reports via Hierarchical Information Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2308.05046v1)  

---


**ABSTRACT**  
We present RadGraph2, a novel dataset for extracting information from radiology reports that focuses on capturing changes in disease state and device placement over time. We introduce a hierarchical schema that organizes entities based on their relationships and show that using this hierarchy during training improves the performance of an information extraction model. Specifically, we propose a modification to the DyGIE++ framework, resulting in our model HGIE, which outperforms previous models in entity and relation extraction tasks. We demonstrate that RadGraph2 enables models to capture a wider variety of findings and perform better at relation extraction compared to those trained on the original RadGraph dataset. Our work provides the foundation for developing automated systems that can track disease progression over time and develop information extraction models that leverage the natural hierarchy of labels in the medical domain.

{{</citation>}}


### (61/109) AspectMMKG: A Multi-modal Knowledge Graph with Aspect-aware Entities (Jingdan Zhang et al., 2023)

{{<citation>}}

Jingdan Zhang, Jiaan Wang, Xiaodan Wang, Zhixu Li, Yanghua Xiao. (2023)  
**AspectMMKG: A Multi-modal Knowledge Graph with Aspect-aware Entities**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.04992v1)  

---


**ABSTRACT**  
Multi-modal knowledge graphs (MMKGs) combine different modal data (e.g., text and image) for a comprehensive understanding of entities. Despite the recent progress of large-scale MMKGs, existing MMKGs neglect the multi-aspect nature of entities, limiting the ability to comprehend entities from various perspectives. In this paper, we construct AspectMMKG, the first MMKG with aspect-related images by matching images to different entity aspects. Specifically, we collect aspect-related images from a knowledge base, and further extract aspect-related sentences from the knowledge base as queries to retrieve a large number of aspect-related images via an online image search engine. Finally, AspectMMKG contains 2,380 entities, 18,139 entity aspects, and 645,383 aspect-related images. We demonstrate the usability of AspectMMKG in entity aspect linking (EAL) downstream task and show that previous EAL models achieve a new state-of-the-art performance with the help of AspectMMKG. To facilitate the research on aspect-related MMKG, we further propose an aspect-related image retrieval (AIR) model, that aims to correct and expand aspect-related images in AspectMMKG. We train an AIR model to learn the relationship between entity image and entity aspect-related images by incorporating entity image, aspect, and aspect image information. Experimental results indicate that the AIR model could retrieve suitable images for a given entity w.r.t different aspects.

{{</citation>}}


### (62/109) Exploring Multilingual Text Data Distillation (Shivam Sahni et al., 2023)

{{<citation>}}

Shivam Sahni, Harsh Patel. (2023)  
**Exploring Multilingual Text Data Distillation**  

---
Primary Category: cs.CL  
Categories: F-2-2, I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2308.04982v1)  

---


**ABSTRACT**  
With the rise of deep learning, large datasets and complex models have become common, requiring significant computing power. To address this, data distillation has emerged as a technique to quickly train models with lower memory and time requirements. However, data distillation on text-based datasets hasn't been explored much because of the challenges rising due to its discrete nature. Additionally, existing dataset distillation methods often struggle to generalize to new architectures. In the paper, we propose several data distillation techniques for multilingual text classification datasets using language-model-based learning methods. We conduct experiments to analyze their performance in terms of classification strength, and cross-architecture generalization. Furthermore, we investigate the language-specific fairness of the data summaries generated by these methods. Our approach builds upon existing techniques, enhancing cross-architecture generalization in the text data distillation domain.

{{</citation>}}


### (63/109) Performance Analysis of Transformer Based Models (BERT, ALBERT and RoBERTa) in Fake News Detection (Shafna Fitria Nur Azizah et al., 2023)

{{<citation>}}

Shafna Fitria Nur Azizah, Hasan Dwi Cahyono, Sari Widya Sihwi, Wisnu Widiarto. (2023)  
**Performance Analysis of Transformer Based Models (BERT, ALBERT and RoBERTa) in Fake News Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: AI, BERT, Fake News, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.04950v1)  

---


**ABSTRACT**  
Fake news is fake material in a news media format but is not processed properly by news agencies. The fake material can provoke or defame significant entities or individuals or potentially even for the personal interests of the creators, causing problems for society. Distinguishing fake news and real news is challenging due to limited of domain knowledge and time constraints. According to the survey, the top three areas most exposed to hoaxes and misinformation by residents are in Banten, DKI Jakarta and West Java. The model of transformers is referring to an approach in the field of artificial intelligence (AI) in natural language processing utilizing the deep learning architectures. Transformers exercise a powerful attention mechanism to process text in parallel and produce rich and contextual word representations. A previous study indicates a superior performance of a transformer model known as BERT over and above non transformer approach. However, some studies suggest the performance can be improved with the use of improved BERT models known as ALBERT and RoBERTa. However, the modified BERT models are not well explored for detecting fake news in Bahasa Indonesia. In this research, we explore those transformer models and found that ALBERT outperformed other models with 87.6% accuracy, 86.9% precision, 86.9% F1-score, and 174.5 run-time (s/epoch) respectively. Source code available at: https://github.com/Shafna81/fakenewsdetection.git

{{</citation>}}


### (64/109) Extrapolating Large Language Models to Non-English by Aligning Languages (Wenhao Zhu et al., 2023)

{{<citation>}}

Wenhao Zhu, Yunzhe Lv, Qingxiu Dong, Fei Yuan, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, Lei Li. (2023)  
**Extrapolating Large Language Models to Non-English by Aligning Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2308.04948v1)  

---


**ABSTRACT**  
Due to the unbalanced training data distribution, the language ability of large language models (LLMs) is often biased towards English. In this paper, we propose to empower pre-trained LLMs on non-English languages by building semantic alignment across languages. We perform instruction-tuning on LLaMA with both translation task data and cross-lingual general task data to obtain cross-lingual models (x-LLaMA). Experiment results on cross-lingual benchmark XQUAD and MLQA show that x-LLaMA models outperform the English instruction-tuned counterpart (Alpaca) by 42.50% on average on six non-English languages. Further experiments on Chinese benchmark C-Eval show that x-LLaMA achieves significant improvement on Chinese humanities tasks, outperforming Alpaca by 8.2%. We also discover that incorporating non-English text on the target side of translation data is particularly effective for boosting non-English ability. Besides, we find that semantic alignment within LLM can be further strengthened as translation task data scales up and we present the formulation of the underlying scaling law. Evaluation results on translation dataset Flores-101 show that \method outperforms previous LLaMA-based models in all evaluated directions. Code and data will be available at: https://github.com/OwenNJU/x-LLM.

{{</citation>}}


### (65/109) LLMeBench: A Flexible Framework for Accelerating LLMs Benchmarking (Fahim Dalvi et al., 2023)

{{<citation>}}

Fahim Dalvi, Maram Hasanain, Sabri Boughorbel, Basel Mousi, Samir Abdaljalil, Nizi Nazar, Ahmed Abdelali, Shammur Absar Chowdhury, Hamdy Mubarak, Ahmed Ali, Majd Hawasly, Nadir Durrani, Firoj Alam. (2023)  
**LLMeBench: A Flexible Framework for Accelerating LLMs Benchmarking**  

---
Primary Category: cs.CL  
Categories: 68T50, F-2-2; I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: AI, BLOOM, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.04945v1)  

---


**ABSTRACT**  
The recent development and success of Large Language Models (LLMs) necessitate an evaluation of their performance across diverse NLP tasks in different languages. Although several frameworks have been developed and made publicly available, their customization capabilities for specific tasks and datasets are often complex for different users. In this study, we introduce the LLMeBench framework. Initially developed to evaluate Arabic NLP tasks using OpenAI's GPT and BLOOM models; it can be seamlessly customized for any NLP task and model, regardless of language. The framework also features zero- and few-shot learning settings. A new custom dataset can be added in less than 10 minutes, and users can use their own model API keys to evaluate the task at hand. The developed framework has been already tested on 31 unique NLP tasks using 53 publicly available datasets within 90 experimental setups, involving approximately 296K data points. We plan to open-source the framework for the community (https://github.com/qcri/LLMeBench/). A video demonstrating the framework is available online (https://youtu.be/FkQn4UjYA0s).

{{</citation>}}


### (66/109) LLaMA-E: Empowering E-commerce Authoring with Multi-Aspect Instruction Following (Kaize Shi et al., 2023)

{{<citation>}}

Kaize Shi, Xueyao Sun, Dingxian Wang, Yinlin Fu, Guandong Xu, Qing Li. (2023)  
**LLaMA-E: Empowering E-commerce Authoring with Multi-Aspect Instruction Following**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: GPT, GPT-3.5, LLaMA  
[Paper Link](http://arxiv.org/abs/2308.04913v1)  

---


**ABSTRACT**  
E-commerce authoring involves creating attractive, abundant, and targeted promotional content to drive product sales. The emergence of large language models (LLMs) introduces an innovative paradigm, offering a unified solution to address various authoring tasks within this scenario. However, mainstream LLMs trained on general corpora with common sense knowledge reveal limitations in fitting complex and personalized features unique to e-commerce products and customers. Furthermore, LLMs like GPT-3.5 necessitate remote accessibility, raising concerns about safeguarding voluminous customer privacy data during transmission. This paper proposes the LLaMA-E, the unified and customized instruction-following language models focusing on diverse e-commerce authoring tasks. Specifically, the domain experts create the seed instruction set from the tasks of ads generation, query-enhanced product title rewriting, product classification, purchase intent speculation, and general Q&A. These tasks enable the models to comprehensively understand precise e-commerce authoring knowledge by interleaving features covering typical service aspects of customers, sellers, and platforms. The GPT-3.5 is introduced as a teacher model, which expands the seed instructions to form a training set for the LLaMA-E models with various scales. The experimental results show that the proposed LLaMA-E models achieve state-of-the-art results in quantitative and qualitative evaluations, also exhibiting the advantage in zero-shot scenes. To the best of our knowledge, this study is the first to serve the LLMs to specific e-commerce authoring scenarios.

{{</citation>}}


### (67/109) Emotion-Conditioned Text Generation through Automatic Prompt Optimization (Yarik Menchaca Resendiz et al., 2023)

{{<citation>}}

Yarik Menchaca Resendiz, Roman Klinger. (2023)  
**Emotion-Conditioned Text Generation through Automatic Prompt Optimization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2308.04857v1)  

---


**ABSTRACT**  
Conditional natural language generation methods often require either expensive fine-tuning or training a large language model from scratch. Both are unlikely to lead to good results without a substantial amount of data and computational resources. Prompt learning without changing the parameters of a large language model presents a promising alternative. It is a cost-effective approach, while still achieving competitive results. While this procedure is now established for zero- and few-shot text classification and structured prediction, it has received limited attention in conditional text generation. We present the first automatic prompt optimization approach for emotion-conditioned text generation with instruction-fine-tuned models. Our method uses an iterative optimization procedure that changes the prompt by adding, removing, or replacing tokens. As objective function, we only require a text classifier that measures the realization of the conditional variable in the generated text. We evaluate the method on emotion-conditioned text generation with a focus on event reports and compare it to manually designed prompts that also act as the seed for the optimization procedure. The optimized prompts achieve 0.75 macro-average F1 to fulfill the emotion condition in contrast to manually designed seed prompts with only 0.22 macro-average F1.

{{</citation>}}


### (68/109) Evaluating the Generation Capabilities of Large Chinese Language Models (Hui Zeng et al., 2023)

{{<citation>}}

Hui Zeng, Jingyuan Xue, Meng Hao, Chen Sun, Bin Ning, Na Zhang. (2023)  
**Evaluating the Generation Capabilities of Large Chinese Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.04823v1)  

---


**ABSTRACT**  
This paper presents CG-Eval, the first comprehensive evaluation of the generation capabilities of large Chinese language models across a wide range of academic disciplines. The models' performance was assessed based on their ability to generate accurate and relevant responses to different types of questions in six disciplines, namely, Science and Engineering, Humanities and Social Sciences, Mathematical Calculations, Medical Practitioner Qualification Examination, Judicial Examination, and Certified Public Accountant Examination. This paper also presents Gscore, a composite index derived from the weighted sum of multiple metrics to measure the quality of model's generation against a reference. The test data and test results can be found at http://cgeval.besteasy.com/.

{{</citation>}}


### (69/109) CLEVA: Chinese Language Models EVAluation Platform (Yanyang Li et al., 2023)

{{<citation>}}

Yanyang Li, Jianqiao Zhao, Duo Zheng, Zi-Yuan Hu, Zhi Chen, Xiaohui Su, Yongfeng Huang, Shijia Huang, Dahua Lin, Michael R. Lyu, Liwei Wang. (2023)  
**CLEVA: Chinese Language Models EVAluation Platform**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.04813v1)  

---


**ABSTRACT**  
With the continuous emergence of Chinese Large Language Models (LLMs), how to evaluate a model's capabilities has become an increasingly significant issue. The absence of a comprehensive Chinese benchmark that thoroughly assesses a model's performance, the unstandardized and incomparable prompting procedure, and the prevalent risk of contamination pose major challenges in the current evaluation of Chinese LLMs. We present CLEVA, a user-friendly platform crafted to holistically evaluate Chinese LLMs. Our platform employs a standardized workflow to assess LLMs' performance across various dimensions, regularly updating a competitive leaderboard. To alleviate contamination, CLEVA curates a significant proportion of new data and develops a sampling strategy that guarantees a unique subset for each leaderboard round. Empowered by an easy-to-use interface that requires just a few mouse clicks and a model API, users can conduct a thorough evaluation with minimal coding. Large-scale experiments featuring 23 influential Chinese LLMs have validated CLEVA's efficacy.

{{</citation>}}


### (70/109) A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with Commonsense Knowledge (Kailai Yang et al., 2023)

{{<citation>}}

Kailai Yang, Tianlin Zhang, Shaoxiong Ji, Sophia Ananiadou. (2023)  
**A Bipartite Graph is All We Need for Enhancing Emotional Reasoning with Commonsense Knowledge**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Commonsense Knowledge, Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2308.04811v1)  

---


**ABSTRACT**  
The context-aware emotional reasoning ability of AI systems, especially in conversations, is of vital importance in applications such as online opinion mining from social media and empathetic dialogue systems. Due to the implicit nature of conveying emotions in many scenarios, commonsense knowledge is widely utilized to enrich utterance semantics and enhance conversation modeling. However, most previous knowledge infusion methods perform empirical knowledge filtering and design highly customized architectures for knowledge interaction with the utterances, which can discard useful knowledge aspects and limit their generalizability to different knowledge sources. Based on these observations, we propose a Bipartite Heterogeneous Graph (BHG) method for enhancing emotional reasoning with commonsense knowledge. In BHG, the extracted context-aware utterance representations and knowledge representations are modeled as heterogeneous nodes. Two more knowledge aggregation node types are proposed to perform automatic knowledge filtering and interaction. BHG-based knowledge infusion can be directly generalized to multi-type and multi-grained knowledge sources. In addition, we propose a Multi-dimensional Heterogeneous Graph Transformer (MHGT) to perform graph reasoning, which can retain unchanged feature spaces and unequal dimensions for heterogeneous node types during inference to prevent unnecessary loss of information. Experiments show that BHG-based methods significantly outperform state-of-the-art knowledge infusion methods and show generalized knowledge infusion ability with higher efficiency. Further analysis proves that previous empirical knowledge filtering methods do not guarantee to provide the most useful knowledge information. Our code is available at: https://github.com/SteveKGYang/BHG.

{{</citation>}}


### (71/109) ADMUS: A Progressive Question Answering Framework Adaptable to Multiple Knowledge Sources (Yirui Zhan et al., 2023)

{{<citation>}}

Yirui Zhan, Yanzeng Li, Minhao Zhang, Lei Zou. (2023)  
**ADMUS: A Progressive Question Answering Framework Adaptable to Multiple Knowledge Sources**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.04800v1)  

---


**ABSTRACT**  
With the introduction of deep learning models, semantic parsingbased knowledge base question answering (KBQA) systems have achieved high performance in handling complex questions. However, most existing approaches primarily focus on enhancing the model's effectiveness on individual benchmark datasets, disregarding the high costs of adapting the system to disparate datasets in real-world scenarios (e.g., multi-tenant platform). Therefore, we present ADMUS, a progressive knowledge base question answering framework designed to accommodate a wide variety of datasets, including multiple languages, diverse backbone knowledge bases, and disparate question answering datasets. To accomplish the purpose, we decouple the architecture of conventional KBQA systems and propose this dataset-independent framework. Our framework supports the seamless integration of new datasets with minimal effort, only requiring creating a dataset-related micro-service at a negligible cost. To enhance the usability of ADUMS, we design a progressive framework consisting of three stages, ranges from executing exact queries, generating approximate queries and retrieving open-domain knowledge referring from large language models. An online demonstration of ADUMS is available at: https://answer.gstore.cn/pc/index.html

{{</citation>}}


### (72/109) Building Interpretable and Reliable Open Information Retriever for New Domains Overnight (Xiaodong Yu et al., 2023)

{{<citation>}}

Xiaodong Yu, Ben Zhou, Dan Roth. (2023)  
**Building Interpretable and Reliable Open Information Retriever for New Domains Overnight**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.04756v1)  

---


**ABSTRACT**  
Information retrieval (IR) or knowledge retrieval, is a critical component for many down-stream tasks such as open-domain question answering (QA). It is also very challenging, as it requires succinctness, completeness, and correctness. In recent works, dense retrieval models have achieved state-of-the-art (SOTA) performance on in-domain IR and QA benchmarks by representing queries and knowledge passages with dense vectors and learning the lexical and semantic similarity. However, using single dense vectors and end-to-end supervision are not always optimal because queries may require attention to multiple aspects and event implicit knowledge. In this work, we propose an information retrieval pipeline that uses entity/event linking model and query decomposition model to focus more accurately on different information units of the query. We show that, while being more interpretable and reliable, our proposed pipeline significantly improves passage coverages and denotation accuracies across five IR and QA benchmarks. It will be the go-to system to use for applications that need to perform IR on a new domain without much dedicated effort, because of its superior interpretability and cross-domain performance.

{{</citation>}}


### (73/109) Slot Induction via Pre-trained Language Model Probing and Multi-level Contrastive Learning (Hoang H. Nguyen et al., 2023)

{{<citation>}}

Hoang H. Nguyen, Chenwei Zhang, Ye Liu, Philip S. Yu. (2023)  
**Slot Induction via Pre-trained Language Model Probing and Multi-level Contrastive Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Contrastive Learning, Dialog, Dialogue, Language Model, NLU, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2308.04712v1)  

---


**ABSTRACT**  
Recent advanced methods in Natural Language Understanding for Task-oriented Dialogue (TOD) Systems (e.g., intent detection and slot filling) require a large amount of annotated data to achieve competitive performance. In reality, token-level annotations (slot labels) are time-consuming and difficult to acquire. In this work, we study the Slot Induction (SI) task whose objective is to induce slot boundaries without explicit knowledge of token-level slot annotations. We propose leveraging Unsupervised Pre-trained Language Model (PLM) Probing and Contrastive Learning mechanism to exploit (1) unsupervised semantic knowledge extracted from PLM, and (2) additional sentence-level intent label signals available from TOD. Our approach is shown to be effective in SI task and capable of bridging the gaps with token-level supervised models on two NLU benchmark datasets. When generalized to emerging intents, our SI objectives also provide enhanced slot label representations, leading to improved performance on the Slot Filling tasks.

{{</citation>}}


### (74/109) Answering Unseen Questions With Smaller Language\\Models Using Rationale Generation and Dense Retrieval (Tim Hartill et al., 2023)

{{<citation>}}

Tim Hartill, Diana Benavides-Prado, Michael Witbrock, Patricia J. Riddle. (2023)  
**Answering Unseen Questions With Smaller Language\\Models Using Rationale Generation and Dense Retrieval**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.04711v1)  

---


**ABSTRACT**  
When provided with sufficient explanatory context, smaller Language Models have been shown to exhibit strong reasoning ability on challenging short-answer question-answering tasks where the questions are unseen in training. We evaluate two methods for further improvement in this setting. Both methods focus on combining rationales generated by a larger Language Model with longer contexts created from a multi-hop dense retrieval system. The first method ($\textit{RR}$) involves training a Rationale Ranking model to score both generated rationales and retrieved contexts with respect to relevance and truthfulness. We then use the scores to derive combined contexts from both knowledge sources using a number of combinatory strategies. For the second method ($\textit{RATD}$) we train a smaller Reasoning model using retrieval-augmented training datasets such that it becomes proficient at utilising relevant information from longer text sequences that may be only partially evidential and frequently contain many irrelevant sentences. Generally we find that both methods are effective but that the $\textit{RATD}$ method is more straightforward to apply and produces the strongest results in the unseen setting on which we focus. Our single best Reasoning model using only 440 million parameters materially improves upon strong comparable prior baselines for unseen evaluation datasets (StrategyQA 58.9 $\rightarrow$ 61.7 acc., CommonsenseQA 63.6 $\rightarrow$ 72.7 acc., ARC-DA 31.6 $\rightarrow$ 52.1 F1, IIRC 25.5 $\rightarrow$ 27.3 F1) and a version utilising our prior knowledge of each type of question in selecting a context combination strategy does even better. Our proposed models also generally outperform direct prompts against much larger models (BLOOM 175B and StableVicuna 13B) in both few-shot chain-of-thought and few-shot answer-only settings.

{{</citation>}}


### (75/109) A Comparative Study of Open-Source Large Language Models, GPT-4 and Claude 2: Multiple-Choice Test Taking in Nephrology (Sean Wu et al., 2023)

{{<citation>}}

Sean Wu, Michael Koo, Lesley Blum, Andy Black, Liyo Kao, Fabien Scalzo, Ira Kurtz. (2023)  
**A Comparative Study of Open-Source Large Language Models, GPT-4 and Claude 2: Multiple-Choice Test Taking in Nephrology**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Falcon, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.04709v1)  

---


**ABSTRACT**  
In recent years, there have been significant breakthroughs in the field of natural language processing, particularly with the development of large language models (LLMs). These LLMs have showcased remarkable capabilities on various benchmarks. In the healthcare field, the exact role LLMs and other future AI models will play remains unclear. There is a potential for these models in the future to be used as part of adaptive physician training, medical co-pilot applications, and digital patient interaction scenarios. The ability of AI models to participate in medical training and patient care will depend in part on their mastery of the knowledge content of specific medical fields. This study investigated the medical knowledge capability of LLMs, specifically in the context of internal medicine subspecialty multiple-choice test-taking ability. We compared the performance of several open-source LLMs (Koala 7B, Falcon 7B, Stable-Vicuna 13B, and Orca Mini 13B), to GPT-4 and Claude 2 on multiple-choice questions in the field of Nephrology. Nephrology was chosen as an example of a particularly conceptually complex subspecialty field within internal medicine. The study was conducted to evaluate the ability of LLM models to provide correct answers to nephSAP (Nephrology Self-Assessment Program) multiple-choice questions. The overall success of open-sourced LLMs in answering the 858 nephSAP multiple-choice questions correctly was 17.1% - 25.5%. In contrast, Claude 2 answered 54.4% of the questions correctly, whereas GPT-4 achieved a score of 73.3%. We show that current widely used open-sourced LLMs do poorly in their ability for zero-shot reasoning when compared to GPT-4 and Claude 2. The findings of this study potentially have significant implications for the future of subspecialty medical training and patient care.

{{</citation>}}


### (76/109) Sci-CoT: Leveraging Large Language Models for Enhanced Knowledge Distillation in Small Models for Scientific QA (Yuhan Ma et al., 2023)

{{<citation>}}

Yuhan Ma, Haiqi Jiang, Chenyou Fan. (2023)  
**Sci-CoT: Leveraging Large Language Models for Enhanced Knowledge Distillation in Small Models for Scientific QA**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLOOM, Knowledge Distillation, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2308.04679v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown outstanding performance across wide range of downstream tasks. This competency is attributed to their substantial parameter size and pre-training on extensive corpus. Moreover, LLMs have exhibited enhanced reasoning capabilities in tackling complex reasoning tasks, owing to the utilization of a method named ``Chain-of-Thought (CoT) prompting''. This method is designed to generate intermediate reasoning steps that guide the inference of the final answer. However, it is essential to highlight that these advanced reasoning abilities appear to emerge in models with a minimum of 10 billion parameters, thereby limiting its efficacy in situations where computational resources are constrained. In this paper, we investigate the possibility of transferring the reasoning capabilities of LLMs to smaller models via knowledge distillation. Specifically, we propose Sci-CoT, a two-stage framework that separates the processes of generating rationales and inferring answers. This method enables a more efficient use of rationales during the answer inference stage, leading to improved performance on scientific question-answering tasks. Utilizing Sci-CoT, our 80-million parameter model is able to exceed the performance of BLOOM-176B in the ARC-Easy dataset under the few shot setting.

{{</citation>}}


### (77/109) Cross-Lingual Constituency Parsing for Middle High German: A Delexicalized Approach (Ercong Nie et al., 2023)

{{<citation>}}

Ercong Nie, Helmut Schmid, Hinrich Schütze. (2023)  
**Cross-Lingual Constituency Parsing for Middle High German: A Delexicalized Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.04645v1)  

---


**ABSTRACT**  
Constituency parsing plays a fundamental role in advancing natural language processing (NLP) tasks. However, training an automatic syntactic analysis system for ancient languages solely relying on annotated parse data is a formidable task due to the inherent challenges in building treebanks for such languages. It demands extensive linguistic expertise, leading to a scarcity of available resources. To overcome this hurdle, cross-lingual transfer techniques which require minimal or even no annotated data for low-resource target languages offer a promising solution. In this study, we focus on building a constituency parser for $\mathbf{M}$iddle $\mathbf{H}$igh $\mathbf{G}$erman $\mathbf{MHG}$ under realistic conditions, where no annotated MHG treebank is available for training. In our approach, we leverage the linguistic continuity and structural similarity between MHG and $\mathbf{M}$odern $\mathbf{G}$erman $\mathbf{MG}$, along with the abundance of MG treebank resources. Specifically, by employing the $\mathit{delexicalization}$ method, we train a constituency parser on MG parse datasets and perform cross-lingual transfer to MHG parsing. Our delexicalized constituency parser demonstrates remarkable performance on the MHG test set, achieving an F1-score of 67.3%. It outperforms the best zero-shot cross-lingual baseline by a margin of 28.6% points. These encouraging results underscore the practicality and potential for automatic syntactic analysis in other ancient languages that face similar challenges as MHG.

{{</citation>}}


## cs.SD (3)



### (78/109) Conformer-based Target-Speaker Automatic Speech Recognition for Single-Channel Audio (Yang Zhang et al., 2023)

{{<citation>}}

Yang Zhang, Krishna C. Puvvada, Vitaly Lavrukhin, Boris Ginsburg. (2023)  
**Conformer-based Target-Speaker Automatic Speech Recognition for Single-Channel Audio**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2308.05218v1)  

---


**ABSTRACT**  
We propose CONF-TSASR, a non-autoregressive end-to-end time-frequency domain architecture for single-channel target-speaker automatic speech recognition (TS-ASR). The model consists of a TitaNet based speaker embedding module, a Conformer based masking as well as ASR modules. These modules are jointly optimized to transcribe a target-speaker, while ignoring speech from other speakers. For training we use Connectionist Temporal Classification (CTC) loss and introduce a scale-invariant spectrogram reconstruction loss to encourage the model better separate the target-speaker's spectrogram from mixture. We obtain state-of-the-art target-speaker word error rate (TS-WER) on WSJ0-2mix-extr (4.2%). Further, we report for the first time TS-WER on WSJ0-3mix-extr (12.4%), LibriSpeech2Mix (4.2%) and LibriSpeech3Mix (7.6%) datasets, establishing new benchmarks for TS-ASR. The proposed model will be open-sourced through NVIDIA NeMo toolkit.

{{</citation>}}


### (79/109) Representation Learning for Audio Privacy Preservation using Source Separation and Robust Adversarial Learning (Diep Luong et al., 2023)

{{<citation>}}

Diep Luong, Minh Tran, Shayan Gharib, Konstantinos Drossos, Tuomas Virtanen. (2023)  
**Representation Learning for Audio Privacy Preservation using Source Separation and Robust Adversarial Learning**  

---
Primary Category: cs.SD  
Categories: cs-CR, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.04960v1)  

---


**ABSTRACT**  
Privacy preservation has long been a concern in smart acoustic monitoring systems, where speech can be passively recorded along with a target signal in the system's operating environment. In this study, we propose the integration of two commonly used approaches in privacy preservation: source separation and adversarial representation learning. The proposed system learns the latent representation of audio recordings such that it prevents differentiating between speech and non-speech recordings. Initially, the source separation network filters out some of the privacy-sensitive data, and during the adversarial learning process, the system will learn privacy-preserving representation on the filtered signal. We demonstrate the effectiveness of our proposed method by comparing our method against systems without source separation, without adversarial learning, and without both. Overall, our results suggest that the proposed system can significantly improve speech privacy preservation compared to that of using source separation or adversarial learning solely while maintaining good performance in the acoustic monitoring task.

{{</citation>}}


### (80/109) Speaker Recognition Using Isomorphic Graph Attention Network Based Pooling on Self-Supervised Representation (Zirui Ge et al., 2023)

{{<citation>}}

Zirui Ge, Xinzhou Xu, Haiyan Guo, Tingting Wang, Zhen Yang. (2023)  
**Speaker Recognition Using Isomorphic Graph Attention Network Based Pooling on Self-Supervised Representation**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Attention, Graph Attention Network, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.04666v1)  

---


**ABSTRACT**  
The emergence of self-supervised representation (i.e., wav2vec 2.0) allows speaker-recognition approaches to process spoken signals through foundation models built on speech data. Nevertheless, effective fusion on the representation requires further investigating, due to the inclusion of fixed or sub-optimal temporal pooling strategies. Despite of improved strategies considering graph learning and graph attention factors, non-injective aggregation still exists in the approaches, which may influence the performance for speaker recognition. In this regard, we propose a speaker recognition approach using Isomorphic Graph ATtention network (IsoGAT) on self-supervised representation. The proposed approach contains three modules of representation learning, graph attention, and aggregation, jointly considering learning on the self-supervised representation and the IsoGAT. Then, we perform experiments for speaker recognition tasks on VoxCeleb1\&2 datasets, with the corresponding experimental results demonstrating the recognition performance for the proposed approach, compared with existing pooling approaches on the self-supervised representation.

{{</citation>}}


## cs.AR (1)



### (81/109) FPGA Resource-aware Structured Pruning for Real-Time Neural Networks (Benjamin Ramhorst et al., 2023)

{{<citation>}}

Benjamin Ramhorst, George A. Constantinides, Vladimir Loncar. (2023)  
**FPGA Resource-aware Structured Pruning for Real-Time Neural Networks**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs.AR  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.05170v1)  

---


**ABSTRACT**  
Neural networks achieve state-of-the-art performance in image classification, speech recognition, scientific analysis and many more application areas. With the ever-increasing need for faster computation and lower power consumption, driven by real-time systems and Internet-of-Things (IoT) devices, FPGAs have emerged as suitable devices for deep learning inference. Due to the high computational complexity and memory footprint of neural networks, various compression techniques, such as pruning, quantization and knowledge distillation, have been proposed in literature. Pruning sparsifies a neural network, reducing the number of multiplications and memory. However, pruning often fails to capture properties of the underlying hardware, causing unstructured sparsity and load-balance inefficiency, thus bottlenecking resource improvements. We propose a hardware-centric formulation of pruning, by formulating it as a knapsack problem with resource-aware tensor structures. The primary emphasis is on real-time inference, with latencies in the order of 1$\mu$s, accelerated with hls4ml, an open-source framework for deep learning inference on FPGAs. Evaluated on a range of tasks, including real-time particle classification at CERN's Large Hadron Collider and fast image classification, the proposed method achieves a reduction ranging between 55% and 92% in the utilization of digital signal processing blocks (DSP) and up to 81% in block memory (BRAM) utilization.

{{</citation>}}


## eess.IV (5)



### (82/109) Improved Multi-Shot Diffusion-Weighted MRI with Zero-Shot Self-Supervised Learning Reconstruction (Jaejin Cho et al., 2023)

{{<citation>}}

Jaejin Cho, Yohan Jun, Xiaoqing Wang, Caique Kobayashi, Berkin Bilgic. (2023)  
**Improved Multi-Shot Diffusion-Weighted MRI with Zero-Shot Self-Supervised Learning Reconstruction**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess.IV  
Keywords: Self-Supervised, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.05103v1)  

---


**ABSTRACT**  
Diffusion MRI is commonly performed using echo-planar imaging (EPI) due to its rapid acquisition time. However, the resolution of diffusion-weighted images is often limited by magnetic field inhomogeneity-related artifacts and blurring induced by T2- and T2*-relaxation effects. To address these limitations, multi-shot EPI (msEPI) combined with parallel imaging techniques is frequently employed. Nevertheless, reconstructing msEPI can be challenging due to phase variation between multiple shots. In this study, we introduce a novel msEPI reconstruction approach called zero-MIRID (zero-shot self-supervised learning of Multi-shot Image Reconstruction for Improved Diffusion MRI). This method jointly reconstructs msEPI data by incorporating deep learning-based image regularization techniques. The network incorporates CNN denoisers in both k- and image-spaces, while leveraging virtual coils to enhance image reconstruction conditioning. By employing a self-supervised learning technique and dividing sampled data into three groups, the proposed approach achieves superior results compared to the state-of-the-art parallel imaging method, as demonstrated in an in-vivo experiment.

{{</citation>}}


### (83/109) Geometric Learning-Based Transformer Network for Estimation of Segmentation Errors (Sneha Sree C et al., 2023)

{{<citation>}}

Sneha Sree C, Mohammad Al Fahim, Keerthi Ram, Mohanasankar Sivaprakasam. (2023)  
**Geometric Learning-Based Transformer Network for Estimation of Segmentation Errors**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2308.05068v2)  

---


**ABSTRACT**  
Many segmentation networks have been proposed for 3D volumetric segmentation of tumors and organs at risk. Hospitals and clinical institutions seek to accelerate and minimize the efforts of specialists in image segmentation. Still, in case of errors generated by these networks, clinicians would have to manually edit the generated segmentation maps. Given a 3D volume and its putative segmentation map, we propose an approach to identify and measure erroneous regions in the segmentation map. Our method can estimate error at any point or node in a 3D mesh generated from a possibly erroneous volumetric segmentation map, serving as a Quality Assurance tool. We propose a graph neural network-based transformer based on the Nodeformer architecture to measure and classify the segmentation errors at any point. We have evaluated our network on a high-resolution micro-CT dataset of the human inner-ear bony labyrinth structure by simulating erroneous 3D segmentation maps. Our network incorporates a convolutional encoder to compute node-centric features from the input micro-CT data, the Nodeformer to learn the latent graph embeddings, and a Multi-Layer Perceptron (MLP) to compute and classify the node-wise errors. Our network achieves a mean absolute error of ~0.042 over other Graph Neural Networks (GNN) and an accuracy of 79.53% over other GNNs in estimating and classifying the node-wise errors, respectively. We also put forth vertex-normal prediction as a custom pretext task for pre-training the CNN encoder to improve the network's overall performance. Qualitative analysis shows the efficiency of our network in correctly classifying errors and reducing misclassifications.

{{</citation>}}


### (84/109) Deep Generative Networks for Heterogeneous Augmentation of Cranial Defects (Kamil Kwarciak et al., 2023)

{{<citation>}}

Kamil Kwarciak, Marek Wodzinski. (2023)  
**Deep Generative Networks for Heterogeneous Augmentation of Cranial Defects**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.04883v1)  

---


**ABSTRACT**  
The design of personalized cranial implants is a challenging and tremendous task that has become a hot topic in terms of process automation with the use of deep learning techniques. The main challenge is associated with the high diversity of possible cranial defects. The lack of appropriate data sources negatively influences the data-driven nature of deep learning algorithms. Hence, one of the possible solutions to overcome this problem is to rely on synthetic data. In this work, we propose three volumetric variations of deep generative models to augment the dataset by generating synthetic skulls, i.e. Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP), WGAN-GP hybrid with Variational Autoencoder pretraining (VAE/WGAN-GP) and Introspective Variational Autoencoder (IntroVAE). We show that it is possible to generate dozens of thousands of defective skulls with compatible defects that achieve a trade-off between defect heterogeneity and the realistic shape of the skull. We evaluate obtained synthetic data quantitatively by defect segmentation with the use of V-Net and qualitatively by their latent space exploration. We show that the synthetically generated skulls highly improve the segmentation process compared to using only the original unaugmented data. The generated skulls may improve the automatic design of personalized cranial implants for real medical cases.

{{</citation>}}


### (85/109) Are Sex-based Physiological Differences the Cause of Gender Bias for Chest X-ray Diagnosis? (Nina Weng et al., 2023)

{{<citation>}}

Nina Weng, Siavash Bigdeli, Eike Petersen, Aasa Feragen. (2023)  
**Are Sex-based Physiological Differences the Cause of Gender Bias for Chest X-ray Diagnosis?**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-CY, cs-LG, eess-IV, eess.IV  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2308.05129v1)  

---


**ABSTRACT**  
While many studies have assessed the fairness of AI algorithms in the medical field, the causes of differences in prediction performance are often unknown. This lack of knowledge about the causes of bias hampers the efficacy of bias mitigation, as evidenced by the fact that simple dataset balancing still often performs best in reducing performance gaps but is unable to resolve all performance differences. In this work, we investigate the causes of gender bias in machine learning-based chest X-ray diagnosis. In particular, we explore the hypothesis that breast tissue leads to underexposure of the lungs and causes lower model performance. Methodologically, we propose a new sampling method which addresses the highly skewed distribution of recordings per patient in two widely used public datasets, while at the same time reducing the impact of label errors. Our comprehensive analysis of gender differences across diseases, datasets, and gender representations in the training set shows that dataset imbalance is not the sole cause of performance differences. Moreover, relative group performance differs strongly between datasets, indicating important dataset-specific factors influencing male/female group performance. Finally, we investigate the effect of breast tissue more specifically, by cropping out the breasts from recordings, finding that this does not resolve the observed performance gaps. In conclusion, our results indicate that dataset-specific factors, not fundamental physiological differences, are the main drivers of male--female performance gaps in chest X-ray analyses on widely used NIH and CheXpert Dataset.

{{</citation>}}


### (86/109) Assessing the performance of deep learning-based models for prostate cancer segmentation using uncertainty scores (Pablo Cesar Quihui-Rubio et al., 2023)

{{<citation>}}

Pablo Cesar Quihui-Rubio, Daniel Flores-Araiza, Gilberto Ochoa-Ruiz, Miguel Gonzalez-Mendoza, Christian Mata. (2023)  
**Assessing the performance of deep learning-based models for prostate cancer segmentation using uncertainty scores**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.04653v1)  

---


**ABSTRACT**  
This study focuses on comparing deep learning methods for the segmentation and quantification of uncertainty in prostate segmentation from MRI images. The aim is to improve the workflow of prostate cancer detection and diagnosis. Seven different U-Net-based architectures, augmented with Monte-Carlo dropout, are evaluated for automatic segmentation of the central zone, peripheral zone, transition zone, and tumor, with uncertainty estimation. The top-performing model in this study is the Attention R2U-Net, achieving a mean Intersection over Union (IoU) of 76.3% and Dice Similarity Coefficient (DSC) of 85% for segmenting all zones. Additionally, Attention R2U-Net exhibits the lowest uncertainty values, particularly in the boundaries of the transition zone and tumor, when compared to the other models.

{{</citation>}}


## math.PR (1)



### (87/109) Mean-Biased Processes for Balanced Allocations (Dimitrios Los et al., 2023)

{{<citation>}}

Dimitrios Los, Thomas Sauerwald, John Sylvester. (2023)  
**Mean-Biased Processes for Balanced Allocations**  

---
Primary Category: math.PR  
Categories: 68W20, 68W27, 68W40, 60C05, G-3; G-2-m; F-2-2, cs-DM, cs-DS, math-CO, math-PR, math.PR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.05087v1)  

---


**ABSTRACT**  
We introduce a new class of balanced allocation processes which bias towards underloaded bins (those with load below the mean load) either by skewing the probability by which a bin is chosen for an allocation (probability bias), or alternatively, by adding more balls to an underloaded bin (weight bias). A prototypical process satisfying the probability bias condition is Mean-Thinning: At each round, we sample one bin and if it is underloaded, we allocate one ball; otherwise, we allocate one ball to a second bin sample. Versions of this process have been in use since at least 1986. An example of a process, introduced by us, which satisfies the weight bias condition is Twinning: At each round, we only sample one bin. If the bin is underloaded, then we allocate two balls; otherwise, we allocate only one ball.   Our main result is that for any process with a probability or weight bias, with high probability the gap between maximum and minimum load is logarithmic in the number of bins. This result holds for any number of allocated balls (heavily loaded case), covers many natural processes that relax the Two-Choice process, and we also prove it is tight for many such processes, including Mean-Thinning and Twinning.   Our analysis employs a delicate interplay between linear, quadratic and exponential potential functions. It also hinges on a phenomenon we call ``mean quantile stabilization'', which holds in greater generality than our framework and may be of independent interest.

{{</citation>}}


## cs.CY (2)



### (88/109) Drones4Good: Supporting Disaster Relief Through Remote Sensing and AI (Nina Merkle et al., 2023)

{{<citation>}}

Nina Merkle, Reza Bahmanyar, Corentin Henry, Seyed Majid Azimi, Xiangtian Yuan, Simon Schopferer, Veronika Gstaiger, Stefan Auer, Anne Schneibel, Marc Wieland, Thomas Kraft. (2023)  
**Drones4Good: Supporting Disaster Relief Through Remote Sensing and AI**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CV, cs-CY, cs.CY  
Keywords: AI, Drone  
[Paper Link](http://arxiv.org/abs/2308.05074v1)  

---


**ABSTRACT**  
In order to respond effectively in the aftermath of a disaster, emergency services and relief organizations rely on timely and accurate information about the affected areas. Remote sensing has the potential to significantly reduce the time and effort required to collect such information by enabling a rapid survey of large areas. To achieve this, the main challenge is the automatic extraction of relevant information from remotely sensed data. In this work, we show how the combination of drone-based data with deep learning methods enables automated and large-scale situation assessment. In addition, we demonstrate the integration of onboard image processing techniques for the deployment of autonomous drone-based aid delivery. The results show the feasibility of a rapid and large-scale image analysis in the field, and that onboard image processing can increase the safety of drone-based aid deliveries.

{{</citation>}}


### (89/109) Where's the Liability in Harmful AI Speech? (Peter Henderson et al., 2023)

{{<citation>}}

Peter Henderson, Tatsunori Hashimoto, Mark Lemley. (2023)  
**Where's the Liability in Harmful AI Speech?**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.04635v1)  

---


**ABSTRACT**  
Generative AI, in particular text-based "foundation models" (large models trained on a huge variety of information including the internet), can generate speech that could be problematic under a wide range of liability regimes. Machine learning practitioners regularly "red team" models to identify and mitigate such problematic speech: from "hallucinations" falsely accusing people of serious misconduct to recipes for constructing an atomic bomb. A key question is whether these red-teamed behaviors actually present any liability risk for model creators and deployers under U.S. law, incentivizing investments in safety mechanisms. We examine three liability regimes, tying them to common examples of red-teamed model behaviors: defamation, speech integral to criminal conduct, and wrongful death. We find that any Section 230 immunity analysis or downstream liability analysis is intimately wrapped up in the technical details of algorithm design. And there are many roadblocks to truly finding models (and their associated parties) liable for generated speech. We argue that AI should not be categorically immune from liability in these scenarios and that as courts grapple with the already fine-grained complexities of platform algorithms, the technical details of generative AI loom above with thornier questions. Courts and policymakers should think carefully about what technical design incentives they create as they evaluate these issues.

{{</citation>}}


## eess.SP (2)



### (90/109) Collaborative Wideband Spectrum Sensing and Scheduling for Networked UAVs in UTM Systems (Sravan Reddy Chintareddy et al., 2023)

{{<citation>}}

Sravan Reddy Chintareddy, Keenan Roach, Kenny Cheung, Morteza Hashemi. (2023)  
**Collaborative Wideband Spectrum Sensing and Scheduling for Networked UAVs in UTM Systems**  

---
Primary Category: eess.SP  
Categories: cs-LG, cs-MA, cs-NI, eess-SP, eess.SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.05036v1)  

---


**ABSTRACT**  
In this paper, we propose a data-driven framework for collaborative wideband spectrum sensing and scheduling for networked unmanned aerial vehicles (UAVs), which act as the secondary users to opportunistically utilize detected spectrum holes. To this end, we propose a multi-class classification problem for wideband spectrum sensing to detect vacant spectrum spots based on collected I/Q samples. To enhance the accuracy of the spectrum sensing module, the outputs from the multi-class classification by each individual UAV are fused at a server in the unmanned aircraft system traffic management (UTM) ecosystem. In the spectrum scheduling phase, we leverage reinforcement learning (RL) solutions to dynamically allocate the detected spectrum holes to the secondary users (i.e., UAVs). To evaluate the proposed methods, we establish a comprehensive simulation framework that generates a near-realistic synthetic dataset using MATLAB LTE toolbox by incorporating base-station~(BS) locations in a chosen area of interest, performing ray-tracing, and emulating the primary users channel usage in terms of I/Q samples. This evaluation methodology provides a flexible framework to generate large spectrum datasets that could be used for developing ML/AI-based spectrum management solutions for aerial devices.

{{</citation>}}


### (91/109) Real-time FPGA Implementation of CNN-based Distributed Fiber Optic Vibration Event Recognition Method (Zhongyao Luo et al., 2023)

{{<citation>}}

Zhongyao Luo, Zhao Ge, Hao Wu, Ming Tang. (2023)  
**Real-time FPGA Implementation of CNN-based Distributed Fiber Optic Vibration Event Recognition Method**  

---
Primary Category: eess.SP  
Categories: cs-SY, eess-SP, eess-SY, eess.SP  
Keywords: Event Recognition  
[Paper Link](http://arxiv.org/abs/2308.04683v1)  

---


**ABSTRACT**  
Utilizing optical fibers to detect and pinpoint vibrations, Distributed Optical Fiber Vibration Sensing (DVS) technology provides real-time monitoring and surveillance of wide-reaching areas. This field has been leveraging Convolutional Neural Networks (CNN). Recently, a study has accomplished end-to-end vibration event recognition, enabling utilization of CNN-based DVS algorithms as real-time embedded system for edge computing in practical application situations. Considering the power consumption of central processing unit (CPU) and graphics processing unit (GPU), and the inflexibility of application-specific integrated circuit (ASIC), field-Programmable gate array (FPGA) is the optimal computing platform for the system. This paper proposes to compress pre-trained network and adopt a novel hardware structure, to design a fully on-chip, pipelined inference accelerator for CNN-based DVS algorithm, without fine tuning or re-training. This design allows for real-time processing with low power consumption and system requirement.An examination has been executed on an existing DVS algorithm based on a 40-layer CNN model comprising 2.7 million parameters. It is completely implemented on-chip, pipelined, with no reduction in accuracy.

{{</citation>}}


## cs.LG (10)



### (92/109) Multi-Class Deep SVDD: Anomaly Detection Approach in Astronomy with Distinct Inlier Categories (Manuel Pérez-Carrasco et al., 2023)

{{<citation>}}

Manuel Pérez-Carrasco, Guillermo Cabrera-Vives, Lorena Hernández-García, Francisco Forster, Paula Sánchez-Sáez, Alejandra Muñoz Arancibia, Nicolás Astorga, Franz Bauer, Amelia Bayo, Martina Cádiz-Leyton, Marcio Catelan. (2023)  
**Multi-Class Deep SVDD: Anomaly Detection Approach in Astronomy with Distinct Inlier Categories**  

---
Primary Category: cs.LG  
Categories: astro-ph-IM, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.05011v2)  

---


**ABSTRACT**  
With the increasing volume of astronomical data generated by modern survey telescopes, automated pipelines and machine learning techniques have become crucial for analyzing and extracting knowledge from these datasets. Anomaly detection, i.e. the task of identifying irregular or unexpected patterns in the data, is a complex challenge in astronomy. In this paper, we propose Multi-Class Deep Support Vector Data Description (MCDSVDD), an extension of the state-of-the-art anomaly detection algorithm One-Class Deep SVDD, specifically designed to handle different inlier categories with distinct data distributions. MCDSVDD uses a neural network to map the data into hyperspheres, where each hypersphere represents a specific inlier category. The distance of each sample from the centers of these hyperspheres determines the anomaly score. We evaluate the effectiveness of MCDSVDD by comparing its performance with several anomaly detection algorithms on a large dataset of astronomical light-curves obtained from the Zwicky Transient Facility. Our results demonstrate the efficacy of MCDSVDD in detecting anomalous sources while leveraging the presence of different inlier categories. The code and the data needed to reproduce our results are publicly available at https://github.com/mperezcarrasco/AnomalyALeRCE.

{{</citation>}}


### (93/109) Adversarial ModSecurity: Countering Adversarial SQL Injections with Robust Machine Learning (Biagio Montaruli et al., 2023)

{{<citation>}}

Biagio Montaruli, Luca Demetrio, Andrea Valenza, Battista Biggio, Luca Compagna, Davide Balzarotti, Davide Ariu, Luca Piras. (2023)  
**Adversarial ModSecurity: Countering Adversarial SQL Injections with Robust Machine Learning**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.04964v1)  

---


**ABSTRACT**  
ModSecurity is widely recognized as the standard open-source Web Application Firewall (WAF), maintained by the OWASP Foundation. It detects malicious requests by matching them against the Core Rule Set, identifying well-known attack patterns. Each rule in the CRS is manually assigned a weight, based on the severity of the corresponding attack, and a request is detected as malicious if the sum of the weights of the firing rules exceeds a given threshold. In this work, we show that this simple strategy is largely ineffective for detecting SQL injection (SQLi) attacks, as it tends to block many legitimate requests, while also being vulnerable to adversarial SQLi attacks, i.e., attacks intentionally manipulated to evade detection. To overcome these issues, we design a robust machine learning model, named AdvModSec, which uses the CRS rules as input features, and it is trained to detect adversarial SQLi attacks. Our experiments show that AdvModSec, being trained on the traffic directed towards the protected web services, achieves a better trade-off between detection and false positive rates, improving the detection rate of the vanilla version of ModSecurity with CRS by 21%. Moreover, our approach is able to improve its adversarial robustness against adversarial SQLi attacks by 42%, thereby taking a step forward towards building more robust and trustworthy WAFs.

{{</citation>}}


### (94/109) Differentially Private Graph Neural Network with Importance-Grained Noise Adaption (Yuxin Qi et al., 2023)

{{<citation>}}

Yuxin Qi, Xi Lin, Jun Wu. (2023)  
**Differentially Private Graph Neural Network with Importance-Grained Noise Adaption**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.04943v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) with differential privacy have been proposed to preserve graph privacy when nodes represent personal and sensitive information. However, the existing methods ignore that nodes with different importance may yield diverse privacy demands, which may lead to over-protect some nodes and decrease model utility. In this paper, we study the problem of importance-grained privacy, where nodes contain personal data that need to be kept private but are critical for training a GNN. We propose NAP-GNN, a node-importance-grained privacy-preserving GNN algorithm with privacy guarantees based on adaptive differential privacy to safeguard node information. First, we propose a Topology-based Node Importance Estimation (TNIE) method to infer unknown node importance with neighborhood and centrality awareness. Second, an adaptive private aggregation method is proposed to perturb neighborhood aggregation from node-importance-grain. Third, we propose to privately train a graph learning algorithm on perturbed aggregations in adaptive residual connection mode over multi-layers convolution for node-wise tasks. Theoretically analysis shows that NAP-GNN satisfies privacy guarantees. Empirical experiments over real-world graph datasets show that NAP-GNN achieves a better trade-off between privacy and accuracy.

{{</citation>}}


### (95/109) An In-Depth Analysis of Discretization Methods for Communication Learning using Backpropagation with Multi-Agent Reinforcement Learning (Astrid Vanneste et al., 2023)

{{<citation>}}

Astrid Vanneste, Simon Vanneste, Kevin Mets, Tom De Schepper, Siegfried Mercelis, Peter Hellinckx. (2023)  
**An In-Depth Analysis of Discretization Methods for Communication Learning using Backpropagation with Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.04938v1)  

---


**ABSTRACT**  
Communication is crucial in multi-agent reinforcement learning when agents are not able to observe the full state of the environment. The most common approach to allow learned communication between agents is the use of a differentiable communication channel that allows gradients to flow between agents as a form of feedback. However, this is challenging when we want to use discrete messages to reduce the message size, since gradients cannot flow through a discrete communication channel. Previous work proposed methods to deal with this problem. However, these methods are tested in different communication learning architectures and environments, making it hard to compare them. In this paper, we compare several state-of-the-art discretization methods as well as a novel approach. We do this comparison in the context of communication learning using gradients from other agents and perform tests on several environments. In addition, we present COMA-DIAL, a communication learning approach based on DIAL and COMA extended with learning rate scaling and adapted exploration. Using COMA-DIAL allows us to perform experiments on more complex environments. Our results show that the novel ST-DRU method, proposed in this paper, achieves the best results out of all discretization methods across the different environments. It achieves the best or close to the best performance in each of the experiments and is the only method that does not fail on any of the tested environments.

{{</citation>}}


### (96/109) Scalability of Message Encoding Techniques for Continuous Communication Learned with Multi-Agent Reinforcement Learning (Astrid Vanneste et al., 2023)

{{<citation>}}

Astrid Vanneste, Thomas Somers, Simon Vanneste, Kevin Mets, Tom De Schepper, Siegfried Mercelis, Peter Hellinckx. (2023)  
**Scalability of Message Encoding Techniques for Continuous Communication Learned with Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.04844v1)  

---


**ABSTRACT**  
Many multi-agent systems require inter-agent communication to properly achieve their goal. By learning the communication protocol alongside the action protocol using multi-agent reinforcement learning techniques, the agents gain the flexibility to determine which information should be shared. However, when the number of agents increases we need to create an encoding of the information contained in these messages. In this paper, we investigate the effect of increasing the amount of information that should be contained in a message and increasing the number of agents. We evaluate these effects on two different message encoding methods, the mean message encoder and the attention message encoder. We perform our experiments on a matrix environment. Surprisingly, our results show that the mean message encoder consistently outperforms the attention message encoder. Therefore, we analyse the communication protocol used by the agents that use the mean message encoder and can conclude that the agents use a combination of an exponential and a logarithmic function in their communication policy to avoid the loss of important information after applying the mean message encoder.

{{</citation>}}


### (97/109) PETformer: Long-term Time Series Forecasting via Placeholder-enhanced Transformer (Shengsheng Lin et al., 2023)

{{<citation>}}

Shengsheng Lin, Weiwei Lin, Wentai Wu, Songbo Wang, Yongxiang Wang. (2023)  
**PETformer: Long-term Time Series Forecasting via Placeholder-enhanced Transformer**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.04791v1)  

---


**ABSTRACT**  
Recently, Transformer-based models have shown remarkable performance in long-term time series forecasting (LTSF) tasks due to their ability to model long-term dependencies. However, the validity of Transformers for LTSF tasks remains debatable, particularly since recent work has shown that simple linear models can outperform numerous Transformer-based approaches. This suggests that there are limitations to the application of Transformer in LTSF. Therefore, this paper investigates three key issues when applying Transformer to LTSF: temporal continuity, information density, and multi-channel relationships. Accordingly, we propose three innovative solutions, including Placeholder Enhancement Technique (PET), Long Sub-sequence Division (LSD), and Multi-channel Separation and Interaction (MSI), which together form a novel model called PETformer. These three key designs introduce prior biases suitable for LTSF tasks. Extensive experiments have demonstrated that PETformer achieves state-of-the-art (SOTA) performance on eight commonly used public datasets for LTSF, outperforming all other models currently available. This demonstrates that Transformer still possesses powerful capabilities in LTSF.

{{</citation>}}


### (98/109) Generative Perturbation Analysis for Probabilistic Black-Box Anomaly Attribution (Tsuyoshi Idé et al., 2023)

{{<citation>}}

Tsuyoshi Idé, Naoki Abe. (2023)  
**Generative Perturbation Analysis for Probabilistic Black-Box Anomaly Attribution**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04708v1)  

---


**ABSTRACT**  
We address the task of probabilistic anomaly attribution in the black-box regression setting, where the goal is to compute the probability distribution of the attribution score of each input variable, given an observed anomaly. The training dataset is assumed to be unavailable. This task differs from the standard XAI (explainable AI) scenario, since we wish to explain the anomalous deviation from a black-box prediction rather than the black-box model itself.   We begin by showing that mainstream model-agnostic explanation methods, such as the Shapley values, are not suitable for this task because of their ``deviation-agnostic property.'' We then propose a novel framework for probabilistic anomaly attribution that allows us to not only compute attribution scores as the predictive mean but also quantify the uncertainty of those scores. This is done by considering a generative process for perturbations that counter-factually bring the observed anomalous observation back to normalcy. We introduce a variational Bayes algorithm for deriving the distributions of per variable attribution scores. To the best of our knowledge, this is the first probabilistic anomaly attribution framework that is free from being deviation-agnostic.

{{</citation>}}


### (99/109) An Analytical Study of Covid-19 Dataset using Graph-Based Clustering Algorithms (Mamata Das et al., 2023)

{{<citation>}}

Mamata Das, P. J. A. Alphonse, Selvakumar K. (2023)  
**An Analytical Study of Covid-19 Dataset using Graph-Based Clustering Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04697v1)  

---


**ABSTRACT**  
Corona VIrus Disease abbreviated as COVID-19 is a novel virus which is initially identified in Wuhan of China in December of 2019 and now this deadly disease has spread all over the world. According to World Health Organization (WHO), a total of 3,124,905 people died from 2019 to 2021, April. In this case, many methods, AI base techniques, and machine learning algorithms have been researched and are being used to save people from this pandemic. The SARS-CoV and the 2019-nCoV, SARS-CoV-2 virus invade our bodies, causing some differences in the structure of cell proteins. Protein-protein interaction (PPI) is an essential process in our cells and plays a very important role in the development of medicines and gives ideas about the disease. In this study, we performed clustering on PPI networks generated from 92 genes of the Covi-19 dataset. We have used three graph-based clustering algorithms to give intuition to the analysis of clusters.

{{</citation>}}


### (100/109) Efficient Bayesian Optimization with Deep Kernel Learning and Transformer Pre-trained on Multiple Heterogeneous Datasets (Wenlong Lyu et al., 2023)

{{<citation>}}

Wenlong Lyu, Shoubo Hu, Jie Chuai, Zhitang Chen. (2023)  
**Efficient Bayesian Optimization with Deep Kernel Learning and Transformer Pre-trained on Multiple Heterogeneous Datasets**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.04660v1)  

---


**ABSTRACT**  
Bayesian optimization (BO) is widely adopted in black-box optimization problems and it relies on a surrogate model to approximate the black-box response function. With the increasing number of black-box optimization tasks solved and even more to solve, the ability to learn from multiple prior tasks to jointly pre-train a surrogate model is long-awaited to further boost optimization efficiency. In this paper, we propose a simple approach to pre-train a surrogate, which is a Gaussian process (GP) with a kernel defined on deep features learned from a Transformer-based encoder, using datasets from prior tasks with possibly heterogeneous input spaces. In addition, we provide a simple yet effective mix-up initialization strategy for input tokens corresponding to unseen input variables and therefore accelerate new tasks' convergence. Experiments on both synthetic and real benchmark problems demonstrate the effectiveness of our proposed pre-training and transfer BO strategy over existing methods.

{{</citation>}}


### (101/109) Sparse Binary Transformers for Multivariate Time Series Modeling (Matt Gorbett et al., 2023)

{{<citation>}}

Matt Gorbett, Hossein Shirazi, Indrakshi Ray. (2023)  
**Sparse Binary Transformers for Multivariate Time Series Modeling**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.04637v1)  

---


**ABSTRACT**  
Compressed Neural Networks have the potential to enable deep learning across new applications and smaller computational environments. However, understanding the range of learning tasks in which such models can succeed is not well studied. In this work, we apply sparse and binary-weighted Transformers to multivariate time series problems, showing that the lightweight models achieve accuracy comparable to that of dense floating-point Transformers of the same structure. Our model achieves favorable results across three time series learning tasks: classification, anomaly detection, and single-step forecasting. Additionally, to reduce the computational complexity of the attention mechanism, we apply two modifications, which show little to no decline in model performance: 1) in the classification task, we apply a fixed mask to the query, key, and value activations, and 2) for forecasting and anomaly detection, which rely on predicting outputs at a single point in time, we propose an attention mask to allow computation only at the current time step. Together, each compression technique and attention modification substantially reduces the number of non-zero operations necessary in the Transformer. We measure the computational savings of our approach over a range of metrics including parameter count, bit size, and floating point operation (FLOPs) count, showing up to a 53x reduction in storage size and up to 10.5x reduction in FLOPs.

{{</citation>}}


## cs.NI (3)



### (102/109) Semantic Communications for Artificial Intelligence Generated Content (AIGC) Toward Effective Content Creation (Guangyuan Liu et al., 2023)

{{<citation>}}

Guangyuan Liu, Hongyang Du, Dusit Niyato, Jiawen Kang, Zehui Xiong, Dong In Kim, Xuemin, Shen. (2023)  
**Semantic Communications for Artificial Intelligence Generated Content (AIGC) Toward Effective Content Creation**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04942v1)  

---


**ABSTRACT**  
Artificial Intelligence Generated Content (AIGC) Services have significant potential in digital content creation. The distinctive abilities of AIGC, such as content generation based on minimal input, hold huge potential, especially when integrating with semantic communication (SemCom). In this paper, a novel comprehensive conceptual model for the integration of AIGC and SemCom is developed. Particularly, a content generation level is introduced on top of the semantic level that provides a clear outline of how AIGC and SemCom interact with each other to produce meaningful and effective content. Moreover, a novel framework that employs AIGC technology is proposed as an encoder and decoder for semantic information, considering the joint optimization of semantic extraction and evaluation metrics tailored to AIGC services. The framework can adapt to different types of content generated, the required quality, and the semantic information utilized. By employing a Deep Q Network (DQN), a case study is presented that provides useful insights into the feasibility of the optimization problem and its convergence characteristics.

{{</citation>}}


### (103/109) GraphCC: A Practical Graph Learning-based Approach to Congestion Control in Datacenters (Guillermo Bernárdez et al., 2023)

{{<citation>}}

Guillermo Bernárdez, José Suárez-Varela, Xiang Shi, Shihan Xiao, Xiangle Cheng, Pere Barlet-Ros, Albert Cabellos-Aparicio. (2023)  
**GraphCC: A Practical Graph Learning-based Approach to Congestion Control in Datacenters**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-LG, cs-MA, cs-NI, cs.NI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.04905v1)  

---


**ABSTRACT**  
Congestion Control (CC) plays a fundamental role in optimizing traffic in Data Center Networks (DCN). Currently, DCNs mainly implement two main CC protocols: DCTCP and DCQCN. Both protocols -- and their main variants -- are based on Explicit Congestion Notification (ECN), where intermediate switches mark packets when they detect congestion. The ECN configuration is thus a crucial aspect on the performance of CC protocols. Nowadays, network experts set static ECN parameters carefully selected to optimize the average network performance. However, today's high-speed DCNs experience quick and abrupt changes that severely change the network state (e.g., dynamic traffic workloads, incast events, failures). This leads to under-utilization and sub-optimal performance. This paper presents GraphCC, a novel Machine Learning-based framework for in-network CC optimization. Our distributed solution relies on a novel combination of Multi-agent Reinforcement Learning (MARL) and Graph Neural Networks (GNN), and it is compatible with widely deployed ECN-based CC protocols. GraphCC deploys distributed agents on switches that communicate with their neighbors to cooperate and optimize the global ECN configuration. In our evaluation, we test the performance of GraphCC under a wide variety of scenarios, focusing on the capability of this solution to adapt to new scenarios unseen during training (e.g., new traffic workloads, failures, upgrades). We compare GraphCC with a state-of-the-art MARL-based solution for ECN tuning -- ACC -- and observe that our proposed solution outperforms the state-of-the-art baseline in all of the evaluation scenarios, showing improvements up to $20\%$ in Flow Completion Time as well as significant reductions in buffer occupancy ($38.0-85.7\%$).

{{</citation>}}


### (104/109) IS2N: Intent-Driven Security Software-Defined Network with Blockchain (Yanbo Song et al., 2023)

{{<citation>}}

Yanbo Song, Tao Feng, Chungang Yang, Xinru Mi, Shanqing Jiang, Mohsen Guizani. (2023)  
**IS2N: Intent-Driven Security Software-Defined Network with Blockchain**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.04641v1)  

---


**ABSTRACT**  
Software-defined network (SDN) is characterized by its programmability, flexibility, and the separation of control and data planes. However, SDN still have many challenges, particularly concerning the security of network information synchronization and network element registration. Blockchain and intent-driven networks are recent technologies to establish secure and intelligent SDN. This article investigates the blockchain-based architecture and intent-driven mechanisms to implement intent-driven security software-defined networks (IS2N). Specifically, we propose a novel four-layer architecture of the IS2N with security capabilities. We integrate an intent-driven security management mechanism in the IS2N to achieve automate network security management. Finally, we develop an IS2N platform with blockchain middle-layer to achieve security capabilities and security store network-level snapshots, such as device registration and OpenFlow messages. Our simulations show that IS2N is more flexible than conventional strategies at resolving problems during network operations and has a minimal effect on the SDN.

{{</citation>}}


## cs.RO (2)



### (105/109) An Autonomous Hybrid Drone-Rover Vehicle for Weed Removal and Spraying Applications in Agriculture (J Krishna Kant et al., 2023)

{{<citation>}}

J Krishna Kant, Mahankali Sripaad, Anand Bharadwaj, Rajashekhar V S, Suresh Sundaram. (2023)  
**An Autonomous Hybrid Drone-Rover Vehicle for Weed Removal and Spraying Applications in Agriculture**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2308.04794v1)  

---


**ABSTRACT**  
The usage of drones and rovers helps to overcome the limitations of traditional agriculture which has been predominantly human-intensive, for carrying out tasks such as removal of weeds and spraying of fertilizers and pesticides. Drones and rovers are helping to realize precision agriculture and farmers with improved monitoring and surveying at affordable costs. Major benefits have come for vertical farming and fields with irrigation canals. However, drones have a limitation of flight time due to payload constraints. Rovers have limitations in vertical farming and obstacles like canals in agricultural fields. To meet the different requirements of multiple terrains and vertical farming in agriculture, we propose an autonomous hybrid drone-rover vehicle that combines the advantages of both rovers and drones. The prototype is described along with experimental results regarding its ability to avoid obstacles, pluck weeds and spray pesticides.

{{</citation>}}


### (106/109) E3-UAV: An Edge-based Energy-Efficient Object Detection System for Unmanned Aerial Vehicles (Jiashun Suo et al., 2023)

{{<citation>}}

Jiashun Suo, Xingzhou Zhang, Weisong Shi, Wei Zhou. (2023)  
**E3-UAV: An Edge-based Energy-Efficient Object Detection System for Unmanned Aerial Vehicles**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.04774v1)  

---


**ABSTRACT**  
Motivated by the advances in deep learning techniques, the application of Unmanned Aerial Vehicle (UAV)-based object detection has proliferated across a range of fields, including vehicle counting, fire detection, and city monitoring. While most existing research studies only a subset of the challenges inherent to UAV-based object detection, there are few studies that balance various aspects to design a practical system for energy consumption reduction. In response, we present the E3-UAV, an edge-based energy-efficient object detection system for UAVs. The system is designed to dynamically support various UAV devices, edge devices, and detection algorithms, with the aim of minimizing energy consumption by deciding the most energy-efficient flight parameters (including flight altitude, flight speed, detection algorithm, and sampling rate) required to fulfill the detection requirements of the task. We first present an effective evaluation metric for actual tasks and construct a transparent energy consumption model based on hundreds of actual flight data to formalize the relationship between energy consumption and flight parameters. Then we present a lightweight energy-efficient priority decision algorithm based on a large quantity of actual flight data to assist the system in deciding flight parameters. Finally, we evaluate the performance of the system, and our experimental results demonstrate that it can significantly decrease energy consumption in real-world scenarios. Additionally, we provide four insights that can assist researchers and engineers in their efforts to study UAV-based object detection further.

{{</citation>}}


## cs.PL (1)



### (107/109) Local Reasoning about Probabilistic Behaviour for Classical-Quantum Programs (Yuxin Deng et al., 2023)

{{<citation>}}

Yuxin Deng, Huiling Wu, Ming Xu. (2023)  
**Local Reasoning about Probabilistic Behaviour for Classical-Quantum Programs**  

---
Primary Category: cs.PL  
Categories: F-3-1, F-3-2, cs-PL, cs.PL, quant-ph  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.04741v1)  

---


**ABSTRACT**  
Verifying the functional correctness of programs with both classical and quantum constructs is a challenging task. The presence of probabilistic behaviour entailed by quantum measurements and unbounded while loops complicate the verification task greatly. We propose a new quantum Hoare logic for local reasoning about probabilistic behaviour by introducing distribution formulas to specify probabilistic properties. We show that the proof rules in the logic are sound with respect to a denotational semantics. To demonstrate the effectiveness of the logic, we formally verify the correctness of non-trivial quantum algorithms including the HHL and Shor's algorithms.

{{</citation>}}


## physics.geo-ph (1)



### (108/109) Optimizing a Transformer-based network for a deep learning seismic processing workflow (Randy Harsuko et al., 2023)

{{<citation>}}

Randy Harsuko, Tariq Alkhalifah. (2023)  
**Optimizing a Transformer-based network for a deep learning seismic processing workflow**  

---
Primary Category: physics.geo-ph  
Categories: cs-LG, physics-geo-ph, physics.geo-ph  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2308.04739v1)  

---


**ABSTRACT**  
StorSeismic is a recently introduced model based on the Transformer to adapt to various seismic processing tasks through its pretraining and fine-tuning training strategy. In the original implementation, StorSeismic utilized a sinusoidal positional encoding and a conventional self-attention mechanism, both borrowed from the natural language processing (NLP) applications. For seismic processing they admitted good results, but also hinted to limitations in efficiency and expressiveness. We propose modifications to these two key components, by utilizing relative positional encoding and low-rank attention matrices as replacements to the vanilla ones. The proposed changes are tested on processing tasks applied to a realistic Marmousi and offshore field data as a sequential strategy, starting from denoising, direct arrival removal, multiple attenuation, and finally root-mean-squared velocity ($V_{RMS}$) prediction for normal moveout (NMO) correction. We observe faster pretraining and competitive results on the fine-tuning tasks and, additionally, fewer parameters to train compared to the vanilla model.

{{</citation>}}


## cs.IR (1)



### (109/109) Pareto Invariant Representation Learning for Multimedia Recommendation (Shanshan Huang et al., 2023)

{{<citation>}}

Shanshan Huang, Haoxuan Li, Qingsong Li, Chunyuan Zheng, Li Liu. (2023)  
**Pareto Invariant Representation Learning for Multimedia Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.04706v1)  

---


**ABSTRACT**  
Multimedia recommendation involves personalized ranking tasks, where multimedia content is usually represented using a generic encoder. However, these generic representations introduce spurious correlations that fail to reveal users' true preferences. Existing works attempt to alleviate this problem by learning invariant representations, but overlook the balance between independent and identically distributed (IID) and out-of-distribution (OOD) generalization. In this paper, we propose a framework called Pareto Invariant Representation Learning (PaInvRL) to mitigate the impact of spurious correlations from an IID-OOD multi-objective optimization perspective, by learning invariant representations (intrinsic factors that attract user attention) and variant representations (other factors) simultaneously. Specifically, PaInvRL includes three iteratively executed modules: (i) heterogeneous identification module, which identifies the heterogeneous environments to reflect distributional shifts for user-item interactions; (ii) invariant mask generation module, which learns invariant masks based on the Pareto-optimal solutions that minimize the adaptive weighted Invariant Risk Minimization (IRM) and Empirical Risk (ERM) losses; (iii) convert module, which generates both variant representations and item-invariant representations for training a multi-modal recommendation model that mitigates spurious correlations and balances the generalization performance within and cross the environmental distributions. We compare the proposed PaInvRL with state-of-the-art recommendation models on three public multimedia recommendation datasets (Movielens, Tiktok, and Kwai), and the experimental results validate the effectiveness of PaInvRL for both within- and cross-environmental learning.

{{</citation>}}
