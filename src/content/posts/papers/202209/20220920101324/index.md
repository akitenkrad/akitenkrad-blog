---
draft: false
title: "A Learning-based Data Augmentation for network anomaly Detection"
date: 2022-09-20
author: "akitenkrad"
description: ""
tags: ["At:Round-2", "Published:2020", "Augmentation", "GNN", "Anomaly Detection", "DS:UNSW-NB15", "DS:CIC-IDS2017"]
menu:
  sidebar:
    name: "A Learning-based Data Augmentation for network anomaly Detection"
    identifier: 20220920
    parent: 202209
    weight: 10
math: true
---

- [x] Round-1: Overview
- [x] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation

{{< citation >}}
al Olaimat, M., Lee, D., Kim, Y., Kim, J., & Kim, J. (2020).  
A Learning-based Data Augmentation for Network Anomaly Detection.  
International Conference on Computer Communications and Networks, ICCCN, 2020-August.  
https://doi.org/10.1109/ICCCN49398.2020.9209598
{{< /citation >}}

## Abstract
> While machine learning technologies have been remarkably advanced over the past several years, one of the fundamental requirements for the success of learning-based approaches would be the availability of high-quality data that thoroughly represent individual classes in a problem space. Unfortunately, it is not uncommon to observe a significant degree of class imbalance with only a few instances for minority classes in many datasets, including network traffic traces highly skewed toward a large number of normal connections while very small in quantity for attack instances. A well-known approach to addressing the class imbalance problem is data augmentation that generates synthetic instances belonging to minority classes. However, traditional statistical techniques may be limited since the extended data through statistical sampling should have the same density as original data instances with a minor degree of variation. This paper takes a learning-based approach to data augmentation to enable effective network anomaly detection. One of the critical challenges for the learning-based approach is the mode collapse problem resulting in a limited diversity of samples, which was also observed from our preliminary experimental result. To this end, we present a novel "Divide-Augment-Combine"(DAC) strategy, which groups the instances based on their characteristics and augments data on a group basis to represent a subset independently using a generative adversarial model. Our experimental results conducted with two recently collected public network datasets (UNSW-NB15 and IDS-2017) show that the proposed technique enhances performances up to 21.5% for identifying network anomalies.

## Background & Wat's New
- Network Anomaly Detection では異常データが少なく，正常データに偏った分布をしていることが多い
- GANによる Augmentation も研究されているが，先行研究では SMOTE よりも少し良い程度にとどまっている
  - シンプルな GAN を検証したところ，ただ適用するだけでは精度向上に寄与しないことがわかった
- mode collapse
  - confines the capability of the generator to produce only a limited diversity of samples
- Divide-Augment-Combine (DAC) による Augmentation を提案
  - サンプルを特徴に応じてグルーピング (divide)
  - GANを用いてグループ単位で Data Augmentation を実施 (augment)
  - モデルを学習 (combine)

## Dataset

- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

## Model Description

基本となる GAN のアーキテクチャに対して改良を実施した．

{{< figure src="BasicGAN.png" width="50%" caption="Basic GAN architecture" >}}

<br/>

- **Number of hidden layers**  
  隠れ層が2層のモデルと5層のモデルをそれぞれ検討した結果，隠れ層の数が処理速度に大きく影響することがわかった．  
  最終的には2層をデフォルト値として選定した．
- **Loss function**  
  MAE (Mean Absolute Error) と BCE (Binary Cross Entropy) を採用した．  
  生成器には MAE を，識別器には BCE をそれぞれ採用した．
- **Activation function**  
  sigmoid と tanh をそれぞれ検討した．  
  デフォルトの設定は sigmoid とした．
- **Number of epochs**  
  モデルが収束するためには一定のエポック数が必要だが，今回使用するデータセットにおいてはエポック数の変動はあまり関係しないことがわかった．
- **Learning Rate**  
  デフォルト値は $1\mathrm{e}{-3}$ とした．
- **Noise vector**  
  出力のためのランダムノイズで一般的には $100$ が使われるため，今回の研究でも同様の設定を採用した．
- **Dropout**  
  未知のデータに対して識別器を頑健にするため，Dropoutを導入した．

### Divide-Augment-Combine (DAC)
- Divide: にた特徴を持つデータをグルーピングする
  - Attack Type によるグルーピング
  - Clustering (k-means) によるグルーピング
- Augment: グループ分けされたデータに対してそれぞれ独立に GAN を適用し Augment する
- Combine: Augment したデータを等号して後続タスクに投入する

## Results

### Results on UNSW-NB15

{{< figure src="table-1.png" width="50%" caption="Classification Performance of Neural Network Classifier for Augmentation by Attack Types on UNSW-NB15" >}}

{{< figure src="table-2.png" width="50%" caption="Classification Performance of Random Forest Classifier for Augmentation by Attack Types on UNSW-NB15" >}}

{{< figure src="table-3.png" width="50%" caption="Classification Performance of Neural Network Classifier for Augmentation by $k$-Means on UNSW-NB15" >}}

{{< figure src="table-4.png" width="50%" caption="Classification Performance of Random Forest Classifier for Augmentation by $k$-Means on UNSW-NB15" >}}

### Results on CIC-IDS2017

{{< figure src="table-5.png" width="50%" caption="Classification Performance of Neural Network Classifier for Augmentation by Attack Types on CIC-IDS2017" >}}

{{< figure src="table-6.png" width="50%" caption="Classification Performance of Random Forest Classifier for Augmentation by Attack Types on CIC-IDS2017" >}}

## References

{{< ci-details summary="Using generative adversarial networks for improving classification effectiveness in credit card fraud detection (Ugo Fiore et al., 2017)">}}
Ugo Fiore, A. D. Santis, F. Perla, P. Zanetti, F. Palmieri. (2017)  
**Using generative adversarial networks for improving classification effectiveness in credit card fraud detection**  
Inf. Sci.  
[Paper Link](https://www.semanticscholar.org/paper/3bf418882a9de772eed10b3177f92f541d1735a6)  
Influential Citation Count (14), SS-ID (3bf418882a9de772eed10b3177f92f541d1735a6)  
{{< /ci-details >}}
{{< ci-details summary="Learning From Imbalanced Data (L. Mathews et al., 2019)">}}
L. Mathews, Seetha Hari. (2019)  
**Learning From Imbalanced Data**  
Advances in Computer and Electrical Engineering  
[Paper Link](https://www.semanticscholar.org/paper/6a7364f6ed2846ea2b705336a4c49dd287102a50)  
Influential Citation Count (154), SS-ID (6a7364f6ed2846ea2b705336a4c49dd287102a50)  
**ABSTRACT**  
A very challenging issue in real-world data is that in many domains like medicine, finance, marketing, web, telecommunication, management, etc. the distribution of data among classes is inherently imbalanced. A widely accepted researched issue is that the traditional classifier algorithms assume a balanced distribution among the classes. Data imbalance is evident when the number of instances representing the class of concern is much lesser than other classes. Hence, the classifiers tend to bias towards the well-represented class. This leads to a higher misclassification rate among the lesser represented class. Hence, there is a need of efficient learners to classify imbalanced data. This chapter aims to address the need, challenges, existing methods, and evaluation metrics identified when learning from imbalanced data sets. Future research challenges and directions are highlighted.
{{< /ci-details >}}
{{< ci-details summary="Benchmarking datasets for Anomaly-based Network Intrusion Detection: KDD CUP 99 alternatives (Abhishek Divekar et al., 2018)">}}
Abhishek Divekar, Meet Parekh, Vaibhav Savla, Rudra Mishra, M. Shirole. (2018)  
**Benchmarking datasets for Anomaly-based Network Intrusion Detection: KDD CUP 99 alternatives**  
2018 IEEE 3rd International Conference on Computing, Communication and Security (ICCCS)  
[Paper Link](https://www.semanticscholar.org/paper/34f9ffd51a173290bfeca44578ed5cc915559bbd)  
Influential Citation Count (6), SS-ID (34f9ffd51a173290bfeca44578ed5cc915559bbd)  
**ABSTRACT**  
Machine Learning has been steadily gaining traction for its use in Anomaly-based Network Intrusion Detection Systems (A-NIDS). Research into this domain is frequently performed using the KDD CUP 99 dataset as a benchmark. Several studies question its usability while constructing a contemporary NIDS, due to the skewed response distribution, non-stationarity, and failure to incorporate modern attacks. In this paper, we compare the performance for KDD-99 alternatives when trained using classification models commonly found in literature: Neural Network, Support Vector Machine, Decision Tree, Random Forest, Naive Bayes and K-Means. Applying the SMOTE oversampling technique and random undersampling, we create a balanced version of NSL-KDD and prove that skewed target classes in KDD-99 and NSL-KDD hamper the efficacy of classifiers on minority classes (U2R and R2L), leading to possible security risks. We explore UNSW-NB15, a modern substitute to KDD-99 with greater uniformity of pattern distribution. We benchmark this dataset before and after SMOTE oversampling to observe the effect on minority performance. Our results indicate that classifiers trained on UNSW-NB15 match or better the Weighted F1-Score of those trained on NSL-KDD and KDD-99 in the binary case, thus advocating UNSW-NB15 as a modern substitute to these datasets.
{{< /ci-details >}}
{{< ci-details summary="Anomaly Detection with Generative Adversarial Networks for Multivariate Time Series (Dan Li et al., 2018)">}}
Dan Li, Dacheng Chen, Jonathan Goh, See-Kiong Ng. (2018)  
**Anomaly Detection with Generative Adversarial Networks for Multivariate Time Series**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/c92ee6ff32fa4833fa1c2bdf29284e2a58ddb640)  
Influential Citation Count (19), SS-ID (c92ee6ff32fa4833fa1c2bdf29284e2a58ddb640)  
**ABSTRACT**  
Today's Cyber-Physical Systems (CPSs) are large, complex, and affixed with networked sensors and actuators that are targets for cyber-attacks. Conventional detection techniques are unable to deal with the increasingly dynamic and complex nature of the CPSs. On the other hand, the networked sensors and actuators generate large amounts of data streams that can be continuously monitored for intrusion events. Unsupervised machine learning techniques can be used to model the system behaviour and classify deviant behaviours as possible attacks. In this work, we proposed a novel Generative Adversarial Networks-based Anomaly Detection (GAN-AD) method for such complex networked CPSs. We used LSTM-RNN in our GAN to capture the distribution of the multivariate time series of the sensors and actuators under normal working conditions of a CPS. Instead of treating each sensor's and actuator's time series independently, we model the time series of multiple sensors and actuators in the CPS concurrently to take into account of potential latent interactions between them. To exploit both the generator and the discriminator of our GAN, we deployed the GAN-trained discriminator together with the residuals between generator-reconstructed data and the actual samples to detect possible anomalies in the complex CPS. We used our GAN-AD to distinguish abnormal attacked situations from normal working conditions for a complex six-stage Secure Water Treatment (SWaT) system. Experimental results showed that the proposed strategy is effective in identifying anomalies caused by various attacks with high detection rate and low false positive rate as compared to existing methods.
{{< /ci-details >}}
{{< ci-details summary="DOPING: Generative Data Augmentation for Unsupervised Anomaly Detection with GAN (Swee Kiat Lim et al., 2018)">}}
Swee Kiat Lim, Yi Loo, Ngoc-Trung Tran, Ngai-Man Cheung, G. Roig, Y. Elovici. (2018)  
**DOPING: Generative Data Augmentation for Unsupervised Anomaly Detection with GAN**  
2018 IEEE International Conference on Data Mining (ICDM)  
[Paper Link](https://www.semanticscholar.org/paper/a47f8794d88c5c27123153c4eb9e08046e2b0c9d)  
Influential Citation Count (2), SS-ID (a47f8794d88c5c27123153c4eb9e08046e2b0c9d)  
**ABSTRACT**  
Recently, the introduction of the generative adversarial network (GAN) and its variants has enabled the generation of realistic synthetic samples, which has been used for enlarging training sets. Previous work primarily focused on data augmentation for semi-supervised and supervised tasks. In this paper, we instead focus on unsupervised anomaly detection and propose a novel generative data augmentation framework optimized for this task. In particular, we propose to oversample infrequent normal samples - normal samples that occur with small probability, e.g., rare normal events. We show that these samples are responsible for false positives in anomaly detection. However, oversampling of infrequent normal samples is challenging for real-world high-dimensional data with multimodal distributions. To address this challenge, we propose to use a GAN variant known as the adversarial autoencoder (AAE) to transform the high-dimensional multimodal data distributions into low-dimensional unimodal latent distributions with well-defined tail probability. Then, we systematically oversample at the 'edge' of the latent distributions to increase the density of infrequent normal samples. We show that our oversampling pipeline is a unified one: it is generally applicable to datasets with different complex data distributions. To the best of our knowledge, our method is the first data augmentation technique focused on improving performance in unsupervised anomaly detection. We validate our method by demonstrating consistent improvements across several real-world datasets.
{{< /ci-details >}}
{{< ci-details summary="Efficient GAN-Based Anomaly Detection (Houssam Zenati et al., 2018)">}}
Houssam Zenati, Chuan-Sheng Foo, Bruno Lecouat, Gaurav Manek, V. Chandrasekhar. (2018)  
**Efficient GAN-Based Anomaly Detection**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/b3acb6f183b5f4b651f53c0eec5cb5c805224ac1)  
Influential Citation Count (59), SS-ID (b3acb6f183b5f4b651f53c0eec5cb5c805224ac1)  
**ABSTRACT**  
Generative adversarial networks (GANs) are able to model the complex highdimensional distributions of real-world data, which suggests they could be effective for anomaly detection. However, few works have explored the use of GANs for the anomaly detection task. We leverage recently developed GAN models for anomaly detection, and achieve state-of-the-art performance on image and network intrusion datasets, while being several hundred-fold faster at test time than the only published GAN-based method.
{{< /ci-details >}}
{{< ci-details summary="Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization (Iman Sharafaldin et al., 2018)">}}
Iman Sharafaldin, Arash Habibi Lashkari, A. Ghorbani. (2018)  
**Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization**  
ICISSP  
[Paper Link](https://www.semanticscholar.org/paper/a27089efabc5f4abd5ddf2be2a409bff41f31199)  
Influential Citation Count (249), SS-ID (a27089efabc5f4abd5ddf2be2a409bff41f31199)  
**ABSTRACT**  
With exponential growth in the size of computer networks and developed applications, the significant increasing of the potential damage that can be caused by launching attacks is becoming obvious. Meanwhile, Intrusion Detection Systems (IDSs) and Intrusion Prevention Systems (IPSs) are one of the most important defense tools against the sophisticated and ever-growing network attacks. Due to the lack of adequate dataset, anomaly-based approaches in intrusion detection systems are suffering from accurate deployment, analysis and evaluation. There exist a number of such datasets such as DARPA98, KDD99, ISC2012, and ADFA13 that have been used by the researchers to evaluate the performance of their proposed intrusion detection and intrusion prevention approaches. Based on our study over eleven available datasets since 1998, many such datasets are out of date and unreliable to use. Some of these datasets suffer from lack of traffic diversity and volumes, some of them do not cover the variety of attacks, while others anonymized packet information and payload which cannot reflect the current trends, or they lack feature set and metadata. This paper produces a reliable dataset that contains benign and seven common attack network flows, which meets real world criteria and is publicly avaliable. Consequently, the paper evaluates the performance of a comprehensive set of network traffic features and machine learning algorithms to indicate the best set of features for detecting the certain attack categories.
{{< /ci-details >}}
{{< ci-details summary="Big data analytics: Understanding its capabilities and potential benefits for healthcare organizations (Yichuan Wang et al., 2018)">}}
Yichuan Wang, LeeAnn Kung, T. Byrd. (2018)  
**Big data analytics: Understanding its capabilities and potential benefits for healthcare organizations**  
  
[Paper Link](https://www.semanticscholar.org/paper/16575f23ff879e6353a55bbfbbcc54e27606bfc5)  
Influential Citation Count (31), SS-ID (16575f23ff879e6353a55bbfbbcc54e27606bfc5)  
{{< /ci-details >}}
{{< ci-details summary="Credit Card Fraud Detection Using AdaBoost and Majority Voting (Kuldeep Randhawa et al., 2018)">}}
Kuldeep Randhawa, C. Loo, M. Seera, C. Lim, A. Nandi. (2018)  
**Credit Card Fraud Detection Using AdaBoost and Majority Voting**  
IEEE Access  
[Paper Link](https://www.semanticscholar.org/paper/aff9897e0acf74aca20acf3d76c621dee9fd169c)  
Influential Citation Count (8), SS-ID (aff9897e0acf74aca20acf3d76c621dee9fd169c)  
**ABSTRACT**  
Credit card fraud is a serious problem in financial services. Billions of dollars are lost due to credit card fraud every year. There is a lack of research studies on analyzing real-world credit card data owing to confidentiality issues. In this paper, machine learning algorithms are used to detect credit card fraud. Standard models are first used. Then, hybrid methods which use AdaBoost and majority voting methods are applied. To evaluate the model efficacy, a publicly available credit card data set is used. Then, a real-world credit card data set from a financial institution is analyzed. In addition, noise is added to the data samples to further assess the robustness of the algorithms. The experimental results positively indicate that the majority voting method achieves good accuracy rates in detecting fraud cases in credit cards.
{{< /ci-details >}}
{{< ci-details summary="Adaptive Swarm Balancing Algorithms for rare-event prediction in imbalanced healthcare data (Jinyan Li et al., 2017)">}}
Jinyan Li, Liansheng Liu, S. Fong, R. Wong, Sabah Mohammed, J. Fiaidhi, Yunsick Sung, Kelvin K. L. Wong. (2017)  
**Adaptive Swarm Balancing Algorithms for rare-event prediction in imbalanced healthcare data**  
PloS one  
[Paper Link](https://www.semanticscholar.org/paper/c6995b573e2d7431140d3deb49f0d94d87aba2c1)  
Influential Citation Count (0), SS-ID (c6995b573e2d7431140d3deb49f0d94d87aba2c1)  
**ABSTRACT**  
Clinical data analysis and forecasting have made substantial contributions to disease control, prevention and detection. However, such data usually suffer from highly imbalanced samples in class distributions. In this paper, we aim to formulate effective methods to rebalance binary imbalanced dataset, where the positive samples take up only the minority. We investigate two different meta-heuristic algorithms, particle swarm optimization and bat algorithm, and apply them to empower the effects of synthetic minority over-sampling technique (SMOTE) for pre-processing the datasets. One approach is to process the full dataset as a whole. The other is to split up the dataset and adaptively process it one segment at a time. The experimental results reported in this paper reveal that the performance improvements obtained by the former methods are not scalable to larger data scales. The latter methods, which we call Adaptive Swarm Balancing Algorithms, lead to significant efficiency and effectiveness improvements on large datasets while the first method is invalid. We also find it more consistent with the practice of the typical large imbalanced medical datasets. We further use the meta-heuristic algorithms to optimize two key parameters of SMOTE. The proposed methods lead to more credible performances of the classifier, and shortening the run time compared to brute-force method.
{{< /ci-details >}}
{{< ci-details summary="Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (T. Schlegl et al., 2017)">}}
T. Schlegl, Philipp Seeböck, S. Waldstein, U. Schmidt-Erfurth, G. Langs. (2017)  
**Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery**  
IPMI  
[Paper Link](https://www.semanticscholar.org/paper/e163a2e89c136cb4442e34c72f7173a0ff46dc79)  
Influential Citation Count (184), SS-ID (e163a2e89c136cb4442e34c72f7173a0ff46dc79)  
{{< /ci-details >}}
{{< ci-details summary="A survey of deep learning-based network anomaly detection (Donghwoon Kwon et al., 2017)">}}
Donghwoon Kwon, Hyunjoo Kim, Jinoh Kim, S. Suh, Ikkyun Kim, Kuinam J. Kim. (2017)  
**A survey of deep learning-based network anomaly detection**  
Cluster Computing  
[Paper Link](https://www.semanticscholar.org/paper/5b59992ca6b77aaec066a0d3142336d2cb1028f1)  
Influential Citation Count (11), SS-ID (5b59992ca6b77aaec066a0d3142336d2cb1028f1)  
{{< /ci-details >}}
{{< ci-details summary="The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set (Nour Moustafa et al., 2016)">}}
Nour Moustafa, J. Slay. (2016)  
**The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set**  
Inf. Secur. J. A Glob. Perspect.  
[Paper Link](https://www.semanticscholar.org/paper/2a56ca15bea4f9a911835bfd08a2f4526091d785)  
Influential Citation Count (52), SS-ID (2a56ca15bea4f9a911835bfd08a2f4526091d785)  
**ABSTRACT**  
ABSTRACT Over the last three decades, Network Intrusion Detection Systems (NIDSs), particularly, Anomaly Detection Systems (ADSs), have become more significant in detecting novel attacks than Signature Detection Systems (SDSs). Evaluating NIDSs using the existing benchmark data sets of KDD99 and NSLKDD does not reflect satisfactory results, due to three major issues: (1) their lack of modern low footprint attack styles, (2) their lack of modern normal traffic scenarios, and (3) a different distribution of training and testing sets. To address these issues, the UNSW-NB15 data set has recently been generated. This data set has nine types of the modern attacks fashions and new patterns of normal traffic, and it contains 49 attributes that comprise the flow based between hosts and the network packets inspection to discriminate between the observations, either normal or abnormal. In this paper, we demonstrate the complexity of the UNSW-NB15 data set in three aspects. First, the statistical analysis of the observations and the attributes are explained. Second, the examination of feature correlations is provided. Third, five existing classifiers are used to evaluate the complexity in terms of accuracy and false alarm rates (FARs) and then, the results are compared with the KDD99 data set. The experimental results show that UNSW-NB15 is more complex than KDD99 and is considered as a new benchmark data set for evaluating NIDSs.
{{< /ci-details >}}
{{< ci-details summary="UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set) (Nour Moustafa et al., 2015)">}}
Nour Moustafa, J. Slay. (2015)  
**UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)**  
2015 Military Communications and Information Systems Conference (MilCIS)  
[Paper Link](https://www.semanticscholar.org/paper/0e7af8e91b8cb2cea1164be5ac5d280b0d12c153)  
Influential Citation Count (170), SS-ID (0e7af8e91b8cb2cea1164be5ac5d280b0d12c153)  
**ABSTRACT**  
One of the major research challenges in this field is the unavailability of a comprehensive network based data set which can reflect modern network traffic scenarios, vast varieties of low footprint intrusions and depth structured information about the network traffic. Evaluating network intrusion detection systems research efforts, KDD98, KDDCUP99 and NSLKDD benchmark data sets were generated a decade ago. However, numerous current studies showed that for the current network threat environment, these data sets do not inclusively reflect network traffic and modern low footprint attacks. Countering the unavailability of network benchmark data set challenges, this paper examines a UNSW-NB15 data set creation. This data set has a hybrid of the real modern normal and the contemporary synthesized attack activities of the network traffic. Existing and novel methods are utilised to generate the features of the UNSWNB15 data set. This data set is available for research purposes and can be accessed from the link.
{{< /ci-details >}}
{{< ci-details summary="Applications of big data to smart cities (Eiman Al Nuaimi et al., 2015)">}}
Eiman Al Nuaimi, Hind Al Neyadi, N. Mohamed, J. Al-Jaroodi. (2015)  
**Applications of big data to smart cities**  
Journal of Internet Services and Applications  
[Paper Link](https://www.semanticscholar.org/paper/edfe9322f47458c937fe01911e0a3aae85a87eeb)  
Influential Citation Count (29), SS-ID (edfe9322f47458c937fe01911e0a3aae85a87eeb)  
{{< /ci-details >}}
{{< ci-details summary="Big-data applications in the government sector (Gang-hoon Kim et al., 2014)">}}
Gang-hoon Kim, S. Trimi, Ji-Hyong Chung. (2014)  
**Big-data applications in the government sector**  
Commun. ACM  
[Paper Link](https://www.semanticscholar.org/paper/ec02b379a635346ec1501e801263ea576a10ed4c)  
Influential Citation Count (20), SS-ID (ec02b379a635346ec1501e801263ea576a10ed4c)  
**ABSTRACT**  
In the same way businesses use big data to pursue profits, governments use it to promote the public good.
{{< /ci-details >}}
{{< ci-details summary="Network Anomaly Detection: Methods, Systems and Tools (M. Bhuyan et al., 2014)">}}
M. Bhuyan, D. Bhattacharyya, J. Kalita. (2014)  
**Network Anomaly Detection: Methods, Systems and Tools**  
IEEE Communications Surveys & Tutorials  
[Paper Link](https://www.semanticscholar.org/paper/e467ff1ea3a0c535a0cef87f7e0daa414d524d9d)  
Influential Citation Count (63), SS-ID (e467ff1ea3a0c535a0cef87f7e0daa414d524d9d)  
**ABSTRACT**  
Network anomaly detection is an important and dynamic research area. Many network intrusion detection methods and systems (NIDS) have been proposed in the literature. In this paper, we provide a structured and comprehensive overview of various facets of network anomaly detection so that a researcher can become quickly familiar with every aspect of network anomaly detection. We present attacks normally encountered by network intrusion detection systems. We categorize existing network anomaly detection methods and systems based on the underlying computational techniques used. Within this framework, we briefly describe and compare a large number of network anomaly detection methods and systems. In addition, we also discuss tools that can be used by network defenders and datasets that researchers in network anomaly detection can use. We also highlight research directions in network anomaly detection.
{{< /ci-details >}}
{{< ci-details summary="Encyclopedia of Social Network Analysis and Mining (F. Stokman, 2014)">}}
F. Stokman. (2014)  
**Encyclopedia of Social Network Analysis and Mining**  
  
[Paper Link](https://www.semanticscholar.org/paper/c535b7c9ca61a75adc0dcd0325de2acb6d83dc78)  
Influential Citation Count (1), SS-ID (c535b7c9ca61a75adc0dcd0325de2acb6d83dc78)  
{{< /ci-details >}}
{{< ci-details summary="Network Anomaly Detection Using Co-clustering (E. Papalexakis et al., 2012)">}}
E. Papalexakis, Alex Beutel, P. Steenkiste. (2012)  
**Network Anomaly Detection Using Co-clustering**  
2012 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining  
[Paper Link](https://www.semanticscholar.org/paper/dd9cd568de5af206b05b6d58c752b4240120d1b7)  
Influential Citation Count (0), SS-ID (dd9cd568de5af206b05b6d58c752b4240120d1b7)  
{{< /ci-details >}}
{{< ci-details summary="Predicting disease risks from highly imbalanced data using random forest (Mohammad Khalilia et al., 2011)">}}
Mohammad Khalilia, S. Chakraborty, M. Popescu. (2011)  
**Predicting disease risks from highly imbalanced data using random forest**  
BMC Medical Informatics Decis. Mak.  
[Paper Link](https://www.semanticscholar.org/paper/f0938481d1439afeca809a643585ccdc1c1234ce)  
Influential Citation Count (8), SS-ID (f0938481d1439afeca809a643585ccdc1c1234ce)  
{{< /ci-details >}}
{{< ci-details summary="Anomaly detection: A survey (V. Chandola et al., 2009)">}}
V. Chandola, A. Banerjee, Vipin Kumar. (2009)  
**Anomaly detection: A survey**  
CSUR  
[Paper Link](https://www.semanticscholar.org/paper/71d1ac92ad36b62a04f32ed75a10ad3259a7218d)  
Influential Citation Count (700), SS-ID (71d1ac92ad36b62a04f32ed75a10ad3259a7218d)  
**ABSTRACT**  
Anomaly detection is an important problem that has been researched within diverse research areas and application domains. Many anomaly detection techniques have been specifically developed for certain application domains, while others are more generic. This survey tries to provide a structured and comprehensive overview of the research on anomaly detection. We have grouped existing techniques into different categories based on the underlying approach adopted by each technique. For each category we have identified key assumptions, which are used by the techniques to differentiate between normal and anomalous behavior. When applying a given technique to a particular domain, these assumptions can be used as guidelines to assess the effectiveness of the technique in that domain. For each category, we provide a basic anomaly detection technique, and then show how the different existing techniques in that category are variants of the basic technique. This template provides an easier and more succinct understanding of the techniques belonging to each category. Further, for each category, we identify the advantages and disadvantages of the techniques in that category. We also provide a discussion on the computational complexity of the techniques since it is an important issue in real application domains. We hope that this survey will provide a better understanding of the different directions in which research has been done on this topic, and how techniques developed in one area can be applied in domains for which they were not intended to begin with.
{{< /ci-details >}}
{{< ci-details summary="Visualizing Data using t-SNE (L. V. D. Maaten et al., 2008)">}}
L. V. D. Maaten, Geoffrey E. Hinton. (2008)  
**Visualizing Data using t-SNE**  
  
[Paper Link](https://www.semanticscholar.org/paper/1c46943103bd7b7a2c7be86859995a4144d1938b)  
Influential Citation Count (919), SS-ID (1c46943103bd7b7a2c7be86859995a4144d1938b)  
**ABSTRACT**  
We present a new technique called “t-SNE” that visualizes high-dimensional data by giving each datapoint a location in a two or three-dimensional map. The technique is a variation of Stochastic Neighbor Embedding (Hinton and Roweis, 2002) that is much easier to optimize, and produces significantly better visualizations by reducing the tendency to crowd points together in the center of the map. t-SNE is better than existing techniques at creating a single map that reveals structure at many different scales. This is particularly important for high-dimensional data that lie on several different, but related, low-dimensional manifolds, such as images of objects from multiple classes seen from multiple viewpoints. For visualizing the structure of very large datasets, we show how t-SNE can use random walks on neighborhood graphs to allow the implicit structure of all of the data to influence the way in which a subset of the data is displayed. We illustrate the performance of t-SNE on a wide variety of datasets and compare it with many other non-parametric visualization techniques, including Sammon mapping, Isomap, and Locally Linear Embedding. The visualizations produced by t-SNE are significantly better than those produced by the other techniques on almost all of the datasets.
{{< /ci-details >}}
{{< ci-details summary="Applying Support Vector Machines to Imbalanced Datasets (R. Akbani et al., 2004)">}}
R. Akbani, Stephen Kwek, N. Japkowicz. (2004)  
**Applying Support Vector Machines to Imbalanced Datasets**  
ECML  
[Paper Link](https://www.semanticscholar.org/paper/45a9e2aa04e91bb511f36c365ef4daa274fe583c)  
Influential Citation Count (83), SS-ID (45a9e2aa04e91bb511f36c365ef4daa274fe583c)  
{{< /ci-details >}}
{{< ci-details summary="SMOTE: Synthetic Minority Over-sampling Technique (N. Chawla et al., 2002)">}}
N. Chawla, K. Bowyer, L. Hall, W. Kegelmeyer. (2002)  
**SMOTE: Synthetic Minority Over-sampling Technique**  
J. Artif. Intell. Res.  
[Paper Link](https://www.semanticscholar.org/paper/8cb44f06586f609a29d9b496cc752ec01475dffe)  
Influential Citation Count (2239), SS-ID (8cb44f06586f609a29d9b496cc752ec01475dffe)  
**ABSTRACT**  
An approach to the construction of classifiers from imbalanced datasets is described. A dataset is imbalanced if the classification categories are not approximately equally represented. Often real-world data sets are predominately composed of "normal" examples with only a small percentage of "abnormal" or "interesting" examples. It is also the case that the cost of misclassifying an abnormal (interesting) example as a normal example is often much higher than the cost of the reverse error. Under-sampling of the majority (normal) class has been proposed as a good means of increasing the sensitivity of a classifier to the minority class. This paper shows that a combination of our method of oversampling the minority (abnormal)cla ss and under-sampling the majority (normal) class can achieve better classifier performance (in ROC space)tha n only under-sampling the majority class. This paper also shows that a combination of our method of over-sampling the minority class and under-sampling the majority class can achieve better classifier performance (in ROC space)t han varying the loss ratios in Ripper or class priors in Naive Bayes. Our method of over-sampling the minority class involves creating synthetic minority class examples. Experiments are performed using C4.5, Ripper and a Naive Bayes classifier. The method is evaluated using the area under the Receiver Operating Characteristic curve (AUC)and the ROC convex hull strategy.
{{< /ci-details >}}
{{< ci-details summary="Who belongs in the family? (R. L. Thorndike, 1953)">}}
R. L. Thorndike. (1953)  
**Who belongs in the family?**  
  
[Paper Link](https://www.semanticscholar.org/paper/47706e9fdfe6b7d33d09579e60d6c9732cfa90e7)  
Influential Citation Count (18), SS-ID (47706e9fdfe6b7d33d09579e60d6c9732cfa90e7)  
{{< /ci-details >}}
