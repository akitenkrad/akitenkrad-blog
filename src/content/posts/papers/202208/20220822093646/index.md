---
draft: true
title: "dynnode2vec: Scalable Dynamic Network Embedding"
date: 2022-08-22
author: "akitenkrad"
description: ""
tags: ["At:Round-2", "Published:2018"]
menu:
  sidebar:
    name: "dynnode2vec: Scalable Dynamic Network Embedding"
    identifier: 20220822
    parent: 202208
    weight: 10
math: true
---

- [x] Round-1: Overview
- [x] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation
{{< citation >}}
Mahdavi, S., Khoshraftar, S., & An, A. (2018).  
dynnode2vec: Scalable Dynamic Network Embedding.  
https://doi.org/10.48550/arxiv.1812.02356
{{< /citation >}}

## Abstract
> Network representation learning in low dimensional vector space has attracted considerable attention in both academic and industrial domains. Most real-world networks are dynamic with addition/deletion of nodes and edges. The existing graph embedding methods are designed for static networks and they cannot capture evolving patterns in a large dynamic network. In this paper, we propose a dynamic embedding method, dynnode2vec, based on the well-known graph embedding method node2vec. Node2vec is a random walk based embedding method for static networks. Applying static network embedding in dynamic settings has two crucial problems: 1) Generating random walks for every time step is time consuming 2) Embedding vector spaces in each timestamp are different. In order to tackle these challenges, dynnode2vec uses evolving random walks and initializes the current graph embedding with previous embedding vectors. We demonstrate the advantages of the proposed dynamic network embedding by conducting empirical evaluations on several large dynamic network datasets.

## Background & Wat's New

## Dataset

## Model Description

## Results

### Settings

## References


{{< ci-details summary="Overview of the 2003 KDD Cup (J. Gehrke et al., 2003)">}}
J. Gehrke, P. Ginsparg, J. Kleinberg. (2003)  
**Overview of the 2003 KDD Cup**  
SKDD  
[Paper Link](https://www.semanticscholar.org/paper/05d5a28fd29fdbd405743cd282888e463c8cb26a)  
Influential Citation Count (19), SS-ID (05d5a28fd29fdbd405743cd282888e463c8cb26a)  
**ABSTRACT**  
This paper surveys the 2003 KDD Cup, a competition held in conjunction with the Ninth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD) in August 2003. The competition focused on mining the complex real-life social network inherent in the e-print arXiv (arXiv.org). We describe the four KDD Cup tasks: citation prediction, download prediction, data cleaning, and an open task.
{{< /ci-details >}}
{{< ci-details summary="LINE: Large-scale Information Network Embedding (Jian Tang et al., 2015)">}}
Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Q. Mei. (2015)  
**LINE: Large-scale Information Network Embedding**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/0834e74304b547c9354b6d7da6fa78ef47a48fa8)  
Influential Citation Count (873), SS-ID (0834e74304b547c9354b6d7da6fa78ef47a48fa8)  
**ABSTRACT**  
This paper studies the problem of embedding very large information networks into low-dimensional vector spaces, which is useful in many tasks such as visualization, node classification, and link prediction. Most existing graph embedding methods do not scale for real world information networks which usually contain millions of nodes. In this paper, we propose a novel network embedding method called the ``LINE,'' which is suitable for arbitrary types of information networks: undirected, directed, and/or weighted. The method optimizes a carefully designed objective function that preserves both the local and global network structures. An edge-sampling algorithm is proposed that addresses the limitation of the classical stochastic gradient descent and improves both the effectiveness and the efficiency of the inference. Empirical experiments prove the effectiveness of the LINE on a variety of real-world information networks, including language networks, social networks, and citation networks. The algorithm is very efficient, which is able to learn the embedding of a network with millions of vertices and billions of edges in a few hours on a typical single machine. The source code of the LINE is available online\footnote{\url{https://github.com/tangjianpku/LINE}}.
{{< /ci-details >}}
{{< ci-details summary="Deep Neural Networks for Learning Graph Representations (Shaosheng Cao et al., 2016)">}}
Shaosheng Cao, Wei Lu, Qiongkai Xu. (2016)  
**Deep Neural Networks for Learning Graph Representations**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/1a37f07606d60df365d74752857e8ce909f700b3)  
Influential Citation Count (62), SS-ID (1a37f07606d60df365d74752857e8ce909f700b3)  
**ABSTRACT**  
In this paper, we propose a novel model for learning graph representations, which generates a low-dimensional vector representation for each vertex by capturing the graph structural information. Different from other previous research efforts, we adopt a random surfing model to capture graph structural information directly, instead of using the sampling-based method for generating linear sequences proposed by Perozzi et al. (2014). The advantages of our approach will be illustrated from both theorical and empirical perspectives. We also give a new perspective for the matrix factorization method proposed by Levy and Goldberg (2014), in which the pointwise mutual information (PMI) matrix is considered as an analytical solution to the objective function of the skip-gram model with negative sampling proposed by Mikolov et al. (2013). Unlike their approach which involves the use of the SVD for finding the low-dimensitonal projections from the PMI matrix, however, the stacked denoising autoencoder is introduced in our model to extract complex features and model non-linearities. To demonstrate the effectiveness of our model, we conduct experiments on clustering and visualization tasks, employing the learned vertex representations as features. Empirical results on datasets of varying sizes show that our model outperforms other stat-of-the-art models in such tasks.
{{< /ci-details >}}
{{< ci-details summary="Visualizing Data using t-SNE (L. V. D. Maaten et al., 2008)">}}
L. V. D. Maaten, Geoffrey E. Hinton. (2008)  
**Visualizing Data using t-SNE**  
  
[Paper Link](https://www.semanticscholar.org/paper/1c46943103bd7b7a2c7be86859995a4144d1938b)  
Influential Citation Count (882), SS-ID (1c46943103bd7b7a2c7be86859995a4144d1938b)  
**ABSTRACT**  
We present a new technique called “t-SNE” that visualizes high-dimensional data by giving each datapoint a location in a two or three-dimensional map. The technique is a variation of Stochastic Neighbor Embedding (Hinton and Roweis, 2002) that is much easier to optimize, and produces significantly better visualizations by reducing the tendency to crowd points together in the center of the map. t-SNE is better than existing techniques at creating a single map that reveals structure at many different scales. This is particularly important for high-dimensional data that lie on several different, but related, low-dimensional manifolds, such as images of objects from multiple classes seen from multiple viewpoints. For visualizing the structure of very large datasets, we show how t-SNE can use random walks on neighborhood graphs to allow the implicit structure of all of the data to influence the way in which a subset of the data is displayed. We illustrate the performance of t-SNE on a wide variety of datasets and compare it with many other non-parametric visualization techniques, including Sammon mapping, Isomap, and Locally Linear Embedding. The visualizations produced by t-SNE are significantly better than those produced by the other techniques on almost all of the datasets.
{{< /ci-details >}}
{{< ci-details summary="DynGEM: Deep Embedding Method for Dynamic Graphs (Palash Goyal et al., 2018)">}}
Palash Goyal, Nitin Kamra, Xinran He, Yan Liu. (2018)  
**DynGEM: Deep Embedding Method for Dynamic Graphs**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/1d49c0dd13911f44418d46ec5fac128d6c4bbf59)  
Influential Citation Count (35), SS-ID (1d49c0dd13911f44418d46ec5fac128d6c4bbf59)  
**ABSTRACT**  
Embedding large graphs in low dimensional spaces has recently attracted significant interest due to its wide applications such as graph visualization, link prediction and node classification. Existing methods focus on computing the embedding for static graphs. However, many graphs in practical applications are dynamic and evolve constantly over time. Naively applying existing embedding algorithms to each snapshot of dynamic graphs independently usually leads to unsatisfactory performance in terms of stability, flexibility and efficiency. In this work, we present an efficient algorithm DynGEM based on recent advances in deep autoencoders for graph embeddings, to address this problem. The major advantages of DynGEM include: (1) the embedding is stable over time, (2) it can handle growing dynamic graphs, and (3) it has better running time than using static embedding methods on each snapshot of a dynamic graph. We test DynGEM on a variety of tasks including graph visualization, graph reconstruction, link prediction and anomaly detection (on both synthetic and real datasets). Experimental results demonstrate the superior stability and scalability of our approach.
{{< /ci-details >}}
{{< ci-details summary="node2vec: Scalable Feature Learning for Networks (Aditya Grover et al., 2016)">}}
Aditya Grover, J. Leskovec. (2016)  
**node2vec: Scalable Feature Learning for Networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/36ee2c8bd605afd48035d15fdc6b8c8842363376)  
Influential Citation Count (1175), SS-ID (36ee2c8bd605afd48035d15fdc6b8c8842363376)  
**ABSTRACT**  
Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader field of representation learning has led to significant progress in automating prediction by learning the features themselves. However, present feature learning approaches are not expressive enough to capture the diversity of connectivity patterns observed in networks. Here we propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes. We define a flexible notion of a node's network neighborhood and design a biased random walk procedure, which efficiently explores diverse neighborhoods. Our algorithm generalizes prior work which is based on rigid notions of network neighborhoods, and we argue that the added flexibility in exploring neighborhoods is the key to learning richer representations. We demonstrate the efficacy of node2vec over existing state-of-the-art techniques on multi-label classification and link prediction in several real-world networks from diverse domains. Taken together, our work represents a new way for efficiently learning state-of-the-art task-independent representations in complex networks.
{{< /ci-details >}}
{{< ci-details summary="Graph Embedding Techniques, Applications, and Performance: A Survey (Palash Goyal et al., 2017)">}}
Palash Goyal, Emilio Ferrara. (2017)  
**Graph Embedding Techniques, Applications, and Performance: A Survey**  
Knowl. Based Syst.  
[Paper Link](https://www.semanticscholar.org/paper/374b4409f6a1d2d853af31e329f025da239d375f)  
Influential Citation Count (48), SS-ID (374b4409f6a1d2d853af31e329f025da239d375f)  
{{< /ci-details >}}
{{< ci-details summary="{SNAP Datasets}: {Stanford} Large Network Dataset Collection (J. Leskovec et al., 2014)">}}
J. Leskovec, A. Krevl. (2014)  
**{SNAP Datasets}: {Stanford} Large Network Dataset Collection**  
  
[Paper Link](https://www.semanticscholar.org/paper/404ae4a2b31d5c2184861cf702f953e47db40cab)  
Influential Citation Count (340), SS-ID (404ae4a2b31d5c2184861cf702f953e47db40cab)  
**ABSTRACT**  
A collection of more than 50 large network datasets from tens of thousands of nodes and edges to tens of millions of nodes and edges. In includes social networks, web graphs, road networks, internet networks, citation networks, collaboration networks, and communication networks.
{{< /ci-details >}}
{{< ci-details summary="Variational Graph Auto-Encoders (Thomas Kipf et al., 2016)">}}
Thomas Kipf, M. Welling. (2016)  
**Variational Graph Auto-Encoders**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/54906484f42e871f7c47bbfe784a358b1448231f)  
Influential Citation Count (403), SS-ID (54906484f42e871f7c47bbfe784a358b1448231f)  
**ABSTRACT**  
We introduce the variational graph auto-encoder (VGAE), a framework for unsupervised learning on graph-structured data based on the variational auto-encoder (VAE). This model makes use of latent variables and is capable of learning interpretable latent representations for undirected graphs. We demonstrate this model using a graph convolutional network (GCN) encoder and a simple inner product decoder. Our model achieves competitive results on a link prediction task in citation networks. In contrast to most existing models for unsupervised learning on graph-structured data and link prediction, our model can naturally incorporate node features, which significantly improves predictive performance on a number of benchmark datasets.
{{< /ci-details >}}
{{< ci-details summary="Personalized entity recommendation: a heterogeneous information network approach (Xiao Yu et al., 2014)">}}
Xiao Yu, Xiang Ren, Yizhou Sun, Quanquan Gu, Bradley Sturt, Urvashi Khandelwal, Brandon Norick, Jiawei Han. (2014)  
**Personalized entity recommendation: a heterogeneous information network approach**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/65a442711000fcc1c0309106caa27a949c325566)  
Influential Citation Count (45), SS-ID (65a442711000fcc1c0309106caa27a949c325566)  
**ABSTRACT**  
Among different hybrid recommendation techniques, network-based entity recommendation methods, which utilize user or item relationship information, are beginning to attract increasing attention recently. Most of the previous studies in this category only consider a single relationship type, such as friendships in a social network. In many scenarios, the entity recommendation problem exists in a heterogeneous information network environment. Different types of relationships can be potentially used to improve the recommendation quality. In this paper, we study the entity recommendation problem in heterogeneous information networks. Specifically, we propose to combine heterogeneous relationship information for each user differently and aim to provide high-quality personalized recommendation results using user implicit feedback data and personalized recommendation models. In order to take full advantage of the relationship heterogeneity in information networks, we first introduce meta-path-based latent features to represent the connectivity between users and items along different types of paths. We then define recommendation models at both global and personalized levels and use Bayesian ranking optimization techniques to estimate the proposed models. Empirical studies show that our approaches outperform several widely employed or the state-of-the-art entity recommendation techniques.
{{< /ci-details >}}
{{< ci-details summary="Node Classification in Social Networks (Smriti Bhagat et al., 2011)">}}
Smriti Bhagat, Graham Cormode, S. Muthukrishnan. (2011)  
**Node Classification in Social Networks**  
Social Network Data Analytics  
[Paper Link](https://www.semanticscholar.org/paper/6e45220c1f3a8a8cbf176a2fc722c7e8380d5dd4)  
Influential Citation Count (12), SS-ID (6e45220c1f3a8a8cbf176a2fc722c7e8380d5dd4)  
{{< /ci-details >}}
{{< ci-details summary="The link-prediction problem for social networks (D. Liben-Nowell et al., 2007)">}}
D. Liben-Nowell, J. Kleinberg. (2007)  
**The link-prediction problem for social networks**  
J. Assoc. Inf. Sci. Technol.  
[Paper Link](https://www.semanticscholar.org/paper/996dfa43f6982bcbff862276ef80cbca7515985a)  
Influential Citation Count (285), SS-ID (996dfa43f6982bcbff862276ef80cbca7515985a)  
**ABSTRACT**  
Given a snapshot of a social network, can we infer which new interactions among its members are likely to occur in the near future? We formalize this question as the link-prediction problem, and we develop approaches to link prediction based on measures for analyzing the “proximity” of nodes in a network. Experiments on large coauthorship networks suggest that information about future interactions can be extracted from network topology alone, and that fairly subtle measures for detecting node proximity can outperform more direct measures. © 2007 Wiley Periodicals, Inc.
{{< /ci-details >}}
{{< ci-details summary="Temporal Analysis of Language through Neural Language Models (Yoon Kim et al., 2014)">}}
Yoon Kim, Yi-I Chiu, K. Hanaki, Darshan Hegde, Slav Petrov. (2014)  
**Temporal Analysis of Language through Neural Language Models**  
LTCSS@ACL  
[Paper Link](https://www.semanticscholar.org/paper/99a94646f09e25bd1ea3f556488f63228f534e1f)  
Influential Citation Count (43), SS-ID (99a94646f09e25bd1ea3f556488f63228f534e1f)  
**ABSTRACT**  
We provide a method for automatically detecting change in language across time through a chronologically trained neural language model. We train the model on the Google Books Ngram corpus to obtain word vector representations specific to each year, and identify words that have changed significantly from 1900 to 2009. The model identifies words such as cell and gay as having changed during that time period. The model simultaneously identifies the specific years during which such words underwent change.
{{< /ci-details >}}
{{< ci-details summary="Dynamic Word Embeddings (Robert Bamler et al., 2017)">}}
Robert Bamler, S. Mandt. (2017)  
**Dynamic Word Embeddings**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/b2535d2e7629f37571046f9abcb91feeced3b3c2)  
Influential Citation Count (19), SS-ID (b2535d2e7629f37571046f9abcb91feeced3b3c2)  
**ABSTRACT**  
We present a probabilistic language model for time-stamped text data which tracks the semantic evolution of individual words over time. The model represents words and contexts by latent trajectories in an embedding space. At each moment in time, the embedding vectors are inferred from a probabilistic version of word2vec [Mikolov et al., 2013]. These embedding vectors are connected in time through a latent diffusion process. We describe two scalable variational inference algorithms--skip-gram smoothing and skip-gram filtering--that allow us to train the model jointly over all times; thus learning on all data while simultaneously allowing word and context vectors to drift. Experimental results on three different corpora demonstrate that our dynamic model infers word embedding trajectories that are more interpretable and lead to higher predictive likelihoods than competing methods that are based on static models trained separately on time slices.
{{< /ci-details >}}
{{< ci-details summary="GraRep: Learning Graph Representations with Global Structural Information (Shaosheng Cao et al., 2015)">}}
Shaosheng Cao, Wei Lu, Qiongkai Xu. (2015)  
**GraRep: Learning Graph Representations with Global Structural Information**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/c2fd72cb2a77941e655b5d949d0d59b01e173c3b)  
Influential Citation Count (143), SS-ID (c2fd72cb2a77941e655b5d949d0d59b01e173c3b)  
**ABSTRACT**  
In this paper, we present {GraRep}, a novel model for learning vertex representations of weighted graphs. This model learns low dimensional vectors to represent vertices appearing in a graph and, unlike existing work, integrates global structural information of the graph into the learning process. We also formally analyze the connections between our work and several previous research efforts, including the DeepWalk model of Perozzi et al. as well as the skip-gram model with negative sampling of Mikolov et al. We conduct experiments on a language network, a social network as well as a citation network and show that our learned global representations can be effectively used as features in tasks such as clustering, classification and visualization. Empirical results demonstrate that our representation significantly outperforms other state-of-the-art methods in such tasks.
{{< /ci-details >}}
{{< ci-details summary="Incremental Skip-gram Model with Negative Sampling (Nobuhiro Kaji et al., 2017)">}}
Nobuhiro Kaji, Hayato Kobayashi. (2017)  
**Incremental Skip-gram Model with Negative Sampling**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/c485fa1e053fe65621bb76bf0ab1789472e21427)  
Influential Citation Count (2), SS-ID (c485fa1e053fe65621bb76bf0ab1789472e21427)  
**ABSTRACT**  
This paper explores an incremental training strategy for the skip-gram model with negative sampling (SGNS) from both empirical and theoretical perspectives. Existing methods of neural word embeddings, including SGNS, are multi-pass algorithms and thus cannot perform incremental model update. To address this problem, we present a simple incremental extension of SGNS and provide a thorough theoretical analysis to demonstrate its validity. Empirical experiments demonstrated the correctness of the theoretical analysis as well as the practical usefulness of the incremental algorithm.
{{< /ci-details >}}
{{< ci-details summary="Streaming Word Embeddings with the Space-Saving Algorithm (Chandler May et al., 2017)">}}
Chandler May, Kevin Duh, Benjamin Van Durme, Ashwin Lall. (2017)  
**Streaming Word Embeddings with the Space-Saving Algorithm**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/ef6fea9d88aa763460b6cf48b2f60d23c6e60e9c)  
Influential Citation Count (1), SS-ID (ef6fea9d88aa763460b6cf48b2f60d23c6e60e9c)  
**ABSTRACT**  
We develop a streaming (one-pass, bounded-memory) word embedding algorithm based on the canonical skip-gram with negative sampling algorithm implemented in word2vec. We compare our streaming algorithm to word2vec empirically by measuring the cosine similarity between word pairs under each algorithm and by applying each algorithm in the downstream task of hashtag prediction on a two-month interval of the Twitter sample stream. We then discuss the results of these experiments, concluding they provide partial validation of our approach as a streaming replacement for word2vec. Finally, we discuss potential failure modes and suggest directions for future work.
{{< /ci-details >}}
{{< ci-details summary="ArnetMiner: extraction and mining of academic social networks (Jie Tang et al., 2008)">}}
Jie Tang, Jing Zhang, Limin Yao, Juan-Zi Li, Li Zhang, Z. Su. (2008)  
**ArnetMiner: extraction and mining of academic social networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/f020b61789112fe7241b871907268f0bdc5c84fa)  
Influential Citation Count (184), SS-ID (f020b61789112fe7241b871907268f0bdc5c84fa)  
**ABSTRACT**  
This paper addresses several key issues in the ArnetMiner system, which aims at extracting and mining academic social networks. Specifically, the system focuses on: 1) Extracting researcher profiles automatically from the Web; 2) Integrating the publication data into the network from existing digital libraries; 3) Modeling the entire academic network; and 4) Providing search services for the academic network. So far, 448,470 researcher profiles have been extracted using a unified tagging approach. We integrate publications from online Web databases and propose a probabilistic framework to deal with the name ambiguity problem. Furthermore, we propose a unified modeling approach to simultaneously model topical aspects of papers, authors, and publication venues. Search services such as expertise search and people association search have been provided based on the modeling results. In this paper, we describe the architecture and main features of the system. We also present the empirical evaluation of the proposed methods.
{{< /ci-details >}}
{{< ci-details summary="Heterogeneous Network Embedding via Deep Architectures (Shiyu Chang et al., 2015)">}}
Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, C. Aggarwal, Thomas S. Huang. (2015)  
**Heterogeneous Network Embedding via Deep Architectures**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/f7172f95a3c0aa4fddfaadbce9908ce20cbf50ef)  
Influential Citation Count (38), SS-ID (f7172f95a3c0aa4fddfaadbce9908ce20cbf50ef)  
**ABSTRACT**  
Data embedding is used in many machine learning applications to create low-dimensional feature representations, which preserves the structure of data points in their original space. In this paper, we examine the scenario of a heterogeneous network with nodes and content of various types. Such networks are notoriously difficult to mine because of the bewildering combination of heterogeneous contents and structures. The creation of a multidimensional embedding of such data opens the door to the use of a wide variety of off-the-shelf mining techniques for multidimensional data. Despite the importance of this problem, limited efforts have been made on embedding a network of scalable, dynamic and heterogeneous data. In such cases, both the content and linkage structure provide important cues for creating a unified feature representation of the underlying network. In this paper, we design a deep embedding algorithm for networked data. A highly nonlinear multi-layered embedding function is used to capture the complex interactions between the heterogeneous data in a network. Our goal is to create a multi-resolution deep embedding function, that reflects both the local and global network structures, and makes the resulting embedding useful for a variety of data mining tasks. In particular, we demonstrate that the rich content and linkage information in a heterogeneous network can be captured by such an approach, so that similarities among cross-modal data can be measured directly in a common embedding space. Once this goal has been achieved, a wide variety of data mining problems can be solved by applying off-the-shelf algorithms designed for handling vector representations. Our experiments on real-world network datasets show the effectiveness and scalability of the proposed algorithm as compared to the state-of-the-art embedding methods.
{{< /ci-details >}}
{{< ci-details summary="DeepWalk: online learning of social representations (Bryan Perozzi et al., 2014)">}}
Bryan Perozzi, Rami Al-Rfou, S. Skiena. (2014)  
**DeepWalk: online learning of social representations**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/fff114cbba4f3ba900f33da574283e3de7f26c83)  
Influential Citation Count (1404), SS-ID (fff114cbba4f3ba900f33da574283e3de7f26c83)  
**ABSTRACT**  
We present DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk's latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information. DeepWalk's representations can provide F1 scores up to 10% higher than competing methods when labeled data is sparse. In some experiments, DeepWalk's representations are able to outperform all baseline methods while using 60% less training data. DeepWalk is also scalable. It is an online learning algorithm which builds useful incremental results, and is trivially parallelizable. These qualities make it suitable for a broad class of real world applications such as network classification, and anomaly detection.
{{< /ci-details >}}
